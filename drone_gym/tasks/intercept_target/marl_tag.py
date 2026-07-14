"""MARL 3D tag (pursuit–evasion) — both drones learn.

This is the multi-agent counterpart of the SARL ``sarl_tag`` task. There, the
runner learned while an *expert* pure-pursuit policy drove the interceptor. Here
**both** drones are learning agents controlled by the trainer:

  * **runner** (``drone_0``) — spawns at the centre and must reach a random 3D
    goal while evading the interceptor.
  * **interceptor** (``drone_1``) — spawns on the runner's likely path and must
    catch the runner (get within ``capture_threshold``) before it reaches the
    goal.

It is a near-zero-sum competitive game: the runner's success is the
interceptor's failure and vice-versa. The reward for each agent is therefore the
mirror of the other's (see :meth:`_calculate_rewards`).

Structurally this subclasses :class:`MarlDroneEnvironment`, so it inherits the
PettingZoo parallel API (dict observations/actions/rewards/terminations) and the
land → teleport → take-off reset that repositions every drone's *model* to its
spawn each episode (the dead-drone fix — nobody flies home from a crash site).

Homogeneous spaces (required by the shared ``observation_space(agent)`` /
``action_space(agent)``):

  * action  = ``[vx, vy, vz]`` in ``[-1, 1]`` (scaled by ``max_velocity`` /
    ``max_velocity_z``), same for both agents.
  * observation (14) = own state (6) + opponent block (4) + goal block (4). Both
    agents observe the goal: the interceptor uses it to anticipate where the
    runner is headed and cut it off.

Launch with the multi-agent SITL so both ports exist::

    ./sitl_multiagent_square.sh -m crazyflie -n 2   # 19850 runner, 19851 interceptor

NOTE (safety): unlike the SARL task, this does not yet run the high-rate
background collision guard that stops both drones the instant they are within
``capture_threshold``. Capture is detected at the step boundary. For hardware or
faster interceptors, port ``_safety_monitor_loop`` from ``sarl_tag`` before use.
"""

import math
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from drone_gym.marl_drone_environment import MarlDroneEnvironment


class MarlTag(MarlDroneEnvironment):
    """Competitive 3D pursuit–evasion for two learning drones."""

    def __init__(
        self,
        use_simulator: Literal[0, 1],
        max_velocity: float = 0.25,
        max_velocity_z: float = 0.03,
        step_time: float = 0.5,
        xy_limit: float = 2.0,
        z_min: float = 0.8,
        z_max: float = 1.2,
        reset_height: float = 1.0,
        reset_spacing: float = 0.5,
        episode_length: int = 80,
        capture_threshold: float = 0.30,
        goal_threshold: float = 0.20,
    ) -> None:
        # Exactly two agents: possible_agents becomes ["drone_0", "drone_1"].
        super().__init__(
            use_simulator=use_simulator,
            num_agents=2,
            max_velocity=max_velocity,
            max_velocity_z=max_velocity_z,
            step_time=step_time,
            xy_limit=xy_limit,
            z_min=z_min,
            z_max=z_max,
            reset_height=reset_height,
            reset_spacing=reset_spacing,
        )

        # Role labels over the base's positional agent ids. Keeping the base's
        # names (rather than renaming possible_agents) leaves all of its
        # bookkeeping — uris, drones, reset_positions — keyed consistently.
        self.RUNNER = self.possible_agents[0]        # drone_0 -> udp 19850
        self.INTERCEPTOR = self.possible_agents[1]   # drone_1 -> udp 19851

        self.episode_length = episode_length
        self.time_tolerance = 0.15  # look-ahead slack for the boundary brake

        # --- Win conditions --------------------------------------------------
        self.capture_threshold = capture_threshold  # interceptor "catches" the runner (3D)
        self.goal_threshold = goal_threshold        # runner reaches the goal (3D)

        # --- Geometry (mirrors sarl_tag) ------------------------------------
        self.runner_spawn = [0.0, 0.0, self.reset_height]
        self.spawn_margin = 0.5
        self.goal_margin = 0.3
        self.z_margin = 0.1
        self.out_of_bounds_tolerance = 0.05

        self.goal_min_distance = 1.5
        self.goal_max_distance = 2.4

        # Fair-race interceptor placement along the runner->goal path.
        self.intercept_frac = (0.45, 0.65)
        self.fairness_jitter = 0.15
        self.interceptor_z_jitter = 0.1
        self.min_runner_clearance = 1.0   # never spawn in point-blank capture range
        self.min_goal_clearance = 0.6     # never spawn camped on the goal
        # Symmetric speeds -> ratio 1.0. Raise if you later give the interceptor
        # a higher max_velocity (see the per-agent speed note in step processing).
        self.interceptor_speed_ratio = 1.0

        # --- Stability: boundary brake + slew limit --------------------------
        # BOTH agents now learn, so BOTH emit high-entropy actions that topple
        # the Crazyflie if applied as instantaneous velocity reversals. The SARL
        # task only needed to smooth the runner (its interceptor ran a smooth
        # expert policy); here the interceptor needs the same treatment.
        self.boundary_brake_margin = 0.25
        self.z_brake_margin = 0.10
        # Max change in commanded velocity per step, per axis (m/s). Matches the
        # SARL effective cap: 0.2 (normalized) * 0.25 (max_velocity) = 0.05 m/s.
        self.max_velocity_delta = 0.05
        self._prev_applied_action: dict[str, list[float]] = {
            agent: [0.0, 0.0, 0.0] for agent in self.possible_agents
        }

        # --- Reward parameters ----------------------------------------------
        self.success_reward = 100.0            # runner reaches goal
        self.capture_reward = 100.0            # interceptor catches runner
        self.out_of_bounds_penalty = 100.0     # runner leaves the arena (magnitude; applied negative)
        self.goal_progress_multiplier = 100.0  # runner: reward closing on the goal
        self.capture_progress_multiplier = 100.0  # interceptor: reward closing on the runner
        self.step_penalty = 1.0                # both: small per-step cost (act fast)
        self.danger_radius = 0.6               # runner: evasion shaping kicks in inside this
        self.danger_penalty = 5.0              # runner: max shaping penalty at zero separation

        # --- Task state ------------------------------------------------------
        self.goal_position: list[float] = [0.0, 0.0, self.reset_height]
        self.caught = False
        self.reached_goal = False
        self.winner: str | None = None
        self.previous_goal_distance = self.max_distance_3d
        self.previous_capture_distance = self.max_distance_3d
        self.runner_success_count = 0
        self.interceptor_success_count = 0

        # --- Spaces (homogeneous across agents) ------------------------------
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # own(6) + opponent(4) + goal(4) = 14
        obs_dim = 6 + 4 + 4
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # global state: 2 positions(6) + 2 velocities(6) + goal(3) = 15
        state_dim = 2 * 3 + 2 * 3 + 3
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Reset — sample fresh geometry, then teleport both drones to spawns
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Sample the episode geometry, write it into ``reset_positions`` so the
        base teleport places each drone at its spawn, then defer to the base
        reset (land -> teleport -> take off -> velocity control)."""
        # The runner always starts at the centre; the goal and interceptor spawn
        # are randomized. Sample here (before the base reset) because the base's
        # _reset_all_drones teleports drones to reset_positions.
        self.goal_position = self._sample_goal(self.runner_spawn)
        interceptor_spawn = self._sample_interceptor_spawn(self.runner_spawn, self.goal_position)

        self.reset_positions[self.RUNNER] = list(self.runner_spawn)
        self.reset_positions[self.INTERCEPTOR] = interceptor_spawn

        return super().reset(seed=seed, options=options)

    def _reset_task_state(self) -> None:
        """Draw the goal marker and reset per-episode task state. Called by the
        base reset AFTER both drones are repositioned to their spawns."""
        self.caught = False
        self.reached_goal = False
        self.winner = None

        # Both agents start the episode at rest.
        for agent in self.possible_agents:
            self._prev_applied_action[agent] = [0.0, 0.0, 0.0]

        # Seed the progress trackers from the true starting geometry so the first
        # step's progress reward is measured against the spawn, not a stale value.
        runner_pos = self.drones[self.RUNNER].get_position()
        interceptor_pos = self.drones[self.INTERCEPTOR].get_position()
        self.previous_goal_distance = self._distance_3d(runner_pos, self.goal_position)
        self.previous_capture_distance = self._distance_3d(interceptor_pos, runner_pos)

        if self.use_simulator:
            self._set_target_marker(position=self.goal_position, marker_name="tag_goal")

    # ------------------------------------------------------------------
    # Geometry sampling (ported from sarl_tag, 3D)
    # ------------------------------------------------------------------

    def _sample_goal(self, runner_pos: list[float]) -> list[float]:
        """Sample a goal between goal_min/max_distance (3D) from the runner."""
        xy = self.xy_limit - self.goal_margin
        z_lo, z_hi = self.z_min + self.z_margin, self.z_max - self.z_margin
        for _ in range(300):
            x = float(np.random.uniform(-xy, xy))
            y = float(np.random.uniform(-xy, xy))
            z = float(np.random.uniform(z_lo, z_hi))
            d = self._distance_3d([x, y, z], runner_pos)
            if self.goal_min_distance <= d <= self.goal_max_distance:
                return [x, y, z]
        # Fallback: mirror the runner across the origin in xy, hold mid altitude.
        return [
            float(np.clip(-runner_pos[0], -xy, xy)),
            float(np.clip(-runner_pos[1], -xy, xy)),
            self.reset_height,
        ]

    def _sample_interceptor_spawn(self, runner_pos: list[float], goal_pos: list[float]) -> list[float]:
        """Seed the interceptor for a FAIR race to contest the runner's path.

        A contest point is chosen a fraction ``f`` along the runner->goal line;
        the interceptor is placed abeam it at a lateral distance scaled by the
        speed ratio so both arrive at roughly the same time. Clearances keep it
        out of point-blank range of the runner and off the goal.
        """
        xy = self.xy_limit - self.spawn_margin
        z_lo, z_hi = self.z_min + self.z_margin, self.z_max - self.z_margin

        dx = goal_pos[0] - runner_pos[0]
        dy = goal_pos[1] - runner_pos[1]
        dz = goal_pos[2] - runner_pos[2]
        D = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        xy_len = math.hypot(dx, dy) or 1.0
        px, py = -dy / xy_len, dx / xy_len  # unit perpendicular to the path in xy
        speed_ratio = self.interceptor_speed_ratio

        def _clearances_ok(p):
            d_runner = self._distance_3d(p, runner_pos)
            d_goal = self._distance_3d(p, goal_pos)
            return d_runner >= self.min_runner_clearance and d_goal >= self.min_goal_clearance

        fallback = None
        for _ in range(400):
            f = float(np.random.uniform(*self.intercept_frac))
            cx = runner_pos[0] + f * dx
            cy = runner_pos[1] + f * dy
            cz = runner_pos[2] + f * dz
            L = speed_ratio * f * D * float(np.random.uniform(1.0 - self.fairness_jitter,
                                                              1.0 + self.fairness_jitter))
            side = 1.0 if np.random.random() < 0.5 else -1.0
            ix = cx + side * px * L
            iy = cy + side * py * L
            iz = float(np.clip(cz + float(np.random.uniform(-self.interceptor_z_jitter,
                                                            self.interceptor_z_jitter)), z_lo, z_hi))
            candidate = [ix, iy, iz]

            if abs(ix) <= xy and abs(iy) <= xy and _clearances_ok(candidate):
                return candidate
            if fallback is None:
                clamped = [float(np.clip(ix, -xy, xy)), float(np.clip(iy, -xy, xy)), iz]
                if _clearances_ok(clamped):
                    fallback = clamped

        if fallback is not None:
            return fallback
        # Last resort: abeam the midpoint at the runner-clearance distance.
        cx = runner_pos[0] + 0.55 * dx
        cy = runner_pos[1] + 0.55 * dy
        cz = runner_pos[2] + 0.55 * dz
        L = max(self.min_runner_clearance, 1.0)
        return [
            float(np.clip(cx + px * L, -xy, xy)),
            float(np.clip(cy + py * L, -xy, xy)),
            float(np.clip(cz, z_lo, z_hi)),
        ]

    # ------------------------------------------------------------------
    # Per-step action processing — boundary brake + slew limit (both agents)
    # ------------------------------------------------------------------

    def _apply_task_action_processing(
        self,
        agent: str,
        vx: float,
        vy: float,
        vz: float,
        current_position: list[float],
    ) -> tuple[float, float, float, dict[str, Any]]:
        """Zero any velocity component that would push a drone further out of
        bounds, then slew-limit the change from last step so a full reversal
        ramps over several steps instead of toppling the Crazyflie.

        (Per-agent speed asymmetry — e.g. a faster interceptor — would go here:
        scale this agent's components and re-clip. Kept symmetric for v1 because
        ``_denormalize_action`` has no agent argument.)
        """
        requested = [vx, vy, vz]
        prediction_time = self.step_time + self.time_tolerance

        px = current_position[0] + vx * prediction_time
        py = current_position[1] + vy * prediction_time
        pz = current_position[2] + vz * prediction_time

        xy_brake = self.xy_limit - self.boundary_brake_margin
        if px > xy_brake and vx > 0:
            vx = 0.0
        elif px < -xy_brake and vx < 0:
            vx = 0.0
        if py > xy_brake and vy > 0:
            vy = 0.0
        elif py < -xy_brake and vy < 0:
            vy = 0.0

        z_hi = self.z_max - self.z_brake_margin
        z_lo = self.z_min + self.z_brake_margin
        if pz > z_hi and vz > 0:
            vz = 0.0
        elif pz < z_lo and vz < 0:
            vz = 0.0

        prev = self._prev_applied_action[agent]
        limited = []
        for cur, p in zip((vx, vy, vz), prev):
            delta = max(-self.max_velocity_delta, min(self.max_velocity_delta, cur - p))
            limited.append(p + delta)
        self._prev_applied_action[agent] = limited

        info = {"requested_velocity": requested, "sent_velocity": list(limited)}
        return limited[0], limited[1], limited[2], info

    # ------------------------------------------------------------------
    # Observations — homogeneous 14-vector: own(6) + opponent(4) + goal(4)
    # ------------------------------------------------------------------

    def _opponent_of(self, agent: str) -> str:
        return self.INTERCEPTOR if agent == self.RUNNER else self.RUNNER

    def _get_observations(self) -> dict[str, np.ndarray]:
        positions = {agent: self.drones[agent].get_position() for agent in self.agents}
        observations: dict[str, np.ndarray] = {}

        for agent in self.agents:
            own_pos = positions[agent]
            vel = self.drones[agent].get_calculated_velocity()
            own_vel = [float(vel.get("x", 0.0)), float(vel.get("y", 0.0)), float(vel.get("z", 0.0))]

            opp = self._opponent_of(agent)
            opp_pos = positions.get(opp) or self.drones[opp].get_position()

            opp_rel = [opp_pos[i] - own_pos[i] for i in range(3)]
            opp_dist = self._distance_3d(opp_pos, own_pos)
            goal_rel = [self.goal_position[i] - own_pos[i] for i in range(3)]
            goal_dist = self._distance_3d(self.goal_position, own_pos)

            obs_parts = [
                self._normalize_position(own_pos),                       # 3
                self._normalize_velocity(own_vel),                       # 3
                self._normalize_relative_position(opp_rel),              # 3
                np.array([opp_dist / self.max_distance_3d], np.float32),  # 1
                self._normalize_relative_position(goal_rel),             # 3
                np.array([goal_dist / self.max_distance_3d], np.float32),  # 1
            ]
            observations[agent] = np.concatenate(obs_parts).astype(np.float32)

        return observations

    # ------------------------------------------------------------------
    # Rewards — near-zero-sum: each agent's reward mirrors the other's
    # ------------------------------------------------------------------

    def _calculate_rewards(self, state_dicts: dict[str, dict[str, Any]]) -> dict[str, float]:
        goal_distance = self._runner_goal_distance(state_dicts)
        capture_distance = self._capture_distance(state_dicts)
        runner_pos = self._runner_position(state_dicts)

        runner_oob = self._is_out_of_task_bounds(runner_pos)
        caught = capture_distance < self.capture_threshold
        reached = goal_distance < self.goal_threshold

        # --- Runner reward ---
        if runner_oob:
            r_runner = -self.out_of_bounds_penalty
        elif caught:
            r_runner = -self.capture_reward
        elif reached:
            r_runner = self.success_reward
        else:
            r_runner = (self.previous_goal_distance - goal_distance) * self.goal_progress_multiplier
            r_runner -= self.step_penalty
            if capture_distance < self.danger_radius:
                closeness = 1.0 - (capture_distance / self.danger_radius)
                r_runner -= self.danger_penalty * closeness

        # --- Interceptor reward (mirror) ---
        if caught:
            r_interceptor = self.capture_reward
        elif reached:
            r_interceptor = -self.success_reward  # runner escaped to the goal
        elif runner_oob:
            r_interceptor = 0.0  # runner eliminated itself; interceptor neither rewarded nor blamed
        else:
            r_interceptor = (self.previous_capture_distance - capture_distance) * self.capture_progress_multiplier
            r_interceptor -= self.step_penalty

        self.previous_goal_distance = goal_distance
        self.previous_capture_distance = capture_distance

        rewards = {self.RUNNER: float(r_runner), self.INTERCEPTOR: float(r_interceptor)}
        return {agent: rewards[agent] for agent in self.agents}

    # ------------------------------------------------------------------
    # Terminations / truncations
    # ------------------------------------------------------------------

    def _check_terminations(self, state_dicts: dict[str, dict[str, Any]]) -> dict[str, bool]:
        goal_distance = self._runner_goal_distance(state_dicts)
        capture_distance = self._capture_distance(state_dicts)
        runner_pos = self._runner_position(state_dicts)

        self.reached_goal = goal_distance < self.goal_threshold
        self.caught = capture_distance < self.capture_threshold
        runner_oob = self._is_out_of_task_bounds(runner_pos)

        terminal = self.reached_goal or self.caught or runner_oob

        if terminal and self.winner is None:
            if self.reached_goal:
                self.winner = self.RUNNER
                self.runner_success_count += 1
                if self._is_evaluating:
                    self.success_counts[self.RUNNER] = self.success_counts.get(self.RUNNER, 0) + 1
            elif self.caught:
                self.winner = self.INTERCEPTOR
                self.interceptor_success_count += 1
                if self._is_evaluating:
                    self.success_counts[self.INTERCEPTOR] = self.success_counts.get(self.INTERCEPTOR, 0) + 1
            else:
                self.winner = "none"  # runner out of bounds — no winner

        # Competitive episode: it ends for both agents at once.
        return {agent: terminal for agent in self.agents}

    def _check_truncations(self, state_dicts: dict[str, dict[str, Any]]) -> dict[str, bool]:
        time_limit_reached = self.steps >= self.episode_length

        any_low_battery = any(
            state_dicts[agent]["battery"] < self.battery_threshold for agent in self.agents
        )

        # A z-band violation (either drone) usually means an EKF thrust-spike
        # launch — truncate before it drifts into the internal kill boundary.
        z_violation = [
            agent for agent in self.agents
            if not (self.z_min <= state_dicts[agent]["position"][2] <= self.z_max)
        ]
        if z_violation:
            print(f"[MarlTag] z-boundary violation {z_violation} — truncating episode")

        truncate_all = time_limit_reached or any_low_battery or bool(z_violation)
        return {agent: truncate_all for agent in self.agents}

    # ------------------------------------------------------------------
    # Info / global state / render
    # ------------------------------------------------------------------

    def _get_infos(
        self,
        state_dicts: dict[str, dict[str, Any]] | None = None,
        denormalized_actions: dict[str, list[float]] | None = None,
        normalized_actions: dict[str, np.ndarray] | None = None,
        old_positions: dict[str, list[float]] | None = None,
        new_positions: dict[str, list[float]] | None = None,
        action_filter_infos: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if state_dicts is None:
            positions = {agent: self.drones[agent].get_position() for agent in self.agents}
            state_dicts = self._generate_state_dicts(positions)

        goal_distance = self._runner_goal_distance(state_dicts)
        capture_distance = self._capture_distance(state_dicts)

        infos: dict[str, dict[str, Any]] = {}
        for agent in self.agents:
            info: dict[str, Any] = {
                "role": "runner" if agent == self.RUNNER else "interceptor",
                "goal_position": self.goal_position[:],
                "distance_to_goal": goal_distance,
                "separation": capture_distance,
                "caught": self.caught,
                "reached_goal": self.reached_goal,
                "winner": self.winner,
                "in_boundaries": state_dicts[agent]["in_boundaries"],
                "battery": state_dicts[agent]["battery"],
            }
            if denormalized_actions is not None:
                info["denormalized_action"] = denormalized_actions.get(agent)
            if normalized_actions is not None:
                a = normalized_actions.get(agent)
                info["normalized_action"] = a.tolist() if a is not None else None
            if action_filter_infos is not None:
                info["action_filter_info"] = action_filter_infos.get(agent)
            if self._is_evaluating:
                info["success_counts"] = dict(self.success_counts)
            infos[agent] = info

        return infos

    def _get_global_state(self) -> np.ndarray:
        parts = []
        for agent in self.possible_agents:
            parts.append(self._normalize_position(self.drones[agent].get_position()))
        for agent in self.possible_agents:
            vel = self.drones[agent].get_calculated_velocity()
            parts.append(self._normalize_velocity(
                [float(vel.get("x", 0.0)), float(vel.get("y", 0.0)), float(vel.get("z", 0.0))]
            ))
        parts.append(self._normalize_position(self.goal_position))
        return np.concatenate(parts).astype(np.float32)

    def _render_task_specific_info(self) -> None:
        runner_pos = self.drones[self.RUNNER].get_position()
        interceptor_pos = self.drones[self.INTERCEPTOR].get_position()
        print(f"Goal:               {[round(v, 2) for v in self.goal_position]}")
        print(f"Runner:             {[round(v, 2) for v in runner_pos]}")
        print(f"Interceptor:        {[round(v, 2) for v in interceptor_pos]}")
        print(f"Distance to goal:   {self._distance_3d(runner_pos, self.goal_position):.2f} "
              f"(threshold {self.goal_threshold:.2f})")
        print(f"Separation:         {self._distance_3d(runner_pos, interceptor_pos):.2f} "
              f"(capture {self.capture_threshold:.2f})")
        print(f"Reached goal: {self.reached_goal} | Caught: {self.caught} | Winner: {self.winner}")

    # ------------------------------------------------------------------
    # Distances / bounds — base hook + helpers
    # ------------------------------------------------------------------

    def _distance_to_target(self, agent: str, position: list[float]) -> float:
        """Base hook feeding ``state_dicts[agent]['distance_to_target']``.

        The runner's "target" is the goal; the interceptor's "target" is the
        runner. This is what lets both progress signals fall straight out of the
        state dict in :meth:`_calculate_rewards`.
        """
        if agent == self.RUNNER:
            return self._distance_3d(position, self.goal_position)
        return self._distance_3d(position, self.drones[self.RUNNER].get_position())

    @staticmethod
    def _distance_3d(a: list[float], b: list[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _is_out_of_task_bounds(self, position: list[float]) -> bool:
        tol = self.out_of_bounds_tolerance
        return (abs(position[0]) > self.xy_limit + tol or
                abs(position[1]) > self.xy_limit + tol or
                position[2] < self.z_min - tol or
                position[2] > self.z_max + tol)

    # --- state_dict accessors (robust to an agent already being deactivated) ---

    def _runner_goal_distance(self, state_dicts: dict[str, dict[str, Any]]) -> float:
        if self.RUNNER in state_dicts:
            return state_dicts[self.RUNNER]["distance_to_target"]
        return self._distance_3d(self.drones[self.RUNNER].get_position(), self.goal_position)

    def _capture_distance(self, state_dicts: dict[str, dict[str, Any]]) -> float:
        if self.INTERCEPTOR in state_dicts:
            return state_dicts[self.INTERCEPTOR]["distance_to_target"]
        return self._distance_3d(
            self.drones[self.INTERCEPTOR].get_position(),
            self.drones[self.RUNNER].get_position(),
        )

    def _runner_position(self, state_dicts: dict[str, dict[str, Any]]) -> list[float]:
        if self.RUNNER in state_dicts:
            return state_dicts[self.RUNNER]["position"]
        return self.drones[self.RUNNER].get_position()
