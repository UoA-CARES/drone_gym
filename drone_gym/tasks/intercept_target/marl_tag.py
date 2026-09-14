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
from collections import deque
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from drone_gym.marl_drone_environment import MarlDroneEnvironment


class MarlTag(MarlDroneEnvironment):
    """Competitive 3D pursuit–evasion for two learning drones."""

    def __init__(
        self,
        use_simulator: Literal[0, 1],
        num_agents: int = 2,
        max_velocity: float = 0.25,
        max_velocity_z: float = 0.03,
        step_time: float = 0.5,
        xy_limit: float = 2.0,
        z_min: float = 0.4,
        z_max: float = 2.5,
        reset_height: float = 1.0,
        reset_spacing: float = 0.5,
        episode_length: int = 80,
        capture_threshold: float = 0.30,
        goal_threshold: float = 0.20,
    ) -> None:

        self.num_interceptor_agents = num_agents - 1
        self.num_runner_agents = 1

        super().__init__(
            use_simulator=use_simulator,
            num_agents=num_agents,
            max_velocity=max_velocity,
            max_velocity_z=max_velocity_z,
            step_time=step_time,
            xy_limit=xy_limit,
            z_min=z_min,
            z_max=z_max,
            reset_height=reset_height,
            reset_spacing=reset_spacing,
        )

        self.runner_agents: list[str] = [self.possible_agents[0]]
        self.interceptor_agents: list[str] = self.possible_agents[1:]

        self.episode_length = episode_length
        self.time_tolerance = 0.15  # look-ahead slack for the boundary brake

        # --- Win conditions --------------------------------------------------
        self.capture_threshold = (
            capture_threshold  # interceptor "catches" the runner (3D)
        )
        self.goal_threshold = goal_threshold  # runner reaches the goal (3D)

        # --- Geometry (mirrors sarl_tag) ------------------------------------
        self.out_of_bounds_tolerance = 0.05
        self.interceptor_speed_ratio = 1.0

        # --- Reset geometry ---------------------------------------------------
        self.goal_min_distance_ratio = 0.60
        self.interceptor_min_distance_ratio = 0.25
        # self.interceptor_goal_min_distance_ratio = 0.15

        self.z_margin = 0.1

        self.max_layout_sampling_attempts = 100
        self.max_position_sampling_attempts = 300

        # --- Interceptor speed curriculum (performance-gated ratchet) --------
        # The interceptor is a learner, so this caps its physical chase speed
        # (applied in _apply_task_action_processing). Start slow so the runner
        # can learn to navigate, then raise the cap as its success rate climbs.
        self.curriculum_enabled = True
        self.interceptor_speed_min = 0.05  # starting cap (m/s)
        self.interceptor_speed_max = self.max_velocity  # ceiling = runner's max speed
        self.curriculum_window = 50
        self.curriculum_success_threshold = 0.6
        self.curriculum_speed_step = 0.02
        self.interceptor_speed = (
            self.interceptor_speed_min if self.curriculum_enabled else self.max_velocity
        )
        self.interceptor_speed_ratio = self.interceptor_speed / max(
            self.max_velocity, 1e-6
        )
        self._episode_count = 0
        self._recent_runner_outcomes = deque(maxlen=self.curriculum_window)

        # --- Stability: boundary brake + slew limit --------------------------
        # BOTH agents now learn, so BOTH emit high-entropy actions that topple
        # the Crazyflie if applied as instantaneous velocity reversals. The SARL
        # task only needed to smooth the runner (its interceptor ran a smooth
        # expert policy); here the interceptor needs the same treatment.
        self.boundary_brake_margin = 0.2
        self.z_brake_margin = 0.10
        # Max change in commanded velocity per step, per axis (m/s). Matches the
        # SARL effective cap: 0.2 (normalized) * 0.25 (max_velocity) = 0.05 m/s.
        self.max_velocity_delta = 0.05
        self._prev_applied_action: dict[str, list[float]] = {
            agent: [0.0, 0.0, 0.0] for agent in self.possible_agents
        }

        # --- Reward parameters ----------------------------------------------
        self.success_reward = 100.0  # runner reaches goal
        self.capture_reward = 100.0  # interceptor catches runner
        self.boundary_penalty_at_limit = -1.0  # penalty at the boundary limit
        self.goal_progress_multiplier = 100.0  # runner: reward closing on the goal
        self.capture_progress_multiplier = (
            100.0  # interceptor: reward closing on the runner
        )
        # self.step_penalty = 1.0  # both: small per-step cost (act fast)
        self.danger_radius = 0.6  # runner: evasion shaping kicks in inside this
        self.danger_penalty = 5.0  # runner: max shaping penalty at zero separation

        # --- Task state ------------------------------------------------------
        self.goal_position: list[float] = [0.0, 0.0, self.reset_height]
        self.caught = False
        self.reached_goal = False
        self.winner: str | None = None
        self.previous_goal_distance = self.max_distance_3d
        self.previous_capture_distances: dict[str, float] = {
            interceptor: self.max_distance_3d for interceptor in self.interceptor_agents
        }
        self.runner_success_count = 0
        self.interceptor_success_count = 0

        # --- Spaces (homogeneous across agents) ------------------------------
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Runner observation:
        # [own vel, own pos, other agents rel pos, other good agents vel, goal rel pos]
        runner_obs_dim = (
            3 + 3 + (num_agents - 1) * 3 + (len(self.runner_agents) - 1) * 3 + 3
        )
        self._runner_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(runner_obs_dim,), dtype=np.float32
        )

        # Interceptor observation:
        # [own vel, own pos, other agents rel pos, other good agents vel]
        interceptor_obs_dim = 3 + 3 + (num_agents - 1) * 3 + len(self.runner_agents) * 3
        self._interceptor_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(interceptor_obs_dim,), dtype=np.float32
        )

        # TODO: This needs to be updated to reflect the actual observation space for each agent.
        # global state: 2 positions(6) + 2 velocities(6) + goal(3) = 15
        state_dim = 2 * 3 + 2 * 3 + 3
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Reset — sample fresh geometry, then teleport both drones to spawns
    # ------------------------------------------------------------------

    def _update_interceptor_curriculum(self, training: bool = True) -> None:
        """Raise the interceptor's speed cap as the runner's success rate climbs.
        Ratchets up only; stalls if the runner plateaus."""
        if not self.curriculum_enabled or not training:
            return
        # self.winner still holds the just-finished episode's result here
        # (_reset_task_state clears it later during super().reset()).
        if self._episode_count > 1:
            self._recent_runner_outcomes.append(
                1.0 if self.winner == self.runner_agents else 0.0
            )
        if len(self._recent_runner_outcomes) < self.curriculum_window:
            return
        success_rate = sum(self._recent_runner_outcomes) / len(
            self._recent_runner_outcomes
        )
        if (
            success_rate >= self.curriculum_success_threshold
            and self.interceptor_speed < self.interceptor_speed_max
        ):
            self.interceptor_speed = min(
                self.interceptor_speed_max,
                self.interceptor_speed + self.curriculum_speed_step,
            )
            self.interceptor_speed_ratio = self.interceptor_speed / max(
                self.max_velocity, 1e-6
            )
            self._recent_runner_outcomes.clear()
            print(
                f"[MarlTag][curriculum] runner success {success_rate:.0%} -> "
                f"interceptor speed {self.interceptor_speed:.3f} m/s"
            )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Sample the episode geometry, write it into ``reset_positions`` so the
        base teleport places each drone at its spawn, then defer to the base
        reset (land -> teleport -> take off -> velocity control)."""
        training = (options or {}).get("training", True)
        self._episode_count += 1
        # Update speed BEFORE sampling the spawn so the new interceptor_speed_ratio
        # feeds the fair-spawn placement.
        self._update_interceptor_curriculum(training)

        self.reset_positions = self._generate_reset_positions()

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

        runner_pos = self.drones[self.runner_agents].get_position()
        self.previous_goal_distance = self._distance_3d(runner_pos, self.goal_position)

        self.previous_capture_distances = {
            interceptor: self._distance_3d(
                self.drones[interceptor].get_position(),
                runner_pos,
            )
            for interceptor in self.interceptor_agents
        }

        if self.use_simulator:
            self._set_target_marker(position=self.goal_position, marker_name="tag_goal")

    # ------------------------------------------------------------------
    # Geometry sampling (ported from sarl_tag, 3D)
    # ------------------------------------------------------------------

    def _sample_random_position(self) -> list[float]:
        xy = self.reset_planner.usable_xy_limit
        z_lo = self.z_min + self.z_margin
        z_hi = self.z_max - self.z_margin

        return [
            float(np.random.uniform(-xy, xy)),
            float(np.random.uniform(-xy, xy)),
            float(np.random.uniform(z_lo, z_hi)),
        ]

    def _runner_has_valid_goal_region(
        self,
        runner_position: list[float],
        minimum_goal_distance: float,
    ) -> bool:
        xy = self.reset_planner.usable_xy_limit

        max_dx = max(
            abs(runner_position[0] - (-xy)),
            abs(runner_position[0] - xy),
        )
        max_dy = max(
            abs(runner_position[1] - (-xy)),
            abs(runner_position[1] - xy),
        )

        max_possible_distance = math.hypot(max_dx, max_dy)

        return max_possible_distance >= minimum_goal_distance

    def _sample_goal(
        self,
        runner_position: list[float],
        minimum_distance: float,
    ) -> list[float] | None:
        while True:
            goal_position = self._sample_random_position()

            if (
                self.reset_planner.distance_xy(
                    runner_position,
                    goal_position,
                )
                >= minimum_distance
            ):
                return goal_position

    def _sample_interceptor_spawn(
        self,
        runner_position: list[float],
        reset_positions: dict[str, list[float]],
        minimum_runner_distance: float,
    ) -> list[float] | None:
        for _ in range(self.max_position_sampling_attempts):
            position = self._sample_random_position()

            if (
                self.reset_planner.distance_xy(
                    position,
                    runner_position,
                )
                < minimum_runner_distance
            ):
                continue

            if not self.reset_planner.is_xy_position_clear(
                position,
                reset_positions,
                self.reset_planner.slot_clearance,
            ):
                continue

            return position

        return None

    def _generate_reset_positions(self) -> dict[str, list[float]]:

        usable_xy_size = 2.0 * self.reset_planner.usable_xy_limit

        goal_min_distance = self.goal_min_distance_ratio * usable_xy_size
        interceptor_runner_min_distance = max(
            self.interceptor_min_distance_ratio * usable_xy_size,
            self.reset_planner.slot_clearance,
        )

        runner_agent = self.runner_agents[0]

        for _ in range(self.max_layout_sampling_attempts):
            reset_positions: dict[str, list[float]] = {}

            # 1. Sample runner.
            runner_position = self._sample_random_position()

            # 2. Reject runners from which no valid goal can exist.
            if not self._runner_has_valid_goal_region(
                runner_position,
                goal_min_distance,
            ):
                continue

            reset_positions[runner_agent] = runner_position

            # 3. Sample goal.
            goal_position = self._sample_goal(
                runner_position,
                goal_min_distance,
            )

            # 4. Sequentially sample interceptors.
            interceptor_positions: list[list[float]] = []
            layout_failed = False

            for _interceptor_agent in self.interceptor_agents:
                interceptor_position = self._sample_interceptor_spawn(
                    runner_position=runner_position,
                    reset_positions=reset_positions,
                    minimum_runner_distance=interceptor_runner_min_distance,
                )

                if interceptor_position is None:
                    layout_failed = True
                    break

                interceptor_positions.append(interceptor_position)

                # Temporary key so subsequent interceptor samples see this slot.
                reset_positions[_interceptor_agent] = interceptor_position

            if layout_failed:
                continue

            # Remove identity bias from sequential placement.
            np.random.shuffle(interceptor_positions)

            for agent, position in zip(
                self.interceptor_agents,
                interceptor_positions,
            ):
                reset_positions[agent] = position

            # 5. Final physical safety validation.
            try:
                self.reset_planner.validate_reset_positions(reset_positions)
            except self.reset_planner.ConfigurationError:
                # 6. Entire layout is rejected.
                continue

            self.goal_position = goal_position

            return reset_positions

        raise RuntimeError(
            "Unable to generate a valid MarlTag reset layout after "
            f"{self.max_layout_sampling_attempts} attempts."
        )

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

        # Curriculum speed cap: scale the interceptor's commanded xy velocity so
        # its top speed = interceptor_speed. The runner is unaffected.
        if agent == self.interceptor_agents and self.curriculum_enabled:
            ratio = self.interceptor_speed / max(self.max_velocity, 1e-6)
            vx *= ratio
            vy *= ratio

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

    # -------------
    # Observations
    # -------------

    def _get_runner_observations(self, agent: str) -> np.ndarray:
        """Runner sees its own state, the interceptor's relative position, and the
        goal's relative position."""
        vel = self.drones[agent].get_calculated_velocity()
        own_vel = [
            float(vel.get("x", 0.0)),
            float(vel.get("y", 0.0)),
            float(vel.get("z", 0.0)),
        ]
        own_pos = self.drones[agent].get_position()

        # Relative position of every other drone
        other_agents_rel_pos = np.array([], dtype=np.float32)
        for other in self.possible_agents:
            if other != agent:
                other_pos = self.drones[other].get_position()
                rel_pos = self._relative_position(other_pos, own_pos)
                other_agents_rel_pos = np.concatenate(
                    (other_agents_rel_pos, self._normalize_relative_pos(rel_pos))
                )

        # Other runner agent vel
        other_good_agents_vel = np.array([], dtype=np.float32)
        for other in self.runner_agents:
            if other != agent:
                vel = self.drones[other].get_calculated_velocity()
                other_vel = [
                    float(vel.get("x", 0.0)),
                    float(vel.get("y", 0.0)),
                    float(vel.get("z", 0.0)),
                ]
                other_good_agents_vel = np.concatenate(
                    (other_good_agents_vel, self._normalize_vel(other_vel))
                )

        goal_pos = self.goal_position
        goal_rel = self._relative_position(goal_pos, own_pos)

        obs_parts = [
            self._normalize_vel(own_vel),  # 3
            self._normalize_pos(own_pos),  # 3
            other_agents_rel_pos,  # 3 * (num_agents - 1)
            other_good_agents_vel,  # 3 * (num_runner_agents - 1)
            self._normalize_relative_pos(goal_rel),  # 3
        ]
        return np.concatenate(obs_parts).astype(np.float32)

    def _get_interceptor_observations(self, agent: str) -> np.ndarray:
        """Interceptor sees its own state, the runner's relative position, and the
        goal's relative position."""
        vel = self.drones[agent].get_calculated_velocity()
        own_vel = [
            float(vel.get("x", 0.0)),
            float(vel.get("y", 0.0)),
            float(vel.get("z", 0.0)),
        ]
        own_pos = self.drones[agent].get_position()

        # Relative position of every other drone
        other_agents_rel_pos = np.array([], dtype=np.float32)
        for other in self.possible_agents:
            if other != agent:
                other_pos = self.drones[other].get_position()
                rel_pos = self._relative_position(other_pos, own_pos)
                other_agents_rel_pos = np.concatenate(
                    (other_agents_rel_pos, self._normalize_relative_pos(rel_pos))
                )

        # Other runner agent vel
        other_good_agents_vel = np.array([], dtype=np.float32)
        for other in self.runner_agents:
            vel = self.drones[other].get_calculated_velocity()
            other_vel = [
                float(vel.get("x", 0.0)),
                float(vel.get("y", 0.0)),
                float(vel.get("z", 0.0)),
            ]
            other_good_agents_vel = np.concatenate(
                (other_good_agents_vel, self._normalize_vel(other_vel))
            )

        obs_parts = [
            self._normalize_vel(own_vel),  # 3
            self._normalize_pos(own_pos),  # 3
            other_agents_rel_pos,  # 3 * (num_agents - 1)
            other_good_agents_vel,  # 3 * num_runner_agents
        ]
        return np.concatenate(obs_parts).astype(np.float32)

    def _get_observations(self) -> dict[str, np.ndarray]:
        observations: dict[str, np.ndarray] = {}

        for agent in self.agents:
            if agent in self.runner_agents:
                observations[agent] = self._get_runner_observations(agent)
            else:
                observations[agent] = self._get_interceptor_observations(agent)

        return observations

    # -------
    # Rewards
    # -------

    def _calculate_rewards(
        self, state_dicts: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        runner_agent = self.runner_agents[0]

        goal_distance = self._runner_goal_distance(state_dicts)
        capture_distances = self._capture_distances(state_dicts)
        runner_pos = self._runner_position(state_dicts)

        caught = self._capture_occurred(capture_distances)
        reached = goal_distance < self.goal_threshold
        runner_oob = self._is_out_of_task_bounds(runner_pos)

        # Runner reward
        runner_reward = (
            self.previous_goal_distance - goal_distance
        ) * self.goal_progress_multiplier

        # Interceptor reward
        interceptor_rewards: dict[str, float] = {}

        for interceptor in self.interceptor_agents:
            capture_distance = capture_distances[interceptor]

            progress = self.previous_capture_distances[interceptor] - capture_distance

            interceptor_rewards[interceptor] = (
                progress * self.capture_progress_multiplier
            )

        if caught:
            runner_reward -= self.capture_reward
            for interceptor in self.interceptor_agents:
                interceptor_rewards[interceptor] += self.capture_reward
        elif reached:
            runner_reward += self.success_reward
            for interceptor in self.interceptor_agents:
                interceptor_rewards[interceptor] -= self.success_reward

        self.previous_goal_distance = goal_distance
        self.previous_capture_distances = capture_distances

        rewards = {
            runner_agent: float(runner_reward),
            **interceptor_rewards,
        }

        for agent in self.agents:
            position = state_dicts[agent]["position"]
            rewards[agent] += self._boundary_penalty(position)

        return rewards

    def _boundary_penalty(self, position: list[float]) -> float:
        """Return the individual boundary shaping penalty for one drone."""

        def _risk(
            distance_from_centre: float,
            penalty_start: float,
            hard_limit: float,
        ) -> float:
            if distance_from_centre <= penalty_start:
                return 0.0

            penalty_width = hard_limit - penalty_start

            if penalty_width <= 0.0:
                return 1.0

            # Linear increase through the boundary margin.
            if distance_from_centre <= hard_limit:
                return (distance_from_centre - penalty_start) / penalty_width

            # Exponential increase beyond the hard boundary.
            overshoot = distance_from_centre - hard_limit
            normalised_overshoot = overshoot / penalty_width

            return float(
                # Exponential boundary rate.
                np.exp(1.0 * normalised_overshoot)
            )

        # XY boundary penalty.
        xy_penalty_start = self.xy_limit - self.boundary_brake_margin

        x_risk = _risk(
            abs(position[0]),
            xy_penalty_start,
            self.xy_limit,
        )

        y_risk = _risk(
            abs(position[1]),
            xy_penalty_start,
            self.xy_limit,
        )

        # Z boundary penalty.
        z_mid = (self.z_min + self.z_max) / 2.0

        if position[2] >= z_mid:
            z_distance = position[2] - z_mid
            hard_z_extent = self.z_max - z_mid
            z_penalty_start = self.z_max - self.z_brake_margin - z_mid
        else:
            z_distance = z_mid - position[2]
            hard_z_extent = z_mid - self.z_min
            z_penalty_start = z_mid - self.z_min - self.z_brake_margin

        z_risk = _risk(
            z_distance,
            z_penalty_start,
            hard_z_extent,
        )

        # Use the most severe boundary risk.
        boundary_risk = max(
            x_risk,
            y_risk,
            z_risk,
        )

        return self.boundary_penalty_at_limit * boundary_risk

    # ------------------------------------------------------------------
    # Terminations / truncations
    # ------------------------------------------------------------------

    def _check_terminations(
        self, state_dicts: dict[str, dict[str, Any]]
    ) -> dict[str, bool]:
        goal_distance = self._runner_goal_distance(state_dicts)
        capture_distance = self._capture_distance(state_dicts)
        runner_pos = self._runner_position(state_dicts)

        self.reached_goal = goal_distance < self.goal_threshold
        self.caught = capture_distance < self.capture_threshold
        runner_oob = self._is_out_of_task_bounds(runner_pos)

        terminal = self.reached_goal or self.caught or runner_oob

        if terminal and self.winner is None:
            if self.reached_goal:
                self.winner = self.runner_agents
                self.runner_success_count += 1
                if self._is_evaluating:
                    self.success_counts[self.runner_agents] = (
                        self.success_counts.get(self.runner_agents, 0) + 1
                    )
            elif self.caught:
                self.winner = self.interceptor_agents
                self.interceptor_success_count += 1
                if self._is_evaluating:
                    self.success_counts[self.interceptor_agents] = (
                        self.success_counts.get(self.interceptor_agents, 0) + 1
                    )
            else:
                self.winner = "none"  # runner out of bounds — no winner

        # Competitive episode: it ends for both agents at once.
        return {agent: terminal for agent in self.agents}

    def _check_truncations(
        self, state_dicts: dict[str, dict[str, Any]]
    ) -> dict[str, bool]:
        time_limit_reached = self.steps >= self.episode_length

        any_low_battery = any(
            state_dicts[agent]["battery"] < self.battery_threshold
            for agent in self.agents
        )

        # A z-band violation (either drone) usually means an EKF thrust-spike
        # launch — truncate before it drifts into the internal kill boundary.
        z_violation = [
            agent
            for agent in self.agents
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
            positions = {
                agent: self.drones[agent].get_position() for agent in self.agents
            }
            state_dicts = self._generate_state_dicts(positions)

        goal_distance = self._runner_goal_distance(state_dicts)
        capture_distance = self._capture_distance(state_dicts)

        infos: dict[str, dict[str, Any]] = {}
        for agent in self.agents:
            info: dict[str, Any] = {
                "role": "runner" if agent == self.runner_agents else "interceptor",
                "goal_position": self.goal_position[:],
                "distance_to_goal": goal_distance,
                "separation": capture_distance,
                "caught": self.caught,
                "reached_goal": self.reached_goal,
                "winner": self.winner,
                # Per-drone outcome flags (1/0), shared across both agents'
                # info dicts. Picked up automatically by the generic
                # success-rate / time-to-outcome plots (any "*_success"
                # column) once MARLDroneEnvironment hoists them to the top
                # level of the logged info.
                "runner_success": int(self.winner == self.RUNNER),
                "interceptor_success": int(self.winner == self.INTERCEPTOR),
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
            parts.append(self._normalize_pos(self.drones[agent].get_position()))
        for agent in self.possible_agents:
            vel = self.drones[agent].get_calculated_velocity()
            parts.append(
                self._normalize_vel(
                    [
                        float(vel.get("x", 0.0)),
                        float(vel.get("y", 0.0)),
                        float(vel.get("z", 0.0)),
                    ]
                )
            )
        parts.append(self._normalize_pos(self.goal_position))
        return np.concatenate(parts).astype(np.float32)

    def _render_task_specific_info(self) -> None:
        runner_pos = self.drones[self.runner_agents].get_position()
        interceptor_pos = self.drones[self.interceptor_agents].get_position()
        print(f"Goal:               {[round(v, 2) for v in self.goal_position]}")
        print(f"Runner:             {[round(v, 2) for v in runner_pos]}")
        print(f"Interceptor:        {[round(v, 2) for v in interceptor_pos]}")
        print(
            f"Distance to goal:   {self._distance_3d(runner_pos, self.goal_position):.2f} "
            f"(threshold {self.goal_threshold:.2f})"
        )
        print(
            f"Separation:         {self._distance_3d(runner_pos, interceptor_pos):.2f} "
            f"(capture {self.capture_threshold:.2f})"
        )
        print(
            f"Reached goal: {self.reached_goal} | Caught: {self.caught} | Winner: {self.winner}"
        )

    # ------------------------------------------------------------------
    # Distances / bounds — base hook + helpers
    # ------------------------------------------------------------------

    def _distance_to_target(self, agent: str, position: list[float]) -> float:
        """Base hook feeding ``state_dicts[agent]['distance_to_target']``.

        The runner's "target" is the goal; the interceptor's "target" is the
        runner. This is what lets both progress signals fall straight out of the
        state dict in :meth:`_calculate_rewards`.
        """
        if agent == self.runner_agents:
            return self._distance_3d(position, self.goal_position)
        return self._distance_3d(
            position, self.drones[self.runner_agents].get_position()
        )

    @staticmethod
    def _distance_3d(a: list[float], b: list[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _is_out_of_task_bounds(self, position: list[float]) -> bool:
        tol = self.out_of_bounds_tolerance
        return (
            abs(position[0]) > self.xy_limit + tol
            or abs(position[1]) > self.xy_limit + tol
            or position[2] < self.z_min - tol
            or position[2] > self.z_max + tol
        )

    # --- state_dict accessors (robust to an agent already being deactivated) ---

    def _runner_goal_distance(self, state_dicts: dict[str, dict[str, Any]]) -> float:
        if self.runner_agents in state_dicts:
            return state_dicts[self.runner_agents]["distance_to_target"]
        return self._distance_3d(
            self.drones[self.runner_agents].get_position(), self.goal_position
        )

    def _capture_distance(self, state_dicts: dict[str, dict[str, Any]]) -> float:
        if self.interceptor_agents in state_dicts:
            return state_dicts[self.interceptor_agents]["distance_to_target"]
        return self._distance_3d(
            self.drones[self.interceptor_agents].get_position(),
            self.drones[self.runner_agents].get_position(),
        )

    def _capture_distances(
        self,
        state_dicts: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        runner_agent = self.runner_agents[0]

        if runner_agent in state_dicts:
            runner_position = state_dicts[runner_agent]["position"]
        else:
            runner_position = self.drones[runner_agent].get_position()

        distances: dict[str, float] = {}

        for interceptor in self.interceptor_agents:
            if interceptor in state_dicts:
                interceptor_position = state_dicts[interceptor]["position"]
            else:
                interceptor_position = self.drones[interceptor].get_position()

            distances[interceptor] = self._distance_3d(
                interceptor_position,
                runner_position,
            )

        return distances

    def _capture_occurred(
        self,
        capture_distances: dict[str, float],
    ) -> bool:
        return any(
            distance < self.capture_threshold for distance in capture_distances.values()
        )

    def _runner_position(self, state_dicts: dict[str, dict[str, Any]]) -> list[float]:
        if self.runner_agents in state_dicts:
            return state_dicts[self.runner_agents]["position"]
        return self.drones[self.runner_agents].get_position()

    def _generate_possible_agents(self) -> list[str]:
        return [
            "runner_0",
            *[f"interceptor_{i}" for i in range(self.num_agents_config - 1)],
        ]

    def observation_space(self, agent: str) -> spaces.Space:
        """
        Return gymnasium Space object for the given agent's observations.
        """
        if agent in self.runner_agents:
            return self._runner_observation_space
        if agent in self.interceptor_agents:
            return self._interceptor_observation_space
