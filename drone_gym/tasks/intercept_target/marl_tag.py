import math
from collections import deque
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from drone_gym.marl_drone_environment import MarlDroneEnvironment


class MarlTag(MarlDroneEnvironment):
    """
    Multi-agent pursuit-evasion task with one runner and one or more interceptors.

    The runner attempts to reach a randomly sampled goal while the interceptors
    attempt to capture it. Runner-interceptor proximity is treated as capture,
    while collisions between non-opposing agents truncate the episode for safety.
    """

    RUNNER = "runner"
    INTERCEPTOR = "interceptor"
    CURRICULUM_STAGES = [
        {"interceptor_speed_factor": 0.40},
        {"interceptor_speed_factor": 0.60},
        {"interceptor_speed_factor": 0.80},
        {"interceptor_speed_factor": 1.00},
    ]

    def __init__(
        self,
        use_simulator: Literal[0, 1],
        num_agents: int = 2,
        max_velocity: float = 0.25,
        max_velocity_z: float = 0.25,
        step_time: float = 0.5,
        xy_limit: float = 2.0,
        z_min: float = 0.4,
        z_max: float = 2.5,
        reset_height: float = 1.0,
        reset_spacing: float = 0.5,
        episode_length: int = 80,
        capture_threshold: float = 0.20,
        goal_threshold: float = 0.20,
    ) -> None:

        self.num_interceptor_agents = num_agents - 1
        self.num_runner_agents = 1

        # Hard safety boundary
        if use_simulator:
            # No risk in sim, allow for larger unbounded flight area.
            boundaries = {"x": 10.0, "y": 10.0, "z_min": 0.1, "z_max": 10.0}
        else:
            boundaries = {"x": 2.5, "y": 2.5, "z_min": 0.1, "z_max": 3.0}

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
            boundaries=boundaries,
            collision_safety_distance=capture_threshold,
        )

        self.runner_agents: list[str] = [self.possible_agents[0]]
        self.interceptor_agents: list[str] = self.possible_agents[1:]

        self.runner_max_velocity = max_velocity
        self.interceptor_max_velocity = max_velocity

        self.episode_length = episode_length

        # Win conditions
        self.capture_threshold = capture_threshold
        self.goal_threshold = goal_threshold

        # Reset geometry
        self.goal_min_distance_ratio = 0.60
        self.interceptor_min_distance_ratio = 0.25

        self.z_margin = 0.1

        self.max_layout_sampling_attempts = 100
        self.max_position_sampling_attempts = 300

        # Interceptor speed curriculum
        self.curriculum_enabled = True
        self.curriculum_stage = 0
        self.curriculum_window = 50
        self.curriculum_success_threshold = 0.6
        self.curriculum_interceptor_vel_factor = 1.0
        self._episode_count = 0
        self._recent_runner_outcomes = deque(maxlen=self.curriculum_window)

        # Reward parameters
        self.success_reward = 100.0  # runner reaches goal
        self.capture_reward = 100.0  # interceptor catches runner
        self.goal_progress_multiplier = 100.0  # runner: goal shaping reward
        self.capture_progress_multiplier = 100.0  # interceptor: capture shaping reward

        # Penalty at the boundary limit
        # Multiplied by linear within margin, exponential beyond limit
        self.boundary_penalty_at_limit = -1.0
        # Margin before the boundary where linear penalty starts
        self.boundary_penalty_margin = 0.2
        # Margin in the z-direction before the boundary limit where linear penalty starts
        self.z_boundary_penalty_margin = 0.10

        # Task state
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
        self._step_collision_outcomes: tuple[bool, bool] = (False, False)
        self.reset_positions: dict[str, list[float]] = {}

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

        # Global state space
        state_dim = (
            runner_obs_dim * self.num_runner_agents
            + interceptor_obs_dim * self.num_interceptor_agents
        )
        self.state_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Agent configuration
    # ------------------------------------------------------------------
    def _generate_possible_agents(self) -> list[str]:
        """Generate agent names for MarlTag task."""
        return [
            f"{self.RUNNER}_0",
            *[f"{self.INTERCEPTOR}_{i}" for i in range(self.num_interceptor_agents)],
        ]

    def observation_space(self, agent: str) -> spaces.Space:
        """Return gymnasium Space object for given agent's observations."""
        if agent in self.runner_agents:
            return self._runner_observation_space
        if agent in self.interceptor_agents:
            return self._interceptor_observation_space
        raise ValueError(f"Unknown agent: {agent}")

    def _get_agent_max_velocity(self, agent: str) -> float:
        if agent in self.runner_agents:
            return self.runner_max_velocity

        if agent in self.interceptor_agents:
            return self.interceptor_max_velocity
        raise ValueError(f"Unknown agent: {agent}")

    # ------------------------------------------------------------------
    # Reset and curriculum learning
    # ------------------------------------------------------------------
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Update curriculum stage and generate a new reset layout before calling
        the base reset."""
        training = (options or {}).get("training", True)
        self._episode_count += 1

        # Update and apply the current curriculum stage.
        self._update_interceptor_curriculum(training)
        self._apply_curriculum_stage()

        self.reset_positions = self._generate_reset_positions()

        return super().reset(seed=seed, options=options)

    def _reset_task_state(self) -> None:
        """Draw the goal marker and reset per-episode task state. Called by the
        base reset after drones are moved to their reset positions."""
        self.caught = False
        self.reached_goal = False
        self.winner = None
        self._step_collision_outcomes = (False, False)

        runner_pos = self.drones[self.runner_agents[0]].get_position()
        self.previous_goal_distance = self._distance_3d(runner_pos, self.goal_position)

        self.previous_capture_distances = {
            interceptor: self._distance_3d(
                self.drones[interceptor].get_position(),
                runner_pos,
            )
            for interceptor in self.interceptor_agents
        }

        if self.use_simulator:
            self._set_target_marker(
                position=self.goal_position, marker_name="runner_goal"
            )

    def _update_interceptor_curriculum(
        self,
        training: bool = True,
    ) -> None:
        """Update the interceptor curriculum based on the current episode's performance."""
        if not self.curriculum_enabled or not training:
            return

        max_stage = len(self.CURRICULUM_STAGES) - 1
        if self.curriculum_stage >= max_stage:
            return

        if self._episode_count > 1:
            self._recent_runner_outcomes.append(
                1.0 if self.winner == self.RUNNER else 0.0
            )

        if len(self._recent_runner_outcomes) < self.curriculum_window:
            return

        success_rate = sum(self._recent_runner_outcomes) / len(
            self._recent_runner_outcomes
        )

        if success_rate >= self.curriculum_success_threshold:
            self.advance_curriculum()
            print(
                f"[MarlTag][curriculum] "
                f"runner success {success_rate:.0%} -> "
                f"stage {self.curriculum_stage}"
            )

    def _apply_curriculum_stage(self) -> None:
        """Apply the current curriculum stage to the interceptor agents."""
        if not self.curriculum_enabled:
            self.curriculum_interceptor_vel_factor = 1.0
            return

        stage = self.CURRICULUM_STAGES[self.curriculum_stage]

        speed_factor = stage["interceptor_speed_factor"]

        self.curriculum_interceptor_vel_factor = speed_factor

    def advance_curriculum(self) -> None:
        """Advance to the next curriculum stage."""
        max_stage = len(self.CURRICULUM_STAGES) - 1

        if self.curriculum_stage < max_stage:
            self.curriculum_stage += 1
            self._recent_runner_outcomes.clear()

    def set_curriculum_stage(self, stage: int) -> None:
        """Set curriculum stage directly."""
        max_stage = len(self.CURRICULUM_STAGES) - 1

        self.curriculum_stage = max(0, min(stage, max_stage))
        self._recent_runner_outcomes.clear()

    # ------------------------------------------------------------------
    # Reset layout and goal generation
    # ------------------------------------------------------------------
    def _generate_reset_positions(self) -> dict[str, list[float]]:
        """Generate a valid reset layout for the MarlTag task. Loops until a
        valid layout is found or the maximum number of attempts is reached."""
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

    def _sample_random_position(self) -> list[float]:
        """Sample a random position within the usable xy area and z limits."""
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
        """Check if a valid goal position can exist given the runner's position."""
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
    ) -> list[float]:
        """Sample a goal position that is at least `minimum_distance` away from the runner."""
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
        """Sample a valid interceptor spawn position that is at least `minimum_runner_distance`
        away from the runner and does not collide with other reset positions."""
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

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _apply_task_action_processing(
        self,
        agent: str,
        vx: float,
        vy: float,
        vz: float,
        current_position: list[float],
    ) -> tuple[float, float, float, dict[str, Any]]:
        """Apply task-specific processing to an agent's velocity command.
        Apply curriculum learning speed cap to the interceptor's commanded velocity.
        """
        requested = [vx, vy, vz]

        if agent in self.interceptor_agents and self.curriculum_enabled:
            vx *= self.curriculum_interceptor_vel_factor
            vy *= self.curriculum_interceptor_vel_factor
            vz *= self.curriculum_interceptor_vel_factor

        sent = [vx, vy, vz]

        info = {
            "requested_velocity": requested,
            "sent_velocity": sent,
        }

        return vx, vy, vz, info

    # ------------------------------------------------------------------
    # Observations and global state
    # ------------------------------------------------------------------
    def _get_runner_observations(self, agent: str) -> np.ndarray:
        """Runner sees its own state, other agents' relative position, other
        good agents' velocity and the goal's relative position."""
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
                    (other_agents_rel_pos, self._normalise_relative_pos(rel_pos))
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
                    (other_good_agents_vel, self._normalise_vel(other_vel, agent=other))
                )

        goal_pos = self.goal_position
        goal_rel = self._relative_position(goal_pos, own_pos)

        obs_parts = [
            self._normalise_vel(own_vel, agent),  # 3
            self._normalise_pos(own_pos),  # 3
            other_agents_rel_pos,  # 3 * (num_agents - 1)
            other_good_agents_vel,  # 3 * (num_runner_agents - 1)
            self._normalise_relative_pos(goal_rel),  # 3
        ]
        return np.concatenate(obs_parts).astype(np.float32)

    def _get_interceptor_observations(self, agent: str) -> np.ndarray:
        """Interceptor sees its own state, other agents' relative position,
        and the runners' velocity."""
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
                    (other_agents_rel_pos, self._normalise_relative_pos(rel_pos))
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
                (other_good_agents_vel, self._normalise_vel(other_vel, agent=other))
            )

        obs_parts = [
            self._normalise_vel(own_vel, agent),  # 3
            self._normalise_pos(own_pos),  # 3
            other_agents_rel_pos,  # 3 * (num_agents - 1)
            other_good_agents_vel,  # 3 * num_runner_agents
        ]
        return np.concatenate(obs_parts).astype(np.float32)

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {agent: self._get_agent_observation(agent) for agent in self.agents}

    def _get_agent_observation(self, agent: str) -> np.ndarray:
        if agent in self.runner_agents:
            return self._get_runner_observations(agent)

        if agent in self.interceptor_agents:
            return self._get_interceptor_observations(agent)

        raise ValueError(f"Unknown agent: {agent}")

    def _get_global_state(self) -> np.ndarray:
        observations = [
            self._get_agent_observation(agent) for agent in self.possible_agents
        ]
        return np.concatenate(observations).astype(np.float32)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _calculate_rewards(
        self, state_dicts: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate rewards for the MarlTag task.
        Shaping rewards are based on progress toward the goal for the runner and
        progress toward capture for the interceptors.
        Terminal rewards are given for catching the runner or reaching the goal.
        Boundary penalties are applied when agents go out of bounds.
        """
        goal_distance = self._runner_goal_distance(state_dicts)
        capture_distances = self._capture_distances(state_dicts)

        self._step_collision_outcomes = self._get_collision_safety_outcomes()
        capture_collision, non_capture_collision = self._step_collision_outcomes

        caught = capture_collision and not non_capture_collision
        reached = goal_distance < self.goal_threshold and not non_capture_collision

        # Runner shaping reward
        runner_reward = (
            self.previous_goal_distance - goal_distance
        ) * self.goal_progress_multiplier

        # Interceptor shaping reward
        interceptor_rewards: dict[str, float] = {}

        for interceptor in self.interceptor_agents:
            capture_distance = capture_distances[interceptor]

            progress = self.previous_capture_distances[interceptor] - capture_distance

            interceptor_rewards[interceptor] = (
                progress * self.capture_progress_multiplier
            )

        # Terminal rewards
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
            self.runner_agents[0]: float(runner_reward),
            **interceptor_rewards,
        }

        # Boundary penalties
        for agent in self.agents:
            position = state_dicts[agent]["position"]
            rewards[agent] += self._boundary_penalty(position)

        return rewards

    def _boundary_penalty(self, position: list[float]) -> float:
        """Return the individual boundary shaping penalty for one drone.
        Linear penalty is applied within the boundary margin, and exponential
        penalty is applied beyond `self.xy_limit`, `self.z_max` and `self.z_min`.
        Only applies penalty to the most severe boundary violation (x, y or z)."""

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
                np.exp(0.8 * normalised_overshoot)
            )

        # XY boundary penalty.
        xy_penalty_start = self.xy_limit - self.boundary_penalty_margin

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
            z_penalty_start = self.z_max - self.z_boundary_penalty_margin - z_mid
        else:
            z_distance = z_mid - position[2]
            hard_z_extent = z_mid - self.z_min
            z_penalty_start = z_mid - self.z_min - self.z_boundary_penalty_margin

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
        self,
        state_dicts: dict[str, dict[str, Any]],
    ) -> dict[str, bool]:
        goal_distance = self._runner_goal_distance(state_dicts)

        capture_collision, non_capture_collision = self._step_collision_outcomes
        if non_capture_collision:
            self.caught = False
            self.reached_goal = False
        else:
            self.caught = capture_collision
            self.reached_goal = goal_distance < self.goal_threshold

        terminal = self.caught or self.reached_goal

        if terminal and self.winner is None:
            # Capture takes priority if capture and goal occur on the same step.
            if self.caught:
                self.winner = self.INTERCEPTOR
                self.interceptor_success_count += 1

                if self._is_evaluating:
                    self.success_counts[self.INTERCEPTOR] = (
                        self.success_counts.get(self.INTERCEPTOR, 0) + 1
                    )

            elif self.reached_goal:
                self.winner = self.RUNNER
                self.runner_success_count += 1

                if self._is_evaluating:
                    self.success_counts[self.RUNNER] = (
                        self.success_counts.get(self.RUNNER, 0) + 1
                    )

        return {agent: terminal for agent in self.agents}

    def _check_truncations(
        self,
        state_dicts: dict[str, dict[str, Any]],
    ) -> dict[str, bool]:
        time_limit_reached = self.steps >= self.episode_length

        _, non_capture_collision = self._step_collision_outcomes
        if non_capture_collision:
            print("[MarlTag] non-capture collision — truncating episode")

        # TODO: Do we keep this or remove this so that episodes don't end early due to
        # low battery? It may be useful for training, but not for evaluation.
        any_low_battery = any(
            state_dicts[agent]["battery"] < self.battery_threshold
            for agent in self.agents
        )
        if any_low_battery:
            print("[MarlTag] low battery — truncating episode")

        z_max = self.z_max + 1 if self.use_simulator else self.z_max
        z_violation = [
            agent
            for agent in self.agents
            if not (self.z_min <= state_dicts[agent]["position"][2] <= z_max)
        ]
        if z_violation:
            print(f"[MarlTag] z-boundary violation {z_violation} — truncating episode")

        truncate_all = (
            time_limit_reached
            or any_low_battery
            or bool(z_violation)
            or non_capture_collision
        )

        return {agent: truncate_all for agent in self.agents}

    def _get_collision_safety_outcomes(
        self,
    ) -> tuple[bool, bool]:
        """Interpret collision-monitor pairs for MarlTag.

        Returns:
            capture_collision:
                True if a runner-interceptor pair triggered the monitor.

            non_capture_collision:
                True if any other pair triggered the monitor.
        """
        if not self._collision_safety_triggered():
            return False, False

        capture_collision = False
        non_capture_collision = False

        for agent_a, agent_b, _distance in self._get_collision_safety_pairs():

            runner_interceptor_pair = (
                agent_a in self.runner_agents and agent_b in self.interceptor_agents
            ) or (agent_b in self.runner_agents and agent_a in self.interceptor_agents)

            if runner_interceptor_pair:
                capture_collision = True
            else:
                non_capture_collision = True

        return capture_collision, non_capture_collision

    # ------------------------------------------------------------------
    # Task info and rendering
    # ------------------------------------------------------------------
    def _get_infos(
        self,
        state_dicts: dict[str, dict[str, Any]] | None = None,
        denormalised_actions: dict[str, list[float]] | None = None,
        normalised_actions: dict[str, np.ndarray] | None = None,
        old_positions: dict[str, list[float]] | None = None,
        new_positions: dict[str, list[float]] | None = None,
        action_filter_infos: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        capture_collision, non_capture_collision = self._step_collision_outcomes

        if state_dicts is None:
            positions = {
                agent: self.drones[agent].get_position() for agent in self.agents
            }
            state_dicts = self._generate_state_dicts(positions)

        goal_distance = self._runner_goal_distance(state_dicts)

        capture_distances = self._capture_distances(state_dicts)
        closest_interceptor_distance = min(capture_distances.values())

        infos: dict[str, dict[str, Any]] = {}
        for agent in self.agents:
            info: dict[str, Any] = {
                "role": "runner" if agent in self.runner_agents else "interceptor",
                "goal_position": self.goal_position[:],
                "distance_to_goal": goal_distance,
                "closest_interceptor_distance": closest_interceptor_distance,
                "caught": self.caught,
                "reached_goal": self.reached_goal,
                "winner": self.winner,
                "runner_success": int(self.winner == self.RUNNER),
                "interceptor_success": int(self.winner == self.INTERCEPTOR),
                "in_boundaries": state_dicts[agent]["in_boundaries"],
                "battery": state_dicts[agent]["battery"],
                "collision_safety_triggered": (self._collision_safety_triggered()),
                "collision_capture": capture_collision,
                "collision_safety_truncation": non_capture_collision,
            }
            if denormalised_actions is not None:
                info["denormalised_action"] = denormalised_actions.get(agent)
            if normalised_actions is not None:
                a = normalised_actions.get(agent)
                info["normalised_action"] = a.tolist() if a is not None else None
            if action_filter_infos is not None:
                info["action_filter_info"] = action_filter_infos.get(agent)
            if self._is_evaluating:
                info["success_counts"] = dict(self.success_counts)
            infos[agent] = info

        return infos

    def _render_task_specific_info(self) -> None:

        print(f"Goal:               {[round(v, 2) for v in self.goal_position]}")
        runner_pos = self.drones[self.runner_agents[0]].get_position()
        print(f"Runner: {[round(v, 2) for v in runner_pos]}")

        for interceptor in self.interceptor_agents:
            interceptor_pos = self.drones[interceptor].get_position()
            separation = self._distance_3d(
                runner_pos,
                interceptor_pos,
            )

            print(
                f"{interceptor}: {[round(v, 2) for v in interceptor_pos]} "
                f"| separation: {separation:.2f} "
                f"(capture {self.capture_threshold:.2f})"
            )
        print(
            f"Distance to goal:   {self._distance_3d(runner_pos, self.goal_position):.2f} "
            f"(threshold {self.goal_threshold:.2f})"
        )
        print(
            f"Reached goal: {self.reached_goal} | Caught: {self.caught} | Winner: {self.winner}"
        )

    # ------------------------------------------------------------------
    # Geometry and state helpers
    # ------------------------------------------------------------------
    def _distance_to_target(self, agent: str, position: list[float]) -> float:
        """Return the distance from a given position to an agent's target."""
        if agent in self.runner_agents:
            return self._distance_3d(position, self.goal_position)
        return self._distance_3d(
            position, self.drones[self.runner_agents[0]].get_position()
        )

    @staticmethod
    def _distance_3d(a: list[float], b: list[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _runner_goal_distance(self, state_dicts: dict[str, dict[str, Any]]) -> float:
        if self.runner_agents[0] in state_dicts:
            return state_dicts[self.runner_agents[0]]["distance_to_target"]
        return self._distance_3d(
            self.drones[self.runner_agents[0]].get_position(), self.goal_position
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
