import math
import time
import io
from collections import deque
from typing import Any, Literal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.markers import MarkerStyle

from drone_gym.drone_environment import DroneEnvironment
from drone_gym.agents.policies import PolicyState, PredictedInterceptPolicy


class SarlTag(DroneEnvironment):
    """
    Single-agent pursuit-evasion task with one RL runner and one or more expert
    interceptors. Modelled after MPE Simple Tag.

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
        exploration_steps: int = 1000,
        episode_length: int = 80,
        interceptor_max_velocity: float = 0.25,
        goal_threshold: float = 0.20,
        capture_threshold: float = 0.2,
    ):

        self.num_interceptor_agents = num_agents - 1
        self.interceptor_agents = [
            f"interceptor_{i}" for i in range(self.num_interceptor_agents)
        ]
        if use_simulator:
            boundaries = {"x": 10.0, "y": 10.0, "z_min": 0.1, "z_max": 10.0}
        else:
            boundaries = {"x": 2.5, "y": 2.5, "z_min": 0.1, "z_max": 3.0}

        super().__init__(
            use_simulator=use_simulator,
            max_velocity=max_velocity,
            step_time=step_time,
            expert_drone_names=self.interceptor_agents,
            boundaries=boundaries,
            collision_safety_distance=capture_threshold,
            max_velocity_z=max_velocity_z,
            xy_limit=xy_limit,
            z_min=z_min,
            z_max=z_max,
        )

        # RL training parameters
        self.episode_length = episode_length
        self.exploration_steps = exploration_steps

        # Reset geometry
        self.goal_min_distance_ratio = 0.60
        self.interceptor_min_distance_ratio = 0.25

        self.z_margin = 0.1

        self.max_layout_sampling_attempts = 100
        self.max_position_sampling_attempts = 300
        self.reset_positions: dict[str, list[float]] = {}

        # Interceptor speed curriculum
        self.interceptor_speed_max = interceptor_max_velocity
        self.interceptor_max_velocity_z = 0.25

        self.curriculum_enabled = True
        self.curriculum_stage = 0
        self.curriculum_window = 50
        self.curriculum_success_threshold = 0.6
        self.curriculum_interceptor_vel_factor = 1.0

        self._recent_runner_outcomes = deque(maxlen=self.curriculum_window)

        # Win conditions
        self.capture_threshold = capture_threshold
        self.goal_threshold = goal_threshold

        # Task geometry
        self.max_xy_range = self.xy_limit * 2
        self.max_z_range = self.z_max - self.z_min
        self.max_distance = math.sqrt(
            self.max_xy_range**2 + self.max_xy_range**2 + self.max_z_range**2
        )
        self.boundary = [self.xy_limit, self.xy_limit, self.z_min, self.z_max]

        # Runner observation:
        # [own vel, own pos, other agents rel pos, other good agents vel, goal rel pos]
        self.observation_space = 3 + 3 + (self.num_interceptor_agents) * 3 + (0) * 3 + 3

        # Reward parameters
        self.success_reward = 100.0
        self.capture_reward = 100.0

        self.goal_progress_multiplier = 100.0

        self.boundary_penalty_at_limit = -1.0
        self.boundary_penalty_margin = 0.2
        self.z_boundary_penalty_margin = 0.10
        self.boundary_penalty_cap = -10.0

        # Task state
        self.caught = False  # True when the interceptor caught the runner
        self.reached_goal = False  # True when the runner reached the goal
        self._step_collision_outcomes: tuple[bool, bool] = (False, False)
        self.winner: str | None = None

        self.goal_position: list[float] = [0.0, 0.0, self.reset_height]

        self.interceptor_policies: dict[str, PredictedInterceptPolicy] = {
            interceptor_name: PredictedInterceptPolicy(
                max_velocity=self.interceptor_speed_max,
                max_velocity_z=self.interceptor_max_velocity_z,
            )
            for interceptor_name in self.interceptor_agents
        }

        # Distance tracking for reward calculation
        self.previous_goal_distance = self.max_distance

        # Evaluation mode tracking — counts episodes the runner reached the goal
        self.successful_episodes_count = 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Reset and curriculum learning
    # ------------------------------------------------------------------
    def reset(
        self,
        training: bool = True,
    ):
        """Update curriculum stage and generate a new reset layout before calling
        the base reset."""

        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        self._episode_count += 1

        self._update_interceptor_curriculum(training)
        self._apply_curriculum_stage()

        self.reset_positions = self._generate_reset_positions()

        return super().reset(training)

    def _reset_task_state(self):
        """Reset task-specific state variables (called from base reset)."""
        self.caught = False
        self.reached_goal = False
        self.winner = None

        runner_pos = self.rl_drone.get_position()
        self.previous_goal_distance = self._distance_to_target(runner_pos)

        if self.use_simulator:
            self._set_target_marker(
                position=self.goal_position, marker_name="runner_goal"
            )

    def _update_interceptor_curriculum(
        self,
        training: bool = True,
    ) -> None:
        """Update the interceptor curriculum based on runner performance."""
        if not self.curriculum_enabled or not training:
            return

        max_stage = len(self.CURRICULUM_STAGES) - 1
        if self.curriculum_stage >= max_stage:
            return

        if self._episode_count > 1:
            self._recent_runner_outcomes.append(1.0 if self.reached_goal else 0.0)

        if len(self._recent_runner_outcomes) < self.curriculum_window:
            return

        success_rate = sum(self._recent_runner_outcomes) / len(
            self._recent_runner_outcomes
        )

        if success_rate >= self.curriculum_success_threshold:
            self.advance_curriculum()
            print(
                f"[SarlTag][curriculum] "
                f"runner success {success_rate:.0%} -> "
                f"stage {self.curriculum_stage}"
            )

    def _apply_curriculum_stage(self) -> None:
        """Apply the current curriculum stage to all expert policies."""
        if not self.curriculum_enabled:
            self.curriculum_interceptor_vel_factor = 1.0
            return

        stage = self.CURRICULUM_STAGES[self.curriculum_stage]

        self.curriculum_interceptor_vel_factor = stage["interceptor_speed_factor"]

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
        """Loops until a valid runner, goal, and interceptor layout is found
        or the maximum number of attempts is reached."""
        usable_xy_size = 2.0 * self.reset_planner.usable_xy_limit

        goal_min_distance = self.goal_min_distance_ratio * usable_xy_size

        interceptor_runner_min_distance = max(
            self.interceptor_min_distance_ratio * usable_xy_size,
            self.reset_planner.slot_clearance,
        )

        for _ in range(self.max_layout_sampling_attempts):
            # 1. Sample runner.
            runner_position = self._sample_random_position()

            # 2. Reject runner positions from which no valid goal can exist.
            if not self._runner_has_valid_goal_region(
                runner_position,
                goal_min_distance,
            ):
                continue

            # 3. Sample goal.
            goal_position = self._sample_goal(
                runner_position,
                goal_min_distance,
            )

            reset_positions = {
                self.RL_DRONE_NAME: runner_position,
            }

            interceptor_positions: list[list[float]] = []
            layout_failed = False

            # 4. Sample interceptors.
            for interceptor_name in self.interceptor_agents:
                interceptor_position = self._sample_interceptor_spawn(
                    runner_position=runner_position,
                    reset_positions=reset_positions,
                    minimum_runner_distance=interceptor_runner_min_distance,
                )

                if interceptor_position is None:
                    layout_failed = True
                    break

                interceptor_positions.append(interceptor_position)

                # Include it immediately so later interceptor samples
                # must keep clear of this position.
                reset_positions[interceptor_name] = interceptor_position

            if layout_failed:
                continue

            # Avoid interceptor identity being correlated with sampling order.
            np.random.shuffle(interceptor_positions)

            for interceptor_name, position in zip(
                self.interceptor_agents,
                interceptor_positions,
            ):
                reset_positions[interceptor_name] = position

            # 5. Final physical safety validation.
            try:
                self.reset_planner.validate_reset_positions(reset_positions)
            except self.reset_planner.ConfigurationError:
                continue

            self.goal_position = goal_position

            return reset_positions

        raise RuntimeError(
            "Unable to generate a valid SarlTag reset layout after "
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
        """Check whether a valid goal can exist for this runner position."""
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
        """Sample a goal sufficiently far from the runner in xy."""
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
        """Sample a valid interceptor reset position."""
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
        """Apply the interceptor curriculum speed cap."""
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
        return (vx, vy, vz, info)

    def _get_velocity_commands(
        self,
        action,
    ) -> dict[str, list[float]]:
        """Generate runner and expert interceptor velocity commands."""
        velocity_commands = super()._get_velocity_commands(action)

        runner_pos = self.rl_drone.get_position()

        runner_velocity = self.rl_drone.get_calculated_velocity()
        runner_vel = [
            float(runner_velocity.get("x", 0.0)),
            float(runner_velocity.get("y", 0.0)),
            float(runner_velocity.get("z", 0.0)),
        ]
        interceptor_speed = (
            self.interceptor_speed_max * self.curriculum_interceptor_vel_factor
        )
        context = {
            "target_position": runner_pos,
            "target_velocity": runner_vel,
            "pursuer_speed": interceptor_speed,
        }

        for interceptor_name, policy in self.interceptor_policies.items():
            interceptor_drone = self.expert_drones[interceptor_name]

            velocity_commands[interceptor_name] = policy.compute(
                state=PolicyState(
                    position=interceptor_drone.get_position(),
                ),
                context=context,
            )

        return velocity_commands

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_runner_observations(self) -> np.ndarray:
        """Runner sees its own state, other agents' relative positions,
        and the goal's relative position."""

        runner = self.rl_drones[self.RL_DRONE_NAME]

        vel = runner.get_calculated_velocity()
        own_vel = [
            float(vel.get("x", 0.0)),
            float(vel.get("y", 0.0)),
            float(vel.get("z", 0.0)),
        ]
        own_pos = runner.get_position()

        # Relative position of every expert drone
        other_agents_rel_pos = np.array([], dtype=np.float32)

        for interceptor_name in self.interceptor_agents:
            other_drone = self.expert_drones[interceptor_name]
            other_pos = other_drone.get_position()
            rel_pos = self._relative_position(other_pos, own_pos)

            other_agents_rel_pos = np.concatenate(
                (
                    other_agents_rel_pos,
                    self._normalise_relative_pos(rel_pos),
                )
            )

        # Goal relative to runner
        goal_rel = self._relative_position(self.goal_position, own_pos)

        obs_parts = [
            self._normalise_vel(own_vel),  # 3
            self._normalise_pos(own_pos),  # 3
            other_agents_rel_pos,  # 3 * num_expert_drones
            self._normalise_relative_pos(goal_rel),  # 3
        ]

        return np.concatenate(obs_parts).astype(np.float32)

    def _get_state(self) -> np.ndarray:
        return self._get_runner_observations()

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _calculate_reward(
        self,
        current_state: dict[str, Any],
    ) -> float:
        """Reward consists of progress towards the goal, terminal rewards for
        reaching the goal or being captured, and boundary shaping."""
        position = current_state["position"]
        goal_distance = current_state["distance_to_target"]

        self._step_collision_outcomes = self._get_collision_safety_outcomes()
        capture_collision, non_capture_collision = self._step_collision_outcomes

        caught = capture_collision and not non_capture_collision
        reached_goal = goal_distance < self.goal_threshold and not non_capture_collision

        # Goal-progress shaping.
        reward = (
            self.previous_goal_distance - goal_distance
        ) * self.goal_progress_multiplier

        # Terminal rewards.
        # Capture takes priority if capture and goal occur on the same step.
        if caught:
            reward -= self.capture_reward

        elif reached_goal:
            reward += self.success_reward

        self.previous_goal_distance = goal_distance

        # Boundary shaping.
        reward += self._boundary_penalty(position)

        return float(reward)

    def _boundary_penalty(
        self,
        position: list[float],
    ) -> float:
        """The penalty increases linearly near the task boundary and
        exponentially after crossing it. Only the most severe axis
        contributes."""

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

            # Linear increase approaching the boundary.
            if distance_from_centre <= hard_limit:
                return (distance_from_centre - penalty_start) / penalty_width

            # Exponential increase outside the task boundary.
            overshoot = distance_from_centre - hard_limit
            normalised_overshoot = overshoot / penalty_width

            return float(np.exp(0.8 * normalised_overshoot))

        # XY boundary risk.
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

        # Z boundary risk.
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

        boundary_risk = max(
            x_risk,
            y_risk,
            z_risk,
        )

        boundary_penalty = max(
            self.boundary_penalty_cap, self.boundary_penalty_at_limit * boundary_risk
        )
        return boundary_penalty

    # ------------------------------------------------------------------
    # Terminations / truncations
    # ------------------------------------------------------------------
    def _check_if_terminated(self, current_state: dict[str, Any]) -> bool:
        """Episode ends on goal reached (success) or interception."""
        goal_distance = current_state["distance_to_target"]

        capture_collision, non_capture_collision = self._step_collision_outcomes

        if non_capture_collision:
            self.caught = False
            self.reached_goal = False
        else:
            self.caught = capture_collision
            self.reached_goal = goal_distance < self.goal_threshold

        terminal = self.caught or self.reached_goal

        if terminal and self.winner is None:
            if self.caught:
                self.winner = self.INTERCEPTOR
            elif self.reached_goal:
                self.winner = self.RUNNER
                if self._is_evaluating:
                    self.successful_episodes_count += 1

        return terminal

    def _check_if_truncated(
        self,
        current_state: dict[str, Any],
    ) -> bool:
        """Truncate the episode on time limit, or an invalid drone altitude."""
        time_limit_reached = self.steps >= self.episode_length

        _, non_capture_collision = self._step_collision_outcomes
        if non_capture_collision:
            print("[SarlTag] Non-capture collision detected. - Truncating episode.")

        z_max = self.z_max + 1 if self.use_simulator else self.z_max
        z_violation_drones = []
        for drone_name, drone in self._iter_drones():
            if drone_name == self.RL_DRONE_NAME:
                position = current_state["position"]
            else:
                position = drone.get_position()

            if not self.z_min <= position[2] <= z_max:
                z_violation_drones.append(drone_name)

        if z_violation_drones:
            print(
                "[SarlTag] Z-boundary violation "
                f"{z_violation_drones} — truncating episode."
            )

        truncate = (
            time_limit_reached or bool(z_violation_drones) or non_capture_collision
        )

        return truncate

    def _get_collision_safety_outcomes(
        self,
    ) -> tuple[bool, bool]:
        if not self._collision_safety_triggered():
            return False, False

        capture_collision = False
        non_capture_collision = False

        for drone_a, drone_b, _distance in self._get_collision_safety_pairs():
            runner_interceptor_pair = (
                drone_a == self.RL_DRONE_NAME and drone_b in self.interceptor_agents
            ) or (drone_b == self.RL_DRONE_NAME and drone_a in self.interceptor_agents)

            if runner_interceptor_pair:
                capture_collision = True
            else:
                non_capture_collision = True

        return capture_collision, non_capture_collision

    # ------------------------------------------------------------------
    # Task info and rendering
    # ------------------------------------------------------------------
    def _get_additional_info(self, current_state: dict[str, Any]) -> dict[str, Any]:
        position = current_state["position"]
        interceptor_positions = {
            name: drone.get_position() for name, drone in self.expert_drones.items()
        }

        interceptor_distances = self._get_interceptor_distances(position)
        info = {
            "goal_position": self.goal_position[:],
            "interceptor_positions": interceptor_positions,
            "distance_to_goal": self._distance_to_target(position),
            "closest_interceptor_distance": min(interceptor_distances.values()),
            "interceptor_distances": interceptor_distances,
            "caught": self.caught,
            "reached_goal": self.reached_goal,
            "success": self.reached_goal,
            "runner_success": int(self.reached_goal),
            "interceptor_success": int(self.caught),
            "in_boundaries": current_state["in_boundaries"],
            "sim_full_restart_count": self.sim_full_restart_count,
            "description": "3D navigate-to-goal under interception — RL runner vs expert interceptor",
        }
        if self._is_evaluating:
            info["success_count"] = self.successful_episodes_count
        return info

    def _render_task_specific_info(self):
        pos = self.rl_drone.get_position()
        d_goal = self._distance_to_target(pos)

        interceptor_distances = self._get_interceptor_distances(pos)

        print(f"Runner Position: [{pos[0]:.2f}, " f"{pos[1]:.2f}, {pos[2]:.2f}]")
        print(
            f"Goal Position:   [{self.goal_position[0]:.2f}, "
            f"{self.goal_position[1]:.2f}, "
            f"{self.goal_position[2]:.2f}]"
        )

        for interceptor_name in self.interceptor_agents:
            interceptor_position = self.expert_drones[interceptor_name].get_position()

            print(
                f"{interceptor_name} Position: "
                f"[{interceptor_position[0]:.2f}, "
                f"{interceptor_position[1]:.2f}, "
                f"{interceptor_position[2]:.2f}] "
                f"(distance {interceptor_distances[interceptor_name]:.2f})"
            )

        print(
            f"Distance to Goal: {d_goal:.2f} " f"(threshold {self.goal_threshold:.2f})"
        )
        print(
            f"Closest Interceptor: "
            f"{min(interceptor_distances.values()):.2f} "
            f"(capture {self.capture_threshold:.2f})"
        )
        print(
            f"Reached goal: {self.reached_goal} | "
            f"Caught: {self.caught} | Winner: {self.winner}"
        )

    def get_overlay_info(self) -> dict[str, Any]:
        position = self.drone.get_position()
        interceptor_positions = {
            name: drone.get_position() for name, drone in self.expert_drones.items()
        }

        interceptor_distances = self._get_interceptor_distances(position)
        return {
            "position": position,
            "goal_position": self.goal_position[:],
            "interceptor_positions": interceptor_positions,
            "interceptor_distances": interceptor_distances,
            "distance_to_goal": self._distance_to_target(position),
            "distance_to_interceptor": min(interceptor_distances.values()),
            "caught": self.caught,
            "reached_goal": self.reached_goal,
        }

    # ------------------------------------------------------------------
    # Geometry and state helpers
    # ------------------------------------------------------------------
    def _distance_to_target(self, position: list[float]) -> float:
        """Base hook: 'target' for this task is the GOAL (used by the info dict)."""
        return math.sqrt(
            (position[0] - self.goal_position[0]) ** 2
            + (position[1] - self.goal_position[1]) ** 2
            + (position[2] - self.goal_position[2]) ** 2
        )

    def _get_interceptor_distances(
        self,
        runner_position: list[float],
    ) -> dict[str, float]:
        distances: dict[str, float] = {}

        for interceptor_name, interceptor_drone in self.expert_drones.items():
            interceptor_position = interceptor_drone.get_position()

            distances[interceptor_name] = math.sqrt(
                (runner_position[0] - interceptor_position[0]) ** 2
                + (runner_position[1] - interceptor_position[1]) ** 2
                + (runner_position[2] - interceptor_position[2]) ** 2
            )

        return distances

    def _distance_to_closest_interceptor(
        self,
        runner_position: list[float],
    ) -> float:
        return min(self._get_interceptor_distances(runner_position).values())

    @property
    def max_action_value(self):
        return 1.0

    @property
    def min_action_value(self):
        return -1.0

    def sample_action(self):
        """Sample a normalized action in [-1, 1]"""
        return np.random.uniform(-1.0, 1.0, size=(3,))

    def grab_frame(self, height: int = 540, width: int = 960) -> np.ndarray:
        fig = plt.figure(figsize=(width / 120, height / 120), dpi=120)

        if not self.episode_positions:
            plt.close(fig)
            return np.full((height, width, 3), 255, dtype=np.uint8)

        pos_array = np.array(self.episode_positions)
        x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]

        from matplotlib.gridspec import GridSpec

        gs = GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])

        gx, gy, gz = self.goal_position

        interceptor_positions = {
            interceptor_name: self.expert_drones[interceptor_name].get_position()
            for interceptor_name in self.interceptor_agents
        }

        # ------------------------------------------------------------------
        # LEFT: 3D trajectory
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0], projection="3d")

        ax1.plot(
            x,
            y,
            z,
            label="Runner Path",
            color="yellow",
            linewidth=2.5,
        )

        ax1.scatter(
            x[0],
            y[0],
            z[0],
            color="green",
            s=80,
            label="Start",
            depthshade=False,
            edgecolors="black",
            linewidth=0.5,
        )

        ax1.scatter(
            x[-1],
            y[-1],
            z[-1],
            color="blue",
            s=80,
            label="Current",
            depthshade=False,
            edgecolors="black",
            linewidth=0.5,
        )

        ax1.scatter(
            gx,
            gy,
            gz,
            color="lime",
            marker="*",
            s=160,
            label="Goal",
            depthshade=False,
            edgecolors="black",
            linewidth=1,
        )

        for i, interceptor_position in enumerate(interceptor_positions.values()):
            ix, iy, iz = interceptor_position

            ax1.scatter(
                ix,
                iy,
                iz,
                color="red",
                marker="^",
                s=120,
                label="Interceptor" if i == 0 else None,
                depthshade=False,
                edgecolors="black",
                linewidth=1,
            )

        ax1.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_zlim(self.z_min - 0.1, self.z_max + 0.1)

        ax1.set_xlabel("X (m)", fontsize=10, labelpad=8)
        ax1.set_ylabel("Y (m)", fontsize=10, labelpad=8)
        ax1.set_zlabel("Z (m)", fontsize=9, labelpad=10)

        ax1.tick_params(axis="x", labelsize=8)
        ax1.tick_params(axis="y", labelsize=8)
        ax1.tick_params(axis="z", labelsize=8)

        ax1.view_init(elev=10, azim=25)
        ax1.set_title("3D Trajectory", fontsize=12, pad=15)

        ax1.legend(
            loc="upper left",
            fontsize=6,
            framealpha=0.9,
            markerscale=0.60,
        )

        ax1.grid(True, alpha=0.3)
        ax1.set_box_aspect([1, 1, 0.67])

        # ------------------------------------------------------------------
        # RIGHT: top-down X-Y
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])

        boundary_x = [
            -self.xy_limit,
            self.xy_limit,
            self.xy_limit,
            -self.xy_limit,
            -self.xy_limit,
        ]

        boundary_y = [
            -self.xy_limit,
            -self.xy_limit,
            self.xy_limit,
            self.xy_limit,
            -self.xy_limit,
        ]

        ax2.plot(
            boundary_x,
            boundary_y,
            "k--",
            linewidth=1,
            alpha=0.5,
            label="Boundary",
            zorder=1,
        )

        ax2.plot(
            x,
            y,
            color="yellow",
            linewidth=2.5,
            label="Runner Path",
            zorder=2,
        )

        ax2.scatter(
            x[0],
            y[0],
            color="green",
            s=80,
            label="Start",
            edgecolors="black",
            linewidth=0.5,
            zorder=4,
        )

        ax2.scatter(
            x[-1],
            y[-1],
            color="blue",
            s=80,
            label="Current",
            edgecolors="black",
            linewidth=0.5,
            zorder=4,
        )

        ax2.scatter(
            gx,
            gy,
            color="lime",
            marker=MarkerStyle("*"),
            s=160,
            label="Goal",
            edgecolors="black",
            linewidth=1,
            zorder=5,
        )

        for i, interceptor_position in enumerate(interceptor_positions.values()):
            ix, iy, _ = interceptor_position

            ax2.scatter(
                ix,
                iy,
                color="red",
                marker=MarkerStyle("^"),
                s=120,
                label="Interceptor" if i == 0 else None,
                edgecolors="black",
                linewidth=1,
                zorder=5,
            )

            ax2.add_patch(
                plt.Circle(
                    (ix, iy),
                    self.capture_threshold,
                    color="red",
                    alpha=0.15,
                    zorder=1,
                )
            )

        ax2.add_patch(
            plt.Circle(
                (gx, gy),
                self.goal_threshold,
                color="lime",
                alpha=0.18,
                zorder=1,
            )
        )

        ax2.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)

        ax2.set_xlabel("X (m)", fontsize=10)
        ax2.set_ylabel("Y (m)", fontsize=10)
        ax2.set_title("Top-Down View (X-Y)", fontsize=12, pad=15)

        ax2.set_aspect("equal", adjustable="box")

        ax2.legend(
            loc="upper left",
            fontsize=6,
            framealpha=0.9,
            markerscale=0.60,
        )

        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="both", labelsize=8)

        # ------------------------------------------------------------------
        # Episode outcome
        # ------------------------------------------------------------------
        outcome = (
            "Reached Goal"
            if self.reached_goal
            else ("Caught" if self.caught else "In Progress")
        )

        fig.suptitle(
            f"SARL Tag (Step {self.steps}) | {outcome}",
            fontsize=13,
            y=0.98,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # ------------------------------------------------------------------
        # Convert matplotlib figure to RGB numpy frame
        # ------------------------------------------------------------------
        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="png",
            dpi=120,
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
        )

        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        buf.close()
        plt.close(fig)

        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            current_h, current_w = frame.shape[:2]

            if current_h != height or current_w != width:
                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        else:
            frame = np.full(
                (height, width, 3),
                255,
                dtype=np.uint8,
            )

        return frame
