import math
import time
import io
import threading
from collections import deque
from typing import Dict, List, Any, Literal
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.markers import MarkerStyle

from drone_gym.drone_environment import DroneEnvironment
from drone_gym.agents.bodies import CrazyflieBody
from drone_gym.agents.policies import CallablePolicy
from drone_gym.agents.sim_agent import SimAgent


class SarlTag(DroneEnvironment):
    """3D navigate-to-goal-under-interception task (Variant A: expert interceptor).

    The learner is the **runner** (Drone 1): it spawns at the centre and must fly
    to a randomly designated **goal** some distance away in 3D, *while evading a
    second drone that is actively trying to intercept it*. The runner therefore
    has to balance two objectives — reach the goal AND avoid the interceptor.

    The **interceptor** (Drone 2) is a *second real SITL Crazyflie*, brought up
    through the shared :class:`SimManager` as a ``crazyflie_pursuer`` agent. In
    this variant its brain is an expert 3D pure-pursuit policy (supplied via the
    manager's ``callable`` policy seam, because the built-in PurePursuitPolicy is
    xy-only); in the MARL variant the same seam takes a learned policy instead, so
    only that one line changes. The interceptor flies faster than the runner so
    that interception is genuinely feasible.

    An "interception" is a 0.20 m proximity event (3D), never a real drone-on-drone
    impact: a high-rate background guard stops both drones the instant they are
    within that distance, so the task is collision-safe in sim and the real arena.

    Episode outcomes:
      * success  — runner reaches within ``goal_threshold`` of the goal.
      * failure  — interceptor gets within ``capture_threshold`` of the runner,
                   or the runner leaves the boundary.
      * truncated — ``episode_length`` steps elapse with neither.

    Launch with the *multi*-agent SITL so both ports exist, e.g.::

        ./sitl_multiagent_square.sh -m crazyflie -n 2   # 19850 (runner), 19851 (interceptor)
    """

    INTERCEPTOR_NAME = "interceptor_0"

    CURRICULUM_STAGES = [
        {"interceptor_speed_factor": 0.40},
        {"interceptor_speed_factor": 0.60},
        {"interceptor_speed_factor": 0.80},
        {"interceptor_speed_factor": 1.00},
    ]

    def __init__(
        self,
        use_simulator: Literal[0, 1],
        max_velocity: float = 0.25,
        step_time: float = 0.5,
        exploration_steps: int = 1000,
        episode_length: int = 80,
        interceptor_max_velocity: float = 0.125,
        capture_threshold: float = 0.2,
    ):
        super().__init__(
            use_simulator=use_simulator,
            max_velocity=max_velocity,
            step_time=step_time,
            expert_drone_names=[self.INTERCEPTOR_NAME],
            collision_safety_distance=capture_threshold,
        )

        # Gentle vertical speed cap — CrazySim's z-velocity control is twitchy and
        # moving up/down fast destabilises the estimator, which makes the firmware
        # command a thrust spike that LAUNCHES the drone to the ceiling (a crash we
        # can't stop from here, since it bypasses our velocity setpoint). Keeping
        # vertical motion very slow keeps the vertical estimator well-conditioned so
        # that spike almost never builds — the task stays 3D but much more stable.
        self.max_velocity_z = 0.03

        # RL training parameters
        self.episode_length = episode_length
        self.exploration_steps = exploration_steps
        self.total_steps = 0
        self.truncate_next = False
        self.learning = True

        # --- Geometry / task parameters -------------------------------------
        # 2.0 is the value the stable sibling tasks use — larger boxes mean longer
        # position-control moves on reset, which is exactly what blows up the EKF.
        self.xy_limit = 2.0
        # z-band around the 1.0 m reset height. Vertical stability comes from the
        # max_velocity_z cap above (slow climbs keep CrazySim's vertical estimator
        # well-conditioned), so the band can be wide enough for real 3D variety —
        # it still ends well short of the containment lines (0.25 / 1.8) and the
        # firmware kill boundary (|z| 2.25).
        self.z_min = 0.6
        self.z_max = 1.4
        self.fixed_z = 1.0  # centre of the z band (default altitude)
        # Runner spawn altitude — resampled every episode. Kept a notch inside the
        # z band so the worst-case vertical gap to a goal (~0.5 m) stays closable
        # at max_velocity_z within an 80-step episode (0.5 / (0.03 * 0.5) ≈ 34 steps).
        self.out_of_bounds_tolerance = 0.05  # small grace for PID overshoot at the wall

        self.interceptor_max_velocity = (
            interceptor_max_velocity  # > max_velocity so capture is feasible
        )
        self.interceptor_max_velocity_z = (
            0.030  # gentle vertical cap for the pursuer too
        )

        # Reset geometry
        self.goal_min_distance_ratio = 0.60
        self.interceptor_min_distance_ratio = 0.25

        self.z_margin = 0.1

        self.max_layout_sampling_attempts = 100
        self.max_position_sampling_attempts = 300
        self.reset_positions: dict[str, list[float]] = {}

        self.num_interceptor_agents = 1  # only one pursuer in this variant (the expert)

        # --- Interceptor speed curriculum ---------------------------------------
        self.interceptor_speed_max = interceptor_max_velocity

        self.curriculum_enabled = True
        self.curriculum_stage = 0
        self.curriculum_window = 50
        self.curriculum_success_threshold = 0.6

        self._recent_runner_outcomes = deque(maxlen=self.curriculum_window)

        self._apply_curriculum_stage()

        self.capture_threshold = capture_threshold
        self.goal_threshold = 0.20  # metres (3D) — runner has reached the goal

        self.max_xy_range = self.xy_limit * 2
        self.max_z_range = self.z_max - self.z_min
        self.max_distance = math.sqrt(
            self.max_xy_range**2 + self.max_xy_range**2 + self.max_z_range**2
        )

        # boundary = (xy, xy, z_low, z_high) — used by the base visual boundary + step clamp
        self.boundary = [self.xy_limit, self.xy_limit, self.z_min, self.z_max]

        # Runner observation:
        # [own vel, own pos, other agents rel pos, other good agents vel, goal rel pos]
        self.observation_space = 3 + 3 + (self.num_interceptor_agents) * 3 + (0) * 3 + 3

        # --- Reward parameters ----------------------------------------------
        self.success_reward = 100.0  # reached the goal — clearly the best outcome
        self.intercepted_penalty = (
            -100.0
        )  # caught by the interceptor — clearly the worst
        self.out_of_bounds_penalty = -100.0
        self.goal_progress_multiplier = (
            100.0  # main drive: reward closing the gap to the goal
        )
        self.step_penalty = 1.0  # small per-step cost — reach the goal FAST
        self.danger_radius = (
            0.6  # within this of the interceptor, apply evasion shaping
        )
        self.danger_penalty = 5.0  # max shaping penalty at zero separation

        # --- Task state ------------------------------------------------------
        self.done = False
        self.caught = False  # True when the interceptor caught the runner
        self.reached_goal = False  # True when the runner reached the goal

        self.goal_position: List[float] = [0.0, 0.0, self.fixed_z]
        self.interceptor_position: List[float] = [0.0, 0.0, self.fixed_z]
        self.interceptor_velocity: List[float] = [0.0, 0.0, 0.0]

        # --- Interceptor agent (second real SITL Crazyflie) ------------------
        # Constructed directly here — agent lifecycle belongs to the environment,
        # not to SimManager.  SimManager is only responsible for Gazebo visuals.
        # self.sim_manager = get_default_sim_manager()
        self.goal_marker_name = "rl_sarl_tag_goal"
        # # Runner is on port 19850; interceptor is drone 2 from sitl_multiagent_square -n 2
        # interceptor_uri = "udp://0.0.0.0:19851"
        # interceptor_body = CrazyflieBody(
        #     use_simulator=use_simulator,
        #     uri=interceptor_uri,
        #     fixed_z=self.fixed_z,
        # )

        interceptor_drone = self.expert_drones[self.INTERCEPTOR_NAME]
        interceptor_body = CrazyflieBody(
            drone=interceptor_drone,
            fixed_z=self.fixed_z,
        )

        interceptor_policy = CallablePolicy(fn=self._interceptor_pursuit)
        self.interceptor = SimAgent(
            agent_id=1,
            body=interceptor_body,
            policy=interceptor_policy,
            role="interceptor",
        )

        # The interceptor repositions via a position-control move to a fresh spawn
        # EVERY episode, which stresses its EKF. The drone's internal safety monitor
        # hard-kills (emergency land + disarm) any drone whose |z| > 2.25 — a death
        # the interceptor can't recover from cleanly. Give its internal boundary
        # VERTICAL headroom only, so a transient EKF z-overshoot during
        # re-convergence doesn't trip the destructive kill. Keep xy at the drone
        # default (2.5, i.e. 0.5 m past the arena wall for PID overshoot) so a
        # lateral drift is still caught before the interceptor roams far outside
        # the arena. The task's own out-of-bounds + collision-guard logic
        # (xy_limit=2.0, z_max=1.4, capture_threshold) still governs episodes.
        # boundaries uses the post-#28 z_min/z_max schema (the boundary monitor now
        # checks z_min <= z <= z_max, not abs(z) <= z). A bare "z" key here would
        # KeyError in the interceptor's boundary thread.
        self._configure_interceptor_drone()

        # Distance tracking for reward calculation
        self.previous_goal_distance = self.max_distance

        # Evaluation mode tracking — counts episodes the runner reached the goal
        self.successful_episodes_count = 0

        # Episode counter (used by reset-time health checks and logging). EKF
        # drift no longer needs an every-N-episodes cap: the teleport reset
        # lands both drones and re-seeds their estimators on the ground EVERY
        # episode, so drift can never accumulate past a single episode.
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Interceptor expert policy — 3D Proportional Navigation (PIP)
    # ------------------------------------------------------------------

    def _interceptor_pursuit(self, state, context) -> List[float]:
        """3D Proportional Navigation via Predicted Intercept Point (PIP).

        Pure pursuit always steers toward the evader's *current* position,
        causing a tail-chase that converges slowly. Proportional Navigation
        (PN) instead drives the line-of-sight angular rate to zero, placing
        the pursuer on a collision course. For a constant-velocity evader
        this is equivalent to steering toward the *Predicted Intercept Point*
        (PIP): where pursuer and evader can arrive simultaneously given the
        evader's current velocity [1, 2].

        The PIP is solved by fixed-point iteration (2–4 steps suffice):
            t_go^(0) = |r| / V_pursuer
            pip^(k)  = runner_pos + runner_vel * t_go^(k)
            t_go^(k+1) = |pip^(k) − pursuer_pos| / V_pursuer

        References:
          [1] Shneydor, N. A. (1998). Missile Guidance and Pursuit, Ch. 4.
          [2] Weintraub, I., Pachter, M., & Garcia, E. (2020). An introduction
              to pursuit-evasion differential games. Proc. American Control
              Conference, pp. 1049–1066.
          [3] Nahin, P. J. (2012). Chases and Escapes, Ch. 3. Princeton UP.
        """
        pos = np.array(state.position, dtype=float)

        # Clamp the evader's position into the arena: a boundary-escaped runner
        # must never pull the aim-point outside it.
        target_pos = np.array(context["runner_pos"], dtype=float)
        runner_vel = np.array(context.get("runner_vel", [0.0, 0.0, 0.0]), dtype=float)

        # Iterative solve for the predicted intercept point.
        pip = target_pos.copy()
        for _ in range(4):
            d = float(np.linalg.norm(pip - pos))
            if d < 1e-6:
                break
            t_go = d / self.interceptor_max_velocity
            pip = target_pos + runner_vel * t_go

        aim = pip - pos
        aim_dist = float(np.linalg.norm(aim))
        if aim_dist < 1e-6:
            return [0.0, 0.0, 0.0]

        scale = self.interceptor_max_velocity / aim_dist
        vx, vy, vz = scale * aim[0], scale * aim[1], scale * aim[2]
        vz = float(
            np.clip(
                vz, -self.interceptor_max_velocity_z, self.interceptor_max_velocity_z
            )
        )

        return [vx, vy, vz]

    # ------------------------------------------------------------------
    # Geometry sampling (3D)
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

            # 4. Sample interceptor.
            interceptor_position = self._sample_interceptor_spawn(
                runner_position=runner_position,
                reset_positions=reset_positions,
                minimum_runner_distance=interceptor_runner_min_distance,
            )

            if interceptor_position is None:
                continue

            reset_positions[self.INTERCEPTOR_NAME] = interceptor_position

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
    # Distances (3D)
    # ------------------------------------------------------------------

    def _distance_to_target(self, position: List[float]) -> float:
        """Base hook: 'target' for this task is the GOAL (used by the info dict)."""
        return math.sqrt(
            (position[0] - self.goal_position[0]) ** 2
            + (position[1] - self.goal_position[1]) ** 2
            + (position[2] - self.goal_position[2]) ** 2
        )

    def _distance_to_interceptor(self, position: List[float]) -> float:
        return math.sqrt(
            (position[0] - self.interceptor_position[0]) ** 2
            + (position[1] - self.interceptor_position[1]) ** 2
            + (position[2] - self.interceptor_position[2]) ** 2
        )

    def _is_out_of_task_bounds(self, position: List[float]) -> bool:
        """Out of the task's 3D boundary (with a small grace for PID overshoot).

        We check the task limits explicitly rather than trusting the drone's own
        in_boundaries flag, which is computed against a different internal limit.
        """
        tol = self.out_of_bounds_tolerance
        return (
            abs(position[0]) > self.xy_limit + tol
            or abs(position[1]) > self.xy_limit + tol
            or position[2] < self.z_min - tol
            or position[2] > self.z_max + tol
        )

    # ------------------------------------------------------------------
    # Interceptor expert control / state tracking
    # ------------------------------------------------------------------

    def _sync_interceptor(self):
        """Copy the interceptor agent's position/velocity into local tracking."""
        self.interceptor_position = list(self.interceptor.position)
        self.interceptor_velocity = list(self.interceptor.velocity)

    def _command_interceptor(self, runner_pos: List[float]):
        """Run the PN pursuit policy and command the interceptor (non-blocking).

        Called before super().step() so the interceptor flies toward the runner
        during the same step_time sleep the runner moves in.
        """
        runner_vel = [
            self.drone.calculated_velocity.get("x", 0.0),
            self.drone.calculated_velocity.get("y", 0.0),
            self.drone.calculated_velocity.get("z", 0.0),
        ]
        self.interceptor.act({"runner_pos": runner_pos, "runner_vel": runner_vel})

    def _configure_interceptor_drone(
        self,
    ) -> None:
        """Apply SarlTag-specific safety limits to the interceptor drone."""
        if not self.use_simulator:
            return

        interceptor_drone = self.expert_drones[self.INTERCEPTOR_NAME]

        interceptor_drone.boundaries = {
            "x": 4,
            "y": 4,
            "z_min": -0.5,
            "z_max": 3.0,
        }

    # ------------------------------------------------------------------
    # Collision safety monitor — zeroes both drones within capture_threshold
    # ------------------------------------------------------------------

    def _freeze_interceptor(self):
        """Zero the interceptor's velocity setpoint (setpoints persist until replaced).

        Must be called before any long runner-handling window (reset, restart,
        ground EKF reset): otherwise the interceptor keeps flying on its stale
        pursuit command — typically toward the wall the runner just died beyond —
        for the whole window (up to 60 s for a restart) and coasts past its own
        internal boundary into the emergency kill. This is the "interceptor
        follows the dead runner and dies too" failure.
        """
        try:
            self.interceptor.body.apply_velocity(0.0, 0.0, 0.0)
            self.interceptor.velocity = [0.0, 0.0, 0.0]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # DroneEnvironment overrides
    # ------------------------------------------------------------------

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
        """Apply the current curriculum stage to the expert interceptor."""
        if not self.curriculum_enabled:
            self.interceptor_max_velocity = self.interceptor_speed_max
            return

        stage = self.CURRICULUM_STAGES[self.curriculum_stage]

        speed_factor = stage["interceptor_speed_factor"]
        self.interceptor_max_velocity = self.interceptor_speed_max * speed_factor

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

    def reset(
        self,
        training: bool = True,
    ):
        """
        Sample the task geometry and use the base environment to
        reset the runner and interceptor together.

        Mirrors marl_tag's reset: sample geometry, populate reset_positions,
        then delegate entirely to the base environment reset. The base
        _reset_all_drones() (shared pattern with MarlDroneEnvironment) already
        lands, teleports, clears a latched emergency, re-seeds the EKF and
        takes off EVERY drone (runner and interceptor alike) EVERY episode —
        that's what makes a dying drone recover. Layering task-level
        pre/post "is it dead, call restart()" checks on top of that (the
        previous approach here) fought with the base reset instead of
        complementing it and was the source of the flaky recovery.
        """

        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        # Stop the existing expert command while reset preparation
        # and any recovery operations are performed.
        self._freeze_interceptor()

        self._episode_count += 1

        # This must happen before interceptor spawn sampling because
        # the configured speed affects fair placement.
        self._update_interceptor_curriculum(training)

        self.reset_positions = self._generate_reset_positions()

        state = super().reset(training)
        # Fatal simulator recovery may recreate the DroneSim objects.
        # Keep the task's interceptor body attached to the current
        # environment-owned expert drone.
        current_interceptor_drone = self.expert_drones[self.INTERCEPTOR_NAME]

        if self.interceptor.body.drone is not current_interceptor_drone:
            print("[SarlTag] Rebinding interceptor body to recreated DroneSim.")
            self.interceptor.body.drone = current_interceptor_drone

        self._configure_interceptor_drone()

        runner_pos = self.rl_drone.get_position()

        # The environment resets the drone lifecycle. The task still
        # resets the expert policy.
        self.interceptor.reset_policy(
            {
                "runner_pos": runner_pos,
                "runner_vel": [0.0, 0.0, 0.0],
            }
        )

        self.interceptor.refresh()
        self._sync_interceptor()

        # Draw the goal after the new geometry has been selected.
        self._set_target_marker(
            self.goal_position,
            marker_name=self.goal_marker_name,
        )

        # _reset_task_state() is already called by super().reset(),
        # so caught, reached_goal and done have been cleared.
        self.previous_goal_distance = self._distance_to_target(runner_pos)

        self._prev_interceptor_velocity = [0.0, 0.0, 0.0]

        time.sleep(0.5)

        return self._get_state()

    def step(self, action):
        """One env step: command the expert interceptor, then move the learner (3D)."""

        self.total_steps += 1

        if self.total_steps == self.exploration_steps and not self.learning:
            print("\nSWITCHING TO LEARNING PHASE...\n")
            self.truncate_next = True
            self.learning = True

        assert len(action) == 3, "action should be length 3"
        if self.learning:
            processed_action = [action[0], action[1], action[2]]
        else:
            processed_action = [action[0] * 2 - 1, action[1] * 2 - 1, action[2] * 2 - 1]

        runner_pos = self.drone.get_position()
        # Command the expert interceptor BEFORE super().step() so both drones fly
        # simultaneously during the step_time sleep inside the parent step.
        self._command_interceptor(runner_pos)

        result = super().step(processed_action)

        # Refresh interceptor tracking after the step (it has flown for step_time).
        self.interceptor.refresh()
        self._sync_interceptor()

        return result

    def _reset_task_state(self):
        """Reset task-specific state variables (called from base reset)."""
        self.done = False
        self.caught = False
        self.reached_goal = False
        # A post-step interceptor-death check (see step()) can set this for
        # the *next* call to step() and never get consumed if that call never
        # comes because the episode ended on this very step. Left uncleared,
        # it silently truncates the following episode after a single step.
        self.truncate_next = False

    def _drones_with_z_boundary_violation(
        self,
        current_state: Dict[str, Any],
    ) -> list[str]:
        """
        Return owned drones whose altitude is outside the task's
        allowed z range.
        """

        violating_drones = []

        for drone_name, drone in self._iter_drones():
            if drone_name == self.RL_DRONE_NAME:
                position = current_state["position"]
            else:
                position = drone.get_position()

            if not (self.z_min <= position[2] <= self.z_max):
                violating_drones.append(drone_name)

        return violating_drones

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

        for other_drone in self.expert_drones.values():
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

    def get_overlay_info(self) -> Dict[str, Any]:
        position = self.drone.get_position()
        return {
            "position": position,
            "goal_position": self.goal_position[:],
            "interceptor_position": self.interceptor_position[:],
            "distance_to_goal": self._distance_to_target(position),
            "distance_to_interceptor": self._distance_to_interceptor(position),
            "caught": self.caught,
            "reached_goal": self.reached_goal,
            "done": self.done,
        }

    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Reward = progress to goal − step cost − evasion shaping, with terminal bonuses."""
        position = current_state["position"]
        goal_distance = current_state["distance_to_target"]
        interceptor_distance = self._distance_to_interceptor(position)

        # Out of bounds is a terminal failure.
        if self._is_out_of_task_bounds(position):
            self.previous_goal_distance = goal_distance
            return self.out_of_bounds_penalty

        # Caught by the interceptor is a terminal failure. The safety monitor may
        # have latched the collision mid-step even if the step-boundary distance
        # reads slightly above threshold, so honour the latched event too.
        if self._collision_safety_triggered():
            self.previous_goal_distance = goal_distance
            return self.intercepted_penalty

        # Reached the goal is a terminal success.
        if goal_distance < self.goal_threshold:
            self.previous_goal_distance = goal_distance
            return self.success_reward

        # Main signal: progress toward the goal.
        progress = self.previous_goal_distance - goal_distance
        reward = progress * self.goal_progress_multiplier

        # Small per-step cost so the runner is rewarded for reaching the goal FAST.
        reward -= self.step_penalty

        # Evasion shaping: ramp up a penalty as the interceptor closes inside the
        # danger radius, so the runner learns to keep clear without ignoring the goal.
        if interceptor_distance < self.danger_radius:
            closeness = 1.0 - (interceptor_distance / self.danger_radius)
            reward -= self.danger_penalty * closeness

        self.previous_goal_distance = goal_distance
        return reward

    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Episode ends on goal reached (success), interception, or out of bounds (failures)."""
        position = current_state["position"]
        goal_distance = current_state["distance_to_target"]
        interceptor_distance = self._distance_to_interceptor(position)

        if goal_distance < self.goal_threshold:
            self.reached_goal = True
            self.done = True
            if self._is_evaluating:
                self.successful_episodes_count += 1
            return True

        if self._collision_safety_triggered():
            self.caught = True
            self.done = True
            return True

        if self._is_out_of_task_bounds(position):
            self.done = True
            return True

        return False

    def is_in_testing_zone(self):
        # Judge against the task's own 3D bounds — the base is_in_boundaries
        # derives its height range from reset_position[2], which now varies
        # with the per-episode spawn altitude.
        return not self._is_out_of_task_bounds(self.drone.get_position())

    def _check_if_truncated(
        self,
        current_state: Dict[str, Any],
    ) -> bool:
        """
        Truncate the episode on time limit, fatal simulator failure,
        or an invalid drone altitude.
        """

        # A fatal DroneSim error requires the base environment to
        # restart CrazySim during the following reset.
        if self.use_simulator and self._get_fatal_sim_drone_names():
            return True

        z_violation_drones = self._drones_with_z_boundary_violation(current_state)

        if z_violation_drones:
            print(
                "[SarlTag] Z boundary violation detected for: "
                f"{z_violation_drones}. "
                "Truncating episode."
            )

            return True

        if self.steps >= self.episode_length:
            return True

        if self.truncate_next:
            self.truncate_next = False
            return True

        return False

    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        position = current_state["position"]
        info = {
            "goal_position": self.goal_position[:],
            "interceptor_position": self.interceptor_position[:],
            "distance_to_goal": self._distance_to_target(position),
            "distance_to_interceptor": self._distance_to_interceptor(position),
            "caught": self.caught,
            "reached_goal": self.reached_goal,
            "success": self.reached_goal,
            # Per-drone outcome flags (1/0), picked up automatically by the
            # generic success-rate / time-to-outcome plots (any "*_success"
            # column). Runner succeeds by reaching the goal; the interceptor
            # succeeds by catching the runner first.
            "runner_success": int(self.reached_goal),
            "interceptor_success": int(self.caught),
            "out_of_bounds": self._is_out_of_task_bounds(position),
            "description": "3D navigate-to-goal under interception — RL runner vs expert interceptor",
        }
        if self._is_evaluating:
            info["success_count"] = self.successful_episodes_count
        return info

    # ------------------------------------------------------------------
    # Action space — keep SARL's denormalize as a no-op so the parent's
    # single multiply-by-max_velocity is the only scaling that happens.
    # Without this, SARL denormalizes [-1,1]→[-0.25,0.25] and the parent
    # then multiplies by 0.25 again → 0.0625 m/s effective (4× too slow).
    # ------------------------------------------------------------------

    @property
    def max_action_value(self):
        return 1.0

    @property
    def min_action_value(self):
        return -1.0

    def sample_action(self):
        """Sample a normalized action in [-1, 1] — the parent will scale to m/s."""
        return np.random.uniform(-1.0, 1.0, size=(3,))

    def close(self) -> None:
        self._stop_safety_monitor()
        super().close()

    def _render_task_specific_info(self):
        pos = self.drone.get_position()
        d_goal = self._distance_to_target(pos)
        d_int = self._distance_to_interceptor(pos)
        print(f"Runner Position:      [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        print(
            f"Goal Position:        [{self.goal_position[0]:.2f}, "
            f"{self.goal_position[1]:.2f}, {self.goal_position[2]:.2f}]"
        )
        print(
            f"Interceptor Position: [{self.interceptor_position[0]:.2f}, "
            f"{self.interceptor_position[1]:.2f}, {self.interceptor_position[2]:.2f}]"
        )
        print(
            f"Distance to Goal:        {d_goal:.2f}  (threshold {self.goal_threshold:.2f})"
        )
        print(
            f"Distance to Interceptor: {d_int:.2f}  (capture {self.capture_threshold:.2f})"
        )
        print(f"Reached Goal: {self.reached_goal} | Caught: {self.caught}")

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
        ix, iy, iz = self.interceptor_position

        # LEFT: 3D trajectory
        ax1 = fig.add_subplot(gs[0, 0], projection="3d")
        ax1.plot(x, y, z, label="Runner Path", color="yellow", linewidth=2.5)
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
        ax1.scatter(
            ix,
            iy,
            iz,
            color="red",
            marker="^",
            s=120,
            label="Interceptor",
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
        ax1.legend(loc="upper left", fontsize=6, framealpha=0.9, markerscale=0.60)
        ax1.grid(True, alpha=0.3)
        ax1.set_box_aspect([1, 1, 0.67])

        # RIGHT: top-down X-Y
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
        ax2.plot(x, y, color="yellow", linewidth=2.5, label="Runner Path", zorder=2)
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
        ax2.scatter(
            ix,
            iy,
            color="red",
            marker=MarkerStyle("^"),
            s=120,
            label="Interceptor",
            edgecolors="black",
            linewidth=1,
            zorder=5,
        )
        ax2.add_patch(
            plt.Circle(
                (gx, gy), self.goal_threshold, color="lime", alpha=0.18, zorder=1
            )
        )
        ax2.add_patch(
            plt.Circle(
                (ix, iy), self.capture_threshold, color="red", alpha=0.15, zorder=1
            )
        )

        ax2.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_xlabel("X (m)", fontsize=10)
        ax2.set_ylabel("Y (m)", fontsize=10)
        ax2.set_title("Top-Down View (X-Y)", fontsize=12, pad=15)
        ax2.set_aspect("equal", adjustable="box")
        ax2.legend(loc="upper left", fontsize=6, framealpha=0.9, markerscale=0.60)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="both", labelsize=8)

        outcome = (
            "Reached Goal"
            if self.reached_goal
            else ("Caught" if self.caught else "In Progress")
        )
        fig.suptitle(f"SARL Tag (Step {self.steps}) | {outcome}", fontsize=13, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

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
                    frame, (width, height), interpolation=cv2.INTER_LANCZOS4
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.full((height, width, 3), 255, dtype=np.uint8)

        return frame
