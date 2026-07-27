from matplotlib.markers import MarkerStyle
import numpy as np
import math
import time
import threading
from collections import deque
from typing import Dict, List, Any, Literal

from drone_gym.drone_environment import DroneEnvironment
from drone_gym.agents.bodies import CrazyflieBody
from drone_gym.agents.policies import CallablePolicy
from drone_gym.agents.sim_agent import SimAgent
import matplotlib.pyplot as plt
import io
import cv2


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

    An "interception" is a 0.30 m proximity event (3D), never a real drone-on-drone
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

    def __init__(self, use_simulator: Literal[0, 1], max_velocity: float = 0.25, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 80,
                 interceptor_max_velocity: float = 0.125):
        super().__init__(
            use_simulator=use_simulator, 
            max_velocity=max_velocity, 
            step_time=step_time,
            expert_drone_names=[self.INTERCEPTOR_NAME],
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
        self.fixed_z = 1.0                  # centre of the z band (default altitude)
        # Runner spawn altitude — resampled every episode. Kept a notch inside the
        # z band so the worst-case vertical gap to a goal (~0.5 m) stays closable
        # at max_velocity_z within an 80-step episode (0.5 / (0.03 * 0.5) ≈ 34 steps).
        self.runner_spawn_z_range = (0.8, 1.2)
        self.spawn_margin = 0.5             # keep spawns clear of the xy wall (PID overshoot safety)
        self.goal_margin = 0.3              # keep the goal clear of the xy wall
        self.z_margin = 0.1                 # keep goal/interceptor spawn off the z floor/ceiling (tight band)
        self.out_of_bounds_tolerance = 0.05  # small grace for PID overshoot at the wall

        # The runner always spawns at the xy centre (0, 0) — lateral diversity
        # comes from the random goal + interceptor placement, NOT from moving the
        # runner. This is the proven-stable pattern from intercept_evader /
        # evade_pursuers: a long-range lateral reset move on the learner stresses
        # CrazySim's EKF and makes the drone tumble and fall. The spawn ALTITUDE
        # is resampled each episode (see runner_spawn_z_range): that only changes
        # the length of the slow, PID-controlled vertical climb out of the ground
        # teleport, which the max_velocity_z cap keeps gentle.
        self.runner_spawn = [0.0, 0.0, self.fixed_z]

        # FAIRNESS — goal placement. The goal must be far enough that a faster
        # interceptor has a real chance to cut the runner off (if the goal were
        # right next to the runner, the runner trivially wins and the interceptor
        # has no chance). goal_max keeps the journey bounded so it stays winnable.
        self.goal_min_distance = 1.5
        self.goal_max_distance = 2.4

        # FAIRNESS — interceptor placement. The interceptor is seeded so it must
        # RACE to contest the runner's path: close enough to threaten, far enough
        # that the runner has a real chance. We pick a contest point along the
        # runner->goal line, then offset the interceptor sideways by a distance
        # scaled by the speed ratio so it arrives at the contest point at roughly
        # the same time as the runner (a fair race — neither side trivially wins).
        # Hard clearances stop the two degenerate cases the task must avoid:
        #   * interceptor right in front of the runner  -> runner has no chance
        #   * interceptor camped on the goal            -> runner has no chance
        self.intercept_frac = (0.45, 0.65)   # where along the runner->goal path the contest is set up
        self.fairness_jitter = 0.15          # ±15% randomness on the fair lateral distance
        self.interceptor_z_jitter = 0.25     # vertical variety for the interceptor spawn
        self.min_runner_clearance = 1.0      # interceptor never starts in (near) capture range of the runner
        self.min_goal_clearance = 0.6        # interceptor can't start camped on the goal

        self.interceptor_max_velocity = interceptor_max_velocity     # > max_velocity so capture is feasible
        self.interceptor_max_velocity_z = 0.030      # gentle vertical cap for the pursuer too

        # --- Interceptor speed curriculum (performance-gated ratchet) --------
        # Start the interceptor slow so the runner can learn to reach the goal,
        # then raise its speed as the runner's success rate climbs. Speed only
        # ever increases, and stalls automatically if the runner stops improving.
        self.curriculum_enabled = True
        self.interceptor_speed_min = 0.05                          # starting speed (m/s)
        self.interceptor_speed_max = self.interceptor_max_velocity  # ceiling = the ctor value
        self.curriculum_window = 50               # episodes judged per difficulty level
        self.curriculum_success_threshold = 0.6   # runner success rate that earns a bump
        self.curriculum_speed_step = 0.01         # speed added per bump (m/s)
        if self.curriculum_enabled:
            self.interceptor_max_velocity = self.interceptor_speed_min
        self._recent_runner_outcomes = deque(maxlen=self.curriculum_window)

        # Brake margins: the per-step clamp zeroes a velocity component before the
        # actual boundary, leaving room for the drone to coast to a stop instead of
        # sailing through the wall/ceiling on momentum (the out-of-bounds loop).
        self.boundary_brake_margin = 0.25   # xy: start braking this far inside xy_limit
        self.z_brake_margin = 0.10          # z: start braking this far inside the z band

        self.capture_threshold = 0.30      # metres (3D) — interceptor "catches" the runner (no real collision)
        self.goal_threshold = 0.20         # metres (3D) — runner has reached the goal

        self.max_xy_range = self.xy_limit * 2
        self.max_z_range = self.z_max - self.z_min
        self.max_distance = math.sqrt(self.max_xy_range ** 2 + self.max_xy_range ** 2 + self.max_z_range ** 2)
        self.time_tolerance = 0.15

        # boundary = (xy, xy, z_low, z_high) — used by the base visual boundary + step clamp
        self.boundary = [self.xy_limit, self.xy_limit, self.z_min, self.z_max]

        # Observation: own(6) + goal block(7) + interceptor block(10) = 23
        self.observation_space = 6 + 7 + 10

        # --- Reward parameters ----------------------------------------------
        self.success_reward = 100.0            # reached the goal — clearly the best outcome
        self.intercepted_penalty = -100.0      # caught by the interceptor — clearly the worst
        self.out_of_bounds_penalty = -100.0
        self.goal_progress_multiplier = 100.0  # main drive: reward closing the gap to the goal
        self.step_penalty = 1.0                # small per-step cost — reach the goal FAST
        self.danger_radius = 0.6               # within this of the interceptor, apply evasion shaping
        self.danger_penalty = 5.0              # max shaping penalty at zero separation

        # --- Task state ------------------------------------------------------
        self.done = False
        self.caught = False           # True when the interceptor caught the runner
        self.reached_goal = False     # True when the runner reached the goal

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
        interceptor_drone = getattr(self.interceptor.body, "drone", None)
        if interceptor_drone is not None and hasattr(interceptor_drone, "boundaries"):
            interceptor_drone.boundaries = {"x": 4, "y": 4, "z_min": -0.5, "z_max": 3.0}

        # --- Collision safety monitor ----------------------------------------
        # The RL step is 0.5 s, but a faster interceptor can close >0.25 m within
        # a single step — far enough to physically overlap before the step-boundary
        # distance check ever runs. A high-rate background monitor watches the 3D
        # separation continuously and the instant the two drones are within
        # capture_threshold it zeroes BOTH velocities (so they stop ~0.3 m apart,
        # never touching) and latches a collision. The episode then ends as a catch.
        self._collision_event = threading.Event()
        self._safety_monitor_running = False
        self._safety_thread = None

        # --- Action smoothing (topple prevention) ---------------------------
        # The velocity controller applies commanded velocity INSTANTLY (its slew
        # limiter is unwired and max_velocity_change_rate=100 ≈ no limit). SAC is a
        # maximum-entropy policy, so it outputs high-variance actions that swing
        # violently between steps (e.g. +0.25 → −0.25 m/s in xy); applying such an
        # instantaneous velocity reversal pitches the Crazyflie over and flips it in
        # CrazySim. TD3's near-deterministic actions are smooth and never hit this.
        # We slew-limit the commanded action per step so every velocity change is
        # gentle — a full reversal ramps over several steps instead of toppling.
        #
        # This slew cap is ALSO a horizontal-acceleration cap, which is the real
        # lever on the launch bug: to accelerate horizontally the quad must TILT,
        # and a tilted drone corrupts its own downward altitude sensor (the ToF
        # z/cos(tilt) model amplifies error), which is what triggers the firmware
        # thrust-spike launch. Smaller per-step velocity change -> smaller tilt ->
        # valid altitude estimate. The implied acceleration cap is
        #   max_action_delta * max_velocity / step_time
        #   = 0.2 * 0.25 / 0.5 = 0.10 m/s^2   (was 0.20 m/s^2 at 0.4)
        # i.e. the drone now tilts about half as hard to change course.
        # Lower this further to reduce tilt/launches more; raise for agility. [0, 2].
        self.max_action_delta = 0.2
        self._prev_applied_action = [0.0, 0.0, 0.0]

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
        soft = self.xy_limit - 0.3

        # Clamp the evader's position into the arena: a boundary-escaped runner
        # must never pull the aim-point outside it.
        runner = context["runner_pos"]
        rx = float(np.clip(runner[0], -soft, soft))
        ry = float(np.clip(runner[1], -soft, soft))
        rz = float(np.clip(runner[2], self.z_min, self.z_max))
        target_pos = np.array([rx, ry, rz])

        runner_vel = np.array(context.get("runner_vel", [0.0, 0.0, 0.0]), dtype=float)

        # Iterative solve for the predicted intercept point.
        pip = target_pos.copy()
        for _ in range(4):
            d = float(np.linalg.norm(pip - pos))
            if d < 1e-6:
                break
            t_go = d / self.interceptor_max_velocity
            pip = np.array([
                float(np.clip(target_pos[0] + runner_vel[0] * t_go, -soft, soft)),
                float(np.clip(target_pos[1] + runner_vel[1] * t_go, -soft, soft)),
                float(np.clip(target_pos[2] + runner_vel[2] * t_go, self.z_min, self.z_max)),
            ])

        aim = pip - pos
        aim_dist = float(np.linalg.norm(aim))
        if aim_dist < 1e-6:
            return [0.0, 0.0, 0.0]

        scale = self.interceptor_max_velocity / aim_dist
        vx, vy, vz = scale * aim[0], scale * aim[1], scale * aim[2]
        vz = float(np.clip(vz, -self.interceptor_max_velocity_z, self.interceptor_max_velocity_z))

        if (pos[0] <= -soft and vx < 0) or (pos[0] >= soft and vx > 0):
            vx = 0.0
        if (pos[1] <= -soft and vy < 0) or (pos[1] >= soft and vy > 0):
            vy = 0.0
        if (pos[2] <= self.z_min and vz < 0) or (pos[2] >= self.z_max and vz > 0):
            vz = 0.0

        return [vx, vy, vz]

    # ------------------------------------------------------------------
    # Geometry sampling (3D)
    # ------------------------------------------------------------------

    def _sample_goal(self, runner_pos: List[float]) -> List[float]:
        """Sample a goal between goal_min/max_distance (3D) from the runner."""
        xy = self.xy_limit - self.goal_margin
        z_lo, z_hi = self.z_min + self.z_margin, self.z_max - self.z_margin
        for _ in range(300):
            x = float(np.random.uniform(-xy, xy))
            y = float(np.random.uniform(-xy, xy))
            z = float(np.random.uniform(z_lo, z_hi))
            d = math.sqrt((x - runner_pos[0]) ** 2 + (y - runner_pos[1]) ** 2 + (z - runner_pos[2]) ** 2)
            if self.goal_min_distance <= d <= self.goal_max_distance:
                return [x, y, z]
        # Fallback: mirror the runner across the origin in xy, hold mid altitude
        return [
            float(np.clip(-runner_pos[0], -xy, xy)),
            float(np.clip(-runner_pos[1], -xy, xy)),
            self.fixed_z,
        ]

    def _sample_interceptor_spawn(self, runner_pos: List[float], goal_pos: List[float]) -> List[float]:
        """Seed the interceptor for a FAIR race to contest the runner's path.

        A contest point P is chosen a fraction ``f`` of the way along the
        runner->goal line. The runner reaches P after ~ ``f * D / v_runner``. We
        place the interceptor abeam P at a lateral distance ``L`` such that it
        reaches P after ~ ``L / v_interceptor`` ≈ the runner's time — i.e.
        ``L = (v_interceptor / v_runner) * f * D``. So the interceptor arrives at
        the contested point at roughly the same moment as the runner: it has a
        real shot, but cannot trivially win. Clearances keep it out of point-blank
        range of the runner and off the goal. Everything is clamped into the box.
        """
        xy = self.xy_limit - self.spawn_margin
        z_lo, z_hi = self.z_min + self.z_margin, self.z_max - self.z_margin

        dx = goal_pos[0] - runner_pos[0]
        dy = goal_pos[1] - runner_pos[1]
        dz = goal_pos[2] - runner_pos[2]
        D = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        xy_len = math.hypot(dx, dy) or 1.0
        px, py = -dy / xy_len, dx / xy_len  # unit perpendicular to the path in xy
        speed_ratio = self.interceptor_max_velocity / max(self.max_velocity, 1e-6)

        def _clearances_ok(p):
            d_runner = math.sqrt((p[0] - runner_pos[0]) ** 2 + (p[1] - runner_pos[1]) ** 2
                                 + (p[2] - runner_pos[2]) ** 2)
            d_goal = math.sqrt((p[0] - goal_pos[0]) ** 2 + (p[1] - goal_pos[1]) ** 2
                               + (p[2] - goal_pos[2]) ** 2)
            return d_runner >= self.min_runner_clearance and d_goal >= self.min_goal_clearance

        fallback = None
        for _ in range(400):
            f = float(np.random.uniform(*self.intercept_frac))
            # contest point on the runner->goal line
            cx = runner_pos[0] + f * dx
            cy = runner_pos[1] + f * dy
            cz = runner_pos[2] + f * dz
            # fair lateral distance so interceptor and runner reach P together
            L = speed_ratio * f * D * float(np.random.uniform(1.0 - self.fairness_jitter,
                                                              1.0 + self.fairness_jitter))
            side = 1.0 if np.random.random() < 0.5 else -1.0
            ix = cx + side * px * L
            iy = cy + side * py * L
            iz = float(np.clip(cz + float(np.random.uniform(-self.interceptor_z_jitter,
                                                            self.interceptor_z_jitter)), z_lo, z_hi))
            candidate = [ix, iy, iz]

            in_box = abs(ix) <= xy and abs(iy) <= xy
            if in_box and _clearances_ok(candidate):
                return candidate
            # Keep a clamped candidate as a fallback in case nothing fits cleanly.
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
    # Distances (3D)
    # ------------------------------------------------------------------

    def _distance_to_target(self, position: List[float]) -> float:
        """Base hook: 'target' for this task is the GOAL (used by the info dict)."""
        return math.sqrt((position[0] - self.goal_position[0]) ** 2 +
                         (position[1] - self.goal_position[1]) ** 2 +
                         (position[2] - self.goal_position[2]) ** 2)

    def _distance_to_interceptor(self, position: List[float]) -> float:
        return math.sqrt((position[0] - self.interceptor_position[0]) ** 2 +
                         (position[1] - self.interceptor_position[1]) ** 2 +
                         (position[2] - self.interceptor_position[2]) ** 2)

    def _is_out_of_task_bounds(self, position: List[float]) -> bool:
        """Out of the task's 3D boundary (with a small grace for PID overshoot).

        We check the task limits explicitly rather than trusting the drone's own
        in_boundaries flag, which is computed against a different internal limit.
        """
        tol = self.out_of_bounds_tolerance
        return (abs(position[0]) > self.xy_limit + tol or
                abs(position[1]) > self.xy_limit + tol or
                position[2] < self.z_min - tol or
                position[2] > self.z_max + tol)

    # ------------------------------------------------------------------
    # Interceptor lifecycle / health (EKF blow-up / fell-to-the-floor recovery)
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
        
    INTERCEPTOR_DEAD_XY = 2.4

    def _interceptor_is_dead(self) -> bool:
        """Detect an interceptor whose EKF blew up, that has fallen, or that has
        drifted clean out of the arena.

        Detection is POSITION-based only, deliberately. We do NOT key off the
        drone's emergency_event: that flag is set by transient connection
        blips during a recovery/reconnect, and reacting to it triggers yet another
        recovery — a feedback loop that turns one death into an unrecoverable
        restart storm across both drones. A real out-of-arena divergence shows up
        as a bad POSITION, which is unambiguous and self-clearing once recovery
        repositions the drone.

        Guard against the (0,0,0) placeholder a drone reports mid-recovery: it is
        NOT a real position.
        """
        try:
            pos = self.interceptor.body.get_position()
        except Exception:
            return False
        if pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 0.0:
            return False  # placeholder during (re)init — not a real reading
        if self._drone_is_dead(pos):
            return True
        if abs(pos[0]) > self.INTERCEPTOR_DEAD_XY or abs(pos[1]) > self.INTERCEPTOR_DEAD_XY:
            return True
        return False

    def _proactive_ekf_reset(self, drone, position: List[float] | None = None) -> None:
        """Kalman-filter reset. DANGEROUS while airborne — DO NOT call in flight.

        Resetting the EKF mid-air makes the estimator emit a brief burst of garbage
        state, and the onboard controller responds with a thrust spike that
        "launches" the drone to the ceiling and out of the arena before it stalls
        and falls. EKF resets must happen on the ground, via the recovery path
        (_recover_drone re-inits the link and resets the filter safely). This helper
        is retained only for that ground-level use.

        When ``position`` is given, the filter is seeded with it (via the
        kalman.initialX/Y/Z firmware params) before the reset, so the estimate
        STARTS at the drone's new true position instead of having to converge
        onto it from sensor data — this is how we "tell" a just-teleported
        drone where it now is. (Velocity needs no seeding: the reset zeroes the
        velocity state, and a landed drone's true velocity IS zero.)
        """
        try:
            if getattr(drone, "cf", None) is None:
                return
            if position is not None:
                try:
                    drone.cf.param.set_value("kalman.initialX", f"{float(position[0])}")
                    drone.cf.param.set_value("kalman.initialY", f"{float(position[1])}")
                    drone.cf.param.set_value("kalman.initialZ", f"{float(position[2])}")
                except Exception as exc:
                    print(f"[SarlTag] EKF position seed warning "
                          f"(continuing with plain reset): {exc}")
            drone.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.4)
        except Exception as exc:
            print(f"[SarlTag] EKF reset warning: {exc}")

    # ------------------------------------------------------------------
    # Teleport reset — land, move the MODELS to their spawns, take off fresh
    # ------------------------------------------------------------------

    # Gazebo model ids: sitl_multiagent spawns crazyflie_{i} bound to udp port
    # 19850+i (the same convention the MARL environment's teleport uses), so
    # the runner (19850) is model 0 and the interceptor (19851) is model 1.
    RUNNER_GAZEBO_ID = 0
    INTERCEPTOR_GAZEBO_ID = 1
    GROUND_Z = 0.02  # resting height for a ground teleport

    def _land_for_teleport(self, drone) -> None:
        """Best-effort: stop controllers and land one drone so its motors are
        idle for the teleport. A dead link just falls through — the teleport
        doesn't need the drone's cooperation, only Gazebo's."""
        try:
            if getattr(drone, "velocity_controller_active", False):
                drone.stop_velocity_control()
            if getattr(drone, "position_controller_active", False):
                drone.stop_position_control()
            drone.set_velocity_vector(0, 0, 0)
            drone.land()
        except Exception as exc:
            print(f"[SarlTag] Land-for-teleport warning "
                  f"({getattr(drone, 'agent_id', '?')}): {exc}")

    def _teleport_reset_both(self, interceptor_spawn: List[float]) -> None:
        """Land both drones, TELEPORT their Gazebo models to the new spawns on
        the floor, and re-seed the estimators on the ground.

        This is the dead-drone fix: nobody flies home between episodes. The old
        reset repositioned the interceptor across the whole arena via position
        control (the #1 EKF stressor) and needed a crashed drone to fly back
        from wherever it died — impossible for a dead one. Instead the episode
        simply *starts* with each model at its spawn:

          1. land both drones together (motors idle — teleporting a model out
             from under a flying controller looks like a massive position
             error, the same crash family as the airborne EKF reset),
          2. set_pose each model to its spawn xy on the floor, LEVEL — which
             also rights a toppled drone, previously an unrecoverable state,
          3. reset each EKF on the ground (the safe reset) so the estimator
             converges at the spawn, not the crash site; a drone with a
             latched emergency gets the full link recovery instead, which
             re-inits + resets its filter as part of bring-up.

        Take-off happens afterwards: the parent reset lifts the runner and
        prepare_reset lifts the interceptor — both now vertical-only moves.
        If a teleport service call fails, the drones are simply left where
        they landed and those same moves degrade to the old fly-to-spawn
        behaviour.
        """
        interceptor_drone = getattr(self.interceptor.body, "drone", None)

        # 1) Land both together (parallel descent), then wait for both.
        self._land_for_teleport(self.drone)
        if interceptor_drone is not None:
            self._land_for_teleport(interceptor_drone)
        for d in (self.drone, interceptor_drone):
            if d is None:
                continue
            try:
                if not d.is_landed_event.wait(timeout=15):
                    print(f"[SarlTag] WARNING: {getattr(d, 'agent_id', '?')} "
                          f"did not confirm landing before teleport")
            except Exception:
                pass

        # 2) Teleport the models to their spawns, flat on the floor.
        if not self.sim_manager.set_drone_pose(
                self.RUNNER_GAZEBO_ID,
                self.runner_spawn[0], self.runner_spawn[1], self.GROUND_Z,
                orientation=(0.0, 0.0, 0.0)):
            print("[SarlTag] WARNING: runner ground teleport failed")
        if not self.sim_manager.set_drone_pose(
                self.INTERCEPTOR_GAZEBO_ID,
                interceptor_spawn[0], interceptor_spawn[1], self.GROUND_Z,
                orientation=(0.0, 0.0, 0.0)):
            print("[SarlTag] WARNING: interceptor ground teleport failed")
        time.sleep(0.3)  # let physics settle the models onto the floor

        # 3) Fresh estimator at the new true position, while safely grounded.
        #    The reset is seeded with the teleported pose (kalman.initialX/Y/Z)
        #    so the estimate starts exactly where the drone now is. Latched-
        #    emergency drones get the full link recovery instead (unseeded —
        #    its re-init resets the filter and the estimate then converges
        #    from sensor data, same as the pre-teleport behaviour).
        if self.drone.emergency_event.is_set():
            print("[SarlTag] Runner emergency latched — ground recovery at spawn")
            self.restart()  # timeout-guarded link recovery (take-off at the end is harmless here)
        else:
            self._proactive_ekf_reset(
                self.drone,
                position=[self.runner_spawn[0], self.runner_spawn[1], self.GROUND_Z],
            )
        if interceptor_drone is not None:
            if interceptor_drone.emergency_event.is_set():
                print("[SarlTag] Interceptor emergency latched — ground recovery at spawn")
                self._recover_interceptor_if_dead(force=True)
            else:
                self._proactive_ekf_reset(
                    interceptor_drone,
                    position=[interceptor_spawn[0], interceptor_spawn[1], self.GROUND_Z],
                )
        time.sleep(0.5)  # give the estimators a beat to converge at the spawns

    def _recover_interceptor_if_dead(self, force: bool = False):
        """Re-initialise + take off the interceptor if it has died, before an episode.

        force=True skips the death heuristics and recovers unconditionally 
        used when the caller already has proof the interceptor is unusable (e.g.
        the spawn-reposition move kept diverging past containment).
        """
        if not force and not self._interceptor_is_dead():
            return
        drone = getattr(self.interceptor.body, "drone", None)
        if drone is None:
            return
        print("[SarlTag] Interceptor appears dead — recovering it")
        # Never recover airborne: _recover_drone's re-init resets the EKF, and an
        # airborne EKF reset is the thrust-spike launch. A drift past the death
        # line leaves the interceptor flying, so land it first (best-effort — a
        # dead link just falls through to the recovery below).
        try:
            if drone.is_flying_event.is_set():
                drone.land()
                drone.is_landed_event.wait(timeout=10)
        except Exception:
            pass
        try:
            # Same survivable-recovery path as the runner: clears the latched
            # emergency, relaunches threads, re-inits the link + resets the EKF.
            if self._recover_drone(drone):
                drone.take_off()
                if not drone.is_flying_event.wait(timeout=15):
                    print("[SarlTag] WARNING: interceptor did not confirm takeoff after recovery")
        except Exception as exc:
            print(f"[SarlTag] Interceptor recovery exception: {exc}")

    def _position_past_containment(self, pos) -> bool:
        """True if `pos` has drifted past the containment lines (toward the kill)."""
        return (abs(pos[0]) > self.CONTAINMENT_XY or
                abs(pos[1]) > self.CONTAINMENT_XY or
                pos[2] > self.CONTAINMENT_Z_HIGH or
                pos[2] < self.CONTAINMENT_Z_LOW)

    def _await_interceptor_reset_safely(self, spawn: List[float], timeout: float = 15.0) -> bool:
        """Wait for the interceptor to reach its spawn, aborting + re-centering if it drifts.

        The position-control move to spawn is the one interceptor motion not covered
        by the per-step velocity clamp, so a diverging EKF can drive it toward the
        fatal kill boundary during the move. Poll the reset event; if the drone
        drifts past a containment line before arriving, abort the move and let it
        settle, then retry from the same target. We do NOT reset the EKF in the air
        here — an airborne Kalman reset triggers the thrust-spike "launch". After a
        few failed attempts, hand off to ground recovery (which resets the EKF
        safely) rather than looping into the kill.
        """
        deadline = time.time() + timeout
        attempts = 0
        max_attempts = 3
        while time.time() < deadline:
            if self.interceptor.await_reset(timeout=0.2):
                return True
            try:
                pos = self.interceptor.body.get_position()
            except Exception:
                continue
            if self._position_past_containment(pos):
                attempts += 1
                print(f"[SarlTag] Interceptor drifting during spawn move "
                      f"(pos={[round(p, 2) for p in pos]}) — aborting move + letting it settle "
                      f"(attempt {attempts}/{max_attempts})")
                drone = getattr(self.interceptor.body, "drone", None)
                if drone is not None:
                    try:
                        drone.stop_position_control()
                        drone.set_velocity_vector(0, 0, 0)
                    except Exception:
                        pass
                if attempts >= max_attempts:
                    print("[SarlTag] Interceptor spawn move kept diverging — ground recovery")
                    # Force it: the move diverging past containment IS the proof it
                    # needs re-init, even if a jumped EKF estimate happens not to
                    # trip the position/emergency death checks.
                    self._recover_interceptor_if_dead(force=True)
                    return False
                time.sleep(0.3)  # let it settle in place before retrying the move
                self.interceptor.prepare_reset(spawn)
        return False

    # ------------------------------------------------------------------
    # Collision safety monitor — zeroes both drones within capture_threshold
    # ------------------------------------------------------------------

    SAFETY_MONITOR_HZ = 20.0  # how often the background monitor checks separation

    # Containment thresholds — the guard steers a drone back inside once it
    # crosses these. Set ABOVE normal operation (the step clamp keeps the runner
    # inside ~xy 1.75 / z 0.7-1.3, and interceptor spawns sit within ±1.5 / 0.7-1.3)
    # but well BELOW the drone's internal fatal kill boundary (xy 2.5, z 2.25;
    # interceptor z 3.0). That gap is the runway the guard uses to correct a
    # drifting/overshooting drone before the destructive emergency-land can fire.
    CONTAINMENT_XY = 2.1
    CONTAINMENT_Z_HIGH = 1.8
    CONTAINMENT_Z_LOW = 0.25

    def _stop_both_drones(self):
        """Immediately command zero velocity to the runner and the interceptor."""
        try:
            self.drone.set_velocity_vector(0, 0, 0)
        except Exception:
            pass
        try:
            self.interceptor.body.apply_velocity(0, 0, 0)
            self.interceptor.velocity = [0.0, 0.0, 0.0]
        except Exception:
            pass

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

    def _start_safety_monitor(self):
        self._collision_event.clear()
        if self._safety_monitor_running:
            return
        self._safety_monitor_running = True
        self._safety_thread = threading.Thread(target=self._safety_monitor_loop, daemon=True)
        self._safety_thread.start()

    def _stop_safety_monitor(self):
        self._safety_monitor_running = False
        if self._safety_thread is not None:
            self._safety_thread.join(timeout=1.0)
            self._safety_thread = None

    def _safety_monitor_loop(self):
        """Background guard: stop both drones on capture, and brake either drone
        that approaches the fatal boundary (3D).

        Runs much faster than the RL step so neither drone can blow past the 0.3 m
        capture distance — nor coast into the internal kill boundary — inside a
        single 0.5 s step. Priorities each tick:
          1. If within capture_threshold: stop both drones and latch the collision.
          2. Otherwise, if a drone has crossed a containment line: BRAKE it (zero
             velocity) so it halts before the drone's internal emergency-land
             boundary (the "outside boundary crash").

        Containment deliberately BRAKES rather than driving the drone back inward:
        commanding an inward velocity from this background thread fights the RL /
        pursuit velocity command and the sudden setpoint reversal topples the
        Crazyflie in CrazySim. Zeroing velocity is the same proven-safe operation
        the collision guard uses, and still halts the drone (~0.4 m short of the
        kill boundary); the episode then ends out-of-bounds and reset recovers.
        """
        dt = 1.0 / self.SAFETY_MONITOR_HZ

        # A drone that is (re)initialising — e.g. mid-recovery, before its
        # position system is up — reports exactly (0,0,0). Treating that as a
        # real position makes the guard "see" a ~0 m separation and latch a
        # BOGUS capture, which cascades into spurious truncations and a
        # two-drone restart storm.
        def _placeholder(p):
            return p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0

        while self._safety_monitor_running:
            # Read each drone independently: one drone being mid-recovery
            # (placeholder position or a raising link) must NOT disable the
            # containment brake for the OTHER drone — that gap is exactly how
            # the interceptor used to sail out after its stale pursuit command
            # while the runner was being recovered.
            rp = ip = None
            try:
                rp = self.drone.get_position()
            except Exception:
                pass
            try:
                ip = self.interceptor.body.get_position()
            except Exception:
                pass
            rp_ok = rp is not None and not _placeholder(rp)
            ip_ok = ip is not None and not _placeholder(ip)

            captured = False
            if rp_ok and ip_ok:
                separation = math.sqrt((rp[0] - ip[0]) ** 2 + (rp[1] - ip[1]) ** 2 + (rp[2] - ip[2]) ** 2)
                if separation < self.capture_threshold:
                    captured = True
                    self._stop_both_drones()
                    if not self._collision_event.is_set():
                        self.caught = True
                        self._collision_event.set()
                        print(f"[SarlTag] COLLISION GUARD: drones within "
                              f"{separation:.2f} m (< {self.capture_threshold:.2f}) — both stopped")

            if not captured and not self._collision_event.is_set():
                # Containment — brake (not reverse) any drone past the line,
                # each judged on its own (valid) reading only.
                if rp_ok and self._position_past_containment(rp):
                    try:
                        self.drone.set_velocity_vector(0, 0, 0)
                    except Exception:
                        pass
                if ip_ok and self._position_past_containment(ip):
                    try:
                        self.interceptor.body.apply_velocity(0, 0, 0)
                        self.interceptor.velocity = [0.0, 0.0, 0.0]
                    except Exception:
                        pass
            time.sleep(dt)

    # ------------------------------------------------------------------
    # Dead / stuck runner recovery (unattended-training safety)
    # ------------------------------------------------------------------

    STUCK_POSITION_TOLERANCE = 1e-4
    STUCK_THRESHOLD_STEPS = 6
    MAX_CONSECUTIVE_RESTART_ATTEMPTS = 5
    RESTART_TIMEOUT_SECONDS = 60  # recover() re-launches threads + re-inits hardware; give it room

    def _reset_stuck_tracker(self):
        self._recent_positions = []

    def _drone_is_dead(self, position: List[float]) -> bool:
        """Detect EKF blow-up or a post-emergency dead-drone state."""
        x, y, z = position
        if abs(x) > 5.0 or abs(y) > 5.0 or abs(z) > 5.0:
            return True
        if x == 0.0 and y == 0.0 and z == 0.0:
            return True
        if z < 0.1:
            return True
        # EKF z-drift: drone is stuck well above the task ceiling after many resets
        if z > self.z_max + 0.5:
            return True
        return False

    def _runner_is_dead(self) -> bool:
        """Runner-specific death check: position heuristics OR a latched emergency.

        The sim emergency stop latches emergency_event WITHOUT disarming, so a
        boundary-killed runner can be re-launched by the parent reset's take_off
        and hover at its crash site while every control loop (all gated on
        emergency_event) refuses commands — a "zombie" whose position (e.g.
        x=2.6, z=1.0) looks perfectly alive to the position heuristics. The
        latched flag is the one unambiguous signature of that state, and
        restart() -> _recover_drone clears it. (Interceptor detection stays
        position-only: its recovery path sees transient emergency blips from
        reconnects, which the runner — recovered synchronously from the env
        thread — does not.)
        """
        if self.drone.emergency_event.is_set():
            return True
        return self._drone_is_dead(self.drone.get_position())

    def _drone_is_stuck(self, position: List[float]) -> bool:
        """Detect a frozen drone reporting the same position every step."""
        if not hasattr(self, "_recent_positions"):
            self._recent_positions = []
        self._recent_positions.append(tuple(position))
        if len(self._recent_positions) > self.STUCK_THRESHOLD_STEPS:
            self._recent_positions = self._recent_positions[-self.STUCK_THRESHOLD_STEPS:]
        if len(self._recent_positions) < self.STUCK_THRESHOLD_STEPS:
            return False
        first = self._recent_positions[0]
        for p in self._recent_positions[1:]:
            if (abs(p[0] - first[0]) > self.STUCK_POSITION_TOLERANCE or
                    abs(p[1] - first[1]) > self.STUCK_POSITION_TOLERANCE or
                    abs(p[2] - first[2]) > self.STUCK_POSITION_TOLERANCE):
                return False
        return True

    def _recover_drone(self, drone) -> bool:
        """Recover a blown-up / emergency-stopped drone, driving its own primitives.

        The drone layer latches ``emergency_event`` (never cleared) and its
        ``_run`` command thread exits on emergency and is never recreated — so a
        plain reconnect hangs and the drone stays dead. We do the recovery here,
        from a neutral thread:

          1. signal all loops to stop and wake queue waiters,
          2. WAIT for the old ``_run`` thread to fully exit before relaunching —
             it shares ``self.cf``/``self.scf`` with the relaunch, so if it is
             still finishing its emergency shutdown when the new thread connects
             it will disarm/close the brand-new link,
          3. join the remaining workers and reset state,
          4. close the link explicitly (the drone's own cleanup doesn't), so the
             next ``open_link()`` to the same URI can't hang on a bound socket,
          5. clear the latched flags,
          6. drain the command queue (the ``"exit"`` used to wake the old threads
             would otherwise immediately stop the freshly launched ``_run``),
          7. relaunch the coordinated worker threads — ``_run`` re-opens the link,
             resets the EKF and re-arms.

        Must NOT be called from one of the drone's own worker threads (the join
        would self-deadlock). Returns True if the hardware re-initialised.
        """
        name = getattr(drone, "agent_id", "drone")
        old_thread = getattr(drone, "thread", None)
        if old_thread is threading.current_thread():
            print(f"[SarlTag] RECOVER {name}: refusing to run from the drone's own thread")
            return False

        print(f"[SarlTag] RECOVER {name}: starting")

        # 1. Stop all loops and wake anything blocked on the command queue.
        try:
            drone._signal_stop_to_all_threads()
        except Exception as e:
            print(f"[SarlTag] RECOVER {name}: signal error: {e}")

        # 2. CRITICAL: let the old _run thread fully exit before relaunching.
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=25.0)
            if old_thread.is_alive():
                print(f"[SarlTag] RECOVER {name}: old control thread won't exit — aborting")
                return False

        # 3. Join the remaining workers and reset shared state.
        for fn in ("_join_all_threads", "_reset_shared_state"):
            try:
                getattr(drone, fn)()
            except Exception as e:
                print(f"[SarlTag] RECOVER {name}: {fn} error: {e}")

        # 4. Close the link explicitly so a re-open to the same URI can't hang.
        try:
            if getattr(drone, "scf", None) is not None:
                drone.scf.close_link()
        except Exception as e:
            print(f"[SarlTag] RECOVER {name}: close_link error: {e}")
        drone.cf = None
        drone.scf = None
        drone.mc = None

        # 5. Clear the latched flags that keep the drone disabled.
        drone.emergency_event.clear()
        drone.hardware_ready_event.clear()
        drone.position_ready_event.clear()
        drone.in_boundaries = True

        # 6. Drain leftover commands (e.g. the "exit" from step 1).
        try:
            drone.clear_command_queue()
        except Exception:
            pass

        # 7. Relaunch the coordinated worker threads.
        drone.set_running(True)
        drone.thread = threading.Thread(target=drone._run)
        try:
            drone._start_threads_coordinated()
        except Exception as e:
            print(f"[SarlTag] RECOVER {name}: relaunch error: {e}")
            return False

        recovered = drone.hardware_ready_event.wait(timeout=20)
        print(f"[SarlTag] RECOVER {name}: {'success' if recovered else 'FAILED'}")
        return recovered

    def restart(self):
        """Auto-restart the runner with no user input (overrides the interactive base).

        open_link() can block indefinitely if the simulator is dead, so the whole
        sequence runs in a daemon thread we abandon after RESTART_TIMEOUT_SECONDS.
        After MAX_CONSECUTIVE_RESTART_ATTEMPTS failures we stop trying.
        """
        if not hasattr(self, "_consecutive_restart_failures"):
            self._consecutive_restart_failures = 0

        if self._consecutive_restart_failures >= self.MAX_CONSECUTIVE_RESTART_ATTEMPTS:
            print("[SarlTag] Skipping restart — too many consecutive "
                  "failures; the simulator may need a manual restart")
            return False

        print("[SarlTag] Auto-restarting runner (no user input required)")

        done_event = threading.Event()
        success_holder = [False]

        def _do_restart():
            try:
                # Never recover airborne: _recover_drone's re-init resets the EKF,
                # and an airborne EKF reset is the thrust-spike launch. A zombie
                # runner (latched emergency) may well be hovering — the parent
                # reset re-launches it because the sim emergency stop never
                # disarms. Land first (best-effort; a dead link falls through).
                if self.drone.is_flying_event.is_set():
                    try:
                        self.drone.land()
                        self.drone.is_landed_event.wait(timeout=10)
                    except Exception:
                        pass
                # _recover_drone clears the latched emergency_event and relaunches
                # the drone's worker threads (which re-init the link + reset the
                # EKF), making a blow-up survivable instead of permanently fatal.
                if self._recover_drone(self.drone):
                    self.drone.take_off()
                    if self.drone.is_flying_event.wait(timeout=15):
                        success_holder[0] = True
            except Exception as exc:
                print(f"[SarlTag] Restart exception: {exc}")
            finally:
                done_event.set()

        thread = threading.Thread(target=_do_restart, daemon=True)
        thread.start()

        if not done_event.wait(timeout=self.RESTART_TIMEOUT_SECONDS):
            print(f"[SarlTag] Restart timed out after "
                  f"{self.RESTART_TIMEOUT_SECONDS}s — abandoning attempt")
            self._consecutive_restart_failures += 1
            return False

        if success_holder[0]:
            self._consecutive_restart_failures = 0
            return True

        print("[SarlTag] WARNING: runner did not confirm takeoff after restart")
        self._consecutive_restart_failures += 1
        return False

    # ------------------------------------------------------------------
    # DroneEnvironment overrides
    # ------------------------------------------------------------------

    def _update_interceptor_curriculum(self, training: bool = True) -> None:
        """Record the finished episode's runner outcome and, once a full window
        is in, raise the interceptor's speed if the runner is succeeding often
        enough. Ratchets up only; stalls if the runner plateaus."""
        if not self.curriculum_enabled or not training:
            return
        # self.reached_goal still holds the just-finished episode's result here
        # (reset clears it later), so record it before the rest of reset runs.
        if self._episode_count > 1:
            self._recent_runner_outcomes.append(1.0 if self.reached_goal else 0.0)
        if len(self._recent_runner_outcomes) < self.curriculum_window:
            return
        success_rate = sum(self._recent_runner_outcomes) / len(self._recent_runner_outcomes)
        if (success_rate >= self.curriculum_success_threshold
                and self.interceptor_max_velocity < self.interceptor_speed_max):
            self.interceptor_max_velocity = min(
                self.interceptor_speed_max,
                self.interceptor_max_velocity + self.curriculum_speed_step,
            )
            self._recent_runner_outcomes.clear()  # re-earn the next bump at the new speed
            print(f"[SarlTag][curriculum] runner success {success_rate:.0%} -> "
                  f"interceptor speed {self.interceptor_max_velocity:.3f} m/s")

    def reset(self, training: bool = True):
        """Teleport both drones to fresh spawns (sim) and start a new episode.

        Sim resets never fly anyone home: both drones land, their MODELS are
        teleported to the new spawn points on the ground, estimators re-seed
        there, and take-off is the only flight before the episode begins. On
        real hardware (no teleport available) this falls back to the old
        fly-to-spawn reset.
        """

        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        self._reset_stuck_tracker()

        # Pause the collision guard while both drones reposition to their spawns
        # (they legitimately fly near each other during repositioning).
        self._stop_safety_monitor()
        self._collision_event.clear()

        # Hold the interceptor for the whole reset: with the guard paused, its
        # last pursuit setpoint would otherwise keep flying it (typically toward
        # the wall a dying runner just crossed) for however long the runner
        # handling below takes.
        self._freeze_interceptor()

        self._episode_count += 1

        # Update the interceptor speed BEFORE sampling its spawn, so the new
        # speed feeds the fair-spawn placement this episode.
        self._update_interceptor_curriculum(training)

        # Vertical variety: resample the runner's spawn altitude each episode.
        # xy stays pinned at the centre (long lateral reset moves are the #1 EKF
        # stressor); the fresh z only changes the length of the slow vertical
        # climb after take-off. The parent reset targets reset_position.
        self.runner_spawn[2] = float(np.random.uniform(*self.runner_spawn_z_range))
        self.reset_position = list(self.runner_spawn)

        # Sample the new geometry up front — the runner always starts at
        # runner_spawn, so nothing here depends on live drone positions.
        self.goal_position = self._sample_goal(self.runner_spawn)
        interceptor_spawn = self._sample_interceptor_spawn(self.runner_spawn, self.goal_position)

        if self.use_simulator:
            # Land + teleport both models to their spawns + ground EKF reset.
            # Dead / toppled / boundary-killed drones are recovered here by
            # construction — no flight back from a crash site is ever needed.
            self._teleport_reset_both(interceptor_spawn)
        elif self._episode_count > 1 and self._runner_is_dead():
            # Real hardware: no teleport available. Recover a dead runner
            # BEFORE the parent reset — super().reset() on a zombie (latched
            # emergency) burns ~30 s re-launching a drone whose control loops
            # refuse every command, times out, and never re-centres it.
            print("[SarlTag] Runner dead at reset entry — recovering before parent reset")
            self.restart()

        # Parent reset: takeoff, move the runner to (0, 0, spawn z), start
        # velocity control. After a teleport the runner is already at the spawn
        # xy, so this is a short vertical-only climb — not a cross-arena move.
        super().reset(training)

        # Stop the velocity controller that the parent started — the runner should hold
        # position quietly while we move the interceptor to its spawn below.  We'll
        # restart it at the end of this reset so the episode begins cleanly.
        if self.drone.velocity_controller_active:
            self.drone.stop_velocity_control()
        self.drone.set_velocity_vector(0, 0, 0)

        # Verify the re-centre actually happened. This catches every failure
        # mode regardless of which heuristic missed it: a dead/zombie runner, an
        # EKF blow-up, or a position move that silently timed out all leave the
        # runner far from the spawn. Recover, then retry the parent reset ONCE
        # so the episode starts with a healthy, centred drone.
        pos = self.drone.get_position()
        off_center = math.dist(pos, self.runner_spawn) > 0.8
        if self._runner_is_dead() or off_center:
            print(f"[SarlTag] Runner unhealthy after parent reset "
                  f"(pos={[round(p, 2) for p in pos]}, off_center={off_center}) — "
                  f"restarting + re-centering")
            self.restart()
            super().reset(training)
            if self.drone.velocity_controller_active:
                self.drone.stop_velocity_control()
            self.drone.set_velocity_vector(0, 0, 0)

        # Real hardware only: recover the interceptor if it died. (In sim the
        # teleport reset above already handled every dead-interceptor case.)
        if not self.use_simulator:
            self._recover_interceptor_if_dead()

        # The runner stays where the parent reset left it — the centre (0, 0, fixed_z).
        runner_pos = self.drone.get_position()

        # Draw the goal marker in Gazebo.
        self._set_target_marker(self.goal_position, marker_name=self.goal_marker_name)

        # Bring the interceptor up at its spawn and arm it for the episode.
        # prepare_reset takes off a grounded drone and position-controls it to
        # the spawn — after the teleport that's a vertical-only climb from the
        # right xy already, with a freshly ground-reset estimator. We never
        # reset the EKF airborne (that's the thrust-spike launch).
        self.interceptor.reset_policy({"runner_pos": runner_pos, "runner_vel": [0.0, 0.0, 0.0]})
        self.interceptor.prepare_reset(interceptor_spawn)
        if not self._await_interceptor_reset_safely(interceptor_spawn, timeout=15.0):
            print("[SarlTag] WARNING: interceptor did not reach spawn cleanly")
        self.interceptor.start_episode()
        self.interceptor.refresh()
        self._sync_interceptor()

        # 4) Reset task state and resume runner velocity control for the episode.
        self.caught = False
        self.reached_goal = False
        self.done = False
        self.previous_goal_distance = self._distance_to_target(runner_pos)
        self._prev_applied_action = [0.0, 0.0, 0.0]  # runner starts the episode at rest

        time.sleep(0.5)  # final settle so both drones are stable before stepping
        self.drone.start_velocity_control()

        # Both drones are at their (well-separated) spawns — arm the collision guard.
        self._start_safety_monitor()

        return self._get_state()

    def step(self, action):
        """One env step: command the expert interceptor, then move the learner (3D)."""

        self.total_steps += 1

        if self.total_steps == self.exploration_steps and not self.learning:
            print("\nSWITCHING TO LEARNING PHASE...\n")
            self.truncate_next = True
            self.learning = True

        assert len(action) == 3, 'action should be length 3'
        if self.learning:
            processed_action = [action[0], action[1], action[2]]
        else:
            processed_action = [action[0] * 2 - 1, action[1] * 2 - 1, action[2] * 2 - 1]

        # If the safety monitor has already latched a collision, end the episode
        # immediately without commanding any further motion. Both drones are held
        # at zero velocity by the monitor; a no-op parent step returns the terminal
        # observation/reward/done (caught is already True).
        if self._collision_event.is_set():
            self._stop_both_drones()
            result = super().step([0, 0, 0])
            self.interceptor.refresh()
            self._sync_interceptor()
            return result

        runner_pos = self.drone.get_position()
        runner_dead = self._runner_is_dead()
        if runner_dead or self._drone_is_stuck(runner_pos):
            reason = "dead" if runner_dead else "stuck"
            print(f"[SarlTag] Runner appears {reason} (pos={runner_pos}) — restarting")
            # Hold the interceptor first: restart() can block for up to 60 s,
            # during which its stale pursuit setpoint would keep flying it. The
            # guard's containment brake only catches it past the 2.1 m line;
            # freezing now keeps it inside the arena for the whole restart.
            self._freeze_interceptor()
            self.restart()
            self._reset_stuck_tracker()
            self.truncate_next = True
            runner_pos = self.drone.get_position()

        # Command the expert interceptor BEFORE super().step() so both drones fly
        # simultaneously during the step_time sleep inside the parent step.
        self._command_interceptor(runner_pos)

        # Per-axis boundary clamp: only zero a velocity component that would push the
        # runner FURTHER out of bounds. Inward motion is always allowed so a drone
        # that has drifted out can return (and the runner can slide along walls).
        position = self.drone.get_position()
        time_step = self.step_time + self.time_tolerance
        clamped_action = list(processed_action)

        xy_brake = self.xy_limit - self.boundary_brake_margin
        z_hi = self.z_max - self.z_brake_margin
        z_lo = self.z_min + self.z_brake_margin

        predicted_x = position[0] + time_step * clamped_action[0] * self.max_velocity
        if predicted_x > xy_brake and clamped_action[0] > 0:
            clamped_action[0] = 0.0
        elif predicted_x < -xy_brake and clamped_action[0] < 0:
            clamped_action[0] = 0.0

        predicted_y = position[1] + time_step * clamped_action[1] * self.max_velocity
        if predicted_y > xy_brake and clamped_action[1] > 0:
            clamped_action[1] = 0.0
        elif predicted_y < -xy_brake and clamped_action[1] < 0:
            clamped_action[1] = 0.0

        predicted_z = position[2] + time_step * clamped_action[2] * self.max_velocity_z
        if predicted_z > z_hi and clamped_action[2] > 0:
            clamped_action[2] = 0.0
        elif predicted_z < z_lo and clamped_action[2] < 0:
            clamped_action[2] = 0.0

        # Slew-rate limit: cap how far the commanded action can move from the last
        # applied action, so no single velocity change is violent enough to topple
        # the drone (SAC's high-entropy actions otherwise swing hard step-to-step).
        limited_action = list(clamped_action)
        for i in range(3):
            delta = clamped_action[i] - self._prev_applied_action[i]
            delta = max(-self.max_action_delta, min(self.max_action_delta, delta))
            limited_action[i] = self._prev_applied_action[i] + delta
        self._prev_applied_action = list(limited_action)

        result = super().step(limited_action)

        # Refresh interceptor tracking after the step (it has flown for step_time).
        self.interceptor.refresh()
        self._sync_interceptor()

        # If the interceptor died this step, end the episode now so we don't keep
        # training against a fallen drone; the next reset re-initialises it.
        if self._interceptor_is_dead():
            try:
                ipos = self.interceptor.body.get_position()
            except Exception:
                ipos = self.interceptor_position
            print(f"[SarlTag] Interceptor died mid-episode at "
                  f"pos={[round(p, 2) for p in ipos]} (z={ipos[2]:.2f}) — truncating. "
                  f"z>~2.0 here means a thrust-spike launch past its boundary.")
            self.truncate_next = True

        return result

    def _reset_task_state(self):
        """Reset task-specific state variables (called from base reset)."""
        self.done = False
        self.caught = False
        self.reached_goal = False

    def _get_state(self) -> np.ndarray:
        """Runner-centric 3D observation: own state + goal block + interceptor block."""
        position = self.drone.get_position()
        vel_x = self.drone.calculated_velocity["x"]
        vel_y = self.drone.calculated_velocity["y"]
        vel_z = self.drone.calculated_velocity["z"]

        z_mid = 0.5 * (self.z_min + self.z_max)
        z_half = 0.5 * self.max_z_range

        # Own state (6): position (3) + velocity (3)
        state: List[float] = [
            position[0] / self.xy_limit,
            position[1] / self.xy_limit,
            (position[2] - z_mid) / z_half,
            vel_x / self.max_velocity,
            vel_y / self.max_velocity,
            vel_z / self.max_velocity_z,
        ]

        # Goal block (7): relative pos (3), distance (1), direction (3)
        gx, gy, gz = self.goal_position
        g_rel_x, g_rel_y, g_rel_z = gx - position[0], gy - position[1], gz - position[2]
        g_dist = math.sqrt(g_rel_x ** 2 + g_rel_y ** 2 + g_rel_z ** 2)
        state += [
            g_rel_x / self.max_xy_range,
            g_rel_y / self.max_xy_range,
            g_rel_z / self.max_z_range,
            g_dist / self.max_distance,
            g_rel_x / (g_dist + 1e-6),
            g_rel_y / (g_dist + 1e-6),
            g_rel_z / (g_dist + 1e-6),
        ]

        # Interceptor block (10): relative pos (3), distance (1), direction (3), velocity (3)
        ix, iy, iz = self.interceptor_position
        i_rel_x, i_rel_y, i_rel_z = ix - position[0], iy - position[1], iz - position[2]
        i_dist = math.sqrt(i_rel_x ** 2 + i_rel_y ** 2 + i_rel_z ** 2)
        state += [
            i_rel_x / self.max_xy_range,
            i_rel_y / self.max_xy_range,
            i_rel_z / self.max_z_range,
            i_dist / self.max_distance,
            i_rel_x / (i_dist + 1e-6),
            i_rel_y / (i_dist + 1e-6),
            i_rel_z / (i_dist + 1e-6),
            self.interceptor_velocity[0] / (self.interceptor_max_velocity + 1e-6),
            self.interceptor_velocity[1] / (self.interceptor_max_velocity + 1e-6),
            self.interceptor_velocity[2] / (self.interceptor_max_velocity_z + 1e-6),
        ]

        return np.array(state, dtype=np.float32)

    def get_overlay_info(self) -> Dict[str, Any]:
        position = self.drone.get_position()
        return {
            'position': position,
            'goal_position': self.goal_position[:],
            'interceptor_position': self.interceptor_position[:],
            'distance_to_goal': self._distance_to_target(position),
            'distance_to_interceptor': self._distance_to_interceptor(position),
            'caught': self.caught,
            'reached_goal': self.reached_goal,
            'done': self.done,
        }

    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Reward = progress to goal − step cost − evasion shaping, with terminal bonuses."""
        position = current_state['position']
        goal_distance = current_state['distance_to_target']
        interceptor_distance = self._distance_to_interceptor(position)

        # Out of bounds is a terminal failure.
        if self._is_out_of_task_bounds(position):
            self.previous_goal_distance = goal_distance
            return self.out_of_bounds_penalty

        # Caught by the interceptor is a terminal failure. The safety monitor may
        # have latched the collision mid-step even if the step-boundary distance
        # reads slightly above threshold, so honour the latched event too.
        if self._collision_event.is_set() or interceptor_distance < self.capture_threshold:
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
        position = current_state['position']
        goal_distance = current_state['distance_to_target']
        interceptor_distance = self._distance_to_interceptor(position)

        if goal_distance < self.goal_threshold:
            self.reached_goal = True
            self.done = True
            if self._is_evaluating:
                self.successful_episodes_count += 1
            return True

        if self._collision_event.is_set() or interceptor_distance < self.capture_threshold:
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

    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        if self.steps >= self.episode_length:
            if self.need_to_change_battery():
                self.change_battery()
            elif current_state["position"][2] <= 0.25:
                self.restart()
            return True

        if self.truncate_next:
            self.truncate_next = False
            return True

        return False

    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        position = current_state['position']
        info = {
            'goal_position': self.goal_position[:],
            'interceptor_position': self.interceptor_position[:],
            'distance_to_goal': self._distance_to_target(position),
            'distance_to_interceptor': self._distance_to_interceptor(position),
            'caught': self.caught,
            'reached_goal': self.reached_goal,
            'success': self.reached_goal,
            'out_of_bounds': self._is_out_of_task_bounds(position),
            'description': "3D navigate-to-goal under interception — RL runner vs expert interceptor",
        }
        if self._is_evaluating:
            info['success_count'] = self.successful_episodes_count
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
        print(f"Goal Position:        [{self.goal_position[0]:.2f}, {self.goal_position[1]:.2f}, {self.goal_position[2]:.2f}]")
        print(f"Interceptor Position: [{self.interceptor_position[0]:.2f}, {self.interceptor_position[1]:.2f}, {self.interceptor_position[2]:.2f}]")
        print(f"Distance to Goal:        {d_goal:.2f}  (threshold {self.goal_threshold:.2f})")
        print(f"Distance to Interceptor: {d_int:.2f}  (capture {self.capture_threshold:.2f})")
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
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax1.plot(x, y, z, label='Runner Path', color='yellow', linewidth=2.5)
        ax1.scatter(x[0], y[0], z[0], color='green', s=80, label='Start',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(x[-1], y[-1], z[-1], color='blue', s=80, label='Current',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(gx, gy, gz, color='lime', marker='*', s=160, label='Goal',
                    depthshade=False, edgecolors='black', linewidth=1)
        ax1.scatter(ix, iy, iz, color='red', marker='^', s=120, label='Interceptor',
                    depthshade=False, edgecolors='black', linewidth=1)
        ax1.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_zlim(self.z_min - 0.1, self.z_max + 0.1)
        ax1.set_xlabel('X (m)', fontsize=10, labelpad=8)
        ax1.set_ylabel('Y (m)', fontsize=10, labelpad=8)
        ax1.set_zlabel('Z (m)', fontsize=9, labelpad=10)
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.tick_params(axis='z', labelsize=8)
        ax1.view_init(elev=10, azim=25)
        ax1.set_title('3D Trajectory', fontsize=12, pad=15)
        ax1.legend(loc='upper left', fontsize=6, framealpha=0.9, markerscale=0.60)
        ax1.grid(True, alpha=0.3)
        ax1.set_box_aspect([1, 1, 0.67])

        # RIGHT: top-down X-Y
        ax2 = fig.add_subplot(gs[0, 1])
        boundary_x = [-self.xy_limit, self.xy_limit, self.xy_limit, -self.xy_limit, -self.xy_limit]
        boundary_y = [-self.xy_limit, -self.xy_limit, self.xy_limit, self.xy_limit, -self.xy_limit]
        ax2.plot(boundary_x, boundary_y, 'k--', linewidth=1, alpha=0.5, label='Boundary', zorder=1)
        ax2.plot(x, y, color='yellow', linewidth=2.5, label='Runner Path', zorder=2)
        ax2.scatter(x[0], y[0], color='green', s=80, label='Start',
                    edgecolors='black', linewidth=0.5, zorder=4)
        ax2.scatter(x[-1], y[-1], color='blue', s=80, label='Current',
                    edgecolors='black', linewidth=0.5, zorder=4)
        ax2.scatter(gx, gy, color='lime', marker=MarkerStyle('*'), s=160, label='Goal',
                    edgecolors='black', linewidth=1, zorder=5)
        ax2.scatter(ix, iy, color='red', marker=MarkerStyle('^'), s=120, label='Interceptor',
                    edgecolors='black', linewidth=1, zorder=5)
        ax2.add_patch(plt.Circle((gx, gy), self.goal_threshold, color='lime', alpha=0.18, zorder=1))
        ax2.add_patch(plt.Circle((ix, iy), self.capture_threshold, color='red', alpha=0.15, zorder=1))

        ax2.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_title('Top-Down View (X-Y)', fontsize=12, pad=15)
        ax2.set_aspect('equal', adjustable='box')
        ax2.legend(loc='upper left', fontsize=6, framealpha=0.9, markerscale=0.60)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=8)

        outcome = "Reached Goal" if self.reached_goal else ("Caught" if self.caught else "In Progress")
        fig.suptitle(f'SARL Tag (Step {self.steps}) | {outcome}', fontsize=13, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120,
                    facecolor='white', edgecolor='none', bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)

        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            current_h, current_w = frame.shape[:2]
            if current_h != height or current_w != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.full((height, width, 3), 255, dtype=np.uint8)

        return frame
