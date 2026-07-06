from matplotlib.markers import MarkerStyle
import numpy as np
import math
import time
import threading
from typing import Dict, List, Any, Literal

from drone_gym.drone_environment import DroneEnvironment
from drone_gym.sim_manager import get_default_sim_manager
from drone_gym.agents.bodies import CrazyflieBody
from drone_gym.agents.policies import CallablePolicy
from drone_gym.agents.sim_agent import SimAgent
import matplotlib.pyplot as plt
import io
import cv2


class InterceptNavigation3D(DroneEnvironment):
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

    def __init__(self, use_simulator: Literal[0, 1], max_velocity: float = 0.25, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 80,
                 interceptor_max_velocity: float = 0.075):

        super().__init__(use_simulator, max_velocity, step_time)
        self.use_simulator = use_simulator

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
        # Tight z-band around the 1.0 m reset height. A narrow vertical corridor
        # means the runner only ever makes small vertical moves, keeping CrazySim's
        # vertical estimator well-conditioned so the thrust-spike launch can't build.
        self.z_min = 0.8
        self.z_max = 1.2
        self.fixed_z = 1.0                  # runner reset altitude (centre of the z band)
        self.spawn_margin = 0.5             # keep spawns clear of the xy wall (PID overshoot safety)
        self.goal_margin = 0.3              # keep the goal clear of the xy wall
        self.z_margin = 0.1                 # keep goal/interceptor spawn off the z floor/ceiling (tight band)
        self.out_of_bounds_tolerance = 0.05  # small grace for PID overshoot at the wall

        # The runner always spawns at the centre (0, 0, fixed_z) — diversity comes
        # from the random goal + interceptor placement, NOT from moving the runner.
        # This is the proven-stable pattern from intercept_evader / evade_pursuers:
        # a long-range reset move on the learner stresses CrazySim's EKF and makes
        # the drone tumble and fall. The runner still travels a real distance every
        # episode — to the goal — which is the whole point of the task.
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
        self.interceptor_z_jitter = 0.1      # vertical variety for the interceptor spawn (tight band)
        self.min_runner_clearance = 1.0      # interceptor never starts in (near) capture range of the runner
        self.min_goal_clearance = 0.6        # interceptor can't start camped on the goal

        self.interceptor_max_velocity = interceptor_max_velocity     # > max_velocity so capture is feasible
        self.interceptor_max_velocity_z = 0.030      # gentle vertical cap for the pursuer too

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
        self.sim_manager = get_default_sim_manager()
        self.goal_marker_name = "rl_intercept_nav_goal"
        # Runner is on port 19850; interceptor is drone 2 from sitl_multiagent_square -n 2
        interceptor_uri = "udp://0.0.0.0:19851"
        interceptor_body = CrazyflieBody(
            use_simulator=use_simulator,
            uri=interceptor_uri,
            fixed_z=self.fixed_z,
        )
        interceptor_policy = CallablePolicy(fn=self._interceptor_pursuit)
        self.interceptor = SimAgent(
            agent_id=1,
            body=interceptor_body,
            policy=interceptor_policy,
            role="interceptor",
        )
        self._interceptor_airborne = False

        # The interceptor repositions via a position-control move to a fresh spawn
        # EVERY episode, which stresses its EKF. The drone's internal safety monitor
        # hard-kills (emergency land + disarm) any drone whose |z| > 2.25 — a death
        # the interceptor can't recover from cleanly. Give its internal boundary
        # VERTICAL headroom only, so a transient EKF z-overshoot during
        # re-convergence doesn't trip the destructive kill. Keep xy at the drone
        # default (2.5, i.e. 0.5 m past the arena wall for PID overshoot) so a
        # lateral drift is still caught before the interceptor roams far outside
        # the arena. The task's own out-of-bounds + collision-guard logic
        # (xy_limit=2.0, z_max=1.5, capture_threshold) still governs episodes.
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

        # EKF drift prevention. Neither drone lands between episodes, so CrazySim's
        # onboard estimator accumulates drift with nothing to reset it. Left
        # unbounded, the drift eventually makes the altitude controller command a
        # thrust spike that LAUNCHES the drone past its internal z boundary — which
        # trips the body's emergency-stop (land + DISARM + kill the control thread),
        # the unrecoverable "the drone just died" crash. We can't reset the EKF in
        # the air (that itself causes a launch), so every N episodes we land the
        # drone, reset the filter on the ground, and take off again — capping the
        # drift well below launch territory.
        #
        # This MUST cover BOTH drones. The interceptor is actually the worse case:
        # it repositions across the whole arena via position control every single
        # episode (a bigger EKF stressor than the runner's short evasive moves) yet
        # historically got no periodic reset at all — so it drifted fastest, spiked
        # past its z=3.0 boundary, and took the emergency kill while the runner was
        # still alive. Resetting only the runner is why "both drones die".
        self._episode_count = 0
        self._ekf_reset_interval = 20

    # ------------------------------------------------------------------
    # Interceptor expert policy — 3D pure pursuit toward the runner
    # ------------------------------------------------------------------

    def _interceptor_pursuit(self, state, context) -> List[float]:
        """3D pure pursuit: head straight at the runner at interceptor_max_velocity.

        Velocity components that would drive the interceptor further past a soft
        boundary are zeroed so it slides along walls/ceiling instead of ramming
        them. ``context['runner_pos']`` is supplied by the task each step.
        """
        runner = context["runner_pos"]
        pos = state.position

        dx = runner[0] - pos[0]
        dy = runner[1] - pos[1]
        dz = runner[2] - pos[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < 1e-6:
            return [0.0, 0.0, 0.0]

        scale = self.interceptor_max_velocity / dist
        vx, vy, vz = scale * dx, scale * dy, scale * dz
        vz = float(np.clip(vz, -self.interceptor_max_velocity_z, self.interceptor_max_velocity_z))

        soft = self.xy_limit - 0.3
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
        """Run the expert pursuit policy and command the interceptor (non-blocking).

        Called before super().step() so the interceptor flies toward the runner
        during the same step_time sleep the runner moves in.
        """
        self.interceptor.act({"runner_pos": runner_pos})
        
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

    def _proactive_ekf_reset(self, drone) -> None:
        """Kalman-filter reset. DANGEROUS while airborne — DO NOT call in flight.

        Resetting the EKF mid-air makes the estimator emit a brief burst of garbage
        state, and the onboard controller responds with a thrust spike that
        "launches" the drone to the ceiling and out of the arena before it stalls
        and falls. EKF resets must happen on the ground, via the recovery path
        (_recover_drone re-inits the link and resets the filter safely). This helper
        is retained only for that ground-level use.
        """
        try:
            if getattr(drone, "cf", None) is not None:
                drone.cf.param.set_value("kalman.resetEstimation", "1")
                time.sleep(0.4)
        except Exception as exc:
            print(f"[InterceptNavigation3D] EKF reset warning: {exc}")

    def _ground_ekf_reset_runner(self) -> None:
        """Land the runner, reset its EKF on the ground, to bound accumulated drift.

        The runner never lands during a run, so CrazySim's estimator drifts
        unbounded until the altitude controller reacts with a thrust spike and
        launches the drone. Resetting the filter airborne causes that same launch,
        so we land first (the drone stays armed — land() doesn't disarm), reset on
        the ground where it's safe, and let the subsequent super().reset() take it
        off and re-centre with a fresh, drift-free estimate.
        """
        print(f"[InterceptNavigation3D] Ground EKF reset for runner (episode {self._episode_count})")
        try:
            if self.drone.velocity_controller_active:
                self.drone.stop_velocity_control()
            self.drone.set_velocity_vector(0, 0, 0)
            self.drone.land()
            self.drone.is_landed_event.wait(timeout=15)
            self._proactive_ekf_reset(self.drone)  # safe: the drone is on the ground now
        except Exception as exc:
            print(f"[InterceptNavigation3D] Runner ground EKF reset warning: {exc}")

    def _ground_ekf_reset_interceptor(self) -> None:
        """Land the interceptor, reset its EKF on the ground, to bound accumulated drift.

        Mirrors :meth:`_ground_ekf_reset_runner` for the second drone. The
        interceptor never lands during a run yet repositions across the whole arena
        via position control every episode, so its estimator drifts even faster than
        the runner's until a thrust spike launches it past its internal z boundary
        and the body's emergency-stop kills it. We land it (it stays armed — land()
        does not disarm), reset the filter on the ground where it's safe, and let the
        subsequent prepare_reset take it off + reposition with a fresh estimate.
        """
        d = getattr(self.interceptor.body, "drone", None)
        if d is None:
            return
        print(f"[InterceptNavigation3D] Ground EKF reset for interceptor (episode {self._episode_count})")
        try:
            if getattr(d, "velocity_controller_active", False):
                d.stop_velocity_control()
            if getattr(d, "controller_active", False):
                d.stop_position_control()
            d.set_velocity_vector(0, 0, 0)
            d.land()
            d.is_landed_event.wait(timeout=15)
            self._proactive_ekf_reset(d)  # safe: the interceptor is on the ground now
        except Exception as exc:
            print(f"[InterceptNavigation3D] Interceptor ground EKF reset warning: {exc}")

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
        print("[InterceptNavigation3D] Interceptor appears dead — recovering it")
        try:
            # Same survivable-recovery path as the runner: clears the latched
            # emergency, relaunches threads, re-inits the link + resets the EKF.
            if self._recover_drone(drone):
                drone.take_off()
                if not drone.is_flying_event.wait(timeout=15):
                    print("[InterceptNavigation3D] WARNING: interceptor did not confirm takeoff after recovery")
        except Exception as exc:
            print(f"[InterceptNavigation3D] Interceptor recovery exception: {exc}")

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
                print(f"[InterceptNavigation3D] Interceptor drifting during spawn move "
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
                    print("[InterceptNavigation3D] Interceptor spawn move kept diverging — ground recovery")
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
    # inside ~xy 1.75 / z 1.35, and interceptor spawns sit within ±1.5 / 0.7-1.3)
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
        while self._safety_monitor_running:
            try:
                rp = self.drone.get_position()
                ip = self.interceptor.body.get_position()

                # Skip placeholder/invalid readings. A drone that is (re)initialising
                # — e.g. mid-recovery, before its position system is up — reports
                # exactly (0,0,0). Treating that as a real position makes the guard
                # "see" a ~0 m separation and latch a BOGUS capture, which cascades
                # into spurious truncations and a two-drone restart storm.
                def _placeholder(p):
                    return p[0] == 0.0 and p[1] == 0.0 and p[2] == 0.0
                if _placeholder(rp) or _placeholder(ip):
                    time.sleep(dt)
                    continue

                separation = math.sqrt((rp[0] - ip[0]) ** 2 + (rp[1] - ip[1]) ** 2 + (rp[2] - ip[2]) ** 2)

                if separation < self.capture_threshold:
                    self._stop_both_drones()
                    if not self._collision_event.is_set():
                        self.caught = True
                        self._collision_event.set()
                        print(f"[InterceptNavigation3D] COLLISION GUARD: drones within "
                              f"{separation:.2f} m (< {self.capture_threshold:.2f}) — both stopped")
                elif not self._collision_event.is_set():
                    # Containment — brake (not reverse) any drone past the line.
                    if self._position_past_containment(rp):
                        try:
                            self.drone.set_velocity_vector(0, 0, 0)
                        except Exception:
                            pass
                    if self._position_past_containment(ip):
                        try:
                            self.interceptor.body.apply_velocity(0, 0, 0)
                            self.interceptor.velocity = [0.0, 0.0, 0.0]
                        except Exception:
                            pass
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
            print(f"[InterceptNavigation3D] RECOVER {name}: refusing to run from the drone's own thread")
            return False

        print(f"[InterceptNavigation3D] RECOVER {name}: starting")

        # 1. Stop all loops and wake anything blocked on the command queue.
        try:
            drone._signal_stop_to_all_threads()
        except Exception as e:
            print(f"[InterceptNavigation3D] RECOVER {name}: signal error: {e}")

        # 2. CRITICAL: let the old _run thread fully exit before relaunching.
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=25.0)
            if old_thread.is_alive():
                print(f"[InterceptNavigation3D] RECOVER {name}: old control thread won't exit — aborting")
                return False

        # 3. Join the remaining workers and reset shared state.
        for fn in ("_join_all_threads", "_reset_shared_state"):
            try:
                getattr(drone, fn)()
            except Exception as e:
                print(f"[InterceptNavigation3D] RECOVER {name}: {fn} error: {e}")

        # 4. Close the link explicitly so a re-open to the same URI can't hang.
        try:
            if getattr(drone, "scf", None) is not None:
                drone.scf.close_link()
        except Exception as e:
            print(f"[InterceptNavigation3D] RECOVER {name}: close_link error: {e}")
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
            print(f"[InterceptNavigation3D] RECOVER {name}: relaunch error: {e}")
            return False

        recovered = drone.hardware_ready_event.wait(timeout=20)
        print(f"[InterceptNavigation3D] RECOVER {name}: {'success' if recovered else 'FAILED'}")
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
            print("[InterceptNavigation3D] Skipping restart — too many consecutive "
                  "failures; the simulator may need a manual restart")
            return False

        print("[InterceptNavigation3D] Auto-restarting runner (no user input required)")

        done_event = threading.Event()
        success_holder = [False]

        def _do_restart():
            try:
                # _recover_drone clears the latched emergency_event and relaunches
                # the drone's worker threads (which re-init the link + reset the
                # EKF), making a blow-up survivable instead of permanently fatal.
                if self._recover_drone(self.drone):
                    self.drone.take_off()
                    if self.drone.is_flying_event.wait(timeout=15):
                        success_holder[0] = True
            except Exception as exc:
                print(f"[InterceptNavigation3D] Restart exception: {exc}")
            finally:
                done_event.set()

        thread = threading.Thread(target=_do_restart, daemon=True)
        thread.start()

        if not done_event.wait(timeout=self.RESTART_TIMEOUT_SECONDS):
            print(f"[InterceptNavigation3D] Restart timed out after "
                  f"{self.RESTART_TIMEOUT_SECONDS}s — abandoning attempt")
            self._consecutive_restart_failures += 1
            return False

        if success_holder[0]:
            self._consecutive_restart_failures = 0
            return True

        print("[InterceptNavigation3D] WARNING: runner did not confirm takeoff after restart")
        self._consecutive_restart_failures += 1
        return False

    # ------------------------------------------------------------------
    # DroneEnvironment overrides
    # ------------------------------------------------------------------

    def reset(self, training: bool = True):
        """Keep the runner centred, place a fresh 3D goal, and seed the interceptor."""

        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        self._reset_stuck_tracker()

        # Pause the collision guard while both drones reposition to their spawns
        # (they legitimately fly near each other during repositioning).
        self._stop_safety_monitor()
        self._collision_event.clear()

        # Periodically land the runner and reset its EKF ON THE GROUND to bound the
        # drift that would otherwise build up over a continuous run and eventually
        # launch it. Done before super().reset(), which then takes off + re-centres
        # with a fresh estimate. (Airborne resets cause the launch, so never here.)
        self._episode_count += 1
        self._ekf_reset_due = (self._episode_count % self._ekf_reset_interval == 0)
        if self._ekf_reset_due:
            self._ground_ekf_reset_runner()

        # Parent reset: takeoff, centre the runner at (0, 0, fixed_z), start velocity control.
        super().reset(training)

        # Stop the velocity controller that the parent started — the runner should hold
        # position quietly while we move the interceptor to its spawn below.  We'll
        # restart it at the end of this reset so the episode begins cleanly.
        if self.drone.velocity_controller_active:
            self.drone.stop_velocity_control()
        self.drone.set_velocity_vector(0, 0, 0)

        if self._drone_is_dead(self.drone.get_position()):
            print("[InterceptNavigation3D] EKF blow-up detected after parent reset — restarting runner")
            self.restart()

        # On the first reset, make sure the interceptor is airborne before we fly it.
        if not self._interceptor_airborne:
            self.interceptor.ensure_airborne()
            self.interceptor.await_airborne(timeout=15.0)
            self._interceptor_airborne = True

        # If the interceptor's EKF blew up (or it fell), re-initialise it before use.
        self._recover_interceptor_if_dead()

        # Periodic ground EKF reset for the interceptor, on the SAME cadence as the
        # runner. This is the fix for the interceptor dying while the runner lived:
        # the interceptor drifts fastest (a full-arena position move every episode)
        # and previously had no drift cap, so it launched past its z boundary and
        # took the emergency kill. We land + reset it here, on the ground, before the
        # prepare_reset below takes it off again to reposition with a fresh estimate.
        # (Skipped on episode 1, when the interceptor was only just brought airborne.)
        if getattr(self, "_ekf_reset_due", False) and self._interceptor_airborne:
            self._ground_ekf_reset_interceptor()

        # 1) The runner stays where the parent reset left it — the centre (0, 0, fixed_z).
        #    No long-range position move on the learner (the EKF stressor).
        runner_pos = self.drone.get_position()

        # 2) Sample the goal (far from the runner) and the interceptor spawn (on the path).
        self.goal_position = self._sample_goal(runner_pos)
        interceptor_spawn = self._sample_interceptor_spawn(runner_pos, self.goal_position)

        # Draw the goal marker in Gazebo.
        if self.use_simulator:
            self.sim_manager.set_visual_target_marker_position(
                self.goal_position[0], self.goal_position[1], self.goal_position[2],
                marker_name=self.goal_marker_name,
            )

        # 3) Fly the interceptor to its spawn, then arm it for the episode.
        # We deliberately do NOT reset the interceptor's EKF here. Resetting the
        # Kalman filter while the drone is airborne makes the estimator emit a brief
        # burst of garbage state, and the onboard controller reacts with a thrust
        # spike — the drone "launches" to the ceiling, leaves the arena, then stalls
        # and falls. (This was the cause of the sudden launch-into-the-air crashes.)
        # EKF resets now happen only on the ground, via the recovery path.
        self.interceptor.reset_policy({"runner_pos": runner_pos})
        self.interceptor.prepare_reset(interceptor_spawn)
        if not self._await_interceptor_reset_safely(interceptor_spawn, timeout=15.0):
            print("[InterceptNavigation3D] WARNING: interceptor did not reach spawn cleanly")
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
        if self._drone_is_dead(runner_pos) or self._drone_is_stuck(runner_pos):
            reason = "dead" if self._drone_is_dead(runner_pos) else "stuck"
            print(f"[InterceptNavigation3D] Runner appears {reason} (pos={runner_pos}) — restarting")
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
            print(f"[InterceptNavigation3D] Interceptor died mid-episode at "
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
        return self.is_in_boundaries()

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

    def close(self):
        """Land the interceptor and runner, then clear the goal marker."""
        self._stop_safety_monitor()
        try:
            self.interceptor.body.close()
        except Exception as e:
            print(f"[InterceptNavigation3D] Error closing interceptor: {e}")
        if self.use_simulator:
            self.sim_manager.remove_visual_marker(self.goal_marker_name)
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
        fig.suptitle(f'Intercept-Navigation 3D (Step {self.steps}) | {outcome}', fontsize=13, y=0.98)
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
