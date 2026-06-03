from matplotlib.markers import MarkerStyle
import numpy as np
import math
import time
import os
import tempfile
import threading
import subprocess
from typing import Dict, List, Any, Literal, Optional, Tuple

from drone_gym.drone_environment import DroneEnvironment
import matplotlib.pyplot as plt
import io
import cv2


class EvadePursuers2DParticle(DroneEnvironment):
    """2D evasion task where pursuers are continuously-moving virtual particles.

    Each pursuer is a Gazebo visual marker (no physics, no flight) updated by a
    background thread at high frequency to give smooth visual motion. This avoids
    all the physics issues of using real flying Crazyflies as pursuers:
    no PID overshoot during reset, no boundary breaches, no collision impulses,
    no altitude drift. Reset is instant.

    Only the evader is a real Crazyflie — launch with the single-agent script:
        sitl_singleagent.sh -m crazyflie -x 0 -y 0
    """

    PURSUER_MARKER_PREFIX = "rl_pursuer_particle_"
    PARTICLE_UPDATE_HZ = 5.0  # background thread update rate (lower = less Gazebo load = firmware EKF gets more CPU)
    WORLD_NAME = "crazysim_default"

    def __init__(self, use_simulator: Literal[0, 1], max_velocity: float = 0.10, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 40,
                 n_pursuers: int = 1, pursuer_max_velocity: float = 0.30):

        super().__init__(use_simulator, max_velocity, step_time)

        # RL training parameters
        self.episode_length = episode_length
        self.exploration_steps = exploration_steps
        self.total_steps = 0
        self.truncate_next = False
        self.learning = True

        # Task parameters
        self.n_pursuers = n_pursuers
        self.pursuer_max_velocity = pursuer_max_velocity
        self.capture_threshold = 0.20
        self.spawn_min_distance = 1.0
        # Spawn margin is intentionally large so PID position-control overshoot during
        # reset can't push the drone past xy_limit. Velocity-control step() already
        # enforces the boundary via per-axis clamp, but position control during reset
        # is a separate code path that can overshoot its target.
        self.spawn_margin = 0.5
        self.fixed_z = 1.0

        self.xy_limit = 2.0
        self.max_xy_range = self.xy_limit * 2
        self.max_distance = np.sqrt(self.max_xy_range ** 2 + self.max_xy_range ** 2)
        self.time_tolerance = 0.15

        self.boundary = [self.xy_limit, self.xy_limit, self.z_limit, self.z_limit + 1]
        self.observation_space = 4 + 7 * n_pursuers

        # Reward parameters
        self.caught_penalty = -50.0
        self.survival_reward = 1.0
        self.distance_reward_multiplier = 2.0
        self.out_of_bounds_penalty = -100.0

        # Task state
        self.done = False
        self.caught = False
        self.boundary_penalise = False
        self.exited_testing_boundary = False

        # Particle state (positions/velocities updated by background thread)
        self.pursuer_positions: List[List[float]] = [[0.0, 0.0, self.fixed_z] for _ in range(n_pursuers)]
        self.pursuer_velocities: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(n_pursuers)]
        self._particle_lock = threading.Lock()

        # Background thread
        self._particle_thread_running = False
        self._particle_thread: Optional[threading.Thread] = None
        self._marker_files: List[str] = [
            os.path.join(tempfile.gettempdir(), f"{self.PURSUER_MARKER_PREFIX}{i}.sdf")
            for i in range(n_pursuers)
        ]
        self._markers_spawned = [False] * n_pursuers

        self.previous_distance = self.max_distance
        self.successful_episodes_count = 0

        # Spawn the particle markers and start the update thread
        self._write_marker_model_files()
        self._spawn_all_markers()
        self._start_particle_thread()

    # ------------------------------------------------------------------
    # Gazebo marker management — direct gz service calls so we can use
    # one marker per pursuer (the built-in drone_sim only supports one)
    # ------------------------------------------------------------------

    def _run_gz_service(self, service: str, reqtype: str, reptype: str, req: str) -> bool:
        cmd = [
            "gz", "service", "-s", service,
            "--reqtype", reqtype, "--reptype", reptype,
            "--timeout", "200", "--req", req,
        ]
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=1.0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        output = ((result.stdout or "") + (result.stderr or "")).lower()
        if result.returncode != 0 and "already exists" not in output:
            return False
        if "data: false" in output:
            return False
        return True

    def _marker_name(self, i: int) -> str:
        return f"{self.PURSUER_MARKER_PREFIX}{i}"

    def _write_marker_model_files(self):
        for i, path in enumerate(self._marker_files):
            sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{self._marker_name(i)}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry><sphere><radius>0.08</radius></sphere></geometry>
        <material>
          <ambient>1 0.2 0.2 0.9</ambient>
          <diffuse>1 0.2 0.2 0.9</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
            with open(path, "w", encoding="utf-8") as f:
                f.write(sdf)

    def _spawn_marker(self, i: int, x: float, y: float, z: float) -> bool:
        req = (
            f'sdf_filename: "{self._marker_files[i]}", '
            f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}, '
            f'name: "{self._marker_name(i)}", allow_renaming: false'
        )
        return self._run_gz_service(
            service=f"/world/{self.WORLD_NAME}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
        )

    def _set_marker_pose(self, i: int, x: float, y: float, z: float) -> bool:
        req = f'name: "{self._marker_name(i)}", position: {{x: {x}, y: {y}, z: {z}}}'
        return self._run_gz_service(
            service=f"/world/{self.WORLD_NAME}/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            req=req,
        )

    def _spawn_all_markers(self):
        for i in range(self.n_pursuers):
            pos = self.pursuer_positions[i]
            if self._spawn_marker(i, pos[0], pos[1], pos[2]):
                self._markers_spawned[i] = True
            else:
                # Probably already exists from a previous run — try setting pose
                self._markers_spawned[i] = self._set_marker_pose(i, pos[0], pos[1], pos[2])

    # ------------------------------------------------------------------
    # Background particle update thread
    # ------------------------------------------------------------------

    def _start_particle_thread(self):
        self._particle_thread_running = True
        self._particle_thread = threading.Thread(target=self._particle_update_loop, daemon=True)
        self._particle_thread.start()

    def _stop_particle_thread(self):
        self._particle_thread_running = False
        if self._particle_thread is not None:
            self._particle_thread.join(timeout=1.0)

    def _particle_update_loop(self):
        """Advance each particle by velocity * dt and push pose to Gazebo at PARTICLE_UPDATE_HZ.

        We deliberately do NOT cap elapsed: if a Gazebo service call lags, the next
        update must compensate so the pursuer stays at its true commanded velocity.
        Capping would silently slow the pursuer down whenever gz is sluggish.
        """
        dt = 1.0 / self.PARTICLE_UPDATE_HZ
        last_time = time.time()
        while self._particle_thread_running:
            now = time.time()
            elapsed = now - last_time
            last_time = now

            with self._particle_lock:
                snapshot: List[Tuple[int, float, float, float]] = []
                for i in range(self.n_pursuers):
                    px, py, pz = self.pursuer_positions[i]
                    vx, vy, _ = self.pursuer_velocities[i]
                    nx = px + vx * elapsed
                    ny = py + vy * elapsed
                    nx = float(np.clip(nx, -self.xy_limit, self.xy_limit))
                    ny = float(np.clip(ny, -self.xy_limit, self.xy_limit))
                    self.pursuer_positions[i] = [nx, ny, self.fixed_z]
                    snapshot.append((i, nx, ny, self.fixed_z))

            for i, x, y, z in snapshot:
                self._set_marker_pose(i, x, y, z)

            sleep_time = dt - (time.time() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # Pursuit policy (Pure Pursuit)
    # ------------------------------------------------------------------

    def _compute_pursuer_velocity(self, pursuer_pos: List[float], evader_pos: List[float]) -> List[float]:
        dx = evader_pos[0] - pursuer_pos[0]
        dy = evader_pos[1] - pursuer_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return [0.0, 0.0, 0.0]
        scale = self.pursuer_max_velocity / dist
        return [scale * dx, scale * dy, 0.0]

    def _update_pursuer_velocities(self, evader_pos: List[float]):
        with self._particle_lock:
            for i in range(self.n_pursuers):
                vx, vy, _ = self._compute_pursuer_velocity(self.pursuer_positions[i], evader_pos)
                self.pursuer_velocities[i] = [vx, vy, 0.0]

    # ------------------------------------------------------------------
    # Position sampling
    # ------------------------------------------------------------------

    def _sample_random_position(self) -> List[float]:
        limit = self.xy_limit - self.spawn_margin
        x = float(np.random.uniform(-limit, limit))
        y = float(np.random.uniform(-limit, limit))
        return [x, y, self.fixed_z]

    def _sample_pursuer_spawn_positions(self, evader_pos: List[float]) -> List[List[float]]:
        limit = self.xy_limit - self.spawn_margin
        targets: List[List[float]] = []
        for i in range(self.n_pursuers):
            placed = False
            for _ in range(200):
                x = float(np.random.uniform(-limit, limit))
                y = float(np.random.uniform(-limit, limit))
                if math.sqrt((x - evader_pos[0]) ** 2 + (y - evader_pos[1]) ** 2) >= self.spawn_min_distance:
                    targets.append([x, y, self.fixed_z])
                    placed = True
                    break
            if not placed:
                angle = i * (2 * math.pi / self.n_pursuers)
                targets.append([math.cos(angle) * limit, math.sin(angle) * limit, self.fixed_z])
        return targets

    # ------------------------------------------------------------------
    # DroneEnvironment overrides
    # ------------------------------------------------------------------

    def reset(self, training: bool = True):
        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        self._reset_stuck_tracker()

        # Parent reset takes the evader to (0, 0, 1) via position control.
        # We deliberately do NOT do a second random-position move — the long-range
        # position command stresses CrazySim's EKF and causes blow-ups. Training
        # diversity comes from the random pursuer spawn instead.
        super().reset(training)

        # If the parent's position-control move blew up the EKF, restart the drone
        if self._drone_is_dead(self.drone.get_position()):
            print("[EvadePursuers2DParticle] EKF blow-up detected after parent reset — restarting drone")
            self.restart()

        # Teleport pursuer particles instantly — no physics, no flight
        evader_pos = self.drone.get_position()
        spawn_positions = self._sample_pursuer_spawn_positions(evader_pos)
        with self._particle_lock:
            for i, sp in enumerate(spawn_positions):
                self.pursuer_positions[i] = list(sp)
                self.pursuer_velocities[i] = [0.0, 0.0, 0.0]

        # Push initial poses to Gazebo immediately
        for i, sp in enumerate(spawn_positions):
            self._set_marker_pose(i, sp[0], sp[1], sp[2])

        self.caught = False
        self.previous_distance = self._distance_to_target(self.drone.get_position())

        return self._get_state()

    STUCK_POSITION_TOLERANCE = 1e-4  # treat positions within this as identical (frozen-log case)
    STUCK_THRESHOLD_STEPS = 6  # how many identical positions before declaring frozen
    UNRESPONSIVE_WINDOW = 8  # steps to check for movement-vs-command mismatch
    UNRESPONSIVE_POSITION_RADIUS = 0.05  # stayed within this radius
    UNRESPONSIVE_MIN_COMMANDED_SPEED = 0.05  # m/s — agent was actually trying to move

    def _drone_is_dead(self, position: List[float]) -> bool:
        """Detect EKF blow-up or post-emergency-landing dead-drone state."""
        x, y, z = position
        if abs(x) > 5.0 or abs(y) > 5.0 or abs(z) > 5.0:
            return True
        if x == 0.0 and y == 0.0 and z == 0.0:
            return True
        if z < 0.1:
            return True
        return False

    def _drone_is_stuck(self, position: List[float]) -> bool:
        """Detect a frozen drone that reports the same position every step despite commands."""
        if not hasattr(self, "_last_observed_positions"):
            self._last_observed_positions = []
        self._last_observed_positions.append(tuple(position))
        # Keep only the recent window
        if len(self._last_observed_positions) > self.STUCK_THRESHOLD_STEPS:
            self._last_observed_positions = self._last_observed_positions[-self.STUCK_THRESHOLD_STEPS:]
        if len(self._last_observed_positions) < self.STUCK_THRESHOLD_STEPS:
            return False
        first = self._last_observed_positions[0]
        for p in self._last_observed_positions[1:]:
            if (abs(p[0] - first[0]) > self.STUCK_POSITION_TOLERANCE or
                    abs(p[1] - first[1]) > self.STUCK_POSITION_TOLERANCE or
                    abs(p[2] - first[2]) > self.STUCK_POSITION_TOLERANCE):
                return False
        return True

    def _drone_is_unresponsive(self, position: List[float], commanded_speed: float) -> bool:
        """Detect a drone that's barely moving despite the agent commanding real velocity.

        Catches the stale-EKF case where position reads jitter slightly but the drone
        isn't actually tracking commanded velocity (e.g. lying on the floor while the
        log is frozen-ish).
        """
        if not hasattr(self, "_unresponsive_history"):
            self._unresponsive_history = []
        self._unresponsive_history.append((tuple(position), commanded_speed))
        if len(self._unresponsive_history) > self.UNRESPONSIVE_WINDOW:
            self._unresponsive_history = self._unresponsive_history[-self.UNRESPONSIVE_WINDOW:]
        if len(self._unresponsive_history) < self.UNRESPONSIVE_WINDOW:
            return False

        # Require that the agent was trying to move on most of those steps
        moving_commands = sum(1 for _, s in self._unresponsive_history
                              if s >= self.UNRESPONSIVE_MIN_COMMANDED_SPEED)
        if moving_commands < self.UNRESPONSIVE_WINDOW - 2:
            return False

        # Check that all positions stayed within a tight radius of the first
        first_pos, _ = self._unresponsive_history[0]
        for pos, _ in self._unresponsive_history[1:]:
            dx = pos[0] - first_pos[0]
            dy = pos[1] - first_pos[1]
            if math.sqrt(dx * dx + dy * dy) > self.UNRESPONSIVE_POSITION_RADIUS:
                return False
        return True

    def _reset_stuck_tracker(self):
        self._last_observed_positions = []
        self._unresponsive_history = []

    def step(self, action):
        self.total_steps += 1

        if self.total_steps == self.exploration_steps and not self.learning:
            print("\nSWITCHING TO LEARNING PHASE...\n")
            self.truncate_next = True
            self.learning = True

        if self.learning:
            assert len(action) == 3, 'action should be length 3'
            processed_action = [action[0], action[1], 0]
        else:
            processed_action = [action[0] * 2 - 1, action[1] * 2 - 1, 0]

        # Detect dead/frozen/unresponsive drone and recover before commanding action.
        # The commanded speed is computed from the processed action so we can tell
        # whether the agent was actually trying to move when checking unresponsiveness.
        evader_pos = self.drone.get_position()
        commanded_speed = math.sqrt(
            (processed_action[0] * self.max_velocity) ** 2 +
            (processed_action[1] * self.max_velocity) ** 2
        )

        is_dead = self._drone_is_dead(evader_pos)
        is_stuck = self._drone_is_stuck(evader_pos)
        is_unresponsive = self._drone_is_unresponsive(evader_pos, commanded_speed)

        if is_dead or is_stuck or is_unresponsive:
            reason = "dead" if is_dead else ("stuck" if is_stuck else "unresponsive")
            print(f"[EvadePursuers2DParticle] Drone appears {reason} (position={evader_pos}) — restarting")
            self.restart()
            self._reset_stuck_tracker()
            self.truncate_next = True
            evader_pos = self.drone.get_position()

        # Update pursuer velocities based on current evader position
        self._update_pursuer_velocities(evader_pos)

        # Per-axis boundary clamp: only block velocity components that would push
        # the drone FURTHER out of bounds. Inward motion is always allowed so a
        # drone that has drifted past the boundary can return to the play area.
        position = self.drone.get_position()
        time_step = self.step_time + self.time_tolerance

        clamped_action = list(processed_action)

        predicted_x = position[0] + time_step * clamped_action[0] * self.max_velocity
        if predicted_x > self.xy_limit and clamped_action[0] > 0:
            clamped_action[0] = 0.0
        elif predicted_x < -self.xy_limit and clamped_action[0] < 0:
            clamped_action[0] = 0.0

        predicted_y = position[1] + time_step * clamped_action[1] * self.max_velocity
        if predicted_y > self.xy_limit and clamped_action[1] > 0:
            clamped_action[1] = 0.0
        elif predicted_y < -self.xy_limit and clamped_action[1] < 0:
            clamped_action[1] = 0.0

        return super().step(clamped_action)

    MAX_CONSECUTIVE_RESTART_ATTEMPTS = 5
    RESTART_TIMEOUT_SECONDS = 30  # hard ceiling so cflib's no-timeout open_link can't hang us

    def restart(self):
        """Auto-restart without user input — needed for unattended overnight runs.

        cflib's open_link() blocks indefinitely if the simulator is dead, so we
        run the whole restart sequence in a daemon thread and abandon it if it
        doesn't complete within RESTART_TIMEOUT_SECONDS. After
        MAX_CONSECUTIVE_RESTART_ATTEMPTS consecutive failures, we stop trying.
        """
        if not hasattr(self, "_consecutive_restart_failures"):
            self._consecutive_restart_failures = 0

        if self._consecutive_restart_failures >= self.MAX_CONSECUTIVE_RESTART_ATTEMPTS:
            print("[EvadePursuers2DParticle] Skipping restart — too many consecutive failures; "
                  "the simulator may need a manual restart")
            return False

        print("[EvadePursuers2DParticle] Auto-restarting drone (no user input required)")

        done_event = threading.Event()
        success_holder = [False]

        def _do_restart():
            try:
                self.drone.pre_battery_change_cleanup()
                time.sleep(2)
                self.drone.initialise_crazyflie()
                self.drone.take_off()
                if self.drone.is_flying_event.wait(timeout=15):
                    success_holder[0] = True
            except Exception as exc:
                print(f"[EvadePursuers2DParticle] Restart exception: {exc}")
            finally:
                done_event.set()

        thread = threading.Thread(target=_do_restart, daemon=True)
        thread.start()

        if not done_event.wait(timeout=self.RESTART_TIMEOUT_SECONDS):
            print(f"[EvadePursuers2DParticle] Restart timed out after {self.RESTART_TIMEOUT_SECONDS}s "
                  "— simulator may be dead, abandoning attempt (background thread leaked)")
            self._consecutive_restart_failures += 1
            return False

        if success_holder[0]:
            self._consecutive_restart_failures = 0
            return True

        print("[EvadePursuers2DParticle] WARNING: drone did not confirm takeoff after restart")
        self._consecutive_restart_failures += 1
        return False

    def _reset_task_state(self):
        self.done = False
        self.caught = False

    def _get_state(self) -> np.ndarray:
        position = self.drone.get_position()
        vel_x = self.drone.calculated_velocity["x"]
        vel_y = self.drone.calculated_velocity["y"]

        state: List[float] = [
            position[0] / self.xy_limit,
            position[1] / self.xy_limit,
            vel_x / self.max_velocity,
            vel_y / self.max_velocity,
        ]

        with self._particle_lock:
            pursuer_snapshot = [(p[:], v[:]) for p, v in zip(self.pursuer_positions, self.pursuer_velocities)]

        for px_py_pz, pvx_pvy_pvz in pursuer_snapshot:
            px, py, _ = px_py_pz
            pvx, pvy, _ = pvx_pvy_pvz

            rel_x = px - position[0]
            rel_y = py - position[1]
            distance = math.sqrt(rel_x ** 2 + rel_y ** 2)
            dir_x = rel_x / (distance + 1e-6)
            dir_y = rel_y / (distance + 1e-6)

            state += [
                rel_x / self.max_xy_range,
                rel_y / self.max_xy_range,
                distance / self.max_distance,
                dir_x,
                dir_y,
                pvx / (self.pursuer_max_velocity + 1e-6),
                pvy / (self.pursuer_max_velocity + 1e-6),
            ]

        return np.array(state, dtype=np.float32)

    def get_overlay_info(self) -> Dict[str, Any]:
        position = self.drone.get_position()
        with self._particle_lock:
            snap = [p[:] for p in self.pursuer_positions]
        return {
            'position': position,
            'pursuer_positions': snap,
            'min_distance_to_pursuer': self._distance_to_target(position),
            'caught': self.caught,
            'done': self.done,
        }

    def _distance_to_target(self, position: List[float]) -> float:
        with self._particle_lock:
            snap = [p[:] for p in self.pursuer_positions]
        if not snap:
            return self.max_distance
        return min(
            math.sqrt((position[0] - p[0]) ** 2 + (position[1] - p[1]) ** 2)
            for p in snap
        )

    def _is_out_of_task_bounds(self, position: List[float]) -> bool:
        """Check if the drone has left the task's xy boundary (with small tolerance for PID overshoot)."""
        tolerance = 0.05
        return (abs(position[0]) > self.xy_limit + tolerance or
                abs(position[1]) > self.xy_limit + tolerance)

    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        # Out-of-bounds is a terminal failure — heavy penalty so the agent never
        # finds it attractive to camp outside the boundary
        if self._is_out_of_task_bounds(current_state['position']):
            self.previous_distance = current_state['distance_to_target']
            return self.out_of_bounds_penalty

        min_distance = current_state['distance_to_target']

        if min_distance < self.capture_threshold:
            self.previous_distance = min_distance
            return self.caught_penalty

        reward = self.survival_reward
        reward += (min_distance / self.max_distance) * self.distance_reward_multiplier
        self.previous_distance = min_distance
        return reward

    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        # Capture ends the episode
        if current_state['distance_to_target'] < self.capture_threshold:
            self.caught = True
            self.done = True
            return True
        # Going out of bounds ends the episode — reset will spawn the drone fresh
        if self._is_out_of_task_bounds(current_state['position']):
            self.exited_testing_boundary = True
            self.done = True
            return True
        return False

    def is_in_testing_zone(self):
        return self.is_in_boundaries()

    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        if self.steps >= self.episode_length:
            if self._is_evaluating and not self.caught:
                self.successful_episodes_count += 1
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
        with self._particle_lock:
            snap = [p[:] for p in self.pursuer_positions]
        info = {
            'pursuer_positions': snap,
            'min_distance_to_pursuer': current_state['distance_to_target'],
            'caught': self.caught,
            'survived': not self.caught,
            'out_of_bounds': not current_state['in_boundaries'],
            'description': "2D evasion task — drone evades virtual particle pursuers",
        }
        if self._is_evaluating:
            info['success_count'] = self.successful_episodes_count
        return info

    def sample_action(self):
        action = np.random.uniform(0, 1, size=(3,))
        action[2] = 0
        return action

    def close(self):
        self._stop_particle_thread()
        # Remove markers
        for i in range(self.n_pursuers):
            req = f'name: "{self._marker_name(i)}"'
            self._run_gz_service(
                service=f"/world/{self.WORLD_NAME}/remove",
                reqtype="gz.msgs.Entity",
                reptype="gz.msgs.Boolean",
                req=req,
            )
        super().close()

    def _render_task_specific_info(self):
        pos = self.drone.get_position()
        print(f"Evader Position: [{pos[0]:.2f}, {pos[1]:.2f}]")
        with self._particle_lock:
            for i, p in enumerate(self.pursuer_positions):
                d = math.sqrt((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2)
                print(f"Pursuer {i}: [{p[0]:.2f}, {p[1]:.2f}] dist={d:.2f}")
        print(f"Capture Threshold: {self.capture_threshold:.2f}")
        print(f"Caught: {self.caught}")

    def grab_frame(self, height: int = 540, width: int = 960) -> np.ndarray:
        fig = plt.figure(figsize=(width / 120, height / 120), dpi=120)

        if not self.episode_positions:
            plt.close(fig)
            return np.full((height, width, 3), 255, dtype=np.uint8)

        pos_array = np.array(self.episode_positions)
        x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])

        pursuer_colors = ['red', 'orange', 'purple', 'brown', 'magenta']

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax1.plot(x, y, z, label='Evader Path', color='yellow', linewidth=2.5)
        ax1.scatter(x[0], y[0], z[0], color='green', s=80, label='Start',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(x[-1], y[-1], z[-1], color='blue', s=80, label='Current',
                    depthshade=False, edgecolors='black', linewidth=0.5)

        with self._particle_lock:
            snap = [p[:] for p in self.pursuer_positions]

        for i, p in enumerate(snap):
            c = pursuer_colors[i % len(pursuer_colors)]
            ax1.scatter(p[0], p[1], p[2], color=c, marker='^', s=120,
                        label=f'Pursuer {i}', depthshade=False, edgecolors='black', linewidth=1)

        ax1.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_zlim(0.25, 1.5)
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

        ax2 = fig.add_subplot(gs[0, 1])
        boundary_x = [-self.xy_limit, self.xy_limit, self.xy_limit, -self.xy_limit, -self.xy_limit]
        boundary_y = [-self.xy_limit, -self.xy_limit, self.xy_limit, self.xy_limit, -self.xy_limit]
        ax2.plot(boundary_x, boundary_y, 'k--', linewidth=1, alpha=0.5, label='Boundary', zorder=1)
        ax2.plot(x, y, color='yellow', linewidth=2.5, label='Evader Path', zorder=2)
        ax2.scatter(x[0], y[0], color='green', s=80, label='Start',
                    edgecolors='black', linewidth=0.5, zorder=4)
        ax2.scatter(x[-1], y[-1], color='blue', s=80, label='Current',
                    edgecolors='black', linewidth=0.5, zorder=4)

        for i, p in enumerate(snap):
            c = pursuer_colors[i % len(pursuer_colors)]
            ax2.scatter(p[0], p[1], color=c, marker=MarkerStyle('^'), s=120, label=f'Pursuer {i}',
                        edgecolors='black', linewidth=1, zorder=5)
            circle = plt.Circle((p[0], p[1]), self.capture_threshold, color=c, alpha=0.15, zorder=1)
            ax2.add_patch(circle)

        ax2.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_title('Top-Down View (X-Y)', fontsize=12, pad=15)
        ax2.set_aspect('equal', adjustable='box')
        ax2.legend(loc='upper left', fontsize=6, framealpha=0.9, markerscale=0.60)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=8)

        fig.suptitle(f'Particle Evasion (Step {self.steps}) | Caught: {self.caught}', fontsize=13, y=0.98)
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
