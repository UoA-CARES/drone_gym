from matplotlib.markers import MarkerStyle
import numpy as np
import math
import time
from typing import Dict, List, Any, Literal
from drone_gym.drone_environment import DroneEnvironment
from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
import matplotlib.pyplot as plt
import io
import cv2


class EvadePursuers2D(DroneEnvironment):
    """2D evasion task — the learning drone evades n_pursuers real CrazyFlie drones.

    Each pursuer is a separate CrazyFlie SITL instance (or real drone) controlled by
    an expert algorithm via continuous velocity commands. All N+1 drones must be
    launched via sitl_multiagent_square.sh -n {1 + n_pursuers} before running.

    Episode ends (done=True) when any pursuer collides with the learning drone
    (distance < capture_threshold). Truncated when episode_length steps survived.
    Boundary hits zero the learning drone's velocity for that step (episode continues).
    """

    # Evader uses port 19850 (matches DroneSim default). Pursuers get 19851, 19852, ...
    PURSUER_PORT_BASE = 19851

    def __init__(self, use_simulator: Literal[0, 1], max_velocity: float = 0.25, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 40,
                 n_pursuers: int = 2, pursuer_max_velocity: float = 0.20):

        super().__init__(use_simulator, max_velocity, step_time)

        # RL Training parameters
        self.episode_length = episode_length
        self.exploration_steps = exploration_steps
        self.total_steps = 0
        self.truncate_next = False

        self.learning = True

        # Task-specific parameters
        self.n_pursuers = n_pursuers
        self.pursuer_max_velocity = pursuer_max_velocity
        self.capture_threshold = 0.20  # metres — pursuer catches evader
        self.spawn_min_distance = 1.0  # minimum start distance between evader and each pursuer
        self.spawn_margin = 0.1
        self.fixed_z = 1.0

        self.xy_limit = 2.0  # Boundary limit for x and y
        self.max_xy_range = self.xy_limit * 2  # Maximum range in x or y direction (for normalizing components)
        self.max_distance = np.sqrt(self.max_xy_range ** 2 + self.max_xy_range ** 2)  # Maximum distance for normalization
        self.time_tolerance = 0.15  # tolerance time for calculating travel distance

        # hard coded z limit
        self.boundary = [self.xy_limit, self.xy_limit, self.z_limit, self.z_limit + 1]

        # Observation space: own_pos(2) + own_vel(2) + per_pursuer(7) * n
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

        # Instantiate the pursuer drones (one DroneSim/Drone per pursuer, each on its own URI)
        # NOTE: DroneSim.__init__ opens a cflib connection synchronously — ensure the SITL
        # firmware instances for ports 19851, 19852, ... are already running.
        self.pursuer_drones: List[Any] = []
        for i in range(n_pursuers):
            if use_simulator:
                pursuer_uri = f"udp://0.0.0.0:{self.PURSUER_PORT_BASE + i}"
                print(f"[EvadePursuers2D] Connecting to pursuer {i} at {pursuer_uri}")
                self.pursuer_drones.append(DroneSim(uri=pursuer_uri))
            else:
                # Real-flight mode: a Drone() per pursuer. Each must have its radio URI
                # configured externally (e.g. via uri_helper / env var) — out of scope here.
                print(f"[EvadePursuers2D] Real-flight pursuer {i} — radio URI must be configured externally")
                self.pursuer_drones.append(Drone())

        # Local tracking (refreshed from real drone state each step)
        self.pursuer_positions: List[List[float]] = [[0.0, 0.0, self.fixed_z] for _ in range(n_pursuers)]
        self.pursuer_velocities: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(n_pursuers)]

        # First-call flag: take off the pursuers on the first reset
        self._pursuers_taken_off = False

        # Distance tracking for reward calculation
        self.previous_distance = self.max_distance

        # Evaluation mode tracking — counts episodes the drone survived
        self.successful_episodes_count = 0

    # ------------------------------------------------------------------
    # Expert pursuer policy — placeholder (Pure Pursuit)
    # Replace _compute_pursuer_velocity() with the expert algorithm.
    # ------------------------------------------------------------------

    def _compute_pursuer_velocity(self, pursuer_pos: List[float], evader_pos: List[float]) -> List[float]:
        """Compute a pursuer's velocity command [vx, vy, 0].

        Placeholder: Pure Pursuit (Isaacs 1965) — steer directly toward the evader
        at max speed. Swap this body with the chosen expert algorithm.
        """
        dx = evader_pos[0] - pursuer_pos[0]
        dy = evader_pos[1] - pursuer_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return [0.0, 0.0, 0.0]
        scale = self.pursuer_max_velocity / dist
        return [scale * dx, scale * dy, 0.0]

    # ------------------------------------------------------------------
    # Pursuer drone management
    # ------------------------------------------------------------------

    def _reset_pursuer_control_properties(self, pursuer):
        """Mirror DroneEnvironment._reset_control_properties for a pursuer drone."""
        pursuer.clear_command_queue()
        time.sleep(0.5)
        pursuer.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        pursuer.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        pursuer.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        pursuer.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        pursuer.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

    def _take_off_pursuers(self):
        """Send takeoff commands to all pursuers and wait for them to be flying."""
        print(f"[EvadePursuers2D] Taking off {self.n_pursuers} pursuers...")
        for pursuer in self.pursuer_drones:
            if not pursuer.is_flying_event.is_set():
                pursuer.take_off()
        # Wait sequentially so a slow pursuer doesn't get skipped
        for i, pursuer in enumerate(self.pursuer_drones):
            if not pursuer.is_flying_event.wait(timeout=15):
                print(f"[EvadePursuers2D] WARNING: pursuer {i} did not confirm takeoff")
        time.sleep(1)
        print("[EvadePursuers2D] All pursuers airborne")

    def _sample_pursuer_target_positions(self, evader_pos: List[float]) -> List[List[float]]:
        """Sample random (x, y, fixed_z) positions for each pursuer with min separation from evader."""
        limit = self.xy_limit - self.spawn_margin
        targets: List[List[float]] = []
        for i in range(self.n_pursuers):
            placed = False
            for _ in range(200):
                x = float(np.random.uniform(-limit, limit))
                y = float(np.random.uniform(-limit, limit))
                dist = math.sqrt((x - evader_pos[0]) ** 2 + (y - evader_pos[1]) ** 2)
                if dist >= self.spawn_min_distance:
                    targets.append([x, y, self.fixed_z])
                    placed = True
                    break
            if not placed:
                # Fallback: spread evenly on the boundary
                angle = i * (2 * math.pi / self.n_pursuers)
                targets.append([math.cos(angle) * limit, math.sin(angle) * limit, self.fixed_z])
        return targets

    def _fly_pursuers_to_positions(self, target_positions: List[List[float]]):
        """Use each pursuer's position controller to fly it to a target (x, y, z)."""
        # Stop velocity controllers, reset PID state, set targets
        for pursuer, target in zip(self.pursuer_drones, target_positions):
            if pursuer.velocity_controller_active:
                pursuer.stop_velocity_control()
            pursuer.set_velocity_vector(0, 0, 0)
            time.sleep(0.1)
            self._reset_pursuer_control_properties(pursuer)
            pursuer.set_target_position(target[0], target[1], target[2])
            pursuer.start_position_control()

        # Wait for all pursuers to arrive at their reset positions
        for i, pursuer in enumerate(self.pursuer_drones):
            if not pursuer.at_reset_position.wait(timeout=12):
                print(f"[EvadePursuers2D] WARNING: pursuer {i} timed out moving to reset position")

        time.sleep(0.5)

        # Switch to velocity control for the episode
        for pursuer in self.pursuer_drones:
            pursuer.stop_position_control()
            pursuer.clear_reset_position_event()
            pursuer.start_velocity_control()

    def _send_pursuer_velocity_commands(self, evader_pos: List[float]):
        """Read each pursuer's actual position and send the expert-policy velocity command."""
        for i, pursuer in enumerate(self.pursuer_drones):
            pos = pursuer.get_position()
            self.pursuer_positions[i] = list(pos)
            vx, vy, _ = self._compute_pursuer_velocity(pos, evader_pos)
            self.pursuer_velocities[i] = [vx, vy, 0.0]
            pursuer.set_velocity_vector(vx, vy, 0.0)

    def _refresh_pursuer_positions(self):
        """Update local position tracking from each pursuer's current state."""
        for i, pursuer in enumerate(self.pursuer_drones):
            self.pursuer_positions[i] = list(pursuer.get_position())

    # ------------------------------------------------------------------
    # DroneEnvironment overrides
    # ------------------------------------------------------------------

    def reset(self, training: bool = True):
        """Reset the learning drone and respawn the pursuers at random positions"""

        # Reset successful episodes count when starting evaluation
        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        # Call parent reset — takes the learning drone off and moves it to (0, 0, 1)
        state = super().reset(training)

        # First reset: take off the pursuer drones (subsequent resets just fly them around)
        if not self._pursuers_taken_off:
            self._take_off_pursuers()
            self._pursuers_taken_off = True

        # Sample fresh random spawn positions for each pursuer
        evader_pos = self.drone.get_position()
        target_positions = self._sample_pursuer_target_positions(evader_pos)

        # Fly each pursuer to its sampled position via position control, then switch to velocity control
        self._fly_pursuers_to_positions(target_positions)

        # Refresh local tracking after they have arrived
        self._refresh_pursuer_positions()

        self.caught = False
        self.previous_distance = self._distance_to_target(self.drone.get_position())

        return self._get_state()

    def step(self, action):
        """Execute one step with RL-specific logic for exploration vs learning phases"""

        # Handle exploration vs learning phase transition
        self.total_steps += 1

        if self.total_steps == self.exploration_steps and not self.learning:
            print("\n")
            print("SWITCHING TO LEARNING PHASE...")
            print("\n")
            self.truncate_next = True
            self.learning = True

        # Modify action normalization based on phase
        if self.learning:
            # Learning phase: action is already in [-1, 1]
            assert len(action) == 3, 'action should be length 3'
            processed_action = [action[0], action[1], 0]  # Add vz=0 to fit 3D action shape of drone_environment
        else:
            # Exploration phase: convert from [0, 1] to [-1, 1]
            processed_action = [action[0] * 2 - 1, action[1] * 2 - 1, 0]

        # Send pursuer velocity commands BEFORE super().step() so all drones fly
        # simultaneously during the step_time sleep inside the parent step
        evader_pos = self.drone.get_position()
        self._send_pursuer_velocity_commands(evader_pos)

        # determine whether not the action we pass will exceed the boundary
        position = self.drone.get_position()
        vx = processed_action[0] * self.max_velocity
        vy = processed_action[1] * self.max_velocity
        vz = 0

        time_step = self.step_time + self.time_tolerance
        sx = time_step * vx
        sy = time_step * vy
        sz = time_step * vz

        new_position = [sx + position[0], sy + position[1], sz + position[2]]
        print(f" new position is: {new_position}")
        # Check if new position would exceed boundaries — zero velocity for this step
        if (new_position[0] < -self.boundary[0] or new_position[0] > self.boundary[0] or
                new_position[1] < -self.boundary[1] or new_position[1] > self.boundary[1] or
                new_position[2] <= self.boundary[2] or new_position[2] > self.boundary[3]):
            result = super().step([0, 0, 0])
            self._refresh_pursuer_positions()
            return result

        # Call parent step method with processed action
        result = super().step(processed_action)

        # Refresh local tracking after the step has completed (pursuers have flown for step_time)
        self._refresh_pursuer_positions()

        return result

    def _reset_task_state(self):
        """Reset task-specific state variables"""
        self.done = False
        self.caught = False

    def _get_state(self) -> np.ndarray:
        """Get the current state representation for the 2D evasion task"""
        position = self.drone.get_position()

        vel_x = self.drone.calculated_velocity["x"]
        vel_y = self.drone.calculated_velocity["y"]

        state: List[float] = [
            # Evader normalised position (2)
            position[0] / self.xy_limit,
            position[1] / self.xy_limit,

            # Evader normalised velocity (2)
            vel_x / self.max_velocity,
            vel_y / self.max_velocity,
        ]

        # Per-pursuer block (7 each): relative pos (2), distance (1), direction to pursuer (2), pursuer velocity (2)
        for i in range(self.n_pursuers):
            px = self.pursuer_positions[i][0]
            py = self.pursuer_positions[i][1]

            rel_x = px - position[0]
            rel_y = py - position[1]
            distance = math.sqrt(rel_x ** 2 + rel_y ** 2)

            dir_x = rel_x / (distance + 1e-6)
            dir_y = rel_y / (distance + 1e-6)

            pvx = self.pursuer_velocities[i][0]
            pvy = self.pursuer_velocities[i][1]

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
        """Get task-specific state information"""
        position = self.drone.get_position()
        return {
            'position': position,
            'pursuer_positions': [p[:] for p in self.pursuer_positions],
            'min_distance_to_pursuer': self._distance_to_target(position),
            'caught': self.caught,
            'done': self.done,
        }

    def _distance_to_target(self, position: List[float]) -> float:
        """Distance to the nearest pursuer (used by base class info dict)"""
        if not self.pursuer_positions:
            return self.max_distance
        return min(
            math.sqrt((position[0] - p[0]) ** 2 + (position[1] - p[1]) ** 2)
            for p in self.pursuer_positions
        )

    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Reward = survival bonus + distance bonus, with large penalty on capture"""

        min_distance = current_state['distance_to_target']

        # Capture is terminal — large penalty
        if min_distance < self.capture_threshold:
            self.previous_distance = min_distance
            return self.caught_penalty

        # Survived this step
        reward = self.survival_reward

        # Encourage staying far from nearest pursuer (normalised 0-1 range)
        reward += (min_distance / self.max_distance) * self.distance_reward_multiplier

        # Update tracking
        self.previous_distance = min_distance

        return reward

    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Episode ends as failure when any pursuer collides with the drone"""

        min_distance = current_state['distance_to_target']

        if min_distance < self.capture_threshold:
            self.caught = True
            self.done = True
            return True

        return False

    def is_in_testing_zone(self):
        """Check if drone is in the testing zone (task-specific boundary logic)"""
        return self.is_in_boundaries()

    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode should be truncated"""

        if self.steps >= self.episode_length:
            # Survived the full episode — count it during evaluation
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
        """Get additional task-specific info"""
        info = {
            'pursuer_positions': [p[:] for p in self.pursuer_positions],
            'min_distance_to_pursuer': current_state['distance_to_target'],
            'caught': self.caught,
            'survived': not self.caught,
            'out_of_bounds': not current_state['in_boundaries'],
            'description': "2D evasion task — drone evades expert CrazyFlie pursuers",
        }

        # Add success count during evaluation
        if self._is_evaluating:
            info['success_count'] = self.successful_episodes_count

        return info

    def sample_action(self):
        """Sample an action for exploration phase - returns action in [0, 1] range"""
        action = np.random.uniform(0, 1, size=(3,))
        action[2] = 0  # Set z-action to 0 for 2D movement
        return action

    def close(self):
        """Land every drone (evader + pursuers) and shut down their threads."""
        # Land pursuers first so they don't crash during the evader landing
        for i, pursuer in enumerate(self.pursuer_drones):
            try:
                if pursuer.velocity_controller_active:
                    pursuer.stop_velocity_control()
                pursuer.set_velocity_vector(0, 0, 0)
                pursuer.land()
            except Exception as e:
                print(f"[EvadePursuers2D] Error landing pursuer {i}: {e}")

        for i, pursuer in enumerate(self.pursuer_drones):
            try:
                pursuer.is_landed_event.wait(timeout=30)
            except Exception:
                pass
            try:
                pursuer.stop()
            except Exception as e:
                print(f"[EvadePursuers2D] Error stopping pursuer {i}: {e}")

        # Land the evader (parent close)
        super().close()

    def _render_task_specific_info(self):
        """Render evasion task specific information"""
        pos = self.drone.get_position()

        print(f"Evader Position: [{pos[0]:.2f}, {pos[1]:.2f}]")
        for i, p in enumerate(self.pursuer_positions):
            d = math.sqrt((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2)
            print(f"Pursuer {i}: [{p[0]:.2f}, {p[1]:.2f}] dist={d:.2f}")
        print(f"Capture Threshold: {self.capture_threshold:.2f}")
        print(f"Caught: {self.caught}")

    def grab_frame(self, height: int = 540, width: int = 960) -> np.ndarray:
        # Create figure with two subplots side by side
        fig = plt.figure(figsize=(width / 120, height / 120), dpi=120)

        # Return white frame if no positions recorded yet
        if not self.episode_positions:
            plt.close(fig)
            return np.full((height, width, 3), 255, dtype=np.uint8)

        # Convert positions to numpy array
        pos_array = np.array(self.episode_positions)
        x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]

        # Use GridSpec with equal widths and minimal spacing
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, figure=fig, wspace=0.25, width_ratios=[1, 1])

        pursuer_colors = ['red', 'orange', 'purple', 'brown', 'magenta']

        # LEFT SUBPLOT: 3D trajectory view
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')

        # Plot the drone's trajectory
        ax1.plot(x, y, z, label='Evader Path', color='yellow', linewidth=2.5)

        # Mark important points with better visibility
        ax1.scatter(x[0], y[0], z[0], color='green', s=80, label='Start',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(x[-1], y[-1], z[-1], color='blue', s=80, label='Current',
                    depthshade=False, edgecolors='black', linewidth=0.5)

        for i, p in enumerate(self.pursuer_positions):
            c = pursuer_colors[i % len(pursuer_colors)]
            ax1.scatter(p[0], p[1], p[2], color=c, marker='^', s=120,
                        label=f'Pursuer {i}', depthshade=False, edgecolors='black', linewidth=1)

        ax1.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax1.set_zlim(0.25, 1.5)

        # Labels and title
        ax1.set_xlabel('X (m)', fontsize=10, labelpad=8)
        ax1.set_ylabel('Y (m)', fontsize=10, labelpad=8)
        ax1.set_zlabel('Z (m)', fontsize=9, labelpad=10)

        # Adjust tick parameters
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.tick_params(axis='z', labelsize=8)

        # Viewing angle
        ax1.view_init(elev=10, azim=25)

        # Title
        ax1.set_title('3D Trajectory', fontsize=12, pad=15)

        # Legend
        ax1.legend(loc='upper left', fontsize=6, framealpha=0.9, markerscale=0.60)

        # Grid
        ax1.grid(True, alpha=0.3)

        # Set equal aspect ratio for 3D plot
        ax1.set_box_aspect([1, 1, 0.67])

        # RIGHT SUBPLOT: 2D X-Y view (top-down view)
        ax2 = fig.add_subplot(gs[0, 1])

        # Boundary box
        boundary_x = [-self.xy_limit, self.xy_limit, self.xy_limit, -self.xy_limit, -self.xy_limit]
        boundary_y = [-self.xy_limit, -self.xy_limit, self.xy_limit, self.xy_limit, -self.xy_limit]
        ax2.plot(boundary_x, boundary_y, 'k--', linewidth=1, alpha=0.5, label='Boundary', zorder=1)

        # Plot the drone's trajectory in X-Y plane
        ax2.plot(x, y, color='yellow', linewidth=2.5, label='Evader Path', zorder=2)

        # Mark important points
        ax2.scatter(x[0], y[0], color='green', s=80, label='Start',
                    edgecolors='black', linewidth=0.5, zorder=4)
        ax2.scatter(x[-1], y[-1], color='blue', s=80, label='Current',
                    edgecolors='black', linewidth=0.5, zorder=4)

        for i, p in enumerate(self.pursuer_positions):
            c = pursuer_colors[i % len(pursuer_colors)]
            ax2.scatter(p[0], p[1], color=c, marker=MarkerStyle('^'), s=120, label=f'Pursuer {i}',
                        edgecolors='black', linewidth=1, zorder=5)
            # Draw capture radius
            circle = plt.Circle((p[0], p[1]), self.capture_threshold, color=c, alpha=0.15, zorder=1)
            ax2.add_patch(circle)

        ax2.set_xlim(-self.xy_limit - 0.2, self.xy_limit + 0.2)
        ax2.set_ylim(-self.xy_limit - 0.2, self.xy_limit + 0.2)

        # Labels and title
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_title('Top-Down View (X-Y)', fontsize=12, pad=15)

        # Equal aspect ratio for accurate representation
        ax2.set_aspect('equal', adjustable='box')

        # Legend
        ax2.legend(loc='upper left', fontsize=6, framealpha=0.9, markerscale=0.60)

        # Grid
        ax2.grid(True, alpha=0.3)

        # Tick parameters
        ax2.tick_params(axis='both', labelsize=8)

        # Add main title at the top
        fig.suptitle(f'Evasion Episode (Step {self.steps}) | Caught: {self.caught}', fontsize=13, y=0.98)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Convert matplotlib figure to image array with higher quality
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120,
                    facecolor='white', edgecolor='none', bbox_inches='tight')
        buf.seek(0)

        # Decode the PNG buffer to numpy array
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)

        # Use cv2 to decode and resize
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            # Only resize if necessary
            current_h, current_w = frame.shape[:2]
            if current_h != height or current_w != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            # Convert BGR to RGB for consistency
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to white frame if decode fails
            frame = np.full((height, width, 3), 255, dtype=np.uint8)

        return frame
