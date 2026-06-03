from matplotlib.markers import MarkerStyle
import numpy as np
import math
import time
import threading
from typing import Dict, List, Any, Literal
from drone_gym.drone_environment import DroneEnvironment
from drone_gym.sim_agent_manager import SimAgentManager
from drone_gym.agents.ticker import AgentTicker
import matplotlib.pyplot as plt
import io
import cv2


class InterceptEvader2DParticle(DroneEnvironment):
    """2D pursuit task — the learning drone must CATCH a fleeing particle target.

    The learner is the predator (the env's own Crazyflie). The target is a
    virtual particle (a Gazebo marker, no physics) driven by an expert FleePolicy
    that runs directly away from the learner and slides along the boundary when
    cornered. Modelling the target as a particle keeps capture collision-safe:
    "catch" is a distance threshold, never a real drone-on-drone impact.

    The target is pulled from a SimAgentManager as a ``particle_evader`` agent,
    so only the learner is a real Crazyflie — launch with the single-agent
    script (e.g. sitl_singleagent.sh -m crazyflie -x 0 -y 0).

    Episode ends (done=True, success) the moment the learner gets within
    capture_threshold of the target. Truncated when episode_length steps elapse
    without a catch.
    """

    PARTICLE_UPDATE_HZ = 5.0  # background ticker rate for smooth target motion

    def __init__(self, use_simulator: Literal[0, 1], max_velocity: float = 0.25, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 60,
                 target_max_velocity: float = 0.13):

        super().__init__(use_simulator, max_velocity, step_time)

        # RL Training parameters
        self.episode_length = episode_length
        self.exploration_steps = exploration_steps
        self.total_steps = 0
        self.truncate_next = False

        self.learning = True

        # Task-specific parameters
        self.target_max_velocity = target_max_velocity  # kept well below the learner's so it is catchable
        self.capture_threshold = 0.30   # metres — learner catches the target
        self.spawn_min_distance = 1.0   # minimum start distance between learner and target
        self.spawn_margin = 0.1
        self.target_spawn_margin = 0.5  # keep the particle clear of the boundary at spawn
        self.fixed_z = 1.0

        self.xy_limit = 2.0  # Boundary limit for x and y
        self.max_xy_range = self.xy_limit * 2
        self.max_distance = np.sqrt(self.max_xy_range ** 2 + self.max_xy_range ** 2)
        self.time_tolerance = 0.15

        # hard coded z limit
        self.boundary = [self.xy_limit, self.xy_limit, self.z_limit, self.z_limit + 1]

        # Observation space: own_pos(2) + own_vel(2) + target block(7)
        self.observation_space = 4 + 7

        # Reward parameters
        self.success_reward = 100.0                    # big terminal bonus — catching is clearly best
        self.distance_improvement_multiplier = 100.0   # main signal: reward closing the gap
        self.step_penalty = 1.0                        # small per-step cost — catch FAST
        self.out_of_bounds_penalty = -100.0

        # Task state
        self.done = False
        self.caught = False  # True when the learner has caught the target
        self.boundary_penalise = False
        self.exited_testing_boundary = False

        # The fleeing target is a particle agent pulled from the manager. The
        # FleePolicy (the expert) runs it away from the learner each step.
        self.agent_manager = SimAgentManager(use_simulator)
        self.target_agent = self.agent_manager.create_agent(
            "particle_evader", role="evader",
            max_velocity=self.target_max_velocity, fixed_z=self.fixed_z,
            bounds=self.xy_limit, render=True,
        )

        # Local tracking, synced from the target agent (used by observation/reward/render)
        self.target_position: List[float] = [0.0, 0.0, self.fixed_z]
        self.target_velocity: List[float] = [0.0, 0.0, 0.0]

        # Background ticker integrates the particle continuously between RL steps
        # for smooth motion + Gazebo marker updates. The manager itself stays a
        # pure factory; this opt-in helper owns the thread.
        self._ticker = AgentTicker([self.target_agent], hz=self.PARTICLE_UPDATE_HZ)

        # Distance tracking for reward calculation
        self.previous_distance = self.max_distance

        # Evaluation mode tracking — counts episodes the learner caught the target
        self.successful_episodes_count = 0

    # ------------------------------------------------------------------
    # Target tracking
    # ------------------------------------------------------------------

    def _sync_target(self):
        """Copy the target agent's position/velocity into local tracking."""
        self.target_position = list(self.target_agent.position)
        self.target_velocity = list(self.target_agent.velocity)

    def _sample_target_spawn(self, learner_pos: List[float]) -> List[float]:
        """Sample a target spawn within the boundary, at least spawn_min_distance away."""
        limit = self.xy_limit - self.target_spawn_margin
        for _ in range(200):
            x = float(np.random.uniform(-limit, limit))
            y = float(np.random.uniform(-limit, limit))
            if math.sqrt((x - learner_pos[0]) ** 2 + (y - learner_pos[1]) ** 2) >= self.spawn_min_distance:
                return [x, y, self.fixed_z]
        # Fallback: opposite corner from the learner
        return [-np.sign(learner_pos[0]) * limit, -np.sign(learner_pos[1]) * limit, self.fixed_z]

    # ------------------------------------------------------------------
    # Dead / stuck learner recovery (unattended-training safety)
    #
    # Earlier runs were corrupted when the learner froze (EKF blow-up / sim
    # death): its position stopped changing and every later episode scored ~-800.
    # We detect that and auto-restart the drone instead of training on garbage.
    # ------------------------------------------------------------------

    STUCK_POSITION_TOLERANCE = 1e-4   # positions within this count as identical
    STUCK_THRESHOLD_STEPS = 6         # identical positions in a row before "stuck"
    MAX_CONSECUTIVE_RESTART_ATTEMPTS = 5
    RESTART_TIMEOUT_SECONDS = 30      # hard ceiling so a hung open_link can't block us

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

    def restart(self):
        """Auto-restart the learner with no user input (overrides the interactive base).

        open_link() can block indefinitely if the simulator is dead, so the whole
        sequence runs in a daemon thread we abandon after RESTART_TIMEOUT_SECONDS.
        After MAX_CONSECUTIVE_RESTART_ATTEMPTS failures we stop trying.
        """
        if not hasattr(self, "_consecutive_restart_failures"):
            self._consecutive_restart_failures = 0

        if self._consecutive_restart_failures >= self.MAX_CONSECUTIVE_RESTART_ATTEMPTS:
            print("[InterceptEvader2DParticle] Skipping restart — too many consecutive "
                  "failures; the simulator may need a manual restart")
            return False

        print("[InterceptEvader2DParticle] Auto-restarting drone (no user input required)")

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
                print(f"[InterceptEvader2DParticle] Restart exception: {exc}")
            finally:
                done_event.set()

        thread = threading.Thread(target=_do_restart, daemon=True)
        thread.start()

        if not done_event.wait(timeout=self.RESTART_TIMEOUT_SECONDS):
            print(f"[InterceptEvader2DParticle] Restart timed out after "
                  f"{self.RESTART_TIMEOUT_SECONDS}s — abandoning attempt")
            self._consecutive_restart_failures += 1
            return False

        if success_holder[0]:
            self._consecutive_restart_failures = 0
            return True

        print("[InterceptEvader2DParticle] WARNING: drone did not confirm takeoff after restart")
        self._consecutive_restart_failures += 1
        return False

    # ------------------------------------------------------------------
    # DroneEnvironment overrides
    # ------------------------------------------------------------------

    def reset(self, training: bool = True):
        """Reset the learner (via parent) and teleport the fleeing target to a fresh spawn."""

        # Reset successful episodes count when starting evaluation
        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        self._reset_stuck_tracker()

        # Parent reset: takeoff, move learner to (0, 0, 1), start velocity control.
        # We keep the learner at the centre (no long-range random move) to avoid
        # stressing the EKF — episode diversity comes from the random target spawn.
        super().reset(training)

        # Pause smooth integration while we teleport the target to its spawn
        self._ticker.stop()

        # Teleport the particle target to a random spawn away from the learner
        learner_pos = self.drone.get_position()
        spawn = self._sample_target_spawn(learner_pos)
        self.target_agent.reset_policy({"threat_pos": learner_pos})
        self.target_agent.prepare_reset(spawn)
        self.target_agent.refresh()
        self._sync_target()

        self.caught = False
        self.done = False
        self.previous_distance = self._distance_to_target(self.drone.get_position())

        # Resume smooth background motion for the episode
        self._ticker.start()

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
            processed_action = [action[0], action[1], 0]
        else:
            # Exploration phase: convert from [0, 1] to [-1, 1]
            processed_action = [action[0] * 2 - 1, action[1] * 2 - 1, 0]

        # Detect a dead/frozen learner (EKF blow-up, emergency landing) and
        # recover before stepping — otherwise the episode just produces garbage
        # (this is what corrupted the tail of earlier runs).
        learner_pos = self.drone.get_position()
        if self._drone_is_dead(learner_pos) or self._drone_is_stuck(learner_pos):
            reason = "dead" if self._drone_is_dead(learner_pos) else "stuck"
            print(f"[InterceptEvader2DParticle] Learner appears {reason} "
                  f"(pos={learner_pos}) — restarting")
            self.restart()
            self._reset_stuck_tracker()
            self.truncate_next = True
            learner_pos = self.drone.get_position()

        # Recompute the fleeing target's velocity from the learner's current
        # position. The frame-to-frame motion and Gazebo marker update happen
        # continuously on the background ticker, so we don't integrate here — we
        # just snapshot the current target position for this step's reward.
        self.target_agent.act({"threat_pos": learner_pos})
        self.target_agent.refresh()
        self._sync_target()

        # determine whether the action we pass will exceed the boundary
        position = self.drone.get_position()
        vx = processed_action[0] * self.max_velocity
        vy = processed_action[1] * self.max_velocity
        vz = 0

        time_step = self.step_time + self.time_tolerance
        sx = time_step * vx
        sy = time_step * vy
        sz = time_step * vz

        new_position = [sx + position[0], sy + position[1], sz + position[2]]
        # Check if new position would exceed boundaries — zero velocity for this step
        if (new_position[0] < -self.boundary[0] or new_position[0] > self.boundary[0] or
                new_position[1] < -self.boundary[1] or new_position[1] > self.boundary[1] or
                new_position[2] <= self.boundary[2] or new_position[2] > self.boundary[3]):
            result = super().step([0, 0, 0])
            # Target kept fleeing during the step (background ticker) — resync
            self.target_agent.refresh()
            self._sync_target()
            return result

        # Call parent step method with processed action (target keeps fleeing
        # during the step_time sleep via the background ticker)
        result = super().step(processed_action)
        self.target_agent.refresh()
        self._sync_target()
        return result

    def _reset_task_state(self):
        """Reset task-specific state variables"""
        self.done = False
        self.caught = False

    def _get_state(self) -> np.ndarray:
        """Get the current state representation for the 2D pursuit task"""
        position = self.drone.get_position()

        vel_x = self.drone.calculated_velocity["x"]
        vel_y = self.drone.calculated_velocity["y"]

        state: List[float] = [
            # Learner normalised position (2)
            position[0] / self.xy_limit,
            position[1] / self.xy_limit,

            # Learner normalised velocity (2)
            vel_x / self.max_velocity,
            vel_y / self.max_velocity,
        ]

        # Target block (7): relative pos (2), distance (1), direction to target (2), target velocity (2)
        tx = self.target_position[0]
        ty = self.target_position[1]

        rel_x = tx - position[0]
        rel_y = ty - position[1]
        distance = math.sqrt(rel_x ** 2 + rel_y ** 2)

        dir_x = rel_x / (distance + 1e-6)
        dir_y = rel_y / (distance + 1e-6)

        tvx = self.target_velocity[0]
        tvy = self.target_velocity[1]

        state += [
            rel_x / self.max_xy_range,
            rel_y / self.max_xy_range,
            distance / self.max_distance,
            dir_x,
            dir_y,
            tvx / (self.target_max_velocity + 1e-6),
            tvy / (self.target_max_velocity + 1e-6),
        ]

        return np.array(state, dtype=np.float32)

    def get_overlay_info(self) -> Dict[str, Any]:
        """Get task-specific state information"""
        position = self.drone.get_position()
        return {
            'position': position,
            'target_position': self.target_position[:],
            'distance_to_target': self._distance_to_target(position),
            'caught': self.caught,
            'done': self.done,
        }

    def _distance_to_target(self, position: List[float]) -> float:
        """2D Euclidean distance from the learner to the target (x, y only)"""
        return math.sqrt(
            (position[0] - self.target_position[0]) ** 2 +
            (position[1] - self.target_position[1]) ** 2
        )

    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Reward = progress toward the target + small time cost, big bonus on capture"""

        distance = current_state['distance_to_target']

        # Capture is terminal — large success reward, unambiguously the best outcome
        if distance < self.capture_threshold:
            self.previous_distance = distance
            return self.success_reward

        # Main signal: reward closing the gap to the target
        distance_improvement = self.previous_distance - distance
        reward = distance_improvement * self.distance_improvement_multiplier

        # Small per-step cost so the agent is rewarded for catching FAST rather
        # than hovering near the target (this is what fixes the ~1.4 m plateau)
        reward -= self.step_penalty

        self.previous_distance = distance

        return reward

    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Episode ends as success when the learner catches the target"""

        distance = current_state['distance_to_target']

        if distance < self.capture_threshold:
            self.caught = True
            self.done = True
            # Count the catch during evaluation
            if self._is_evaluating:
                self.successful_episodes_count += 1
            return True

        return False

    def is_in_testing_zone(self):
        """Check if drone is in the testing zone (task-specific boundary logic)"""
        return self.is_in_boundaries()

    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode should be truncated"""

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
        """Get additional task-specific info"""
        info = {
            'target_position': self.target_position[:],
            'distance_to_target': current_state['distance_to_target'],
            'caught': self.caught,
            'success': self.caught,
            'out_of_bounds': not current_state['in_boundaries'],
            'description': "2D pursuit task — drone catches a fleeing particle target",
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
        """Stop the background ticker, despawn the target marker and land the learner."""
        self._ticker.stop()
        self.agent_manager.close_all()
        super().close()

    def _render_task_specific_info(self):
        """Render pursuit task specific information"""
        pos = self.drone.get_position()
        d = self._distance_to_target(pos)

        print(f"Predator Position: [{pos[0]:.2f}, {pos[1]:.2f}]")
        print(f"Target Position: [{self.target_position[0]:.2f}, {self.target_position[1]:.2f}]")
        print(f"Distance to Target: {d:.2f}")
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

        tx, ty, tz = self.target_position[0], self.target_position[1], self.target_position[2]

        # LEFT SUBPLOT: 3D trajectory view
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')

        # Plot the predator's trajectory
        ax1.plot(x, y, z, label='Predator Path', color='yellow', linewidth=2.5)

        # Mark important points with better visibility
        ax1.scatter(x[0], y[0], z[0], color='green', s=80, label='Start',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(x[-1], y[-1], z[-1], color='blue', s=80, label='Current',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(tx, ty, tz, color='red', marker='*', s=140, label='Target',
                    depthshade=False, edgecolors='black', linewidth=1)

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

        # Plot the predator's trajectory in X-Y plane
        ax2.plot(x, y, color='yellow', linewidth=2.5, label='Predator Path', zorder=2)

        # Mark important points
        ax2.scatter(x[0], y[0], color='green', s=80, label='Start',
                    edgecolors='black', linewidth=0.5, zorder=4)
        ax2.scatter(x[-1], y[-1], color='blue', s=80, label='Current',
                    edgecolors='black', linewidth=0.5, zorder=4)
        ax2.scatter(tx, ty, color='red', marker=MarkerStyle('*'), s=140, label='Target',
                    edgecolors='black', linewidth=1, zorder=5)

        # Draw capture radius around the target
        circle = plt.Circle((tx, ty), self.capture_threshold, color='red', alpha=0.15, zorder=1)
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
        fig.suptitle(f'Pursuit Episode (Step {self.steps}) | Caught: {self.caught}', fontsize=13, y=0.98)

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
