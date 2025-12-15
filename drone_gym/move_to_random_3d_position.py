import numpy as np
import math
import time
from typing import Dict, List, Any
from drone_gym.drone_environment import DroneEnvironment
import matplotlib.pyplot as plt
import io
import cv2


class MoveToRandom3DPosition(DroneEnvironment):
    """Reinforcement learning task for drone navigation to a target position"""

    def __init__(self, max_velocity: float = 0.20, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 40,
                 x_range: List[float] = [-1.0, 1.0],
                 y_range: List[float] = [-1.0, 1.0],
                 z_range: List[float] = [0.5, 1.5]):

        super().__init__(max_velocity, step_time)
        

        # Store ranges
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        # RL Training parameters
        self.episode_length = episode_length
        self.exploration_steps = exploration_steps
        self.total_steps = 0
        self.truncate_next = False

        self.learning = True

        # Task-specific parameters
        self.goal_position = [0.2, 0.9, 0.7]  # Goal position
        self.distance_threshold = 0.05  # Distance threshold to consider target reached
        self.max_xy_range = 2.0  # Maximum range in x or y direction (for normalizing components)
        self.max_distance = 5.74  # Maximum distance for normalization (diagonal of 2m x 2m x 2m space)
        self.time_tolerance = 0.15 # tolerance time for calculating travel distance

        # hard coded z limit
        self.boundary = [self.xy_limit, self.xy_limit, self.z_limit, self.z_limit + 1]

        # Reward parameters
        self.success_reward = 50
        self.reward_multiplier = 10.0
        self.out_of_bounds_penalty = -100.0
        self.distance_improvement_multiplier = 300.0

        # Task state
        self.done = False
        self.boundary_penalise = False
        self.exited_testing_boundary = False

        # Distance tracking for reward calculation
        self.previous_distance = self.max_distance

        # Evaluation mode tracking
        self.successful_episodes_count = 0

        # Randomize the initial goal position
        # self._randomize_goal_position()


    # def _randomize_goal_position(self):
    #     """Randomize the goal position within specified ranges"""
    #     self.goal_position = [
    #         np.random.uniform(self.x_range[0], self.x_range[1]),
    #         np.random.uniform(self.y_range[0], self.y_range[1]),
    #         np.random.uniform(self.z_range[0], self.z_range[1])
    #     ]
    #     print(f"New goal position: [{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}, {self.goal_position[2]:.3f}]")


    def reset(self, training: bool = True):
        """Reset the drone and randomize the target position"""
        # # Randomize goal position before calling parent reset
        # self._randomize_goal_position()

        # Reset successful episodes count when starting evaluation
        if not training and not self._is_evaluating:
            self.successful_episodes_count = 0

        # Handle boundary penalty from previous episode
        if self.exited_testing_boundary:
            self.boundary_penalise = True
            self.exited_testing_boundary = False
            print("--------")
            print(f"total steps: {self.total_steps}")
            print("--------")

        # Call parent reset
        state = super().reset(training)

        # Initialize previous_distance for reward calculation
        self.previous_distance = self._distance_to_target(self.drone.get_position())

        return state
    

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
            processed_action = action
            print(f" action is: {action}")
            assert len(action)==3,'action should be length 3'

        else:
            # Exploration phase: convert from [0, 1] to [-1, 1]
            # processed_action = [action[0] * 2 - 1, action[1] * 2 - 1, action[2] * 2 - 1,]
            processed_action = [action[0] * 2 - 1, action[1] * 2 - 1, action[2] * 2 - 1]

        # determine whether not the action we pass will exceed the boundary
        position = self.drone.get_position()
        vx = processed_action[0] * self.max_velocity
        vy = processed_action[1] * self.max_velocity
        vz = processed_action[2] * self.max_velocity
        time_step = self.step_time + self.time_tolerance
        sx = time_step * vx
        sy = time_step * vy
        sz = time_step * vz

        new_position = [sx + position[0], sy + position[1], sz + position[2]]
        print(f" new position is: {new_position}")
        # Check if new position would exceed boundaries
        # X boundary check
        # step is [0, 0] isn't of [0, 0, 0]
        if new_position[0] < -self.boundary[0] or new_position[0] > self.boundary[0]:
            return super().step([0, 0, 0])

        # Y boundary check
        if new_position[1] < -self.boundary[1] or new_position[1] > self.boundary[1]:
            return super().step([0, 0, 0])

        if new_position[2] <= self.boundary[2] or new_position[2] > self.boundary[3]:
            return super().step([0, 0, 0])
    
        # LOG EVERYTHING
        if self.steps % 5 == 0:  # Every 5 steps
            pos = self.drone.get_position()
            goal = self.goal_position
            distance = self._distance_to_target(pos)
            
            # Calculate direction agent SHOULD go
            should_go = [goal[0] - pos[0], goal[1] - pos[1], goal[2] - pos[2]]
            should_go_norm = np.linalg.norm(should_go)
            
            # Calculate direction agent IS going
            vel = [self.drone.calculated_velocity["x"], 
                self.drone.calculated_velocity["y"],
                self.drone.calculated_velocity["z"]]
            vel_norm = np.linalg.norm(vel)
            
            # Dot product: how aligned is velocity with goal direction?
            if should_go_norm > 0 and vel_norm > 0:
                alignment = np.dot(vel, should_go) / (vel_norm * should_go_norm)
                print(f"Step {self.steps}: Distance={distance:.3f}m, "
                    f"Alignment={alignment:.3f} "
                    f"(1.0=perfect, -1.0=opposite, 0=perpendicular)")
                print("goal we are heading to:", goal)
                print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                print(f"  Goal:     [{goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}]")
                print(f"  Velocity: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
                print(f"  Should move toward: [{should_go[0]:.2f}, {should_go[1]:.2f}, {should_go[2]:.2f}]")
                print()

            # Call parent step method with processed action
        return super().step(processed_action)


    def _reset_task_state(self):
        """Reset task-specific state variables"""
        self.done = False

    def _get_state(self) -> np.ndarray:
        """Get the current state representation for the 2D navigation task"""
        position = self.drone.get_position()

        # Use relative position instead of absolute positions
        relative_x = self.goal_position[0] - position[0]
        relative_y = self.goal_position[1] - position[1]
        relative_z = self.goal_position[2] - position[2]

        distance = np.sqrt(relative_x**2 + relative_y**2 + relative_z**2)

        # Direction to goal (normalized) - helps agent understand "which way"
        direction_x = relative_x / (distance + 1e-6)
        direction_y = relative_y / (distance + 1e-6)
        direction_z = relative_z / (distance + 1e-6)

        # Velocity information
        vel_x = self.drone.calculated_velocity["x"]
        vel_y = self.drone.calculated_velocity["y"]
        vel_z = self.drone.calculated_velocity["z"]
        velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

        # How well velocity aligns with goal direction (1 = perfect, -1 = opposite)
        velocity_alignment = (vel_x * direction_x + vel_y * direction_y + vel_z * direction_z) / (velocity_magnitude + 1e-6) if velocity_magnitude > 0 else 0

        max_z_range = self.z_range[1] - self.z_range[0]

        state = [
            # Relative position to goal (2) - better than absolute positions
            relative_x / self.max_xy_range,
            relative_y / self.max_xy_range,
            relative_z / max_z_range,

            # Distance to goal (1)
            distance / self.max_distance,

            # Direction to goal - unit vector (2) - helps with directional awareness
            direction_x,
            direction_y,
            direction_z,

            # Current velocity (3)
            vel_x / self.max_velocity,
            vel_y / self.max_velocity,
            vel_z / self.max_velocity,

            # Velocity magnitude (1) - overall speed
            velocity_magnitude / self.max_velocity,

            # Velocity alignment with goal (1) - are we heading the right way?
            velocity_alignment
        ]

        return np.array(state, dtype=np.float32)

    def get_overlay_info(self) -> Dict[str, Any]:
        """Get task-specific state information"""
        position = self.drone.get_position()
        return {
            'position': position,
            'goal_position': self.goal_position,
            'distance_to_target': self._distance_to_target(position),
            'done': self.done
        }

    def _distance_to_target(self, position: List[float]) -> float:
        """Calculate 3D Euclidean distance to target position (x, y, z)"""
        return math.sqrt(
            (position[0] - self.goal_position[0])**2 +
            (position[1] - self.goal_position[1])**2 +
            (position[2] - self.goal_position[2])**2
        )

    
    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        position = current_state['position']
        distance = self._distance_to_target(position)
        
        # Progress-based reward
        distance_improvement = self.previous_distance - distance
        reward = distance_improvement * 100  # Strong signal for getting closer
        
        # Penalize being far away
        reward -= distance * 5
        
        # Reaching target leads to high reward
        if distance < self.distance_threshold:
            reward += 50
            
        # Update tracking
        self.previous_distance = distance
        
        return reward

    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Check if navigation task is complete"""
        distance = current_state['distance_to_target']

        # Success condition
        if distance < self.distance_threshold:
            self.done = True
            # Increment success counter only during evaluation
            if self.need_to_change_battery():
                self.change_battery()

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
        # elif not self.is_in_testing_zone():
        #     self.exited_testing_boundary = True
        #     return True

        return False

    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional task-specific info"""
        info = {
            'goal_position': self.goal_position,
            'success': current_state['distance_to_target'] < self.distance_threshold,
            'out_of_bounds': not current_state['in_boundaries'],
            'description': "Gym environment for reinforcement learning control of drones"
        }

        # Add success count during evaluation
        if self._is_evaluating:
            info['success_count'] = self.successful_episodes_count

        return info

    def sample_action(self):
        """Sample an action for exploration phase - returns action in [0, 1] range"""
        return np.random.uniform(0, 1, size=(3,))

    def _render_task_specific_info(self):
        """Render navigation task specific information"""
        pos = self.drone.get_position()
        target = self.goal_position
        distance = self._distance_to_target(pos)

        print(f"Target Position: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
        print(f"Distance to Target: {distance:.2f}")
        print(f"Success Threshold: {self.distance_threshold:.2f}")
        print(f"Done: {self.done}")

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

        # LEFT SUBPLOT: 3D trajectory view
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')

        # Plot the drone's trajectory
        ax1.plot(x, y, z, label='Drone Path', color='yellow', linewidth=2.5)

        # Mark important points with better visibility
        ax1.scatter(x[0], y[0], z[0], color='green', s=80, label='Start',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(x[-1], y[-1], z[-1], color='blue', s=80, label='Current',
                    depthshade=False, edgecolors='black', linewidth=0.5)
        ax1.scatter(self.goal_position[0], self.goal_position[1], self.goal_position[2],
                    color='red', marker='*', s=120, label='Goal',
                    depthshade=False, edgecolors='black', linewidth=1)

        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_zlim(0.25, 1.25)

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

        # Plot the drone's trajectory in X-Y plane
        ax2.plot(x, y, color='yellow', linewidth=2.5, label='Drone Path', zorder=1)

        # Mark important points
        ax2.scatter(x[0], y[0], color='green', s=80, label='Start',
                    edgecolors='black', linewidth=0.5, zorder=3)
        ax2.scatter(x[-1], y[-1], color='blue', s=80, label='Current',
                    edgecolors='black', linewidth=0.5, zorder=3)
        ax2.scatter(self.goal_position[0], self.goal_position[1],
                    color='red', marker='*', s=120, label='Goal',
                    edgecolors='black', linewidth=1, zorder=3)

        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)

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
        fig.suptitle(f'Episode Trajectory (Step {self.steps})', fontsize=13, y=0.98)

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


if __name__ == "__main__":
    # quick sanity test
    env = MoveToRandom3DPosition()
    env.reset()
    env.drone.set_velocity_vector(2, 0, 0)
    time.sleep(2)
    env.reset()
    env.drone.set_velocity_vector(0, 2, 0)
    time.sleep(2)
    env.reset()
    # for _ in range(3):
    #     a = env.sample_action()
    #     s, r, d, t, i = env.step(a)
    #     assert s.shape == (8,)  # Updated observation space size
    #     assert -50 <= r <= 150  # Updated reward range
    env.close()
    print("Sanity-check passed")
