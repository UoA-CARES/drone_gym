import numpy as np
import math
import time
from typing import Dict, List, Any
from drone_gym.drone_environment import DroneEnvironment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import cv2


class MoveToPosition(DroneEnvironment):
    """Reinforcement learning task for drone navigation to a target position"""

    def __init__(self, max_velocity: float = 0.25, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 40):
        super().__init__(max_velocity, step_time)

        # RL Training parameters
        self.episode_length = episode_length
        self.exploration_steps = exploration_steps
        self.total_steps = 0
        self.truncate_next = False

        # Task-specific parameters
        self.goal_position = [0, 0.5, 1.0]  # Goal position
        self.distance_threshold = 0.1  # Distance threshold to consider target reached
        self.max_distance = 1  # Maximum distance for normalization
        self.time_tolerance = 0.15 # tolerance time for calculating travel distance

        # hard coded z limit
        self.boundary = [self.xy_limit, self.xy_limit, self.z_limit, self.z_limit + 1]

        # Reward parameters
        self.success_reward = 100.0
        self.out_of_bounds_penalty = -100.0
        self.distance_improvement_multiplier = 100.0

        # Task state
        self.done = False
        self.boundary_penalise = False
        self.exited_testing_boundary = False

        # Distance tracking for reward calculation
        self.previous_distance = self.max_distance

        # Evaluation mode tracking
        self.successful_episodes_count = 0


    def reset(self, training: bool = True):
        """Reset the drone to initial position and handle task-specific logic"""

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

        if self.total_steps == self.exploration_steps:
            print("\n")
            print("SWITCHING TO LEARNING PHASE...")
            print("\n")
            self.truncate_next = True

        # Modify action normalization based on phase
        if self.total_steps > self.exploration_steps:
            # Learning phase: action is already in [-1, 1]
            processed_action = action
        else:
            # Exploration phase: convert from [0, 1] to [-1, 1]
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
        if new_position[0] < -self.boundary[0] or new_position[0] > self.boundary[0]:
            return super().step([0, 0, 0])

        # Y boundary check
        if new_position[1] < -self.boundary[1] or new_position[1] > self.boundary[1]:
            return super().step([0, 0, 0])

        if new_position[2] <= self.boundary[2] or new_position[2] > self.boundary[3]:
            return super().step([0, 0, 0])

        # Call parent step method with processed action
        return super().step(processed_action)


    def _reset_task_state(self):
        """Reset task-specific state variables"""
        self.done = False

    def _get_state(self) -> np.ndarray:
        """Get the current state representation for the navigation task"""
        position = self.drone.get_position()

        # State includes: current position, target position, distance to target, normalized
        state = [
            position[0], position[1], position[2],  # Current position
            self.goal_position[0], self.goal_position[1], self.goal_position[2],  # Target position
            self._distance_to_target(position) / self.max_distance,  # Normalized distance
            1.0 if self.drone.in_boundaries else 0.0,  # In boundaries flag
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
        """Calculate Euclidean distance to target position"""
        return math.sqrt(
            (position[0] - self.goal_position[0])**2 +
            (position[1] - self.goal_position[1])**2 +
            (position[2] - self.goal_position[2])**2
        )

    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward for navigation task"""
        current_distance = current_state['distance_to_target']

        distance_improvement = self.previous_distance - current_distance
        reward = self.distance_improvement_multiplier * distance_improvement

        if current_distance < self.distance_threshold:
            reward += self.success_reward

        if self.boundary_penalise:
            reward += self.out_of_bounds_penalty
            self.boundary_penalise = False

        # Update for the next step
        self.previous_distance = current_distance

        return reward

    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Check if navigation task is complete"""
        distance = current_state['distance_to_target']
        position = current_state['position']

        # Success condition
        if distance < self.distance_threshold:
            self.done = True
            # Increment success counter only during evaluation
            if self._is_evaluating:
                self.successful_episodes_count += 1
            return True

        # Failure conditions
        if not current_state['in_boundaries'] or position[2] <= 0:
            self.done = True
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

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        """
        Generate a 3D plot of the drone's trajectory, convert to image array for video recording
        """
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Return black frame if no positions recorded yet
        if not self.episode_positions:
            plt.close(fig)
            return np.full((height, width, 3), 255, dtype=np.uint8)

        # Convert positions to numpy array for easier manipulation
        pos_array = np.array(self.episode_positions)
        x, y, z = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2]

        # Plot the drone's trajectory
        ax.plot(x, y, z, label='Drone Path', color='cyan', linewidth=2)

        # Mark important points
        ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start', depthshade=False)
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, label='Current', depthshade=False)
        ax.scatter(self.goal_position[0], self.goal_position[1], self.goal_position[2],
                   color='blue', marker='*', s=200, label='Goal', depthshade=False)

        # Set consistent plot limits based on environment boundaries
        ax.set_xlim([-self.xy_limit, self.xy_limit])
        ax.set_ylim([-self.xy_limit, self.xy_limit])
        ax.set_zlim([0, self.z_limit + self.reset_position[2]])

        # Labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Add episode info to title
        success_info = f"Successes: {self.successful_episodes_count}" if self._is_evaluating else ""
        ax.set_title(f'Episode Trajectory (Step {self.steps}) {success_info}')
        ax.legend()

        # Convert matplotlib figure to image array
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)

        # Decode the PNG buffer to numpy array
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close(fig)

        # Use cv2 to decode and resize
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.resize(frame, (width, height))
            # Convert BGR to RGB for consistency
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to black frame if decode fails
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        return frame


if __name__ == "__main__":
    # quick sanity test
    env = MoveToPosition()
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
