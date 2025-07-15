import numpy as np
import math
from typing import Dict, List, Any
from environment import DroneEnvironment


class DroneNavigationTask(DroneEnvironment):
    """Drone navigation task - reach a target position"""

    def __init__(self, max_velocity: float = 1.0, step_time: float = 0.1):
        super().__init__(max_velocity, step_time)

        # Task-specific parameters
        self.target_position = [1, 1, 1]  # Goal position
        self.distance_threshold = 0.2  # Distance threshold to consider target reached
        self.max_distance = 5.0  # Maximum distance for normalization
        self.max_steps = 1000  # Maximum steps before truncation

        # Reward parameters
        self.success_reward = 100.0
        self.out_of_bounds_penalty = -50.0
        self.distance_improvement_multiplier = 10.0

        # Task state
        self.done = False
        self.prior_state = None

    def _reset_task_state(self):
        """Reset task-specific state variables"""
        self.done = False
        self.prior_state = None

    def _get_state(self) -> np.ndarray:
        """Get the current state representation for the navigation task"""
        position = self.drone.get_position()

        # State includes: current position, target position, distance to target, normalized
        state = [
            position[0], position[1], position[2],  # Current position
            self.target_position[0], self.target_position[1], self.target_position[2],  # Target position
            self._distance_to_target(position) / self.max_distance,  # Normalized distance
            1.0 if self.drone.in_boundaries else 0.0,  # In boundaries flag
            self.steps / self.max_steps  # Normalized step count
        ]

        return np.array(state, dtype=np.float32)

    def _get_task_specific_state(self) -> Dict[str, Any]:
        """Get task-specific state information"""
        position = self.drone.get_position()
        return {
            'target_position': self.target_position,
            'distance_to_target': self._distance_to_target(position),
            'done': self.done
        }

    def _distance_to_target(self, position: List[float]) -> float:
        """Calculate Euclidean distance to target position"""
        return math.sqrt(
            (position[0] - self.target_position[0])**2 +
            (position[1] - self.target_position[1])**2 +
            (position[2] - self.target_position[2])**2
        )

    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward for navigation task"""
        position = current_state['position']
        distance = current_state['distance_to_target']

        # Base reward is negative distance (closer = higher reward)
        reward = -distance

        # Reward for moving closer to target
        if self.prior_state is not None:
            prev_distance = self.prior_state['distance_to_target']
            distance_improvement = prev_distance - distance
            reward += distance_improvement * self.distance_improvement_multiplier

        # Bonus for reaching target
        if distance < self.distance_threshold:
            reward += self.success_reward

        # Penalty for going out of bounds
        if not current_state['in_boundaries']:
            reward += self.out_of_bounds_penalty

        return reward

    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Check if navigation task is complete"""
        distance = current_state['distance_to_target']
        position = current_state['position']

        # Success condition
        if distance < self.distance_threshold:
            self.done = True
            return True

        # Failure conditions
        if not current_state['in_boundaries'] or position[2] <= 0:
            self.done = True
            return True

        return False

    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode should be truncated"""
        return current_state['steps'] >= self.max_steps

    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional task-specific info"""
        return {
            'target_position': self.target_position,
            'success': current_state['distance_to_target'] < self.distance_threshold,
            'out_of_bounds': not current_state['in_boundaries']
        }

    def _render_task_specific_info(self):
        """Render navigation task specific information"""
        pos = self.drone.get_position()
        target = self.target_position
        distance = self._distance_to_target(pos)

        print(f"Target Position: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
        print(f"Distance to Target: {distance:.2f}")
        print(f"Success Threshold: {self.distance_threshold:.2f}")
        print(f"Done: {self.done}")

    # Task-specific methods
    def set_target_position(self, target: List[float]):
        """Set a new target position for the drone to reach"""
        if len(target) != 3:
            raise ValueError("Target must be a 3-element array [x, y, z]")
        self.target_position = target
        print(f"Target position set to: {target}")

    def set_distance_threshold(self, threshold: float):
        """Set the distance threshold for considering target reached"""
        self.distance_threshold = threshold
        print(f"Distance threshold set to: {threshold}")

    def set_max_steps(self, max_steps: int):
        """Set maximum steps before truncation"""
        self.max_steps = max_steps
        print(f"Max steps set to: {max_steps}")

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current task"""
        return {
            'task_type': 'navigation',
            'target_position': self.target_position,
            'distance_threshold': self.distance_threshold,
            'max_steps': self.max_steps,
            'success_reward': self.success_reward,
            'out_of_bounds_penalty': self.out_of_bounds_penalty
        }


# Example usage
if __name__ == "__main__":
    env = DroneNavigationTask()

    # Set task parameters
    env.set_target_position([1.5, 1.0, 1.2])
    env.set_distance_threshold(0.3)
    env.set_max_steps(500)

    # Reset environment
    state = env.reset()

    # Run a few steps
    for i in range(10):
        # Random action
        action = np.random.uniform(-1, 1, 3)
        state, reward, done, info = env.step(action)

        print(f"Step {i+1}: Reward={reward:.2f}, Done={done}")
        env.render()

        if done:
            break

    env.close()
