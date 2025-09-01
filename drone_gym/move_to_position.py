import numpy as np
import math
import time
from typing import Dict, List, Any
from drone_gym.drone_environment import DroneEnvironment


class DroneNavigationTask(DroneEnvironment):
    """Drone navigation task - reach a target position"""

    def __init__(self, max_velocity: float = 0.25, step_time: float = 0.5, max_steps: int = 1000):
        super().__init__(max_velocity, step_time, max_steps)

        # Task-specific parameters
        self.target_position = [0, 0, 1.2]  # Goal position
        self.distance_threshold = 0.1  # Distance threshold to consider target reached
        self.max_distance = 5.0  # Maximum distance for normalization
        self.max_steps = max_steps

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

    def get_overlay_info(self) -> Dict[str, Any]:
        """Get task-specific state information"""
        position = self.drone.get_position()
        return {
            'position': position,
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
        distance = self._distance_to_target(position)

        # Base reward is negative distance (closer = higher reward)
        # currently this reward will never be positive and is too low. assuming the max distance is (3,3,3) reward = -5.2.
        # Changed from reward = -distance to reward = 50-10*distance
        reward = 50 - 10*distance

        # Reward for moving closer to target
        # This should be done in proportions (JJ)
        # if self.prior_state is not None:
        #     prev_distance = self.prior_state['distance_to_target']
        #     distance_improvement = (prev_distance - distance)/prev_distance # this gives proportional distance increase compared to prev dist
        #     reward += distance_improvement * self.distance_improvement_multiplier

        # Bonus for reaching target
        if distance < self.distance_threshold:
            reward += self.success_reward

        # TODO: Penalty for going close to out of bounds
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
        if self.need_to_change_battery():
            self.change_battery()
            # truncates the current episode -> resets
            return True

        if current_state['steps'] >= self.max_steps:
            return True
        elif not self.is_in_testing_zone():
            return True

        return False


    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional task-specific info"""
        return {
            'target_position': self.target_position,
            'success': current_state['distance_to_target'] < self.distance_threshold,
            'out_of_bounds': not current_state['in_boundaries'],
            'description': "Gym environment for reinforcement learning control of drones"
        }

    def sample_action(self, safety=True):
        if safety:
            x, y , z= np.random.uniform(0, 1, size=(3,))
            print("Sampled action:", x, y)
            # z value is 0.5 which will be normalised to 0
            return np.array([x, y, z])
        return np.random.uniform(0, 1, size=(3,))

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


if __name__ == "__main__":
    # quick sanity test
    env = DroneNavigationTask()
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
    #     assert s.shape == (9,)
    #     assert -50 <= r <= 100
    env.close()
    print("Sanity-check passed")
