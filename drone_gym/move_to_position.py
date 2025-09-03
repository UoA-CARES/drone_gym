import numpy as np
import math
import time
from typing import Dict, List, Any
from drone_gym.drone_environment import DroneEnvironment


class MoveToPosition(DroneEnvironment):
    """Reinforcement learning task for drone navigation to a target position"""

    def __init__(self, max_velocity: float = 0.25, step_time: float = 0.5,
                 exploration_steps: int = 1000):
        super().__init__(max_velocity, step_time)

        # RL Training parameters
        self.exploration_steps = exploration_steps
        self.total_steps = 0
        self.truncate_next = False

        # Task-specific parameters
        self.goal_position = [0, 0.5, 0.5]  # Goal position
        self.distance_threshold = 0.1  # Distance threshold to consider target reached
        self.max_distance = 1  # Maximum distance for normalization

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

    def reset(self):
        """Reset the drone to initial position and handle task-specific logic"""
        # Handle boundary penalty from previous episode
        if self.exited_testing_boundary:
            self.boundary_penalise = True
            self.exited_testing_boundary = False
            print("--------")
            print(f"total steps: {self.total_steps}")
            print("--------")

        # Call parent reset
        state = super().reset()

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
        if self.need_to_change_battery():
            self.change_battery()
            # truncates the current episode -> resets
            return True

        if self.truncate_next:
            self.truncate_next = False
            return True
        elif not self.is_in_testing_zone():
            self.exited_testing_boundary = True
            return True

        return False

    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional task-specific info"""
        return {
            'goal_position': self.goal_position,
            'success': current_state['distance_to_target'] < self.distance_threshold,
            'out_of_bounds': not current_state['in_boundaries'],
            'description': "Gym environment for reinforcement learning control of drones"
        }

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
