import numpy as np
from typing import List
from drone_gym.move_to_position import MoveToPosition


class MoveToRandomPosition(MoveToPosition):
    """Reinforcement learning task for drone navigation to randomized target positions"""

    def __init__(self, max_velocity: float = 0.20, step_time: float = 0.5,
                 exploration_steps: int = 1000, episode_length: int = 40,
                 x_range: List[float] = [-1.0, 1.0],
                 y_range: List[float] = [-1.0, 1.0]):

        # Store ranges
        self.x_range = x_range
        self.y_range = y_range

        # Initialize parent class (this sets initial goal_position)
        super().__init__(max_velocity, step_time, exploration_steps, episode_length)

        # Randomize the initial goal position
        self._randomize_goal_position()

    def _randomize_goal_position(self):
        """Randomize the goal position within specified ranges"""
        self.goal_position = [
            np.random.uniform(self.x_range[0], self.x_range[1]),
            np.random.uniform(self.y_range[0], self.y_range[1])
        ]
        print(f"New goal position: [{self.goal_position[0]:.3f}, {self.goal_position[1]:.3f}, {self.goal_position[2]:.3f}]")

    def reset(self, training: bool = True):
        """Reset the drone and randomize the target position"""
        # Randomize goal position before calling parent reset
        self._randomize_goal_position()

        # Call parent reset which will use the new goal_position
        return super().reset(training)
