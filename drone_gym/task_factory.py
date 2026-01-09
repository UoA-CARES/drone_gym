

from typing import Literal
from drone_gym.move_to_2d_position import MoveToPosition
from drone_gym.move_to_random_2d_position import MoveToRandomPosition
from drone_gym.move_to_3d_position import MoveTo3DPosition
from drone_gym.move_to_random_3d_position import MoveToRandom3DPosition


def make(task_name: str, use_simulator: Literal[0,1], **kwargs):
    """Factory function to create drone tasks based on the task name."""
    if task_name == "move_2d":
        env = MoveToPosition(use_simulator, **kwargs)
    elif task_name == "move_random_2d":
        env = MoveToRandomPosition(use_simulator, **kwargs)
    elif task_name == "move_3d":
        env = MoveTo3DPosition(use_simulator, **kwargs)
    elif task_name == "move_random_3d":
        env = MoveToRandom3DPosition(use_simulator, **kwargs)
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    return env
