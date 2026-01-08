from drone_gym.drone_environment import DroneEnvironment
from drone_gym.move_to_2d_position import MoveToPosition
from drone_gym.move_to_random_2d_position import MoveToRandomPosition
from drone_gym import (
    move_to_2d_position,
    move_to_random_2d_position,
    move_to_3d_position,
    move_to_random_3d_position,
)

__all__ = ['DroneEnvironment', 'MoveToPosition', 'MoveToRandomPosition']
task_map = {
    "move_to_2d_position": move_to_2d_position.MoveToPosition,
    "move_to_random_2d_position": move_to_random_2d_position.MoveToRandomPosition,
    "move_to_3d_position": move_to_3d_position.MoveTo3DPosition,
    "move_to_random_3d_position": move_to_random_3d_position.MoveToRandom3DPosition,
}