

from typing import Literal
from drone_gym.tasks.intercept_target.intercept_target_line_2d import InterceptTargetLine2D
from drone_gym.tasks.intercept_target.intercept_target_line_3d import InterceptTargetLine3D
from drone_gym.tasks.intercept_target.intercept_evader_2d_particle import InterceptEvader2DParticle
from drone_gym.tasks.intercept_target.intercept_navigation_3d import InterceptNavigation3D
from drone_gym.tasks.move_to_2d_position import MoveToPosition
from drone_gym.tasks.move_to_random_2d_position import MoveToRandomPosition
from drone_gym.tasks.move_to_3d_position import MoveTo3DPosition
from drone_gym.tasks.move_to_random_3d_position import MoveToRandom3DPosition
from drone_gym.tasks.move_circle.move_circle_2d import MoveCircle2D
from drone_gym.tasks.move_circle.move_circle_2d_velocity import MoveCircle2DVelocity
from drone_gym.tasks.move_circle.move_circle_2d_acceleration import MoveCircle2DAcceleration
from drone_gym.tasks.move_circle.move_circle_2d_slow import MoveCircle2DSlow
from drone_gym.tasks.evade_pursuers.evade_pursuers_2d import EvadePursuers2D
from drone_gym.tasks.evade_pursuers.evade_pursuers_2d_particle import EvadePursuers2DParticle


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
    elif task_name == "move_circle":
        env = MoveCircle2D(use_simulator, **kwargs)
    elif task_name == "move_circle_velocity":
        env = MoveCircle2DVelocity(use_simulator, **kwargs)
    elif task_name == "move_circle_acceleration":
        env = MoveCircle2DAcceleration(use_simulator, **kwargs)
    elif task_name == "move_circle_slow":
        env = MoveCircle2DSlow(use_simulator, **kwargs)
    elif task_name == "intercept_line_2d":
        env = InterceptTargetLine2D(use_simulator, **kwargs)
    elif task_name == "intercept_line_3d":
        env = InterceptTargetLine3D(use_simulator, **kwargs)
    elif task_name == "intercept_evader_2d_particle":
        env = InterceptEvader2DParticle(use_simulator, **kwargs)
    elif task_name == "intercept_navigation_3d":
        env = InterceptNavigation3D(use_simulator, **kwargs)
    elif task_name == "evade_pursuers_2d":
        env = EvadePursuers2D(use_simulator, **kwargs)
    elif task_name == "evade_pursuers_2d_particle":
        env = EvadePursuers2DParticle(use_simulator, **kwargs)
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    return env
