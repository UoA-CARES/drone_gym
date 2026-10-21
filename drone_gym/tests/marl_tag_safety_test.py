"""Checks MarlTag's real-drone boundary guard and flight band. No drones or simulator needed.

    python drone_gym/tests/marl_tag_safety_test.py
"""
from types import SimpleNamespace

import numpy as np

from drone_gym.tasks.intercept_target.marl_tag import MarlTag


def make_env(use_simulator: int) -> MarlTag:
    # Skip __init__ (it connects to drones); set only what the methods under test read.
    env = MarlTag.__new__(MarlTag)
    env.use_simulator = use_simulator
    env.xy_limit = 2.0
    env.flight_z_min, env.flight_z_max = (0.4, 2.5) if use_simulator else (0.6, 1.1)
    env.guard_margin = 0.05
    env.step_time = 0.5
    env.z_margin = 0.1
    env.interceptor_agents = ["interceptor_0"]
    env.curriculum_enabled = False
    env.reset_planner = SimpleNamespace(usable_xy_limit=1.95)
    return env


def sent(env, position, velocity):
    vx, vy, vz, info = env._apply_task_action_processing("runner_0", *velocity, current_position=position)
    return [vx, vy, vz], info["boundary_guard_axes"]


real = make_env(use_simulator=0)

# Climbing past the 1.05 m guard ceiling is stopped; x and y keep going.
assert sent(real, [0.0, 0.0, 1.0], [0.1, -0.1, 0.25]) == ([0.1, -0.1, 0.0], ["z"])
# Descending past the 0.65 m guard floor is stopped.
assert sent(real, [0.0, 0.0, 0.7], [0.0, 0.0, -0.25]) == ([0.0, 0.0, 0.0], ["z"])
# Moves that stay inside, or head back in from outside, are untouched.
assert sent(real, [0.0, 0.0, 0.9], [0.0, 0.0, 0.25]) == ([0.0, 0.0, 0.25], [])
assert sent(real, [0.0, 0.0, 1.3], [0.0, 0.0, -0.2]) == ([0.0, 0.0, -0.2], [])
# Leaving the +/- 2.0 m xy box is stopped on each axis separately.
assert sent(real, [1.95, -1.95, 0.9], [0.25, -0.25, 0.0]) == ([0.0, 0.0, 0.0], ["x", "y"])
assert sent(real, [1.95, 0.0, 0.9], [-0.25, 0.0, 0.0]) == ([-0.25, 0.0, 0.0], [])

# The simulator is never guarded, so training is unchanged.
sim = make_env(use_simulator=1)
assert sent(sim, [0.0, 0.0, 2.4], [0.0, 0.0, 0.25]) == ([0.0, 0.0, 0.25], [])

# Goals and spawns stay inside the flight band (0.7 to 1.0 m on real drones).
heights = np.array([real._sample_random_position()[2] for _ in range(2000)])
assert 0.7 <= heights.min() and heights.max() <= 1.0, (heights.min(), heights.max())
sim_heights = np.array([sim._sample_random_position()[2] for _ in range(2000)])
assert 0.5 <= sim_heights.min() and sim_heights.max() <= 2.4

print("marl_tag safety checks OK")
