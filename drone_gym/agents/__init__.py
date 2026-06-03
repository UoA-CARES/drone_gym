"""Multi-agent / expert-agent support for drone_gym tasks.

A task adds non-learning agents (expert pursuers, moving targets, obstacles,
other crazyflies) through :class:`~drone_gym.sim_agent_manager.SimAgentManager`,
which owns these agents and drives their reset/step/close lifecycle.

The pieces:

* :mod:`drone_gym.agents.bodies`    — how an agent physically moves
  (a real CrazyFlie vs. a software-integrated Gazebo marker).
* :mod:`drone_gym.agents.policies`  — what velocity an agent *wants*
  (pure pursuit, straight-line motion, stationary, or any callable).
* :mod:`drone_gym.agents.sim_agent` — a :class:`SimAgent` couples one body
  with one policy and a role label.
"""

from drone_gym.agents.sim_agent import SimAgent, AgentState
from drone_gym.agents.bodies import AgentBody, CrazyflieBody, SimulatedBody
from drone_gym.agents.ticker import AgentTicker
from drone_gym.agents.policies import (
    BasePolicy,
    PurePursuitPolicy,
    FleePolicy,
    LineMotionPolicy,
    StationaryPolicy,
    CallablePolicy,
)

__all__ = [
    "SimAgent",
    "AgentState",
    "AgentBody",
    "CrazyflieBody",
    "SimulatedBody",
    "AgentTicker",
    "BasePolicy",
    "PurePursuitPolicy",
    "FleePolicy",
    "LineMotionPolicy",
    "StationaryPolicy",
    "CallablePolicy",
]
