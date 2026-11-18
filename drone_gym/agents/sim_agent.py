"""SimAgent — one body + one policy + a role label.

A :class:`SimAgent` is the unit the
:class:`~drone_gym.sim_manager.SimManager` manages. It delegates
physical movement to its :class:`~drone_gym.agents.bodies.AgentBody` and
decision-making to its :class:`~drone_gym.agents.policies.BasePolicy`, exposing
a uniform lifecycle so the manager never has to care whether an agent is a real
CrazyFlie or a virtual particle.
"""

from dataclasses import dataclass, field
from typing import Any

from drone_gym.agents.bodies import AgentBody
from drone_gym.agents.policies import BasePolicy


@dataclass
class AgentState:
    """A snapshot of one agent, handed to policies and back to the task."""

    agent_id: int
    role: str
    position: list[float]
    velocity: list[float]
    extra: dict[str, Any] = field(default_factory=dict)


class SimAgent:
    """Couples a body and a policy under a role label.

    Per-step flow used by the manager:

    * :meth:`act` reads the live body position, asks the policy for a velocity,
      caches it, and commands the body (non-blocking — real drones then fly
      during the task's own ``step_time`` sleep).
    * :meth:`refresh` re-reads the body position after motion has happened.
    * :meth:`state` returns the cached position (post-refresh) and the last
      commanded velocity — exactly what the observation needs.
    """

    def __init__(
        self, agent_id: int, body: AgentBody, policy: BasePolicy, role: str = "agent"
    ):
        self.agent_id = agent_id
        self.body = body
        self.policy = policy
        self.role = role
        self.position: list[float] = body.get_position()
        self.velocity: list[float] = [0.0, 0.0, 0.0]

    @property
    def needs_ticking(self) -> bool:
        """Whether the manager must advance this agent on its background tick."""
        return self.body.is_software_integrated

    def state(self) -> AgentState:
        return AgentState(
            self.agent_id, self.role, list(self.position), list(self.velocity)
        )

    # --- lifecycle (delegated to the body, with policy reset) ---
    def reset_policy(self, context: dict[str, Any]) -> None:
        self.policy.reset(self.state(), context)

    def ensure_airborne(self) -> None:
        self.body.ensure_airborne()

    def await_airborne(self, timeout: float) -> bool:
        return self.body.await_airborne(timeout)

    def prepare_reset(self, position: list[float]) -> None:
        self.body.prepare_reset(position)
        self.position = [position[0], position[1], position[2]]
        self.velocity = [0.0, 0.0, 0.0]

    def await_reset(self, timeout: float) -> bool:
        return self.body.await_reset(timeout)

    def start_episode(self) -> None:
        self.body.start_episode()

    # --- per-step ---
    def compute_velocity(self, context: dict[str, Any]) -> list[float]:
        """Compute and cache the policy velocity without commanding the body."""
        self.position = self.body.get_position()

        velocity = self.policy.compute(
            self.state(),
            context,
        )

        self.velocity = [
            velocity[0],
            velocity[1],
            velocity[2] if len(velocity) > 2 else 0.0,
        ]

        return list(self.velocity)

    def act(self, context: dict[str, Any]) -> list[float]:
        """Compute the policy velocity and command the body."""
        velocity = self.compute_velocity(context)

        self.body.apply_velocity(
            velocity[0],
            velocity[1],
            velocity[2],
        )

        return velocity

    def tick(self, dt: float) -> None:
        self.body.integrate(dt)

    def refresh(self) -> None:
        self.position = self.body.get_position()

    def close(self) -> None:
        self.body.close()
