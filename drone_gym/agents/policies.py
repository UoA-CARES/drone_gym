"""Agent policies — the "brain" that decides what velocity an agent wants.

A policy is decoupled from the body it controls: the same
:class:`PurePursuitPolicy` can drive a real CrazyFlie pursuer or a virtual
particle pursuer. This is what lets a task plug in arbitrary expert behaviour
(or even a trained RL policy via :class:`CallablePolicy`).

All policies implement::

    compute(state, context) -> [vx, vy, vz]

where ``state`` is the agent's :class:`~drone_gym.agents.sim_agent.AgentState`
and ``context`` is a free-form dict the task passes in each step (e.g.
``{"evader_pos": [x, y, z]}``). ``reset(state, context)`` is an optional hook
called once per episode for stateful policies.
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Callable

import numpy as np


class BasePolicy(ABC):
    """Base class for all agent policies."""

    def reset(self, state: "Any", context: dict[str, Any]) -> None:
        """Optional per-episode reset hook. Override for stateful policies."""

    @abstractmethod
    def compute(self, state: "Any", context: dict[str, Any]) -> list[float]:
        """Return the desired velocity command ``[vx, vy, vz]`` for this step."""
        raise NotImplementedError


class PurePursuitPolicy(BasePolicy):
    """Chase a moving target at a fixed speed (classic pure pursuit).

    The target position is read from ``context[target_key]`` each step, so the
    task only has to supply e.g. ``{"evader_pos": [...]}``. When
    ``boundary_limit`` is set, velocity components that would drive the agent
    further past a soft boundary are zeroed, preventing it from building
    momentum into a wall (mirrors the original EvadePursuers2D behaviour).
    """

    def __init__(
        self,
        max_velocity: float,
        target_key: str = "target_position",
        boundary_limit: float | None = None,
        soft_margin: float = 0.3,
    ):
        self.max_velocity = max_velocity
        self.target_key = target_key
        self.boundary_limit = boundary_limit
        self.soft_margin = soft_margin

    def compute(self, state: "Any", context: dict[str, Any]) -> list[float]:
        if self.target_key not in context:
            raise KeyError(
                f"PurePursuitPolicy needs context['{self.target_key}'] "
                f"(the target position) but it was not provided"
            )
        target = context[self.target_key]
        pos = state.position

        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            vx, vy = 0.0, 0.0
        else:
            scale = self.max_velocity / dist
            vx = scale * dx
            vy = scale * dy

        if self.boundary_limit is not None:
            soft_limit = self.boundary_limit - self.soft_margin
            if (pos[0] <= -soft_limit and vx < 0) or (pos[0] >= soft_limit and vx > 0):
                vx = 0.0
            if (pos[1] <= -soft_limit and vy < 0) or (pos[1] >= soft_limit and vy > 0):
                vy = 0.0

        return [vx, vy, 0.0]


class PredictedInterceptPolicy(BasePolicy):
    """Pursue a moving target using a predicted intercept point."""

    def __init__(
        self,
        max_velocity: float,
        max_velocity_z: float | None = None,
        prediction_iterations: int = 4,
    ) -> None:
        if max_velocity <= 0:
            raise ValueError("max_velocity must be greater than zero.")

        if max_velocity_z is not None and max_velocity_z <= 0:
            raise ValueError("max_velocity_z must be greater than zero.")

        if prediction_iterations < 1:
            raise ValueError("prediction_iterations must be at least 1.")

        self.max_velocity = max_velocity
        self.max_velocity_z = max_velocity_z
        self.prediction_iterations = prediction_iterations

    def set_max_velocity(self, max_velocity: float) -> None:
        """Update the policy's horizontal/overall speed limit."""
        if max_velocity <= 0:
            raise ValueError("max_velocity must be greater than zero.")

        self.max_velocity = max_velocity

    def compute(self, state, context) -> list[float]:
        """Return a velocity command towards the predicted intercept point."""
        pursuer_position = np.asarray(
            state.position,
            dtype=float,
        )

        target_position = np.asarray(
            context["target_position"],
            dtype=float,
        )

        target_velocity = np.asarray(
            context.get(
                "target_velocity",
                [0.0, 0.0, 0.0],
            ),
            dtype=float,
        )

        intercept_point = target_position.copy()

        for _ in range(self.prediction_iterations):
            distance = float(np.linalg.norm(intercept_point - pursuer_position))

            if distance < 1e-6:
                return [0.0, 0.0, 0.0]

            time_to_go = distance / self.max_velocity

            intercept_point = target_position + target_velocity * time_to_go

        direction = intercept_point - pursuer_position
        distance = float(np.linalg.norm(direction))

        if distance < 1e-6:
            return [0.0, 0.0, 0.0]

        velocity = self.max_velocity * direction / distance

        if self.max_velocity_z is not None:
            velocity[2] = np.clip(
                velocity[2],
                -self.max_velocity_z,
                self.max_velocity_z,
            )

        return velocity.tolist()


class FleePolicy(BasePolicy):
    """Flee from a threat at fixed speed — the mirror of pure pursuit.

    The threat position (e.g. the chasing learner) is read from
    ``context[threat_key]`` each step and the agent moves directly away from it.
    When ``boundary_limit`` is set, the velocity component that would drive the
    agent further past a soft boundary is zeroed, so a cornered evader slides
    along the wall instead of pinning itself into it. If the threat is exactly on
    top of the agent, a random escape heading is chosen.
    """

    def __init__(
        self,
        max_velocity: float,
        threat_key: str = "threat_pos",
        boundary_limit: float | None = None,
        soft_margin: float = 0.3,
    ):
        self.max_velocity = max_velocity
        self.threat_key = threat_key
        self.boundary_limit = boundary_limit
        self.soft_margin = soft_margin

    def compute(self, state: "Any", context: dict[str, Any]) -> list[float]:
        if self.threat_key not in context:
            raise KeyError(
                f"FleePolicy needs context['{self.threat_key}'] "
                f"(the threat position) but it was not provided"
            )
        threat = context[self.threat_key]
        pos = state.position

        dx = pos[0] - threat[0]  # vector pointing AWAY from the threat
        dy = pos[1] - threat[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            angle = float(np.random.uniform(0, 2 * math.pi))
            vx = self.max_velocity * math.cos(angle)
            vy = self.max_velocity * math.sin(angle)
        else:
            scale = self.max_velocity / dist
            vx = scale * dx
            vy = scale * dy

        if self.boundary_limit is not None:
            soft_limit = self.boundary_limit - self.soft_margin
            if (pos[0] <= -soft_limit and vx < 0) or (pos[0] >= soft_limit and vx > 0):
                vx = 0.0
            if (pos[1] <= -soft_limit and vy < 0) or (pos[1] >= soft_limit and vy > 0):
                vy = 0.0

        return [vx, vy, 0.0]


class LineMotionPolicy(BasePolicy):
    """Move in a straight line at fixed speed, reflecting off the boundary.

    A random heading is sampled on every :meth:`reset`. When ``reflect`` is
    True the velocity component is flipped whenever the agent reaches a
    boundary, so it bounces around the play area (mirrors the InterceptTarget
    line target).
    """

    def __init__(self, speed: float, bounds: float, reflect: bool = True):
        self.speed = speed
        self.bounds = bounds
        self.reflect = reflect
        self.velocity: list[float] = [0.0, 0.0, 0.0]

    def reset(self, state: "Any", context: dict[str, Any]) -> None:
        angle = float(np.random.uniform(0, 2 * math.pi))
        self.velocity = [
            self.speed * math.cos(angle),
            self.speed * math.sin(angle),
            0.0,
        ]

    def compute(self, state: "Any", context: dict[str, Any]) -> list[float]:
        if self.reflect:
            x, y = state.position[0], state.position[1]
            if (x <= -self.bounds and self.velocity[0] < 0) or (
                x >= self.bounds and self.velocity[0] > 0
            ):
                self.velocity[0] *= -1
            if (y <= -self.bounds and self.velocity[1] < 0) or (
                y >= self.bounds and self.velocity[1] > 0
            ):
                self.velocity[1] *= -1
        return list(self.velocity)


class StationaryPolicy(BasePolicy):
    """Hold position (e.g. for a static obstacle)."""

    def compute(self, state: "Any", context: dict[str, Any]) -> list[float]:
        return [0.0, 0.0, 0.0]


class CallablePolicy(BasePolicy):
    """Wrap an arbitrary ``fn(state, context) -> [vx, vy, vz]`` as a policy.

    This is the escape hatch for fully custom expert behaviour or a trained
    network: ``CallablePolicy(lambda s, c: my_model(s, c))``. An optional
    ``reset_fn(state, context)`` can be supplied for per-episode setup.
    """

    def __init__(
        self,
        fn: Callable[[Any, dict[str, Any]], list[float]],
        reset_fn: Callable[[Any, dict[str, Any]], None] | None = None,
    ):
        self._fn = fn
        self._reset_fn = reset_fn

    def reset(self, state: "Any", context: dict[str, Any]) -> None:
        if self._reset_fn is not None:
            self._reset_fn(state, context)

    def compute(self, state: "Any", context: dict[str, Any]) -> list[float]:
        v = self._fn(state, context)
        return [v[0], v[1], v[2] if len(v) > 2 else 0.0]
