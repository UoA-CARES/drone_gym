"""SimAgentManager — a factory/catalog of simulation objects.

This is NOT an orchestrator. It does not run a loop, reset agents, or step
anything. It is a catalog you pull ready-made objects from so you can write
your own script around them:

    from drone_gym.sim_agent_manager import SimAgentManager

    manager = SimAgentManager(use_simulator=1)

    # pull a connected crazyflie (a real DroneSim/Drone you drive yourself)
    pursuer = manager.create_crazyflie(uri="udp://0.0.0.0:19851")

    # pull an expert system (a policy that maps state+context -> velocity)
    expert = manager.create_policy("pure_pursuit", max_velocity=0.2, boundary_limit=2.0)

    # ... your own loop ...
    vx, vy, vz = expert.compute(some_state, {"evader_pos": evader_xyz})
    pursuer.set_velocity_vector(vx, vy, vz)

    manager.close_all()   # land/despawn everything it handed out

What you can pull:

* ``create_crazyflie(...)``  — a connected DroneSim (sim) or Drone (real radio).
* ``create_policy(name, ...)`` — an expert policy. Built-ins:
  ``pure_pursuit``, ``line_motion``, ``stationary``, ``callable``.
* ``create_agent(type, ...)`` — a :class:`SimAgent` (body + policy already paired).
  Built-ins: ``crazyflie_pursuer``, ``particle_pursuer``, ``virtual_target``,
  ``stationary_obstacle``.
* ``create_particle(...)``   — a :class:`SimulatedBody` (a Gazebo marker you can
  move around in software, no physics).

Extend the catalog with :meth:`register_policy` / :meth:`register_agent_type`.

The manager keeps a handle on everything it creates so :meth:`close_all` can
land drones and despawn markers in one call — that is the only lifecycle help it
provides; stepping/looping is entirely up to your script.
"""

from typing import Any, Callable, Dict, List, Optional

from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
from drone_gym.agents.bodies import CrazyflieBody, SimulatedBody
from drone_gym.agents.policies import (
    BasePolicy,
    PurePursuitPolicy,
    FleePolicy,
    LineMotionPolicy,
    StationaryPolicy,
    CallablePolicy,
)
from drone_gym.agents.sim_agent import SimAgent


# Default colours for marker bodies, cycled per marker for visual distinction.
_MARKER_COLORS = [
    (1.0, 0.2, 0.2, 0.9),   # red
    (1.0, 0.6, 0.0, 0.9),   # orange
    (0.6, 0.2, 0.8, 0.9),   # purple
    (0.6, 0.3, 0.1, 0.9),   # brown
    (1.0, 0.0, 1.0, 0.9),   # magenta
]


class SimAgentManager:
    """A catalog of simulation objects (crazyflies, expert policies, agents)."""

    # Crazyflies pulled without an explicit URI get ports 19851, 19852, ...
    # (a task's own evader typically uses 19850 / the DroneSim default).
    CRAZYFLIE_PORT_BASE = 19851

    def __init__(self, use_simulator: int, world_name: str = "crazysim_default"):
        self.use_simulator = use_simulator
        self.world_name = world_name

        self._next_port = self.CRAZYFLIE_PORT_BASE
        self._next_marker_index = 0
        self._next_agent_id = 0

        # Catalog registries (extendable via register_*)
        self._policies: Dict[str, Callable[..., BasePolicy]] = dict(DEFAULT_POLICIES)
        self._agent_builders: Dict[str, Callable[..., SimAgent]] = dict(DEFAULT_AGENT_TYPES)

        # Everything handed out, so close_all() can tear it down.
        self._created: List[Any] = []

    # ------------------------------------------------------------------
    # Allocation helpers (used by builders)
    # ------------------------------------------------------------------

    def allocate_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def allocate_marker_index(self) -> int:
        idx = self._next_marker_index
        self._next_marker_index += 1
        return idx

    def marker_color(self, index: int) -> tuple:
        return _MARKER_COLORS[index % len(_MARKER_COLORS)]

    # ------------------------------------------------------------------
    # Pull a crazyflie drone
    # ------------------------------------------------------------------

    def create_crazyflie(self, uri: Optional[str] = None):
        """Create and return a connected crazyflie you drive yourself.

        Returns a :class:`DroneSim` when ``use_simulator`` is set (auto-assigning
        the next UDP port if ``uri`` is None), otherwise a real :class:`Drone`.
        The connection is opened synchronously, so the matching SITL/firmware
        instance must already be running.
        """
        if self.use_simulator:
            if uri is None:
                uri = f"udp://0.0.0.0:{self.allocate_port()}"
            print(f"[SimAgentManager] creating crazyflie at {uri}")
            drone = DroneSim(uri=uri)
        else:
            print("[SimAgentManager] creating real crazyflie (radio URI configured externally)")
            drone = Drone()
        self._created.append(drone)
        return drone

    # ------------------------------------------------------------------
    # Pull an expert policy
    # ------------------------------------------------------------------

    def create_policy(self, name: str, **kwargs) -> BasePolicy:
        """Create an expert policy by name (e.g. ``"pure_pursuit"``)."""
        if name not in self._policies:
            raise ValueError(
                f"Unknown policy: {name!r}. Known policies: {sorted(self._policies)}"
            )
        return self._policies[name](**kwargs)

    def register_policy(self, name: str, factory: Callable[..., BasePolicy]) -> None:
        """Register a new policy name. ``factory(**kwargs)`` returns a policy."""
        self._policies[name] = factory

    def available_policies(self) -> List[str]:
        return sorted(self._policies)

    # ------------------------------------------------------------------
    # Pull a pre-paired agent (body + policy)
    # ------------------------------------------------------------------

    def create_agent(self, agent_type: str, role: Optional[str] = None,
                     policy: Optional[BasePolicy] = None, **cfg) -> SimAgent:
        """Create a :class:`SimAgent` (a body coupled with a policy and role).

        ``policy`` overrides the type's default policy; ``cfg`` is forwarded to
        the type's builder.
        """
        if agent_type not in self._agent_builders:
            raise ValueError(
                f"Unknown agent type: {agent_type!r}. "
                f"Known types: {sorted(self._agent_builders)}"
            )
        agent = self._agent_builders[agent_type](self, role=role, policy=policy, **cfg)
        agent.agent_id = self._next_agent_id
        self._next_agent_id += 1
        self._created.append(agent)
        return agent

    def register_agent_type(self, name: str, builder: Callable[..., SimAgent]) -> None:
        """Register a new agent-type string. ``builder(manager, role, policy, **cfg)``
        must return a :class:`SimAgent`."""
        self._agent_builders[name] = builder

    def available_agent_types(self) -> List[str]:
        return sorted(self._agent_builders)

    # ------------------------------------------------------------------
    # Pull a particle (software-integrated Gazebo marker)
    # ------------------------------------------------------------------

    def create_particle(self, render: bool = True, fixed_z: float = 1.0,
                        bounds: float = 2.0, color: Optional[tuple] = None,
                        marker_name: Optional[str] = None, radius: float = 0.08
                        ) -> SimulatedBody:
        """Create a :class:`SimulatedBody` — a marker you move in software."""
        idx = self.allocate_marker_index()
        body = SimulatedBody(
            fixed_z=fixed_z, bounds=bounds, render=render,
            world_name=self.world_name,
            marker_name=marker_name or f"rl_particle_{idx}",
            color=color or self.marker_color(idx), radius=radius,
        )
        self._created.append(body)
        return body

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close_all(self) -> None:
        """Land every crazyflie and despawn every marker this manager created."""
        for obj in reversed(self._created):
            try:
                if isinstance(obj, SimAgent):
                    obj.close()
                elif isinstance(obj, SimulatedBody):
                    obj.close()
                else:  # a raw crazyflie (DroneSim / Drone)
                    self._land_crazyflie(obj)
            except Exception as e:
                print(f"[SimAgentManager] Error closing {type(obj).__name__}: {e}")
        self._created.clear()

    @staticmethod
    def _land_crazyflie(drone) -> None:
        if getattr(drone, "velocity_controller_active", False):
            drone.stop_velocity_control()
        drone.set_velocity_vector(0, 0, 0)
        drone.land()
        drone.is_landed_event.wait(timeout=30)
        drone.stop()


# ======================================================================
# Built-in policy factories
# ======================================================================

DEFAULT_POLICIES: Dict[str, Callable[..., BasePolicy]] = {
    "pure_pursuit": lambda **kw: PurePursuitPolicy(**kw),
    "flee": lambda **kw: FleePolicy(**kw),
    "line_motion": lambda **kw: LineMotionPolicy(**kw),
    "stationary": lambda **kw: StationaryPolicy(),
    "callable": lambda fn=None, **kw: CallablePolicy(fn, **kw),
}


# ======================================================================
# Built-in agent-type builders
#
# Each builder has signature builder(manager, role, policy, **cfg) -> SimAgent.
# agent_id is assigned by create_agent after construction.
# ======================================================================

def _build_crazyflie_pursuer(manager: SimAgentManager, role: Optional[str] = None,
                             policy: Optional[BasePolicy] = None, max_velocity: float = 0.20,
                             fixed_z: float = 1.0, boundary_limit: float = 2.0,
                             soft_margin: float = 0.3, target_key: str = "evader_pos",
                             uri: Optional[str] = None, **_) -> SimAgent:
    if uri is None and manager.use_simulator:
        uri = f"udp://0.0.0.0:{manager.allocate_port()}"
        print(f"[SimAgentManager] crazyflie_pursuer connecting at {uri}")
    body = CrazyflieBody(manager.use_simulator, uri=uri, fixed_z=fixed_z)
    if policy is None:
        policy = PurePursuitPolicy(max_velocity, target_key=target_key,
                                   boundary_limit=boundary_limit, soft_margin=soft_margin)
    return SimAgent(0, body, policy, role or "pursuer")


def _build_particle_pursuer(manager: SimAgentManager, role: Optional[str] = None,
                            policy: Optional[BasePolicy] = None, max_velocity: float = 0.30,
                            fixed_z: float = 1.0, bounds: float = 2.0,
                            render: bool = True, target_key: str = "evader_pos",
                            **_) -> SimAgent:
    idx = manager.allocate_marker_index()
    body = SimulatedBody(fixed_z=fixed_z, bounds=bounds, render=render,
                         world_name=manager.world_name,
                         marker_name=f"rl_pursuer_particle_{idx}",
                         color=manager.marker_color(idx))
    if policy is None:
        policy = PurePursuitPolicy(max_velocity, target_key=target_key)
    return SimAgent(0, body, policy, role or "pursuer")


def _build_virtual_target(manager: SimAgentManager, role: Optional[str] = None,
                          policy: Optional[BasePolicy] = None, speed: float = 0.25,
                          fixed_z: float = 1.0, bounds: float = 2.0,
                          render: bool = True, reflect: bool = True, **_) -> SimAgent:
    idx = manager.allocate_marker_index()
    body = SimulatedBody(fixed_z=fixed_z, bounds=bounds, render=render,
                         world_name=manager.world_name,
                         marker_name=f"rl_target_{idx}",
                         color=manager.marker_color(idx))
    if policy is None:
        policy = LineMotionPolicy(speed=speed, bounds=bounds, reflect=reflect)
    return SimAgent(0, body, policy, role or "target")


def _build_particle_evader(manager: SimAgentManager, role: Optional[str] = None,
                           policy: Optional[BasePolicy] = None, max_velocity: float = 0.18,
                           fixed_z: float = 1.0, bounds: float = 2.0,
                           render: bool = True, threat_key: str = "threat_pos",
                           soft_margin: float = 0.3, **_) -> SimAgent:
    idx = manager.allocate_marker_index()
    body = SimulatedBody(fixed_z=fixed_z, bounds=bounds, render=render,
                         world_name=manager.world_name,
                         marker_name=f"rl_evader_particle_{idx}",
                         color=manager.marker_color(idx))
    if policy is None:
        policy = FleePolicy(max_velocity, threat_key=threat_key,
                            boundary_limit=bounds, soft_margin=soft_margin)
    return SimAgent(0, body, policy, role or "evader")


def _build_stationary_obstacle(manager: SimAgentManager, role: Optional[str] = None,
                               policy: Optional[BasePolicy] = None, fixed_z: float = 1.0,
                               bounds: float = 2.0, render: bool = True, **_) -> SimAgent:
    idx = manager.allocate_marker_index()
    body = SimulatedBody(fixed_z=fixed_z, bounds=bounds, render=render,
                         world_name=manager.world_name,
                         marker_name=f"rl_obstacle_{idx}",
                         color=manager.marker_color(idx))
    if policy is None:
        policy = StationaryPolicy()
    return SimAgent(0, body, policy, role or "obstacle")


DEFAULT_AGENT_TYPES: Dict[str, Callable[..., SimAgent]] = {
    "crazyflie_pursuer": _build_crazyflie_pursuer,
    "particle_pursuer": _build_particle_pursuer,
    "particle_evader": _build_particle_evader,
    "virtual_target": _build_virtual_target,
    "stationary_obstacle": _build_stationary_obstacle,
}
