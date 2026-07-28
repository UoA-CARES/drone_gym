"""Agent bodies — how an agent physically moves and renders.

A body hides the difference between a real flying CrazyFlie and a software
"particle" that only exists as a Gazebo visual marker. Tasks drive every body
through the same lifecycle::

    ensure_airborne() / await_airborne()   # takeoff (real drones only)
    prepare_reset(pos) / await_reset()     # move to spawn
    start_episode()                        # switch to per-step velocity control
    apply_velocity(vx, vy, vz)             # each step: command a velocity
    integrate(dt)                          # background tick (software bodies only)
    get_position()                         # read current position
    close()

Real drones integrate motion physically, so their integrate() is a no-op.
Software bodies have no physics, so the AgentTicker advances them via integrate().
"""

from abc import ABC, abstractmethod
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone


class AgentBody(ABC):
    """Base class for agent bodies."""

    # Software bodies (no physics) need a ticker to advance them each tick.
    is_software_integrated: bool = False

    def ensure_airborne(self) -> None:
        """Begin takeoff if not already flying (non-blocking)."""

    def await_airborne(self, timeout: float = 15.0) -> bool:
        """Block until airborne. Returns True on success."""
        return True

    @abstractmethod
    def prepare_reset(self, position: List[float]) -> None:
        """Begin moving the body to position (non-blocking)."""
        raise NotImplementedError

    def await_reset(self, timeout: float = 12.0) -> bool:
        """Block until the body has reached its reset position."""
        return True

    def start_episode(self) -> None:
        """Switch to per-step velocity control for the episode."""

    @abstractmethod
    def apply_velocity(self, vx: float, vy: float, vz: float) -> None:
        """Command a velocity for this step."""
        raise NotImplementedError

    def integrate(self, dt: float) -> None:
        """Advance the body by its current velocity (software bodies only)."""

    @abstractmethod
    def get_position(self) -> List[float]:
        """Return current position [x, y, z]."""
        raise NotImplementedError

    def close(self) -> None:
        """Land / clean up."""


class CrazyflieBody(AgentBody):
    """
    Agent-body adapter for a physical or simulated Crazyflie.

    A pre-created Drone or DroneSim may be supplied through ``drone``.
    In that case, the body does not own the drone and ``close()`` leaves
    its lifecycle to the creating environment.

    For backwards compatibility, the body can still create its own drone
    when ``use_simulator`` is supplied. In that mode, the body owns and
    closes the resulting drone interface.
    """

    is_software_integrated = False

    def __init__(
        self,
        use_simulator: int | None = None,
        uri: Optional[str] = None,
        fixed_z: float = 1.0,
        takeoff_timeout: float = 15.0,
        drone: Drone | DroneSim | None = None,
    ) -> None:
        if drone is not None:
            self.drone = drone
            self.owns_drone = False

        else:
            if use_simulator is None:
                raise ValueError(
                    "use_simulator is required when CrazyflieBody "
                    "constructs its own drone."
                )

            if use_simulator:
                self.drone = (
                    DroneSim(uri=uri)
                    if uri is not None
                    else DroneSim()
                )
            else:
                self.drone = Drone(uri=uri)

            self.owns_drone = True

        self.fixed_z = fixed_z
        self.takeoff_timeout = takeoff_timeout

    def ensure_airborne(self) -> None:
        if not self.drone.is_flying_event.is_set():
            self.drone.take_off()

    def await_airborne(self, timeout: float = 15.0) -> bool:
        return self.drone.is_flying_event.wait(timeout=timeout)

    def _reset_control_properties(self) -> None:
        d = self.drone
        d.clear_command_queue()
        time.sleep(0.5)
        d.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        d.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        d.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        d.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        d.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

    def prepare_reset(self, position: List[float]) -> None:
        d = self.drone
        if not d.is_flying_event.is_set():
            d.take_off()
            d.is_flying_event.wait(timeout=self.takeoff_timeout)
        if d.velocity_controller_active:
            d.stop_velocity_control()
        d.set_velocity_vector(0, 0, 0)
        time.sleep(0.1)
        self._reset_control_properties()
        d.set_target_position(position[0], position[1], position[2])
        d.start_position_control()

    def await_reset(self, timeout: float = 12.0) -> bool:
        return self.drone.at_reset_position.wait(timeout=timeout)

    def start_episode(self) -> None:
        d = self.drone
        d.stop_position_control()
        d.clear_reset_position_event()
        d.start_velocity_control()

    def apply_velocity(self, vx: float, vy: float, vz: float) -> None:
        self.drone.set_velocity_vector(vx, vy, vz)

    def get_position(self) -> List[float]:
        return list(self.drone.get_position())

    def close(self) -> None:
        if not self.owns_drone:
            return

        d = self.drone

        try:
            if d.velocity_controller_active:
                d.stop_velocity_control()

            d.set_velocity_vector(0, 0, 0)
            d.land()

        except Exception as exc:
            print(
                f"[CrazyflieBody] Error landing: {exc}"
            )

        try:
            d.is_landed_event.wait(timeout=30)
        except Exception:
            pass

        try:
            d.stop()
        except Exception as exc:
            print(
                f"[CrazyflieBody] Error stopping: {exc}"
            )


class SimulatedBody(AgentBody):
    """A software-integrated agent rendered as a Gazebo sphere marker (no physics).

    Position is advanced by position += velocity * dt on the AgentTicker's
    background tick and clamped to ±bounds. All Gazebo service calls are
    delegated to the SimManager that created this body, keeping Gazebo I/O in
    one place. Reset is an instant teleport.
    """

    is_software_integrated = True

    def __init__(self, sim_manager, fixed_z: float = 1.0, bounds: float = 2.0,
                 render: bool = True, marker_name: str = "rl_sim_agent",
                 color: Tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.9),
                 radius: float = 0.08):
        self._sim_manager = sim_manager  # used for all Gazebo calls
        self.fixed_z = fixed_z
        self.bounds = bounds
        self.render = render
        self.marker_name = marker_name
        self.color = color
        self.radius = radius

        self.position: List[float] = [0.0, 0.0, fixed_z]
        self.velocity: List[float] = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()
        self._marker_spawned = False

        if self.render:
            self._spawn_marker(0.0, 0.0, fixed_z)

    def _spawn_marker(self, x: float, y: float, z: float) -> None:
        ok = self._sim_manager._spawn_sphere_entity(
            self.marker_name, x, y, z, self.color, self.radius
        )
        self._marker_spawned = ok

    def _push_pose(self, x: float, y: float, z: float) -> None:
        if not self.render:
            return
        if not self._marker_spawned:
            self._spawn_marker(x, y, z)
            return
        if not self._sim_manager._set_entity_pose(self.marker_name, x, y, z):
            self._marker_spawned = False

    def prepare_reset(self, position: List[float]) -> None:
        with self._lock:
            self.position = [position[0], position[1], self.fixed_z]
            self.velocity = [0.0, 0.0, 0.0]
        self._push_pose(position[0], position[1], self.fixed_z)

    def apply_velocity(self, vx: float, vy: float, vz: float) -> None:
        with self._lock:
            self.velocity = [vx, vy, vz]

    def integrate(self, dt: float) -> None:
        with self._lock:
            px, py, _ = self.position
            vx, vy, _ = self.velocity
            nx = float(np.clip(px + vx * dt, -self.bounds, self.bounds))
            ny = float(np.clip(py + vy * dt, -self.bounds, self.bounds))
            self.position = [nx, ny, self.fixed_z]
        self._push_pose(nx, ny, self.fixed_z)

    def get_position(self) -> List[float]:
        with self._lock:
            return list(self.position)

    def close(self) -> None:
        if self.render:
            self._sim_manager.remove_entity(self.marker_name)
