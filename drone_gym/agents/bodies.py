"""Agent bodies — how an agent physically moves and renders.

A body hides the difference between a real flying CrazyFlie and a software
"particle" that only exists as a Gazebo visual marker. The
:class:`~drone_gym.sim_agent_manager.SimAgentManager` drives every body through
the same lifecycle::

    ensure_airborne() / await_airborne()   # takeoff (real drones only)
    prepare_reset(pos) / await_reset()     # move to spawn
    start_episode()                        # switch to per-step velocity control
    apply_velocity(vx, vy, vz)             # each step: command a velocity
    integrate(dt)                          # background tick (software bodies only)
    get_position()                         # read current position
    close()

Real drones integrate motion physically, so their :meth:`integrate` is a no-op.
Software bodies have no physics, so the manager's background thread advances
them via :meth:`integrate`.
"""

from abc import ABC, abstractmethod
import math
import os
import subprocess
import tempfile
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone


# ----------------------------------------------------------------------
# Gazebo marker helper (shared by software-integrated bodies)
# ----------------------------------------------------------------------

def run_gz_service(service: str, reqtype: str, reptype: str, req: str,
                   timeout: float = 1.0) -> bool:
    """Call a Gazebo transport service via the ``gz`` CLI. Returns success."""
    cmd = [
        "gz", "service", "-s", service,
        "--reqtype", reqtype, "--reptype", reptype,
        "--timeout", "200", "--req", req,
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True,
                                text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    output = ((result.stdout or "") + (result.stderr or "")).lower()
    if result.returncode != 0 and "already exists" not in output:
        return False
    if "data: false" in output:
        return False
    return True


class AgentBody(ABC):
    """Base class for agent bodies."""

    # Software bodies (no physics) need the manager to advance them each tick.
    is_software_integrated: bool = False

    # --- takeoff phase (real drones only; no-op otherwise) ---
    def ensure_airborne(self) -> None:
        """Begin takeoff if not already flying (non-blocking)."""

    def await_airborne(self, timeout: float = 15.0) -> bool:
        """Block until airborne. Returns True on success."""
        return True

    # --- reset phase ---
    @abstractmethod
    def prepare_reset(self, position: List[float]) -> None:
        """Begin moving the body to ``position`` (non-blocking)."""
        raise NotImplementedError

    def await_reset(self, timeout: float = 12.0) -> bool:
        """Block until the body has reached its reset position."""
        return True

    def start_episode(self) -> None:
        """Switch to per-step velocity control for the episode."""

    # --- per-step ---
    @abstractmethod
    def apply_velocity(self, vx: float, vy: float, vz: float) -> None:
        """Command a velocity for this step."""
        raise NotImplementedError

    def integrate(self, dt: float) -> None:
        """Advance the body by its current velocity (software bodies only)."""

    @abstractmethod
    def get_position(self) -> List[float]:
        """Return current position ``[x, y, z]``."""
        raise NotImplementedError

    def close(self) -> None:
        """Land / clean up."""


class CrazyflieBody(AgentBody):
    """A real (or SITL) CrazyFlie controlled like the task's own drone.

    Wraps a :class:`DroneSim` (simulator) or :class:`Drone` (real radio). The
    lifecycle mirrors the per-pursuer logic that used to live inside
    EvadePursuers2D: takeoff, fly to a spawn via position control, then run
    velocity control for the episode.

    NOTE: constructing this opens a cflib connection synchronously, so the SITL
    firmware instance for ``uri`` must already be running.
    """

    is_software_integrated = False

    def __init__(self, use_simulator: int, uri: Optional[str] = None,
                 fixed_z: float = 1.0, takeoff_timeout: float = 15.0):
        if use_simulator:
            self.drone = DroneSim(uri=uri) if uri is not None else DroneSim()
        else:
            # Real-flight: the radio URI must be configured externally
            # (uri_helper / env var) — multi-radio addressing is out of scope here.
            self.drone = Drone()
        self.fixed_z = fixed_z
        self.takeoff_timeout = takeoff_timeout

    def ensure_airborne(self) -> None:
        if not self.drone.is_flying_event.is_set():
            self.drone.take_off()

    def await_airborne(self, timeout: float = 15.0) -> bool:
        return self.drone.is_flying_event.wait(timeout=timeout)

    def _reset_control_properties(self) -> None:
        """Mirror DroneEnvironment._reset_control_properties for this drone."""
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
        # Re-take off if a previous emergency stop landed the drone
        if not d.is_flying_event.is_set():
            d.take_off()
            d.is_flying_event.wait(timeout=self.takeoff_timeout)
        # Stop velocity control, reset PID state, then position-control to spawn
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
        d = self.drone
        try:
            if d.velocity_controller_active:
                d.stop_velocity_control()
            d.set_velocity_vector(0, 0, 0)
            d.land()
        except Exception as e:
            print(f"[CrazyflieBody] Error landing: {e}")
        try:
            d.is_landed_event.wait(timeout=30)
        except Exception:
            pass
        try:
            d.stop()
        except Exception as e:
            print(f"[CrazyflieBody] Error stopping: {e}")


class SimulatedBody(AgentBody):
    """A software-integrated agent rendered as a Gazebo marker (no physics).

    Position is advanced by ``position += velocity * dt`` on the manager's
    background tick and clamped to ``±bounds``. When ``render`` is True a
    coloured sphere marker is spawned in Gazebo and moved to match. Reset is an
    instant teleport. This is the cheap, robust pursuer/target body — no PID
    overshoot, no boundary breaches, no collision impulses.
    """

    is_software_integrated = True

    def __init__(self, fixed_z: float = 1.0, bounds: float = 2.0,
                 render: bool = True, world_name: str = "crazysim_default",
                 marker_name: str = "rl_sim_agent",
                 color: Tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.9),
                 radius: float = 0.08):
        self.fixed_z = fixed_z
        self.bounds = bounds
        self.render = render
        self.world_name = world_name
        self.marker_name = marker_name
        self.color = color
        self.radius = radius

        self.position: List[float] = [0.0, 0.0, fixed_z]
        self.velocity: List[float] = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()

        self._marker_file = os.path.join(
            tempfile.gettempdir(), f"{self.marker_name}.sdf"
        )
        self._marker_spawned = False
        if self.render:
            self._write_marker_file()
            self._spawn_marker(self.position[0], self.position[1], self.position[2])

    # --- Gazebo marker management ---
    def _write_marker_file(self) -> None:
        r, g, b, a = self.color
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{self.marker_name}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry><sphere><radius>{self.radius}</radius></sphere></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        with open(self._marker_file, "w", encoding="utf-8") as f:
            f.write(sdf)

    def _spawn_marker(self, x: float, y: float, z: float) -> bool:
        req = (
            f'sdf_filename: "{self._marker_file}", '
            f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}, '
            f'name: "{self.marker_name}", allow_renaming: false'
        )
        ok = run_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
        )
        if not ok:
            # Probably already exists from a previous run — fall back to a pose set
            ok = self._set_marker_pose(x, y, z)
        self._marker_spawned = ok
        return ok

    def _set_marker_pose(self, x: float, y: float, z: float) -> bool:
        req = f'name: "{self.marker_name}", position: {{x: {x}, y: {y}, z: {z}}}'
        return run_gz_service(
            service=f"/world/{self.world_name}/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            req=req,
        )

    def _push_pose(self, x: float, y: float, z: float) -> None:
        if not self.render:
            return
        if not self._marker_spawned:
            self._spawn_marker(x, y, z)
            return
        if not self._set_marker_pose(x, y, z):
            # World may have restarted — try to respawn next time
            self._marker_spawned = False

    # --- lifecycle ---
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
        if not self.render:
            return
        run_gz_service(
            service=f"/world/{self.world_name}/remove",
            reqtype="gz.msgs.Entity",
            reptype="gz.msgs.Boolean",
            req=f'name: "{self.marker_name}"',
        )
