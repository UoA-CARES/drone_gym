import os
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# drone_gym.agents.* is imported lazily inside methods, not here: a top-level
# import would loop (agents/__init__ -> bodies -> drone_sim -> sim_manager).

@dataclass
class MarkerVisualSettings:
    """
    Visual/model-level settings for a Gazebo target marker.

    These describe what the marker looks like and how often it should be
    retried/updated.
    """
    radius: float = 0.02
    rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.6)
    specular: Tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)
    pose_tolerance: float = 0.005
    retry_backoff: float = 1.0

@dataclass
class MarkerState:
    """
    Runtime cache state for one Gazebo marker entity.

    Each marker name gets its own MarkerState, so multiple markers do not share:
    - spawned/not-spawned state
    - last pose
    - retry timer
    - lock
    - temporary SDF file path
    """
    spawned: bool = False
    last_attempt_time: float = 0.0
    last_pose: Optional[Tuple[float, float, float]] = None
    model_file: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

@dataclass
class EntityHandle:
    """
    Lightweight handle for a named Gazebo entity (e.g. a collision object).

    Stored in the manager's cleanup registry so close_all() can despawn the
    entity by name.
    """
    name: str
    sim_manager: "SimManager"

    def remove(self) -> None:
        self.sim_manager.remove_entity(self.name)

class SimManager:
    """
    World-level simulation manager for Gazebo/CrazySim.

    This class owns simulation-wide responsibilities such as:
    - Gazebo service calls
    - target visual markers
    - boundary visual markers
    - collision / obstacle objects (boxes, cylinders)
    - expert policies and particle/agent factories

    It does not own per-drone control logic such as:
    - Crazyflie connection
    - position logging
    - velocity commands
    - takeoff/landing
    (those live in DroneSim.)
    """

    # Marker colours cycled per particle for visual distinction
    MARKER_COLORS: List[Tuple[float, float, float, float]] = [
        (1.0, 0.2, 0.2, 0.9),  # red
        (1.0, 0.6, 0.0, 0.9),  # orange
        (0.6, 0.2, 0.8, 0.9),  # purple
        (0.6, 0.3, 0.1, 0.9),  # brown
        (1.0, 0.0, 1.0, 0.9),  # magenta
    ]

    def __init__(
        self,
        world_name: str = "crazysim_default",
        enable_target_marker: bool = True,
        enable_boundary_lines: bool = True,
        marker_name: str = "rl_target_marker",
        boundary_name: str = "rl_boundary",
        marker_retry_backoff: float = 1.0,
        boundary_retry_backoff: float = 1.0,
        marker_pose_tolerance: float = 0.005,
        boundary_line_thickness: float = 0.02,
        boundary_visual_height: float = 0.8,
        gz_timeout: float = 1.5,
    ):
        self.world_name = world_name

        # Target marker settings/state
        self.enable_target_marker = enable_target_marker
        self.marker_name = marker_name

        self.marker_visual_settings = MarkerVisualSettings(
            pose_tolerance=marker_pose_tolerance,
            retry_backoff=marker_retry_backoff,
        )

        self._marker_states: Dict[str, MarkerState] = {}
        self._marker_states_lock = threading.Lock()

        # Boundary visual settings/state
        self.enable_boundary_lines = enable_boundary_lines
        self.boundary_name = boundary_name
        self.boundary_retry_backoff = boundary_retry_backoff
        self.boundary_line_thickness = boundary_line_thickness
        self.boundary_visual_height = boundary_visual_height
        self._last_boundary_signature = None
        self._last_boundary_attempt_time = 0.0
        self._boundary_lock = threading.Lock()

        # Gazebo command settings
        self.gz_timeout = gz_timeout

        # Agent / particle factory state
        # Ports are auto-assigned to extra SITL instances (the task's own drone
        # uses 19850; additional ones start here)
        self._next_port = 19851
        self._next_marker_index = 0
        self._next_agent_id = 0

        # Built-in expert policies. Extend with register_policy.
        # Imported here (not at module top) to avoid the agents -> drone_sim ->
        # sim_manager import loop; safe because __init__ runs after import time.
        from drone_gym.agents.policies import (
            PurePursuitPolicy, FleePolicy, LineMotionPolicy,
            StationaryPolicy, CallablePolicy,
        )
        self._policies: Dict[str, Callable] = {
            "pure_pursuit": lambda **kw: PurePursuitPolicy(**kw),
            "flee": lambda **kw: FleePolicy(**kw),
            "line_motion": lambda **kw: LineMotionPolicy(**kw),
            "stationary": lambda **kw: StationaryPolicy(),
            "callable": lambda fn=None, **kw: CallablePolicy(fn, **kw),
        }

        # Built-in agent types (body + policy pairs). Extend with register_agent_type.
        self._agent_builders: Dict[str, Callable] = {
            "crazyflie_pursuer": self._build_crazyflie_pursuer,
            "particle_pursuer": self._build_particle_pursuer,
            "particle_evader": self._build_particle_evader,
            "virtual_target": self._build_virtual_target,
            "stationary_obstacle": self._build_stationary_obstacle,
        }

        # Everything the manager hands out, so close_all() can clean up
        self._created: List[Any] = []

    def _safe_file_name(self, name: str) -> str:
        """
        Convert a Gazebo entity name into a safe file-name.
        """
        return (
            name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

    def _get_target_marker_model_file_path(self, marker_name: str) -> str:
        """
        Return the temporary SDF file path for a specific marker.
        """
        safe_name = self._safe_file_name(marker_name)

        return os.path.join(
            tempfile.gettempdir(),
            f"{safe_name}.sdf",
        )

    def _get_marker_state(self, marker_name: str) -> MarkerState:
        """
        Get or create the runtime cache state for one marker entity.
        """
        with self._marker_states_lock:
            if marker_name not in self._marker_states:
                self._marker_states[marker_name] = MarkerState(
                    model_file=self._get_target_marker_model_file_path(marker_name)
                )

            return self._marker_states[marker_name]

    def set_visual_target_marker_position(
        self,
        x: float,
        y: float,
        z: float,
        marker_name: Optional[str] = None,
    ) -> None:
        """
        Spawn or update a named Gazebo visual marker for a task target.

        If marker_name is not supplied, the default marker name is used.
        """
        if not self.enable_target_marker:
            return

        marker_name = marker_name or self.marker_name
        marker_state = self._get_marker_state(marker_name)
        settings = self.marker_visual_settings
        now = time.time()

        with marker_state.lock:
            if marker_state.last_pose is not None:
                dx = abs(x - marker_state.last_pose[0])
                dy = abs(y - marker_state.last_pose[1])
                dz = abs(z - marker_state.last_pose[2])

                if (
                    dx < settings.pose_tolerance
                    and dy < settings.pose_tolerance
                    and dz < settings.pose_tolerance
                ):
                    return

            if not marker_state.spawned:
                if now - marker_state.last_attempt_time < settings.retry_backoff:
                    return

                marker_state.last_attempt_time = now
                marker_state.spawned = self._spawn_target_marker(
                    marker_name=marker_name,
                    x=x,
                    y=y,
                    z=z,
                )

                if not marker_state.spawned:
                    return

            if not self._set_entity_pose(marker_name, x, y, z):
                # The Gazebo world may have restarted or the marker may have been deleted.
                # Mark this specific marker as missing and retry on a later call.
                marker_state.spawned = False
                return

            marker_state.last_pose = (x, y, z)

    def set_visual_boundary_lines(self, xy_limit: float, z_level: float) -> None:
        """
        Draw a square boundary line overlay in Gazebo.
        """
        if not self.enable_boundary_lines:
            return

        signature = (
            round(float(xy_limit), 3),
            round(float(z_level), 3),
        )

        with self._boundary_lock:
            if signature == self._last_boundary_signature:
                return

            now = time.time()
            if now - self._last_boundary_attempt_time < self.boundary_retry_backoff:
                return

            self._last_boundary_attempt_time = now

            boundary_ok = self._spawn_or_replace_boundary_model(
                name=self.boundary_name,
                xy_limit=max(0.05, float(xy_limit)),
                z_level=float(z_level),
                rgba=(0.2, 0.55, 1.0, 0.8),
            )

            if boundary_ok:
                self._last_boundary_signature = signature

    def reset_visual_state(self) -> None:
        """
        Reset cached visual state.

        Useful if Gazebo is restarted without restarting Python.
        This does not remove entities by itself; it only makes the manager
        attempt to respawn/update them on the next visual call.
        """
        with self._marker_states_lock:
            for marker_state in self._marker_states.values():
                with marker_state.lock:
                    marker_state.spawned = False
                    marker_state.last_pose = None
                    marker_state.last_attempt_time = 0.0

        with self._boundary_lock:
            self._last_boundary_signature = None
            self._last_boundary_attempt_time = 0.0

    def remove_entity(self, entity_name: str) -> bool:
        """
        Remove an entity from Gazebo by name.
        """
        req = f'name: "{entity_name}"'

        ok, _ = self._run_gz_service(
            service=f"/world/{self.world_name}/remove",
            reqtype="gz.msgs.Entity",
            reptype="gz.msgs.Boolean",
            req=req,
        )

        return ok

    def remove_visual_marker(self, marker_name: Optional[str] = None) -> bool:
        """
        Remove a visual target marker from Gazebo and clear its cached state.
        """
        marker_name = marker_name or self.marker_name

        removed = self.remove_entity(marker_name)

        with self._marker_states_lock:
            self._marker_states.pop(marker_name, None)

        return removed

    def remove_all_visual_markers(self) -> None:
        """
        Remove all known visual target markers from Gazebo.
        """
        with self._marker_states_lock:
            marker_names = list(self._marker_states.keys())

        for marker_name in marker_names:
            self.remove_visual_marker(marker_name)

    def _run_gz_service(
        self,
        service: str,
        reqtype: str,
        reptype: str,
        req: str,
    ) -> Tuple[bool, str]:
        """
        Call a Gazebo transport service using the gz CLI.
        """
        cmd = [
            "gz",
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            "200",
            "--req",
            req,
        ]

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.gz_timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, ""

        output = (result.stdout or "") + (result.stderr or "")

        if result.returncode != 0:
            return False, output

        output_lower = output.lower()

        if "[err]" in output_lower:
            return False, output

        if "data: false" in output_lower:
            return False, output

        if "data: true" in output_lower:
            return True, output

        # Some Gazebo service responses do not include explicit data: true/false.
        # If there was no return-code failure and no explicit error, treat it as ok.
        return True, output

    def _write_target_marker_model_file(
        self,
        marker_name: str,
        model_file_path: str,
    ) -> None:
        settings = self.marker_visual_settings
        r, g, b, a = settings.rgba
        sr, sg, sb, sa = settings.specular

        model_sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
<model name='{marker_name}'>
    <static>true</static>
    <link name='link'>
    <visual name='visual'>
        <geometry>
        <sphere>
            <radius>{settings.radius}</radius>
        </sphere>
        </geometry>
        <material>
        <ambient>{r} {g} {b} {a}</ambient>
        <diffuse>{r} {g} {b} {a}</diffuse>
        <specular>{sr} {sg} {sb} {sa}</specular>
        </material>
    </visual>
    </link>
</model>
</sdf>
"""

        with open(model_file_path, "w", encoding="utf-8") as model_file:
            model_file.write(model_sdf)

    def _spawn_target_marker(
        self,
        marker_name: str,
        x: float,
        y: float,
        z: float,
    ) -> bool:
        marker_state = self._get_marker_state(marker_name)

        if marker_state.model_file is None:
            marker_state.model_file = self._get_target_marker_model_file_path(marker_name)

        self._write_target_marker_model_file(
            marker_name=marker_name,
            model_file_path=marker_state.model_file,
        )

        req = (
            f'sdf_filename: "{marker_state.model_file}", '
            + f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}, '
            + f'name: "{marker_name}", allow_renaming: false'
        )

        ok, output = self._run_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
        )

        if not ok and "already exists" in output.lower():
            return True

        return ok

    def _set_entity_pose(self, entity_name: str, x: float, y: float, z: float) -> bool:
        req = f'name: "{entity_name}", position: {{x: {x}, y: {y}, z: {z}}}'

        ok, _ = self._run_gz_service(
            service=f"/world/{self.world_name}/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            req=req,
        )

        return ok

    def _boundary_model_file_path(
        self,
        name: str,
        xy_limit: float,
        z_level: float,
    ) -> str:
        safe_name = name.replace("/", "_")

        return os.path.join(
            tempfile.gettempdir(),
            f"{safe_name}_{xy_limit:.3f}_{z_level:.3f}.sdf",
        )

    def _write_boundary_model_file(
        self,
        name: str,
        xy_limit: float,
        z_level: float,
        rgba: Tuple[float, float, float, float],
    ) -> str:
        model_file_path = self._boundary_model_file_path(name, xy_limit, z_level)

        half = float(xy_limit)
        thickness = max(0.005, float(self.boundary_line_thickness))
        wall_height = max(0.05, float(self.boundary_visual_height))
        wall_center_z = wall_height / 2.0
        full = max(0.1, 2.0 * half)
        r, g, b, a = rgba

        model_sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{name}'>
    <static>true</static>
    <pose>0 0 {z_level} 0 0 0</pose>
    <link name='link'>
      <visual name='north'>
        <pose>0 {half} {wall_center_z} 0 0 0</pose>
        <geometry><box><size>{full} {thickness} {wall_height}</size></box></geometry>
        <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
      <visual name='south'>
        <pose>0 {-half} {wall_center_z} 0 0 0</pose>
        <geometry><box><size>{full} {thickness} {wall_height}</size></box></geometry>
        <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
      <visual name='east'>
        <pose>{half} 0 {wall_center_z} 0 0 0</pose>
        <geometry><box><size>{thickness} {full} {wall_height}</size></box></geometry>
        <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
      <visual name='west'>
        <pose>{-half} 0 {wall_center_z} 0 0 0</pose>
        <geometry><box><size>{thickness} {full} {wall_height}</size></box></geometry>
        <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""

        with open(model_file_path, "w", encoding="utf-8") as model_file:
            model_file.write(model_sdf)

        return model_file_path

    def _spawn_or_replace_boundary_model(
        self,
        name: str,
        xy_limit: float,
        z_level: float,
        rgba: Tuple[float, float, float, float],
    ) -> bool:
        self.remove_entity(name)

        model_file_path = self._write_boundary_model_file(
            name=name,
            xy_limit=xy_limit,
            z_level=z_level,
            rgba=rgba,
        )

        req = (
            f'sdf_filename: "{model_file_path}", '
            + f'pose: {{position: {{x: 0, y: 0, z: 0}}}}, '
            + f'name: "{name}", allow_renaming: false'
        )

        ok, output = self._run_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
        )

        if not ok and "already exists" in output.lower():
            return True

        return ok

    # ==================================================================
    # Allocation helpers (used by the factory methods/builders)
    # ==================================================================

    def allocate_port(self) -> int:
        """Return the next UDP port for an extra CrazySim SITL instance."""
        port = self._next_port
        self._next_port += 1
        return port

    def allocate_marker_index(self) -> int:
        """Return the next marker colour index."""
        idx = self._next_marker_index
        self._next_marker_index += 1
        return idx

    def marker_color(self, index: int) -> Tuple[float, float, float, float]:
        """Cycle through the marker colour palette."""
        return self.MARKER_COLORS[index % len(self.MARKER_COLORS)]

    # ==================================================================
    # Factory — CrazyFlie drones
    # ==================================================================

    def create_crazyflie(self, use_simulator: int, uri: Optional[str] = None):
        """
        Create and return a connected CrazyFlie you drive yourself.

        use_simulator=1 returns a DroneSim (CrazySim/Gazebo).
        use_simulator=0 returns a real Drone (radio).
        Lazy imports avoid the circular DroneSim -> SimManager dependency.
        """
        if use_simulator:
            from drone_gym.drone_sim import DroneSim
            if uri is None:
                uri = f"udp://0.0.0.0:{self.allocate_port()}"
            print(f"[SimManager] creating crazyflie at {uri}")
            drone = DroneSim(uri=uri, sim_manager=self)
        else:
            from drone_gym.drone import Drone
            print("[SimManager] creating real crazyflie (radio URI configured externally)")
            drone = Drone()
        self._created.append(drone)
        return drone

    # ==================================================================
    # Factory — expert policies
    # ==================================================================

    def create_policy(self, name: str, **kwargs):
        """Create an expert policy by name (e.g. 'pure_pursuit', 'flee')."""
        if name not in self._policies:
            raise ValueError(f"Unknown policy: {name!r}. Known: {sorted(self._policies)}")
        return self._policies[name](**kwargs)

    def register_policy(self, name: str, factory: Callable) -> None:
        """Register a custom policy. factory(**kwargs) must return a policy."""
        self._policies[name] = factory

    def available_policies(self) -> List[str]:
        return sorted(self._policies)

    # ==================================================================
    # Factory — agents (body + policy pairs)
    # ==================================================================

    def create_agent(self, agent_type: str, role: Optional[str] = None,
                     policy=None, **cfg):
        """
        Create a SimAgent (body + policy) by type name.

        Built-in types: particle_evader, particle_pursuer, virtual_target,
        crazyflie_pursuer, stationary_obstacle.
        """
        if agent_type not in self._agent_builders:
            raise ValueError(
                f"Unknown agent type: {agent_type!r}. Known: {sorted(self._agent_builders)}"
            )
        agent = self._agent_builders[agent_type](role=role, policy=policy, **cfg)
        agent.agent_id = self._next_agent_id
        self._next_agent_id += 1
        self._created.append(agent)
        return agent

    def register_agent_type(self, name: str, builder: Callable) -> None:
        """Register a custom agent type. builder(role, policy, **cfg) -> SimAgent."""
        self._agent_builders[name] = builder

    def available_agent_types(self) -> List[str]:
        return sorted(self._agent_builders)

    # ==================================================================
    # Factory — bare particle (SimulatedBody without a policy)
    # ==================================================================

    def create_particle(self, render: bool = True, fixed_z: float = 1.0,
                        bounds: float = 2.0,
                        color: Optional[Tuple[float, float, float, float]] = None,
                        marker_name: Optional[str] = None,
                        radius: float = 0.08):
        """Create a SimulatedBody — a Gazebo sphere you move purely in software."""
        from drone_gym.agents.bodies import SimulatedBody
        idx = self.allocate_marker_index()
        body = SimulatedBody(
            sim_manager=self,
            fixed_z=fixed_z,
            bounds=bounds,
            render=render,
            marker_name=marker_name or f"rl_particle_{idx}",
            color=color or self.marker_color(idx),
            radius=radius,
        )
        self._created.append(body)
        return body

    # ==================================================================
    # Factory — collision / obstacle objects
    # ==================================================================

    def create_box_obstacle(self, x: float, y: float, z: float,
                            size: Tuple[float, float, float] = (0.2, 0.2, 0.5),
                            rgba: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.8),
                            name: Optional[str] = None) -> str:
        """Spawn a static box obstacle in Gazebo. Returns the entity name."""
        idx = self.allocate_marker_index()
        entity_name = name or f"rl_obstacle_box_{idx}"
        self._spawn_box_entity(entity_name, x, y, z, size, rgba)
        self._created.append(EntityHandle(entity_name, self))
        return entity_name

    def create_cylinder_obstacle(self, x: float, y: float, z: float,
                                 radius: float = 0.1, length: float = 0.5,
                                 rgba: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.8),
                                 name: Optional[str] = None) -> str:
        """Spawn a static cylinder obstacle in Gazebo. Returns the entity name."""
        idx = self.allocate_marker_index()
        entity_name = name or f"rl_obstacle_cyl_{idx}"
        self._spawn_cylinder_entity(entity_name, x, y, z, radius, length, rgba)
        self._created.append(EntityHandle(entity_name, self))
        return entity_name

    # ==================================================================
    # Teardown
    # ==================================================================

    def close_all(self) -> None:
        """Despawn every particle, obstacle, and agent this manager created."""
        from drone_gym.agents.sim_agent import SimAgent
        from drone_gym.agents.bodies import SimulatedBody
        for obj in reversed(self._created):
            try:
                if isinstance(obj, (SimAgent, SimulatedBody)):
                    obj.close()
                elif isinstance(obj, EntityHandle):
                    obj.remove()
                else:
                    # Raw crazyflie handed out by create_crazyflie
                    self._land_crazyflie(obj)
            except Exception as e:
                print(f"[SimManager] Error closing {type(obj).__name__}: {e}")
        self._created.clear()

    # ==================================================================
    # Internal — generic entity spawners (sphere / box / cylinder)
    # ==================================================================

    def _spawn_entity_from_sdf(self, name: str, model_file: str,
                               x: float, y: float, z: float) -> bool:
        req = (
            f'sdf_filename: "{model_file}", '
            + f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}, '
            + f'name: "{name}", allow_renaming: false'
        )
        ok, output = self._run_gz_service(
            service=f"/world/{self.world_name}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
        )
        if not ok and "already exists" in output.lower():
            return True
        return ok

    def _spawn_sphere_entity(self, name: str, x: float, y: float, z: float,
                             rgba: Tuple[float, float, float, float],
                             radius: float) -> bool:
        """Spawn a static coloured sphere (used by SimulatedBody for its marker)."""
        model_file = os.path.join(tempfile.gettempdir(), f"{self._safe_file_name(name)}.sdf")
        self._write_sphere_sdf(model_file, name, rgba, radius)
        return self._spawn_entity_from_sdf(name, model_file, x, y, z)

    def _spawn_box_entity(self, name: str, x: float, y: float, z: float,
                          size: Tuple[float, float, float],
                          rgba: Tuple[float, float, float, float]) -> bool:
        model_file = os.path.join(tempfile.gettempdir(), f"{self._safe_file_name(name)}.sdf")
        self._write_box_sdf(model_file, name, size, rgba)
        return self._spawn_entity_from_sdf(name, model_file, x, y, z)

    def _spawn_cylinder_entity(self, name: str, x: float, y: float, z: float,
                               radius: float, length: float,
                               rgba: Tuple[float, float, float, float]) -> bool:
        model_file = os.path.join(tempfile.gettempdir(), f"{self._safe_file_name(name)}.sdf")
        self._write_cylinder_sdf(model_file, name, radius, length, rgba)
        return self._spawn_entity_from_sdf(name, model_file, x, y, z)

    # ==================================================================
    # Internal — SDF writers
    # ==================================================================

    @staticmethod
    def _write_sphere_sdf(model_file: str, name: str,
                          rgba: Tuple[float, float, float, float],
                          radius: float) -> None:
        r, g, b, a = rgba
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
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
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(sdf)

    @staticmethod
    def _write_box_sdf(model_file: str, name: str,
                       size: Tuple[float, float, float],
                       rgba: Tuple[float, float, float, float]) -> None:
        sx, sy, sz = size
        r, g, b, a = rgba
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
      <visual name='visual'>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
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
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(sdf)

    @staticmethod
    def _write_cylinder_sdf(model_file: str, name: str,
                            radius: float, length: float,
                            rgba: Tuple[float, float, float, float]) -> None:
        r, g, b, a = rgba
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
      </collision>
      <visual name='visual'>
        <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
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
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(sdf)

    # ==================================================================
    # Internal — built-in agent builders
    # Each builder: builder(role, policy, **cfg) -> SimAgent
    # ==================================================================

    def _build_crazyflie_pursuer(self, role=None, policy=None,
                                 max_velocity=0.20, fixed_z=1.0,
                                 boundary_limit=2.0, soft_margin=0.3,
                                 use_simulator=1, target_key="evader_pos",
                                 uri=None, **_):
        from drone_gym.agents.bodies import CrazyflieBody
        from drone_gym.agents.sim_agent import SimAgent
        from drone_gym.agents.policies import PurePursuitPolicy
        if uri is None and use_simulator:
            uri = f"udp://0.0.0.0:{self.allocate_port()}"
            print(f"[SimManager] crazyflie_pursuer at {uri}")
        body = CrazyflieBody(use_simulator, uri=uri, fixed_z=fixed_z)
        if policy is None:
            policy = PurePursuitPolicy(max_velocity, target_key=target_key,
                                       boundary_limit=boundary_limit, soft_margin=soft_margin)
        return SimAgent(0, body, policy, role or "pursuer")

    def _build_particle_pursuer(self, role=None, policy=None,
                                max_velocity=0.30, fixed_z=1.0, bounds=2.0,
                                render=True, target_key="evader_pos", **_):
        from drone_gym.agents.bodies import SimulatedBody
        from drone_gym.agents.sim_agent import SimAgent
        from drone_gym.agents.policies import PurePursuitPolicy
        idx = self.allocate_marker_index()
        body = SimulatedBody(sim_manager=self, fixed_z=fixed_z, bounds=bounds,
                             render=render, marker_name=f"rl_pursuer_particle_{idx}",
                             color=self.marker_color(idx))
        if policy is None:
            policy = PurePursuitPolicy(max_velocity, target_key=target_key)
        return SimAgent(0, body, policy, role or "pursuer")

    def _build_particle_evader(self, role=None, policy=None,
                               max_velocity=0.18, fixed_z=1.0, bounds=2.0,
                               render=True, threat_key="threat_pos",
                               soft_margin=0.3, **_):
        from drone_gym.agents.bodies import SimulatedBody
        from drone_gym.agents.sim_agent import SimAgent
        from drone_gym.agents.policies import FleePolicy
        idx = self.allocate_marker_index()
        body = SimulatedBody(sim_manager=self, fixed_z=fixed_z, bounds=bounds,
                             render=render, marker_name=f"rl_evader_particle_{idx}",
                             color=self.marker_color(idx))
        if policy is None:
            policy = FleePolicy(max_velocity, threat_key=threat_key,
                                boundary_limit=bounds, soft_margin=soft_margin)
        return SimAgent(0, body, policy, role or "evader")

    def _build_virtual_target(self, role=None, policy=None,
                              speed=0.25, fixed_z=1.0, bounds=2.0,
                              render=True, reflect=True, **_):
        from drone_gym.agents.bodies import SimulatedBody
        from drone_gym.agents.sim_agent import SimAgent
        from drone_gym.agents.policies import LineMotionPolicy
        idx = self.allocate_marker_index()
        body = SimulatedBody(sim_manager=self, fixed_z=fixed_z, bounds=bounds,
                             render=render, marker_name=f"rl_target_{idx}",
                             color=self.marker_color(idx))
        if policy is None:
            policy = LineMotionPolicy(speed=speed, bounds=bounds, reflect=reflect)
        return SimAgent(0, body, policy, role or "target")

    def _build_stationary_obstacle(self, role=None, policy=None,
                                   fixed_z=1.0, bounds=2.0, render=True, **_):
        from drone_gym.agents.bodies import SimulatedBody
        from drone_gym.agents.sim_agent import SimAgent
        from drone_gym.agents.policies import StationaryPolicy
        idx = self.allocate_marker_index()
        body = SimulatedBody(sim_manager=self, fixed_z=fixed_z, bounds=bounds,
                             render=render, marker_name=f"rl_obstacle_{idx}",
                             color=self.marker_color(idx))
        if policy is None:
            policy = StationaryPolicy()
        return SimAgent(0, body, policy, role or "obstacle")

    # ==================================================================
    # Internal — land a raw crazyflie handed out by create_crazyflie
    # ==================================================================

    @staticmethod
    def _land_crazyflie(drone) -> None:
        if getattr(drone, "velocity_controller_active", False):
            drone.stop_velocity_control()
        drone.set_velocity_vector(0, 0, 0)
        drone.land()
        drone.is_landed_event.wait(timeout=30)
        drone.stop()


_DEFAULT_SIM_MANAGER: Optional[SimManager] = None
_DEFAULT_SIM_MANAGER_LOCK = threading.Lock()


def get_default_sim_manager() -> SimManager:
    """
    Return a shared default SimManager.

    This lets the current single-agent code keep using DroneSim() with no
    extra arguments, while also making it possible for future multiple
    DroneSim objects to share one simulation manager.
    """
    global _DEFAULT_SIM_MANAGER

    with _DEFAULT_SIM_MANAGER_LOCK:
        if _DEFAULT_SIM_MANAGER is None:
            _DEFAULT_SIM_MANAGER = SimManager()

        return _DEFAULT_SIM_MANAGER
