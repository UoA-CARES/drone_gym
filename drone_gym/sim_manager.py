import os
import subprocess
import tempfile
import threading
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

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

class SimManager:
    """
    World-level simulation manager for Gazebo/CrazySim.

    This class should own simulation-wide responsibilities such as:
    - Gazebo service calls
    - target visual markers
    - boundary visual markers

    It should not own per-drone control logic such as:
    - Crazyflie connection
    - position logging
    - velocity commands
    - takeoff/landing
    """

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