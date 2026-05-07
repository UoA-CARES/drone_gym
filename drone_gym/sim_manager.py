import os
import subprocess
import tempfile
import threading
import time
from typing import Optional, Tuple


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
        self.marker_retry_backoff = marker_retry_backoff
        self.marker_pose_tolerance = marker_pose_tolerance
        self._marker_spawned = False
        self._last_marker_attempt_time = 0.0
        self._last_marker_pose = None
        self._marker_lock = threading.Lock()
        self._marker_model_file = os.path.join(
            tempfile.gettempdir(),
            f"{self.marker_name}.sdf",
        )

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

    def set_visual_target_marker_position(self, x: float, y: float, z: float) -> None:
        """
        Spawn or update the Gazebo visual marker for a task target.
        """
        if not self.enable_target_marker:
            return

        now = time.time()

        with self._marker_lock:
            if self._last_marker_pose is not None:
                dx = abs(x - self._last_marker_pose[0])
                dy = abs(y - self._last_marker_pose[1])
                dz = abs(z - self._last_marker_pose[2])

                if (
                    dx < self.marker_pose_tolerance
                    and dy < self.marker_pose_tolerance
                    and dz < self.marker_pose_tolerance
                ):
                    return

            if not self._marker_spawned:
                if now - self._last_marker_attempt_time < self.marker_retry_backoff:
                    return

                self._last_marker_attempt_time = now
                self._marker_spawned = self._spawn_target_marker(x, y, z)

                if not self._marker_spawned:
                    return

            if not self._set_entity_pose(self.marker_name, x, y, z):
                # The Gazebo world may have restarted or the marker may have been deleted.
                # Mark it as missing and retry on a later call.
                self._marker_spawned = False
                return

            self._last_marker_pose = (x, y, z)

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
        with self._marker_lock:
            self._marker_spawned = False
            self._last_marker_pose = None
            self._last_marker_attempt_time = 0.0

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

    def _write_target_marker_model_file(self) -> None:
        model_sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{self.marker_name}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <sphere>
            <radius>0.02</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1 0 0 0.6</ambient>
          <diffuse>1 0 0 0.6</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

        with open(self._marker_model_file, "w", encoding="utf-8") as model_file:
            model_file.write(model_sdf)

    def _spawn_target_marker(self, x: float, y: float, z: float) -> bool:
        self._write_target_marker_model_file()

        req = (
            f'sdf_filename: "{self._marker_model_file}", '
            + f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}, '
            + f'name: "{self.marker_name}", allow_renaming: false'
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