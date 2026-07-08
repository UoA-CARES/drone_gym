import math
import os
import subprocess
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field

@dataclass
class MarkerVisualSettings:
    """
    Visual/model-level settings for a Gazebo target marker.

    These describe what the marker looks like and how often it should be
    retried/updated. 
    """
    radius: float = 0.02
    rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.6)
    specular: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)
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
    last_pose: tuple[float, float, float] | None = None
    model_file: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

@dataclass
class SimLaunchConfig:
    """
    Configuration for launching and managing the CrazySim/Gazebo simulator process.

    Args:
        firmware_dir: Path to the CrazySim Crazyflie firmware directory. This is
            used as the working directory when running the launch script. Defaults to
            "CrazySim/crazyflie-firmware".
        launch_script: Relative path to the CrazySim/Gazebo launch script from
            inside firmware_dir. Defaults to
            "tools/crazyflie-simulation/simulator_files/gazebo/launch/sitl_multiagent_square.sh".
        model: Name of the robot model to launch. Defaults to "crazyflie".
        num_agents: Number of simulated Crazyflie agents to launch. Defaults to 1.
        startup_timeout: Maximum time, in seconds, to wait for the simulator to
            become responsive after launching. Defaults to 60.0.
        shutdown_timeout: Maximum time, in seconds, to wait for the simulator
            process to shut down cleanly before forcing termination. Defaults to
            10.0.
        restart_delay: Time, in seconds, to wait between stopping and restarting
            the simulator. Defaults to 8.0.
        log_file: Path to the file where simulator stdout and stderr output should
            be written. Defaults to "logs/crazysim.log".
    """

    firmware_dir: str = "CrazySim/crazyflie-firmware"
    launch_script: str = (
        "tools/crazyflie-simulation/simulator_files/gazebo/launch/"
        "sitl_multiagent_square.sh"
    )

    model: str = "crazyflie"
    num_agents: int = 1

    startup_timeout: float = 60.0
    shutdown_timeout: float = 10.0
    restart_delay: float = 8.0

    log_file: str = "logs/crazysim.log"

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
        sim_launch_config: SimLaunchConfig | None = None,
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
    ) -> None:

        # Gazebo launch configuration
        self.world_name = world_name
        self.sim_launch_config = sim_launch_config
        self.sim_process: subprocess.Popen | None = None
        self._sim_process_lock = threading.Lock()
        self._sim_log_handle = None

        # Target marker settings/state
        self.enable_target_marker = enable_target_marker
        self.marker_name = marker_name
        
        self.marker_visual_settings = MarkerVisualSettings(
            pose_tolerance=marker_pose_tolerance,
            retry_backoff=marker_retry_backoff,
        )

        self._marker_states: dict[str, MarkerState] = {}
        self._marker_states_lock = threading.Lock()

        # Boundary visual settings/state
        self.enable_boundary_lines = enable_boundary_lines
        self.boundary_name = boundary_name
        self.boundary_retry_backoff = boundary_retry_backoff
        self.boundary_line_thickness = boundary_line_thickness
        self.boundary_visual_height = boundary_visual_height
        self._last_boundary_signature: tuple[float, float] | None = None
        self._last_boundary_attempt_time = 0.0
        self._boundary_lock = threading.Lock()

        # Gazebo command settings
        self.gz_timeout = gz_timeout

    def start_sim(self) -> bool:
        """
        Start the CrazySim/Gazebo simulator process if it is not already running.

        Returns:
            True if the simulator process is running or was launched successfully.
            False if the simulator could not be launched.
        """
        if self.sim_launch_config is None:
            raise RuntimeError(
                "Cannot start simulator because sim_launch_config was not provided."
            )

        with self._sim_process_lock:
            if self.is_sim_process_alive():
                print("[SIM MANAGER] Simulator process is already running.")
                return True

            firmware_dir = self.sim_launch_config.firmware_dir
            if not os.path.isdir(firmware_dir):
                print(
                    "[SIM MANAGER] Cannot start simulator. "
                    f"Firmware directory does not exist: {firmware_dir}"
                )
                return False

            launch_script_path = os.path.join(
                firmware_dir,
                self.sim_launch_config.launch_script,
            )
            if not os.path.isfile(launch_script_path):
                print(
                    "[SIM MANAGER] Cannot start simulator. "
                    f"Launch script does not exist: {launch_script_path}"
                )
                return False

            log_dir = os.path.dirname(self.sim_launch_config.log_file)

            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            cmd =[
                "bash",
                self.sim_launch_config.launch_script,
                "-m",
                self.sim_launch_config.model,
                "-n",
                str(self.sim_launch_config.num_agents),
            ]

            try:
                self._sim_log_handle = open(self.sim_launch_config.log_file, "a", encoding="utf-8")

                print("[SIM MANAGER] Starting CrazySim/Gazebo...")
                print(f"[SIM MANAGER] Command: {' '.join(cmd)}")
                print(f"[SIM MANAGER] Working directory: {firmware_dir}")
                print(f"[SIM MANAGER] Log file: {self.sim_launch_config.log_file}")

                self.sim_process = subprocess.Popen(
                    cmd,
                    cwd=firmware_dir,
                    stdout=self._sim_log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )

            except OSError as exc:
                print(f"[SIM MANAGER] Failed to start simulator: {exc}")
                self._close_sim_log_handle()
                self.sim_process = None
                return False

            return True

    def is_sim_process_alive(self) -> bool:
        """
        Return True if SimManager has launched a simulator process and it is still running.
        """
        return self.sim_process is not None and self.sim_process.poll() is None
    
    def _close_sim_log_handle(self) -> None:
        """
        Close the simulator log file handle if it is open.
        """
        if self._sim_log_handle is not None:
            self._sim_log_handle.close()
            self._sim_log_handle = None

    def stop_sim(self) -> None:
        """
        Stop the CrazySim/Gazebo simulator process launched by SimManager.

        The simulator is launched with start_new_session=True, so this method first
        tries to terminate the whole process group. If the simulator does not stop
        within shutdown_timeout, the process group is force-killed.
        """
        with self._sim_process_lock:
            if self.sim_process is None:
                print("[SIM MANAGER] No simulator process to stop.")
                self._close_sim_log_handle()
                return

            process = self.sim_process

            if process.poll() is not None:
                print(
                    "[SIM MANAGER] Simulator process has already stopped "
                    f"with return code {process.returncode}."
                )
                self.sim_process = None
                self._close_sim_log_handle()
                return

            shutdown_timeout = 10.0
            if self.sim_launch_config is not None:
                shutdown_timeout = self.sim_launch_config.shutdown_timeout

            print("[SIM MANAGER] Stopping CrazySim/Gazebo...")

            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=shutdown_timeout)
                print("[SIM MANAGER] Simulator stopped cleanly.")

            except subprocess.TimeoutExpired:
                print(
                    "[SIM MANAGER] Simulator did not stop within "
                    f"{shutdown_timeout} seconds. Force killing..."
                )

                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=5.0)
                    print("[SIM MANAGER] Simulator force killed.")

                except OSError as exc:
                    print(f"[SIM MANAGER] Failed to force kill simulator: {exc}")

            except ProcessLookupError:
                print("[SIM MANAGER] Simulator process was already gone.")

            except OSError as exc:
                print(
                    "[SIM MANAGER] Failed to stop simulator process group. "
                    f"Trying direct process termination. Error: {exc}"
                )

                try:
                    process.terminate()
                    process.wait(timeout=shutdown_timeout)
                    print("[SIM MANAGER] Simulator stopped using direct termination.")

                except subprocess.TimeoutExpired:
                    print("[SIM MANAGER] Direct termination timed out. Force killing process.")
                    process.kill()
                    process.wait(timeout=5.0)

            finally:
                self.sim_process = None
                self._close_sim_log_handle()

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
        marker_name: str | None = None,
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
    
    def remove_visual_marker(self, marker_name: str | None = None) -> bool:
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
    ) -> tuple[bool, str]:
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

    def _set_entity_pose(
        self, 
        entity_name: str,
        x: float, y: float, z: float,
        orientation: tuple[float, float, float] | None = None,
    ) -> bool:
        """
        Set an entity's world position and optionally its orientation.

        Args:
            entity_name: Gazebo entity name.
            x: World x position in metres.
            y: World y position in metres.
            z: World z position in metres.
            orientation: Optional (roll, pitch, yaw) tuple in radians.
                When None, only position is changed and the current
                orientation is preserved.
        """
        req_parts = [
            f'name: "{entity_name}"',
            (
                "position: {"
                f"x: {float(x)}, "
                f"y: {float(y)}, "
                f"z: {float(z)}"
                "}"
            ),
        ]        
        
        if orientation is not None:
 
            roll, pitch, yaw = orientation

            qx, qy, qz, qw = self._rpy_to_quaternion(
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
            )

            req_parts.append(
                "orientation: {"
                f"x: {qx}, "
                f"y: {qy}, "
                f"z: {qz}, "
                f"w: {qw}"
                "}"
            )

        req = ", ".join(req_parts)

        ok, _ = self._run_gz_service(
            service=f"/world/{self.world_name}/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            req=req,
        )

        return ok

    def set_drone_pose(
        self, 
        drone_id: str, 
        x: float, y: float, z: float,
        orientation: tuple[float, float, float] | None = None,
        ) -> bool:
        """
        Set a Crazyflie model's position and optionally its orientation.

        Args:
            orientation: Optional (roll, pitch, yaw) in radians.
                When None, the current orientation is preserved.
        """
        drone_name = f"crazyflie_{drone_id}"
        return self._set_entity_pose(drone_name, x, y, z, orientation)

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
        rgba: tuple[float, float, float, float],
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
        rgba: tuple[float, float, float, float],
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

    @staticmethod
    def _rpy_to_quaternion(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> tuple[float, float, float, float]:
        """
        Convert roll, pitch and yaw in radians into a quaternion.

        Returns:
            Quaternion in Gazebo's x, y, z, w order.
        """
        cr = math.cos(roll / 2.0)
        sr = math.sin(roll / 2.0)
        cp = math.cos(pitch / 2.0)
        sp = math.sin(pitch / 2.0)
        cy = math.cos(yaw / 2.0)
        sy = math.sin(yaw / 2.0)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy

        return qx, qy, qz, qw


_DEFAULT_SIM_MANAGER: SimManager | None = None
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