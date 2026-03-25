import queue
import time
import os
import subprocess
import tempfile
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from drone_gym.drone_setup import DroneSetup
import warnings

warnings.filterwarnings('ignore', message='Using legacy TYPE_HOVER_LEGACY')


class DroneSim(DroneSetup):
    def __init__(self, uri="udp://0.0.0.0:19850", simulation=True):
        super().__init__(uri=uri)
        # Drone Properties
        self.simulation = simulation

        # Gazebo target marker 
        self._enable_target_marker = True
        self._marker_world = "crazysim_default"
        self._marker_name = "rl_target_marker"
        self._marker_retry_backoff = 1.0
        self._marker_spawned = False
        self._last_marker_attempt_time = 0.0
        self._last_marker_pose = None
        self._marker_lock = threading.Lock()
        self._marker_model_file = os.path.join(
            tempfile.gettempdir(), f"{self._marker_name}.sdf"
        )

        # Gazebo boundary line visuals 
        self._enable_boundary_lines = True
        self._boundary_retry_backoff = 1.0
        self._boundary_line_thickness = 0.02
        self._boundary_visual_height = 0.8
        self._boundary_name = "rl_boundary"
        self._last_boundary_signature = None
        self._last_boundary_attempt_time = 0.0
        self._boundary_lock = threading.Lock()

    def _update_position(self):
        """Update position from Gazebo via state estimate logs"""
        print("[Drone] Position tracking thread started")
        
        # Wait for CF to be ready
        timeout = time.time() + 10
        while self.cf is None and time.time() < timeout:
            time.sleep(0.1)
        
        if self.cf is None:
            print("[Drone] ERROR: Crazyflie not initialized")
            return

        try:
            # Setup position logging from state estimate
            position_log_config = LogConfig(name="Position", period_in_ms=50)
            position_log_config.add_variable("stateEstimate.x", "float")
            position_log_config.add_variable("stateEstimate.y", "float")
            position_log_config.add_variable("stateEstimate.z", "float")

            self.cf.log.add_config(position_log_config)
            position_log_config.data_received_cb.add_callback(self._position_callback)
            position_log_config.start()
            print("[Drone] Position logging started")

            # Wait for first position
            time.sleep(2)
            self.position_ready_event.set()
            self.last_velocity_calculation_time = time.time()

            # Keep thread alive for logging
            while self.is_running() and not self.emergency_event.is_set():
                current_time = time.time()
                
                # Calculate velocity periodically
                if (current_time - self.last_velocity_calculation_time) >= self.velocity_update_rate:
                    if len(self.position_history) >= 2:
                        self._calculate_velocity()
                    self.last_velocity_calculation_time = current_time
                
                time.sleep(0.05)

        except Exception as e:
            print(f"[Drone] Error in position thread: {str(e)}")
        finally:
            if 'position_log_config' in locals():
                try:
                    position_log_config.stop()
                    position_log_config.delete()
                except:
                    pass

    def initialise_crazyflie(self):
        """Initialise Crazyflie connection for CrazySim"""
        try:
            cflib.crtp.init_drivers()
            print("[Drone] Initializing CRTP drivers...")
            
            # Add retries for connection
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    print(f"[Drone] Connection attempt {attempt + 1}/{max_retries} to {self.URI}...")
                    self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache="./cache"))
                    self.scf.open_link()
                    self.cf = self.scf.cf
                    print("[Drone] Successfully connected to CrazySim!")
                    break
                except Exception as e:
                    print(f"[Drone] Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"[Drone] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("[Drone] All connection attempts failed")
                        print("[Drone] Make sure:")
                        print("  1. Gazebo is running")
                        print("  2. The SITL firmware is started")
                        print("  3. The drone model is spawned in Gazebo")
                        raise

            # For CrazySim, we can skip deck detection or set it immediately I believe
            print("[Drone] Setting deck attached (simulated)")
            self.deck_attached_event.set()

            print("[Drone] Waiting for firmware to be ready...")
            time.sleep(2)

            print("[Drone] Resetting all log configurations")
            self.cf.log.reset()
            time.sleep(0.5)

            print("[Drone] Resetting state estimation (EKF)...")
            self.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.5)

            # Arm the drone
            print("[Drone] Arming Crazyflie...")
            self.cf.platform.send_arming_request(True)
            time.sleep(1.5)
            self.armed = True
            print("[Drone] Crazyflie armed.")

            self._setup_battery_logging()
            self._setup_velocity_logging()

            # Signal that hardware is ready
            self.hardware_ready_event.set()
            print("[Drone] Hardware initialisation complete - ready to fly!")
            return True

        except Exception as e:
            print(f"[Drone] Failed to initialize Crazyflie: {str(e)}")
            return False

    def _position_callback(self, timestamp, data, logconf):
        """Callback for position data from Gazebo"""
        current_time = time.time()
        
        with self.position_lock:
            self.position = {
                "x": data["stateEstimate.x"],
                "y": data["stateEstimate.y"],
                "z": data["stateEstimate.z"],
            }
            current_pos = self.position.copy()
        
        # Store for velocity calculation
        self.position_history.append((current_time, current_pos))

    def set_visual_target_marker_position(self, x: float, y: float, z: float) -> None:
        """Update Gazebo visual marker for task targets"""
        self._update_target_marker_if_enabled(x, y, z)

    def _update_target_marker_if_enabled(self, x, y, z):
        if not self._enable_target_marker:
            return

        now = time.time()

        with self._marker_lock:
            if self._last_marker_pose is not None:
                dx = abs(x - self._last_marker_pose[0])
                dy = abs(y - self._last_marker_pose[1])
                dz = abs(z - self._last_marker_pose[2])
                if dx < 0.005 and dy < 0.005 and dz < 0.005:
                    return

            if not self._marker_spawned:
                if now - self._last_marker_attempt_time < self._marker_retry_backoff:
                    return

                self._last_marker_attempt_time = now
                self._marker_spawned = self._spawn_target_marker(x, y, z)
                if not self._marker_spawned:
                    return

            if not self._set_target_marker_pose(x, y, z):
                # World may have restarted. Mark as missing and retry on next calls.
                self._marker_spawned = False
                return

            self._last_marker_pose = (x, y, z)

    def set_visual_boundary_lines(self, drone_xy_limit: float, z_level: float) -> None:
        """Draw a single square boundary line overlay shared by drone and target zones"""
        if not self._enable_boundary_lines:
            return

        signature = (
            round(float(drone_xy_limit), 3),
            round(float(z_level), 3),
        )

        with self._boundary_lock:
            if signature == self._last_boundary_signature:
                return

            now = time.time()
            if now - self._last_boundary_attempt_time < self._boundary_retry_backoff:
                return
            self._last_boundary_attempt_time = now

            boundary_ok = self._spawn_or_replace_boundary_model(
                name=self._boundary_name,
                xy_limit=max(0.05, float(drone_xy_limit)),
                z_level=float(z_level),
                rgba=(0.2, 0.55, 1.0, 0.8),
            )

            if boundary_ok:
                self._last_boundary_signature = signature

    def _run_gz_service(self, service, reqtype, reptype, req):
        """Call Gazebo transport service using gz CLI."""
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
                timeout=1.5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, ""

        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            return False, output

        # Some Gazebo service responses omit explicit `data: true/false` text.
        # Still treat explicit Gazebo runtime errors as failures.
        output_lower = output.lower()
        if "[err]" in output_lower:
            return False, output
        if "data: false" in output_lower:
            return False, output
        if "data: true" in output_lower:
            return True, output
        return True, output

    def _ensure_marker_model_file(self):
        model_sdf = f"""<?xml version='1.0'?>
<sdf version='1.9'>
  <model name='{self._marker_name}'>
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

    def _spawn_target_marker(self, x, y, z):
        self._ensure_marker_model_file()
        req = (
            f'sdf_filename: "{self._marker_model_file}", '
            + f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}, '
            + f'name: "{self._marker_name}", allow_renaming: false'
        )
        ok, output = self._run_gz_service(
            service=f"/world/{self._marker_world}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
        )
        if not ok and "already exists" in output.lower():
            return True
        return ok

    def _set_target_marker_pose(self, x, y, z):
        req = f'name: "{self._marker_name}", position: {{x: {x}, y: {y}, z: {z}}}'
        ok, _ = self._run_gz_service(
            service=f"/world/{self._marker_world}/set_pose",
            reqtype="gz.msgs.Pose",
            reptype="gz.msgs.Boolean",
            req=req,
        )
        return ok

    def _remove_entity_if_exists(self, entity_name):
        req = f'name: "{entity_name}"'
        self._run_gz_service(
            service=f"/world/{self._marker_world}/remove",
            reqtype="gz.msgs.Entity",
            reptype="gz.msgs.Boolean",
            req=req,
        )

    def _boundary_model_file_path(self, name, xy_limit, z_level):
        safe_name = name.replace("/", "_")
        return os.path.join(
            tempfile.gettempdir(),
            f"{safe_name}_{xy_limit:.3f}_{z_level:.3f}.sdf",
        )

    def _write_boundary_model_file(self, name, xy_limit, z_level, rgba):
        model_file_path = self._boundary_model_file_path(name, xy_limit, z_level)
        half = float(xy_limit)
        thickness = max(0.005, float(self._boundary_line_thickness))
        wall_height = max(0.05, float(self._boundary_visual_height))
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

    def _spawn_or_replace_boundary_model(self, name, xy_limit, z_level, rgba):
        self._remove_entity_if_exists(name)

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
            service=f"/world/{self._marker_world}/create",
            reqtype="gz.msgs.EntityFactory",
            reptype="gz.msgs.Boolean",
            req=req,
        )

        if not ok and "already exists" in output.lower():
            return True
        return ok

    def stop(self):
        """
        Fully stop the drone and optionally prepare for a clean restart.

        Args

        """
        print("[Drone] Stopping...")
        self.set_running(False)
        self.controller_active = False
        self.velocity_controller_active = False
        
        # Clear queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

        print("[Drone] Stopped")
        self._signal_stop_to_all_threads()
        self._join_all_threads()
        self._reset_shared_state()
        self._final_cleanup()

if __name__ == "__main__":
    drone = DroneSim()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if drone.is_flying_event.is_set():
        print("Hovering for 5 seconds...")
        time.sleep(5)
        
        print("Moving forward...")
        drone.set_velocity_vector(0.2, 0, 0)
        time.sleep(3)
        
        drone.set_velocity_vector(0, 0, 0)
        time.sleep(1)

    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()