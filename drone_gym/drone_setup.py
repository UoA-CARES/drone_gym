import threading
import queue
import time
import math
from threading import Event
from collections import deque
from typing import Any

from cflib.crazyflie.log import LogConfig
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from drone_gym.utils.position_source import (
    PositionSample,
    PositionSource,
)


class DroneSetup:
    def __init__(
            self,
            uri: str | None = None,
            agent_id: str = "Drone",
            boundaries: dict[str, float] | None = None,
            position_source: PositionSource | None = None,
        ) -> None:
        # Drone Properties

        self.URI = uri if uri is not None else uri_helper.uri_from_env(
            default="radio://0/100/2M/E7E7E7E7E7"
        )  # changed radio channel in 22/9
        self.agent_id = agent_id
        self.default_height = 0.5
        self.deck_attached_event = Event()
        self.battery_lock = threading.Lock()
        self.battery_log_config = None
        self.battery_level = 5.0  # default value
        self.velocity_log_lock = threading.Lock()
        self.velocity_log_config = None
        self.internal_vx = 0.0
        self.internal_vy = 0.0
        self.internal_vz = 0.0

        self._last_position_print = 0

        # Drone Events
        self.is_flying_event = Event()
        self.is_landed_event = Event()
        self.is_landed_event.set()

        # Command processing
        self.command_queue = queue.Queue()
        self.velocity = 0.0
        self.velocity_lock = threading.Lock()
        self.running = True
        self.running_lock = threading.Lock()

        # Controller parameters
        self.position_controller_active = False
        self.controller_thread = None
        self.at_reset_position = Event()

        # Velocity controller parameters
        self.velocity_controller_active = False
        self.velocity_controller_thread = None
        self.velocity_control_rate = 0.1  # 10Hz velocity control rate

        # filtered velocity tracking
        self.velocity_filter_alpha = 1

        # PID gains - separate for each axis for better tuning
        self.gains = {
            "x": {"kp": 0.35, "kd": 0.1425, "ki": 0.08},
            "y": {"kp": 0.35, "kd": 0.1425, "ki": 0.08},
            "z": {"kp": 0.4, "kd": 0.03, "ki": 0},
        }
        self.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.integral = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Velocity control PID gains and state
        self.velocity_gains = {
            "x": {"kp": 0.105, "kd": 0.05, "ki": 0.03},
            "y": {"kp": 0.105, "kd": 0.05, "ki": 0.03},  # Higher Kd for Y due to oscillations
            "z": {"kp": 0.05, "kd": 0.01, "ki": 0.0},
            # "z": {"kp": 0, "kd": 0, "ki": 0},
        }
        self.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.max_velocity = {"x": 0.40, "y": 0.40, "z": 0.40} # Maximum velocity in m/s
        # overridden by environment and task max_vel when vel is set using set_velocity_vector and velocity controller is active
        self.position_deadband = (
            0.05  # Position error below which velocity will be zero (in meters)
            # Changed from 0.1 to 0.05 along with position control error
        )

        # NEW: Velocity ramping for smooth transitions
        self.current_commanded_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.max_velocity_change_rate = 100 #Maximum change in velocity per second (m/s²)
        self.velocity_command_lock = threading.Lock()

        # Crazyflie objects - will be initialized in _run
        self.scf: SyncCrazyflie | None = None
        self.cf: Crazyflie | None = None
        self.mc: MotionCommander | None = None
        self.armed = False

        # Drone position tracking
        self.position_source = position_source
        self.last_position_update_time: float | None = None
        self._position_source_started = False
        self._position_source_lock = threading.Lock()
        # Vicon Integration
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.position_thread = None
        self.position_lock = threading.Lock()

        # Velocity calculation (from position differentiation)
        self.calculated_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity_calculation_lock = threading.Lock()
        self.position_history = deque(maxlen=15)  # Store last 15 positions for moving average
        self.velocity_update_rate = 0.20  # 20Hz velocity calculation rate
        self.position_update_rate = 0.0166666  # 60Hz position update rate
        self.last_velocity_calculation_time = 0.0

        # Drone Safety
        self.boundaries = {"x": 2.5, "y": 2.5, "z_min": 0.1, "z_max": 3.0} if boundaries is None else boundaries
        self.safety_thread = None
        self.in_boundaries = True
        self.emergency_event = Event()
        self.safety_thread_active = False

        # Objective related
        self.target_position = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Thread coordination events
        self.hardware_ready_event = Event()
        self.position_ready_event = Event()

        # Debugging parameters
        self.control_target_velocities = [0, 0, 0]
        self.control_target_lock = threading.Lock()

        # Start threads in coordinated sequence
        self.thread = threading.Thread(target=self._run)
        self._start_threads_coordinated()

    def _start_threads_coordinated(self):
        print(f"[{self.agent_id}] Starting threads...")

        # Start main thread first (initialises hardware)
        self.thread.start()
        print(f"[{self.agent_id}] Main thread started, waiting for hardware initialization...")

        # Wait for hardware to be ready
        if not self.hardware_ready_event.wait(timeout=15):
            print(f"[{self.agent_id}] ERROR: Hardware failed to initialize in time")
            print(f"[{self.agent_id}] IF SIM: Make sure Gazebo is running and the drone model is loaded")
            self.set_running(False)
            return

        print(f"[{self.agent_id}] Hardware ready, starting position tracking...")

        # Start position tracking after hardware is ready
        if not self._start_position_tracking():
            print(
                f"[{self.agent_id}] ERROR: Position tracking failed to start"
            )
            self.set_running(False)
            self.command_queue.put("exit")
            return

        # Wait for position system to stabilize
        if not self.position_ready_event.wait(timeout=10):
            print(f"[{self.agent_id}] WARNING: Position system may not be ready")
        else:
            print(f"[{self.agent_id}] Position system ready, starting safety monitoring...")

        # Start safety monitoring last
        self.safety_thread_active = True
        self.safety_thread = threading.Thread(target=self._check_boundaries)
        self.safety_thread.start()

        print(f"[{self.agent_id}] All threads started successfully with coordination")

    def _check_boundaries(self) -> None:
        """Monitor drone boundaries with safety checks for valid position data"""
        print(f"[{self.agent_id}] Boundary monitoring thread started")

        # Wait for position system to be ready before checking boundaries
        if not self.position_ready_event.wait(timeout=15):
            print(
                f"[{self.agent_id}] WARNING: Starting boundary check without confirmed position data"
            )

        # Additional startup delay to let system stabilize
        time.sleep(2)
        print(f"[{self.agent_id}] Boundary checking now active")

        while self.is_running() and self.safety_thread_active:
            try:
                if self.emergency_event.is_set():
                    break

                # Safety check: only monitor boundaries if position system is ready
                if not self.position_ready_event.is_set():
                    time.sleep(0.1)
                    continue

                # Check if position is within boundaries for each axis
                with self.position_lock:
                    current_pos = self.position.copy()

                # Additional safety: skip check if position is clearly invalid (all zeros)
                if (
                    current_pos["x"] == 0.0
                    and current_pos["y"] == 0.0
                    and current_pos["z"] == 0.0
                ):
                    time.sleep(0.01)
                    continue

                x_in_bounds = self.boundaries["x"] >= abs(current_pos["x"])
                y_in_bounds = self.boundaries["y"] >= abs(current_pos["y"])
                # z_in_bounds = self.boundaries["z_min"] <= current_pos["z"] <= self.boundaries["z_max"]
                z_in_bounds = current_pos["z"] <= self.boundaries["z_max"]

                # Set in_boundaries status
                in_bounds = x_in_bounds and y_in_bounds and z_in_bounds
                # Start threads in coordinated sequence

                if not in_bounds:
                    self.in_boundaries = False
                    print(
                        f"[{self.agent_id}] BOUNDARY VIOLATION: Position {current_pos} exceeds limits {self.boundaries}"
                    )
                    self._execute_emergency_stop()
                    break
                else:
                    self.in_boundaries = True

                time.sleep(0.01)
            except Exception as e:
                print(f"[{self.agent_id}] Error in boundary checking: {str(e)}")
                time.sleep(0.1)

        print(f"[{self.agent_id}] Boundary monitoring thread stopped")

    def _execute_emergency_stop(self) -> None:
        if not self.emergency_event.is_set():
            self.emergency_event.set()
            print(f"[{self.agent_id}] Emergency stop event triggered!!!")

        if self.mc:
            self.mc.land()
            time.sleep(1)

        self.position_controller_active = False

        if self.armed and self.cf:
            # self.cf.platform.send_arming_request(False)
            self.cf.supervisor.send_arming_request(False)
            self.armed = False

        self.command_queue.put("exit")
        self.stop()

    def is_running(self) -> bool:
        """Check if the drone is running"""
        with self.running_lock:
            return self.running

    def set_running(self, value: bool) -> None:
        """Set the running state"""
        with self.running_lock:
            self.running = value

    def _update_position(self) -> None:
        """
        This method is meant to be overridden by any child classes.
        """
        pass

    def _handle_position_sample(
        self,
        sample: PositionSample,
    ) -> None:
        """
        Validate and store a position received from the configured source.

        All position sources publish through this function so that position
        storage, history, readiness and velocity handling are source-independent.
        """

        coordinates = (sample.x, sample.y, sample.z)

        if not all(math.isfinite(value) for value in coordinates):
            source_name = (
                self.position_source.name
                if self.position_source is not None
                else "unknown source"
            )

            print(
                f"[{self.agent_id}] Ignoring invalid position sample "
                f"from {source_name}: {coordinates}"
            )
            return

        position = sample.as_dict()

        with self.position_lock:
            self.position = position

            self.position_history.append(
                (
                    sample.timestamp,
                    position.copy(),
                )
            )

            self.last_position_update_time = sample.timestamp

        if not self.position_ready_event.is_set():
            self.position_ready_event.set()

            source_name = (
                self.position_source.name
                if self.position_source is not None
                else "position source"
            )

            print(
                f"[{self.agent_id}] First valid position received "
                f"from {source_name}: {position}"
            )

        if (
            sample.timestamp - self.last_velocity_calculation_time
            >= self.velocity_update_rate
        ):
            self._calculate_velocity()
            self.last_velocity_calculation_time = sample.timestamp

    def _start_position_tracking(self) -> bool:
        """
        Start the configured position source.

        If no PositionSource has been supplied, temporarily fall back to the
        existing subclass _update_position() implementation.
        """

        self.position_ready_event.clear()
        self.last_position_update_time = None
        self.last_velocity_calculation_time = 0.0

        if self.position_source is None:
            print(
                f"[{self.agent_id}] No PositionSource configured. "
                "Using legacy _update_position() implementation."
            )

            self.position_thread = threading.Thread(
                target=self._update_position,
                name=f"{self.agent_id}-legacy-position",
            )
            self.position_thread.start()
            return True

        with self._position_source_lock:
            if self._position_source_started:
                print(
                    f"[{self.agent_id}] Position source "
                    f"{self.position_source.name} is already running"
                )
                return True

            try:
                print(
                    f"[{self.agent_id}] Starting position source: "
                    f"{self.position_source.name}"
                )

                self.position_source.start(
                    callback=self._handle_position_sample,
                )

                self._position_source_started = True
                return True

            except Exception as exc:
                print(
                    f"[{self.agent_id}] Failed to start position source "
                    f"{self.position_source.name}: {exc}"
                )
                return False
        
    def _stop_position_tracking(self) -> None:
        """
        Stop the configured PositionSource.

        Legacy _update_position() threads are stopped through self.running and
        joined by _join_all_threads().
        """

        if self.position_source is None:
            return

        with self._position_source_lock:
            if not self._position_source_started:
                return

            try:
                print(
                    f"[{self.agent_id}] Stopping position source: "
                    f"{self.position_source.name}"
                )

                self.position_source.stop()

            except Exception as exc:
                print(
                    f"[{self.agent_id}] Error stopping position source "
                    f"{self.position_source.name}: {exc}"
                )

            finally:
                self._position_source_started = False

    def _calculate_velocity(self) -> None:
        """Calculate velocity using moving average filter over position history with additional low-pass filtering"""

        with self.position_lock:
            position_history = list(self.position_history)
        if len(position_history) < 2:
            return

        velocities = {"x": [], "y": [], "z": []}

        for i in range(1, len(position_history)):
            t_prev, pos_prev = position_history[i - 1]
            t_curr, pos_curr = position_history[i]

            dt = t_curr - t_prev

            if dt > 0:
                for axis in ["x", "y", "z"]:
                    vel = (pos_curr[axis] - pos_prev[axis]) / dt
                    velocities[axis].append(vel)

        # Apply moving average filter
        with self.velocity_calculation_lock:
            for axis in ["x", "y", "z"]:
                if len(velocities[axis]) > 0:
                    raw_velocity = sum(velocities[axis]) / len(velocities[axis])

                    # Apply exponential low-pass filter for smoothing
                    self.calculated_velocity[axis] = (
                        self.velocity_filter_alpha * raw_velocity +
                        (1 - self.velocity_filter_alpha) * self.calculated_velocity[axis]
                    )

    def get_calculated_velocity(self) -> dict[str, float]:
        """Get the calculated velocity from position differentiation (x, y, and z)"""
        with self.velocity_calculation_lock:
            return self.calculated_velocity.copy()

    # TODO - check for unsuccessful arming attempts

    def initialise_crazyflie(self) -> bool:
        """
        This method is meant to be overridden by any child classes.
        Return True on success, False on failure.
        """
        pass

    def _run(self, display: bool = False) -> None:
        """Main drone control loop"""
        if not self.initialise_crazyflie():
            self.set_running(False)
            print(f"[{self.agent_id}] Hardware Initialisation failed...")
            self.hardware_ready_event.set()  # Even though the hardware did not initialise set to prevent deadlock
            return

        try:
            # Main command processing loop
            while self.is_running():
                try:
                    if self.emergency_event.is_set() and not self.simulation:
                        print(f"[{self.agent_id} - Drone Setup] Emergency event detected in _run thread. Exiting main loop.")
                        self._handle_command({"land": True})
                        self.is_landed_event.wait(timeout=10)
                        break
                    # Get command with timeout
                    command = self.command_queue.get(timeout=0.1)
                    if command == "exit":
                        self.set_running(False)
                        print(f"[{self.agent_id}] Shutting down.")
                        break
                    elif command == "emergency_stop":
                        if self.position_controller_active:
                            self.stop_position_control()
                        if self.is_flying_event.is_set() and self.mc:
                            print(f"[{self.agent_id}] EMERGENCY STOP: Landing and Stopping.")
                            self.land_and_stop()
                        self.set_running(False)
                        print(f"[{self.agent_id}] EMERGENCY STOP initiated.")
                        break
                    else:
                        self._handle_command(command)

                    self.command_queue.task_done()

                except queue.Empty:
                    pass  # No command received; continue

                # Display current position periodically
                if display:
                    if hasattr(self, "_last_position_print"):
                        if time.time() - self._last_position_print > 0.2:
                            print(f"[{self.agent_id}] Current position: {self.position}")
                            self._last_position_print = time.time()
                    else:
                        self._last_position_print = time.time()

                time.sleep(0.05)  # Shorter sleep for more responsive command processing

        finally:
            self._shutdown_crazyflie()

    # minorly modified from cearlier drone
    def _shutdown_crazyflie(self) -> None:
        """Properly shutdown Crazyflie connection"""
        try:
            print(f"[{self.agent_id}] Shutting down Crazyflie...")

            if self.is_flying_event.is_set() and self.mc:
                print(f"[{self.agent_id}] Landing before shutdown...")
                self.mc.land()
                time.sleep(2)
                self.is_flying_event.clear()

            if self.mc:
                self.mc = None

            if self.armed and self.cf:
                print(f"[{self.agent_id}] Disarming Crazyflie...")
                # self.cf.platform.send_arming_request(False)
                self.cf.supervisor.send_arming_request(False)
                self.armed = False

            #Added try catch for safety
            if self.battery_log_config:
                try:
                    self.battery_log_config.stop()
                    self.battery_log_config.delete()
                except:
                    pass
                self.battery_log_config = None

            if self.velocity_log_config:
                try:
                    self.velocity_log_config.stop()
                    self.velocity_log_config.delete()
                except:
                    pass
                self.velocity_log_config = None

            if self.scf:
                self.scf.close_link()
                self.scf = None

            self.set_running(False)
            print(f"[{self.agent_id}] Shutdown complete")

        except Exception as e:
            print(f"[{self.agent_id}] Error during shutdown: {str(e)}")

    def _handle_command(self, command: Any) -> None:
        """Handle different types of commands"""

        # Handle string commands first
        if not isinstance(command, dict):
            print(f"[{self.agent_id}] String command received: {command}")
            return

        # From here on, we know command is a dictionary
        try:
            if "velocity" in command:
                with self.velocity_lock:
                    self.velocity = command["velocity"]
                # print(f"[{self.agent_id}] Velocity set to: {self.velocity}")

            elif "position" in command:
                # This is a position command
                with self.position_lock:
                    x = command["position"].get("x", self.position["x"])
                    y = command["position"].get("y", self.position["y"])
                    z = command["position"].get("z", self.position["z"])
                    self.target_position = {"x": x, "y": y, "z": z}
                    print(f"[{self.agent_id}] Target position set: x={x}, y={y}, z={z}")

            if "take_off" in command:
                if not self.is_flying_event.is_set() and self.armed:
                    print(f"[{self.agent_id}] Taking off...")
                    self.mc = MotionCommander(self.scf, default_height=self.default_height)
                    self.mc.take_off()
                    self.is_landed_event.clear()
                    self.is_flying_event.set()
                    # Check if the radio link is still alive and connected
                    if not self.mc._cf.is_connected():
                        print(f"[{self.agent_id}] Error: The Crazyflie instance is disconnected or invalid!")
                    print(f"[{self.agent_id}] Take-off successful")
                else:
                    print(f"[{self.agent_id}] Cannot take off - already flying or not armed")

            elif "land" in command:
                if self.is_flying_event.is_set() and self.mc:
                    print(f"[{self.agent_id}] Landing...")
                    self.mc.land()
                    self.is_landed_event.set()
                    self.is_flying_event.clear()
                    print(f"[{self.agent_id}] Landing successful")
                else:
                    print(f"[{self.agent_id}] Cannot land - not currently flying")

            elif "move" in command:
                if self.is_flying_event.is_set() and self.mc:
                    self.mc.start_linear_motion(0, 0, 0)

            elif "velocity_vector" in command:
                if self.is_flying_event.is_set() and self.mc:
                    vel_vector = command["velocity_vector"]
                    vx = vel_vector.get("x", 0.0)
                    vy = vel_vector.get("y", 0.0)
                    vz = vel_vector.get("z", 0.0)

                    # Apply velocity limits for safety
                    vx = max(-self.max_velocity["x"], min(self.max_velocity["x"], vx))
                    vy = max(-self.max_velocity["y"], min(self.max_velocity["y"], vy))
                    vz = max(-self.max_velocity["z"], min(self.max_velocity["z"], vz))

                    if self.velocity_controller_active:
                        # Set target velocity for velocity controller
                        with self.velocity_lock:
                            self.target_velocity = {"x": vx, "y": vy, "z": vz}
                        # print(f"[{self.agent_id}] Target velocity set for controller: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
                    else:
                        # print(f"start linear motion: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
                        # Send direct velocity command
                        self.mc.start_linear_motion(vx, vy, vz)
                        # print(f"[{self.agent_id}] Direct velocity vector set: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")
            else:
                print(f"[{self.agent_id}] Unknown command: {command}")

        except Exception as e:
            print(f"[{self.agent_id}] Error handling command: {str(e)}")

    def take_off(self) -> None:
        """Public method to take off"""
        self.send_command({"take_off": True})
        print(f"[{self.agent_id}] Take off command sent")

    def land(self) -> None:
        """Public method to land"""
        self.send_command({"land": True})

    def move(self) -> None:
        self.send_command({"move": True})


    def start_boundary_monitoring(self) -> None:
        """Start the boundary monitoring thread for safety checks"""
        print(f"[{self.agent_id} - DroneSetup] Entering start boundary function")
        # Wait for position system to stabilize
        if not self.position_ready_event.wait(timeout=10):
            print(f"[{self.agent_id}] WARNING: Position system may not be ready")
        else:
            print(f"[{self.agent_id}] Position system ready, starting safety monitoring...")
       
        if not self.safety_thread_active:
            self.safety_thread_active = True
            self.safety_thread = threading.Thread(
                target=self._check_boundaries
            )
            self.safety_thread.start()
            print(f"[{self.agent_id}] Boundary monitoring started")
        else:
            print(f"[{self.agent_id}] Boundary monitoring already active")

    def stop_boundary_monitoring(self) -> None:
        """Stop the boundary monitoring thread for safety checks"""
        print(f"[{self.agent_id} - DroneSetup] Entering stop boundary function")
        if self.safety_thread_active:
            self.safety_thread_active = False
            if self.safety_thread and self.safety_thread.is_alive():
                self.safety_thread.join(timeout=1)
            print(f"[{self.agent_id}] Boundary monitoring stopped")
        else:
            print(f"[{self.agent_id}] Boundary monitoring already stopped")


    def start_position_control(self) -> None:
        """Start the position controller thread for automatic position tracking"""
        if not self.position_controller_active:
            self.position_controller_active = True
            self.controller_thread = threading.Thread(
                target=self._position_control_loop
            )
            self.controller_thread.start()
            print(f"[{self.agent_id}] Position controller started")
        else:
            print(f"[{self.agent_id}] Position controller already active")

    def stop_position_control(self) -> None:
        """Stop the position controller thread"""
        if self.position_controller_active:
            self.position_controller_active = False
            if self.controller_thread and self.controller_thread.is_alive():
                self.controller_thread.join()
            print(f"[{self.agent_id}] Position controller stopped")
        else:
            print(f"[{self.agent_id}] Position controller already stopped")

    def start_velocity_control(self) -> None:
        """Start the velocity controller thread for automatic velocity tracking"""
        if not self.velocity_controller_active:
            self.velocity_controller_active = True
            self.velocity_controller_thread = threading.Thread(
                target=self._velocity_control_loop
            )
            self.velocity_controller_thread.start()
            print(f"[{self.agent_id}] Velocity controller started")
        else:
            print(f"[{self.agent_id}] Velocity controller already active")

    def stop_velocity_control(self) -> None:
        """Stop the velocity controller thread"""
        if self.velocity_controller_active:
            self.velocity_controller_active = False
            if self.velocity_controller_thread and self.velocity_controller_thread.is_alive():
                self.velocity_controller_thread.join()
            print(f"[{self.agent_id}] Velocity controller stopped")
        else:
            print(f"[{self.agent_id}] Velocity controller already stopped")

    def _position_control_loop(self, first_instance: int = 0, debugging: bool = True) -> None:
        """
        Main control loop for position-based velocity control
        
        Used for moving drone to reset positions
        """
        # BUG Is control rate meant to be 0.5 (2Hz) or 0.05 (20Hz)??
        control_rate = 0.5  # Control rate in seconds (20hz)
        error_threshold = 0.05  # Error threshold to consider position reached (meters)
                                # Changed from 0.17 to 0.05 along with position_deadband        

        print(f"[{self.agent_id}] Position control loop started")
        while (
            self.is_running()
            and self.position_controller_active
            and not self.emergency_event.is_set()
        ):
            try:
                # Get current and target positions
                current_pos = self.get_position_dict()
                with self.position_lock:
                    target_pos = self.target_position.copy()

                # if first_instance == 0:
                    # print(f"target position = {self.target_position.copy()}")
                    # print(f"current position = {self.get_position_dict()}")

                # Calculate position error
                error = {
                    "x": target_pos["x"] - current_pos["x"],
                    "y": target_pos["y"] - current_pos["y"],
                    "z": target_pos["z"] - current_pos["z"],
                }

                if debugging:
                    # print(f"[{self.agent_id}] Position({current_pos['x']}, {current_pos['y']}, {current_pos['z']})")
                    pass

                # Calculate error magnitude to determine if position is reached
                error_magnitude = (
                    abs(error["x"]) ** 2 + abs(error["y"]) ** 2 + abs(error["z"] ** 2)
                ) ** 0.5

                # print(error_magnitude)
                # print(self.get_battery())

                if error_magnitude < error_threshold:
                    # print( f"[{self.agent_id}: Controller] Position reached! Error: {error_magnitude:.2f}m")
                    self.at_reset_position.set()
                # Calculate velocity command using PID control
                velocity = self._calculate_pid_velocity(error, control_rate)

                # Apply velocity command if flying
                if self.is_flying_event.is_set() and self.mc:
                    # Change velocity to be handled by the main thread
                    self.set_velocity_vector(
                        velocity["x"], velocity["y"], velocity["z"]
                    )

                    # if debugging:
                    #     print(
                    #         f"[Controller] Pos error: ({error['x']:.2f}, {error['y']:.2f}, {error['z']:.2f}) → "
                    #         + f"Vel: ({velocity['x']:.2f}, {velocity['y']:.2f}, {velocity['z']:.2f})"
                    #     )

                # Sleep to maintain control rate
                time.sleep(control_rate)

            except Exception as e:
                print(f"[{self.agent_id}] Error in position control loop: {str(e)}")
                time.sleep(0.5)  # Sleep longer on error

        print(f"[{self.agent_id}] Position control loop stopped")

    def _velocity_control_loop(self) -> None:
        """Main control loop for velocity tracking using outer PID control with gradual ramping"""
        print(f"[{self.agent_id}] Velocity control loop started")

        while (
            self.is_running()
            and self.velocity_controller_active
            and not self.emergency_event.is_set()
        ):
            try:
                # FIXME This currently uses target_velocity which is set by the position controller and is open loop velocity control
                # The closed loop velocity control found from _calculate_velocity_pid is ignored as well as the ramped velocity function
                # What is the intended behaviour and why, how is performance changed
                # If _calculate_velocity_pid is used then the drones control properties need to be cleared after position control for reset

                # Get target and actual velocities
                with self.velocity_lock:
                    target_vel = self.target_velocity.copy()

                actual_vel = self.get_calculated_velocity()

                # Calculate corrected velocity command using PID
                desired_velocity = self._calculate_velocity_pid(target_vel, actual_vel, self.velocity_control_rate)

                # Apply gradual ramping to the velocity command
                ramped_velocity = self.target_velocity.copy()

                # print(f"Target: ({target_vel['x']:.3f}, {target_vel['y']:.3f}), "
                #     f"Actual: ({actual_vel['x']:.3f}, {actual_vel['y']:.3f}), "
                #     f"Desired: ({desired_velocity['x']:.3f}, {desired_velocity['y']:.3f}), "
                #     f"Ramped: ({ramped_velocity['x']:.3f}, {ramped_velocity['y']:.3f})")

                # Apply ramped velocity command if flying
                if self.is_flying_event.is_set() and self.mc:
                    # Send ramped velocity to MotionCommander
                    self.mc.start_linear_motion(
                        ramped_velocity["x"],
                        ramped_velocity["y"],
                        ramped_velocity["z"],
                    )

                # Sleep to maintain control rate
                time.sleep(self.velocity_control_rate)

            except Exception as e:
                print(f"[{self.agent_id}] Error in velocity control loop: {str(e)}")
                time.sleep(0.5)  # Sleep longer on error

        print(f"[{self.agent_id}] Velocity control loop stopped")

    def _apply_velocity_ramping(self, desired_velocity: dict[str, float], dt: float) -> dict[str, float]:

        ramped_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Calculate maximum allowed velocity change in this time step
        max_delta = self.max_velocity_change_rate * dt

        with self.velocity_command_lock:
            # Calculate the desired change vector (x, y and z)
            delta = {
                "x": desired_velocity["x"] - self.current_commanded_velocity["x"],
                "y": desired_velocity["y"] - self.current_commanded_velocity["y"],
                "z": desired_velocity["z"] - self.current_commanded_velocity["z"],
            }

            # Calculate the magnitude of the change vector (x, y, and z)
            delta_magnitude = (delta["x"]**2 + delta["y"]**2 + delta["z"]**2)**0.5

            # If the desired change is larger than allowed, scale it down while preserving direction
            if delta_magnitude > max_delta:
                scale = max_delta / delta_magnitude
                delta["x"] *= scale
                delta["y"] *= scale
                delta["z"] *= scale

            # Apply the limited change for x, y, and z
            ramped_velocity["x"] = self.current_commanded_velocity["x"] + delta["x"]
            ramped_velocity["y"] = self.current_commanded_velocity["y"] + delta["y"]
            ramped_velocity["z"] = self.current_commanded_velocity["z"] + delta["z"]

            # Update the current commanded velocity for next iteration
            self.current_commanded_velocity["x"] = ramped_velocity["x"]
            self.current_commanded_velocity["y"] = ramped_velocity["y"]
            self.current_commanded_velocity["z"] = ramped_velocity["z"]

        return ramped_velocity

    def clear_reset_position_event(self) -> None:
        self.at_reset_position.clear()

    def clear_emergency_event(self) -> None:
        self.emergency_event.clear()

    def _calculate_pid_velocity(self, error: dict[str, float], dt: float) -> dict[str, float]:
        """Calculate velocity vector using PID control

        Args:
            error: Position error in each axis as dictionary
        """
        velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        # For each axis (x, y, z)
        for axis in ["x", "y", "z"]:
            # Apply deadband to reduce jitter when close to target
            if abs(error[axis]) < self.position_deadband:
                velocity[axis] = 0.0
                continue

            # Proportional term
            p_term = self.gains[axis]["kp"] * error[axis]
            # Derivative term (rate of change of error)
            d_term = self.gains[axis]["kd"] * (error[axis] - self.last_error[axis]) / dt
            # Integral term (accumulating error)
            self.integral[axis] += error[axis] * dt
            # Anti-windup: reset integral when changing direction
            if (error[axis] * self.last_error[axis]) < 0:
                self.integral[axis] = 0.0
            # FIXME There is no limit applied to prevent integral windup 
            # Apply integral term with limits to prevent windup
            i_term = self.gains[axis]["ki"] * self.integral[axis]

            # Calculate raw velocity command (sum of PID terms)
            raw_velocity = p_term + d_term + i_term
            # Apply velocity limits
            velocity[axis] = max(
                -self.max_velocity[axis], min(self.max_velocity[axis], raw_velocity)
            )
            # Update last error for next iteration
            self.last_error[axis] = error[axis]

        return velocity

    def _calculate_velocity_pid(
            self, 
            target_velocity: dict[str, float], 
            actual_velocity: dict[str, float], 
            dt: float
            ) -> dict[str, float]:
        corrected_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        for axis in ["x", "y", "z"]:
            error = target_velocity[axis] - actual_velocity[axis]

            # PID terms
            p_term = self.velocity_gains[axis]["kp"] * error
            d_term = self.velocity_gains[axis]["kd"] * (error - self.velocity_last_error[axis]) / dt

            self.velocity_integral[axis] += error * dt
            max_integral = 0.2
            self.velocity_integral[axis] = max(-max_integral, min(max_integral, self.velocity_integral[axis]))
            i_term = self.velocity_gains[axis]["ki"] * self.velocity_integral[axis]

            # Feed-forward term (NEW)
            ff_term = target_velocity[axis]  # Direct pass-through of target

            # Combine all terms
            raw_velocity = ff_term + p_term + d_term + i_term
            corrected_velocity[axis] = max(-self.max_velocity[axis], min(self.max_velocity[axis], raw_velocity))

            self.velocity_last_error[axis] = error

        return corrected_velocity

    def set_velocity_vector(self, vx: float, vy: float, vz: float) -> None:
        if self.velocity_controller_active:
            # Set target velocity for velocity controller
            with self.velocity_lock:
                self.target_velocity = {"x": vx, "y": vy, "z": vz}
        else:
            # Direct velocity command to MotionCommander
            velocity_command = {"velocity_vector": {"x": vx, "y": vy, "z": vz}}
            self.send_command(velocity_command)

    def set_target_position(self, x: float, y: float, z: float) -> None:
        """Set target position with boundary checking"""
        # Check the target position is within boundaries
        if not (
            abs(x) <= self.boundaries["x"]
            and abs(y) <= self.boundaries["y"]
            # and self.boundaries["z_min"] <= z <= self.boundaries["z_max"]
            and z <= self.boundaries["z_max"]

        ):
            print(
                f"[{self.agent_id}] WARNING: Target position {x}, {y}, {z} is outside safe boundaries. Command rejected."
            )
            return

        # Create and send a position command
        position_command = {"position": {"x": x, "y": y, "z": z}}

        # Send the command to the queue
        self.send_command(position_command)
        # print(f"[{self.agent_id}] Target position command sent: x={x}, y={y}, z={z}")

    def get_position(self) -> list[float]:
        """Get current position as list"""
        with self.position_lock:
            return [self.position["x"], self.position["y"], self.position["z"]]

    def get_position_dict(self) -> dict[str, float]:
        """Get current position as dictionary"""
        with self.position_lock:
            return self.position.copy()

    def get_internal_velocity(self) -> tuple[float, float, float]:
        """Get current velocity from Crazyflie internal state estimate"""
        with self.velocity_log_lock:
            return (self.internal_vx, self.internal_vy, self.internal_vz)

    def send_command(self, command: Any) -> None:
        """Send command to the command queue"""
        self.command_queue.put(command)

    def clear_command_queue(self) -> None:
        """Clear all pending commands from the command queue."""
        while True:
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

    def _setup_battery_logging(self) -> None:
        # Check if Crazyflie object is available
        if self.cf is None:
            print(
                f"[{self.agent_id}] Could not start battery logging, Crazyflie object not available."
            )
            return

        # Check if log interface is available
        if not hasattr(self.cf, "log") or self.cf.log is None:
            print(
                f"[{self.agent_id}] Could not start battery logging, log interface not available."
            )
            return

        try:
            self.battery_log_config = LogConfig(name="Battery", period_in_ms=1000)
            self.battery_log_config.add_variable("pm.vbat", "float")

            self.cf.log.add_config(self.battery_log_config)
            # Register the callback function that will receive the data
            self.battery_log_config.data_received_cb.add_callback(
                self._battery_callback
            )
            # Start the logging
            self.battery_log_config.start()
            print(f"[{self.agent_id}] Battery logging started.")
        except KeyError as e:
            print(f"[{self.agent_id}] Could not start battery logging: {e}")
        except AttributeError:
            print(
                f"[{self.agent_id}] Could not start battery logging, Crazyflie object not available."
            )

    def _battery_callback(self, timestamp: int, data: dict[str, Any], logconf: Any) -> None:
        """Callback for when new battery data is received from the drone."""
        voltage = data["pm.vbat"]
        with self.battery_lock:
            self.battery_level = voltage

    def _setup_velocity_logging(self) -> None:
        if self.cf is None:
            print(
                f"[{self.agent_id}] Could not start velocity logging, Crazyflie object not available."
            )
            return

        if not hasattr(self.cf, "log") or self.cf.log is None:
            print(
                f"[{self.agent_id}] Could not start velocity logging, log interface not available."
            )
            return

        try:
            self.velocity_log_config = LogConfig(name="Velocity", period_in_ms=100)
            self.velocity_log_config.add_variable("stateEstimate.vx", "float")
            self.velocity_log_config.add_variable("stateEstimate.vy", "float")
            self.velocity_log_config.add_variable("stateEstimate.vz", "float")

            self.cf.log.add_config(self.velocity_log_config)
            self.velocity_log_config.data_received_cb.add_callback(
                self._velocity_callback
            )
            self.velocity_log_config.start()
            print(f"[{self.agent_id}] Velocity logging started.")
        except KeyError as e:
            print(f"[{self.agent_id}] Could not start velocity logging: {e}")
        except AttributeError:
            print(
                f"[{self.agent_id}] Could not start velocity logging, Crazyflie object not available."
            )

    def _velocity_callback(self, timestamp: int, data: dict[str, Any], logconf: Any) -> None:
        """Callback for when new velocity data is received from the drone."""
        with self.velocity_log_lock:
            self.internal_vx = data["stateEstimate.vx"]
            self.internal_vy = data["stateEstimate.vy"]
            self.internal_vz = data["stateEstimate.vz"]

    def get_motion_commander_setpoint(self) -> list[float]:
        """Get the current hover setpoint from MotionCommander's internal thread"""
        if self.mc and hasattr(self.mc, "_thread") and self.mc._thread:
            try:
                # Access the internal hover setpoint from the MotionCommander thread
                hover_setpoint = self.mc._thread._hover_setpoint.copy()
                # Return as [vx, vy, vz] (first 3 elements, ignoring yaw)
                return hover_setpoint[:3]
            except AttributeError as e:
                print(
                    f"[{self.agent_id}] Warning: Could not access MotionCommander setpoint: {e}"
                )
                return [0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    def land_and_stop(self) -> None:
        self.land()
        self.is_landed_event.wait(timeout=10)
        if not self.is_landed_event.is_set():
            print(f"[{self.agent_id}] Drone is failing to land....")
            print(f"[{self.agent_id}] Forcing stop")
        self.stop()

    def get_battery(self) -> float:
        with self.battery_lock:
            return self.battery_level

    def stop(self) -> None:
        """
        This method is meant to be overridden by any child classes.
        """
        pass

    def _signal_stop_to_all_threads(self) -> None:
        """Set all shutdown flags so every thread leaves its loop ASAP."""

        self.set_running(False)

        self.position_controller_active = False
        self.velocity_controller_active = False
        self.safety_thread_active = False

        self._stop_position_tracking()

        # Purge the command queue so no stale commands run after restart
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
        # Force threads waiting on Queue.get() to wake up
        self.command_queue.put("exit")

    def _join_all_threads(self) -> None:
        """Wait until every managed thread has exited."""
        threads_to_join = [
            ("position", self.position_thread),
            ("safety", self.safety_thread),
            ("controller", self.controller_thread),
            ("velocity_controller", self.velocity_controller_thread),
            ("main", self.thread),
        ]

        for name, thr in threads_to_join:
            if thr and thr.is_alive():
                if name == "main":
                    thr.join(timeout=5.0)
                else:
                    thr.join(timeout=2.0)
                if thr.is_alive():
                    print(f"[{self.agent_id}] WARNING: {name} thread did not join in time.")

    def _reset_shared_state(self) -> None:
        """Reset all state variables to their initial values."""
        # Position and target
        with self.position_lock:
            self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.target_position = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.position_history.clear()
            self.last_position_update_time = None
        # Velocity calculation
        with self.velocity_calculation_lock:
            self.calculated_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        # PID
        self.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        # Velocity PID
        self.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        # NEW: Reset ramped velocity
        with self.velocity_command_lock:
            self.current_commanded_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        # Events
        self.is_flying_event.clear()
        self.is_landed_event.set()
        self.deck_attached_event.clear()
        # Misc
        with self.velocity_lock:
            self.velocity = 0.0
        with self.battery_lock:
            self.battery_level = 5.0
        self.in_boundaries = True

    def _final_cleanup(self) -> None:
        """Delete big objects so garbage collection can reclaim them."""
        # These will be re-created if the user ever calls start() again
        # TODO :close link first then remove the scf instance
        self.cf = None
        self.scf = None
        self.mc = None

    def pre_battery_change_cleanup(self) -> None:

        if self.position_controller_active:
            self.stop_position_control()
        self.cf = None
        self.scf = None
        self.mc = None
        self.clear_command_queue()
        self._reset_shared_state()
