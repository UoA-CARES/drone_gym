import threading
import queue
import time
from threading import Event
from collections import deque
from drone_gym.utils.vicon_connection_class import ViconInterface as vi

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.power_switch import PowerSwitch


class Drone:
    def __init__(self):
        # Drone Properties
        self.URI = uri_helper.uri_from_env(
            default="radio://0/100/2M/E7E7E7E7E7"
        )  # changed radio channel in 22/9
        self.default_height = 0.5
        self.deck_attached_event = Event()
        self.battery_lock = threading.Lock()
        self.battery_log_config = None
        self.battery_level = 5.0  # default value
        self.ps = PowerSwitch(
            "radio://0/100/2M/E7E7E7E7E7"
        )  # changed radio channel in 22/9

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
        self.controller_active = False
        self.controller_thread = None
        self.at_reset_position = Event()

        # Velocity controller parameters
        self.velocity_controller_active = False
        self.velocity_controller_thread = None
        self.velocity_control_rate = 0.1  # 10Hz velocity control rate

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
            "x": {"kp": 1.30, "kd": 0.25, "ki": 0},
            "y": {"kp": 1.35, "kd": 0.25, "ki": 0},
            "z": {"kp": 0, "kd": 0, "ki": 0},
        }
        self.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.max_velocity = 0.25  # Maximum velocity in m/s
        self.position_deadband = (
            0.10  # Position error below which velocity will be zero (in meters)
        )

        # NEW: Velocity ramping for smooth transitions
        self.current_commanded_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.max_velocity_change_rate = 0.075 #Maximum change in velocity per second (m/s²)
        self.velocity_command_lock = threading.Lock()

        # Crazyflie objects - will be initialized in _run
        self.scf = None
        self.cf = None
        self.mc = None
        self.armed = False

        # Vicon Integration
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone_name = "Crzayme"
        self.vicon = vi()
        self.position_thread = None
        self.position_lock = threading.Lock()

        # Velocity calculation (from position differentiation)
        self.calculated_velocity = {"x": 0.0, "y": 0.0}
        self.velocity_calculation_lock = threading.Lock()
        self.position_history = deque(maxlen=15)  # Store last 15 positions for moving average
        self.velocity_update_rate = 0.10  # 20Hz velocity calculation rate
        self.position_update_rate = 0.0166666  # 60Hz position update rate
        self.last_velocity_calculation_time = 0.0

        # Drone Safety
        self.boundaries = {"x": 2.5, "y": 2.5, "z": 2.25}
        self.safety_thread = None
        self.in_boundaries = True
        self.emergency_event = Event()

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
        print("[Drone] Starting threads with")

        # Start main thread first (initialises hardware)
        self.thread.start()
        print("[Drone] Main thread started, waiting for hardware initialization...")

        # Wait for hardware to be ready
        if not self.hardware_ready_event.wait(timeout=15):
            print("[Drone] ERROR: Hardware failed to initialize in time")
            self.set_running(False)
            return

        print("[Drone] Hardware ready, starting position thread...")

        # Start position tracking thread
        self.position_thread = threading.Thread(target=self._update_position)
        self.position_thread.start()

        # Wait for position system to stabilize
        if not self.position_ready_event.wait(timeout=10):
            print("[Drone] WARNING: Position system may not be ready")
        else:
            print("[Drone] Position system ready, starting safety monitoring...")

        # Start safety monitoring last
        self.safety_thread = threading.Thread(target=self._check_boundaries)
        self.safety_thread.start()

        print("[Drone] All threads started successfully with coordination")

    def _check_boundaries(self):
        """Monitor drone boundaries with safety checks for valid position data"""
        print("[Drone] Boundary monitoring thread started")

        # Wait for position system to be ready before checking boundaries
        if not self.position_ready_event.wait(timeout=15):
            print(
                "[Drone] WARNING: Starting boundary check without confirmed position data"
            )

        # Additional startup delay to let system stabilize
        time.sleep(2)
        print("[Drone] Boundary checking now active")

        while self.is_running():
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
                    z_in_bounds = self.boundaries["z"] >= abs(current_pos["z"])

                # Set in_boundaries status
                in_bounds = x_in_bounds and y_in_bounds and z_in_bounds
                # Start threads in coordinated sequence

                if not in_bounds:
                    self.in_boundaries = False
                    print(
                        f"[Drone] BOUNDARY VIOLATION: Position {current_pos} exceeds limits {self.boundaries}"
                    )
                    self._execute_emergency_stop()
                    break
                else:
                    self.in_boundaries = True

                time.sleep(0.01)
            except Exception as e:
                print(f"[Drone] Error in boundary checking: {str(e)}")
                time.sleep(0.1)

        print("[Drone] Boundary monitoring thread stopped")

    def _execute_emergency_stop(self):
        if not self.emergency_event.is_set():
            self.emergency_event.set()
            print("Emergency stop event triggered!!!")

        if self.mc:
            self.mc.land()
            time.sleep(1)

        self.controller_active = False

        if self.armed and self.cf:
            self.cf.platform.send_arming_request(False)
            self.armed = False

        self.command_queue.put("exit")
        self.stop()

    def is_running(self):
        """Check if the drone is running"""
        with self.running_lock:
            return self.running

    def set_running(self, value):
        """Set the running state"""
        with self.running_lock:
            self.running = value

    def _update_position(self):
        vicon_thread = None
        try:
            # It takes some time for the vicon to get values
            vicon_thread = threading.Thread(target=self.vicon.main_loop)
            vicon_thread.start()
            print("[Drone] Vicon thread started, waiting for position data...")
            time.sleep(4)

            # Wait for first valid position reading before signaling ready
            position_ready = False
            ready_timeout = time.time() + 6  # 6 second timeout for first position
            self.last_velocity_calculation_time = time.time()

            while self.is_running() and not self.emergency_event.is_set():
                try:
                    position_array = self.vicon.getPos(self.drone_name)
                    if position_array is not None:
                        current_time = time.time()

                        with self.position_lock:
                            self.position = {
                                "x": position_array[0],
                                "y": position_array[1],
                                "z": position_array[2],
                            }
                            current_pos = self.position.copy()

                        # Store position with timestamp for velocity calculation
                        self.position_history.append((current_time, current_pos))

                        # Calculate velocity at 20Hz (every 0.05s)
                        if (current_time - self.last_velocity_calculation_time) >= self.velocity_update_rate:
                            if len(self.position_history) >= 2:
                                self._calculate_velocity()
                            self.last_velocity_calculation_time = current_time

                        # Signal ready on first successful position read
                        if not position_ready:
                            self.position_ready_event.set()
                            position_ready = True
                            print(f"[Drone] First position acquired: {self.position}")

                    else:
                        print("Drone position is not being updated")
                        # If timeout reached without position, signal anyway to prevent deadlock
                        if not position_ready and time.time() > ready_timeout:
                            print(
                                "[Drone] WARNING: Position timeout - signaling ready anyway"
                            )
                            self.position_ready_event.set()
                            position_ready = True

                    time.sleep(self.position_update_rate)  # 60 Hz
                except Exception as e:
                    print(
                        f"[Drone] Error: Position data could not be parsed correctly - {str(e)}"
                    )

        except Exception as e:
            print(f"[Drone] Critical error in position thread: {str(e)}")
        finally:
            # Signal the vicon thread to join
            self.vicon.run_interface = False
            if vicon_thread is not None:
                vicon_thread.join()

    def _calculate_velocity(self):
        """Calculate velocity using moving average filter over position history (x and y only)"""
        if len(self.position_history) < 2:
            return

        # Calculate velocities for each time step in history
        velocities = {"x": [], "y": []}

        for i in range(1, len(self.position_history)):
            t_prev, pos_prev = self.position_history[i - 1]
            t_curr, pos_curr = self.position_history[i]

            dt = t_curr - t_prev

            if dt > 0:
                # Calculate velocity for x and y axes only
                for axis in ["x", "y"]:
                    vel = (pos_curr[axis] - pos_prev[axis]) / dt
                    velocities[axis].append(vel)

        # Apply moving average filter
        with self.velocity_calculation_lock:
            for axis in ["x", "y"]:
                if len(velocities[axis]) > 0:
                    self.calculated_velocity[axis] = sum(velocities[axis]) / len(velocities[axis])

    def get_calculated_velocity(self):
        """Get the calculated velocity from position differentiation (x and y only)"""
        with self.velocity_calculation_lock:
            return self.calculated_velocity.copy()

    # TODO - check for unsuccessful arming attempts

    def initialise_crazyflie(self):
        """Initialise Crazyflie connection and setup"""
        try:
            cflib.crtp.init_drivers()
            print("[Drone] Connecting to Crazyflie...")

            self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache="./cache"))
            self.scf.open_link()
            self.cf = self.scf.cf

            # Setup deck detection
            self.cf.param.add_update_callback(
                group="deck", name="bcFlow2", cb=self._param_deck_flow
            )
            time.sleep(1)

            if not self.deck_attached_event.wait(timeout=5):
                print("No flow deck is detected! Exiting....")
                self.stop()
                return False

            print("[Drone] Resetting all log configurations")
            self.cf.log.reset()
            time.sleep(0.5)

            print("[Drone] Resetting state estimation (EKF)...")
            self.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.1)

            # Arm the drone
            print("[Crazyflie] Arming Crazyflie...")
            self.cf.platform.send_arming_request(True)
            time.sleep(1.0)
            self.armed = True
            print("[Crazyflie] Crazyflie armed.")

            self._setup_battery_logging()

            # Signal that hardware is ready
            self.hardware_ready_event.set()
            print("[Drone] Hardware initialisation complete - signaling ready")
            return True

        except Exception as e:
            print(f"[Drone] Failed to initialize Crazyflie: {str(e)}")
            return False

    def _run(self, display=False):
        """Main drone control loop"""
        if not self.initialise_crazyflie():
            self.set_running(False)
            print("[Drone] Hardware Initialisation failed...")
            self.hardware_ready_event.set()  # Even though the hardware did not initialise set to prevent deadlock
            return

        try:
            # Main command processing loop
            while self.is_running():
                try:
                    if self.emergency_event.is_set():
                        self._handle_command({"land": True})
                        self.is_landed_event.wait(timeout=10)
                        break
                    # Get command with timeout
                    command = self.command_queue.get(timeout=0.1)
                    if command == "exit":
                        self.set_running(False)
                        print("[Drone] Shutting down.")
                        break
                    elif command == "emergency_stop":
                        if self.controller_active:
                            self.stop_position_control()
                        if self.is_flying_event.is_set() and self.mc:
                            print("EMERGENCY STOP: Landing and Stopping.")
                            self.land_and_stop()
                        self.set_running(False)
                        print("[Drone] EMERGENCY STOP initiated. Shutting down.")
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
                            print(f"[Drone] Current position: {self.position}")
                            self._last_position_print = time.time()
                    else:
                        self._last_position_print = time.time()

                time.sleep(0.05)  # Shorter sleep for more responsive command processing

        finally:
            self._shutdown_crazyflie()

    def _shutdown_crazyflie(self):
        """Properly shutdown Crazyflie connection"""
        try:
            print("In shutdown crazyflie")
            if self.is_flying_event.is_set() and self.mc:
                print("[Drone] Landing before shutdown...")
                self.is_flying_event.clear()

            if self.mc:
                print("mc")
                # self.mc.stop()
                self.mc = None

            if self.armed and self.cf:
                print("[Drone] Disarming Crazyflie...")
                self.cf.platform.send_arming_request(False)
                self.armed = False

            if self.scf:
                self.scf.close_link()
                self.scf = None
                print("scf object cleaned")

            if self.battery_log_config:
                self.battery_log_config.stop()
                self.battery_log_config.delete()
                self.battery_log_config = None

            self.set_running(False)

        except Exception as e:
            print(f"[Drone] Error during shutdown: {str(e)}")

    def _handle_command(self, command):
        """Handle different types of commands"""
        # Handle string commands first
        if not isinstance(command, dict):
            print(f"[Drone] String command received: {command}")
            return

        # From here on, we know command is a dictionary
        try:
            if "velocity" in command:
                with self.velocity_lock:
                    self.velocity = command["velocity"]
                print(f"[Drone] Velocity set to: {self.velocity}")

            elif "position" in command:
                # This is a position command
                with self.position_lock:
                    x = command["position"].get("x", self.position["x"])
                    y = command["position"].get("y", self.position["y"])
                    z = command["position"].get("z", self.position["z"])
                    self.target_position = {"x": x, "y": y, "z": z}
                    print(f"[Drone] Target position set: x={x}, y={y}, z={z}")

            elif "take_off" in command:
                if not self.is_flying_event.is_set() and self.armed:
                    print("[Drone] Processing the take off command")
                    self.mc = MotionCommander(
                        self.scf, default_height=self.default_height
                    )
                    self.mc.take_off()
                    self.is_landed_event.clear()
                    self.is_flying_event.set()
                    print("[Drone] Take-off successful")
                else:
                    print("[Drone] Cannot take off - already flying or not armed")

            elif "land" in command:
                if self.is_flying_event.is_set() and self.mc:
                    print("[Drone] Landing drone")
                    self.mc.land()
                    self.is_landed_event.set()
                    self.is_flying_event.clear()
                    print("[Drone] Landing successful")
                else:
                    print("[Drone] Cannot land - not currently flying")
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
                    max_vel = getattr(self, "max_velocity", 0.5)
                    vx = max(-max_vel, min(max_vel, vx))
                    vy = max(-max_vel, min(max_vel, vy))
                    vz = max(-max_vel, min(max_vel, vz))

                    if self.velocity_controller_active:
                        # Set target velocity for velocity controller
                        with self.velocity_lock:
                            self.target_velocity = {"x": vx, "y": vy, "z": vz}
                        print(
                            f"[Drone] Target velocity set for controller: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}"
                        )
                    else:
                        # Send direct velocity command
                        self.mc.start_linear_motion(vx, vy, vz)
                        print(
                            f"[Drone] Direct velocity vector set: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}"
                        )
                else:
                    print(
                        "[Drone] Cannot set velocity - drone not flying or motion commander not available"
                    )

            else:
                print(f"[Drone] Unknown command: {command}")

        except Exception as e:
            print(f"[Drone] Error handling command {command}: {str(e)}")

    def _param_deck_flow(self, _, value_str):
        """Callback for deck detection"""
        value = int(value_str)
        if value:
            self.deck_attached_event.set()
            print("Deck is attached!")
        else:
            print("Deck is NOT attached")

    def take_off(self):
        """Public method to take off"""
        self.send_command({"take_off": True})
        print("Take off command sent")

    def land(self):
        """Public method to land"""
        self.send_command({"land": True})

    def move(self):
        self.send_command({"move": True})

    def start_position_control(self):
        """Start the position controller thread for automatic position tracking"""
        if not self.controller_active:
            self.controller_active = True
            self.controller_thread = threading.Thread(
                target=self._position_control_loop
            )
            self.controller_thread.start()
            print("[Drone] Position controller started")
        else:
            print("[Drone] Position controller already active")

    def stop_position_control(self):
        """Stop the position controller thread"""
        if self.controller_active:
            self.controller_active = False
            if self.controller_thread and self.controller_thread.is_alive():
                self.controller_thread.join()
            print("[Drone] Position controller stopped")
        else:
            print("[Drone] Position controller already stopped")

    def start_velocity_control(self):
        """Start the velocity controller thread for automatic velocity tracking"""
        if not self.velocity_controller_active:
            self.velocity_controller_active = True
            self.velocity_controller_thread = threading.Thread(
                target=self._velocity_control_loop
            )
            self.velocity_controller_thread.start()
            print("[Drone] Velocity controller started")
        else:
            print("[Drone] Velocity controller already active")

    def stop_velocity_control(self):
        """Stop the velocity controller thread"""
        if self.velocity_controller_active:
            self.velocity_controller_active = False
            if self.velocity_controller_thread and self.velocity_controller_thread.is_alive():
                self.velocity_controller_thread.join()
            print("[Drone] Velocity controller stopped")
        else:
            print("[Drone] Velocity controller already stopped")

    def _position_control_loop(self, first_instance=0, debugging=True):
        """Main control loop for position-based velocity control"""
        control_rate = 0.5  # Control rate in seconds (20hz)
        error_threshold = 0.17  # Error threshold to consider position reached (meters)

        print("[Drone] Position control loop started")
        while (
            self.is_running()
            and self.controller_active
            and not self.emergency_event.is_set()
        ):
            try:
                # Get current and target positions
                current_pos = self.get_position_dict()
                with self.position_lock:
                    target_pos = self.target_position.copy()

                if first_instance == 0:
                    print(f"target position = {self.target_position.copy()}")
                    print(f"current position = {self.get_position_dict()}")

                # Calculate position error
                error = {
                    "x": target_pos["x"] - current_pos["x"],
                    "y": target_pos["y"] - current_pos["y"],
                    "z": target_pos["z"] - current_pos["z"],
                }

                if debugging:
                    print(
                        f"[Controller]: Position({current_pos['x']}, {current_pos['y']}, {current_pos['z']})"
                    )

                # Calculate error magnitude to determine if position is reached
                error_magnitude = (
                    abs(error["x"]) ** 2 + abs(error["y"]) ** 2 + abs(error["z"] ** 2)
                ) ** 0.5

                # print(error_magnitude)
                # print(self.get_battery())

                if error_magnitude < error_threshold:
                    print(
                        f"[Controller] Position reached! Error: {error_magnitude:.2f}m"
                    )
                    self.at_reset_position.set()
                # Calculate velocity command using PID control
                velocity = self._calculate_pid_velocity(error, control_rate)

                # Apply velocity command if flying
                if self.is_flying_event.is_set() and self.mc:
                    # Change velocity to be handled by the main thread
                    self.set_velocity_vector(
                        velocity["x"], velocity["y"], velocity["z"]
                    )

                    if debugging:
                        print(
                            f"[Controller] Pos error: ({error['x']:.2f}, {error['y']:.2f}, {error['z']:.2f}) → "
                            + f"Vel: ({velocity['x']:.2f}, {velocity['y']:.2f}, {velocity['z']:.2f})"
                        )

                # Sleep to maintain control rate
                time.sleep(control_rate)

            except Exception as e:
                print(f"[Drone] Error in position control loop: {str(e)}")
                time.sleep(0.5)  # Sleep longer on error

        print("[Drone] Position control loop stopped")

    def _velocity_control_loop(self):
        """Main control loop for velocity tracking using outer PID control with gradual ramping"""
        print("[Drone] Velocity control loop started")

        while (
            self.is_running()
            and self.velocity_controller_active
            and not self.emergency_event.is_set()
        ):
            try:
                # Get target and actual velocities
                with self.velocity_lock:
                    target_vel = self.target_velocity.copy()

                actual_vel = self.get_calculated_velocity()

                # Calculate corrected velocity command using PID
                desired_velocity = self._calculate_velocity_pid(target_vel, actual_vel, self.velocity_control_rate)

                # Apply gradual ramping to the velocity command
                ramped_velocity = self._apply_velocity_ramping(desired_velocity, self.velocity_control_rate)

                print(f"Target: ({target_vel['x']:.3f}, {target_vel['y']:.3f}), "
                    f"Actual: ({actual_vel['x']:.3f}, {actual_vel['y']:.3f}), "
                    f"Desired: ({desired_velocity['x']:.3f}, {desired_velocity['y']:.3f}), "
                    f"Ramped: ({ramped_velocity['x']:.3f}, {ramped_velocity['y']:.3f})")

                # Apply ramped velocity command if flying
                if self.is_flying_event.is_set() and self.mc:
                    # Send ramped velocity to MotionCommander
                    self.mc.start_linear_motion(
                        ramped_velocity["x"],
                        ramped_velocity["y"],
                        0
                    )

                # Sleep to maintain control rate
                time.sleep(self.velocity_control_rate)

            except Exception as e:
                print(f"[Drone] Error in velocity control loop: {str(e)}")
                time.sleep(0.5)  # Sleep longer on error

        print("[Drone] Velocity control loop stopped")

    def _apply_velocity_ramping(self, desired_velocity, dt):

        ramped_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Calculate maximum allowed velocity change in this time step
        max_delta = self.max_velocity_change_rate * dt

        with self.velocity_command_lock:
            for axis in ["x", "y", "z"]:
                # Calculate the desired change in velocity
                delta = desired_velocity[axis] - self.current_commanded_velocity[axis]

                # Limit the change to the maximum allowed rate
                if abs(delta) > max_delta:
                    delta = max_delta if delta > 0 else -max_delta

                # Apply the limited change
                ramped_velocity[axis] = self.current_commanded_velocity[axis] + delta

                # Update the current commanded velocity for next iteration
                self.current_commanded_velocity[axis] = ramped_velocity[axis]

        return ramped_velocity

    def clear_reset_position_event(self):
        self.at_reset_position.clear()

    def _calculate_pid_velocity(self, error, dt):
        """Calculate velocity vector using PID control

        Args:
            error: Position error in each axis as dictionary
        """
        velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        # For each axis (x, y, z)``
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
            # Apply integral term with limits to prevent windup
            i_term = self.gains[axis]["ki"] * self.integral[axis]

            # Calculate raw velocity command (sum of PID terms)
            raw_velocity = p_term + d_term + i_term
            # Apply velocity limits
            velocity[axis] = max(
                -self.max_velocity, min(self.max_velocity, raw_velocity)
            )
            # Update last error for next iteration
            self.last_error[axis] = error[axis]

        return velocity

    def _calculate_velocity_pid(self, target_velocity, actual_velocity, dt):
        """Calculate velocity corrections using PID control for velocity tracking (x, y only)

        Args:
            target_velocity: Target velocity in each axis as dictionary
            actual_velocity: Actual velocity in each axis as dictionary
            dt: Time step for derivative and integral calculations
        """
        corrected_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        # For x and y axes only (z is always 0)
        for axis in ["x", "y"]:
            # Calculate velocity error
            error = target_velocity[axis] - actual_velocity[axis]

            # Proportional term
            p_term = self.velocity_gains[axis]["kp"] * error

            # Derivative term (rate of change of error)
            d_term = self.velocity_gains[axis]["kd"] * (error - self.velocity_last_error[axis]) / dt

            # Integral term (accumulating error)
            self.velocity_integral[axis] += error * dt
            # Anti-windup: reset integral when changing direction
            if (error * self.velocity_last_error[axis]) < 0:
                self.velocity_integral[axis] = 0.0
            # Apply integral term with limits to prevent windup
            i_term = self.velocity_gains[axis]["ki"] * self.velocity_integral[axis]

            # Calculate corrected velocity command (sum of PID terms)
            raw_velocity = p_term + d_term + i_term
            # Apply velocity limits
            corrected_velocity[axis] = max(
                -self.max_velocity, min(self.max_velocity, raw_velocity)
            )
            # Update last error for next iteration
            self.velocity_last_error[axis] = error

        return corrected_velocity

    def set_velocity_vector(self, vx: float, vy: float, vz: float) -> None:
        if self.velocity_controller_active:
            # Set target velocity for velocity controller
            with self.velocity_lock:
                self.target_velocity = {"x": vx, "y": vy, "z": vz}
            print(f"[Drone] Target velocity set for controller: vx={vx}, vy={vy}, vz={vz}")
        else:
            # Direct velocity command to MotionCommander
            velocity_command = {"velocity_vector": {"x": vx, "y": vy, "z": vz}}
            self.send_command(velocity_command)
            print(f"[Drone] Direct velocity command sent: vx={vx}, vy={vy}, vz={vz}")

    def set_velocity(self, velocity_vector) -> None:
        """Set velocity vector from a list or array [vx, vy, vz]"""

        if len(velocity_vector) != 3:
            raise ValueError(
                "Velocity vector must have exactly 3 elements [vx, vy, vz]"
            )

        self.set_velocity_vector(
            velocity_vector[0], velocity_vector[1], velocity_vector[2]
        )

    def set_target_position(self, x: float, y: float, z: float) -> None:
        """Set target position with boundary checking"""
        # Check the target position is within boundaries
        if not (
            abs(x) <= self.boundaries["x"]
            and abs(y) <= self.boundaries["y"]
            and abs(z) <= self.boundaries["z"]
        ):
            print(
                f"[Drone] WARNING: Target position {x}, {y}, {z} is outside safe boundaries. Command rejected."
            )
            return

        # Create and send a position command
        position_command = {"position": {"x": x, "y": y, "z": z}}

        # Send the command to the queue
        self.send_command(position_command)
        print(f"[Drone] Target position command sent: x={x}, y={y}, z={z}")

    def get_position(self):
        """Get current position as list"""
        with self.position_lock:
            return [self.position["x"], self.position["y"], self.position["z"]]

    def get_position_dict(self):
        """Get current position as dictionary"""
        with self.position_lock:
            return self.position.copy()

    def send_command(self, command):
        """Send command to the command queue"""
        self.command_queue.put(command)

    def clear_command_queue(self):
        """Clear all pending commands from the command queue."""
        while True:
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

    def _setup_battery_logging(self):
        # Check if Crazyflie object is available
        if self.cf is None:
            print(
                "[Drone] Could not start battery logging, Crazyflie object not available."
            )
            return

        # Check if log interface is available
        if not hasattr(self.cf, "log") or self.cf.log is None:
            print(
                "[Drone] Could not start battery logging, log interface not available."
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
            print("[Drone] Battery logging started.")
        except KeyError as e:
            print(f"[Drone] Could not start battery logging: {e}")
        except AttributeError:
            print(
                "[Drone] Could not start battery logging, Crazyflie object not available."
            )

    def _battery_callback(self, timestamp, data, logconf):
        """Callback for when new battery data is received from the drone."""
        voltage = data["pm.vbat"]
        with self.battery_lock:
            self.battery_level = voltage

    def get_motion_commander_setpoint(self):
        """Get the current hover setpoint from MotionCommander's internal thread"""
        if self.mc and hasattr(self.mc, "_thread") and self.mc._thread:
            try:
                # Access the internal hover setpoint from the MotionCommander thread
                hover_setpoint = self.mc._thread._hover_setpoint.copy()
                # Return as [vx, vy, vz] (first 3 elements, ignoring yaw)
                return hover_setpoint[:3]
            except AttributeError as e:
                print(
                    f"[Drone] Warning: Could not access MotionCommander setpoint: {e}"
                )
                return [0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    def land_and_stop(self):
        self.land()
        self.is_landed_event.wait(timeout=10)
        if not self.is_landed_event.is_set():
            print("Drone is failing to land....")
            print("Forcing stop")
        self.stop()

    def get_battery(self):
        with self.battery_lock:
            return self.battery_level

    def stop(self):
        """
        Fully stop the drone and optionally prepare for a clean restart.

        Args

        """
        print("In the new stop function")
        self._signal_stop_to_all_threads()
        self._close_vicon()
        self._join_all_threads()
        self._reset_shared_state()
        self._final_cleanup()

    def _signal_stop_to_all_threads(self):
        """Set all shutdown flags so every thread leaves its loop ASAP."""
        self.set_running(False)
        self.controller_active = False
        self.velocity_controller_active = False
        # Purge the command queue so no stale commands run after restart
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
        # Force threads waiting on Queue.get() to wake up
        self.command_queue.put("exit")

    def _join_all_threads(self):
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
                    print(f"[Drone] WARNING: {name} thread did not join in time.")

    def _close_vicon(self):
        """Tell the Vicon interface to stop its background thread."""
        try:
            self.vicon.run_interface = False
            # Give Vicon a moment to shut down its socket
            time.sleep(0.2)
        except Exception as e:
            print(f"[Drone] Error while closing Vicon: {e}")

    def _reset_shared_state(self):
        """Reset all state variables to their initial values."""
        # Position and target
        with self.position_lock:
            self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.target_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        # Velocity calculation
        with self.velocity_calculation_lock:
            self.calculated_velocity = {"x": 0.0, "y": 0.0}
        self.position_history.clear()
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

    def _final_cleanup(self):
        """Delete big objects so garbage collection can reclaim them."""
        # These will be re-created if the user ever calls start() again
        # todo :close link first then remove the scf instance
        self.cf = None
        self.scf = None
        self.mc = None

    def pre_battery_change_cleanup(self):
        self.cf = None
        self.scf = None
        self.mc = None
        self.clear_command_queue()
        self._reset_shared_state()

    def reboot(self):
        print("[Drone] Initiating remote reboot sequence...")
        # Step 3: Perform the power cycle
        print("[Drone] Executing STM32 power cycle via PowerSwitch...")
        try:
            # Re-initialize the PowerSwitch as the old connection may be stale
            ps = PowerSwitch(self.URI)
            ps.stm_power_cycle()
            print("[Drone] Power cycle complete. Waiting for reboot...")
            time.sleep(5)  # Give the Crazyflie time to reboot and restart
        except Exception as e:
            print(f"[Drone] ERROR during power cycle: {e}")
            return False  # Return failure status

        return True


if __name__ == "__main__":
    # Testing instructions

    drone = Drone()
    print("Drone class initiated")
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if not drone.is_flying_event.is_set():
        print("Drone failed to take off")
        drone.stop()

    drone.start_position_control()
    time.sleep(2)  # Let the controller stabilise first
    print("Setting target position")
    drone.set_target_position(0, 0, 1)  # Move 1m forward on x-axis
    time.sleep(30)
    drone.stop_position_control()
    drone.land()
    drone.is_landed_event.wait(timeout=30)
    if not drone.is_landed_event.is_set():
        print("Drone is failing to land....")
        print("Forcing stop")
    drone.stop()
