import threading
import queue
import time
from threading import Event
from vicon_connection_class import ViconInterface as vi

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper


class Drone:

    def __init__(self):
        # Drone Properties
        self.URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
        self.default_height = 0.5
        self.deck_attached_event = Event()

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
        # PID gains - separate for each axis for better tuning
        self.gains = {
            "x": {"kp": 0.5, "kd": 0, "ki": 0},
            "y": {"kp": 0.5, "kd": 0, "ki": 0},
            "z": {"kp": 0.5, "kd": 0, "ki": 0}
        }
        self.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.max_velocity = 0.5  # Maximum velocity in m/s
        self.position_deadband = 0.15  # Position error below which velocity will be zero (in meters)

        # Crazyflie objects - will be initialized in _run
        self.scf = None
        self.cf = None
        self.mc = None
        self.armed = False

        # Vicon Integration
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone_name = "AtlasCrazyflie"
        self.vicon = vi()
        self.position_thread = threading.Thread(target=self._update_position)
        self.position_thread.start()
        self.position_lock = threading.Lock()

        # Drone Safety
        self.boundaries = {"x": 3, "y": 3, "z": 3}
        self.safety_thread = threading.Thread(target=self._check_boundaries)
        self.safety_thread.start()
        self.in_boundaries = True

        # Objective related
        self.target_position = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Start the main drone thread
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _check_boundaries(self):
        time.sleep(5)
        while self.is_running():
            try:
                # Check if position is within boundaries for each axis
                with self.position_lock:
                    current_pos = self.position.copy()
                    x_in_bounds = self.boundaries["x"] >= abs(current_pos["x"])
                    y_in_bounds = self.boundaries["y"] >= abs(current_pos["y"])
                    z_in_bounds = self.boundaries["z"] >= abs(current_pos["z"])

                # Set in_boundaries status
                current_status = x_in_bounds and y_in_bounds and z_in_bounds

                if current_status != self.in_boundaries:
                    self.in_boundaries = current_status
                    if self.in_boundaries:
                        print("Drone currently in bounds")
                    else:
                        print("[Drone] WARNING: Out of bounds!")
                        self.send_command("emergency_stop")
                        print("Emergency stop command sent")
                time.sleep(0.01)

            except Exception as e:
                print(f"[Drone] Error in boundary checking: {str(e)}")
                time.sleep(0.1)

    def is_running(self):
        """Check if the drone is running"""
        with self.running_lock:
            return self.running

    def set_running(self, value):
        """Set the running state"""
        with self.running_lock:
            self.running = value

    def _update_position(self):
        # It takes some time for the vicon to get values
        time.sleep(1)
        vicon_thread = threading.Thread(target=self.vicon.main_loop)
        vicon_thread.start()
        while self.is_running():
            try:
                position_array = self.vicon.getPos(self.drone_name)
                if position_array is not None:
                    with self.position_lock:
                        self.position = {
                            "x": position_array[0],
                            "y": position_array[1],
                            "z": position_array[2]
                        }
                else:
                    print("Drone position is not being updated")

                time.sleep(0.0166666)  # 60 Hz
            except Exception as e:
                print(f"[Drone] Error: Position data could not be parsed correctly - {str(e)}")
        # Signal the vicon thread to join
        self.vicon.run_interface = False
        vicon_thread.join()

    def _initialize_crazyflie(self):
        """Initialize Crazyflie connection and setup"""
        try:
            cflib.crtp.init_drivers()
            print("[Drone] Connecting to Crazyflie...")

            self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache='./cache'))
            self.scf.open_link()
            self.cf = self.scf.cf

            # Setup deck detection
            self.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=self._param_deck_flow)
            time.sleep(1)

            if not self.deck_attached_event.wait(timeout=5):
                print("No flow deck is detected! Exiting....")
                return False

            # Arm the drone
            print("[Crazyflie] Arming Crazyflie...")
            self.cf.platform.send_arming_request(True)
            time.sleep(1.0)
            self.armed = True
            print("[Crazyflie] Crazyflie armed.")

            return True

        except Exception as e:
            print(f"[Drone] Failed to initialize Crazyflie: {str(e)}")
            return False

    def _run(self):
        """Main drone control loop"""
        if not self._initialize_crazyflie():
            self.set_running(False)
            return

        try:
            # Main command processing loop
            while self.is_running():
                try:
                    # Get command with timeout
                    command = self.command_queue.get(timeout=0.1)

                    if command == "exit":
                        self.set_running(False)
                        print("[Drone] Shutting down.")
                        break
                    elif command == "emergency_stop":
                        self.set_running(False)
                        print("[Drone] EMERGENCY STOP initiated. Shutting down.")
                        break
                    else:
                        self._handle_command(command)

                    self.command_queue.task_done()

                except queue.Empty:
                    pass  # No command received; continue

                # Display current position periodically
                if hasattr(self, '_last_position_print'):
                    if time.time() - self._last_position_print > 1.5:
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
                print("scf")
                self.scf.close_link()
                self.scf = None

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
                    print("[Drone] Executing take-off command")
                    self.mc = MotionCommander(self.scf, default_height=self.default_height)
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
                    # self.mc.stop()
                    # self.mc = None
                    self.is_landed_event.set()
                    self.is_flying_event.clear()
                    print("[Drone] Landing successful")
                else:
                    print("[Drone] Cannot land - not currently flying")
            elif "move" in command:
                if self.is_flying_event.is_set() and self.mc:
                    self.mc.start_linear_motion(0,-0.2,0)
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
        self.send_command({"move" : True})

    def start_position_control(self):
        """Start the position controller thread for automatic position tracking"""
        if not self.controller_active:
            self.controller_active = True
            self.controller_thread = threading.Thread(target=self._position_control_loop)
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

    def _position_control_loop(self):
        """Main control loop for position-based velocity control"""
        control_rate = 0.05  # Control rate in seconds (50Hz)
        error_threshold = 0.1  # Error threshold to consider position reached (meters)
        position_reached = False

        print("[Drone] Position control loop started")
        while self.is_running() and self.controller_active:
            try:
                # Get current and target positions
                current_pos = self.get_position_dict()
                with self.position_lock:
                    target_pos = self.target_position.copy()

                # Calculate position error
                error = {
                    "x": target_pos["x"] - current_pos["x"],
                    "y": target_pos["y"] - current_pos["y"],
                    "z": target_pos["z"] - current_pos["z"]
                }
                print(f"[Controller]: Position({current_pos['x']}, {current_pos['y']}, {current_pos['z']})")
                # Calculate error magnitude to determine if position is reached
                error_magnitude = (error["x"]**2 + error["y"]**2 + error["z"]**2)**0.5

                if error_magnitude < error_threshold and not position_reached:
                    print(f"[Controller] Position reached! Error: {error_magnitude:.2f}m")
                    position_reached = True

                elif error_magnitude >= error_threshold * 1.5 and position_reached:
                    # Reset flag when error increases significantly (e.g., new target set)
                    position_reached = False

                # Calculate velocity command using PID control
                velocity = self._calculate_pid_velocity(error, control_rate)

                # Apply velocity command if flying
                if self.is_flying_event.is_set() and self.mc:
                    self.mc.start_linear_motion(
                        velocity["x"],
                        velocity["y"],
                        velocity["z"]
                    )
                    print(f"[Controller] Pos error: ({error['x']:.2f}, {error['y']:.2f}, {error['z']:.2f}) → "
                          f"Vel: ({velocity['x']:.2f}, {velocity['y']:.2f}, {velocity['z']:.2f})")

                # Sleep to maintain control rate
                time.sleep(control_rate)

            except Exception as e:
                print(f"[Drone] Error in position control loop: {str(e)}")
                time.sleep(0.5)  # Sleep longer on error

        print("[Drone] Position control loop stopped")

    def _calculate_pid_velocity(self, error, dt):
        """Calculate velocity vector using PID control

        Args:
            error: Position error in each axis as dictionary
            dt: Time step for derivative and integral calculations
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
            velocity[axis] = max(-self.max_velocity, min(self.max_velocity, raw_velocity))
            # Update last error for next iteration
            self.last_error[axis] = error[axis]

        return velocity

    def test_velocity(self, x, y, z):

        if self.is_flying_event.is_set() and self.mc:
            self.mc.start_linear_motion(x, y, z)

    def set_target_position(self, x: float, y: float, z: float) -> None:
        """Set target position with boundary checking"""
        # Check the target position is within boundaries
        if not (abs(x) <= self.boundaries["x"] and
                abs(y) <= self.boundaries["y"] and
                abs(z) <= self.boundaries["z"]):
            print(f"[Drone] WARNING: Target position {x}, {y}, {z} is outside safe boundaries. Command rejected.")
            return

        # Create and send a position command
        position_command = {
            "position": {
                "x": x,
                "y": y,
                "z": z
            }
        }

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

    def set_pid_gains(self, axis, kp=None, ki=None, kd=None):
        """Update PID gains for a specific axis"""
        axes = ["x", "y", "z"] if axis == "all" else [axis]

        for ax in axes:
            if ax not in self.gains:
                print(f"[Drone] Invalid axis '{ax}'. Use 'x', 'y', 'z', or 'all'")
                continue

            if kp is not None:
                self.gains[ax]["kp"] = kp
            if ki is not None:
                self.gains[ax]["ki"] = ki
            if kd is not None:
                self.gains[ax]["kd"] = kd

            print(f"[Drone] Updated PID gains for {ax}-axis: kp={self.gains[ax]['kp']}, "
                  f"ki={self.gains[ax]['ki']}, kd={self.gains[ax]['kd']}")

    def set_max_velocity(self, velocity):
        """Set the maximum velocity limit for the controller"""
        self.max_velocity = float(velocity)
        print(f"[Drone] Maximum velocity set to {self.max_velocity} m/s")

    def set_deadband(self, value):
        """Set position error deadband"""
        self.position_deadband = float(value)
        print(f"[Drone] Position deadband set to {self.position_deadband} meters")

    def stop(self):
        """Stop the drone and all threads"""
        # Stop position controller if active
        if self.controller_active:
            self.stop_position_control()

        self.set_running(False)

        self.send_command("exit")

        time.sleep(0.2)

        # Join the run thread
        if self.thread.is_alive():
            self.thread.join()
        # Join the vicon thread
        if self.position_thread.is_alive():
            self.position_thread.join()
        # Join the safety thread
        if self.safety_thread.is_alive():
            self.safety_thread.join()


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
    time.sleep(2) # Let the controller stabilise first
    print("Setting target position")
    drone.set_target_position(0, 1.0, 0.5)  # Move 1m forward on x-axis
    # Let the position controller run for 15 seconds
    time.sleep(15)
    print("post 35 seconds pause")
    drone.set_target_position(0.0, 0.0, 0.5)  # Return to origin (x,y)
    time.sleep(15)
    drone.set_target_position(1.0, 1.0, 0.5)  # Move along y-axis and change height
    time.sleep(15)
    drone.stop_position_control()
    # # Land and stop
    drone.land()
    drone.is_landed_event.wait(timeout=30)
    if not drone.is_landed_event.is_set():
        print("Drone is failing to land....")
        print("Forcing stop")
    # time.sleep(5)
    drone.stop()
