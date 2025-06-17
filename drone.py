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

        # Command processing
        self.command_queue = queue.Queue()
        self.velocity = 0.0
        self.velocity_lock = threading.Lock()
        self.running = True
        self.running_lock = threading.Lock()

        # Crazyflie objects - will be initialized in _run
        self.scf = None
        self.cf = None
        self.mc = None
        self.armed = False
        self.flying = False

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
                    x_in_bounds = self.boundaries["x"] >= abs(self.position["x"])
                    y_in_bounds = self.boundaries["y"] >= abs(self.position["y"])
                    z_in_bounds = self.boundaries["z"] >= abs(self.position["z"])

                # Set in_boundaries status
                current_status = x_in_bounds and y_in_bounds and z_in_bounds

                if current_status != self.in_boundaries:
                    self.in_boundaries = current_status
                    if self.in_boundaries:
                        print("Drone currently in bounds")
                    else:
                        print(f"[Drone] WARNING: Out of bounds!")
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
        time.sleep(3)
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

                time.sleep(0.090)  # 90 Hz
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
            if self.flying and self.mc:
                print("[Drone] Landing before shutdown...")
                self.mc.land()
                self.flying = False

            if self.mc:
                self.mc.stop()
                self.mc = None

            if self.armed and self.cf:
                print("[Drone] Disarming Crazyflie...")
                self.cf.platform.send_arming_request(False)
                self.armed = False

            if self.scf:
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

                    # Execute the movement if we have a motion commander
                    if self.mc and self.flying:
                        rel_x = x - self.position["x"]
                        rel_y = y - self.position["y"]
                        rel_z = z - self.position["z"]
                        self.mc.move_distance(rel_x, rel_y, rel_z)
                        print(f"[Drone] Moving to position: x={x}, y={y}, z={z}")

            elif "take_off" in command:
                if not self.flying and self.armed:
                    print("[Drone] Executing take-off command")
                    self.mc = MotionCommander(self.scf, default_height=self.default_height)
                    self.mc.take_off()
                    self.flying = True
                    print("[Drone] Take-off successful")
                else:
                    print("[Drone] Cannot take off - already flying or not armed")

            elif "land" in command:
                if self.flying and self.mc:
                    print("[Drone] Landing drone")
                    self.mc.land()
                    self.mc.stop()
                    self.mc = None
                    self.flying = False
                    print("[Drone] Landing successful")
                else:
                    print("[Drone] Cannot land - not currently flying")
            elif "move" in command:
                if self.flying and self.mc:
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

    def stop(self):
        """Stop the drone and all threads"""
        self.send_command("exit")
        # Ensure running is set to False so all threads can terminate
        self.set_running(False)
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
    # Wait for initialization
    time.sleep(10)
    drone.take_off()

    for i in range(30):
        drone.move()
        time.sleep(1)

    # time.sleep(10)
    # drone.take_off()
    # time.sleep(10)

    # drone.stop()
