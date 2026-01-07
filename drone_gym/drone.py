import threading
import queue
import time
from threading import Event
from collections import deque
from drone_gym.drone_setup import DroneSetup
from drone_gym.utils.vicon_connection_class import ViconInterface as vi

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.power_switch import PowerSwitch


class Drone(DroneSetup):
    def __init__(self):
        super().__init__()
        # Drone Properties
        self.URI = uri_helper.uri_from_env(
            default="radio://0/100/2M/E7E7E7E7E7"
        )  # changed radio channel in 22/9

        self.ps = PowerSwitch(
            "radio://0/100/2M/E7E7E7E7E7"
        )  # changed radio channel in 22/9

        # Vicon Integration
        self.drone_name = "Crzayme"
        self.vicon = vi()

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
            self._setup_velocity_logging()

            # Signal that hardware is ready
            self.hardware_ready_event.set()
            print("[Drone] Hardware initialisation complete - signaling ready")
            return True

        except Exception as e:
            print(f"[Drone] Failed to initialize Crazyflie: {str(e)}")
            return False

    def _param_deck_flow(self, _, value_str):
        """Callback for deck detection"""
        value = int(value_str)
        if value:
            self.deck_attached_event.set()
            print("Deck is attached!")
        else:
            print("Deck is NOT attached")

    def set_velocity(self, velocity_vector) -> None:
        """Set velocity vector from a list or array [vx, vy, vz]"""

        if len(velocity_vector) != 3:
            raise ValueError(
                "Velocity vector must have exactly 3 elements [vx, vy, vz]"
            )

        self.set_velocity_vector(
            velocity_vector[0], velocity_vector[1], velocity_vector[2]
        )

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

    def _close_vicon(self):
        """Tell the Vicon interface to stop its background thread."""
        try:
            self.vicon.run_interface = False
            # Give Vicon a moment to shut down its socket
            time.sleep(0.2)
        except Exception as e:
            print(f"[Drone] Error while closing Vicon: {e}")

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
