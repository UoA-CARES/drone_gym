import threading
import time
from drone_gym.drone_setup import DroneSetup
from drone_gym.utils.vicon_connection_class import ViconInterface as vi
from drone_gym.utils.position_source import PositionSource
from drone_gym.utils.vicon_position_source import (
    ViconPositionSource,
)

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.utils.power_switch import PowerSwitch


class Drone(DroneSetup):
    """
    Drone class for Crazyflie 

    Args:
        agent_id (str): Unique identifier for the drone instance.
        boundaries (dict[str, float] | None): Optional dictionary defining the safe operational boundaries for the drone. If not provided, default boundaries will be used. (e.g. {"x": 2.5, "y": 2.5, "z_min": 0.1, "z_max": 3.0})
        uri (str | None): The URI for the Crazyflie drone.
        position_source (PositionSource): The source for obtaining the drone's position.
    """
    def __init__(
            self, 
            position_source: PositionSource,
            agent_id: str = "Drone",
            boundaries: dict[str, float] | None = None,
            uri: str = "radio://0/100/2M/E7E7E7E700",
        ) -> None:
        # Use either legacy_vicon or source_vicon
        # source_vicon is the new implementation that abstracts position source 
        self.position_tracking_mode: str = "legacy_vicon"

        if self.position_tracking_mode == "legacy_vicon":
            position_source = None
            self.vicon = vi()
        elif self.position_tracking_mode == "source_vicon":
            self.vicon = None
        else:
            raise ValueError(
                f"Invalid position_tracking_mode: {self.position_tracking_mode}. "
                "Must be either 'legacy_vicon' or 'source_vicon'."
            )

        # Vicon Integration
        self.drone_name = f"Crzayme_{agent_id}"

        super().__init__(
            boundaries=boundaries, 
            agent_id=agent_id, 
            uri=uri, 
            position_source=position_source
        )

        # Drone Properties
        self.ps = PowerSwitch(self.URI)

        self.agent_id = agent_id

    def _update_position(self) -> None:
        vicon_thread = None
        try:
            while (not hasattr(self, "vicon")):
                print("Waiting for vicon initialization...")
            # It takes some time for the vicon to get values
            vicon_thread = threading.Thread(target=self.vicon.main_loop)
            vicon_thread.start()
            print(f"[{self.agent_id}] Vicon thread started, waiting for position data...")
            time.sleep(4)

            # Wait for first valid position reading before signaling ready
            position_ready = False
            ready_timeout = time.monotonic() + 6  # 6 second timeout for first position
            self.last_velocity_calculation_time = time.monotonic()

            while self.is_running() and not self.emergency_event.is_set():
                try:
                    position_array = self.vicon.getPos(self.drone_name)

                    if position_array is None:
                        print(
                            f"[{self.agent_id}] Waiting for Vicon position "
                            f"for {self.drone_name!r}..."
                        )
                        time.sleep(self.position_update_rate)
                        continue

                    if position_array is not None:
                        current_time = time.monotonic()

                        with self.position_lock:
                            self.position = {
                                "x": position_array[0],
                                "y": position_array[1],
                                "z": position_array[2],
                            }
                            current_pos = self.position.copy()

                        # Store position with timestamp for velocity calculation
                        self.position_history.append((current_time, current_pos))
                        self.last_position_update_time = current_time

                        # Calculate velocity at 20Hz (every 0.05s)
                        if (current_time - self.last_velocity_calculation_time) >= self.velocity_update_rate:
                            if len(self.position_history) >= 2:
                                self._calculate_velocity()
                            self.last_velocity_calculation_time = current_time

                        # Signal ready on first successful position read
                        if not position_ready:
                            self.position_ready_event.set()
                            position_ready = True
                            print(f"[{self.agent_id}] First position acquired: {self.position}")

                    else:
                        print(f"[{self.agent_id}] Drone position is not being updated")
                        # If timeout reached without position, signal anyway to prevent deadlock
                        if not position_ready and time.monotonic() > ready_timeout:
                            print(
                                f"[{self.agent_id}] WARNING: Position timeout - signaling ready anyway"
                            )
                            self.position_ready_event.set()
                            position_ready = True

                    time.sleep(self.position_update_rate)  # 60 Hz
                except Exception as e:
                    print(
                        f"[{self.agent_id}] Error: Position data could not be parsed correctly - {str(e)}"
                    )

        except Exception as e:
            print(f"[{self.agent_id}] Critical error in position thread:", e)
        finally:
            # Signal the vicon thread to join
            self.vicon.run_interface = False
            if vicon_thread is not None:
                vicon_thread.join()

    def initialise_crazyflie(self) -> bool:
        """Initialise Crazyflie connection and setup"""
        try:
            cflib.crtp.init_drivers()
            print(f"[{self.agent_id}] Connecting to Crazyflie...")

            self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache="./cache"))
            self.scf.open_link()
            self.cf = self.scf.cf

            # Setup deck detection
            self.cf.param.add_update_callback(
                group="deck", name="bcFlow2", cb=self._param_deck_flow
            )
            time.sleep(1)

            if not self.deck_attached_event.wait(timeout=5):
                print(f"[{self.agent_id}] No flow deck is detected! Exiting....")
                self.stop()
                return False

            print(f"[{self.agent_id}] Resetting all log configurations")
            self.cf.log.reset()
            time.sleep(0.5)

            print(f"[{self.agent_id}] Resetting state estimation (EKF)...")
            self.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.1)

            # Arm the drone
            print(f"[{self.agent_id}] Arming Crazyflie...")
            self.cf.platform.send_arming_request(True)
            time.sleep(1.0)
            self.armed = True
            print(f"[{self.agent_id}] Crazyflie armed.")

            self._setup_battery_logging()
            self._setup_velocity_logging()

            # Signal that hardware is ready
            self.hardware_ready_event.set()
            print(f"[{self.agent_id}] Hardware initialisation complete - signaling ready")
            return True

        except Exception as e:
            print(f"[{self.agent_id}] Failed to initialize Crazyflie: {str(e)}")
            return False

    def _param_deck_flow(self, _, value_str) -> None:
        """Callback for deck detection"""
        value = int(value_str)
        if value:
            self.deck_attached_event.set()
            print(f"[{self.agent_id}] Deck is attached!")
        else:
            print(f"[{self.agent_id}] Deck is NOT attached")

    def set_velocity(self, velocity_vector: list[float] | tuple[float, float, float]) -> None:
        """Set velocity vector from a list or array [vx, vy, vz]"""

        if len(velocity_vector) != 3:
            raise ValueError(
                "Velocity vector must have exactly 3 elements [vx, vy, vz]"
            )

        self.set_velocity_vector(
            velocity_vector[0], velocity_vector[1], velocity_vector[2]
        )

    def stop(self) -> None:
        """
        Fully stop the drone and optionally prepare for a clean restart.

        Args

        """
        print(f"[{self.agent_id}] In the new stop function")
        self._signal_stop_to_all_threads()
        self._close_vicon()
        self._join_all_threads()
        self._reset_shared_state()
        self._final_cleanup()

    def _close_vicon(self) -> None:
        """Tell the Vicon interface to stop its background thread."""
        if self.vicon is None:
            print(f"[{self.agent_id}] Exiting close_vicon() - vicon object is None, no action needed")
            return
        if self.position_tracking_mode != "legacy_vicon":
            print(f"[{self.agent_id}] Exiting close_vicon() - not in legacy_vicon mode, no action needed")
            return
        try:
            self.vicon.run_interface = False
            # Give Vicon a moment to shut down its socket
            time.sleep(0.2)
        except Exception as e:
            print(f"[{self.agent_id}] Error while closing Vicon: {e}")

    def reboot(self) -> bool:
        print(f"[{self.agent_id}] Initiating remote reboot sequence...")
        # Step 3: Perform the power cycle
        print(f"[{self.agent_id}] Executing STM32 power cycle via PowerSwitch...")
        try:
            # Re-initialize the PowerSwitch as the old connection may be stale
            ps = PowerSwitch(self.URI)
            ps.stm_power_cycle()
            print(f"[{self.agent_id}] Power cycle complete. Waiting for reboot...")
            time.sleep(5)  # Give the Crazyflie time to reboot and restart
        except Exception as e:
            print(f"[{self.agent_id}] ERROR during power cycle: {e}")
            return False  # Return failure status

        return True

if __name__ == "__main__":
    # Testing instructions
    drone = Drone()
    print("Drone class initiated")
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if not drone.is_flying_event.is_set():
        print(f"[{drone.agent_id}] Drone failed to take off")
        drone.stop()

    drone.start_position_control()
    time.sleep(2)  # Let the controller stabilise first
    print(f"[{drone.agent_id}] Setting target position")
    drone.set_target_position(0, 0, 1)  # Move 1m forward on x-axis
    time.sleep(30)
    drone.stop_position_control()
    drone.land()
    drone.is_landed_event.wait(timeout=30)
    if not drone.is_landed_event.is_set():
        print(f"[{drone.agent_id}] Drone is failing to land....")
        print(f"[{drone.agent_id}] Forcing stop")
    drone.stop()
