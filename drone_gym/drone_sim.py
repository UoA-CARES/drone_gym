from multiprocessing import Event, Lock
import queue
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from drone_gym.drone_setup import DroneSetup
from drone_gym.sim_manager import SimManager
import warnings
from drone_gym.utils.crazyflie_log_position_source import (
    CrazyflieLogPositionSource,
)
from drone_gym.utils.position_source import PositionSource

warnings.filterwarnings('ignore', message='Using legacy TYPE_HOVER_LEGACY')

class DroneSim(DroneSetup):
    """
    Drone class for CrazySim (Gazebo simulation)

    Args:
        uri (str): The URI for the Crazyflie drone.
        agent_id (str): Unique identifier for the drone instance.
        simulation (bool): Flag indicating whether the drone is in simulation mode.
        sim_manager (SimManager | None): Optional SimManager instance for managing simulation interactions. If not provided, a default SimManager will be used.
        boundaries (dict[str, float] | None): Optional dictionary defining the safe operational boundaries for the drone. If not provided, default boundaries will be used. (e.g. {"x": 2.5, "y": 2.5, "z_min": 0.1, "z_max": 3.0})
        position_source (PositionSource | None): Optional PositionSource instance for providing position data. If not provided, a default CrazyflieLogPositionSource will be used.
    """
    def __init__(
            self,
            uri: str = "udp://0.0.0.0:19850",
            agent_id: str = "Drone",
            simulation: bool = True,
            sim_manager: SimManager | None = None,
            boundaries: dict[str, float] | None = None,
            position_source: PositionSource | None = None,
        ) -> None:
        # Drone Properties
        self.simulation = simulation
        self.agent_id = agent_id
        self.sim_manager = sim_manager
        if position_source is None:
            position_source = CrazyflieLogPositionSource(
                crazyflie_getter=lambda: self.cf,
                period_ms=50,
                label=self.agent_id,
            )

        self.fatal_error_event = Event()
        self.fatal_error_lock = Lock()
    
        super().__init__(
            uri=uri, 
            agent_id=agent_id, 
            simulation=simulation, 
            boundaries=boundaries, 
            position_source=position_source
        )

    def initialise_crazyflie(self) -> bool:
        """Initialise Crazyflie connection for CrazySim"""
        try:
            cflib.crtp.init_drivers()
            print(f"[{self.agent_id}] Initializing CRTP drivers...")
            
            # Add retries for connection
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    print(f"[{self.agent_id}] Connection attempt {attempt + 1}/{max_retries} to {self.URI}...")
                    self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache=f"./cache/{self.agent_id}/"))
                    self.scf.open_link()
                    self.cf = self.scf.cf
                    print(f"[{self.agent_id}] Successfully connected to CrazySim!")
                    break
                except Exception as e:
                    print(f"[{self.agent_id}] Connection attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"[{self.agent_id}] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"[{self.agent_id}] All connection attempts failed")
                        print(f"[{self.agent_id}] Make sure:")
                        print(f"  1. Gazebo is running")
                        print(f"  2. The SITL firmware is started")
                        print(f"  3. The drone model is spawned in Gazebo")
                        raise

            # For CrazySim, we can skip deck detection or set it immediately I believe
            print(f"[{self.agent_id}] Setting deck attached (simulated)")
            self.deck_attached_event.set()

            print(f"[{self.agent_id}] Waiting for firmware to be ready...")
            time.sleep(2)

            print(f"[{self.agent_id}] Resetting all log configurations")
            self.cf.log.reset()
            time.sleep(0.5)

            print(f"[{self.agent_id}] Resetting state estimation (EKF)...")
            self.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.5)

            # Arm the drone
            print(f"[{self.agent_id}] Arming Crazyflie...")
            # self.cf.platform.send_arming_request(True)
            self.cf.supervisor.send_arming_request(True)
            time.sleep(1.5)
            self.armed = True
            print(f"[{self.agent_id}] Crazyflie armed.")

            self._setup_battery_logging()
            self._setup_velocity_logging()
            # self.cf.disconnected.add_callback(self._disconnected)
            self.cf.connection_lost.add_callback(self._connection_lost)
    
            # Signal that hardware is ready
            self.hardware_ready_event.set()
            print(f"[{self.agent_id}] Hardware initialisation complete - ready to fly!")
            return True

        except Exception as e:
            print(f"[{self.agent_id}] Failed to initialize Crazyflie: {str(e)}")
            return False

    def stop(self) -> None:
        """
        Fully stop the drone and optionally prepare for a clean restart.

        Args

        """
        print(f"[{self.agent_id}] Stopping...")
        self.set_running(False)
        self.position_controller_active = False
        self.velocity_controller_active = False
        
        # Clear queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break

        print(f"[{self.agent_id}] Stopped")
        self._signal_stop_to_all_threads()
        self._join_all_threads()
        self._reset_shared_state()
        self._final_cleanup()

    def _execute_emergency_stop(self):
        """
        Simulation-specific emergency stop / hard-boundary response.

        This overrides DroneSetup._execute_emergency_stop().

        For physical drones, the inherited behaviour lands, disarms, exits the
        command thread, and calls stop(). That is safe for real hardware but 
        causes issues in simulation.
        """
        current_position = self.get_position()
        if not self.emergency_event.is_set():
            self.emergency_event.set()
            print(
            f"[{self.agent_id}] SIM HARD BOUNDARY VIOLATION. "
            f"Position={current_position}, limits={self.boundaries}. "
            )

        if self.mc:
            print(f"[{self.agent_id}] Initiating simulated emergency landing...")
            self.mc.land()
            self.is_landed_event.set()
            self.is_flying_event.clear()
            time.sleep(1)
        print(f"[{self.agent_id} - DRONE SIM] Drone landed due to boundary violation. Current position: {self.get_position()}")

        # Cancel any currently commanded motion.
        try:
            self.set_velocity_vector(0.0, 0.0, 0.0)
        except Exception as exc:
            print(f"[{self.agent_id}] Failed to send zero velocity after boundary violation: {exc}")

        # Disable high-level controllers so they do not keep pushing the drone.
        self.position_controller_active = False
        self.velocity_controller_active = False

    # def _disconnected(self, link_uri):
    #     print(f"\n[{self.agent_id}] CRITICAL: Drone ({link_uri}) disconnected!")

    def _connection_lost(self, link_uri: str, message: str) -> None:
        print(f"\n[{self.agent_id}] CRITICAL: Connection no longer works!")
        print(f"[{self.agent_id}] Link: {link_uri}")
        print(f"[{self.agent_id}] Reason: {message}")

        with self.fatal_error_lock:
            self.fatal_error_event.set()
        self.emergency_event.set()

if __name__ == "__main__":
    drone = DroneSim()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if drone.is_flying_event.is_set():
        print(f"[{drone.agent_id}] Hovering for 5 seconds...")
        time.sleep(5)

        print(f"[{drone.agent_id}] Moving forward...")
        drone.set_velocity_vector(0.2, 0, 0)
        time.sleep(3)
        
        drone.set_velocity_vector(0, 0, 0)
        time.sleep(1)

    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()