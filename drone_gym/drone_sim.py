import queue
import time
from typing import Any

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from drone_gym.drone_setup import DroneSetup
from drone_gym.sim_manager import SimManager, get_default_sim_manager
import warnings

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
    """
    def __init__(
            self,
            uri: str = "udp://0.0.0.0:19850",
            agent_id: str = "Drone",
            simulation: bool = True,
            sim_manager: SimManager | None = None,
            boundaries: dict[str, float] | None = None,
        ) -> None:
        # Drone Properties
        self.simulation = simulation
        self.agent_id = agent_id
        self.sim_manager = sim_manager or get_default_sim_manager()

        super().__init__(uri=uri, agent_id=agent_id, simulation=simulation, boundaries=boundaries)

    def set_visual_target_marker_position(
        self,
        x: float,
        y: float,
        z: float,
        marker_name: str | None = None,
    ) -> None:
        """
        Backwards-compatible wrapper.

        If marker_name is provided, pass it to SimManager.
        Otherwise, call SimManager without marker_name so it uses its own default.
        """
        if marker_name is not None:
            self.sim_manager.set_visual_target_marker_position(
                x=x,
                y=y,
                z=z,
                marker_name=marker_name,
            )
        else:
            self.sim_manager.set_visual_target_marker_position(
                x=x,
                y=y,
                z=z,
            )

    def set_visual_boundary_lines(self, drone_xy_limit: float, z_level: float) -> None:
        """
        Backwards-compatible wrapper
        """
        self.sim_manager.set_visual_boundary_lines(
            xy_limit=drone_xy_limit,
            z_level=z_level,
        )

    def _update_position(self) -> None:
        """Update position from Gazebo via state estimate logs"""
        print(f"[{self.agent_id}] Position tracking thread started")
        
        # Wait for CF to be ready
        timeout = time.time() + 10
        while self.cf is None and time.time() < timeout:
            time.sleep(0.1)
        
        if self.cf is None:
            print(f"[{self.agent_id}] ERROR: Crazyflie not initialized")
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
            print(f"[{self.agent_id}] Position logging started")

            # Wait for first position
            time.sleep(2)
            self.position_ready_event.set()
            self.last_velocity_calculation_time = time.time()

            # Keep thread alive for logging
            # while self.is_running() and not self.emergency_event.is_set():
            while self.is_running():
                current_time = time.time()
                
                # Calculate velocity periodically
                if (current_time - self.last_velocity_calculation_time) >= self.velocity_update_rate:
                    if len(self.position_history) >= 2:
                        self._calculate_velocity()
                    self.last_velocity_calculation_time = current_time
                
                time.sleep(0.05)

        except Exception as e:
            print(f"[{self.agent_id}] Error in position thread: {str(e)}")
        finally:
            if 'position_log_config' in locals():
                try:
                    position_log_config.stop()
                    position_log_config.delete()
                except:
                    pass
        print(f'[{self.agent_id} - Drone Sim] Position tracking thread ending')

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
            self.cf.platform.send_arming_request(True)
            time.sleep(1.5)
            self.armed = True
            print(f"[{self.agent_id}] Crazyflie armed.")

            self._setup_battery_logging()
            self._setup_velocity_logging()
            self.cf.disconnected.add_callback(self._connection_lost)
            self.cf.connection_lost.add_callback(self._connection_lost)
    
            # Signal that hardware is ready
            self.hardware_ready_event.set()
            print(f"[{self.agent_id}] Hardware initialisation complete - ready to fly!")
            return True

        except Exception as e:
            print(f"[{self.agent_id}] Failed to initialize Crazyflie: {str(e)}")
            return False

    def _position_callback(self, timestamp: int, data: dict[str, Any], logconf: Any) -> None:
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

    def _connection_lost(self, link_uri):
        print(f"\n[{self.agent_id}] CRITICAL: Connection no longer works!")
        print(f"[{self.agent_id}] Reason: Simulation crashed or link was severed.")
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