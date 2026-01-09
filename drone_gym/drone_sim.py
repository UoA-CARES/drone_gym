import queue
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from drone_gym.drone_setup import DroneSetup


class DroneSim(DroneSetup):
    def __init__(self, uri="udp://0.0.0.0:19850", simulation=True):
        super().__init__(uri=uri)
        # Drone Properties
        self.simulation = simulation

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