from abc import ABC, abstractmethod
import os
import subprocess
from drone_gym.sim_manager import SimManager, SimLaunchConfig
from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
import time
import numpy as np
from typing import Dict, List, Any, Literal

# TODO - make naming more consistent


class DroneEnvironment(ABC):
    """Base drone environment that handles common drone operations"""

    def __init__(
            self, 
            use_simulator: Literal[0,1],
            max_velocity: float = 0.5,
            step_time: float = 0.5,
        ) -> None:
        self._closed = False  # Track if the environment has been closed
        # Set the appropriate drone instance based on use_simulator flag
        print("use_simulator", use_simulator)
        self.num_agents_config = 1  # Default to 1 agent; can be overridden by tasks

        if use_simulator:
            self.sim_manager = SimManager(
                sim_launch_config=SimLaunchConfig(
                    num_agents=self.num_agents_config
                )
            )

            time.sleep(1)  # Allow time for the sim manager to initialize
            print("Starting simulator...")
            sim_started = self.sim_manager.start_sim()
            print("start_sim returned:", sim_started)

            print("cwd:", os.getcwd())
            print("DISPLAY:", os.environ.get("DISPLAY"))
            print("XDG_RUNTIME_DIR:", os.environ.get("XDG_RUNTIME_DIR"))

            subprocess.run(
                'ps -eo pid,ppid,pgid,sid,stat,etime,cmd | '
                'grep -i "sitl_multiagent_square\\|gz sim\\|cf2" | grep -v grep',
                shell=True,
            )

            print("launch process alive:", self.sim_manager.is_sim_process_alive())
            print("sitl ready:", self.sim_manager._sitl_processes_ready())
            print("drones spawned:", self.sim_manager.are_drones_spawned(timeout=2.0))

            if not sim_started:
                raise RuntimeError("Failed to start CrazySim/Gazebo simulator.")
            # if not self.sim_manager.start_sim():
            #     raise RuntimeError("Failed to start CrazySim/Gazebo simulator.")
        else:
            self.sim_manager = None
            
        if use_simulator:
            print("Made DroneSim")
            self.drone = DroneSim(
                sim_manager=self.sim_manager,
            )
        else:
            print("Made Drone")
            self.drone = Drone()
            
        self.reset_position = [0, 0, 1]
        self.max_velocity = max_velocity
        self.max_velocity_z = 0.5
        self.step_time = step_time
        self.steps = 0
        self.seed = 0

        self.battery_threshold = 3.25
        self.observation_space = 12

        # Movement Boundary - can be overridden by tasks
        self.xy_limit = 1.0
        self.z_limit = 0.5

        # Reset target position optimization
        self._reset_target_set = False

        # Episode tracking for evaluation mode
        self._is_evaluating = False
        self.episode_positions = []
        self._log_path = None

        # Success tracking for learning phase
        self.success_count = 0

    def _update_visual_boundaries(self):
        """Push per-task boundary limits to Gazebo visual overlays when available."""
        if not hasattr(self.drone, "set_visual_boundary_lines"):
            return

        drone_xy = float(self.xy_limit)
        z_level = float(self.reset_position[2])

        if hasattr(self, "boundary") and isinstance(self.boundary, list) and len(self.boundary) >= 4:
            drone_xy = float(self.boundary[0])
            z_level = float(max(self.boundary[2], self.reset_position[2]))

        self.drone.set_visual_boundary_lines(
            drone_xy_limit=drone_xy,
            z_level=z_level,
        )

    def _reset_control_properties(self):
        self.drone.clear_command_queue()
        time.sleep(0.5)  # Allow any in-flight commands to be processed
        self.drone.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

    def reset(self, training: bool = True):
        """Reset the drone to initial position and state"""

        # Handle evaluation mode detection
        if not training and not self._is_evaluating:
            print("--- STARTING NEW EVALUATION BLOCK ---")
            self._is_evaluating = True
        elif training:
            self._is_evaluating = False

        # Clear episode position tracking
        self.episode_positions = []

        self._reset_control_properties()

        # Stop velocity controller to prevent conflict with position controller
        if self.drone.velocity_controller_active:
            print("[Reset] Stopping velocity controller for reset")
            self.drone.stop_velocity_control()

        # Stop the current velocity
        self.drone.set_velocity_vector(0, 0, 0)
        time.sleep(0.5)

        self.steps = 0
        # Check that the drone is not already flying
        print("DRONE RESET")

        if isinstance(self.drone, DroneSim):
            self._prepare_sim_drone_for_reset()

        self._move_drone_to_reset_position()
        # if not self.drone.is_flying_event.is_set():
        #     print("Control: The drone is already flying")
        #     self.drone.take_off()
        #     time.sleep(1)

        # # Ensure drone is flying before setting target position
        # self.drone.is_flying_event.wait(timeout=15)

        # # Move to the task's reset position (defaults to [0, 0, 1])
        # self.drone.set_target_position(*self.reset_position)
        # time.sleep(0.1)  # Allow target position to be set
        # self.drone.start_position_control()

        # # Wait for position to be reached
        # self.drone.at_reset_position.wait(timeout=12)
        # time.sleep(1)
        # self.drone.stop_position_control()

        # self.drone.clear_reset_position_event()

        # Reset task-specific state
        self._reset_task_state()

        # Refresh Gazebo boundary lines using current task limits.
        self._update_visual_boundaries()

        # Record initial position
        initial_position = self.drone.get_position()
        self.episode_positions.append(initial_position)

        # Start velocity controller for episode steps
        self.drone.start_velocity_control()

        return self._get_state()
    
    def _move_drone_to_reset_position(self) -> None:
        """
        Take off when necessary and move the drone to the configured reset position.

        This behaviour is shared by physical and simulated drones.
        """
        if not self.drone.is_flying_event.is_set():
            print("[Reset] Taking off before moving to reset position.")
            self.drone.take_off()
            time.sleep(1)

        # Ensure the drone is flying before enabling position control
        if not self.drone.is_flying_event.wait(timeout=15):
            print(
                "[Reset] Drone failed to confirm take-off before "
                "moving to the reset position."
            )

        self.drone.set_target_position(*self.reset_position)
        time.sleep(0.1)

        self.drone.start_position_control()

        reset_reached = self.drone.at_reset_position.wait(timeout=12)

        if not reset_reached:
            print(
                "[Reset] Drone failed to reach the reset position "
                f"within the timeout. Target: {self.reset_position}, "
                f"current: {self.drone.get_position()}"
            )

        time.sleep(1)

        self.drone.stop_position_control()
        self.drone.clear_reset_position_event()
        self.drone.set_velocity_vector(0.0, 0.0, 0.0)

    def _prepare_sim_drone_for_reset(self) -> None:
        """
        Prepare a simulated drone for a new episode.

        The drone is landed, its emergency and boundary-monitoring state is
        restored, its Gazebo model is moved to the reset spawn, and its EKF is
        reseeded before take-off.
        """
        if self.sim_manager is None:
            print("[Reset] No simulator manager available; skipping simulated drone reset.")
            return

        print("[Reset] Preparing simulated drone for reset.")

        # Ensure the simulated drone is on the ground before teleporting it
        self.drone.land()

        if not self.drone.is_landed_event.wait(timeout=10):
            print(
                "[Reset] Simulated drone did not confirm landing before teleport."
            )

        # A hard-boundary response leaves this set and disables controllers
        if self.drone.emergency_event.is_set():
            print("[Reset] Clearing simulated drone emergency state.")
            self.drone.clear_emergency_event()

        # Restore boundary monitoring if it stopped following an emergency
        if self.drone.safety_thread_active:
            self.drone.stop_boundary_monitoring()

        time.sleep(0.2)
        self.drone.start_boundary_monitoring()

        spawn_position = [
            float(self.reset_position[0]),
            float(self.reset_position[1]),
            0.02,
        ]

        print(
            "[Reset] Teleporting simulated drone to spawn position: "
            f"{spawn_position}"
        )

        self.sim_manager.set_drone_pose(
            drone_id=0,
            x=spawn_position[0],
            y=spawn_position[1],
            z=spawn_position[2],
            orientation=(0.0, 0.0, 0.0),
        )

        # Allow the model to settle onto the Gazebo floor
        time.sleep(0.3)

        self._reset_ekf(
            drone=self.drone,
            position=spawn_position,
        )

        # Give the estimator time to converge before taking off
        time.sleep(0.5)
        
    def _reset_ekf(
        self,
        drone: DroneSim,
        position: list[float] | None = None,
    ) -> None:
        """
        Seed the Kalman filter with the drone's true position and reset it.

        Only call this after the simulated drone has landed and has been
        teleported. Otherwise, the estimator may still use the old position
        and produce a large corrective command during take-off.
        """
        try:
            if getattr(drone, "cf", None) is None:
                return

            if position is not None:
                try:
                    drone.cf.param.set_value(
                        "kalman.initialX",
                        f"{float(position[0])}",
                    )
                    drone.cf.param.set_value(
                        "kalman.initialY",
                        f"{float(position[1])}",
                    )
                    drone.cf.param.set_value(
                        "kalman.initialZ",
                        f"{float(position[2])}",
                    )

                except Exception as exc:
                    print(
                        f"[{getattr(drone, 'agent_id', '?')}] "
                        "EKF seed warning; continuing with plain reset: "
                        f"{exc}"
                    )

            drone.cf.param.set_value(
                "kalman.resetEstimation",
                "1",
            )
            time.sleep(0.4)

        except Exception as exc:
            print(
                f"[{getattr(drone, 'agent_id', '?')}] "
                f"EKF reset warning: {exc}"
            )

    def _stop_drone_motion(self, reason: str = "") -> None:
        """
        Cancel the drone's current commanded motion.

        This does not land, disconnect, or stop the velocity-control thread.
        It only replaces the current velocity target with zero.
        """
        try:
            self.drone.set_velocity_vector(0.0, 0.0, 0.0)

            if reason:
                print(
                    "[SARL ENV] Zero velocity command sent: "
                    f"{reason}"
                )

        except Exception as exc:
            print(
                "[SARL ENV] Failed to send zero velocity command: "
                f"{exc}"
            )

    def step(self, action):
        """Execute one step in the environment"""

        # print(self.episode_positions)
        # Check that the current drone battery is above the threshold
        self.current_battery = self.drone.get_battery()
        print(f"Battery level: {self.current_battery}")

        self.steps += 1

        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [vx, vy, vz]")

        # Denormalize action from [-1, 1] to [-max_velocity, max_velocity]
        vx = action[0] * self.max_velocity
        vy = action[1] * self.max_velocity
        vz = action[2] * self.max_velocity_z # topples when moving up --> limit z velocity
        # vz = 0

        print("Normalised action aka velocity is:", [vx, vy, vz])

        current_pos = self.drone.get_position()
        # Store previous state for reward calculation
        self.prior_state = self._generate_state_dict(current_pos)

        # Send velocity command to drone
        self.drone.set_velocity_vector(vx, vy, vz)
        # Apply velocity for specified time - can improve this to be non-blocking
        time.sleep(self.step_time)

        new_pos = self.drone.get_position()
        current_state = self._generate_state_dict(new_pos)

        # Track position for episode trajectory
        self.episode_positions.append(new_pos)

        # Calculate reward using task-specific logic
        reward = self._calculate_reward(current_state)

        # Debugging
        print("\n")
        print("******")
        print(f"REWARD for current step {reward}")
        print(f"POSITION IN ACTION SPACE: {new_pos}")
        print(f"Episode steps: {self.steps}")

        # Check if episode is done using task-specific logic
        done = self._check_if_done(current_state)
        truncated = self._check_if_truncated(current_state)
        if done or truncated:
            self._stop_drone_motion()

        # Generate info dict
        info = {
            "current_position": new_pos,
            "previous_position": current_pos,
            "distance_to_target": self._distance_to_target(new_pos),
            "applied_velocity": [vx, vy, vz],  # Store the denorm
            "normalized_action": action,  # Store the original normalized action
            "in_boundaries": self.drone.in_boundaries,
            "steps": self.steps,
            "success_count": self.success_count,
            **self._get_additional_info(current_state),
        }
        gotten_state = self._get_state()
        return gotten_state, reward, done, truncated, info

    def _generate_state_dict(self, position: List[float]) -> Dict[str, Any]:
        """Generate a state dictionary with common drone information"""
        return {
            "position": position,
            "in_boundaries": self.drone.in_boundaries,
            "steps": self.steps,
            "distance_to_target": self._distance_to_target(position),
        }

    # def _generate_action_dict(self, action: List[float]) -> np.ndarray:
    #     """Generate a compact action representation as numpy array"""
    #     action = [
    #         action[0],  # x velocity
    #         action[1],  # y velocity
    #         action[2],  # z velocity
    #     ]
    #     return np.array(action, dtype=np.float32)

    def _distance_to_target(self, position: List[float]) -> float:
        """Calculate distance to target - to be overridden by task"""
        return 0.0

    def get_action_bounds(self) -> Dict:
        """Get the bounds for action space"""
        return {"low": [-1.0, -1.0, -1.0], "high": [1.0, 1.0, 1.0], "shape": (3,)}

    def get_action_space_info(self) -> Dict:
        """Get detailed action space information"""
        return {
            "type": "continuous",
            "shape": (3,),
            "low": [-1.0, -1.0, -1.0],
            "high": [1.0, 1.0, 1.0],
            "description": {
                "vx": "Velocity in the X direction (-1 to 1, scaled to m/s)",
                "vy": "Velocity in the Y direction (-1 to 1, scaled to m/s)",
                "vz": "Velocity in the Z direction (-1 to 1, scaled to m/s)",
            },
            "max_velocity_ms": self.max_velocity,
            "step_duration_s": self.step_time,
        }

    def set_reset_position(self, position: List[float]):
        """Set a new reset position and invalidate the cached target"""
        if len(position) != 3:
            raise ValueError("Reset position must be a 3-element list [x, y, z]")

        self.reset_position = position.copy()
        self._reset_target_set = False  # Force re-setting on next reset
        print(f"Reset position updated to {self.reset_position}")

    def close(self) -> None:
        """
        Stop the drone interface and any simulator owned by the environment.
        """
        if self._closed:
            return

        self._closed = True

        try:
            try:
                self.drone.land()

                if not self.drone.is_landed_event.wait(timeout=30):
                    print(
                        "[SARL ENV] Drone failed to confirm landing. "
                        "Forcing drone shutdown."
                    )

            except Exception as exc:
                print(f"[SARL ENV] Error while landing drone: {exc}")

            finally:
                try:
                    self.drone.stop()
                except Exception as exc:
                    print(f"[SARL ENV] Error while stopping drone: {exc}")

        finally:
            if self.sim_manager is not None:
                try:
                    self.sim_manager.stop_sim()
                except Exception as exc:
                    print(f"[SARL ENV] Error while stopping simulator: {exc}")

    def render(self, mode="human"):
        """Render the environment state"""
        if mode == "human":
            pos = self.drone.get_position()
            print(f"Drone Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            print(f"In Bounds: {self.drone.in_boundaries}")
            print(f"Steps: {self.steps}")
            self._render_task_specific_info()
            print("-" * 50)

    def set_seed(self, seed: int):
        self.seed = seed
        np.random.seed(seed)

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        """Generate a frame for video recording - to be overridden by tasks"""
        # Default implementation returns white frame
        return np.full((height, width, 3), 255, dtype=np.uint8)

    def is_in_boundaries(self, position=None):
        """Check if drone is within movement boundaries - can be overridden by tasks"""
        if position is None:
            x, y, z = self.drone.get_position()
        else:
            x, y, z = position

        in_height_range = self.z_limit < z < self.z_limit + self.reset_position[2]
        in_xy_range = abs(x) <= self.xy_limit and abs(y) <= self.xy_limit

        return in_xy_range and in_height_range

    def need_to_change_battery(self):
        self.drone.battery_level = self.drone.get_battery()
        if self.drone.battery_level <= self.battery_threshold:
            return True
        return False

    def change_battery(self):
        """
        Land the drone, wait for user confirmation of battery change, and then re-initialize and take off.
        This method is intended for use with physical drones and will not perform any actions if the drone is a simulation instance.
        """
        if isinstance(self.drone, DroneSim):
            return 
        
        if self.drone.velocity_controller_active:
            self.drone.stop_velocity_control()
        print("[Drone] Beginning battery change operation.")
        self.drone.land()
        self.drone.is_landed_event.wait(timeout=15)

        self.drone.pre_battery_change_cleanup()
        time.sleep(2)
        # loop until you get a clear 'y' or 'n' from the user
        while True:
            response = input("Is the battery changed and ready to fly? (y/n): ").lower()
            if response == "y":
                break  # Exit the loop and continue with take-off
            elif response == "n":
                print("[Drone] Operation aborted by user.")
                return False  # Exit the function
            else:
                print(
                    "[Drone] Invalid input. Please enter 'y' for yes or 'n' to abort."
                )

        # re-initialize and take off
        print("[Drone] Re-initializing...")
        self.drone.initialise_crazyflie()

        self.drone.take_off()
        if not self.drone.is_flying_event.wait(timeout=15):
            print(
                "[ERROR] Drone failed to confirm take-off. MANUAL INTERVENTION REQUIRED."
            )
            return False  # Exit because the drone is in an uncertain state

        print("[Drone] Battery change operation complete.")
        return True

    def restart(self):
        """
        Restart the drone after it has landed or crashed, ensuring it is ready for flight. 
        Does the necessary cleanup, waits for user confirmation, and does re-initialization steps before take-off.
        This method is intended for use with physical drones and will not perform any actions if the drone is a simulation instance.
        """
        if isinstance(self.drone, DroneSim):
            return 
        
        self.drone.pre_battery_change_cleanup()
        time.sleep(2)

        while True:
            response = input("Is the drone ready to fly again? (y/n): ").lower()
            if response == "y":
                break  # Exit the loop and continue with take-off
            elif response == "n":
                print("[Drone] Operation aborted by user.")
                return False  # Exit the function
            else:
                print(
                    "[Drone] Invalid input. Please enter 'y' for yes or 'n' to abort."
                )

        self.drone.initialise_crazyflie()

        self.drone.take_off()
        if not self.drone.is_flying_event.wait(timeout=15):
            print(
                "[ERROR] Drone failed to confirm take-off. MANUAL INTERVENTION REQUIRED."
            )
            return False  # Exit because the drone is in an uncertain state
        return True

    @property
    def max_action_value(self):
        return self.max_velocity

    @property
    def min_action_value(self):
        return -self.max_velocity

    @property
    def action_num(self):
        # return 3
        # on x and y values
        return 3

    # Abstract methods to be implemented by task-specific environments

    @abstractmethod
    def sample_action(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def _reset_task_state(self):
        """Reset task-specific state variables"""
        pass

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        """Get the current state representation"""
        pass

    @abstractmethod
    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward based on current state"""
        pass

    @abstractmethod
    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode is done"""
        pass

    @abstractmethod
    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode should be truncated"""
        pass

    @abstractmethod
    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional task-specific info for the info dict"""
        pass

    @abstractmethod
    def _render_task_specific_info(self):
        """Render task-specific information during rendering"""
        pass
