from abc import ABC, abstractmethod
from drone_gym.drone import Drone
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# TODO - make naming more consistent

class DroneEnvironment(ABC):
    """Base drone environment that handles common drone operations"""
    def __init__(self, max_velocity: float = 0.3,  step_time: float = 1, max_steps: int = 200):

        self.drone = Drone()
        self.reset_position = [0, 0, 1]
        self.max_velocity = max_velocity
        self.step_time = step_time
        self.steps = 0
        self.max_steps = max_steps
        self.seed = 0

        self.battery_threshold = 3.15
        # self.battery_threshold = 3.45

        self.observation_space = 6

        # Movement Boundary
        self.xy_limit = 1.0
        self.z_limit = 0.5

    def _reset_control_properties(self):
        print("[DEBUG] DroneEnvironment: _reset_control_properties() - Clearing command queue and PID state")
        queue_size_before = self.drone.command_queue.qsize()
        self.drone.clear_command_queue()
        print(f"[DEBUG] DroneEnvironment: Cleared {queue_size_before} commands from queue")
        time.sleep(0.1)  # Allow any in-flight commands to be processed

        # Log PID state before reset
        print(f"[DEBUG] DroneEnvironment: PID state before reset - Last error: {self.drone.last_error}, Integral: {self.drone.integral}")
        self.drone.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        print("[DEBUG] DroneEnvironment: PID state reset to zero")

    def reset(self):
        """Reset the drone to initial position and state"""
        print("[DEBUG] DroneEnvironment: reset() called.")

        self._reset_control_properties()

        # Stop the current velocity
        print("[DEBUG] DroneEnvironment: Stopping current velocity.")
        current_pos_before_stop = self.drone.get_position()
        print(f"[DEBUG] DroneEnvironment: Position before velocity stop: {current_pos_before_stop}")
        self.drone.set_velocity_vector(0, 0, 0)
        print("[DEBUG] DroneEnvironment: Zero velocity command sent")
        time.sleep(2.5)
        current_pos_after_stop = self.drone.get_position()
        print(f"[DEBUG] DroneEnvironment: Position after velocity stop: {current_pos_after_stop}")
        print(f"[DEBUG] DroneEnvironment: Position change during stop: {[current_pos_after_stop[i] - current_pos_before_stop[i] for i in range(3)]}")
        self.steps = 0
        # Check that the drone is not already flying
        #
        print("DRONE RESET")

        if not self.drone.is_flying_event.is_set():
            print("Control: The drone is already flying")
            self.drone.take_off()

        # Ensure drone is flying before setting target position
        self.drone.is_flying_event.wait(timeout=15)

        print(f"[DEBUG] DroneEnvironment: Setting target position to {self.reset_position}")
        self.drone.set_target_position(self.reset_position[0], self.reset_position[1], self.reset_position[2])
        time.sleep(0.1)  # Allow target position to be set
        print("[DEBUG] DroneEnvironment: Starting position control.")
        self.drone.start_position_control()

        # Wait for position to be reached
        print("[DEBUG] DroneEnvironment: Waiting for drone to reach reset position.")
        self.drone.at_reset_position.wait(timeout=12)
        print("[DEBUG] DroneEnvironment: Drone reached reset position.")
        time.sleep(1)
        print("[DEBUG] DroneEnvironment: Stopping position control.")
        controller_was_active = self.drone.controller_active
        self.drone.stop_position_control()
        print(f"[DEBUG] DroneEnvironment: Position controller stopped (was active: {controller_was_active})")
        self.drone.clear_reset_position_event()
        print("[DEBUG] DroneEnvironment: Reset position event cleared")

        # Check final position before task reset
        final_reset_pos = self.drone.get_position()
        print(f"[DEBUG] DroneEnvironment: Final position after reset sequence: {final_reset_pos}")
        print(f"[DEBUG] DroneEnvironment: Distance from target reset position: {[abs(final_reset_pos[i] - self.reset_position[i]) for i in range(3)]}")

        # Reset task-specific state
        print("[DEBUG] DroneEnvironment: Resetting task-specific state")
        self._reset_task_state()

        return self._get_state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        print(f"[DEBUG] DroneEnvironment: step() called with action: {action}")
        # Check that the current drone battery is above the threshold
        self.current_battery = self.drone.get_battery()
        print(f"Battery level: {self.current_battery}")
        # if self.current_battery is not None:
        #     if self.current_battery <= self.battery_threshold:
        #         self.drone.land()

        self.steps += 1
        print(f"action: {action}")
        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [vx, vy, vz]")

        # Denormalize action from [0, 1] to [-max_velocity, max_velocity]
        vx = action[0] * (2 * self.max_velocity) - self.max_velocity  # [0,1] -> [-max_velocity, max_velocity]
        vy = action[1] * (2 * self.max_velocity) - self.max_velocity
        vz = action[2] * (2 * self.max_velocity) - self.max_velocity
        print(f"[DEBUG] DroneEnvironment: Calculated velocity: [vx={vx}, vy={vy}, vz={vz}]")

        current_pos = self.drone.get_position()
        print(f"[DEBUG] DroneEnvironment: Current position before action: {current_pos}")
        # Store previous state for reward calculation
        self.prior_state = self._generate_state_dict(current_pos)

        # Send velocity command to drone
        print("[DEBUG] DroneEnvironment: Setting velocity vector.")
        print(f"[DEBUG] DroneEnvironment: Drone flying status: {self.drone.is_flying_event.is_set()}")
        print(f"[DEBUG] DroneEnvironment: Motion commander available: {self.drone.mc is not None}")
        print(f"[DEBUG] DroneEnvironment: Controller active: {self.drone.controller_active}")
        print(f"[DEBUG] DroneEnvironment: Command queue size before velocity command: {self.drone.command_queue.qsize()}")

        self.drone.set_velocity_vector(vx, vy, vz)
        print("[DEBUG] DroneEnvironment: Velocity vector command queued")
        print(f"[DEBUG] DroneEnvironment: Command queue size after velocity command: {self.drone.command_queue.qsize()}")

        # Apply velocity for specified time - can improve this to be non-blocking
        print(f"[DEBUG] DroneEnvironment: Sleeping for {self.step_time}s to apply velocity")
        time.sleep(self.step_time)
        print("[DEBUG] DroneEnvironment: Velocity application period completed")

        new_pos = self.drone.get_position()
        print(f"[DEBUG] DroneEnvironment: New position after action: {new_pos}")
        actual_movement = [new_pos[i] - current_pos[i] for i in range(3)]
        expected_movement = [vx * self.step_time, vy * self.step_time, vz * self.step_time]
        print(f"[DEBUG] DroneEnvironment: Expected movement: {expected_movement}")
        print(f"[DEBUG] DroneEnvironment: Actual movement: {actual_movement}")
        movement_error = [actual_movement[i] - expected_movement[i] for i in range(3)]
        print(f"[DEBUG] DroneEnvironment: Movement error: {movement_error}")

        # Check if drone responded to velocity command
        total_movement = sum([abs(x) for x in actual_movement])
        if total_movement < 0.01:  # Very small movement threshold
            print(f"[WARNING] DroneEnvironment: Minimal movement detected ({total_movement:.6f}m) - drone may not be responding to velocity commands")

        current_state = self._generate_state_dict(new_pos)

        # Calculate reward using task-specific logic
        reward = self._calculate_reward(current_state)
        print(f"[DEBUG] DroneEnvironment: Calculated reward: {reward}")
        # Check if episode is done using task-specific logic
        done = self._check_if_done(current_state)
        print(f"[DEBUG] DroneEnvironment: Episode done: {done}")
        truncated = self._check_if_truncated(current_state)
        print(f"[DEBUG] DroneEnvironment: Episode truncated: {truncated}")

        # Generate info dict
        info = {
            'current_position': new_pos,
            'previous_position': current_pos,
            'distance_to_target': self._distance_to_target(new_pos),
            'applied_velocity': [vx, vy, vz],  # Store the denorm
            'normalized_action': action,  # Store the original normalized action
            'in_boundaries': self.drone.in_boundaries,
            'steps': self.steps,
            **self._get_additional_info(current_state)
        }

        return self._get_state(), reward, done, truncated, info

    def _generate_state_dict(self, position: List[float]) -> Dict[str, Any]:
        """Generate a state dictionary with common drone information"""
        return {
            'position': position,
            'in_boundaries': self.drone.in_boundaries,
            'steps': self.steps,
            'distance_to_target' : self._distance_to_target(position),
        }

    def _generate_action_dict(self, action: List[float]) -> np.ndarray:
        """Generate a compact action representation as numpy array"""
        action = [
            action[0],  # x velocity
            action[1],  # y velocity
            action[2],  # z velocity
        ]
        return np.array(action, dtype=np.float32)

    def _distance_to_target(self, position: List[float]) -> float:
        """Calculate distance to target - to be overridden by task"""
        return 0.0

    def get_action_bounds(self) -> Dict:
        """Get the bounds for action space"""
        return {
            'low': [-1.0, -1.0, -1.0],
            'high': [1.0, 1.0, 1.0],
            'shape': (3,)
        }

    def get_action_space_info(self) -> Dict:
        """Get detailed action space information"""
        return {
            'type': 'continuous',
            'shape': (3,),
            'low': [-1.0, -1.0, -1.0],
            'high': [1.0, 1.0, 1.0],
            'description': {
                'vx': 'Velocity in the X direction (-1 to 1, scaled to m/s)',
                'vy': 'Velocity in the Y direction (-1 to 1, scaled to m/s)',
                'vz': 'Velocity in the Z direction (-1 to 1, scaled to m/s)'
            },
            'max_velocity_ms': self.max_velocity,
            'step_duration_s': self.step_time
        }

    def set_max_velocity(self, max_vel: float):
        """Set the maximum velocity for action scaling"""
        self.max_velocity = max_vel
        print(f"Max velocity set to {max_vel} m/s")

    def set_step_time(self, step_time: float):
        """Set the duration for each velocity command"""
        self.step_time = step_time
        print(f"Step time set to {step_time} seconds")

    def close(self):
        """Clean up the drone environment"""
        self.drone.land()
        self.drone.is_landed_event.wait(timeout=30)
        if not self.drone.is_landed_event.is_set():
            print("Drone is failing to land....")
            print("Forcing stop")
        # time.sleep(5)
        self.drone.stop()

    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            pos = self.drone.get_position()
            print(f"Drone Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            print(f"In Bounds: {self.drone.in_boundaries}")
            print(f"Steps: {self.steps}")
            self._render_task_specific_info()
            print("-" * 50)

    def set_seed(self):
        """Generate the random seed for the environment"""
        self.seed = np.random.randint(0, 2**32 - 1)

    def is_in_testing_zone(self):
        x, y, z = self.drone.get_position()
        in_height_range = self.z_limit < z < self.z_limit + self.reset_position[2]
        if abs(x) > self.xy_limit or abs(y) > self.xy_limit:
            return False
        elif not in_height_range:
            return False

        return True

    def need_to_change_battery(self):
        self.drone.battery_level = self.drone.get_battery()
        if self.drone.battery_level <= self.battery_threshold:
            return True
        return False

    def change_battery(self):
        print("[Drone] Beginning battery change operation.")
        self.drone.land()
        self.drone.is_landed_event.wait(timeout=15)

        self.drone.pre_battery_change_cleanup()

        time.sleep(2)

        print("Battery changed? y/n")
        battery_changed = input()

        if battery_changed == "y":
            print("Continue training")

        # Reinitialise crazyflie parametres
        self.drone._initialise_crazyflie()
        self.drone.take_off()
        self.drone.is_flying_event.wait(timeout=15)
        print("[Drone] Taking off after battery change successful.")
        print("[Drone] Battery change operation complete.")


    @property
    def max_action_value(self):
        return self.max_velocity

    @property
    def min_action_value(self):
        return -self.max_velocity

    @property
    def action_num(self):
        return 3

    # Abstract methods to be implemented by task-specific environments

    @abstractmethod
    def sample_action(self) -> Any:
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
