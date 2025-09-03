from abc import ABC, abstractmethod
from drone_gym.drone import Drone
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# TODO - make naming more consistent

class DroneEnvironment(ABC):
    """Base drone environment that handles common drone operations"""
    def __init__(self, max_velocity: float = 0.25,  step_time: float = 0.5, exploration_steps: int = 1000):

        self.drone = Drone()
        self.reset_position = [0, 0, 1]
        self.max_velocity = max_velocity
        self.step_time = step_time
        self.steps = 0
        self.exploration_steps = exploration_steps
        self.truncate_next = False
        self.seed = 0

        self.total_steps = 0

        self.battery_threshold = 3.15

        self.observation_space = 8

        # Movement Boundary
        self.xy_limit = 1.0
        self.z_limit = 0.5

        # Reset target position optimization
        self._reset_target_set = False

        # Reward specific properties
        self.exited_testing_boundary = False
        self.boundary_penalise = False

    def _reset_control_properties(self):
        self.drone.clear_command_queue()
        time.sleep(0.5)  # Allow any in-flight commands to be processed
        self.drone.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone.integral = {"x": 0.0, "y": 0.0, "z": 0.0}

    def reset(self):
        """Reset the drone to initial position and state"""

        self._reset_control_properties()

        # Stop the current velocity
        self.drone.set_velocity_vector(0, 0, 0)
        time.sleep(0.5)


        self.steps = 0
        # Check that the drone is not already flying
        print("DRONE RESET")

        if not self.drone.is_flying_event.is_set():
            print("Control: The drone is already flying")
            self.drone.take_off()
            time.sleep(1)

        # Ensure drone is flying before setting target position
        self.drone.is_flying_event.wait(timeout=15)

        # Set reset target position only once (lazy initialization)
        # self._ensure_reset_target_set()
        self.drone.set_target_position(0, 0, 1)
        time.sleep(0.1)  # Allow target position to be set
        self.drone.start_position_control()

        # Wait for position to be reached
        self.drone.at_reset_position.wait(timeout=12)
        time.sleep(1)
        self.drone.stop_position_control()

        if self.exited_testing_boundary:
            self.boundary_penalise = True
            self.exited_testing_boundary = False
            print("--------")
            print(f"total steps: {self.total_steps}")
            print("--------")
        self.drone.clear_reset_position_event()

        # Reset task-specific state
        self._reset_task_state()

        return self._get_state()

    def step(self, action):
        """Execute one step in the environment"""
        # Check that the current drone battery is above the threshold
        self.current_battery = self.drone.get_battery()
        print(f"Battery level: {self.current_battery}")

        self.steps += 1
        self.total_steps += 1

        if self.total_steps == self.exploration_steps:

            print("\n")
            print("SWITCHING TO LEARNING PHASE...")
            print("\n")

            self.truncate_next = True

        print(f"action: {action}")
        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [vx, vy, vz]")

        if self.total_steps > self.exploration_steps:
            # Denormalize action from [-1, 1] to [-max_velocity, max_velocity]
            vx = action[0] * self.max_velocity
            vy = action[1] * self.max_velocity
            vz = action[2] * self.max_velocity
        else:
            # Denormalize action from [0, 1] to [-max_velocity, max_velocity]
            vx = action[0] * (2 * self.max_velocity) - self.max_velocity  # [0,1] -> [-max_velocity, max_velocity]
            vy = action[1] * (2 * self.max_velocity) - self.max_velocity
            vz = action[2] * (2 * self.max_velocity) - self.max_velocity

        current_pos = self.drone.get_position()
        # Store previous state for reward calculation
        self.prior_state = self._generate_state_dict(current_pos)

        # Send velocity command to drone
        self.drone.set_velocity_vector(vx, vy, vz)
        # Apply velocity for specified time - can improve this to be non-blocking
        time.sleep(self.step_time)

        new_pos = self.drone.get_position()
        current_state = self._generate_state_dict(current_pos)

        # Calculate reward using task-specific logic
        reward = self._calculate_reward(current_state)

        # Debugging
        print("\n")
        print("******")
        print(f"REWARD for current step {reward}")
        print(f"POSITION IN ACTION SPACE: {new_pos}")
        print(f"Total steps: {self.total_steps}")

        # Check if episode is done using task-specific logic
        done = self._check_if_done(current_state)
        truncated = self._check_if_truncated(current_state)

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

    def set_reset_position(self, position: List[float]):
        """Set a new reset position and invalidate the cached target"""
        if len(position) != 3:
            raise ValueError("Reset position must be a 3-element list [x, y, z]")

        self.reset_position = position.copy()
        self._reset_target_set = False  # Force re-setting on next reset
        print(f"Reset position updated to {self.reset_position}")

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
