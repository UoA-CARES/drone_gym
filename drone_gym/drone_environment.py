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

        self.current_battery = self.drone.get_battery()
        self.battery_threshold = 3.5
        self.observation_space = 6

        # Movement Boundary
        self.xy_limit = 1.5
        self.z_limit = 0.5

    def _reset_control_properties(self):
        self.drone.clear_command_queue()
        self.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.integral = {"x": 0.0, "y": 0.0, "z": 0.0}

    def reset(self):
        """Reset the drone to initial position and state"""

        self._reset_control_properties()

        # Stop the current velocity
        self.drone.set_velocity_vector(0, 0, 0)
        time.sleep(0.5)
        self.steps = 0
        # Check that the drone is not already flying
        #
        print("DRONE RESET")
        if not self.drone.is_flying_event.is_set():
            print("Control: The drone is already flying")
            self.drone.take_off()

        self.drone.set_target_position(self.reset_position[0], self.reset_position[1], self.reset_position[2])
        self.drone.is_flying_event.wait(timeout=15)
        self.drone.start_position_control()
        self.drone.at_reset_position.wait(timeout=12)
        time.sleep(1)
        self.drone.stop_position_control()
        self.drone.clear_reset_position_event()
        # Reset task-specific state
        self._reset_task_state()

        return self._get_state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Check that the current drone battery is above the threshold
        # self.current_battery = self.drone.get_battery()
        # if self.current_battery <= self.battery_threshold:
        #     self.drone.land_and_stop()
        self.steps += 1
        print(f"action: {action}")
        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [vx, vy, vz]")

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
        # Check if episode is done using task-specific logic
        done = self._check_if_done(current_state)
        truncated = self._check_if_truncated(current_state)

        # Generate info dict
        info = {
            'current_position': new_pos,
            'previous_position': current_pos,
            'distance_to_target': self._distance_to_target(new_pos),
            'applied_velocity': [vx, vy, vz],  # Store the denormalized velocities
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

    def _is_in_testing_zone(self):
        x, y, z = self.drone.get_position()
        in_height_range = self.z_limit < z < self.z_limit + self.reset_position[2]
        if abs(x) > self.xy_limit or abs(y) > self.xy_limit:
            return False
        elif not in_height_range:
            return False

        return True

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
