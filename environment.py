from abc import ABC, abstractmethod
from drone import Drone
import time
import numpy as np
from typing import Dict, List, Tuple, Any


class DroneEnvironment(ABC):
    """Base drone environment that handles common drone operations"""
    def __init__(self, max_velocity: float = 1.0, step_time: float = 0.1, max_steps: int = 1000):

        self.drone = Drone()
        self.reset_position = [0, 0, 0.5]
        self.max_velocity = max_velocity
        self.step_time = step_time
        self.steps = 0
        self.max_steps = max_steps

        self.current_battery = self.drone.get_battery()
        self.battery_threshold = 3.5


    def reset(self):
        """Reset the drone to initial position and state"""
        self.steps = 0

        # Check that the drone is not already flying
        self.drone.set_target_position(self.reset_position[0], self.reset_position[1], self.reset_position[2])

        if self.drone.is_flying_event.is_set():
            print("Control: The drone is already flying")
            self.drone.land_and_stop()

        self.drone.take_off()
        self.drone.is_flying_event.wait(timeout=15)
        self.drone.start_position_control()
        time.sleep(10)
        self.drone.stop_position_control()

        # Reset task-specific state
        self._reset_task_state()

        return self._get_state(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""

        # Check that the current drone battery is above the threshold
        self.current_battery = self.drone.get_battery()
        if self.current_battery <= self.battery_threshold:
            self.drone.land_and_stop()

        self.steps += 1

        # Parse action - expecting [vx, vy, vz] in range [-1, 1]
        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [vx, vy, vz]")

        current_pos = self.drone.get_position()

        # Store previous state for reward calculation
        self.prior_state = self._generate_state_dict(current_pos)

        vx, vy, vz = action

        # Send velocity command to drone
        self.drone.set_velocity_vector(vx, vy, vz)

        # Apply velocity for specified time - can improve this to be non-blocking
        time.sleep(self.step_time)

        # Stop velocity after step duration
        self.drone.stop_velocity()

        new_pos = self.drone.get_position()
        current_state = self._generate_state_dict(new_pos)

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
            'applied_velocity': [vx, vy, vz],
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
            **self._get_task_specific_state()
        }

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

    @property
    def max_action_value(self):
        return self.max_velocity

    @property
    def min_action_value(self):
        return -self.max_velocity

    # Abstract methods to be implemented by task-specific environments

    @abstractmethod
    def sample_action(self):
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
    def _get_task_specific_state(self) -> Dict[str, Any]:
        """Get task-specific state information"""
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
