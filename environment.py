from drone import Drone
import time
import numpy as np
import math

class DroneEnvironment:

    def __init__(self):
        self.drone = Drone()
        self.reset_position = [0, 0, 1]
        self.reward = 0.0
        self.done = False
        self.target_position = [1, 1, 1]  # Goal position
        self.max_distance = 5.0  # Maximum distance for normalization
        self.step_penalty = -0.01  # Small penalty per step
        self.success_reward = 100.0  # Reward for reaching target
        self.out_of_bounds_penalty = -50.0  # Penalty for going out of bounds
        self.distance_threshold = 0.2  # Distance threshold to consider target reached

    def reset(self):
        # Check that the drone is not already flying
        self.drone.set_target_position(self.reset_position[0], self.reset_position[1], self.reset_position[2])

        if self.drone.is_flying_event.is_set():
            print("Control: The drone is already flying")
            self.drone.land()
            self.drone.is_landed_event.wait(timeout=30)
            self.drone.stop()

        self.drone.take_off()
        self.drone.is_flying_event.wait(timeout=15)
        self.drone.start_position_control()
        time.sleep(10)

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action: A list/array of [x, y, z] position deltas or target position

        Returns:
            next_state: Current drone position [x, y, z]
            reward: Reward for this step
            done: Whether the episode has ended
            info: Additional information dictionary
        """
        # Parse action - assuming action is [dx, dy, dz] movement deltas
        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [dx, dy, dz]")

        # Get current position
        current_pos = self.drone.get_position()

        # Calculate new target position based on action
        new_target_x = current_pos[0] + action[0]
        new_target_y = current_pos[1] + action[1]
        new_target_z = current_pos[2] + action[2]

        # Send command to drone
        self.drone.set_target_position(new_target_x, new_target_y, new_target_z)

        # Wait for drone to move (adjust timing as needed)
        time.sleep(0.1)

        # Get new position after movement
        new_pos = self.drone.get_position()

        # Calculate reward
        reward = self._calculate_reward(new_pos)

        # Check if episode is done
        done = self._is_done(new_pos)

        # Create info dictionary
        info = {
            'current_position': new_pos,
            'target_position': self.target_position,
            'distance_to_target': self._distance_to_target(new_pos),
            'in_boundaries': self.drone.in_boundaries
        }

        return new_pos, reward, done, info

    def _calculate_reward(self, position):
        """Calculate reward based on current position"""
        # Distance to target
        distance = self._distance_to_target(position)

        # Base reward is negative distance (closer = higher reward)
        reward = -distance

        # Add step penalty to encourage efficiency
        reward += self.step_penalty

        # Bonus for reaching target
        if distance < self.distance_threshold:
            reward += self.success_reward

        # Penalty for going out of bounds
        if not self.drone.in_boundaries:
            reward += self.out_of_bounds_penalty

        return reward

    def _distance_to_target(self, position):
        """Calculate Euclidean distance to target position"""
        return math.sqrt(
            (position[0] - self.target_position[0])**2 +
            (position[1] - self.target_position[1])**2 +
            (position[2] - self.target_position[2])**2
        )

    def _is_done(self, position):
        """Check if episode should end"""
        # Episode ends if:
        # 1. Target is reached
        # 2. Drone goes out of bounds
        # 3. Drone crashes (z <= 0)

        distance = self._distance_to_target(position)

        # Success condition
        if distance < self.distance_threshold:
            self.done = True
            return True

        # Failure conditions
        if not self.drone.in_boundaries or position[2] <= 0:
            self.done = True
            return True

        return False

    def set_target_position(self, target):
        """Set a new target position for the drone to reach"""
        if len(target) != 3:
            raise ValueError("Target must be a 3-element array [x, y, z]")
        self.target_position = target

    def get_observation_space(self):
        """Get the observation space (current position)"""
        return 3  # [x, y, z]

    def get_action_space(self):
        """Get the action space (position deltas)"""
        return 3  # [dx, dy, dz]

    def get_state(self):
        """Get current state of the environment"""
        position = self.drone.get_position()
        return {
            'position': position,
            'target': self.target_position,
            'distance_to_target': self._distance_to_target(position),
            'in_boundaries': self.drone.in_boundaries,
            'done': self.done
        }

    def close(self):
        """Clean up the environment"""
        if hasattr(self.drone, 'is_flying_event') and self.drone.is_flying_event.is_set():
            self.drone.land()
            self.drone.is_landed_event.wait(timeout=30)
        self.drone.stop()

    def render(self, mode='human'):
        """Render the environment state"""
        if mode == 'human':
            pos = self.drone.get_position()
            target = self.target_position
            distance = self._distance_to_target(pos)
            print(f"Drone Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            print(f"Target Position: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
            print(f"Distance to Target: {distance:.2f}")
            print(f"In Bounds: {self.drone.in_boundaries}")
            print(f"Done: {self.done}")
            print("-" * 50)





if __name__ == "__main__":
    env = DroneEnvironment()

    # Reset environment
    env.reset()

    # Set a target position
    env.set_target_position([1.5, 1.0, 1.2])

    # Example episode
    for step in range(50):
        # Get current state
        current_state = env.get_state()
        print(f"Step {step}: Position {current_state['position']}, Distance: {current_state['distance_to_target']:.2f}")

        # Simple action - move towards target
        current_pos = env.drone.get_position()
        target = env.target_position

        # Calculate direction to target (simple proportional control)
        dx = (target[0] - current_pos[0]) * 0.1
        dy = (target[1] - current_pos[1]) * 0.1
        dz = (target[2] - current_pos[2]) * 0.1

        # Limit movement per step
        max_step = 0.2
        dx = max(-max_step, min(max_step, dx))
        dy = max(-max_step, min(max_step, dy))
        dz = max(-max_step, min(max_step, dz))

        action = [dx, dy, dz]

        # Take step
        next_state, reward, done, info = env.step(action)

        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}")

        if done:
            print("Episode finished!")
            break

        time.sleep(0.5)  # Small delay for visualization

    # Clean up
    env.close()
