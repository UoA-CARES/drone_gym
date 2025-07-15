from drone import Drone
import time

class DroneEnvironment:

    def __init__(self):
        self.drone = Drone()
        self.reset_position = [0, 0, 1]
        self.reward = 0.0
        self.done = False

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
        # to do




if __name__ == "__main__":
    env = DroneEnvironment()
    env.reset()
