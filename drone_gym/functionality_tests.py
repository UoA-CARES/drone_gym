import time

from drone_gym.drone import Drone


def test_delay_time():

    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    for i in range(12):  # Increased to 12 for 3 complete cycles of 4 vectors
        if i % 4 == 0:
            drone.set_velocity_vector(0, 0.5, 0)    # Forward
        elif i % 4 == 1:
            drone.set_velocity_vector(0.5, 0, 0)    # Right
        elif i % 4 == 2:
            drone.set_velocity_vector(0, -0.5, 0)   # Backward
        else:
            drone.set_velocity_vector(-0.5, 0, 0)   # Left
        time.sleep(1)

    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()

def test_position_control():

    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    drone.set_velocity_vector(0.5, 0, 0)
    time.sleep(3)
    drone.start_position_control()
    drone.set_target_position(0, 0, 0.5)
    drone.at_reset_position.wait(timeout = 30)
    drone.stop_position_control()

    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()

if __name__ == "__main__":
    test_position_control()
