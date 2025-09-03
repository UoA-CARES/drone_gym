import time

from drone_gym.drone import Drone


def test_delay_time():

    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    for i in range(52):
        if i % 4 == 0:
            drone.set_velocity_vector(0, 0.5, 0)    # Forward
        elif i % 4 == 1:
            drone.set_velocity_vector(0.5, 0, 0)    # Right
        elif i % 4 == 2:
            drone.set_velocity_vector(0, -0.5, 0)   # Backward
        else:
            drone.set_velocity_vector(-0.5, 0, 0)   # Left
        time.sleep(2)

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


def taking_off():
    drone = Drone()

    for i in range(10):
        print(f"Flight cycle {i + 1}/10")

        # Take off and fly
        drone.take_off()
        drone.is_flying_event.wait(timeout=15)
        time.sleep(3)

        # Land
        drone.land()
        drone.is_landed_event.wait(timeout=15)

        if i < 9:  # Don't ask for battery change on the last cycle
            drone.pre_battery_change_cleanup()
            print("input:")
            battery_changed = input()
            print(battery_changed)
            drone._initialise_crazyflie()
        else:
            time.sleep(3)

    drone.stop()


if __name__ == "__main__":
    taking_off()
