import time
import numpy as np

from drone_gym.drone import Drone


def test_delay_time():
    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    for i in range(52):
        if i % 4 == 0:
            drone.set_velocity_vector(0, 0.5, 0)  # Forward
        elif i % 4 == 1:
            drone.set_velocity_vector(0.5, 0, 0)  # Right
        elif i % 4 == 2:
            drone.set_velocity_vector(0, -0.5, 0)  # Backward
        else:
            drone.set_velocity_vector(-0.5, 0, 0)  # Left
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
    drone.at_reset_position.wait(timeout=30)
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
            drone.initialise_crazyflie()
        else:
            time.sleep(3)

    drone.stop()


def test_velocity_controller():
    """Test the outer velocity controller tracking accuracy"""
    drone = Drone()
    
    # Take off
    print("[Test] Taking off...")
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)
    time.sleep(2)
    
    # Start velocity controller
    print("[Test] Starting velocity controller...")
    drone.start_velocity_control()
    time.sleep(1)  # Let controller thread start
    
    # Test 1: Command constant velocity and measure tracking
    print("\n[Test 1] Testing velocity tracking accuracy...")
    target_vx, target_vy = 0.2, 0.0
    drone.set_velocity_vector(target_vx, target_vy, 0)
    
    # Measure velocity over multiple samples
    velocities = []
    for i in range(10):
        time.sleep(0.2)
        pos1 = drone.get_position()
        time.sleep(1)  # 100ms sampling
        pos2 = drone.get_position()
        
        measured_vx = (pos2[0] - pos1[0]) / 0.1
        measured_vy = (pos2[1] - pos1[1]) / 0.1
        velocities.append([measured_vx, measured_vy])
        print(f"  Sample {i+1}: vx={measured_vx:.3f} m/s (target: {target_vx}), vy={measured_vy:.3f} m/s (target: {target_vy})")
    
    # Calculate average and error
    avg_velocities = np.mean(velocities, axis=0)
    vx_error = abs(avg_velocities[0] - target_vx)
    vy_error = abs(avg_velocities[1] - target_vy)
    print(f"\n  Average: vx={avg_velocities[0]:.3f} m/s, vy={avg_velocities[1]:.3f} m/s")
    print(f"  Error: vx={vx_error:.3f} m/s, vy={vy_error:.3f} m/s")
    
    # Stop and reset
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(1)
    
    # Test 2: Simulate step() behavior - change velocity multiple times
    print("\n[Test 2] Simulating environment step() behavior...")
    test_velocities = [
        (0.15, 0.15),   # Diagonal
        (-0.1, 0.2),    # Mixed direction
        (0.0, -0.15),   # Y-axis only
        (0.25, 0.0),    # X-axis only (max)
    ]
    
    for idx, (vx, vy) in enumerate(test_velocities):
        print(f"\n  Step {idx+1}: Commanding vx={vx}, vy={vy}")
        pos_start = drone.get_position()
        
        drone.set_velocity_vector(vx, vy, 0)
        time.sleep(0.5)  # Typical step_time
        
        pos_end = drone.get_position()
        actual_displacement = [
            pos_end[0] - pos_start[0],
            pos_end[1] - pos_start[1]
        ]
        expected_displacement = [vx * 0.5, vy * 0.5]
        
        error = [
            abs(actual_displacement[0] - expected_displacement[0]),
            abs(actual_displacement[1] - expected_displacement[1])
        ]
        
        print(f"    Expected displacement: x={expected_displacement[0]:.4f}, y={expected_displacement[1]:.4f}")
        print(f"    Actual displacement:   x={actual_displacement[0]:.4f}, y={actual_displacement[1]:.4f}")
        print(f"    Error: x={error[0]:.4f} m, y={error[1]:.4f} m")
    
    # Stop velocity controller and land
    print("\n[Test] Stopping velocity controller...")
    drone.stop_velocity_control()
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(1)
    
    print("[Test] Landing...")
    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()
    
    print("\n[Test] Velocity controller test complete!")


if __name__ == "__main__":
    test_velocity_controller()
