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
    """Test the outer velocity controller tracking accuracy with detailed logging and graphs"""
    import matplotlib.pyplot as plt
    
    drone = Drone()
    
    # Take off
    print("[Test] Taking off...")
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)
    time.sleep(2)
    
    # Start velocity controller
    print("[Test] Starting velocity controller...")
    drone.start_velocity_control()
    time.sleep(1)
    
    # Data collection for graphs
    timestamps = []
    target_velocities_x = []
    target_velocities_y = []
    internal_velocities_x = []
    internal_velocities_y = []
    calculated_velocities_x = []
    calculated_velocities_y = []
    
    # Test: Command constant velocity in X direction and measure tracking
    print("\n[Test] Testing velocity tracking - moving in +X direction...")
    target_vx, target_vy = 0.2, 0.0
    drone.set_velocity_vector(target_vx, target_vy, 0)
    
    start_time = time.time()
    test_duration = 10.0
    sample_rate = 0.1
    
    print(f"  Collecting data for {test_duration} seconds at {1/sample_rate}Hz...")
    
    prev_pos = drone.get_position()
    prev_time = start_time
    
    while (time.time() - start_time) < test_duration:
        current_time = time.time()
        elapsed = current_time - start_time
        
        current_pos = drone.get_position()
        internal_vel = drone.get_internal_velocity()
        
        dt = current_time - prev_time
        if dt > 0:
            calc_vx = (current_pos[0] - prev_pos[0]) / dt
            calc_vy = (current_pos[1] - prev_pos[1]) / dt
        else:
            calc_vx = 0.0
            calc_vy = 0.0
        
        timestamps.append(elapsed)
        target_velocities_x.append(target_vx)
        target_velocities_y.append(target_vy)
        internal_velocities_x.append(internal_vel[0])
        internal_velocities_y.append(internal_vel[1])
        calculated_velocities_x.append(calc_vx)
        calculated_velocities_y.append(calc_vy)
        
        if len(timestamps) % 10 == 0:
            print(f"    t={elapsed:.1f}s: target_vx={target_vx:.3f}, internal_vx={internal_vel[0]:.3f}, calc_vx={calc_vx:.3f}")
        
        prev_pos = current_pos
        prev_time = current_time
        time.sleep(sample_rate)
    
    # Stop movement
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(1)
    
    # Calculate statistics
    internal_vel_x_array = np.array(internal_velocities_x)
    calculated_vel_x_array = np.array(calculated_velocities_x)
    
    internal_mean_vx = np.mean(internal_vel_x_array)
    internal_error_vx = abs(internal_mean_vx - target_vx)
    calculated_mean_vx = np.mean(calculated_vel_x_array)
    calculated_error_vx = abs(calculated_mean_vx - target_vx)
    
    print(f"\n[Results]")
    print(f"  Target velocity: vx={target_vx:.3f} m/s")
    print(f"  Internal velocity (from Crazyflie): mean={internal_mean_vx:.3f} m/s, error={internal_error_vx:.3f} m/s")
    print(f"  Calculated velocity (from Vicon): mean={calculated_mean_vx:.3f} m/s, error={calculated_error_vx:.3f} m/s")
    
    # Generate graphs
    print("\n[Test] Generating velocity tracking graphs...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # X-axis velocity plot
    ax1.plot(timestamps, target_velocities_x, 'k--', label='Target vx', linewidth=2)
    ax1.plot(timestamps, internal_velocities_x, 'b-', label='Internal vx (Crazyflie)', alpha=0.7)
    ax1.plot(timestamps, calculated_velocities_x, 'r-', label='Calculated vx (Vicon)', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity X (m/s)')
    ax1.set_title('Velocity Tracking Performance - X Axis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Y-axis velocity plot
    ax2.plot(timestamps, target_velocities_y, 'k--', label='Target vy', linewidth=2)
    ax2.plot(timestamps, internal_velocities_y, 'b-', label='Internal vy (Crazyflie)', alpha=0.7)
    ax2.plot(timestamps, calculated_velocities_y, 'r-', label='Calculated vy (Vicon)', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity Y (m/s)')
    ax2.set_title('Velocity Tracking Performance - Y Axis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_tracking_performance.png', dpi=150)
    print("  Graph saved to: velocity_tracking_performance.png")
    plt.show()
    
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
