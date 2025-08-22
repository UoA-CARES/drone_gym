import time
import threading

from drone_gym.drone import Drone


def test_continuous_commands():
    """Test that the continuous command system maintains velocity setpoints"""
    print("=== Testing Continuous Command System ===")

    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if not drone.is_flying_event.is_set():
        print("ERROR: Failed to take off")
        return False

    print("Testing velocity setpoint persistence...")

    # Test 1: Set velocity and check it persists for multiple seconds
    print("Test 1: Velocity persistence")
    drone.set_velocity_vector(0.1, 0, 0)  # Move forward slowly
    time.sleep(5)  # Should maintain velocity for 5 seconds

    # Check current position has changed (indicating continuous movement)
    initial_pos = drone.get_position()
    time.sleep(2)
    final_pos = drone.get_position()

    movement = abs(final_pos[0] - initial_pos[0])
    if movement > 0.05:  # Should have moved at least 5cm
        print(f"✓ Velocity maintained - moved {movement:.3f}m in 2 seconds")
    else:
        print(f"✗ Velocity not maintained - only moved {movement:.3f}m")

    # Test 2: Change velocity mid-flight
    print("Test 2: Velocity change responsiveness")
    drone.set_velocity_vector(0, 0.2, 0)  # Change to sideways
    time.sleep(3)
    drone.set_velocity_vector(0, 0, 0)  # Stop
    time.sleep(1)

    # Test 3: Square pattern with sustained velocity
    print("Test 3: Square pattern with continuous commands")
    movements = [
        (0.3, 0, 0),    # Forward
        (0, 0.3, 0),    # Right
        (-0.3, 0, 0),   # Backward
        (0, -0.3, 0)    # Left
    ]

    for i, (vx, vy, vz) in enumerate(movements):
        print(f"  Movement {i+1}: vx={vx}, vy={vy}, vz={vz}")
        drone.set_velocity_vector(vx, vy, vz)
        time.sleep(4)  # Each movement for 4 seconds

    # Stop all movement
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(1)

    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()

    print("=== Continuous Command Test Complete ===")
    return True


def test_command_frequency():
    """Test that commands are being sent at the expected frequency"""
    print("=== Testing Command Frequency ===")

    drone = Drone()

    # Wait for hardware to be ready
    drone.hardware_ready_event.wait(timeout=20)

    if not drone.commander:
        print("✗ Commander not available for frequency testing")
        drone.stop()
        return False

    # Access the continuous command sender directly for testing
    command_count = 0
    original_send_method = drone.commander.send_velocity_world_setpoint

    def count_commands(vx, vy, vz, yaw_rate):
        nonlocal command_count
        command_count += 1
        if original_send_method:
            original_send_method(vx, vy, vz, yaw_rate)

    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if not drone.is_flying_event.is_set():
        print("✗ Failed to take off for frequency test")
        drone.stop()
        return False

    # Monkey patch to count commands
    drone.commander.send_velocity_world_setpoint = count_commands

    # Set a velocity and measure command frequency
    drone.set_velocity_vector(0.1, 0, 0)
    initial_count = command_count
    time.sleep(3)  # Measure for 3 seconds for better accuracy
    final_count = command_count

    commands_per_second = (final_count - initial_count) / 3.0
    expected_rate = drone.command_rate

    print(f"Commands sent: {final_count - initial_count} in 3 seconds")
    print(f"Measured rate: {commands_per_second:.1f} Hz")
    print(f"Expected rate: {expected_rate} Hz")

    frequency_ok = abs(commands_per_second - expected_rate) < 3  # Allow 3Hz tolerance
    if frequency_ok:
        print("✓ Command frequency is correct")
    else:
        print("✗ Command frequency is incorrect")

    # Restore original method
    drone.commander.send_velocity_world_setpoint = original_send_method

    drone.set_velocity_vector(0, 0, 0)
    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()

    print("=== Command Frequency Test Complete ===")
    return frequency_ok


def test_thread_coordination():
    """Test that all threads start and stop properly"""
    print("=== Testing Thread Coordination ===")

    drone = Drone()

    # Check that all expected threads are running
    expected_threads = [
        'position_thread',
        'safety_thread',
        'command_sender_thread',
        'thread'  # main thread
    ]

    time.sleep(2)  # Let threads start

    active_threads = []
    for thread_name in expected_threads:
        thread_obj = getattr(drone, thread_name, None)
        if thread_obj and thread_obj.is_alive():
            active_threads.append(thread_name)
            print(f"✓ {thread_name} is running")
        else:
            print(f"✗ {thread_name} is not running")

    print(f"Active threads: {len(active_threads)}/{len(expected_threads)}")

    # Test proper shutdown
    print("Testing shutdown...")
    drone.stop()

    time.sleep(3)  # Wait for shutdown

    stopped_threads = []
    for thread_name in expected_threads:
        thread_obj = getattr(drone, thread_name, None)
        if not thread_obj or not thread_obj.is_alive():
            stopped_threads.append(thread_name)
            print(f"✓ {thread_name} stopped properly")
        else:
            print(f"✗ {thread_name} failed to stop")

    print(f"Stopped threads: {len(stopped_threads)}/{len(expected_threads)}")
    print("=== Thread Coordination Test Complete ===")


def test_position_control():
    """Test position control with continuous commands"""
    print("=== Testing Position Control ===")

    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if not drone.is_flying_event.is_set():
        print("ERROR: Failed to take off")
        return False


    drone.start_position_control()

    drone.set_target_position(0, 0, 1)

    print("Waiting for position to be reached...")
    position_reached = drone.at_reset_position.wait(timeout=30)

    if position_reached:
        print("✓ Position control reached target")
    else:
        print("✗ Position control failed to reach target")

    drone.stop_position_control()

    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()

    print("=== Position Control Test Complete ===")
    return position_reached


def run_all_tests():
    """Run all functionality tests"""
    print("Starting comprehensive drone functionality tests...")

    tests = [
        ("Thread Coordination", test_thread_coordination),
        ("Continuous Commands", test_continuous_commands),
        ("Command Frequency", test_command_frequency),
        ("Position Control", test_position_control)
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")

        try:
            result = test_func()
            results[test_name] = result if result is not None else True
        except Exception as e:
            print(f"ERROR in {test_name}: {str(e)}")
            results[test_name] = False

        time.sleep(2)  # Pause between tests

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    return passed == total


def test_world_frame_directions():
    """Test world frame directions by moving drone and observing position changes"""
    print("=== Testing World Frame Directions ===")
    print("This test will move the drone in each world axis direction")
    print("and record the position changes to determine the coordinate system.")

    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)

    if not drone.is_flying_event.is_set():
        print("ERROR: Failed to take off")
        drone.stop()
        return False

    print("Drone took off successfully. Starting direction tests...")

    # Wait for position to stabilize after takeoff
    time.sleep(3)

    # Stop any existing position control to get clean velocity control
    drone.stop_position_control()
    time.sleep(1)

    # Record initial position
    initial_pos = drone.get_position()
    print(f"Initial position: x={initial_pos[0]:.3f}, y={initial_pos[1]:.3f}, z={initial_pos[2]:.3f}")

    # Test parameters
    test_velocity = 0.3  # m/s - moderate speed for clear observation
    test_duration = 4    # seconds per test
    settle_time = 2      # seconds to settle between tests

    results = {}

    # Test +X direction
    print("\n--- Testing +X direction (sending vx=+0.3 m/s) ---")
    drone.set_velocity_vector(test_velocity, 0, 0)
    time.sleep(test_duration)

    # Stop and measure position change
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(settle_time)

    pos_after_x = drone.get_position()
    dx = pos_after_x[0] - initial_pos[0]
    dy = pos_after_x[1] - initial_pos[1]
    dz = pos_after_x[2] - initial_pos[2]

    print(f"Position after +X: x={pos_after_x[0]:.3f}, y={pos_after_x[1]:.3f}, z={pos_after_x[2]:.3f}")
    print(f"Position change: Δx={dx:.3f}, Δy={dy:.3f}, Δz={dz:.3f}")

    # Determine which axis moved most
    abs_changes = [abs(dx), abs(dy), abs(dz)]
    max_change_idx = abs_changes.index(max(abs_changes))
    axis_names = ['X', 'Y', 'Z']
    direction_signs = ['+' if dx > 0 else '-', '+' if dy > 0 else '-', '+' if dz > 0 else '-']

    results['X_positive'] = {
        'command': '+X velocity',
        'primary_axis': axis_names[max_change_idx],
        'direction': direction_signs[max_change_idx],
        'movement': [dx, dy, dz]
    }

    print(f"→ +X command primarily moved {direction_signs[max_change_idx]}{axis_names[max_change_idx]} axis")

    # Return to initial position approximately
    drone.set_velocity_vector(-test_velocity, 0, 0)
    time.sleep(test_duration)
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(settle_time)

    # Test +Y direction
    print("\n--- Testing +Y direction (sending vy=+0.3 m/s) ---")
    current_pos = drone.get_position()

    drone.set_velocity_vector(0, test_velocity, 0)
    time.sleep(test_duration)

    # Stop and measure
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(settle_time)

    pos_after_y = drone.get_position()
    dx = pos_after_y[0] - current_pos[0]
    dy = pos_after_y[1] - current_pos[1]
    dz = pos_after_y[2] - current_pos[2]

    print(f"Position after +Y: x={pos_after_y[0]:.3f}, y={pos_after_y[1]:.3f}, z={pos_after_y[2]:.3f}")
    print(f"Position change: Δx={dx:.3f}, Δy={dy:.3f}, Δz={dz:.3f}")

    abs_changes = [abs(dx), abs(dy), abs(dz)]
    max_change_idx = abs_changes.index(max(abs_changes))
    direction_signs = ['+' if dx > 0 else '-', '+' if dy > 0 else '-', '+' if dz > 0 else '-']

    results['Y_positive'] = {
        'command': '+Y velocity',
        'primary_axis': axis_names[max_change_idx],
        'direction': direction_signs[max_change_idx],
        'movement': [dx, dy, dz]
    }

    print(f"→ +Y command primarily moved {direction_signs[max_change_idx]}{axis_names[max_change_idx]} axis")

    # Return to center
    drone.set_velocity_vector(0, -test_velocity, 0)
    time.sleep(test_duration)
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(settle_time)

    # Test +Z direction
    print("\n--- Testing +Z direction (sending vz=+0.2 m/s) ---")
    current_pos = drone.get_position()

    drone.set_velocity_vector(0, 0, 0.2)  # Slower for Z to avoid ceiling
    time.sleep(test_duration)

    # Stop and measure
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(settle_time)

    pos_after_z = drone.get_position()
    dx = pos_after_z[0] - current_pos[0]
    dy = pos_after_z[1] - current_pos[1]
    dz = pos_after_z[2] - current_pos[2]

    print(f"Position after +Z: x={pos_after_z[0]:.3f}, y={pos_after_z[1]:.3f}, z={pos_after_z[2]:.3f}")
    print(f"Position change: Δx={dx:.3f}, Δy={dy:.3f}, Δz={dz:.3f}")

    abs_changes = [abs(dx), abs(dy), abs(dz)]
    max_change_idx = abs_changes.index(max(abs_changes))
    direction_signs = ['+' if dx > 0 else '-', '+' if dy > 0 else '-', '+' if dz > 0 else '-']

    results['Z_positive'] = {
        'command': '+Z velocity',
        'primary_axis': axis_names[max_change_idx],
        'direction': direction_signs[max_change_idx],
        'movement': [dx, dy, dz]
    }

    print(f"→ +Z command primarily moved {direction_signs[max_change_idx]}{axis_names[max_change_idx]} axis")

    # Return to safe height
    drone.set_velocity_vector(0, 0, -0.2)
    time.sleep(test_duration)
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(settle_time)

    # Summary of findings
    print(f"\n{'='*60}")
    print("WORLD FRAME COORDINATE SYSTEM ANALYSIS")
    print(f"{'='*60}")

    print("Command → Physical Movement:")
    for cmd, data in results.items():
        print(f"  {data['command']} → {data['direction']}{data['primary_axis']} axis movement")

    # Determine coordinate system type
    print(f"\nCoordinate System Analysis:")
    x_moves = results['X_positive']['primary_axis'] + results['X_positive']['direction']
    y_moves = results['Y_positive']['primary_axis'] + results['Y_positive']['direction']
    z_moves = results['Z_positive']['primary_axis'] + results['Z_positive']['direction']

    print(f"  +X velocity command moves: {x_moves}")
    print(f"  +Y velocity command moves: {y_moves}")
    print(f"  +Z velocity command moves: {z_moves}")

    # Try to identify common coordinate systems
    if x_moves == 'X+' and y_moves == 'Y+' and z_moves == 'Z+':
        print("  → Standard Right-Handed Coordinate System")
    elif x_moves == 'Y+' and y_moves == 'X-' and z_moves == 'Z+':
        print("  → 90° Rotated Coordinate System")
    else:
        print("  → Custom or Rotated Coordinate System")

    print(f"\nOrigin appears to be at: x≈{initial_pos[0]:.3f}, y≈{initial_pos[1]:.3f}, z≈{initial_pos[2]:.3f}")

    # Land the drone
    print("\nLanding drone...")
    drone.land()
    drone.is_landed_event.wait(timeout=15)
    drone.stop()

    print("=== World Frame Direction Test Complete ===")

    return results

def height():
    """Test position control with continuous commands"""
    print("=== Testing Position Control ===")

    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)
    drone.set_velocity_vector(0, 0, 0.09)
    time.sleep(5)
    drone.land()
    time.sleep(1)
    drone.stop()

def control():
    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)
    time.sleep(1)
    drone.start_position_control()

def beep():
    drone = Drone()
    drone.take_off()
    drone.is_flying_event.wait(timeout=15)
    time.sleep(2)
    for i in range(12):
        drone.set_velocity_vector(0, 0.5, 0.05)
        time.sleep(3)
        with drone.velocity_setpoint_lock:
            print(drone.current_velocity_setpoint.copy())
        drone.set_velocity_vector(0, -0.5, 0)
        time.sleep(3)
        with drone.velocity_setpoint_lock:
            print(drone.current_velocity_setpoint.copy())
    drone.land()
    time.sleep(2)

if __name__ == "__main__":
    print(beep())
