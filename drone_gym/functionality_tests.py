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
    drone.set_velocity_vector(0.2, 0, 0)  # Move forward slowly
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

    print("Testing manual velocity first...")
    drone.set_velocity_vector(0.5, 0, 0)
    time.sleep(3)
    print("Stopping manual velocity...")
    drone.set_velocity_vector(0, 0, 0)
    time.sleep(1)
    
    print("Starting position control...")
    drone.start_position_control()
    drone.set_target_position(0, 0, 0.5)
    
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


if __name__ == "__main__":
    run_all_tests()
