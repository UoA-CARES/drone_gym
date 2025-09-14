import logging
import sys
import time
from threading import Event
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
DEFAULT_HEIGHT = 0.5
deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

# Global variable to store position estimate
position_estimate = [0.0, 0.0]

def log_pos_callback(timestamp, data, logconf):
    """Callback for logging position data and updating the global variable."""
    global position_estimate
    position_estimate[0] = data['stateEstimate.x']
    position_estimate[1] = data['stateEstimate.y']

def param_deck_flow(_, value_str):
    """Callback for deck parameter updates."""
    value = int(value_str)
    if value:
        deck_attached_event.set()
        print('Flow deck is attached!')
    else:
        print('Flow deck is NOT attached!')

def test_move_duration(scf, duration):
    """Test a specific move duration and return success/failure."""
    try:
        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            print(f"Testing move_duration: {duration:.3f}s")

            # Test all 4 directions with this duration
            directions = [
                (0, 0.2, 0, "forward"),
                (0.2, 0, 0, "right"),
                (0, -0.2, 0, "backward"),
                (-0.2, 0, 0, "left")
            ]

            for vx, vy, vz, direction in directions:
                print(f"  Moving {direction}...")
                mc.start_linear_motion(vx, vy, vz)
                time.sleep(duration)
                mc.stop()
                time.sleep(0.05)  # Minimal pause between moves

            print(f"✓ Duration {duration:.3f}s completed successfully")
            return True

    except Exception as e:
        print(f"✗ Duration {duration:.3f}s failed: {e}")
        return False

def find_shortest_duration(scf):
    """Binary search to find the shortest stable move duration."""
    print("=== Finding Shortest Move Duration ===")

    # Define search range (in seconds)
    min_duration = 0.001  # 1ms - very aggressive start
    max_duration = 0.5    # 500ms - conservative upper bound
    tolerance = 0.001     # Stop when we're within 1ms

    successful_durations = []
    failed_durations = []

    # Test some key points first to establish bounds
    test_points = [0.3, 0.4, 0.5]

    print("\n--- Initial Range Testing ---")
    for duration in test_points:
        if test_move_duration(scf, duration):
            successful_durations.append(duration)
        else:
            failed_durations.append(duration)
        time.sleep(2)  # Rest between tests

    if not successful_durations:
        print("ERROR: No durations worked! Try increasing the test range.")
        return None

    # Find the working range
    min_working = min(successful_durations)
    max_failed = max(failed_durations) if failed_durations else 0

    print(f"\nWorking durations found: {successful_durations}")
    print(f"Failed durations: {failed_durations}")
    print(f"Shortest working duration so far: {min_working:.3f}s")

    # Binary search between max_failed and min_working
    if max_failed < min_working:
        left = max_failed
        right = min_working

        print(f"\n--- Binary Search between {left:.3f}s and {right:.3f}s ---")

        iteration = 0
        while (right - left) > tolerance and iteration < 10:
            iteration += 1
            mid = (left + right) / 2
            print(f"\nIteration {iteration}: Testing {mid:.3f}s")

            if test_move_duration(scf, mid):
                right = mid
                print(f"  Success! New upper bound: {right:.3f}s")
            else:
                left = mid
                print(f"  Failed. New lower bound: {left:.3f}s")

            time.sleep(2)  # Rest between tests

        final_duration = right
    else:
        final_duration = min_working

    print("\n=== RESULT ===")
    print(f"Shortest stable move_duration: {final_duration:.3f}s ({final_duration*1000:.1f}ms)")
    return final_duration

def verify_shortest_duration(scf, duration, cycles=3):
    """Verify the shortest duration with multiple cycles."""
    print(f"\n=== Verifying {duration:.3f}s with {cycles} cycles ===")

    try:
        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            for cycle in range(cycles):
                print(f"Verification cycle {cycle + 1}/{cycles}")

                # Square pattern
                movements = [
                    (0, 0.2, 0, "forward"),
                    (0.2, 0, 0, "right"),
                    (0, -0.2, 0, "backward"),
                    (-0.2, 0, 0, "left")
                ]

                for vx, vy, vz, direction in movements:
                    mc.start_linear_motion(vx, vy, vz)
                    time.sleep(duration)
                    mc.stop()
                    time.sleep(0.01)  # Minimal pause

                time.sleep(0.5)  # Pause between cycles

        print(f"✓ Verification successful! {duration:.3f}s is stable.")
        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        try:
            scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                             cb=param_deck_flow)
            print('Waiting for flow deck to be attached...')

            if not deck_attached_event.wait(timeout=5):
                print('No flow deck detected!')
                sys.exit(1)

            logconf = LogConfig(name='Position', period_in_ms=10)
            logconf.add_variable('stateEstimate.x', 'float')
            logconf.add_variable('stateEstimate.y', 'float')
            scf.cf.log.add_config(logconf)
            logconf.data_received_cb.add_callback(log_pos_callback)
            logconf.start()

            time.sleep(1.0)

            # Find the shortest duration
            shortest = find_shortest_duration(scf)

            if shortest:
                time.sleep(3)  # Rest before verification
                # Verify it works consistently
                verify_shortest_duration(scf, shortest, cycles=3)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
        finally:
            print("Disarming...")
            try:
                scf.cf.platform.send_arming_request(False)
            except:
                pass

    print("Test finished.")
