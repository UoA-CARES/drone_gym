import logging
import sys
import time

import cflib
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander


# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_SIM_PORT = 19850
DEFAULT_RADIO_CHANNEL = "100"
DEFAULT_RADIO_DATARATE = "2M"
DEFAULT_RADIO_PREFIX = "E7E7E7E7"

_DRIVERS_INITIALIZED = False


def init_drivers() -> None:
    global _DRIVERS_INITIALIZED

    if not _DRIVERS_INITIALIZED:
        cflib.crtp.init_drivers()
        _DRIVERS_INITIALIZED = True


def generate_physical_uris(count: int) -> list[str]:
    if count < 1:
        raise ValueError("NUM_DRONES must be at least 1")

    if count > 256:
        raise ValueError(
            "NUM_DRONES must be 256 or less for the generated radio addresses"
        )

    return [
        (
            f"radio://0/"
            f"{DEFAULT_RADIO_CHANNEL}/"
            f"{DEFAULT_RADIO_DATARATE}/"
            f"{DEFAULT_RADIO_PREFIX}{index:02X}"
        )
        for index in range(count)
    ]


def generate_sim_uris(count: int) -> list[str]:
    if count < 1:
        raise ValueError("NUM_DRONES must be at least 1")

    return [
        f"udp://0.0.0.0:{DEFAULT_SIM_PORT + index}"
        for index in range(count)
    ]


def generate_drone_uris(
    simulator: bool,
    count: int,
) -> list[str]:
    if simulator:
        return generate_sim_uris(count)

    return generate_physical_uris(count)


def drone_label(index: int) -> str:
    return f"drone {index + 1}"


def test_basic_connection(
    uri: str,
    label: str,
) -> bool:
    """Test basic connection to a single drone URI."""

    print("=" * 60)
    print(f"TEST 1: Basic Connection Test for {label}")
    print("=" * 60)

    try:
        init_drivers()

        print("\n[1/4] Initializing CRTP drivers...")
        print("Drivers initialized")

        print(f"\n[2/4] Connecting to {uri}...")

        with SyncCrazyflie(
            uri,
            cf=Crazyflie(rw_cache="./cache"),
        ) as scf:
            print("Connected successfully!")

            print("\n[3/4] Testing communication...")
            cf = scf.cf

            print("      Reading firmware information...")
            time.sleep(1)

            print("Communication working")

            print("\n[4/4] Testing arming...")

            cf.platform.send_arming_request(True)
            time.sleep(1)
            print("Arming command sent")

            cf.platform.send_arming_request(False)
            print("Disarming command sent")

        print("\n" + "=" * 60)
        print("✓ TEST 1 PASSED - Basic connection works!")
        print("=" * 60)

        return True

    except Exception as exc:
        print("\n✗ TEST 1 FAILED")
        print(f"Error: {exc}")

        return False


def run_motion_sequence(
    scf: SyncCrazyflie,
    label: str,
) -> None:
    """
    Run the flight sequence for one drone.

    This function is executed in parallel for every drone by
    Swarm.parallel_safe().
    """

    cf = scf.cf

    print(f"[{label}] Connected")

    try:
        print(f"[{label}] Arming...")
        cf.platform.send_arming_request(True)
        time.sleep(1)

        print(f"[{label}] Armed")

        print(f"[{label}] Taking off...")

        with MotionCommander(
            scf,
            default_height=0.5,
        ) as mc:
            print(f"[{label}] Took off - hovering at 0.5 m")

            time.sleep(3)

            print(f"[{label}] Moving forward 0.3 m...")

            mc.forward(
                0.3,
                velocity=0.2,
            )

            time.sleep(1)

            print(f"[{label}] Moving back 0.3 m...")

            mc.back(
                0.3,
                velocity=0.2,
            )

            time.sleep(1)

            print(f"[{label}] Returned to starting position")

            time.sleep(2)

            print(f"[{label}] Landing...")

        print(f"[{label}] Landed")

    finally:
        try:
            cf.platform.send_arming_request(False)
            print(f"[{label}] Disarmed")

        except Exception as exc:
            print(
                f"[{label}] WARNING: "
                f"Could not send disarm command: {exc}"
            )


def test_simultaneous_motion_commander(
    uris: list[str],
) -> bool:
    """
    Connect to all drones simultaneously and run the same flight
    sequence on all drones in parallel.
    """

    print("\n" + "=" * 60)
    print("TEST 2: Simultaneous Motion Commander Test")
    print("=" * 60)

    try:
        init_drivers()

        print("\nConnecting to all drones simultaneously...")

        factory = CachedCfFactory(
            rw_cache="./cache"
        )

        args_dict = {
            uri: [drone_label(index)]
            for index, uri in enumerate(uris)
        }

        with Swarm(
            uris,
            factory=factory,
        ) as swarm:
            print(
                f"Connected to all {len(uris)} drones"
            )

            print("\nStarting simultaneous flight...")
            print(
                "All drones will take off, move forward, "
                "move back, and land."
            )

            swarm.parallel_safe(
                run_motion_sequence,
                args_dict=args_dict,
            )

        print("\n" + "=" * 60)
        print(
            "✓ TEST 2 PASSED - "
            "Simultaneous flight test successful!"
        )
        print("=" * 60)

        return True

    except Exception as exc:
        print("\n✗ TEST 2 FAILED")
        print(f"Error: {exc}")

        return False


def check_environment(
    simulator: bool,
    uris: list[str],
) -> bool:
    """Check if environment is set up correctly."""

    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)

    print(
        f"\nMode: "
        f"{'simulator' if simulator else 'physical drones'}"
    )

    print("Generated URIs:")

    for index, uri in enumerate(uris):
        print(
            f"  - {drone_label(index)}: {uri}"
        )

    if len(uris) != len(set(uris)):
        print("✗ ERROR: Duplicate drone URIs detected")
        return False

    if simulator:
        valid = all(
            uri.startswith("udp://0.0.0.0:")
            for uri in uris
        )

        if valid:
            print("URI list looks correct for CrazySim")
        else:
            print("✗ ERROR: Invalid CrazySim URI detected")

        return valid

    valid = all(
        uri.startswith(
            "radio://0/"
            f"{DEFAULT_RADIO_CHANNEL}/"
            f"{DEFAULT_RADIO_DATARATE}/"
            f"{DEFAULT_RADIO_PREFIX}"
        )
        for uri in uris
    )

    if valid:
        print("URI list looks correct for physical drones")
    else:
        print("✗ ERROR: Invalid physical drone URI detected")

    return valid


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CrazySim / Physical Drone Connection Test Suite")
    print("=" * 60)

    # simulator=True:
    #     Uses udp:// URIs for CrazySim.
    #
    # simulator=False:
    #     Uses radio:// URIs for physical Crazyflies.
    simulator = False

    # Number of drones to connect to and test simultaneously.
    drone_count = 2

    uris = generate_drone_uris(
        simulator=simulator,
        count=drone_count,
    )

    if not check_environment(
        simulator=simulator,
        uris=uris,
    ):
        print(
            "\n✗ Environment check failed. "
            "Fix issues and try again."
        )
        sys.exit(1)

    input(
        "\n⏎ Press ENTER when the drones/simulator are ready..."
    )

    # First test every drone individually. This makes it easy to
    # identify a bad URI, radio connection, or individual drone.
    for index, uri in enumerate(uris):
        label = drone_label(index)

        if not test_basic_connection(
            uri=uri,
            label=label,
        ):
            print(
                f"\n✗ Basic connection failed for {label}. "
                "Cannot proceed with flight test."
            )
            sys.exit(1)

    print("\n" + "=" * 60)
    print("INDIVIDUAL CONNECTION TESTS PASSED")
    print("=" * 60)

    print(
        "\nThe next test will connect to ALL drones simultaneously."
    )
    print(
        "They will take off to 0.5 m, move forward 0.3 m, "
        "move back 0.3 m, then land."
    )

    input(
        "\n⏎ Ensure the drones have sufficient separation, "
        "then press ENTER to start..."
    )

    if not test_simultaneous_motion_commander(
        uris=uris,
    ):
        print(
            "\n✗ Simultaneous flight test failed."
        )
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("=" * 60)

    if simulator:
        print(
            "\nYour CrazySim multi-drone setup is working correctly!"
        )
    else:
        print(
            "\nYour physical multi-drone setup is working correctly!"
        )