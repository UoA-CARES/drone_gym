from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

from drone_gym.utils.position_source import PositionSample
from drone_gym.utils.vicon_connection_class import ViconInterface as LegacyViconInterface
from drone_gym.utils.vicon_position_source import ViconPositionSource, ViconProvider


@dataclass
class DroneStats:
    sample_count: int = 0
    last_position: tuple[float, float, float] | None = None


def generate_object_names(count: int, prefix: str) -> list[str]:
    if count < 1:
        raise ValueError("num_drones must be at least 1")
    return [f"{prefix}{index}" for index in range(count)]


def run_legacy_vicon_test(
    object_names: list[str],
    udp_ip: str,
    udp_port: int,
    test_seconds: float,
    poll_period: float,
) -> dict[str, DroneStats]:
    stats = {name: DroneStats() for name in object_names}

    vicon = LegacyViconInterface(udp_ip=udp_ip, udp_port=udp_port)
    vicon_thread = threading.Thread(
        target=vicon.main_loop,
        name="legacy-vicon",
        daemon=True,
    )
    vicon_thread.start()

    end_time = time.monotonic() + test_seconds

    try:
        while time.monotonic() < end_time:
            for name in object_names:
                position = vicon.getPos(name)

                if position is None:
                    continue

                stats[name].sample_count += 1
                stats[name].last_position = (
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                )

            time.sleep(poll_period)

    finally:
        vicon.end()
        vicon_thread.join(timeout=2.0)

    return stats


def print_tracked_objects(provider: ViconProvider) -> None:
    interface = provider._interface

    if interface is None:
        print("\nNo Vicon data received")
        return

    with interface.tracked_object_lock:
        tracked_objects = {
            name: list(data)
            for name, data in interface.tracked_object.items()
        }

    print("\n" + "-" * 72)
    print("VICON TRACKED OBJECTS")
    print("-" * 72)

    if not tracked_objects:
        print("No tracked objects received")
        return

    for name, data in tracked_objects.items():
        print(
            f"{name}: "
            f"pos=({data[0]:.3f}, {data[1]:.3f}, {data[2]:.3f}), "
            f"rot=({data[3]:.3f}, {data[4]:.3f}, {data[5]:.3f}), "
            f"vel=({data[6]:.3f}, {data[7]:.3f}, {data[8]:.3f}), "
            f"angular_vel=({data[9]:.3f}, {data[10]:.3f}, {data[11]:.3f})"
        )


def run_position_source_test(
    object_names: list[str],
    udp_ip: str,
    udp_port: int,
    test_seconds: float,
    poll_period: float,
) -> dict[str, DroneStats]:
    stats = {name: DroneStats() for name in object_names}
    stats_lock = threading.Lock()

    provider = ViconProvider(
        udp_ip=udp_ip,
        udp_port=udp_port,
    )

    sources: list[ViconPositionSource] = []

    def make_callback(object_name: str):
        def callback(sample: PositionSample) -> None:
            with stats_lock:
                stats[object_name].sample_count += 1
                stats[object_name].last_position = (
                    float(sample.x),
                    float(sample.y),
                    float(sample.z),
                )

        return callback

    try:
        for name in object_names:
            source = ViconPositionSource(
                object_name=name,
                provider=provider,
                poll_period=poll_period,
                label=name,
            )

            source.start(make_callback(name))
            sources.append(source)

        end_time = time.monotonic() + test_seconds

        while time.monotonic() < end_time:
            print_tracked_objects(provider)
            time.sleep(1.0)

    finally:
        for source in sources:
            source.stop()

    return stats


def print_summary(stats: dict[str, DroneStats]) -> None:
    print("\n" + "=" * 72)
    print("VICON SAMPLE SUMMARY")
    print("=" * 72)

    for object_name, object_stats in stats.items():
        print(
            f"{object_name}: "
            f"samples={object_stats.sample_count}, "
            f"last_position={object_stats.last_position}"
        )


def all_drones_received_samples(
    stats: dict[str, DroneStats],
    minimum_samples_per_drone: int,
) -> bool:
    return all(
        object_stats.sample_count >= minimum_samples_per_drone
        for object_stats in stats.values()
    )


if __name__ == "__main__":
    # Choose which implementation to test:
    #   "legacy" -> drone_gym.utils.vicon_connection_class.ViconInterface
    #   "source" -> drone_gym.utils.vicon_position_source.ViconPositionSource
    backend: Literal["legacy", "source"] = "legacy"

    # Number of tracked drones/objects to verify.
    num_drones = 2

    # Legacy backend is intentionally limited to one hardcoded object.
    legacy_object_name = "Crzayme"

    # Expected Vicon object naming format.
    object_name_prefix = "Crzayme_"

    # Vicon UDP bind settings.
    udp_ip = "0.0.0.0"
    udp_port = 51001

    # How long to listen for data and how often to poll.
    test_seconds = 10.0
    poll_period = 1.0 / 60.0

    # Test passes only if every drone has at least this many samples.
    minimum_samples_per_drone = 1

    if backend == "legacy":
        object_names = [legacy_object_name]
    else:
        object_names = generate_object_names(
            count=num_drones,
            prefix=object_name_prefix,
        )

    print("\n" + "=" * 72)
    print("Vicon Position Source Integration Test")
    print("=" * 72)
    print(f"Backend: {backend}")
    print(f"Objects: {object_names}")
    print(f"UDP: {udp_ip}:{udp_port}")
    print(f"Duration: {test_seconds}s")

    if backend == "legacy":
        stats = run_legacy_vicon_test(
            object_names=object_names,
            udp_ip=udp_ip,
            udp_port=udp_port,
            test_seconds=test_seconds,
            poll_period=poll_period,
        )
    elif backend == "source":
        stats = run_position_source_test(
            object_names=object_names,
            udp_ip=udp_ip,
            udp_port=udp_port,
            test_seconds=test_seconds,
            poll_period=poll_period,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    print_summary(stats)

    if all_drones_received_samples(stats, minimum_samples_per_drone):
        print("\nPASS: all drones received Vicon samples")
        raise SystemExit(0)

    print("\nFAIL: one or more drones did not receive enough Vicon samples")
    raise SystemExit(1)