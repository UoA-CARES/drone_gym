"""Staged hardware checks for flying several Crazyflies with Vicon.

    python drone_gym/tests/multi_drone_lab.py vicon
    python drone_gym/tests/multi_drone_lab.py ground --drones 0 1
    python drone_gym/tests/multi_drone_lab.py hover --drones 0 1
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from drone_gym.drone import Drone
from drone_gym.utils.vicon_position_source import ViconInterface

HOVER_HEIGHT = 1.0
HOVER_SPREAD = 1.0  # drones spaced evenly on x in [-HOVER_SPREAD, HOVER_SPREAD]
PRINT_PERIOD = 0.5


def drone_uri(index: int) -> str:
    return f"radio://0/100/2M/E7E7E7E7{0xE7 + index:02X}"


def hover_targets(count: int) -> list[tuple[float, float, float]]:
    if count == 1:
        return [(0.0, 0.0, HOVER_HEIGHT)]
    step = 2 * HOVER_SPREAD / (count - 1)
    return [(-HOVER_SPREAD + k * step, 0.0, HOVER_HEIGHT) for k in range(count)]


def fmt(position: list[float]) -> str:
    return "[" + ", ".join(f"{v:6.3f}" for v in position) + "]"


def run_vicon(seconds: float) -> bool:
    vicon = ViconInterface()
    threading.Thread(target=vicon.main_loop, daemon=True).start()
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(1.0)
            if not vicon.have_recv_packet:
                print("No Vicon packets yet")
                continue
            with vicon.tracked_object_lock:
                names = sorted(vicon.tracked_object)
            now = time.monotonic()
            for name in names:
                snapshot = vicon.get_position_snapshot(name)
                if snapshot is not None:
                    position, received = snapshot
                    print(f"{name:>24} {fmt(position)}  age {now - received:.2f}s")
            print("-" * 60)
        return vicon.have_recv_packet
    finally:
        vicon.end()


def create_drones(args: argparse.Namespace, drones: dict[str, Drone]) -> bool:
    """Fill drones in place so the caller can clean up a partial set."""
    for index, name in zip(args.drones, args.names):
        print(f"Connecting drone {index}: {drone_uri(index)} / Vicon {name!r} / {args.mode}")
        drone = Drone(
            agent_id=f"drone_{index}",
            uri=drone_uri(index),
            position_tracking_mode=args.mode,
            vicon_object_name=name,
        )
        drones[name] = drone
        if not drone.is_running():
            print(f"{name}: failed to initialise")
            return False
        if not drone.position_ready_event.is_set():
            print(f"{name}: no Vicon position received")
            return False
    return True


def print_positions(drones: dict[str, Drone], seconds: float) -> bool:
    """Return False early if any drone raises an emergency."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for name, drone in drones.items():
            if drone.emergency_event.is_set():
                print(f"{name}: emergency triggered")
                return False
        print("  ".join(f"{name} {fmt(d.get_position())}" for name, d in drones.items()))
        time.sleep(PRINT_PERIOD)
    return True


def wait_for_target(drone: Drone, target: tuple[float, float, float], timeout: float = 2.0) -> bool:
    # set_target_position is queued; confirm it applied before control starts.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with drone.position_lock:
            applied = drone.target_position
            if (applied["x"], applied["y"], applied["z"]) == target:
                return True
        time.sleep(0.05)
    return False


def fly(drones: dict[str, Drone], seconds: float) -> bool:
    # Lowest current x gets the lowest target x, so paths never cross.
    order = sorted(drones, key=lambda name: drones[name].get_position()[0])
    targets = dict(zip(order, hover_targets(len(drones))))

    for drone in drones.values():
        drone.take_off()
    for name, drone in drones.items():
        if not drone.is_flying_event.wait(timeout=15):
            print(f"{name}: take-off not confirmed")
            return False

    # Target before control, else the controller flies every drone to (0, 0, 0).
    for name, drone in drones.items():
        drone.set_target_position(*targets[name])
        if not wait_for_target(drone, targets[name]):
            print(f"{name}: target {targets[name]} not applied")
            return False

    for name, drone in drones.items():
        print(f"{name}: flying to {targets[name]}")
        drone.start_position_control()

    return print_positions(drones, seconds)


def land_and_stop(drones: dict[str, Drone]) -> None:
    for drone in drones.values():
        drone.stop_position_control()
        drone.set_velocity_vector(0.0, 0.0, 0.0)
        drone.land()
    for name, drone in drones.items():
        if not drone.is_landed_event.wait(timeout=15):
            print(f"{name}: landing not confirmed")
    for drone in drones.values():
        drone.stop()


def run_drones(args: argparse.Namespace) -> bool:
    drones: dict[str, Drone] = {}
    try:
        if not create_drones(args, drones):
            return False
        if args.command == "ground":
            return print_positions(drones, args.seconds)
        return fly(drones, args.seconds)
    finally:
        land_and_stop(drones)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    vicon = commands.add_parser("vicon", help="print every Vicon object, no radio")
    vicon.add_argument("--seconds", type=float, default=10.0)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--drones", type=int, nargs="+", required=True, help="drone indices, e.g. 0 1")
    shared.add_argument("--names", nargs="+", help="Vicon object names (default Crzayme_<index>)")
    shared.add_argument("--mode", choices=["source_vicon", "legacy_vicon"], default="source_vicon")
    shared.add_argument("--seconds", type=float, default=20.0)
    commands.add_parser("ground", parents=[shared], help="connect and print positions, no take-off")
    commands.add_parser("hover", parents=[shared], help="take off, hold spaced targets, land")

    args = parser.parse_args(argv)
    if args.command == "vicon":
        return args

    if len(set(args.drones)) != len(args.drones):
        parser.error("--drones must not repeat")
    if any(not 0 <= i <= 0xFF - 0xE7 for i in args.drones):
        parser.error(f"--drones must be in 0..{0xFF - 0xE7}")
    if args.names is None:
        args.names = [f"Crzayme_{i}" for i in args.drones]
    elif len(args.names) != len(args.drones):
        parser.error("--names needs one name per drone")
    if len(set(args.names)) != len(args.names):
        parser.error("--names must not repeat")
    if args.mode == "legacy_vicon" and len(args.drones) > 1:
        parser.error("legacy_vicon supports one drone only")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        ok = run_vicon(args.seconds) if args.command == "vicon" else run_drones(args)
    except KeyboardInterrupt:
        print("Interrupted")
        return 130
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
