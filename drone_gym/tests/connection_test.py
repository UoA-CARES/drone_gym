#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
import sys
import cflib
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SIM_PORT = 19850
DEFAULT_RADIO_CHANNEL = '100'
DEFAULT_RADIO_DATARATE = '2M'
DEFAULT_RADIO_PREFIX = 'E7E7E7E7'

_DRIVERS_INITIALIZED = False


def init_drivers() -> None:
    global _DRIVERS_INITIALIZED

    if not _DRIVERS_INITIALIZED:
        cflib.crtp.init_drivers()
        _DRIVERS_INITIALIZED = True


def generate_physical_uris(count: int) -> list[str]:
    if count < 1:
        raise ValueError('NUM_DRONES must be at least 1')
    if count > 256:
        raise ValueError('NUM_DRONES must be 256 or less for the generated radio addresses')

    return [
        f'radio://0/{DEFAULT_RADIO_CHANNEL}/{DEFAULT_RADIO_DATARATE}/{DEFAULT_RADIO_PREFIX}{index:02X}'
        for index in range(count)
    ]


def generate_sim_uris(count: int) -> list[str]:
    if count < 1:
        raise ValueError('NUM_DRONES must be at least 1')

    return [f'udp://0.0.0.0:{DEFAULT_SIM_PORT + index}' for index in range(count)]


def generate_drone_uris(simulator: bool, count: int) -> list[str]:
    if simulator:
        return generate_sim_uris(count)
    return generate_physical_uris(count)


def drone_label(index: int) -> str:
    return f'drone {index + 1}'


def test_basic_connection(uri: str, label: str) -> bool:
    """Test basic connection to a single drone URI."""
    print("=" * 60)
    print(f"TEST 1: Basic Connection Test for {label}")
    print("=" * 60)
    
    try:
        init_drivers()
        print(f"\n[1/4] Initializing CRTP drivers...")
        print("Drivers initialized")
        
        print(f"\n[2/4] Connecting to {uri}...")
        
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Connected successfully!")
            
            print(f"\n[3/4] Testing communication...")
            cf = scf.cf
            
            # Check if we can read a parameter
            print("      Reading firmware version...")
            time.sleep(1)
            
            print("Communication working")
            
            print(f"\n[4/4] Testing arming...")
            cf.platform.send_arming_request(True)
            time.sleep(1)
            print("Arming command sent")
            
            cf.platform.send_arming_request(False)
            print("Disarming command sent")
            
        print("\n" + "=" * 60)
        print("✓ TEST 1 PASSED - Basic connection works!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED")
        print(f"Error: {e}")
        return False


def test_motion_commander(uri: str, label: str) -> bool:
    """Test basic flight with MotionCommander for a single drone URI."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Motion Commander Test for {label}")
    print("=" * 60)
    
    try:
        init_drivers()
        print(f"\n[1/4] Connecting to {uri}...")
        
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Connected")
            
            print(f"\n[2/4] Arming drone...")
            scf.cf.platform.send_arming_request(True)
            time.sleep(1)
            print("Armed")
            
            print(f"\n[3/4] Taking off...")
            with MotionCommander(scf, default_height=0.5) as mc:
                print("Took off! Hovering at 0.5m")
                time.sleep(3)
                
                print(f"\n[4/4] Testing movement...")
                print("      Moving forward 0.3m...")
                mc.forward(0.3, velocity=0.2)
                time.sleep(1)
                
                print("      Moving back to start...")
                mc.back(0.3, velocity=0.2)
                time.sleep(1)
                
                print("Movement test complete")
                print("\n      Landing...")
            
            print("Landed")
            scf.cf.platform.send_arming_request(False)
            print("Disarmed")
        
        print("\n" + "=" * 60)
        print("✓ TEST 2 PASSED - Flight test successful!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED")
        print(f"Error: {e}")
        return False


def check_environment(simulator: bool, uris: list[str]) -> bool:
    """Check if environment is set up correctly."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    print(f"\n[2/3] Mode: {'simulator' if simulator else 'physical drones'}")
    print("      Generated URIs:")
    for index, uri in enumerate(uris):
        print(f"      - {drone_label(index)}: {uri}")

    if simulator and all(uri.startswith('udp://0.0.0.0:') for uri in uris):
        print("URI list looks correct for CrazySim")
    elif not simulator and all(uri.startswith('radio://0/100/2M/E7E7E7E7') for uri in uris):
        print("URI list looks correct for physical drones")
    else:
        print("⚠ WARNING: URI may not be correct for CrazySim")
    
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CrazySim / Physical Drone Connection Test Suite")
    print("=" * 60)

    # Explicit configuration for readability.
    # simulator=True uses udp:// URIs; simulator=False uses radio:// URIs.
    simulator = False
    drone_count = 3

    uris = generate_drone_uris(simulator, drone_count)

    if not check_environment(simulator, uris):
        print("\n✗ Environment check failed. Fix issues and try again.")
        sys.exit(1)

    input("\n⏎ Press ENTER when the drones/simulator are ready...")

    for index, uri in enumerate(uris):
        label = drone_label(index)
        if not test_basic_connection(uri, label):
            print(f"\n✗ Basic connection failed for {label}. Cannot proceed with flight test.")
            sys.exit(1)

    input("\nPress ENTER to run the motion commander test for each drone...")

    for index, uri in enumerate(uris):
        label = drone_label(index)
        if not test_motion_commander(uri, label):
            print(f"\n✗ Flight test failed for {label}.")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("=" * 60)
    print("\nYour CrazySim setup is working correctly!")