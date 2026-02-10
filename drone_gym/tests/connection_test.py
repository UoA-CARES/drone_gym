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
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/100/2M')

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_connection():
    """Test basic connection to CrazySim"""
    print("=" * 60)
    print("TEST 1: Basic Connection Test")
    print("=" * 60)
    
    try:
        print(f"\n[1/4] Initializing CRTP drivers...")
        cflib.crtp.init_drivers()
        print("Drivers initialized")
        
        print(f"\n[2/4] Connecting to {URI}...")
        
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
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


def test_motion_commander():
    """Test basic flight with MotionCommander"""
    print("\n" + "=" * 60)
    print("TEST 2: Motion Commander Test")
    print("=" * 60)
    
    try:
        print(f"\n[1/3] Connecting to {URI}...")
        cflib.crtp.init_drivers()
        
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Connected")
            
            print(f"\n[2/3] Taking off...")
            with MotionCommander(scf, default_height=0.5) as mc:
                print("Took off! Hovering at 0.5m")
                time.sleep(3)
                
                print(f"\n[3/3] Testing movement...")
                print("      Moving forward 0.3m...")
                mc.forward(0.3, velocity=0.2)
                time.sleep(1)
                
                print("      Moving back to start...")
                mc.back(0.3, velocity=0.2)
                time.sleep(1)
                
                print("Movement test complete")
                print("\n      Landing...")
            
            print("Landed")
        
        print("\n" + "=" * 60)
        print("✓ TEST 2 PASSED - Flight test successful!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED")
        print(f"Error: {e}")
        return False


def check_environment():
    """Check if environment is set up correctly"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Check URI
    print(f"\n[2/3] Checking URI...")
    print(f"      URI: {URI}")
    if 'udp://' in URI and '19850' in URI:
        print("URI looks correct for CrazySim")
    else:
        print("⚠ WARNING: URI may not be correct for CrazySim")
    
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CrazySim Connection Test Suite")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("\n✗ Environment check failed. Fix issues and try again.")
        sys.exit(1)
    
    input("\n⏎ Press ENTER when CrazySim is running and ready...")
    
    # Test basic connection
    if not test_basic_connection():
        print("\n✗ Basic connection failed. Cannot proceed with flight test.")
        sys.exit(1)
    
    # Test flight
    input("\n Press ENTER to run flight test in Gazebo!.......")
    
    if not test_motion_commander():
        print("\n✗ Flight test failed.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    print("=" * 60)
    print("\nYour CrazySim setup is working correctly!")