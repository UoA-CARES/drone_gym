"""Hover one Crazyflie in place for 10 s using only cflib (no Vicon, no drone_gym control).

If the drone drifts here, the flow deck / floor / lighting is the problem, not drone_gym.

    python drone_gym/tests/flow_hover_test.py E7E7E7E7E7
"""
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

address = sys.argv[1]
cflib.crtp.init_drivers()
with SyncCrazyflie(f"radio://0/100/2M/{address}", cf=Crazyflie(rw_cache=f"./cache/flow_{address}/")) as scf:
    cf = scf.cf
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2)
    cf.platform.send_arming_request(True)
    time.sleep(1)
    with MotionCommander(scf, default_height=0.5):  # takes off; lands on exit or Ctrl+C
        time.sleep(10)
    cf.platform.send_arming_request(False)
