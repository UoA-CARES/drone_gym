"""Take off both Crazyflies together, hover in place 10 s, land. Only cflib: no Vicon, no drone_gym.

Ctrl+C does not land them cleanly (links close mid-air); let it finish or use the kill switch.

    python drone_gym/tests/swarm_hover_test.py
"""
import time

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.positioning.motion_commander import MotionCommander

URIS = ["radio://0/100/2M/E7E7E7E7E7", "radio://0/100/2M/E7E7E7E7E8"]


def prepare(scf):
    cf = scf.cf
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2)
    cf.platform.send_arming_request(True)
    time.sleep(1)


def hover(scf):
    with MotionCommander(scf, default_height=0.5):  # takes off here, lands on exit
        time.sleep(10)
    scf.cf.platform.send_arming_request(False)


cflib.crtp.init_drivers()
with Swarm(URIS, factory=CachedCfFactory(rw_cache="./cache")) as swarm:
    swarm.parallel_safe(prepare)
    swarm.parallel_safe(hover)
