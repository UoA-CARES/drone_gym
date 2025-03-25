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
deck_attached_event = Event()


DEFAULT_HEIGHT = 0.5
BOX_LIMIT = 0.5


def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print("Flow deck is attached! - Woohoo")
    else:
        print("Deck is NOT attached!")

def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()

if __name__ == "__main__":
    cflib.crtp.init_drivers()
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)

        scf.cf.add_update_callback(group="deck", name="bcFlow2", cb=param_deck_flow)
        
        time.sleep(1.0)

        take_off_simple(scf)

