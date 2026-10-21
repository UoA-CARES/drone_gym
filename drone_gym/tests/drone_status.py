"""Print one Crazyflie's battery voltage and firmware supervisor state (why it won't arm/fly).

    python drone_gym/tests/drone_status.py E7E7E7E7E8
"""
import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

# Bit order of supervisor.info, as in cflib's Supervisor.STATES
STATES = ["Can be armed", "Is armed", "Is auto armed", "Can fly", "Is flying", "Is tumbled",
          "Is locked", "Is crashed", "HL control is active", "Finished HL trajectory",
          "HL control is disabled"]

cflib.crtp.init_drivers()
with SyncCrazyflie(f"radio://0/100/2M/{sys.argv[1]}", cf=Crazyflie(rw_cache="./cache")) as scf:
    config = LogConfig("status", period_in_ms=100)
    config.add_variable("pm.vbat", "float")
    config.add_variable("supervisor.info", "uint16_t")
    with SyncLogger(scf, config) as logger:
        _, data, _ = next(iter(logger))

bits = int(data["supervisor.info"])
print(f"battery {data['pm.vbat']:.2f} V")
print("supervisor:", [name for i, name in enumerate(STATES) if bits & (1 << i)])
