# Multi-Drone Setup and Running MARL on Real Drones

This guide covers flying two Crazyflie drones together with Vicon: one-time setup, quick tests, and running a trained `marl_tag` policy on the real drones.

**📚 Related Documentation:**
- [← Back to Main README](../README.md)
- [Vicon System Setup](VICON_SETUP.md) - Configure motion capture system
- [Hardware Setup Guide](HARDWARE_SETUP.md) - Drone positioning, battery management, and troubleshooting

## Prerequisites

⚠️ **Important**: Complete the [Vicon System Setup](VICON_SETUP.md) and the [Hardware Setup Guide](HARDWARE_SETUP.md) pre-flight checklist first.

You need:
- **2 × Crazyflie 2.x**, each with a flow deck and Vicon markers. Use a **different marker layout on each drone**, or Vicon can mix them up.
- **1 × Crazyradio**. One dongle is enough for both drones.
- **Charged batteries** (> 4.1V, see [Battery Management](HARDWARE_SETUP.md#battery-management)).
- **Marker boards under the whole flight area**, not just the take-off spot. The flow deck loses track over plain floor and the drone drifts away.
- **A second person on the kill switch.**

All commands are run from the `drone_gym` repository root, in the `drone_gym` virtual environment.

## One-Time Setup

### Drone Addresses and Vicon Names

Each drone needs a radio address and a Vicon object name that match the code:

| Drone | Radio URI | Vicon object |
|-------|-----------|--------------|
| 0 | `radio://0/100/2M/E7E7E7E700` | `Crzayme_0` |
| 1 | `radio://0/100/2M/E7E7E7E701` | `Crzayme_1` |

> **Note**: The names are spelled `Crzayme` in the code (`marl_drone_environment.py`). The Vicon objects must match exactly. The `Crzayme_0` object must be on the drone with address `...00`.

The radio address is stored **on the drone**, not in the code. To set it over USB:

1. Plug **only one drone** into the computer with a micro-USB cable and switch it on
2. Close cfclient
3. Run the script below. Change the address on its first line to `E7E7E7E701` for drone 1
4. Unplug the cable and switch the drone off and on

<details>
<summary><b>Radio address script (click to expand)</b></summary>

```bash
perl -pe 's/\xE2\x80\x8B|\xEF\xBB\xBF//g; s/\xC2\xA0/ /g' <<'EOF' | python - E7E7E7E700
import sys, threading, cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

ADDRESS = int(sys.argv[1], 16)

def show(eeprom, label):
    done = threading.Event()
    eeprom.update(lambda _: done.set())
    done.wait(5)
    e = eeprom.elements
    print(label, "channel", e.get("radio_channel"), "speed", e.get("radio_speed"), "address", hex(e.get("radio_address", 0)))
    return e

cflib.crtp.init_drivers()
with SyncCrazyflie("usb://0", cf=Crazyflie()) as scf:
    eeprom = scf.cf.mem.get_mems(MemoryElement.TYPE_I2C)[0]
    e = show(eeprom, "before:")
    if (e.get("radio_channel"), e.get("radio_speed"), e.get("radio_address")) == (100, 2, ADDRESS):
        print("already correct, nothing written")
    else:
        e.update(version=1, radio_channel=100, radio_speed=2, radio_address=ADDRESS)
        e.setdefault("pitch_trim", 0.0)
        e.setdefault("roll_trim", 0.0)
        done = threading.Event()
        eeprom.write_data(lambda *_: done.set())
        print("written" if done.wait(5) else "WRITE TIMED OUT")
        show(eeprom, "after: ")
EOF
```

It should end with `after:  channel 100 speed 2 address 0xe7e7e7e700`. Speed `2` means 2M.

</details>

You can also set the address in cfclient: **Connect → Configure 2.x**, then set Channel `100`, Bandwidth `2M` and the Address, and click **Write**.

### Vicon Settings for Two Drones

In Vicon Tracker, under **System → Local Vicon System → UDP OBJECT STREAM** (see [Vicon System Setup](VICON_SETUP.md#3-configure-udp-object-stream)):

| Setting | Value |
|---------|-------|
| Enabled | ✅ ticked |
| IP Address | `192.168.10.3`. This is the Vicon PC's own network adapter, chosen from the drop-down |
| Port | `51001` |
| Object Per Port | ❌ **unticked**. If it's ticked, drone 1 is sent to a different port and never seen |

In the **OBJECT** tab, make sure **both** `Crzayme_0` and `Crzayme_1` are checked.

## Before Every Session

- [ ] Hardware pre-flight checklist done (see [Pre-Flight Checklist](HARDWARE_SETUP.md#pre-flight-checklist))
- [ ] Marker boards cover the whole flight area
- [ ] Both drones' fronts face the **positive x-direction** (see [Coordinate System](HARDWARE_SETUP.md#coordinate-system))
- [ ] Drones switched on where they'll take off, then left **level and untouched for ~15 seconds**
- [ ] cfclient closed (it holds the radio dongle)
- [ ] All USB cables unplugged from the drones
- [ ] Someone on the kill switch

## Two-Drone Tests

Run these in order. Each one should pass before you go on.

### 1. Radio Check

Switch on **only drone 0**:

```bash
python -c "import cflib.crtp as c; c.init_drivers(); print(c.scan_interfaces(0xE7E7E7E700))"
```

Then switch on **only drone 1**:

```bash
python -c "import cflib.crtp as c; c.init_drivers(); print(c.scan_interfaces(0xE7E7E7E701))"
```

✅ **Pass**: each prints its own URI, e.g. `[['radio://0/100/2M/E7E7E7E700', '']]`

❌ **`[]`**: the drone is off or has the wrong address. See [Drone Addresses and Vicon Names](#drone-addresses-and-vicon-names).

### 2. Vicon Check

```bash
python -c 'import threading,time; from drone_gym.utils.vicon_position_source import ViconInterface as V; v=V(); threading.Thread(target=v.main_loop,daemon=True).start(); time.sleep(3); print({n: [round(x,2) for x in v.getPos(n)[:3]] for n in list(v.tracked_object)} or "NO VICON DATA"); v.end()'
```

✅ **Pass**: `{'Crzayme_0': [x, y, z], 'Crzayme_1': [x, y, z]}`, with positions matching where the drones sit. Lift each drone by hand and run it again: that drone's own z value should go up.

❌ **`NO VICON DATA`**: check the [Vicon Settings for Two Drones](#vicon-settings-for-two-drones).

### 3. Both Drones Hover

Both drones take off to 0.5 m, hover in place for 10 seconds and land. This test uses only the Crazyflie library, so Vicon doesn't steer the drones. Place them **at least 1 m apart** on the boards.

⚠️ **Do not press Ctrl+C during this test.** It cuts the radio and the drones drop. Use the kill switch instead.

```bash
perl -pe 's/\xE2\x80\x8B|\xEF\xBB\xBF//g; s/\xC2\xA0/ /g' <<'EOF' | python -
import time
import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from cflib.positioning.motion_commander import MotionCommander

URIS = ["radio://0/100/2M/E7E7E7E700", "radio://0/100/2M/E7E7E7E701"]

def prepare(scf):
    cf = scf.cf
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2)
    cf.platform.send_arming_request(True)
    time.sleep(1)

def hover(scf):
    with MotionCommander(scf, default_height=0.5):
        time.sleep(10)
    scf.cf.platform.send_arming_request(False)

cflib.crtp.init_drivers()
with Swarm(URIS, factory=CachedCfFactory(rw_cache="./cache")) as swarm:
    swarm.parallel_safe(prepare)
    swarm.parallel_safe(hover)
EOF
```

✅ **Pass**: both drones lift, stay roughly in place, and land.

❌ **A drone doesn't lift**: switch it off and on, wait ~15 seconds, and run the test again.

## Running `marl_tag` with Two Drones

### 1. Apply the Code Fixes

⚠️ **Important**: In `drone_gym/drone.py`, the real-drone position mode is hard-coded to `legacy_vicon`. With two drones, the second one crashes with `OSError: [Errno 98] Address already in use`. Change it to `source_vicon`:

```bash
sed -i 's/self.position_tracking_mode: str = "legacy_vicon"/self.position_tracking_mode: str = "source_vicon"/' drone_gym/drone.py
```

Check it worked. It should show `"source_vicon"`:

```bash
grep -n 'position_tracking_mode: str' drone_gym/drone.py
```

**Only on `main` (commit `704a13a`)**: the height limits (task 2.5 m, emergency stop 3.0 m) are above the lab ceiling. Lower them:

```bash
sed -i 's/"z_min": 0.1, "z_max": 3.0}/"z_min": 0.1, "z_max": 2.0}/' drone_gym/tasks/intercept_target/marl_tag.py
```

```bash
sed -i 's/z_hi = self.z_max - self.z_margin/z_hi = min(self.z_max - self.z_margin, 1.8)/' drone_gym/tasks/intercept_target/marl_tag.py
```

Check which commit you're on with `git log -1 --oneline`. The `fixing_graph_bug` branch (`aa35c40`) already has lower limits (1.5 m), so skip those two commands there.

> **Note**: `git pull` and `git checkout` undo these edits. Re-check them before every session.

### 2. Point the Trained Run at the Real Drones

Set `RUN` to your trained run folder:

```bash
RUN=/path/to/your/run_folder
```

Switch it from the simulator to the real drones:

```bash
sed -i 's/"use_simulator": *1/"use_simulator": 0/' "$RUN/env_config.json"
```

Check the settings. It should show `"use_simulator": 0`, `"num_agents": 2` and `"task": "marl_tag"`:

```bash
grep -o '"use_simulator": *[01]\|"num_agents": *[0-9]*\|"task": *"[a-z_]*"' "$RUN/env_config.json"
```

> **Note**: For real drones only `task`, `use_simulator` and `num_agents` are read from `env_config.json`. The arena size, height limits and speeds come from the `drone_gym` code you have checked out. Fly the policy on the same `drone_gym` version it was trained on.

### 3. Run One Episode

1. **Switch both drones off and on**, then leave them still for ~15 seconds
2. **Run the policy**, then press ENTER at the prompt:

```bash
cares-rl test --data_path "$RUN" --eval_seed 20 --episodes 1
```

Use the seed the run was trained with for `--eval_seed`.

### What Happens During a Run

| Stage | What you see | What to do |
|-------|--------------|------------|
| Connecting | `[runner_0]` and `[interceptor_0]` setup messages | Nothing |
| Reset | Both drones take off and fly to their start positions | Watch for drones flying toward each other |
| Episode | The trained policy flies both drones, 0.5 s per step | Stay on the kill switch |
| Battery low | `Please replace the batteries for the above drones and confirm when done. (y/n)` | Swap the batteries, then type `y` |
| Reset failed | `Please reposition and power on the failed drones, then confirm when done. (y/n)` | Reposition the drones, then type `y` |

⚠️ **Ctrl+C does not land the drones** during `cares-rl test`. They keep flying their last command until they cross the safety boundary. **The kill switch is the only reliable stop.**

### After the Session

Undo the local code edits before your next `git pull`:

```bash
git checkout -- drone_gym/drone.py drone_gym/tasks/intercept_target/marl_tag.py
```

## Safety Reference

| Action | Does it land the drones? |
|--------|--------------------------|
| Ctrl+C during the two-drone hover test | ❌ No, the drones drop |
| Ctrl+C during `cares-rl test` | ❌ No, they keep flying until the safety boundary |
| Drone crosses the safety boundary | ✅ That drone lands |
| Kill switch | ✅ Always |

| Safety limit (real drones) | `main` (`704a13a`) | `fixing_graph_bug` (`aa35c40`) |
|----------------------------|--------------------|--------------------------------|
| x, y emergency boundary | ±2.5 m | ±2.5 m |
| Height emergency boundary | 3.0 m | 1.5 m |
| Task height range | 0.4 – 2.5 m | 0.6 – 1.1 m |

> **Note**: The minimum height in the boundary is **not** checked by the code. Only the maximum height and the x/y limits trigger an emergency landing.

## Troubleshooting

### Quick Reference Guide

| Problem | Symptoms | Cause | Solution |
|---------|----------|-------|----------|
| **Second drone crashes** | `OSError: [Errno 98] Address already in use` | `drone.py` is in `legacy_vicon` mode | Apply the [code fix](#1-apply-the-code-fixes) again. A `git pull` or `git checkout` undoes it |
| **Radio scan finds nothing** | Scan prints `[]` | Drone off, or wrong address | Check the address with the [radio address script](#drone-addresses-and-vicon-names) |
| **Drone fails to connect** | `Hardware failed to initialize in time` | The first connection downloads the drone's settings and is slow | Run it again from the same folder |
| **No Vicon positions** | `NO VICON DATA` or `no Vicon position received` | UDP stream settings | Check [Vicon Settings for Two Drones](#vicon-settings-for-two-drones) |
| **Logs say take-off, drone stays down** | `Take-off successful` but motors don't spin | Drone refused to arm (just switched on, moved, or locked after a crash) | Switch it off and on, and leave it still for ~15 s |
| **Drones spiral away** | Drones curve around their targets instead of flying straight | Drones not facing +x | Turn both fronts to the positive x-direction |
| **Drone drifts or flies off** | Slides away at speed, even while landing | Flow deck lost track over plain floor | Put marker boards under the whole flight area |
| **Drone climbs to the ceiling** | Drone keeps climbing | Height limits above the ceiling (`main`) | Apply the [height edits](#1-apply-the-code-fixes) |
| **Positions swap between drones** | One drone briefly shows the other's position | Both drones have the same marker layout | Move one marker on one drone and recreate its Vicon object |
| **Pasted command fails** | `invalid non-printable character U+200B` | Invisible characters copied from a web page or chat | Use the blocks above (the first line strips them), or retype the command |
| **Vicon check fails** | `Address already in use` | Another drone script is still running | Close it and try again |

**First time setup?** Complete the [Vicon System Setup](VICON_SETUP.md) and [Hardware Setup Guide](HARDWARE_SETUP.md) first.
