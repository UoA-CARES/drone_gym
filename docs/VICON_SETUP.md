# Vicon Motion Capture System Setup

This guide covers how to connect and configure the Vicon motion capture system for use with the CARES Drone Gym.

**📚 Related Documentation:**
- [← Back to Main README](../README.md)
- [Hardware Setup Guide](HARDWARE_SETUP.md) - Drone positioning and troubleshooting

## Prerequisites

⚠️ **Important**: Before you start, make sure you are inducted in the Motion Capture Lab!


⚠️ **Important**: Confirm that Vicon tracking is working before attempting to fly. Each drone must receive valid and sufficiently recent position measurements. Missing or stale position data may prevent the drone from becoming ready or trigger safety-related behaviour.

## Setup Steps

### 1. Set Up Hardware

Click on the two buttons shown in the picture below. **Both top and bottom should be on**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/721651c8-8104-4b07-b570-9dceae6f8fc1" width="400" alt="Vicon cameras on/off buttons">
</p>

If using a separate device to control the drones, connect the device to the local network. The easiest way is to use an ethernet cable to connect directly to the router (ask the technician for a cable).

<p align="center">
  <img src="https://github.com/user-attachments/assets/cde34acd-26e2-4846-a7b4-b5de495cb2fd" width="400" alt="Connecting ethernet cable to router">
  <img src="https://github.com/user-attachments/assets/fa3b6788-31bc-41c0-92be-9cab619794c9" width="400" alt="Connecting ethernet cable to device">
</p>

### 2. Open Vicon Tracker Application

1. Launch the **Vicon Tracker app** (the green one)
2. Navigate to the **Systems** tab
3. Click on the **"Local Vicon System"** tab
4. Press **"SHOW ADVANCED"**

<div align="center">
  <img src="https://github.com/user-attachments/assets/a9cbd0de-4a1a-4e6b-9bcf-9ccdb84f2231" width="400" alt="Vicon Tracker app Systems tab">
</div>

### 3. Configure UDP Object Stream

1. Scroll down to the **"UDP OBJECT STREAM"** section
2. Enable the UDP stream
3. Change the IP address to: **`192.168.10.3`** (used to be `10.104.144.214`)
4. Change the port to: **`51001`**

<div align="center">
  <img src="https://github.com/user-attachments/assets/edff9550-3619-4c04-8682-79550698886c" width="400" alt="UDP OBJECT STREAM configuration">
</div>


### 4. Enable Crazyflie Object Tracking

1. Navigate to the **OBJECT** tab in Vicon Tracker.
2. Ensure each Crazyflie has a corresponding tracked object.
3. Enable tracking for all Crazyflie objects that will be used.

The default object naming convention in `drone_gym` is:

- `Crzayme_0` — First drone
- `Crzayme_1` — Second drone
- And so on.

The object names must match those configured in `drone_gym`.

Both `DroneEnvironment` (SARL) and `MarlDroneEnvironment` (MARL) automatically assign Vicon object names based on the order of their configured agents.

These mappings are stored in `self.vicon_object_names` and passed to each drone's `ViconPositionSource`.

If the object names in Vicon Tracker differ, update the mappings in the corresponding environment.


<div align="center">
  <img src="https://github.com/user-attachments/assets/d97d3e0d-b75d-451b-9609-e2a5ca4da27f" width="400" alt="Crazyflie object enabled in OBJECT tab">
</div>

## Useful Vicon Shortcuts

| Action | Control |
|--------|---------|
| **Zoom in/out** | Press and hold the right mouse button, then drag |
| **Pan** | Press down on the scroll wheel and move the mouse |

## Coordinate System

The Vicon system uses the following coordinate frame:
- **X-axis (red)**: Right direction (drone "Front" should face this direction)
- **Y-axis (green)**: Forward direction (towards the window)
- **Z-axis**: Vertical (up)

See the [Hardware Setup Guide](HARDWARE_SETUP.md#drone-positioning-and-coordinate-system) for more details on drone positioning.

## Troubleshooting

### Position Not Updating

If a drone does not receive its first position measurement or its position data becomes stale, check the following:

1. Check that you're on the correct network
2. Verify that the Vicon cameras are powered on and tracking correctly.
3. Confirm that the UDP Object Stream is enabled.
4. Check that the UDP destination IP matches the computer running `drone_gym`.
5. Confirm that UDP port `51001` is configured correctly and is not blocked by a firewall.
6. Verify that the tracked object's name matches the `object_name` configured in `ViconPositionSource`.
7. Ensure that the tracked object is enabled and visible to the Vicon cameras.
8. Check the console output for Vicon receiver, position-source, or position-readiness errors.

The new `ViconPositionSource` publishes position samples only when a new Vicon measurement is available.

If Vicon stops delivering measurements, the last known position is not continuously republished as new data. Position freshness monitoring can detect when measurements become stale.


### Object Not Detected
If the Vicon system doesn't detect the drone:
1. Make sure the reflective markers are clean and properly attached
2. Ensure there's sufficient lighting in the capture volume
3. Recalibrate the object in Vicon Tracker if necessary

## Technical Details

Vicon position tracking is implemented in:

`drone_gym/utils/vicon_position_source.py`

The implementation consists of three main classes:

### ViconInterface

- Creates and manages the UDP socket used to receive Vicon packets.
- Listens on `0.0.0.0:51001` by default.
- Parses incoming Vicon packets containing tracked-object measurements.
- Converts position measurements from millimetres to metres.
- Stores the latest position, orientation, linear velocity, angular velocity, and reception timestamp for each tracked object.
- Calculates linear and angular velocities using differences between consecutive measurements.
- Uses `time.monotonic()` for reception timestamps.

Each tracked object stores the following data:

| Index | Data |
|---|---|
| 0–2 | Position (X, Y, Z), metres |
| 3–5 | Orientation (roll, pitch, yaw), radians |
| 6–8 | Linear velocity (X, Y, Z), m/s |
| 9–11 | Angular velocity (roll, pitch, yaw rates), rad/s |
| 12 | Reception timestamp (`time.monotonic()`) |

The `get_position_snapshot()` method retrieves only the position and reception timestamp needed by `ViconPositionSource`. The additional orientation and velocity values remain stored within `ViconInterface`.

### ViconProvider

- Manages a shared `ViconInterface` instance.
- Starts the Vicon receiver when the first position source requests it.
- Allows multiple drones to use the same UDP connection.
- Tracks active subscribers.
- Stops the receiver when the final subscriber releases it.

### ViconPositionSource

- Implements the common `PositionSource` interface.
- Tracks one Vicon object specified by `object_name`.
- Retrieves position measurements from `ViconProvider`.
- Publishes new measurements as `PositionSample` objects.
- Avoids publishing the same Vicon measurement repeatedly.
- Releases its provider subscription when stopped.

The default position-source polling interval is `1/120` seconds. This is the interval at which the source checks for new measurements, not necessarily the frequency at which Vicon sends packets.

### Position Processing

Position samples are forwarded to `DroneSetup._handle_position_sample()`.

`DroneSetup` handles:

- Updating the drone's current position.
- Maintaining position history.
- Calculating and filtering linear velocity.
- Tracking the time of the latest position update.
- Signalling readiness after receiving a valid position sample.

Although `ViconInterface` calculates linear and angular velocities internally, these values are not currently forwarded through `ViconPositionSource`.

Instead, `DroneSetup` independently calculates linear velocity from received position samples. This allows the same velocity calculation to be used regardless of whether positions originate from Vicon or Crazyflie onboard logging.

### Coordinate System

Vicon provides positions in its global coordinate frame. The Vicon receiver converts position units from millimetres to metres but does not automatically rotate the global X and Y axes into the Crazyflie's body frame.

Ensure that the Vicon global frame, Crazyflie estimator frame, and commanded motion frame are aligned appropriately for the intended control behaviour.

For physical drone positioning instructions, refer to the [Hardware Setup Guide](HARDWARE_SETUP.md#drone-positioning-and-coordinate-system).

