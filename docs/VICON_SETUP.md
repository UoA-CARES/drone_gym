# Vicon Motion Capture System Setup

This guide covers how to connect and configure the Vicon motion capture system for use with the CARES Drone Gym.

**📚 Related Documentation:**
- [← Back to Main README](../README.md)
- [Hardware Setup Guide](HARDWARE_SETUP.md) - Drone positioning and troubleshooting

## Prerequisites

⚠️ **Important**: Before you start, make sure you are inducted in the Motion Capture Lab!

If the Vicon system is not connected properly, you will the following message in the console: `"Drone position is not being updated"`

## Setup Steps

### 1. Turn on the Vicon Cameras

Click on the two buttons shown in the picture below. **Both top and bottom should be on**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/721651c8-8104-4b07-b570-9dceae6f8fc1" width="400" alt="Vicon cameras on/off buttons">
</p>


![20260130_153850](https://github.com/user-attachments/assets/cde34acd-26e2-4846-a7b4-b5de495cb2fd)
![20260130_153900](https://github.com/user-attachments/assets/fa3b6788-31bc-41c0-92be-9cab619794c9)


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
3. Change the IP address to: **`10.104.144.214`**

<div align="center">
  <img src="https://github.com/user-attachments/assets/edff9550-3619-4c04-8682-79550698886c" width="400" alt="UDP OBJECT STREAM configuration">
</div>

### 4. Enable Crazyflie Object Tracking

1. Navigate to the **OBJECT** tab
2. Make sure **"Crzayme"** is checked

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
If you see the error "Drone position is not being updated":
1. Verify all cameras are powered on
2. Check that the UDP stream is enabled with the correct IP address
3. Ensure the "Crzayme" object is checked in the OBJECT tab
4. Verify the drone is visible to at least 3 cameras
5. Check that you're on the correct network

### Object Not Detected
If the Vicon system doesn't detect the drone:
1. Make sure the reflective markers are clean and properly attached
2. Ensure there's sufficient lighting in the capture volume
3. Recalibrate the object in Vicon Tracker if necessary

## Technical Details

The `vicon_connection_class.py` module handles:
- **UDP socket binding** on port `51001`
- **Real-time position updates** at 60Hz
- **Velocity calculation** from position differentiation
- **Coordinate transformation** from Vicon frame to drone frame

For more technical information, see `drone_gym/utils/vicon_connection_class.py`.
