# Hardware Setup Guide

This guide covers physical drone setup, positioning, battery management, and troubleshooting for the Crazyflie drone.

** Related Documentation:**
- [← Back to Main README](../README.md)
- [Vicon System Setup](VICON_SETUP.md) - Configure motion capture system

## Drone Positioning and Coordinate System

### Initial Setup

Before turning on the drone, follow these steps:

1. **Place the marker board on the ground** - This stabilizes the drone takeoff and flight, as the optical flow deck relies on texture changes (verified by trial and error)
2. **Position the drone on the ground** - Turn on the drone while it's on the ground using the small push button on the side

### Coordinate System

The drone uses the following coordinate frame:

<div align="center">
  <img src="https://github.com/user-attachments/assets/93923ad5-50b7-4fd1-ad6a-6e8a362e8e78" width="400" alt="Drone coordinate system diagram 1">
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/d8177450-425e-4ec1-803f-796a6c5fd4ef" width="400" alt="Drone coordinate system diagram 2">
</div>

**Coordinate Frame:**
- **Positive X-axis (red)**: Right direction - **Drone "Front" should face this direction**
- **Positive Y-axis (green)**: Forward direction (towards the window)
- **Z-axis**: Vertical (up)

⚠️ **Important**: Ensure the drone's front is aligned with the positive x-direction before takeoff.

> **Note**: This coordinate system matches the Vicon system configuration. See the [Vicon Setup Guide](VICON_SETUP.md#coordinate-system) for more information about the motion capture coordinate frame.

## Battery Management

### Battery Specifications

| Battery Type | Capacity | Flight Time (Fully Charged) |
|--------------|----------|------------------------------|
| Regular      | 250mAh   | ~5 minutes                   |
| Extended     | 350mAh   | ~7 minutes                   |

### Voltage Levels

| Voltage | Battery State |
|---------|---------------|
| **4.2V** | Fully charged |
| **4.1V** | Recommended minimum for flying |
| **3.8V** | Needs charging (do not discharge below this) |

### Charging

- **Charge time**: ~20-30 minutes per battery
- **Measurement**: Use a multimeter to check voltage levels

### Safety Guidelines

✅ **Use batteries with voltage > 4.1V** for safe and stable flight

❌ **Do not fly with batteries < 3.8V** - risk of voltage drop during flight

## Troubleshooting

Check the **Issues** tab in the repository for known problems and solutions.

### Quick Reference Guide

| Problem | Symptoms | Diagnosis | Solution |
|---------|----------|-----------|----------|
| **Loose Motor Mount** | Drone tips toward one motor<br>• Motor wobbles or has play | Pull on motor mount - if it slides easily, it's too loose | Swap motor mount with a new one from the Crazyflie kit (ask lab technician) |
| **Overheating Motor** | Motor hot to touch<br>• Inconsistent thrust<br>• Drone drifts during flight | Check motor temperature after flight | 1. Let motor cool down<br>2. Replace motor (extras labelled "New" in CARES container) |
| **Flow Deck Not Detected** | Error: "No flow deck is detected!" | Check connection and LED status | 1. Re-seat flow deck firmly<br> 2. Check for bent pins <br> 3. Verify deck LED is on |
| **Vicon Position Not Updating** | Error: "Drone position is not being updated" | Check Vicon system status | See [Vicon Setup Guide](VICON_SETUP.md#troubleshooting) |
| **Low Battery** |Reduced flight time<br>• Unstable hovering<br>• Unexpected landing | Check voltage with multimeter | Replace battery if < 4.1V |
| **Drone Uncontrollable** | Drone not responding to commands | Emergency situation | Software auto-lands if out of bounds<br> |

### Motor Replacement Procedure

For detailed instructions, see: 👉 [Bitcraze Crazyflie Motor Replacement Guide](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/)

**Quick Steps:**

| Step | Action |
|------|--------|
| 1 | Power off drone completely |
| 2 | Gently pull out old motor from mount |
| 3 | Insert new motor (ensure correct orientation) |
| 4 | Verify all motors spin freely |
| 5 | Test flight in safe area |

### Pre-Flight Checklist

Before first flight session:
- [ ] Battery voltage > 4.1V
- [ ] Flow deck LED is on
- [ ] All motors spin freely
- [ ] No loose motor mounts
- [ ] Marker board on the ground
- [ ] Vicon system connected and tracking (see [Vicon Setup Guide](VICON_SETUP.md))
- [ ] Drone positioned with front facing positive x-direction
- [ ] Safety boundaries configured in code

**First time setup?** Complete the [Vicon System Setup](VICON_SETUP.md) before your first flight.

## Hardware Resources

### Lab Equipment Location

- **Crazyflie spare parts kit**: Ask the lab technician
- **New motors**: CARES container, labelled "New"
- **Batteries**: Charging station in the drone lab
- **Marker boards**: Motion capture lab storage

### External Documentation

- [Bitcraze Crazyflie Documentation](https://www.bitcraze.io/documentation/tutorials/getting-started-with-crazyflie-2-x/)
- [Flow Deck v2 Documentation](https://www.bitcraze.io/documentation/hardware/flow_deck_v2/)
- [Crazyflie Client User Guide](https://www.bitcraze.io/documentation/repository/crazyflie-clients-python/master/)

## Contact

For lab access, induction, or equipment issues, contact the Motion Capture Lab technician.
