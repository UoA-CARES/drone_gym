<h1 align="center">CARES Drone Gym</h1>

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains the code used to control and train the grippers currently being designed and used in the CARES lab at the The University of Auckland. 
</div>

## Installation

### Prerequisites
- Python 3.8 or higher (tested with Python 3.10)
- Crazyflie drone with flow deck attached
- Access to Vicon motion capture system

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/UoA-CARES/drone_gym.git
   cd drone_gym
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## Project Structure

```
drone_gym/
├── drone_gym/              # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── drone.py            with PID control and Crazyflie integration
│   ├── drone_environment.py  # Base DroneEnvironment class for RL tasks
│   ├── move_to_position.py   # Example: Move to specific position
│   ├── move_to_random_position.py  # Example: Move to random positions
│   ├── functionality_tests.py      # Testing utilities
│   ├── utils/             # Utility modules
│   │   ├── vicon_connection_class.py  # Vicon motion capture interface
│   │   └── test_grab_frame.py        # Frame capture testing
│       ├── single_cf_grounded.py
│       └── test_link.py
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation configuration
└── README.md             # This file
```

### Key Components

- **`drone.py`**: Implements the low-level drone control with PID velocity/position controllers, Vicon integration, and safety boundary checking
- **`drone_environment.py`**: Abstract base class for creating custom RL environments following the Gymnasium API
- **`utils/vicon_connection_class.py`**: Handles UDP communication with Vicon motion capture system for precise position tracking

## Hardware Setup

For detailed hardware setup instructions, please refer to the documentation in the `docs/` folder:

- **[Vicon System Setup](docs/VICON_SETUP.md)** - Connect and configure the motion capture system
- **[Hardware Setup Guide](docs/HARDWARE_SETUP.md)** - Drone positioning, battery management, and troubleshooting

## Quick Start

```python
from drone_gym.drone import Drone

# Initialize the drone
drone = Drone()

# Take off
drone.take_off()
drone.is_flying_event.wait(timeout=15)

# Set a target position (x, y, z in meters)
drone.start_position_control()
drone.set_target_position(0.5, 0.5, 1.0)

# Wait for position to be reached
time.sleep(10)

# Land
drone.stop_position_control()
drone.land()
drone.is_landed_event.wait(timeout=15)
drone.stop()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- University of Auckland CARES Lab
- Bitcraze Crazyflie platform
- Vicon Motion Capture Systems
