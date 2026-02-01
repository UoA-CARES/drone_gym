<h1 align="center">CARES Drone Gym</h1>

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Gymnasium-compatible environment for reinforcement learning with Crazyflie drones in real-world settings using Vicon motion capture.

Developed by the CARES Lab at the University of Auckland.

</div>

> **Note**: This environment is designed to work with the [CARES Gymnasium Environments](https://github.com/UoA-CARES/gymnasium_envrionments) framework for running RL training tasks.

## Installation

### Prerequisites
- Python 3.10 (incompatible with Python 3.12)
- `setup.sh` downloaded (can be found in the same repository as this README)
- Docker
- For physical drone testing:
  - Crazyflie drone with flow deck attached
  - Access to Vicon motion capture system

Setup is now automated - run `bash setup.sh`. If you run into any issues refer to the original instructions at the repositories below.

Note that if you simply want to run a training, you may not need to install locally - instead refer to the docker instructions in 'Running the Simulator' or 'Running the Simulator Standalone' depending on your use case.

### Source Repositories

- [CARES Gymnasium Environments](https://github.com/UoA-CARES/gymnasium_envrionments) - Framework for running RL tasks
- [CARES Reinforcement Learning](https://github.com/UoA-CARES/cares_reinforcement_learning) - RL algorithms library
- [Bitcraze Crazyflie](https://www.bitcraze.io/) - Open-source micro quadcopter platform

## Project Structure

```
drone_gym/
├── drone_gym/              # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── drone.py            with PID control and Crazyflie integration
│   ├── drone_environment.py  # Base DroneEnvironment class for RL tasks
│   ├── move_to_2d_position.py   # Example: Move to specific position
│   ├── move_to_random_2d_position.py  # Example: Move to random positions
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

## Usage

### Running using a Live Crazyflie

Refer to the instructions under Hardware Setup. When running RL tasks, include the `--use_simulator 0` flag.

### Running the Simulator

To run the simulator, in the `drone_gym` directory, use
```bash
docker compose up
```
In a separate terminal (still in the `drone_gym` directory), run
```bash
docker compose exec cares bash
```
Refer to Running RL Tasks for how to execute training runs in a second terminal. The simulator will need to be shutdown and restarted after each training run.

Note that you can run `docker compose up` with the `-d` flag to reuse the same terminal. In that case, use `docker compose down` to shutdown the simulator.

### Running the Simulator Standalone

If you'd prefer not to use docker compose (e.g. when actively modifying drone_gym files), you can run the simulator and CARES RL training environment separately.

**To run the simulator:**
```bash
docker run --rm -p 19850:19850/udp --gpus all --name crazysim -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:ro oculux314/cares:CrazySim
```
If you're in an environment that doesn't support graphics (e.g. when ssh'd into a remote desktop), you can run the simulator in headless mode:
```bash
docker run --rm -p 19850:19850/udp --gpus all --name crazysim oculux314/cares:CrazySim-headless
```
You can run multiple simulator/training instances in parallel by changing the host port referred to by the `-p` flag.

**To run the CARES RL training gym:**
```bash
docker run -it --gpus all --net host oculux314/cares:drone
```
This image contains the `cares_reinforcement_learning`, `gymnasium_envrionments`, and `drone_gym` repositories in the `/app` folder. Logs are saved to `/app/cares_rl_logs`. If you need to modify the image, you can edit `Dockerfile` and rebuild with `docker build -t oculux314/cares:drone .`. The base Dockerfile `oculux314/cares:base` and instructions to run can be found at https://github.com/UoA-CARES/gymnasium_envrionments.

### Running RL Tasks

This environment is designed to be used with the [CARES Gymnasium Environments](https://github.com/UoA-CARES/gymnasium_envrionments) framework. To run RL training tasks:

```bash
# Navigate to the gymnasium_envrionments directory
cd gymnasium_envrionments/scripts

# Run a training task (example)
python train.py run --env drone_gym --task move_to_2d_position
```

Refer to the [gymnasium_envrionments documentation](https://github.com/UoA-CARES/gymnasium_envrionments) for detailed instructions on running tasks and configuring training parameters.

### Direct Drone Control (Example)

Position Based PID Control:

```python
from drone_gym.drone import Drone
import time

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

### Creating Custom RL Tasks

Extend the `DroneEnvironment` base class to create custom tasks:

```python
from drone_gym.drone_environment import DroneEnvironment
import numpy as np

class MyCustomTask(DroneEnvironment):
    def _reset_task_state(self):
        # Initialize task-specific state
        self.target = np.array([1.0, 1.0, 1.0])

    def _get_state(self):
        # Return observation for the RL agent
        pos = self.drone.get_position()
        return np.array([*pos, *self.target])

    def _calculate_reward(self, current_state):
        # Define reward function
        distance = np.linalg.norm(current_state['position'] - self.target)
        return -distance

    # Implement other abstract methods...
```

See `move_to_2d_position.py` and `move_to_random_2d_position.py` for complete examples.

### Example Task Demos

**Move to Position Task**
- Video demonstration of the model evaluation: [Watch on YouTube](https://www.youtube.com/watch?v=MFenj1JX5Cs)
- Implementation example: [gymnasium_environments/drone_gym](https://github.com/UoA-CARES/gymnasium_envrionments/tree/drone_gym)
