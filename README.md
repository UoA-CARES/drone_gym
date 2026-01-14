<h1 align="center">CARES Drone Gym</h1>

<div align="center">

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Gymnasium-compatible environment for reinforcement learning with Crazyflie drones in real-world settings using Vicon motion capture.

Developed by the CARES Lab at the University of Auckland.

</div>

> **Note**: This environment is designed to work with the [CARES Gymnasium Environments](https://github.com/UoA-CARES/gymnasium_envrionments) framework for running RL training tasks.

## Setup Instructions
For all installation, run the **[setup.sh](setup.sh)** script.

### Hardware Setup
If using real Crazyflie, for detailed hardware setup instructions, please refer to the documentation in the `docs/` folder:
- **[Vicon System Setup](docs/VICON_SETUP.md)** - Connect and configure the motion capture system
- **[Hardware Setup Guide](docs/HARDWARE_SETUP.md)** - Drone positioning, battery management, and troubleshooting


### Useful links for reference
- [CARES Gymnasium Environments](https://github.com/UoA-CARES/gymnasium_envrionments) (for running RL tasks)
- [CARES Reinforcement Learning](https://github.com/UoA-CARES/cares_reinforcement_learning) (RL algorithms)
- [CrazySim](https://github.com/gtfactslab/CrazySim)



## Project Structure

```text
drone_gym/
├── drone_gym/                # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── drone_setup.py        # Base DroneSetup class for drone setup and control
│   ├── drone.py              # Additional functions with PID control and Crazyflie integration
│   ├── drone_sim.py          # Additional functions needed for running in simulation
│   ├── drone_environment.py  # Base DroneEnvironment class for RL tasks
│   ├── task_factory.py       # Selects appropriate task
│   ├── tasks/                # Task examples
│   │   ├── move_to_2d_position.py           # Example: Move to specific 2D position
│   │   ├── move_to_3d_position.py           # Example: Move to specific 3D position
│   │   ├── move_to_random_2d_position.py    # Example: Move to random 2D positions
│   │   ├── move_to_random_3d_position.py    # Example: Move to random 3D positions
│   │   └── move_circle                      # Example: Move to a target moving in a circle
│   ├── tests/                # Testing for various things 
│   │   ├── connection_test.py               # Testing connection with CrazySim
│   │   ├── functionality_tests.py           # Testing utilities
│   │   ├── sim_functionality_tests.py       # Testing utilities in simulation
│   │   └── move_to_position.py              # Testing movement of drone to position 
│   ├── utils/                # Utility modules
│       ├── vicon_connection_class.py        # Vicon motion capture interface
│       └── test_grab_frame.py               # Frame capture testing
├── Dockerfile                # Setting up Docker
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation configuration
├── setup.sh                  # Manage installation
└── README.md                 # This file
```

### Key Components

- **`drone_setup.py`**: Base class for implementing the low-level drone setup and control.
- **`drone.py`**: Child class of `drone_setup.py` with PID velocity/position controllers, Vicon integration, and safety boundary checking
- **`drone_sim.py`**: Child class of `drone_setup.py` with functions for running the drone in simulation
- **`drone_environment.py`**: Abstract base class for creating custom RL environments following the Gymnasium API
- **`utils/vicon_connection_class.py`**: Handles UDP communication with Vicon motion capture system for precise position tracking


## Usage

### Running RL Tasks

This environment is designed to be used with the [CARES Gymnasium Environments](https://github.com/UoA-CARES/gymnasium_envrionments) framework. To run RL training tasks:

```bash
# Navigate to the gymnasium_envrionments directory
cd gymnasium_envrionments/scripts

# Run a training task (example)
python train.py run --env drone_gym --task move_2d
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

Extend the `DroneEnvironment` base class to create custom tasks in the `tasks` folder:

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

### Running using Docker

Simply run `docker run -it --gpus all oculux314/cares:drone` for a prebuilt Docker image to run the drone environment on. This image contains the `cares_reinforcement_learning`, `gymnasium_envrionments`, and `drone_gym` repositories in the `/app` folder. Logs are saved to `/app/cares_rl_logs`.

If you need to modify the image, you can edit `Dockerfile` and rebuild with `docker build -t oculux314/cares:drone .`. The base Dockerfile `oculux314/cares:base` and instructions to run can be found at https://github.com/UoA-CARES/gymnasium_envrionments.

## Related Projects

- [CARES Gymnasium Environments](https://github.com/UoA-CARES/gymnasium_envrionments) - Framework for running RL tasks
- [CARES Reinforcement Learning](https://github.com/UoA-CARES/cares_reinforcement_learning) - RL algorithms library
- [Bitcraze Crazyflie](https://www.bitcraze.io/) - Open-source micro quadcopter platform
