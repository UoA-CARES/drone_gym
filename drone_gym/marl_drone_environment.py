from abc import abstractmethod
import time
from typing import Any, Dict, List, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
from drone_gym.sim_manager import SimManager

########### NOTES ################
# [ ] Need to update Drone class to work with multiple drones and probably 
# have a shared viacon class or smth like above
# [ ] Add type hinting to this class

class MarlDroneEnvironment(ParallelEnv):
    metadata = {"name": "marl_drone_environment_v0", "render_modes": ["human"]}

    def __init__(
        self,
        use_simulator: Literal[0, 1],
        num_agents: int = 2,
        max_velocity: float = 0.5,
        max_velocity_z: float = 0.5,
        step_time: float = 0.5,
        xy_limit: float = 1.0,
        z_min: float = 0.5,
        z_max: float = 1.5,
        reset_height: float = 1.0,
        reset_spacing: float = 0.3,
    ):
        super().__init__()

        # Environment setup
        self.use_simulator = use_simulator
        self.num_agents_config = num_agents

        if self.use_simulator:
            self.sim_manager = SimManager(world_name="crazysim_default")
        else:
            self.sim_manager = None

        # Control limits
        self.max_velocity = max_velocity
        self.max_velocity_z = max_velocity_z
        self.step_time = step_time

        # Movement boundary
        self.xy_limit = xy_limit
        self.z_min = z_min
        self.z_max = z_max
        self.max_xy_range = xy_limit*2
        self.max_z_range = (self.z_max - self.z_min)
        self.max_distance_2d = np.sqrt(self.max_xy_range**2 + self.max_xy_range**2)
        self.max_distance_3d = np.sqrt(self.max_xy_range**2 + self.max_xy_range**2 + self.max_z_range**2)

        # Reset layout
        self.reset_height = reset_height
        self.reset_spacing = reset_spacing

        # Episode state
        self.seed_value = 0
        self.steps = 0

        # Battery threshold
        self.battery_threshold = 3.25

        # PettingZoo agent lists
        self.possible_agents = [f"drone_{i}" for i in range(num_agents)]
        self.agents = []

        # Optional helper mapping
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }

        self.drone_uris = self._generate_default_sim_uris()

        # Per-agent drone objects and state containers
        self.drones: Dict[str, Any] = {}

        # Reset positions must be separated so drones do not start in collision
        self.reset_positions: Dict[str, List[float]] = self._generate_default_reset_positions()

        self.episode_positions: Dict[str, List[List[float]]] = {
            agent: [] for agent in self.possible_agents
        }

        self.prior_states: Dict[str, Dict[str, Any]] = {}
        self.current_batteries: Dict[str, float] = {}

        # Evaluation episode tracking
        self._is_evaluating = False
        self.success_counts: Dict[str, int] = {
            agent: 0 for agent in self.possible_agents
        }
        self.team_success_count = 0

        # Action:
        # [vx, vy, vz] 
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation:
        # [x, y, z, vx, vy, vz, target_dx, target_dy, target_dz]
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self._create_drones()

    # PettingZoo-required API

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        """Reset the drones to initial position and state"""

        # PettingZoo talks about setting seed in reset but this might be handled by the cares rl wrapper??
        # if seed is not None:
        #     self.set_seed(seed)

        options = options or {}
        training = options.get("training", True)
        
        # Handle evaluation mode detection
        if not training and not self._is_evaluating:
            print("--- STARTING NEW EVALUATION BLOCK ---")
            self._is_evaluating = True
        elif training:
            self._is_evaluating = False

        self.agents = self.possible_agents[:]
        self.steps = 0

        # Clear episode data
        self.episode_positions = {agent: [] for agent in self.possible_agents}
        self.prior_states = {}
        self.current_batteries = {}

        # Reset each drone to its own separated reset position
        self._reset_all_drones()

        # Reset task-specific state
        self._reset_task_state()

        # Refresh Gazebo boundary visual
        self._update_visual_boundaries()

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):

        self.steps += 1

        # Read old positions before applying actions
        old_positions = {
            agent: self.drones[agent].get_position()
            for agent in self.agents
        }

        self.prior_states = self._generate_state_dicts(old_positions)

        # Apply all actions
        denormalized_actions = {}
        action_filter_infos = {}

        for agent in self.agents:
            vx, vy, vz = self._denormalize_action(actions[agent])

            vx, vy, vz, action_filter_info = self._apply_task_action_filter(
                agent=agent,
                vx=vx,
                vy=vy,
                vz=vz,
                current_position=old_positions[agent],
            )

            denormalized_actions[agent] = [vx, vy, vz]
            action_filter_infos[agent] = action_filter_info

            self.drones[agent].set_velocity_vector(vx, vy, vz)

        # Advance simulation / wait control interval
        time.sleep(self.step_time)

        # Read new positions
        new_positions = {
            agent: self.drones[agent].get_position()
            for agent in self.agents
        }

        for agent in self.agents:
            self.episode_positions[agent].append(new_positions[agent])
            self.current_batteries[agent] = self.drones[agent].get_battery()

        state_dicts = self._generate_state_dicts(new_positions)

        rewards = self._calculate_rewards(state_dicts)
        terminations = self._check_terminations(state_dicts)
        truncations = self._check_truncations(state_dicts)

        infos = self._get_infos(
            state_dicts=state_dicts,
            denormalized_actions=denormalized_actions,
            normalized_actions=actions,
            old_positions=old_positions,
            new_positions=new_positions,
            action_filter_infos=action_filter_infos,
        )

        observations = self._get_observations()

        # Keep only active agents
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations.get(agent, False) or truncations.get(agent, False))
        ]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        for agent in self.possible_agents:
            if agent in self.drones:
                pos = self.drones[agent].get_position()
                print(f"{agent}: pos={[round(v, 3) for v in pos]}")

        self._render_task_specific_info()
        print("-" * 60)

    def close(self):
        for agent, drone in self.drones.items():
            try:
                drone.land()
                drone.is_landed_event.wait(timeout=30)
                drone.stop()
            except Exception as exc:
                print(f"Error closing {agent}: {exc}")

    def state(self):
        """
        Optional global state for centralized training.
        """
        return self._get_global_state()

    # Drone env helpers

    def _create_drones(self):
        for agent in self.possible_agents:
            if self.use_simulator:
                # TODO: sim_manager shoudn't be passed to each drone
                # SARL tasks need to be updated to use sim_manager directly
                # before this can be removed from the drone constructor 
                self.drones[agent] = DroneSim(
                    uri=self.drone_uris[agent],
                    agent_id=agent,
                    sim_manager=self.sim_manager
                )
            else:
                self.drones[agent] = Drone()

    def _generate_default_reset_positions(self) -> Dict[str, List[float]]:
        """
        Generate reset positions with 30 cm spacing by default.

        For 2 agents:
            drone_0 -> [-0.15, 0.0, reset_height]
            drone_1 -> [ 0.15, 0.0, reset_height]

        Error: Currently ignores area boundary 
        """
        reset_positions = {}

        centre_index = (self.num_agents_config - 1) / 2.0

        for i, agent in enumerate(self.possible_agents):
            x = (i - centre_index) * self.reset_spacing
            y = 0.0
            z = self.reset_height

            reset_positions[agent] = [x, y, z]

        return reset_positions
    
    def _generate_default_sim_uris(self) -> Dict[str, str]:
        return {
            agent: f"udp://0.0.0.0:{19850 + i}"
            for i, agent in enumerate(self.possible_agents)
        }

    def _reset_all_drones(self):
        """
        Resets each drone one by one 
        Can be improved to do all drones at once
        """
        for agent in self.possible_agents:
            drone = self.drones[agent]
            self._reset_control_properties(agent)

            if drone.velocity_controller_active:
                drone.stop_velocity_control()

            drone.set_velocity_vector(0, 0, 0)
            time.sleep(0.5)

            if not drone.is_flying_event.is_set():
                drone.take_off()
                time.sleep(1)

            drone.is_flying_event.wait(timeout=15)

            reset_pos = self.reset_positions[agent]
            drone.set_target_position(*reset_pos)
            time.sleep(0.1)

            drone.start_position_control()
            drone.at_reset_position.wait(timeout=12)
            time.sleep(1)

            drone.stop_position_control()
            drone.clear_reset_position_event()

            initial_position = drone.get_position()
            self.episode_positions[agent].append(initial_position)

            drone.start_velocity_control()

    def _reset_control_properties(self, agent: str):
        drone = self.drones[agent]

        drone.clear_command_queue()
        time.sleep(0.5)

        drone.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

    def _denormalize_action(self, action: np.ndarray):
        vx = float(action[0]) * self.max_velocity
        vy = float(action[1]) * self.max_velocity
        vz = float(action[2]) * self.max_velocity_z

        return vx, vy, vz
    
    def _apply_task_action_processing(
        self,
        agent: str,
        vx: float,
        vy: float,
        vz: float,
        current_position: List[float],
    ):
        """
        Task-specific action processing.

        Default behaviour:
            Do not modify the action.

        Task environments can override this to implement behaviours such as:
            - zeroing actions that would leave the boundary
            - clipping actions near walls
            - bouncing at boundaries
            - slowing down near obstacles

        Returns:
            vx, vy, vz, action_info
        """
        return vx, vy, vz, {}

    def _normalize_position(self, position: List[float]) -> np.ndarray:
        x, y, z = position

        x_norm = np.clip(x / self.xy_limit, -1.0, 1.0)
        y_norm = np.clip(y / self.xy_limit, -1.0, 1.0)

        z_mid = 0.5 * (self.z_min + self.z_max)
        z_half = 0.5 * (self.z_max - self.z_min)
        z_norm = np.clip((z - z_mid) / z_half, -1.0, 1.0)

        return np.array([x_norm, y_norm, z_norm], dtype=np.float32)

    def _normalize_velocity(self, velocity_xyz: List[float]) -> np.ndarray:
        vx, vy, vz = velocity_xyz

        return np.array(
            [
                np.clip(vx / self.max_velocity, -1.0, 1.0),
                np.clip(vy / self.max_velocity, -1.0, 1.0),
                np.clip(vz / self.max_velocity_z, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _normalize_relative_position(self, rel_xyz: List[float]) -> np.ndarray:
        rx, ry, rz = rel_xyz

        return np.array(
            [
                np.clip(rx / self.max_xy_range, -1.0, 1.0),
                np.clip(ry / self.max_xy_range, -1.0, 1.0),
                np.clip(rz / self.max_z_range, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _generate_state_dicts(
        self,
        positions: Dict[str, List[float]],
    ) -> Dict[str, Dict[str, Any]]:
        state_dicts = {}

        for agent, position in positions.items():
            state_dicts[agent] = {
                "position": position,
                "in_boundaries": self.is_in_boundaries(position),
                "steps": self.steps,
                "battery": self.drones[agent].get_battery(),
                "distance_to_target": self._distance_to_target(agent, position),
            }

        return state_dicts

    def is_in_boundaries(self, position: List[float]) -> bool:
        x, y, z = position

        in_xy_range = abs(x) <= self.xy_limit and abs(y) <= self.xy_limit
        in_z_range = self.z_min <= z <= self.z_max

        return in_xy_range and in_z_range

    def set_reset_position(self, agent: str, position: List[float]):
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent: {agent}")

        if len(position) != 3:
            raise ValueError("Reset position must be a 3-element list [x, y, z].")

        self.reset_positions[agent] = position.copy()

    def set_seed(self, seed: int):
        self.seed_value = seed
        np.random.seed(seed)

    # Simulation helpers

    def _set_target_marker(
        self,
        position: List[float],
        marker_name: str = "target",
    ):
        """
        Draw or update one Gazebo target marker

        This should be used by MARL tasks instead of going through DroneSim,
        because the target belongs to the task/environment, not the drone object.
        """
        if self.sim_manager is None:
            return

        self.sim_manager.set_visual_target_marker_position(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            marker_name=marker_name,
        )

    def _update_visual_boundaries(self):
        """
        Draw or update the shared MARL flight boundary.
        """
        if self.sim_manager is None:
            return

        self.sim_manager.set_visual_boundary_lines(
            xy_limit=float(self.xy_limit),
            z_level=float(self.reset_height),
        )

    # Abstract methods to be implemented by task-specific environments

    @abstractmethod
    def _reset_task_state(self):
        pass

    @abstractmethod
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Return one normalized observation vector per active agent.
        """
        pass

    @abstractmethod
    def _calculate_rewards(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        pass

    @abstractmethod
    def _check_terminations(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, bool]:
        pass

    @abstractmethod
    def _check_truncations(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, bool]:
        pass

    @abstractmethod
    def _get_infos(
        self,
        state_dicts: Dict[str, Dict[str, Any]] | None = None,
        denormalized_actions: Dict[str, List[float]] | None = None,
        normalized_actions: Dict[str, np.ndarray] | None = None,
        old_positions: Dict[str, List[float]] | None = None,
        new_positions: Dict[str, List[float]] | None = None,
        action_filter_infos: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def _distance_to_target(self, agent: str, position: List[float]) -> float:
        pass

    @abstractmethod
    def _get_global_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def _render_task_specific_info(self):
        pass