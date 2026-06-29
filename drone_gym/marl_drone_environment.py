from abc import abstractmethod
import time
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
from drone_gym.sim_manager import SimManager

########### NOTES ################
# [ ] Need to update Drone class to work with multiple drones and probably 
# have a shared vicon class or smth like above


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
        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents_config)]
        self.agents = []

        # Optional helper mapping
        self.agent_name_mapping = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }

        self.drone_uris = self._generate_default_sim_uris()

        # Per-agent drone objects and state containers
        self.drones: dict[str, Drone | DroneSim] = {}

        # Reset positions must be separated so drones do not start in collision
        self.reset_positions: dict[str, list[float]] = self._generate_grid_reset_positions()

        self.episode_positions: dict[str, list[list[float]]] = {
            agent: [] for agent in self.possible_agents
        }

        self.prior_states: dict[str, dict[str, Any]] = {}
        self.current_batteries: dict[str, float] = {}

        # Evaluation episode tracking
        self._is_evaluating = False
        self.success_counts: dict[str, int] = {
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

    def observation_space(self, agent: str) -> spaces.Space:
        """
        Return gymnasium Space object for the given agent's observations.
        """
        return self._observation_space

    def action_space(self, agent: str) -> spaces.Space:
        """
        Return gymnasium Space object for the given agent's actions.
        """
        return self._action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """
        Reset the environment and return initial observations.

        Resets all drones, episode bookkeeping, task-specific state, and visual
        boundaries before starting a new episode.

        Args:
            seed: Optional random seed accepted for PettingZoo API compatibility.
                Currently unused.
            options: Optional reset options. Supports `training`, which defaults
                to True. When `training` is False, evaluation-mode bookkeeping is
                enabled.

        Returns:
            A tuple containing:

            - observations: Initial observations for each active agent.
            - infos: Initial info dictionaries for each active agent.
        """
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
        while True:
            self._reset_all_drones()
            if self._all_drones_safe():
                for agent in self.possible_agents:
                    drone = self.drones[agent]
                    if not drone.safety_thread_active:
                        drone.start_boundary_monitoring()
                break
            time.sleep(20)

        # Reset task-specific state
        self._reset_task_state()

        # Refresh Gazebo boundary visual
        self._update_visual_boundaries()

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def step(self, actions: dict[str, np.ndarray]
        ) -> tuple[dict[str, np.ndarray], 
                   dict[str, float], 
                   dict[str, bool], 
                   dict[str, bool], 
                   dict[str, dict[str, Any]]]:
        """
        Take a step in the environment using the provided actions.

        Actions are expected to be normalized in the range [-1, 1]. They are
        denormalized to velocity commands based on `max_velocity` and
        `max_velocity_z`.

        Args:
            actions: Normalized action commands for each active agent.

        Returns:
            A tuple containing:
                - observations: New observations for each active agent.
                - rewards: Rewards for each active agent.
                - terminations: Whether each agent's episode has terminated.
                - truncations: Whether each agent's episode has been truncated.
                - infos: Additional info for each active agent.
        """
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

            vx, vy, vz, action_filter_info = self._apply_task_action_processing(
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

        finished_agents = [
            agent
            for agent in self.agents
            if terminations.get(agent, False) or truncations.get(agent, False)
        ]

        if finished_agents:
            self._stop_drones(
                agents=finished_agents,
                reason="episode terminated/truncated",
            )

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

    def render(self) -> None:
        """Render the environment state"""

        for agent in self.possible_agents:
            if agent in self.drones:
                pos = self.drones[agent].get_position()
                print(f"{agent}: pos={[round(v, 3) for v in pos]}")

        self._render_task_specific_info()
        print("-" * 60)

    def close(self) -> None:
        """Clean up the drone environment"""
        
        for agent, drone in self.drones.items():
            try:
                drone.land()
                drone.is_landed_event.wait(timeout=30)
                drone.stop()
            except Exception as exc:
                print(f"Error closing {agent}: {exc}")

    def state(self) -> np.ndarray:
        """
        Optional global state for centralized training.
        """
        return self._get_global_state()

    # Drone env helpers

    def _create_drones(self) -> None:
        """Create drone instances for all possible agents."""
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
                self.drones[agent] = Drone(agent_id=agent)
    
    def _generate_grid_reset_positions(self) -> dict[str, list[float]]:
        """
        Generate reset positions in a centred 2D grid that respects xy_limit.

        The grid uses reset_spacing where possible. If the requested spacing
        cannot fit inside the allowed x-y area, an error is raised rather than
        silently placing drones outside the boundary.

        Example for 4 agents:
            drone_0 -> [-0.15, -0.15, reset_height]
            drone_1 -> [ 0.15, -0.15, reset_height]
            drone_2 -> [-0.15,  0.15, reset_height]
            drone_3 -> [ 0.15,  0.15, reset_height]
        """
        reset_positions = {}

        num_agents = len(self.possible_agents)

        if num_agents == 0:
            return reset_positions

        # Keep reset positions slightly away from the boundary.
        reset_margin = 0.05
        usable_xy_limit = self.xy_limit - reset_margin

        if num_agents == 1:
            agent = self.possible_agents[0]
            reset_positions[agent] = [0.0, 0.0, self.reset_height]
            return reset_positions

        # Maximum number of grid positions that can fit along one axis.
        # Example:
        #   xy_limit = 1.0, margin = 0.05, spacing = 0.3
        #   usable range is [-0.95, 0.95], width = 1.9
        #   floor(1.9 / 0.3) + 1 = 7 positions along an axis
        max_positions_per_axis = int((2.0 * usable_xy_limit) // self.reset_spacing) + 1

        if max_positions_per_axis <= 0:
            raise ValueError(
                f"reset_spacing={self.reset_spacing} is too large for xy_limit={self.xy_limit}."
            )

        max_grid_positions = max_positions_per_axis * max_positions_per_axis

        if num_agents > max_grid_positions:
            raise ValueError(
                f"Cannot place {num_agents} drones inside xy_limit={self.xy_limit} "
                f"with reset_spacing={self.reset_spacing}. "
                f"Maximum possible grid positions: {max_grid_positions}."
            )

        # Choose a compact grid shape.
        num_cols = min(max_positions_per_axis, int(np.ceil(np.sqrt(num_agents))))
        num_rows = int(np.ceil(num_agents / num_cols))

        # If rows do not fit, increase columns until rows fit.
        while num_rows > max_positions_per_axis:
            num_cols += 1

            if num_cols > max_positions_per_axis:
                raise ValueError(
                    f"Cannot create valid reset grid for {num_agents} drones with "
                    f"xy_limit={self.xy_limit} and reset_spacing={self.reset_spacing}."
                )

            num_rows = int(np.ceil(num_agents / num_cols))

        # Centre the grid around (0, 0).
        x_centre_index = (num_cols - 1) / 2.0
        y_centre_index = (num_rows - 1) / 2.0

        for i, agent in enumerate(self.possible_agents):
            row = i // num_cols
            col = i % num_cols

            x = (col - x_centre_index) * self.reset_spacing
            y = (row - y_centre_index) * self.reset_spacing
            z = self.reset_height

            if abs(x) > usable_xy_limit or abs(y) > usable_xy_limit:
                raise ValueError(
                    f"Generated reset position for {agent} is outside usable bounds: "
                    f"position={[x, y, z]}, usable_xy_limit={usable_xy_limit}"
                )

            reset_positions[agent] = [float(x), float(y), float(z)]

        return reset_positions
    
    def _generate_default_sim_uris(self) -> dict[str, str]:
        """Generate default simulator URIs for all possible agents."""
        return {
            agent: f"udp://0.0.0.0:{19850 + i}"
            for i, agent in enumerate(self.possible_agents)
        }

    def _reset_all_drones(self) -> None:
        """
        Resets all drones at once to their respective reset positions.
        Ends by enabling velocity control for all drones so they are ready for the first step.
        
        This method can be improved to avoid collision during reset
        """
        print("Resetting all drones to initial positions...")
        print(f"Reset positions: {self.reset_positions}")

        for agent in self.possible_agents:
            drone = self.drones[agent]

            if drone.velocity_controller_active:
                drone.stop_velocity_control()

            time.sleep(0.5)
            self._reset_control_properties(agent)
            drone.set_velocity_vector(0, 0, 0)

            if isinstance(drone, DroneSim):
                drone.land()

        for agent in self.possible_agents:
            drone = self.drones[agent]
            if isinstance(drone, DroneSim):
                drone.is_landed_event.wait(timeout=10)

                if drone.emergency_event.is_set():
                    drone.emergency_event.clear()
                    if drone.safety_thread_active:
                        drone.stop_boundary_monitoring()
                        time.sleep(0.2)
                        drone.start_boundary_monitoring()

                self.sim_manager.set_drone_pose(
                    drone_id=self.possible_agents.index(agent), 
                    x=self.reset_positions[agent][0],
                    y=self.reset_positions[agent][1],
                    z=0.02
                )

        for agent in self.possible_agents:
            drone = self.drones[agent]

            if not drone.is_flying_event.is_set():
                print(f"[{agent}] Taking off before reset...")
                drone.take_off()

        time.sleep(1)

        for agent in self.possible_agents:
            drone = self.drones[agent]

            if not drone.is_flying_event.wait(timeout=15):
                if isinstance(drone, Drone):
                    raise RuntimeError(f"[{agent}] Failed to confirm take-off during reset.")
                else:
                    print(f"[{agent}] Failed to confirm take-off during reset.")

        for agent in self.possible_agents:
            drone = self.drones[agent]

            reset_pos = self.reset_positions[agent]
            drone.set_target_position(*reset_pos)

        time.sleep(0.1)

        for agent in self.possible_agents:
            drone = self.drones[agent]

            drone.start_position_control()
        
        reset_success = self._wait_for_all_reset_events(timeout=10)

        if not reset_success:
            print("[ERROR] Not all drones reached reset positions in time. Check drone states and reset position configuration.")

        time.sleep(1)

        for agent in self.possible_agents:
            drone = self.drones[agent]

            drone.stop_position_control()
            drone.clear_reset_position_event()
            drone.set_velocity_vector(0.0, 0.0, 0.0)

        for agent in self.possible_agents:
            drone = self.drones[agent]

            initial_position = drone.get_position()
            print(f"[{agent}] Current position after reset: {initial_position}")
            print(f"[{agent}] Desired reset target: {self.reset_positions[agent]}")
            self.episode_positions[agent].append(initial_position)

            drone.start_velocity_control()

    def _reset_control_properties(self, agent: str) -> None:
        """
        Reset control state for the given agent's drone.

        Clears queued commands, waits briefly for the queue to settle, and resets
        position and velocity controller errors to zero.
        """
        drone = self.drones[agent]

        drone.clear_command_queue()
        time.sleep(0.5)

        drone.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

    def _wait_for_all_reset_events(self, timeout: float = 12.0) -> bool:
        """
        Wait for all drones to signal that they reached their reset positions.

        Polls each drone's `at_reset_position` event until every possible agent has
        reached its reset target or the timeout expires.

        Args:
            timeout: Maximum number of seconds to wait before giving up.

        Returns:
            bool: True if all drones reach their reset positions before the timeout;
            otherwise False.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            reached_agents = [
                agent
                for agent in self.possible_agents
                if self.drones[agent].at_reset_position.is_set()
            ]

            if len(reached_agents) == len(self.possible_agents):
                print("[RESET] All drones reached their reset targets.")
                return True

            waiting_agents = [
                agent
                for agent in self.possible_agents
                if agent not in reached_agents
            ]

            print(
                "[RESET] Waiting for drones: "
                f"{waiting_agents}. Reached: {reached_agents}"
            )

            time.sleep(0.5)

        timed_out_agents = [
            agent
            for agent in self.possible_agents
            if not self.drones[agent].at_reset_position.is_set()
        ]

        print(f"[RESET] Timeout waiting for drones: {timed_out_agents}")
        return False
    
    def _all_drones_safe(self) -> bool:
        """
        Check if all drones are currently in a safe state (i.e., no emergency events).

        This could be used apart of battery checks or other safety-related terminations in the future.
        """
        return all(
            not self.drones[agent].emergency_event.is_set()
            for agent in self.possible_agents
        )

    def _denormalize_action(self, action: np.ndarray) -> tuple[float, float, float]:
        """
        Convert a normalized action into velocity commands.

        The x and y velocity components are scaled by `max_velocity`, while the z
        velocity component is scaled by `max_velocity_z`.
        """
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
        current_position: list[float],
    ) -> tuple[float, float, float, dict[str, Any]]:
        """
        Apply task-specific processing to velocity commands before execution.

        This hook allows task environments to modify, filter, or annotate the
        velocity commands produced from the policy action before they are sent to
        the drone. 
        
        The default implementation returns the commands unchanged.

        Task environments can override this to implement behaviours such as:
            - zeroing actions that would leave the boundary
            - clipping actions near walls
            - bouncing at boundaries
            - slowing down near obstacles

        Args:
            agent: Agent whose action is being processed.
            vx: Desired x-axis velocity command.
            vy: Desired y-axis velocity command.
            vz: Desired z-axis velocity command.
            current_position: Current position of the agent's drone.

        Returns:
            A tuple containing:

            - vx: Processed x-axis velocity command.
            - vy: Processed y-axis velocity command.
            - vz: Processed z-axis velocity command.
            - action_info: Additional task-specific action metadata.
        """
        return vx, vy, vz, {}

    def _normalize_position(self, position: list[float]) -> np.ndarray:
        """Normalize a 3D position based on environment boundaries."""
        x, y, z = position

        x_norm = x / self.xy_limit
        y_norm = y / self.xy_limit

        z_mid = 0.5 * (self.z_min + self.z_max)
        z_half = 0.5 * (self.z_max - self.z_min)
        z_norm = (z - z_mid) / z_half

        return np.array([x_norm, y_norm, z_norm], dtype=np.float32)

    def _normalize_velocity(self, velocity_xyz: list[float]) -> np.ndarray:
        """Normalize a velocity vector based on maximum velocity limits."""
        vx, vy, vz = velocity_xyz

        return np.array(
            [
                vx / self.max_velocity,
                vy / self.max_velocity,
                vz / self.max_velocity_z,
            ],
            dtype=np.float32,
        )

    def _normalize_relative_position(self, rel_xyz: list[float]) -> np.ndarray:
        """Normalize a relative position vector based on maximum possible distances."""
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
        positions: dict[str, list[float]],
    ) -> dict[str, dict[str, Any]]:
        """Generate a dictionary of state information for each agent."""
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

    def is_in_boundaries(self, position: list[float]) -> bool:
        """Check if a given position is within the defined flight boundaries (xy range and z range)."""
        x, y, z = position

        in_xy_range = abs(x) <= self.xy_limit and abs(y) <= self.xy_limit
        in_z_range = self.z_min <= z <= self.z_max

        return in_xy_range and in_z_range

    def set_reset_position(self, agent: str, position: list[float]) -> None:
        """Set the reset position for a specific agent (x, y, z)."""
        if agent not in self.possible_agents:
            raise ValueError(f"Unknown agent: {agent}")

        if len(position) != 3:
            raise ValueError("Reset position must be a 3-element list [x, y, z].")

        self.reset_positions[agent] = position.copy()

    def set_seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.
        CARES RL compatible wrapper
        """
        self.seed_value = seed
        np.random.seed(seed)

    # Simulation helpers

    def _set_target_marker(
        self,
        position: list[float],
        marker_name: str = "target",
    ) -> None:
        """
        For simulation only:

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

    def _update_visual_boundaries(self) -> None:
        """
        For simulation only:
        
        Draw or update the shared MARL flight boundary.
        """
        if self.sim_manager is None:
            return

        self.sim_manager.set_visual_boundary_lines(
            xy_limit=float(self.xy_limit),
            z_level=float(self.reset_height),
        )

    # TODO Add support for real drones regarding battery changes  
    
    def _stop_drones(
        self,
        agents: list[str] | None = None,
        reason: str = "",
    ) -> None:
        """
        Send zero velocity commands to selected drones.

        This does not stop, land, disconnect, or clean up the underlying DroneSim or
        Drone objects. It only cancels the currently commanded motion so drones do
        not continue flying with their last velocity after an episode terminates or
        truncates.

        Args:
            agents: Agents whose drones should be stopped. If None, all possible
                agents are stopped.
            reason: Optional message included in the debug output.
        """
        agents_to_stop = list(self.possible_agents) if agents is None else agents

        for agent in agents_to_stop:
            if agent not in self.drones:
                continue

            try:
                self.drones[agent].set_velocity_vector(0.0, 0.0, 0.0)

                if reason:
                    print(f"[{agent}] Zero velocity command sent: {reason}")

            except Exception as e:
                print(f"[{agent}] Failed to send zero velocity command: {e}")

    # Abstract methods to be implemented by task-specific environments

    @abstractmethod
    def _reset_task_state(self) -> None:
        pass

    @abstractmethod
    def _get_observations(self) -> dict[str, np.ndarray]:
        """Return one normalized observation vector per active agent."""
        pass

    @abstractmethod
    def _calculate_rewards(
        self,
        state_dicts: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        pass

    @abstractmethod
    def _check_terminations(
        self,
        state_dicts: dict[str, dict[str, Any]],
    ) -> dict[str, bool]:
        pass

    @abstractmethod
    def _check_truncations(
        self,
        state_dicts: dict[str, dict[str, Any]],
    ) -> dict[str, bool]:
        pass

    @abstractmethod
    def _get_infos(
        self,
        state_dicts: dict[str, dict[str, Any]] | None = None,
        denormalized_actions: dict[str, list[float]] | None = None,
        normalized_actions: dict[str, np.ndarray] | None = None,
        old_positions: dict[str, list[float]] | None = None,
        new_positions: dict[str, list[float]] | None = None,
        action_filter_infos: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        pass

    @abstractmethod
    def _distance_to_target(self, agent: str, position: list[float]) -> float:
        pass

    @abstractmethod
    def _get_global_state(self) -> np.ndarray:
        """Generate a global state representation for centralized training."""
        pass

    @abstractmethod
    def _render_task_specific_info(self) -> None:
        pass