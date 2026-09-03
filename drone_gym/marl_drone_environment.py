from abc import abstractmethod
import time
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from drone_gym.reset_planner import ResetPlanner
from drone_gym.utils.vicon_position_source import ViconPositionSource, ViconProvider
from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
from drone_gym.sim_manager import SimManager, SimLaunchConfig


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
        reset_spacing: float = 0.5,
        reset_safety_distance: float = 0.25,
        position_max_age: float | None = 0.05,
    ):
        super().__init__()

        # Environment setup
        self.use_simulator = use_simulator
        self.num_agents_config = num_agents

        if self.use_simulator:
            self.sim_manager = SimManager(
                sim_launch_config=SimLaunchConfig(num_agents=self.num_agents_config)
            )

            time.sleep(1)  # Allow time for the sim manager to initialize
            print("Starting simulator...")
            sim_started = self.sim_manager.start_sim()

            if not sim_started:
                raise RuntimeError("Failed to start CrazySim/Gazebo simulator.")
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
        self.max_xy_range = xy_limit * 2
        self.max_z_range = self.z_max - self.z_min
        self.max_distance_2d = np.sqrt(self.max_xy_range**2 + self.max_xy_range**2)
        self.max_distance_3d = np.sqrt(
            self.max_xy_range**2 + self.max_xy_range**2 + self.max_z_range**2
        )

        # Reset
        self.reset_height = reset_height
        self.reset_hover_height = reset_height
        self.reset_spacing = reset_spacing
        self.reset_safety_distance = reset_safety_distance
        self.position_max_age = position_max_age
        self.max_manual_interventions = 2
        self.max_sim_reset_attempts = 3

        # Episode state
        self.seed_value = 0
        self.steps = 0

        # Battery threshold
        self.battery_threshold = 3.25
        self.battery_reset_margin = 0.2
        self.battery_episode_margin = 0.2

        # PettingZoo agent lists
        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents_config)]
        self.agents = []

        if self.use_simulator:
            self.drone_uris = self._generate_default_sim_uris()
        else:
            self.drone_uris = self._generate_default_crazyflie_uris()
            self.vicon_object_names = {
                agent: f"Crzayme_{i}" for i, agent in enumerate(self.possible_agents)
            }
            self.vicon_provider = ViconProvider()

        # Per-agent drone objects and state containers
        self.drones: dict[str, Drone | DroneSim] = {}

        # Reset positions must be separated so drones do not start in collision
        self.reset_positions: dict[str, list[float]] = (
            self._generate_grid_reset_positions()
        )

        self.episode_positions: dict[str, list[list[float]]] = {
            agent: [] for agent in self.possible_agents
        }

        self.prior_states: dict[str, dict[str, Any]] = {}

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

        self.reset_planner = ResetPlanner(
            lambda: self.drones,
            hover_height=lambda: self.reset_hover_height,
            xy_limit=lambda: self.xy_limit,
            z_min=lambda: self.z_min,
            z_max=lambda: self.z_max,
            safety_distance=self.reset_safety_distance,
            position_max_age=self.position_max_age,
        )

        self.reset_planner.validate_reset_positions(self.reset_positions)

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

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
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

        self._reset_all_drones()

        for agent in self.possible_agents:
            drone = self.drones[agent]
            if not drone.safety_thread_active:
                drone.start_boundary_monitoring()

        # Reset task-specific state
        self._reset_task_state()

        # Refresh Gazebo boundary visual
        self._update_visual_boundaries()

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, infos

    def step(self, actions: dict[str, np.ndarray]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
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
            agent: self.drones[agent].get_position() for agent in self.agents
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
            agent: self.drones[agent].get_position() for agent in self.agents
        }

        for agent in self.agents:
            self.episode_positions[agent].append(new_positions[agent])

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
            self._stop_drones_motion(
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
        """Clean up all drone interfaces and the simulator."""

        drones = list(self.drones.items())

        try:
            # First tell all drones to stop moving and land.
            for agent, drone in drones:
                try:
                    if drone.velocity_controller_active:
                        drone.stop_velocity_control()

                    drone.set_velocity_vector(0.0, 0.0, 0.0)
                    drone.land()

                except Exception as exc:
                    print(f"[MARL ENV] Error while landing " f"{agent!r}: {exc}")

            # Then wait for all drones to finish landing.
            for agent, drone in drones:
                try:
                    if not drone.is_landed_event.wait(timeout=30):
                        print(
                            f"[MARL ENV] {agent!r} failed "
                            "to confirm landing. Forcing shutdown."
                        )

                except Exception as exc:
                    print(
                        f"[MARL ENV] Error while waiting for "
                        f"{agent!r} to land: {exc}"
                    )

        finally:
            # Stop every drone interface regardless of landing success.
            for agent, drone in drones:
                try:
                    drone.stop()

                except Exception as exc:
                    print(f"[MARL ENV] Error while stopping " f"{agent!r}: {exc}")

            self.drones.clear()

            if self.sim_manager is not None:
                try:
                    self.sim_manager.stop_sim()

                except Exception as exc:
                    print(f"[MARL ENV] Error while stopping simulator: {exc}")

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
                self.drones[agent] = DroneSim(
                    uri=self.drone_uris[agent],
                    agent_id=agent,
                )
            else:
                self.drones[agent] = Drone(
                    agent_id=agent,
                    position_source=ViconPositionSource(
                        object_name=self.vicon_object_names[agent],
                        provider=self.vicon_provider,
                        label=agent,
                    ),
                    uri=self.drone_uris[agent],
                )

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
                f"reset_spacing={self.reset_spacing} "
                f"is too large for xy_limit={self.xy_limit}."
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

    def _generate_default_crazyflie_uris(self) -> dict[str, str]:
        """Generate default URIs for all physical drones."""
        return {
            agent: f"radio://0/100/2M/E7E7E7E7{index:02X}"
            for index, agent in enumerate(self.possible_agents)
        }

    def _get_battery_levels(self) -> dict[str, float]:
        """Return battery levels for all physical drones."""

        if self.use_simulator:
            return {}

        battery_levels = {}

        for agent in self.possible_agents:
            drone = self.drones[agent]
            if drone.battery_ready_event.wait(timeout=5.0):
                battery_levels[agent] = drone.get_battery()
            else:
                print(f"[{agent}] No battery telemetry available.")

        return battery_levels

    def _get_low_battery_agents(self) -> list[str]:
        """Return physical agents whose batteries are below the threshold."""

        battery_levels = self._get_battery_levels()

        return [
            agent
            for agent, battery_level in battery_levels.items()
            if battery_level <= self.battery_threshold
        ]

    def _land_all_drones_and_disable_safety(
        self,
        label: str,
    ) -> bool:
        """
        Land all drones, disable boundary monitoring, and disarm
        them before a physical service operation.
        """

        if not self._land_all_drones(label=label):
            return False

        for agent in self.possible_agents:
            drone = self.drones[agent]

            if not drone.safety_thread_active:
                continue
            try:
                drone.stop_boundary_monitoring()
            except Exception as exc:
                print(
                    f"[{label}] Failed to stop boundary monitoring for {agent}: {exc}"
                )
                return False

        for agent in self.possible_agents:
            drone = self.drones[agent]

            if not drone.armed or drone.cf is None:
                continue
            try:
                print(f"[{label}] Disarming {agent}...")
                drone.cf.supervisor.send_arming_request(False)
                drone.armed = False
            except Exception as exc:
                print(f"[{label}] Failed to disarm {agent}: {exc}")

                return False

        return True

    def _recover_service_drones(
        self,
        drones_to_service: list[str],
        label: str,
        prompt_text: str,
    ) -> bool:
        """
        Wait for physical servicing of selected drones, reconnect them,
        confirm fresh position data, and re-arm drones that remained
        connected.
        """

        print(f"\n[{label}] {prompt_text}")

        while True:
            response = input(f"{prompt_text} (y/n): ").strip().lower()

            if response == "y":
                break

            if response == "n":
                raise RuntimeError(f"[{label}] Service aborted by user.")

            print(f"[{label}] Invalid input. " "Please enter 'y' or 'n'.")

        # Serviced drones may have physically moved.
        for agent in drones_to_service:
            drone = self.drones[agent]

            with drone.position_lock:
                drone.last_position_update_time = None

        # Reconnect serviced drones.
        for agent in drones_to_service:
            drone = self.drones[agent]

            print(f"[{label}] Reinitialising {agent}...")

            if not drone.initialise_crazyflie():
                print(f"[{label}] Failed to reinitialise " f"{agent}.")

                return False

        # Require fresh position information.
        for agent in drones_to_service:
            drone = self.drones[agent]

            deadline = time.monotonic() + 5.0

            while (
                drone.last_position_update_time is None and time.monotonic() < deadline
            ):
                time.sleep(0.05)

            if drone.last_position_update_time is None:
                print(f"[{label}] No fresh position " f"received for {agent}.")

                return False

            print(f"[{label}] {agent} fresh position: " f"{drone.get_position()}")

        # Re-arm drones which remained connected.
        for agent in self.possible_agents:
            if agent in drones_to_service:
                continue

            drone = self.drones[agent]

            try:
                print(f"[{label}] Re-arming {agent}...")

                drone.cf.supervisor.send_arming_request(True)

                time.sleep(1.0)

                drone.armed = True

            except Exception as exc:
                print(f"[{label}] Failed to re-arm " f"{agent}: {exc}")

                return False

        return True

    def change_battery(
        self,
        low_battery_agents: list[str],
        minimum_voltage: float | None = None,
    ) -> bool:
        """
        Handle battery replacement for physical drones.

        All drones are landed and disarmed before battery replacement.
        Only drones requiring new batteries are powered down and
        reinitialised.

        Args:
            low_battery_agents:
                Agents whose batteries need replacing.
            minimum_voltage:
                Minimum acceptable voltage for replacement batteries.
                Defaults to battery_threshold.

        Returns:
            True when battery replacement completes successfully.
            False otherwise.
        """
        if self.use_simulator:
            return True

        if not low_battery_agents:
            return True

        if minimum_voltage is None:
            minimum_voltage = self.battery_threshold

        print("\n[BATTERY] Beginning battery change operation.")

        print("[BATTERY] Batteries requiring replacement: " f"{low_battery_agents}")

        print("[BATTERY] Required replacement voltage: " f"{minimum_voltage:.2f} V")

        if not self._land_all_drones_and_disable_safety(label="BATTERY"):
            return False

        # Only affected drones are powered down.
        for agent in low_battery_agents:
            try:
                self.drones[agent].pre_battery_change_cleanup()

                print(f"[BATTERY] {agent} ready " "for battery replacement.")

            except Exception as exc:
                print(
                    f"[BATTERY] Failed to prepare " f"{agent} for battery change: {exc}"
                )

                return False

        print(
            "[BATTERY] Low-battery drones are powered down "
            "and ready for battery replacement."
        )

        if not self._recover_service_drones(
            drones_to_service=low_battery_agents,
            label="BATTERY",
            prompt_text=(
                "Please replace the batteries for the above "
                "drones and confirm when done."
            ),
        ):
            self._land_all_drones_and_disable_safety(label="BATTERY")
            return False

        # Validate replacement batteries.
        for agent in low_battery_agents:
            drone = self.drones[agent]

            if not drone.battery_ready_event.wait(timeout=5.0):
                print(
                    f"[BATTERY] No battery telemetry received "
                    f"from {agent} after reconnect."
                )

                return False

            battery_level = drone.get_battery()

            print(f"[BATTERY] {agent} battery: " f"{battery_level:.2f} V")

            if battery_level <= minimum_voltage:
                print(
                    f"[BATTERY] Replacement battery for "
                    f"{agent} is below the required voltage "
                    f"({minimum_voltage:.2f} V)."
                )

                self._land_all_drones_and_disable_safety(label="BATTERY")

                return False

        # Servicing is complete. Restore safety monitoring
        # before flight is allowed again.
        for agent in self.possible_agents:
            drone = self.drones[agent]

            if not drone.safety_thread_active:
                drone.start_boundary_monitoring()

        print("[BATTERY] Battery change operation complete.")

        return True

    def _reset_all_drones(self) -> None:
        """Reset all drones to their respective reset positions."""
        print("Resetting all drones to initial positions...")
        print(f"Reset positions: {self.reset_positions}")

        if self.use_simulator:
            self._reset_all_sim_drones()
        else:
            self._reset_all_physical_drones()

    def _reset_all_sim_drones(
        self,
    ) -> None:
        """
        Reset all simulated drones using the fast Gazebo reset process.

        Retries up to max_sim_reset_attempts when the simulator,
        take-off, reset movement, or final safety state is invalid.
        """

        for attempt in range(
            1,
            self.max_sim_reset_attempts + 1,
        ):
            try:
                self._recover_sim_if_fatal()

                # Stop all existing control before landing.
                for agent in self.possible_agents:
                    drone = self.drones[agent]

                    if drone.velocity_controller_active:
                        drone.stop_velocity_control()

                    if drone.position_controller_active:
                        drone.stop_position_control()

                    self._reset_control_properties(agent)

                    drone.set_velocity_vector(
                        0.0,
                        0.0,
                        0.0,
                    )

                    drone.land()

                # Confirm landing before teleporting.
                failed_landings = [
                    agent
                    for agent in self.possible_agents
                    if not self.drones[agent].is_landed_event.wait(timeout=10)
                ]

                if failed_landings:
                    raise RuntimeError(
                        "Drones failed to confirm landing "
                        f"during sim reset: {failed_landings}"
                    )

                # Clear emergencies and teleport.
                for agent in self.possible_agents:
                    drone = self.drones[agent]

                    if drone.emergency_event.is_set():
                        drone.clear_emergency_event()

                        if drone.safety_thread_active:
                            drone.stop_boundary_monitoring()
                            time.sleep(0.2)
                            drone.start_boundary_monitoring()

                    self.sim_manager.set_drone_pose(
                        drone_id=self.possible_agents.index(agent),
                        x=self.reset_positions[agent][0],
                        y=self.reset_positions[agent][1],
                        z=0.02,
                        orientation=(
                            0.0,
                            0.0,
                            0.0,
                        ),
                    )

                # Allow Gazebo physics to settle.
                time.sleep(0.3)

                # Reset estimators at the teleported positions.
                for agent in self.possible_agents:
                    drone = self.drones[agent]

                    spawn_position = [
                        self.reset_positions[agent][0],
                        self.reset_positions[agent][1],
                        0.02,
                    ]

                    self._reset_ekf(
                        drone,
                        spawn_position,
                    )

                time.sleep(0.5)

                # Take off.
                for agent in self.possible_agents:
                    drone = self.drones[agent]

                    if not drone.is_flying_event.is_set():
                        print(f"[{agent}] " "Taking off before reset...")

                        drone.take_off()

                time.sleep(1.0)

                grounded = [
                    agent
                    for agent in self.possible_agents
                    if not self.drones[agent].is_flying_event.wait(timeout=15)
                ]

                if grounded:
                    raise RuntimeError(
                        "Drones failed to confirm take-off "
                        f"during sim reset: {grounded}"
                    )

                # Send all drones to their actual reset positions.
                for agent in self.possible_agents:
                    self.drones[agent].set_target_position(*self.reset_positions[agent])

                time.sleep(0.1)

                for agent in self.possible_agents:
                    self.drones[agent].start_position_control()

                reset_success = self._wait_for_all_reset_events(timeout=10)

                if not reset_success:
                    raise RuntimeError(
                        "Not all drones reached their reset "
                        "positions before timeout."
                    )

                time.sleep(1.0)
                self._switch_to_velocity_control()

                for agent in self.possible_agents:
                    drone = self.drones[agent]

                    initial_position = drone.get_position()

                    print(
                        f"[{agent}] Current position "
                        f"after reset: {initial_position}"
                    )

                    print(
                        f"[{agent}] Desired reset target: "
                        f"{self.reset_positions[agent]}"
                    )

                    self.episode_positions[agent].append(initial_position)

                if self._all_drones_safe():
                    print("[SIM RESET] All drones safe after reset.")

                    return

                unsafe = self._unsafe_agents()

                raise RuntimeError("Drones unsafe after reset: " f"{unsafe}")

            except Exception as exc:
                if attempt >= self.max_sim_reset_attempts:
                    raise RuntimeError(
                        "Sim reset failed after "
                        f"{self.max_sim_reset_attempts} attempts."
                    ) from exc

                print(
                    f"[SIM RESET] Attempt "
                    f"{attempt}/{self.max_sim_reset_attempts} "
                    f"failed: {exc}. Retrying..."
                )

                time.sleep(5.0)

    def _reset_all_physical_drones(
        self,
    ) -> None:
        """
        Run the physical reset sequence for all drones.

        Batteries are checked before the reset for enough charge to
        complete the reset, and again after the reset for enough charge
        to complete the next episode.

        If a post-reset battery change occurs, the entire physical reset
        is repeated.
        """

        self.reset_planner.validate_reset_positions(self.reset_positions)

        interventions = 0

        while True:
            # Ensure sufficient battery to complete potentially
            # lengthy reset movement.
            self._change_batteries_if_needed(margin=self.battery_reset_margin)

            while True:
                try:
                    # Keep this inside the try because preparation itself
                    # can raise InterventionRequired.
                    self._prepare_drones_for_physical_reset()

                    outcome = self.reset_planner.execute(self.reset_positions)

                    break

                except ResetPlanner.InterventionRequired as escalation:
                    print("[RESET] Automatic reset requires " "manual intervention.")

                    print(f"[RESET] Affected drones: " f"{escalation.agents}")

                    print(f"[RESET] Reason: " f"{escalation.reason}")

                    if not self._land_all_drones(label="RESET"):
                        raise RuntimeError(
                            "Could not safely land all drones "
                            "for manual reset intervention."
                        ) from escalation

                    if interventions >= self.max_manual_interventions:
                        raise RuntimeError(
                            f"Reset failed after "
                            f"{interventions} interventions: "
                            f"{escalation}"
                        ) from escalation

                    if not self._manual_reset_intervention(
                        escalation.agents,
                        escalation.reason,
                    ):
                        raise RuntimeError(
                            "Manual reset recovery failed."
                        ) from escalation

                    interventions += 1

                    # Manual handling changes physical geometry.
                    # Restart planning from the beginning.
                    continue

                except Exception:
                    self._land_all_drones(label="RESET")

                    raise

            # Preserve slot assignment before any battery service.
            self.reset_positions = outcome["assigned_positions"]

            # Reset movement may have consumed significant battery.
            # Before an episode starts, ensure enough remains for the
            # complete episode.
            battery_changed = self._change_batteries_if_needed(
                margin=self.battery_episode_margin
            )

            if battery_changed:
                print(
                    "[RESET] Battery changed after reset. "
                    "Restarting the full physical reset before "
                    "starting the episode."
                )

                # Battery replacement landed/disarmed the fleet and
                # may have physically moved serviced drones.
                continue

            print(f"Reset: " f"{ResetPlanner.summarise(outcome)}")

            self._log_initial_state_error(outcome)

            self._switch_to_velocity_control()

            return

    def _log_initial_state_error(self, outcome: dict) -> None:
        """
        Track how far off each episode actually starts from
        desired reset position."""
        for agent, error in sorted(outcome["final_errors"].items()):
            self.episode_positions[agent].append(self.drones[agent].get_position())
            if error > self.reset_planner.hold_error * 2:
                print(f"{agent} started {error:.3f}m from its reset slot")

    def _unsafe_agents(self) -> list[str]:
        """Which drones are in a fault state."""
        unsafe = []

        for agent in self.possible_agents:
            drone = self.drones[agent]

            if drone.emergency_event.is_set():
                unsafe.append(agent)
            elif isinstance(drone, DroneSim) and drone.fatal_error_event.is_set():
                unsafe.append(agent)

        return unsafe

    def _all_drones_safe(self) -> bool:
        return not self._unsafe_agents()

    def _change_batteries_if_needed(
        self,
        margin: float = 0.0,
    ) -> bool:
        """
        Replace batteries below the required operating voltage.

        Returns:
            True if one or more batteries were changed.
            False otherwise.
        """
        if self.use_simulator:
            return False

        minimum_voltage = self.battery_threshold + margin
        battery_levels = self._get_battery_levels()

        low_battery_agents = [
            agent
            for agent, battery_level in battery_levels.items()
            if battery_level <= minimum_voltage
        ]

        if not low_battery_agents:
            return False

        print(
            "[BATTERY] Drones below required voltage "
            f"({minimum_voltage:.2f} V): "
            f"{low_battery_agents}"
        )

        if not self.change_battery(
            low_battery_agents,
            minimum_voltage=minimum_voltage,
        ):
            raise RuntimeError("Battery change failed for " f"{low_battery_agents}")

        return True

    def _recover_sim_if_fatal(self) -> None:
        """Restart the simulator if any DroneSim has a fatal error."""
        fatal = self._get_fatal_sim_drone_agents()
        if not fatal:
            return

        print(f"Fatal simulator error for: {fatal}")

        if not self._recover_from_fatal_sim_error(max_attempts=3, retry_delay=10.0):
            raise RuntimeError("Failed to recover the simulator after all retries")

    def _prepare_drones_for_physical_reset(self) -> None:
        """Prepare the drones for a ResetPlanner reset.

        Stop active controllers, ensure every drone is airborne, and command a hold
        pose before planning begins.

        Raises:
            ResetPlanner.InterventionRequired: If any drone cannot confirm take-off
                or the required reset-ready state.
        """
        for agent in self.possible_agents:
            drone = self.drones[agent]

            if drone.velocity_controller_active:
                drone.stop_velocity_control()
            if drone.position_controller_active:
                drone.stop_position_control()

            self._reset_control_properties(agent)
            drone.set_velocity_vector(0.0, 0.0, 0.0)

        for agent in self.possible_agents:
            drone = self.drones[agent]
            if not drone.is_flying_event.is_set():
                print(f"{agent} taking off before reset")
                drone.take_off()

        grounded = [
            agent
            for agent in self.possible_agents
            if not self.drones[agent].is_flying_event.wait(timeout=15)
        ]
        if grounded:
            raise ResetPlanner.InterventionRequired(
                grounded, "failed to confirm take-off"
            )

        # Hold where they are. The planner assumes nothing is drifting under a
        # stale command before it starts planning.
        for agent in self.possible_agents:
            drone = self.drones[agent]
            drone.set_target_position(*drone.get_position())

        for agent in self.possible_agents:
            self.drones[agent].start_position_control()

        time.sleep(1.0)

    def _land_all_drones(
        self,
        timeout: float = 15.0,
        label: str = "",
    ) -> bool:
        """Land all drones and return whether all confirmed landing."""

        success = True

        for agent in self.possible_agents:
            drone = self.drones[agent]

            try:
                drone.clear_command_queue()
                if drone.position_controller_active:
                    drone.stop_position_control()
                if drone.velocity_controller_active:
                    drone.stop_velocity_control()
                drone.set_velocity_vector(0.0, 0.0, 0.0)
                drone.land()

            except Exception as exc:
                print(f"[{label}] Failed to request landing for {agent}: {exc}")
                success = False

        for agent in self.possible_agents:
            try:
                if not self.drones[agent].is_landed_event.wait(timeout=timeout):
                    print(f"[{label}] {agent} did not confirm landing.")
                    success = False

            except Exception as exc:
                print(f"[{label}] Error waiting for {agent} to land: {exc}")
                success = False

        return success

    def _switch_to_velocity_control(self) -> None:
        for agent in self.possible_agents:
            drone = self.drones[agent]

            drone.stop_position_control()
            drone.clear_reset_position_event()
            drone.set_velocity_vector(0.0, 0.0, 0.0)
            drone.start_velocity_control()

    def _manual_reset_intervention(
        self,
        failed_agents: list[str],
        reason: str | None = None,
    ) -> bool:
        """
        Allow manual repositioning when automatic physical reset fails.

        All drones are landed and disarmed so the flight area can be entered
        safely. Failed drones are powered down and disconnected using the same
        cleanup process as a battery change. Other drones remain connected.

        After the user repositions and powers the failed drones back on, they
        are reinitialised and all drones are re-armed. The normal physical reset
        should then be restarted from the beginning.
        """
        if self.use_simulator:
            return False

        if not failed_agents:
            return True

        print("\n[RESET RECOVERY] Manual intervention required.")
        print(f"[RESET RECOVERY] Automatic reset failed for: " f"{failed_agents}")
        if reason is not None:
            print(f"[RESET RECOVERY] Reason: {reason}")

        for agent in failed_agents:
            print(
                f"[RESET RECOVERY] {agent} reset target: "
                f"{self.reset_positions[agent]}"
            )

        if not self._land_all_drones_and_disable_safety(label="RESET RECOVERY"):
            return False

        # Only failed drones are powered down/disconnected.
        for agent in failed_agents:
            try:
                self.drones[agent].pre_battery_change_cleanup()
                print(f"[RESET RECOVERY] {agent} powered down.")
            except Exception as exc:
                print(f"[RESET RECOVERY] Failed to prepare {agent}: {exc}")
                return False
        print("\n[RESET RECOVERY] It is now safe to enter the flight area.")

        # Wait for user confirmation that the failed drones have been repositioned
        # and powered back on.
        if not self._recover_service_drones(
            drones_to_service=failed_agents,
            label="RESET RECOVERY",
            prompt_text="Please reposition and power on the failed drones, "
            "then confirm when done.",
        ):
            self._land_all_drones_and_disable_safety(label="RESET RECOVERY")
            return False

        print(
            "[RESET RECOVERY] Manual intervention complete. "
            "Restarting automatic reset."
        )
        for agent in self.possible_agents:
            drone = self.drones[agent]

            if not drone.safety_thread_active:
                drone.start_boundary_monitoring()

        return True

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
                agent for agent in self.possible_agents if agent not in reached_agents
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

    def _reset_ekf(self, drone: DroneSim, position: list[float] | None = None) -> None:
        """Seed the Kalman filter with the drone's true position and reset it.

        DANGEROUS while airborne — only call after the drone has landed and been
        teleported to its spawn. Without this, the EKF still estimates the old
        crash position after a teleport and the position controller fires a large
        corrective thrust that launches the drone into the ceiling.
        """
        try:
            if getattr(drone, "cf", None) is None:
                return
            if position is not None:
                try:
                    drone.cf.param.set_value("kalman.initialX", f"{float(position[0])}")
                    drone.cf.param.set_value("kalman.initialY", f"{float(position[1])}")
                    drone.cf.param.set_value("kalman.initialZ", f"{float(position[2])}")
                except Exception as exc:
                    print(
                        f"[{getattr(drone, 'agent_id', '?')}] EKF seed warning "
                        f"(continuing with plain reset): {exc}"
                    )
            drone.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.4)
        except Exception as exc:
            print(f"[{getattr(drone, 'agent_id', '?')}] EKF reset warning: {exc}")

    def _get_fatal_sim_drone_agents(self) -> list[str]:
        """Return simulated agents whose drone interface has a fatal error."""
        if self.sim_manager is None:
            return []

        return [
            agent
            for agent, drone in self.drones.items()
            if (isinstance(drone, DroneSim) and drone.fatal_error_event.is_set())
        ]

    def _clear_all_drones_instances(self) -> None:
        """
        Shuts down drone interfaces and clears drone objects

        IMPORTANT: Does not land drones
        """
        for agent, drone in list(self.drones.items()):
            try:
                print(f"[MARL] Stopping old interface for {agent}...")
                drone.stop()
            except Exception as exc:
                print(f"[MARL] Error stopping {agent}: {exc}.")

        self.drones.clear()

    def _recover_from_fatal_sim_error(
        self,
        max_attempts: int = 3,
        retry_delay: float = 5.0,
    ) -> bool:
        """
        Restart the simulator and recreate all DroneSim interfaces.

        Args:
            max_attempts: Maximum number of complete simulator restart attempts.
            retry_delay: Delay in seconds between failed attempts.

        Returns:
            True if the simulator and all drone interfaces were recovered.
            False if every recovery attempt failed.
        """
        if self.sim_manager is None:
            raise RuntimeError(
                "Fatal simulated-drone error detected, but no SimManager exists."
            )

        print("[SIM RECOVERY] Preparing to restart the simulation...")

        # The current objects refer to the old SITL processes and cannot be reused.
        self._clear_all_drones_instances()

        for attempt in range(1, max_attempts + 1):
            print(f"[SIM RECOVERY] Recovery attempt " f"{attempt}/{max_attempts}...")

            try:
                sim_restarted = self.sim_manager.restart_sim()

                if not sim_restarted:
                    print(
                        f"[SIM RECOVERY] Simulator restart failed on "
                        f"attempt {attempt}/{max_attempts}."
                    )
                else:
                    print(
                        "[SIM RECOVERY] Simulator restarted. "
                        "Recreating drone interfaces..."
                    )

                    # Ensure a previous partial creation attempt is removed.
                    if self.drones:
                        self._clear_all_drones_instances()

                    self._create_drones()

                    failed_agents = [
                        agent
                        for agent in self.possible_agents
                        if (
                            agent not in self.drones
                            or not self.drones[agent].is_running()
                            or (
                                isinstance(self.drones[agent], DroneSim)
                                and self.drones[agent].fatal_error_event.is_set()
                            )
                        )
                    ]

                    if not failed_agents:
                        print(
                            "[SIM RECOVERY] Simulator and drone interfaces "
                            "recovered successfully."
                        )
                        return True

                    print(
                        "[SIM RECOVERY] The following drone interfaces "
                        f"failed to initialise: {failed_agents}"
                    )

            except Exception as exc:
                print(
                    f"[SIM RECOVERY] Recovery attempt "
                    f"{attempt}/{max_attempts} raised an error: {exc}"
                )

            # Remove any objects created during an unsuccessful attempt.
            if self.drones:
                self._clear_all_drones_instances()

            if attempt < max_attempts:
                print(f"[SIM RECOVERY] Retrying in " f"{retry_delay} seconds...")
                time.sleep(retry_delay)

        print(
            "[SIM RECOVERY] Failed to recover the simulation after "
            f"{max_attempts} attempts."
        )
        return False

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
        """
        Check if a given position is within the defined flight
        boundaries (xy range and z range)."""
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

    def _stop_drones_motion(
        self,
        agents: list[str] | None = None,
        reason: str = "",
    ) -> None:
        """
        Send zero velocity commands to selected drones.

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

    @abstractmethod
    def _render_task_specific_info(self) -> None:
        pass
