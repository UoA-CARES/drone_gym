from abc import ABC, abstractmethod
from collections.abc import Iterator
from drone_gym.utils.vicon_position_source import ViconPositionSource, ViconProvider
from drone_gym.sim_manager import SimManager, SimLaunchConfig
from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
import time
import numpy as np
from typing import Dict, List, Any, Literal

RL_DRONE_NAME = "rl_drone"

class DroneEnvironment(ABC):
    """
    Base environment for single-agent reinforcement-learning drone tasks.

    The environment owns exactly one RL-controlled drone and may also own
    task-controlled expert drones. Tasks remain responsible for expert
    policies, observations, rewards and other task-specific behaviour.

    The environment manages the lifecycle of all owned drones, including
    creation, coordinated reset, stopping motion and shutdown.
    """

    def __init__(
            self, 
            use_simulator: Literal[0,1],
            max_velocity: float = 0.5,
            step_time: float = 0.5,
            expert_drone_names: list[str] | None = None
        ) -> None:
        """
        Args:
            use_simulator: Use simulated drones when 1, or physical drones when 0.
            max_velocity: Maximum x and y velocity in metres per second.
            step_time: Duration each action is applied, in seconds.
            expert_drone_names: Ordered names of optional expert drones.
        """        
        self._closed = False  # Track if the environment has been closed
        # Set the appropriate drone instance based on use_simulator flag
        print("use_simulator", use_simulator)
        self.use_simulator = use_simulator

        self.expert_drone_names = list(expert_drone_names or [])
        self.num_drones_config = len(self.expert_drone_names) + 1

        if self.use_simulator:
            self.sim_manager = SimManager(
                sim_launch_config=SimLaunchConfig(
                    num_agents=self.num_drones_config
                )
            )
            time.sleep(1)  # Allow time for the sim manager to initialize
            print("Starting simulator...")
            sim_started = self.sim_manager.start_sim()
            if not sim_started:
                raise RuntimeError("Failed to start CrazySim/Gazebo simulator.")
  
        else:
            self.sim_manager = None

        # Initialize the drone instances
        self.rl_drones: dict[str, Drone | DroneSim] = {}
        self.expert_drones: dict[str, Drone | DroneSim] = {}

        self.possible_agents = [RL_DRONE_NAME]
        self.possible_agents.extend(self.expert_drone_names)

        self._create_drones()

        # Initialize reset positions for RL drone followed by possible expert drones
        # The reset_positions list must match the order of _iter_drones()
        self.reset_positions: list[list[float]] = []
        self.reset_position = [0, 0, 1]

        self.max_velocity = max_velocity
        self.max_velocity_z = 0.5
        self.step_time = step_time
        self.steps = 0
        self.seed = 0

        self.battery_threshold = 3.25
        self.observation_space = 12

        # Movement Boundary - can be overridden by tasks
        self.xy_limit = 1.0
        self.z_limit = 0.5

        # Reset target position optimization
        self._reset_target_set = False

        # Episode tracking for evaluation mode
        self._is_evaluating = False
        self.episode_positions = []
        self._log_path = None

        # Success tracking for learning phase
        self.success_count = 0

    @property
    def rl_drone(self) -> Drone | DroneSim:
        """Return the environment's single RL-controlled drone."""

        try:
            return self.rl_drones[RL_DRONE_NAME]
        except KeyError as exc:
            raise RuntimeError(
                "The RL drone has not been created."
            ) from exc


    @property
    def drone(self) -> Drone | DroneSim:
        """
        Backwards-compatible alias for the single RL drone.

        Existing SARL tasks currently use self.drone.
        """
        return self.rl_drone

    @property
    def reset_position(self) -> list[float]:
        """
        Return the RL drone's reset position.

        This compatibility property preserves the existing single-drone
        SARL interface. The RL drone is always the first drone returned
        by _iter_drones(), so its reset position is index zero.
        """
        return self.reset_positions[0]


    @reset_position.setter
    def reset_position(
        self,
        position: List[float],
    ) -> None:
        """
        Set the RL drone's reset position.

        Existing SARL tasks may continue assigning self.reset_position.
        """
        new_position = list(position)

        if (
            hasattr(self, "reset_positions")
            and self.reset_positions
        ):
            self.reset_positions[0] = new_position
        else:
            self.reset_positions = [
                new_position,
            ]

    def _iter_drones(
        self,
    ) -> Iterator[tuple[str, Drone | DroneSim]]:
        """Iterate over the RL drone followed by expert drones."""

        yield from self.rl_drones.items()
        yield from self.expert_drones.items()

    def _generate_default_sim_uris(self) -> dict[str, str]:
        """Generate default simulator URIs for all drones."""
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

    def _get_sim_drone_id(
        self,
        drone_name: str,
    ) -> int:
        """
        Return the CrazySim model index associated with a drone.

        drone_uris is created in the same order as the SITL instances:
        index 0 corresponds to crazyflie_0 and port 19850,
        index 1 corresponds to crazyflie_1 and port 19851, and so on.
        """
        return list(
            self.drone_uris.keys()
        ).index(drone_name)
    
    def _create_drones(self) -> None:
        """Create drone instances for all drones."""

        if self.use_simulator:
            self.drone_uris = self._generate_default_sim_uris()
        else:
            self.drone_uris = self._generate_default_crazyflie_uris()
            self.vicon_object_names = { 
                agent: f"Crzayme_{i}" 
                for i, agent in enumerate(self.possible_agents)
            }
            self.vicon_provider = ViconProvider()

        if self.use_simulator:
            self.rl_drones[RL_DRONE_NAME] = DroneSim(
                uri=self.drone_uris[RL_DRONE_NAME],
                agent_id=RL_DRONE_NAME,
            )
            print(f"[SARL ENV] RL drone simulator created with URI: {self.drone_uris[RL_DRONE_NAME]}")
        else:
            
            self.rl_drones[RL_DRONE_NAME] = Drone(
                agent_id=RL_DRONE_NAME,
                position_source= ViconPositionSource(
                    object_name=self.vicon_object_names[RL_DRONE_NAME],
                    provider=self.vicon_provider,
                    label=RL_DRONE_NAME,
                ),
                uri=self.drone_uris[RL_DRONE_NAME]
            )
            print(f"[SARL ENV] RL drone physical instance created with agent ID: {RL_DRONE_NAME}")

        for expert_agent in self.expert_drone_names:
            if self.use_simulator:
                self.expert_drones[expert_agent] = DroneSim(
                    uri=self.drone_uris[expert_agent],
                    agent_id=expert_agent,
                )
                print(f"[SARL ENV] Expert drone simulator created with URI: {self.drone_uris[expert_agent]}")
            else:
                self.expert_drones[expert_agent] = Drone(
                    agent_id=expert_agent,
                    position_source=ViconPositionSource(
                        object_name=self.vicon_object_names[expert_agent],
                        provider=self.vicon_provider,
                        label=expert_agent,
                    ),
                    uri=self.drone_uris[expert_agent]
                )
                print(f"[SARL ENV] Expert drone physical instance created with agent ID: {expert_agent}")

    def _update_visual_boundaries(self) -> None:
        """
        Draw or update the task flight boundary in Gazebo.

        This is a simulation-only world operation. Physical-drone
        environments do not have a SimManager, so this method is a no-op.
        """
        if self.sim_manager is None:
            return

        xy_limit = float(self.xy_limit)
        z_level = float(self.z_limit)

        if (
            hasattr(self, "boundary")
            and isinstance(self.boundary, list)
            and len(self.boundary) >= 4
        ):
            xy_limit = float(self.boundary[0])
            z_level = float(self.boundary[2])

        self.sim_manager.set_visual_boundary_lines(
            xy_limit=xy_limit,
            z_level=z_level,
        )

    def _set_target_marker(
        self,
        position: List[float] | np.ndarray,
        marker_name: str = "target",
    ) -> None:
        """
        Draw or update a Gazebo marker for a task target.

        This method is a no-op when the environment is controlling
        a physical drone.
        """
        if self.sim_manager is None:
            return

        if len(position) != 3:
            raise ValueError(
                "Target marker position must contain [x, y, z]."
            )

        self.sim_manager.set_visual_target_marker_position(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            marker_name=marker_name,
        )

    def _reset_control_properties(
            self,
            drone: Drone | DroneSim,
        ) -> None:
        drone.clear_command_queue()
        time.sleep(0.5)  # Allow any in-flight commands to be processed
        drone.last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.velocity_last_error = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.velocity_integral = {"x": 0.0, "y": 0.0, "z": 0.0}
        drone.target_velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

    def reset(
        self,
        training: bool = True,
    ):
        """Reset all owned drones and task state."""

        if (
            not training
            and not self._is_evaluating
        ):
            print(
                "--- STARTING NEW EVALUATION BLOCK ---"
            )
            self._is_evaluating = True

        elif training:
            self._is_evaluating = False

        self.episode_positions = []
        self.steps = 0

        print("DRONE RESET")

        if not self.use_simulator:
            low_battery_drones = self._get_low_battery_drones()

            if low_battery_drones:
                if not self.change_battery(low_battery_drones):
                    raise RuntimeError(
                        "Physical drone battery change failed."
                    )

        self._reset_all_drones()
        for _, drone in self._iter_drones():
            if not drone.safety_thread_active:
                drone.start_boundary_monitoring()

        # Preserve the existing ordering: task state is reset
        # after physical drone reset has completed.
        self._reset_task_state()

        self._update_visual_boundaries()

        initial_position = (
            self.rl_drone.get_position()
        )
        self.episode_positions.append(
            initial_position
        )

        return self._get_state()
        
    def _reset_ekf(
        self,
        drone: DroneSim,
        position: list[float] | None = None,
    ) -> None:
        """
        Seed the Kalman filter with the drone's true position and reset it.

        Only call this after the simulated drone has landed and has been
        teleported. Otherwise, the estimator may still use the old position
        and produce a large corrective command during take-off.
        """
        try:
            if getattr(drone, "cf", None) is None:
                return

            if position is not None:
                try:
                    drone.cf.param.set_value(
                        "kalman.initialX",
                        f"{float(position[0])}",
                    )
                    drone.cf.param.set_value(
                        "kalman.initialY",
                        f"{float(position[1])}",
                    )
                    drone.cf.param.set_value(
                        "kalman.initialZ",
                        f"{float(position[2])}",
                    )

                except Exception as exc:
                    print(
                        f"[{getattr(drone, 'agent_id', '?')}] "
                        "EKF seed warning; continuing with plain reset: "
                        f"{exc}"
                    )

            drone.cf.param.set_value(
                "kalman.resetEstimation",
                "1",
            )
            time.sleep(0.4)

        except Exception as exc:
            print(
                f"[{getattr(drone, 'agent_id', '?')}] "
                f"EKF reset warning: {exc}"
            )

    def _stop_drone_motion(
            self,
            drone: Drone | DroneSim,
            reason: str = "",
        ) -> None:
        """
        Cancel the drone's current commanded motion.

        This does not land, disconnect, or stop the velocity-control thread.
        It only replaces the current velocity target with zero.
        """
        try:
            drone.set_velocity_vector(0.0, 0.0, 0.0)

            if reason:
                print(
                    "[SARL ENV] Zero velocity command sent: "
                    f"{reason}"
                )

        except Exception as exc:
            print(
                "[SARL ENV] Failed to send zero velocity command: "
                f"{exc}"
            )

    def _stop_all_drone_motion(
        self,
        reason: str = "",
    ) -> None:
        """Command zero velocity for every owned drone."""

        for _, drone in self._iter_drones():
            self._stop_drone_motion(
                drone=drone,
                reason=reason,
            )

    def _wait_for_all_reset_events(
        self,
        timeout: float = 12.0,
    ) -> bool:
        """
        Wait for all owned drones to signal that they reached
        their reset positions.

        Returns:
            True when every drone reaches its reset target before
            the timeout; otherwise False.
        """
        drones = list(self._iter_drones())
        deadline = time.time() + timeout

        while time.time() < deadline:
            reached_drones = [
                drone_name
                for drone_name, drone in drones
                if drone.at_reset_position.is_set()
            ]

            if len(reached_drones) == len(drones):
                print(
                    "[RESET] All drones reached their "
                    "reset targets."
                )
                return True

            waiting_drones = [
                drone_name
                for drone_name, _ in drones
                if drone_name not in reached_drones
            ]

            print(
                "[RESET] Waiting for drones: "
                f"{waiting_drones}. "
                f"Reached: {reached_drones}"
            )

            time.sleep(0.5)

        timed_out_drones = [
            drone_name
            for drone_name, drone in drones
            if not drone.at_reset_position.is_set()
        ]

        print(
            "[RESET] Timeout waiting for drones: "
            f"{timed_out_drones}"
        )

        return False

    def _reset_all_drones(self) -> None:
        """
        Reset all RL and expert drones to their corresponding
        reset positions.

        reset_positions must follow the same order as _iter_drones():
        the RL drone first, followed by expert drones.
        """
        drones = list(self._iter_drones())

        print(
            "Resetting all drones to initial positions..."
        )
        print(
            f"Reset positions: {self.reset_positions}"
        )

        for _, drone in drones:
            if drone.velocity_controller_active:
                drone.stop_velocity_control()

            time.sleep(0.5)

            self._reset_control_properties(
                drone=drone,
            )

            drone.set_velocity_vector(
                0.0,
                0.0,
                0.0,
            )

            if isinstance(drone, DroneSim):
                drone.land()

        for reset_index, (
            drone_name,
            drone,
        ) in enumerate(drones):
            if not isinstance(drone, DroneSim):
                continue

            drone.is_landed_event.wait(
                timeout=10
            )

            if drone.emergency_event.is_set():
                drone.clear_emergency_event()
                if drone.safety_thread_active:
                    drone.stop_boundary_monitoring()
                    time.sleep(0.2)
                    drone.start_boundary_monitoring()

            reset_position = self.reset_positions[
                reset_index
            ]

            self.sim_manager.set_drone_pose(
                drone_id=self._get_sim_drone_id(
                    drone_name
                ),
                x=float(reset_position[0]),
                y=float(reset_position[1]),
                z=0.02,
                orientation=(0.0, 0.0, 0.0),
            )

        time.sleep(0.3)

        for reset_index, (
            _,
            drone,
        ) in enumerate(drones):
            if not isinstance(drone, DroneSim):
                continue

            reset_position = self.reset_positions[
                reset_index
            ]

            spawn_position = [
                float(reset_position[0]),
                float(reset_position[1]),
                0.02,
            ]

            self._reset_ekf(
                drone=drone,
                position=spawn_position,
            )
        if self.use_simulator:
            time.sleep(0.5)

        for drone_name, drone in drones:
            if not drone.is_flying_event.is_set():
                print(
                    f"[{drone_name}] "
                    "Taking off before reset..."
                )
                drone.take_off()

        time.sleep(1)

        for drone_name, drone in drones:
            if drone.is_flying_event.wait(
                timeout=15
            ):
                continue

            if isinstance(drone, Drone):
                raise RuntimeError(
                    f"[{drone_name}] Failed to confirm "
                    "take-off during reset."
                )

            print(
                f"[{drone_name}] Failed to confirm "
                "take-off during reset."
            )

        for reset_index, (
            _,
            drone,
        ) in enumerate(drones):
            reset_position = self.reset_positions[
                reset_index
            ]

            drone.set_target_position(
                *reset_position
            )

        time.sleep(0.1)

        for _, drone in drones:
            drone.start_position_control()

        reset_success = (
            self._wait_for_all_reset_events(
                timeout=10,
            )
        )

        if not reset_success:
            print(
                "[ERROR] Not all drones reached reset positions in time. Check drone states and reset position configuration."
            )

        time.sleep(1)

        for _, drone in drones:
            drone.stop_position_control()
            drone.clear_reset_position_event()
            drone.set_velocity_vector(
                0.0,
                0.0,
                0.0,
            )

        for reset_index, (
            drone_name,
            drone,
        ) in enumerate(drones):
            initial_position = (
                drone.get_position()
            )

            print(
                f"[{drone_name}] Current position "
                f"after reset: {initial_position}"
            )
            print(
                f"[{drone_name}] Desired reset target: "
                f"{self.reset_positions[reset_index]}"
            )

            drone.start_velocity_control()

    def step(self, action):
        """Execute one step in the environment"""

        # print(self.episode_positions)
        # Check that the current drone battery is above the threshold
        self.current_battery = self.drone.get_battery()
        print(f"Battery level: {self.current_battery}")

        self.steps += 1

        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [vx, vy, vz]")

        # Denormalize action from [-1, 1] to [-max_velocity, max_velocity]
        vx = action[0] * self.max_velocity
        vy = action[1] * self.max_velocity
        vz = action[2] * self.max_velocity_z # topples when moving up --> limit z velocity
        # vz = 0

        print("Normalised action aka velocity is:", [vx, vy, vz])

        current_pos = self.drone.get_position()
        # Store previous state for reward calculation
        self.prior_state = self._generate_state_dict(current_pos)

        # Send velocity command to drone
        self.drone.set_velocity_vector(vx, vy, vz)
        # Apply velocity for specified time - can improve this to be non-blocking
        time.sleep(self.step_time)

        new_pos = self.drone.get_position()
        current_state = self._generate_state_dict(new_pos)

        # Track position for episode trajectory
        self.episode_positions.append(new_pos)

        # Calculate reward using task-specific logic
        reward = self._calculate_reward(current_state)

        # Debugging
        print("\n")
        print("******")
        print(f"REWARD for current step {reward}")
        print(f"POSITION IN ACTION SPACE: {new_pos}")
        print(f"Episode steps: {self.steps}")

        # Check if episode is done using task-specific logic
        done = self._check_if_done(current_state)
        truncated = self._check_if_truncated(current_state)
        if done or truncated:
            self._stop_all_drone_motion()

        # Generate info dict
        info = {
            "current_position": new_pos,
            "previous_position": current_pos,
            "distance_to_target": self._distance_to_target(new_pos),
            "applied_velocity": [vx, vy, vz],  # Store the denorm
            "normalized_action": action,  # Store the original normalized action
            "in_boundaries": self.drone.in_boundaries,
            "steps": self.steps,
            "success_count": self.success_count,
            **self._get_additional_info(current_state),
        }
        gotten_state = self._get_state()
        return gotten_state, reward, done, truncated, info

    def _generate_state_dict(self, position: List[float]) -> Dict[str, Any]:
        """Generate a state dictionary with common drone information"""
        return {
            "position": position,
            "in_boundaries": self.drone.in_boundaries,
            "steps": self.steps,
            "distance_to_target": self._distance_to_target(position),
        }

    # def _generate_action_dict(self, action: List[float]) -> np.ndarray:
    #     """Generate a compact action representation as numpy array"""
    #     action = [
    #         action[0],  # x velocity
    #         action[1],  # y velocity
    #         action[2],  # z velocity
    #     ]
    #     return np.array(action, dtype=np.float32)

    def _distance_to_target(self, position: List[float]) -> float:
        """Calculate distance to target - to be overridden by task"""
        return 0.0

    def get_action_bounds(self) -> Dict:
        """Get the bounds for action space"""
        return {"low": [-1.0, -1.0, -1.0], "high": [1.0, 1.0, 1.0], "shape": (3,)}

    def get_action_space_info(self) -> Dict:
        """Get detailed action space information"""
        return {
            "type": "continuous",
            "shape": (3,),
            "low": [-1.0, -1.0, -1.0],
            "high": [1.0, 1.0, 1.0],
            "description": {
                "vx": "Velocity in the X direction (-1 to 1, scaled to m/s)",
                "vy": "Velocity in the Y direction (-1 to 1, scaled to m/s)",
                "vz": "Velocity in the Z direction (-1 to 1, scaled to m/s)",
            },
            "max_velocity_ms": self.max_velocity,
            "step_duration_s": self.step_time,
        }

    def set_reset_position(self, position: List[float]):
        """Set a new reset position and invalidate the cached target"""
        if len(position) != 3:
            raise ValueError("Reset position must be a 3-element list [x, y, z]")

        self.reset_position = position.copy()
        self._reset_target_set = False  # Force re-setting on next reset
        print(f"Reset position updated to {self.reset_position}")

    def close(self) -> None:
        """
        Stop the drone interface and any simulator owned by the environment.
        """
        if self._closed:
            return

        self._closed = True

        drones = list(self._iter_drones())

        try:
            for drone_name, drone in drones:
                try:
                    if drone.velocity_controller_active:
                        drone.stop_velocity_control()

                    drone.set_velocity_vector(0.0, 0.0, 0.0)
                    drone.land()

                except Exception as exc:
                    print(
                        f"[SARL ENV] Error while landing "
                        f"{drone_name!r}: {exc}"
                    )

            for drone_name, drone in drones:
                        try:
                            if not drone.is_landed_event.wait(
                                timeout=30
                            ):
                                print(
                                    f"[SARL ENV] {drone_name!r} failed "
                                    "to confirm landing. Forcing shutdown."
                                )
                        except Exception as exc:
                            print(
                                f"[SARL ENV] Error while waiting for "
                                f"{drone_name!r} to land: {exc}"
                            )
        finally:
            for drone_name, drone in drones:
                try:
                    drone.stop()
                except Exception as exc:
                    print(
                        f"[SARL ENV] Error while stopping "
                        f"{drone_name!r}: {exc}"
                    )

            self.rl_drones.clear()
            self.expert_drones.clear()

            if self.sim_manager is not None:
                try:
                    self.sim_manager.stop_sim()
                except Exception as exc:
                    print(f"[SARL ENV] Error while stopping simulator: {exc}")

    def render(self, mode="human"):
        """Render the environment state"""
        if mode == "human":
            pos = self.drone.get_position()
            print(f"Drone Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            print(f"In Bounds: {self.drone.in_boundaries}")
            print(f"Steps: {self.steps}")
            self._render_task_specific_info()
            print("-" * 50)

    def set_seed(self, seed: int):
        self.seed = seed
        np.random.seed(seed)

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        """Generate a frame for video recording - to be overridden by tasks"""
        # Default implementation returns white frame
        return np.full((height, width, 3), 255, dtype=np.uint8)

    def is_in_boundaries(self, position=None):
        """Check if drone is within movement boundaries - can be overridden by tasks"""
        if position is None:
            x, y, z = self.drone.get_position()
        else:
            x, y, z = position

        in_height_range = self.z_limit < z < self.z_limit + self.reset_position[2]
        in_xy_range = abs(x) <= self.xy_limit and abs(y) <= self.xy_limit

        return in_xy_range and in_height_range

    def _get_battery_levels(self) -> dict[str, float]:
        if self.use_simulator:
            return {}

        battery_levels = {}

        for drone_name, drone in self._iter_drones():
            if drone.battery_ready_event.wait(timeout=5.0):
                battery_levels[drone_name] = drone.get_battery()
            else:
                print(f"[{drone_name}] No battery telemetry available.")

        return battery_levels

    def _get_low_battery_drones(self) -> list[str]:

        battery_levels = self._get_battery_levels()

        return [
            drone_name
            for drone_name, battery_level in battery_levels.items()
            if battery_level <= self.battery_threshold
        ]

    def need_to_change_battery(self) -> bool:
        if self.use_simulator:
            return False

        return bool(self._get_low_battery_drones())

    def change_battery(
        self,
        low_battery_drones: list[str],
    ) -> bool:
        """
        Handle battery replacement for physical drones.

        All physical drones are landed and disarmed before battery replacement.
        Only drones with low batteries have their Crazyflie connection powered
        down and reinitialised.

        The drones are not taken off in this method. Normal reset behaviour
        handles take-off and movement to reset positions afterwards.

        Args:
            low_battery_drones: Names of drones whose batteries need replacing.

        Returns:
            True if the battery-change process completes successfully.
            False if landing, battery replacement, reconnection, or validation
            fails.
        """
        if self.use_simulator:
            return True

        if not low_battery_drones:
            return True

        drones = dict(self._iter_drones())

        print("\n[BATTERY] Beginning battery change operation.")
        print(f"[BATTERY] Batteries requiring replacement: {low_battery_drones}")

        # Stop all drone controllers and land
        for drone_name, drone in drones.items():
            try:
                drone.clear_command_queue()

                if drone.velocity_controller_active:
                    drone.stop_velocity_control()

                if drone.position_controller_active:
                    drone.stop_position_control()

                drone.set_velocity_vector(
                    0.0,
                    0.0,
                    0.0,
                )

                drone.land()

            except Exception as exc:
                print(f"[BATTERY] Failed to request landing for {drone_name}: {exc}")
                return False

        # Wait for every drone to finish landing.
        for drone_name, drone in drones.items():
            if not drone.is_landed_event.wait(timeout=15):
                print(f"[BATTERY] {drone_name} failed to confirm landing. Battery change aborted.")
                return False

        print("[BATTERY] All drones landed.")

        # Disable boundary monitoring while drones may be handled/moved
        for drone_name, drone in drones.items():
            if drone.safety_thread_active:
                try:
                    drone.stop_boundary_monitoring()
                except Exception as exc:
                    print(f"[BATTERY] Failed to stop boundary monitoring for {drone_name}: {exc}")
                    return False

        # Disarm healthy drones
        for drone_name, drone in drones.items():
            if drone.armed and drone.cf is not None:
                try:
                    print(f"[BATTERY] Disarming {drone_name}...")

                    drone.cf.supervisor.send_arming_request(False)
                    drone.armed = False

                except Exception as exc:
                    print(f"[BATTERY] Failed to disarm {drone_name}: {exc}")
                    return False

        # Power down and disconnect drones needing new batteries
        for drone_name in low_battery_drones:
            drone = drones[drone_name]
            try:
                drone.pre_battery_change_cleanup()

            except Exception as exc:
                print(f"[BATTERY] Failed to prepare {drone_name} for battery change: {exc}")
                return False

        print(
            "\n[BATTERY] Low-battery drones are powered down and ready for battery replacement."
        )

        # Wait for user to replace batteries.
        while True:
            response = input(
                "Are the batteries changed and the drones ready? (y/n): "
            ).strip().lower()

            if response == "y":
                break

            if response == "n":
                raise RuntimeError("[BATTERY] Battery change aborted by user.")

            print(
                "[BATTERY] Invalid input. "
                "Please enter 'y' or 'n'."
            )

        # Invalidate possible stale positions
        for drone_name in low_battery_drones:
            drone = drones[drone_name]

            with drone.position_lock:
                drone.last_position_update_time = None

        # Reinitialise affected Crazyflies
        for drone_name in low_battery_drones:
            drone = drones[drone_name]

            print(
                f"[BATTERY] Reinitialising {drone_name}..."
            )

            if not drone.initialise_crazyflie():
                print(f"[BATTERY] Failed to reinitialise {drone_name}.")    
                return False

        for drone_name in low_battery_drones:
            drone = drones[drone_name]

            if not drone.battery_ready_event.wait(timeout=5.0):
                print(
                    f"[BATTERY] No battery telemetry received "
                    f"from {drone_name} after reconnect."
                )
                return False

            battery_level = drone.get_battery()

            print(f"[BATTERY] {drone_name} battery: {battery_level:.2f} V")

            if battery_level <= self.battery_threshold:
                print(
                    f"[BATTERY] Replacement battery for {drone_name} is still below the threshold ({self.battery_threshold:.2f} V)."
                )
                return False

        # Wait for fresh position information after physical handling
        for drone_name in low_battery_drones:
            drone = drones[drone_name]

            deadline = time.monotonic() + 5.0

            while (
                drone.last_position_update_time is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)

            if drone.last_position_update_time is None:
                print(f"[BATTERY] No fresh position received for {drone_name} after battery replacement.")
                return False

            print(f"[BATTERY] {drone_name} fresh position: {drone.get_position()}")

        # Re-arm drones that remained connected
        for drone_name, drone in drones.items():
            if drone_name in low_battery_drones:
                continue
            try:
                print(f"[BATTERY] Re-arming {drone_name}...")

                drone.cf.platform.send_arming_request(True)
                time.sleep(1.0)

                drone.armed = True

            except Exception as exc:
                print(f"[BATTERY] Failed to re-arm {drone_name}: {exc}")
                return False

        print("[BATTERY] Battery change operation complete.")

        return True

    def restart(self):
        """
        Restart the drone after it has landed or crashed, ensuring it is ready for flight. 
        Does the necessary cleanup, waits for user confirmation, and does re-initialization steps before take-off.
        This method is intended for use with physical drones and will not perform any actions if the drone is a simulation instance.
        """
        if isinstance(self.drone, DroneSim):
            return 
        
        self.drone.pre_battery_change_cleanup()
        time.sleep(2)

        while True:
            response = input("Is the drone ready to fly again? (y/n): ").lower()
            if response == "y":
                break  # Exit the loop and continue with take-off
            elif response == "n":
                print("[Drone] Operation aborted by user.")
                return False  # Exit the function
            else:
                print(
                    "[Drone] Invalid input. Please enter 'y' for yes or 'n' to abort."
                )

        self.drone.initialise_crazyflie()

        self.drone.take_off()
        if not self.drone.is_flying_event.wait(timeout=15):
            print(
                "[ERROR] Drone failed to confirm take-off. MANUAL INTERVENTION REQUIRED."
            )
            return False  # Exit because the drone is in an uncertain state
        return True

    @property
    def max_action_value(self):
        return self.max_velocity

    @property
    def min_action_value(self):
        return -self.max_velocity

    @property
    def action_num(self):
        # return 3
        # on x and y values
        return 3

    # Abstract methods to be implemented by task-specific environments

    @abstractmethod
    def sample_action(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        pass

    @abstractmethod
    def _reset_task_state(self):
        """Reset task-specific state variables"""
        pass

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        """Get the current state representation"""
        pass

    @abstractmethod
    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward based on current state"""
        pass

    @abstractmethod
    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode is done"""
        pass

    @abstractmethod
    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode should be truncated"""
        pass

    @abstractmethod
    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional task-specific info for the info dict"""
        pass

    @abstractmethod
    def _render_task_specific_info(self):
        """Render task-specific information during rendering"""
        pass
