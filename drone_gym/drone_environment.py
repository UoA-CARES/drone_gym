from abc import ABC, abstractmethod
from collections.abc import Iterator
import time
from typing import Dict, List, Any, Literal
import numpy as np

from drone_gym.utils.vicon_position_source import ViconPositionSource, ViconProvider
from drone_gym.sim_manager import SimManager, SimLaunchConfig
from drone_gym.drone_sim import DroneSim
from drone_gym.drone import Drone
from drone_gym.reset_planner import ResetPlanner


class DroneEnvironment(ABC):
    """
    Base environment for single-agent reinforcement-learning drone tasks.

    The environment owns exactly one RL-controlled drone and may also own
    task-controlled expert drones. Tasks remain responsible for expert
    policies, observations, rewards and other task-specific behaviour.

    The environment manages the lifecycle of all owned drones, including
    creation, coordinated reset, stopping motion and shutdown.
    """

    RL_DRONE_NAME = "rl_drone"

    def __init__(
        self,
        use_simulator: Literal[0, 1],
        max_velocity: float = 0.5,
        step_time: float = 0.5,
        expert_drone_names: list[str] | None = None,
        xy_limit: float = 1.0,
        z_min: float = 0.5,
        z_max: float = 1.5,
        reset_height: float = 1.0,
        reset_safety_distance: float = 0.25,
        position_max_age: float | None = 0.05,
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
                sim_launch_config=SimLaunchConfig(num_agents=self.num_drones_config)
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

        self.possible_agents = [self.RL_DRONE_NAME]
        self.possible_agents.extend(self.expert_drone_names)

        self.max_velocity = max_velocity
        self.max_velocity_z = 0.5
        self.step_time = step_time
        self.steps = 0
        self.seed = 0

        self.battery_threshold = 3.25
        self.observation_space = 12

        # Movement Boundary - can be overridden by tasks
        self.xy_limit = xy_limit
        self.z_min = z_min
        self.z_max = z_max
        self.z_limit = z_min
        # Task-specific environments may replace this with their boundary.
        self.boundary: list[float] | None = None

        self.reset_height = reset_height
        self.reset_hover_height = reset_height
        self.reset_safety_distance = reset_safety_distance
        self.position_max_age = position_max_age
        self.battery_reset_margin = 0.2
        self.battery_episode_margin = 0.2
        self.max_manual_interventions = 2
        self.max_sim_reset_attempts = 3

        # Initialize reset positions keyed by drone name.
        self.reset_positions: dict[str, list[float]] = {}
        self.reset_positions[self.RL_DRONE_NAME] = [0, 0, self.reset_height]

        # Reset target position optimization
        self._reset_target_set = False

        # Episode tracking for evaluation mode
        self._is_evaluating = False
        self.episode_positions = []
        self._log_path = None
        self.prior_state: dict[str, Any] = []

        # Success tracking for learning phase
        self.success_count = 0

        self._create_drones()

        self.reset_planner = ResetPlanner(
            self._get_drone_mapping,
            hover_height=lambda: self.reset_hover_height,
            xy_limit=lambda: self.xy_limit,
            z_min=lambda: self.z_min,
            z_max=lambda: self.z_max,
            safety_distance=self.reset_safety_distance,
            position_max_age=self.position_max_age,
        )

    @property
    def rl_drone(self) -> Drone | DroneSim:
        """Return the environment's single RL-controlled drone."""

        try:
            return self.rl_drones[self.RL_DRONE_NAME]
        except KeyError as exc:
            raise RuntimeError("The RL drone has not been created.") from exc

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
        SARL interface.
        """
        return self.reset_positions[self.RL_DRONE_NAME]

    @reset_position.setter
    def reset_position(
        self,
        position: List[float],
    ) -> None:
        """
        Set the RL drone's reset position.

        Existing SARL tasks may continue assigning self.reset_position.
        """
        self.reset_positions[self.RL_DRONE_NAME] = list(position)

    def _iter_drones(
        self,
    ) -> Iterator[tuple[str, Drone | DroneSim]]:
        """Iterate over the RL drone followed by expert drones."""

        yield from self.rl_drones.items()
        yield from self.expert_drones.items()

    def _get_drone_mapping(
        self,
    ) -> dict[str, Drone | DroneSim]:
        return dict(self._iter_drones())

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
        return list(self.drone_uris.keys()).index(drone_name)

    def _create_drones(self) -> None:
        """Create drone instances for all drones."""

        if self.use_simulator:
            self.drone_uris = self._generate_default_sim_uris()
        else:
            self.drone_uris = self._generate_default_crazyflie_uris()
            self.vicon_object_names = {
                agent: f"Crzayme_{i}" for i, agent in enumerate(self.possible_agents)
            }
            self.vicon_provider = ViconProvider()

        if self.use_simulator:
            self.rl_drones[self.RL_DRONE_NAME] = DroneSim(
                uri=self.drone_uris[self.RL_DRONE_NAME],
                agent_id=self.RL_DRONE_NAME,
            )
            print(
                f"[SARL ENV] RL drone simulator created with URI: "
                f"{self.drone_uris[self.RL_DRONE_NAME]}"
            )
        else:

            self.rl_drones[self.RL_DRONE_NAME] = Drone(
                agent_id=self.RL_DRONE_NAME,
                position_source=ViconPositionSource(
                    object_name=self.vicon_object_names[self.RL_DRONE_NAME],
                    provider=self.vicon_provider,
                    label=self.RL_DRONE_NAME,
                ),
                uri=self.drone_uris[self.RL_DRONE_NAME],
            )
            print(
                f"[SARL ENV] RL drone physical instance created "
                f"with agent ID: {self.RL_DRONE_NAME}"
            )

        for expert_agent in self.expert_drone_names:
            if self.use_simulator:
                self.expert_drones[expert_agent] = DroneSim(
                    uri=self.drone_uris[expert_agent],
                    agent_id=expert_agent,
                )
                print(
                    f"[SARL ENV] Expert drone simulator created with "
                    f"URI: {self.drone_uris[expert_agent]}"
                )
            else:
                self.expert_drones[expert_agent] = Drone(
                    agent_id=expert_agent,
                    position_source=ViconPositionSource(
                        object_name=self.vicon_object_names[expert_agent],
                        provider=self.vicon_provider,
                        label=expert_agent,
                    ),
                    uri=self.drone_uris[expert_agent],
                )
                print(
                    f"[SARL ENV] Expert drone physical instance created "
                    f"with agent ID: {expert_agent}"
                )

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

        if self.boundary is not None and len(self.boundary) >= 4:
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
            raise ValueError("Target marker position must contain [x, y, z].")

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

        if not training and not self._is_evaluating:
            print("--- STARTING NEW EVALUATION BLOCK ---")
            self._is_evaluating = True

        elif training:
            self._is_evaluating = False

        self.episode_positions = []
        self.steps = 0

        print("DRONE RESET")

        self._reset_all_drones()
        for _, drone in self._iter_drones():
            if not drone.safety_thread_active:
                drone.start_boundary_monitoring()

        # Preserve the existing ordering: task state is reset
        # after physical drone reset has completed.
        self._reset_task_state()

        self._update_visual_boundaries()

        initial_position = self.rl_drone.get_position()
        self.episode_positions.append(initial_position)

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
            print(f"[{getattr(drone, 'agent_id', '?')}] " f"EKF reset warning: {exc}")

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
                print("[SARL ENV] Zero velocity command sent: " f"{reason}")

        except Exception as exc:
            print("[SARL ENV] Failed to send zero velocity command: " f"{exc}")

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
                print("[RESET] All drones reached their reset targets.")
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

        print("[RESET] Timeout waiting for drones: " f"{timed_out_drones}")

        return False

    def _reset_all_drones(self) -> None:
        """
        Reset all RL and expert drones to their corresponding
        reset positions.
        """
        print("Resetting all drones to initial positions...")
        print(f"Reset positions: {self.reset_positions}")

        if self.use_simulator:
            self._reset_all_sim_drones()
        else:
            self._reset_all_physical_drones()

    def _reset_all_sim_drones(
        self,
    ) -> None:
        """Reset all simulated drones using the Gazebo reset process."""

        for attempt in range(
            1,
            self.max_sim_reset_attempts + 1,
        ):
            try:
                self._recover_sim_if_fatal()
                drones = list(self._iter_drones())

                # Stop all existing control before landing.
                for _, drone in drones:
                    if drone.velocity_controller_active:
                        drone.stop_velocity_control()

                    if drone.position_controller_active:
                        drone.stop_position_control()

                    self._reset_control_properties(drone=drone)
                    drone.set_velocity_vector(0.0, 0.0, 0.0)
                    drone.land()

                # Confirm landing before teleporting.
                failed_landings = [
                    drone_name
                    for drone_name, drone in drones
                    if not drone.is_landed_event.wait(timeout=10)
                ]

                if failed_landings:
                    raise RuntimeError(
                        "Drones failed to confirm landing "
                        f"during sim reset: {failed_landings}"
                    )

                # Clear emergency states and teleport drones.
                for drone_name, drone in drones:
                    if drone.emergency_event.is_set():
                        drone.clear_emergency_event()

                        if drone.safety_thread_active:
                            drone.stop_boundary_monitoring()
                            time.sleep(0.2)
                            drone.start_boundary_monitoring()

                    self.sim_manager.set_drone_pose(
                        drone_id=self._get_sim_drone_id(drone_name),
                        x=float(self.reset_positions[drone_name][0]),
                        y=float(self.reset_positions[drone_name][1]),
                        z=0.02,
                        orientation=(0.0, 0.0, 0.0),
                    )

                # Allow Gazebo physics to settle.
                time.sleep(0.3)

                # Reset each estimator at the teleported position.
                for drone_name, drone in drones:
                    spawn_position = [
                        float(self.reset_positions[drone_name][0]),
                        float(self.reset_positions[drone_name][1]),
                        0.02,
                    ]
                    self._reset_ekf(drone=drone, position=spawn_position)

                time.sleep(0.5)

                # Take off.
                for drone_name, drone in drones:
                    if not drone.is_flying_event.is_set():
                        print(f"[{drone_name}] " "Taking off before reset...")

                        drone.take_off()

                time.sleep(1.0)

                grounded_drones = [
                    drone_name
                    for drone_name, drone in drones
                    if not drone.is_flying_event.wait(timeout=15)
                ]

                if grounded_drones:
                    raise RuntimeError(
                        "Drones failed to confirm take-off "
                        f"during sim reset: {grounded_drones}"
                    )

                # Move all drones to their actual reset positions.
                for drone_name, drone in drones:
                    drone.set_target_position(*self.reset_positions[drone_name])

                time.sleep(0.1)

                for _, drone in drones:
                    drone.start_position_control()

                reset_success = self._wait_for_all_reset_events(timeout=10)

                if not reset_success:
                    raise RuntimeError(
                        "Not all drones reached their reset positions before timeout."
                    )

                time.sleep(1.0)

                self._switch_to_velocity_control()

                # Log final reset positions.
                for drone_name, drone in drones:
                    initial_position = drone.get_position()

                    print(
                        f"[{drone_name}] Current position "
                        f"after reset: {initial_position}"
                    )

                    print(
                        f"[{drone_name}] Desired reset target: "
                        f"{self.reset_positions[drone_name]}"
                    )

                # Make sure no drone entered an emergency/fatal
                # state during the reset.
                if self._all_drones_safe():
                    print("[SIM RESET] All drones safe after reset.")

                    return

                unsafe_drones = self._unsafe_drones()

                raise RuntimeError("Drones unsafe after reset: " f"{unsafe_drones}")

            except Exception as exc:
                if attempt >= self.max_sim_reset_attempts:
                    raise RuntimeError(
                        "Sim reset failed after "
                        f"{self.max_sim_reset_attempts} "
                        "attempts."
                    ) from exc

                print(
                    f"[SIM RESET] Attempt "
                    f"{attempt}/"
                    f"{self.max_sim_reset_attempts} "
                    f"failed: {exc}. Retrying..."
                )

                time.sleep(5.0)

    def _unsafe_drones(
        self,
    ) -> list[str]:
        """Return drones currently in an emergency or fatal state."""

        unsafe = []

        for drone_name, drone in self._iter_drones():
            if drone.emergency_event.is_set():
                unsafe.append(drone_name)

            elif isinstance(drone, DroneSim) and drone.fatal_error_event.is_set():
                unsafe.append(drone_name)

        return unsafe

    def _all_drones_safe(
        self,
    ) -> bool:
        """Return whether all owned drones are free of fault states."""
        return not self._unsafe_drones()

    def _recover_sim_if_fatal(
        self,
    ) -> None:
        """Restart the simulator if any DroneSim has a fatal error."""

        fatal_drones = self._get_fatal_sim_drone_names()

        if not fatal_drones:
            return

        print(f"Fatal simulator error for: {fatal_drones}")
        if not self._recover_from_fatal_sim_error(max_attempts=3, retry_delay=10.0):
            raise RuntimeError("Failed to recover the simulator after all retries.")

    def _recover_from_fatal_sim_error(
        self,
        max_attempts: int = 3,
        retry_delay: float = 5.0,
    ) -> bool:
        """
        Restart the simulator and recreate all DroneSim interfaces.

        Args:
            max_attempts:
                Maximum number of complete simulator restart attempts.
            retry_delay:
                Delay between failed recovery attempts.

        Returns:
            True if the simulator and all drone interfaces recover.
            False if every recovery attempt fails.
        """

        if self.sim_manager is None:
            raise RuntimeError(
                "Fatal simulated-drone error detected, but no SimManager exists."
            )

        print("[SIM RECOVERY] Preparing to restart the simulation...")

        # Existing DroneSim objects refer to the old SITL
        # processes and cannot be reused.
        self._clear_all_drones_instances()

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            print(f"[SIM RECOVERY] Recovery attempt {attempt}/{max_attempts}...")

            try:
                sim_restarted = self.sim_manager.restart_sim()

                if not sim_restarted:
                    print(
                        "[SIM RECOVERY] Simulator restart "
                        f"failed on attempt {attempt}/{max_attempts}."
                    )

                else:
                    print(
                        "[SIM RECOVERY] Simulator restarted. "
                        "Recreating drone interfaces..."
                    )

                    # Remove any interfaces left by a previous
                    # partial creation attempt.
                    if self.rl_drones or self.expert_drones:
                        self._clear_all_drones_instances()

                    self._create_drones()

                    drones = self._get_drone_mapping()

                    failed_drones = [
                        drone_name
                        for drone_name in self.possible_agents
                        if (
                            drone_name not in drones
                            or not drones[drone_name].is_running()
                            or (
                                isinstance(
                                    drones[drone_name],
                                    DroneSim,
                                )
                                and drones[drone_name].fatal_error_event.is_set()
                            )
                        )
                    ]

                    if not failed_drones:
                        print(
                            "[SIM RECOVERY] Simulator and "
                            "drone interfaces recovered "
                            "successfully."
                        )

                        return True

                    print(
                        "[SIM RECOVERY] The following drone "
                        "interfaces failed to initialise: "
                        f"{failed_drones}"
                    )

            except Exception as exc:
                print(
                    "[SIM RECOVERY] Recovery attempt "
                    f"{attempt}/{max_attempts} "
                    f"raised an error: {exc}"
                )

            # Clean up objects created during a failed attempt.
            if self.rl_drones or self.expert_drones:
                self._clear_all_drones_instances()

            if attempt < max_attempts:
                print("[SIM RECOVERY] Retrying in " f"{retry_delay} seconds...")

                time.sleep(retry_delay)

        print(
            "[SIM RECOVERY] Failed to recover the "
            "simulation after "
            f"{max_attempts} attempts."
        )

        return False

    def _get_fatal_sim_drone_names(
        self,
    ) -> list[str]:
        """Return simulated drones whose interfaces have a fatal error."""

        if self.sim_manager is None:
            return []

        return [
            drone_name
            for drone_name, drone in self._iter_drones()
            if (isinstance(drone, DroneSim) and drone.fatal_error_event.is_set())
        ]

    def _clear_all_drones_instances(
        self,
    ) -> None:
        """
        Stop and remove all current drone interfaces.

        IMPORTANT:
            This does not land drones. It is intended for simulator
            recovery where the existing SITL processes/interfaces are
            already considered invalid.
        """

        for drone_name, drone in list(self._iter_drones()):
            try:
                print(f"[SARL] Stopping old interface " f"for {drone_name}...")

                drone.stop()

            except Exception as exc:
                print(f"[SARL] Error stopping " f"{drone_name}: {exc}.")

        self.rl_drones.clear()
        self.expert_drones.clear()

    def _reset_all_physical_drones(
        self,
    ) -> None:
        """Reset all physical drones using ResetPlanner."""

        self.reset_planner.validate_reset_positions(self.reset_positions)

        manual_interventions = 0

        while True:
            # Before taking off, ensure there is enough battery
            # headroom to safely complete the reset procedure.
            self._change_batteries_if_needed(margin=self.battery_reset_margin)

            while True:
                try:
                    # Must be repeated after manual intervention or
                    # battery servicing because all drones are grounded.
                    self._prepare_drones_for_physical_reset()

                    outcome = self.reset_planner.execute(self.reset_positions)

                    break

                except ResetPlanner.InterventionRequired as escalation:
                    print("[RESET] Automatic reset requires manual intervention.")
                    print(f"[RESET] Affected drones: {escalation.agents}")
                    print(f"[RESET] Reason: {escalation.reason}")

                    if not self._land_all_drones(label="RESET"):
                        raise RuntimeError(
                            "Could not safely land all drones "
                            "for manual reset intervention."
                        ) from escalation

                    if self.use_simulator:
                        raise RuntimeError(
                            "ResetPlanner requested manual intervention "
                            "while using the simulator: "
                            f"{escalation}"
                        ) from escalation

                    if manual_interventions >= self.max_manual_interventions:
                        raise RuntimeError(
                            "Automatic reset still failed after "
                            f"{manual_interventions} manual "
                            "intervention(s): {escalation}"
                        ) from escalation

                    recovery_success = self._manual_reset_intervention(
                        escalation.agents,
                        escalation.reason,
                    )

                    if not recovery_success:
                        raise RuntimeError(
                            "Manual reset recovery failed."
                        ) from escalation

                    manual_interventions += 1

                    # Physical geometry has changed, so restart
                    # ResetPlanner completely.
                    continue

                except Exception:
                    print(
                        "[RESET] Physical reset failed unexpectedly. "
                        "Landing all drones."
                    )

                    self._land_all_drones(label="RESET")

                    raise

            self.reset_positions = outcome["assigned_positions"]

            # The reset may have taken significant time.
            # Check again before starting the episode.
            battery_changed = self._change_batteries_if_needed(
                margin=self.battery_episode_margin
            )

            if battery_changed:
                print(
                    "[RESET] Battery changed after reset. "
                    "Restarting the full physical reset before "
                    "starting the episode."
                )

                # change_battery() landed/disarmed the drones and they
                # may have been physically moved. The previous planner
                # result is therefore no longer a valid episode start.
                continue

            print(f"Reset: " f"{ResetPlanner.summarise(outcome)}")

            for drone_name, error in sorted(outcome["final_errors"].items()):
                if error > self.reset_planner.hold_error * 2:
                    print(f"{drone_name} started " f"{error:.3f}m from its reset slot")

            self._switch_to_velocity_control()

            return

    def _prepare_drones_for_physical_reset(
        self,
    ) -> None:
        """Prepare all owned drones for a ResetPlanner reset."""

        drones = list(self._iter_drones())

        # Stop old controllers and commands.
        for _, drone in drones:
            if drone.velocity_controller_active:
                drone.stop_velocity_control()

            if drone.position_controller_active:
                drone.stop_position_control()

            self._reset_control_properties(
                drone=drone,
            )

            drone.set_velocity_vector(
                0.0,
                0.0,
                0.0,
            )

        # Take off.
        for drone_name, drone in drones:
            if not drone.is_flying_event.is_set():
                print(f"{drone_name} taking off before reset")
                drone.take_off()

        grounded_drones = [
            drone_name
            for drone_name, drone in drones
            if not drone.is_flying_event.wait(timeout=15)
        ]

        if grounded_drones:
            raise ResetPlanner.InterventionRequired(
                grounded_drones,
                "failed to confirm take-off",
            )

        # Hold each drone where it currently is.
        for _, drone in drones:
            drone.set_target_position(*drone.get_position())

        for _, drone in drones:
            drone.start_position_control()

        time.sleep(1.0)

    def _land_all_drones(
        self,
        timeout: float = 15.0,
        label: str = "",
    ) -> bool:
        """Land all owned drones and return whether all confirmed landing."""

        drones = list(self._iter_drones())
        success = True

        for drone_name, drone in drones:
            try:
                drone.clear_command_queue()

                if drone.position_controller_active:
                    drone.stop_position_control()

                if drone.velocity_controller_active:
                    drone.stop_velocity_control()

                drone.set_velocity_vector(
                    0.0,
                    0.0,
                    0.0,
                )

                drone.land()

            except Exception as exc:
                print(
                    f"[{label}] Failed to request landing " f"for {drone_name}: {exc}"
                )
                success = False

        for drone_name, drone in drones:
            try:
                if not drone.is_landed_event.wait(timeout=timeout):
                    print(f"[{label}] {drone_name} " "did not confirm landing.")
                    success = False

            except Exception as exc:
                print(f"[{label}] Error waiting for " f"{drone_name} to land: {exc}")
                success = False

        return success

    def _land_all_drones_and_disable_safety(
        self,
        label: str,
    ) -> bool:
        """
        Land all drones, stop boundary monitoring, and disarm them
        before a physical service operation.
        """

        if not self._land_all_drones(label=label):
            return False

        drones = self._get_drone_mapping()

        # Disable boundary monitoring while drones may be
        # physically handled or moved.
        for drone_name, drone in drones.items():
            if not drone.safety_thread_active:
                continue

            try:
                drone.stop_boundary_monitoring()

            except Exception as exc:
                print(
                    f"[{label}] Failed to stop boundary "
                    f"monitoring for {drone_name}: {exc}"
                )
                return False

        # Disarm every connected drone.
        for drone_name, drone in drones.items():
            if not drone.armed or drone.cf is None:
                continue

            try:
                print(f"[{label}] Disarming {drone_name}...")

                drone.cf.supervisor.send_arming_request(False)

                drone.armed = False

            except Exception as exc:
                print(f"[{label}] Failed to disarm " f"{drone_name}: {exc}")
                return False

        return True

    def _switch_to_velocity_control(
        self,
    ) -> None:
        """Switch all owned drones from position to velocity control."""

        for _, drone in self._iter_drones():
            drone.stop_position_control()
            drone.clear_reset_position_event()

            drone.set_velocity_vector(
                0.0,
                0.0,
                0.0,
            )

            drone.start_velocity_control()

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

        drones = self._get_drone_mapping()

        print(f"\n[{label}] {prompt_text}")

        while True:
            response = input(f"{prompt_text} (y/n): ").strip().lower()

            if response == "y":
                break

            if response == "n":
                raise RuntimeError(f"[{label}] Service aborted by user.")

            print(f"[{label}] Invalid input. " "Please enter 'y' or 'n'.")

        # Serviced drones may have physically moved.
        # Their previous position information is invalid.
        for drone_name in drones_to_service:
            drone = drones[drone_name]

            with drone.position_lock:
                drone.last_position_update_time = None

        # Reconnect/reinitialise serviced drones.
        for drone_name in drones_to_service:
            drone = drones[drone_name]

            print(f"[{label}] Reinitialising " f"{drone_name}...")

            if not drone.initialise_crazyflie():
                print(f"[{label}] Failed to reinitialise " f"{drone_name}.")
                return False

        # Require fresh position information.
        for drone_name in drones_to_service:
            drone = drones[drone_name]

            deadline = time.monotonic() + 5.0

            while (
                drone.last_position_update_time is None and time.monotonic() < deadline
            ):
                time.sleep(0.05)

            if drone.last_position_update_time is None:
                print(f"[{label}] No fresh position " f"received for {drone_name}.")
                return False

            print(f"[{label}] {drone_name} fresh position: " f"{drone.get_position()}")

        # Serviced drones are armed by initialise_crazyflie().
        # Re-arm drones that remained connected.
        for drone_name, drone in drones.items():
            if drone_name in drones_to_service:
                continue

            try:
                print(f"[{label}] Re-arming " f"{drone_name}...")

                drone.cf.supervisor.send_arming_request(True)

                time.sleep(1.0)

                drone.armed = True

            except Exception as exc:
                print(f"[{label}] Failed to re-arm " f"{drone_name}: {exc}")
                return False

        return True

    def _manual_reset_intervention(
        self,
        failed_drones: list[str],
        reason: str | None = None,
    ) -> bool:
        """
        Allow manual repositioning when automatic physical reset fails.

        All drones are landed and disarmed before the flight area is
        entered. Only failed drones are powered down and reinitialised.
        """
        if self.use_simulator:
            return False

        if not failed_drones:
            return True

        drones = self._get_drone_mapping()

        unknown_drones = [
            drone_name for drone_name in failed_drones if drone_name not in drones
        ]

        if unknown_drones:
            print("[RESET RECOVERY] Unknown failed drones: " f"{unknown_drones}")
            return False

        print("\n[RESET RECOVERY] Manual intervention required.")
        print("[RESET RECOVERY] Automatic reset failed for: " f"{failed_drones}")

        if reason is not None:
            print(f"[RESET RECOVERY] Reason: {reason}")

        for drone_name in failed_drones:
            print(
                f"[RESET RECOVERY] {drone_name} reset target: "
                f"{self.reset_positions[drone_name]}"
            )

        if not self._land_all_drones_and_disable_safety(label="RESET RECOVERY"):
            return False

        # Only failed drones are powered down/disconnected.
        for drone_name in failed_drones:
            try:
                drones[drone_name].pre_battery_change_cleanup()

                print(f"[RESET RECOVERY] {drone_name} " "powered down.")

            except Exception as exc:
                print(f"[RESET RECOVERY] Failed to prepare " f"{drone_name}: {exc}")
                return False

        print("\n[RESET RECOVERY] It is now safe to enter the flight area.")

        if not self._recover_service_drones(
            drones_to_service=failed_drones,
            label="RESET RECOVERY",
            prompt_text=(
                "Please reposition and power on the failed drones, "
                "clear the flight area, then confirm when done."
            ),
        ):
            self._land_all_drones_and_disable_safety(label="RESET RECOVERY")
            return False

        # Manual handling is finished. Reinstate monitoring before
        # starting the complete automatic reset again.
        for _, drone in drones.items():
            if not drone.safety_thread_active:
                drone.start_boundary_monitoring()

        print(
            "[RESET RECOVERY] Manual intervention complete. "
            "Restarting automatic reset."
        )

        return True

    def step(self, action):
        """Execute one step in the environment"""

        self.steps += 1

        if len(action) != 3:
            raise ValueError("Action must be a 3-element array [vx, vy, vz]")

        # Denormalize action from [-1, 1] to [-max_velocity, max_velocity]
        vx = action[0] * self.max_velocity
        vy = action[1] * self.max_velocity
        vz = (
            action[2] * self.max_velocity_z
        )  # topples when moving up --> limit z velocity
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
                    print(f"[SARL ENV] Error while landing " f"{drone_name!r}: {exc}")

            for drone_name, drone in drones:
                try:
                    if not drone.is_landed_event.wait(timeout=30):
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
                    print(f"[SARL ENV] Error while stopping " f"{drone_name!r}: {exc}")

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
        minimum_voltage: float | None = None,
    ) -> bool:
        """
        Handle battery replacement for physical drones.

        All drones are landed and disarmed before battery replacement.
        Only drones requiring new batteries are powered down and
        reinitialised.

        Args:
            low_battery_drones:
                Names of drones whose batteries need replacing.
            minimum_voltage:
                Minimum acceptable voltage for the replacement batteries.
                Defaults to battery_threshold.

        Returns:
            True if battery replacement completes successfully.
            False otherwise.
        """
        if self.use_simulator:
            return True

        if not low_battery_drones:
            return True

        if minimum_voltage is None:
            minimum_voltage = self.battery_threshold

        drones = self._get_drone_mapping()

        print("\n[BATTERY] Beginning battery change operation.")
        print("[BATTERY] Batteries requiring replacement: " f"{low_battery_drones}")
        print("[BATTERY] Required replacement voltage: " f"{minimum_voltage:.2f} V")

        if not self._land_all_drones_and_disable_safety(label="BATTERY"):
            return False

        # Only affected drones are powered down/disconnected.
        for drone_name in low_battery_drones:
            try:
                drones[drone_name].pre_battery_change_cleanup()

                print(f"[BATTERY] {drone_name} ready " "for battery replacement.")

            except Exception as exc:
                print(
                    f"[BATTERY] Failed to prepare "
                    f"{drone_name} for battery change: {exc}"
                )
                return False

        print(
            "[BATTERY] Low-battery drones are powered down "
            "and ready for battery replacement."
        )

        if not self._recover_service_drones(
            drones_to_service=low_battery_drones,
            label="BATTERY",
            prompt_text=(
                "Please replace the batteries for the above "
                "drones and confirm when done."
            ),
        ):
            self._land_all_drones_and_disable_safety(label="BATTERY")
            return False

        # Verify every replacement battery is suitable for the
        # operation that triggered the battery change.
        for drone_name in low_battery_drones:
            drone = drones[drone_name]

            if not drone.battery_ready_event.wait(timeout=5.0):
                print(
                    f"[BATTERY] No battery telemetry received "
                    f"from {drone_name} after reconnect."
                )
                return False

            battery_level = drone.get_battery()

            print(f"[BATTERY] {drone_name} battery: " f"{battery_level:.2f} V")

            if battery_level <= minimum_voltage:
                print(
                    f"[BATTERY] Replacement battery for "
                    f"{drone_name} is still below the required "
                    f"voltage ({minimum_voltage:.2f} V)."
                )

                # _recover_service_drones() has re-armed the fleet.
                # Make sure everything is returned to a safe,
                # landed/disarmed state before reporting failure.
                self._land_all_drones_and_disable_safety(label="BATTERY")

                return False
        for _, drone in drones.items():
            if not drone.safety_thread_active:
                drone.start_boundary_monitoring()

        print("[BATTERY] Battery change operation complete.")

        return True

    def _change_batteries_if_needed(
        self,
        margin: float = 0.0,
    ) -> bool:
        """
        Replace batteries below the required operating voltage.

        The required voltage is battery_threshold + margin.

        Args:
            margin:
                Additional voltage required above battery_threshold.

        Returns:
            True if one or more batteries were changed.
            False if no battery change was required.

        Raises:
            RuntimeError:
                If a required battery change fails.
        """
        if self.use_simulator:
            return False

        minimum_voltage = self.battery_threshold + margin

        battery_levels = self._get_battery_levels()

        low_battery_drones = [
            drone_name
            for drone_name, battery_level in battery_levels.items()
            if battery_level <= minimum_voltage
        ]

        if not low_battery_drones:
            return False

        print(
            "[BATTERY] Drones below required voltage "
            f"({minimum_voltage:.2f} V): "
            f"{low_battery_drones}"
        )

        if not self.change_battery(
            low_battery_drones,
            minimum_voltage=minimum_voltage,
        ):
            raise RuntimeError("Battery change failed for " f"{low_battery_drones}")

        return True

    def restart(self):
        """
        Restart the drone after it has landed or crashed, ensuring it is ready for
        flight. Does the necessary cleanup, waits for user confirmation, and does
        re-initialization steps before take-off. This method is intended for use
        with physical drones and will not perform any actions if the drone is a
        simulation instance.
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
                "[ERROR] Drone failed to confirm take-off. "
                "MANUAL INTERVENTION REQUIRED."
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

    @abstractmethod
    def _get_state(self) -> np.ndarray:
        """Get the current state representation"""

    @abstractmethod
    def _calculate_reward(self, current_state: Dict[str, Any]) -> float:
        """Calculate reward based on current state"""

    @abstractmethod
    def _check_if_done(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode is done"""

    @abstractmethod
    def _check_if_truncated(self, current_state: Dict[str, Any]) -> bool:
        """Check if episode should be truncated"""

    @abstractmethod
    def _get_additional_info(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional task-specific info for the info dict"""

    @abstractmethod
    def _render_task_specific_info(self):
        """Render task-specific information during rendering"""
