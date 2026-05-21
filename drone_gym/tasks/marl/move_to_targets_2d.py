import numpy as np
from gymnasium import spaces
from typing import Any, Dict, List, Literal, Optional

from drone_gym.marl_drone_environment import MarlDroneEnvironment


class MarlMoveToTargets2D(MarlDroneEnvironment):
    """
    Cooperative MARL task where each drone moves to its own target position.

    This is inspired by MPE2 Simple Spread:
    - N agents
    - N target positions
    - each agent observes target relative positions and other-agent relative positions
    - reward combines local target progress, team coverage, and collision penalties

    Uses assigned targets:
        drone_0 -> target_drone_0
        drone_1 -> target_drone_1
        ...
    """

    def __init__(
        self,
        use_simulator: Literal[0, 1],
        num_agents: int = 3,
        max_velocity: float = 0.3,
        max_velocity_z: float = 0.0,
        step_time: float = 0.5,
        xy_limit: float = 1.0,
        z_min: float = 0.5,
        z_max: float = 1.5,
        reset_height: float = 1.0,
        reset_spacing: float = 0.4,
        episode_length: int = 80,
        target_threshold: float = 0.05,
        collision_distance: float = 0.15,
        collision_penalty: float = 0.5,
        local_ratio: float = 0.5,
        success_bonus: float = 2.0,
        target_xy_margin: float = 0.2,
    ):
        super().__init__(
            use_simulator=use_simulator,
            num_agents=num_agents,
            max_velocity=max_velocity,
            max_velocity_z=max_velocity_z,
            step_time=step_time,
            xy_limit=xy_limit,
            z_min=z_min,
            z_max=z_max,
            reset_height=reset_height,
            reset_spacing=reset_spacing,
        )

        self.episode_length = episode_length
        self.target_threshold = target_threshold
        self.collision_distance = collision_distance
        self.collision_penalty = collision_penalty
        self.local_ratio = local_ratio
        self.success_bonus = success_bonus
        self.target_xy_margin = target_xy_margin


        self.target_positions: Dict[str, List[float]] = {}
        self.agent_success: Dict[str, bool] = {
            agent: False for agent in self.possible_agents
        }

        # 2D action:
        # [vx, vy]
        self._action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation:
        # self xy velocity: 2
        # self xy position: 2
        # own target relative position: 2
        # other drone relative positions: 2 * (N - 1)
        obs_dim = 6 + 2 * (self.num_agents_config - 1)

        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _denormalize_action(self, action: np.ndarray):
        """
        Convert normalized 2D action [ax, ay] into drone velocity command.

        The base MARL step loop expects this method to return vx, vy, vz.
        For this 2D task, vz is always 0.0.
        """
        vx = float(action[0]) * self.max_velocity
        vy = float(action[1]) * self.max_velocity
        vz = 0.0

        return vx, vy, vz

    def _apply_task_action_filter(
        self,
        agent: str,
        vx: float,
        vy: float,
        vz: float,
        current_position: List[float],
    ):
        """
        If the proposed velocity command is predicted to move the drone outside
        the allowed flight boundary, replace it with zero velocity.
        """
        prediction_time = self.step_time + self.time_tolerance

        predicted_position = [
            current_position[0] + vx * prediction_time,
            current_position[1] + vy * prediction_time,
            current_position[2] + vz * prediction_time,
        ]

        would_leave_boundaries = not self._is_in_xy_boundaries(predicted_position)

        if would_leave_boundaries:
            return 0.0, 0.0, 0.0, {
                "boundary_guard_triggered": True,
                "requested_action": [vx, vy, vz],
                "sent_action": [0.0, 0.0, 0.0],
                "predicted_position": predicted_position,
            }

        return vx, vy, vz, {}

    def _reset_task_state(self):
        """
        Generate one target per drone and draw/update target markers.
        """
        self.target_positions = {}
        self.agent_success = {
            agent: False for agent in self.possible_agents
        }

        for agent in self.possible_agents:
            self.target_positions[agent] = self._sample_target_position()

        if self.use_simulator:
            for agent, target_position in self.target_positions.items():
                self._set_target_marker(
                    position=target_position,
                    marker_name=f"target_{agent}",
                )

    def _sample_target_position(self) -> List[float]:
        """
        Randomly sample a position for target inside the allowed flight area.
        """
        xy_max = max(0.05, self.xy_limit - self.target_xy_margin)

        x = float(np.random.uniform(-xy_max, xy_max))
        y = float(np.random.uniform(-xy_max, xy_max))
        z = float(self.reset_height)

        return [x, y, z]

    def _get_observations(self) -> Dict[str, np.ndarray]:
        observations = {}

        # TODO: get_position and get_velocity functions may not exist or be wrong
        # Check agaisnt SARL tasks
        positions = {
            agent: self.drones[agent].get_position()
            for agent in self.agents
        }

        velocities = {
            agent: self.drones[agent].get_velocity()
            for agent in self.agents
        }

        for agent in self.agents:
            own_position = positions[agent]
            own_velocity = velocities[agent]
            own_target = self.target_positions[agent]

            obs_parts = []

            # Own velocity and position
            obs_parts.append(self._normalize_velocity_xy(own_velocity))
            obs_parts.append(self._normalize_position_xy(own_position))

            # Own target relative position
            own_target_rel = [
                own_target[0] - own_position[0],
                own_target[1] - own_position[1],
            ]
            obs_parts.append(self._normalize_relative_xy(own_target_rel))

            # Other drones
            for other_agent in self.possible_agents:
                if other_agent == agent:
                    continue

                if other_agent in positions:
                    other_position = positions[other_agent]
                    other_drone_rel = [
                        other_position[0] - own_position[0],
                        other_position[1] - own_position[1],
                    ]
                else:
                    # If the other agent is no longer active, zero-pad.
                    other_drone_rel = [0.0, 0.0]

                obs_parts.append(
                    self._normalize_relative_xy(other_drone_rel)
                )

            observations[agent] = np.concatenate(obs_parts).astype(np.float32)

        return observations

    def _normalize_position_xy(self, position: List[float]) -> np.ndarray:
        x, y = position[0], position[1]

        return np.array(
            [
                x / self.xy_limit,
                y / self.xy_limit,
            ],
            dtype=np.float32,
        )

    def _normalize_velocity_xy(self, velocity_xyz: List[float]) -> np.ndarray:
        vx, vy = velocity_xyz[0], velocity_xyz[1]

        return np.array(
            [
                vx / self.max_velocity,
                vy / self.max_velocity,
            ],
            dtype=np.float32,
        )

    def _normalize_relative_xy(self, rel_xy: List[float]) -> np.ndarray:
        rx, ry = rel_xy

        return np.array(
            [
                rx / self.max_xy_range,
                ry / self.max_xy_range,
            ],
            dtype=np.float32,
        )

    def _calculate_rewards(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        distances = {
            agent: state_dicts[agent]["distance_to_target"]
            for agent in self.agents
        }

        normalized_distances = {
            agent: distances[agent] / max(self.max_distance_2d, 1e-6)
            for agent in self.agents
        }

        # Team coverage reward: assigned-target version.
        # Later, this can be changed to true Simple Spread style:
        # for each target, use the distance to the closest drone. 
        global_reward = -float(np.mean(list(normalized_distances.values())))

        rewards = {}

        for agent in self.agents:
            local_reward = -float(normalized_distances[agent])

            reward = (
                self.local_ratio * local_reward
                + (1.0 - self.local_ratio) * global_reward
            )

            reward += self._collision_penalty_for_agent(agent)

            if self._all_targets_reached(state_dicts):
                reward += self.success_bonus

            rewards[agent] = float(reward)

        return rewards

    def _collision_penalty_for_agent(self, agent: str) -> float:
        """
        Penalise this agent for being too close to other active agents.

        TODO check if position can be foudn from state_dicts instead of get_position() calls
        TODO consider multiple levels of penalty based on distance, similar design to a paper that used multiple circle radii for different penalty levels 
        """
        if agent not in self.agents:
            return 0.0

        own_position = self.drones[agent].get_position()
        own_xy = np.array(own_position[:2], dtype=np.float32)
        penalty = 0.0

        for other_agent in self.agents:
            if other_agent == agent:
                continue

            other_position = self.drones[other_agent].get_position()
            other_xy = np.array(other_position[:2], dtype=np.float32)
            distance = float(np.linalg.norm(own_xy - other_xy))

            if distance < self.collision_distance:
                penalty -= self.collision_penalty

        return penalty

    def _is_in_xy_boundaries(self, position: List[float]) -> bool:
        """
        Check only the x-y flight boundary.
        """
        x, y = position[0], position[1]

        return abs(x) <= self.xy_limit and abs(y) <= self.xy_limit
    
    def _is_in_z_boundaries(self, position: List[float]) -> bool:
        """
        Check z-only truncation condition.
        """
        z = float(position[2])

        if z <= self.z_truncation_min:
            return False

        if self.truncate_on_high_z and z > self.z_max:
            return False

        return True


    def _agents_with_z_boundary_violation(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        Return active agents whose z position violates z boundary.
        """
        violating_agents = []

        for agent in self.agents:
            position = state_dicts[agent]["position"]

            if not self._is_in_z_boundaries(position):
                violating_agents.append(agent)

        return violating_agents

    def _check_terminations(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, bool]:
        """
        Cooperative termination: end the episode for all agents only when
        every drone has reached its assigned target.
        """
        all_reached = self._all_targets_reached(state_dicts)

        if all_reached:
            for agent in self.possible_agents:
                self.agent_success[agent] = True

            self.team_success_count += 1

        return {
            agent: all_reached
            for agent in self.agents
        }

    def _check_truncations(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, bool]:
        """
        Truncate all agents together if the episode is too long or unsafe.
        """
        time_limit_reached = self.steps >= self.episode_length

        # TODO: Add proper handling of low battery state such as landing for battery swap
        any_low_battery = any(
            state_dicts[agent]["battery"] < self.battery_threshold
            for agent in self.agents
        )

        # TODO: consider what needs to be done for the specific drones breaking z bounderies (e.g. battery change)
        z_violation_agents = self._agents_with_z_boundary_violation(state_dicts)
        any_z_violation = len(z_violation_agents) > 0

        if any_z_violation:
            print(
                "[TASK] Z boundary violation detected for agents: "
                f"{z_violation_agents}. Truncating episode."
            )

        truncate_all = (
            time_limit_reached
            or any_low_battery
            or any_z_violation
        )

        return {
            agent: truncate_all
            for agent in self.agents
        }

    def _get_infos(
        self,
        state_dicts: Optional[Dict[str, Dict[str, Any]]] = None,
        denormalized_actions: Optional[Dict[str, List[float]]] = None,
        normalized_actions: Optional[Dict[str, np.ndarray]] = None,
        old_positions: Optional[Dict[str, List[float]]] = None,
        new_positions: Optional[Dict[str, List[float]]] = None,
        action_filter_infos: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        infos = {}


        # TODO: Check this function in SARL and see what .get_position() is
        if state_dicts is None:
            positions = {
                agent: self.drones[agent].get_position()
                for agent in self.agents
            }
            state_dicts = self._generate_state_dicts(positions)

        for agent in self.agents:
            distance = state_dicts[agent]["distance_to_target"]
            reached = distance <= self.target_threshold

            infos[agent] = {
                "target_position": self.target_positions.get(agent),
                "distance_to_target": distance,
                "target_reached": reached,
                "team_success": self._all_targets_reached(state_dicts),
                "in_boundaries": state_dicts[agent]["in_boundaries"],
                "battery": state_dicts[agent]["battery"],
            }

            if denormalized_actions is not None:
                infos[agent]["denormalized_action"] = denormalized_actions.get(agent)

            if normalized_actions is not None:
                action = normalized_actions.get(agent)
                infos[agent]["normalized_action"] = (
                    action.tolist() if action is not None else None
                )

            if old_positions is not None:
                infos[agent]["old_position"] = old_positions.get(agent)

            if new_positions is not None:
                infos[agent]["new_position"] = new_positions.get(agent)
            
            if action_filter_infos is not None:
                infos[agent]["action_filter_info"] = action_filter_infos.get(agent)

        return infos

    def _distance_to_target(self, agent: str, position: List[float]) -> float:
        target_position = self.target_positions.get(agent)

        if target_position is None:
            return float("inf")
        
        current_xy = np.array(position[:2], dtype=np.float32)
        target_xy = np.array(target_position[:2], dtype=np.float32)
        return float(np.linalg.norm(current_xy - target_xy))

    def _all_targets_reached(
        self,
        state_dicts: Dict[str, Dict[str, Any]],
    ) -> bool:
        if not self.agents:
            return False

        return all(
            state_dicts[agent]["distance_to_target"] <= self.target_threshold
            for agent in self.agents
        )

    def _get_global_state(self) -> np.ndarray:
        """
        Centralised-training state.

        Contains:
        - all drone positions
        - all drone velocities
        - all target positions
        """

        # TODO: Ask Henry what the order of the global state should be
        state_parts = []

        for agent in self.possible_agents:
            position = self.drones[agent].get_position()
            state_parts.append(self._normalize_position_xy(position))

        for agent in self.possible_agents:
            velocity = self.drones[agent].get_velocity()
            state_parts.append(self._normalize_velocity_xy(velocity))

        for agent in self.possible_agents:
            target_position = self.target_positions.get(
                agent,
                [0.0, 0.0, self.reset_height],
            )
            state_parts.append(self._normalize_position_xy(target_position))

        return np.concatenate(state_parts).astype(np.float32)  

    def _render_task_specific_info(self):
        print("Targets:")

        for agent in self.possible_agents:
            target = self.target_positions.get(agent)
            if target is None:
                print(f"  {agent}: target=None")
                continue

            position = self.drones[agent].get_position()
            distance = self._distance_to_target(agent, position)

            print(  
                f"  {agent}: "
                f"target={[round(v, 3) for v in target[:2]]}, "
                f"distance={distance:.3f}"
            )