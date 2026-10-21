from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

class ResetPlanner:
    """Moves a fleet of physical drones to their reset positions safely.

    **Precondition:** every drone is airborne and under position control. The
    planner commands targets; it never arms, takes off, lands, or power-cycles.

    **Postcondition:** every drone is within tolerance of its assigned reset
    position, or :class:`ResetPlanner.InterventionRequired` was raised.

    **Escalation:** the planner does not know how to ask a human for help, so it
    raises. The environment owns landing, disarming, prompting, and retrying.


    Clearance ladder
    --------------------
    ======================  ==============================  ====================================
    Quantity                Definition                      Answers
    ======================  ==============================  ====================================
    ``safety_distance``     hard floor                      "how close is a collision risk?"
    ``hold_error``          max position_error_threshold    "how far off can a holding drone be?"
    ``obstacle_clearance``  safety + hold                   "may I command a waypoint here?"
    ``path_clearance``      safety + hold + transit         "may I fly this segment?"
    ``reserved_clearance``  path + hold                     "may I park on a reserved slot?"
    ``escape_clearance``    path + escape margin            "is this pair separated *enough*?"
    ``slot_clearance``      escape                          "are two reset slots far enough apart?"
    ======================  ==============================  ====================================
    """

    # ------------------------------------------------------------------
    # Escalation types
    # ------------------------------------------------------------------

    class Error(RuntimeError):
        """Base class for every reset planning failure."""

    class ConfigurationError(Error, ValueError):
        """Reset positions violate the planner's own safety model.

        Raised during validation, before anything moves. A programming or
        configuration fault; slots too close, outside the flight volume, an
        agent missing. Not a hardware fault.
        """

    class MoveFailure(Error):
        """A commanded move did not complete within its timeout.

        The planner always commands the drone to hold its current position
        before raising. Whether that hold took effect is checked separately by
        :meth:`ResetPlanner._confirm_holding`.
        """

        def __init__(
            self,
            agent: str,
            target: list[float],
            current_position: list[float],
            timeout: float,
        ) -> None:
            super().__init__(
                f"[{agent}] failed to reach {[round(v, 3) for v in target]} "
                f"within {timeout:.1f}s. Currently at "
                f"{[round(v, 3) for v in current_position]}."
            )
            self.agent = agent
            self.target = list(target)
            self.current_position = list(current_position)
            self.timeout = timeout

    class InterventionRequired(Error):
        """Automatic recovery is exhausted; a human must reposition drones.

        Args:
            agents: Every agent the operator needs to deal with.
            reason: Short phrase for the operator prompt and the logs.
        """

        def __init__(self, agents: Iterable[str], reason: str) -> None:
            self.agents = sorted(set(agents))
            self.reason = reason
            super().__init__(f"{reason}: {self.agents}")

    class TrackingLost(InterventionRequired):
        """Position data went stale during the reset."""

        def __init__(self, agents: Iterable[str], max_age: float) -> None:
            super().__init__(agents, f"position data older than {max_age:.2f}s")
            self.max_age = max_age

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        drones: Mapping[str, object] | Callable[[], Mapping[str, object]],
        *,
        # Flight volume. Each may be a value or a zero-argument callable.
        hover_height: float | Callable[[], float],
        xy_limit: float | Callable[[], float],
        z_min: float | Callable[[], float],
        z_max: float | Callable[[], float],
        boundary_margin: float = 0.05,
        # Clearances
        safety_distance: float = 0.30,
        transit_margin: float = 0.10,
        escape_margin: float = 0.05,
        # Retry limits
        max_moves_per_drone: int = 4,
        max_passes: int = 2,
        max_displacements: int = 2,
        # Timing
        move_timeout: float = 30.0,
        hold_settle_time: float = 1.0,
        hold_tolerance_scale: float = 2.0,
        # Search
        search_step: float = 0.10,
        max_search_distance: float = 1.0,
        along_offsets: tuple[float, ...] = (0.0, -0.2, 0.2, -0.4, 0.4),
        escape_progress_epsilon: float = 0.02,
        # Behaviour
        position_max_age: float | None = None,
    ) -> None:
        """
        Args:
            drones: Every drone, keyed by agent name, or a callable returning that mapping.
            hover_height: Height to which all drones move to before moving to xy 
                reset positions to reduce the risk of collisions. Must be within 
                ``[z_min, z_max]``.
            xy_limit, z_min, z_max: Safe flight volume boundaries.
            boundary_margin: Minimum distance target positions must be from the
                boundary ``xy_limit``.
            safety_distance: Hard minimum separation between drones.
            transit_margin: Additional buffer for moving drones, to account for 
                overshoot and position error.
            escape_margin: Drones that are initially too close to each other are 
                separated by ``path_clearance + escape_margin``. This is the 
                minimum separation that must be achieved before the planner 
                considers the pair to be "safely separated" and moves on to the next step.
            max_moves_per_drone: Waypoints per drone per pass, including intermediate 
                points and the final direct move.
            max_passes: Attempts at the full pending set before escalating error.
            max_displacements: Times one drone may be pushed off its position
                before escalating.                
            move_timeout: Seconds to wait for a drone to reach a commanded waypoint 
                before raising :class:`MoveFailure`.
            hold_settle_time: Time waited for a drone to settle before checking 
                its position against the requested hold position.
            hold_tolerance_scale: Multiple of ``hold_error`` allowed when
                confirming a hold or an arrival.
            search_step: Radial increment for waypoint searches.
            max_search_distance: Largest offset considered for a waypoint.
            along_offsets: Offsets along a blocked path at which to try lateral
                detours.
            escape_progress_epsilon: Minimum separation gained per escape move
                before declaring no progress to stop drone from making no progress 
                and oscillating.
            position_max_age: Maximum age of position data in seconds, or None
                to disable.            
        """
        self._drones = drones
        self._hover_height = hover_height
        self._xy_limit = xy_limit
        self._z_min = z_min
        self._z_max = z_max

        self.boundary_margin = boundary_margin

        self.safety_distance = safety_distance
        self.transit_margin = transit_margin
        self.escape_margin = escape_margin
        self.max_moves_per_drone = max_moves_per_drone
        self.max_passes = max_passes
        self.max_displacements = max_displacements

        self.move_timeout = move_timeout
        self.hold_settle_time = hold_settle_time
        self.hold_tolerance_scale = hold_tolerance_scale

        self.search_step = search_step
        self.max_search_distance = max_search_distance
        self.along_offsets = tuple(along_offsets)
        self.escape_progress_epsilon = escape_progress_epsilon

        self.position_max_age = position_max_age

        self._displacement_counts: dict[str, int] = {}
        self._known_agents: set[str] | None = None

    # ------------------------------------------------------------------
    # Live values -- resolved on every access, never snapshotted
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(value):
        return value() if callable(value) else value

    @property
    def drones(self) -> Mapping[str, object]:
        return self._resolve(self._drones)

    @property
    def hover_height(self) -> float:
        return float(self._resolve(self._hover_height))

    @property
    def xy_limit(self) -> float:
        return float(self._resolve(self._xy_limit))

    @property
    def z_min(self) -> float:
        return float(self._resolve(self._z_min))

    @property
    def z_max(self) -> float:
        return float(self._resolve(self._z_max))

    # ------------------------------------------------------------------
    # Geometry 
    # ------------------------------------------------------------------

    @staticmethod
    def as_xy(position: list[float]) -> np.ndarray:
        return np.asarray(position[:2], dtype=float)

    @staticmethod
    def distance(position_1: list[float], position_2: list[float]) -> float:
        """Euclidean distance over however many components are supplied."""
        return float(
            np.linalg.norm(
                np.asarray(position_1, dtype=float)
                - np.asarray(position_2, dtype=float)
            )
        )

    @classmethod
    def distance_xy(cls, position_1: list[float], position_2: list[float]) -> float:
        return cls.distance(position_1[:2], position_2[:2])

    @staticmethod
    def point_to_segment_distance(
        point: list[float], segment_start: list[float], segment_end: list[float]
    ) -> float:
        """Shortest distance from a point to a line segment.

        Args:
            point: Position of the stationary obstacle.
            segment_start: Where the moving drone starts.
            segment_end: Where the moving drone is commanded to.

        Returns:
            Distance in metres. 
        """
        point_array = np.asarray(point, dtype=float)
        start_array = np.asarray(segment_start, dtype=float)
        end_array = np.asarray(segment_end, dtype=float)

        segment = end_array - start_array
        segment_length_squared = float(np.dot(segment, segment))

        if segment_length_squared == 0.0:
            return float(np.linalg.norm(point_array - start_array))

        t = float(np.dot(point_array - start_array, segment) / segment_length_squared)
        t = min(1.0, max(0.0, t))

        return float(np.linalg.norm(point_array - (start_array + t * segment)))

    @classmethod
    def point_to_xy_segment_distance(
        cls, point: list[float], segment_start: list[float], segment_end: list[float]
    ) -> float:
        return cls.point_to_segment_distance(
            point[:2], segment_start[:2], segment_end[:2]
        )

    @classmethod
    def closest_point_on_xy_segment(
        cls, point: list[float], segment_start: list[float], segment_end: list[float]
    ) -> np.ndarray:
        """XY point on a xy segment that is nearest to ``point``. Used to seed 
        detours when a drone is near another drones planned path.
        """
        start_xy = cls.as_xy(segment_start)
        segment = cls.as_xy(segment_end) - start_xy
        segment_length_squared = float(np.dot(segment, segment))

        if segment_length_squared == 0.0:
            return start_xy

        t = float(np.dot(cls.as_xy(point) - start_xy, segment) / segment_length_squared)
        return start_xy + min(1.0, max(0.0, t)) * segment

    @classmethod
    def evaluate_xy_segment(
        cls,
        start: list[float],
        end: list[float],
        obstacles: Mapping[str, list[float]],
        required_clearance: float,
    ) -> tuple[int, float, str | None]:
        """Score an XY path against all stationary drones.

        Returns:
            ``(blocker_count, minimum_clearance, closest_blocker)``.
        """
        blocker_count = 0
        minimum_clearance = float("inf")
        closest_blocker: str | None = None
        closest_blocking_distance = float("inf")

        for agent in sorted(obstacles):
            obstacle_distance = cls.point_to_xy_segment_distance(
                obstacles[agent], start, end
            )
            minimum_clearance = min(minimum_clearance, obstacle_distance)

            if obstacle_distance < required_clearance:
                blocker_count += 1
                if obstacle_distance < closest_blocking_distance:
                    closest_blocking_distance = obstacle_distance
                    closest_blocker = agent

        return blocker_count, minimum_clearance, closest_blocker

    @classmethod
    def is_xy_segment_safe(
        cls,
        start: list[float],
        end: list[float],
        obstacles: Mapping[str, list[float]],
        path_clearance: float,
        hard_minimum: float,
        allow_initial_overlap: bool = False,
    ) -> tuple[bool, str | None]:
        """Decide whether a straight XY move is safe to command.

        The move is considered safe if no obstacle comes within ``path_clearance``
        of the segment, except for a special case where the drone starts already
        too close to an obstacle and is intentionally moving away from it.        

        Args:
            start: Current XY position of the moving drone.
            end: Target XY position for the move.
            obstacles: Other drones treated as static obstacles.
            path_clearance: Minimum horizontal separation required during transit.
            hard_minimum: Absolute minimum allowed separation before escape logic.
            allow_initial_overlap: If True, permit a move that starts inside the
                hard minimum when the goal is to move away from the obstacle.

        Returns:
            ``(True, None)`` if safe, else ``(False, blocking_agent)``.
        """
        start_xy = cls.as_xy(start)
        end_xy = cls.as_xy(end)
        movement = end_xy - start_xy

        blocking_agent: str | None = None
        closest_blocking_distance = float("inf")

        for agent in sorted(obstacles):
            obstacle_xy = cls.as_xy(obstacles[agent])
            initial_distance = float(np.linalg.norm(start_xy - obstacle_xy))

            if initial_distance < path_clearance:
                if initial_distance < hard_minimum and not allow_initial_overlap:
                    return False, agent
                if float(np.linalg.norm(end_xy - obstacle_xy)) < path_clearance:
                    return False, agent
                if float(np.dot(start_xy - obstacle_xy, movement)) < 0.0:
                    return False, agent
                continue

            obstacle_distance = cls.point_to_xy_segment_distance(
                obstacles[agent], start, end
            )
            if (
                obstacle_distance < path_clearance
                and obstacle_distance < closest_blocking_distance
            ):
                blocking_agent = agent
                closest_blocking_distance = obstacle_distance

        return blocking_agent is None, blocking_agent

    @classmethod
    def is_xy_position_clear(
        cls,
        position: list[float],
        obstacles: Mapping[str, list[float]],
        minimum_clearance: float,
    ) -> bool:
        """Whether a waypoint is far enough from every stationary drone."""
        return all(
            cls.distance_xy(position, obstacle) >= minimum_clearance
            for obstacle in obstacles.values()
        )

    @staticmethod
    def xy_search_directions() -> list[np.ndarray]:
        """Eight unit directions on the XY plane, for outward waypoint search."""
        directions = []
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                if x == 0.0 and y == 0.0:
                    continue
                direction = np.array([x, y], dtype=float)
                directions.append(direction / np.linalg.norm(direction))
        return directions

    # ------------------------------------------------------------------
    # Clearances
    # ------------------------------------------------------------------

    @property
    def usable_xy_limit(self) -> float:
        return self.xy_limit - self.boundary_margin

    @property
    def hold_error(self) -> float:
        """Worst-case steady-state position error across the fleet."""
        return max(
            (float(drone.position_error_threshold) for drone in self.drones.values()),
            default=0.0,
        )
    
    @property
    def obstacle_clearance(self) -> float:
        """May I command a waypoint here, given a drone is holding nearby?
        ``safety_distance + hold_error``
        ``safety_distance`` is the hard minimum, and ``hold_error`` is the 
        worst-case steady-state position error across the fleet."""
        return self.safety_distance + self.hold_error

    @property
    def path_clearance(self) -> float:
        """May I fly this segment past a holding drone?
        ``safety_distance + hold_error + transit_margin``
        ``safety_distance`` is the hard minimum, ``hold_error`` is the 
        worst-case, ``transit_margin`` is an extra buffer for moving drones 
        to account for overshoot and position error."""
        return self.safety_distance + self.hold_error + self.transit_margin

    @property
    def reserved_clearance(self) -> float:
        """Does this position clear other drones reset positions/waypoints?
        ``path_clearance + hold_error``
        ``path_clearance`` is the minimum separation required during transit, 
        and ``hold_error`` is the worst-case steady-state position error."""
        return self.path_clearance + self.hold_error

    @property
    def escape_clearance(self) -> float:
        """Is this pair separated enough that it will not be re-flagged?
        ``path_clearance + escape_margin``
        ``path_clearance`` is the minimum separation required during transit, 
        and ``escape_margin`` is an extra buffer for escaping drones."""
        return self.path_clearance + self.escape_margin

    @property
    def slot_clearance(self) -> float:
        """Are two reset slots far enough apart to rest at?
        ``escape_clearance`` is the minimum separation that must be achieved"""
        return self.escape_clearance

    def _check_clearance_ladder(self) -> None:
        """The whole planner depends on this ordering."""
        assert (
            self.safety_distance
            <= self.obstacle_clearance
            <= self.path_clearance
            < self.escape_clearance
        )
        assert self.path_clearance <= self.reserved_clearance

    def describe_clearances(self) -> str:
        return (
            f"hold={self.hold_error:.3f} obstacle={self.obstacle_clearance:.3f} "
            f"path={self.path_clearance:.3f} reserved={self.reserved_clearance:.3f} "
            f"escape={self.escape_clearance:.3f}"
        )

    # ------------------------------------------------------------------
    # Reset position validation
    # ------------------------------------------------------------------

    def validate_reset_positions(self, reset_positions: Mapping[str, list[float]]) -> None:
        """Check a reset layout against the planner's own safety model.

        Run this every time positions are generated, not once at construction.
        Tasks that randomise positions per episode produce a new layout each
        reset, and a layout the planner considers unsafe should fail on the
        ground with a clear message rather than in the air.

        Raises:
            ResetPlanner.ConfigurationError: A slot is outside the flight
                volume, an agent is missing, or two slots are too close for
                one another.
        """
        missing = set(self.drones) - set(reset_positions)
        if missing:
            raise self.ConfigurationError(f"No reset position for: {sorted(missing)}")

        agents = sorted(reset_positions)

        for agent in agents:
            position = reset_positions[agent]
            if len(position) != 3:
                raise self.ConfigurationError(
                    f"Reset position for {agent} must be [x, y, z], got {position}"
                )
            if not self.is_within_bounds(position):
                raise self.ConfigurationError(
                    f"Reset position for {agent} outside usable flight volume: "
                    f"{position} (usable xy limit {self.usable_xy_limit:.3f})"
                )

        for index, agent in enumerate(agents):
            for other in agents[index + 1 :]:
                position = reset_positions[agent]
                other_position = reset_positions[other]
                separation = self.distance_xy(position, other_position)
                required = self.slot_clearance

                if separation < required:
                    raise self.ConfigurationError(
                        f"Reset slots for {agent} and {other} are "
                        f"{separation:.3f}m apart in XY but need {required:.3f}m. "
                        "Increase the spacing between reset slots."
                    )

    def is_within_bounds(self, position: list[float]) -> bool:
        """Whether a waypoint sits inside the usable flight volume."""
        x, y, z = position
        return (
            abs(x) <= self.usable_xy_limit
            and abs(y) <= self.usable_xy_limit
            and self.z_min <= z <= self.z_max
        )

    # ------------------------------------------------------------------
    # Position access
    # ------------------------------------------------------------------

    def position(self, agent: str) -> list[float]:
        """Current position, with an optional freshness guard.

        ``get_position`` returns the last known value, so without this check a
        stale position leaves the planner routing other drones around a position
        the drone left some time ago.
        """
        drone = self.drones[agent]

        if self.position_max_age is not None:
            updated_at = getattr(drone, "last_position_update_time", None)
            if (
                updated_at is None
                or (time.monotonic() - updated_at) > self.position_max_age
            ):
                raise self.TrackingLost([agent], self.position_max_age)

        return list(drone.get_position())

    def positions(self, agents: Sequence[str] | None = None) -> dict[str, list[float]]:
        return {agent: self.position(agent) for agent in (agents or self.drones)}

    def obstacles_for(self, agent: str) -> dict[str, list[float]]:
        """Every other drone's position, treated as a static obstacle."""
        return {other: self.position(other) for other in self.drones if other != agent}

    def _raise_if_emergency(self) -> None:
        triggered = [
            agent
            for agent, drone in self.drones.items()
            if getattr(drone, "emergency_event", None) is not None
            and drone.emergency_event.is_set()
        ]
        if triggered:
            raise self.InterventionRequired(triggered, "emergency state during reset")

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(self, agent: str, target: list[float], purpose: str = "") -> None:
        """Command a move and wait for it, verifying the drone is really there.

        Raises:
            ResetPlanner.MoveFailure: Timed out, or arrived and left again. The
                drone is commanded to hold where it is before raising.
            ResetPlanner.InterventionRequired: An emergency fired mid-move.
        """
        drone = self.drones[agent]
        start = self.position(agent)
        timeout = self.move_timeout

        print(
            f"[Reset Planner] {agent} -> {[round(v, 3) for v in target]} "
            f"({purpose or 'move'}, timeout {timeout:.1f}s)"
        )

        drone.set_target_position(*target)

        deadline = time.monotonic() + timeout
        reached = False
        while time.monotonic() < deadline:
            if drone.at_reset_position.wait(timeout=0.1):
                reached = True
                break
            self._raise_if_emergency()

        if not reached:
            current = self.position(agent)
            self._hold_position(agent)
            raise self.MoveFailure(agent, target, current, timeout)

        settled = self.position(agent)
        error = self.distance(settled, target)

        if error > self.hold_error * self.hold_tolerance_scale:
            print(
                f"[Reset Planner] {agent} signalled arrival at {target} but is "
                f"{error:.3f}m away"
            )
            self._hold_position(agent)
            raise self.MoveFailure(agent, target, settled, timeout)

    def _hold_position(self, agent: str) -> list[float]:
        """Command a drone where it currently is.

        Called after any failed move; a drone still chasing an unreachable
        target keeps drifting, and every other drone's path safety is computed
        against the assumption that it is stationary.
        """
        current = self.position(agent)
        self.drones[agent].set_target_position(*current)
        return current

    def _confirm_holding(self, agent: str) -> bool:
        """Whether a drone commanded to hold is actually holding."""
        hold_point = self._hold_position(agent)
        time.sleep(self.hold_settle_time)

        drift = self.distance(self.position(agent), hold_point)

        if drift > self.hold_error * self.hold_tolerance_scale:
            print(f"[Reset Planner] {agent} drifted {drift:.3f}m while commanded to hold")
            return False
        return True

    # ------------------------------------------------------------------
    # Phase 1 -- separation
    # ------------------------------------------------------------------

    def separate_close_drones(self) -> None:
        """Push apart any initial drones closer than the path clearance."""
        max_iterations = len(self.drones) * 3

        for _ in range(max_iterations):
            self._raise_if_emergency()

            pair = self._closest_offending_pair(self.positions(), self.path_clearance)
            if pair is None:
                return

            agent_1, agent_2, separation = pair
            print(
                f"[Reset Planner] {agent_1} and {agent_2} are {separation:.3f}m apart "
                f"(need {self.path_clearance:.3f}m); separating"
            )

            if not self._escape_one_of(agent_1, agent_2):
                raise self.InterventionRequired(
                    [agent_1, agent_2], "no safe separation move available"
                )

            new_separation = self.distance_xy(
                self.position(agent_1), self.position(agent_2)
            )
            if new_separation <= separation + self.escape_progress_epsilon:
                raise self.InterventionRequired(
                    [agent_1, agent_2],
                    f"separation not converging ({separation:.3f}m -> "
                    f"{new_separation:.3f}m)",
                )

        raise self.InterventionRequired(
            list(self.drones), "exceeded separation attempts"
        )

    def _closest_offending_pair(
        self, positions: Mapping[str, list[float]], minimum: float
    ) -> tuple[str, str, float] | None:
        """Worst offending pair, so the most urgent conflict is fixed first."""
        worst: tuple[str, str, float] | None = None
        agents = sorted(positions)

        for index, agent in enumerate(agents):
            for other in agents[index + 1 :]:
                separation = self.distance_xy(positions[agent], positions[other])
                if separation < minimum and (worst is None or separation < worst[2]):
                    worst = (agent, other, separation)

        return worst

    def _escape_one_of(self, agent_1: str, agent_2: str) -> bool:
        """Try to move either member of a too-close pair to open space."""
        for moving_agent in (agent_1, agent_2):
            escape = self.find_clear_position(
                start=self.position(moving_agent),
                obstacles=self.obstacles_for(moving_agent),
                reserved_positions=[],
                minimum_clearance=self.escape_clearance,
                allow_initial_overlap=True,
            )
            if escape is None:
                continue

            self.move(moving_agent, escape, purpose="separation")
            return True

        return False

    # ------------------------------------------------------------------
    # Waypoint search
    # ------------------------------------------------------------------

    def find_clear_position(
        self,
        start: list[float],
        obstacles: Mapping[str, list[float]],
        reserved_positions: Sequence[list[float]],
        minimum_clearance: float,
        allow_initial_overlap: bool = False,
    ) -> list[float] | None:
        """Nearest open spot around ``start``, maximising separation.

        Searches outward in fixed increments and, at the first radius yielding
        any candidate, returns the best of them rather than the first found.

        Args:
            start: Where the drone currently is.
            obstacles: Other drones treated as static obstacles.
            reserved_positions: Positions that are temporarily off-limits.
            minimum_clearance: How far away the candidate must be from every
                obstacle and reserved position.
            allow_initial_overlap: If True, permit a candidate that starts inside the
                hard minimum when the goal is to move away from the obstacle.
        """
        start_array = np.asarray(start, dtype=float)
        directions = self.xy_search_directions()

        search_distance = self.search_step
        while search_distance <= self.max_search_distance:
            best: list[float] | None = None
            best_clearance = -float("inf")

            for direction in directions:
                candidate_xy = start_array[:2] + search_distance * direction
                candidate = [
                    float(candidate_xy[0]),
                    float(candidate_xy[1]),
                    float(start_array[2]),
                ]

                if not self._is_candidate_safe(
                    start=start,
                    candidate=candidate,
                    obstacles=obstacles,
                    reserved_positions=reserved_positions,
                    minimum_clearance=minimum_clearance,
                    allow_initial_overlap=allow_initial_overlap,
                ):
                    continue

                achieved = min(
                    (self.distance_xy(candidate, o) for o in obstacles.values()),
                    default=float("inf"),
                )
                if achieved > best_clearance:
                    best, best_clearance = candidate, achieved

            if best is not None:
                return best

            search_distance += self.search_step

        return None

    def _is_candidate_safe(
        self,
        start: list[float],
        candidate: list[float],
        obstacles: Mapping[str, list[float]],
        reserved_positions: Sequence[list[float]],
        minimum_clearance: float,
        allow_initial_overlap: bool = False,
    ) -> bool:
        """Return whether a candidate waypoint is clear and safe to move toward.

        The waypoint must stay within the flight volume, remain outside the
        clearance radius of all obstacles and reserved slots, and not pass through
        an unsafe XY path segment.
        """
        if not self.is_within_bounds(candidate):
            return False

        if not self.is_xy_position_clear(candidate, obstacles, minimum_clearance):
            return False

        if any(
            self.distance_xy(candidate, reserved) < self.reserved_clearance
            for reserved in reserved_positions
        ):
            return False

        safe, _ = self.is_xy_segment_safe(
            start=start,
            end=candidate,
            obstacles=obstacles,
            path_clearance=self.path_clearance,
            hard_minimum=self.safety_distance,
            allow_initial_overlap=allow_initial_overlap,
        )
        return safe

    def find_detour(
        self,
        start: list[float],
        target: list[float],
        obstacles: Mapping[str, list[float]],
        blocking_agent: str,
        reserved_positions: Sequence[list[float]],
        visited_positions: Sequence[list[float]],
    ) -> list[float] | None:
        """Intermediate waypoint that gets around a blocking drone.

        Candidates are offsets perpendicular to the blocked path, searched at
        increasing lateral distance and several points along it, then ranked by:
        1. obstacles still between the candidate and the target
        2. whether the candidate sits on another drone's reserved slot 
        3. progress toward the target, then total path length, then clearance.
        """
        start_xy = self.as_xy(start)
        target_xy = self.as_xy(target)

        segment = target_xy - start_xy
        segment_length = float(np.linalg.norm(segment))
        if segment_length == 0.0:
            return None

        direction = segment / segment_length
        perpendicular = np.array([-direction[1], direction[0]], dtype=float)
        anchor = self.closest_point_on_xy_segment(
            obstacles[blocking_agent], start, target
        )

        revisit_distance = 2.0 * self.hold_error
        candidates: list[tuple] = []

        search_distance = self.path_clearance
        while search_distance <= self.max_search_distance:
            for along_offset in self.along_offsets:
                for side in (-1.0, 1.0):
                    candidate_xy = (
                        anchor
                        + along_offset * direction
                        + side * search_distance * perpendicular
                    )
                    candidate = [
                        float(candidate_xy[0]),
                        float(candidate_xy[1]),
                        float(target[2]),
                    ]

                    if any(
                        self.distance_xy(candidate, previous) < revisit_distance
                        for previous in visited_positions
                    ):
                        continue

                    if not self._is_candidate_safe(
                        start=start,
                        candidate=candidate,
                        obstacles=obstacles,
                        reserved_positions=[],
                        minimum_clearance=self.obstacle_clearance,
                    ):
                        continue

                    blockers, clearance, _ = self.evaluate_xy_segment(
                        candidate, target, obstacles, self.path_clearance
                    )
                    blocks_reserved = int(
                        any(
                            self.distance_xy(candidate, reserved)
                            < self.reserved_clearance
                            for reserved in reserved_positions
                        )
                    )
                    remaining = float(np.linalg.norm(target_xy - candidate_xy))

                    candidates.append(
                        (
                            blockers,
                            blocks_reserved,
                            remaining,
                            float(np.linalg.norm(candidate_xy - start_xy)) + remaining,
                            -clearance,
                            candidate,
                        )
                    )

            search_distance += self.search_step

        if not candidates:
            return None

        candidates.sort(key=lambda entry: entry[:5])
        return candidates[0][5]

    # ------------------------------------------------------------------
    # Phases 2 -- Move to reset height
    # ------------------------------------------------------------------
  
    def _move_all_to_hover_height(self) -> None:
        """Bring every drone to the shared reset altitude.

        Sequential and purely vertical. XY separation is done first, so
        no horizontal conflict can occur from these moves.
        """
        for agent in sorted(self.drones):
            current = self.position(agent)
            if abs(current[2] - self.hover_height) <= self.hold_error:
                continue
            self.move(
                agent,
                [current[0], current[1], self.hover_height],
                purpose="hover height",
            )

    # ------------------------------------------------------------------
    # Phase 3 -- XY reset
    # ------------------------------------------------------------------

    def _reset_one_drone_xy(
        self,
        agent: str,
        reset_positions: Mapping[str, list[float]],
        outcome: dict,
    ) -> bool:
        """Move one drone to its slot at the hover height.

        Returns:
            True on arrival, False if the drone should be retried in a later
            pass. Never leaves the drone chasing a target it cannot reach.
        """
        target = [
            reset_positions[agent][0],
            reset_positions[agent][1],
            self.hover_height,
        ]
        reserved = [reset_positions[other] for other in self.drones if other != agent]
        revisit_distance = 2.0 * self.hold_error
        visited: list[list[float]] = []

        for move_number in range(1, self.max_moves_per_drone + 1):
            current = self.position(agent)

            if any(
                self.distance_xy(current, previous) < revisit_distance
                for previous in visited
            ):
                print(
                    f"[Reset Planner] {agent} is oscillating after {move_number - 1} moves; "
                    "deferring"
                )
                self._hold_position(agent)
                return False

            visited.append(list(current))
            obstacles = self.obstacles_for(agent)

            direct_safe, blocking_agent = self.is_xy_segment_safe(
                start=current,
                end=target,
                obstacles=obstacles,
                path_clearance=self.path_clearance,
                hard_minimum=self.safety_distance,
            )

            if direct_safe:
                self.move(agent, target, purpose=f"reset {move_number}")
                outcome["moves_commanded"] += 1
                return True

            detour = self.find_detour(
                start=current,
                target=target,
                obstacles=obstacles,
                blocking_agent=blocking_agent,
                reserved_positions=reserved,
                visited_positions=visited,
            )

            if detour is None:
                print(
                    f"[Reset Planner] No safe detour for {agent} around {blocking_agent}; "
                    "deferring"
                )
                self._hold_position(agent)
                return False

            self.move(agent, detour, purpose=f"detour {move_number}")
            outcome["moves_commanded"] += 1
            outcome["intermediate_moves"] += 1

        print(
            f"[Reset Planner] {agent} exceeded {self.max_moves_per_drone} moves; deferring"
        )
        self._hold_position(agent)
        return False

    def _clear_slot(
        self, agent: str, reset_positions: Mapping[str, list[float]]
    ) -> list[str]:
        """Move any drone sitting on ``agent``'s slot out of the way.

        Prefers sending the blocker to its own slot. That completes its reset
        instead of creating a new temporary obstacle for other drones.

        Returns:
            Agents that were displaced. The caller must re-queue any that had
            already finished, or the pass will report success with a drone
            parked somewhere it does not belong.

        Raises:
            ResetPlanner.InterventionRequired: The blocker cannot be moved
                anywhere safe.
        """
        slot = reset_positions[agent]
        displaced: list[str] = []

        for other in sorted(self.drones):
            if other == agent:
                continue

            other_position = self.position(other)
            if self.distance_xy(slot, other_position) >= self.path_clearance:
                continue

            print(
                f"[Reset Planner] {other} is blocking {agent}'s slot "
                f"({self.distance_xy(slot, other_position):.3f}m away)"
            )

            obstacles = self.obstacles_for(other)
            reserved = [
                reset_positions[reserved_agent]
                for reserved_agent in self.drones
                if reserved_agent != other
            ]

            own_slot = [
                reset_positions[other][0],
                reset_positions[other][1],
                self.hover_height,
            ]

            own_slot_safe, _ = self.is_xy_segment_safe(
                start=other_position,
                end=own_slot,
                obstacles=obstacles,
                path_clearance=self.path_clearance,
                hard_minimum=self.safety_distance,
            )
            if own_slot_safe and self.is_xy_position_clear(
                own_slot, obstacles, self.obstacle_clearance
            ):
                destination: list[float] | None = own_slot
                print(f"[Reset Planner] Sending {other} to its own slot instead of parking it")
            else:
                destination = self.find_clear_position(
                    start=other_position,
                    obstacles=obstacles,
                    reserved_positions=reserved,
                    minimum_clearance=self.obstacle_clearance,
                )

            if destination is None:
                raise self.InterventionRequired(
                    [agent, other], f"cannot clear {agent}'s reset slot"
                )

            self.move(other, destination, purpose="clear slot")
            displaced.append(other)

        return displaced

    def _run_xy_passes(
        self, reset_positions: Mapping[str, list[float]], outcome: dict
    ) -> None:
        """Work the pending set until everyone is over their slot, or escalate."""
        pending = sorted(self.drones)

        for pass_number in range(1, self.max_passes + 1):
            if not pending:
                return

            print(
                f"[Reset Planner] XY reset pass {pass_number}/{self.max_passes}, "
                f"pending: {pending}"
            )

            failed: list[str] = []
            unattempted = list(pending)

            while unattempted:
                self._raise_if_emergency()

                # Recompute each time. Earlier moves changed the geometry, and
                # nearest-first keeps drones from crossing the whole arena
                # while others wait.
                agent = min(
                    unattempted,
                    key=lambda candidate: self.distance_xy(
                        self.position(candidate), reset_positions[candidate]
                    ),
                )
                unattempted.remove(agent)

                try:
                    displaced = self._clear_slot(agent, reset_positions)
                    succeeded = self._reset_one_drone_xy(agent, reset_positions, outcome)
                except self.MoveFailure as failure:
                    if not self._confirm_holding(failure.agent):
                        raise self.InterventionRequired(
                            [failure.agent], "drone will not hold position"
                        ) from failure

                    print(f"[Reset Planner] {agent}: {failure} -- deferring to next pass")
                    for deferred in (failure.agent, agent):
                        if deferred not in failed:
                            failed.append(deferred)
                    continue

                if not succeeded:
                    failed.append(agent)

                self._requeue_displaced(displaced, unattempted, failed, outcome)

            if not failed:
                print(f"[Reset Planner] All drones reset during pass {pass_number}")
                return

            print(f"[Reset Planner] Pass {pass_number} deferred: {failed}")
            pending = failed

        raise self.InterventionRequired(pending, "automatic XY reset failed")

    def _requeue_displaced(
        self,
        displaced: Sequence[str],
        unattempted: list[str],
        failed: list[str],
        outcome: dict,
    ) -> None:
        """Put drones pushed off their slots back in the queue."""
        for agent in displaced:
            if agent in unattempted or agent in failed:
                continue

            count = self._displacement_counts.get(agent, 0) + 1
            self._displacement_counts[agent] = count
            outcome["displacements"][agent] = count

            if count > self.max_displacements:
                raise self.InterventionRequired(
                    [agent], f"displaced from its slot {count} times"
                )

            print(f"[Reset Planner] Re-queuing {agent} after displacement {count}")
            unattempted.append(agent)

    # ------------------------------------------------------------------
    # Phases 4 -- Move to final heights
    # ------------------------------------------------------------------

    def _verify_all_at_slots(self, reset_positions: Mapping[str, list[float]]) -> None:
        """Precondition for the final vertical move: everyone over their own slot.

        The descent skips collision checking because a purely vertical move
        cannot cause a horizontal collision.
        """
        tolerance = self.hold_error * self.hold_tolerance_scale
        stragglers = [
            agent
            for agent in sorted(self.drones)
            if self.distance_xy(self.position(agent), reset_positions[agent]) > tolerance
        ]
        if stragglers:
            raise self.InterventionRequired(
                stragglers, "not over reset slot before final vertical move"
            )

    def _move_all_to_final_heights(
        self, reset_positions: Mapping[str, list[float]]
    ) -> None:
        """Apply per-drone reset heights, all at once."""
        moving = []
        for agent in sorted(self.drones):
            target = reset_positions[agent]
            if abs(self.position(agent)[2] - target[2]) <= self.hold_error:
                continue
            self.drones[agent].set_target_position(*target)
            moving.append((agent, target))

        if not moving:
            return

        deadline = time.monotonic() + self.move_timeout

        while time.monotonic() < deadline:
            self._raise_if_emergency()
            if all(
                self.drones[agent].at_reset_position.is_set() for agent, _ in moving
            ):
                return
            time.sleep(0.1)

        late = [
            agent
            for agent, _ in moving
            if not self.drones[agent].at_reset_position.is_set()
        ]
        for agent in late:
            self._hold_position(agent)
        raise self.InterventionRequired(late, "failed to reach final reset height")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def execute(self, reset_positions: Mapping[str, list[float]]) -> dict:
        """Move every drone to its reset position.

        Args:
            reset_positions: Target ``[x, y, z]`` per agent. 
            Supplied per call so tasks can vary the layout each episode.

        Returns:
            A dict with ``assigned_positions``, ``final_errors``,
            ``moves_commanded``, ``intermediate_moves``, ``displacements``,
            ``duration``.

        Raises:
            ResetPlanner.ConfigurationError: The layout is unusable. Nothing
                has moved.
            ResetPlanner.InterventionRequired: A human must reposition
                hardware. Drones are commanded to hold; the caller decides
                whether to land them.
        """
        started = time.monotonic()

        self._check_clearance_ladder()

        self.validate_reset_positions(reset_positions)
        assigned = dict(reset_positions)

        outcome: dict = {
            "assigned_positions": assigned,
            "final_errors": {},
            "moves_commanded": 0,
            "intermediate_moves": 0,
            "displacements": {},
            "duration": 0.0,
        }
        self._displacement_counts = {}

        print(f"[Reset Planner] Starting physical reset. Clearances: {self.describe_clearances()}")

        for agent in sorted(self.drones):
            self._hold_position(agent)

        self.separate_close_drones()
        self._move_all_to_hover_height()
        self.separate_close_drones()

        self._run_xy_passes(assigned, outcome)

        self._verify_all_at_slots(assigned)
        self._move_all_to_final_heights(assigned)

        outcome["final_errors"] = {
            agent: self.distance(self.position(agent), assigned[agent])
            for agent in sorted(self.drones)
        }
        outcome["duration"] = time.monotonic() - started

        print(f"[Reset Planner] Reset complete: {self.summarise(outcome)}")
        return outcome

    @staticmethod
    def summarise(outcome: Mapping) -> str:
        """One-line description of a reset, for logs."""
        worst = max(outcome["final_errors"].values(), default=0.0)
        return (
            f"{outcome['moves_commanded']} moves "
            f"({outcome['intermediate_moves']} intermediate) in "
            f"{outcome['duration']:.1f}s, worst final error {worst:.3f}m"
        )