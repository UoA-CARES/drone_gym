"""AgentTicker — opt-in background integration for software-integrated agents.

Particle/marker bodies have no physics, so between RL steps they only move when
something advances them. An :class:`AgentTicker` runs a background thread that
calls ``tick(dt)`` on each software-integrated agent at a fixed rate, giving
smooth motion and continuous Gazebo marker updates instead of one jump per step.

This is deliberately separate from :class:`~drone_gym.sim_manager.SimManager`
(which stays a pure factory): a task that wants smooth particles creates its own
ticker over the agents it pulled, and owns its lifecycle::

    self._ticker = AgentTicker([target_agent], hz=5.0)
    # reset():  self._ticker.stop(); ...teleport...; self._ticker.start()
    # close():  self._ticker.stop()

Real-drone agents (``needs_ticking == False``) are ignored — they move via their
own controller threads.
"""

import threading
import time
from typing import List


class AgentTicker:
    """Advances software-integrated agents on a background thread."""

    def __init__(self, agents: List["object"], hz: float = 5.0):
        # Only agents that need software integration are ticked.
        self.agents = [a for a in agents if getattr(a, "needs_ticking", False)]
        self.hz = hz
        self._running = False
        self._thread = None

    def start(self) -> None:
        """Start the background thread (no-op if nothing needs ticking or already running)."""
        if not self.agents or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and wait for it to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        """Tick at ``hz``.

        Elapsed time is deliberately not capped: if a Gazebo pose call lags, the
        next tick compensates so the agent keeps its true commanded velocity
        rather than silently slowing down.
        """
        dt_target = 1.0 / self.hz
        last = time.time()
        while self._running:
            now = time.time()
            elapsed = now - last
            last = now
            for agent in self.agents:
                agent.tick(elapsed)
            sleep_time = dt_target - (time.time() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)
