"""Temporary, low-overhead Crazyflie flight diagnostics.

The monitor only observes state. It does not issue flight commands, change
setpoint rates, or interfere with reset/emergency-stop decisions.
"""

import atexit
import logging
import os
import threading
import time
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import SimpleQueue

_LOG = logging.getLogger("drone_gym.flight_debug")
_LOG.setLevel(logging.INFO)
_LOG.propagate = False
_LOG_PATH = Path(
    os.environ.get(
        "DRONE_GYM_DEBUG_LOG",
        str(
            Path.cwd()
            / "flight_debug_logs"
            / f"flight_{datetime.now():%Y%m%d_%H%M%S}.log"
        ),
    )
)
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_handler = QueueHandler(SimpleQueue())
_LOG.addHandler(_handler)
_format = logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file = logging.FileHandler(_LOG_PATH, encoding="utf-8")
_console = logging.StreamHandler()
_file.setFormatter(_format)
_console.setFormatter(_format)
_listener = QueueListener(_handler.queue, _file, _console)
_listener.start()
atexit.register(_listener.stop)
print(f"[FLIGHT DIAG] Logging to {_LOG_PATH}", flush=True)


def diag(event, **fields):
    """Timestamp an event without performing disk/console I/O in the caller."""
    parts = " ".join(f"{key}={value}" for key, value in fields.items())
    _LOG.info("mono=%.6f %-18s %s", time.monotonic(), event, parts)


def _age(now, timestamp):
    return None if timestamp is None else round(max(0.0, now - timestamp), 3)


def _point(point):
    if point is None:
        return "?"
    return "(" + ",".join(f"{point[axis]:.3f}" for axis in ("x", "y", "z")) + ")"


class FlightDebugMonitor:
    def __init__(self, drone):
        self.drone = drone
        self._stop = threading.Event()
        self._link_lost = threading.Event()
        self._link_quality = None
        self._quality_at = None
        self._callback_lock = threading.RLock()
        self._attached_cf = None
        self._attached_callbacks = []
        self._thread = threading.Thread(
            target=self._run,
            name=f"{drone.agent_id}-flight-diag",
            daemon=True,
        )

    def start(self):
        """Start the monitoring thread once, independently of the radio link."""
        if self._stop.is_set():
            raise RuntimeError("Cannot restart a stopped flight monitor")

        if self._thread.is_alive():
            return

        diag("MONITOR_START", agent=self.drone.agent_id, uri=self.drone.URI)
        self._thread.start()

    def attach_to(self, cf):
        """Attach callbacks to the current Crazyflie connection."""
        if cf is None:
            raise ValueError("Cannot attach flight monitor to None")

        with self._callback_lock:
            # Already attached to this exact Crazyflie instance.
            if self._attached_cf is cf:
                return

            # Remove callbacks belonging to any previous connection.
            self._detach_locked()

            # The new connection must not inherit old link information.
            self._link_lost.clear()
            self._link_quality = None
            self._quality_at = None

            # Each callback belongs to this particular connection.
            # Ignore events from an old connection after reconnection.
            def is_current():
                return self._attached_cf is cf

            def on_connected(uri):
                if is_current():
                    diag(
                        "MONITOR_CONNECTED",
                        agent=self.drone.agent_id,
                        uri=uri,
                    )

            def on_connection_failed(uri, message):
                if is_current():
                    diag(
                        "CONNECTION_FAILED",
                        agent=self.drone.agent_id,
                        uri=uri,
                        reason=repr(message),
                    )

            def on_connection_lost(uri, message):
                if is_current():
                    self._on_connection_lost(uri, message)

            def on_disconnected(uri):
                if is_current():
                    self._on_disconnected(uri)

            def on_link_quality(value):
                if is_current():
                    self._on_link_quality(value)

            # Keep the callback references for later removal.
            self._attached_cf = cf
            self._attached_callbacks = [
                (cf.connected, on_connected),
                (cf.connection_failed, on_connection_failed),
                (cf.connection_lost, on_connection_lost),
                (cf.disconnected, on_disconnected),
                (
                    cf.link_statistics.link_quality_updated,
                    on_link_quality,
                ),
            ]

            for caller, callback in self._attached_callbacks:
                caller.add_callback(callback)

        diag(
            "MONITOR_ATTACHED",
            agent=self.drone.agent_id,
            cf_id=id(cf),
        )

    def _detach_locked(self):
        """Remove callbacks. Caller must hold _callback_lock."""
        old_cf = self._attached_cf

        # Invalidate callbacks from the previous connection first.
        self._attached_cf = None

        callbacks = self._attached_callbacks
        self._attached_callbacks = []

        for caller, callback in callbacks:
            try:
                caller.remove_callback(callback)
            except ValueError:
                # Callback was already removed.
                pass

        return old_cf

    def detach(self):
        """Detach callbacks without stopping the monitor thread."""
        with self._callback_lock:
            old_cf = self._detach_locked()

        if old_cf is not None:
            diag(
                "MONITOR_DETACHED",
                agent=self.drone.agent_id,
                cf_id=id(old_cf),
            )

    def stop(self):
        """Permanently stop the monitor and remove its callbacks."""
        self._stop.set()
        self.detach()

        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

        diag("MONITOR_STOP", agent=self.drone.agent_id)

    def _on_connection_lost(self, uri, message):
        self._link_lost.set()
        diag(
            "CONNECTION_LOST", agent=self.drone.agent_id, uri=uri, reason=repr(message)
        )

    def _on_disconnected(self, uri):
        diag("DISCONNECTED", agent=self.drone.agent_id, uri=uri)

    def _on_link_quality(self, value):
        # Called at radio packet rate. Store only; do not print per packet.
        self._link_quality = round(float(value), 1)
        self._quality_at = time.monotonic()

    def _run(self):
        while not self._stop.wait(1.0):
            try:
                self.snapshot()
            except Exception as exc:
                diag("MONITOR_ERROR", agent=self.drone.agent_id, error=repr(exc))

    def snapshot(self):
        drone = self.drone
        now = time.monotonic()
        pos = target = last_position = None

        # Do not hang the diagnostic thread if position_lock is stuck.
        pos_lock_acquired = drone.position_lock.acquire(timeout=0.01)
        if pos_lock_acquired:
            try:
                pos = drone.position.copy()
                target = drone.target_position.copy()
                last_position = drone.last_position_update_time
            finally:
                drone.position_lock.release()

        voltage = None
        if drone.battery_lock.acquire(timeout=0.01):
            try:
                voltage = (
                    drone.battery_level if drone.battery_ready_event.is_set() else None
                )
            finally:
                drone.battery_lock.release()

        mc_thread = getattr(getattr(drone, "mc", None), "_thread", None)
        mc_queue = getattr(mc_thread, "_queue", None)
        cf = drone.cf
        link = getattr(cf, "link", None) if cf is not None else None
        radio_thread = getattr(link, "_thread", None)

        extra = {}
        vicon_frame_age = None
        if hasattr(drone, "get_vicon_diagnostics"):
            details = drone.get_vicon_diagnostics()
            extra = {
                "vicon_counts": details.get("counts"),
                "vicon_issue": details.get("last_issue"),
                "vicon_send_age_s": details.get("last_sent_age"),
            }
            vicon_frame_age = details.get("position_age")

        battery_age = _age(now, getattr(drone, "_diag_battery_at", None))
        if drone.is_flying_event.is_set() and (
            battery_age is None or battery_age > 2.0
        ):
            diag(
                "BATTERY_TELEMETRY_STALE",
                agent=drone.agent_id,
                age_s=battery_age,
                log_started=getattr(drone.battery_log_config, "started", None),
                log_err_no=getattr(drone.battery_log_config, "err_no", None),
            )

        if not getattr(drone, "using_flow_deck", True):
            last_tx = getattr(drone, "vicon_last_sent_time", None)
            last_rx = getattr(drone, "_diag_extpos_rx_last_change_at", None)

            tx_is_recent = last_tx is not None and now - last_tx < 0.3
            rx_is_stale = last_rx is None or now - last_rx > 0.5

            if tx_is_recent and rx_is_stale:
                diag(
                    "EXTPOS_RX_STALE",
                    agent=drone.agent_id,
                    tx_age_s=_age(now, last_tx),
                    rx_change_age_s=_age(now, last_rx),
                    rx_tick=getattr(drone, "_extpos_rx_last_tick", None),
                )
        diag(
            "STATUS",
            agent=drone.agent_id,
            flying=drone.is_flying_event.is_set(),
            emergency=drone.emergency_event.is_set(),
            connected=cf.is_connected() if cf is not None else None,
            link_lost=self._link_lost.is_set(),
            radio_thread_alive=radio_thread.is_alive() if radio_thread else None,
            main_thread_alive=drone.thread.is_alive(),
            pid_thread_alive=(
                drone.controller_thread.is_alive() if drone.controller_thread else None
            ),
            mc_thread_alive=mc_thread.is_alive() if mc_thread else None,
            position_lock_acquired=pos_lock_acquired,
            position=_point(pos),
            target=_point(target),
            pos_age_s=_age(now, last_position),
            vicon_frame_age_s=vicon_frame_age,
            command_queue=drone.command_queue.qsize(),
            mc_queue=mc_queue.qsize() if mc_queue else None,
            last_command_age_s=_age(now, getattr(drone, "_diag_last_run_tick", None)),
            last_pid_age_s=_age(now, getattr(drone, "_diag_last_pid_tick", None)),
            last_queue_lag_ms=round(
                getattr(drone, "_diag_last_queue_lag_s", 0.0) * 1000, 1
            ),
            last_mc_update_age_s=_age(
                now, getattr(drone, "_diag_last_mc_update", None)
            ),
            quality=self._link_quality,
            quality_age_s=_age(now, self._quality_at),
            voltage=round(voltage, 3) if voltage is not None else None,
            voltage_sample_age_s=_age(now, getattr(drone, "_diag_battery_at", None)),
            extpos_rx_tick=getattr(drone, "_extpos_rx_last_tick", None),
            extpos_rx_log_age_s=_age(
                now, getattr(drone, "_diag_extpos_rx_last_sample_at", None)
            ),
            extpos_rx_change_age_s=_age(
                now, getattr(drone, "_diag_extpos_rx_last_change_at", None)
            ),
            **extra,
        )
