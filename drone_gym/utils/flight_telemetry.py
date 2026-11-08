"""Time-aligned, nonblocking CSV flight diagnostics for drone_gym.

Contains no flight control actions. Each event is timestamped at its source and
written by a separate thread. Retain separate 'vicon', 'estimate', 'battery',
'extra' rows: never pretend asynchronous sources are exactly simultaneous.
"""

import atexit
import csv
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time


FIELDS = (
    "utc_received", "host_mono_s", "sample_mono_s", "stream", "agent",
    "cf_timestamp_ms", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s",
    "vz_m_s", "yaw_deg", "target_z_m", "range_z_m", "thrust",
    "battery_v", "detail",
)


class FlightTelemetryRecorder:
    def __init__(self, agent_id, directory="flight_debug_logs"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in agent_id)
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"telemetry_{stamp}_{safe_id}.csv"
        self.agent_id = agent_id
        self.queue = Queue(maxsize=20000)
        self.dropped = 0
        self._closed = False
        self._close_lock = threading.Lock()
        self._stop = object()
        self._thread = threading.Thread(target=self._write_loop,
                                        name=f"{agent_id}-telemetry-writer",
                                        daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def record(self, stream, *, sample_mono_s=None, **values):
        """Fast producer API. Call from Vicon or cflib callbacks."""
        if self._closed:
            return
        row = dict.fromkeys(FIELDS, "")
        row.update(values)
        row.update(
            utc_received=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            host_mono_s=time.monotonic(),
            sample_mono_s=sample_mono_s if sample_mono_s is not None else "",
            stream=stream,
            agent=self.agent_id,
        )
        try:
            self.queue.put_nowait(row)
        except Full:
            self.dropped += 1

    def _write_loop(self):
        try:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                last_flush = time.monotonic()
                while True:
                    try:
                        row = self.queue.get(timeout=0.2)
                    except Empty:
                        row = None
                    if row is self._stop:
                        break
                    if row is not None:
                        writer.writerow(row)
                    if time.monotonic() - last_flush >= 0.5:
                        handle.flush()
                        last_flush = time.monotonic()
                handle.flush()
        except Exception as exc:
            # The writer is the only thread allowed to perform CSV I/O.
            print(f"[flight-telemetry] Writer failed: {exc!r}", flush=True)

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            # The queue may be full after a prolonged disk failure. Do not hang.
            try:
                self.queue.put_nowait(self._stop)
            except Full:
                try:
                    self.queue.get_nowait()
                    self.queue.put_nowait(self._stop)
                except (Empty, Full):
                    pass
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
