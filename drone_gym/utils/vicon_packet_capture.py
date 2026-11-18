"""Loss-observable, packet-level Vicon Tracker UDP recording.

Capture in the *shared Vicon UDP receiver*, not in a per-drone position callback.
Every datagram that reaches the process is queued without disk I/O. A separate
writer thread stores exact bytes and decodes Tracker UDP Object Stream data.

The UDP stream exposes frame number, per-object XYZ translation (mm) and
Euler XYZ rotation (radians). Differentiated velocities are computed locally,
not sent by Tracker. This class never sends Crazyflie radio messages.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
from queue import Empty, Full, Queue
import struct
import threading
import time


_PACKET_FIELDS = (
    "packet_seq", "utc_received", "recv_wall_ns", "recv_mono_ns",
    "sender_ip", "sender_port", "packet_bytes", "frame_number",
    "reported_items", "decoded_objects", "unknown_items", "frame_delta",
    "candidate_missing_frames", "trailing_bytes", "parse_error",
)
_OBJECT_FIELDS = (
    "packet_seq", "utc_received", "recv_mono_ns", "frame_number",
    "item_index", "object_name", "x_mm", "y_mm", "z_mm",
    "x_m", "y_m", "z_m", "rot_x_rad", "rot_y_rad", "rot_z_rad",
    "dt_s", "vx_est_m_s", "vy_est_m_s", "vz_est_m_s",
    "rot_x_rate_est_rad_s", "rot_y_rate_est_rad_s", "rot_z_rate_est_rad_s",
    "object_frame_delta", "candidate_missing_object_frames", "non_finite_pose",
)


class ViconPacketCapture:
    """One recorder per ViconInterface/socket, shared by all drones."""

    def __init__(self, directory: str = "flight_debug_logs", max_packets: int = 10000):
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.raw_path = directory_path / f"vicon_udp_{stamp}.bin"
        self.packet_path = directory_path / f"vicon_packets_{stamp}.csv"
        self.object_path = directory_path / f"vicon_objects_{stamp}.csv"

        self.queue: Queue[tuple[int, int, int, bytes, str, int]] = Queue(maxsize=max_packets)
        self._closing = threading.Event()
        self._accepting = True
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._writer_loop, name="vicon-packet-writer", daemon=True
        )
        self.received = 0
        self.queue_dropped = 0
        self.written = 0
        self.parse_errors = 0
        self.writer_error: str | None = None
        self._thread.start()

    def record(self, packet: bytes, sender: tuple[str, int],
               recv_mono_ns: int, recv_wall_ns: int) -> None:
        """Called from recvfrom() thread: bounded, no disk I/O or decoding."""
        if not self._accepting:
            return
        self.received += 1
        sequence = self.received
        try:
            self.queue.put_nowait((sequence, recv_mono_ns, recv_wall_ns,
                                   packet, sender[0], sender[1]))
        except Full:
            # Never block the receiver or control loop to save diagnostics.
            self.queue_dropped += 1

    def _writer_loop(self) -> None:
        last_pose: dict[str, tuple[int, tuple[float, ...], int]] = {}
        last_frame: int | None = None
        try:
            with (self.raw_path.open("wb") as raw,
                  self.packet_path.open("w", newline="", encoding="utf-8") as pf,
                  self.object_path.open("w", newline="", encoding="utf-8") as of):
                raw.write(b"VICONUDP1\n")
                packet_writer = csv.DictWriter(pf, fieldnames=_PACKET_FIELDS)
                object_writer = csv.DictWriter(of, fieldnames=_OBJECT_FIELDS)
                packet_writer.writeheader()
                object_writer.writeheader()
                last_flush = time.monotonic()

                while not self._closing.is_set() or not self.queue.empty():
                    try:
                        seq, mono_ns, wall_ns, packet, sender_ip, sender_port = (
                            self.queue.get(timeout=0.1)
                        )
                    except Empty:
                        continue

                    # Raw file records: <uint64 sequence, uint64 mono_ns,
                    # uint64 Unix_ns, uint32 datagram_size>, then exact bytes.
                    raw.write(struct.pack("<QQQI", seq, mono_ns, wall_ns, len(packet)))
                    raw.write(packet)
                    utc = datetime.fromtimestamp(
                        wall_ns / 1e9, tz=timezone.utc
                    ).isoformat(timespec="microseconds")
                    frame, frame_delta, missing, metadata, objects, error = (
                        self._decode_packet(packet, last_frame)
                    )
                    if frame is not None and frame_delta is not None and 0 < frame_delta < 0x80000000:
                        last_frame = frame
                    elif frame is not None and last_frame is None:
                        last_frame = frame

                    packet_writer.writerow({
                        "packet_seq": seq, "utc_received": utc,
                        "recv_wall_ns": wall_ns, "recv_mono_ns": mono_ns,
                        "sender_ip": sender_ip, "sender_port": sender_port,
                        "packet_bytes": len(packet),
                        "frame_number": frame if frame is not None else "",
                        "reported_items": metadata["reported_items"],
                        "decoded_objects": len(objects),
                        "unknown_items": metadata["unknown_items"],
                        "frame_delta": frame_delta if frame_delta is not None else "",
                        "candidate_missing_frames": missing,
                        "trailing_bytes": metadata["trailing_bytes"],
                        "parse_error": error,
                    })
                    if error:
                        self.parse_errors += 1

                    for item_index, name, values in objects:
                        # Six values are actual Vicon UDP measurements:
                        # XYZ in mm, Euler XYZ in radians.
                        xyz = values[:3]
                        rot = values[3:]
                        estimates = [""] * 6
                        dt_s = ""
                        object_frame_delta = ""
                        missing_object = 0
                        previous = last_pose.get(name)
                        valid = all(math.isfinite(v) for v in values)
                        if previous is not None and frame is not None:
                            prev_mono, prev_values, prev_frame = previous
                            delta_ns = mono_ns - prev_mono
                            if delta_ns > 0:
                                dt = delta_ns / 1e9
                                dt_s = dt
                                if valid and all(math.isfinite(v) for v in prev_values):
                                    estimates = [
                                        (values[i] - prev_values[i]) / dt / 1000.0
                                        for i in range(3)
                                    ] + [
                                        self._angle_diff(values[i], prev_values[i]) / dt
                                        for i in range(3, 6)
                                    ]
                            delta = (frame - prev_frame) & 0xFFFFFFFF
                            if delta < 0x80000000:
                                object_frame_delta = delta
                                if 1 < delta < 1000:
                                    missing_object = delta - 1

                        if valid and frame is not None:
                            last_pose[name] = (mono_ns, values, frame)

                        object_writer.writerow({
                            "packet_seq": seq, "utc_received": utc,
                            "recv_mono_ns": mono_ns, "frame_number": frame,
                            "item_index": item_index, "object_name": name,
                            "x_mm": xyz[0], "y_mm": xyz[1], "z_mm": xyz[2],
                            "x_m": xyz[0] / 1000.0,
                            "y_m": xyz[1] / 1000.0,
                            "z_m": xyz[2] / 1000.0,
                            "rot_x_rad": rot[0], "rot_y_rad": rot[1],
                            "rot_z_rad": rot[2], "dt_s": dt_s,
                            "vx_est_m_s": estimates[0],
                            "vy_est_m_s": estimates[1],
                            "vz_est_m_s": estimates[2],
                            "rot_x_rate_est_rad_s": estimates[3],
                            "rot_y_rate_est_rad_s": estimates[4],
                            "rot_z_rate_est_rad_s": estimates[5],
                            "object_frame_delta": object_frame_delta,
                            "candidate_missing_object_frames": missing_object,
                            "non_finite_pose": not valid,
                        })
                    self.written += 1
                    if time.monotonic() - last_flush >= 0.5:
                        raw.flush()
                        pf.flush()
                        of.flush()
                        last_flush = time.monotonic()
                raw.flush()
                pf.flush()
                of.flush()
        except Exception as exc:
            self.writer_error = repr(exc)
            print(f"[ViconCapture] Writer failed: {exc!r}", flush=True)

    @staticmethod
    def _angle_diff(now: float, before: float) -> float:
        return (now - before + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _decode_packet(packet: bytes, previous_frame: int | None):
        """Return decoded UDP metadata/objects without affecting flight control."""
        meta = {"reported_items": "", "unknown_items": 0, "trailing_bytes": 0}
        objects: list[tuple[int, str, tuple[float, ...]]] = []
        frame = None
        frame_delta = None
        missing = 0
        error = ""
        if len(packet) < 5:
            return frame, frame_delta, missing, meta, objects, "short_header"
        frame = struct.unpack_from("<I", packet, 0)[0]
        meta["reported_items"] = packet[4]
        if previous_frame is not None:
            frame_delta = (frame - previous_frame) & 0xFFFFFFFF
            if 1 < frame_delta < 1000:
                missing = frame_delta - 1
        offset = 5
        for index in range(packet[4]):
            if offset + 3 > len(packet):
                error = f"truncated_item_header_at_{index}"
                break
            item_id, size = struct.unpack_from("<BH", packet, offset)
            offset += 3
            if offset + size > len(packet):
                error = f"truncated_item_{index}_id_{item_id}_size_{size}"
                break
            item = packet[offset:offset + size]
            offset += size
            if item_id != 0:
                meta["unknown_items"] += 1
                continue
            if size < 72:
                error = f"short_object_{index}_size_{size}"
                break
            name = item[:24].split(b"\0", 1)[0].decode("utf-8", errors="replace")
            values = struct.unpack_from("<6d", item, 24)
            objects.append((index, name, values))
        meta["trailing_bytes"] = len(packet) - offset
        return frame, frame_delta, missing, meta, objects, error

    def close(self, timeout: float = 5.0) -> None:
        """Drain received packet backlog after UDP receive loop has stopped."""
        with self._close_lock:
            if self._closing.is_set():
                return
            self._accepting = False
            self._closing.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout)
        print(
            f"[ViconCapture] received={self.received} written={self.written} "
            f"queue_dropped={self.queue_dropped} parse_errors={self.parse_errors} "
            f"writer_alive={self._thread.is_alive()} writer_error={self.writer_error} "
            f"raw={self.raw_path} objects={self.object_path}", flush=True
        )
