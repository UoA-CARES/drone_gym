from __future__ import annotations

import datetime
import math
import socket
import threading
import time
from struct import unpack

from drone_gym.utils.position_source import (
    PositionCallback,
    PositionSample,
    PositionSource,
)


ViconSnapshot = tuple[list[float], float]


class ViconInterface:
    """
    Low-level Vicon UDP receiver.

    This class:
      - owns the UDP socket;
      - parses incoming Vicon packets;
      - stores the latest data for every tracked object.

    It does not know about Drone, DroneSetup, SARL, MARL, or PositionSource.
    """

    def __init__(
        self,
        udp_ip: str = "0.0.0.0",
        udp_port: int = 51001,
    ) -> None:
        self.udp_ip = udp_ip
        self.udp_port = udp_port

        print(
            f"[ViconInterface] Connecting to "
            f"{self.udp_ip}:{self.udp_port}"
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.sock.bind(
            (
                self.udp_ip,
                self.udp_port,
            )
        )

        # A timeout lets the receive loop periodically check its stop event.
        self.sock.settimeout(0.5)

        print("[ViconInterface] Connected")

        self.time_last_packet = 0
        self.FPS = 0

        # Maps object name to:
        # [
        #   x, y, z,
        #   roll, pitch, yaw,
        #   vx, vy, vz,
        #   roll_rate, pitch_rate, yaw_rate,
        #   reception_time,
        # ]
        self.tracked_object: dict[str, list] = {}
        self.tracked_object_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.have_recv_packet = False

    def main_loop(self) -> None:
        """Receive and process Vicon packets until end() is called."""

        print("[ViconInterface] Receiver loop started")

        try:
            while not self.stop_event.is_set():
                try:
                    packet, _ = self.sock.recvfrom(4096)

                except socket.timeout:
                    continue

                except OSError as exc:
                    # Closing the socket during normal shutdown interrupts
                    # recvfrom(). Do not report that as an error.
                    if not self.stop_event.is_set():
                        print(
                            "[ViconInterface] Socket error: "
                            f"{exc}"
                        )
                    break

                try:
                    self._process_packet(packet)

                except Exception as exc:
                    print(
                        "[ViconInterface] Could not parse packet: "
                        f"{exc}"
                    )

        finally:
            try:
                self.sock.close()
            except OSError:
                pass

            print("[ViconInterface] Receiver loop stopped")

    def _process_packet(self, packet: bytes) -> None:
        """Extract every tracked object contained in one Vicon packet."""

        if len(packet) < 5:
            return

        number_of_items = packet[4]
        byte_offset = 5

        for _ in range(number_of_items):
            # An item needs at least its ID and two-byte data length.
            if byte_offset + 3 > len(packet):
                return

            item_id = packet[byte_offset]
            byte_offset += 1

            item_data_size = int.from_bytes(
                packet[byte_offset:byte_offset + 2],
                byteorder="little",
            )
            byte_offset += 2

            item_end = byte_offset + item_data_size

            if item_end > len(packet):
                return

            item_data = packet[byte_offset:item_end]
            byte_offset = item_end

            # Item ID zero represents a tracked object.
            if item_id == 0:
                self._process_tracked_object(item_data)

    def _process_tracked_object(
        self,
        item_data: bytes,
    ) -> None:
        """Parse and store one tracked-object item."""

        # Existing packet layout:
        #   24 bytes: object name
        #   48 bytes: six doubles
        if len(item_data) < 72:
            return

        name = str(
            item_data[0:24],
            "utf-8",
        ).strip("\x00")

        (
            raw_x,
            raw_y,
            raw_z,
            roll,
            pitch,
            yaw,
        ) = unpack(
            "6d",
            item_data[24:72],
        )

        # Vicon supplies millimetres. drone_gym uses metres.
        x = raw_x / 1000.0
        y = raw_y / 1000.0
        z = raw_z / 1000.0

        reception_time = time.monotonic()

        with self.tracked_object_lock:
            previous_data = self.tracked_object.get(name)

            if previous_data is None:
                x_velocity = 0.0
                y_velocity = 0.0
                z_velocity = 0.0

                roll_rate = 0.0
                pitch_rate = 0.0
                yaw_rate = 0.0

            else:
                previous_time = previous_data[12]

                dt = reception_time - previous_time

                if dt > 0:
                    x_velocity = (
                        x - previous_data[0]
                    ) / dt
                    y_velocity = (
                        y - previous_data[1]
                    ) / dt
                    z_velocity = (
                        z - previous_data[2]
                    ) / dt

                    roll_rate = self._angular_rate(
                        current_angle=roll,
                        previous_angle=previous_data[3],
                        dt=dt,
                    )
                    pitch_rate = self._angular_rate(
                        current_angle=pitch,
                        previous_angle=previous_data[4],
                        dt=dt,
                    )
                    yaw_rate = self._angular_rate(
                        current_angle=yaw,
                        previous_angle=previous_data[5],
                        dt=dt,
                    )

                else:
                    x_velocity = 0.0
                    y_velocity = 0.0
                    z_velocity = 0.0

                    roll_rate = 0.0
                    pitch_rate = 0.0
                    yaw_rate = 0.0

            self.tracked_object[name] = [
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
                x_velocity,
                y_velocity,
                z_velocity,
                roll_rate,
                pitch_rate,
                yaw_rate,
                reception_time,
            ]

            self.have_recv_packet = True

    @staticmethod
    def _angular_rate(
        current_angle: float,
        previous_angle: float,
        dt: float,
    ) -> float:
        """Calculate wrapped angular velocity in radians per second."""

        angle_change = (
            (
                current_angle
                - previous_angle
                + math.pi
            )
            % (2 * math.pi)
        ) - math.pi

        return angle_change / dt

    def get_position_snapshot(
        self,
        name: str,
    ) -> ViconSnapshot | None:
        """
        Return a thread-safe snapshot of one object's latest position.

        The datetime indicates when that Vicon measurement was received.
        It lets ViconPositionSource avoid repeatedly publishing an old value.
        """

        with self.tracked_object_lock:
            object_data = self.tracked_object.get(name)

            if object_data is None:
                return None

            position = [
                float(object_data[0]),
                float(object_data[1]),
                float(object_data[2]),
            ]

            reception_time = object_data[12]

        return position, reception_time

    def getPos(
        self,
        name: str,
    ) -> list[float] | None:
        """
        Return position and attitude for backwards compatibility.

        This method can be removed after all old callers are migrated.
        """

        with self.tracked_object_lock:
            object_data = self.tracked_object.get(name)

            if object_data is None:
                return None

            return list(object_data[0:6])

    def getVel(
        self,
        name: str,
    ) -> list[float] | None:
        """Return linear and angular velocities."""

        with self.tracked_object_lock:
            object_data = self.tracked_object.get(name)

            if object_data is None:
                return None

            return list(object_data[6:12])

    def end(self) -> None:
        """
        Stop the receiver and interrupt a blocked recvfrom() call.

        Calling this more than once is safe.
        """

        if self.stop_event.is_set():
            return

        self.stop_event.set()

        try:
            self.sock.close()
        except OSError:
            pass


class ViconProvider:
    """
    Owns one shared ViconInterface.

    Several ViconPositionSource objects may acquire this provider. The UDP
    connection starts for the first source and stops after the final source
    releases it.
    """

    def __init__(
        self,
        udp_ip: str = "0.0.0.0",
        udp_port: int = 51001,
    ) -> None:
        self._udp_ip = udp_ip
        self._udp_port = udp_port

        self._interface: ViconInterface | None = None
        self._receiver_thread: threading.Thread | None = None

        self._subscriber_count = 0
        self._lifecycle_lock = threading.Lock()

    def acquire(self) -> None:
        """
        Register an active ViconPositionSource.

        The first subscriber creates and starts the shared receiver.
        """

        with self._lifecycle_lock:
            if self._subscriber_count > 0:
                self._subscriber_count += 1
                return

            interface = ViconInterface(
                udp_ip=self._udp_ip,
                udp_port=self._udp_port,
            )

            receiver_thread = threading.Thread(
                target=interface.main_loop,
                name="vicon-receiver",
                daemon=True,
            )

            self._interface = interface
            self._receiver_thread = receiver_thread

            try:
                receiver_thread.start()

            except Exception:
                self._interface = None
                self._receiver_thread = None

                interface.end()
                raise

            self._subscriber_count = 1

    def release(self) -> None:
        """
        Release one source's use of this provider.

        The shared receiver remains active while any subscribers remain.
        """

        with self._lifecycle_lock:
            if self._subscriber_count == 0:
                return

            self._subscriber_count -= 1

            if self._subscriber_count > 0:
                return

            interface = self._interface
            receiver_thread = self._receiver_thread

            self._interface = None
            self._receiver_thread = None

        if interface is not None:
            interface.end()

        if (
            receiver_thread is not None
            and receiver_thread.is_alive()
            and receiver_thread is not threading.current_thread()
        ):
            receiver_thread.join(timeout=2.0)

            if receiver_thread.is_alive():
                print(
                    "[ViconProvider] WARNING: "
                    "receiver thread did not stop in time"
                )

    def get_position_snapshot(
        self,
        object_name: str,
    ) -> ViconSnapshot | None:
        """Return the latest snapshot for one tracked object."""

        with self._lifecycle_lock:
            interface = self._interface

        if interface is None:
            return None

        return interface.get_position_snapshot(object_name)

    @property
    def subscriber_count(self) -> int:
        with self._lifecycle_lock:
            return self._subscriber_count


_default_vicon_provider: ViconProvider | None = None
_default_vicon_provider_lock = threading.Lock()


def get_default_vicon_provider() -> ViconProvider:
    """
    Return the process-wide default provider.

    Separately created physical Drone objects therefore share one Vicon
    socket unless a different provider is explicitly supplied.
    """

    global _default_vicon_provider

    with _default_vicon_provider_lock:
        if _default_vicon_provider is None:
            _default_vicon_provider = ViconProvider()

        return _default_vicon_provider


class ViconPositionSource(PositionSource):
    """
    PositionSource for one tracked Vicon object.

    Each physical drone gets its own ViconPositionSource, while those sources
    can share one ViconProvider and one UDP connection.
    """

    def __init__(
        self,
        object_name: str,
        provider: ViconProvider | None = None,
        poll_period: float = 1.0 / 120.0,
        label: str | None = None,
    ) -> None:
        if not object_name:
            raise ValueError(
                "object_name must not be empty"
            )

        if poll_period <= 0:
            raise ValueError(
                "poll_period must be greater than zero"
            )

        self._object_name = object_name
        self._provider = (
            provider
            if provider is not None
            else get_default_vicon_provider()
        )

        self._poll_period = poll_period
        self._label = label or object_name

        self._callback: PositionCallback | None = None
        self._worker_thread: threading.Thread | None = None

        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()

        self._active = False
        self._provider_acquired = False

    def start(
        self,
        callback: PositionCallback,
    ) -> None:
        """
        Start publishing Vicon positions and return immediately.
        """

        with self._lifecycle_lock:
            if self._active:
                return

            self._callback = callback
            self._stop_event.clear()

        try:
            self._provider.acquire()

            with self._lifecycle_lock:
                self._provider_acquired = True
                self._active = True

                worker_thread = threading.Thread(
                    target=self._run,
                    name=(
                        "vicon-position-source-"
                        f"{self._object_name}"
                    ),
                    daemon=True,
                )

                self._worker_thread = worker_thread

            worker_thread.start()

            print(
                f"[{self._label}] Vicon source started for "
                f"{self._object_name!r}"
            )

        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """
        Stop publishing and release the shared provider.

        Calling this more than once is safe.
        """

        with self._lifecycle_lock:
            worker_thread = self._worker_thread
            provider_acquired = self._provider_acquired

            was_started = (
                self._active
                or self._provider_acquired
                or self._worker_thread is not None
            )

            self._active = False
            self._callback = None
            self._worker_thread = None
            self._provider_acquired = False

            self._stop_event.set()

        if (
            worker_thread is not None
            and worker_thread.is_alive()
            and worker_thread is not threading.current_thread()
        ):
            worker_thread.join(timeout=2.0)

            if worker_thread.is_alive():
                print(
                    f"[{self._label}] WARNING: "
                    "position source thread did not stop in time"
                )

        if provider_acquired:
            self._provider.release()

        if was_started:
            print(
                f"[{self._label}] Vicon position source stopped"
            )

    def _run(self) -> None:
        """
        Poll the shared provider and publish each Vicon update once.
        """

        last_reception_time: float | None = None

        while not self._stop_event.is_set():
            snapshot = self._provider.get_position_snapshot(
                self._object_name
            )

            if snapshot is not None:
                position, reception_time = snapshot

                # Only publish when the underlying Vicon receiver has
                # received a genuinely new measurement.
                if reception_time != last_reception_time:
                    last_reception_time = reception_time

                    with self._lifecycle_lock:
                        if not self._active:
                            break

                        callback = self._callback

                    if callback is not None:
                        try:
                            callback(
                                PositionSample.now(
                                    x=position[0],
                                    y=position[1],
                                    z=position[2],
                                )
                            )

                        except Exception as exc:
                            # Do not let a consumer exception terminate the
                            # position-source worker thread.
                            print(
                                f"[{self._label}] Error publishing "
                                f"Vicon position: {exc}"
                            )

            self._stop_event.wait(
                self._poll_period
            )

    @property
    def object_name(self) -> str:
        return self._object_name