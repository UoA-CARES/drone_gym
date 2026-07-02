from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from drone_gym.utils.position_source import (
    PositionCallback,
    PositionSample,
    PositionSource,
)


CrazyflieGetter = Callable[[], Crazyflie | None]


class CrazyflieLogPositionSource(PositionSource):
    """
    Supplies positions from Crazyflie state-estimate log variables.

    This source reads:

        stateEstimate.x
        stateEstimate.y
        stateEstimate.z

    and publishes them as PositionSample objects.
    """

    def __init__(
        self,
        crazyflie_getter: CrazyflieGetter,
        period_ms: int = 50,
        label: str = "Crazyflie",
    ) -> None:
        if period_ms <= 0:
            raise ValueError("period_ms must be greater than zero")

        self._crazyflie_getter = crazyflie_getter
        self._period_ms = period_ms
        self._label = label

        self._callback: PositionCallback | None = None
        self._log_config: LogConfig | None = None

        self._lifecycle_lock = threading.Lock()
        self._active = False

    def start(self, callback: PositionCallback) -> None:
        """
        Start Crazyflie position logging.

        This method configures logging and then returns. Position samples are
        subsequently delivered by cflib through its logging callback.
        """

        with self._lifecycle_lock:
            if self._active:
                return

            crazyflie = self._crazyflie_getter()

            if crazyflie is None:
                raise RuntimeError(
                    "Cannot start position logging because the "
                    "Crazyflie instance is not available"
                )

            log_config = LogConfig(
                name="Position",
                period_in_ms=self._period_ms,
            )

            log_config.add_variable(
                "stateEstimate.x",
                "float",
            )
            log_config.add_variable(
                "stateEstimate.y",
                "float",
            )
            log_config.add_variable(
                "stateEstimate.z",
                "float",
            )

            self._callback = callback
            self._log_config = log_config

        try:
            crazyflie.log.add_config(log_config)

            if not log_config.valid:
                raise RuntimeError(
                    "One or more stateEstimate position variables "
                    "were not found in the Crazyflie log TOC"
                )

            log_config.data_received_cb.add_callback(
                self._position_callback
            )
            log_config.error_cb.add_callback(
                self._logging_error_callback
            )

            # Mark active before starting so the first callback is accepted.
            with self._lifecycle_lock:
                self._active = True

            log_config.start()

            print(
                f"[{self._label}] Crazyflie position logging started"
            )

        except Exception:
            # start() must not leave behind a partially configured source.
            self.stop()
            raise

    def stop(self) -> None:
        """
        Stop position logging and remove registered callbacks.

        Calling this method multiple times is safe.
        """

        with self._lifecycle_lock:
            log_config = self._log_config

            self._active = False
            self._callback = None
            self._log_config = None

        if log_config is None:
            return

        try:
            log_config.stop()
        except Exception as exc:
            print(
                f"[{self._label}] Failed to stop position logging: "
                f"{exc}"
            )

        try:
            log_config.delete()
        except Exception as exc:
            print(
                f"[{self._label}] Failed to delete position log "
                f"configuration: {exc}"
            )

        try:
            log_config.data_received_cb.remove_callback(
                self._position_callback
            )
        except Exception:
            pass

        try:
            log_config.error_cb.remove_callback(
                self._logging_error_callback
            )
        except Exception:
            pass

        print(
            f"[{self._label}] Crazyflie position logging stopped"
        )

    def _position_callback(
        self,
        timestamp: int,
        data: dict[str, Any],
        log_config: LogConfig,
    ) -> None:
        """
        Convert one Crazyflie log packet into a PositionSample.
        """

        del timestamp, log_config

        with self._lifecycle_lock:
            if not self._active:
                return

            callback = self._callback

        if callback is None:
            return

        try:
            sample = PositionSample.now(
                x=data["stateEstimate.x"],
                y=data["stateEstimate.y"],
                z=data["stateEstimate.z"],
            )

            callback(sample)

        except KeyError as exc:
            print(
                f"[{self._label}] Position log packet was missing "
                f"a required variable: {exc}"
            )

        except (TypeError, ValueError) as exc:
            print(
                f"[{self._label}] Invalid position log data: {exc}"
            )

        except Exception as exc:
            # Do not allow an exception to escape into cflib's log thread.
            print(
                f"[{self._label}] Error publishing position sample: "
                f"{exc}"
            )

    def _logging_error_callback(
        self,
        log_config: LogConfig,
        message: str,
    ) -> None:
        print(
            f"[{self._label}] Position logging error in "
            f"{log_config.name}: {message}"
        )