from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PositionSample:
    """One position measurement in metres in the drone_gym world frame."""

    x: float
    y: float
    z: float
    timestamp: float

    @classmethod
    def now(
        cls,
        x: float,
        y: float,
        z: float,
    ) -> PositionSample:
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            timestamp=time.monotonic(),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


PositionCallback = Callable[[PositionSample], None]


class PositionSource(ABC):
    """
    Standard interface for a single drone's position stream.

    Implementations hide source-specific communication and publish positions
    in metres in the drone_gym world coordinate frame.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def start(self, callback: PositionCallback) -> None:
        """
        Start publishing valid position samples through callback.

        This method must initiate any source-specific background work and return.
        It must not remain blocked for the lifetime of the source.
        """
    @abstractmethod
    def stop(self) -> None:
        """Stop publishing and release source-specific resources."""