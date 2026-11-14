"""Shared immutable BGR frame. Timestamp clock domains are source-specific.

Offline received_at_ns uses Unix epoch time; live RTP uses local monotonic time.
Live captured_at_ns is unknown (None). See README before comparing timestamps.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class VideoFrame:
    frame_id: int
    captured_at_ns: int | None
    received_at_ns: int
    source: str
    image_bgr: NDArray[np.uint8]

    def __post_init__(self) -> None:
        for name in ("frame_id", "received_at_ns", "captured_at_ns"):
            value = getattr(self, name)
            if value is None and name == "captured_at_ns":
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a nonempty string")
        image = self.image_bgr
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("image_bgr must be a nonempty H x W x 3 uint8 array")
        # Immutable bytes backing prevents mutation even via setflags(write=True).
        frozen = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(image.shape)
        object.__setattr__(self, "image_bgr", frozen)
