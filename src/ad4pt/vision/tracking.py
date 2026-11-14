"""Validate the small local BoT-SORT recipe before opening the camera."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray


class _IndexedBoxes:
    """Keep source-row indices when BoT-SORT splits strong/weak detections."""

    def __init__(self, boxes: Any, indices: NDArray[np.intp] | None = None) -> None:
        self._boxes = boxes
        self.indices = (
            np.arange(len(boxes), dtype=np.intp) if indices is None else indices
        )

    def __len__(self) -> int:
        return len(self._boxes)

    def __getitem__(self, key: Any) -> "_IndexedBoxes":
        return _IndexedBoxes(self._boxes[key], np.atleast_1d(self.indices[key]))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._boxes, name)


@dataclass(frozen=True)
class TrackingConfig:
    path: Path
    confirmation_frames: int
    track_buffer: int
    track_low_thresh: float
    settings: dict[str, Any]


def load_tracking_config(path: Path) -> TrackingConfig:
    with path.open("rb") as handle:
        data = handle.read(65537)
    if len(data) > 65536:
        raise ValueError("tracker configuration exceeds 64 KiB")
    try:
        values = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid tracker YAML: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError("tracker configuration must be a mapping")
    if values.get("tracker_type") != "botsort" or values.get("with_reid") is not False:
        raise ValueError("use BoT-SORT with with_reid: false (no extra model)")
    if values.get("gmc_method") != "sparseOptFlow":
        raise ValueError("gmc_method must be sparseOptFlow for camera compensation")
    for key in ("confirmation_frames", "track_buffer"):
        if type(values.get(key)) is not int or not 1 <= values[key] <= 300:
            raise ValueError(f"{key} must be an integer in [1, 300]")
    for key in (
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "match_thresh",
        "proximity_thresh",
        "appearance_thresh",
    ):
        value = values.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{key} must be a number in [0, 1]")
    if values["track_low_thresh"] > values["track_high_thresh"]:
        raise ValueError("track_low_thresh must not exceed track_high_thresh")
    if type(values.get("fuse_score")) is not bool:
        raise ValueError("fuse_score must be a boolean")
    return TrackingConfig(
        path.resolve(strict=True),
        values["confirmation_frames"],
        values["track_buffer"],
        values["track_low_thresh"],
        values,
    )


def create_tracker(path: Path, detection_threshold: float) -> Any:
    config = load_tracking_config(path)
    # Check the dependency before Ultralytics can attempt an automatic install.
    try:
        import_module("lap")
    except ImportError as exc:
        raise ValueError("tracking requires lap; install ad4pt[vision]") from exc
    settings = config.settings.copy()
    # Every detection eligible for display/counting must be able to start an ID.
    settings["track_high_thresh"] = detection_threshold
    settings["new_track_thresh"] = detection_threshold
    settings["track_low_thresh"] = min(config.track_low_thresh, detection_threshold)
    module = import_module("ultralytics.trackers.bot_sort")

    class IndexedBOTSORT(module.BOTSORT):  # type: ignore[misc, name-defined]
        def update(self, results: Any, img: Any = None, feats: Any = None) -> Any:
            return super().update(_IndexedBoxes(results), img, feats)

        def init_track(self, results: _IndexedBoxes, img: Any = None) -> list[Any]:
            tracks: list[Any] = super().init_track(results, img)
            # Upstream init_track enumerates each confidence subset from zero.
            # Replace that subset index with the original YOLO row index.
            for track, index in zip(tracks, results.indices, strict=True):
                track.idx = int(index)
            return tracks

    return IndexedBOTSORT(SimpleNamespace(**settings), frame_rate=30)
