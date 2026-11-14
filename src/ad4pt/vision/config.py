"""Small versioned JSON experiment configuration for trusted local checkpoints."""

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ad4pt.json_io import decode, open_regular


@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path
    artifact_sha256: str
    model_id: str
    classes: tuple[str, ...]
    device: str = "cpu"
    image_size_px: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7
    max_detections: int = 300
    ignore_top_fraction: float = 0.0
    max_box_area_fraction: float = 1.0
    unripe_confidence_threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, Path):
            raise ValueError("model_path must be a local pathlib.Path")
        if self.model_path.suffix.lower() != ".pt" or not self.model_path.is_file():
            raise ValueError(
                f"model_path must be an existing local .pt file: {self.model_path}"
            )
        if not isinstance(self.artifact_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.artifact_sha256
        ):
            raise ValueError(
                "artifact_sha256 must be 64 lowercase hexadecimal characters"
            )
        if not isinstance(self.model_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.model_id
        ):
            raise ValueError("model_id must be a short identifier")
        if (
            not isinstance(self.classes, tuple)
            or not 1 <= len(self.classes) <= 100
            or any(
                not isinstance(c, str) or not c or len(c) > 128 for c in self.classes
            )
            or len(set(self.classes)) != len(self.classes)
        ):
            raise ValueError("classes must contain 1–100 unique nonempty labels")
        if self.device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        if (
            type(self.image_size_px) is not int
            or not 32 <= self.image_size_px <= 2048
            or self.image_size_px % 32
        ):
            raise ValueError("image_size_px must be a multiple of 32 in [32, 2048]")
        if type(self.max_detections) is not int or not 1 <= self.max_detections <= 1000:
            raise ValueError("max_detections must be an integer in [1, 1000]")
        for key in ("confidence_threshold", "iou_threshold"):
            value = getattr(self, key)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{key} must be finite and in [0, 1]")
        if (
            type(self.ignore_top_fraction) not in (float, int)
            or not math.isfinite(self.ignore_top_fraction)
            or not 0 <= self.ignore_top_fraction <= 0.5
        ):
            raise ValueError("ignore_top_fraction must be finite and in [0, 0.5]")
        if self.unripe_confidence_threshold is not None and (
            type(self.unripe_confidence_threshold) not in (int, float)
            or not math.isfinite(self.unripe_confidence_threshold)
            or not self.confidence_threshold <= self.unripe_confidence_threshold <= 1
        ):
            raise ValueError(
                "unripe_confidence_threshold must be between confidence_threshold and 1"
            )
        if (
            type(self.max_box_area_fraction) not in (float, int)
            or not math.isfinite(self.max_box_area_fraction)
            or not 0 < self.max_box_area_fraction <= 1
        ):
            raise ValueError("max_box_area_fraction must be finite and in (0, 1]")
        object.__setattr__(self, "model_path", self.model_path.resolve(strict=True))

    def threshold_for(self, label: str) -> float:
        if label in ("strawberry-unripe", "unripe"):
            if self.unripe_confidence_threshold is not None:
                return self.unripe_confidence_threshold
        return self.confidence_threshold


def load_detector_config(path: Path) -> DetectorConfig:
    """Paths are relative to the config file, independent of working directory."""
    try:
        with os.fdopen(open_regular(path, os.O_RDONLY), "rb") as handle:
            data = handle.read(65537)
        if len(data) > 65536:
            raise ValueError("detector configuration exceeds 64 KiB")
        values = decode(data)
        if values.pop("config_version", None) != "1.0":
            raise ValueError("detector config_version must be 1.0")
        if not isinstance(values.get("model_path"), str):
            raise ValueError("model_path must be a local file path")
        values["model_path"] = path.resolve().parent / values["model_path"]
        if not isinstance(values.get("classes"), list):
            raise ValueError("classes must be an ordered JSON array")
        values["classes"] = tuple(values["classes"])
        return DetectorConfig(**values)
    except (
        OSError,
        RuntimeError,
        RecursionError,
        TypeError,
        json.JSONDecodeError,
        UnicodeError,
    ) as exc:
        raise ValueError(f"cannot load detector configuration {path}: {exc}") from exc
