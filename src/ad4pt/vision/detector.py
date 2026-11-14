"""Replaceable single-frame detector, with a lazy Ultralytics adapter."""

import hashlib
import logging
import math
import os
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

from ad4pt.json_io import open_regular
from ad4pt.types import VideoFrame
from ad4pt.vision.config import DetectorConfig
from ad4pt.vision.tracking import create_tracker


class DetectorError(Exception):
    """Model load, inference, or output validation failed."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy_px: tuple[float, float, float, float]
    track_id: int | None = None

    def __post_init__(self) -> None:
        if type(self.class_id) is not int or self.class_id < 0:
            raise ValueError("class_id must be a nonnegative integer")
        if self.track_id is not None and (
            type(self.track_id) is not int or self.track_id < 1
        ):
            raise ValueError("track_id must be a positive integer or None")
        if not isinstance(self.label, str) or not 1 <= len(self.label) <= 128:
            raise ValueError(
                "label must be a nonempty string of at most 128 characters"
            )
        if (
            type(self.confidence) not in (int, float)
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite and in [0, 1]")
        box = self.bbox_xyxy_px
        if (
            not isinstance(box, tuple)
            or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
        ):
            raise ValueError("bbox_xyxy_px must contain four finite numbers")
        if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
            raise ValueError("bbox_xyxy_px must be nonnegative with positive area")


class Detector(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...
    def detect(self, frame: VideoFrame) -> tuple[Detection, ...]: ...
    def close(self) -> None: ...


class UltralyticsDetector:
    """One owner, one frame at a time. Only load explicitly trusted .pt files.

    SHA256 verifies identity, not safety: PyTorch checkpoints may execute code.
    The adapter never accepts URLs or model aliases and never downloads weights.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model: Any = None
        self._metadata: dict[str, Any] = {}
        self._tracker: Any = None
        self._tracker_path: Path | None = None
        try:
            with os.fdopen(
                open_regular(config.model_path, os.O_RDONLY), "rb"
            ) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != config.artifact_sha256:
                raise DetectorError("checkpoint SHA256 mismatch; model was not loaded")
            # Load the inference framework only when the detector starts.
            logging.getLogger(__name__).info("Loading Ultralytics and PyTorch")
            ultralytics = import_module("ultralytics")
            torch = import_module("torch")
            if ultralytics.__version__ != "8.4.14":
                raise DetectorError(
                    "install the vision extra: Ultralytics 8.4.14 is required"
                )
            if config.device == "mps" and not torch.backends.mps.is_available():
                raise DetectorError("MPS is unavailable; use --device cpu")
            logging.getLogger(__name__).info("Loading verified model checkpoint")
            self._model = ultralytics.YOLO(str(config.model_path), task="detect")
            expected = dict(enumerate(config.classes))
            if self._model.task != "detect" or self._model.names != expected:
                raise DetectorError(
                    "checkpoint task/classes differ from detector configuration"
                )
            logging.getLogger(__name__).info("Detector ready")
            settings = asdict(config)
            settings.pop("model_path")
            settings.pop("artifact_sha256")
            settings.pop("model_id")
            settings.pop("classes")
            self._metadata = {
                "model_id": config.model_id,
                "artifact_sha256": digest,
                "classes": list(config.classes),
                "configuration": settings,
                "runtime": {
                    k: version(k)
                    for k in ("ultralytics", "torch", "numpy", "opencv-python")
                },
            }
        except BaseException as exc:
            self.close()
            if isinstance(exc, (KeyboardInterrupt, SystemExit, DetectorError)):
                raise
            raise DetectorError(
                f"cannot load detector (install ad4pt[vision]): {exc}"
            ) from exc

    @property
    def metadata(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self._metadata)

    def detect(self, frame: VideoFrame) -> tuple[Detection, ...]:
        return self._infer(frame)

    def track(
        self, frame: VideoFrame, tracker: Path, low_confidence_threshold: float = 0.1
    ) -> tuple[Detection, ...]:
        return self._infer(frame, tracker, low_confidence_threshold)

    def _infer(
        self,
        frame: VideoFrame,
        tracker: Path | None = None,
        low_confidence_threshold: float = 0.1,
    ) -> tuple[Detection, ...]:
        if self._model is None:
            raise DetectorError("detector is closed")
        try:
            # Crop OSD out before inference; restore boxes to preview coordinates.
            height, width = frame.image_bgr.shape[:2]
            top = min(math.ceil(height * self.config.ignore_top_fraction), height - 1)
            image = frame.image_bgr[top:].copy()
            confidence_threshold = self.config.confidence_threshold
            if tracker is not None:
                confidence_threshold = min(
                    confidence_threshold, low_confidence_threshold
                )
                path = tracker.resolve()
                if self._tracker is None:
                    self._tracker = create_tracker(
                        path, self.config.confidence_threshold
                    )
                    self._tracker_path = path
                elif path != self._tracker_path:
                    raise ValueError("cannot change tracker configuration during a run")
            results = self._model.predict(
                source=image,
                device=self.config.device,
                imgsz=self.config.image_size_px,
                conf=confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                rect=True,
                half=False,
                augment=False,
                agnostic_nms=False,
                save=False,
                save_txt=False,
                save_crop=False,
                show=False,
                verbose=False,
                stream=False,
            )
            if len(results) != 1 or tuple(results[0].orig_shape) != image.shape[:2]:
                raise ValueError("expected one result in inference-image coordinates")
            boxes = results[0].boxes
            if boxes is None:
                raise ValueError("detector returned no boxes object")
            rows = boxes.data.detach().cpu().numpy()
            if (
                rows.ndim != 2
                or rows.shape[1] != 6
                or len(rows) > self.config.max_detections
            ):
                raise ValueError("expected bounded detection/tracking output")
            detections: list[Detection] = []
            accepted_indices: list[int] = []
            for detection_index, row in enumerate(rows):
                x1, y1, x2, y2 = map(float, row[:4])
                confidence, class_id = map(float, row[-2:])
                if (
                    not math.isfinite(class_id)
                    or not class_id.is_integer()
                    or not 0 <= class_id < len(self.config.classes)
                ):
                    raise ValueError("invalid output class ID")
                detection = Detection(
                    int(class_id),
                    self.config.classes[int(class_id)],
                    confidence,
                    (x1, y1, x2, y2),
                )
                if x2 > width or y2 > image.shape[0]:
                    raise ValueError("bounding box exceeds inference image bounds")
                if confidence < confidence_threshold:
                    raise ValueError("confidence below configured threshold")
                # Reject scene-sized false positives before they can steal track IDs.
                if (x2 - x1) * (y2 - y1) > (
                    width * image.shape[0] * self.config.max_box_area_fraction
                ):
                    continue
                accepted_indices.append(detection_index)
                detections.append(
                    replace(detection, bbox_xyxy_px=(x1, y1 + top, x2, y2 + top))
                )
            if tracker is not None:
                if len(accepted_indices) != len(rows):
                    boxes = boxes[accepted_indices]
                # BoT-SORT returns an index into the ORIGINAL detection rows.
                # Keep only its identity; boxes, scores and labels remain YOLO's.
                tracks = self._tracker.update(boxes.cpu().numpy(), image)
                assigned: set[int] = set()
                identities: set[int] = set()
                for row in tracks:
                    identity, index = float(row[4]), float(row[-1])
                    if (
                        not math.isfinite(identity)
                        or not identity.is_integer()
                        or identity < 1
                        or not math.isfinite(index)
                        or not index.is_integer()
                        or not 0 <= index < len(detections)
                    ):
                        raise ValueError("invalid track ID or detection index")
                    position, track_id = int(index), int(identity)
                    if position in assigned or track_id in identities:
                        raise ValueError("duplicate tracker assignment")
                    assigned.add(position)
                    identities.add(track_id)
                    detections[position] = replace(
                        detections[position], track_id=track_id
                    )
            # Weaker observations enter association only: no boxes or ripeness votes.
            return tuple(
                d
                for d in detections
                if d.confidence >= self.config.threshold_for(d.label)
            )
        except Exception as exc:
            raise DetectorError(
                f"inference failed for frame {frame.frame_id}: {exc}"
            ) from exc

    def close(self) -> None:
        # Drop model, predictor, and retained last-image/tensor references together.
        self._model = None
        self._tracker = None
        self._tracker_path = None
