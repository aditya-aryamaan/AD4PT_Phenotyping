"""Count confirmed BoT-SORT IDs once, with confidence votes for ripeness."""

import math
from collections import Counter
from dataclasses import dataclass

from ad4pt.vision.detector import Detection

LABELS = {
    "ripe": "ripe",
    "unripe": "unripe",
    "strawberry-ripe": "ripe",
    "strawberry-unripe": "unripe",
}


@dataclass
class _Votes:
    hits: int = 0
    ripe: float = 0.0
    unripe: float = 0.0

    @property
    def label(self) -> str:
        return "ripe" if self.ripe > self.unripe else "unripe"


class StrawberryCounter:
    """Retain only small vote totals per ID, never frames or detection history.

    Tracker IDs persist across short gaps. Three tracked observations are needed
    before a fruit contributes to totals. Returning IDs keep their existing count.
    """

    def __init__(
        self, confirmation_frames: int = 3, confidence_threshold: float = 0.25
    ) -> None:
        if type(confirmation_frames) is not int or confirmation_frames < 1:
            raise ValueError("confirmation_frames must be a positive integer")
        self.confirmation_frames = confirmation_frames
        if (
            type(confidence_threshold) not in (int, float)
            or not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be finite and in [0, 1]")
        self.confidence_threshold = confidence_threshold
        self._tracks: dict[int, _Votes] = {}
        self.frames_processed = 0

    def update(self, detections: tuple[Detection, ...]) -> None:
        if any(d.label not in LABELS for d in detections):
            raise ValueError("only ripe and unripe strawberries are supported")
        ids = [d.track_id for d in detections if d.track_id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate track ID in a frame")
        for detection in detections:
            if (
                detection.track_id is None
                or detection.confidence < self.confidence_threshold
            ):
                continue
            votes = self._tracks.setdefault(detection.track_id, _Votes())
            votes.hits += 1
            label = LABELS[detection.label]
            setattr(votes, label, getattr(votes, label) + detection.confidence)
        self.frames_processed += 1

    @property
    def counts(self) -> dict[str, int]:
        counts = {"ripe": 0, "unripe": 0}
        for votes in self._tracks.values():
            if votes.hits >= self.confirmation_frames:
                counts[votes.label] += 1
        return counts


class StaticSceneCounter:
    """Most frequent per-frame count pair for one fixed, fully visible scene.

    This estimates simultaneous detections, not unique fruit along a flight path.
    No track IDs or images are retained. Ties retain the first observed pair.
    """

    def __init__(
        self, confirmation_frames: int = 3, confidence_threshold: float = 0.25
    ) -> None:
        # Share configuration validation with the cumulative counter.
        StrawberryCounter(confirmation_frames, confidence_threshold)
        self.confirmation_frames = confirmation_frames
        self.confidence_threshold = confidence_threshold
        self.frames_processed = 0
        self._pairs: Counter[tuple[int, int]] = Counter()

    def update(self, detections: tuple[Detection, ...]) -> None:
        if any(d.label not in LABELS for d in detections):
            raise ValueError("only ripe and unripe strawberries are supported")
        counts = {"ripe": 0, "unripe": 0}
        for detection in detections:
            if detection.confidence >= self.confidence_threshold:
                counts[LABELS[detection.label]] += 1
        self._pairs[(counts["ripe"], counts["unripe"])] += 1
        self.frames_processed += 1

    @property
    def support_frames(self) -> int:
        return self._pairs.most_common(1)[0][1] if self._pairs else 0

    @property
    def counts(self) -> dict[str, int]:
        if self.support_frames < self.confirmation_frames:
            return {"ripe": 0, "unripe": 0}
        ripe, unripe = self._pairs.most_common(1)[0][0]
        return {"ripe": ripe, "unripe": unripe}


class SceneDetectionFilter:
    """Confirm boxes in consecutive frames of a stationary view, without IDs.

    One missed frame resets confirmation. Matching is one-to-one by box overlap
    and class. Only the previous frame's boxes and streak lengths are retained.
    """

    def __init__(self, confirmation_frames: int = 5, min_iou: float = 0.3) -> None:
        if type(confirmation_frames) is not int or not 1 <= confirmation_frames <= 60:
            raise ValueError("scene confirmation frames must be an integer in [1, 60]")
        if type(min_iou) not in (int, float) or not 0 < min_iou <= 1:
            raise ValueError("scene min_iou must be in (0, 1]")
        self.confirmation_frames = confirmation_frames
        self.min_iou = min_iou
        self._previous: list[tuple[Detection, int]] = []

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ax1, ay1, ax2, ay2 = a.bbox_xyxy_px
        bx1, by1, bx2, by2 = b.bbox_xyxy_px
        overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
        return overlap / union

    def update(self, detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
        candidates = []
        for current, detection in enumerate(detections):
            for previous, (old, _) in enumerate(self._previous):
                if old.label == detection.label:
                    overlap = self._iou(old, detection)
                    if overlap >= self.min_iou:
                        candidates.append((overlap, current, previous))
        matched_current: set[int] = set()
        matched_previous: set[int] = set()
        streaks = [1] * len(detections)
        for _, current, previous in sorted(candidates, reverse=True):
            if current not in matched_current and previous not in matched_previous:
                streaks[current] = min(
                    self._previous[previous][1] + 1, self.confirmation_frames
                )
                matched_current.add(current)
                matched_previous.add(previous)
        self._previous = list(zip(detections, streaks, strict=True))
        return tuple(
            detection
            for detection, streak in self._previous
            if streak >= self.confirmation_frames
        )
