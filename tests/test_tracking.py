from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from ad4pt.counting import StrawberryCounter
from ad4pt.vision.detector import Detection
from ad4pt.vision.tracking import load_tracking_config

RECIPE = Path(__file__).resolve().parents[1] / "config" / "strawberry_botsort.yaml"


@pytest.mark.parametrize(
    "changes",
    [
        {"with_reid": True},
        {"gmc_method": "none"},
        {"track_buffer": 0},
        {"confirmation_frames": True},
        {"track_low_thresh": 0.9},
        {"new_track_thresh": float("nan")},
        {"fuse_score": "true"},
    ],
)
def test_invalid_tracking_recipe(tmp_path, changes):
    data = yaml.safe_load(RECIPE.read_text())
    data.update(changes)
    path = tmp_path / "tracker.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_tracking_config(path)


def test_actual_botsort_survives_pan_and_missed_detections():
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT

    config = load_tracking_config(RECIPE)
    args = SimpleNamespace(**yaml.safe_load(config.path.read_text()))
    tracker = BOTSORT(args, frame_rate=30)
    counter = StrawberryCounter(config.confirmation_frames)
    base = np.random.default_rng(123).integers(0, 256, (160, 240, 3), dtype=np.uint8)
    identities = set()
    for frame_index in range(24):
        dx, dy = 2 * frame_index, frame_index
        image = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]), (240, 160))
        rows = [[50 + dx, 45 + dy, 80 + dx, 75 + dy, 0.9, 0]]
        # Unripe fruit is briefly missed, while the camera continues panning.
        if not 8 <= frame_index <= 12:
            rows.append([130 + dx, 45 + dy, 160 + dx, 75 + dy, 0.9, 1])
        # A one-frame false positive must never become a fruit count.
        if frame_index == 5:
            rows.append([10, 115, 30, 140, 0.9, 0])
        boxes = Boxes(np.array(rows, dtype=np.float32), image.shape[:2])
        tracks = tracker.update(boxes, image)
        detections = tuple(
            Detection(
                int(row[6]),
                "ripe" if int(row[6]) == 0 else "unripe",
                float(row[5]),
                tuple(map(float, row[:4])),
                int(row[4]),
            )
            for row in tracks
        )
        counter.update(detections)
        if frame_index == 4:
            identities = {d.track_id for d in detections}
        if frame_index == 23:
            assert {d.track_id for d in detections} == identities
    assert counter.counts == {"ripe": 1, "unripe": 1}
    assert counter.frames_processed == 24
    assert tracker.encoder is None  # No recognition network.


def test_actual_tracker_starts_valid_032_fruit_and_only_retains_weak_observations():
    from ultralytics.engine.results import Boxes

    from ad4pt.vision.tracking import create_tracker

    tracker = create_tracker(RECIPE, 0.25)
    counter = StrawberryCounter()
    image = np.random.default_rng(7).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    identities = set()
    scores = [0.32, 0.32, 0.19, 0.19, 0.32]
    for score in scores:
        boxes = Boxes(
            np.array([[50, 40, 80, 70, score, 1]], dtype=np.float32), image.shape[:2]
        )
        tracks = tracker.update(boxes, image)
        assert len(tracks) == 1
        identity = int(tracks[0, 4])
        identities.add(identity)
        counter.update((Detection(1, "unripe", score, (50, 40, 80, 70), identity),))
    assert len(identities) == 1
    assert counter.counts == {"ripe": 0, "unripe": 1}
    assert tracker.args.new_track_thresh == tracker.args.track_high_thresh == 0.25


def test_original_indices_survive_mixed_confidence_groups_and_reordering():
    from ultralytics.engine.results import Boxes

    from ad4pt.vision.tracking import create_tracker

    tracker = create_tracker(RECIPE, 0.25)
    image = np.random.default_rng(11).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    ripe = [20, 30, 45, 55, 0.9, 0]
    unripe = [90, 30, 115, 55, 0.9, 1]
    weak_unripe = [90, 30, 115, 55, 0.19, 1]
    weak_background = [10, 90, 25, 110, 0.12, 0]
    frames = [
        [ripe, unripe],
        [ripe, weak_unripe],  # Both subsets used to return local index zero.
        [weak_unripe, ripe],  # Confidence filtering changes original positions.
        [weak_background, ripe, unripe],
    ]
    ids_by_class = {}
    for frame_index, rows in enumerate(frames):
        boxes = Boxes(np.array(rows, dtype=np.float32), image.shape[:2])
        tracks = tracker.update(boxes, image)
        assert len(tracks) == 2
        indices = [int(row[-1]) for row in tracks]
        assert len(set(indices)) == 2
        for row in tracks:
            identity, index = int(row[4]), int(row[-1])
            original = rows[index]
            assert int(row[6]) == original[5]
            assert float(row[5]) == pytest.approx(original[4])
            if frame_index == 0:
                ids_by_class[int(row[6])] = identity
            else:
                assert identity == ids_by_class[original[5]]
