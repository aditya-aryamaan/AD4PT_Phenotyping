"""Offline detector boundary tests; no checkpoint, torch, network, or GPU required."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from ad4pt.types import VideoFrame
from ad4pt.vision import detector
from ad4pt.vision.config import DetectorConfig, load_detector_config
from ad4pt.vision.detector import Detection, DetectorError, UltralyticsDetector


@pytest.fixture
def config(tmp_path):
    model = tmp_path / "synthetic.pt"
    model.write_bytes(b"not executable; mocked backend only")
    return DetectorConfig(
        model,
        hashlib.sha256(model.read_bytes()).hexdigest(),
        "test",
        ("ripe", "unripe"),
    )


def fake_model(monkeypatch, rows=()):
    model = Mock(task="detect", names={0: "ripe", 1: "unripe"})
    boxes = Mock()
    boxes.data.detach.return_value.cpu.return_value.numpy.return_value = np.asarray(
        rows, dtype=float
    ).reshape((-1, 6))
    model.predict.return_value = [SimpleNamespace(orig_shape=(24, 32), boxes=boxes)]
    framework = SimpleNamespace(__version__="8.4.14", YOLO=Mock(return_value=model))
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setattr(
        detector,
        "import_module",
        lambda name: framework if name == "ultralytics" else torch,
    )
    monkeypatch.setattr(detector, "version", lambda name: "test")
    return model, framework


def frame(index=0):
    return VideoFrame(
        index, None, 123, "synthetic", np.zeros((24, 32, 3), dtype=np.uint8)
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": True},
        {"iou_threshold": -1},
        {"iou_threshold": 1.1},
        {"device": "cuda"},
        {"max_detections": 0},
        {"max_detections": True},
        {"image_size_px": 33},
        {"image_size_px": 4096},
        {"artifact_sha256": "bad"},
        {"model_id": "../bad"},
        {"classes": ("ripe", "ripe")},
        {"classes": ()},
        {"classes": (1,)},
        {"ignore_top_fraction": -0.1},
        {"ignore_top_fraction": 0.6},
        {"ignore_top_fraction": True},
        {"ignore_top_fraction": float("nan")},
    ],
)
def test_config_validation(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def write_config(config, tmp_path):
    path = tmp_path / "detector.json"
    path.write_text(
        json.dumps(
            {
                "config_version": "1.0",
                "model_path": config.model_path.name,
                "model_id": config.model_id,
                "artifact_sha256": config.artifact_sha256,
                "classes": list(config.classes),
            }
        )
    )
    return path


def test_config_file_boundaries(config, tmp_path, monkeypatch):
    path = write_config(config, tmp_path)
    monkeypatch.chdir("/")
    assert load_detector_config(path) == config
    for contents in (
        '{"config_version":"1.0","config_version":"1.0"}',
        "{}",
        "[]",
        '{"x":NaN}',
        "x" * 65537,
    ):
        path.write_text(contents)
        with pytest.raises(ValueError):
            load_detector_config(path)
    write_config(config, tmp_path)
    data = json.loads(path.read_text())
    data["extra"] = "no"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_detector_config(path)
    with pytest.raises(ValueError):
        load_detector_config(tmp_path)
    with pytest.raises(ValueError):
        load_detector_config(tmp_path / "absent")


def test_hash_verified_before_import(config, monkeypatch):
    importer = Mock(side_effect=AssertionError("must not import"))
    monkeypatch.setattr(detector, "import_module", importer)
    with pytest.raises(DetectorError, match="SHA256"):
        UltralyticsDetector(replace(config, artifact_sha256="0" * 64))
    importer.assert_not_called()


def test_backend_contract_and_cleanup(config, monkeypatch):
    model, _ = fake_model(monkeypatch, [[1, 2, 30, 20, 0.9, 1]])
    adapter = UltralyticsDetector(config)
    original = frame()
    assert adapter.detect(original) == (
        Detection(1, "unripe", 0.9, (1.0, 2.0, 30.0, 20.0)),
    )
    kwargs = model.predict.call_args.kwargs
    assert kwargs["source"].flags.writeable
    kwargs["source"][:] = 255
    assert not original.image_bgr.any()
    assert kwargs["save"] is False and kwargs["augment"] is False
    assert adapter.metadata["artifact_sha256"] == config.artifact_sha256
    adapter.close()
    adapter.close()
    assert adapter._model is None
    with pytest.raises(DetectorError, match="closed"):
        adapter.detect(original)


def test_osd_is_cropped_before_inference_and_boxes_restored(config, monkeypatch):
    model, _ = fake_model(monkeypatch, [[1, 2, 30, 18, 0.9, 1]])
    model.predict.return_value[0].orig_shape = (19, 32)
    adapter = UltralyticsDetector(replace(config, ignore_top_fraction=0.2))
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)
    pixels[:5] = (0, 0, 255)  # Red OSD pixels must never reach the model.
    original = replace(frame(), image_bgr=pixels)
    assert adapter.detect(original) == (
        Detection(1, "unripe", 0.9, (1.0, 7.0, 30.0, 23.0)),
    )
    image = model.predict.call_args.kwargs["source"]
    assert image.shape == (19, 32, 3)
    assert not image.any()
    assert (original.image_bgr[:5, :, 2] == 255).all()
    adapter.close()


@pytest.mark.parametrize(
    "row",
    [
        [-1, 0, 4, 4, 0.9, 0],
        [0, 0, 33, 4, 0.9, 0],
        [4, 0, 4, 4, 0.9, 0],
        [0, 0, 4, 4, float("nan"), 0],
        [0, 0, 4, 4, 0.9, 2],
        [0, 0, 4, 4, 0.9, 0.5],
        [0, 0, 4, 4, 0.1, 0],
        [0, 0, float("inf"), 4, 0.9, 0],
    ],
)
def test_bad_model_outputs(config, monkeypatch, row):
    fake_model(monkeypatch, [row])
    adapter = UltralyticsDetector(config)
    with pytest.raises(DetectorError, match="frame 0"):
        adapter.detect(frame())
    adapter.close()


def test_empty_results_and_model_errors(config, monkeypatch):
    model, _ = fake_model(monkeypatch)
    adapter = UltralyticsDetector(config)
    assert adapter.detect(frame()) == ()
    model.predict.return_value = []
    with pytest.raises(DetectorError):
        adapter.detect(frame())
    model.predict.side_effect = RuntimeError("bad tensor")
    with pytest.raises(DetectorError, match="bad tensor"):
        adapter.detect(frame())
    adapter.close()
    model.names = {0: "wrong"}
    with pytest.raises(DetectorError, match="classes"):
        UltralyticsDetector(config)
    with pytest.raises(DetectorError, match="MPS"):
        UltralyticsDetector(replace(config, device="mps"))


def test_optional_dependency_is_lazy_and_imports_are_independent():
    import subprocess
    import sys

    for module in (
        "ad4pt.vision.config",
        "ad4pt.vision.detector",
        "ad4pt.cli",
    ):
        subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {module}; import sys; "
                "assert 'torch' not in sys.modules; "
                "assert 'ultralytics' not in sys.modules",
            ],
            check=True,
        )


def test_missing_dependency_and_wrong_framework(config, monkeypatch):
    monkeypatch.setattr(
        detector, "import_module", Mock(side_effect=ImportError("missing torch"))
    )
    with pytest.raises(DetectorError, match="install"):
        UltralyticsDetector(config)
    _, framework = fake_model(monkeypatch)
    framework.__version__ = "unexpected"
    with pytest.raises(DetectorError, match="8.4.14"):
        UltralyticsDetector(config)


def test_tracking_preserves_yolo_boxes_scores_labels_and_pending_fruit(
    config, monkeypatch, tmp_path
):
    model, _ = fake_model(
        monkeypatch,
        [
            [1, 2, 30, 18, 0.9, 1],
            [2, 1, 8, 8, 0.19, 0],
            [4, 4, 12, 12, 0.32, 1],
        ],
    )
    model.predict.return_value[0].orig_shape = (19, 32)
    tracker = Mock()
    # Only ID/index matter: ignore smoothed geometry, class and score.
    tracker.update.return_value = np.array(
        [
            [-50, -60, 1000, 1000, 7, 0.1, 0, 0],
            [0, 0, 31, 18, 8, 0.9, 1, 1],
        ],
        dtype=float,
    )
    factory = Mock(return_value=tracker)
    monkeypatch.setattr(detector, "create_tracker", factory)
    adapter = UltralyticsDetector(replace(config, ignore_top_fraction=0.2))
    path = tmp_path / "botsort.yaml"
    expected = (
        Detection(1, "unripe", 0.9, (1.0, 7.0, 30.0, 23.0), track_id=7),
        Detection(1, "unripe", 0.32, (4.0, 9.0, 12.0, 17.0)),
    )
    assert adapter.track(frame(), path) == expected
    assert model.predict.call_args.kwargs["conf"] == 0.1
    assert model.predict.call_args.kwargs["source"].shape == (19, 32, 3)
    assert tracker.update.call_args.args[1].shape == (19, 32, 3)
    factory.assert_called_once_with(path.resolve(), 0.25)
    model.track.assert_not_called()
    assert adapter.track(frame(), path) == expected
    assert factory.call_count == 1
    # Stateless detection still works after tracking: no model callbacks changed.
    model.predict.return_value[
        0
    ].boxes.data.detach.return_value.cpu.return_value.numpy.return_value = np.array(
        [
            [1, 2, 30, 18, 0.9, 1],
            [4, 4, 12, 12, 0.32, 1],
        ]
    )
    assert adapter.detect(frame()) == tuple(replace(d, track_id=None) for d in expected)
    adapter.close()
    assert adapter._tracker is None


@pytest.mark.parametrize(
    "row",
    [
        [0, 0, 10, 10, 1, 0.9, 0, 99],
        [0, 0, 10, 10, 1.5, 0.9, 0, 0],
        [0, 0, 10, 10, 0, 0.9, 0, 0],
    ],
)
def test_tracker_assignment_validation(config, monkeypatch, tmp_path, row):
    fake_model(monkeypatch, [[1, 2, 30, 18, 0.9, 1]])
    tracker = Mock()
    tracker.update.return_value = np.array([row], dtype=float)
    monkeypatch.setattr(detector, "create_tracker", lambda *args: tracker)
    adapter = UltralyticsDetector(config)
    with pytest.raises(DetectorError, match="track ID or detection index"):
        adapter.track(frame(), tmp_path / "botsort.yaml")
    adapter.close()


def test_live_adapter_handles_mixed_strong_and_weak_tracks(config, monkeypatch):
    from pathlib import Path

    import torch
    from ultralytics.engine.results import Boxes

    from ad4pt.vision.tracking import create_tracker

    model, _ = fake_model(monkeypatch)
    tracker = create_tracker(
        Path(__file__).resolve().parents[1] / "config" / "strawberry_botsort.yaml", 0.25
    )
    monkeypatch.setattr(detector, "create_tracker", lambda *args: tracker)
    adapter = UltralyticsDetector(config)
    rows_by_frame = [
        [[1, 2, 10, 12, 0.9, 0], [20, 2, 30, 12, 0.9, 1]],
        [[1, 2, 10, 12, 0.9, 0], [20, 2, 30, 12, 0.19, 1]],
        [[20, 2, 30, 12, 0.19, 1], [1, 2, 10, 12, 0.9, 0]],
        [[1, 2, 10, 12, 0.9, 0], [20, 2, 30, 12, 0.32, 1]],
    ]
    original_ids = {}
    for index, rows in enumerate(rows_by_frame):
        model.predict.return_value = [
            SimpleNamespace(
                orig_shape=(24, 32),
                boxes=Boxes(torch.tensor(rows, dtype=torch.float32), (24, 32)),
            )
        ]
        result = adapter.track(frame(index), Path("test-tracker.yaml"))
        assert len(result) == (2 if index in (0, 3) else 1)
        assert all(d.confidence >= 0.25 for d in result)
        for detection in result:
            if index == 0:
                original_ids[detection.class_id] = detection.track_id
            else:
                assert detection.track_id == original_ids[detection.class_id]
    adapter.close()


@pytest.mark.parametrize("value", [0, -0.1, 1.1, True, float("nan")])
def test_box_area_config_validation(config, value):
    with pytest.raises(ValueError, match="max_box_area_fraction"):
        replace(config, max_box_area_fraction=value)


def test_large_boxes_removed_before_real_tracker_and_ids_stay_aligned(
    config, monkeypatch
):
    from pathlib import Path

    import torch
    from ultralytics.engine.results import Boxes

    from ad4pt.counting import StrawberryCounter
    from ad4pt.vision.tracking import create_tracker

    model, _ = fake_model(monkeypatch)
    tracker = create_tracker(Path("config/strawberry_botsort.yaml"), 0.25)
    monkeypatch.setattr(detector, "create_tracker", lambda *args: tracker)
    adapter = UltralyticsDetector(replace(config, max_box_area_fraction=0.15))
    counter = StrawberryCounter(3, 0.25)
    ids = {}
    for index in range(4):
        # Giant false ripe box first; accepted rows must get new relative indices.
        rows = [[0, 0, 32, 24, 0.95, 0], [1, 2, 9, 12, 0.9, 0], [20, 2, 28, 12, 0.9, 1]]
        if index == 3:
            rows = rows[:1]  # All rejected: tracker still receives an empty frame.
        model.predict.return_value = [
            SimpleNamespace(
                orig_shape=(24, 32), boxes=Boxes(torch.tensor(rows), (24, 32))
            )
        ]
        result = adapter.track(frame(index), Path("tracker.yaml"))
        assert len(result) == (0 if index == 3 else 2)
        for d in result:
            if index == 0:
                ids[d.class_id] = d.track_id
            assert d.track_id == ids[d.class_id]
        counter.update(result)
    assert counter.counts == {"ripe": 1, "unripe": 1}
    adapter.close()


def test_area_filter_uses_cropped_image_and_can_be_disabled(config, monkeypatch):
    model, _ = fake_model(monkeypatch, [[0, 0, 8, 12, 0.9, 1]])
    model.predict.return_value[0].orig_shape = (12, 32)
    adapter = UltralyticsDetector(
        replace(config, ignore_top_fraction=0.5, max_box_area_fraction=0.15)
    )
    assert adapter.detect(frame()) == ()  # 25% of cropped area, 12.5% of full frame.
    adapter.close()
    adapter = UltralyticsDetector(replace(config, ignore_top_fraction=0.5))
    assert len(adapter.detect(frame())) == 1
    adapter.close()


@pytest.mark.parametrize("value", [True, 0.1, -1, 1.1, float("nan"), float("inf")])
def test_unripe_threshold_validation(config, value):
    with pytest.raises(ValueError, match="unripe_confidence_threshold"):
        replace(config, unripe_confidence_threshold=value)


def test_stricter_unripe_threshold_preserves_ripe_scores_and_raw_model_input(
    config, monkeypatch, tmp_path
):
    fake_model(
        monkeypatch,
        [[1, 2, 8, 12, 0.32, 0], [10, 2, 17, 12, 0.42, 1], [20, 2, 28, 12, 0.5, 1]],
    )
    tracker = Mock()
    tracker.update.return_value = np.array(
        [
            [1, 2, 8, 12, 1, 0.32, 0, 0],
            [10, 2, 17, 12, 2, 0.42, 1, 1],
            [20, 2, 28, 12, 3, 0.5, 1, 2],
        ]
    )
    monkeypatch.setattr(detector, "create_tracker", lambda *args: tracker)
    adapter = UltralyticsDetector(replace(config, unripe_confidence_threshold=0.5))
    plain = adapter.detect(frame())
    assert [(d.label, d.confidence) for d in plain] == [("ripe", 0.32), ("unripe", 0.5)]
    tracked = adapter.track(frame(), tmp_path / "tracker.yaml")
    assert tuple(replace(d, track_id=None) for d in tracked) == plain
    assert [d.track_id for d in tracked] == [1, 3]
    adapter.close()
