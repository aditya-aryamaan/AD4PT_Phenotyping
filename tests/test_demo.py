import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_detection import config as config  # pytest fixture
from test_detection import frame

from ad4pt.cli import main
from ad4pt.latest_frame import LatestFrameBuffer
from ad4pt.live import StreamTimeout
from ad4pt.vision import demo
from ad4pt.vision.detector import Detection, DetectorError


@pytest.mark.parametrize(
    "stop,code,reason,frames",
    [
        ("window", 0, "window_closed", 1),
        ("interrupt", 130, "interrupted", 1),
        ("error", 3, "error", 0),
        ("idle", 0, "window_closed", 0),
        ("eos", 0, "stream_ended", 1),
        ("timeout", 0, "stream_timeout", 1),
        ("no_stream", 4, "no_stream", 0),
        ("limit", 0, "max_frames", 1),
    ],
)
def test_demo_saves_counts_and_cleans_up(
    config, tmp_path, monkeypatch, capsys, stop, code, reason, frames
):
    model = Mock()
    model.track.return_value = (Detection(0, "ripe", 0.9, (1, 2, 10, 12), track_id=1),)
    preview = Mock()
    preview.open.side_effect = [True, False]
    source = Mock()
    source.__enter__ = Mock(return_value=source)
    source.__exit__ = Mock(return_value=False)
    source.poll.return_value = frame()
    if stop == "idle":
        source.poll.side_effect = TimeoutError
    elif stop == "interrupt":
        preview.show.side_effect = KeyboardInterrupt
    elif stop == "error":
        model.track.side_effect = DetectorError("test")
    elif stop in ("eos", "limit", "timeout"):
        preview.open.side_effect = [True, True]
        end = StreamTimeout("lost stream") if stop == "timeout" else StopIteration
        source.poll.side_effect = [frame(), end]
    elif stop == "no_stream":
        source.poll.side_effect = StreamTimeout("no stream")
    monkeypatch.setattr(demo, "load_detector_config", lambda path: config)
    monkeypatch.setattr(demo, "UltralyticsDetector", lambda cfg: model)
    monkeypatch.setattr(demo, "LatestFrameSource", lambda *a, **kw: source)
    monkeypatch.setattr(demo, "Preview", lambda **kwargs: preview)
    root = tmp_path / "runs"
    args = ["demo", "--device", "cpu", "--run-root", str(root)]
    if stop == "limit":
        args += ["--max-frames", "1"]
    assert main(args) == code
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    path = Path(result.pop("counts_json"))
    assert json.loads(path.read_text()) == result
    assert list(root.iterdir()) == [path]  # A single file, no per-frame logs.
    assert result["stop_reason"] == reason
    assert result["frames_processed"] == frames
    assert result["ripe"] == 0
    assert result["unripe"] == 0
    assert result["total"] == 0
    assert result["started_at_utc"].endswith("Z")
    assert result["ended_at_utc"].endswith("Z")
    model.close.assert_called_once()
    preview.close.assert_called_once()
    source.__exit__.assert_called_once()


def test_repeated_frames_count_fruit_once(config, tmp_path, monkeypatch, capsys):
    model = Mock()
    model.track.return_value = (
        Detection(0, "ripe", 0.9, (1, 2, 10, 12), track_id=1),
        Detection(1, "unripe", 0.9, (15, 2, 25, 12), track_id=2),
    )
    preview = Mock()
    preview.open.side_effect = [True] * 20 + [False]
    source = Mock()
    source.__enter__ = Mock(return_value=source)
    source.__exit__ = Mock(return_value=False)
    source.poll.return_value = frame()
    monkeypatch.setattr(demo, "load_detector_config", lambda path: config)
    monkeypatch.setattr(demo, "UltralyticsDetector", lambda cfg: model)
    monkeypatch.setattr(demo, "LatestFrameSource", lambda *a, **kw: source)
    monkeypatch.setattr(demo, "Preview", lambda **kwargs: preview)
    output = tmp_path / "counts.json"
    assert main(["demo", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert (result["ripe"], result["unripe"], result["total"]) == (1, 1, 2)
    assert result["frames_processed"] == 20
    capsys.readouterr()


def test_model_failure_still_saves_zero_counts(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, "load_detector_config", lambda path: config)
    monkeypatch.setattr(
        demo, "UltralyticsDetector", Mock(side_effect=DetectorError("model failed"))
    )
    output = tmp_path / "counts.json"
    assert main(["demo", "--output", str(output)]) == 3
    assert json.loads(output.read_text())["total"] == 0
    assert json.loads(capsys.readouterr().out)["stop_reason"] == "error"


def test_save_failure_is_reported(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(demo, "load_detector_config", lambda path: config)
    monkeypatch.setattr(
        demo, "UltralyticsDetector", Mock(side_effect=DetectorError("model failed"))
    )
    monkeypatch.setattr(demo, "save_json", Mock(side_effect=OSError("disk full")))
    assert main(["demo", "--output", str(tmp_path / "counts.json")]) == 5
    captured = capsys.readouterr()
    assert not captured.out
    assert "disk full" in captured.err


def test_timed_buffer_does_not_finish_source():
    buffer = LatestFrameBuffer()
    with pytest.raises(TimeoutError):
        buffer.take(0.001)
    buffer.publish(frame())
    assert buffer.take(0.001).frame_id == 0
    buffer.finish()
    with pytest.raises(StopIteration):
        buffer.take(0.001)


def test_optional_capture_saves_raw_pixels_and_default_saves_nothing(
    tmp_path, monkeypatch
):
    import cv2
    import numpy as np

    monkeypatch.setattr(demo.cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(demo.cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(demo.cv2, "waitKey", lambda *a: ord("s"))
    monkeypatch.setattr(demo.cv2, "getWindowProperty", lambda *a: 1)
    raw = frame()
    preview = demo.Preview(capture_dir=tmp_path)
    preview.show(raw, (Detection(0, "ripe", 0.9, (1, 2, 10, 12)),))
    assert preview.open()
    saved = list(tmp_path.glob("strawberry-frame_*.png"))
    assert len(saved) == 1
    assert np.array_equal(cv2.imread(str(saved[0])), raw.image_bgr)
    preview = demo.Preview()
    preview.show(raw, ())
    assert preview.open()
    assert len(list(tmp_path.iterdir())) == 1


def test_fixed_scene_mode_bypasses_tracker_and_saves_its_method(
    config, tmp_path, monkeypatch, capsys
):
    model = Mock()
    model.detect.return_value = tuple(
        Detection(0, "ripe", 0.9, (i * 5, 2, i * 5 + 4, 12)) for i in range(4)
    )
    preview = Mock()
    preview.open.side_effect = [True] * 20 + [False]
    source = Mock()
    source.__enter__ = Mock(return_value=source)
    source.__exit__ = Mock(return_value=False)
    source.poll.return_value = frame()
    monkeypatch.setattr(demo, "load_detector_config", lambda path: config)
    monkeypatch.setattr(demo, "UltralyticsDetector", lambda cfg: model)
    monkeypatch.setattr(demo, "LatestFrameSource", lambda *a, **kw: source)
    monkeypatch.setattr(demo, "Preview", lambda **kwargs: preview)
    output = tmp_path / "scene.json"
    assert main(["demo", "--counting-mode", "scene", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["ripe"] == 4 and result["unripe"] == 0
    assert result["counting_method"] == "static_scene_confirmed_modal_detections"
    assert result["scene_mode_support_frames"] == 16
    assert result["track_buffer_frames"] is None
    assert model.detect.call_count == 20
    model.track.assert_not_called()
    capsys.readouterr()
