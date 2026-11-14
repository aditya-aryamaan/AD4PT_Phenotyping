import logging
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from ad4pt import gstreamer
from ad4pt.gstreamer import GstBackend, sample_to_frame
from ad4pt.live_config import RtpConfig
from ad4pt.sources import SourceError


def sample_parts(width=3, height=2, stride=12, offset=0, data=None):
    if data is None:
        data = bytearray(range(offset + stride * height))
    mapping = NS(data=data)
    buffer = Mock()
    buffer.map.return_value = (True, mapping)
    sample = Mock()
    sample.get_buffer.return_value = buffer
    info = NS(
        width=width,
        height=height,
        stride=[stride],
        offset=[offset],
        finfo=NS(name="BGR"),
    )
    video = NS(
        VideoInfo=Mock(),
        buffer_get_video_meta=Mock(return_value=None),
        VideoFormat=NS(BGR="BGR"),
    )
    video.VideoInfo.new_from_caps.return_value = info
    gst = NS(MapFlags=NS(READ=1))
    return sample, gst, video, buffer, mapping, info


def convert(parts):
    return sample_to_frame(
        *parts[:3], frame_id=7, received_at_ns=123, source="rtp://test"
    )


def test_bgr_stride_copy_and_unmap():
    parts = sample_parts(offset=4)
    frame = convert(parts)
    assert frame.frame_id == 7 and frame.received_at_ns == 123
    assert frame.captured_at_ns is None
    assert frame.image_bgr.shape == (2, 3, 3)
    np.testing.assert_array_equal(frame.image_bgr[0].ravel(), range(4, 13))
    np.testing.assert_array_equal(frame.image_bgr[1].ravel(), range(16, 25))
    parts[4].data[:] = bytes(len(parts[4].data))
    assert frame.image_bgr[0, 0, 0] == 4
    with pytest.raises(ValueError):
        frame.image_bgr.setflags(write=True)
    parts[3].unmap.assert_called_once_with(parts[4])


def test_video_meta_overrides_stride():
    parts = sample_parts(stride=16)
    parts[2].buffer_get_video_meta.return_value = NS(
        n_planes=1, width=3, height=2, format="BGR", stride=[12], offset=[2]
    )
    frame = convert(parts)
    assert frame.image_bgr[1, 0, 0] == 14
    parts[2].buffer_get_video_meta.return_value.n_planes = 2
    with pytest.raises(SourceError, match="metadata"):
        convert(parts)


@pytest.mark.parametrize(
    "problem", ["caps", "buffer", "format", "size", "stride", "map", "short"]
)
def test_bad_sample(problem):
    parts = sample_parts()
    sample, gst, video, buffer, mapping, info = parts
    if problem == "caps":
        sample.get_caps.return_value = None
    elif problem == "buffer":
        sample.get_buffer.return_value = None
    elif problem == "format":
        info.finfo.name = "RGB"
    elif problem == "size":
        info.width = 10000
    elif problem == "stride":
        info.stride = [1]
    elif problem == "map":
        buffer.map.return_value = (False, None)
    elif problem == "short":
        mapping.data = b"short"
    with pytest.raises(SourceError):
        convert(parts)
    assert buffer.unmap.call_count == (1 if problem == "short" else 0)


def native_backend(monkeypatch, samples=None):
    sample, gst, video, *_ = sample_parts()
    gst.State = NS(PLAYING=1, NULL=0)
    gst.StateChangeReturn = NS(FAILURE=-1, SUCCESS=1)
    gst.MessageType = NS(ERROR=1, WARNING=2, EOS=3, STATE_CHANGED=4)
    gst.ElementFactory = Mock()
    sink = Mock()
    sink.emit.side_effect = samples if samples is not None else [sample]
    sink.get_property.return_value = False
    bus = Mock()
    bus.pop.return_value = None
    pipeline = Mock()
    pipeline.get_by_name.return_value = sink
    pipeline.get_bus.return_value = bus
    pipeline.set_state.return_value = 1
    gst.parse_launch = Mock(return_value=pipeline)
    monkeypatch.setattr(gstreamer, "_load_gst", lambda: (gst, video))
    clock = iter([100, 200, 300])
    backend = GstBackend(
        RtpConfig(expected_width_px=3, expected_height_px=2), clock=lambda: next(clock)
    )
    return backend, gst, pipeline, sink, bus


def test_native_start_pull_and_close(monkeypatch):
    backend, gst, pipeline, sink, bus = native_backend(monkeypatch)
    sink.emit.side_effect = [sample_parts()[0], sample_parts()[0]]
    backend.start()
    frame = backend.pull(1000, 0)
    assert frame.received_at_ns == 100 and frame.frame_id == 0
    assert frame.source == "rtp://0.0.0.0:5600"
    sink.emit.assert_called_once_with("try-pull-sample", 1000)
    second = backend.pull(1000, 1)
    assert second.frame_id == 1
    assert second.received_at_ns == 200 > frame.received_at_ns
    backend.close()
    backend.close()
    assert [c.args[0] for c in pipeline.set_state.call_args_list] == [1, 0]


@pytest.mark.parametrize("problem", ["plugin", "parse", "state", "sink", "bus"])
def test_native_start_failure_cleanup(monkeypatch, problem):
    backend, gst, pipeline, sink, bus = native_backend(monkeypatch)
    if problem == "plugin":
        gst.ElementFactory.find.return_value = None
    elif problem == "parse":
        gst.parse_launch.side_effect = RuntimeError("bad pipeline")
    elif problem == "state":
        pipeline.set_state.return_value = -1
    elif problem == "sink":
        pipeline.get_by_name.return_value = None
    elif problem == "bus":
        pipeline.get_bus.return_value = None
    with pytest.raises(SourceError):
        backend.start()
    if problem not in ("plugin", "parse"):
        assert pipeline.set_state.call_args.args == (0,)


def test_bus_warning_error_state_and_eos(monkeypatch, caplog):
    backend, gst, pipeline, sink, bus = native_backend(monkeypatch)
    backend.start()
    warning = Mock(type=gst.MessageType.WARNING)
    warning.parse_warning.return_value = ("warning detail", "debug detail")
    state = Mock(type=gst.MessageType.STATE_CHANGED, src=pipeline)
    state.parse_state_changed.return_value = (0, 1, 0)
    eos = Mock(type=gst.MessageType.EOS)
    bus.pop.side_effect = [warning, state, eos, None, None]
    with caplog.at_level(logging.INFO):
        assert backend.pull(1000, 0) is not None  # queued sample survives EOS
    assert backend.eos
    assert "warning detail" in caplog.text and "entered PLAYING" in caplog.text
    error = Mock(type=gst.MessageType.ERROR)
    error.parse_error.return_value = ("decoder failed", "debug")
    bus.pop.side_effect = [error]
    with pytest.raises(SourceError, match="decoder failed"):
        backend.pull(1000, 1)
    backend.close()


def test_none_sample_eos_and_native_exception(monkeypatch):
    backend, gst, pipeline, sink, bus = native_backend(monkeypatch, [None])
    backend.start()
    sink.get_property.return_value = True
    assert backend.pull(1000, 0) is None
    assert backend.eos
    sink.emit.side_effect = RuntimeError("native failure")
    with pytest.raises(SourceError, match="native failure"):
        backend.pull(1000, 0)
    backend.close()


def test_missing_gi_is_actionable(monkeypatch):
    monkeypatch.setattr(
        gstreamer.importlib, "import_module", Mock(side_effect=ImportError)
    )
    with pytest.raises(SourceError, match="PyGObject"):
        gstreamer._load_gst()
