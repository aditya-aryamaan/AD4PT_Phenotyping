import logging
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from ad4pt.live import RtpFrameSource, StreamTimeout
from ad4pt.live_config import RtpConfig
from ad4pt.sources import SourceError
from ad4pt.types import VideoFrame


def backend_with_frames(count):
    backend = Mock(eos=False)
    backend.pull.side_effect = [
        VideoFrame(i, None, 100 + i, "rtp://test", np.zeros((2, 3, 3), np.uint8))
        for i in range(count)
    ]
    return backend


def test_sequential_frames_and_limit(caplog):
    backend = backend_with_frames(3)
    with (
        caplog.at_level(logging.INFO),
        RtpFrameSource(RtpConfig(max_frames=3), backend=backend) as source,
    ):
        frames = list(source)
    assert [f.frame_id for f in frames] == [0, 1, 2]
    assert [f.received_at_ns for f in frames] == [100, 101, 102]
    assert all(f.captured_at_ns is None for f in frames)
    assert [call.args[1] for call in backend.pull.call_args_list] == [0, 1, 2]
    assert all(0 < call.args[0] <= 100_000_000 for call in backend.pull.call_args_list)
    assert caplog.text.count("First frame received") == 1
    backend.start.assert_called_once()
    backend.close.assert_called_once()
    source.close()
    assert list(source) == []


def test_stop_before_start_and_close_before_start():
    event = threading.Event()
    event.set()
    backend = Mock()
    source = RtpFrameSource(RtpConfig(), backend=backend, stop_event=event)
    assert list(source) == []
    backend.start.assert_not_called()
    backend.close.assert_called_once()
    backend = Mock()
    source = RtpFrameSource(RtpConfig(), backend=backend)
    source.close()
    assert list(source) == []
    backend.start.assert_not_called()


def test_cancel_while_waiting_no_spin():
    backend = Mock(eos=False)
    backend.pull.return_value = None  # simulate immediate native return before PLAYING
    event = threading.Event()
    timer = threading.Timer(0.05, event.set)
    timer.start()
    try:
        started = time.monotonic()
        with RtpFrameSource(RtpConfig(), backend=backend, stop_event=event) as source:
            assert list(source) == []
        assert time.monotonic() - started < 1
        assert backend.pull.call_count <= 2
        backend.close.assert_called_once()
    finally:
        timer.join()


def test_no_stream_and_loss_timeout():
    for prior_frames in (0, 1):
        backend = backend_with_frames(prior_frames)
        frames = list(backend.pull.side_effect)
        backend.pull.side_effect = None
        backend.pull.return_value = None
        with RtpFrameSource(RtpConfig(timeout_s=0.03), backend=backend) as source:
            if prior_frames:
                backend.pull.side_effect = frames
                next(source)
                backend.pull.side_effect = None
            with pytest.raises(StreamTimeout, match="UDP port 5600"):
                next(source)
        backend.close.assert_called_once()
        assert backend.pull.call_count <= prior_frames + 2


def test_delayed_first_frame_and_eos():
    backend = backend_with_frames(1)
    frame = next(backend.pull.side_effect)
    backend.pull.side_effect = [None, frame, None]
    backend.eos = False
    with RtpFrameSource(RtpConfig(timeout_s=1), backend=backend) as source:
        assert next(source).frame_id == 0
        backend.eos = True
        assert list(source) == []
    backend.close.assert_called_once()


@pytest.mark.parametrize("stage", ["start", "pull", "consumer", "interrupt"])
def test_exception_cleanup(stage):
    backend = backend_with_frames(1)
    if stage in ("start", "pull"):
        getattr(backend, stage).side_effect = SourceError("native failed")
    if stage == "interrupt":
        backend.pull.side_effect = KeyboardInterrupt
    with pytest.raises((SourceError, RuntimeError, KeyboardInterrupt)):
        with RtpFrameSource(RtpConfig(), backend=backend) as source:
            next(source)
            if stage == "consumer":
                raise RuntimeError("consumer failed")
    assert source.closed
    backend.close.assert_called_once()
