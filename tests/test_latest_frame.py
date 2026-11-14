import threading
import weakref
from unittest.mock import Mock

import numpy as np
import pytest

from ad4pt.latest_frame import FrameCounts, LatestFrameBuffer, LatestFrameSource
from ad4pt.live import StreamTimeout
from ad4pt.types import VideoFrame


def frame(index):
    return VideoFrame(index, None, index + 10, "test", np.zeros((2, 3, 3), np.uint8))


class Source:
    def __init__(self, values):
        self.values = iter(values)
        self.closed_on = None

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed_on = threading.get_ident()
        if hasattr(self.values, "close"):
            self.values.close()


def test_slot_replacement_memory_and_accounting():
    buffer = LatestFrameBuffer()
    refs = []
    for i in range(1000):
        value = frame(i)
        refs.append(weakref.ref(value))
        buffer.publish(value)
    del value
    assert sum(ref() is not None for ref in refs) == 1
    assert buffer.counts == FrameCounts(1000, 0, 999, 0, 1)
    assert buffer.take().frame_id == 999
    buffer.finish()
    with pytest.raises(StopIteration):
        buffer.take()
    assert buffer.counts == FrameCounts(1000, 1, 999, 0, 0)


def test_finish_drains_final_frame_then_error():
    buffer = LatestFrameBuffer()
    buffer.publish(frame(0))
    buffer.finish(StreamTimeout("lost stream"))
    assert buffer.take().frame_id == 0
    with pytest.raises(StreamTimeout, match="lost stream"):
        buffer.take()
    with pytest.raises(StopIteration):
        buffer.take()


def test_close_wakes_reader_and_discards_late_publish():
    buffer = LatestFrameBuffer()
    done = threading.Event()

    def reader():
        with pytest.raises(StopIteration):
            buffer.take()
        done.set()

    thread = threading.Thread(target=reader)
    thread.start()
    buffer.close()
    thread.join(1)
    assert done.is_set() and not thread.is_alive()
    buffer.publish(frame(0))
    assert buffer.counts == FrameCounts(1, 0, 0, 1, 0)
    buffer.close()


def test_slow_consumer_gets_newest_not_backlog():
    advance, burst_done = threading.Event(), threading.Event()
    owners = []
    raw = None

    def factory(stop):
        nonlocal raw
        owners.append(threading.get_ident())

        def values():
            yield frame(0)
            assert advance.wait(1)
            for index in range(1, 101):
                yield frame(index)
            burst_done.set()
            stop.wait(1)

        raw = Source(values())
        return raw

    with LatestFrameSource(factory) as source:
        assert next(source).frame_id == 0
        advance.set()
        assert burst_done.wait(1)
        newest = next(source)
        assert newest.frame_id == 100
        assert newest.received_at_ns == 110
        assert source.counts == FrameCounts(101, 2, 99, 0, 0)
    assert raw.closed_on == owners[0] != threading.get_ident()
    assert not source._thread.is_alive()


def test_cancel_before_first_frame_and_waiting_worker():
    started = threading.Event()
    raw = None

    def factory(stop):
        nonlocal raw

        def values():
            started.set()
            assert stop.wait(1)
            yield from ()

        raw = Source(values())
        return raw

    source = LatestFrameSource(factory)

    # Main consumer can be woken by close from a control thread.
    def cancel():
        assert started.wait(1)
        source.close()

    control = threading.Thread(target=cancel)
    control.start()
    assert list(source) == []
    control.join(1)
    assert not control.is_alive()
    source._thread.join(1)
    assert not source._thread.is_alive() and raw.closed_on is not None


def test_eos_limits_and_close_before_start():
    factory = Mock()
    source = LatestFrameSource(factory)
    source.close()
    assert list(source) == []
    factory.assert_not_called()
    for limit in (None, 1):
        raw = Source([frame(i) for i in range(10)])
        with LatestFrameSource(lambda stop, raw=raw: raw, max_frames=limit) as source:
            result = list(source)
        assert 1 <= len(result) <= (limit or 10)
        assert result[-1].frame_id <= 9
        counts = source.counts
        assert counts.captured == counts.processed + counts.replaced + counts.discarded
        assert counts.pending == 0
        assert raw.closed_on is not None


@pytest.mark.parametrize("where", ["factory", "read", "consumer"])
def test_exceptions_join_and_propagate(where):
    raw = None

    def factory(stop):
        nonlocal raw
        if where == "factory":
            raise StreamTimeout("failed")

        def values():
            if where == "read":
                raise StreamTimeout("failed")
            yield frame(0)
            stop.wait(1)

        raw = Source(values())
        return raw

    with pytest.raises((StreamTimeout, RuntimeError)):
        with LatestFrameSource(factory) as source:
            next(source)
            raise RuntimeError("consumer failed")
    assert source.closed and not source._thread.is_alive()
    if raw is not None:
        assert raw.closed_on is not None


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        LatestFrameSource(Mock(), max_frames=limit)
