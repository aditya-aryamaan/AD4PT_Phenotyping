"""One-slot frame exchange and a cooperative capture worker for live sources."""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from threading import Condition, Event, Thread
from types import TracebackType
from typing import Self

from ad4pt.sources import FrameStream, SourceError
from ad4pt.types import VideoFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameCounts:
    captured: int
    processed: int
    replaced: int
    discarded: int
    pending: int


class LatestFrameBuffer:
    """Thread-safe single slot. Taking a frame removes it; no duplicate delivery."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._frame: VideoFrame | None = None
        self._finished = False
        self._error: BaseException | None = None
        self._captured = self._processed = self._replaced = self._discarded = 0

    @property
    def counts(self) -> FrameCounts:
        with self._condition:
            return FrameCounts(
                self._captured,
                self._processed,
                self._replaced,
                self._discarded,
                int(self._frame is not None),
            )

    def publish(self, frame: VideoFrame) -> None:
        with self._condition:
            self._captured += 1
            if self._finished:
                self._discarded += 1
                return
            if self._frame is not None:
                self._replaced += 1
            self._frame = frame
            self._condition.notify_all()

    def take(self, timeout_s: float | None = None) -> VideoFrame:
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._frame is not None or self._finished, timeout=timeout_s
            ):
                raise TimeoutError("waiting for a new frame")
            if self._frame is not None:
                frame, self._frame = self._frame, None
                self._processed += 1
                return frame
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            raise StopIteration

    def finish(self, error: BaseException | None = None) -> None:
        """Producer completion: drain the final slot, then report any failure."""
        with self._condition:
            if not self._finished:
                self._finished = True
                self._error = error
            self._condition.notify_all()

    def close(self) -> None:
        """Consumer cancellation: discard pending work and wake blocked readers."""
        with self._condition:
            self._finished = True
            self._discarded += int(self._frame is not None)
            self._frame = None
            self._error = None
            self._condition.notify_all()


class LatestFrameSource(Iterator[VideoFrame]):
    """Single consumer; factory creates/closes the source on the capture thread.

    The factory must honor the supplied stop event (RtpFrameSource does). Use a
    context manager for early breaks. max_frames limits delivered, not captured,
    frames. Source frame IDs and timestamps are preserved, including ID gaps.
    """

    def __init__(
        self, factory: Callable[[Event], FrameStream], *, max_frames: int | None = None
    ) -> None:
        if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer")
        self._factory = factory
        self._max_frames = max_frames
        self._stop = Event()
        self._buffer = LatestFrameBuffer()
        self._thread: Thread | None = None
        self.closed = False

    @property
    def counts(self) -> FrameCounts:
        return self._buffer.counts

    @property
    def frames_read(self) -> int:
        return self.counts.processed

    def _capture(self) -> None:
        error: BaseException | None = None
        try:
            with self._factory(self._stop) as source:
                for frame in source:
                    self._buffer.publish(frame)
                    if self._stop.is_set():
                        break
        except BaseException as exc:
            error = exc
        finally:
            self._buffer.finish(error)

    def __iter__(self) -> Self:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __next__(self) -> VideoFrame:
        return self.poll()

    def poll(self, timeout_s: float | None = None) -> VideoFrame:
        """Timed wait lets a GUI service events while capture is idle."""
        if self.closed:
            raise StopIteration
        try:
            if self._thread is None:
                # Daemon is only a last resort for a wedged native plugin. Normal
                # shutdown always joins and never closes Gst from a second thread.
                self._thread = Thread(
                    target=self._capture, name="ad4pt-capture", daemon=True
                )
                self._thread.start()
            frame = self._buffer.take(timeout_s)
            if self.frames_read == self._max_frames:
                self.close()
            return frame
        except TimeoutError:
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        self._buffer.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise SourceError(
                    "capture worker did not stop within 2 s; native source may be stuck"
                )
        counts = self.counts
        logger.info(
            "Latest-frame worker stopped: captured=%d processed=%d "
            "replaced=%d discarded=%d",
            counts.captured,
            counts.processed,
            counts.replaced,
            counts.discarded,
        )
