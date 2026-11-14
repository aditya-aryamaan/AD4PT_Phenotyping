"""Cancellable live iterator; all buffering belongs to GStreamer, not Python."""

import logging
import time
from collections.abc import Callable, Iterator
from threading import Event
from types import TracebackType
from typing import Protocol, Self

from ad4pt.live_config import RtpConfig
from ad4pt.sources import SourceError
from ad4pt.types import VideoFrame

logger = logging.getLogger(__name__)


class StreamTimeout(SourceError):
    """No decoded frame arrived within the configured wait interval."""


class StreamBackend(Protocol):
    """Narrow injection boundary for the optional native GStreamer runtime."""

    @property
    def eos(self) -> bool: ...
    def start(self) -> None: ...
    def pull(self, timeout_ns: int, frame_id: int) -> VideoFrame | None: ...
    def close(self) -> None: ...


class RtpFrameSource(Iterator[VideoFrame]):
    """Single-consumer live source with an iterator/context contract.

    Set stop_event from another thread to cancel a wait; close on the consumer
    thread. No worker thread or application frame queue is created here.
    """

    def __init__(
        self,
        config: RtpConfig,
        *,
        backend: StreamBackend | None = None,
        stop_event: Event | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if backend is None:
            from ad4pt.gstreamer import GstBackend

            backend = GstBackend(config)
        self.config = config
        self._backend = backend
        self._stop = stop_event if stop_event is not None else Event()
        self._clock = clock
        self._started = False
        self.closed = False
        self.frames_read = 0

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
        if self.closed:
            raise StopIteration
        try:
            if self._stop.is_set() or self.frames_read == self.config.max_frames:
                self.close()
                raise StopIteration
            if not self._started:
                self._backend.start()
                self._started = True
            deadline_ns = self._clock() + int(self.config.timeout_s * 1e9)
            while not self._stop.is_set():
                remaining_ns = deadline_ns - self._clock()
                if remaining_ns <= 0:
                    logger.warning("No decoded frame for %.3f s", self.config.timeout_s)
                    raise StreamTimeout(
                        f"no decoded frame within {self.config.timeout_s:g} s on "
                        f"UDP port {self.config.port}; check Wi-Fi and RTP sender"
                    )
                poll_ns = min(remaining_ns, 100_000_000)
                poll_started_ns = self._clock()
                frame = self._backend.pull(poll_ns, self.frames_read)
                if self._stop.is_set():
                    break
                if frame is not None:
                    if self.frames_read == 0:
                        logger.info(
                            "First frame received: %s x %s px BGR",
                            frame.image_bgr.shape[1],
                            frame.image_bgr.shape[0],
                        )
                    self.frames_read += 1
                    if self.frames_read == self.config.max_frames:
                        self.close()
                    return frame
                if self._backend.eos:
                    logger.info("GStreamer EOS")
                    break
                # Some native states return immediately; avoid spinning before PLAYING.
                rest_ns = poll_ns - (self._clock() - poll_started_ns)
                if rest_ns > 0:
                    self._stop.wait(rest_ns / 1e9)
            self.close()
            raise StopIteration
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._stop.set()
            self._backend.close()
            logger.info("RTP receiver stopped (%d frames read)", self.frames_read)
