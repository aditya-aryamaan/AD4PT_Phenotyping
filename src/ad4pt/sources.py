"""Shared contract and errors for the live frame receiver."""

from collections.abc import Iterator
from types import TracebackType
from typing import Protocol, Self

from ad4pt.types import VideoFrame


class FrameStream(Protocol):
    """Structural contract for live sources and the capture worker."""

    def __iter__(self) -> Iterator[VideoFrame]: ...
    def __next__(self) -> VideoFrame: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class SourceError(Exception):
    """Input cannot be opened or decoded reliably."""
