"""Direct PyGObject appsink adapter; gi is imported only on live startup."""

import importlib
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ad4pt.live_config import RtpConfig
from ad4pt.sources import SourceError
from ad4pt.types import VideoFrame

logger = logging.getLogger(__name__)


def build_pipeline(config: RtpConfig) -> str:
    """Only validated integers and one element identifier enter the launch string."""
    queue = "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    return (
        f"udpsrc port={config.port} buffer-size={config.udp_buffer_size_bytes} "
        f'caps="application/x-rtp,media=video,clock-rate={config.clock_rate},'
        f'encoding-name={config.encoding},payload={config.payload_type}" '
        f"! rtph264depay ! h264parse ! {queue} ! {config.decoder} ! {queue} "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink name=frames max-buffers=1 drop=true sync=false "
        "async=false emit-signals=false wait-on-eos=false enable-last-sample=false"
    )


def _load_gst() -> tuple[Any, Any]:
    # Dynamic GI namespaces have no bundled static stubs. Keep Any at this boundary.
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        gst = importlib.import_module("gi.repository.Gst")
        video = importlib.import_module("gi.repository.GstVideo")
        gst.init(None)
        if gst.version()[:2] < (1, 20):
            raise SourceError("live input requires GStreamer >= 1.20")
        return gst, video
    except (ImportError, ValueError, OSError) as exc:
        raise SourceError(
            "live input requires PyGObject (gi), GStreamer >= 1.20 and GstVideo "
            "in this Python environment; see README live setup"
        ) from exc


def sample_to_frame(
    sample: Any,
    gst: Any,
    video: Any,
    *,
    frame_id: int,
    received_at_ns: int,
    source: str,
) -> VideoFrame:
    """Copy mapped BGR pixels before unmapping, respecting row padding/VideoMeta."""
    caps, buffer = sample.get_caps(), sample.get_buffer()
    if caps is None or buffer is None:
        raise SourceError("GStreamer sample has no caps or buffer")
    info = video.VideoInfo.new_from_caps(caps)
    if info is None or info.finfo.name != "BGR":
        raise SourceError("GStreamer sample must negotiate BGR video")
    width, height = int(info.width), int(info.height)
    if not (0 < width <= 8192 and 0 < height <= 8192):
        raise SourceError(f"unsupported decoded size: {width} x {height} px")
    stride, offset = int(info.stride[0]), int(info.offset[0])
    meta = video.buffer_get_video_meta(buffer)
    if meta is not None:
        if (
            meta.n_planes != 1
            or meta.width != width
            or meta.height != height
            or meta.format != video.VideoFormat.BGR
        ):
            raise SourceError("inconsistent GStreamer BGR video metadata")
        stride, offset = int(meta.stride[0]), int(meta.offset[0])
    if stride < width * 3 or offset < 0:
        raise SourceError("invalid BGR row stride or offset")
    success, mapping = buffer.map(gst.MapFlags.READ)
    if not success:
        raise SourceError("cannot map GStreamer video buffer")
    try:
        required_bytes = offset + (height - 1) * stride + width * 3
        if required_bytes > len(mapping.data):
            raise SourceError("GStreamer BGR buffer is shorter than its dimensions")
        image = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=mapping.data,
            offset=offset,
            strides=(stride, 3, 1),
        )
        return VideoFrame(frame_id, None, received_at_ns, source, image)
    finally:
        buffer.unmap(mapping)


class GstBackend:
    """Own one pipeline. The calling thread polls samples and the bus."""

    def __init__(
        self, config: RtpConfig, *, clock: Callable[[], int] = time.monotonic_ns
    ) -> None:
        self.config = config
        self._clock = clock
        self._gst: Any = None
        self._video: Any = None
        self._pipeline: Any = None
        self._sink: Any = None
        self._bus: Any = None
        self._eos = False
        self._playing_logged = False
        self._size_warning_logged = False

    @property
    def eos(self) -> bool:
        return self._eos

    def start(self) -> None:
        logger.info("Starting live RTP receiver on UDP port %d", self.config.port)
        try:
            self._gst, self._video = _load_gst()
            for name in (
                "udpsrc",
                "rtph264depay",
                "h264parse",
                "queue",
                self.config.decoder,
                "videoconvert",
                "appsink",
            ):
                if self._gst.ElementFactory.find(name) is None:
                    raise SourceError(
                        f"required GStreamer element is unavailable: {name}"
                    )
            self._pipeline = self._gst.parse_launch(build_pipeline(self.config))
            self._sink = self._pipeline.get_by_name("frames")
            self._bus = self._pipeline.get_bus()
            if self._sink is None or self._bus is None:
                raise SourceError("GStreamer pipeline is missing its appsink or bus")
            logger.info("GStreamer pipeline created (decoder=%s)", self.config.decoder)
            result = self._pipeline.set_state(self._gst.State.PLAYING)
            if result == self._gst.StateChangeReturn.FAILURE:
                self._check_bus()
                raise SourceError("GStreamer pipeline failed to enter PLAYING")
            self._check_bus()
            logger.info("Listening for H.264 RTP on UDP port %d", self.config.port)
        except BaseException as exc:
            self.close()
            if isinstance(exc, (SourceError, KeyboardInterrupt, SystemExit)):
                raise
            raise SourceError(f"cannot start GStreamer receiver: {exc}") from exc

    def _check_bus(self) -> None:
        # Bound work per poll, even if a broken plugin floods the bus.
        for _ in range(64):
            message = self._bus.pop()
            if message is None:
                return
            kind = message.type
            if kind == self._gst.MessageType.ERROR:
                error, debug = message.parse_error()
                logger.error("GStreamer error: %s (%s)", error, debug)
                raise SourceError(f"GStreamer error: {error}")
            if kind == self._gst.MessageType.WARNING:
                warning, debug = message.parse_warning()
                logger.warning("GStreamer warning: %s (%s)", warning, debug)
            elif kind == self._gst.MessageType.EOS:
                self._eos = True
            elif (
                kind == self._gst.MessageType.STATE_CHANGED
                and message.src == self._pipeline
            ):
                _, new, _ = message.parse_state_changed()
                if new == self._gst.State.PLAYING and not self._playing_logged:
                    logger.info("GStreamer pipeline entered PLAYING")
                    self._playing_logged = True

    def pull(self, timeout_ns: int, frame_id: int) -> VideoFrame | None:
        try:
            self._check_bus()
            sample = self._sink.emit("try-pull-sample", timeout_ns)
            received_at_ns = self._clock()
            self._check_bus()
            if sample is None:
                # Bus EOS can precede the final queued sample. Drain that sample first.
                self._eos = self._eos or bool(self._sink.get_property("eos"))
                return None
            frame = sample_to_frame(
                sample,
                self._gst,
                self._video,
                frame_id=frame_id,
                received_at_ns=received_at_ns,
                source=f"rtp://0.0.0.0:{self.config.port}",
            )
            height, width = frame.image_bgr.shape[:2]
            if not self._size_warning_logged and (
                width != self.config.expected_width_px
                or height != self.config.expected_height_px
            ):
                logger.warning(
                    "Decoded size %d x %d px differs from expected %d x %d px",
                    width,
                    height,
                    self.config.expected_width_px,
                    self.config.expected_height_px,
                )
                self._size_warning_logged = True
            return frame
        except SourceError:
            raise
        except Exception as exc:
            raise SourceError(f"GStreamer frame read failed: {exc}") from exc

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        self._sink = self._bus = None
        if pipeline is not None:
            try:
                if (
                    pipeline.set_state(self._gst.State.NULL)
                    == self._gst.StateChangeReturn.FAILURE
                ):
                    logger.error(
                        "GStreamer pipeline failed to enter NULL during cleanup"
                    )
            except Exception:
                logger.exception("GStreamer cleanup failed")
