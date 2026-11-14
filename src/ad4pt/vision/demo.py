"""Live ripe/unripe preview with one count summary at shutdown."""

import argparse
import json
import sys
from contextlib import closing, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ad4pt.counting import SceneDetectionFilter, StaticSceneCounter, StrawberryCounter
from ad4pt.json_io import save_json, save_run_json
from ad4pt.latest_frame import LatestFrameSource
from ad4pt.live import RtpFrameSource, StreamTimeout
from ad4pt.live_config import load_rtp_config
from ad4pt.sources import SourceError
from ad4pt.types import VideoFrame
from ad4pt.vision.config import load_detector_config
from ad4pt.vision.detector import Detection, DetectorError, UltralyticsDetector
from ad4pt.vision.tracking import load_tracking_config

TITLE = "AD4PT - ripe / unripe (Q or Esc to stop)"


class Preview:
    def __init__(self, capture_dir: Path | None = None, track_ids: bool = True) -> None:
        self.capture_dir = capture_dir
        self.track_ids = track_ids
        self._last_frame: VideoFrame | None = None
        print("Opening preview window...", file=sys.stderr, flush=True)
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.imshow(TITLE, np.zeros((480, 640, 3), dtype=np.uint8))
        cv2.waitKey(1)

    def open(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if (
            key == ord("s")
            and self.capture_dir is not None
            and self._last_frame is not None
        ):
            try:
                self.capture_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f_%Z")
                path = self.capture_dir / f"strawberry-frame_{stamp}.png"
                if not cv2.imwrite(str(path), self._last_frame.image_bgr):
                    raise OSError("image writer failed")
                print(f"Saved raw camera frame: {path}", file=sys.stderr, flush=True)
            except (OSError, cv2.error) as exc:
                print(f"Cannot save camera frame: {exc}", file=sys.stderr)
        return (
            key not in (27, ord("q"))
            and cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) >= 1
        )

    def show(self, frame: VideoFrame, detections: tuple[Detection, ...]) -> None:
        if self.capture_dir is not None:
            self._last_frame = frame
        image = frame.image_bgr.copy()
        for detection in detections:
            identity = (
                f"#{detection.track_id}"
                if detection.track_id is not None
                else "pending"
            )
            x1, y1, x2, y2 = map(round, detection.bbox_xyxy_px)
            color = (
                (0, 200, 0)
                if detection.label in ("strawberry-ripe", "ripe")
                else (0, 200, 255)
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{identity + ' ' if self.track_ids else ''}"
                f"{detection.label} {detection.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.imshow(TITLE, image)

    def close(self) -> None:
        self._last_frame = None
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def add_demo_command(commands: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = commands.add_parser(
        "demo", help="live strawberry preview; save ripe/unripe counts on stop"
    )
    parser.add_argument("--config", type=Path, default=Path("config/live_apfpv.yaml"))
    parser.add_argument(
        "--model-config", type=Path, default=Path("config/strawberry_yolo11s.json")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, help="summary JSON path")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--unripe-confidence",
        type=float,
        help="optional stricter confidence for unripe boxes; may miss green fruit",
    )
    parser.add_argument(
        "--counting-mode",
        choices=("tracks", "scene"),
        default="tracks",
        help=(
            "tracks: cumulative IDs for moving views; "
            "scene: modal counts for a fixed view"
        ),
    )
    parser.add_argument(
        "--scene-confirmation-frames",
        type=int,
        default=5,
        help=(
            "consecutive matching frames before showing/counting "
            "a box in scene mode (1-60)"
        ),
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        help="enable S in preview to save a raw camera frame for diagnosis",
    )
    parser.add_argument(
        "--tracker-config", type=Path, default=Path("config/strawberry_botsort.yaml")
    )
    return parser


def demo_command(args: argparse.Namespace) -> int:
    print(f"AD4PT demo starting ({sys.executable})", file=sys.stderr, flush=True)
    try:
        live = load_rtp_config(args.config)
        if args.max_frames is not None:
            live = replace(live, max_frames=args.max_frames)
        config = replace(load_detector_config(args.model_config), device=args.device)
        if args.unripe_confidence is not None:
            config = replace(config, unripe_confidence_threshold=args.unripe_confidence)
        tracking = load_tracking_config(args.tracker_config)
        scene_filter = SceneDetectionFilter(args.scene_confirmation_frames)
        if set(config.classes) not in (
            {"strawberry-ripe", "strawberry-unripe"},
            {"ripe", "unripe"},
        ):
            raise ValueError(
                "model must have exactly ripe and unripe strawberry classes"
            )
        if args.output is not None and args.output.suffix.lower() != ".json":
            raise ValueError("--output must be a .json file")
    except (ValueError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    started = datetime.now(UTC)
    output = args.output
    directory = output.parent if output is not None else args.run_root
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"cannot create output directory: {exc}", file=sys.stderr)
        return 5
    scene_mode = args.counting_mode == "scene"
    counter: StrawberryCounter | StaticSceneCounter = (
        StaticSceneCounter if scene_mode else StrawberryCounter
    )(tracking.confirmation_frames, config.confidence_threshold)
    reason = "window_closed"
    code = 0
    print(f"Loading detector on {config.device}...", file=sys.stderr, flush=True)
    if config.unripe_confidence_threshold is not None:
        print(
            f"Unripe confidence threshold: {config.unripe_confidence_threshold:g}; "
            "stricter filtering can also hide genuine green fruit.",
            file=sys.stderr,
            flush=True,
        )
    if scene_mode:
        print(
            "Fixed-scene counts: most frequent ripe/unripe detection pair; "
            f"each box requires {scene_filter.confirmation_frames} consecutive frames; "
            f"requires {tracking.confirmation_frames} matching frames. "
            "Keep the entire scene visible; do not use while flying across crops.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"BoT-SORT: confirm after {tracking.confirmation_frames} "
            "tracked observations; "
            f"retain lost tracks for {tracking.track_buffer} processed frames.",
            file=sys.stderr,
            flush=True,
        )
    if config.ignore_top_fraction:
        print(
            f"Excluding top {config.ignore_top_fraction:.0%} of footage (OSD).",
            file=sys.stderr,
            flush=True,
        )
    if config.max_box_area_fraction < 1:
        print(
            f"Rejecting boxes covering more than {config.max_box_area_fraction:.0%} "
            "of the image below the OSD.",
            file=sys.stderr,
            flush=True,
        )
    try:
        with (
            redirect_stdout(sys.stderr),
            closing(UltralyticsDetector(config)) as detector,
            closing(
                Preview(capture_dir=args.capture_dir, track_ids=not scene_mode)
            ) as preview,
            LatestFrameSource(
                lambda stop: RtpFrameSource(
                    replace(live, max_frames=None), stop_event=stop
                ),
                max_frames=live.max_frames,
            ) as source,
        ):
            print(
                f"Waiting for APFPV video on UDP {live.port} "
                f"(timeout {live.timeout_s:g} seconds)...",
                file=sys.stderr,
                flush=True,
            )
            while preview.open():
                try:
                    frame = source.poll(0.05)
                except TimeoutError:
                    continue
                except StopIteration:
                    reason = (
                        "max_frames"
                        if counter.frames_processed == live.max_frames
                        else "stream_ended"
                    )
                    break
                detections = (
                    detector.detect(frame)
                    if scene_mode
                    else detector.track(frame, tracking.path, tracking.track_low_thresh)
                )
                if scene_mode:
                    detections = scene_filter.update(detections)
                counter.update(detections)
                preview.show(frame, detections)
    except StreamTimeout as exc:
        reason = "stream_timeout" if counter.frames_processed else "no_stream"
        code = 0 if counter.frames_processed else 4
        print(f"Stream stopped: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        reason, code = "interrupted", 130
    except (SourceError, DetectorError, OSError, cv2.error) as exc:
        reason, code = "error", 3
        print(f"demo stopped: {exc}", file=sys.stderr)
    finally:
        counts = counter.counts
        summary = {
            **counts,
            "total": sum(counts.values()),
            "frames_processed": counter.frames_processed,
            "started_at_utc": started.isoformat().replace("+00:00", "Z"),
            "ended_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "stop_reason": reason,
            "counting_method": (
                "static_scene_confirmed_modal_detections"
                if scene_mode
                else "botsort_raw_detections"
            ),
            "count_confidence_threshold": config.confidence_threshold,
            "unripe_confidence_threshold": config.threshold_for("strawberry-unripe"),
            "confirmation_frames": tracking.confirmation_frames,
            "track_buffer_frames": None if scene_mode else tracking.track_buffer,
            "max_box_area_fraction": config.max_box_area_fraction,
        }
        if isinstance(counter, StaticSceneCounter):
            summary["scene_mode_support_frames"] = counter.support_frames
            summary["scene_confirmation_frames"] = scene_filter.confirmation_frames
            summary["scene_match_iou"] = scene_filter.min_iou
        try:
            if output is None:
                output = save_run_json(directory, started, summary)
            else:
                save_json(output, summary)
            print(json.dumps({"counts_json": str(output), **summary}, indent=2))
        except OSError as exc:
            print(
                f"cannot save counts to {output or directory}: {exc}", file=sys.stderr
            )
            code = 5
    return code
