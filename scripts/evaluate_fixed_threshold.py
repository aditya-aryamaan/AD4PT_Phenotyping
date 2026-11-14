"""Compare detector weights at fixed confidence, without tracking or scene filters."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import yaml

from ad4pt.types import VideoFrame
from ad4pt.vision.config import load_detector_config
from ad4pt.vision.detector import UltralyticsDetector


def iou(a, b):
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    intersection = max(0.0, min(x2, u2) - max(x1, u1)) * max(
        0.0, min(y2, v2) - max(y1, v1)
    )
    return intersection / ((x2 - x1) * (y2 - y1) + (u2 - u1) * (v2 - v1) - intersection)


def evaluate(config_path, data_path, split, confidence):
    data = yaml.safe_load(data_path.read_text())
    root = (data_path.resolve().parent / data["path"]).resolve()
    config = replace(
        load_detector_config(config_path),
        device="cpu",
        confidence_threshold=confidence,
        unripe_confidence_threshold=None,
        ignore_top_fraction=0,
        max_box_area_fraction=1,
        image_size_px=640,
        iou_threshold=0.7,
    )
    counts = [{k: 0 for k in ("TP", "FP", "FN")} for _ in range(2)]
    images = []
    detector = UltralyticsDetector(config)
    try:
        for index, path in enumerate(sorted((root / "images" / split).glob("*"))):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unreadable image: {path}")
            h, w = image.shape[:2]
            label = root / "labels" / split / path.with_suffix(".txt").name
            ground_truth = []
            for line in label.read_text().splitlines():
                cls, x, y, bw, bh = map(float, line.split())
                ground_truth.append(
                    (
                        int(cls),
                        (
                            (x - bw / 2) * w,
                            (y - bh / 2) * h,
                            (x + bw / 2) * w,
                            (y + bh / 2) * h,
                        ),
                    )
                )
            detections = detector.detect(VideoFrame(index, None, 0, str(path), image))
            matched = set()
            for detection in sorted(
                detections, key=lambda d: d.confidence, reverse=True
            ):
                options = [
                    (iou(detection.bbox_xyxy_px, box), j)
                    for j, (cls, box) in enumerate(ground_truth)
                    if j not in matched and cls == detection.class_id
                ]
                overlap, j = max(options, default=(0.0, -1))
                if overlap >= 0.5:
                    counts[detection.class_id]["TP"] += 1
                    matched.add(j)
                else:
                    counts[detection.class_id]["FP"] += 1
            for j, (cls, _) in enumerate(ground_truth):
                if j not in matched:
                    counts[cls]["FN"] += 1
            images.append(
                {
                    "image": path.name,
                    "reference": [
                        sum(cls == k for cls, _ in ground_truth) for k in (0, 1)
                    ],
                    "detected": [
                        sum(d.class_id == k for d in detections) for k in (0, 1)
                    ],
                }
            )
    finally:
        detector.close()
    metrics = {}
    for cls, label in enumerate(config.classes):
        row = counts[cls]
        tp, fp, fn = row["TP"], row["FP"], row["FN"]
        metrics[label] = {
            **row,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    return {
        "model_id": config.model_id,
        "artifact_sha256": config.artifact_sha256,
        "confidence_threshold": confidence,
        "matching_iou": 0.5,
        "nms_iou": 0.7,
        "image_size_px": 640,
        "device": "cpu",
        "split": split,
        "images": len(images),
        "note": (
            "Raw detection evaluation. No crop, area gate, tracking, "
            "temporal confirmation or class-specific confidence change."
        ),
        "per_class": metrics,
        "per_image": images,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=Path("config/strawberry_yolo11s.json")
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=Path("config/strawberry_training.yaml")
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation/fixed-threshold.json"),
    )
    args = parser.parse_args()
    results = {
        key: evaluate(path, args.data, args.split, args.confidence)
        for key, path in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {key: value["per_class"] for key, value in results.items()}, indent=2
        )
    )
