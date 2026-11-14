"""Convert the supplied six-stage export to two classes, with an audit manifest."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

STAGES = ["Early-Turning", "Green", "Late-Turning", "Red", "Turning", "White"]


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(
                f"Refusing to overwrite different existing content: {path}"
            )
    else:
        path.write_text(text)


def source_id(path):
    return path.stem.split(".rf.")[0]


def convert_rows(text):
    result, corrections = [], 0
    for line in text.splitlines():
        values = line.split()
        if len(values) != 5 or values[0] not in tuple(map(str, range(6))):
            raise ValueError(f"Invalid source label row: {line}")
        cls = int(values[0])
        x, y, w, h = map(float, values[1:])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid source geometry: {line}")
        original = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if min(original[:2]) < -0.001 or max(original[2:]) > 1.001:
            raise ValueError(f"Box substantially outside image: {line}")
        x1, y1, x2, y2 = (
            max(0.0, original[0]),
            max(0.0, original[1]),
            min(1.0, original[2]),
            min(1.0, original[3]),
        )
        corrections += int(original != (x1, y1, x2, y2))
        result.append(
            (
                0 if STAGES[cls] == "Red" else 1,
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            )
        )
    return result, corrections


def label_text(rows):
    return "".join(
        f"{row[0]} " + " ".join(repr(float(v)) for v in row[1:]) + "\n" for row in rows
    )


def prepare(source, output, camera_labels=None):
    metadata = yaml.safe_load((source / "data.yaml").read_text())
    if metadata.get("names") != STAGES:
        raise ValueError(
            "Unexpected maturity classes; verify mapping before conversion"
        )
    # Keep provider validation/test memberships; discard matching TRAIN source copies.
    records, memberships = [], defaultdict(set)
    for src_split, split in [("test", "test"), ("valid", "val"), ("train", "train")]:
        for image in sorted((source / src_split / "images").iterdir()):
            if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            records.append((image, src_split, split))
            memberships[source_id(image)].add(split)
    preference = {"train": 0, "val": 1, "test": 2}
    seen_hashes, exported, skipped = {}, [], []
    counts, corrections = {s: Counter() for s in preference}, 0
    for image, src_split, split in records:
        identity = source_id(image)
        target_split = max(memberships[identity], key=preference.get)
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if split != target_split or (
            digest in seen_hashes and seen_hashes[digest] != split
        ):
            skipped.append(
                {
                    "source": str(image),
                    "reason": "source/hash also in higher-priority held-out split",
                }
            )
            continue
        seen_hashes[digest] = split
        label = source / src_split / "labels" / image.with_suffix(".txt").name
        rows, fixed = convert_rows(label.read_text())
        corrections += fixed
        name = f"public-{identity}__{image.name}"
        target = output / "images" / split / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(image, target)
            except OSError:
                shutil.copy2(image, target)
        elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing image differs: {target}")
        write_text(
            output / "labels" / split / Path(name).with_suffix(".txt"), label_text(rows)
        )
        counts[split]["images"] += 1
        counts[split]["ripe_boxes"] += sum(r[0] == 0 for r in rows)
        counts[split]["unripe_boxes"] += sum(r[0] == 1 for r in rows)
        exported.append(
            {
                "source": str(image),
                "target": str(target),
                "split": split,
                "source_sha256": digest,
                "clipped_boxes": fixed,
                "labels": "supplied source annotations; classes remapped",
            }
        )
    if camera_labels:
        for record in json.loads(camera_labels.read_text()):
            if record.get("review_status") != "assistant_visual_review":
                raise ValueError(
                    "Camera boxes must be visually reviewed, not raw predictions"
                )
            image = Path(record["source"])
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f"Cannot read {image}")
            pixels = pixels[record["crop_top"] :]
            h, w = pixels.shape[:2]
            boxes = record["boxes"]
            if len(boxes) != 4:
                raise ValueError(
                    "This reviewed replica scene must have four ripe boxes"
                )
            rows = []
            for x1, y1, x2, y2 in boxes:
                if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                    raise ValueError(f"Invalid camera box: {image}")
                rows.append(
                    (
                        0,
                        (x1 + x2) / (2 * w),
                        (y1 + y2) / (2 * h),
                        (x2 - x1) / w,
                        (y2 - y1) / h,
                    )
                )
            name = "apfpv-replica-20261015__" + image.name
            target = output / "images/train" / name
            if not target.exists() and not cv2.imwrite(str(target), pixels):
                raise OSError(f"Cannot save {target}")
            write_text(
                output / "labels/train" / Path(name).with_suffix(".txt"),
                label_text(rows),
            )
            counts["train"]["images"] += 1
            counts["train"]["ripe_boxes"] += 4
            exported.append(
                {
                    "source": str(image),
                    "target": str(target),
                    "split": "train",
                    "source_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "labels": (
                        "ripe box proposals visually checked "
                        "against four known replicas"
                    ),
                    "reviewer": "assistant",
                    "crop_top_px": record["crop_top"],
                }
            )
    manifest = {
        "mapping": {name: ("ripe" if name == "Red" else "unripe") for name in STAGES},
        "source": str(source),
        "source_export_metadata": metadata.get("roboflow"),
        "counts": counts,
        "clipped_boxes": corrections,
        "excluded": skipped,
        "records": exported,
        "limitations": [
            "Provider splits retained except known source/hash leakage removals.",
            "Source-photo IDs do not establish independent acquisition scenes.",
            "All replica frames are training-only, not camera validation.",
            (
                "Camera labels are AI-assisted visual annotations; "
                "review before final use."
            ),
        ],
    }
    write_text(
        output / "preparation-manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("counts", "clipped_boxes", "excluded")}, indent=2
        )
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/strawberry-adaptation")
    )
    parser.add_argument("--camera-labels", type=Path)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.camera_labels)
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        parser.exit(2, f"Dataset preparation error: {exc}\n")
