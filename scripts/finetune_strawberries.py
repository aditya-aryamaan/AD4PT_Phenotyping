"""Check a reviewed dataset; optionally fine-tune the existing local checkpoint."""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from ad4pt.vision.config import load_detector_config

CLASSES = {0: "strawberry-ripe", 1: "strawberry-unripe"}


def check_dataset(config_path):
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict) or data.get("names") != CLASSES:
        raise ValueError("dataset classes must preserve 0=ripe, 1=unripe")
    root = (config_path.resolve().parent / data["path"]).resolve()
    summary, hashes, groups = {}, {}, {}
    for split in ("train", "val", "test"):
        if data.get(split) != f"images/{split}":
            raise ValueError(f"Use {split}: images/{split} in the dataset YAML")
        folder = root / "images" / split
        images = sorted(
            p
            for p in folder.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not images and split != "test":
            raise ValueError(
                f"No {split} images in {folder}; reviewed labels needed first"
            )
        counts, backgrounds = [0, 0], 0
        for image in images:
            if cv2.imread(str(image)) is None:
                raise ValueError(f"Unreadable image: {image}")
            scene, separator, _ = image.stem.partition("__")
            if not separator or not scene:
                raise ValueError(f"Use sceneID__frameName filenames: {image}")
            if scene in groups and groups[scene] != split:
                raise ValueError(f"Scene {scene} appears in multiple splits")
            groups[scene] = split
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"Duplicate image across splits: {image}")
            hashes[digest] = split
            relative = image.relative_to(folder).with_suffix(".txt")
            label = root / "labels" / split / relative
            if not label.is_file():
                raise ValueError(f"Missing reviewed label: {label}")
            lines = label.read_text().splitlines()
            if not lines:
                backgrounds += 1
            for line in lines:
                parts = line.split()
                if len(parts) != 5 or parts[0] not in ("0", "1"):
                    raise ValueError(f"Invalid YOLO row in {label}: {line}")
                x, y, w, h = map(float, parts[1:])
                if not (
                    all(math.isfinite(v) for v in (x, y, w, h))
                    and 0 < w <= 1
                    and 0 < h <= 1
                    and w / 2 <= x <= 1 - w / 2
                    and h / 2 <= y <= 1 - h / 2
                ):
                    raise ValueError(f"Invalid normalized box in {label}: {line}")
                counts[int(parts[0])] += 1
        if images and split != "test" and min(counts) == 0:
            raise ValueError(f"{split} must contain labelled ripe AND unripe examples")
        summary[split] = {
            "images": len(images),
            "ripe_boxes": counts[0],
            "unripe_boxes": counts[1],
            "background_images": backgrounds,
        }
    data = {
        "path": str(root),
        "names": CLASSES,
        **{split: f"images/{split}" for split in ("train", "val", "test")},
    }
    return data, summary


def evaluate(model, data_path, device, output, name):
    metrics = model.val(
        data=str(data_path),
        split="val",
        imgsz=640,
        device=device,
        batch=4,
        workers=0,
        conf=0.001,
        iou=0.7,
        project=str(output),
        name=name,
        plots=True,
    )
    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        per_class[CLASSES[int(class_id)]] = dict(
            zip(
                ("precision", "recall", "AP50", "AP50_95"),
                map(float, metrics.box.class_result(index)),
                strict=True,
            )
        )
    return {
        "overall": {k: float(v) for k, v in metrics.results_dict.items()},
        "per_class": per_class,
        "split": "val",
        "note": "Development validation, not an independent final test.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("config/strawberry_training.yaml")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("config/strawberry_yolo11s.json")
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="actually start training; default only checks data",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--freeze-layers", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--candidate",
        type=Path,
        help="compare a trusted, already trained local checkpoint without retraining",
    )
    args = parser.parse_args()
    try:
        data, summary = check_dataset(args.data)
        if args.epochs < 1 or args.batch < 1:
            raise ValueError("epochs and batch must be positive")
        if args.freeze_layers < 0 or not 0 < args.learning_rate <= 1:
            raise ValueError(
                "freeze-layers must be nonnegative; learning-rate in (0, 1]"
            )
        print(json.dumps(summary, indent=2))
        if not args.train and args.candidate is None:
            print("Dataset checks passed. Labels still require human visual review.")
            return
        config = load_detector_config(args.model_config)
        if (
            hashlib.sha256(config.model_path.read_bytes()).hexdigest()
            != config.artifact_sha256
        ):
            raise ValueError("Checkpoint SHA256 mismatch")
        import ultralytics

        if ultralytics.__version__ != "8.4.14":
            raise ValueError("Use the project's pinned Ultralytics 8.4.14 environment")
        from ultralytics import YOLO

        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S-%f_%Z")
        output = (Path("runs/training") / stamp).resolve()
        output.mkdir(parents=True, exist_ok=False)
        resolved = output / "data.yaml"
        resolved.write_text(yaml.safe_dump(data))
        (output / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
        fingerprints = []
        for split in ("train", "val", "test"):
            folder = Path(data["path"]) / "images" / split
            for image in sorted(folder.rglob("*")):
                if image.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label = (
                    Path(data["path"])
                    / "labels"
                    / split
                    / image.relative_to(folder).with_suffix(".txt")
                )
                fingerprints.append(
                    {
                        "image": str(image),
                        "split": split,
                        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                    }
                )
        (output / "dataset-fingerprints.json").write_text(
            json.dumps(fingerprints, indent=2)
        )
        model = YOLO(str(config.model_path), task="detect")
        if model.names != CLASSES:
            raise ValueError("Checkpoint classes differ from dataset")
        baseline = evaluate(
            model, resolved.resolve(), args.device, output, "baseline-val"
        )
        if args.candidate is not None:
            best = args.candidate.resolve(strict=True)
            if best.suffix != ".pt":
                raise ValueError("Candidate must be a trusted local .pt checkpoint")
        else:
            # Fresh object: validation does not alter fine-tuning initialization.
            model = YOLO(str(config.model_path), task="detect")
            model.train(
                data=str(resolved.resolve()),
                device=args.device,
                epochs=args.epochs,
                imgsz=640,
                batch=args.batch,
                workers=0,
                optimizer="AdamW",
                lr0=args.learning_rate,
                freeze=args.freeze_layers,
                patience=10,
                seed=42,
                amp=False,
                cache=False,
                close_mosaic=5,
                project=str(output),
                name="finetune",
            )
            best = Path(model.trainer.best)
        candidate = evaluate(
            YOLO(str(best)), resolved.resolve(), args.device, output, "candidate-val"
        )
        (output / "validation-comparison.json").write_text(
            json.dumps({"baseline": baseline, "candidate": candidate}, indent=2)
        )
        deployment = json.loads(args.model_config.read_text())
        deployment.update(
            model_path=str(best.resolve()),
            artifact_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
            model_id=f"ad4pt-finetuned-{stamp}",
            unripe_confidence_threshold=None,
        )
        (output / "candidate-detector.json").write_text(
            json.dumps(deployment, indent=2)
        )
        print(
            f"Candidate saved to {output}. "
            "Active model not replaced; compare before deployment."
        )
    except (ValueError, KeyError, OSError, yaml.YAMLError) as exc:
        parser.exit(2, f"Training setup error: {exc}\n")


if __name__ == "__main__":
    main()
