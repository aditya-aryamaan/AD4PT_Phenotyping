"""Validate the gates used before expensive training; no training is run."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "finetune", Path(__file__).parents[1] / "scripts/finetune_strawberries.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    for index, split in enumerate(("train", "val")):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        name = f"{split}-scene__one"
        cv2.imwrite(
            str(root / "images" / split / f"{name}.png"),
            np.full((32, 32, 3), index * 30, dtype=np.uint8),
        )
        (root / "labels" / split / f"{name}.txt").write_text(
            "0 0.25 0.5 0.2 0.2\n1 0.75 0.5 0.2 0.2\n"
        )
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": "data",
                "names": module.CLASSES,
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
            }
        )
    )
    return root, config


def test_training_preflight_resolves_relative_paths_and_accepts_reviewed_negatives(
    dataset,
):
    root, config = dataset
    cv2.imwrite(
        str(root / "images/train/train-scene__negative.png"),
        np.full((32, 32, 3), 77, dtype=np.uint8),
    )
    (root / "labels/train/train-scene__negative.txt").write_text("")
    data, summary = module.check_dataset(config)
    assert data["path"] == str(root.resolve())
    assert summary["train"]["background_images"] == 1
    assert summary["train"]["ripe_boxes"] == 1
    assert summary["val"]["unripe_boxes"] == 1
    assert summary["test"]["images"] == 0


def test_training_preflight_rejects_missing_label_and_out_of_bounds_box(dataset):
    root, config = dataset
    label = root / "labels/train/train-scene__one.txt"
    label.unlink()
    with pytest.raises(ValueError, match="Missing reviewed label"):
        module.check_dataset(config)
    label.write_text("0 0.95 0.5 0.5 0.2\n")
    with pytest.raises(ValueError, match="Invalid normalized box"):
        module.check_dataset(config)


def test_training_preflight_rejects_scene_leakage(dataset):
    root, config = dataset
    (root / "images/val/val-scene__one.png").rename(
        root / "images/val/train-scene__two.png"
    )
    with pytest.raises(ValueError, match="multiple splits"):
        module.check_dataset(config)


def test_training_preflight_rejects_exact_duplicate_images_across_splits(dataset):
    root, config = dataset
    (root / "images/val/val-scene__one.png").write_bytes(
        (root / "images/train/train-scene__one.png").read_bytes()
    )
    with pytest.raises(ValueError, match="Duplicate image"):
        module.check_dataset(config)
