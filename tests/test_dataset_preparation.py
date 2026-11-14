import importlib.util
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "prepare", Path(__file__).parents[1] / "scripts/prepare_strawberry_dataset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_six_stage_mapping_preserves_every_annotation():
    text = "\n".join(f"{i} .5 .5 .2 .2" for i in range(6))
    rows, _ = module.convert_rows(text)
    assert [row[0] for row in rows] == [1, 1, 1, 0, 1, 1]
    assert len(rows) == 6


def test_tiny_boundary_error_is_clipped_but_large_error_rejected():
    rows, count = module.convert_rows("1 .01 .02 .02 .04001")
    assert count == 1
    assert rows[0][2] == rows[0][4] / 2
    with pytest.raises(ValueError, match="substantially outside"):
        module.convert_rows("1 .01 .02 .02 .2")


def test_shared_source_kept_out_of_training_without_changing_original(tmp_path):
    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    (source / "data.yaml").write_text(yaml.safe_dump({"names": module.STAGES}))
    for split in ("train", "valid", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
    for split in ("train", "test"):
        name = f"DSC_01.rf.{split}"
        (source / split / "images" / f"{name}.jpg").write_bytes(split.encode())
        (source / split / "labels" / f"{name}.txt").write_text("3 .5 .5 .2 .2")
    manifest = module.prepare(source, target)
    assert manifest["counts"]["train"]["images"] == 0
    assert manifest["counts"]["test"]["images"] == 1
    assert len(manifest["excluded"]) == 1
    assert (source / "train/labels/DSC_01.rf.train.txt").read_text() == "3 .5 .5 .2 .2"
    assert module.prepare(source, target)["counts"] == manifest["counts"]


def test_reviewed_alignment_transform_keeps_class_and_rotates_geometry():
    path = Path(__file__).parents[1] / "scripts/apply_strawberry_label_corrections.py"
    spec = importlib.util.spec_from_file_location("alignment", path)
    alignment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alignment)
    row = [1, 0.2, 0.3, 0.1, 0.15]
    rotated = alignment.transform_row(row, "ccw90")
    assert rotated == pytest.approx([1, 0.3, 0.8, 0.15, 0.1])
    for _ in range(3):
        rotated = alignment.transform_row(rotated, "ccw90")
    assert rotated == pytest.approx(row)
    assert alignment.transform_row(row, "180") == pytest.approx(
        [1, 0.8, 0.7, 0.1, 0.15]
    )
