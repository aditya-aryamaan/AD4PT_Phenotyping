import importlib.util
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import yaml
from test_detection import config as config

from ad4pt.vision.detector import Detection

spec = importlib.util.spec_from_file_location(
    "fixed_eval", Path(__file__).parents[1] / "scripts/evaluate_fixed_threshold.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_and_wrong_class_boxes_are_false_positives(
    config, tmp_path, monkeypatch
):
    root = tmp_path / "data"
    (root / "images/val").mkdir(parents=True)
    (root / "labels/val").mkdir(parents=True)
    cv2.imwrite(str(root / "images/val/one.png"), np.zeros((32, 32, 3), dtype=np.uint8))
    (root / "labels/val/one.txt").write_text(
        "0 .1875 .1875 .25 .25\n1 .75 .1875 .25 .25\n"
    )
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"path": "data"}))
    detector = Mock()
    detector.detect.return_value = (
        Detection(0, "ripe", 0.9, (2, 2, 10, 10)),
        Detection(0, "ripe", 0.8, (2.5, 2.5, 10.5, 10.5)),
        Detection(0, "ripe", 0.7, (20, 2, 28, 10)),
    )
    monkeypatch.setattr(module, "load_detector_config", lambda path: config)
    monkeypatch.setattr(module, "UltralyticsDetector", lambda config: detector)
    result = module.evaluate(Path("unused"), path, "val", 0.25)
    assert result["per_class"]["ripe"]["TP"] == 1
    assert result["per_class"]["ripe"]["FP"] == 2
    assert result["per_class"]["ripe"]["FN"] == 0
    assert result["per_class"]["unripe"]["FN"] == 1
    assert result["per_class"]["unripe"]["TP"] == 0
    assert result["per_class"]["ripe"]["precision"] == 1 / 3
    assert result["per_class"]["unripe"]["recall"] == 0
    detector.close.assert_called_once()
