from pathlib import Path

import numpy as np

from scripts.anti_uav.build_hard_positive_rehearsal_mix import build_rehearsal_mix
from scripts.anti_uav.mine_yolo_hard_positives import classify_difficulty


def test_classify_missed_localized_weak_and_easy() -> None:
    gt = np.array([[0.1, 0.1, 0.2, 0.2]], dtype=np.float32)
    assert classify_difficulty(gt, np.empty((0, 4)), np.empty((0,)), 0.5, 0.15)["category"] == "missed"

    far = np.array([[0.7, 0.7, 0.8, 0.8]], dtype=np.float32)
    assert classify_difficulty(gt, far, np.array([0.9]), 0.5, 0.15)["category"] == "localization"

    assert classify_difficulty(gt, gt.copy(), np.array([0.10]), 0.5, 0.15)["category"] == "weak"
    assert classify_difficulty(gt, gt.copy(), np.array([0.80]), 0.5, 0.15)["category"] == "easy"


def write_example(root: Path, name: str, positive: bool) -> Path:
    image = root / "images" / "gray" / "Video00001" / f"{name}.jpg"
    label = root / "labels" / "gray" / "Video00001" / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"test")
    label.write_text("0 0.5 0.5 0.1 0.1\n" if positive else "")
    return image


def test_rehearsal_keeps_negatives_and_unique_positive_coverage(tmp_path: Path) -> None:
    first = write_example(tmp_path, "000001", True)
    second = write_example(tmp_path, "000002", True)
    negative = write_example(tmp_path, "000003", False)
    source = [first, second, negative, first, second, first]
    records = [
        {
            "image": str(first),
            "group": "Video00001",
            "category": "missed",
            "extra_repeats": 3,
        }
    ]

    output, manifest = build_rehearsal_mix(source, records, hard_positive_fraction=0.4, seed=3)

    assert len(output) == len(source)
    assert [path for path in output if path == negative] == [negative]
    assert {first, second}.issubset(output)
    assert manifest["negative_samples"] == 1
    assert manifest["negative_fraction"] == 1 / 6
    assert manifest["negative_samples_unchanged"] is True
    assert manifest["unique_positive_coverage_unchanged"] is True
