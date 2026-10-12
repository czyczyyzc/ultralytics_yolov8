from pathlib import Path
import json
import sys

import cv2
import numpy as np
import yaml

from scripts.anti_uav import prepare_stratified_rknn_calibration as calibration
from scripts.anti_uav.blend_rknn_calibration import choose_stratified


def test_blended_selection_preserves_requested_scale_mix():
    rows = [dict(category=category, calibration_image=f"/{category}_{index}.png")
            for category in calibration.CATEGORIES for index in range(100)]
    selected = choose_stratified(rows, 192, np.random.default_rng(7))
    expected = calibration.category_budgets(192)
    assert len(selected) == 192
    assert {category: sum(row["category"] == category for row in selected)
            for category in calibration.CATEGORIES} == expected


def test_stratified_calibration_writes_fixed_letterbox_and_audit(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    image_dir, label_dir = root / "images" / "train", root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    paths = []
    boxes = ("", "0 .5 .5 .01 .01\n", "0 .5 .5 .08 .08\n", "0 .5 .5 .4 .4\n", "")
    for index, label in enumerate(boxes):
        image = np.full((100, 200, 3), 30 + index * 20, np.uint8)
        path = image_dir / f"frame_{index}.png"
        assert cv2.imwrite(str(path), image)
        (label_dir / f"frame_{index}.txt").write_text(label)
        paths.append(str(path))
    listing = root / "train.txt"
    listing.write_text("\n".join(paths) + "\n")
    cached_labels = []
    for path, label in zip(paths, boxes):
        parsed = [line.split() for line in label.splitlines() if line.strip()]
        bboxes = np.asarray([[float(value) for value in fields[1:5]] for fields in parsed], dtype=np.float32)
        cached_labels.append(dict(
            im_file=path,
            shape=(100, 200),
            cls=np.zeros((len(parsed), 1), np.float32),
            bboxes=bboxes.reshape(-1, 4),
            normalized=True,
            bbox_format="xywh",
        ))
    with listing.with_suffix(".cache").open("wb") as handle:
        np.save(handle, dict(labels=cached_labels))
    data = root / "data.yaml"
    data.write_text(yaml.safe_dump(dict(train=str(listing))))
    output = tmp_path / "calibration"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_stratified_rknn_calibration.py",
            "--data", str(data),
            "--output-dir", str(output),
            "--samples", "5",
            "--height", "64",
            "--width", "96",
        ],
    )
    calibration.main()
    generated = [Path(line) for line in (output / "dataset.txt").read_text().splitlines()]
    assert len(generated) == 5
    assert all(cv2.imread(str(path)).shape == (64, 96, 3) for path in generated)
    audit = json.loads((output / "audit.json").read_text())
    assert audit["label_cache"].endswith("train.cache")


def test_calibration_rejects_evaluation_source(tmp_path, monkeypatch):
    listing = tmp_path / "train.txt"
    listing.write_text("/tmp/Video00009/images/frame.png\n")
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump(dict(train=str(listing))))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_stratified_rknn_calibration.py",
            "--data", str(data),
            "--output-dir", str(tmp_path / "out"),
            "--exclude-token", "Video00009",
        ],
    )
    try:
        calibration.main()
    except ValueError as error:
        assert "excluded evaluation source" in str(error)
    else:
        raise AssertionError("Expected evaluation-source leakage rejection")
