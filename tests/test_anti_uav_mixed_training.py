from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_detector_mixed_dataset(tmp_path):
    anti_root = tmp_path / "anti"
    extra_root = tmp_path / "extra"
    anti_root.mkdir()
    (anti_root / "train_rgb.txt").write_text("/tmp/a.jpg\n", encoding="utf-8")
    (anti_root / "val_rgb.txt").write_text("/tmp/b.jpg\n", encoding="utf-8")
    (extra_root / "images" / "train").mkdir(parents=True)
    (extra_root / "labels" / "train").mkdir(parents=True)
    (extra_root / "images" / "val").mkdir(parents=True)
    (extra_root / "labels" / "val").mkdir(parents=True)
    for split, stem in (("train", "x"), ("val", "y")):
        image_path = extra_root / "images" / split / f"{stem}.jpg"
        label_path = extra_root / "labels" / split / f"{stem}.txt"
        image_path.write_bytes(b"x")
        label_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    output_root = tmp_path / "merged"
    script = ROOT / "scripts" / "anti_uav" / "prepare_detector_mixed_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--antiuav-root", str(anti_root), "--extra-yolo-root", str(extra_root), "--output-root", str(output_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["train_items"] == 2
    assert summary["val_items"] == 2
    assert (output_root / "AntiUAV300PlusHanlueRGB.yaml").exists()


def test_convert_tracker_sequences_nanotrack(tmp_path):
    tracker_root = tmp_path / "tracker_sequences"
    image_root = tmp_path / "images"
    seq_dir = tracker_root / "train" / "train_000001"
    seq_dir.mkdir(parents=True)
    (tracker_root / "val").mkdir(parents=True)
    (image_root / "train").mkdir(parents=True)
    frame_path = image_root / "train" / "frame_000001.jpg"
    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(frame_path), frame)
    (seq_dir / "frames.txt").write_text("/missing/frame_000001.jpg\n", encoding="utf-8")
    (seq_dir / "groundtruth.txt").write_text("10,12,20,16\n", encoding="utf-8")
    (seq_dir / "meta.json").write_text("{}", encoding="utf-8")

    output_root = tmp_path / "nanotrack"
    script = ROOT / "scripts" / "anti_uav" / "convert_tracker_sequences_nanotrack.py"
    subprocess.run(
        [sys.executable, str(script), "--source-root", str(tracker_root), "--image-root", str(image_root), "--output-root", str(output_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    train_json = json.loads((output_root / "rgb" / "train.json").read_text(encoding="utf-8"))
    assert "train_000001" in train_json
    crop_files = list((output_root / "rgb" / "crop511" / "train_000001").glob("*.jpg"))
    assert crop_files


def test_merge_nanotrack_datasets(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root, name in ((root_a, "seq_a"), (root_b, "seq_b")):
        (root / "crop511" / name).mkdir(parents=True)
        (root / "crop511" / name / "000000.00.x.jpg").write_bytes(b"x")
        (root / "train.json").write_text(json.dumps({name: {"00": {"000000": [0, 0, 10, 10]}}}), encoding="utf-8")
        (root / "val.json").write_text("{}", encoding="utf-8")
        (root / "split_manifest.json").write_text(json.dumps({"train": [{"name": name}], "val": []}), encoding="utf-8")
    output_root = tmp_path / "merged"
    script = ROOT / "scripts" / "anti_uav" / "merge_nanotrack_datasets.py"
    subprocess.run(
        [sys.executable, str(script), "--input-roots", str(root_a), str(root_b), "--output-root", str(output_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    merged = json.loads((output_root / "train.json").read_text(encoding="utf-8"))
    assert {"seq_a", "seq_b"} <= set(merged)
    assert (output_root / "crop511" / "seq_a").is_symlink()


def test_convert_anti_uav300_supports_train_layout(tmp_path):
    source_root = tmp_path / "train"
    sequence_dir = source_root / "seq_0001"
    sequence_dir.mkdir(parents=True)
    frame = np.full((32, 32, 3), 255, dtype=np.uint8)
    writer = cv2.VideoWriter(str(sequence_dir / "visible.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (32, 32))
    writer.write(frame)
    writer.release()
    (sequence_dir / "visible.json").write_text(json.dumps({"gt_rect": [[4, 6, 12, 10]]}), encoding="utf-8")
    output_root = tmp_path / "yolo"
    script = ROOT / "scripts" / "anti_uav" / "convert_anti_uav300.py"
    subprocess.run(
        [sys.executable, str(script), "--source-root", str(source_root), "--output-root", str(output_root), "--modalities", "rgb"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output_root / "train_rgb.txt").exists() or (output_root / "val_rgb.txt").exists()
