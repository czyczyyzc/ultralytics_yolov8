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
    flat_root = tmp_path / "flat_extra"
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
    (flat_root / "images").mkdir(parents=True)
    (flat_root / "labels").mkdir(parents=True)
    for stem in ("train_000001", "val_000002"):
        (flat_root / "images" / f"{stem}.jpg").write_bytes(b"x")
        (flat_root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    output_root = tmp_path / "merged"
    script = ROOT / "scripts" / "anti_uav" / "prepare_detector_mixed_dataset.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--antiuav-root",
            str(anti_root),
            "--extra-yolo-root",
            str(extra_root),
            "--extra-yolo-root",
            str(flat_root),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["train_items"] == 3
    assert summary["val_items"] == 3
    assert len(summary["extra_yolo_roots"]) == 2
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


def test_convert_tracker_sequences_nanotrack_sequence_roots(tmp_path):
    sequence_root = tmp_path / "asset_a" / "sequences"
    seq_dir = sequence_root / "seq_0001"
    frames_dir = seq_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame = np.full((64, 64, 3), 255, dtype=np.uint8)
    for index in range(2):
        cv2.imwrite(str(frames_dir / f"{index:06d}.jpg"), frame)
    (seq_dir / "groundtruth.txt").write_text("10,12,20,16\n11,13,20,16\n", encoding="utf-8")

    output_root = tmp_path / "nanotrack"
    script = ROOT / "scripts" / "anti_uav" / "convert_tracker_sequences_nanotrack.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--sequence-root",
            str(sequence_root),
            "--output-root",
            str(output_root),
            "--val-ratio",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    train_json = json.loads((output_root / "rgb" / "train.json").read_text(encoding="utf-8"))
    assert "asset_a__seq_0001" in train_json
    crop_files = list((output_root / "rgb" / "crop511" / "asset_a__seq_0001").glob("*.jpg"))
    assert len(crop_files) == 2


def test_batch_tracker_sequence_eval_reads_frames_dir(tmp_path):
    seq_dir = tmp_path / "sequences" / "seq_0001"
    frames_dir = seq_dir / "frames"
    frames_dir.mkdir(parents=True)
    for index in range(2):
        frame = np.full((32, 32, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{index:06d}.jpg"), frame)
    (seq_dir / "groundtruth.txt").write_text("4,6,12,10\n0,0,0,0\n", encoding="utf-8")

    sys.path.insert(0, str(ROOT))
    from scripts.anti_uav.batch_tracker_sequence_eval import read_sequence

    frame_paths, gt = read_sequence(seq_dir, tmp_path / "unused_images", "val")
    assert [path.name for path in frame_paths] == ["000000.jpg", "000001.jpg"]
    assert gt[1] == (4.0, 6.0, 16.0, 16.0)
    assert gt[2] is None


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


def test_prepare_external_rgb_static_and_video(tmp_path):
    raw_root = tmp_path / "raw"
    dut_root = raw_root / "dut_anti_uav" / "train"
    (dut_root / "img").mkdir(parents=True)
    (dut_root / "xml").mkdir(parents=True)
    image = np.full((64, 96, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(dut_root / "img" / "dut_img.jpg"), image)
    (dut_root / "xml" / "dut_img.xml").write_text(
        """
        <annotation>
          <object>
            <name>drone</name>
            <bndbox><xmin>10</xmin><ymin>12</ymin><xmax>40</xmax><ymax>32</ymax></bndbox>
          </object>
        </annotation>
        """,
        encoding="utf-8",
    )
    tracking_dir = raw_root / "dut_anti_uav" / "extracted" / "Anti-UAV-Tracking-V0" / "video01"
    gt_dir = raw_root / "dut_anti_uav" / "extracted" / "Anti-UAV-Tracking-V0GT"
    tracking_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    cv2.imwrite(str(tracking_dir / "00001.jpg"), image)
    cv2.imwrite(str(tracking_dir / "00002.jpg"), image)
    (gt_dir / "video01_gt.txt").write_text("10 12 20 16\n11 12 20 16\n", encoding="utf-8")

    halmstad_root = raw_root / "halmstad_drone_detection"
    halmstad_root.mkdir(parents=True)
    video_path = halmstad_root / "VISIBLE_BIRD_001.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (96, 64))
    writer.write(image)
    writer.release()

    output_yolo = tmp_path / "external_yolo"
    output_nano = tmp_path / "external_nano"
    script = ROOT / "scripts" / "anti_uav" / "prepare_external_rgb_datasets.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--raw-root",
            str(raw_root),
            "--yolo-root",
            str(output_yolo),
            "--nanotrack-root",
            str(output_nano),
            "--datasets",
            "dut",
            "halmstad",
            "--frame-step",
            "1",
            "--negative-frame-step",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["yolo"]["positive"] == 3
    assert summary["yolo"]["hard_negative"] == 1
    assert (output_yolo / "ExternalRGBDrone.yaml").exists()
    label_files = list((output_yolo / "labels").rglob("*.txt"))
    assert len(label_files) == 4
    assert any(path.read_text(encoding="utf-8").strip() for path in label_files)
    assert list((output_nano / "rgb" / "crop511").rglob("*.jpg"))


def test_prepare_external_rgb_aod4_coco(tmp_path):
    raw_root = tmp_path / "raw"
    aod_root = raw_root / "aod4" / "AOD 4"
    for split in ("train", "valid"):
        (aod_root / "Images" / split).mkdir(parents=True)
        (aod_root / "Annotations" / "COCO Annotation format" / split).mkdir(parents=True)
    image = np.full((64, 96, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(aod_root / "Images" / "train" / "drone.jpg"), image)
    cv2.imwrite(str(aod_root / "Images" / "valid" / "bird.jpg"), image)
    categories = [
        {"id": 1, "name": "airplane"},
        {"id": 2, "name": "bird"},
        {"id": 3, "name": "drone"},
        {"id": 4, "name": "helicopter"},
    ]
    (aod_root / "Annotations" / "COCO Annotation format" / "train" / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "categories": categories,
                "images": [{"id": 1, "file_name": "drone.jpg", "width": 96, "height": 64}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 12, 20, 16]}],
            }
        ),
        encoding="utf-8",
    )
    (aod_root / "Annotations" / "COCO Annotation format" / "valid" / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "categories": categories,
                "images": [{"id": 2, "file_name": "bird.jpg", "width": 96, "height": 64}],
                "annotations": [{"id": 2, "image_id": 2, "category_id": 2, "bbox": [10, 12, 20, 16]}],
            }
        ),
        encoding="utf-8",
    )

    output_yolo = tmp_path / "external_yolo"
    output_nano = tmp_path / "external_nano"
    script = ROOT / "scripts" / "anti_uav" / "prepare_external_rgb_datasets.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--raw-root",
            str(raw_root),
            "--yolo-root",
            str(output_yolo),
            "--nanotrack-root",
            str(output_nano),
            "--datasets",
            "aod4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["yolo"]["positive"] == 1
    assert summary["yolo"]["hard_negative"] == 1
    assert summary["yolo"]["train"] == 1
    assert summary["yolo"]["val"] == 1


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
