#!/usr/bin/env python3
"""Preserve the old schedule; append unseen train videos and isolate gray validation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from scripts.anti_uav.build_approved_gray_rehearsal import validate_frame_sets
from scripts.anti_uav.build_manual_gray_video_rehearsal import label_path, sha256_file
from scripts.anti_uav.large_target_augmentation import context_zoom, mild_capture_effects, yolo_rows


def append_unseen(old_paths, new_records, old_hashes, validation_hashes):
    additions = []
    for record in new_records:
        if record["sha256"] in old_hashes | validation_hashes:
            continue
        additions.extend(sorted(str(p) for p in Path(record["image_directory"]).glob("*.jpg")))
    assert len(additions) == len(set(additions))
    result = list(old_paths) + additions
    assert Counter(result[:len(old_paths)]) == Counter(old_paths)
    return result


def extract_validation(record, output, stride):
    task = Path(record["task"])
    manifest = json.loads((task/"manifest.json").read_text())
    included, _ = validate_frame_sets(manifest)
    assert sha256_file(Path(record["video"])) == record["sha256"]
    assert sha256_file(task/"manifest.json") == record["manifest_sha256"]
    selected = {i for i in included if i % stride == 0}
    image_dir = output/"images"/"gray_val"/record["sha256"][:12]
    label_dir = output/"labels"/"gray_val"/record["sha256"][:12]
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    paths, labels = [], []
    cap = cv2.VideoCapture(record["video"])
    index = 0
    while cap.grab():
        if index in selected:
            ok, image = cap.retrieve()
            assert ok and image.shape[:2] == (manifest["video"]["frameHeight"], manifest["video"]["frameWidth"])
            path = image_dir/f"{index:09d}.jpg"
            assert cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            text = (task/"yolo/labels"/f"{index:09d}.txt").read_text()
            (label_dir/f"{index:09d}.txt").write_text(text)
            boxes = np.array([[float(v) for v in line.split()[1:]] for line in text.splitlines() if line.strip()]).reshape(-1, 4)
            labels.append(dict(im_file=str(path), shape=image.shape[:2], bboxes=boxes))
            paths.append(str(path))
        index += 1
    cap.release()
    assert index == manifest["video"]["frameCount"] and len(paths) == len(selected)
    return paths, labels


def build_zoom(records, output, split, count, seed):
    image_dir, label_dir = output/"images"/split, output/"labels"/split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    eligible = []
    for label in records:
        h, w = label["shape"]
        boxes = np.asarray(label["bboxes"]).reshape(-1, 4)
        if len(boxes) == 1 and min(boxes[0, 2:]*[w, h]) >= 48 and max(boxes[0, 2:]*[w, h]) >= 96:
            eligible.append(label)
    if not eligible:
        return [], []
    paths, reports, usage = [], [], Counter()
    for attempt in range(count*300):
        if len(paths) == count:
            break
        slot = len(paths) % 10
        area = ((.05, .1) if slot == 0 else (.1, .25) if slot < 3 else
                (.25, .5) if slot < 5 else (.5, .75) if slot < 7 else
                (.75, .99) if slot < 9 else (.99, 1.001))
        label = eligible[int(rng.integers(len(eligible)))]
        if usage[label["im_file"]] >= 4:
            continue
        sh, sw = label["shape"]
        _, _, source_w, source_h = np.asarray(label["bboxes"])[0]*[sw, sh, sw, sh]
        if source_w*source_h*16/(960*544) < area[0]:
            continue
        if slot == 9 and min(source_w, source_h*960/544)*.85 < 240:
            continue
        image = cv2.imread(label["im_file"])
        if image is None:
            raise FileNotFoundError(label["im_file"])
        h, w = image.shape[:2]
        xywh = np.asarray(label["bboxes"])*[w, h, w, h]
        xyxy = np.column_stack((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2))
        # Fixed quotas cover transition, close-up and full-frame truncated targets.
        result = context_zoom(image, xyxy, rng, area_range=area,
                              partial=rng.random() < .2 or slot >= 7, full_frame=slot == 9)
        if result is None:
            continue
        augmented, boxes, meta = result
        augmented, effects = mild_capture_effects(augmented, rng)
        name = f"{len(paths):05d}_{hashlib.sha256(label['im_file'].encode()).hexdigest()[:12]}"
        path = image_dir/f"{name}.jpg"
        assert cv2.imwrite(str(path), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (label_dir/f"{name}.txt").write_text(yolo_rows(boxes))
        paths.append(str(path))
        reports.append(dict(source=label["im_file"], image=str(path), effects=effects, **meta))
        usage[label["im_file"]] += 1
    if split == "zoom_train" and len(paths) != count:
        raise RuntimeError(f"Insufficient high-quality zoom donors: {len(paths)}/{count}")
    return paths, reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-data", type=Path, required=True)
    p.add_argument("--approved-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation-video-token", default="stationary_video00009")
    p.add_argument("--val-stride", type=int, default=10)
    p.add_argument("--zoom-count", type=int, default=1000)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cv2.setNumThreads(1)
    old_data = yaml.safe_load(a.old_data.read_text())
    old_manifest = json.loads((a.old_data.parent/"manifest.json").read_text())
    new_manifest = json.loads((a.approved_data.parent/"manifest.json").read_text())
    old_paths = Path(old_data["train"]).read_text().splitlines()
    old_hashes = {v["sha256"] for v in old_manifest["videos"].values()}
    val = [r for r in new_manifest["approved_tasks"] if a.validation_video_token in r["video"]]
    assert len(val) == 1, "Explicitly select exactly one whole video"
    val_hashes = {r["sha256"] for r in val}
    assert not val_hashes & old_hashes
    assert not val_hashes & set(new_manifest["base_audit"]["training_video_hashes"].values())
    assert new_manifest["base_audit"]["holdout"] == "Video00004"
    schedule = append_unseen(old_paths, new_manifest["approved_tasks"], old_hashes, val_hashes)
    all_labels = {}
    for data_path in (a.old_data, a.approved_data):
        data = yaml.safe_load(data_path.read_text())
        cache = np.load(Path(data["train"]).with_suffix(".cache"), allow_pickle=True).item()
        assert Counter(x["im_file"] for x in cache["labels"]) == Counter(Path(data["train"]).read_text().splitlines())
        all_labels.update((x["im_file"], x) for x in cache["labels"])
    assert set(schedule) <= all_labels.keys()
    for path in schedule:
        assert "Video00004" not in Path(path).parts
        assert not any(Path(path).is_relative_to(Path(v["image_directory"])) for v in val)
    a.output.mkdir(parents=True)
    base_list = a.output/"train_append_only.txt"
    base_list.write_text("\n".join(schedule)+"\n")
    native_val, val_labels = extract_validation(val[0], a.output, a.val_stride)
    train_zoom, zoom_reports = build_zoom([all_labels[x] for x in sorted(set(schedule))], a.output,
                                         "zoom_train", a.zoom_count, 20260915)
    val_zoom, val_zoom_reports = build_zoom(val_labels, a.output, "zoom_val", 64, 20260916)
    final = schedule + train_zoom
    positive = sum(bool(len(all_labels[x]["bboxes"])) for x in schedule)
    negatives = len(schedule)-positive
    (a.output/"train_with_zoom.txt").write_text("\n".join(final)+"\n")
    (a.output/"val_native.txt").write_text("\n".join(native_val)+"\n")
    (a.output/"val_zoom.txt").write_text("\n".join(val_zoom)+("\n" if val_zoom else ""))
    (a.output/"val_monitor.txt").write_text("\n".join(native_val+val_zoom)+"\n")
    train_data = dict(path=str(a.output), train=str(a.output/"train_with_zoom.txt"),
                      val=str(a.output/"val_monitor.txt"), names={0: "drone"})
    (a.output/"train_gray_monitor.yaml").write_text(yaml.safe_dump(train_data, sort_keys=False))
    manifest = dict(schema="rebalanced_gray.v1", old_data=str(a.old_data), approved_data=str(a.approved_data),
                    old_train_sha256=sha256_file(Path(old_data["train"])), old_schedule_entries=len(old_paths),
                    old_prefix_exactly_preserved=True, original_gray_training_videos=14,
                    validation_video=val[0], validation_stride=a.val_stride, validation_frames=len(native_val),
                    test_video="Video00004", test_is_not_used_for_selection=True,
                    append_only_entries=len(schedule), append_only_positive=positive, negative=negatives,
                    append_only_negative_fraction=negatives/len(schedule), zoom_training_samples=len(train_zoom),
                    final_entries=len(final), final_negative_fraction=negatives/len(final),
                    zoom_validation_samples=len(val_zoom),
                    source_training_positive_paths=[x for x in sorted(set(schedule)) if len(all_labels[x]["bboxes"])],
                    train_video_hashes=sorted(set(new_manifest["base_audit"]["training_video_hashes"].values()) |
                                             {r["sha256"] for r in new_manifest["approved_tasks"] if r["sha256"] not in val_hashes}),
                    limitation="Whole video held out; other same-day videos remain in train at operator request. Not a session-independent guarantee.")
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (a.output/"zoom_manifest.json").write_text(json.dumps(dict(train=zoom_reports, validation=val_zoom_reports), indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k != "source_training_positive_paths"}, indent=2))


if __name__ == "__main__":
    main()
