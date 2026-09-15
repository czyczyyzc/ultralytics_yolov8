#!/usr/bin/env python3
"""Replace easy negatives within each source video, never change positive exposure."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import yaml

from scripts.anti_uav.build_manual_gray_video_rehearsal import label_path
from ultralytics import YOLO


def video_key(path):
    p = Path(path)
    if p.parent.name == "rgb":
        return "RGB/"+p.stem.rsplit("_", 1)[0]
    return p.parent.name


def replace_easy_negatives(schedule, scores, fraction=.2, minimum_score=.1, repeat_cap=3):
    output = list(schedule)
    groups, counts = defaultdict(list), Counter(schedule)
    for index, path in enumerate(schedule):
        if path in scores:
            groups[video_key(path)].append(index)
    changes = []
    for group, indices in sorted(groups.items()):
        candidates = sorted({schedule[i] for i in indices}, key=lambda p: -scores[p]["score"])
        donors, used_frames = [], []
        for path in candidates:
            if scores[path]["score"] < minimum_score:
                break
            frame = int(Path(path).stem.rsplit("_", 1)[-1])
            if any(abs(frame-f) < 20 for f in used_frames):
                continue
            donors.append(path)
            used_frames.append(frame)
        easy = sorted((i for i in indices if scores[schedule[i]]["score"] < .01),
                      key=lambda i: (scores[schedule[i]]["score"], -scores[schedule[i]]["sharpness"]))
        quota = min(int(len(indices)*fraction), len(easy))
        slots = [p for p in donors for _ in range(max(0, repeat_cap-counts[p]))]
        for index, donor in zip(easy[:quota], slots):
            previous = output[index]
            output[index] = donor
            changes.append(dict(group=group, removed=previous, added=donor, score=scores[donor]["score"]))
    assert len(output) == len(schedule)
    assert Counter(p for p in output if p not in scores) == Counter(p for p in schedule if p not in scores)
    assert Counter(video_key(p) for p in output if p in scores) == Counter(video_key(p) for p in schedule if p in scores)
    return output, changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--device", default="6")
    args = p.parse_args()
    cv2.setNumThreads(1)
    manifest = json.loads((args.dataset/"manifest.json").read_text())
    source = (args.dataset/"train_append_only.txt").read_text().splitlines()
    positives = set(manifest["source_training_positive_paths"])
    negatives = sorted(set(source)-positives)
    assert all(not label_path(Path(p)).read_text().strip() for p in negatives)
    assert all("gray_val" not in Path(p).parts and "Video00004" not in Path(p).parts for p in negatives)
    negative_list = args.dataset/"negative_mining_inputs.txt"
    negative_list.write_text("\n".join(negatives)+"\n")
    scores_path = args.dataset/"negative_scores.json"
    model = YOLO(str(args.model))
    train_data = Path(model.ckpt["train_args"]["data"])
    trained_manifest = json.loads((train_data.parent/"manifest.json").read_text())
    trained_hashes = {x["sha256"] for x in trained_manifest["videos"].values()}
    assert manifest["validation_video"]["sha256"] not in trained_hashes
    scores = {}
    for index, result in enumerate(model.predict(source=str(negative_list), imgsz=[544, 960], batch=32,
                                                  conf=.01, iou=.45, max_det=100, device=args.device,
                                                  rect=False, stream=True, verbose=False), 1):
        gray = cv2.cvtColor(cv2.resize(result.orig_img, (480, 270)), cv2.COLOR_BGR2GRAY)
        boxes = result.boxes.data.cpu().numpy()
        scores[result.path] = dict(score=float(boxes[:, 4].max()) if len(boxes) else 0.,
                                   sharpness=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                                   boxes=boxes.tolist())
        if index % 1000 == 0:
            print(f"Mined {index}/{len(negatives)} verified train negatives", flush=True)
    assert set(scores) == set(negatives)
    scores_path.write_text(json.dumps(dict(model=str(args.model), scores=scores), indent=2)+"\n")
    schedule = (args.dataset/"train_with_zoom.txt").read_text().splitlines()
    final, changes = replace_easy_negatives(schedule, scores)
    final_list = args.dataset/"train_hardneg.txt"
    final_list.write_text("\n".join(final)+"\n")
    data = yaml.safe_load((args.dataset/"train_gray_monitor.yaml").read_text())
    data["train"] = str(final_list)
    (args.dataset/"train_hardneg_gray_monitor.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    report = dict(model=str(args.model), unique_negative_candidates=len(negatives), replacements=len(changes),
                  positive_exposure_unchanged=True, per_video_negative_counts_unchanged=True,
                  fraction_cap=.2, hard_min_conf=.1, easy_max_conf=.01, repeat_cap=3,
                  selection="Only explicit empty training labels; no validation/test frames; old 10-video teacher.",
                  changes=changes)
    (args.dataset/"hard_negative_manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "changes"}, indent=2))


if __name__ == "__main__":
    main()
