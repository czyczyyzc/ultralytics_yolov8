#!/usr/bin/env python3
"""Audit cached training exposure and metadata without modifying data or weights."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import yaml


def quantiles(values):
    return dict(zip(("min", "p10", "p50", "p90", "max"),
                    map(float, np.quantile(values, [0, .1, .5, .9, 1])))) if len(values) else {}


def bucket(long_edge):
    for limit in (4, 6, 8, 12, 16, 32):
        if long_edge <= limit:
            return f"le{limit}"
    return "gt32"


def pixel_boxes(label):
    h, w = label["shape"]
    boxes = np.asarray(label["bboxes"], dtype=float).reshape(-1, 4).copy()
    assert label["normalized"] and label["bbox_format"] == "xywh"
    return boxes * [w, h, w, h]


def box_iou_xywh(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lo = np.maximum(a[:2] - a[2:] / 2, b[:2] - b[2:] / 2)
    hi = np.minimum(a[:2] + a[2:] / 2, b[:2] + b[2:] / 2)
    intersection = np.maximum(hi - lo, 0).prod()
    return float(intersection / max(a[2:].prod() + b[2:].prod() - intersection, 1e-12))


def audit_dataset(data_path):
    data = yaml.safe_load(data_path.read_text())
    manifest = json.loads((data_path.parent / "manifest.json").read_text())
    schedule = Path(data["train"]).read_text().splitlines()
    counts = Counter(p.strip() for p in schedule if p.strip())
    # Only load trusted, locally generated Ultralytics caches, never downloaded pickles.
    cache = np.load(Path(data["train"]).with_suffix(".cache"), allow_pickle=True).item()
    assert Counter(x["im_file"] for x in cache["labels"]) == counts, "Stale/different cache schedule"
    labels = {x["im_file"]: x for x in cache["labels"]}
    old_videos = manifest.get("videos", {})
    hashes = {Path(name).stem: v["sha256"] for name, v in old_videos.items()}
    hashes.update({Path(v["image_directory"]).name: v["sha256"]
                   for v in manifest.get("approved_tasks", [])})
    groups, identities = defaultdict(list), {}
    for path, count in counts.items():
        p, label = Path(path), labels[path]
        if "approved_gray" in p.parts or "manual_gray" in p.parts:
            group = p.parent.name
            key = (hashes[group], int(p.stem))
            category = "approved_or_manual"
        elif "anti_uav300_yolo" in p.parts:
            group, key, category = "RGB", (path, -1), "RGB"
        else:
            group, key, category = p.parent.name, (path, -1), "old_gray"
        assert key not in identities, f"Duplicate decoded frame path: {key}"
        identities[key] = (path, label, count)
        groups[group].append((path, label, count, category))
    summaries, totals = {}, defaultdict(list)
    for group, records in groups.items():
        category = records[0][3]
        totals[category].extend(records)
        summaries[group] = summarize(records)
    report = dict(data=str(data_path), cache_results=cache["results"],
                  scheduled=len(schedule), unique=len(counts),
                  groups=summaries, categories={k: summarize(v) for k, v in totals.items()},
                  sampling=manifest.get("sampling", manifest.get("rehearsal")),
                  source_hash=manifest["source_train_sha256"])
    return report, identities


def summarize(records):
    npos = nneg = upos = uneg = 0
    size_hist, unique_hist = Counter(), Counter()
    sizes, repeats, pixel_min = [], [], []
    for _, label, count, _ in records:
        boxes = pixel_boxes(label)
        if not len(boxes):
            nneg += count
            uneg += 1
            continue
        npos += count
        upos += 1
        h, w = label["shape"]
        gain = min(544 / h, 960 / w)
        for box in boxes:
            edge = float(max(box[2:]) * gain)
            sizes.extend([edge] * count)
            pixel_min.append(float(min(box[2:]) * gain))
            size_hist[bucket(edge)] += count
            unique_hist[bucket(edge)] += 1
        repeats.append(count)
    return dict(positive_exposures=npos, negative_exposures=nneg,
                unique_positive=upos, unique_negative=uneg,
                positive_repeat_quantiles=quantiles(repeats),
                long_edge_at_960x544_weighted=quantiles(sizes),
                short_edge_at_960x544_unique=quantiles(pixel_min),
                positive_boxes_by_long_edge=dict(size_hist),
                unique_boxes_by_long_edge=dict(unique_hist))


def compare_labels(old, new):
    common = old.keys() & new.keys()
    mismatch = []
    ious, differences = [], []
    for key in sorted(common):
        op, ol, _ = old[key]
        np_, nl, _ = new[key]
        a, b = pixel_boxes(ol), pixel_boxes(nl)
        if len(a) != len(b):
            mismatch.append(dict(old=op, new=np_, old_boxes=len(a), new_boxes=len(b)))
        elif len(a):
            differences.append(float(np.max(abs(a-b))))
            ious.extend(box_iou_xywh(x, y) for x, y in zip(a, b))
    return dict(common_unique_frames=len(common), old_only=len(old.keys()-new.keys()),
                new_only=len(new.keys()-old.keys()), presence_mismatches=mismatch,
                max_pixel_coordinate_difference=max(differences, default=0),
                common_box_iou=quantiles(ious),
                changed_boxes_over_0_1px=sum(x > .1 for x in differences))


def run_metadata(run):
    result = {}
    for stage in ("p3", "addon"):
        path = next((run / f"training_{stage}").glob("*/args.yaml"))
        args = yaml.safe_load(path.read_text())
        with (path.parent / "results.csv").open() as stream:
            rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        best = max(rows, key=lambda r: .1*r["metrics/mAP50(B)"] + .9*r["metrics/mAP50-95(B)"])
        result[stage] = dict(args=args, rgb_selected_epoch=best["epoch"],
                             best_rgb=best, last=rows[-1], curve=rows,
                             manifest=json.loads((path.parent / "training_manifest.json").read_text()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--new-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    metadata = {name: run_metadata(run) for name, run in (("old", a.old_run), ("new", a.new_run))}
    old, oi = audit_dataset(Path(metadata["old"]["p3"]["args"]["data"]))
    new, ni = audit_dataset(Path(metadata["new"]["p3"]["args"]["data"]))
    diffs = {}
    for stage in ("p3", "addon"):
        x, y = metadata["old"][stage]["args"], metadata["new"][stage]["args"]
        diffs[stage] = {k: [x.get(k), y.get(k)] for k in x.keys() | y.keys() if x.get(k) != y.get(k)}
    result = dict(metadata=metadata, args_diff=diffs, old_dataset=old, new_dataset=new,
                  common_label_comparison=compare_labels(oi, ni),
                  size_policy="Long-edge exclusive bins in input pixels before augmentation; schedule weighted.",
                  limitation="Exposure/label audit identifies changes, not causal effect. No holdout-based model selection.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("args_diff", "common_label_comparison")}, indent=2))
    for name, d in (("old", old), ("new", new)):
        print(name, json.dumps(d["categories"], indent=2))


if __name__ == "__main__":
    main()
