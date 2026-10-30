#!/usr/bin/env python3
"""Read-only audit of recipe, cache, augmentation and actual epoch composition."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.label_pool_sampling import TinyAwareSampler, _video_identity
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


class AuditDataset:
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)


def size_category(row):
    if not len(row["cls"]):
        return "negative"
    h, w = row["shape"]
    boxes = np.asarray(row["bboxes"], dtype=np.float32)
    edges = np.maximum(boxes[:, 2] * w, boxes[:, 3] * h) * min(544 / h, 960 / w)
    if np.any((edges >= 4) & (edges <= 8)):
        return "4to8"
    if np.any(edges < 4):
        return "lt4"
    return "8to16" if np.any(edges <= 16) else "gt16"


def origin(path, baseline):
    if path not in baseline:
        return "new_gray"
    return "RGB" if "/anti_uav300_yolo/" in path else "old_gray"


def proportions(counter):
    counts = dict(counter)
    total = sum(counts.values())
    return dict(counts=counts, fraction={k: v / total for k, v in counts.items()} if total else {})


def label_equal(a, b):
    return (tuple(a["shape"]) == tuple(b["shape"])
            and a.get("normalized") == b.get("normalized")
            and a.get("bbox_format") == b.get("bbox_format")
            and np.array_equal(a["cls"], b["cls"])
            and np.array_equal(a["bboxes"], b["bboxes"]))


def strip_locations(args):
    return {k: v for k, v in args.items()
            if k not in {"data", "model", "project", "save_dir", "device"}}


def best_curve(path):
    with path.open() as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    def score(row):
        return (.5 * row["native/c0.03/F2"] + .3 * row["native/mAP50"]
                + .2 * row["native/mAP50-95"])
    keys = ("epoch", "native/c0.03/R", "native/c0.03/P", "native/c0.03/long_4to8px/R",
            "native/c0.03/FP", "native/mAP50", "native/mAP50-95", "train/box_loss",
            "train/cls_loss", "train/dfl_loss")
    best = max(rows, key=score)
    return dict(completed_epochs=len(rows), best={k: best[k] for k in keys if k in best},
                curve=[{k: row[k] for k in keys if k in row} for row in rows])


def epoch_statistics(labels, config, baseline, augmentation_index, epoch):
    dataset = AuditDataset(labels)
    sampler = TinyAwareSampler(dataset, config.get("label_sampling"), config["tiny_sampling"], 64, 20261015)
    sampler.set_epoch(epoch)
    order = list(sampler)
    categories = [size_category(row) for row in labels]
    assert all(int(sampler.categories[i]) == (2 if category == "negative" else 0 if category == "4to8" else 1)
               for i, category in enumerate(categories))
    origins = [origin(row["im_file"], baseline) for row in labels]
    videos = [_video_identity(row["im_file"]) for row in labels]

    def summary(indices):
        indices = list(indices)
        counter = Counter(categories[i] for i in indices)
        sources = Counter(origins[i] for i in indices)
        positive = [i for i in indices if categories[i] != "negative"]
        eligible = sum(labels[i]["im_file"] in augmentation_index for i in positive)
        return dict(slots=len(indices), categories=proportions(counter), origins=proportions(sources),
                    replacement_eligible_positive=eligible,
                    eligible_fraction_of_positive=eligible / len(positive) if positive else 0,
                    tiny_per_batch64=64 * counter["4to8"] / max(len(indices), 1),
                    top_videos=Counter(videos[i] for i in indices).most_common(5))

    chunks = {}
    for j in range(10):
        chunks[f"decile_{j+1}"] = summary(order[len(order)*j//10:len(order)*(j+1)//10])
    return dict(epoch_zero_based=epoch, whole=summary(order), deciles=chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 29])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    configs, labels, protocols, listings = {}, {}, {}, {}
    for key, run in (("old52", args.old_run), ("new73", args.new_run)):
        protocols[key] = json.loads((run / "protocol.json").read_text())
        configs[key] = yaml.safe_load((run / "configs/foundation.yaml").read_text())
        listing = Path(configs[key]["train"])
        listings[key] = listing.read_text().splitlines()
        labels[key] = np.load(listing.with_suffix(".cache"), allow_pickle=True).item()["labels"]
        assert Counter(row["im_file"] for row in labels[key]) == Counter(listings[key])
    old_by = {row["im_file"]: row for row in labels["old52"]}
    new_by = {row["im_file"]: row for row in labels["new73"]}
    common = old_by.keys() & new_by.keys()
    changed = [p for p in common if not label_equal(old_by[p], new_by[p])]
    baseline = set(old_by)
    augmentation = Path(configs["new73"]["online_replacement"]["cache"]) / "index.json"
    augmentation_index = json.loads(augmentation.read_text())["images"]
    per_video = defaultdict(Counter)
    for row in labels["new73"]:
        if row["im_file"] not in baseline:
            per_video[_video_identity(row["im_file"])][size_category(row)] += 1
    recipe, curves = {}, {}
    for stage in ("foundation/p3", "teacher/p2p3", "distill_fp32/p3_student"):
        a = yaml.safe_load((args.old_run / stage / "args.yaml").read_text())
        b = yaml.safe_load((args.new_run / stage / "args.yaml").read_text())
        a, b = strip_locations(a), strip_locations(b)
        recipe[stage] = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        curves[stage] = {key: best_curve(run / stage / "results.csv")
                         for key, run in (("old52", args.old_run), ("new73", args.new_run))}
    tracked = ["scripts/anti_uav/label_pool_sampling.py", "scripts/anti_uav/gray_deployment_trainer.py",
               "scripts/anti_uav/tiny_aware_loss.py", "scripts/anti_uav/p3_student_distillation.py",
               "scripts/anti_uav/run_p3_independent_curriculum_training.py", "ultralytics/data"]
    changes = subprocess.check_output(["git", "diff", "--stat", protocols["old52"]["git_commit"],
                                      protocols["new73"]["git_commit"], "--", *tracked], cwd=ROOT, text=True)
    report = dict(schema="anti_uav.labeler_expansion_regression.v1", training_unmodified=True,
        data_integrity=dict(old_unique_paths=len(old_by), new_unique_paths=len(new_by),
            old_missing=len(old_by.keys()-new_by.keys()), changed_old_labels=changed,
            exact_old_schedule_prefix=listings["new73"][:len(listings["old52"])] == listings["old52"],
            validation_list_identical=Path(configs["old52"]["val"]).read_bytes() == Path(configs["new73"]["val"]).read_bytes(),
            validation_cache_identical=sha256(Path(configs["old52"]["val"]).with_suffix(".cache")) == sha256(Path(configs["new73"]["val"]).with_suffix(".cache")),
            public_initialization_identical=protocols["old52"]["foundation_sha256"] == protocols["new73"]["foundation_sha256"]),
        recipe_differences=recipe, training_code_diff=changes, new_videos=dict(per_video), curves=curves,
        augmentation=dict(cache=str(augmentation), index_sha256=sha256(augmentation),
            new_unique_paths_with_background=sum(p in augmentation_index for p in new_by.keys()-old_by.keys())),
        epochs={})
    for key in ("old52", "new73"):
        report["epochs"][key] = []
        for epoch in args.epochs:
            result = epoch_statistics(labels[key], configs[key], baseline, augmentation_index, epoch)
            report["epochs"][key].append(result)
            print(json.dumps(dict(version=key, epoch=epoch, whole=result["whole"],
                                  last_decile=result["deciles"]["decile_10"])), flush=True)
            dump(args.output / "partial.json", report)
    snapshot_path = Path(protocols["new73"]["data"]).parent / "annotation_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    report["snapshot"] = dict(captured_at=snapshot["captured_at"],
        task_statuses=dict(Counter(item["status"] for item in snapshot["videos"])),
        uncertain_policy=snapshot["uncertain_policy"], unreviewed_absent_policy=snapshot["unreviewed_absent_policy"])
    dump(args.output / "audit.json", report)
    print('AUDIT_COMPLETE', args.output / "audit.json", flush=True)


if __name__ == "__main__":
    main()
