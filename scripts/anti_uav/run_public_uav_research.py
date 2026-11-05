#!/usr/bin/env python3
"""Append audited public positives to the unchanged native curriculum for research."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_positive_label(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        values = [float(x) for x in line.split()]
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError(f"Invalid YOLO row: {path}")
        cls, x, y, w, h = values
        if cls != 0 or min(w, h) <= 0 or not all(0 <= v <= 1 for v in values[1:]):
            raise ValueError(f"Unexpected class or box: {path}")
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
            raise ValueError(f"Box extends beyond the image: {path}")
        rows.append(values)
    if not rows:
        raise ValueError(f"Public supplement must be an annotated positive: {path}")
    return rows


def sampling_extension(config, base_count, added_positive):
    result = deepcopy(config)
    sample = result["label_sampling"]
    if base_count - sample["negative_pool_count"] != sample["anchor_slots"]:
        raise ValueError("Baseline schedule differs from its audited sampler")
    old_epoch = sample["anchor_slots"] + sample["negatives_per_epoch"]
    fraction = float(sample["target_negative_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("Invalid native negative fraction")
    old_negative = round(old_epoch * fraction)
    old_positive = old_epoch - old_negative
    fixed_negative = old_negative - sample["negatives_per_epoch"]
    desired = round((old_positive + added_positive) * fraction / (1-fraction))
    budget = desired - fixed_negative
    if not 0 < budget <= sample["negative_pool_count"]:
        raise ValueError("Cannot retain negative fraction with the reviewed pool")
    sample.update(anchor_slots=sample["anchor_slots"]+added_positive, negatives_per_epoch=budget)
    return result, dict(native_epoch=old_epoch, native_positive=old_positive,
        native_negative=old_negative, added_positive=added_positive,
        epoch_positive=old_positive+added_positive, epoch_negative=desired,
        epoch_slots=old_positive+added_positive+desired,
        negative_fraction=desired/(old_positive+added_positive+desired))


def inspect_image(path):
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    identity = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    return identity, image.shape[:2]


def check_groups(records):
    groups = {}
    for record in records:
        key = record["dataset"], record["group"]
        split = record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown public split: {split}")
        if key in groups and groups[key] != split:
            raise ValueError(f"A public source group crosses splits: {key}")
        groups[key] = split


def completed_manifest(path):
    summary = json.loads((path.parent / "summary.json").read_text())
    if summary.get("complete") is not True or summary.get("manifest_sha256") != sha256(path):
        raise ValueError("Public ingestion is incomplete or its manifest changed")
    return summary


def bind_control(args):
    import yaml
    control = json.loads((args.baseline_run / "protocol.json").read_text())
    if Path(control["data"]).resolve() != args.baseline_yaml.resolve():
        raise ValueError("Baseline YAML is not the control run's data")
    source = yaml.safe_load(args.baseline_yaml.read_text())
    if source.get("trusted_label_caches") != control["trusted_label_caches"]:
        raise ValueError("Baseline cache pins differ from the control run")
    return control


def prepare(args):
    import cv2
    import numpy as np
    import yaml
    cv2.setNumThreads(1)
    source = yaml.safe_load(args.baseline_yaml.read_text())
    base_manifest = json.loads((args.baseline_yaml.parent / "manifest.json").read_text())
    if len(base_manifest["train_video_hashes"]) != 69:
        raise ValueError("This research experiment requires the frozen 69-video control")
    if source.get("online_scale"):
        raise ValueError("Do not add scale-view augmentation to this data-only experiment")
    args.data_output.mkdir(parents=True, exist_ok=False)
    base = Path(source["train"]).read_text().splitlines()
    val_text = Path(source["val"]).read_text()
    from scripts.anti_uav.run_native_pool_comparison import HOLDOUT
    protected = set(val_text.splitlines())
    protected.update(Path(yaml.safe_load(HOLDOUT.read_text())["val"]).read_text().splitlines())
    records, public_sources = [], []
    for manifest in args.public_manifest:
        completed_manifest(manifest)
        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        records.extend(entries)
        public_sources.append(dict(path=str(manifest), sha256=sha256(manifest), entries=len(entries)))
    check_groups(records)
    heldout = [x["image"] for x in records if x["split"] != "train"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        protected_hashes = {x[0] for x in pool.map(inspect_image, sorted(protected | set(heldout)))}
    candidates = [x for x in records if x["split"] == "train"]
    images = [x["image"] for x in candidates]
    if len(set(images)) != len(images) or set(images) & set(base):
        raise ValueError("Public manifest contains repeated or native image paths")
    accepted, labels, excluded, seen = [], [], [], set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, (identity, shape) in zip(candidates, pool.map(inspect_image, images)):
            if identity in protected_hashes or identity in seen:
                excluded.append(dict(image=row["image"], reason="exact_pixel_duplicate_heldout_or_public"))
                continue
            seen.add(identity)
            boxes = np.asarray(read_positive_label(row["label"]), dtype=np.float32)
            accepted.append(dict(row, pixel_sha256=identity))
            labels.append(dict(im_file=row["image"], shape=shape, cls=boxes[:, :1],
                bboxes=boxes[:, 1:], segments=[], keypoints=None, normalized=True, bbox_format="xywh"))
    if not accepted:
        raise ValueError("No safe public training positives remain")
    config, exposure = sampling_extension(source, len(base), len(accepted))
    if (exposure["native_positive"], exposure["native_negative"]) != (
            base_manifest["epoch_positive"], base_manifest["epoch_negative"]):
        raise ValueError("Native exposure counts differ from the frozen manifest")
    # Keep the exact repeated native prefix, not merely the same set of files.
    train = args.data_output / "train_public.txt"
    val = args.data_output / "val_monitor.txt"
    train.write_text("\n".join(base + [row["image"] for row in accepted]) + "\n")
    val.write_text(val_text)
    base_cache_path = Path(source["train"]).with_suffix(".cache")
    pinned = source["trusted_label_caches"][str(Path(source["train"]))]
    if sha256(base_cache_path) != pinned["cache_sha256"] or sha256(source["train"]) != pinned["source_sha256"]:
        raise ValueError("Native cache/list differs from the baseline protocol")
    cache = np.load(base_cache_path, allow_pickle=True).item()
    if Counter(x["im_file"] for x in cache["labels"]) != Counter(base):
        raise ValueError("Native cache does not preserve the repeated image schedule")
    cache["labels"] = sorted(cache["labels"] + labels, key=lambda row: row["im_file"])
    count = len(cache["labels"])
    cache.update(hash=hashlib.sha256((sha256(base_cache_path)+sha256(train)+json.dumps(public_sources)).encode()).hexdigest(),
        results=(count, 0, sum(not len(x["cls"]) for x in cache["labels"]), 0, count), msgs=[])
    with train.with_suffix(".cache").open("wb") as stream:
        np.save(stream, cache, allow_pickle=True)
    val_source = Path(source["val"])
    val_pin = source["trusted_label_caches"][str(val_source)]
    if (sha256(val_source) != val_pin["source_sha256"] or
            sha256(val_source.with_suffix(".cache")) != val_pin["cache_sha256"]):
        raise ValueError("Native validation cache/list changed")
    if any(sha256(row["path"]) != row["sha256"] for row in public_sources):
        raise ValueError("A public manifest changed during preparation")
    shutil.copy2(val_source.with_suffix(".cache"), val.with_suffix(".cache"))
    config.update(path=str(args.data_output), train=str(train), val=str(val))
    pins = {}
    for listing in (train, val):
        cached = np.load(listing.with_suffix(".cache"), allow_pickle=True).item()
        pins[str(listing)] = dict(path=str(listing.with_suffix(".cache")), source_sha256=sha256(listing),
            cache_sha256=sha256(listing.with_suffix(".cache")), dataset_hash=str(cached["hash"]), entries=len(cached["labels"]))
    config["trusted_label_caches"] = pins
    output = args.data_output / "train_public_gray_monitor.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(base_manifest, schema="anti_uav.public_research.v1", source_dataset=str(args.baseline_yaml.parent),
        public_sources=public_sources, public_training_images=len(accepted), **exposure,
        research_only=True, commercial_clearance=False, native_prefix_preserved=True,
        source_train_sha256=sha256(source["train"]), source_validation_sha256=sha256(source["val"]),
        external_splits_not_used_for_selection=True,
        deduplication="Exact decoded pixels against public heldouts, native validation/test, and public train; not perceptual/near-duplicate deduplication")
    dump(args.data_output / "manifest.json", manifest)
    dump(args.data_output / "public_accepted.json", accepted)
    dump(args.data_output / "public_excluded.json", excluded)
    dump(args.data_output / "summary.json", dict(stage="complete", **exposure,
        dataset_counts=dict(Counter(x["dataset"] for x in accepted)), excluded=len(excluded),
        research_only=True, config=str(output)))
    return output


def compare(args, device):
    import numpy as np
    import torch
    import yaml
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator, pool_native
    from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
    from scripts.anti_uav.run_approved_data_only_refresh import recipe_difference
    from ultralytics import YOLO
    deadline = time.monotonic() + 24*3600
    while True:
        state = args.baseline_run / "refresh_status.json"
        value = json.loads(state.read_text()) if state.exists() else {}
        if value.get("stage") == "complete":
            break
        if value.get("stage") == "failed" or time.monotonic() > deadline:
            raise RuntimeError("The native control did not complete; do not compare partial checkpoints")
        dump(args.run_dir / "research_status.json", dict(stage="waiting_for_control_completion"))
        time.sleep(120)
    old = json.loads((args.baseline_run / "protocol.json").read_text())
    new = json.loads((args.run_dir / "protocol.json").read_text())
    if recipe_difference(old, new):
        raise ValueError("Training recipe changed beyond its public-data supplement")
    output = args.run_dir / "comparison"
    output.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    splits = {"Video00004": HOLDOUT, "Video00009": args.baseline_yaml}
    counts = {"Video00004": (2359, 448), "Video00009": (1421, 797)}
    result = {}
    for arm, folder in (("native69", args.baseline_run), ("native69_public", args.run_dir)):
        for stage, relative in (("p3_initial", "foundation/p3/weights/best.pt"),
                                ("teacher", "teacher/p2p3/weights/best.pt"),
                                ("student", "p3_student_distilled_best_fp32.pt")):
            weights = folder / relative
            identity = sha256(weights)
            name = arm + "_" + stage
            entries, result[name] = [], dict(weights=str(weights), sha256=identity, splits={})
            for split, data in splits.items():
                dump(args.run_dir / "research_status.json", dict(stage="evaluating", model=name, split=split))
                model = YOLO(str(weights))
                metrics = model.val(data=str(data), validator=PairValidator, imgsz=[544, 960],
                    rect=False, device=str(device), batch=32, workers=4, conf=.001, iou=.45,
                    max_det=100, half=False, plots=False, verbose=False,
                    project=str(output / "val"), name=name+"_"+split)
                native = {k: v for k, v in clean(metrics.gray_selection).items() if k.startswith("native/")}
                if (native["native/c0.03/FRAMES"], native["native/c0.03/TP"]+native["native/c0.03/FN"]) != counts[split]:
                    raise ValueError("Holdout coverage changed")
                result[name]["splits"][split] = native
                entries.append(dict(metrics=native, arrays=metrics.native_pr_arrays, scales=metrics.native_scale_counts))
                np.savez_compressed(output / (name+"_"+split+".npz"), **metrics.native_pr_arrays)
                del model
                torch.cuda.empty_cache()
            pooled, arrays = pool_native(entries)
            result[name]["splits"]["pooled"] = pooled
            np.savez_compressed(output / (name+"_pooled.npz"), **arrays)
            if sha256(weights) != identity:
                raise ValueError("A checkpoint changed while evaluating")
            dump(output / "results.json", result)
    dump(output / "protocol.json", dict(research_only=True, conf=.03, iou=.5, nms_iou=.45,
        precision="FP32", input_hw=[544, 960], tracker=False,
        pooled_contains_selection_validation=True, train_recipe_changes={},
        splits={k: dict(yaml=str(v), yaml_sha256=sha256(v),
                        list_sha256=sha256(yaml.safe_load(v.read_text())["val"])) for k, v in splits.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-yaml", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--public-manifest", required=True, nargs="+", type=Path)
    parser.add_argument("--data-output", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpus", type=int, nargs="+", default=[6])
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args()
    for key in ("baseline_yaml", "baseline_run", "data_output", "run_dir"):
        setattr(args, key, getattr(args, key).resolve())
    args.public_manifest = [x.resolve() for x in args.public_manifest]
    if args.workers < 1:
        parser.error("workers must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    lock = (args.run_dir / "research.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        dump(args.run_dir / "research_status.json", dict(stage="preparing", research_only=True, pid=os.getpid()))
        control = bind_control(args)
        data = prepare(args)
        from scripts.anti_uav.run_approved_data_only_refresh import choose_idle_gpu
        if sha256(control["foundation"]) != control["foundation_sha256"]:
            raise ValueError("Public initialization checkpoint changed")
        usage = ""
        while True:
            usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            device = choose_idle_gpu(usage, args.gpus, args.allow_shared_gpu)
            if device is not None:
                break
            dump(args.run_dir / "research_status.json", dict(stage="waiting_for_allowed_gpu", usage=usage))
            time.sleep(60)
        command = [sys.executable, "-u", str(ROOT / "scripts/anti_uav/run_p3_independent_curriculum_training.py"),
            "--data", str(data), "--foundation", control["foundation"], "--run-dir", str(args.run_dir),
            "--device", str(device), "--batch", "64", "--workers", "16", "--foundation-epochs", "30",
            "--teacher-epochs", "12", "--student-epochs", "15", "--localization-epochs", "0",
            "--replacement-probability", "0.5", "--expected-training-videos", "69"]
        dump(args.run_dir / "launch.json", dict(command=command, gpu=device, usage_before_launch=usage,
            allow_shared_gpu=args.allow_shared_gpu, research_only=True, commercial_clearance=False,
            control_unchanged=str(args.baseline_run), public_sources=[str(x) for x in args.public_manifest]))
        dump(args.run_dir / "research_status.json", dict(stage="training", device=device, pid=os.getpid()))
        with (args.run_dir / "training.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        compare(args, device)
        dump(args.run_dir / "research_status.json", dict(stage="complete", research_only=True))
    except BaseException as error:
        dump(args.run_dir / "research_status.json", dict(stage="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()
