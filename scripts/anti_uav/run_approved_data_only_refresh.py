#!/usr/bin/env python3
"""Freeze approved additions, retain the legacy recipe, then train and compare."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def select_approved_additions(items, baseline, expected=15):
    blocked = set(baseline["train_video_hashes"]) | {
        baseline["test_sha256"], baseline["validation_sha256"]}
    selected, seen = [], set()
    for item in items:
        digest = item["source_sha256"]
        if digest in blocked or item["status"] != "approved":
            continue
        if item["revision"] != item["approved_revision"]:
            raise ValueError("Approved task has an unapproved current revision")
        if digest in seen:
            raise ValueError("Duplicate source SHA in approved additions")
        seen.add(digest)
        selected.append(item)
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} approved additions, found {len(selected)}")
    return selected


def verify_frozen_scope(expected, actual):
    def identities(items):
        return {x["task_id"]: (x["source_sha256"], x["status"], x["revision"],
                              x["approved_revision"], x["annotations"]) for x in items}
    if identities(expected) != identities(actual):
        raise ValueError("Approved annotations changed while preparing the snapshot")


def recipe_difference(reference, candidate):
    keys = ("method", "foundation_sha256", "task_specific_initialization", "input_hw",
            "batch", "nbs", "seed", "epochs", "augmentation", "distillation", "qat",
            "checkpoint_selection", "final_graph")
    return {key: {"reference": reference.get(key), "candidate": candidate.get(key)}
            for key in keys if reference.get(key) != candidate.get(key)}


def choose_idle_gpu(text, allowed):
    for row in text.splitlines():
        index, memory, utilization = map(int, row.split(","))
        if index in allowed and memory <= 64 and utilization == 0:
            return index
    return None


def compare_models(run, reference, previous, device, output):
    """Evaluate every stage under one numerical/runtime protocol, without tracking."""
    import numpy as np
    import torch
    import yaml
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator, pool_native
    from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    output.mkdir()
    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False,
                         hub=False, mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((run / "protocol.json").read_text())
    models = {}
    for prefix, folder in (("old52", reference), ("previous73", previous), ("approved67", run)):
        for stage, subpath in (
            ("p3_initial", "foundation/p3/weights/best.pt"),
            ("p2p3_teacher", "teacher/p2p3/weights/best.pt"),
            ("p3_student", "p3_student_distilled_best_fp32.pt"),
        ):
            models[f"{prefix}_{stage}"] = folder / subpath
    val9 = Path(json.loads((reference / "protocol.json").read_text())["data"])
    splits = {"Video00004_test": HOLDOUT, "Video00009_selection_subset": val9}
    expected = {"Video00004_test": (2359, 448), "Video00009_selection_subset": (1421, 797)}
    frozen = {name: dict(path=str(path), sha256=sha256(path)) for name, path in models.items()}
    listings = {name: Path(yaml.safe_load(path.read_text())["val"]) for name, path in splits.items()}
    split_hashes = {name: sha256(path) for name, path in listings.items()}
    dump(output / "protocol.json", dict(models=frozen, input_hw=[544, 960], precision="FP32",
        conf_floor=.001, reported_conf=.03, nms_iou=.45, matching_iou=.5, batch=32,
        max_det=100, tf32=False, tracker=False, pooled_contains_selection_validation=True,
        checkpoint_selection=False, splits={name: dict(list=str(path), sha256=split_hashes[name])
                                            for name, path in listings.items()},
        training_recipe=protocol))
    results = {}
    for name, path in models.items():
        entries = []
        results[name] = dict(**frozen[name], splits={})
        for split, data in splits.items():
            dump(output / "status.json", dict(stage="evaluating", model=name, split=split))
            model = YOLO(str(path))
            metrics = model.val(data=str(data), validator=PairValidator, imgsz=[544, 960],
                rect=False, device=str(device), batch=32, workers=4, conf=.001, iou=.45,
                max_det=100, half=False, plots=False, verbose=False,
                project=str(output / "val"), name=f"{name}_{split}")
            native = {k: v for k, v in clean(metrics.gray_selection).items() if k.startswith("native/")}
            if (native["native/c0.03/FRAMES"], native["native/c0.03/TP"] + native["native/c0.03/FN"]) != expected[split]:
                raise ValueError("Comparative evaluation frame/GT coverage changed")
            results[name]["splits"][split] = native
            entries.append(dict(metrics=native, arrays=metrics.native_pr_arrays,
                                scales=metrics.native_scale_counts))
            np.savez_compressed(output / f"{name}_{split}_pr.npz", **metrics.native_pr_arrays)
            del model
            torch.cuda.empty_cache()
        pooled, arrays = pool_native(entries)
        results[name]["splits"]["pooled_native"] = pooled
        np.savez_compressed(output / f"{name}_pooled_pr.npz", **arrays)
        if sha256(path) != frozen[name]["sha256"]:
            raise ValueError("A comparison checkpoint changed")
        dump(output / "results.json", results)
    if any(sha256(path) != split_hashes[name] for name, path in listings.items()):
        raise ValueError("Evaluation listings changed")
    dump(output / "status.json", dict(stage="complete", models=len(models), pooled_frames=3780))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-data", type=Path, required=True)
    p.add_argument("--baseline-run", type=Path, required=True)
    p.add_argument("--previous-run", type=Path, required=True)
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--data-output", type=Path, required=True)
    p.add_argument("--cache-output", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--expected-additions", type=int, default=15)
    p.add_argument("--gpus", type=int, nargs="+", default=[2, 6])
    p.add_argument("--extraction-workers", type=int, default=8)
    p.add_argument("--background-workers", type=int, default=32)
    args = p.parse_args()
    import yaml
    from scripts.anti_uav.prepare_labeler_training_expansion import snapshot_tasks

    run = args.run_dir.resolve()
    run.mkdir(exist_ok=False)
    (run / "logs").mkdir()
    (run / "audit").mkdir()
    lock = (run / "refresh.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(stage, **extra):
        dump(run / "refresh_status.json", dict(stage=stage, pid=os.getpid(),
            time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra))
        print(json.dumps(dict(stage=stage, **extra)), flush=True)

    def command(label, cmd):
        status(label, command=[str(x) for x in cmd])
        with (run / "logs" / f"{label}.log").open("x") as log:
            subprocess.run([str(x) for x in cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        baseline = json.loads((args.baseline_data / "manifest.json").read_text())
        reference = json.loads((args.baseline_run / "protocol.json").read_text())
        previous = json.loads((args.previous_run / "protocol.json").read_text())
        if recipe_difference(reference, previous):
            raise ValueError("The old52 and previous73 training recipes are not identical")
        if len(baseline["train_video_hashes"]) != 52 or reference["training_videos"] != 52:
            raise ValueError("Unexpected legacy training split")
        if sha256(reference["foundation"]) != reference["foundation_sha256"]:
            raise ValueError("Public YOLOv8n initialization changed")
        blocked = set(baseline["train_video_hashes"]) | {baseline["test_sha256"], baseline["validation_sha256"]}
        items = select_approved_additions(snapshot_tasks(args.database, args.video_root, blocked),
                                          baseline, args.expected_additions)
        dump(run / "audit/requested_approved_scope.json", dict(videos=items,
             heldout_sha256=[baseline["test_sha256"], baseline["validation_sha256"]]))
        dump(run / "refresh_plan.json", dict(source_data=str(args.baseline_data),
            baseline_training_videos=52, new_approved_videos=len(items),
            training_videos=52 + len(items), validation_videos=1,
            validation_name="Video00009", test_name="Video00004", batch=64, seed=20261015,
            epochs=reference["epochs"], augmentation=reference["augmentation"], qat=False,
            old_samples_and_repetitions_preserved=True, existing_jobs_stopped=False,
            data_output=str(args.data_output), background_output=str(args.cache_output),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        command("prepare_data", [sys.executable, "-u", ROOT / "scripts/anti_uav/prepare_labeler_training_expansion.py",
            "--source", args.baseline_data, "--baseline-protocol", args.baseline_run / "protocol.json",
            "--database", args.database, "--video-root", args.video_root, "--output", args.data_output,
            "--workers", args.extraction_workers, "--task-snapshot", run / "audit/requested_approved_scope.json"])
        frozen = json.loads((args.data_output / "annotation_snapshot.json").read_text())
        verify_frozen_scope(items, frozen["videos"])
        config = yaml.safe_load((args.data_output / "train_online_gray_monitor.yaml").read_text())
        old_cache = Path(config["online_replacement"]["cache"])
        command("prepare_backgrounds", [sys.executable, "-u", ROOT / "scripts/anti_uav/prepare_snapshot_replacement_cache.py",
            "--data-dir", args.data_output, "--previous-cache", old_cache,
            "--output", args.cache_output, "--workers", args.background_workers])
        old_index = json.loads((old_cache / "index.json").read_text())
        index = json.loads((args.cache_output / "index.json").read_text())
        if len(index["asset_ids"]) != 328 or old_index["asset_ids"] != index["asset_ids"]:
            raise ValueError("The legacy 328-asset catalog changed")
        if any(index["images"].get(path) != info for path, info in old_index["images"].items()):
            raise ValueError("An existing augmentation cache entry changed")
        coverage = Counter(info["video_sha256"] for path, info in index["images"].items()
                           if path.startswith(str(args.data_output / "images/labeler_snapshot") + "/"))
        dump(run / "audit/new_augmentation_coverage.json", dict(asset_count=328, probability=.5,
            videos=[dict(name=x.get("original_path") or x["name"], sha256=x["source_sha256"],
                         safe_cached_positive_frames=coverage[x["source_sha256"]]) for x in items],
            unsafe_or_uncached_frames_retained_as_original=True))
        status("verify_augmentation")
        from scripts.anti_uav.verify_snapshot_replacement_cache import verify_cache
        verify_cache(args.cache_output, args.data_output, run / "audit/replacement_preview")
        config["online_replacement"]["cache"] = str(args.cache_output)
        # This contains size definitions only, deliberately no new batch quotas/repeats.
        config["tiny_sampling"] = dict(input_hw=[544, 960], tiny_min_px=4.0, tiny_max_px=8.0)
        data_yaml = args.data_output / "train_legacy52_gray_monitor.yaml"
        data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
        while True:
            usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], text=True)
            device = choose_idle_gpu(usage, args.gpus)
            if device is not None:
                break
            status("waiting_for_idle_gpu", allowed=args.gpus)
            time.sleep(60)
        command("training", [sys.executable, "-u", ROOT / "scripts/anti_uav/run_p3_independent_curriculum_training.py",
            "--data", data_yaml, "--foundation", reference["foundation"], "--run-dir", run,
            "--device", device, "--batch", "64", "--workers", "16", "--foundation-epochs", "30",
            "--teacher-epochs", "12", "--student-epochs", "15", "--localization-epochs", "0",
            "--replacement-probability", "0.5", "--expected-training-videos", 52 + len(items)])
        actual = json.loads((run / "protocol.json").read_text())
        difference = recipe_difference(reference, actual)
        dump(run / "audit/recipe_comparison.json", dict(differences=difference, data_only_change=not difference))
        if difference:
            raise ValueError("Training recipe changed unexpectedly")
        status("comparison", device=device)
        compare_models(run, args.baseline_run, args.previous_run, device, run / "data_only_comparison")
        status("complete", comparison=str(run / "data_only_comparison/results.json"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
