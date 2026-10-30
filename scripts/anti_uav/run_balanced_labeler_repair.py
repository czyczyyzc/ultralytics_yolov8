#!/usr/bin/env python3
"""Validate all four repairs, then run only the repaired public P3 foundation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--previous-cache", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--foundation", type=Path, required=True)
    p.add_argument("--reference-snapshot", type=Path, required=True)
    p.add_argument("--preparation-pid", type=int, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--cache-workers", type=int, default=16)
    a = p.parse_args()
    for key in ("data_dir", "previous_cache", "cache_dir", "run_dir", "foundation", "reference_snapshot"):
        setattr(a, key, getattr(a, key).resolve())
    if a.run_dir.exists() or a.cache_dir.exists() or not a.device.isdecimal():
        raise ValueError("Use fresh run/cache directories and a single physical GPU ID")
    a.run_dir.mkdir(parents=True)
    (a.run_dir/"logs").mkdir()
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", WANDB_MODE="disabled", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    def status(stage, **extra):
        dump(a.run_dir/"workflow_status.json", dict(stage=stage, pid=os.getpid(),
            time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **extra))
        print(json.dumps(dict(stage=stage, **extra)), flush=True)
    def execute(stage, command):
        status(stage, command=command)
        with (a.run_dir/"logs"/(stage+".log")).open("a") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        status("waiting_for_latest_frozen_labels")
        deadline = time.monotonic()+7200
        while True:
            path = a.data_dir/"status.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get("stage") == "complete":
                break
            if not Path(f'/proc/{a.preparation_pid}').exists() or time.monotonic() > deadline:
                raise RuntimeError("Frozen data preparation failed or timed out")
            time.sleep(10)
        manifest = json.loads((a.data_dir/"manifest.json").read_text())
        if len(manifest["train_video_hashes"]) != 73:
            raise ValueError("Repair must not silently change the 73-video scope")
        reference = json.loads(a.reference_snapshot.read_text())
        snapshot = json.loads((a.data_dir/"annotation_snapshot.json").read_text())
        before = {x["task_id"]: x for x in reference["videos"]}
        after = {x["task_id"]: x for x in snapshot["videos"]}
        if before.keys() != after.keys():
            raise ValueError("Frozen repair scope differs from the stopped experiment")
        from scripts.anti_uav.prepare_snapshot_replacement_cache import pixel_plan
        changes = []
        for key, item in after.items():
            old_boxes, _ = pixel_plan(before[key])
            new_boxes, _ = pixel_plan(item)
            changes.append(dict(task_id=key, name=item["name"], revision=[before[key]["revision"], item["revision"]],
                changed_used_frames=sorted(f for f in old_boxes.keys()&new_boxes.keys() if old_boxes[f] != new_boxes[f]),
                removed_frames=len(old_boxes.keys()-new_boxes.keys()), added_frames=len(new_boxes.keys()-old_boxes.keys())))
        dump(a.run_dir/"label_refresh_audit.json", dict(snapshot_sha256=sha256(a.data_dir/"annotation_snapshot.json"),
            changes=changes, source_annotations_modified=False))
        execute("10_background_cache", [sys.executable, "-u", str(ROOT/"scripts/anti_uav/prepare_snapshot_replacement_cache.py"),
            "--data-dir", str(a.data_dir), "--previous-cache", str(a.previous_cache), "--output", str(a.cache_dir),
            "--workers", str(a.cache_workers)])
        status("11_online_composition_qa")
        from scripts.anti_uav.verify_snapshot_replacement_cache import verify_cache
        composition_qa = verify_cache(a.cache_dir, a.data_dir, a.run_dir/"replacement_qa")
        import numpy as np
        import yaml
        from scripts.anti_uav.label_pool_sampling import BalancedTinyAwareSampler
        from scripts.anti_uav.audit_labeler_expansion_regression import AuditDataset
        config_path = a.data_dir/"train_online_gray_monitor.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["tiny_sampling"] = dict(policy="balanced_v2", input_hw=[544, 960], tiny_min_px=4., tiny_max_px=8.,
            target_tiny_fraction=.30, max_extra_repeat_per_image=1)
        config["online_replacement"] = dict(config["online_replacement"], cache=str(a.cache_dir), replacement_probability=.5)
        rows = np.load(Path(config["train"]).with_suffix(".cache"), allow_pickle=True).item()["labels"]
        dataset = AuditDataset(rows)
        sampler = BalancedTinyAwareSampler(dataset, config["label_sampling"], config["tiny_sampling"], 64, 20261015)
        order = list(sampler)
        counts = Counter(order)
        positive = np.flatnonzero(sampler.categories != 2)
        if any(counts[int(i)] < 1 for i in positive):
            raise ValueError("Repair removed native positive exposure")
        index = json.loads((a.cache_dir/"index.json").read_text())
        prefix = str(a.data_dir/"images/labeler_snapshot")
        new_ready = [p for p in index["images"] if p.startswith(prefix+"/")]
        if not new_ready:
            raise ValueError("No new background passed safety gates; do not silently train an unaugmented repair")
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        dump(a.data_dir/"native_preparation_manifest.json", manifest)
        manifest.update(schema="anti_uav.balanced_labeler_repair.v1", tiny_sampling=config["tiny_sampling"],
            online_replacement=config["online_replacement"], sampler_audit=sampler.audit,
            epoch_positive=sampler.audit["epoch_positive"], epoch_negative=sampler.audit["epoch_negative"],
            final_entries=sampler.audit["epoch_total"], final_negative_fraction=sampler.audit["negative_fraction"],
            label_refresh_audit=str(a.run_dir/"label_refresh_audit.json"))
        dump(a.data_dir/"manifest.json", manifest)
        acceptance = dict(sampler.audit, native_positive_minimum_exposure=min(counts[int(i)] for i in positive),
            new_ready_backgrounds=len(new_ready), foundation_sha256=sha256(a.foundation),
            data_snapshot_sha256=sha256(a.data_dir/"annotation_snapshot.json"),
            training_videos=73, old_experiment_restarted=False, qat=False,
            online_composition_qa_frames=composition_qa["rendered"], asset_count=composition_qa["asset_count"])
        dump(a.run_dir/"repair_acceptance.json", acceptance)
        dump(a.data_dir/"status.json", dict(state, stage="complete", sampler_audit=sampler.audit,
             cache_ready=True, new_ready_backgrounds=len(new_ready)))
        usage = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, usage.strip().split(','))
        if memory > 256 or utilization > 5:
            raise RuntimeError(f"GPU {a.device} is no longer idle; refuse to disturb other jobs: {usage}")
        execute("20_repaired_foundation", [sys.executable, "-u", str(ROOT/"scripts/anti_uav/run_p3_independent_curriculum_training.py"),
            "--data", str(config_path), "--foundation", str(a.foundation), "--run-dir", str(a.run_dir/"training"),
            "--device", a.device, "--batch", "64", "--workers", "16", "--foundation-epochs", "30",
            "--teacher-epochs", "12", "--student-epochs", "15", "--localization-epochs", "0",
            "--replacement-probability", "0.5", "--expected-training-videos", "73", "--foundation-only"])
        training = json.loads((a.run_dir/"training/status.json").read_text())
        if training.get("stage") != "foundation_complete":
            raise RuntimeError("Repaired foundation did not finish successfully")
        status("foundation_complete", result=training, teacher_student_started=False,
               next_action="Review repaired foundation before continuing Teacher and Student")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
