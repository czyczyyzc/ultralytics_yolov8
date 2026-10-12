#!/usr/bin/env python3
"""Refine a deployable P3 Student for recall and small-box localization."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from scripts.anti_uav.p3_student_distillation import (  # noqa: E402
    P3StudentFP32DistillationTrainer,
    copy_teacher_auxiliary,
    export_pure_p3_student,
)
from scripts.anti_uav.rknn_qat import canonicalize_dfl  # noqa: E402
from scripts.anti_uav.run_p3_generalization_training import (  # noqa: E402
    P3_CFG,
    common_train,
    stage_yaml,
    validate,
)
from scripts.anti_uav.train_frozen_p3_addon_p2 import DEFAULT_CFG, initialize_addon_model  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402


SEED = 20261010


def write_status(run_dir: Path, stage: str, **extra) -> None:
    value = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    temporary = run_dir / "status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(run_dir / "status.json")
    print(json.dumps(value), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--p3-initial", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="6")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    args.data = args.data.expanduser().resolve()
    args.p3_initial = args.p3_initial.expanduser().resolve()
    args.teacher = args.teacher.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    for path in (args.data, args.p3_initial, args.teacher):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    lock = (args.run_dir / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (args.run_dir / "configs").mkdir()
    (args.run_dir / "audit").mkdir()

    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    init_seeds(SEED, deterministic=True)
    source = yaml.safe_load(args.data.read_text())
    train_paths = Path(source["train"]).read_text().splitlines()
    val_paths = Path(source["val"]).read_text().splitlines()
    if set(train_paths) & set(val_paths):
        raise ValueError("Training and validation paths overlap")
    if any("Video00004" in Path(path).parts for path in train_paths + val_paths):
        raise ValueError("Video00004 must remain test-only")
    data = stage_yaml(source, args.run_dir / "configs/refine.yaml", 0)

    loss_options = dict(
        auxiliary_weight=.25,
        p3_cls_weight=.10,
        p3_dfl_weight=.05,
        p2_cls_weight=.30,
        p2_box_weight=.15,
        feature_weight=.10,
        tiny_multiplier=1.85,
        localization_weight=.15,
        localization_min_px=4.0,
        localization_max_px=64.0,
        temperature=2.0,
    )
    protocol = dict(
        method="P2-teacher FP32 P3 localization refinement",
        data=str(args.data),
        p3_initial=str(args.p3_initial),
        teacher=str(args.teacher),
        input_hw=[544, 960],
        final_graph="standard YOLOv8n P3-P5",
        epochs=args.epochs,
        batch=args.batch,
        nbs=128,
        seed=SEED,
        lr0=5e-5,
        loss=loss_options,
        checkpoint_selection="Video00009: 0.5 F2@0.03 + 0.3 AP50 + 0.2 AP50-95",
        independent_test="Video00004; evaluated only after checkpoint selection",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    (args.run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    try:
        write_status(args.run_dir, "initialize")
        canonical_initial = args.run_dir / "p3_initial_canonical_dfl.pt"
        canonical_wrapper = YOLO(str(args.p3_initial))
        canonical_report = canonicalize_dfl(canonical_wrapper.model)
        if canonical_report["modules"] != 1:
            raise RuntimeError(f"Expected one DFL module, got {canonical_report}")
        canonical_wrapper.save(canonical_initial)
        (args.run_dir / "audit/dfl_canonicalization.json").write_text(
            json.dumps(canonical_report, indent=2) + "\n"
        )
        student_init = args.run_dir / "student_four_scale_initialized.pt"
        student, report = initialize_addon_model(canonical_initial, DEFAULT_CFG, student_init)
        report["student_dfl"] = canonicalize_dfl(student.model)
        report["teacher_auxiliary_copy"] = copy_teacher_auxiliary(YOLO(str(args.teacher)).model, student.model)
        student.model.model[-1].auxiliary_training_only = True
        student.save(student_init)
        (args.run_dir / "audit/initialization.json").write_text(json.dumps(report, indent=2) + "\n")

        P3StudentFP32DistillationTrainer.teacher_weights = args.teacher
        P3StudentFP32DistillationTrainer.loss_options = loss_options
        student.add_callback("on_fit_epoch_end", lambda trainer: write_status(
            args.run_dir, "train", epoch=trainer.epoch + 1, fitness=float(trainer.fitness)
        ))
        write_status(args.run_dir, "train", epoch=0)
        student.train(
            trainer=P3StudentFP32DistillationTrainer,
            project=str(args.run_dir / "refine_fp32"),
            name="p3_student",
            lr0=5e-5,
            warmup_epochs=0,
            **common_train(data, args, args.epochs, min(4, args.epochs)),
        )
        weights = args.run_dir / "refine_fp32/p3_student/weights"
        exports = {}
        for checkpoint in ("best", "last"):
            exports[checkpoint] = export_pure_p3_student(
                weights / f"{checkpoint}.pt",
                args.run_dir / f"p3_student_{checkpoint}_fp32.pt",
                P3_CFG,
            )
        (args.run_dir / "audit/export.json").write_text(json.dumps(exports, indent=2) + "\n")

        write_status(args.run_dir, "compare_video00009")
        comparison = {
            "baseline": validate(
                canonical_initial, data, args, args.run_dir / "audit/video00009_baseline.json"
            ),
            "refined": validate(
                args.run_dir / "p3_student_best_fp32.pt",
                data,
                args,
                args.run_dir / "audit/video00009_refined.json",
            ),
        }
        (args.run_dir / "comparison_video00009.json").write_text(json.dumps(comparison, indent=2) + "\n")
        write_status(args.run_dir, "complete", best=str(args.run_dir / "p3_student_best_fp32.pt"))
    except Exception as error:
        previous = json.loads((args.run_dir / "status.json").read_text()) if (args.run_dir / "status.json").exists() else {}
        write_status(args.run_dir, "failed", failed_stage=previous.get("stage"), error=str(error))
        raise


if __name__ == "__main__":
    main()
