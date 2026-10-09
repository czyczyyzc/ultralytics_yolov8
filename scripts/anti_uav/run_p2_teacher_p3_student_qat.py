#!/usr/bin/env python3
"""Append three approved videos and train a deployable QAT P3 student from a P2+P3 teacher."""

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
    P3StudentDistillationTrainer,
    copy_teacher_auxiliary,
    export_pure_p3_student,
)
from scripts.anti_uav.train_frozen_p3_addon_p2 import (  # noqa: E402
    DEFAULT_CFG,
    initialize_addon_model,
    verify_legacy_outputs,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402


P3_CFG = ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"


def write_status(run: Path, stage: str, **extra) -> None:
    value = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    temporary = run / "status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(run / "status.json")
    print(json.dumps(value), flush=True)


def execute(log: Path, command: list[str], env: dict[str, str]) -> None:
    with log.open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--expanded-dataset", type=Path, required=True)
    parser.add_argument("--approved-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--p3-initial", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="6")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.run_dir / "protocol.json").exists():
        raise FileExistsError("Refuse to overwrite an existing distillation experiment")
    (args.run_dir / "logs").mkdir()
    snapshot = args.run_dir / "approved_expansion_snapshot.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT), WANDB_MODE="disabled", WANDB_DISABLED="true",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MPLBACKEND="Agg")
    scripts = ROOT / "scripts/anti_uav"
    python = sys.executable
    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    try:
        write_status(args.run_dir, "audit_new_approved_videos")
        execute(args.run_dir / "logs/audit.log", [
            python, str(scripts / "audit_approved_gray_expansion.py"),
            "--dataset", str(args.source_dataset), "--approved-root", str(args.approved_root),
            "--video-root", str(args.video_root), "--output", str(snapshot),
        ], env)
        audit = json.loads(snapshot.read_text())
        if audit["new_videos"] != 3:
            raise ValueError(f"Expected exactly three reviewed new videos, found {audit['new_videos']}")

        write_status(args.run_dir, "prepare_47_video_dataset")
        execute(args.run_dir / "logs/prepare_dataset.log", [
            python, str(scripts / "append_approved_gray_native.py"),
            "--source", str(args.source_dataset), "--source-yaml", "train_online_gray_monitor.yaml",
            "--snapshot", str(snapshot), "--video-root", str(args.video_root),
            "--old-root", str(args.old_root), "--output", str(args.expanded_dataset),
            "--positive-stride", "1", "--negative-stride", "1", "--negative-fraction", "0.15",
        ], env)
        data = args.expanded_dataset / "train_online_gray_monitor.yaml"
        config = yaml.safe_load(data.read_text())
        if not config.get("online_replacement"):
            raise ValueError("The existing online target replacement policy was not preserved")
        manifest = json.loads((args.expanded_dataset / "manifest.json").read_text())
        if manifest["original_gray_training_videos"] != 47:
            raise ValueError("Expanded dataset is not the expected 47-video set")

        write_status(args.run_dir, "initialize_student")
        init_seeds(20260915, deterministic=True)
        initialized_path = args.run_dir / "student_four_scale_initialized.pt"
        student, initialization = initialize_addon_model(args.p3_initial, DEFAULT_CFG, initialized_path)
        teacher = YOLO(str(args.teacher))
        auxiliary_copy = copy_teacher_auxiliary(teacher.model, student.model)
        initialization["teacher_auxiliary_copy"] = auxiliary_copy
        initialization["post_copy_legacy_regression"] = verify_legacy_outputs(YOLO(str(args.p3_initial)).model, student.model)
        student.model.model[-1].auxiliary_training_only = True
        student.save(initialized_path)
        (args.run_dir / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")

        protocol = dict(
            method="P2 Teacher -> P3 Student distillation + training-only auxiliary P2 + RKNN INT8 QAT",
            source_dataset=str(args.source_dataset.resolve()), expanded_dataset=str(args.expanded_dataset.resolve()),
            training_videos=47, newly_added_videos=3,
            new_positive_frames=audit["new_positive_frames"], new_negative_frames=audit["new_negative_frames"],
            uncertain_frames_excluded=sum(row.get("uncertain_frames", 0) for row in audit["new_rows"]),
            p3_initial=str(args.p3_initial.resolve()), teacher=str(args.teacher.resolve()),
            final_graph="standard YOLOv8n P3-P5; auxiliary P2 physically removed",
            input_hw=[544, 960], epochs=args.epochs, batch=args.batch, nbs=128, seed=20260915,
            validation_video=manifest["validation_video"]["video"], test_video="Video00004",
            replacement_assets=config["online_replacement"].get("asset_count", 328),
            replacement_probability=config["online_replacement"]["replacement_probability"],
            negative_fraction=config["label_sampling"]["target_negative_fraction"],
            qat=dict(weight="symmetric per-output-channel INT8", activation="EMA asymmetric per-tensor INT8",
                     start_epoch=0, observer_freeze_epoch=min(12, args.epochs)),
            distillation=dict(auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
                              p2_cls_weight=.25, p2_box_weight=.10, temperature=2.0),
            checkpoint_selection="pure P3 validation fitness: 0.5 F2@0.03 + 0.3 AP50 + 0.2 AP50-95",
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        )
        (args.run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

        P3StudentDistillationTrainer.teacher_weights = args.teacher.resolve()
        P3StudentDistillationTrainer.qat_start_epoch = 0
        P3StudentDistillationTrainer.qat_observer_freeze_epoch = min(12, args.epochs)
        P3StudentDistillationTrainer.loss_options = protocol["distillation"]
        student.add_callback("on_fit_epoch_end", lambda trainer: write_status(
            args.run_dir, "training", epoch=trainer.epoch + 1, fitness=float(trainer.fitness)
        ))
        write_status(args.run_dir, "training", epoch=0)
        student.train(
            trainer=P3StudentDistillationTrainer,
            data=str(data), project=str(args.run_dir / "training"), name="p3_student_qat",
            exist_ok=False, epochs=args.epochs, patience=args.epochs, batch=args.batch,
            imgsz=[544, 960], device=args.device, workers=args.workers, save=True, save_period=1,
            pretrained=True, optimizer="AdamW", lr0=.0001, lrf=.1, momentum=.937,
            weight_decay=.0005, warmup_epochs=0, warmup_momentum=.8, warmup_bias_lr=.01,
            cos_lr=True, amp=True, deterministic=True, seed=20260915, nbs=128,
            rect=False, cache=False, val=True, plots=True, single_cls=True,
            mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=.05, scale=.2,
            shear=0., perspective=0., flipud=0., fliplr=.5, hsv_h=.015, hsv_s=.4,
            hsv_v=.3, close_mosaic=0, verbose=True,
        )

        write_status(args.run_dir, "extract_pure_p3")
        reports = {}
        weights = args.run_dir / "training/p3_student_qat/weights"
        for checkpoint in ("best", "last"):
            reports[checkpoint] = export_pure_p3_student(
                weights / f"{checkpoint}.pt", args.run_dir / f"p3_student_{checkpoint}_qat.pt", P3_CFG
            )
        (args.run_dir / "pure_p3_export.json").write_text(json.dumps(reports, indent=2) + "\n")
        write_status(args.run_dir, "complete", best=str(args.run_dir / "p3_student_best_qat.pt"))
    except Exception as error:
        previous = json.loads((args.run_dir / "status.json").read_text()) if (args.run_dir / "status.json").exists() else {}
        write_status(args.run_dir, "failed", failed_stage=previous.get("stage"), error=str(error))
        raise


if __name__ == "__main__":
    main()
