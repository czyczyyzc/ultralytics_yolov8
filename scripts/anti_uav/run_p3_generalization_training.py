#!/usr/bin/env python3
"""Train a 47-video Teacher, FP32-distilled P3 Student, then short RKNN QAT."""

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

from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator  # noqa: E402
from scripts.anti_uav.p3_student_distillation import (  # noqa: E402
    P3QATFineTuneTrainer,
    P3StudentFP32DistillationTrainer,
    TinyAwareAddOnTrainer,
    copy_teacher_auxiliary,
    export_pure_p3_student,
    export_standard_p3_qat,
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
SEED = 20261010


def write_status(run_dir: Path, stage: str, **extra) -> None:
    value = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    temporary = run_dir / "status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(run_dir / "status.json")
    print(json.dumps(value), flush=True)


def stage_yaml(source: dict, output: Path, epoch_offset: int) -> Path:
    data = dict(source)
    data["tiny_sampling"] = dict(input_hw=[544, 960], tiny_min_px=4.0, tiny_max_px=8.0)
    replacement = data.get("online_replacement")
    if replacement:
        data["online_replacement"] = dict(
            replacement,
            epoch_offset=int(replacement.get("epoch_offset", 0)) + epoch_offset,
        )
    output.write_text(yaml.safe_dump(data, sort_keys=False))
    return output


def common_train(data: Path, args, epochs: int, patience: int) -> dict:
    return dict(
        data=str(data), imgsz=[544, 960], epochs=epochs, patience=patience,
        batch=args.batch, device=args.device, workers=args.workers, save=True, save_period=-1,
        pretrained=True, optimizer="AdamW", lrf=.1, momentum=.937, weight_decay=.0005,
        warmup_momentum=.8, warmup_bias_lr=.01, cos_lr=True, amp=True,
        deterministic=True, seed=SEED, nbs=128, rect=False, cache=False, val=True,
        plots=True, single_cls=True, mosaic=0., mixup=0., copy_paste=0., degrees=0.,
        translate=.05, scale=.2, shear=0., perspective=0., flipud=0., fliplr=.5,
        hsv_h=.015, hsv_s=.4, hsv_v=.3, close_mosaic=0, verbose=True, exist_ok=False,
    )


def validate(model_path: Path, data: Path, args, output: Path) -> dict:
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data), validator=FixedShapeGrayValidator, imgsz=[544, 960], device=args.device,
        batch=min(args.batch, 64), workers=args.workers, rect=False, conf=.001, iou=.45,
        max_det=100, plots=False, project=str(output.parent / "validation"), name=output.stem,
    )
    result = dict(metrics.gray_selection)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--p3-initial", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="7")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--teacher-epochs", type=int, default=12)
    parser.add_argument("--distill-epochs", type=int, default=15)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Continue after a completed pipeline stage")
    args = parser.parse_args()

    args.data, args.p3_initial, args.run_dir = args.data.resolve(), args.p3_initial.resolve(), args.run_dir.resolve()
    if not args.data.is_file() or not args.p3_initial.is_file():
        raise FileNotFoundError("Training YAML or initial P3 checkpoint is missing")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol_path = args.run_dir / "protocol.json"
    if protocol_path.exists() and not args.resume:
        raise FileExistsError("Refuse to overwrite an existing generalization experiment")
    (args.run_dir / "configs").mkdir(exist_ok=args.resume)
    (args.run_dir / "audit").mkdir(exist_ok=args.resume)

    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    init_seeds(SEED, deterministic=True)
    source = yaml.safe_load(args.data.read_text())
    if not source.get("label_sampling") or not source.get("online_replacement"):
        raise ValueError("This protocol requires the audited negative pool and online replacement cache")
    train_paths = Path(source["train"]).read_text().splitlines()
    val_paths = Path(source["val"]).read_text().splitlines()
    if set(train_paths) & set(val_paths):
        raise ValueError("Training and validation paths overlap")
    if any("Video00004" in Path(path).parts for path in train_paths + val_paths):
        raise ValueError("Video00004 must remain test-only")

    teacher_data = stage_yaml(source, args.run_dir / "configs/teacher.yaml", 0)
    distill_data = stage_yaml(source, args.run_dir / "configs/distill_fp32.yaml", args.teacher_epochs)
    qat_data = stage_yaml(
        source,
        args.run_dir / "configs/qat.yaml",
        args.teacher_epochs + args.distill_epochs,
    )
    protocol = dict(
        method="47-video P2+P3 Teacher -> FP32 localized P2-to-P3 distillation -> short pure-P3 RKNN QAT",
        data=str(args.data), p3_initial=str(args.p3_initial), input_hw=[544, 960],
        training_videos=47, batch=args.batch, nbs=128, seed=SEED,
        epochs=dict(teacher=args.teacher_epochs, distill_fp32=args.distill_epochs, qat=args.qat_epochs),
        tiny_sampling=dict(native_exposure_preserved=True, input_hw=[544, 960], range_px=[4, 8],
                           expected_batch_mix="~31% tiny positive / ~54% other positive / 15% negative"),
        tiny_loss=dict(multiplier=1.75, applies_to="assigned positive classification, box and DFL only", cap=2.0),
        distillation=dict(auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
                          p2_cls_weight=.25, p2_box_weight=.10, feature_weight=.10,
                          feature_region="4-8 px GT neighborhoods", temperature=2.0),
        qat=dict(epochs=args.qat_epochs, lr0=1e-5, observer_freeze_epoch=min(3, args.qat_epochs)),
        final_graph="standard YOLOv8n P3-P5; temporary P2 and projection physically removed",
        checkpoint_selection="Video00009: 0.5 F2@0.03 + 0.3 AP50 + 0.2 AP50-95",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    if protocol_path.exists():
        previous_protocol = json.loads(protocol_path.read_text())
        for key in ("data", "p3_initial", "input_hw", "training_videos", "batch", "seed", "epochs"):
            if previous_protocol.get(key) != protocol.get(key):
                raise ValueError(f"Resume protocol mismatch for {key}")
        previous_protocol["resume"] = dict(
            time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            git_commit=protocol["git_commit"],
            reason="continue from completed Teacher after Student initialization compatibility fix",
        )
        protocol_path.write_text(json.dumps(previous_protocol, indent=2) + "\n")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    try:
        teacher_best = args.run_dir / "teacher/p2p3/weights/best.pt"
        if args.resume and teacher_best.is_file():
            write_status(args.run_dir, "resume_from_completed_teacher", best=str(teacher_best))
        else:
            write_status(args.run_dir, "initialize_47_video_teacher")
            teacher_init = args.run_dir / "teacher_initialized.pt"
            teacher, teacher_report = initialize_addon_model(args.p3_initial, DEFAULT_CFG, teacher_init)
            (args.run_dir / "audit/teacher_initialization.json").write_text(json.dumps(teacher_report, indent=2) + "\n")
            teacher.add_callback("on_fit_epoch_end", lambda trainer: write_status(
                args.run_dir, "train_47_video_teacher", epoch=trainer.epoch + 1, fitness=float(trainer.fitness)
            ))
            write_status(args.run_dir, "train_47_video_teacher", epoch=0)
            teacher.train(
                trainer=TinyAwareAddOnTrainer,
                project=str(args.run_dir / "teacher"), name="p2p3", lr0=.001, warmup_epochs=1,
                **common_train(teacher_data, args, args.teacher_epochs, min(5, args.teacher_epochs)),
            )
            frozen = verify_legacy_outputs(YOLO(str(args.p3_initial)).model, YOLO(str(teacher_best)).model)
            (args.run_dir / "audit/teacher_frozen_p3.json").write_text(json.dumps(frozen, indent=2) + "\n")

        write_status(args.run_dir, "initialize_fp32_student")
        student_init = args.run_dir / "student_four_scale_initialized.pt"
        student, student_report = initialize_addon_model(args.p3_initial, DEFAULT_CFG, student_init)
        auxiliary_copy = copy_teacher_auxiliary(YOLO(str(teacher_best)).model, student.model)
        student.model.model[-1].auxiliary_training_only = True
        student.save(student_init)
        student_report["teacher_auxiliary_copy"] = auxiliary_copy
        student_report["post_copy_legacy_regression"] = verify_legacy_outputs(
            YOLO(str(args.p3_initial)).model, student.model
        )
        (args.run_dir / "audit/student_initialization.json").write_text(json.dumps(student_report, indent=2) + "\n")

        P3StudentFP32DistillationTrainer.teacher_weights = teacher_best.resolve()
        P3StudentFP32DistillationTrainer.loss_options = dict(
            auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
            p2_cls_weight=.25, p2_box_weight=.10, feature_weight=.10,
            tiny_multiplier=1.75, temperature=2.0,
        )
        student.add_callback("on_fit_epoch_end", lambda trainer: write_status(
            args.run_dir, "distill_fp32_student", epoch=trainer.epoch + 1, fitness=float(trainer.fitness)
        ))
        write_status(args.run_dir, "distill_fp32_student", epoch=0)
        student.train(
            trainer=P3StudentFP32DistillationTrainer,
            project=str(args.run_dir / "distill_fp32"), name="p3_student", lr0=.0001, warmup_epochs=0,
            **common_train(distill_data, args, args.distill_epochs, min(4, args.distill_epochs)),
        )
        fp32_weights = args.run_dir / "distill_fp32/p3_student/weights"
        fp32_exports = {}
        for checkpoint in ("best", "last"):
            fp32_exports[checkpoint] = export_pure_p3_student(
                fp32_weights / f"{checkpoint}.pt",
                args.run_dir / f"p3_student_{checkpoint}_fp32.pt",
                P3_CFG,
            )
        (args.run_dir / "audit/fp32_export.json").write_text(json.dumps(fp32_exports, indent=2) + "\n")

        write_status(args.run_dir, "qat_p3_student", epoch=0)
        P3QATFineTuneTrainer.observer_freeze_epoch = min(3, args.qat_epochs)
        qat = YOLO(str(args.run_dir / "p3_student_best_fp32.pt"))
        qat.add_callback("on_fit_epoch_end", lambda trainer: write_status(
            args.run_dir, "qat_p3_student", epoch=trainer.epoch + 1, fitness=float(trainer.fitness)
        ))
        qat.train(
            trainer=P3QATFineTuneTrainer,
            project=str(args.run_dir / "qat"), name="p3_student", lr0=1e-5, warmup_epochs=0,
            **common_train(qat_data, args, args.qat_epochs, args.qat_epochs),
        )
        qat_weights = args.run_dir / "qat/p3_student/weights"
        qat_exports = {}
        for checkpoint in ("best", "last"):
            qat_exports[checkpoint] = export_standard_p3_qat(
                qat_weights / f"{checkpoint}.pt",
                args.run_dir / f"p3_student_{checkpoint}_qat.pt",
            )
        (args.run_dir / "audit/qat_export.json").write_text(json.dumps(qat_exports, indent=2) + "\n")

        write_status(args.run_dir, "compare_video00009")
        models = {
            "baseline_qat": args.p3_initial,
            "new_fp32": args.run_dir / "p3_student_best_fp32.pt",
            "new_qat": args.run_dir / "p3_student_best_qat.pt",
        }
        comparison = {
            name: validate(path, qat_data, args, args.run_dir / f"audit/video00009_{name}.json")
            for name, path in models.items()
        }
        (args.run_dir / "comparison_video00009.json").write_text(json.dumps(comparison, indent=2) + "\n")
        write_status(args.run_dir, "complete", best=str(args.run_dir / "p3_student_best_qat.pt"))
    except Exception as error:
        previous = json.loads((args.run_dir / "status.json").read_text()) if (args.run_dir / "status.json").exists() else {}
        write_status(args.run_dir, "failed", failed_stage=previous.get("stage"), error=str(error))
        raise


if __name__ == "__main__":
    main()
