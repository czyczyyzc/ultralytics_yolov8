#!/usr/bin/env python3
"""Jointly fine-tune an existing P2+P3 Teacher and distilled P3 Student."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
from scripts.anti_uav.p3_student_distillation import (
    P3StudentFP32DistillationTrainer,
    TinyAwareP3Trainer,
    copy_teacher_auxiliary,
)
from scripts.anti_uav.run_p3_independent_curriculum_training import (
    add_status_callback,
    common_train,
    export_student_pair,
    sha256,
    validate,
)
from scripts.anti_uav.train_frozen_p3_addon_p2 import (
    DEFAULT_CFG,
    initialize_addon_model,
)
from ultralytics import YOLO
from ultralytics.nn.modules import FrozenP3AddOnP2Detect
from ultralytics.utils import SETTINGS
from ultralytics.utils.torch_utils import de_parallel, init_seeds


SEED = 20261022


class FullTeacherFineTuneTrainer(TinyAwareP3Trainer):
    """Explicitly unlock the inherited P3 graph and existing P2 add-on."""

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        for parameter in de_parallel(model).parameters():
            parameter.requires_grad_(True)
        return super().build_optimizer(model, name, lr, momentum, decay, iterations)


def status(run_dir: Path, stage: str, **extra) -> None:
    value = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    temporary = run_dir / "status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(run_dir / "status.json")
    print(json.dumps(value), flush=True)


def assert_levels(path: Path, expected: int) -> None:
    detector = YOLO(str(path)).model.model[-1]
    if len(detector.stride) != expected:
        raise ValueError(f"Expected {expected} detector levels in {path}, found {len(detector.stride)}")


def clean_validate(model_path: Path, data: Path, args, output: Path) -> dict:
    """Validate sequential models without retaining multiprocessing DataLoaders."""
    validation_args = argparse.Namespace(**vars(args))
    validation_args.workers = 0
    try:
        return validate(model_path, data, validation_args, output)
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def prepare_data(source_path: Path, run_dir: Path) -> tuple[Path, Path, dict]:
    source = yaml.safe_load(source_path.read_text())
    if not source.get("label_sampling") or not source.get("online_replacement"):
        raise ValueError("Expected audited negative sampling and online replacement")
    if source.get("online_scale"):
        raise ValueError("Cross-target fine-tuning uses native frames, not generated scale views")
    source["tiny_sampling"] = dict(input_hw=[544, 960], tiny_min_px=4.0, tiny_max_px=8.0)
    source["online_replacement"] = dict(source["online_replacement"], epoch_offset=0)
    train_data = run_dir / "configs/train_52videos.yaml"
    train_data.write_text(yaml.safe_dump(source, sort_keys=False))

    manifest = json.loads((source_path.parent / "manifest.json").read_text())
    new_videos = manifest.get("latest_appended_videos", [])
    if len(new_videos) != 2:
        raise ValueError(f"Expected exactly two newly appended videos, found {len(new_videos)}")
    cross_images = sorted(
        str(image)
        for row in new_videos
        for image in Path(row["image_directory"]).glob("*.jpg")
    )
    if len(cross_images) != sum(row["positive_samples"] + row["negative_samples"] for row in new_videos):
        raise ValueError("Cross-target extraction count differs from the audited manifest")
    cross_list = run_dir / "configs/cross_training_fit.txt"
    cross_list.write_text("\n".join(cross_images) + "\n")
    cross_data = run_dir / "configs/cross_training_fit.yaml"
    cross_data.write_text(yaml.safe_dump(dict(path=str(source_path.parent), train=str(cross_list),
                                              val=str(cross_list), names={0: "drone"}), sort_keys=False))
    return train_data, cross_data, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--teacher-epochs", type=int, default=6)
    parser.add_argument("--student-epochs", type=int, default=8)
    parser.add_argument("--expected-training-videos", type=int, default=52)
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.teacher = args.teacher.resolve()
    args.student = args.student.resolve()
    args.run_dir = args.run_dir.resolve()
    if any(not path.is_file() for path in (args.data, args.teacher, args.student)):
        raise FileNotFoundError("Data or initialization checkpoint is missing")
    if min(args.batch, args.workers, args.teacher_epochs, args.student_epochs) < 1:
        raise ValueError("Batch, workers and epochs must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "configs").mkdir()
    (args.run_dir / "audit").mkdir()
    lock = (args.run_dir / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False,
                         hub=False, mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    init_seeds(SEED, deterministic=True)
    assert_levels(args.teacher, 4)
    assert_levels(args.student, 3)
    train_data, cross_data, manifest = prepare_data(args.data, args.run_dir)
    if len(manifest.get("train_video_hashes", [])) != args.expected_training_videos:
        raise ValueError("Expanded dataset does not contain the expected number of training videos")

    protocol = dict(
        method="joint 50-video replay + fourfold new cross-target exposure; full P2+P3 Teacher fine-tune; Teacher-to-P3 Student distillation",
        data=str(args.data), data_manifest_sha256=sha256(args.data.parent / "manifest.json"),
        initial_teacher=str(args.teacher), initial_teacher_sha256=sha256(args.teacher),
        initial_student=str(args.student), initial_student_sha256=sha256(args.student),
        training_videos=args.expected_training_videos, input_hw=[544, 960], batch=args.batch,
        seed=SEED, epochs=dict(teacher=args.teacher_epochs, student=args.student_epochs),
        learning_rates=dict(teacher=1e-4, student=5e-5), qat=False,
        checkpoint_selection="unchanged Video00009 monitor; cross-video metric is training-fit only",
        augmentation=dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
                          translate=0.05, scale=0.2, shear=0.0, perspective=0.0,
                          flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
                          online_replacement_probability=0.5, online_scale=False),
        limitation="Both new cross-target videos are used in training; their reported cross metrics measure adaptation fit, not unseen-video generalization.",
    )
    (args.run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    try:
        status(args.run_dir, "baseline_evaluation")
        baseline = {
            "teacher_video00009": clean_validate(args.teacher, train_data, args, args.run_dir / "audit/baseline_teacher_video00009.json"),
            "student_video00009": clean_validate(args.student, train_data, args, args.run_dir / "audit/baseline_student_video00009.json"),
            "teacher_cross_training_fit": clean_validate(args.teacher, cross_data, args, args.run_dir / "audit/baseline_teacher_cross.json"),
            "student_cross_training_fit": clean_validate(args.student, cross_data, args, args.run_dir / "audit/baseline_student_cross.json"),
        }

        status(args.run_dir, "teacher_finetune", epoch=0)
        FullTeacherFineTuneTrainer.validator_class = FixedShapeGrayValidator
        FullTeacherFineTuneTrainer.tiny_multiplier = 1.5
        FullTeacherFineTuneTrainer.localization_weight = .08
        FullTeacherFineTuneTrainer.localization_min_px = 4.0
        FullTeacherFineTuneTrainer.localization_max_px = 960.0
        teacher = YOLO(str(args.teacher))
        add_status_callback(teacher, args.run_dir, "teacher_finetune")
        teacher.train(
            trainer=FullTeacherFineTuneTrainer,
            project=str(args.run_dir / "teacher"), name="p2p3",
            lr0=1e-4, warmup_epochs=0,
            **common_train(train_data, args, args.teacher_epochs),
        )
        teacher_best = args.run_dir / "teacher/p2p3/weights/best.pt"
        assert_levels(teacher_best, 4)

        status(args.run_dir, "student_initialization")
        student_init = args.run_dir / "student_four_scale_initialized.pt"
        student, initialization = initialize_addon_model(args.student, DEFAULT_CFG, student_init)
        teacher_model = YOLO(str(teacher_best)).model
        if not isinstance(teacher_model.model[-1], FrozenP3AddOnP2Detect):
            raise TypeError("Fine-tuned Teacher is not a P2+P3 model")
        initialization["teacher_auxiliary_copy"] = copy_teacher_auxiliary(teacher_model, student.model)
        student.model.model[-1].auxiliary_training_only = True
        student.save(student_init)
        (args.run_dir / "audit/student_initialization.json").write_text(
            json.dumps(initialization, indent=2) + "\n"
        )

        P3StudentFP32DistillationTrainer.teacher_weights = teacher_best.resolve()
        P3StudentFP32DistillationTrainer.validator_class = FixedShapeGrayValidator
        P3StudentFP32DistillationTrainer.loss_options = dict(
            auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
            p2_cls_weight=.20, p2_box_weight=.08, feature_weight=.08,
            p2_min_px=0.0, p2_max_px=16.0, feature_min_px=4.0, feature_max_px=8.0,
            tiny_multiplier=1.5, localization_weight=.08,
            localization_min_px=4.0, localization_max_px=960.0, temperature=2.0,
        )
        add_status_callback(student, args.run_dir, "student_distillation")
        status(args.run_dir, "student_distillation", epoch=0)
        student.train(
            trainer=P3StudentFP32DistillationTrainer,
            project=str(args.run_dir / "student"), name="p3",
            lr0=5e-5, warmup_epochs=0,
            **common_train(train_data, args, args.student_epochs),
        )
        student_weights = args.run_dir / "student/p3/weights"
        exports = export_student_pair(student_weights, args.run_dir, "cross_finetuned")
        (args.run_dir / "audit/student_exports.json").write_text(json.dumps(exports, indent=2) + "\n")
        student_best = args.run_dir / "p3_student_cross_finetuned_best_fp32.pt"
        assert_levels(student_best, 3)

        status(args.run_dir, "final_evaluation")
        final = {
            "teacher_video00009": clean_validate(teacher_best, train_data, args, args.run_dir / "audit/final_teacher_video00009.json"),
            "student_video00009": clean_validate(student_best, train_data, args, args.run_dir / "audit/final_student_video00009.json"),
            "teacher_cross_training_fit": clean_validate(teacher_best, cross_data, args, args.run_dir / "audit/final_teacher_cross.json"),
            "student_cross_training_fit": clean_validate(student_best, cross_data, args, args.run_dir / "audit/final_student_cross.json"),
        }
        result = dict(baseline=baseline, final=final, teacher=str(teacher_best), student=str(student_best))
        (args.run_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        status(args.run_dir, "complete", teacher=str(teacher_best), student=str(student_best))
    except BaseException as error:
        previous = json.loads((args.run_dir / "status.json").read_text()) if (args.run_dir / "status.json").exists() else {}
        status(args.run_dir, "failed", failed_stage=previous.get("stage"), error=repr(error))
        raise


if __name__ == "__main__":
    main()
