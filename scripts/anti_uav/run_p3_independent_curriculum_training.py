#!/usr/bin/env python3
"""Train a P2 Teacher and deployable P3 Student without an old anti-UAV Student dependency."""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.gray_deployment_trainer import (  # noqa: E402
    FixedShapeGrayValidator,
)
from scripts.anti_uav.p3_student_distillation import (  # noqa: E402
    P3StudentFP32DistillationTrainer,
    TinyAwareAddOnTrainer,
    TinyAwareP3Trainer,
    copy_teacher_auxiliary,
    export_pure_p3_student,
)
from scripts.anti_uav.train_frozen_p3_addon_p2 import (  # noqa: E402
    DEFAULT_CFG,
    initialize_addon_model,
    verify_legacy_outputs,
)
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.dataset import DATASET_CACHE_VERSION  # noqa: E402
from ultralytics.data.utils import load_dataset_cache_file  # noqa: E402
from ultralytics.utils import SETTINGS  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402


P3_CFG = ROOT / "ultralytics/cfg/models/v8/yolov8.yaml"
SEED = 20261015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_status(run_dir: Path, stage: str, **extra) -> None:
    value = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    temporary = run_dir / "status.tmp"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(run_dir / "status.json")
    print(json.dumps(value), flush=True)


def audit_label_caches(source: dict) -> dict:
    """Pin train/validation cache snapshots so NFS startup does not stat every image and label."""
    records = {}
    for split in ("train", "val"):
        list_path = Path(source[split]).resolve()
        if not list_path.is_file():
            raise ValueError(f"Audited cache mode requires a file list for {split}: {list_path}")
        cache_path = list_path.with_suffix(".cache")
        if not cache_path.is_file():
            raise FileNotFoundError(f"Missing {split} label cache: {cache_path}")
        cache = load_dataset_cache_file(cache_path)
        if cache.get("version") != DATASET_CACHE_VERSION:
            raise ValueError(f"Stale {split} label cache version: {cache_path}")
        results = cache.get("results")
        if not results or len(results) != 5:
            raise ValueError(f"Invalid {split} label cache results: {cache_path}")
        records[str(list_path)] = {
            "path": str(cache_path),
            "source_sha256": sha256(list_path),
            "cache_sha256": sha256(cache_path),
            "dataset_hash": str(cache["hash"]),
            "entries": int(results[4]),
        }
    return records


def _scale_slots(config: dict | None) -> tuple[int, int]:
    if not config:
        return 0, 0
    donors = json.loads(Path(config["donor_manifest"]).read_text())["donors"]
    return len(donors), int(config.get("views_per_donor", 0))


def stage_yaml(
    source: dict,
    output: Path,
    epoch_offset: int,
    replacement_probability: float = 0.5,
) -> Path:
    """Build a delivery-style stage with replacement but no appended scale views."""
    data = deepcopy(source)
    data["tiny_sampling"] = dict(input_hw=[544, 960], tiny_min_px=4.0, tiny_max_px=8.0)
    replacement = data.get("online_replacement")
    if not replacement or not 0.0 <= replacement_probability <= 1.0:
        raise ValueError("Delivery-style training requires a valid online replacement configuration")
    data["online_replacement"] = dict(
        replacement,
        replacement_probability=float(replacement_probability),
        epoch_offset=int(replacement.get("epoch_offset", 0)) + epoch_offset,
    )

    scale = data.pop("online_scale", None)
    donors, original_views = _scale_slots(scale)
    if donors and data.get("label_sampling"):
        sampling = dict(data["label_sampling"])
        sampling["anchor_slots"] = int(sampling["anchor_slots"]) - donors * original_views
        if sampling["anchor_slots"] <= 0:
            raise ValueError("Removing scale views produced an invalid anchor count")
        data["label_sampling"] = sampling

    output.write_text(yaml.safe_dump(data, sort_keys=False))
    return output


def common_train(data: Path, args, epochs: int, patience: int = 0) -> dict:
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


def validate(
    model_path: Path,
    data: Path,
    args,
    output: Path,
    validator_class=FixedShapeGrayValidator,
) -> dict:
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data), validator=validator_class, imgsz=[544, 960], device=args.device,
        batch=min(args.batch, 64), workers=args.workers, rect=False, conf=.001, iou=.45,
        max_det=100, plots=False, project=str(output.parent / "validation"), name=output.stem,
    )
    result = dict(metrics.gray_selection)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def add_status_callback(model: YOLO, run_dir: Path, stage: str) -> None:
    model.add_callback("on_fit_epoch_end", lambda trainer: write_status(
        run_dir, stage, epoch=trainer.epoch + 1, fitness=float(trainer.fitness)
    ))


def export_student_pair(source_dir: Path, run_dir: Path, prefix: str) -> dict:
    exports = {}
    for checkpoint in ("best", "last"):
        output = run_dir / f"p3_student_{prefix}_{checkpoint}_fp32.pt"
        exports[checkpoint] = export_pure_p3_student(
            source_dir / f"{checkpoint}.pt", output, P3_CFG
        )
    return exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--foundation", type=Path, required=True,
                        help="Standard public YOLOv8n checkpoint, not a task-specific Student")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="2")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--foundation-epochs", type=int, default=30)
    parser.add_argument("--teacher-epochs", type=int, default=12)
    parser.add_argument("--student-epochs", type=int, default=15)
    parser.add_argument("--localization-epochs", type=int, default=10)
    parser.add_argument("--replacement-probability", type=float, default=0.5)
    parser.add_argument("--expected-training-videos", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.foundation = args.foundation.resolve()
    args.run_dir = args.run_dir.resolve()
    if not args.data.is_file() or not args.foundation.is_file():
        raise FileNotFoundError("Training YAML or standard foundation checkpoint is missing")
    if "approved" in args.foundation.name.lower() or "student" in args.foundation.name.lower():
        raise ValueError("Foundation must not be a previous anti-UAV Student checkpoint")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.run_dir / "protocol.json").exists() and not args.resume:
        raise FileExistsError("Refuse to overwrite an existing curriculum experiment")
    (args.run_dir / "configs").mkdir(exist_ok=True)
    (args.run_dir / "audit").mkdir(exist_ok=True)

    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    init_seeds(SEED, deterministic=True)
    source = yaml.safe_load(args.data.read_text())
    if not source.get("label_sampling") or not source.get("online_replacement"):
        raise ValueError("Training requires audited negatives and replacement assets")
    train_paths = Path(source["train"]).read_text().splitlines()
    val_paths = Path(source["val"]).read_text().splitlines()
    if set(train_paths) & set(val_paths):
        raise ValueError("Training and validation paths overlap")
    if any("Video00004" in Path(path).parts for path in train_paths + val_paths):
        raise ValueError("Video00004 must remain test-only")

    source["trusted_label_caches"] = audit_label_caches(source)
    (args.run_dir / "audit/label_caches.json").write_text(
        json.dumps(source["trusted_label_caches"], indent=2) + "\n"
    )

    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    training_videos = len(manifest.get("train_video_hashes", []))
    if training_videos != args.expected_training_videos:
        raise ValueError(f"Expected {args.expected_training_videos} training videos, found {training_videos}")

    offsets = 0
    foundation_data = stage_yaml(
        source, args.run_dir / "configs/foundation.yaml", offsets, args.replacement_probability
    )
    offsets += args.foundation_epochs
    teacher_data = stage_yaml(source, args.run_dir / "configs/teacher.yaml", offsets, args.replacement_probability)
    offsets += args.teacher_epochs
    student_data = stage_yaml(
        source, args.run_dir / "configs/student_fp32.yaml", offsets, args.replacement_probability
    )
    offsets += args.student_epochs
    localization_data = stage_yaml(
        source, args.run_dir / "configs/localization.yaml", offsets, args.replacement_probability
    )

    protocol = dict(
        method="standard YOLOv8n -> delivery-style P3 foundation -> frozen-P3 P2 Teacher -> "
               "FP32 P3 Student distillation -> FP32 localization refinement",
        data=str(args.data), foundation=str(args.foundation), foundation_sha256=sha256(args.foundation),
        task_specific_initialization=False, input_hw=[544, 960], training_videos=training_videos,
        batch=args.batch, nbs=128, seed=SEED,
        epochs=dict(foundation=args.foundation_epochs, teacher=args.teacher_epochs,
                    student=args.student_epochs, localization=args.localization_epochs),
        augmentation=dict(
            online_replacement_probability=args.replacement_probability,
            online_scale_views=0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
            translate=0.05, scale=0.2, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        ),
        distillation=dict(
            student=dict(auxiliary=0.25, p3_cls=0.10, p3_dfl=0.05,
                         p2_cls=0.25, p2_box=0.10, feature=0.10),
            localization=dict(auxiliary=0.25, p3_cls=0.10, p3_dfl=0.05,
                              p2_cls=0.30, p2_box=0.15, feature=0.10,
                              localization_weight=0.15, localization_range_px=[4, 64]),
        ),
        qat=False,
        checkpoint_selection="Video00009: 0.5 F2@0.03 + 0.3 AP50 + 0.2 AP50-95",
        trusted_label_caches=source["trusted_label_caches"],
        final_graph="standard YOLOv8n P3-P5; all P2/projection modules removed",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    protocol_path = args.run_dir / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        for key in ("data", "foundation", "foundation_sha256", "training_videos", "epochs", "augmentation"):
            if previous.get(key) != protocol.get(key):
                raise ValueError(f"Resume protocol mismatch for {key}")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    try:
        foundation_best = args.run_dir / "foundation/p3/weights/best.pt"
        if not (args.resume and foundation_best.is_file()):
            TinyAwareP3Trainer.validator_class = FixedShapeGrayValidator
            TinyAwareP3Trainer.tiny_multiplier = 1.75
            TinyAwareP3Trainer.localization_weight = .12
            TinyAwareP3Trainer.localization_min_px = 4.0
            TinyAwareP3Trainer.localization_max_px = 64.0
            foundation = YOLO(str(args.foundation))
            add_status_callback(foundation, args.run_dir, "train_native_p3_foundation")
            write_status(args.run_dir, "train_native_p3_foundation", epoch=0)
            foundation.train(
                trainer=TinyAwareP3Trainer, project=str(args.run_dir / "foundation"), name="p3",
                lr0=.001, warmup_epochs=1,
                **common_train(foundation_data, args, args.foundation_epochs),
            )

        teacher_best = args.run_dir / "teacher/p2p3/weights/best.pt"
        if not (args.resume and teacher_best.is_file()):
            write_status(args.run_dir, "initialize_teacher")
            teacher_init = args.run_dir / "teacher_initialized.pt"
            teacher, report = initialize_addon_model(foundation_best, DEFAULT_CFG, teacher_init)
            (args.run_dir / "audit/teacher_initialization.json").write_text(json.dumps(report, indent=2) + "\n")
            TinyAwareAddOnTrainer.validator_class = FixedShapeGrayValidator
            TinyAwareAddOnTrainer.tiny_multiplier = 1.75
            add_status_callback(teacher, args.run_dir, "train_teacher")
            teacher.train(
                trainer=TinyAwareAddOnTrainer, project=str(args.run_dir / "teacher"), name="p2p3",
                lr0=.001, warmup_epochs=1,
                **common_train(teacher_data, args, args.teacher_epochs),
            )
            frozen = verify_legacy_outputs(YOLO(str(foundation_best)).model, YOLO(str(teacher_best)).model)
            (args.run_dir / "audit/teacher_frozen_p3.json").write_text(json.dumps(frozen, indent=2) + "\n")

        student_weights = args.run_dir / "distill_fp32/p3_student/weights"
        if not (args.resume and (student_weights / "best.pt").is_file()):
            write_status(args.run_dir, "initialize_student")
            student_init = args.run_dir / "student_four_scale_initialized.pt"
            student, report = initialize_addon_model(foundation_best, DEFAULT_CFG, student_init)
            report["teacher_auxiliary_copy"] = copy_teacher_auxiliary(YOLO(str(teacher_best)).model, student.model)
            student.model.model[-1].auxiliary_training_only = True
            student.save(student_init)
            report["post_copy_legacy_regression"] = verify_legacy_outputs(
                YOLO(str(foundation_best)).model, student.model
            )
            (args.run_dir / "audit/student_initialization.json").write_text(json.dumps(report, indent=2) + "\n")

            P3StudentFP32DistillationTrainer.teacher_weights = teacher_best.resolve()
            P3StudentFP32DistillationTrainer.validator_class = FixedShapeGrayValidator
            P3StudentFP32DistillationTrainer.loss_options = dict(
                auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
                p2_cls_weight=.25, p2_box_weight=.10, feature_weight=.10,
                p2_min_px=0.0, p2_max_px=16.0, feature_min_px=4.0, feature_max_px=8.0,
                tiny_multiplier=1.75, localization_weight=.12,
                localization_min_px=4.0, localization_max_px=960.0, temperature=2.0,
            )
            add_status_callback(student, args.run_dir, "distill_fp32")
            student.train(
                trainer=P3StudentFP32DistillationTrainer,
                project=str(args.run_dir / "distill_fp32"), name="p3_student",
                lr0=1e-4, warmup_epochs=0,
                **common_train(student_data, args, args.student_epochs),
            )
            exports = export_student_pair(student_weights, args.run_dir, "distilled")
            (args.run_dir / "audit/student_exports.json").write_text(json.dumps(exports, indent=2) + "\n")

        localization_weights = args.run_dir / "localization/p3_student/weights"
        if not (args.resume and (localization_weights / "best.pt").is_file()):
            localization = YOLO(str(student_weights / "best.pt"))
            P3StudentFP32DistillationTrainer.teacher_weights = teacher_best.resolve()
            P3StudentFP32DistillationTrainer.loss_options = dict(
                auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
                p2_cls_weight=.30, p2_box_weight=.15, feature_weight=.10,
                p2_min_px=0.0, p2_max_px=16.0, feature_min_px=4.0, feature_max_px=16.0,
                tiny_multiplier=1.85, localization_weight=.15,
                localization_min_px=4.0, localization_max_px=64.0, temperature=2.0,
            )
            add_status_callback(localization, args.run_dir, "localization_refinement")
            localization.train(
                trainer=P3StudentFP32DistillationTrainer,
                project=str(args.run_dir / "localization"), name="p3_student",
                lr0=5e-5, warmup_epochs=0,
                **common_train(localization_data, args, args.localization_epochs),
            )
            exports = export_student_pair(localization_weights, args.run_dir, "localization")
            (args.run_dir / "audit/localization_exports.json").write_text(json.dumps(exports, indent=2) + "\n")

        write_status(args.run_dir, "compare_video00009")
        models = {
            "foundation_p3": foundation_best,
            "teacher_p2p3": teacher_best,
            "student_fp32": args.run_dir / "p3_student_distilled_best_fp32.pt",
            "student_localization_fp32": args.run_dir / "p3_student_localization_best_fp32.pt",
        }
        comparison = {
            name: validate(path, localization_data, args, args.run_dir / f"audit/video00009_{name}.json")
            for name, path in models.items()
        }
        (args.run_dir / "comparison_video00009.json").write_text(json.dumps(comparison, indent=2) + "\n")
        fp32_candidates = {key: value for key, value in comparison.items() if key.endswith("fp32")}
        selected_fp32 = max(fp32_candidates, key=lambda key: fp32_candidates[key]["fitness"])
        summary = dict(
            best_fp32=selected_fp32,
            best_fp32_path=str(models[selected_fp32]),
            qat_candidate="",
            qat_is_global_best=False,
        )
        (args.run_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_status(args.run_dir, "complete", **summary)
    except Exception as error:
        previous = json.loads((args.run_dir / "status.json").read_text()) if (args.run_dir / "status.json").exists() else {}
        write_status(args.run_dir, "failed", failed_stage=previous.get("stage"), error=str(error))
        raise


if __name__ == "__main__":
    main()
