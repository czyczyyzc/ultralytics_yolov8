#!/usr/bin/env python3
"""Train a P2 Teacher and deployable P3 Student without an old anti-UAV Student dependency."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
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
from ultralytics.data.dataset import DATASET_CACHE_VERSION, load_audited_dataset_cache  # noqa: E402
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
        if source.get("trusted_label_caches") and str(list_path) not in source["trusted_label_caches"]:
            raise ValueError(f"Missing audited cache pin for {split}: {list_path}")
        cache = load_audited_dataset_cache(cache_path, list_path, source)
        if cache is None:
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
    data["tiny_sampling"] = dict(data.get("tiny_sampling", {}), input_hw=[544, 960], tiny_min_px=4.0, tiny_max_px=8.0)
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
    gray_input = yaml.safe_load(data.read_text()).get("input_color", "source") == "gray3"
    return dict(
        data=str(data), imgsz=[544, 960], epochs=epochs, patience=patience,
        batch=args.batch, device=args.device, workers=args.workers, save=True, save_period=-1,
        pretrained=True, optimizer="AdamW", lrf=.1, momentum=.937, weight_decay=.0005,
        warmup_momentum=.8, warmup_bias_lr=.01, cos_lr=True, amp=True,
        deterministic=True, seed=SEED, nbs=128, rect=False, cache=False, val=True,
        plots=True, single_cls=True, mosaic=0., mixup=0., copy_paste=0., degrees=0.,
        translate=.05, scale=.2, shear=0., perspective=0., flipud=0., fliplr=.5,
        hsv_h=0. if gray_input else .015, hsv_s=0. if gray_input else .4,
        hsv_v=.3, close_mosaic=0, verbose=True, exist_ok=bool(getattr(args, "resume", False)),
    )


def stage_resume_state(stage_dir: Path, epochs: int, resume: bool) -> dict:
    """A best checkpoint is not a completion marker; resume partial stages from last."""
    result = dict(complete=False, completed_epochs=0, checkpoint=None)
    if not resume:
        return result
    csv_path = stage_dir / "results.csv"
    recorded = []
    if csv_path.is_file():
        with csv_path.open() as stream:
            recorded = [int(float(row["epoch"])) for row in csv.DictReader(stream, skipinitialspace=True)]
    if recorded != list(range(1, len(recorded) + 1)) or len(recorded) > epochs:
        raise ValueError(f"Invalid epoch history for {stage_dir}")
    last, best = stage_dir / "weights/last.pt", stage_dir / "weights/best.pt"
    if not last.is_file():
        if recorded or best.exists():
            raise ValueError(f"Missing last checkpoint for partial stage {stage_dir}")
        return result
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    saved_epoch = int(checkpoint.get("epoch", -1))
    if int(checkpoint["train_args"]["epochs"]) != epochs:
        raise ValueError(f"Saved epoch budget differs for {stage_dir}")
    stripped = saved_epoch == -1 and checkpoint.get("optimizer") is None
    if not stripped and saved_epoch + 1 != len(recorded):
        raise ValueError(f"Checkpoint and CSV epoch history differ for {stage_dir}")
    if len(recorded) == epochs:
        if not best.is_file():
            raise ValueError(f"Completed stage has no best checkpoint: {stage_dir}")
        return dict(result, complete=True, completed_epochs=len(recorded))
    if (saved_epoch < 0 or checkpoint.get("optimizer") is None
            or checkpoint.get("ema") is None):
        raise ValueError(f"Partial stage lacks resumable optimizer/EMA state: {stage_dir}")
    return dict(result, completed_epochs=len(recorded), checkpoint=str(last), checkpoint_sha256=sha256(last))


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
    def record_resume(trainer):
        if not trainer.resume:
            return
        value = dict(stage=stage, device=str(trainer.args.device),
                     checkpoint=str(trainer.args.resume), start_epoch_zero_based=trainer.start_epoch,
                     batch=trainer.batch_size, optimizer_state_entries=len(trainer.optimizer.state),
                     ema_updates=trainer.ema.updates if trainer.ema else None)
        (run_dir / f"audit/active_resume_{stage}.json").write_text(json.dumps(value, indent=2) + "\n")

    model.add_callback("on_train_start", record_resume)
    model.add_callback("on_fit_epoch_end", lambda trainer: write_status(
        run_dir, stage, epoch=trainer.epoch + 1, fitness=float(trainer.fitness), device=str(trainer.args.device),
        batches_per_epoch=len(trainer.train_loader), optimizer_updates=trainer.ema.updates,
    ))


def retain_matched_epoch(model: YOLO, run_dir: Path, epoch: int) -> None:
    def keep(trainer):
        if trainer.epoch + 1 != epoch:
            return
        directory = run_dir / "audit" / "matched_epoch"
        directory.mkdir(exist_ok=False)
        checkpoint = directory / "last.pt"
        temporary = directory / "last.pt.tmp"
        shutil.copy2(trainer.last, temporary)
        temporary.replace(checkpoint)
        receipt = dict(epoch=epoch, optimizer_updates=trainer.ema.updates,
                       batches=len(trainer.train_loader), checkpoint=str(checkpoint), sha256=sha256(checkpoint))
        temporary = directory / "snapshot.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(directory / "snapshot.json")

    model.add_callback("on_model_save", keep)


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
    parser.add_argument("--foundation-only", action="store_true", help="Validate the repaired foundation before training Teacher/Student")
    parser.add_argument("--foundation-compare-epoch", type=int, help="Retain an exact last.pt copy at a matched comparison epoch")
    parser.add_argument("--accepted-p3-gate", type=Path, help="Use the exact signed P3 checkpoint that passed the matched-step gate")
    parser.add_argument("--regression-repair", action="store_true", help="Opt-in source/scale-normalized regression; legacy defaults unchanged")
    p3_variant = parser.add_mutually_exclusive_group()
    p3_variant.add_argument("--p3-nearest-offset", action="store_true", help="Test P3-only off-grid tiny fallback and DFL reference shift")
    p3_variant.add_argument("--p3-signed-dfl", action="store_true", help="Test off-grid P3 fallback, signed DFL, and matched-only regression budget")
    parser.add_argument("--repair-balance-mix", type=float, default=.5)
    parser.add_argument("--repair-warmup-epochs", type=int, default=3)
    args = parser.parse_args()
    if args.foundation_compare_epoch is not None and not (args.foundation_only and
            1 <= args.foundation_compare_epoch <= args.foundation_epochs):
        parser.error("--foundation-compare-epoch requires an epoch within the foundation-only schedule")
    if (args.p3_nearest_offset or args.p3_signed_dfl) and not (
            args.foundation_only and args.regression_repair and args.localization_epochs == 0):
        parser.error("P3 variants require foundation-only regression repair with no localization stage")
    if args.accepted_p3_gate and (args.foundation_only or not args.regression_repair or args.localization_epochs
                                 or min(args.teacher_epochs, args.student_epochs) < 1):
        parser.error("An accepted P3 checkpoint requires the FP32 Teacher/Student regression-repair stages")

    p3_trainer, teacher_trainer, student_trainer = TinyAwareP3Trainer, TinyAwareAddOnTrainer, P3StudentFP32DistillationTrainer
    repair = None
    if args.regression_repair:
        from scripts.anti_uav.regression_repair_trainers import (
            RegressionRepairP3Trainer, RegressionRepairAddOnTrainer, RegressionRepairStudentTrainer,
        )
        if not 0 < args.repair_balance_mix <= .75 or args.repair_warmup_epochs < 1:
            raise ValueError("Invalid bounded regression repair schedule")
        repair = dict(balance_mix=args.repair_balance_mix, warmup_epochs=args.repair_warmup_epochs,
                      tiny_log_size_weight=.5)
        p3_trainer, teacher_trainer, student_trainer = RegressionRepairP3Trainer, RegressionRepairAddOnTrainer, RegressionRepairStudentTrainer
        for trainer in (p3_trainer, teacher_trainer, student_trainer):
            trainer.repair_options = repair

    args.data = args.data.resolve()
    args.foundation = args.foundation.resolve()
    args.run_dir = args.run_dir.resolve()
    if not args.data.is_file() or not args.foundation.is_file():
        raise FileNotFoundError("Training YAML or standard foundation checkpoint is missing")
    if "approved" in args.foundation.name.lower() or "student" in args.foundation.name.lower():
        raise ValueError("Foundation must not be a previous anti-UAV Student checkpoint")
    if min(args.foundation_epochs, args.teacher_epochs, args.student_epochs, args.localization_epochs) < 0:
        raise ValueError("Training epoch counts must be non-negative")
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
    accepted = None
    if args.accepted_p3_gate:
        gate_path = args.accepted_p3_gate.resolve()
        accepted = json.loads(gate_path.read_text())
        checkpoint = Path(accepted["candidate_checkpoint"])
        if (accepted["passed"] is not True or sha256(checkpoint) != accepted["candidate_sha256"]
                or sha256(args.data) != accepted["data_sha256"] or training_videos != accepted["training_videos"]
                or sha256(Path(accepted["comparison"])) != accepted["comparison_sha256"]
                or not 1 <= accepted["epoch"] <= args.foundation_epochs
                or not hasattr(YOLO(str(checkpoint)).model.model[-1], "p3_negative_bin")):
            raise ValueError("The signed P3 gate or approved checkpoint does not match the training protocol")

    offsets = 0
    foundation_data = stage_yaml(
        source, args.run_dir / "configs/foundation.yaml", offsets, args.replacement_probability
    )
    offsets += accepted["epoch"] if accepted else args.foundation_epochs
    teacher_data = stage_yaml(source, args.run_dir / "configs/teacher.yaml", offsets, args.replacement_probability)
    offsets += args.teacher_epochs
    student_data = stage_yaml(
        source, args.run_dir / "configs/student_fp32.yaml", offsets, args.replacement_probability
    )
    offsets += args.student_epochs
    localization_data = None
    if args.localization_epochs:
        localization_data = stage_yaml(
            source, args.run_dir / "configs/localization.yaml", offsets, args.replacement_probability
        )

    stages = [
        "standard YOLOv8n",
        "delivery-style P3 foundation",
        "frozen-P3 P2 Teacher",
        "FP32 P3 Student distillation",
    ]
    if args.localization_epochs:
        stages.append("FP32 localization refinement")
    if args.foundation_only:
        stages = stages[:2]
    elif accepted:
        stages = ["accepted signed-DFL P3 foundation", *stages[2:]]

    protocol = dict(
        method=" -> ".join(stages),
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
            flipud=0.0, fliplr=0.5,
            hsv_h=0.0 if source.get("input_color") == "gray3" else 0.015,
            hsv_s=0.0 if source.get("input_color") == "gray3" else 0.4, hsv_v=0.3,
        ),
        distillation=dict(
            student=dict(auxiliary=0.25, p3_cls=0.10, p3_dfl=0.05,
                         p2_cls=0.25, p2_box=0.10, feature=0.10),
            localization=dict(auxiliary=0.25, p3_cls=0.10, p3_dfl=0.05,
                              p2_cls=0.30, p2_box=0.15, feature=0.10,
                              localization_weight=0.15, localization_range_px=[4, 64]),
        ),
        qat=False,
        input_color=source.get("input_color", "source"),
        sampling_policy=source.get("tiny_sampling", {"policy": "legacy"}),
        execution_scope="foundation_only" if args.foundation_only else "full_curriculum",
        checkpoint_selection="Video00009: 0.5 F2@0.03 + 0.3 AP50 + 0.2 AP50-95",
        trusted_label_caches=source["trusted_label_caches"],
        final_graph="standard YOLOv8n P3-P5; all P2/projection modules removed",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    )
    if source.get("visible_box_guard"):
        protocol["augmentation"]["visible_box_guard"] = source["visible_box_guard"]
        if repair is None:
            raise ValueError("The protected box recipe requires the regression repair trainer audit")
    protocol_path = args.run_dir / "protocol.json"
    if args.foundation_compare_epoch is not None:
        protocol["foundation_compare_epoch"] = args.foundation_compare_epoch
    if accepted:
        protocol["accepted_p3_gate"] = dict(path=str(gate_path), sha256=sha256(gate_path),
            checkpoint=str(checkpoint), checkpoint_sha256=accepted["candidate_sha256"], foundation_epochs=accepted["epoch"])
        protocol["final_graph"] = "P3 signed DFL (-1..15), P4/P5 legacy DFL (0..15); training-only P2 removed"
    if repair is not None:
        protocol["regression_repair"] = dict(options=repair,
            source="per_image_gt_credit_and_augmented_source_scale_buckets",
            classification_unchanged=True, assigner_unchanged=True,
            inference_graph_unchanged=True, unmatched_gt_not_fabricated=True,
            extra_size_range_px=[2,16], localization_min_px=2,
            stages=["foundation", "teacher", "student"])
    if args.p3_nearest_offset:
        from scripts.anti_uav.p3_nearest_offset import FALLBACK_SCORE, OFFSET_LOSS_WEIGHT, TINY_MAX_PX

        protocol["p3_nearest_offset"] = dict(tiny_long_edge_max_px=TINY_MAX_PX, fallback_score=FALLBACK_SCORE,
            offset_loss_weight=OFFSET_LOSS_WEIGHT, preserve_dfl=True, preserve_p4_p5=True)
        protocol["regression_repair"].update(classification_unchanged=False, assigner_unchanged=False,
            inference_graph_unchanged=False)
        protocol["final_graph"] = "P3-P5 with a signed P3 DFL reference shift; no P2"
    if args.p3_signed_dfl:
        from scripts.anti_uav.p3_signed_dfl import FALLBACK_SCORE, MAX_BIN, MIN_BIN

        protocol["p3_signed_dfl"] = dict(bins=[MIN_BIN, MAX_BIN], fallback_score=FALLBACK_SCORE,
            eligible="no_interior_p3_grid_point", fallback="nearest_free_representable_point",
            normal_tal_unchanged=True, matched_gt_only_regression_budget=True, no_reference_offset=True)
        protocol["regression_repair"].update(classification_unchanged=False, assigner_unchanged=False,
            inference_graph_unchanged=False, stages=["foundation"])
        protocol["final_graph"] = "P3 signed DFL (-1..15), P4/P5 legacy DFL (0..15); no P2"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if any(previous.get(key) != protocol.get(key) for key in
               ("regression_repair", "p3_nearest_offset", "p3_signed_dfl", "foundation_compare_epoch", "accepted_p3_gate")):
            raise ValueError("Resume regression repair differs from saved experiment")
        for key in ("data", "foundation", "foundation_sha256", "training_videos", "epochs", "augmentation",
                    "batch", "nbs", "seed", "qat", "trusted_label_caches"):
            if previous.get(key) != protocol.get(key):
                raise ValueError(f"Resume protocol mismatch for {key}")
        if previous.get("input_color", "source") != protocol["input_color"]:
            raise ValueError("Resume input color mode differs from the saved experiment")
        if "sampling_policy" in previous and previous["sampling_policy"] != protocol["sampling_policy"]:
            raise ValueError("Resume protocol mismatch for sampling_policy")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    try:
        progress = {
            name: stage_resume_state(args.run_dir / relative, epochs, args.resume)
            for name, relative, epochs in (
                ("foundation", "foundation/p3", args.foundation_epochs),
                ("teacher", "teacher/p2p3", args.teacher_epochs),
                ("student", "distill_fp32/p3_student", args.student_epochs),
                ("localization", "localization/p3_student", args.localization_epochs),
            ) if epochs
        }
        if accepted:
            progress["foundation"] = dict(complete=True, completed_epochs=accepted["epoch"], checkpoint=str(checkpoint))
        ordered = list(progress.values())
        for index, state in enumerate(ordered):
            if state["completed_epochs"] and any(not earlier["complete"] for earlier in ordered[:index]):
                raise ValueError("Downstream checkpoints exist before the upstream stage completed")
        if args.resume:
            (args.run_dir / "audit/resume_plan.json").write_text(json.dumps(
                dict(device=args.device, stages=progress, optimizer_state_requested=True), indent=2
            ) + "\n")
        foundation_best = checkpoint if accepted else args.run_dir / "foundation/p3/weights/best.pt"
        if not progress["foundation"]["complete"]:
            p3_trainer.validator_class = FixedShapeGrayValidator
            p3_trainer.tiny_multiplier = 1.75
            p3_trainer.localization_weight = .12
            p3_trainer.localization_min_px = 4.0
            p3_trainer.localization_max_px = 64.0
            foundation = YOLO(progress["foundation"]["checkpoint"] or str(args.foundation))
            if args.p3_nearest_offset and not progress["foundation"]["checkpoint"]:
                foundation.model.yaml["p3_nearest_offset_v1"] = True
            if args.p3_signed_dfl and not progress["foundation"]["checkpoint"]:
                foundation.model.yaml["p3_signed_dfl_v1"] = True
            add_status_callback(foundation, args.run_dir, "train_native_p3_foundation")
            if args.foundation_compare_epoch is not None and args.foundation_compare_epoch > progress["foundation"]["completed_epochs"]:
                retain_matched_epoch(foundation, args.run_dir, args.foundation_compare_epoch)
            write_status(args.run_dir, "train_native_p3_foundation",
                         epoch=progress["foundation"]["completed_epochs"], device=args.device)
            foundation.train(
                trainer=p3_trainer, project=str(args.run_dir / "foundation"), name="p3",
                lr0=.001, warmup_epochs=1, resume=bool(progress["foundation"]["checkpoint"]),
                **common_train(foundation_data, args, args.foundation_epochs),
            )

        if args.foundation_only:
            training_status = json.loads((args.run_dir / "status.json").read_text())
            metrics_path = args.run_dir / "audit/video00009_foundation_p3.json"
            metrics = validate(foundation_best, foundation_data, args, metrics_path)
            write_status(args.run_dir, "foundation_complete", foundation_best=str(foundation_best),
                         metrics=str(metrics_path), fitness=metrics["fitness"], teacher_student_started=False,
                         epoch=training_status.get("epoch"), batches_per_epoch=training_status.get("batches_per_epoch"),
                         optimizer_updates=training_status.get("optimizer_updates"))
            return

        teacher_best = args.run_dir / "teacher/p2p3/weights/best.pt"
        if not progress["teacher"]["complete"]:
            write_status(args.run_dir, "initialize_teacher")
            if progress["teacher"]["checkpoint"]:
                teacher = YOLO(progress["teacher"]["checkpoint"])
            else:
                teacher_init = args.run_dir / "teacher_initialized.pt"
                teacher, report = initialize_addon_model(foundation_best, DEFAULT_CFG, teacher_init)
                (args.run_dir / "audit/teacher_initialization.json").write_text(json.dumps(report, indent=2) + "\n")
            teacher_trainer.validator_class = FixedShapeGrayValidator
            teacher_trainer.tiny_multiplier = 1.75
            add_status_callback(teacher, args.run_dir, "train_teacher")
            teacher.train(
                trainer=teacher_trainer, project=str(args.run_dir / "teacher"), name="p2p3",
                lr0=.001, warmup_epochs=1, resume=bool(progress["teacher"]["checkpoint"]),
                **common_train(teacher_data, args, args.teacher_epochs),
            )
        frozen = verify_legacy_outputs(YOLO(str(foundation_best)).model, YOLO(str(teacher_best)).model)
        (args.run_dir / "audit/teacher_frozen_p3.json").write_text(json.dumps(frozen, indent=2) + "\n")

        student_weights = args.run_dir / "distill_fp32/p3_student/weights"
        if not progress["student"]["complete"]:
            write_status(args.run_dir, "initialize_student")
            if progress["student"]["checkpoint"]:
                student = YOLO(progress["student"]["checkpoint"])
            else:
                student_init = args.run_dir / "student_four_scale_initialized.pt"
                student, report = initialize_addon_model(foundation_best, DEFAULT_CFG, student_init)
                report["teacher_auxiliary_copy"] = copy_teacher_auxiliary(YOLO(str(teacher_best)).model, student.model)
                student.model.model[-1].auxiliary_training_only = True
                student.save(student_init)
                report["post_copy_legacy_regression"] = verify_legacy_outputs(
                    YOLO(str(foundation_best)).model, student.model
                )
                (args.run_dir / "audit/student_initialization.json").write_text(json.dumps(report, indent=2) + "\n")

            student_trainer.teacher_weights = teacher_best.resolve()
            student_trainer.validator_class = FixedShapeGrayValidator
            student_trainer.loss_options = dict(
                auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
                p2_cls_weight=.25, p2_box_weight=.10, feature_weight=.10,
                p2_min_px=0.0, p2_max_px=16.0, feature_min_px=4.0, feature_max_px=8.0,
                tiny_multiplier=1.75, localization_weight=.12,
                localization_min_px=4.0, localization_max_px=960.0, temperature=2.0,
            )
            add_status_callback(student, args.run_dir, "distill_fp32")
            student.train(
                trainer=student_trainer,
                project=str(args.run_dir / "distill_fp32"), name="p3_student",
                lr0=1e-4, warmup_epochs=0, resume=bool(progress["student"]["checkpoint"]),
                **common_train(student_data, args, args.student_epochs),
            )
        if not (args.resume and (args.run_dir / "audit/student_exports.json").is_file()
                and all((args.run_dir / f"p3_student_distilled_{key}_fp32.pt").is_file()
                        for key in ("best", "last"))):
            exports = export_student_pair(student_weights, args.run_dir, "distilled")
            (args.run_dir / "audit/student_exports.json").write_text(json.dumps(exports, indent=2) + "\n")

        if args.localization_epochs:
            localization_weights = args.run_dir / "localization/p3_student/weights"
            if not progress["localization"]["complete"]:
                localization = YOLO(progress["localization"]["checkpoint"] or str(student_weights / "best.pt"))
                student_trainer.teacher_weights = teacher_best.resolve()
                student_trainer.loss_options = dict(
                    auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
                    p2_cls_weight=.30, p2_box_weight=.15, feature_weight=.10,
                    p2_min_px=0.0, p2_max_px=16.0, feature_min_px=4.0, feature_max_px=16.0,
                    tiny_multiplier=1.85, localization_weight=.15,
                    localization_min_px=4.0, localization_max_px=64.0, temperature=2.0,
                )
                add_status_callback(localization, args.run_dir, "localization_refinement")
                localization.train(
                    trainer=student_trainer,
                    project=str(args.run_dir / "localization"), name="p3_student",
                    lr0=5e-5, warmup_epochs=0, resume=bool(progress["localization"]["checkpoint"]),
                    **common_train(localization_data, args, args.localization_epochs),
                )
            if not (args.resume and (args.run_dir / "audit/localization_exports.json").is_file()
                    and all((args.run_dir / f"p3_student_localization_{key}_fp32.pt").is_file()
                            for key in ("best", "last"))):
                exports = export_student_pair(localization_weights, args.run_dir, "localization")
                (args.run_dir / "audit/localization_exports.json").write_text(
                    json.dumps(exports, indent=2) + "\n"
                )

        write_status(args.run_dir, "compare_video00009")
        models = {
            "foundation_p3": foundation_best,
            "teacher_p2p3": teacher_best,
            "student_fp32": args.run_dir / "p3_student_distilled_best_fp32.pt",
        }
        if args.localization_epochs:
            models["student_localization_fp32"] = args.run_dir / "p3_student_localization_best_fp32.pt"
        evaluation_data = localization_data or student_data
        comparison = {
            name: validate(path, evaluation_data, args, args.run_dir / f"audit/video00009_{name}.json")
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
