#!/usr/bin/env python3
"""Continue old52 frozen-Teacher/P3-Student training on the complete 52+21 schedule."""

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.prepare_fixedwing_finetune import fit_monitor, read_pinned_cache, write_split
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


LOSS_OPTIONS = dict(auxiliary_weight=.25, p3_cls_weight=.10, p3_dfl_weight=.05,
    p2_cls_weight=.25, p2_box_weight=.10, feature_weight=.10,
    p2_min_px=0., p2_max_px=16., feature_min_px=4., feature_max_px=8.,
    tiny_multiplier=1.75, localization_weight=.12,
    localization_min_px=4., localization_max_px=960., temperature=2.)


def verify_full_replay(old_rows, expanded_rows):
    """Every legacy slot and label, including duplicate exposures, must survive."""
    legacy = Counter(x["im_file"] for x in old_rows)
    current = Counter(x["im_file"] for x in expanded_rows)
    if any(current[path] != count for path, count in legacy.items()):
        raise ValueError("Legacy exposure multiplicity changed")
    by_path = {x["im_file"]: x for x in expanded_rows}
    for old in old_rows:
        new = by_path[old["im_file"]]
        if (tuple(old["shape"]) != tuple(new["shape"])
                or not np.array_equal(old["cls"], new["cls"])
                or not np.array_equal(old["bboxes"], new["bboxes"])):
            raise ValueError("Legacy cached label or image geometry changed")
    return dict(legacy_candidate_slots=len(old_rows),
        legacy_positive_slots=sum(bool(len(x["cls"])) for x in old_rows),
        complete_legacy_exposure_preserved=True, legacy_labels_exact=True)


def retainable(metrics, baseline):
    keys = ("native/c0.03/R", "native/c0.03/long_4to8px/R", "fitness")
    return (all(metrics.get(k) is not None and math.isfinite(metrics[k]) for k in keys)
        and metrics[keys[0]] >= baseline[keys[0]]-.03
        and metrics[keys[1]] >= baseline[keys[1]]-.05)


def choose_eligible(records, baseline):
    eligible = [(name, value) for name, value in records.items() if retainable(value, baseline)]
    return max(eligible, key=lambda pair: pair[1]["fitness"])[0] if eligible else None


def safe_metrics(metrics):
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in metrics.items()}


def prepare_data(source, old_data, output):
    full = yaml.safe_load((source/"train_online_gray_monitor.yaml").read_text())
    old = yaml.safe_load((old_data/"train_online_gray_monitor.yaml").read_text())
    expanded = read_pinned_cache(full["train"], full)
    original = read_pinned_cache(old["train"], old)
    audit = verify_full_replay(original["labels"], expanded["labels"])
    old_paths, full_paths = (Path(config["train"]).read_text().splitlines() for config in (old, full))
    if full_paths[:len(old_paths)] != old_paths:
        raise ValueError("Old candidate schedule prefix was not preserved exactly")
    m = json.loads((source/"manifest.json").read_text())
    if len(m["train_video_hashes"]) != 73 or len(m["latest_appended_videos"]) != 21:
        raise ValueError("Expected the same frozen 52+21 video scope")
    if {m["test_sha256"], m["validation_sha256"]} & set(m["train_video_hashes"]):
        raise ValueError("Held-out video occurs in training")
    if full.get("tiny_sampling", {}).get("policy") != "balanced_v2":
        raise ValueError("Do not restore the known-broken legacy sampler ordering")
    roots = {v["image_directory"] for v in m["latest_appended_videos"]}
    new = [x for x in expanded["labels"] if str(Path(x["im_file"]).parent) in roots]
    if len(new) != m["added_positive_samples"]+m["added_negative_samples"]:
        raise ValueError("New frozen-label frame coverage changed")
    native = read_pinned_cache(full["val"], full)
    if set(full_paths) & {x["im_file"] for x in native["labels"]}:
        raise ValueError("Native validation overlaps training")
    output.mkdir()
    val = output/"val.txt"
    pin = write_split(val, native["labels"]+fit_monitor(new), native["version"],
                      sha256(source/"annotation_snapshot.json"))
    full = dict(full, val=str(val), trusted_label_caches=dict(full["trusted_label_caches"], **{str(val): pin}))
    teacher_data, student_data = output/"teacher.yaml", output/"train.yaml"
    for path, offset in ((teacher_data, 42), (student_data, 57)):
        config = dict(full, online_replacement=dict(full["online_replacement"], epoch_offset=offset))
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    audit.update(source=str(source), source_snapshot_sha256=sha256(source/"annotation_snapshot.json"),
        source_manifest_sha256=sha256(source/"manifest.json"), new_video_count=21,
        training_video_hashes=m["train_video_hashes"], new_videos=m["latest_appended_videos"],
        new_positive=m["added_positive_samples"], new_reviewed_negative=m["added_negative_samples"],
        replay_positive=audit["legacy_positive_slots"], adaptation_roots=sorted(roots),
        native_epoch_positive=m["epoch_positive"], native_epoch_negative=m["epoch_negative"],
        sampler_audit=m["sampler_audit"], fit_monitor_frames=len(fit_monitor(new)),
        monitor_is_training_fit=True, old_source=str(old_data), old_source_train_sha256=sha256(old["train"]))
    dump(output/"manifest.json", audit)
    return teacher_data, student_data, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", default="2")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--teacher-epochs", type=int, default=6)
    p.add_argument("--student-epochs", type=int, default=8)
    a = p.parse_args()
    a.data_dir, a.old_run, a.run_dir = (getattr(a, k).resolve() for k in ("data_dir", "old_run", "run_dir"))
    if a.run_dir.exists() or not a.device.isdecimal() or (a.teacher_epochs, a.student_epochs) != (6, 8):
        raise ValueError("Use a fresh audited single-GPU 6+8 continuation")
    a.run_dir.mkdir(parents=True)
    (a.run_dir/"audit").mkdir()
    lock = (a.run_dir/"pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        dump(a.run_dir/"status.json", dict(stage=stage, pid=os.getpid(),
            time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra))
    try:
        used = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, used.strip().split(','))
        if memory > 256 or utilization > 5:
            raise RuntimeError("Selected GPU is not idle; no sharing/preemption attempted")
        status("verifying_full_legacy_replay")
        old_protocol = json.loads((a.old_run/"protocol.json").read_text())
        teacher_data, student_data, audit = prepare_data(a.data_dir, Path(old_protocol["data"]).parent, a.run_dir/"data")
        import torch
        from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
        from scripts.anti_uav.p3_student_distillation import P3StudentFP32DistillationTrainer, TinyAwareAddOnTrainer
        from scripts.anti_uav.run_p3_independent_curriculum_training import common_train, export_student_pair, validate
        from scripts.anti_uav.train_frozen_p3_addon_p2 import verify_legacy_outputs
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False,
            hub=False, mlflow=False, neptune=False, raytune=False))
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        teacher_weights = a.old_run/"teacher/p2p3/weights/best.pt"
        student_weights = a.old_run/"distill_fp32/p3_student/weights/best.pt"
        reference = a.old_run/"p3_student_distilled_best_fp32.pt"
        original_p3 = a.old_run/"foundation/p3/weights/best.pt"
        pins = {str(path): sha256(path) for path in (teacher_weights, student_weights, reference, original_p3)}
        teacher, student = YOLO(str(teacher_weights)), YOLO(str(student_weights))
        initial_checks = dict(teacher=verify_legacy_outputs(YOLO(str(original_p3)).model, teacher.model),
            student=verify_legacy_outputs(YOLO(str(reference)).model, student.model))
        if not hasattr(student.model, "tiny_feature_projection"):
            raise ValueError("Original trained feature projection is missing")
        dump(a.run_dir/"audit/initial_state_verification.json", initial_checks)
        roots = set(audit["adaptation_roots"])
        class MonitorValidator(FixedShapeGrayValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                self.group_stats["adaptation_fit"] = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

            def image_group(self, image):
                return "adaptation_fit" if str(Path(image).parent) in roots else super().image_group(image)

        class ContinuedStudentTrainer(P3StudentFP32DistillationTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                model = super().get_model(cfg, weights, verbose)
                if not hasattr(weights, "tiny_feature_projection"):
                    raise ValueError("Continuation must retain the trained feature projection")
                model.tiny_feature_projection.load_state_dict(weights.tiny_feature_projection.state_dict())
                verify_legacy_outputs(YOLO(str(reference)).model, model)
                return model

        protocol = dict(method_id="continued_frozen_teacher_student_v1",
            method="full legacy replay; continue old frozen-P3 Teacher P2; continue trained four-scale FP32 Student distillation",
            initial_teacher=str(teacher_weights), initial_student=str(reference),
            initial_student_sha256=sha256(reference), initial_four_scale_student=str(student_weights),
            initialization_pins=pins, data=str(student_data), dataset=audit,
            teacher_epochs=6, epochs=8, batch=a.batch, input_hw=[544, 960], seed=20261015,
            learning_rates=dict(teacher=1e-4, student=5e-5), warmup_epochs=0,
            distillation=True, auxiliary_p2=True, qat=False, loss_options=LOSS_OPTIONS,
            feature_projection_reinitialized=False, teacher_p3_and_bn_frozen=True,
            checkpoint_selection="original Video00009 fitness only; hard native Recall and tiny Recall eligibility; Video00004 remains test-only",
            previous_supervised_candidate=str(a.run_dir.parent/"old52_p3_fixedwing21_finetune_20261030_v2/p3_student_fixedwing_best_fp32.pt"),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        dump(a.run_dir/"protocol.json", protocol)
        status("baseline_evaluation")
        baseline = validate(reference, student_data, a, a.run_dir/"baseline.json", MonitorValidator)
        native_before = validate(teacher_weights, teacher_data, a, a.run_dir/"audit/baseline_teacher.json", MonitorValidator)
        def callback(stage):
            return lambda trainer: status(stage, epoch=trainer.epoch+1,
                fitness=float(trainer.fitness), old_recall_retained=retainable(
                    dict(trainer.metrics, fitness=float(trainer.fitness)),
                    native_before if stage.startswith("teacher") else baseline))
        TinyAwareAddOnTrainer.validator_class = MonitorValidator
        TinyAwareAddOnTrainer.tiny_multiplier = 1.75
        teacher.add_callback("on_fit_epoch_end", callback("teacher_finetune_frozen_p3"))
        settings = common_train(teacher_data, a, 6)
        settings.update(save_period=1)
        status("teacher_finetune_frozen_p3", epoch=0)
        teacher.train(trainer=TinyAwareAddOnTrainer, project=str(a.run_dir/"teacher"), name="p2p3",
            lr0=1e-4, warmup_epochs=0, **settings)
        teacher_directory = a.run_dir/"teacher/p2p3/weights"
        teacher_candidates = dict(best=teacher_directory/"best.pt",
            **{f"epoch{i+1}": teacher_directory/f"epoch{i}.pt" for i in range(6)})
        teacher_records = {name: validate(path, teacher_data, a, a.run_dir/f"audit/teacher_{name}.json", MonitorValidator)
                           for name, path in teacher_candidates.items()}
        teacher_choice = choose_eligible(teacher_records, native_before)
        teacher_best = teacher_candidates[teacher_choice] if teacher_choice else teacher_weights
        dump(a.run_dir/"audit/teacher_checkpoint_selection.json", dict(selected=teacher_choice or "original_teacher",
            weights=str(teacher_best), sha256=sha256(teacher_best),
            records={k: safe_metrics(v) for k,v in teacher_records.items()}))
        frozen = verify_legacy_outputs(YOLO(str(original_p3)).model, YOLO(str(teacher_best)).model)
        dump(a.run_dir/"audit/teacher_frozen_p3.json", frozen)
        teacher_after = validate(teacher_best, teacher_data, a, a.run_dir/"audit/final_teacher.json", MonitorValidator)
        ContinuedStudentTrainer.teacher_weights = teacher_best
        ContinuedStudentTrainer.validator_class = MonitorValidator
        ContinuedStudentTrainer.loss_options = dict(LOSS_OPTIONS)
        student.add_callback("on_fit_epoch_end", callback("student_distillation"))
        status("student_distillation", epoch=0)
        settings = common_train(student_data, a, 8)
        settings.update(save_period=1)
        student.train(trainer=ContinuedStudentTrainer, project=str(a.run_dir/"training"), name="p3",
            lr0=5e-5, warmup_epochs=0, **settings)
        status("strict_student_checkpoint_selection")
        weights = a.run_dir/"training/p3/weights"
        exports = export_student_pair(weights, a.run_dir, "continued_distilled")
        dump(a.run_dir/"audit/student_exports.json", exports)
        # Eligibility is a real selection constraint, not a constant score penalty.
        candidates = dict(best=weights/"best.pt", **{f"epoch{i+1}": weights/f"epoch{i}.pt" for i in range(8)})
        records = {name: validate(path, student_data, a, a.run_dir/f"audit/student_{name}.json", MonitorValidator)
                   for name, path in candidates.items()}
        selected = choose_eligible(records, baseline)
        eligible = selected is not None
        selected = selected or max(records, key=lambda name: records[name]["fitness"])
        from scripts.anti_uav.p3_student_distillation import export_pure_p3_student
        output = a.run_dir/"p3_student_fixedwing_best_fp32.pt"
        export_pure_p3_student(candidates[selected], output, ROOT/"ultralytics/cfg/models/v8/yolov8.yaml")
        final = validate(output, student_data, a, a.run_dir/"final.json", MonitorValidator)
        improved = final["adaptation_fit/c0.03/R"] > baseline["adaptation_fit/c0.03/R"]
        accepted = eligible and retainable(final, baseline) and improved
        dump(a.run_dir/"audit/student_checkpoint_selection.json", dict(selected=selected,
            eligible=eligible, records={k: safe_metrics(v) for k,v in records.items()},
            initial_student_is_fallback_not_overwritten=True))
        for path, digest in pins.items():
            if sha256(path) != digest:
                raise ValueError("Original checkpoint changed during continuation")
        dump(a.run_dir/"comparison.json", dict(baseline=safe_metrics(baseline), final=safe_metrics(final),
            teacher_baseline=safe_metrics(native_before), teacher_final=safe_metrics(teacher_after),
            candidate=str(output), sha256=sha256(output), adaptation_recall_improved=improved,
            ready_for_quantization_review=accepted, independent_fixedwing_test=False,
            strict_eligible_checkpoint_found=eligible))
        status("complete", candidate=str(output), ready_for_quantization_review=accepted,
            eligible_checkpoint_found=eligible)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
