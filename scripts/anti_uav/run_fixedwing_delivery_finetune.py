#!/usr/bin/env python3
"""Urgently adapt the delivered pure P3 Student, independently of public retraining."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.prepare_fixedwing_finetune import prepare
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256


def selection_score(stats, baseline=None):
    prefix = "adaptation_fit/"
    fit = .5*stats[prefix+"c0.03/F2"]+.3*stats[prefix+"mAP50"]+.2*stats[prefix+"mAP50-95"]
    retained = (baseline is None or (stats["native/c0.03/R"] >= baseline["native/c0.03/R"]-.03
        and stats["native/c0.03/long_4to8px/R"] >= baseline["native/c0.03/long_4to8px/R"]-.05))
    return .5*stats["fitness"]+.5*fit-(0 if retained else 1), retained


def finite_metrics(stats):
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in stats.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    for key in ("data_dir", "student", "run_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.run_dir.exists() or min(args.epochs, args.batch, args.workers) < 1 or not args.device.isdecimal():
        raise ValueError("Use a fresh run and a single physical GPU")
    if not 0 < args.lr <= 1e-4:
        raise ValueError("Urgent fine-tuning uses a conservative learning rate <=1e-4")
    args.run_dir.mkdir(parents=True)
    lock = (args.run_dir/"pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        dump(args.run_dir/"status.json", dict(stage=stage, pid=os.getpid(),
            time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra))
    try:
        usage = subprocess.check_output(["nvidia-smi", f"--id={args.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, usage.strip().split(','))
        if memory > 256 or utilization > 5:
            raise RuntimeError(f"GPU {args.device} is not idle; refuse to share/preempt: {usage}")
        status("preparing_frozen_adaptation_schedule")
        data, audit = prepare(args.data_dir, args.run_dir/"data")
        import torch
        from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
        from scripts.anti_uav.p3_student_distillation import TinyAwareP3Trainer
        from scripts.anti_uav.run_p3_independent_curriculum_training import common_train, validate
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False,
            hub=False, mlflow=False, neptune=False, raytune=False))
        torch.set_num_threads(4)
        model = YOLO(str(args.student))
        if len(model.model.model[-1].stride) != 3:
            raise ValueError("Initialize only from the delivered pure three-scale P3 Student")
        class AdaptationValidator(FixedShapeGrayValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                self.group_stats["adaptation_fit"] = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

            def image_group(self, image):
                if str(Path(image).parent) in roots:
                    return "adaptation_fit"
                return super().image_group(image)

            def get_stats(self):
                stats = super().get_stats()
                stats["validation_fitness"] = stats["fitness"]
                score, retained = selection_score(stats, baseline)
                stats["fitness"], stats["old_recall_retained"] = score, int(retained)
                self.metrics.gray_selection = dict(stats)
                return stats

        roots, baseline = set(audit["adaptation_roots"]), None
        status("baseline_evaluation")
        baseline = validate(args.student, data, args, args.run_dir/"baseline.json", AdaptationValidator)
        protocol = dict(method="supervised pure-P3 fine-tune from delivered old52 Student + stratified old replay",
            initial_student=str(args.student), initial_student_sha256=sha256(args.student),
            data=str(data), dataset=audit, input_hw=[544, 960], epochs=args.epochs, batch=args.batch,
            learning_rate=args.lr, qat=False, auxiliary_p2=False, distillation=False,
            candidate_selection="equal native Video00009 validation and new-video adaptation-fit fitness; penalize native Recall loss>3pp or tiny Recall loss>5pp",
            adaptation_fit_is_independent_test=False, inference_graph="unchanged pure P3-P5",
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        dump(args.run_dir/"protocol.json", protocol)
        TinyAwareP3Trainer.validator_class = AdaptationValidator
        TinyAwareP3Trainer.tiny_multiplier = 1.75
        TinyAwareP3Trainer.localization_weight = 0.
        model.add_callback("on_fit_epoch_end", lambda trainer: status("finetuning_p3",
            epoch=trainer.epoch+1, epochs=args.epochs, fitness=float(trainer.fitness)))
        training = common_train(data, args, args.epochs)
        training.update(save_period=1, seed=20261030)
        status("finetuning_p3", epoch=0, epochs=args.epochs)
        model.train(trainer=TinyAwareP3Trainer, project=str(args.run_dir/"training"), name="p3",
            lr0=args.lr, warmup_epochs=.5, **training)
        best = args.run_dir/"training/p3/weights/best.pt"
        output = args.run_dir/"p3_student_fixedwing_best_fp32.pt"
        shutil.copy2(best, output)
        status("final_evaluation")
        final = validate(output, data, args, args.run_dir/"final.json", AdaptationValidator)
        improved = final["adaptation_fit/c0.03/R"] > baseline["adaptation_fit/c0.03/R"]
        accepted = bool(final["old_recall_retained"]) and improved
        dump(args.run_dir/"comparison.json", dict(baseline=finite_metrics(baseline), final=finite_metrics(final),
            candidate=str(output), sha256=sha256(output), adaptation_recall_improved=improved,
            ready_for_quantization_review=accepted, independent_fixedwing_test=False))
        status("complete", candidate=str(output), ready_for_quantization_review=accepted)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
