#!/usr/bin/env python3
"""Prepare all reviewed new frames, train two controlled arms, then compare detectors."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = Path("/mnt/andrew/anti_uav_model_refinement/data")
INITIAL = ROOT / "runs/anti_uav/real_gray_yolov8n_strict_holdout_Video00004_newclips01_20260902/training/strict_holdout_Video00004_neg15_newclips01_v1_20260902/weights/best.pt"
OLD = ROOT / "runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_manual_clips0123_20260904"
HOLDOUT = DATA / "real_gray_yolo_lovo_positive_mixed_v1_20260828/folds/holdout_Video00004/holdout_all.yaml"


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_and_report(run, baseline, expanded, device):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    models = {}
    for arm in ("baseline_14", "expanded_28"):
        for branch, relative in (("p3", "training_p3/p3"), ("addon", "training_addon/p2")):
            models[f"{arm}_{branch}"] = run / arm / relative / "weights/best.pt"
    models["deployment_old_p3"] = next((OLD / "training_p3").glob("*/weights/best.pt"))
    models["deployment_old_addon"] = OLD / "training_addon/final/weights/best.pt"
    output = run / "evaluation"
    output.mkdir(exist_ok=True)
    results = {}
    for split, data in (("gray_validation", baseline / "train_hardneg_gray_monitor.yaml"), ("Video00004", HOLDOUT)):
        for name, weights in models.items():
            status(run, "evaluation", split=split, model=name)
            metrics = YOLO(str(weights)).val(data=str(data), imgsz=[544, 960], rect=False,
                        validator=FixedShapeGrayValidator, device=device, batch=32, workers=4,
                        conf=.001, iou=.45, max_det=100, half=False, plots=False,
                        project=str(output / "val"), name=f"{split}_{name}", verbose=False)
            value = dict(weights=str(weights), split=split, metrics=clean(metrics.gray_selection))
            results[f"{split}_{name}"] = value
            (output / f"{split}_{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    lines = ["# Native Data Expansion: Controlled Detector Comparison", "",
             "P3 15 epochs, then frozen-P3 add-on P2 15 epochs per arm. Same initialization,",
             "seed, base augmentation and fixed 960x544 validation/selection policy.",
             "More data means more optimizer steps at matched epochs; compute is not matched.",
             "PT FP32 detector results, not RKNN INT8, tracker metrics or board FPS.",
             "Old deployment weights are a practical reference, not the controlled data-only baseline.",
             "Video00004 is test-only; no checkpoint selection or mining uses this test.", ""]
    for split in ("gray_validation", "Video00004"):
        lines += [f"## {split}", "", "| Model | Conf | P | R | FP | mAP50 | 4-8px R |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for conf in (.01, .03, .05):
            for name in models:
                m = results[f"{split}_{name}"]["metrics"]
                key = f"native/c{conf:.2f}"
                tiny = m[key + "/long_4to8px/R"]
                tiny_text = "N/A" if tiny is None else f"{tiny*100:.2f}%"
                lines.append(f"| {name} | {conf:.2f} | {m[key+'/P']*100:.2f}% | {m[key+'/R']*100:.2f}% | "
                             f"{m[key+'/FP']} | {m['native/mAP50']*100:.2f}% | {tiny_text} |")
        lines.append("")
    deltas = {}
    for split in ("gray_validation", "Video00004"):
        for branch in ("p3", "addon"):
            before = results[f"{split}_baseline_14_{branch}"]["metrics"]
            after = results[f"{split}_expanded_28_{branch}"]["metrics"]
            deltas[f"{split}_{branch}"] = {k: after[k]-before[k] for k in before
                                           if before[k] is not None and after.get(k) is not None}
    (run / "comparison.json").write_text(json.dumps(dict(results=results, deltas=deltas), indent=2, allow_nan=False) + "\n")
    (run / "COMPARISON.md").write_text("\n".join(lines) + "\n")


def status(run, stage, **extra):
    value = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    tmp = run / "status.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(run / "status.json")
    print(json.dumps(value), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--expanded", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--device", default="6")
    p.add_argument("--epochs", type=int, default=15)
    a = p.parse_args()
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "protocol.json").exists():
        raise FileExistsError("Do not overwrite an existing paired experiment")
    (run / "logs").mkdir(exist_ok=True)
    (run / "yolo_config").mkdir(exist_ok=True)
    font = ROOT / "runs/anti_uav/frozen_p3_addon_p2_all398_online_scale_20260916/yolo_config/Arial.ttf"
    if font.is_file():
        shutil.copy2(font, run / "yolo_config/Arial.ttf")
    env = dict(os.environ, PYTHONPATH=str(ROOT), WANDB_MODE="disabled", WANDB_DISABLED="true",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MPLBACKEND="Agg", YOLO_CONFIG_DIR=str(run / "yolo_config"))
    for key in ("ANTI_UAV_TRUST_DATASET_CACHE", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    os.environ.update({k: env[k] for k in ("YOLO_CONFIG_DIR", "WANDB_MODE", "WANDB_DISABLED")})
    def execute(stage, command):
        status(run, stage, command=command)
        with (run / "logs" / f"{stage}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    scripts, python = ROOT / "scripts/anti_uav", sys.executable
    snapshot = json.loads(a.snapshot.read_text())
    if snapshot["new_videos"] != 14:
        raise ValueError("This paired protocol expects the frozen 14-new-video snapshot")
    protocol = dict(baseline=str(a.baseline), expanded=str(a.expanded), snapshot=str(a.snapshot),
                    new_video_count=14, epochs_per_stage=a.epochs, stages_per_arm=2, batch=64, nbs=128,
                    input_hw=[544, 960], device=a.device, seed=20260915, initial_p3=str(INITIAL),
                    augmentation="Original scale=.2, translate=.05, fliplr=.5; no extra crop views",
                    sampling="Keep every old slot and every reviewed new positive; cycle all reviewed new negatives at global ~15%",
                    evaluation="Fixed 960x544, conf=.01/.03/.05, NMS IoU=.45, match IoU=.5, max_det=100",
                    selection="Native gray F2/AP, not Video00004 or synthetic stress views",
                    limitation="Matched epochs, not matched optimizer steps; whole-video split, not session-independent",
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (run / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    try:
        execute("prepare_expanded", [python, str(scripts / "append_approved_gray_native.py"),
                "--source", str(a.baseline), "--snapshot", str(a.snapshot), "--output", str(a.expanded),
                "--video-root", "/mnt/andrew/video-labeler/videos", "--old-root", str(DATA / "seven_old_videos"),
                "--positive-stride", "1", "--negative-stride", "1", "--negative-fraction", "0.15"])
        import yaml
        before = yaml.safe_load((a.baseline / "train_hardneg_gray_monitor.yaml").read_text())
        after = yaml.safe_load((a.expanded / "train_hardneg_gray_monitor.yaml").read_text())
        if Path(before["val"]).read_bytes() != Path(after["val"]).read_bytes():
            raise ValueError("Validation data changed between experiment arms")
        for arm, dataset in (("expanded_28", a.expanded), ("baseline_14", a.baseline)):
            execute(f"train_{arm}", [python, str(scripts / "run_rebalanced_fullscale_training.py"),
                    "--dataset", str(dataset), "--run-dir", str(run / arm), "--initial-p3", str(INITIAL),
                    "--old-run", str(OLD), "--device", a.device, "--epochs", str(a.epochs),
                    "--fixed-validation", "--skip-final-test"])
        evaluate_and_report(run, a.baseline, a.expanded, a.device)
        status(run, "complete", report=str(run / "COMPARISON.md"))
    except KeyboardInterrupt:
        status(run, "stopped_by_user")
        raise
    except Exception as error:
        previous = json.loads((run / "status.json").read_text())
        status(run, "failed", failed_stage=previous["stage"], error=str(error))
        raise


if __name__ == "__main__":
    main()
