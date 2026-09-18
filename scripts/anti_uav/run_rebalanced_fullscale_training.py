#!/usr/bin/env python3
"""Train P3 then frozen add-on with predeclared gray selection; evaluate test only afterwards."""

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

import torch
import yaml

from scripts.anti_uav.gray_deployment_trainer import GrayP3Trainer, GrayAddOnTrainer, GrayDeploymentValidator
from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayP3Trainer, FixedShapeGrayAddOnTrainer, FixedShapeGrayValidator
from scripts.anti_uav.train_frozen_p3_addon_p2 import initialize_addon_model, DEFAULT_CFG, verify_legacy_outputs
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.torch_utils import init_seeds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--data-yaml", type=Path, help="Explicit controlled-arm YAML; defaults to the existing dataset schedule")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--initial-p3", type=Path, required=True)
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--device", default="6")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--resume-p3", action="store_true", help="Resume saved P3 optimizer/EMA, then run the pending add-on stage.")
    p.add_argument("--fixed-validation", action="store_true", help="Use an asserted 544x960 canvas in new controlled experiments.")
    p.add_argument("--skip-final-test", action="store_true", help="Let a parent paired-experiment runner evaluate both arms together.")
    a = p.parse_args()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.run_dir/"pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    init_seeds(20260915, deterministic=True)
    data = a.data_yaml or a.dataset/"train_hardneg_gray_monitor.yaml"
    manifest = json.loads((a.dataset/"manifest.json").read_text())
    split = yaml.safe_load(data.read_text())
    if split.get("label_sampling") and not a.fixed_validation:
        raise ValueError("Label-pool datasets require --fixed-validation and the native exposure trainer")
    train_paths = Path(split["train"]).read_text().splitlines()
    val_paths = Path(split["val"]).read_text().splitlines()
    assert not set(train_paths) & set(val_paths)
    assert all("Video00004" not in Path(x).parts for x in train_paths+val_paths)
    assert all("gray_val" not in Path(x).parts and "zoom_val" not in Path(x).parts for x in train_paths)
    model = YOLO(str(a.initial_p3))
    initial_data = Path(model.ckpt["train_args"]["data"])
    initial_train = Path(yaml.safe_load(initial_data.read_text())["train"]).read_text()
    assert "stationary_video00009" not in initial_train
    protocol = dict(dataset=str(a.dataset), initial_p3=str(a.initial_p3), device=a.device, epochs=a.epochs,
                    batch=64, nbs=128, input_hw=[544, 960], seed=20260915, val_video=manifest["validation_video"]["video"],
                    fitness="0.5 * native-gray F2(conf=.03) + 0.3 * native-gray AP50 + 0.2 * native-gray AP50-95",
                    nms_iou=.45, conf_floor=.001, test_selection=False,
                    zoom_validation="Reported separately; never affects checkpoint selection",
                    online_scale=split.get("online_scale"), label_sampling=split.get("label_sampling"), fixed_validation=a.fixed_validation,
                    online_replacement=split.get("online_replacement"), data_yaml=str(data),
                    initial_train_data=str(initial_data), git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    if a.resume_p3:
        previous = json.loads((a.run_dir/"protocol.json").read_text())
        if previous.get("fixed_validation", False) != a.fixed_validation:
            raise ValueError("Cannot change validation shape when resuming checkpoint selection")
        if previous.get("label_sampling") != split.get("label_sampling"):
            raise ValueError("Cannot change native exposure sampling when resuming")
        if previous.get("online_replacement") != split.get("online_replacement"):
            raise ValueError("Cannot change online replacement policy when resuming")
        for key in ("dataset", "initial_p3", "epochs", "batch", "input_hw", "seed", "online_scale"):
            if previous[key] != protocol[key]:
                raise ValueError(f"Resume must preserve protocol field: {key}")
        checkpoint = a.run_dir/"training_p3/p3/weights/last.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("optimizer") is None or not 0 <= state["epoch"] < a.epochs-1:
            raise ValueError("P3 checkpoint lacks a resumable optimizer or has already completed")
        if Path(state["train_args"]["data"]).resolve() != data.resolve() or (a.run_dir/"training_addon").exists():
            raise ValueError("Unexpected dataset or an existing add-on stage")
        recovery = dict(checkpoint=str(checkpoint), completed_epochs=state["epoch"]+1,
                        original_commit=previous["git_commit"], recovery_commit=protocol["git_commit"],
                        time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        note="Optimizer/EMA restored; interrupted epoch rerun. RNG stream is not bit-exact across restart.")
        recovery_path = a.run_dir/f"recovery_{time.time_ns()}.json"
        recovery_path.write_text(json.dumps(recovery, indent=2)+"\n")
        del state
        model = YOLO(str(checkpoint))
    else:
        if not a.validate_only and ((a.run_dir/"training_p3").exists() or (a.run_dir/"training_addon").exists()):
            raise FileExistsError("Refuse to overwrite existing training")
        (a.run_dir/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    def status(stage, **extra):
        state = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        temp = a.run_dir/"status.tmp"
        temp.write_text(json.dumps(state,indent=2)+"\n")
        temp.replace(a.run_dir/"status.json")
        print(json.dumps(state), flush=True)
    try:
        if a.validate_only:
            status("validation_smoke")
            metrics = model.val(data=str(data), validator=FixedShapeGrayValidator if a.fixed_validation else GrayDeploymentValidator, imgsz=[544,960], device=a.device,
                                batch=32, workers=4, rect=False, conf=.001, iou=.45, max_det=100,
                                plots=False, project=str(a.run_dir), name="smoke")
            (a.run_dir/"validation_smoke.json").write_text(json.dumps(metrics.gray_selection,indent=2)+"\n")
            status("validation_smoke_complete")
            return
        if not a.resume_p3 and ((a.run_dir/"training_p3").exists() or (a.run_dir/"training_addon").exists()):
            raise FileExistsError("Refuse to overwrite existing training")
        common = dict(data=str(data), imgsz=[544,960], epochs=a.epochs, patience=a.epochs,
                      batch=64, device=a.device, workers=8, exist_ok=False, optimizer="AdamW",
                      lrf=.1, momentum=.937, weight_decay=.0005, warmup_momentum=.8,
                      warmup_bias_lr=.01, cos_lr=True, amp=True, deterministic=True, seed=20260915,
                      nbs=128, rect=False, cache=False, val=True, save=True, save_period=1, plots=True,
                      single_cls=True, mosaic=0., mixup=0., copy_paste=0., degrees=0., translate=.05,
                      scale=.2, shear=0., perspective=0., flipud=0., fliplr=.5, hsv_h=.015,
                      hsv_s=.4, hsv_v=.3, close_mosaic=0, verbose=True)
        status("training_p3")
        model.add_callback("on_fit_epoch_end", lambda trainer: status("training_p3", epoch=trainer.epoch+1,
                                                                       fitness=float(trainer.fitness)))
        model.train(trainer=FixedShapeGrayP3Trainer if a.fixed_validation else GrayP3Trainer, resume=a.resume_p3, project=str(a.run_dir/"training_p3"), name="p3",
                    lr0=.0001, warmup_epochs=0, **common)
        p3 = a.run_dir/"training_p3/p3/weights/best.pt"
        status("initialize_addon")
        init_seeds(20260915, deterministic=True)
        addon, init = initialize_addon_model(p3, DEFAULT_CFG, a.run_dir/"addon_initialized.pt")
        (a.run_dir/"addon_initialization.json").write_text(json.dumps(init,indent=2)+"\n")
        addon.add_callback("on_fit_epoch_end", lambda trainer: status("training_addon", epoch=trainer.epoch+1,
                                                                       fitness=float(trainer.fitness)))
        status("training_addon")
        if split.get("online_replacement"):
            addon_split = dict(split, online_replacement=dict(split["online_replacement"],
                epoch_offset=int(split["online_replacement"].get("epoch_offset", 0))+a.epochs))
            addon_data = a.run_dir/"addon_online_data.yaml"
            addon_data.write_text(yaml.safe_dump(addon_split, sort_keys=False))
            common["data"] = str(addon_data)
        addon.train(trainer=FixedShapeGrayAddOnTrainer if a.fixed_validation else GrayAddOnTrainer, project=str(a.run_dir/"training_addon"), name="p2",
                    lr0=.001, warmup_epochs=1, **common)
        freezes = {}
        for ckpt in ("best", "last"):
            freezes[ckpt] = verify_legacy_outputs(YOLO(str(p3)).model,
                             YOLO(str(a.run_dir/f"training_addon/p2/weights/{ckpt}.pt")).model)
        (a.run_dir/"frozen_checks.json").write_text(json.dumps(freezes,indent=2)+"\n")
        if a.skip_final_test:
            status("weights_ready", note="No holdout evaluation performed; parent runner will compare both arms.")
            return
        holdout = Path("/mnt/andrew/anti_uav_model_refinement/data/real_gray_yolo_lovo_positive_mixed_v1_20260828/folds/holdout_Video00004")
        models = {"old_p3":next((a.old_run/"training_p3").glob("*/weights/best.pt")),
                  "old_addon":a.old_run/"training_addon/final/weights/best.pt", "new_p3":p3,
                  "new_addon":a.run_dir/"training_addon/p2/weights/best.pt",
                  "new_addon_last":a.run_dir/"training_addon/p2/weights/last.pt"}
        for name, weights in models.items():
            status("final_test", model=name)
            subprocess.run([sys.executable,str(ROOT/"scripts/anti_uav/evaluate_real_gray_yolo_lovo_fold.py"),
                            "--model",str(weights),"--fold-dir",str(holdout),"--rgb-data",str(data),"--skip-rgb",
                            "--output",str(a.run_dir/"evaluation"/f"{name}.json"),"--device",a.device,
                            "--batch","32","--workers","4","--thresholds","0.01","0.03","0.05","0.10","0.25","0.40","0.45"],check=True)
        from scripts.anti_uav.run_approved_frozen_p3_p2 import report
        report(a.run_dir)
        comparison = a.run_dir/"COMPARISON.md"
        comparison.write_text(comparison.read_text().replace("10-video vs 15-video training", "old 10-video vs rebalanced 14-video training")
                              .replace("Checkpoint selection: Anti-UAV300 RGB validation only. No holdout epoch selection.",
                                       "New checkpoint selection: independent native gray validation deployment fitness. Video00004 is test-only."))
        status("complete")
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
