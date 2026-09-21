#!/usr/bin/env python3
"""Wait for audited cache preparation, smoke-test, train two stages and compare."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--initial-p3", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--prepare-timeout-hours", type=float, default=24)
    p.add_argument("--comparison-kind", choices=("combined", "assets-only"), default="combined")
    p.add_argument("--reference-label", default="previous_28")
    p.add_argument("--experiment-label", default="expanded_online")
    p.add_argument("--preview-count", type=int, default=20)
    p.add_argument("--wait-reference", action="store_true")
    a = p.parse_args()
    for key in ("cache", "run_dir", "initial_p3", "reference_run", "old_run"):
        setattr(a, key, getattr(a, key).resolve())
    if a.epochs < 1 or a.prepare_timeout_hours <= 0 or a.preview_count < 1:
        raise ValueError("Invalid training/wait budget")
    if a.reference_label == a.experiment_label or any(
            not label.replace("_", "").isalnum() for label in (a.reference_label, a.experiment_label)):
        raise ValueError("Use distinct alphanumeric/underscore arm labels")
    a.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.run_dir/"experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.run_dir/"experiment.json").exists():
        raise FileExistsError("Use a new experiment; do not overwrite training")
    from scripts.anti_uav.synthesize_gray_drone_replacements import sha256
    comparison_note = ("Only asset catalog changes; same data, sampling, initialization, epochs and replacement probability"
                       if a.comparison_kind == "assets-only" else
                       "Combined data expansion and online replacement; not an augmentation-only ablation")
    protocol = dict(cache=str(a.cache), initial_p3=str(a.initial_p3), initial_sha256=sha256(a.initial_p3),
                    reference_run=str(a.reference_run), epochs_per_stage=a.epochs, device=a.device,
                    batch=64, input_hw=[544, 960], initialization="Same pretrained P3 as the 28-video experiment; fresh optimizer, not random initialization",
                    stages=["train_p3", "freeze_p3_train_addon_p2"],
                    comparison=comparison_note, reference_label=a.reference_label, experiment_label=a.experiment_label,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (a.run_dir/"experiment.json").write_text(json.dumps(protocol, indent=2)+"\n")
    (a.run_dir/"logs").mkdir()
    config = a.run_dir/"yolo_config"
    config.mkdir()
    for font in (a.reference_run/"yolo_config/Arial.ttf", a.reference_run.parent/"yolo_config/Arial.ttf"):
        if font.is_file():
            shutil.copy2(font, config/"Arial.ttf")
            break
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               WANDB_MODE="disabled", WANDB_DISABLED="true", MPLBACKEND="Agg", YOLO_CONFIG_DIR=str(config))
    for key in ("CUDA_VISIBLE_DEVICES", "ANTI_UAV_TRUST_DATASET_CACHE"):
        env.pop(key, None)
    os.environ.update({k: env[k] for k in ("YOLO_CONFIG_DIR", "WANDB_MODE", "WANDB_DISABLED")})
    def status(stage, **extra):
        value = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        tmp = a.run_dir/"experiment_status.tmp"
        tmp.write_text(json.dumps(value, indent=2)+"\n")
        tmp.replace(a.run_dir/"experiment_status.json")
        print(json.dumps(value), flush=True)
    def execute(stage, script, arguments):
        command = [sys.executable, "-u", str(ROOT/"scripts/anti_uav"/script), *arguments]
        status(stage, command=command)
        with (a.run_dir/"logs"/f"{stage}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        status("waiting_for_cache", training_started=False)
        deadline = time.monotonic()+a.prepare_timeout_hours*3600
        while True:
            state = json.loads((a.cache/"extension_status.json").read_text())
            if state["stage"] == "complete":
                break
            if state["stage"] == "failed":
                raise RuntimeError(f"Cache preparation failed: {state.get('error')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Cache preparation exceeded wait budget")
            os.kill(state["pid"], 0)
            time.sleep(30)
        if a.comparison_kind == "assets-only":
            import yaml
            from scripts.anti_uav.prepare_gray_asset_ablation import assert_asset_only_configs
            reference = json.loads((a.reference_run/"protocol.json").read_text())
            if (reference["epochs"] != a.epochs or reference["batch"] != 64
                    or reference["input_hw"] != [544, 960] or reference["seed"] != 20260915
                    or not reference.get("fixed_validation")
                    or sha256(Path(reference["initial_p3"])) != protocol["initial_sha256"]):
                raise ValueError("Reference training protocol differs beyond asset catalog")
            assert_asset_only_configs(yaml.safe_load(Path(reference["data_yaml"]).read_text()),
                                      yaml.safe_load((a.cache/"train_online_gray_monitor.yaml").read_text()))
        execute("augmentation_smoke", "benchmark_online_gray_replacement.py",
                ["--cache", str(a.cache), "--output", str(a.run_dir/"augmentation_smoke")])
        execute("augmentation_preview", "preview_online_gray_replacements.py",
                ["--cache", str(a.cache), "--output", str(a.run_dir/"augmentation_preview"), "--count", str(a.preview_count), "--seed", "20260921"])
        if a.comparison_kind == "assets-only":
            preview = json.loads((a.run_dir/"augmentation_preview/summary.json").read_text())
            assets = json.loads((a.cache/"index.json").read_text())["asset_ids"]
            if set(preview["asset_ids_used"]) != set(assets):
                raise ValueError("Expanded experiment requires at least one successful preview per asset")
        usage = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, usage.strip().split(","))
        if memory > 1024 or utilization > 10:
            raise RuntimeError(f"GPU {a.device} is no longer idle ({usage.strip()}); not starting training")
        execute("training", "run_rebalanced_fullscale_training.py",
                ["--dataset", str(a.cache), "--data-yaml", str(a.cache/"train_online_gray_monitor.yaml"),
                 "--run-dir", str(a.run_dir), "--initial-p3", str(a.initial_p3), "--old-run", str(a.old_run),
                 "--device", str(a.device), "--epochs", str(a.epochs), "--fixed-validation", "--skip-final-test"])
        if a.wait_reference:
            status("waiting_for_reference")
            deadline = time.monotonic()+a.prepare_timeout_hours*3600
            while True:
                reference_state = json.loads((a.reference_run/"experiment_status.json").read_text())
                if reference_state["stage"] == "complete":
                    break
                if reference_state["stage"] == "failed":
                    raise RuntimeError(f"Reference run failed: {reference_state.get('error')}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Reference run did not finish in time")
                os.kill(reference_state["pid"], 0)
                time.sleep(30)
        from ultralytics import YOLO
        from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
        from scripts.anti_uav.run_native_pool_comparison import clean, HOLDOUT
        models = {}
        for group, run in ((a.reference_label, a.reference_run), (a.experiment_label, a.run_dir)):
            for branch, subdir in (("p3", "training_p3/p3"), ("addon", "training_addon/p2")):
                models[f"{group}_{branch}"] = run/subdir/"weights/best.pt"
        output = a.run_dir/"comparison"
        output.mkdir()
        results = {}
        for split, data in (("Video00009_selection_subset", a.cache/"train_online_gray_monitor.yaml"), ("Video00004_test", HOLDOUT)):
            for name, weights in models.items():
                status("evaluation", split=split, model=name)
                metrics = YOLO(str(weights)).val(data=str(data), imgsz=[544, 960], rect=False,
                    validator=FixedShapeGrayValidator, device=str(a.device), batch=32, workers=4,
                    conf=.001, iou=.45, max_det=100, half=False, plots=False,
                    project=str(output/"val"), name=f"{split}_{name}", verbose=False)
                result = dict(weights=str(weights), sha256=sha256(weights), metrics=clean(metrics.gray_selection))
                results[f"{split}_{name}"] = result
                (output/f"{split}_{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        (output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
        lines = ["# Online Augmentation and Approved-Video Expansion", "",
                 "PT FP32 detector evaluation at 960x544, not RKNN or tracking metrics.",
                 "Video00009's unchanged validation subset selects checkpoints; it is not an untouched test.",
                 "Video00004 is test-only. " + comparison_note + ".", "",
                 "| Split | Model | Conf | Precision | Recall | FP | mAP50 | 4-8px recall |",
                 "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, result in results.items():
            m = result["metrics"]
            for conf in (.01, .03, .05):
                prefix = f"native/c{conf:.2f}"
                tiny = m[prefix+"/long_4to8px/R"]
                tiny_text = "N/A" if tiny is None else f"{tiny:.2%}"
                lines.append(f"| {name.rsplit('_', 1)[0]} | {name.rsplit('_', 1)[1]} | {conf:.2f} | {m[prefix+'/P']:.2%} | "
                             f"{m[prefix+'/R']:.2%} | {m[prefix+'/FP']} | {m['native/mAP50']:.2%} | {tiny_text} |")
        (a.run_dir/"COMPARISON.md").write_text("\n".join(lines)+"\n")
        status("complete", report=str(a.run_dir/"COMPARISON.md"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
