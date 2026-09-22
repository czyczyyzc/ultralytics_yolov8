#!/usr/bin/env python3
"""Run an isolated replacement-probability ablation; never tune on the test video."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probability_config(reference, probability):
    if not 0 <= probability <= 1 or not reference.get("online_replacement"):
        raise ValueError("A reference replacement config and probability in [0, 1] are required")
    result = deepcopy(reference)
    if probability == 0:
        result.pop("online_replacement")
    else:
        result["online_replacement"]["replacement_probability"] = probability
    return result


def assert_probability_only(reference, candidate):
    a, b = deepcopy(reference), deepcopy(candidate)
    source = a.pop("online_replacement")
    policy = b.pop("online_replacement", None)
    if policy:
        source.pop("replacement_probability")
        probability = policy.pop("replacement_probability")
        if not 0 < probability <= 1 or source != policy:
            raise ValueError("Replacement policy changed beyond probability")
    if a != b:
        raise ValueError("Data, validation or sampling changed beyond replacement probability")


def assert_split_isolation(train, val):
    heldout_parts = {"video00004", "holdout_video00004"}
    if (not train or not val or set(train) & set(val)
            or any(heldout_parts & {p.lower() for p in Path(s).parts} for s in train + val)):
        raise ValueError("Train/validation/test leakage or empty split")


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--probability", type=float, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--detach", action="store_true")
    a = parser.parse_args()
    a.reference_run, a.run_dir = a.reference_run.resolve(), a.run_dir.resolve()
    if a.detach:
        a.run_dir.mkdir(parents=True, exist_ok=False)
        with (a.run_dir / "console.log").open("x") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       *[v for v in sys.argv[1:] if v != "--detach"]]
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps(dict(pid=child.pid, run_dir=str(a.run_dir))))
        return
    a.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.run_dir / "experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.run_dir / "experiment.json").exists():
        raise FileExistsError("Use a new run directory; existing experiments are immutable")

    def status(stage, **extra):
        record = dict(stage=stage, pid=os.getpid(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        write_json(a.run_dir / "experiment_status.json", record)
        print(json.dumps(record), flush=True)

    try:
        status("preflight")
        reference = json.loads((a.reference_run / "protocol.json").read_text())
        completed = json.loads((a.reference_run / "experiment_status.json").read_text())
        if completed["stage"] != "complete" or not reference.get("fixed_validation"):
            raise ValueError("Reference must be complete with fixed-shape validation")
        if (reference["input_hw"] != [544, 960] or reference["batch"] != 64
                or reference["seed"] != 20260915 or reference["epochs"] != 15):
            raise ValueError("Reference differs from the controlled training protocol")
        import torch
        with (a.reference_run / "logs/training.log").open() as log:
            header = log.read(8192)
        expected = re.search(r"Python-([\d.]+) torch-([^\s]+)", header)
        if not expected or (platform.python_version(), torch.__version__) != expected.groups():
            raise ValueError("Use the same Python/PyTorch environment as the completed reference run")
        source = yaml.safe_load(Path(reference["data_yaml"]).read_text())
        candidate = probability_config(source, a.probability)
        assert_probability_only(source, candidate)
        cache = Path(reference["dataset"])
        summary = json.loads((cache / "summary.json").read_text())
        index = json.loads((cache / "index.json").read_text())
        if (summary["stage"] != "complete" or summary["is_smoke_subset"]
                or sha256(cache / "index.json") != summary["index_sha256"]
                or sha256(index["catalog"]) != index["catalog_sha256"]):
            raise ValueError("Reference cache/catalog is incomplete or changed")
        if any(row["video_sha256"] in index["heldout_sha256"] for row in index["images"].values()):
            raise ValueError("Holdout video in replacement cache")
        train = Path(source["train"]).read_text().splitlines()
        val = Path(source["val"]).read_text().splitlines()
        assert_split_isolation(train, val)
        hashes = {key: sha256(source[key]) for key in ("train", "val")}
        hashes["negative_pool"] = sha256(source["label_sampling"]["negative_pool"])
        hashes["initial_p3"] = sha256(reference["initial_p3"])
        source_experiment = json.loads((a.reference_run / "experiment.json").read_text())
        if hashes["initial_p3"] != source_experiment["initial_sha256"]:
            raise ValueError("Reference initialization changed")
        data_yaml = a.run_dir / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(candidate, sort_keys=False))
        protocol = dict(reference_run=str(a.reference_run), probability=a.probability,
                        device=a.device, epochs_per_stage=15, asset_count=len(index["asset_ids"]),
                        source_hashes=hashes, input_hw=[544, 960], batch=64, seed=20260915,
                        python_executable=sys.executable, python_version=platform.python_version(),
                        torch_version=torch.__version__,
                        selection="Unchanged reference gray fitness; validation-only branch calibration afterwards",
                        test_video_evaluated=False, independent_test_claim=False,
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
        write_json(a.run_dir / "experiment.json", protocol)
        config = a.run_dir / "yolo_config"
        config.mkdir()
        font = a.reference_run / "yolo_config/Arial.ttf"
        if font.is_file():
            shutil.copy2(font, config / "Arial.ttf")
        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   WANDB_MODE="disabled", WANDB_DISABLED="true", MPLBACKEND="Agg", YOLO_CONFIG_DIR=str(config))
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("ANTI_UAV_TRUST_DATASET_CACHE", None)
        os.environ.update(env)
        gpu_lock = Path(f"/tmp/anti_uav_probability_gpu{a.device}.lock").open("a")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        usage = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, usage.strip().split(","))
        if memory > 1024 or utilization > 10:
            raise RuntimeError(f"GPU is not idle: {usage.strip()}")
        command = [sys.executable, "-u", str(ROOT / "scripts/anti_uav/run_rebalanced_fullscale_training.py"),
                   "--dataset", str(cache), "--data-yaml", str(data_yaml), "--run-dir", str(a.run_dir),
                   "--initial-p3", reference["initial_p3"], "--old-run", str(a.reference_run),
                   "--device", str(a.device), "--epochs", "15", "--fixed-validation", "--skip-final-test"]
        status("training", command=command)
        with (a.run_dir / "training.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        current_hashes = {key: sha256(source[key]) for key in ("train", "val")}
        current_hashes["negative_pool"] = sha256(source["label_sampling"]["negative_pool"])
        current_hashes["initial_p3"] = sha256(reference["initial_p3"])
        if hashes != current_hashes:
            raise ValueError("Training source files changed during the experiment")

        from ultralytics import YOLO
        from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
        from scripts.anti_uav.run_native_pool_comparison import clean
        old_results = json.loads((a.reference_run / "comparison/results.json").read_text())
        results = {}
        for name, subdir in (("p3", "training_p3/p3"), ("addon", "training_addon/p2")):
            old_weights = a.reference_run / subdir / "weights/best.pt"
            old = [row for key, row in old_results.items()
                   if key.startswith("Video00009_selection_subset_") and row["sha256"] == sha256(old_weights)]
            if len(old) != 1:
                raise ValueError("Missing unambiguous completed validation reference")
            results[f"reference_prob50_{name}"] = old[0]
            weights = a.run_dir / subdir / "weights/best.pt"
            status("validation", model=name)
            metrics = YOLO(str(weights)).val(data=str(data_yaml), imgsz=[544, 960], rect=False,
                validator=FixedShapeGrayValidator, device=str(a.device), batch=32, workers=4,
                conf=.001, iou=.45, max_det=100, half=False, plots=False,
                project=str(a.run_dir / "validation"), name=name, verbose=False)
            results[f"candidate_{name}"] = dict(weights=str(weights), sha256=sha256(weights),
                                               metrics=clean(metrics.gray_selection))
        write_json(a.run_dir / "validation_results.json", results)
        lines = ["# Replacement probability ablation", "",
                 "Video00009 selection subset only; not independent test improvement. PT FP32, 960x544, conf .03.", "",
                 "| Model | P | R | FP | FN | AP50 | AP50-95 | 4-8px R |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, result in results.items():
            m, q = result["metrics"], "native/c0.03/"
            lines.append(f"| {name} | {m[q+'P']:.2%} | {m[q+'R']:.2%} | {m[q+'FP']} | {m[q+'FN']} | "
                         f"{m['native/mAP50']:.2%} | {m['native/mAP50-95']:.2%} | {m[q+'long_4to8px/R']:.2%} |")
        (a.run_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n")
        status("branch_calibration")
        calibration = [sys.executable, "-u", str(ROOT / "scripts/anti_uav/calibrate_addon_branches.py"),
                       "--model", str(a.run_dir / "training_addon/p2/weights/best.pt"),
                       "--reference-p3", str(a.run_dir / "training_p3/p3/weights/best.pt"),
                       "--data", str(data_yaml), "--output", str(a.run_dir / "branch_calibration"),
                       "--device", str(a.device)]
        with (a.run_dir / "calibration.log").open("x") as log:
            subprocess.run(calibration, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status("complete", report=str(a.run_dir / "COMPARISON.md"), test_video_evaluated=False)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
