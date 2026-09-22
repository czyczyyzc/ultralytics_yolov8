#!/usr/bin/env python3
"""Evaluate fixed P3 checkpoints on two native splits and pool prediction statistics."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
from scripts.anti_uav.run_gray_probability_ablation import sha256, write_json
from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.metrics import ap_per_class


class PairValidator(FixedShapeGrayValidator):
    def get_stats(self):
        stats = super().get_stats()
        self.metrics.native_pr_arrays = {
            key: np.concatenate(value) if value else np.empty((0, 10) if key == "tp" else (0,))
            for key, value in self.group_stats["native"].items()}
        self.metrics.native_scale_counts = {
            (conf, size): dict(counts) for (group, conf, size), counts in self.scale_counts.items()
            if group == "native"}
        return stats


def pool_native(entries):
    """Micro-average counts; calculate AP from globally ranked detections, never mean APs."""
    output = {}
    for conf in (.01, .03, .05):
        q = f"native/c{conf:.2f}/"
        for key in ("TP", "FP", "FN", "FRAMES"):
            output[q + key] = sum(entry["metrics"][q + key] for entry in entries)
        tp, fp, fn = (output[q + key] for key in ("TP", "FP", "FN"))
        output[q + "P"] = tp / max(tp + fp, 1)
        output[q + "R"] = tp / max(tp + fn, 1)
        output[q + "FP1000"] = 1000 * fp / max(output[q + "FRAMES"], 1)
        sizes = set().union(*(entry["scales"].keys() for entry in entries))
        for level_conf, size in sorted(sizes):
            if level_conf != conf:
                continue
            counts = [entry["scales"].get((conf, size), {}) for entry in entries]
            total = sum(c.get("gt", 0) for c in counts)
            matched = sum(c.get("tp", 0) for c in counts)
            output[q + size + "/GT"] = total
            output[q + size + "/R"] = matched / total if total else None
    arrays = {key: np.concatenate([entry["arrays"][key] for entry in entries])
              for key in ("tp", "conf", "pred_cls", "target_cls")}
    output["native/mAP50"] = output["native/mAP50-95"] = 0.
    if len(arrays["conf"]) and len(arrays["target_cls"]):
        ap = ap_per_class(**arrays, plot=False)[5]
        output["native/mAP50"] = float(ap[:, 0].mean())
        output["native/mAP50-95"] = float(ap.mean())
    return output, arrays


MODELS = {
    "old28_p3": "native_fullpool_14_vs_28_20260916/expanded_28",
    "gray40_assets53_prob50_p3": "approved40_online53_20260921",
    "gray40_assets328_prob50_p3": "approved40_online328_20260922",
    "gray40_assets328_prob15_p3": "approved40_prob15_v2_20260922",
    "gray40_prob00_p3": "approved40_prob00_v2_20260922",
}


def write_report(output, results):
    lines = ["# Pure P3: Video00004 + Video00009", "",
             "FP32 detector only; no P2, tracking, branch calibration or RKNN quantization.",
             "Fixed best.pt, 960x544 input, NMS IoU .45, matching IoU .5, max_det 100.",
             "P/R/counts use conf .03; AP is ranked from conf floor .001.",
             "Video00004: 2359 test frames / 448 GT. Video00009: 1421 selection-validation frames / 797 GT.",
             "The 64 synthetic zoom views are excluded. The pooled set contains 3780 frames / 1245 GT.",
             "The pooled set is NOT an independent test: Video00009 selected checkpoints.",
             "No checkpoint or threshold is selected using this evaluation.",
             "Pooled counts are summed and ratios recomputed; pooled AP is recomputed from all native predictions.", ""]
    for split in ("Video00004_test", "Video00009_selection_subset", "pooled_native"):
        lines += [f"## {split}", "",
                  "| Model | P | R | TP | FP | FN | AP50 | AP50-95 | 4-8px R |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name in MODELS:
            m, q = results[name]["splits"][split], "native/c0.03/"
            lines.append(f"| {name} | {m[q+'P']:.2%} | {m[q+'R']:.2%} | {m[q+'TP']} | {m[q+'FP']} | "
                         f"{m[q+'FN']} | {m['native/mAP50']:.2%} | {m['native/mAP50-95']:.2%} | "
                         f"{m[q+'long_4to8px/R']:.2%} |")
        lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)

    def status(stage, **extra):
        write_json(a.output / "status.json", dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                   pid=os.getpid(), **extra))
        print(json.dumps(dict(stage=stage, **extra)), flush=True)

    try:
        status("preflight")
        if platform.python_version() != "3.10.12" or torch.__version__ != "2.5.1+cu121":
            raise ValueError("Use the reference Python 3.10.12 / PyTorch 2.5.1+cu121 runtime")
        usage = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        mem, util = map(int, usage.strip().split(","))
        if mem > 1024 or util > 10:
            raise RuntimeError(f"GPU {a.device} is not idle: {usage.strip()}")
        SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                             mlflow=False, neptune=False, raytune=False))
        torch.set_num_threads(4)
        runs = ROOT / "runs/anti_uav"
        val = Path("/mnt/andrew/anti_uav_model_refinement/data/real_gray_online_replacement_40videos_assets328_20260922/train_online_gray_monitor.yaml")
        splits = {"Video00004_test": HOLDOUT, "Video00009_selection_subset": val}
        expected = {"Video00004_test": (2359, 448), "Video00009_selection_subset": (1421, 797)}
        manifests, native_sets = {}, {}
        for name, config_path in splits.items():
            config = yaml.safe_load(config_path.read_text())
            listing = Path(config["val"])
            paths = [line for line in listing.read_text().splitlines() if line.strip()]
            native = [line for line in paths if "/zoom_val/" not in line]
            if len(native) != len(set(native)) or len(native) != expected[name][0]:
                raise ValueError(f"Unexpected coverage for {name}")
            manifests[name] = dict(config=str(config_path), list=str(listing), list_sha256=sha256(listing),
                                   native_frames=len(native), excluded_zoom_views=len(paths)-len(native))
            native_sets[name] = set(native)
        if native_sets["Video00004_test"] & native_sets["Video00009_selection_subset"]:
            raise ValueError("Duplicate frames across splits")
        weights = {name: runs / subdir / "training_p3/p3/weights/best.pt" for name, subdir in MODELS.items()}
        frozen_weights = {name: dict(path=str(path), sha256=sha256(path)) for name, path in weights.items()}
        write_json(a.output / "protocol.json", dict(models=frozen_weights, datasets=manifests,
            input_hw=[544, 960], dtype="FP32", nms_iou=.45, matching_iou=.5, conf_floor=.001,
            thresholds=[.01, .03, .05], max_det=100, checkpoint_selection=False, threshold_tuning=False,
            pooled_is_independent_test=False, python=platform.python_version(), torch=torch.__version__,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        results = {}
        for name, path in weights.items():
            entries = []
            results[name] = dict(**frozen_weights[name], splits={})
            for split, data in splits.items():
                status("evaluation", model=name, split=split)
                model = YOLO(str(path))
                if len(model.model.model[-1].stride) != 3:
                    raise ValueError("This evaluation is for pure P3 only")
                metrics = model.val(data=str(data), validator=PairValidator, imgsz=[544, 960], rect=False,
                    device=str(a.device), batch=32, workers=4, conf=.001, iou=.45, max_det=100,
                    half=False, plots=False, verbose=False, project=str(a.output / "val"), name=f"{name}_{split}")
                native = {k: v for k, v in clean(metrics.gray_selection).items() if k.startswith("native/")}
                if (native["native/c0.03/FRAMES"], native["native/c0.03/TP"] + native["native/c0.03/FN"]) != expected[split]:
                    raise ValueError(f"Frames/GT changed for {split}")
                results[name]["splits"][split] = native
                entries.append(dict(metrics=native, arrays=metrics.native_pr_arrays, scales=metrics.native_scale_counts))
                np.savez_compressed(a.output / f"{name}_{split}_pr.npz", **metrics.native_pr_arrays)
                del model
                torch.cuda.empty_cache()
            combined, arrays = pool_native(entries)
            results[name]["splits"]["pooled_native"] = combined
            np.savez_compressed(a.output / f"{name}_pooled_native_pr.npz", **arrays)
            if sha256(path) != frozen_weights[name]["sha256"]:
                raise ValueError("Checkpoint changed during evaluation")
            write_json(a.output / f"{name}.json", results[name])
            write_json(a.output / "results.json", results)
        for manifest in manifests.values():
            if sha256(manifest["list"]) != manifest["list_sha256"]:
                raise ValueError("Dataset list changed during evaluation")
        write_report(a.output, results)
        status("complete", report=str(a.output / "REPORT.md"), models=len(results), pooled_frames=3780)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
