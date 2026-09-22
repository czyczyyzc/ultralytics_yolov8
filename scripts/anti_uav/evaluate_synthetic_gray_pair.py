#!/usr/bin/env python3
"""Matched original/synthetic evaluation of the five frozen standalone P3 models."""
import argparse
from collections import Counter
from functools import partial
import json
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
from scripts.anti_uav.evaluate_p3_gray_pair import MODELS, PairValidator, pool_native
from scripts.anti_uav.run_gray_probability_ablation import sha256, write_json
from scripts.anti_uav.run_native_pool_comparison import clean
from ultralytics import YOLO
from ultralytics.utils import SETTINGS, ops


GROUPS = ("all", "replaced", "original_retained", "negative_unchanged", "replaced_original_4to8px")


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k,v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return clean(value)


def groups_for(row):
    groups = ["all", row["state"]]
    if row["state"] == "replaced":
        edge = max(row["box"][2:])*min(960/row["width"], 544/row["height"])
        if 4 <= edge <= 8:
            groups.append("replaced_original_4to8px")
    return groups


def frame_entry(correct, scores, pred_cls, target_cls, counts):
    metrics = {}
    for conf, (tp, fp, fn) in counts.items():
        q = f"native/c{conf:.2f}/"
        metrics.update({q+k: v for k, v in dict(TP=tp, FP=fp, FN=fn, FRAMES=1).items()})
    return dict(metrics=metrics, scales={}, arrays=dict(tp=correct, conf=scores, pred_cls=pred_cls, target_cls=target_cls))


def pool_frames(frames, group):
    selected = [f["entry"] for f in frames if group in f["groups"]]
    if not selected:
        raise ValueError(f"Empty evaluation group: {group}")
    return pool_native(selected)


class FrameValidator(PairValidator):
    def __init__(self, *args, frame_metadata, **kwargs):
        self.frame_metadata = frame_metadata
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.frames = []

    def postprocess(self, preds):
        # Offline accuracy must not silently omit the tail of a slow NMS batch.
        return ops.non_max_suppression(preds, self.args.conf, self.args.iou, labels=self.lb,
            multi_label=True, agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det, max_time_img=float("inf"))

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            source = batch["im_file"][si]
            row = self.frame_metadata[source]
            prepared = self._prepare_batch(si, batch)
            boxes, classes = prepared["bbox"], prepared["cls"]
            if len(boxes) != int(bool(row["box"])):
                raise ValueError("Per-frame GT count differs from manifest")
            native = self._prepare_pred(pred, prepared)
            correct = self._process_batch(native, boxes, classes).cpu().numpy()
            counts = {}
            for conf in (.01, .03, .05):
                selected = native[native[:, 4] >= conf]
                tp = int(self._process_batch(selected, boxes, classes)[:, 0].sum())
                counts[conf] = (tp, len(selected)-tp, len(boxes)-tp)
            entry = frame_entry(correct, native[:, 4].cpu().numpy(), native[:, 5].cpu().numpy(),
                                classes.cpu().numpy(), counts)
            self.frames.append(dict(video=row["video"], frame=row["frame"], state=row["state"],
                asset_id=row["asset_id"], groups=groups_for(row), entry=entry,
                predictions=native.cpu().numpy().tolist(), gt=boxes.cpu().numpy().tolist()))

    def get_stats(self):
        stats = super().get_stats()
        self.metrics.paired_frames = self.frames
        if len(self.frames) != len(self.frame_metadata):
            raise ValueError("Missing or unexpected evaluation frames")
        pooled, _ = pool_frames(self.frames, "all")
        for key, value in pooled.items():
            if not np.isclose(value, stats[key], atol=1e-12, rtol=0):
                raise AssertionError(f"Per-frame aggregation disagrees with validator: {key}")
        return stats


def verify_dataset(root, reference):
    summary = json.loads((root/"summary.json").read_text())
    if summary["stage"] != "complete" or summary["smoke_per_video"]:
        raise ValueError("Use the complete non-smoke dataset")
    if sha256(root/"manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["training_allowed"] is not False:
        raise ValueError("Dataset is not evaluation-only")
    rows = manifest["rows"]
    if len(rows) != 3780 or len({(r["video"], r["frame"]) for r in rows}) != len(rows):
        raise ValueError("Unexpected frame identities")
    if Counter(r["state"] for r in rows) != Counter(replaced=981, original_retained=264, negative_unchanged=2535):
        raise ValueError("Dataset state counts changed")
    if sum(bool(r["box"]) for r in rows) != 1245:
        raise ValueError("GT coverage changed")
    protected = {str(root/n): sha256(root/n) for n in ("plan.json", "manifest.json", "summary.json")}
    plan = json.loads((root/"plan.json").read_text())
    if sha256(reference/"protocol.json") != plan["protected"][str(reference/"protocol.json")]:
        raise ValueError("Reference evaluation changed")
    protected.update(plan["protected"])
    for row in rows:
        for key, hash_key in (("original", "source_sha256"), ("original_label", "label_sha256"),
                              ("replacement", "output_sha256"), ("output_label", "output_label_sha256")):
            path = root/row[key]
            if sha256(path) != row[hash_key]:
                raise ValueError(f"Input pixels/labels changed: {path}")
            protected[str(path)] = row[hash_key]
        if row["state"] != "replaced" and row["output_sha256"] != row["source_sha256"]:
            raise ValueError("Unchanged control was altered")
    for variant, key in (("original", "original"), ("synthetic", "replacement")):
        for video in ("Video00004", "Video00009"):
            config = root/f"{variant}_{video}.yaml"
            data = yaml.safe_load(config.read_text())
            listing = Path(data["val"])
            expected = [str(root/r[key]) for r in rows if r["video"] == video]
            if data["train"] is not None or listing.read_text().splitlines() != expected:
                raise ValueError("Evaluation list/config differs from frozen manifest")
            protected[str(config)], protected[str(listing)] = sha256(config), sha256(listing)
    for path, digest in protected.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Protected input changed: {path}")
    return rows, protected


def write_report(root, results):
    lines = ["# Frozen standalone P3: synthetic appearance evaluation", "",
        "960x544 FP32; conf .03 for counts, matching IoU .5, NMS .45, max_det 100; AP floor .001.",
        "Five fixed best.pt files; no training, checkpoint selection, threshold tuning, tracker or RKNN.",
        "Full sets: 3780 frames / 1245 GT. Replaced pairs: 981 positive frames only.",
        "The 328 source assets were used in training; Video00009 selected checkpoints. Not an independent or unseen-type test.",
        "Original-size 4-8px group membership is frozen before replacement; do not compare shifting bbox-size denominators.", ""]
    for split in ("combined", "Video00004", "Video00009"):
        for group in ("all", "replaced", "replaced_original_4to8px"):
            lines += [f"## {split} / {group}", "",
                "| Model | View | P | R | TP | FP | FN | AP50 | AP50-95 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for name, item in results.items():
                for variant in ("original", "synthetic"):
                    m = item["variants"][variant][split][group]
                    q = "native/c0.03/"
                    lines.append(f"| {name} | {variant} | {m[q+'P']:.2%} | {m[q+'R']:.2%} | "
                        f"{m[q+'TP']} | {m[q+'FP']} | {m[q+'FN']} | {m['native/mAP50']:.2%} | {m['native/mAP50-95']:.2%} |")
            lines.append("")
    (root/"REPORT.md").write_text("\n".join(lines).rstrip()+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    a = p.parse_args()
    root, dataset, reference = a.output.resolve(), a.dataset.resolve(), a.reference.resolve()
    root.mkdir(parents=True, exist_ok=False)
    def status(stage, **kwargs):
        value = dict(stage=stage, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kwargs)
        write_json(root/"status.json", value)
        print(json.dumps(value), flush=True)
    try:
        status("preflight")
        if platform.python_version() != "3.10.12" or torch.__version__ != "2.5.1+cu121":
            raise ValueError("Use the matching Python 3.10.12 / PyTorch 2.5.1+cu121 environment")
        usage = subprocess.check_output(["nvidia-smi", f"--id={a.device}",
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, usage.strip().split(","))
        if memory > 1024 or utilization > 10:
            raise RuntimeError(f"Selected GPU is not idle: {usage}")
        SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                             mlflow=False, neptune=False, raytune=False))
        torch.set_num_threads(4)
        rows, protected = verify_dataset(dataset, reference)
        ref = json.loads((reference/"protocol.json").read_text())
        previous = json.loads((reference/"results.json").read_text())
        weights = {n: ref["models"][n] for n in MODELS}
        for item in weights.values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("Reference checkpoint changed")
            protected[item["path"]] = item["sha256"]
        write_json(root/"protocol.json", dict(dataset=str(dataset), reference=str(reference), models=weights,
            input_hw=[544,960], dtype="FP32", nms_iou=.45, matching_iou=.5, conf_floor=.001,
            thresholds=[.01,.03,.05], max_det=100, batch=32, groups=list(GROUPS),
            nms_wall_clock_truncation=False,
            checkpoint_selection=False, threshold_tuning=False, assets_seen_during_training=True,
            independent_test=False, protected=protected, python=platform.python_version(), torch=torch.__version__,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        results = {}
        for name, weight in weights.items():
            result = dict(**weight, variants={}, control_check={})
            frames_by_variant = {}
            for variant, key in (("original", "original"), ("synthetic", "replacement")):
                all_frames, splits = [], {}
                for video in ("Video00004", "Video00009"):
                    status("evaluation", model=name, variant=variant, video=video)
                    metadata = {str(dataset/r[key]): r for r in rows if r["video"] == video}
                    model = YOLO(weight["path"])
                    if len(model.model.model[-1].stride) != 3:
                        raise ValueError("Only standalone P3 checkpoints are allowed")
                    metrics = model.val(data=str(dataset/f"{variant}_{video}.yaml"),
                        validator=partial(FrameValidator, frame_metadata=metadata), imgsz=[544,960], rect=False,
                        device=str(a.device), batch=32, workers=4, conf=.001, iou=.45, max_det=100,
                        half=False, plots=False, verbose=False, project=str(root/"val"), name=f"{name}_{variant}_{video}")
                    frames = metrics.paired_frames
                    splits[video] = {group: pool_frames(frames, group)[0] for group in GROUPS}
                    all_frames.extend(frames)
                    del model
                    torch.cuda.empty_cache()
                splits["combined"] = {}
                for group in GROUPS:
                    m, arrays = pool_frames(all_frames, group)
                    splits["combined"][group] = m
                    np.savez_compressed(root/f"{name}_{variant}_{group}_pr.npz", **arrays)
                result["variants"][variant] = splits
                frames_by_variant[variant] = {(f["video"], f["frame"]): f for f in all_frames}
                write_json(root/f"{name}_{variant}_frames.json", serializable(all_frames))
            changes = []
            for identity, original in frames_by_variant["original"].items():
                synthetic = frames_by_variant["synthetic"][identity]
                if original["state"] != "replaced" and original["entry"]["metrics"] != synthetic["entry"]["metrics"]:
                    changes.append(dict(video=identity[0], frame=identity[1],
                        original=original["entry"]["metrics"], synthetic=synthetic["entry"]["metrics"]))
            result["control_check"]["unchanged_frame_count_differences"] = changes
            if changes:
                write_json(root/f"{name}_invalid_controls.json", changes)
                raise AssertionError("Unchanged frames have different counts; investigate before reporting accuracy")
            old = previous[name]["splits"]["pooled_native"]
            current = result["variants"]["original"]["combined"]["all"]
            result["control_check"]["reference_original_differences"] = {
                k: dict(previous=old[k], current=v) for k,v in current.items() if old[k] != v}
            results[name] = result
            write_json(root/f"{name}.json", clean(result))
            write_json(root/"results.json", clean(results))
        status("final_integrity_check")
        for path, digest in protected.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Protected input mutated during evaluation: {path}")
        write_report(root, results)
        status("complete", models=len(results), report=str(root/"REPORT.md"))
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
