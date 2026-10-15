#!/usr/bin/env python3
"""Evaluate four delivery detectors and lossless native tracking on one pooled test set."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.efficient_gmc import EfficientGMC
from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator, pool_native
from scripts.anti_uav.motion_native_runtime import CAUSAL_DEFAULTS, NativeMotion
from scripts.anti_uav.run_gray_probability_ablation import sha256, write_json
from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.metrics import box_iou


RUNS = ROOT / "runs/anti_uav"
MODELS = {
    "original_p2p3": {
        "label": "Original P2+P3 (44 videos)",
        "weights": RUNS / "approved44_direct_online328_20260929/training_addon/p2/weights/best.pt",
        "strides": 4,
    },
    "original_p3": {
        "label": "Original P3 (44 videos)",
        "weights": RUNS / "approved44_direct_online328_20260929/training_p3/p3/weights/best.pt",
        "strides": 3,
    },
    "p2p3_teacher": {
        "label": "P2+P3 Teacher (47 videos)",
        "weights": RUNS / "approved47_p3_generalization_v2_20261010/teacher/p2p3/weights/best.pt",
        "strides": 4,
    },
    "p3_student": {
        "label": "P3 Student (47 videos, delivered)",
        "weights": RUNS / "approved47_p3_localization_v3_canonical_20261012/p3_student_best_fp32.pt",
        "strides": 3,
    },
}

VIDEO00009 = Path(
    "/mnt/andrew/anti_uav_model_refinement/data/real_gray_online_replacement_40videos_assets328_20260922/"
    "train_online_gray_monitor.yaml"
)
SPLITS = {
    "Video00004": {
        "data": HOLDOUT,
        "frames": 2359,
        "gt": 448,
        "fps": 100.0,
        "step": 1,
        "source": Path("/mnt/andrew/anti_uav_model_refinement/external_eval/real_gray_mp4/Video00004.mp4"),
    },
    "Video00009": {
        "data": VIDEO00009,
        "frames": 1421,
        "gt": 797,
        "fps": 100.0,
        "step": 10,
        "source": Path(
            "/mnt/andrew/video-labeler/videos/"
            "20260805_multirotor_sunny_frontlight_stationary_video00009_90af30d9.mp4"
        ),
    },
}


class CaptureValidator(PairValidator):
    """Retain native-frame predictions and GT after the validator's coordinate conversion."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.native_frame_records = []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for index, pred in enumerate(preds):
            path = str(batch["im_file"][index])
            if "/zoom_val/" in path:
                continue
            prepared = self._prepare_batch(index, batch)
            native_pred = self._prepare_pred(pred, prepared).detach().cpu().numpy()
            gt_boxes = prepared["bbox"].detach().cpu().numpy()
            gt_classes = prepared["cls"].detach().cpu().numpy()
            self.native_frame_records.append(
                {
                    "path": path,
                    "frame_index": int(Path(path).stem),
                    "original_hw": [int(value) for value in prepared["ori_shape"]],
                    "gt_xyxy_class": [
                        [*map(float, box), int(cls)] for box, cls in zip(gt_boxes, gt_classes)
                    ],
                    "pred_xyxy_score_class": native_pred.tolist(),
                }
            )

    def get_stats(self):
        stats = super().get_stats()
        self.metrics.native_frame_records = self.native_frame_records
        return stats


def status(output: Path, stage: str, **extra) -> None:
    value = {
        "stage": stage,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        **extra,
    }
    write_json(output / "status.json", value)
    print(json.dumps(value), flush=True)


def native_paths(config_path: Path) -> tuple[list[str], dict]:
    config = yaml.safe_load(config_path.read_text())
    listing = Path(config["val"])
    all_paths = [line for line in listing.read_text().splitlines() if line.strip()]
    native = [line for line in all_paths if "/zoom_val/" not in line]
    return native, {
        "config": str(config_path),
        "list": str(listing),
        "list_sha256": sha256(listing),
        "native_frames": len(native),
        "excluded_zoom_views": len(all_paths) - len(native),
    }


def build_motion_library(output: Path) -> Path:
    native = ROOT / "scripts/anti_uav/dist_native"
    library = output / "libmotion_tracker.so"
    subprocess.run(
        [
            "g++", "-std=c++17", "-O3", "-DNDEBUG", "-ffp-contract=off", "-fPIC", "-shared",
            str(native / "motion_tracker.cpp"), str(native / "third_party/lap/lapjv.cpp"),
            "-o", str(library),
        ],
        check=True,
    )
    return library


def greedy_matches(gt_boxes: np.ndarray, predictions: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    if len(gt_boxes) == 0 or len(predictions) == 0:
        return []
    iou = box_iou(torch.from_numpy(gt_boxes), torch.from_numpy(predictions[:, :4])).numpy()
    pairs = np.array(np.nonzero(iou >= threshold)).T
    if len(pairs) > 1:
        pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return [(int(gt), int(pred)) for gt, pred in pairs]


def count_boxes(frames: list[dict], selector) -> dict:
    counts = Counter(frames=len(frames), gt=sum(len(frame["gt_xyxy_class"]) for frame in frames))
    for frame in frames:
        gt = np.asarray([row[:4] for row in frame["gt_xyxy_class"]], dtype=np.float32).reshape(-1, 4)
        predictions = np.asarray(selector(frame), dtype=np.float32).reshape(-1, 5)
        matches = greedy_matches(gt, predictions)
        counts["tp"] += len(matches)
        counts["fp"] += len(predictions) - len(matches)
        counts["fn"] += len(gt) - len(matches)
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        **dict(counts),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def track_split(records: list[dict], spec: dict, library: Path) -> tuple[dict, list[dict]]:
    config = dict(CAUSAL_DEFAULTS, nominal_fps=spec["fps"])
    tracker = NativeMotion(library, fps=spec["fps"], config=config)
    gmc = EfficientGMC(320, 128, 5, True)
    tracked = []
    status_counts = Counter()
    matched_status = Counter()
    assigned_ids: set[int] = set()
    adjacent_id_changes = 0
    previous = None
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261015)
    try:
        for position, frame in enumerate(records):
            frame_index = int(frame["frame_index"])
            if position and frame_index - int(records[position - 1]["frame_index"]) != spec["step"]:
                raise ValueError(f"Non-causal frame order at {frame['path']}")
            image = cv2.imread(frame["path"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read {frame['path']}")
            warp = gmc.apply(image)
            all_predictions = np.asarray(frame["pred_xyxy_score_class"], dtype=np.float32).reshape(-1, 6)
            selected = all_predictions[all_predictions[:, 4] >= 0.03, :5]
            tracker.update(selected, warp, gmc.last_quality, frame_index / spec["fps"])
            observations = tracker.observations()
            if len(observations) != len(selected):
                raise RuntimeError("Tracker swallowed detector observations")
            for index, observation in enumerate(observations):
                expected = selected[index]
                actual = np.asarray([*observation["box"], observation["score"]], dtype=np.float32)
                if observation["detection_index"] != index or not np.array_equal(actual, expected):
                    raise RuntimeError("Tracker changed detector coordinates, score, or order")
                status_counts[observation["status"]] += 1
                if observation["id"] is not None:
                    assigned_ids.add(int(observation["id"]))
            gt = np.asarray([row[:4] for row in frame["gt_xyxy_class"]], dtype=np.float32).reshape(-1, 4)
            if len(gt) > 1:
                raise ValueError("Identity continuity audit needs at most one GT object per frame")
            matches = greedy_matches(gt, selected)
            current = None
            if matches:
                observation = observations[matches[0][1]]
                matched_status[observation["status"]] += 1
                current = observation["id"]
            if (previous is not None and current is not None and len(gt)
                    and frame_index - previous[0] == spec["step"] and int(current) != previous[1]):
                adjacent_id_changes += 1
            previous = (frame_index, int(current)) if current is not None and len(gt) else None
            tracked.append({**frame, "observations": observations})
    finally:
        tracker_stats = tracker.stats()
        tracker.close()
    lossless = count_boxes(
        tracked,
        lambda frame: [[*row["box"], row["score"]] for row in frame["observations"]],
    )
    confirmed = count_boxes(
        tracked,
        lambda frame: [
            [*row["box"], row["score"]] for row in frame["observations"] if row["confirmed"]
        ],
    )
    return {
        "lossless_box_metrics": lossless,
        "confirmed_only_box_metrics": confirmed,
        "observation_status": dict(status_counts),
        "matched_gt_status": dict(matched_status),
        "assigned_track_ids": len(assigned_ids),
        "adjacent_matched_id_changes": adjacent_id_changes,
        "tracker_stats": tracker_stats,
        "gmc_counts": gmc.counts,
    }, tracked


def run_model(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_spec = MODELS[args.model]
    weights = model_spec["weights"]
    if not weights.is_file():
        raise FileNotFoundError(weights)
    try:
        status(output, "preflight", model=args.model)
        SETTINGS.update(
            dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                 mlflow=False, neptune=False, raytune=False)
        )
        torch.set_num_threads(4)
        paths_by_split, manifests = {}, {}
        for split, spec in SPLITS.items():
            paths, manifest = native_paths(spec["data"])
            if len(paths) != spec["frames"] or len(paths) != len(set(paths)):
                raise ValueError(f"Unexpected native coverage for {split}")
            paths_by_split[split], manifests[split] = paths, manifest
        if set(paths_by_split["Video00004"]) & set(paths_by_split["Video00009"]):
            raise ValueError("Duplicate frames across source videos")
        library = build_motion_library(output)
        protocol = {
            "model": args.model,
            "label": model_spec["label"],
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "datasets": manifests,
            "sources": {name: {"path": str(spec["source"]), "sha256": sha256(spec["source"])}
                        for name, spec in SPLITS.items()},
            "pooled_frames": sum(spec["frames"] for spec in SPLITS.values()),
            "pooled_gt": sum(spec["gt"] for spec in SPLITS.values()),
            "input_hw": [544, 960],
            "dtype": "FP32",
            "conf_floor": 0.001,
            "operating_conf": 0.03,
            "nms_iou": 0.45,
            "matching_iou": 0.5,
            "max_det": 100,
            "tracking": "quality-gated EfficientGMC + native C++ causal global-assignment motion tracker",
            "tracking_output": "all current detector observations; no predicted boxes",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        }
        write_json(output / "protocol.json", protocol)
        split_entries, split_tracking = [], {}
        for split, spec in SPLITS.items():
            status(output, "detector", model=args.model, split=split)
            model = YOLO(str(weights))
            if len(model.model.model[-1].stride) != model_spec["strides"]:
                raise ValueError(f"Unexpected detection scales for {args.model}")
            metrics = model.val(
                data=str(spec["data"]), validator=CaptureValidator, imgsz=[544, 960], rect=False,
                device=str(args.device), batch=32, workers=4, conf=0.001, iou=0.45, max_det=100,
                half=False, plots=False, verbose=False, project=str(output / "val"), name=split,
            )
            native = {key: value for key, value in clean(metrics.gray_selection).items()
                      if key.startswith("native/")}
            if (native["native/c0.03/FRAMES"], native["native/c0.03/TP"] + native["native/c0.03/FN"]) != (
                spec["frames"], spec["gt"]
            ):
                raise ValueError(f"Frame or GT coverage changed for {split}")
            records = metrics.native_frame_records
            if [record["path"] for record in records] != paths_by_split[split]:
                raise ValueError(f"Validator changed frame order for {split}")
            gt_counts = [len(record["gt_xyxy_class"]) for record in records]
            if max(gt_counts, default=0) > 1 or sum(gt_counts) != spec["gt"]:
                raise ValueError(f"Unexpected GT identity coverage for {split}")
            with (output / f"{split}_detections.jsonl").open("x") as stream:
                for record in records:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            status(output, "tracking", model=args.model, split=split)
            tracking, tracked = track_split(records, spec, library)
            split_tracking[split] = tracking
            with (output / f"{split}_tracking.jsonl").open("x") as stream:
                for record in tracked:
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            split_entries.append(
                {"metrics": native, "arrays": metrics.native_pr_arrays, "scales": metrics.native_scale_counts}
            )
            np.savez_compressed(output / f"{split}_pr.npz", **metrics.native_pr_arrays)
            del model, metrics
            torch.cuda.empty_cache()
        pooled, arrays = pool_native(split_entries)
        np.savez_compressed(output / "pooled_pr.npz", **arrays)
        lossless = {
            key: sum(split_tracking[split]["lossless_box_metrics"][key] for split in SPLITS)
            for key in ("frames", "gt", "tp", "fp", "fn")
        }
        lossless["precision"] = lossless["tp"] / max(lossless["tp"] + lossless["fp"], 1)
        lossless["recall"] = lossless["tp"] / max(lossless["tp"] + lossless["fn"], 1)
        confirmed = {
            key: sum(split_tracking[split]["confirmed_only_box_metrics"][key] for split in SPLITS)
            for key in ("frames", "gt", "tp", "fp", "fn")
        }
        confirmed["precision"] = confirmed["tp"] / max(confirmed["tp"] + confirmed["fp"], 1)
        confirmed["recall"] = confirmed["tp"] / max(confirmed["tp"] + confirmed["fn"], 1)
        detector_counts = {key: pooled[f"native/c0.03/{key.upper()}"] for key in ("tp", "fp", "fn")}
        if any(int(lossless[key]) != int(detector_counts[key]) for key in detector_counts):
            raise RuntimeError("Lossless tracker output differs from detector metrics")
        result = {
            "model": args.model,
            "label": model_spec["label"],
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "pooled_detector": pooled,
            "pooled_tracking_lossless": {
                **lossless,
                "mAP50": pooled["native/mAP50"],
                "mAP50-95": pooled["native/mAP50-95"],
            },
            "pooled_tracking_confirmed_only": confirmed,
            "tracking_diagnostics": {
                "split_state_resets": len(SPLITS),
                "dropped_detector_boxes": 0,
                "changed_detector_boxes": 0,
                "adjacent_matched_id_changes": sum(
                    value["adjacent_matched_id_changes"] for value in split_tracking.values()
                ),
                "observation_status": dict(sum(
                    (Counter(value["observation_status"]) for value in split_tracking.values()), Counter()
                )),
                "matched_gt_status": dict(sum(
                    (Counter(value["matched_gt_status"]) for value in split_tracking.values()), Counter()
                )),
                "per_source_audit": split_tracking,
            },
        }
        write_json(output / "result.json", result)
        if sha256(weights) != protocol["weights_sha256"]:
            raise RuntimeError("Checkpoint changed during evaluation")
        for manifest in manifests.values():
            if sha256(manifest["list"]) != manifest["list_sha256"]:
                raise RuntimeError("Dataset manifest changed during evaluation")
        status(output, "complete", model=args.model, result=str(output / "result.json"))
    except BaseException as error:
        status(output, "failed", model=args.model, error=repr(error))
        raise


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def merge_results(args: argparse.Namespace) -> None:
    root = args.merge.resolve()
    results = {name: json.loads((root / name / "result.json").read_text()) for name in MODELS}
    protocols = {name: json.loads((root / name / "protocol.json").read_text()) for name in MODELS}
    dataset_hashes = {
        name: tuple(protocols[name]["datasets"][split]["list_sha256"] for split in SPLITS)
        for name in MODELS
    }
    if len(set(dataset_hashes.values())) != 1:
        raise RuntimeError("Models were not evaluated on identical dataset manifests")
    write_json(root / "results.json", results)
    lines = [
        "# Four-model pooled detector and tracker comparison",
        "",
        "Video00004 and Video00009 are treated as one pooled set: 3,780 frames and 1,245 GT boxes.",
        "No per-video metric is reported. Tracker state is reset only at the source-video boundary.",
        "FP32, fixed 960x544, conf=0.03, NMS IoU=0.45, GT matching IoU=0.50.",
        "AP is recomputed from globally ranked pooled predictions at conf floor 0.001.",
        "Tracking uses quality-gated GMC and the native C++ causal global-assignment tracker.",
        "The deployment output is the lossless observation stream: every current detector box is retained;",
        "pending or ambiguous identity is explicit and no predicted-only box is emitted.",
        "Video00009 was used for checkpoint selection, so this pooled result is not an independent test.",
        "",
        "| Model / output | Recall | Precision | TP | FP | FN | mAP50 | mAP50-95 | 4-8 px Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in MODELS:
        result = results[name]
        detector = result["pooled_detector"]
        q = "native/c0.03/"
        for output_name, metrics in (
            ("detector", {
                "recall": detector[q + "R"], "precision": detector[q + "P"],
                "tp": detector[q + "TP"], "fp": detector[q + "FP"], "fn": detector[q + "FN"],
                "mAP50": detector["native/mAP50"], "mAP50-95": detector["native/mAP50-95"],
            }),
            ("detector + GMC/Dist", result["pooled_tracking_lossless"]),
        ):
            lines.append(
                f"| {result['label']} / {output_name} | {percent(metrics['recall'])} | "
                f"{percent(metrics['precision'])} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
                f"{percent(metrics['mAP50'])} | {percent(metrics['mAP50-95'])} | "
                f"{percent(detector[q + 'long_4to8px/R'])} |"
            )
    lines += [
        "",
        "## Tracking audit",
        "",
        "| Model | Dropped boxes | Changed boxes | Adjacent matched-ID changes | Confirmed-only Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in MODELS:
        result = results[name]
        audit = result["tracking_diagnostics"]
        lines.append(
            f"| {result['label']} | {audit['dropped_detector_boxes']} | {audit['changed_detector_boxes']} | "
            f"{audit['adjacent_matched_id_changes']} | "
            f"{percent(result['pooled_tracking_confirmed_only']['recall'])} |"
        )
    lines += [
        "",
        "Confirmed-only is an audit column, not the deployment output. Consuming confirmed tracks alone",
        "suppresses pending/ambiguous detections; production must consume the lossless observation stream.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(root / "REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--merge", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.model:
        if args.output is None:
            parser.error("--output is required with --model")
        run_model(args)
    else:
        merge_results(args)


if __name__ == "__main__":
    main()
