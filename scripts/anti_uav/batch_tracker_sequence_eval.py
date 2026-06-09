#!/usr/bin/env python3
"""Batch replay evaluation for tracker_sequences folders.

The expected sequence layout is:

    <root>/<split>/<sequence>/frames.txt
    <root>/<split>/<sequence>/groundtruth.txt

`frames.txt` contains one image path per frame. Paths are remapped to
`--image-root/<split>/<basename>` when the recorded absolute path is not valid
on the current machine. `groundtruth.txt` stores one xywh box per frame.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

import cv2


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav import replay_eval  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics import solutions  # noqa: E402


DEFAULT_TRACKER_ROOT = Path("/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525/tracker_sequences")
DEFAULT_IMAGE_ROOT = Path("/mnt/hanlue/hanlue_multirotor_assets_phase1_20260525/images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Detector weights.")
    parser.add_argument("--tracker-root", type=Path, default=DEFAULT_TRACKER_ROOT, help="tracker_sequences root.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Fallback image root.")
    parser.add_argument(
        "--sequence-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Sequence parent root to evaluate directly. Can be repeated. "
            "When provided, --tracker-root/--split are ignored."
        ),
    )
    parser.add_argument("--split", default="val", help="Split folder under --tracker-root.")
    parser.add_argument("--output-root", required=True, help="Directory for per-sequence summaries.")
    parser.add_argument("--limit", type=int, default=0, help="Optional sequence cap.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-sequence summary.json files.")
    parser.add_argument("--save-video", action="store_true", help="Save one annotated replay mp4 per sequence.")
    parser.add_argument("--tracker", default="template_match", choices=solutions.available_trackers(), help="Tracker backend.")
    parser.add_argument("--nanotrack-root", default="", help="Optional upstream NanoTrack workspace root.")
    parser.add_argument("--nanotrack-config", default="", help="Optional NanoTrack config yaml path.")
    parser.add_argument("--nanotrack-snapshot", default="", help="Optional NanoTrack checkpoint path.")
    parser.add_argument("--nanotrack-device", default="", help="Optional NanoTrack torch device.")
    parser.add_argument(
        "--presence-verifier",
        default="",
        choices=("", "none", "heuristic", "mlp", "pair_head", "pair_head_edl"),
        help="Optional lightweight presence verifier.",
    )
    parser.add_argument("--presence-model", default="", help="Optional presence verifier checkpoint path.")
    parser.add_argument("--presence-device", default="", help="Optional torch device for presence verifier.")
    parser.add_argument("--presence-score-thresh", type=float, default=0.45)
    parser.add_argument("--presence-uncertainty-thresh", type=float, default=0.25)
    parser.add_argument("--presence-refresh-streak", type=int, default=2)
    parser.add_argument("--target-classes", default="drone,uav", help="Comma-separated class-name allowlist.")
    parser.add_argument("--conf", type=float, default=0.45, help="Detector confidence threshold.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="State-machine detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Detector input size.")
    parser.add_argument("--device", default=None, help="Torch device for detector inference.")
    parser.add_argument("--detector-assist-policy", default="edtc_like", choices=("granular", "edtc_like"))
    parser.add_argument("--detect-interval", type=int, default=2)
    parser.add_argument("--max-lost", type=int, default=30)
    parser.add_argument("--tracker-score-thresh", type=float, default=0.4)
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--input-mode", default="rgb", choices=("rgb", "gray", "ir"))
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--area-min-px", type=float, default=9.0)
    parser.add_argument("--area-max-ratio", type=float, default=0.25)
    parser.add_argument("--aspect-min", type=float, default=0.1)
    parser.add_argument("--aspect-max", type=float, default=10.0)
    parser.add_argument("--border-margin", type=int, default=1)
    parser.add_argument("--disable-roi-redetect", action="store_true")
    parser.add_argument("--disable-full-frame-fallback", action="store_true")
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--auto-confirm", action="store_true")
    parser.add_argument("--min-confirm-detections", type=int, default=2)
    return parser.parse_args()


def remap_frame_path(raw_path: str, image_root: Path, split: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    fallback = image_root / split / candidate.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Frame path not found: {raw_path} (fallback: {fallback})")


def read_frame_paths(sequence_dir: Path, image_root: Path, split: str) -> list[Path]:
    frames_path = sequence_dir / "frames.txt"
    if frames_path.exists():
        return [
            remap_frame_path(line.strip(), image_root, split)
            for line in frames_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    frames_dir = sequence_dir / "frames"
    if frames_dir.exists():
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        frame_paths = sorted(path.resolve() for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
        if frame_paths:
            return frame_paths

    raise FileNotFoundError(f"Missing frames.txt or frames/ images in {sequence_dir}")


def read_sequence(sequence_dir: Path, image_root: Path, split: str) -> tuple[list[Path], dict[int, Optional[tuple[float, float, float, float]]]]:
    groundtruth_path = sequence_dir / "groundtruth.txt"
    if not groundtruth_path.exists():
        raise FileNotFoundError(f"Missing groundtruth.txt in {sequence_dir}")

    frame_paths = read_frame_paths(sequence_dir, image_root, split)
    ground_truth: dict[int, Optional[tuple[float, float, float, float]]] = {}
    rows = [line.strip() for line in groundtruth_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(frame_paths) != len(rows):
        raise ValueError(f"Frame/groundtruth count mismatch in {sequence_dir}: {len(frame_paths)} vs {len(rows)}")
    for frame_index, raw in enumerate(rows, start=1):
        values = [float(part) for part in raw.replace(",", " ").split()]
        if len(values) < 4:
            raise ValueError(f"Invalid groundtruth row in {groundtruth_path}: {raw}")
        x, y, w, h = values[:4]
        ground_truth[frame_index] = (x, y, x + w, y + h) if w > 0 and h > 0 else None
    return frame_paths, ground_truth


def open_writer(path: Path, sample_frame, fps: float = 30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = sample_frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {path}")
    return writer


def evaluate_sequence(
    sequence_dir: Path,
    model,
    detector,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    frame_paths, ground_truth = read_sequence(sequence_dir, args.image_root, args.split)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]

    tracker_args = deepcopy(args)
    tracker = replay_eval.build_tracker(tracker_args)
    system = solutions.AntiUAVSystem(
        detector,
        tracker=tracker,
        presence_verifier=replay_eval.build_presence_verifier(args),
        detector_assist_policy=args.detector_assist_policy,
        presence_score_thresh=args.presence_score_thresh,
        presence_uncertainty_thresh=(None if args.presence_uncertainty_thresh < 0 else args.presence_uncertainty_thresh),
        presence_refresh_streak=args.presence_refresh_streak,
        detect_interval=args.detect_interval,
        max_lost=args.max_lost,
        tracker_score_thresh=args.tracker_score_thresh,
        min_confidence=args.min_confidence,
        roi_redetect=not args.disable_roi_redetect,
        full_frame_fallback=not args.disable_full_frame_fallback,
        manual_confirmation=not args.auto_confirm,
        min_confirm_detections=args.min_confirm_detections,
    )
    recorder = solutions.AlertRecorder(
        state_path=output_dir / "states.jsonl",
        alert_path=output_dir / "alerts.jsonl",
        crop_dir=None,
    )

    metrics = replay_eval.ReplayMetrics()
    writer = None
    error_path = output_dir / "errors.jsonl"
    error_file = error_path.open("w", encoding="utf-8")
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Unable to read frame: {frame_path}")
            metrics.total_frames += 1
            frame_index = metrics.total_frames
            gt_bbox = ground_truth.get(frame_index)
            if gt_bbox is not None:
                metrics.gt_present_frames += 1
                if metrics.first_gt_frame == 0:
                    metrics.first_gt_frame = frame_index

            state = system.step(frame)
            if args.auto_confirm and state.confirmation_state == "pending":
                system.confirm_current_target(True, note="offline_eval_auto_confirm")

            events = system.drain_alerts()
            recorder.record_state(state)
            recorder.record_events(frame, events)

            pred_bbox = system.bbox
            pred_present = pred_bbox is not None
            if pred_present:
                metrics.predicted_frames += 1

            iou = replay_eval.bbox_iou(pred_bbox, gt_bbox)
            match = pred_present and gt_bbox is not None and iou >= args.iou_thresh
            if match:
                metrics.tp_frames += 1
                metrics.matched_frames += 1
                metrics.iou_sum += iou
            elif pred_present and gt_bbox is None:
                metrics.fp_frames += 1
                error_file.write(json.dumps({"frame_index": frame_index, "type": "false_positive", "bbox": list(pred_bbox)}) + "\n")
            elif gt_bbox is not None and not pred_present:
                metrics.fn_frames += 1
                error_file.write(json.dumps({"frame_index": frame_index, "type": "false_negative", "gt_bbox": list(gt_bbox)}) + "\n")
            elif pred_present and gt_bbox is not None and not match:
                metrics.fp_frames += 1
                metrics.fn_frames += 1
                error_file.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "type": "localization_error",
                            "bbox": list(pred_bbox),
                            "gt_bbox": list(gt_bbox),
                            "iou": iou,
                        }
                    )
                    + "\n"
                )

            for event in events:
                if event.event_type == "alert_raised":
                    metrics.alert_raised += 1
                    if metrics.first_alert_frame == 0:
                        metrics.first_alert_frame = frame_index
                    if match:
                        metrics.alert_hit += 1

            if args.save_video:
                annotated = replay_eval.draw_gt(frame, gt_bbox)
                annotated = system.annotate(annotated, state)
                if writer is None:
                    writer = open_writer(output_dir / "replay.mp4", annotated)
                writer.write(annotated)
    finally:
        error_file.close()
        if writer is not None:
            writer.release()

    summary = metrics.as_dict()
    summary["sequence"] = sequence_dir.name
    replay_eval.write_json(output_dir / "summary.json", summary)
    return summary


def aggregate_metrics(sequence_summaries: list[dict]) -> dict:
    return replay_eval_batch_aggregate(sequence_summaries)


def json_safe(value):
    """Convert argparse values such as nested Path lists into JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def replay_eval_batch_aggregate(sequence_summaries: list[dict]) -> dict:
    totals = {
        "sequence_count": len(sequence_summaries),
        "total_frames": 0,
        "gt_present_frames": 0,
        "predicted_frames": 0,
        "tp_frames": 0,
        "fp_frames": 0,
        "fn_frames": 0,
        "matched_frames": 0,
        "iou_sum": 0.0,
        "alert_raised": 0,
        "alert_hit": 0,
    }
    for summary in sequence_summaries:
        for key in totals:
            if key != "sequence_count":
                totals[key] += summary.get(key, 0)
    tp, fp, fn = totals["tp_frames"], totals["fp_frames"], totals["fn_frames"]
    matched = totals["matched_frames"]
    totals["precision"] = tp / max(tp + fp, 1)
    totals["recall"] = tp / max(tp + fn, 1)
    totals["alert_precision"] = totals["alert_hit"] / max(totals["alert_raised"], 1)
    totals["avg_iou"] = totals["iou_sum"] / max(matched, 1)
    totals["mean_sequence_precision"] = sum(item["precision"] for item in sequence_summaries) / max(len(sequence_summaries), 1)
    totals["mean_sequence_recall"] = sum(item["recall"] for item in sequence_summaries) / max(len(sequence_summaries), 1)
    totals["mean_sequence_avg_iou"] = sum(item["avg_iou"] for item in sequence_summaries) / max(len(sequence_summaries), 1)
    return totals


def main() -> None:
    args = parse_args()
    args.tracker_root = args.tracker_root.expanduser().resolve()
    args.image_root = args.image_root.expanduser().resolve()
    args.sequence_root = [path.expanduser().resolve() for path in args.sequence_root]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_roots = args.sequence_root or [args.tracker_root / args.split]
    for sequence_root in sequence_roots:
        if not sequence_root.exists():
            raise FileNotFoundError(f"Sequence root does not exist: {sequence_root}")

    model = YOLO(args.model)
    detector = replay_eval.build_detector(model, args)
    sequence_summaries: list[dict] = []
    failures: list[dict] = []

    sequence_dirs: list[tuple[Path, Path]] = []
    for sequence_root in sequence_roots:
        sequence_dirs.extend((sequence_root, path) for path in sorted(item for item in sequence_root.iterdir() if item.is_dir()))

    for index, (sequence_root, sequence_dir) in enumerate(sequence_dirs, start=1):
        if args.limit and len(sequence_summaries) + len(failures) >= args.limit:
            break
        sequence_name = sequence_dir.name
        if len(sequence_roots) > 1:
            sequence_name = f"{sequence_root.parent.name}__{sequence_dir.name}"
        sequence_out = output_root / sequence_name
        summary_path = sequence_out / "summary.json"
        if args.skip_existing and summary_path.exists():
            sequence_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        sequence_out.mkdir(parents=True, exist_ok=True)
        try:
            summary = evaluate_sequence(sequence_dir, model, detector, args, sequence_out)
            summary["sequence_root"] = str(sequence_root)
            summary["sequence"] = sequence_name
            replay_eval.write_json(summary_path, summary)
            sequence_summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            failures.append({"sequence": sequence_name, "sequence_root": str(sequence_root), "error": repr(exc)})
            (sequence_out / "failure.log").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"[{index}] {sequence_name} done", flush=True)

    aggregate = {
        "tracker_root": str(args.tracker_root),
        "sequence_roots": [str(path) for path in sequence_roots],
        "image_root": str(args.image_root),
        "split": args.split,
        "model": args.model,
        "args": json_safe(vars(args)),
        "aggregate": aggregate_metrics(sequence_summaries),
        "sequence_summaries": sequence_summaries,
        "failures": failures,
    }
    replay_eval.write_json(output_root / "aggregate_summary.json", aggregate)
    print(json.dumps(aggregate["aggregate"], indent=2, ensure_ascii=False))
    if failures:
        print(f"{len(failures)} sequences failed. See {output_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
