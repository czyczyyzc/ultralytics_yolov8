#!/usr/bin/env python3
"""Export side-by-side tracking comparison videos from replay state logs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Sequence

import cv2
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav import batch_tracker_sequence_eval, replay_eval  # noqa: E402


@dataclass
class SequenceCandidate:
    dataset: str
    sequence: str
    frames: int
    left_precision: float
    left_recall: float
    left_avg_iou: float
    left_errors: int
    right_precision: float
    right_recall: float
    right_avg_iou: float
    right_errors: int
    recall_delta: float
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True, help="Evaluation root containing tracking/<model>/<dataset>.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to write comparison videos and manifests.")
    parser.add_argument("--left-model", default="960x960", help="Model shown on the left side.")
    parser.add_argument("--right-model", default="640x640", help="Model shown on the right side.")
    parser.add_argument("--datasets", nargs="+", default=["anti_uav", "hanlue_old", "hanlue_new"])
    parser.add_argument("--max-per-dataset", type=int, default=0, help="0 exports all failing sequences.")
    parser.add_argument("--min-error-frames", type=int, default=1, help="Minimum FP+FN in either model to export.")
    parser.add_argument("--min-recall-delta", type=float, default=0.0, help="Optional absolute recall-delta filter.")
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="Frame-level match IoU threshold.")
    parser.add_argument("--panel-width", type=int, default=960, help="Resize each panel to this width. 0 keeps source size.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional per-video frame cap for quick previews.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos that already exist.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_states(path: Path) -> dict[int, dict]:
    states: dict[int, dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        states[int(item["frame_index"])] = item
    return states


def bbox_iou(box1: Optional[Sequence[float]], box2: Optional[Sequence[float]]) -> float:
    return replay_eval.bbox_iou(box1, box2)


def error_count(summary: dict) -> int:
    return int(summary.get("fp_frames", 0)) + int(summary.get("fn_frames", 0))


def classify_reason(left: dict, right: dict, min_recall_delta: float) -> str:
    left_errors = error_count(left)
    right_errors = error_count(right)
    recall_delta = float(left.get("recall", 0.0)) - float(right.get("recall", 0.0))
    if recall_delta >= max(min_recall_delta, 0.15):
        return "left_covers_better"
    if recall_delta <= -max(min_recall_delta, 0.15):
        return "right_covers_better"
    if left_errors or right_errors:
        return "both_have_errors"
    return "no_failure"


def collect_candidates(args: argparse.Namespace) -> list[SequenceCandidate]:
    candidates: list[SequenceCandidate] = []
    tracking_root = args.eval_root / "tracking"
    for dataset in args.datasets:
        right_root = tracking_root / args.right_model / dataset
        left_root = tracking_root / args.left_model / dataset
        if not right_root.exists() or not left_root.exists():
            print(f"Skipping {dataset}: missing {left_root} or {right_root}", file=sys.stderr)
            continue
        dataset_rows: list[SequenceCandidate] = []
        for right_seq_dir in sorted(path for path in right_root.iterdir() if path.is_dir()):
            sequence = right_seq_dir.name
            left_summary_path = left_root / sequence / "summary.json"
            right_summary_path = right_seq_dir / "summary.json"
            if not left_summary_path.exists() or not right_summary_path.exists():
                continue
            left = load_json(left_summary_path)
            right = load_json(right_summary_path)
            left_errors = error_count(left)
            right_errors = error_count(right)
            recall_delta = float(left.get("recall", 0.0)) - float(right.get("recall", 0.0))
            if max(left_errors, right_errors) < args.min_error_frames:
                continue
            if args.min_recall_delta and abs(recall_delta) < args.min_recall_delta:
                continue
            reason = classify_reason(left, right, args.min_recall_delta)
            dataset_rows.append(
                SequenceCandidate(
                    dataset=dataset,
                    sequence=sequence,
                    frames=int(right.get("total_frames", left.get("total_frames", 0))),
                    left_precision=float(left.get("precision", 0.0)),
                    left_recall=float(left.get("recall", 0.0)),
                    left_avg_iou=float(left.get("avg_iou", 0.0)),
                    left_errors=left_errors,
                    right_precision=float(right.get("precision", 0.0)),
                    right_recall=float(right.get("recall", 0.0)),
                    right_avg_iou=float(right.get("avg_iou", 0.0)),
                    right_errors=right_errors,
                    recall_delta=recall_delta,
                    reason=reason,
                )
            )
        dataset_rows.sort(key=lambda row: (max(row.left_errors, row.right_errors), abs(row.recall_delta)), reverse=True)
        if args.max_per_dataset:
            dataset_rows = dataset_rows[: args.max_per_dataset]
        candidates.extend(dataset_rows)
    return candidates


def resolve_frames_and_gt(eval_root: Path, model: str, dataset: str, sequence: str) -> tuple[list[Path] | Path, dict[int, Optional[tuple[float, float, float, float]]]]:
    seq_dir = eval_root / "tracking" / model / dataset / sequence
    summary = load_json(seq_dir / "summary.json")
    if dataset == "anti_uav":
        video_path = Path(summary["video"])
        annotations = Path(summary["annotations"])
        return video_path, replay_eval.load_anti_uav_annotations(annotations)

    sequence_root = Path(summary["sequence_root"])
    sequence_dir = sequence_root / sequence
    frame_paths, gt = batch_tracker_sequence_eval.read_sequence(sequence_dir, Path("/"), "val")
    return frame_paths, gt


def read_video_frames(video_path: Path, max_frames: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            yield index, frame
            if max_frames and index >= max_frames:
                break
    finally:
        cap.release()


def read_image_frames(frame_paths: list[Path], max_frames: int = 0):
    for index, path in enumerate(frame_paths, start=1):
        if max_frames and index > max_frames:
            break
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Unable to read frame: {path}")
        yield index, frame


def draw_box(frame: np.ndarray, bbox: Optional[Sequence[float]], color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def put_label(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def frame_status(state: dict | None, gt_bbox: Optional[Sequence[float]], iou_thresh: float) -> tuple[str, float]:
    pred = state.get("bbox") if state else None
    iou = bbox_iou(pred, gt_bbox)
    if pred is not None and gt_bbox is not None and iou >= iou_thresh:
        return "TP", iou
    if pred is None and gt_bbox is not None:
        return "FN", iou
    if pred is not None and gt_bbox is None:
        return "FP", iou
    if pred is not None and gt_bbox is not None:
        return "LOC", iou
    return "TN", iou


def annotate_panel(
    frame: np.ndarray,
    model: str,
    summary: dict,
    state: dict | None,
    gt_bbox: Optional[Sequence[float]],
    frame_index: int,
    iou_thresh: float,
) -> np.ndarray:
    panel = frame.copy()
    draw_box(panel, gt_bbox, (0, 255, 0), "GT", 2)
    pred = state.get("bbox") if state else None
    status, iou = frame_status(state, gt_bbox, iou_thresh)
    pred_color = (255, 180, 0) if status == "TP" else (0, 0, 255)
    draw_box(panel, pred, pred_color, f"{model} {status}", 2)
    put_label(panel, f"{model}  P={summary.get('precision', 0):.3f} R={summary.get('recall', 0):.3f} IoU={summary.get('avg_iou', 0):.3f}", 24)
    conf = state.get("confidence", 0.0) if state else 0.0
    track = state.get("track_score", 0.0) if state else 0.0
    presence = state.get("presence_score", 0.0) if state else 0.0
    put_label(panel, f"frame={frame_index} status={status} iou={iou:.2f} conf={conf:.2f} track={track:.2f} pres={presence:.2f}", 52)
    return panel


def resize_panel(frame: np.ndarray, panel_width: int) -> np.ndarray:
    if panel_width <= 0 or frame.shape[1] == panel_width:
        return frame
    scale = panel_width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (panel_width, height), interpolation=cv2.INTER_AREA)


def open_writer(path: Path, frame: np.ndarray, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {path}")
    return writer


def export_one(args: argparse.Namespace, candidate: SequenceCandidate) -> Path:
    out_path = args.output_root / candidate.dataset / candidate.reason / f"{candidate.sequence}.mp4"
    if args.skip_existing and out_path.exists():
        return out_path

    left_dir = args.eval_root / "tracking" / args.left_model / candidate.dataset / candidate.sequence
    right_dir = args.eval_root / "tracking" / args.right_model / candidate.dataset / candidate.sequence
    left_summary = load_json(left_dir / "summary.json")
    right_summary = load_json(right_dir / "summary.json")
    left_states = load_states(left_dir / "states.jsonl")
    right_states = load_states(right_dir / "states.jsonl")
    frames_source, gt = resolve_frames_and_gt(args.eval_root, args.right_model, candidate.dataset, candidate.sequence)
    frame_iter = read_video_frames(frames_source, args.max_frames) if isinstance(frames_source, Path) else read_image_frames(frames_source, args.max_frames)

    writer = None
    try:
        for frame_index, frame in frame_iter:
            gt_bbox = gt.get(frame_index)
            left = annotate_panel(frame, args.left_model, left_summary, left_states.get(frame_index), gt_bbox, frame_index, args.iou_thresh)
            right = annotate_panel(frame, args.right_model, right_summary, right_states.get(frame_index), gt_bbox, frame_index, args.iou_thresh)
            left = resize_panel(left, args.panel_width)
            right = resize_panel(right, args.panel_width)
            if left.shape[0] != right.shape[0]:
                target_h = min(left.shape[0], right.shape[0])
                left = cv2.resize(left, (left.shape[1], target_h), interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, (right.shape[1], target_h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([left, right], axis=1)
            if writer is None:
                writer = open_writer(out_path, combined, args.fps)
            writer.write(combined)
    finally:
        if writer is not None:
            writer.release()
    return out_path


def write_manifest(path: Path, rows: list[SequenceCandidate], video_paths: dict[tuple[str, str], Path] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "sequence",
        "frames",
        "reason",
        "left_precision",
        "left_recall",
        "left_avg_iou",
        "left_errors",
        "right_precision",
        "right_recall",
        "right_avg_iou",
        "right_errors",
        "recall_delta",
        "video",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            data = row.__dict__.copy()
            data["video"] = str(video_paths.get((row.dataset, row.sequence), "")) if video_paths else ""
            writer.writerow(data)


def main() -> None:
    args = parse_args()
    args.eval_root = args.eval_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(args)
    write_manifest(args.output_root / "selected_sequences.csv", candidates)
    video_paths: dict[tuple[str, str], Path] = {}
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate.dataset}/{candidate.sequence} {candidate.reason}", flush=True)
        video_paths[(candidate.dataset, candidate.sequence)] = export_one(args, candidate)
    write_manifest(args.output_root / "export_manifest.csv", candidates, video_paths)
    print(f"Exported {len(video_paths)} videos to {args.output_root}")


if __name__ == "__main__":
    main()
