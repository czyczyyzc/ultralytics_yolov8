#!/usr/bin/env python3
"""Export template/current ROI verifier samples from replay state logs and source video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.replay_eval import load_ground_truth
from ultralytics import solutions
from ultralytics.solutions import anti_uav as anti_uav_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Source video used during replay.")
    parser.add_argument("--annotations", type=Path, required=True, help="Ground-truth annotation file.")
    parser.add_argument("--states-jsonl", type=Path, required=True, help="Replay state log from AntiUAVSystem.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for verifier crops and manifest.")
    parser.add_argument("--sequence-name", default="", help="Optional sequence name stored in the manifest.")
    parser.add_argument(
        "--dataset-format",
        default="anti-uav-json",
        choices=("anti-uav-json", "drone-vs-bird-txt", "jsonl-bbox"),
        help="Ground-truth annotation format.",
    )
    parser.add_argument("--patch-scale", type=float, default=1.2, help="Patch expansion scale around each bbox.")
    parser.add_argument("--patch-size", type=int, default=64, help="Saved patch side length.")
    parser.add_argument("--positive-iou", type=float, default=0.5, help="Positive-label IoU threshold.")
    parser.add_argument("--negative-iou", type=float, default=0.2, help="Negative-label IoU threshold.")
    parser.add_argument(
        "--anchor-statuses",
        default="detected,reacquired",
        help="Comma-separated states that refresh the template anchor.",
    )
    return parser.parse_args()


def read_state_rows(path: Path) -> list[dict]:
    """Load replay state rows."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    rows.sort(key=lambda row: int(row.get("frame_index", 0)))
    return rows


def bbox_iou(box1: list[float], box2: tuple[float, float, float, float]) -> float:
    """Compute IoU between logged xyxy boxes and GT xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(value) for value in box1[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box2[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area2 = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = max(area1 + area2 - inter, 1e-6)
    return float(inter / union)


def normalize_patch(frame, bbox, patch_scale: float, patch_size: int):
    """Extract a fixed-size grayscale patch."""
    patch = anti_uav_solution._extract_patch(frame, bbox, patch_scale)
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return cv2.resize(gray, (patch_size, patch_size), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    template_dir = args.output_root / "templates"
    search_dir = args.output_root / "search"
    template_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"

    anchor_statuses = {item.strip().lower() for item in args.anchor_statuses.split(",") if item.strip()}
    sequence_name = args.sequence_name or args.states_jsonl.parent.name
    feature_names = tuple(solutions.HeuristicPresenceVerifier.feature_names)
    state_rows = read_state_rows(args.states_jsonl.expanduser().resolve())
    ground_truth = load_ground_truth(args.annotations.expanduser().resolve(), args.dataset_format)

    frames_by_index = {int(row["frame_index"]): row for row in state_rows}
    if not frames_by_index:
        raise ValueError(f"No state rows found in {args.states_jsonl}")

    cap = cv2.VideoCapture(str(args.video.expanduser().resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    anchor_patch = None
    anchor_ref = ""
    counts = {"written": 0, "positive": 0, "negative": 0, "skipped": 0}
    frame_index = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame_index += 1
                row = frames_by_index.get(frame_index)
                if row is None:
                    continue

                bbox = row.get("bbox")
                if not bbox:
                    counts["skipped"] += 1
                    continue
                clipped_bbox = anti_uav_solution._clip_bbox(bbox, frame.shape)
                status = str(row.get("status", "")).lower()
                if anchor_patch is None or status in anchor_statuses:
                    template = normalize_patch(frame, clipped_bbox, args.patch_scale, args.patch_size)
                    if template is None:
                        counts["skipped"] += 1
                        continue
                    anchor_ref = f"{sequence_name}_anchor_{frame_index:06d}.png"
                    cv2.imwrite(str(template_dir / anchor_ref), template)
                    anchor_patch = template

                if anchor_patch is None:
                    counts["skipped"] += 1
                    continue

                search_patch = normalize_patch(frame, clipped_bbox, args.patch_scale, args.patch_size)
                if search_patch is None:
                    counts["skipped"] += 1
                    continue

                gt_bbox = ground_truth.get(frame_index)
                label = None
                iou = 0.0
                if gt_bbox is None:
                    label = 0
                else:
                    iou = bbox_iou(bbox, gt_bbox)
                    if iou >= args.positive_iou:
                        label = 1
                    elif iou <= args.negative_iou:
                        label = 0

                if label is None:
                    counts["skipped"] += 1
                    continue

                search_ref = f"{sequence_name}_{frame_index:06d}.png"
                cv2.imwrite(str(search_dir / search_ref), search_patch)
                payload = {
                    "sequence": sequence_name,
                    "frame_index": frame_index,
                    "status": row.get("status", ""),
                    "label": int(label),
                    "iou": float(iou),
                    "template_path": str((template_dir / anchor_ref).resolve()),
                    "search_path": str((search_dir / search_ref).resolve()),
                    "track_score": float(row.get("track_score", 0.0)),
                    "presence_score": float(row.get("presence_score", 0.0)),
                    "presence_uncertainty": float(row.get("presence_uncertainty", 0.0)),
                    "features": {
                        name: float((row.get("presence_features") or {}).get(name, 0.0))
                        for name in feature_names
                    },
                }
                manifest.write(json.dumps(payload, ensure_ascii=False) + "\n")
                counts["written"] += 1
                counts["positive" if label == 1 else "negative"] += 1
        finally:
            cap.release()

    print(json.dumps({"sequence": sequence_name, "manifest": str(manifest_path), **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
