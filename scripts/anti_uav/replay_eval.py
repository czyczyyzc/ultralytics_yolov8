#!/usr/bin/env python3
"""Replay and evaluate the alerting-only anti-UAV pipeline on annotated video."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO, solutions


DEFAULT_PRESENCE_MODEL_ENV = "ANTI_UAV_DEFAULT_PRESENCE_MODEL"
DEFAULT_PRESENCE_MODEL_NAME = "pair_presence_edl.pt"


def resolve_default_presence_model() -> Path | None:
    """Find the current default pair-head presence verifier checkpoint."""
    candidates: list[Path] = []
    env_path = os.getenv(DEFAULT_PRESENCE_MODEL_ENV, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    runs_root = ROOT / "runs" / "anti_uav"
    preferred = runs_root / "presence_pair_trainonly24_a52f825_model" / DEFAULT_PRESENCE_MODEL_NAME
    candidates.append(preferred)
    if runs_root.exists():
        dynamic = sorted(
            runs_root.glob(f"presence_pair*_model/{DEFAULT_PRESENCE_MODEL_NAME}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(dynamic)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


@dataclass
class ReplayMetrics:
    total_frames: int = 0
    gt_present_frames: int = 0
    predicted_frames: int = 0
    tp_frames: int = 0
    fp_frames: int = 0
    fn_frames: int = 0
    matched_frames: int = 0
    iou_sum: float = 0.0
    alert_raised: int = 0
    alert_hit: int = 0
    first_gt_frame: int = 0
    first_alert_frame: int = 0

    def as_dict(self) -> dict:
        precision = self.tp_frames / max(self.tp_frames + self.fp_frames, 1)
        recall = self.tp_frames / max(self.tp_frames + self.fn_frames, 1)
        alert_precision = self.alert_hit / max(self.alert_raised, 1)
        avg_iou = self.iou_sum / max(self.matched_frames, 1)
        return {
            **asdict(self),
            "precision": precision,
            "recall": recall,
            "alert_precision": alert_precision,
            "avg_iou": avg_iou,
            "time_to_first_alert_frames": (
                self.first_alert_frame - self.first_gt_frame if self.first_gt_frame and self.first_alert_frame else None
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path.")
    parser.add_argument(
        "--dataset-format",
        default="anti-uav-json",
        choices=("anti-uav-json", "drone-vs-bird-txt", "jsonl-bbox"),
        help="Ground-truth annotation format.",
    )
    parser.add_argument("--video", default="", help="Video path. Optional when using --sequence-root with Anti-UAV.")
    parser.add_argument("--annotations", default="", help="Annotation file path.")
    parser.add_argument("--sequence-root", default="", help="Sequence directory for Anti-UAV style datasets.")
    parser.add_argument("--modality", default="rgb", choices=("rgb", "ir", "auto"), help="Anti-UAV modality selector.")
    parser.add_argument("--tracker", default="template_match", choices=solutions.available_trackers(), help="Tracker backend.")
    parser.add_argument(
        "--opencv-tracker-type",
        default="csrt",
        help="OpenCV tracker type when --tracker opencv is selected, for example mil.",
    )
    parser.add_argument("--nanotrack-root", default="", help="Optional upstream NanoTrack workspace root.")
    parser.add_argument("--nanotrack-config", default="", help="Optional NanoTrack config yaml path.")
    parser.add_argument("--nanotrack-snapshot", default="", help="Optional NanoTrack checkpoint path.")
    parser.add_argument("--nanotrack-device", default="", help="Optional NanoTrack torch device, for example cpu or 0.")
    parser.add_argument(
        "--presence-verifier",
        default="",
        choices=("", "none", "heuristic", "mlp", "pair_head", "pair_head_edl"),
        help="Optional lightweight presence verifier over tracker outputs. Defaults to pair_head_edl when a default checkpoint is available; use 'none' to disable it explicitly.",
    )
    parser.add_argument("--presence-model", default="", help="Optional MLPPresenceVerifier checkpoint path.")
    parser.add_argument("--presence-device", default="", help="Optional torch device for the presence verifier.")
    parser.add_argument(
        "--presence-score-thresh",
        type=float,
        default=0.45,
        help="Presence score threshold below which the tracker is treated as suspect.",
    )
    parser.add_argument(
        "--presence-uncertainty-thresh",
        type=float,
        default=0.25,
        help="Optional uncertainty threshold for evidential presence verifiers. Negative disables it.",
    )
    parser.add_argument(
        "--presence-refresh-streak",
        type=int,
        default=2,
        help="Require this many low-presence frames before forcing detector refresh.",
    )
    parser.add_argument("--target-classes", default="drone,uav", help="Comma-separated class-name allowlist.")
    parser.add_argument("--conf", type=float, default=0.45, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Detector input size.")
    parser.add_argument("--device", default=None, help="Torch device for detector inference, for example 0 or cpu.")
    parser.add_argument("--detect-interval", type=int, default=2, help="Run detector every N frames while tracking.")
    parser.add_argument("--max-lost", type=int, default=30, help="Frames to wait before dropping a lost target.")
    parser.add_argument(
        "--tracker-score-thresh",
        type=float,
        default=0.4,
        help="Minimum acceptable tracker score before forcing detector recovery.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        help="Minimum detector confidence accepted by the single-target state machine.",
    )
    parser.add_argument("--tile-size", type=int, default=0, help="Enable tiled detection with square tile size.")
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap ratio.")
    parser.add_argument("--input-mode", default="rgb", choices=("rgb", "gray", "ir"), help="Detector preprocessing mode.")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE in gray/IR preprocessing.")
    parser.add_argument("--area-min-px", type=float, default=9.0, help="Reject detections smaller than this area.")
    parser.add_argument(
        "--area-max-ratio",
        type=float,
        default=0.25,
        help="Reject detections covering more than this fraction of the frame.",
    )
    parser.add_argument("--aspect-min", type=float, default=0.1, help="Minimum bbox aspect ratio kept by the filter.")
    parser.add_argument("--aspect-max", type=float, default=10.0, help="Maximum bbox aspect ratio kept by the filter.")
    parser.add_argument("--border-margin", type=int, default=1, help="Reject detections sitting on the border margin.")
    parser.add_argument(
        "--disable-roi-redetect",
        action="store_true",
        help="Disable ROI-based re-detection around the tracked target.",
    )
    parser.add_argument(
        "--disable-full-frame-fallback",
        action="store_true",
        help="Do not fall back to full-frame detector passes after ROI re-detect misses.",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="IoU threshold for frame-level matches.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means no limit.")
    parser.add_argument("--auto-confirm", action="store_true", help="Auto-confirm pending targets during offline replay.")
    parser.add_argument(
        "--min-confirm-detections",
        type=int,
        default=2,
        help="Require this many detector-backed hits before a target can enter pending/confirmed review state.",
    )
    parser.add_argument("--save-video", default="", help="Optional annotated replay output video path.")
    parser.add_argument("--summary-json", default="", help="Optional path to write replay metrics as JSON.")
    parser.add_argument("--error-log", default="", help="Optional JSONL file for FP/FN/alert review records.")
    parser.add_argument("--state-log", default="", help="Optional state JSONL output path.")
    parser.add_argument("--alert-log", default="", help="Optional alert event JSONL output path.")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve video and annotation paths from direct inputs or an Anti-UAV sequence directory."""
    if args.sequence_root:
        return resolve_anti_uav_sequence(Path(args.sequence_root), args.modality)
    if not args.video or not args.annotations:
        raise ValueError("Either provide --video and --annotations, or use --sequence-root for Anti-UAV sequences.")
    return Path(args.video).expanduser().resolve(), Path(args.annotations).expanduser().resolve()


def resolve_anti_uav_sequence(sequence_root: Path, modality: str) -> tuple[Path, Path]:
    """Locate the most likely video/json pair inside an Anti-UAV sequence directory."""
    sequence_root = sequence_root.expanduser().resolve()
    if not sequence_root.exists():
        raise FileNotFoundError(f"Sequence root does not exist: {sequence_root}")

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    json_candidates = list(sequence_root.rglob("*.json"))
    video_candidates = [path for path in sequence_root.rglob("*") if path.suffix.lower() in video_extensions]
    if not json_candidates or not video_candidates:
        raise FileNotFoundError(f"Could not find both video and annotation files under: {sequence_root}")

    def score(path: Path, token: str) -> int:
        lower = path.name.lower()
        if token == "auto":
            return 1
        if token in lower:
            return 3
        if token == "rgb" and "ir" not in lower:
            return 2
        return 0

    token = modality.lower()
    video_path = sorted(video_candidates, key=lambda path: (score(path, token), path.name), reverse=True)[0]
    annotation_path = sorted(json_candidates, key=lambda path: (score(path, token), path.name), reverse=True)[0]
    return video_path, annotation_path


def build_detector(model, args: argparse.Namespace):
    class_names = [name.strip() for name in args.target_classes.split(",") if name.strip()]
    filters = [
        solutions.AreaFilter(min_area_px=args.area_min_px, max_area_ratio=args.area_max_ratio),
        solutions.AspectRatioFilter(min_ratio=args.aspect_min, max_ratio=args.aspect_max),
        solutions.BorderFilter(margin_px=args.border_margin),
    ]
    return solutions.YOLODetectionAdapter(
        model,
        class_names=class_names or None,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        tile_size=args.tile_size if args.tile_size > 0 else None,
        tile_overlap=args.tile_overlap,
        enable_tiling=args.tile_size > 0,
        preprocess_mode=args.input_mode,
        clahe=args.clahe,
        filters=filters,
    )


def build_tracker(args: argparse.Namespace):
    """Instantiate an optional tracker object when extra backend configuration is required."""
    if args.tracker == "opencv":
        return solutions.build_tracker("opencv", tracker_type=args.opencv_tracker_type)
    if args.tracker != "nanotrack":
        return args.tracker
    return solutions.build_tracker(
        "nanotrack",
        nanotrack_root=args.nanotrack_root or None,
        config_path=args.nanotrack_config or None,
        snapshot_path=args.nanotrack_snapshot or None,
        device=args.nanotrack_device or None,
        score_threshold=args.tracker_score_thresh,
    )


def build_presence_verifier(args: argparse.Namespace):
    """Instantiate an optional lightweight tracker-presence verifier."""
    verifier_name = (args.presence_verifier or "").strip()
    checkpoint_path = (args.presence_model or "").strip()
    if verifier_name.lower() == "none":
        return None
    if not verifier_name:
        default_checkpoint = resolve_default_presence_model()
        if default_checkpoint is None:
            return None
        verifier_name = "pair_head_edl"
        checkpoint_path = str(default_checkpoint)

    if verifier_name in {"mlp", "pair_head", "pair_head_edl"}:
        if not checkpoint_path:
            raise ValueError("--presence-model is required when --presence-verifier is mlp/pair_head/pair_head_edl.")
        return solutions.build_presence_verifier(
            verifier_name,
            checkpoint_path=checkpoint_path,
            device=args.presence_device or None,
        )
    return solutions.build_presence_verifier("heuristic")


def load_ground_truth(annotation_path: Path, dataset_format: str) -> dict[int, Optional[tuple[float, float, float, float]]]:
    """Load frame-indexed ground-truth boxes."""
    if dataset_format == "anti-uav-json":
        return load_anti_uav_annotations(annotation_path)
    if dataset_format == "drone-vs-bird-txt":
        return load_drone_vs_bird_annotations(annotation_path)
    if dataset_format == "jsonl-bbox":
        return load_bbox_jsonl(annotation_path)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def load_anti_uav_annotations(annotation_path: Path) -> dict[int, Optional[tuple[float, float, float, float]]]:
    """Parse Anti-UAV style JSON annotations with broad schema tolerance."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    frames = {}

    if isinstance(data, dict) and "gt_rect" in data:
        rects = data.get("gt_rect", [])
        exists = data.get("exist") or data.get("exists") or data.get("existence")
        for index, rect in enumerate(rects, start=1):
            present = bool(exists[index - 1]) if exists and index - 1 < len(exists) else _rect_present(rect)
            frames[index] = _xywh_to_xyxy(rect) if present else None
        return frames

    if isinstance(data, dict):
        def sort_key(item):
            try:
                return int(str(item[0]).split(".")[0])
            except ValueError:
                return 0

        iterable = sorted(data.items(), key=sort_key)
        for raw_index, item in iterable:
            try:
                frame_index = int(str(raw_index).split(".")[0])
            except ValueError:
                continue
            frames[frame_index] = _parse_generic_annotation_entry(item)
        return frames

    if isinstance(data, list):
        for index, item in enumerate(data, start=1):
            frames[index] = _parse_generic_annotation_entry(item)
        return frames

    raise ValueError(f"Unsupported Anti-UAV annotation schema: {annotation_path}")


def load_drone_vs_bird_annotations(annotation_path: Path) -> dict[int, Optional[tuple[float, float, float, float]]]:
    """Parse Drone-vs-Bird custom txt annotations."""
    frames = {}
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        frame_index = int(parts[0])
        count = int(parts[1])
        if count <= 0 or len(parts) < 6:
            frames[frame_index] = None
            continue
        x, y, w, h = map(float, parts[2:6])
        frames[frame_index] = (x, y, x + w, y + h)
    return frames


def load_bbox_jsonl(annotation_path: Path) -> dict[int, Optional[tuple[float, float, float, float]]]:
    """Parse generic JSONL records containing frame_index and bbox."""
    frames = {}
    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        frame_index = int(record["frame_index"])
        bbox = record.get("bbox")
        frames[frame_index] = tuple(map(float, bbox)) if bbox else None
    return frames


def _parse_generic_annotation_entry(item) -> Optional[tuple[float, float, float, float]]:
    """Extract a bbox from a permissive dict/list annotation entry."""
    if item is None or item is False or item == 0 or item == []:
        return None
    if isinstance(item, dict):
        if any(key in item for key in ("exist", "exists", "visible")):
            exists = item.get("exist", item.get("exists", item.get("visible", True)))
            if not exists:
                return None
        if "bbox" in item:
            bbox = item["bbox"]
            return _xywh_to_xyxy(bbox) if len(bbox) == 4 and bbox[2] >= 0 and bbox[3] >= 0 else None
        if "gt_rect" in item:
            return _xywh_to_xyxy(item["gt_rect"])
        if all(key in item for key in ("x", "y", "w", "h")):
            return _xywh_to_xyxy((item["x"], item["y"], item["w"], item["h"]))
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 4:
        return _xywh_to_xyxy(item[:4])
    return None


def _xywh_to_xyxy(box: Sequence[float]) -> Optional[tuple[float, float, float, float]]:
    """Convert an xywh box to xyxy, returning None for empty boxes."""
    if not box or len(box) < 4:
        return None
    x, y, w, h = [float(value) for value in box[:4]]
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def _rect_present(box: Sequence[float]) -> bool:
    return bool(box and len(box) >= 4 and float(box[2]) > 0 and float(box[3]) > 0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def draw_gt(frame: np.ndarray, bbox: Optional[tuple[float, float, float, float]]) -> np.ndarray:
    """Overlay ground-truth bbox for visual replay."""
    annotated = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, "gt", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return annotated


def bbox_iou(box1: Optional[Sequence[float]], box2: Optional[Sequence[float]]) -> float:
    if box1 is None or box2 is None:
        return 0.0
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 0.0)
    area2 = max((box2[2] - box2[0]) * (box2[3] - box2[1]), 0.0)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    video_path, annotation_path = resolve_inputs(args)
    ground_truth = load_ground_truth(annotation_path, args.dataset_format)

    model = YOLO(args.model)
    detector = build_detector(model, args)
    system = solutions.AntiUAVSystem(
        detector,
        tracker=build_tracker(args),
        presence_verifier=build_presence_verifier(args),
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
        state_path=args.state_log or None,
        alert_path=args.alert_log or None,
        crop_dir=None,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    writer = None
    error_file = None
    if args.error_log:
        error_path = Path(args.error_log).expanduser().resolve()
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_file = open(error_path, "w", encoding="utf-8")

    metrics = ReplayMetrics()
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

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

            iou = bbox_iou(pred_bbox, gt_bbox)
            match = pred_present and gt_bbox is not None and iou >= args.iou_thresh
            if match:
                metrics.tp_frames += 1
                metrics.matched_frames += 1
                metrics.iou_sum += iou
            elif pred_present and gt_bbox is None:
                metrics.fp_frames += 1
                if error_file is not None:
                    error_file.write(json.dumps({"frame_index": frame_index, "type": "false_positive", "bbox": list(pred_bbox)}) + "\n")
            elif gt_bbox is not None and not pred_present:
                metrics.fn_frames += 1
                if error_file is not None:
                    error_file.write(json.dumps({"frame_index": frame_index, "type": "false_negative", "gt_bbox": list(gt_bbox)}) + "\n")
            elif pred_present and gt_bbox is not None and not match:
                metrics.fp_frames += 1
                metrics.fn_frames += 1
                if error_file is not None:
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
                    elif error_file is not None:
                        error_file.write(
                            json.dumps(
                                {
                                    "frame_index": frame_index,
                                    "type": "bad_alert",
                                    "bbox": list(pred_bbox) if pred_bbox else None,
                                    "gt_bbox": list(gt_bbox) if gt_bbox else None,
                                    "iou": iou,
                                }
                            )
                            + "\n"
                        )

            if args.save_video:
                if writer is None:
                    output_path = Path(args.save_video).expanduser().resolve()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                annotated = system.annotate(frame)
                annotated = draw_gt(annotated, gt_bbox)
                writer.write(annotated)

            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if error_file is not None:
            error_file.close()
        recorder.close()

    summary = metrics.as_dict()
    summary["video"] = str(video_path)
    summary["annotations"] = str(annotation_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)


if __name__ == "__main__":
    main()
