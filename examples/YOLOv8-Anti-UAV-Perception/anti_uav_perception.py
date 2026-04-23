from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an alerting-only anti-UAV perception pipeline on video.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path.")
    parser.add_argument("--source", required=True, help="Video path or camera index.")
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
        help="Presence score threshold below which tracking is treated as suspect.",
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
    parser.add_argument(
        "--detector-assist-policy",
        default="granular",
        choices=("granular", "edtc_like"),
        help="Detector-assisted tracking policy. 'granular' uses the current multi-stage fusion logic, while 'edtc_like' mimics anti_uav_edtc_jit style detect/search and track/until-uncertain switching.",
    )
    parser.add_argument("--detect-interval", type=int, default=2, help="Run detector every N frames while tracking.")
    parser.add_argument("--max-lost", type=int, default=30, help="Frames to wait before dropping a lost target.")
    parser.add_argument(
        "--min-confirm-detections",
        type=int,
        default=2,
        help="Require this many detector-backed hits before a target enters pending/confirmed review state.",
    )
    parser.add_argument("--input-mode", default="rgb", choices=("rgb", "gray", "ir"), help="Input preprocessing mode.")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE in gray/IR preprocessing.")
    parser.add_argument("--tile-size", type=int, default=0, help="Enable tiled detection with square tile size.")
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="Tile overlap ratio.")
    parser.add_argument("--disable-roi-redetect", action="store_true", help="Disable ROI re-detection around the active target.")
    parser.add_argument("--no-manual-confirmation", action="store_true", help="Auto-confirm alerts after a short warmup.")
    parser.add_argument("--save-video", default="", help="Optional annotated output video path.")
    parser.add_argument("--state-log", default="", help="Optional JSONL state log path.")
    parser.add_argument("--alert-log", default="", help="Optional JSONL alert event log path.")
    parser.add_argument("--alert-crops", default="", help="Optional directory for confirmed alert crops.")
    parser.add_argument("--show", action="store_true", help="Show annotated frames.")
    return parser.parse_args()


def open_source(source: str):
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def build_detector(model, args: argparse.Namespace):
    class_names = [name.strip() for name in args.target_classes.split(",") if name.strip()]
    filters = [
        solutions.AreaFilter(min_area_px=16),
        solutions.AspectRatioFilter(min_ratio=0.25, max_ratio=4.0),
        solutions.BorderFilter(margin_px=6),
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
        enable_roi=not args.disable_roi_redetect,
        preprocess_mode=args.input_mode,
        clahe=args.clahe,
        filters=filters,
    )


def build_tracker(args: argparse.Namespace):
    """Instantiate tracker backends that need extra runtime parameters."""
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


def maybe_handle_keypress(key: int, system: solutions.AntiUAVSystem) -> None:
    if key in {ord("c"), ord("C")}:
        system.confirm_current_target(True, note="operator_confirmed")
    elif key in {ord("r"), ord("R")}:
        system.confirm_current_target(False, note="operator_rejected")


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    detector = build_detector(model, args)
    system = solutions.AntiUAVSystem(
        detector,
        tracker=build_tracker(args),
        presence_verifier=build_presence_verifier(args),
        detector_assist_policy=args.detector_assist_policy,
        presence_score_thresh=args.presence_score_thresh,
        presence_uncertainty_thresh=(None if args.presence_uncertainty_thresh < 0 else args.presence_uncertainty_thresh),
        presence_refresh_streak=args.presence_refresh_streak,
        detect_interval=args.detect_interval,
        max_lost=args.max_lost,
        tracker_score_thresh=0.4,
        min_confidence=0.45,
        roi_redetect=not args.disable_roi_redetect,
        manual_confirmation=not args.no_manual_confirmation,
        min_confirm_detections=args.min_confirm_detections,
    )
    recorder = solutions.AlertRecorder(
        state_path=args.state_log or None,
        alert_path=args.alert_log or None,
        crop_dir=args.alert_crops or None,
    )

    cap = open_source(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    writer = None
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            state = system.step(frame)
            events = system.drain_alerts()
            recorder.record_state(state)
            recorder.record_events(frame, events)

            annotated = system.annotate(frame, state)
            if args.show:
                cv2.imshow("anti-uav-alerting", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                maybe_handle_keypress(key, system)

            if args.save_video:
                if writer is None:
                    output_path = Path(args.save_video).expanduser().resolve()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                writer.write(annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
