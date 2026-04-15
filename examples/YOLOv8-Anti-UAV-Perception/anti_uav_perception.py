from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO, solutions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an alerting-only anti-UAV perception pipeline on video.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path.")
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--tracker", default="template_match", choices=solutions.available_trackers(), help="Tracker backend.")
    parser.add_argument("--target-classes", default="drone,uav", help="Comma-separated class-name allowlist.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Detector input size.")
    parser.add_argument("--device", default=None, help="Torch device for detector inference, for example 0 or cpu.")
    parser.add_argument("--detect-interval", type=int, default=8, help="Run detector every N frames while tracking.")
    parser.add_argument("--max-lost", type=int, default=30, help="Frames to wait before dropping a lost target.")
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
    filters = [solutions.AreaFilter(min_area_px=9), solutions.AspectRatioFilter(), solutions.BorderFilter()]
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
        tracker=args.tracker,
        detect_interval=args.detect_interval,
        max_lost=args.max_lost,
        roi_redetect=not args.disable_roi_redetect,
        manual_confirmation=not args.no_manual_confirmation,
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
