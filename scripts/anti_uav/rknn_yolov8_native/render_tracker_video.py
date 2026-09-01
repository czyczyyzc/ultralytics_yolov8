#!/usr/bin/env python3
"""Render RKNN detector and RK-BoT-SORT CSV output on the source video."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", default="YOLOv8n INT8", help="Model name shown in the video header.")
    parser.add_argument("--platform-label", default="RK3588S")
    parser.add_argument("--footer-label", default="")
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_tracks(path: Path) -> dict[int, list[dict[str, str]]]:
    rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["frame"])].append(row)
    return rows


def load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def track_color(track_id: int) -> tuple[int, int, int]:
    hue = int((track_id * 47 + 79) % 180)
    hsv = np.uint8([[[hue, 215, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.62,
    color: tuple[int, int, int] = (245, 248, 250),
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 9, 12), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, y1: int, y2: int, alpha: float = 0.78) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (13, 21, 26), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, width - 1).astype(int)
    y1, y2 = np.clip([y1, y2], 0, height - 1).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    corner = max(10, min(28, (x2 - x1) // 5, (y2 - y1) // 5))
    for start, end in (
        ((x1, y1), (x1 + corner, y1)),
        ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)),
        ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)),
        ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)),
        ((x2, y2), (x2, y2 - corner)),
    ):
        cv2.line(frame, start, end, (255, 255, 255), 2, cv2.LINE_AA)

    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2)
    label_y1 = max(68, y1 - text_height - baseline - 12)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_width + 18, label_y1 + text_height + baseline + 12), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 9, label_y1 + text_height + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (7, 12, 16),
        2,
        cv2.LINE_AA,
    )


def timestamp(frame_index: int, fps: float) -> str:
    seconds = frame_index / max(fps, 1e-6)
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def main() -> None:
    args = parse_args()
    tracks = load_tracks(args.tracks)
    benchmark = load_json(args.benchmark)
    summary = load_json(args.summary)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 20.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(source_frames, args.max_frames) if args.max_frames else source_frames
    hold_frames = max(0, int(round(args.hold_sec * fps)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video: {args.output}")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=45))
    last_seen: dict[int, int] = {}
    pipeline_fps = float(benchmark.get("fps_mean", 0.0))
    tracker_ms = float(benchmark.get("detail_mean_ms", {}).get("tracker_association", 0.0))
    input_size = benchmark.get("input_size", [544, 960])
    input_height = int(input_size[0]) if len(input_size) >= 2 else 544
    input_width = int(input_size[1]) if len(input_size) >= 2 else 960
    accuracy = summary.get("accuracy_iou_0_5", {})

    frame_index = 0
    try:
        while frame_index < total_frames:
            ok, frame = capture.read()
            if not ok:
                break
            current_rows = tracks.get(frame_index, [])

            for row in current_rows:
                track_id = int(row["track_id"])
                x = float(row["x"])
                y = float(row["y"])
                box_width = float(row["width"])
                box_height = float(row["height"])
                confidence = float(row["confidence"])
                box = (
                    int(round(x)),
                    int(round(y)),
                    int(round(x + box_width)),
                    int(round(y + box_height)),
                )
                color = track_color(track_id)
                center = (int(round(x + box_width / 2)), int(round(y + box_height / 2)))
                if track_id in last_seen and frame_index - last_seen[track_id] > hold_frames:
                    trails[track_id].clear()
                trails[track_id].append(center)
                last_seen[track_id] = frame_index

                points = list(trails[track_id])
                for point_index in range(1, len(points)):
                    fade = point_index / len(points)
                    trail_color = tuple(int(channel * (0.25 + 0.75 * fade)) for channel in color)
                    cv2.line(frame, points[point_index - 1], points[point_index], trail_color, 2, cv2.LINE_AA)
                predicted = int(row.get("predicted", "0")) == 1
                state = "PRED" if predicted else "DET"
                draw_box(frame, box, color, f"ID {track_id:02d}  {confidence:.2f}  {state}")

            retained = [
                (track_id, frame_index - seen_frame)
                for track_id, seen_frame in last_seen.items()
                if 0 < frame_index - seen_frame <= hold_frames
            ]
            if current_rows:
                visible_ids = ", ".join(f"{int(row['track_id']):02d}" for row in current_rows)
                state_text = f"DETECTED  |  TRACK ID {visible_ids}"
                state_color = (95, 242, 173)
            elif retained:
                retained.sort(key=lambda item: item[1])
                retained_id, missing_frames = retained[0]
                state_text = f"TRACK BUFFER  |  ID {retained_id:02d} retained for {missing_frames / fps:.2f}s  |  no predicted box"
                state_color = (80, 198, 255)
            else:
                state_text = "NO DETECTOR-BACKED OUTPUT"
                state_color = (180, 190, 198)

            draw_panel(frame, 0, 66)
            cv2.rectangle(frame, (0, 0), (8, 66), (54, 218, 255), -1)
            put_text(
                frame,
                f"{args.platform_label}  |  {args.model_label} {input_width}x{input_height} + RK-BoT-SORT",
                (24, 29),
                0.69,
            )
            put_text(frame, state_text, (24, 56), 0.54, state_color)
            frame_text = f"FRAME {frame_index + 1:04d}/{total_frames:04d}  |  {timestamp(frame_index, fps)}"
            (frame_text_width, _), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            put_text(frame, frame_text, (width - frame_text_width - 24, 39), 0.62)

            draw_panel(frame, height - 62, height)
            perf_text = args.footer_label or (
                f"Board benchmark  {pipeline_fps:.2f} FPS  |  3 NPU cores  |  "
                f"Tracker  {tracker_ms:.4f} ms/frame  |  Source  {fps:.0f} FPS"
            )
            put_text(frame, perf_text, (24, height - 34), 0.59)
            if accuracy:
                quality_text = (
                    f"P {float(accuracy.get('precision', 0.0)) * 100:.2f}%  "
                    f"R {float(accuracy.get('recall', 0.0)) * 100:.2f}%  "
                    f"F1 {float(accuracy.get('f1', 0.0)) * 100:.2f}%"
                )
                (quality_width, _), _ = cv2.getTextSize(quality_text, cv2.FONT_HERSHEY_SIMPLEX, 0.59, 2)
                put_text(frame, quality_text, (width - quality_width - 24, height - 34), 0.59, (95, 242, 173))

            writer.write(frame)
            frame_index += 1
            if frame_index % 100 == 0 or frame_index == total_frames:
                print(f"Rendered {frame_index}/{total_frames}", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"Wrote {frame_index} frames to {args.output}")


if __name__ == "__main__":
    main()
