#!/usr/bin/env python3
"""Render a prediction-only side-by-side comparison of two RK-BoT-SORT outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anti_uav.render_pt_detector_pair_video import (  # noqa: E402
    COLORS,
    draw_corner_box,
    load_gt,
    match_counts,
    metrics,
    read_manifest,
    source_fps,
    text_with_shadow,
)


@dataclass(frozen=True)
class Track:
    track_id: int
    box: np.ndarray
    confidence: float
    predicted: bool
    hits: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Ordered image manifest")
    parser.add_argument("--source-video", type=Path, help="Original video, used only for output FPS")
    parser.add_argument("--left-tracks", type=Path, required=True)
    parser.add_argument("--right-tracks", type=Path, required=True)
    parser.add_argument("--left-label", default="P3 + RK-BoT-SORT")
    parser.add_argument("--right-label", default="Frozen-P3 + Add-on P2 + RK-BoT-SORT")
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(544, 960), metavar=("H", "W"))
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--fps", type=float, help="Override output FPS")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"))
    parser.add_argument("--trail-length", type=int, default=24)
    parser.add_argument("--track-high-thresh", type=float, default=0.03)
    parser.add_argument("--track-low-thresh", type=float, default=0.01)
    parser.add_argument("--new-track-thresh", type=float, default=0.03)
    parser.add_argument("--track-min-hits", type=int, default=2)
    return parser.parse_args()


def load_tracks(path: Path, frame_count: int, min_hits: int) -> dict[int, list[Track]]:
    tracks: dict[int, list[Track]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_index = int(row["frame"])
            if not 0 <= frame_index < frame_count:
                raise ValueError(f"Track frame {frame_index} is outside [0, {frame_count}) in {path}")
            hits = int(row.get("hits", "1"))
            if hits < min_hits:
                continue
            x = float(row["x"])
            y = float(row["y"])
            width = float(row["width"])
            height = float(row["height"])
            tracks[frame_index].append(
                Track(
                    track_id=int(row["track_id"]),
                    box=np.asarray((x, y, x + width, y + height), dtype=np.float32),
                    confidence=float(row["confidence"]),
                    predicted=bool(int(row.get("predicted", "0"))),
                    hits=hits,
                )
            )
    return tracks


def prediction_array(tracks: list[Track]) -> np.ndarray:
    if not tracks:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray([(*track.box, track.confidence) for track in tracks], dtype=np.float32)


def scaled_box(box: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    result = box.copy()
    result[[0, 2]] *= scale_x
    result[[1, 3]] *= scale_y
    return result


def draw_trails(
    panel: np.ndarray,
    histories: dict[int, deque[tuple[float, float]]],
    color: tuple[int, int, int],
    scale_x: float,
    scale_y: float,
) -> None:
    for points in histories.values():
        if len(points) < 2:
            continue
        scaled = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in points]
        for index in range(1, len(scaled)):
            fade = 0.25 + 0.75 * index / len(scaled)
            faded = tuple(int(channel * fade) for channel in color)
            cv2.line(panel, scaled[index - 1], scaled[index], faded, 1, cv2.LINE_AA)


def update_histories(
    histories: dict[int, deque[tuple[float, float]]],
    last_seen: dict[int, int],
    tracks: list[Track],
    frame_index: int,
    trail_length: int,
) -> None:
    reset_gap = max(2, trail_length)
    for track in tracks:
        if frame_index - last_seen.get(track.track_id, frame_index) > reset_gap:
            histories[track.track_id].clear()
        x1, y1, x2, y2 = track.box
        histories[track.track_id].append(((x1 + x2) / 2, (y1 + y2) / 2))
        last_seen[track.track_id] = frame_index
    for track_id, seen_at in list(last_seen.items()):
        if frame_index - seen_at > reset_gap:
            histories.pop(track_id, None)
            last_seen.pop(track_id, None)


def focus_box(gt: np.ndarray, tracks: list[Track]) -> np.ndarray | None:
    if len(gt):
        return gt[0]
    if tracks:
        return max(tracks, key=lambda track: track.confidence).box
    return None


def add_zoom(
    panel: np.ndarray,
    clean_frame: np.ndarray,
    gt: np.ndarray,
    tracks: list[Track],
    color: tuple[int, int, int],
) -> None:
    focus = focus_box(gt, tracks)
    if focus is None:
        return
    panel_h, panel_w = panel.shape[:2]
    source_h, source_w = clean_frame.shape[:2]
    x1, y1, x2, y2 = focus
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    crop_w, crop_h = min(160, source_w), min(100, source_h)
    left = int(np.clip(center_x - crop_w // 2, 0, max(0, source_w - crop_w)))
    top = int(np.clip(center_y - crop_h // 2, 0, max(0, source_h - crop_h)))
    crop = clean_frame[top : top + crop_h, left : left + crop_w]
    if crop.shape[:2] != (crop_h, crop_w):
        return

    zoom_w, zoom_h = 320, 200
    zoom = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
    visible_tracks: list[Track] = []
    for track in tracks:
        tx1, ty1, tx2, ty2 = track.box
        if tx2 < left or tx1 >= left + crop_w or ty2 < top or ty1 >= top + crop_h:
            continue
        zoom_box = np.asarray(
            (
                (tx1 - left) * zoom_w / crop_w,
                (ty1 - top) * zoom_h / crop_h,
                (tx2 - left) * zoom_w / crop_w,
                (ty2 - top) * zoom_h / crop_h,
            ),
            dtype=np.float32,
        )
        draw_corner_box(zoom, zoom_box, color, 1)
        visible_tracks.append(track)

    zoom_x, zoom_y = panel_w - 330, 66
    cv2.rectangle(panel, (zoom_x - 3, zoom_y - 3), (zoom_x + 323, zoom_y + 203), (255, 255, 255), 2)
    panel[zoom_y : zoom_y + zoom_h, zoom_x : zoom_x + zoom_w] = zoom
    crop_box = np.asarray(
        (
            left * panel_w / source_w,
            top * panel_h / source_h,
            (left + crop_w) * panel_w / source_w,
            (top + crop_h) * panel_h / source_h,
        )
    )
    draw_corner_box(panel, crop_box, (225, 225, 225), 1, corner_length=8)

    if visible_tracks:
        best = max(visible_tracks, key=lambda track: track.confidence)
        state = "PRED" if best.predicted else "DET"
        legend = f"TARGET VIEW  |  ID {best.track_id:02d}  {state}  {best.confidence:.2f}"
    else:
        legend = "TARGET VIEW  |  NO TRACK"
    text_with_shadow(panel, legend, (zoom_x + 8, zoom_y + 20), 0.43, (255, 255, 255), 1)


def track_status(tracks: list[Track]) -> str:
    if not tracks:
        return "NO TRACK OUTPUT"
    parts = []
    for track in sorted(tracks, key=lambda item: item.confidence, reverse=True)[:3]:
        state = "PRED" if track.predicted else "DET"
        parts.append(f"ID {track.track_id:02d} {state} {track.confidence:.2f}")
    if len(tracks) > 3:
        parts.append(f"+{len(tracks) - 3}")
    return "  |  ".join(parts)


def make_panel(
    frame: np.ndarray,
    gt: np.ndarray,
    tracks: list[Track],
    label: str,
    color: tuple[int, int, int],
    panel_size: tuple[int, int],
    histories: dict[int, deque[tuple[float, float]]],
) -> np.ndarray:
    panel_w, panel_h = panel_size
    source_h, source_w = frame.shape[:2]
    scale_x, scale_y = panel_w / source_w, panel_h / source_h
    panel = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    draw_trails(panel, histories, color, scale_x, scale_y)
    for track in tracks:
        draw_corner_box(panel, scaled_box(track.box, scale_x, scale_y), color, 1)
    add_zoom(panel, frame, gt, tracks, color)

    cv2.rectangle(panel, (0, 0), (panel_w, 58), (18, 18, 18), -1)
    text_with_shadow(panel, label, (14, 25), 0.66, (245, 245, 245), 2)
    status = track_status(tracks)
    status_color = color if tracks else COLORS["clear"]
    text_with_shadow(panel, status, (14, 49), 0.48, status_color, 1)
    return panel


def empty_stats() -> dict[str, object]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "output_frames": 0,
        "track_outputs": 0,
        "predicted_outputs": 0,
        "unique_ids": set(),
    }


def update_stats(stats: dict[str, object], counts: tuple[int, int, int], tracks: list[Track]) -> None:
    for key, value in zip(("tp", "fp", "fn"), counts):
        stats[key] = int(stats[key]) + value
    stats["output_frames"] = int(stats["output_frames"]) + bool(tracks)
    stats["track_outputs"] = int(stats["track_outputs"]) + len(tracks)
    stats["predicted_outputs"] = int(stats["predicted_outputs"]) + sum(track.predicted for track in tracks)
    unique_ids = stats["unique_ids"]
    assert isinstance(unique_ids, set)
    unique_ids.update(track.track_id for track in tracks)


def final_stats(stats: dict[str, object]) -> dict[str, object]:
    detection_metrics = metrics({key: int(stats[key]) for key in ("tp", "fp", "fn")})
    unique_ids = stats["unique_ids"]
    assert isinstance(unique_ids, set)
    return {
        **detection_metrics,
        "output_frames": int(stats["output_frames"]),
        "track_outputs": int(stats["track_outputs"]),
        "predicted_outputs": int(stats["predicted_outputs"]),
        "unique_track_ids": sorted(unique_ids),
    }


def main() -> None:
    args = parse_args()
    images = read_manifest(args.images)
    left_tracks = load_tracks(args.left_tracks, len(images), args.track_min_hits)
    right_tracks = load_tracks(args.right_tracks, len(images), args.track_min_hits)
    fps = source_fps(args.source_video, args.fps)
    panel_size = tuple(args.panel_size)
    output_size = (panel_size[0] * 2, panel_size[1])

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.output_video}")

    trail_length = max(0, args.trail_length)
    histories = {
        "left": defaultdict(lambda: deque(maxlen=trail_length or 1)),
        "right": defaultdict(lambda: deque(maxlen=trail_length or 1)),
    }
    last_seen: dict[str, dict[int, int]] = {"left": {}, "right": {}}
    totals = {"left": empty_stats(), "right": empty_stats()}
    try:
        for frame_index, image_path in enumerate(images):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")
            gt = load_gt(image_path, frame.shape[1], frame.shape[0])
            current = {
                "left": left_tracks.get(frame_index, []),
                "right": right_tracks.get(frame_index, []),
            }
            for side in ("left", "right"):
                update_histories(
                    histories[side], last_seen[side], current[side], frame_index, trail_length
                )
                counts = match_counts(gt, prediction_array(current[side]), args.match_iou)
                update_stats(totals[side], counts, current[side])

            left_panel = make_panel(
                frame,
                gt,
                current["left"],
                args.left_label,
                COLORS["left"],
                panel_size,
                histories["left"],
            )
            right_panel = make_panel(
                frame,
                gt,
                current["right"],
                args.right_label,
                COLORS["right"],
                panel_size,
                histories["right"],
            )
            combined = np.concatenate((left_panel, right_panel), axis=1)
            cv2.line(combined, (panel_size[0], 0), (panel_size[0], panel_size[1]), (255, 255, 255), 2)
            footer = (
                f"Video00004  |  frame {frame_index + 1:04d}/{len(images)}  |  "
                f"{frame_index / fps:06.2f}s  |  input {args.imgsz[1]}x{args.imgsz[0]}  |  "
                f"RK-BoT-SORT high/low {args.track_high_thresh:.2f}/{args.track_low_thresh:.2f}"
            )
            text_with_shadow(combined, footer, (390, panel_size[1] - 14), 0.50, (245, 245, 245), 1)
            writer.write(combined)
            if (frame_index + 1) % 250 == 0 or frame_index + 1 == len(images):
                print(f"Rendered {frame_index + 1}/{len(images)} frames", flush=True)
    finally:
        writer.release()

    summary = {
        "schema_version": "anti_uav.rk_botsort_pair_video.v1",
        "images_manifest": str(args.images.resolve()),
        "source_video": str(args.source_video.resolve()) if args.source_video else None,
        "frame_count": len(images),
        "fps": fps,
        "output_resolution": list(output_size),
        "input_resolution": [args.imgsz[1], args.imgsz[0]],
        "visualization_style": "prediction_only_confirmed_tracks_corner_boxes_and_short_trail_v1",
        "ground_truth_drawn": False,
        "unconfirmed_tracks_drawn": False,
        "tracker": {
            "name": "RK-BoT-SORT",
            "high_threshold": args.track_high_thresh,
            "low_threshold": args.track_low_thresh,
            "new_track_threshold": args.new_track_thresh,
            "min_hits": args.track_min_hits,
            "trail_length": trail_length,
        },
        "left": {
            "label": args.left_label,
            "tracks_csv": str(args.left_tracks.resolve()),
            "metrics": final_stats(totals["left"]),
        },
        "right": {
            "label": args.right_label,
            "tracks_csv": str(args.right_tracks.resolve()),
            "metrics": final_stats(totals["right"]),
        },
        "output_video": str(args.output_video.resolve()),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
