#!/usr/bin/env python3
"""Render audited detector caches without rerunning model inference."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np


COLORS = {
    "teacher": (30, 205, 255),
    "student": (255, 205, 40),
    "text": (242, 244, 246),
    "muted": (190, 195, 200),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--student-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--video-label", required=True)
    parser.add_argument("--teacher-label", default="50-video P2+P3 Teacher FP32")
    parser.add_argument("--student-label", default="50-video P3 Student FP32")
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--source-fps", type=Fraction, required=True)
    parser.add_argument("--sample-stride", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def validate(teacher: list[dict], student: list[dict]) -> None:
    if not teacher or len(teacher) != len(student):
        raise ValueError("Teacher and Student caches must have the same non-zero length")
    for index, (left, right) in enumerate(zip(teacher, student)):
        for key in ("path", "frame_index", "original_hw", "gt_xyxy_class"):
            if left[key] != right[key]:
                raise ValueError(f"Cache mismatch at row {index}: {key}")
        if not Path(left["path"]).is_file():
            raise FileNotFoundError(left["path"])


def boxes(row: dict, confidence: float) -> np.ndarray:
    predictions = np.asarray(row["pred_xyxy_score_class"], dtype=np.float32).reshape(-1, 6)
    predictions = predictions[predictions[:, 4] >= confidence]
    if len(predictions):
        predictions = predictions[np.argsort(-predictions[:, 4], kind="stable")]
    return predictions[:, :5]


def corner(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.rint(box[:4]).astype(int)
    x1, x2 = sorted((int(np.clip(x1, 0, width - 1)), int(np.clip(x2, 0, width - 1))))
    y1, y2 = sorted((int(np.clip(y1, 0, height - 1)), int(np.clip(y2, 0, height - 1))))
    length = max(2, min(10, round(max(x2 - x1 + 1, y2 - y1 + 1) * 0.35)))
    segments = (
        ((x1, y1), (min(x2, x1 + length), y1)), ((x1, y1), (x1, min(y2, y1 + length))),
        ((x2, y1), (max(x1, x2 - length), y1)), ((x2, y1), (x2, min(y2, y1 + length))),
        ((x1, y2), (min(x2, x1 + length), y2)), ((x1, y2), (x1, max(y1, y2 - length))),
        ((x2, y2), (max(x1, x2 - length), y2)), ((x2, y2), (x2, max(y1, y2 - length))),
    )
    for start, end in segments:
        cv2.line(image, start, end, color, 1, cv2.LINE_8)


def text(image: np.ndarray, value: str, origin: tuple[int, int], scale: float,
         color: tuple[int, int, int] = COLORS["text"]) -> None:
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def crop_anchor(row: dict, teacher: np.ndarray, student: np.ndarray) -> np.ndarray | None:
    gt = np.asarray(row["gt_xyxy_class"], dtype=np.float32).reshape(-1, 5)
    if len(gt):
        return gt[0, :4]
    if len(teacher):
        return teacher[0, :4]
    if len(student):
        return student[0, :4]
    return None


def crop_bounds(anchor: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    target_w = max(1.0, float(anchor[2] - anchor[0]))
    target_h = max(1.0, float(anchor[3] - anchor[1]))
    crop_w = min(width, max(160, int(np.ceil(max(target_w * 1.8, target_h * 1.8 * 16 / 9)))))
    crop_h = min(height, max(90, int(np.ceil(crop_w * 9 / 16))))
    center_x, center_y = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
    left = int(np.clip(center_x - crop_w / 2, 0, width - crop_w))
    top = int(np.clip(center_y - crop_h / 2, 0, height - crop_h))
    return left, top, crop_w, crop_h


def draw_boxes(image: np.ndarray, predictions: np.ndarray, color: tuple[int, int, int],
               scale_x: float, scale_y: float, offset_x: float = 0, offset_y: float = 0) -> None:
    for prediction in predictions:
        box = prediction.copy()
        box[[0, 2]] = (box[[0, 2]] - offset_x) * scale_x
        box[[1, 3]] = (box[[1, 3]] - offset_y) * scale_y
        if box[2] >= 0 and box[0] < image.shape[1] and box[3] >= 0 and box[1] < image.shape[0]:
            corner(image, box, color)


def panel(frame: np.ndarray, predictions: np.ndarray, anchor: np.ndarray | None,
          label: str, color: tuple[int, int, int], status: str) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    output = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    draw_boxes(output, predictions, color, 960 / source_w, 540 / source_h)
    cv2.rectangle(output, (0, 0), (960, 40), (18, 18, 18), -1)
    text(output, label, (14, 27), 0.63)
    text(output, f"pred {len(predictions)}", (825, 27), 0.55, color)
    if anchor is not None:
        left, top, crop_w, crop_h = crop_bounds(anchor, source_w, source_h)
        zoom = cv2.resize(frame[top:top + crop_h, left:left + crop_w], (320, 180),
                          interpolation=cv2.INTER_CUBIC if crop_w < 320 else cv2.INTER_AREA)
        draw_boxes(zoom, predictions, color, 320 / crop_w, 180 / crop_h, left, top)
        output[47:227, 633:953] = zoom
        cv2.rectangle(output, (630, 44), (956, 230), (245, 245, 245), 1)
        text(output, "TARGET VIEW", (640, 66), 0.43)
    text(output, status, (14, 522), 0.43, COLORS["muted"])
    return output


def encoder(path: Path, size: tuple[int, int], fps: Fraction) -> subprocess.Popen:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


def finish(process: subprocess.Popen, name: str) -> None:
    process.stdin.close()
    error = process.stderr.read().decode("utf-8", errors="replace")
    if process.wait() != 0:
        raise RuntimeError(f"{name} encoder failed: {error}")


def main() -> None:
    args = parse_args()
    if args.output.exists() or not 0 < args.conf < 1 or args.sample_stride < 1:
        raise ValueError("Output must be fresh and parameters must be valid")
    teacher_rows, student_rows = load(args.teacher_cache), load(args.student_cache)
    validate(teacher_rows, student_rows)
    args.output.mkdir(parents=True)
    output_fps = args.source_fps / args.sample_stride
    paths = {
        "teacher": args.output / f"{args.video_label}_50video_P2P3_Teacher_FP32_conf003.mp4",
        "student": args.output / f"{args.video_label}_50video_P3_Student_FP32_conf003.mp4",
        "comparison": args.output / f"{args.video_label}_Teacher_vs_Student_FP32_conf003.mp4",
    }
    processes = {
        "teacher": encoder(paths["teacher"], (960, 540), output_fps),
        "student": encoder(paths["student"], (960, 540), output_fps),
        "comparison": encoder(paths["comparison"], (1920, 540), output_fps),
    }
    started = time.monotonic()
    totals = {"teacher": 0, "student": 0}
    try:
        for index, (teacher_row, student_row) in enumerate(zip(teacher_rows, student_rows)):
            frame = cv2.imread(teacher_row["path"], cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read {teacher_row['path']}")
            teacher_boxes = boxes(teacher_row, args.conf)
            student_boxes = boxes(student_row, args.conf)
            anchor = crop_anchor(teacher_row, teacher_boxes, student_boxes)
            seconds = teacher_row["frame_index"] / float(args.source_fps)
            status = (f"{args.video_label} | frame {teacher_row['frame_index']} | {seconds:.2f}s | "
                      f"input 960x544 | conf {args.conf:.2f} | FP32")
            teacher_panel = panel(frame, teacher_boxes, anchor, args.teacher_label,
                                  COLORS["teacher"], status)
            student_panel = panel(frame, student_boxes, anchor, args.student_label,
                                  COLORS["student"], status)
            processes["teacher"].stdin.write(teacher_panel.tobytes())
            processes["student"].stdin.write(student_panel.tobytes())
            processes["comparison"].stdin.write(np.concatenate((teacher_panel, student_panel), axis=1).tobytes())
            totals["teacher"] += len(teacher_boxes)
            totals["student"] += len(student_boxes)
            if index in {0, len(teacher_rows) // 2, len(teacher_rows) - 1}:
                cv2.imwrite(str(args.output / f"preview_{index:06d}.jpg"),
                            np.concatenate((teacher_panel, student_panel), axis=1))
            if (index + 1) % 256 == 0 or index + 1 == len(teacher_rows):
                dump(args.output / "status.json", {
                    "stage": "rendering", "frames": index + 1, "total": len(teacher_rows),
                    "seconds": round(time.monotonic() - started, 2),
                })
                print(json.dumps({"frames": index + 1, "total": len(teacher_rows)}), flush=True)
        for name, process in processes.items():
            finish(process, name)
    except BaseException:
        for process in processes.values():
            if process.poll() is None:
                process.kill()
        raise
    summary = {
        "video_label": args.video_label,
        "frames": len(teacher_rows),
        "source_fps": str(args.source_fps),
        "sample_stride": args.sample_stride,
        "output_fps": str(output_fps),
        "input_hw": [544, 960],
        "conf": args.conf,
        "ground_truth_drawn": False,
        "ground_truth_used_for_common_crop_only": True,
        "inference_rerun": False,
        "teacher_cache": str(args.teacher_cache.resolve()),
        "teacher_cache_sha256": sha256(args.teacher_cache),
        "student_cache": str(args.student_cache.resolve()),
        "student_cache_sha256": sha256(args.student_cache),
        "detections_at_operating_conf": totals,
        "videos": {key: str(value.resolve()) for key, value in paths.items()},
        "video_sha256": {key: sha256(value) for key, value in paths.items()},
        "visualization": "prediction-only 1px corner boxes; no GT/crosshair; clean-source target crop",
        "seconds": round(time.monotonic() - started, 2),
    }
    dump(args.output / "summary.json", summary)
    dump(args.output / "status.json", {"stage": "complete", "frames": len(teacher_rows)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
