#!/usr/bin/env python3
"""Render a full source video with one detector, no GT or tracker, and clean crops."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def probe(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration,codec_name,pix_fmt",
        "-of", "json", str(path)], text=True))
    return data["streams"][0]


def panel(frame, boxes, index, count, fps, title):
    from scripts.anti_uav.render_pt_detector_pair_video import draw_corner_box, boxes_in_crop

    height, width = frame.shape[:2]
    color = (255, 210, 40)
    canvas = np.full((784, 1600, 3), (25, 29, 35), np.uint8)
    main = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    sx, sy = 1280 / width, 720 / height
    for box in boxes:
        scaled = box.copy()
        scaled[:4] *= [sx, sy, sx, sy]
        draw_corner_box(main, scaled, color, 1)
    canvas[64:, :1280] = main
    cv2.putText(canvas, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, .67, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {index+1}/{count} | {index/fps:.2f}s | predictions {len(boxes)} | NO GT / NO TRACKER",
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, .55, (210, 215, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "TOP-SCORE CROPS", (1292, 91), cv2.FONT_HERSHEY_SIMPLEX, .52, (220, 225, 230), 1, cv2.LINE_AA)
    for rank in range(2):
        y = 125 + rank * 285
        if rank >= len(boxes):
            cv2.putText(canvas, "NO DETECTION", (1325, y + 105), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (130, 135, 140), 1, cv2.LINE_AA)
            continue
        box = boxes[rank]
        crop_w, crop_h = min(160, width), min(100, height)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        left = int(np.clip(cx - crop_w / 2, 0, width - crop_w))
        top = int(np.clip(cy - crop_h / 2, 0, height - crop_h))
        # Resize the clean source first, then draw at one display pixel.
        crop = cv2.resize(frame[top:top+crop_h, left:left+crop_w], (320, 200), interpolation=cv2.INTER_CUBIC)
        zoom_boxes = boxes_in_crop(boxes, left, top, crop_w, crop_h, 320/crop_w, 200/crop_h)
        for zoom_box in zoom_boxes:
            draw_corner_box(crop, zoom_box, color, 1)
        canvas[y:y+200, 1280:] = crop
        cv2.putText(canvas, f"rank {rank+1} | score {box[4]:.3f} | {320/crop_w:.1f}x", (1290, y+225),
                    cv2.FONT_HERSHEY_SIMPLEX, .47, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Crop positions follow", (1292, 724), cv2.FONT_HERSHEY_SIMPLEX, .46,
                (190, 195, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "scores, not target IDs.", (1292, 746), cv2.FONT_HERSHEY_SIMPLEX, .46,
                (190, 195, 200), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--output-name", default="detector_visualization.mp4")
    parser.add_argument("--label", default="Frozen-P3 + Add-on P2")
    parser.add_argument("--device", default="6")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=.03)
    parser.add_argument("--nms-iou", type=float, default=.45)
    parser.add_argument("--max-frames", type=int, help="Explicit smoke-test prefix; default the entire video")
    args = parser.parse_args()
    if args.batch < 1 or not 0 < args.conf < 1 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("Invalid batch, confidence or frame limit")
    if args.output.exists():
        raise FileExistsError(args.output)
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".mp4"):
        raise ValueError("Output name must be an MP4 basename")
    src = probe(args.source_video)
    if src["width"] * 9 != src["height"] * 16:
        raise ValueError("This layout requires a 16:9 source; refusing to distort other aspect ratios")
    numerator, denominator = map(int, src["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    if fps <= 0 or src["avg_frame_rate"] != src["r_frame_rate"]:
        raise ValueError("This renderer requires constant-rate source video")
    expected = int(src["nb_frames"])
    limit = min(expected, args.max_frames) if args.max_frames else expected
    args.output.mkdir(parents=True)
    cv2.setNumThreads(2)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise ValueError("Cannot open source video")
    destination = args.output / args.output_name
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "1600x784", "-r", src["avg_frame_rate"], "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination)]
    started = time.monotonic()
    frames_written = detections = detected_frames = 0
    stats = dict(source_video=str(args.source_video), source_sha256=sha256(args.source_video), source_info=src,
        weights=str(args.model), weights_sha256=sha256(args.model), input_hw=[544, 960], conf=args.conf,
        nms_iou=args.nms_iou, max_det=100, precision="PT FP32", device=args.device,
        full_source_video=args.max_frames is None, ground_truth_used=False, tracker_used=False,
        output_fps=fps, frame_stride=1, audio_included=False,
        visualization="1px prediction corners, two top-score crops from clean originals; no cross or IDs",
        video=str(destination), command=command)
    dump(args.output / "protocol.json", stats)
    title = f"{args.label} | input 960x544 | conf {args.conf:.2f} | PT FP32"
    with (args.output / "encode.log").open("x") as log, (args.output / "predictions.jsonl").open("x") as predictions:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            while frames_written < limit:
                batch = []
                for _ in range(min(args.batch, limit-frames_written)):
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f"Source ended at frame {frames_written+len(batch)}, expected {limit}")
                    batch.append(image)
                results = model.predict(source=batch, imgsz=[544, 960], rect=False, conf=args.conf,
                    iou=args.nms_iou, max_det=100, device=args.device, half=False, batch=args.batch,
                    verbose=False, save=False, stream=False)
                if len(results) != len(batch):
                    raise AssertionError("Frame/result count mismatch")
                for frame, result in zip(batch, results):
                    if result.boxes is None or len(result.boxes) == 0:
                        boxes = np.empty((0, 5), np.float32)
                    else:
                        boxes = np.column_stack((result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()))
                        boxes = boxes[np.argsort(-boxes[:, 4], kind="stable")]
                    rendered = panel(frame, boxes, frames_written, limit, fps, title)
                    encoder.stdin.write(rendered.tobytes())
                    predictions.write(json.dumps(dict(frame_index=frames_written, time_seconds=frames_written/fps,
                        boxes_xyxy_score=boxes.tolist())) + "\n")
                    if frames_written in (0, min(limit-1, 300), min(limit-1, 4000), min(limit-1, 9000)):
                        if not cv2.imwrite(str(args.output / f"preview_{frames_written:06d}.jpg"), rendered):
                            raise RuntimeError("Failed to write preview")
                    frames_written += 1
                    detections += len(boxes)
                    detected_frames += bool(len(boxes))
                progress = dict(stage="rendering", frames=frames_written, total=limit, detections=detections,
                                seconds=round(time.monotonic()-started, 2))
                dump(args.output / "status.json", progress)
                if frames_written % 512 == 0 or frames_written == limit:
                    print(json.dumps(progress), flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed; see encode.log")
            out = probe(destination)
            if int(out["nb_frames"]) != limit or abs(float(out["duration"])-limit/fps) > .02:
                raise AssertionError("Encoded video frame count/duration mismatch")
            stats.update(frames=frames_written, total_detections=detections, frames_with_detections=detected_frames,
                output_info=out, seconds=round(time.monotonic()-started, 2), output_sha256=sha256(destination),
                note="Source playback FPS is not measured board inference FPS. Full-frame video is not the sampled evaluation set.")
            dump(args.output / "summary.json", stats)
            dump(args.output / "status.json", dict(stage="complete", frames=frames_written, video=str(destination)))
            print(json.dumps(stats), flush=True)
        except BaseException as error:
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
            dump(args.output / "status.json", dict(stage="failed", frames=frames_written, error=repr(error)))
            raise
        finally:
            cap.release()


if __name__ == "__main__":
    main()
