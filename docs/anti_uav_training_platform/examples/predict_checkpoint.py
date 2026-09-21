#!/usr/bin/env python3
"""Offline reference worker: trusted local PT checkpoint, fixed 544x960 canvas.

This is not an HTTP server or an upload sandbox. The platform must authorize
and resolve checkpoint/media IDs before invoking it.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=.03)
    parser.add_argument("--nms-iou", type=float, default=.45)
    parser.add_argument("--preview-frames", type=int, default=3)
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or not 0 < args.nms_iou <= 1 or args.preview_frames < 0:
        parser.error("Invalid thresholds or preview count")
    if not args.checkpoint.is_file() or args.checkpoint.suffix != ".pt" or not args.source.is_file():
        parser.error("A trusted local .pt checkpoint and local media file are required")
    if args.source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}:
        parser.error("Unsupported media extension")

    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect.predict import DetectionPredictor

    class FixedCanvasPredictor(DetectionPredictor):
        def pre_transform(self, images):
            letterbox = LetterBox(new_shape=(544, 960), auto=False, stride=32)
            return [letterbox(image=image) for image in images]

        def preprocess(self, images):
            tensor = super().preprocess(images)
            if tuple(tensor.shape[1:]) != (3, 544, 960):
                raise ValueError(f"Unexpected inference shape: {tuple(tensor.shape)}")
            return tensor

    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = YOLO(str(args.checkpoint))
    options = dict(imgsz=[544, 960], conf=args.conf, iou=args.nms_iou, max_det=100,
                   device=args.device, half=False, save=False, verbose=False,
                   mode="predict", task="detect", batch=1)
    # This repository's Model.predict accepts a predictor instance, not a class.
    predictor = FixedCanvasPredictor(overrides=options, _callbacks=model.callbacks)
    frames = detections = 0
    with (args.output / "predictions.jsonl").open("x") as stream:
        for frame_index, result in enumerate(model.predict(
            source=str(args.source), predictor=predictor, stream=True, **options,
        )):
            boxes = []
            if result.boxes is not None:
                for xyxy, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                            result.boxes.conf.cpu().tolist(),
                                            result.boxes.cls.cpu().tolist()):
                    boxes.append(dict(class_id=int(cls), score=score, xyxy=xyxy))
            height, width = result.orig_shape
            stream.write(json.dumps(dict(frame_index=frame_index, width=width, height=height,
                                         boxes=boxes), allow_nan=False) + "\n")
            if frame_index < args.preview_frames:
                preview = result.plot(line_width=1, labels=True, conf=True)
                if not cv2.imwrite(str(args.output / f"preview_{frame_index:06d}.jpg"), preview):
                    raise RuntimeError("Failed to write prediction preview")
            frames += 1
            detections += len(boxes)
    if not frames:
        raise RuntimeError("No frame was decoded")
    summary = dict(backend="pytorch", input_hw=[544, 960], conf=args.conf,
                   nms_iou=args.nms_iou, max_det=100, frames=frames, detections=detections,
                   checkpoint_sha256=sha256(args.checkpoint), source_sha256=sha256(args.source),
                   coordinate_system="original-image pixel xyxy; frame_index is zero-based",
                   scope="Detector only, no tracker, no GT scoring, not an RKNN/FPS benchmark")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
