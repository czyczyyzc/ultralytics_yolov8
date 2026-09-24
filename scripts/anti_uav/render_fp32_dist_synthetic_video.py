#!/usr/bin/env python3
"""Render paired full-length Video00009 FP32 + Dist/GMC visualizations.

The paired synthetic dataset is sparse. This preserves the original full frame
sequence and substitutes only its successfully replaced frames. It is a visual
experiment, not a temporally consistent synthetic tracking benchmark.
"""
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


def digest(path: Path) -> str:
    hash_value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def manifest_rows(root: Path) -> dict[int, dict]:
    raw = json.loads((root / "manifest.json").read_text())
    if raw.get("training_allowed") is not False:
        raise ValueError("Expected the evaluation-only paired dataset")
    rows = {int(row["frame"]): row for row in raw["rows"] if row["video"] == "Video00009"}
    if len(rows) != 1421 or sum(row["state"] == "replaced" for row in rows.values()) != 565:
        raise ValueError("Video00009 paired manifest coverage changed")
    if any(row["state"] not in {"replaced", "original_retained", "negative_unchanged"} for row in rows.values()):
        raise ValueError("Unknown synthesis state")
    return rows


def video_info(path: Path) -> tuple[int, float, int, int]:
    source = cv2.VideoCapture(str(path))
    if not source.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(source.get(cv2.CAP_PROP_FPS))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source.release()
    if count != 14201 or fps != 100 or (width, height) != (1920, 1080):
        raise ValueError(f"Unexpected source video: {count} frames, {fps} FPS, {width}x{height}")
    return count, fps, width, height


def replacement_frame(root: Path, row: dict, width: int, height: int) -> np.ndarray:
    frame = cv2.imread(str(root / row["replacement"]), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (height, width):
        raise ValueError(f"Invalid replacement frame {row['frame']}")
    return frame


def selected_frame(frame: np.ndarray, root: Path, rows: dict[int, dict], index: int,
                   variant: str) -> np.ndarray:
    row = rows.get(index)
    if variant == "synthetic" and row is not None and row["state"] == "replaced":
        return replacement_frame(root, row, frame.shape[1], frame.shape[0])
    return frame


def predict_boxes(results) -> list[list[list[float]]]:
    outputs = []
    for result in results:
        if result.boxes is None:
            outputs.append([])
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        if any(int(cls) != 0 for cls in classes):
            raise ValueError("Unexpected non-UAV detector class")
        rows = np.column_stack((boxes, scores)).astype(np.float32).tolist()
        if any(not np.isfinite(row).all() or row[2] <= row[0] or row[3] <= row[1]
               or row[4] < .03 for row in map(np.asarray, rows)):
            raise ValueError("Invalid detector observation")
        outputs.append(rows)
    return outputs


def infer(args) -> None:
    from ultralytics import YOLO

    rows = manifest_rows(args.manifest)
    count, fps, width, height = video_info(args.source)
    if digest(args.source) != "2629efe7da36dfd079985344653a7a4ecaa25f651954e5258a53c50e960d6b86":
        raise ValueError("Source video identity changed")
    limit = count if args.max_frames is None else min(count, args.max_frames)
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    model = YOLO(str(args.model))
    source = cv2.VideoCapture(str(args.source))
    if not source.isOpened():
        raise ValueError("Could not reopen source video")
    real_count = synthetic_count = changed = 0
    started = time.monotonic()
    with (args.output / "real_detections.jsonl").open("x") as real_file, \
            (args.output / "synthetic_detections.jsonl").open("x") as synthetic_file:
        try:
            for start in range(0, limit, args.batch):
                originals = []
                for index in range(start, min(limit, start + args.batch)):
                    ok, frame = source.read()
                    if not ok or frame.shape[:2] != (height, width):
                        raise RuntimeError(f"Source decode failed at frame {index}")
                    originals.append(frame)
                params = dict(imgsz=(544, 960), conf=.03, iou=.45, max_det=100,
                              device=args.device, half=False, rect=False, verbose=False)
                real_boxes = predict_boxes(model.predict(source=originals, **params))
                changed_indices = [offset for offset in range(len(originals))
                                   if (start + offset in rows and rows[start + offset]["state"] == "replaced")]
                replacements = [replacement_frame(args.manifest, rows[start + offset], width, height)
                                for offset in changed_indices]
                synthetic_boxes = list(real_boxes)
                if replacements:
                    replacement_boxes = predict_boxes(model.predict(source=replacements, **params))
                    for offset, boxes in zip(changed_indices, replacement_boxes):
                        synthetic_boxes[offset] = boxes
                    changed += len(replacements)
                for offset, (original, synthetic) in enumerate(zip(real_boxes, synthetic_boxes)):
                    index = start + offset
                    real_file.write(json.dumps(dict(frame_index=index, time_seconds=index/fps,
                                                    boxes_xyxy_score=original)) + "\n")
                    synthetic_file.write(json.dumps(dict(frame_index=index, time_seconds=index/fps,
                                                         boxes_xyxy_score=synthetic)) + "\n")
                    real_count += len(original)
                    synthetic_count += len(synthetic)
                if start % 1024 < args.batch:
                    print(json.dumps(dict(stage="infer", frames=min(limit, start+len(originals)),
                                          total=limit, replaced=changed)), flush=True)
        finally:
            source.release()
    if limit == count and changed != 565:
        raise AssertionError("Did not run all synthetic replacements")
    write_json(args.output / "summary.json", dict(stage="complete", frames=limit, fps=fps,
        source=str(args.source), source_sha256=digest(args.source), model=str(args.model),
        model_sha256=digest(args.model), manifest=str(args.manifest),
        manifest_sha256=digest(args.manifest / "manifest.json"), input_wh=[960, 544],
        dtype="FP32", conf=.03, nms_iou=.45, max_det=100, batch=args.batch,
        changed_frames=changed, original_detections=real_count,
        synthetic_detections=synthetic_count, seconds=time.monotonic()-started,
        real_cache_sha256=digest(args.output / "real_detections.jsonl"),
        synthetic_cache_sha256=digest(args.output / "synthetic_detections.jsonl"),
        note="Full consecutive video; only 565 approved synthetic frames are substituted."))


def iter_cache(path: Path, limit: int):
    with path.open() as stream:
        for expected, line in enumerate(stream):
            if expected >= limit:
                raise ValueError("Cache exceeds video frame count")
            row = json.loads(line)
            if row["frame_index"] != expected or abs(row["time_seconds"] - expected/100) > 1e-8:
                raise ValueError(f"Cache frame order mismatch at {expected}")
            yield row


def track(args) -> None:
    from types import SimpleNamespace
    from scripts.anti_uav.dist_numpy_runtime import CONFIG, as_results, load_dist, observations
    from scripts.anti_uav.efficient_gmc import EfficientGMC

    rows = manifest_rows(args.manifest)
    count, fps, _, _ = video_info(args.source)
    meta = json.loads((args.detector_dir / "summary.json").read_text())
    if (meta["frames"] != count or meta["model_sha256"] != digest(args.model)
            or meta["manifest_sha256"] != digest(args.manifest / "manifest.json")
            or meta["source_sha256"] != digest(args.source)):
        raise ValueError("Detector provenance differs")
    cache = args.detector_dir / f"{args.variant}_detections.jsonl"
    if digest(cache) != meta[f"{args.variant}_cache_sha256"]:
        raise ValueError("Detector cache changed")
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(20260924)
    tracker = load_dist(args.upstream)(SimpleNamespace(**CONFIG), frame_rate=int(fps))
    tracker.gmc = EfficientGMC(320, 128, 5, True)
    source = cv2.VideoCapture(str(args.source))
    displayed = tracked_frames = 0
    identities = set()
    started = time.monotonic()
    try:
        with (args.output / "tracks.jsonl").open("x") as stream:
            for index, row in enumerate(iter_cache(cache, count)):
                ok, original = source.read()
                if not ok:
                    raise RuntimeError(f"Source decode failed at {index}")
                frame = selected_frame(original, args.manifest, rows, index, args.variant)
                boxes = np.asarray(row["boxes_xyxy_score"], np.float32).reshape(-1, 5)
                tracker.update(as_results(boxes), img=frame)
                shown = observations(tracker, boxes)
                stream.write(json.dumps(dict(frame_index=index, time_seconds=index/fps,
                    boxes_xyxy_score=row["boxes_xyxy_score"], displayed_tracks=shown)) + "\n")
                displayed += len(shown)
                tracked_frames += bool(shown)
                identities.update(item["id"] for item in shown)
                if index % 1000 == 0:
                    print(json.dumps(dict(stage="track", variant=args.variant,
                                          frame=index, total=count)), flush=True)
        if index + 1 != count:
            raise ValueError("Tracker did not cover every source frame")
    finally:
        source.release()
    write_json(args.output / "summary.json", dict(stage="complete", variant=args.variant,
        frames=count, fps=fps, detector_cache_sha256=digest(cache),
        tracks_sha256=digest(args.output / "tracks.jsonl"), model_sha256=digest(args.model),
        config=CONFIG, gmc="causal EfficientGMC(320,128,refresh=5,resize_first=True)",
        gmc_counts=tracker.gmc.counts, displayed_tracks=displayed,
        frames_with_track=tracked_frames, distinct_displayed_ids=len(identities),
        seconds=time.monotonic()-started,
        note="Synthetic targets vary by sampled frame; IDs are visual only, not tracking accuracy."))


def render(args) -> None:
    from scripts.anti_uav.render_cached_tracker_result_video import panel

    rows = manifest_rows(args.manifest)
    count, fps, _, _ = video_info(args.source)
    tracked = json.loads((args.tracker_dir / "summary.json").read_text())
    if tracked["variant"] != args.variant or tracked["frames"] != count:
        raise ValueError("Tracker result does not match requested variant")
    cache = args.tracker_dir / "tracks.jsonl"
    if digest(cache) != tracked["tracks_sha256"]:
        raise ValueError("Tracker cache changed")
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / f"Video00009_{args.variant}_FP32_960x544_Dist_GMC_conf003.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", "1600x784", "-r", str(int(fps)), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "6",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    source = cv2.VideoCapture(str(args.source))
    title = (f"Video00009 {args.variant.upper()} | Frozen-P3 + Add-on P2 | "
             "FP32 960x544 | Dist+GMC | conf 0.03")
    written = 0
    started = time.monotonic()
    with (args.output / "encode.log").open("x") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=log)
        try:
            for index, row in enumerate(iter_cache(cache, count)):
                ok, original = source.read()
                if not ok:
                    raise RuntimeError(f"Source decode failed at {index}")
                frame = selected_frame(original, args.manifest, rows, index, args.variant)
                for item in row["displayed_tracks"]:
                    observation = row["boxes_xyxy_score"][item["detection_index"]]
                    if not np.allclose(item["box"] + [item["score"]], observation, rtol=0, atol=1e-4):
                        raise ValueError(f"Track is not current detection at {index}")
                canvas = panel(frame, row["displayed_tracks"], index, count, fps, title)
                if args.variant == "synthetic" and index in rows and rows[index]["state"] == "replaced":
                    cv2.putText(canvas, "SYNTHETIC TARGET FRAME", (1290, 31),
                                cv2.FONT_HERSHEY_SIMPLEX, .51, (255, 210, 40), 1, cv2.LINE_AA)
                encoder.stdin.write(canvas.tobytes())
                if index in {0, 1060, 3760, 6960, 12130, count-1}:
                    cv2.imwrite(str(args.output / f"preview_{index:06d}.jpg"), canvas)
                written += 1
                if index % 1000 == 0:
                    print(json.dumps(dict(stage="render", variant=args.variant,
                                          frame=index, total=count)), flush=True)
        finally:
            source.release()
            if encoder.stdin:
                encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed, see {args.output / 'encode.log'}")
    if written != count:
        raise ValueError("Video does not cover every source frame")
    write_json(args.output / "summary.json", dict(stage="complete", variant=args.variant,
        frames=written, fps=fps, video=str(destination), video_sha256=digest(destination),
        tracks_sha256=tracked["tracks_sha256"], source_sha256=digest(args.source),
        substituted_frames=565 if args.variant == "synthetic" else 0,
        seconds=time.monotonic()-started, ground_truth_used=False,
        note="Playback rate is source timing, not inference FPS; only currently observed tracks shown."))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="stage", required=True)
    for name in ("infer", "track", "render"):
        cmd = actions.add_parser(name)
        cmd.add_argument("--source", type=Path, required=True)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        if name in ("track", "render"):
            cmd.add_argument("--variant", choices=("real", "synthetic"), required=True)
        if name in ("infer", "track"):
            cmd.add_argument("--model", type=Path, required=True)
        if name == "infer":
            cmd.add_argument("--device", default="0")
            cmd.add_argument("--batch", type=int, default=16)
            cmd.add_argument("--max-frames", type=int)
        if name == "track":
            cmd.add_argument("--detector-dir", type=Path, required=True)
            cmd.add_argument("--upstream", type=Path, required=True)
        if name == "render":
            cmd.add_argument("--tracker-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "infer":
        if args.batch < 1 or (args.max_frames is not None and args.max_frames < 1):
            parser.error("Invalid batch or frame count")
        infer(args)
    elif args.stage == "track":
        track(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
