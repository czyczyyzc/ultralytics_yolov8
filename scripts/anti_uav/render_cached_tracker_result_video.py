#!/usr/bin/env python3
"""Render verified, observed tracker-cache results without rerunning inference."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.render_pt_detector_video import dump, probe, sha256
from scripts.anti_uav.review_tracker_video_frames import corner


def load_records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def validate_records(detections, tracks, count, fps):
    if len(detections) != count or len(tracks) != count:
        raise ValueError("Caches must cover every source frame, including empty frames")
    for index, (det, row) in enumerate(zip(detections, tracks)):
        for record in (det, row):
            if record["frame_index"] != index or not np.isclose(
                    record["time_seconds"], index / fps, rtol=0, atol=1e-7):
                raise ValueError(f"Frame/timestamp mismatch at {index}")
        identities, observations = set(), set()
        for track in row["displayed_tracks"]:
            if track["confirmed"] is not True or track["predicted"] is not False:
                raise ValueError(f"Unconfirmed or predicted output at {index}")
            if track not in row["raw_tracks"]:
                raise ValueError(f"Displayed output not in original tracks at {index}")
            identity, observation = track["id"], track["detection_index"]
            if (not isinstance(identity, int) or identity <= 0 or identity in identities
                    or not isinstance(observation, int) or observation < 0
                    or observation >= len(det["boxes_xyxy_score"]) or observation in observations):
                raise ValueError(f"Invalid/duplicate ID or observation at {index}")
            actual = np.asarray([*track["box"], track["score"]], dtype=float)
            expected = np.asarray(det["boxes_xyxy_score"][observation], dtype=float)
            if (actual.shape != (5,) or not np.isfinite(actual).all()
                    or actual[2] <= actual[0] or actual[3] <= actual[1]
                    or not np.allclose(actual, expected, rtol=0, atol=1e-5)):
                raise ValueError(f"Tracker output is not the associated detection at {index}")
            identities.add(identity)
            observations.add(observation)


def crop_bounds(box, width, height):
    # Keep large targets in view too, rather than clipping them to a tiny inset.
    crop_w = min(width, max(160, int(np.ceil(max((box[2]-box[0])*1.4,
                                               (box[3]-box[1])*1.4*1.6)))))
    crop_h = min(height, max(1, int(np.ceil(crop_w / 1.6))))
    left = int(np.clip((box[0]+box[2]-crop_w)/2, 0, width-crop_w))
    top = int(np.clip((box[1]+box[3]-crop_h)/2, 0, height-crop_h))
    return left, top, crop_w, crop_h


def panel(frame, tracks, index, count, fps, title):
    height, width = frame.shape[:2]
    canvas = np.full((784, 1600, 3), (25, 29, 35), np.uint8)
    color = (255, 210, 40)
    main = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
    shown = sorted(tracks, key=lambda t: (-t["score"], t["id"]))
    for track in shown:
        box = np.asarray(track["box"]) * [1280/width, 720/height, 1280/width, 720/height]
        corner(main, box, color)
        xy = (int(np.clip(box[0], 1, 1180)), int(np.clip(box[1]-7, 16, 713)))
        cv2.putText(main, f"ID {track['id']}", xy, cv2.FONT_HERSHEY_SIMPLEX,
                    .45, color, 1, cv2.LINE_AA)
    canvas[64:, :1280] = main
    cv2.putText(canvas, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX,
                .61, (240, 240, 240), 1, cv2.LINE_AA)
    status = (f"frame {index+1}/{count} | {index/fps:.2f}s | confirmed {len(shown)}"
              f" | NO GT | original speed {fps:g} FPS")
    cv2.putText(canvas, status, (16, 52), cv2.FONT_HERSHEY_SIMPLEX,
                .53, (210, 215, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CLEAN-SOURCE CROPS", (1291, 91), cv2.FONT_HERSHEY_SIMPLEX,
                .49, (220, 225, 230), 1, cv2.LINE_AA)
    for rank in range(2):
        y = 125 + rank*285
        if rank >= len(shown):
            cv2.putText(canvas, "NO CONFIRMED TRACK", (1290, y+105),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (130, 135, 140), 1, cv2.LINE_AA)
            continue
        track = shown[rank]
        left, top, cw, ch = crop_bounds(track["box"], width, height)
        scale = min(320/cw, 200/ch)
        rw, rh = max(1, round(cw*scale)), max(1, round(ch*scale))
        dx, dy = (320-rw)//2, (200-rh)//2
        crop = np.full((200, 320, 3), (25, 29, 35), np.uint8)
        crop[dy:dy+rh, dx:dx+rw] = cv2.resize(
            frame[top:top+ch, left:left+cw], (rw, rh),
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        for other in shown:
            box = ((np.asarray(other["box"]) - [left, top, left, top])
                   * [rw/cw, rh/ch, rw/cw, rh/ch] + [dx, dy, dx, dy])
            if box[2] >= 0 and box[0] < 320 and box[3] >= 0 and box[1] < 200:
                corner(crop, box, color)
        canvas[y:y+200, 1280:] = crop
        cv2.putText(canvas, f"ID {track['id']} | score {track['score']:.3f} | {scale:.1f}x",
                    (1290, y+225), cv2.FONT_HERSHEY_SIMPLEX, .46, color, 1, cv2.LINE_AA)
    for y, text in ((713, "Crops follow top scores."), (737, "Original IDs; no remapping."),
                    (761, "1px corners; no crosshair.")):
        cv2.putText(canvas, text, (1290, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (190, 195, 200), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--tracker-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-name", default="Video00009_Dist_public_GMC_conf003_full.mp4")
    parser.add_argument("--label", default="Dist public-code + GMC")
    parser.add_argument("--max-frames", type=int, help="Optional smoke-test prefix, not a full video")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (Path(args.output_name).name != args.output_name or not args.output_name.endswith(".mp4")
            or (args.max_frames is not None and args.max_frames < 1)):
        raise ValueError("Invalid output name or frame limit")
    detector = json.loads((args.detector_dir / "summary.json").read_text())
    tracker = json.loads((args.tracker_dir / "summary.json").read_text())
    source_hash = sha256(args.source)
    detector_path, tracks_path = args.detector_dir / "predictions.jsonl", args.tracker_dir / "tracks.jsonl"
    detection_hash = sha256(detector_path)
    if source_hash != detector["source_sha256"] or source_hash != tracker["provenance"]["source_sha256"]:
        raise ValueError("Source does not match the evaluated detector/tracker inputs")
    if detection_hash != tracker["detector_cache_sha256"]:
        raise ValueError("Detection cache does not match the evaluated tracker input")
    if detector["weights_sha256"] != tracker["provenance"]["weights_sha256"]:
        raise ValueError("Detector weights provenance mismatch")
    src = probe(args.source)
    count, fps = int(src["nb_frames"]), float(Fraction(src["avg_frame_rate"]))
    if (fps <= 0 or Fraction(src["avg_frame_rate"]) != Fraction(src["r_frame_rate"])
            or src["width"]*9 != src["height"]*16):
        raise ValueError("Layout requires a constant-frame-rate 16:9 source")
    detections, tracks = load_records(detector_path), load_records(tracks_path)
    validate_records(detections, tracks, count, fps)
    limit = min(count, args.max_frames) if args.max_frames else count
    args.output.mkdir(parents=True)
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise ValueError("Cannot open original source")
    destination = args.output / args.output_name
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", "1600x784", "-r", src["avg_frame_rate"], "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "4",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    stats = dict(source=str(args.source.resolve()), source_sha256=source_hash, source_info=src,
                 detector_cache=str(detector_path.resolve()), detector_cache_sha256=detection_hash,
                 tracks_cache=str(tracks_path.resolve()), tracks_sha256=sha256(tracks_path),
                 weights_sha256=detector["weights_sha256"], detector_conf=detector["conf"],
                 input_hw=detector["input_hw"], precision=detector["precision"],
                 tracker_config=tracker["config"], tracker_provenance=tracker["provenance"],
                 tracker_label=args.label, full_source_video=limit == count, frame_stride=1,
                 ground_truth_used=False, predicted_or_pending_boxes_shown=False,
                 inference_rerun=False, box_mode="associated_detection", id_remapping=False,
                 audio_included=False, output_fps=fps, video=str(destination.resolve()), command=command,
                 visualization="1px corners, no crosshair/GT; resize clean source before overlays; adaptive crops",
                 note="Playback FPS is source timing, not board inference throughput.")
    dump(args.output / "protocol.json", stats)
    title = (f"Frozen-P3 + Add-on P2 | {args.label} | "
             f"input {detector['input_hw'][1]}x{detector['input_hw'][0]} | conf {detector['conf']:.2f}")
    frames_written = outputs = frames_with_output = 0
    ids, started = set(), time.monotonic()
    with (args.output / "encode.log").open("x") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for index in range(limit):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Source ended at {index}, expected {limit}")
                shown = tracks[index]["displayed_tracks"]
                rendered = panel(frame, shown, index, limit, fps, title)
                encoder.stdin.write(rendered.tobytes())
                if index in {0, 970, 8999, 11954, limit-1}:
                    if not cv2.imwrite(str(args.output / f"preview_{index:06d}.jpg"), rendered):
                        raise RuntimeError("Preview write failed")
                frames_written += 1
                outputs += len(shown)
                frames_with_output += bool(shown)
                ids.update(t["id"] for t in shown)
                if frames_written % 512 == 0 or frames_written == limit:
                    progress = dict(stage="rendering", frames=frames_written, total=limit,
                                    seconds=round(time.monotonic()-started, 2))
                    dump(args.output / "status.json", progress)
                    print(json.dumps(progress), flush=True)
            if limit == count and cap.read()[0]:
                raise ValueError("Source contains more frames than its metadata/caches")
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed; inspect encode.log")
            out = probe(destination)
            if (int(out["nb_frames"]) != limit or Fraction(out["avg_frame_rate"]) != Fraction(src["avg_frame_rate"])
                    or abs(float(out["duration"])-limit/fps) > .02):
                raise ValueError("Encoded frame count, FPS or duration mismatch")
            stats.update(frames=frames_written, total_outputs=outputs, frames_with_output=frames_with_output,
                         visible_ids=len(ids), output_info=out, output_sha256=sha256(destination),
                         seconds=round(time.monotonic()-started, 2))
            dump(args.output / "summary.json", stats)
            dump(args.output / "status.json", dict(stage="complete", frames=frames_written, video=str(destination.resolve())))
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
