#!/usr/bin/env python3
"""Render diagnosis clips with separate detection visibility and track identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.anti_uav.review_tracker_video_frames import corner


def display_detections(boxes, raw_tracks):
    observed = {tuple(t["box"]+[t["score"]]): t for t in raw_tracks if not t["predicted"]}
    result = []
    for box in boxes:
        track = observed.get(tuple(box))
        confirmed = track is not None and track["confirmed"]
        result.append(dict(box=box[:4], score=box[4],
                           id=track["id"] if confirmed else None,
                           state="confirmed" if confirmed else "pending_detection"))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--detector-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    def read_rows(path):
        return [json.loads(line) for line in path.read_text().splitlines()]
    det = read_rows(args.detector_dir / "predictions.jsonl")
    old, new = (read_rows(folder / "tracks.jsonl") for folder in (args.baseline,args.candidate))
    summary = json.loads((args.detector_dir / "summary.json").read_text())
    candidate_summary = json.loads((args.candidate / "summary.json").read_text())
    candidate_config = candidate_summary["config"]
    if hashlib.sha256(args.coco.read_bytes()).hexdigest() != candidate_summary["provenance"]["coco_sha256"]:
        raise ValueError("Diagnostic crop annotations differ from audited GT")
    fps = summary["output_fps"]
    if fps != 100:
        raise ValueError("This diagnostic montage expects the audited 100 FPS Video00009")
    for folder in (args.baseline,args.candidate):
        s = json.loads((folder / "summary.json").read_text())
        if s["detector_cache_sha256"] != hashlib.sha256((args.detector_dir / "predictions.jsonl").read_bytes()).hexdigest():
            raise ValueError("Tracker cache uses different detections")
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != summary["source_sha256"]:
        raise ValueError("Source video hash mismatch")
    if len(det) != len(old) or len(det) != len(new):
        raise ValueError("Frame count mismatch")
    if not all(d["frame_index"] == o["frame_index"] == n["frame_index"] == i for i,(d,o,n) in enumerate(zip(det,old,new))):
        raise ValueError("Frame order mismatch")
    coco = json.loads(args.coco.read_text())
    images = {i["id"]:i["frame_index"] for i in coco["images"]}
    gt = {images[a["image_id"]]:a["bbox"] for a in coco["annotations"]}
    intervals = [(950,990),(2190,2230),(8985,9025),(11940,11980)]
    args.output.mkdir(parents=True)
    destination = args.output / "Video00009_tracker_visual_diagnosis_10x_slow.mp4"
    cap = cv2.VideoCapture(str(args.source))
    command = ["ffmpeg","-hide_banner","-loglevel","error","-n","-f","rawvideo","-pix_fmt","bgr24",
               "-s","1920x720","-r","10","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast",
               "-crf","18","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)]
    titles = ["Detector conf 0.03", "Original: confirmed only", "Candidate: confirmed only", "Candidate: DET + stable ID"]
    count = 0
    with (args.output / "encode.log").open("x") as log:
        encoder = subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for start,end in intervals:
                cap.set(cv2.CAP_PROP_POS_FRAMES,start)
                for index in range(start,end):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Cannot decode frame {index}")
                    sets = [[dict(box=b[:4],score=b[4],id=None,state="detector") for b in det[index]["boxes_xyxy_score"]],
                            [dict(t,state="confirmed") for t in old[index]["displayed_tracks"]],
                            [dict(t,state="confirmed") for t in new[index]["displayed_tracks"]],
                            display_detections(det[index]["boxes_xyxy_score"],new[index]["raw_tracks"])]
                    x,y,w,h = gt[index]
                    cw = min(frame.shape[1],int(max(120,w*1.5,h*2.5)))
                    ch = min(frame.shape[0],int(cw*.6))
                    x0 = int(np.clip(x+w/2-cw/2,0,frame.shape[1]-cw))
                    y0 = int(np.clip(y+h/2-ch/2,0,frame.shape[0]-ch))
                    canvas = np.full((720,1920,3),22,np.uint8)
                    for col,records in enumerate(sets):
                        overview = cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)
                        crop = cv2.resize(frame[y0:y0+ch,x0:x0+cw],(480,288),interpolation=cv2.INTER_CUBIC)
                        for r in records:
                            color = (255,215,40) if col==0 else ((50,210,255) if r["id"] is None else (80,230,120))
                            label = f'ID {r["id"]}' if r["id"] is not None else ("DET" if col==0 else "DET pending")
                            box = np.array(r["box"])
                            corner(overview,box*np.array([480/frame.shape[1],270/frame.shape[0]]*2),color)
                            b = (box-[x0,y0,x0,y0])*np.array([480/cw,288/ch]*2)
                            if b[2]>=0 and b[0]<480 and b[3]>=0 and b[1]<288:
                                corner(crop,b,color,f'{label} {r["score"]:.3f}')
                        xcol=col*480
                        canvas[72:342,xcol:xcol+480]=overview
                        canvas[378:666,xcol:xcol+480]=crop
                        cv2.putText(canvas,titles[col],(xcol+8,25),cv2.FONT_HERSHEY_SIMPLEX,.56,(235,235,235),1,cv2.LINE_AA)
                        cv2.putText(canvas,f'frame {index} | {index/100:.2f}s | boxes {len(records)}',(xcol+8,53),cv2.FONT_HERSHEY_SIMPLEX,.48,(200,205,210),1,cv2.LINE_AA)
                        cv2.putText(canvas,"CLEAN SOURCE CROP / NO GT BOX",(xcol+8,366),cv2.FONT_HERSHEY_SIMPLEX,.47,(190,195,200),1,cv2.LINE_AA)
                        footer = ("Same detector cache / NO predicted boxes", "Original matching / confirmed only",
                                  f'Active-first {int(candidate_config["active_first"])} / match {candidate_config["first"]:.2f} / hits {candidate_config["min_hits"]}',
                                  "Yellow DET pending is NOT a stable ID")[col]
                        cv2.putText(canvas,footer,(xcol+8,696),cv2.FONT_HERSHEY_SIMPLEX,.41,(190,195,200),1,cv2.LINE_AA)
                    encoder.stdin.write(canvas.tobytes())
                    if index in (970,2196,8999,11954):
                        if not cv2.imwrite(str(args.output / f"frame_{index:05d}.png"),canvas):
                            raise RuntimeError("Preview write failed")
                    count+=1
            encoder.stdin.close()
            if encoder.wait()!=0:
                raise RuntimeError("Encoding failed")
        finally:
            cap.release()
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
    (args.output / "protocol.json").write_text(json.dumps(dict(
        source_sha256=summary["source_sha256"],intervals_zero_based_end_exclusive=intervals,
        frames=count,source_fps=100,playback_fps=10,candidate=str(args.candidate),
        note="10x slow diagnostic montage. Offline GT used ONLY to center inspection crops, never drawn or supplied to tracker. All four columns use original detector boxes. Last column separates pending detections from stable IDs; it is NOT improved tracking recall.",
        output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
