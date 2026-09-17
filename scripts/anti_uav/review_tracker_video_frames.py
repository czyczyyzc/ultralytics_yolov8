#!/usr/bin/env python3
"""Extract contact sheets from delivered videos and clean source for visual audit."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def corner(image, box, color, label=""):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    length = max(2, min(9, (x2-x1)//3, (y2-y1)//3))
    for x, y, sx, sy in ((x1,y1,1,1), (x2,y1,-1,1), (x1,y2,1,-1), (x2,y2,-1,-1)):
        cv2.line(image, (x,y), (x+sx*length,y), color, 1)
        cv2.line(image, (x,y), (x,y+sy*length), color, 1)
    if label:
        cv2.putText(image, label, (max(1,x1), max(15,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detector-dir", type=Path, required=True)
    p.add_argument("--tracker-dir", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--coco", type=Path)
    p.add_argument("--frames", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.source:
        summary = json.loads((args.detector_dir / "summary.json").read_text())
        if hashlib.sha256(args.source.read_bytes()).hexdigest() != summary["source_sha256"]:
            raise ValueError("Source video incomplete or changed")
    args.output.mkdir(parents=True)
    frames = [int(x) for x in args.frames.split(",")]
    videos = [next(folder.glob("*.mp4")) for folder in (args.detector_dir, args.tracker_dir)]
    caps = [cv2.VideoCapture(str(v)) for v in videos]
    for start in range(0, len(frames), 4):
        chosen = frames[start:start+4]
        sheet = np.full((len(chosen)*392, 1600, 3), 20, np.uint8)
        for row, index in enumerate(chosen):
            for col, cap in enumerate(caps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode delivered video frame {index}")
                sheet[row*392:(row+1)*392,col*800:(col+1)*800] = cv2.resize(image,(800,392),interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.output / f"delivered_{start:03d}.jpg"), sheet)
    for cap in caps:
        cap.release()
    if not args.source:
        return
    det = [json.loads(l) for l in (args.detector_dir / "predictions.jsonl").read_text().splitlines()]
    tr = [json.loads(l) for l in (args.tracker_dir / "tracks.jsonl").read_text().splitlines()]
    coco = json.loads(args.coco.read_text())
    byid = {i["id"]:i["frame_index"] for i in coco["images"]}
    gt = {byid[a["image_id"]]:a["bbox"] for a in coco["annotations"]}
    cap = cv2.VideoCapture(str(args.source))
    metadata = []
    for start in range(0, len(frames), 6):
        chosen = frames[start:start+6]
        sheet = np.full((len(chosen)*270, 1200, 3), 25, np.uint8)
        for row, index in enumerate(chosen):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode source frame {index}")
            x,y,w,h = gt[index]
            crop_w = int(max(120, w*1.5, h*2.5))
            crop_h = int(crop_w*.6)
            crop_w, crop_h = min(crop_w,frame.shape[1]), min(crop_h,frame.shape[0])
            x0 = int(np.clip(x+w/2-crop_w/2,0,frame.shape[1]-crop_w))
            y0 = int(np.clip(y+h/2-crop_h/2,0,frame.shape[0]-crop_h))
            clean = cv2.resize(frame[y0:y0+crop_h,x0:x0+crop_w],(400,240),interpolation=cv2.INTER_CUBIC)
            tiles = [clean.copy() for _ in range(3)]
            for b in det[index]["boxes_xyxy_score"]:
                # Explicit scaling avoids resizing already drawn overlays.
                transformed = (np.array(b[:4])-[x0,y0,x0,y0])*np.array([400/crop_w,240/crop_h,400/crop_w,240/crop_h])
                if transformed[2]>=0 and transformed[0]<400 and transformed[3]>=0 and transformed[1]<240:
                    corner(tiles[1],transformed,(255,215,40),f'{b[4]:.3f}')
            shown = tr[index]["displayed_tracks"]
            for t in shown:
                transformed = (np.array(t["box"])-[x0,y0,x0,y0])*np.array([400/crop_w,240/crop_h,400/crop_w,240/crop_h])
                if transformed[2]>=0 and transformed[0]<400 and transformed[3]>=0 and transformed[1]<240:
                    corner(tiles[2],transformed,(80,230,120),f'ID {t["id"]} {t["score"]:.3f}')
            for col,tile in enumerate(tiles):
                sheet[row*270+30:(row+1)*270,col*400:(col+1)*400] = tile
                label = (f"Source f{index} {index/100:.2f}s", "Detector conf .03", "Tracker confirmed only")[col]
                cv2.putText(sheet,label,(col*400+6,row*270+21),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1,cv2.LINE_AA)
            metadata.append(dict(tr[index], crop_xywh=[x0,y0,crop_w,crop_h],
                                 detections=det[index]["boxes_xyxy_score"]))
        cv2.imwrite(str(args.output / f"source_crops_{start:03d}.jpg"),sheet)
    cap.release()
    (args.output / "frames.json").write_text(json.dumps(dict(
        note="Diagnostic GT-centered crops only; no GT boxes drawn or used in tracker decisions. Original source resized before 1px overlays.",
        frames=metadata),indent=2)+"\n")


if __name__ == "__main__":
    main()
