#!/usr/bin/env python3
"""Render four prediction-only panels and a shared clean-frame target zoom."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.anti_uav.render_pt_detector_pair_video import draw_corner_box, boxes_in_crop, text_with_shadow
from scripts.anti_uav.summarize_causal_roi_comparison import ARMS, TITLES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.root/ARMS[0]/"manifest.json").read_text())
    traces={a:[json.loads(line) for line in (args.root/a/"frames.jsonl").read_text().splitlines()] for a in ARMS}
    output=args.root/"visualization"
    output.mkdir(exist_ok=True)
    video=output/"Video00004_P3_ROI2x_ROI4x_AddonP2_RKBoTSORT.mp4"
    capture=cv2.VideoCapture(manifest["source"])
    process=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","bgr24",
                              "-s","1920x1080","-r",str(manifest["source_fps"]),"-i","pipe:0",
                              "-an","-c:v","libx264","-preset","fast","-crf","18",
                              "-pix_fmt","yuv420p","-movflags","+faststart",str(video)],stdin=subprocess.PIPE)
    colors=[(60,180,255),(60,235,160),(255,180,60),(255,225,60)]
    completed=False
    try:
        for index in range(manifest["frames"]):
            ok,frame=capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at {index}")
            height,width=frame.shape[:2]
            candidates=traces["addon_full"][index]["tracks"] or traces["p3_roi2x"][index]["tracks"] or traces["p3_full"][index]["tracks"]
            focus=max(candidates,key=lambda t:t["score"])["box"] if candidates else None
            panels=[]
            for arm,color in zip(ARMS,colors):
                r=traces[arm][index]
                panel=cv2.resize(frame,(960,540),interpolation=cv2.INTER_AREA)
                scale=np.array([960/width,540/height,960/width,540/height])
                if r["mode"] == "roi":
                    draw_corner_box(panel,np.array(r["roi"])*scale,(215,215,215),1,20)
                for t in r["tracks"]:
                    box=np.array(t["box"])*scale
                    draw_corner_box(panel,box,color,1)
                    text_with_shadow(panel,f"ID{t['id']} {t['score']:.2f}",
                                     (int(box[0])+5,int(box[1])-5),.4,color,1)
                if focus:
                    crop_w,crop_h=160,100
                    left=int(np.clip((focus[0]+focus[2])/2-crop_w/2,0,width-crop_w))
                    top=int(np.clip((focus[1]+focus[3])/2-crop_h/2,0,height-crop_h))
                    zoom=cv2.resize(frame[top:top+crop_h,left:left+crop_w],(320,200),interpolation=cv2.INTER_CUBIC)
                    arr=np.asarray([t["box"] for t in r["tracks"]],dtype=float).reshape(-1,4)
                    for box in boxes_in_crop(arr,left,top,crop_w,crop_h,2,2):
                        draw_corner_box(zoom,box,color,1)
                    panel[50:250,630:950]=zoom
                    cv2.rectangle(panel,(630,50),(950,250),(215,215,215),1)
                    text_with_shadow(panel,"2x DISPLAY ZOOM",(638,68),.4,(240,240,240),1)
                cv2.rectangle(panel,(0,0),(960,39),(18,18,18),-1)
                text_with_shadow(panel,TITLES[arm]+" + RK-BoT-SORT",(10,25),.60,color,1)
                cv2.rectangle(panel,(0,510),(960,540),(18,18,18),-1)
                text_with_shadow(panel,f"frame {index:04d}/{manifest['frames']}  {index/manifest['source_fps']:.2f}s  {r['mode']}  input 960x544",
                                 (10,530),.46,(240,240,240),1)
                panels.append(panel)
            canvas=np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])))
            process.stdin.write(canvas.tobytes())
            if index in (635,700,850,950):
                cv2.imwrite(str(output/f"comparison_frame_{index:04d}.jpg"),canvas)
            if (index+1)%500 == 0:
                print(f"rendered {index+1}/{manifest['frames']}",flush=True)
        completed=True
    finally:
        capture.release()
        process.stdin.close()
        returncode=process.wait()
    if not completed or returncode:
        raise RuntimeError(f"Rendering failed: {returncode}")
    print(video,flush=True)


if __name__ == "__main__":
    main()
