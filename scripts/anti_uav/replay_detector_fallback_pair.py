"""Compare two FP32 detectors with the same empty-frame top-one fallback."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from scripts.anti_uav.prepare_labeler_training_expansion import approved_frame_binding, dump, frame_plan, sha256, snapshot_tasks
from scripts.anti_uav.render_pt_detector_pair_video import box_iou, make_panel, metrics, text_with_shadow
from scripts.anti_uav.render_pt_detector_video import probe
from ultralytics import YOLO


def select_predictions(candidates, primary=.01, fallback=.001):
    high = candidates[candidates[:, 4] >= primary]
    if len(high):
        return high, False
    low = candidates[candidates[:, 4] >= fallback]
    if not len(low):
        return high, False
    best = low[:, 4].argmax()
    return low[best:best + 1], True


def score(gt, detections):
    overlaps = box_iou(gt, detections[:, :4])
    remaining = set(range(len(gt)))
    matches = []
    for i in np.argsort(-detections[:, 4], kind="stable"):
        if remaining:
            g = max(sorted(remaining), key=lambda j: overlaps[j, i])
            if overlaps[g, i] >= .5:
                matches.append((g, int(i)))
                remaining.remove(g)
    return dict(tp=len(matches), fp=len(detections)-len(matches), fn=len(remaining)), matches


def explain_misses(gt, selected, candidates, matched):
    counts = Counter()
    kept_gt = {g for g, _ in matched}
    for idx, box in enumerate(gt):
        if idx in kept_gt:
            continue
        overlap = box_iou(box[None], candidates[:, :4])[0]
        if (box_iou(box[None], selected[:, :4])[0] >= .5).any():
            counts["candidate_used_by_another_gt"] += 1
        elif (overlap >= .5).any():
            reason = "valid_low_score_blocked_by_primary_box" if (selected[:, 4] >= .01).any() else "valid_low_score_not_top1"
            counts[reason] += 1
        elif len(candidates):
            wh = box[2:]-box[:2]
            centers = (candidates[:, :2]+candidates[:, 2:4]-box[:2]-box[2:])*.5
            nearby = np.linalg.norm(centers, axis=1) <= max(4., float(wh.max())*.5)
            if nearby.any():
                other = candidates[nearby]
                sizes = np.maximum(other[:, 2:4]-other[:, :2], 0)
                inter = np.minimum(sizes,wh).prod(1)
                centered_iou = inter/(sizes.prod(1)+wh.prod()-inter)
                counts["nearby_box_center_error" if (centered_iou >= .5).any() else "nearby_box_size_or_shape_error"] += 1
            else:
                counts["no_nearby_post_nms_candidate"] += 1
        else:
            counts["no_nearby_post_nms_candidate"] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("database", "video-root", "frame-assets", "old", "new", "output"):
        p.add_argument(f"--{name}", type=lambda s: Path(s).resolve(), required=True)
    for name in ("source-sha", "old-sha", "new-sha"):
        p.add_argument(f"--{name}", required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    item = next(x for x in snapshot_tasks(a.database, a.video_root, set(), submitted_positives_only=True)
                if x["source_sha256"] == a.source_sha)
    identity, fingerprints = approved_frame_binding(a.database, a.frame_assets, item)
    if sha256(item["video"]) != a.source_sha:
        raise ValueError("Source bytes do not match the approved video")
    w, h, n, fps = (item[key] for key in ("frame_width","frame_height","frame_count","fps"))
    gt_plan, _ = frame_plan(item["annotations"], n, w, h, True)
    if set(gt_plan) != set(range(n)):
        raise ValueError("Video is not fully reviewed")
    specs = {
        "74_student": (a.old,a.old_sha,"74-video P3 Student",(40,175,255)),
        "97_initial": (a.new,a.new_sha,"97-video P3 initial best (epoch 7)",(255,215,40)),
    }
    models = {}
    for name,(weights,digest,*_) in specs.items():
        if sha256(weights) != digest:
            raise ValueError(f"Wrong checkpoint: {name}")
        models[name] = YOLO(str(weights))
    opts = [cv2.CAP_PROP_N_THREADS,1] if hasattr(cv2,"CAP_PROP_N_THREADS") else []
    cap = cv2.VideoCapture(item["video"],cv2.CAP_FFMPEG,opts)
    if not cap.isOpened():
        raise OSError("Cannot decode the original video")
    a.output.mkdir(parents=True,exist_ok=False)
    video = a.output/"00008_74student_vs_97initial_FP32_conf001_fallback0001.mp4"
    command = ["ffmpeg","-hide_banner","-loglevel","error","-n","-f","rawvideo","-pix_fmt","bgr24",
               "-s","1920x580","-r",str(fps),"-i","pipe:0","-an","-c:v","libx264","-preset","veryfast",
               "-crf","17","-threads","3","-pix_fmt","yuv420p","-movflags","+faststart",str(video)]
    totals = {k:{t:Counter(tp=0,fp=0,fn=0) for t in ("0.030","0.010","0.001","adaptive")} for k in specs}
    audit = {k:Counter(empty_primary=0,fallback_frames=0,fallback_tp=0,fallback_fp=0) for k in specs}
    causes = {k:Counter() for k in specs}
    rows = []
    try:
        with subprocess.Popen(command,stdin=subprocess.PIPE) as enc:
            for start in range(0,n,12):
                frames = []
                for i in range(start,min(n,start+12)):
                    ok, im = cap.read()
                    if not ok or im.shape != (h,w,3) or hashlib.sha256(im.tobytes()).hexdigest() != fingerprints[i]:
                        raise ValueError(f"Frame does not match approved pixels: {i}")
                    frames.append(im)
                gray = [cv2.cvtColor(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR) for im in frames]
                # Ultralytics uses a strict comparison at its first score filter.
                floor = float(np.nextafter(np.float32(.001),np.float32(0)))
                results = {k:m.predict(gray,imgsz=[544,960],rect=False,device="cpu",conf=floor,iou=.45,
                    max_det=100,half=False,verbose=False,save=False) for k,m in models.items()}
                for offset, frame in enumerate(frames):
                    i = start+offset
                    gt = np.asarray([[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
                                     for x,y,bw,bh in gt_plan[i]],np.float32).reshape(-1,4)
                    row = dict(frame_number=i+1,gt=gt.tolist(),models={})
                    panels = []
                    for name, predictions in results.items():
                        candidates = predictions[offset].boxes.data.cpu().numpy()[:,:5]
                        candidates = candidates[candidates[:,4]>=.001]
                        selected, used = select_predictions(candidates)
                        counts, matched = score(gt,selected)
                        totals[name]["adaptive"].update(counts)
                        for c in (.03,.01,.001):
                            totals[name][f"{c:.3f}"].update(score(gt,candidates[candidates[:,4]>=c])[0])
                        audit[name].update(empty_primary=int(not (candidates[:,4]>=.01).any()),
                            fallback_frames=int(used),fallback_tp=counts["tp"] if used else 0,
                            fallback_fp=counts["fp"] if used else 0)
                        causes[name].update(explain_misses(gt,selected,candidates,matched))
                        row["models"][name] = dict(pred=selected.tolist(),candidates=candidates.tolist(),
                            fallback_used=used,**counts)
                        panel,_ = make_panel(frame,gt,selected,specs[name][2],specs[name][3],(960,540),.5,show_gt=False)
                        if used:
                            text_with_shadow(panel,f"Fallback top 1 | {selected[0,4]:.5f}",(14,520),.55,(240,240,240))
                        panels.append(panel)
                    canvas = np.full((580,1920,3),18,np.uint8)
                    canvas[:540,:960],canvas[:540,960:] = panels
                    cv2.line(canvas,(959,0),(959,539),(210,210,210),1)
                    text_with_shadow(canvas,f"00008.ts | frame {i+1}/{n} | FP32 | 960x544 | conf 0.01 | "
                        "if empty: top 1 >= 0.001 | predictions only",(16,566),.61,(240,240,240))
                    enc.stdin.write(canvas.tobytes())
                    if i+1 in (61,169,174):
                        if not cv2.imwrite(str(a.output/f"frame_{i+1:04}.png"),canvas):
                            raise OSError("Could not write preview")
                    rows.append(row)
            enc.stdin.close()
            if enc.wait() != 0 or cap.read()[0] or int(probe(video)["nb_frames"]) != n:
                raise ValueError("Decoded or encoded frame coverage differs")
    finally:
        cap.release()
    for name in specs:
        assert sum(causes[name].values()) == totals[name]["adaptive"]["fn"]
        for key in ("tp","fp"):
            assert totals[name]["adaptive"][key]-totals[name]["0.010"][key] == audit[name][f"fallback_{key}"]
    summary = dict(clip=item["original_path"],video_sha256=sha256(video),source_sha256=a.source_sha,
        source_frames=n,source_fps=fps,gt_identity=identity,gt=sum(len(r["gt"]) for r in rows),
        weights={k:dict(path=str(v[0]),sha256=v[1]) for k,v in specs.items()},
        protocol=dict(primary=.01,fallback=.001,input_wh=[960,544],gray3=True,nms_iou=.45,matching_iou=.5,
                      detector_only=True,fp32=True,gt_not_used_for_selection=True),
        metrics={k:{t:metrics(v) for t,v in group.items()} for k,group in totals.items()},
        fallback_audit=audit,miss_causes=causes,last_frame={k:{x:y for x,y in v.items() if x!="candidates"}
            for k,v in rows[-1]["models"].items()})
    dump(a.output/"predictions_and_gt.json",rows)
    dump(a.output/"summary.json",summary)
    print(a.output/"summary.json")


if __name__ == "__main__":
    main()
