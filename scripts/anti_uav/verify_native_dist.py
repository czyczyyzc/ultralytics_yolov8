#!/usr/bin/env python3
"""Check native ID/observation equivalence; cached replay is NOT pipeline FPS."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from dist_numpy_runtime import CONFIG, as_results, load_dist, observations
from native_dist_runtime import NativeDist


def python_snapshot(tracker):
    rows = []
    for kind,tracks in enumerate((tracker.tracked_stracks,tracker.lost_stracks)):
        for t in tracks:
            rows.append([kind,t.track_id,t.state,t.is_activated,t.frame_id,t.start_frame,t.idx,
                         *t.mean.tolist(),*t.covariance.ravel().tolist()])
    return np.asarray(rows,dtype=float).reshape(-1,79)


def synthetic_cases():
    scores = [.00999,.01,.01001,.02999,.03,.09999,.1,.10001,.9]
    rng = np.random.default_rng(39)
    frames = []
    for frame in range(450):
        boxes = []
        if frame<180 or frame>300:
            for target in range(5):
                if rng.random()<.20:
                    continue
                x=30+target*12+frame*.21+rng.normal(0,.5)
                y=40+target*12+rng.normal(0,.3)
                w,h=(5,4) if target%2 else (25,20)
                boxes.append([x,y,x+w,y+h,scores[(frame+target)%len(scores)]])
        if frame%19==0:
            boxes.extend([[500,100,1500,900,.9],[500,100,1500,900,.9]])
        angle=.002*np.sin(frame*.1)
        warp=np.array([[np.cos(angle),-np.sin(angle),.15],[np.sin(angle),np.cos(angle),-.08]])
        frames.append((np.asarray(boxes,np.float32).reshape(-1,5),warp))
    yield "thresholds_duplicates_crossings_expiry", frames
    yield "expiry_reappearance", [(np.array([[100,100,110,110,.9]],np.float32) if i in (0,1,103,104,205,206)
                                   else np.empty((0,5),np.float32),np.eye(2,3)) for i in range(240)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cls=load_dist(args.upstream)
    cases=[]
    for name,frames in synthetic_cases():
        py=cls(SimpleNamespace(**CONFIG),frame_rate=100)
        cpp=NativeDist(args.library)
        maximum=0.
        try:
            for index,(boxes,warp) in enumerate(frames):
                py.gmc=SimpleNamespace(apply=lambda *unused:warp)
                py.update(as_results(boxes),img=np.empty((1,1,3),np.uint8))
                a=observations(py,boxes);b=cpp.update(boxes,warp)
                assert a==b, (name,index,a,b)
                a=python_snapshot(py);b=cpp.snapshot()
                np.testing.assert_array_equal(a[:,:7],b[:,:7],err_msg=f"{name}:{index} lifecycle")
                np.testing.assert_allclose(a[:,7:],b[:,7:],rtol=1e-6,atol=1e-6,err_msg=f"{name}:{index} Kalman")
                maximum=max(maximum,float(np.max(np.abs(a[:,7:]-b[:,7:]),initial=0)))
        finally:
            cpp.close()
        cases.append(dict(name=name,frames=len(frames),maximum_state_absolute_difference=maximum))
    cpp=NativeDist(args.library)
    times=[];count=0;tracks=0
    try:
        with args.cache.open() as stream:
            for index,line in enumerate(stream):
                row=json.loads(line)
                assert row["frame_index"]==index
                boxes=np.asarray(row["boxes_xyxy_score"],np.float32).reshape(-1,5)
                start=time.perf_counter()
                result=cpp.update(boxes,np.asarray(row["warp"]))
                times.append((time.perf_counter()-start)*1000)
                assert result==row["displayed_tracks"], ("video",index,result,row["displayed_tracks"])
                count+=1;tracks+=len(result)
    finally:
        cpp.close()
    summary=dict(passed=True,synthetic=cases,frames=count,observations=tracks,
        mean_native_bridge_ms=float(np.mean(times)),p95_native_bridge_ms=float(np.percentile(times,95)),
        scope="Cached exact ID/current-detection association check. Includes ctypes/output conversion; excludes live GMC and detector. NOT pipeline FPS.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


if __name__=="__main__":
    main()
