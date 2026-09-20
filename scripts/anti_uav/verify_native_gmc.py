#!/usr/bin/env python3
"""Compare native GMC (cached and uncached) with the live Python reference."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from efficient_gmc import EfficientGMC
from native_dist_runtime import NativeGMC


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--video",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frames",type=int,default=0)
    args=p.parse_args()
    cv2.setNumThreads(1)
    flows=[EfficientGMC(320,128,5,True),NativeGMC(args.library,cached=False),NativeGMC(args.library,cached=True)]
    rng=np.random.default_rng(79)
    gray=rng.integers(0,256,(540,960),dtype=np.uint8)
    synthetic=[cv2.cvtColor(np.roll(gray,i,axis=1),cv2.COLOR_GRAY2BGR) for i in range(12)]
    synthetic.extend([np.zeros((180,320),np.uint8),np.zeros((180,320,3),np.uint8)])
    for frame in synthetic:
        values=[]
        for flow in flows:
            cv2.setRNGSeed(20260917);values.append(flow.apply(frame))
        for actual in values[1:]: np.testing.assert_allclose(actual,values[0],rtol=0,atol=1e-9)
    for flow in flows[1:]:flow.close()
    flows=[EfficientGMC(320,128,5,True),NativeGMC(args.library,cached=False),NativeGMC(args.library,cached=True)]
    cap=cv2.VideoCapture(str(args.video));assert cap.isOpened()
    count=args.frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timing=[[],[],[]];maximum=[0.,0.];exact=[0,0]
    try:
        for index in range(count):
            ok,frame=cap.read();assert ok,index
            values=[]
            for j,flow in enumerate(flows):
                cv2.setRNGSeed(20260917)
                start=time.perf_counter();values.append(flow.apply(frame));timing[j].append((time.perf_counter()-start)*1000)
            for j,actual in enumerate(values[1:]):
                maximum[j]=max(maximum[j],float(np.max(np.abs(actual-values[0]))))
                exact[j]+=int(np.array_equal(actual,values[0]))
                np.testing.assert_allclose(actual,values[0],rtol=0,atol=1e-9,err_msg=f"frame={index} variant={j}")
            if index%2000==0:print(json.dumps(dict(frame=index)),flush=True)
        assert flows[0].counts==flows[1].counts==flows[2].counts
        report=dict(passed=True,frames=count,exact_warp_frames=exact,maximum_warp_difference=maximum,
                    counts=flows[0].counts,mean_ms=[float(np.mean(x)) for x in timing],
                    native_cached_seconds=flows[2].seconds,
                    scope="Sequential same-frame GMC correctness/microbenchmark; Python, C++ raw, C++ cached. NOT end-to-end FPS.")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)
    finally:
        cap.release()
        for flow in flows[1:]:flow.close()


if __name__=="__main__":main()
