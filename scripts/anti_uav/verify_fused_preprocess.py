#!/usr/bin/env python3
"""Offline pixel equivalence test; does not measure RK3588 inference speed."""
import argparse
import ctypes
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    lib=ctypes.CDLL(str(args.library.resolve()))
    fn=lib.fused_half_rgb_test
    fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
    fn.restype=ctypes.c_int
    rng=np.random.default_rng(20260920);cases=0
    cv2.setNumThreads(1)
    for w in [2,4,14,16,18,30,32,34,62,128,1920]:
        for h in [2,6,18,1080]:
            for pad in [0,1,17]:
                for gray in [False,True]:
                    storage=rng.integers(0,256,(h,w+pad,3),dtype=np.uint8)
                    frame=storage[:,:w]
                    if gray:
                        frame[:]=frame[:,:,0:1]
                    actual=np.full((h//2+4,w//2+6,3),157,np.uint8)
                    expected=actual.copy()
                    expected[2:-2,3:-3]=cv2.cvtColor(cv2.resize(frame,(w//2,h//2),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
                    view=actual[2:-2,3:-3]
                    assert fn(frame.ctypes.data,w,h,frame.strides[0],view.ctypes.data,view.strides[0])==0
                    np.testing.assert_array_equal(actual,expected,err_msg=f'{w}x{h} pad={pad} gray={gray}')
                    cases+=1
    report=dict(passed=True,cases=cases,opencv=cv2.__version__,scope='Pixel and guard-region equivalence only; not board latency or FPS.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
