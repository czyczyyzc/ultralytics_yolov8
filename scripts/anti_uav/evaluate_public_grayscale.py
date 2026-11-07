#!/usr/bin/env python3
"""Evaluate frozen old/new students on the same public fixedwing gray3 holdout."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--new-run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import yaml
    from ultralytics import YOLO
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator
    from scripts.anti_uav.prepare_labeler_training_expansion import dump,sha256
    from scripts.anti_uav.run_native_pool_comparison import clean
    torch.set_num_threads(4)
    data=yaml.safe_load(a.data.read_text())
    if data.get('input_color')!='gray3':raise ValueError('Require explicit gray3 evaluation')
    listing=Path(data['val']);identity=sha256(listing)
    result={}
    for name,run in [('native69',a.old_run),('all_aircraft',a.new_run)]:
        weights=run/'p3_student_distilled_best_fp32.pt';digest=sha256(weights)
        dump(a.output/'status.json',dict(stage='evaluating',model=name,device='cpu'))
        model=YOLO(str(weights))
        metrics=model.val(data=str(a.data),validator=PairValidator,imgsz=[544,960],rect=False,
            device='cpu',batch=8,workers=4,conf=.001,iou=.45,max_det=100,half=False,
            plots=False,verbose=False,project=str(a.output/'val'),name=name)
        result[name]=dict(weights=str(weights),sha256=digest,
            metrics={k:v for k,v in clean(metrics.gray_selection).items() if k.startswith('native/')})
        if sha256(weights)!=digest or sha256(listing)!=identity:raise ValueError('Evaluation input changed')
        dump(a.output/'results.json',result)
        del model
    dump(a.output/'protocol.json',dict(input_color='gray3',frames=3451,gt=3451,positive_only=True,
        checkpoint_selection=False,threshold_tuning=False,conf=.03,ap_floor=.001,match_iou=.5,
        nms_iou=.45,input_hw=[544,960],precision='FP32',device='cpu',list_sha256=identity,
        caveat='This is a same-color diagnostic, not a new trained candidate or full-video background FP evaluation'))
    for entry in result.values():
        m=entry['metrics']
        if (m['native/c0.03/FRAMES'],m['native/c0.03/TP']+m['native/c0.03/FN'])!=(3451,3451):raise ValueError('Coverage changed')
    dump(a.output/'status.json',dict(stage='complete'))


if __name__=='__main__':main()
