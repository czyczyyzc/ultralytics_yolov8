#!/usr/bin/env python3
"""Read-only per-GT localization/confidence diagnosis; never exports training samples."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--new-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator
    from scripts.anti_uav.gray_deployment_trainer import matched_ground_truth
    from scripts.anti_uav.prepare_labeler_training_expansion import dump,sha256

    class CaptureValidator(PairValidator):
        def init_metrics(self,model):
            super().init_metrics(model);self.frames={}

        def update_metrics(self,preds,batch):
            super().update_metrics(preds,batch)
            for index,pred in enumerate(preds):
                path=batch['im_file'][index]
                if self.image_group(path)!='native':continue
                prepared=self._prepare_batch(index,batch)
                gt=prepared['bbox'];pred=self._prepare_pred(pred,prepared)
                full=box_iou(gt,pred[:,:4]).cpu().numpy()
                selected=pred[:,4]>=.03
                iou=full[:,selected.cpu().numpy()]
                matched=set(matched_ground_truth(iou).tolist())
                h,w=prepared['ori_shape'];gain=min(544/h,960/w)
                objects=[]
                for j,box in enumerate(gt.cpu().numpy()):
                    best=int(full[j].argmax()) if full.shape[1] else None
                    objects.append(dict(gt=box.tolist(),long_edge_px=float(max(box[2]-box[0],box[3]-box[1])*gain),
                        matched=j in matched,best_iou_at_floor=float(full[j,best]) if best is not None else 0.,
                        best_iou_at_conf003=float(iou[j].max()) if iou.shape[1] else 0.,
                        best_box_conf=float(pred[best,4]) if best is not None else 0.))
                self.frames[path]=dict(objects=objects,predictions_at_conf003=int(selected.sum()),
                    fp_at_conf003=int(selected.sum())-len(matched))

        def get_stats(self):
            out=super().get_stats();self.metrics.frame_audit=self.frames;return out

    torch.set_num_threads(4)
    data=json.loads((a.new_run/'protocol.json').read_text())['data']
    models={};identities={}
    for name,run in [('old',a.old_run),('new',a.new_run)]:
        dump(a.output/'status.json',dict(stage='evaluating',model=name,device='cpu'))
        weights=run/'p3_student_distilled_best_fp32.pt';digest=sha256(weights)
        model=YOLO(str(weights))
        metrics=model.val(data=data,validator=CaptureValidator,imgsz=[544,960],rect=False,
            device='cpu',batch=8,workers=4,conf=.001,iou=.45,max_det=100,half=False,
            plots=False,verbose=False,project=str(a.output/'val'),name=name)
        models[name]=metrics.frame_audit;identities[name]=dict(path=str(weights),sha256=digest)
        if sha256(weights)!=digest:raise ValueError('Checkpoint changed')
        dump(a.output/(name+'_frames.json'),models[name]);del model
    if models['old'].keys()!=models['new'].keys() or len(models['old'])!=1421:raise ValueError('Frame coverage changed')
    losses=[];gains=[];sizes=Counter();reasons=Counter()
    for path,frame in models['old'].items():
        other=models['new'][path]
        if [x['gt'] for x in frame['objects']]!=[x['gt'] for x in other['objects']]:raise ValueError('GT changed')
        for index,(old,new) in enumerate(zip(frame['objects'],other['objects'])):
            if old['matched'] and not new['matched']:
                size='lt4' if old['long_edge_px']<4 else '4to8' if old['long_edge_px']<=8 else 'gt8'
                reason='low_confidence' if new['best_iou_at_floor']>=.5 else 'localization_0.3to0.5' if new['best_iou_at_conf003']>=.3 else 'no_box_iou_ge0.3'
                sizes[size]+=1;reasons[reason]+=1
                losses.append(dict(image=path,gt_index=index,size=size,reason=reason,old=old,new=new))
            elif new['matched'] and not old['matched']:
                gains.append(dict(image=path,gt_index=index,old=old,new=new))
    dump(a.output/'diagnosis.json',dict(complete=True,models=identities,frames=1421,
        lost_matches=len(losses),gained_matches=len(gains),lost_sizes=dict(sizes),lost_reasons=dict(reasons),
        losses=losses,gains=gains,training_samples_exported=False,
        note='CPU FP32 diagnostic; primary comparison remains the frozen unified GPU evaluation. IoU failure does not prove visual absence.'))
    dump(a.output/'status.json',dict(stage='complete'))


if __name__=='__main__':main()
