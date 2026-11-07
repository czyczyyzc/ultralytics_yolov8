#!/usr/bin/env python3
"""Snapshot P3 weights and compare per-frame gray validation failures read-only."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np


def edge_bin(edge):
    return 'lt4' if edge < 4 else '4to8' if edge <= 8 else '8to16' if edge <= 16 else 'gt16'


def compare_cases(reference, candidate):
    before = {row['image']: row for row in reference}
    after = {row['image']: row for row in candidate}
    if len(before) != len(reference) or len(after) != len(candidate) or before.keys() != after.keys():
        raise ValueError('Comparison frame coverage differs')
    changes, counts, by_size, reasons = [], Counter(), {}, Counter()
    for image, old in before.items():
        new = after[image]
        if old['gt'] != new['gt'] or old['gt_classes'] != new['gt_classes']:
            raise ValueError('Comparison ground truth differs')
        for index, edge in enumerate(old['gt_long_edge_input_px']):
            old_hit, new_hit = index in old['matched_gt'], index in new['matched_gt']
            state = 'retained' if old_hit and new_hit else 'lost' if old_hit else 'gained' if new_hit else 'missed_both'
            size = edge_bin(edge)
            counts[state] += 1
            by_size.setdefault(size, Counter())[state] += 1
            if state not in ('lost', 'gained'):
                continue
            reason = None
            if state == 'lost':
                if new['gt_best_iou'][index] >= .5:
                    reason = 'iou50_candidate_below_conf_or_assignment'
                elif new['gt_best_iou'][index] >= .1:
                    reason = 'partial_overlap_below_iou50'
                else:
                    reason = 'no_overlapping_candidate_above_0001_floor'
                reasons[reason] += 1
            changes.append(dict(image=image, gt_index=index, input_long_edge_px=edge,
                state=state, diagnostic=reason, old_best_iou=old['gt_best_iou'][index],
                new_best_iou=new['gt_best_iou'][index],
                old_iou50_score=old['gt_best_iou50_score'][index],
                new_iou50_score=new['gt_best_iou50_score'][index]))
    return dict(counts=dict(counts), by_size={k: dict(v) for k, v in by_size.items()},
                lost_diagnostics=dict(reasons), changes=changes)


def case_summary(rows):
    positive = [row for row in rows if row['gt']]
    negative = [row for row in rows if not row['gt']]
    return dict(frames=len(rows), positive_frames=len(positive), negative_frames=len(negative),
        gt=sum(len(row['gt']) for row in rows),
        positive_fp=sum(row['fp'] for row in positive), negative_fp=sum(row['fp'] for row in negative),
        positive_frames_with_fp=sum(row['fp'] > 0 for row in positive),
        negative_frames_with_fp=sum(row['fp'] > 0 for row in negative))


def draw_examples(output, models, comparison, limit=8):
    import cv2
    from PIL import Image, ImageDraw
    lookup = [{row['image']: row for row in rows} for rows in models]
    examples = [x for x in comparison['changes'] if x['state'] == 'lost']
    examples.sort(key=lambda x: (-x['old_iou50_score'], x['image']))
    selected, buckets = [], Counter()
    for row in examples:
        size = edge_bin(row['input_long_edge_px'])
        if buckets[size] >= 3:
            continue
        # Avoid filling the evidence sheet with adjacent nearly identical frames.
        stem = Path(row['image']).stem
        frame = int(stem) if stem.isdigit() else None
        if frame is not None and any(abs(frame - prior) < 30 for prior in selected):
            continue
        if frame is not None:
            selected.append(frame)
        buckets[size] += 1
        source = cv2.imread(row['image'])
        if source is None:
            raise FileNotFoundError(row['image'])
        h, w = source.shape[:2]
        gt = lookup[0][row['image']]['gt'][row['gt_index']]
        cx, cy = (gt[0]+gt[2])/2, (gt[1]+gt[3])/2
        radius = max(32, max(gt[2]-gt[0], gt[3]-gt[1])*3)
        x0, y0 = max(0, int(cx-radius)), max(0, int(cy-radius))
        x1, y1 = min(w, int(cx+radius)), min(h, int(cy+radius))
        sheet = Image.new('RGB', (1200, 700), '#20242a')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), f"Frame {Path(row['image']).stem} | input target {row['input_long_edge_px']:.2f}px | GT green, prediction cyan", fill='white')
        draw.text((12, 27), str(row['diagnostic']), fill='white')
        for column, table in enumerate(lookup):
            record = table[row['image']]
            image = source.copy()
            for box in record['gt']:
                a, b, c, d = map(round, box)
                cv2.rectangle(image, (a,b), (c,d), (40,220,80), 1)
            for box in record['predictions']:
                if box[4] < .03:
                    continue
                a, b, c, d = map(round, box[:4])
                cv2.rectangle(image, (a,b), (c,d), (255,220,0), 1)
                cv2.putText(image, f'{box[4]:.3f}', (a,max(10,b-2)), cv2.FONT_HERSHEY_SIMPLEX,.4,(255,220,0),1)
            full = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((580,326))
            sheet.paste(full,(10+column*600,76))
            # Enlarge clean pixels first, then draw crisp boxes in the inset.
            crop = cv2.resize(source[y0:y1,x0:x1],(280,280),interpolation=cv2.INTER_NEAREST)
            sx,sy=280/(x1-x0),280/(y1-y0)
            for box in [*record['gt'], *[p for p in record['predictions'] if p[4]>=.03]]:
                if box[2]<x0 or box[0]>=x1 or box[3]<y0 or box[1]>=y1:
                    continue
                color=(40,220,80) if len(box)==4 else (255,220,0)
                a,b,c,d=map(round,[(box[0]-x0)*sx,(box[1]-y0)*sy,(box[2]-x0)*sx,(box[3]-y0)*sy])
                for x,y,dx,dy in [(a,b,1,1),(c,b,-1,1),(a,d,1,-1),(c,d,-1,-1)]:
                    cv2.line(crop,(x,y),(x+dx*7,y),color,1)
                    cv2.line(crop,(x,y),(x,y+dy*7),color,1)
            sheet.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(10+column*600,414))
            draw.text((12+column*600,54), 'OLD native69 P3' if column==0 else 'CURRENT74 P3', fill='white')
        sheet.save(output/f"lost_{Path(row['image']).stem}.jpg",quality=95)
        if len(selected) >= limit:
            break


def main():
    import torch
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.metrics import box_iou
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator
    from scripts.anti_uav.gray_deployment_trainer import matched_ground_truth
    from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256
    from scripts.anti_uav.run_native_pool_comparison import clean

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--models',nargs='+',required=True,help='name=/absolute/weights.pt; first is reference')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='6')
    p.add_argument('--batch',type=int,default=16)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'weights').mkdir()
    config=yaml.safe_load(a.data.read_text());config['input_color']='gray3'
    data=a.output/'gray3.yaml';data.write_text(yaml.safe_dump(config,sort_keys=False))
    specs={}
    for spec in a.models:
        name,path=spec.split('=',1);path=Path(path)
        if not name.replace('_','').isalnum() or name in specs:
            raise ValueError('Invalid or duplicate model name')
        target=a.output/'weights'/f'{name}.pt'
        for _ in range(3):
            before=sha256(path);shutil.copy2(path,target)
            if before==sha256(path)==sha256(target):break
        else:raise ValueError('Checkpoint is being rewritten; retry a stable snapshot')
        specs[name]=dict(source=str(path),snapshot=str(target),sha256=before)
    dump(a.output/'protocol.json',dict(models=specs,input_hw=[544,960],input_color='gray3',
        precision='FP32',conf=.03,ap_floor=.001,nms_iou=.45,match_iou=.5,max_det=100,
        training_unmodified=True,validation_used_for_checkpoint_selection=True))

    class CaseValidator(PairValidator):
        def init_metrics(self,model):
            super().init_metrics(model);self.cases=[]

        def update_metrics(self,preds,batch):
            super().update_metrics(preds,batch)
            for si,pred in enumerate(preds):
                if self.image_group(batch['im_file'][si])!='native':continue
                prepared=self._prepare_batch(si,batch)
                boxes,classes=prepared['bbox'],prepared['cls']
                native=self._prepare_pred(pred,prepared)
                selected=native[native[:,4]>=.03]
                iou=box_iou(boxes,native[:,:4])*(classes[:,None]==native[:,5])
                selected_iou=box_iou(boxes,selected[:,:4])*(classes[:,None]==selected[:,5])
                matched=matched_ground_truth(selected_iou.cpu().numpy()).tolist()
                tp=int(self._process_batch(selected,boxes,classes)[:,0].sum())
                gain=min(544/prepared['ori_shape'][0],960/prepared['ori_shape'][1])
                best_iou=iou.max(dim=1).values.tolist() if len(native) else [0.]*len(boxes)
                scores=[float(native[:,4][iou[i]>=.5].max()) if bool((iou[i]>=.5).any()) else 0. for i in range(len(boxes))]
                self.cases.append(dict(image=batch['im_file'][si],gt=boxes.cpu().tolist(),
                    gt_classes=classes.cpu().tolist(),gt_long_edge_input_px=((boxes[:,2:]-boxes[:,:2]).max(dim=1).values*gain).tolist(),
                    matched_gt=matched,gt_best_iou=best_iou,gt_best_iou50_score=scores,
                    predictions=native.cpu().tolist(),tp=tp,fp=len(selected)-tp,fn=len(boxes)-tp))

        def get_stats(self):
            stats=super().get_stats();self.metrics.frame_cases=self.cases;return stats

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results={};cases={}
    for name,spec in specs.items():
        model=YOLO(spec['snapshot'])
        m=model.val(data=str(data),validator=CaseValidator,imgsz=[544,960],rect=False,
            device=a.device,batch=a.batch,workers=4,conf=.001,iou=.45,max_det=100,
            half=False,plots=False,verbose=False,project=str(a.output/'val'),name=name)
        metric={k:v for k,v in clean(m.gray_selection).items() if k.startswith('native/')}
        rows=m.frame_cases
        if len(rows)!=1421 or sum(len(row['gt']) for row in rows)!=797:
            raise ValueError('Video00009 frame/GT coverage differs')
        dump(a.output/f'{name}_cases.json',rows)
        results[name]=dict(metrics=metric,case_summary=case_summary(rows));cases[name]=rows
        dump(a.output/'results.json',results)
        del model;torch.cuda.empty_cache()
        print('MODEL_DONE',name,json.dumps(results[name]['case_summary']),flush=True)
    reference=next(iter(specs));comparisons={}
    for name in list(specs)[1:]:
        comparisons[name]=compare_cases(cases[reference],cases[name])
    dump(a.output/'case_comparisons.json',comparisons)
    if 'current74' in cases:
        draw_examples(a.output,[cases[reference],cases['current74']],comparisons['current74'])
    print('AUDIT_COMPLETE',str(a.output),flush=True)


if __name__=='__main__':main()
