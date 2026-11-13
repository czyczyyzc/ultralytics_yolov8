#!/usr/bin/env python3
"""Exercise four full-size real augmented training batches, then discard all weights."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class PreflightComplete(Exception):
    pass


def main():
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    from ultralytics.utils.torch_utils import de_parallel
    from scripts.anti_uav.regression_repair_trainers import RegressionRepairP3Trainer
    from scripts.anti_uav.run_p3_independent_curriculum_training import common_train
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--foundation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',required=True)
    variant=p.add_mutually_exclusive_group()
    variant.add_argument('--p3-nearest-offset',action='store_true')
    variant.add_argument('--p3-signed-dfl',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    SETTINGS.update(dict(sync=False,wandb=False,clearml=False,comet=False,dvc=False,hub=False,
        mlflow=False,neptune=False,raytune=False))
    torch.set_num_threads(4)
    trainer=RegressionRepairP3Trainer
    trainer.tiny_multiplier=1.75;trainer.localization_weight=.12
    trainer.localization_min_px=4.;trainer.localization_max_px=64.
    from scripts.anti_uav.gray_deployment_trainer import FixedShapeGrayValidator
    trainer.validator_class=FixedShapeGrayValidator
    state=dict(batches=0)
    def force_full_repair(t):
        if state['batches']==0:de_parallel(t.model).criterion.begin_epoch(3)
    def inspect(t):
        if not torch.isfinite(t.loss).all():raise FloatingPointError('Non-finite real-batch repair loss')
        state['batches']+=1
        audit=de_parallel(t.model).criterion.audit_summary()
        if audit['input_hw']!=[544,960] or audit['batch_size']!=64 or audit['balance_mix']!=.5:
            raise ValueError('Preflight did not exercise the intended real canvas/batch/full repair')
        if a.p3_nearest_offset and (audit['p3_offset']['no_free_p3']
                                   or (state['batches']>=4 and not audit['p3_offset']['reserved'])):
            raise ValueError('Preflight did not reserve P3 points for the real unmatched tiny GTs')
        if a.p3_signed_dfl and state['batches']>=4 and not audit['p3_signed_dfl']['reserved']:
            raise ValueError('Signed DFL preflight did not reserve any representable off-grid points')
        report=dict(complete=state['batches']>=4,real_augmented_batches=state['batches'],
            input_hw=audit['input_hw'],
            batch_size=64,amp=True,full_repair_enabled=True,
            loss=t.loss.detach().float().cpu().tolist(),weights_discarded_not_used_for_initialization=True,
            audit=audit)
        if t.data.get('visible_box_guard'):
            report['visible_box_guard']=t.data['visible_box_guard']
            report['visible_box_guard_totals']=t.visible_box_guard_totals
        (a.output/'acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
        if state['batches']>=4:raise PreflightComplete()
    model=YOLO(str(a.foundation))
    if a.p3_nearest_offset:
        model.model.yaml['p3_nearest_offset_v1']=True
    if a.p3_signed_dfl:
        model.model.yaml['p3_signed_dfl_v1']=True
    model.add_callback('on_train_batch_start',force_full_repair)
    model.add_callback('on_train_batch_end',inspect)
    args=SimpleNamespace(batch=64,device=a.device,workers=16)
    try:
        model.train(trainer=trainer,project=str(a.output),name='probe',lr0=.001,warmup_epochs=1,
                    **common_train(a.data,args,1))
    except PreflightComplete:
        print('PREFLIGHT_PASSED',str(a.output/'acceptance.json'),flush=True)
    else:
        raise RuntimeError('Preflight failed to reach its bounded callback')


if __name__=='__main__':main()
