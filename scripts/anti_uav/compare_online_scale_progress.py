#!/usr/bin/env python3
"""Fixed-input, matched-epoch P3 comparison while online add-on training is pending."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from scripts.anti_uav.gray_deployment_trainer import GrayDeploymentValidator


class FixedShapeValidator(GrayDeploymentValidator):
    def preprocess(self, batch):
        result = super().preprocess(batch)
        assert tuple(result['img'].shape[2:]) == (544, 960), result['img'].shape
        return result

    def get_stats(self):
        stats = super().get_stats()
        self.metrics.gray_selection['native_frames'] = self.fixed[('native', .03)]['frames']
        self.metrics.gray_selection['zoom_frames'] = self.fixed[('zoom', .03)]['frames']
        return stats


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--static-run', type=Path, required=True)
    p.add_argument('--online-run', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=14)
    p.add_argument('--device', default='6')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    SETTINGS.update(dict(sync=False, wandb=False, clearml=False, comet=False, dvc=False, hub=False,
                         mlflow=False, neptune=False, raytune=False))
    torch.set_num_threads(4)
    results = {}
    for name, run in [('static_p3', a.static_run), ('online_p3', a.online_run)]:
        weights = run/f'training_p3/p3/weights/epoch{a.epoch-1}.pt'
        assert weights.is_file(), weights
        model = YOLO(str(weights))
        assert model.ckpt['epoch'] == a.epoch-1
        metrics = model.val(data=str(a.data), imgsz=[544, 960], rect=False, validator=FixedShapeValidator,
                            device=a.device, batch=32, workers=4, conf=.001, iou=.45, max_det=100,
                            half=False, plots=False, project=str(a.output/'val'), name=name, verbose=False)
        results[name] = dict(weights=str(weights), epoch=a.epoch, metrics=clean(metrics.gray_selection))
        (a.output/f'{name}.json').write_text(json.dumps(results[name], indent=2, allow_nan=False)+'\n')
    prior = {}
    for name in ('old_addon', 'new_addon'):
        prior[name] = json.loads((a.static_run/'evaluation'/f'{name}.json').read_text())
    output = dict(input_hw=[544, 960], conf_floor=.001, nms_iou=.45, max_det=100,
                  checkpoint_policy='Matched completed P3 epoch, not test-selected; online P2 is not yet available.',
                  limitation='Native validation has no >=80% boxes; augmented stress frames are not real close-range footage.',
                  p3_interim=results, previous_completed_Video00004=prior)
    (a.output/'comparison.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    lines = ['# Interim Online Random-Scale Comparison', '',
             f'P3 only, matched epoch {a.epoch}; actual input 960x544. Online add-on P2 has not completed.', '',
             '## Independent Native Gray Validation', '',
             '| Model | Conf | P | R | FP/1000 | mAP50 | mAP50-95 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for conf in (.01, .03, .05):
        for name, result in results.items():
            m = result['metrics']; k = f'native/c{conf:.2f}'
            lines.append(f'| {name} | {conf:.2f} | {100*m[k+"/P"]:.2f}% | {100*m[k+"/R"]:.2f}% | '
                         f'{m[k+"/FP1000"]:.2f} | {100*m["native/mAP50"]:.2f}% | {100*m["native/mAP50-95"]:.2f}% |')
    lines += ['', '## Completed Add-On Models On Video00004', '',
              'These are previous completed models, not the unfinished online model. Conf=0.03.', '',
              '| Model | P | R | FP | mAP50 |', '| --- | ---: | ---: | ---: | ---: |']
    for name, value in prior.items():
        m = value['gray_holdout_fixed_thresholds']['0.03']
        lines.append(f'| {name} | {100*m["precision"]:.2f}% | {100*m["recall"]:.2f}% | {m["fp"]} | '
                     f'{100*value["gray_holdout_standard"]["map50"]:.2f}% |')
    lines += ['', 'Old add-on: original manual-clips0123 model. New add-on: completed fixed-480-augmentation model.',
              'Do not use these test results to select epochs or add Video00004 frames to training.']
    (a.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(clean(output), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
