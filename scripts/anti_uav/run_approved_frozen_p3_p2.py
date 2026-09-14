#!/usr/bin/env python3
"""Run audited data preparation, sequential P3/P2 training and matched holdout comparison."""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/mnt/andrew/anti_uav_model_refinement/data')
OLD_RUN = ROOT / 'runs/anti_uav/real_gray_yolov8n_frozen_p3_addon_p2_manual_clips0123_20260904'
OLD_FOLD = 'real_gray_yolo_strict_holdout_Video00004_manual_clips0123_neg5fps_v1_20260904'
INITIAL = ROOT / 'runs/anti_uav/real_gray_yolov8n_strict_holdout_Video00004_newclips01_20260902/training/strict_holdout_Video00004_neg15_newclips01_v1_20260902/weights/best.pt'
HOLDOUT = DATA_ROOT / 'real_gray_yolo_lovo_positive_mixed_v1_20260828/folds/holdout_Video00004'
RGB_ROOT = Path('/mnt/chenziye/datasets/anti_uav/anti_uav300_yolo')


def idle_gpus() -> list[str]:
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                         '--format=csv,noheader'], text=True)
    busy = set(processes.splitlines())
    return [row[0].strip() for row in csv.reader(io.StringIO(query))
            if row[1].strip() not in busy and int(row[2]) < 256 and int(row[3]) < 5]


def write_status(run: Path, stage: str, **extra):
    data = dict(stage=stage, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra)
    temporary = run / 'status.tmp'
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(run / 'status.json')
    print(json.dumps(data), flush=True)


def execute(run: Path, stage: str, command: list[str], env: dict):
    write_status(run, stage, command=command)
    with (run / 'logs' / f'{stage}.log').open('w') as log:
        subprocess.run(command, check=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def report(run: Path):
    results = {name: json.loads((run / 'evaluation' / f'{name}.json').read_text())
               for name in ('old_p3', 'new_p3', 'old_addon', 'new_addon')}
    lines = ['# Video00004: 10-video vs 15-video training', '',
             'PT-reference detector evaluation, 960x544. This is not RKNN INT8 or board FPS.',
             'Checkpoint selection: Anti-UAV300 RGB validation only. No holdout epoch selection.', '',
             '| Model | Precision (validator operating point) | Recall | mAP50 | mAP50-95 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, value in results.items():
        m = value['gray_holdout_standard']
        lines.append('| '+name+' | '+' | '.join(f'{100*m[k]:.2f}%' for k in ('precision','recall','map50','map50_95'))+' |')
    lines.extend(['', 'Fixed confidence, NMS IoU=0.45, matching IoU=0.50:', '',
                  '| conf | Model | TP | FP | FN | Precision | Recall | F1 |',
                  '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for conf in results['new_addon']['gray_holdout_fixed_thresholds']:
        for name in ('old_p3', 'new_p3', 'old_addon', 'new_addon'):
            m = results[name]['gray_holdout_fixed_thresholds'][conf]
            lines.append(f'| {conf} | {name} | {m["tp"]} | {m["fp"]} | {m["fn"]} | '
                         f'{100*m["precision"]:.2f}% | {100*m["recall"]:.2f}% | {100*m["f1"]:.2f}% |')
    deltas = {}
    for branch in ('p3', 'addon'):
        old, new = results[f'old_{branch}'], results[f'new_{branch}']
        deltas[branch] = {
            'standard_percentage_point_delta': {k:100*(new['gray_holdout_standard'][k]-old['gray_holdout_standard'][k])
                                                 for k in old['gray_holdout_standard']},
            'fixed_threshold_delta': {conf:{k:new['gray_holdout_fixed_thresholds'][conf][k]-old['gray_holdout_fixed_thresholds'][conf][k]
                                             for k in ('tp','fp','fn','precision','recall','f1')}
                                      for conf in new['gray_holdout_fixed_thresholds']}}
    (run / 'comparison.json').write_text(json.dumps(dict(results=results,deltas=deltas),indent=2)+'\n')
    (run / 'COMPARISON.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--device', help='Explicit GPU selection; shared use requires operator authorization')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--poll-seconds', type=int, default=60)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run / 'logs').mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), WANDB_MODE='disabled', WANDB_DISABLED='true',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    env.pop('ANTI_UAV_TRUST_DATASET_CACHE', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    python = sys.executable
    scripts = ROOT / 'scripts/anti_uav'
    old_p3 = OLD_RUN / 'training_p3' / OLD_FOLD / 'weights/best.pt'
    old_addon = OLD_RUN / 'training_addon/final/weights/best.pt'
    for file in (INITIAL, old_p3, old_addon, HOLDOUT/'holdout_all.yaml'):
        if not file.is_file():
            raise FileNotFoundError(file)
    try:
        if not (a.dataset_dir / 'manifest.json').exists():
            execute(run, 'prepare_data', [python, str(scripts/'build_approved_gray_rehearsal.py'),
                '--approved-root', str(DATA_ROOT/'approved_tasks'),
                '--video-root', '/mnt/andrew/video-labeler/videos',
                '--old-root', str(DATA_ROOT/'seven_old_videos'),
                '--source-data', str(DATA_ROOT/'real_gray_yolo_strict_holdout_Video00004_newclips01_v1_20260902/base_neg15/train_rgb_monitor.yaml'),
                '--rgb-root', str(RGB_ROOT), '--output', str(a.dataset_dir)], env)
        dataset = json.loads((a.dataset_dir/'manifest.json').read_text())
        if dataset['training_gray_videos'] != 15 or dataset['base_audit']['holdout'] != 'Video00004':
            raise ValueError('Dataset differs from the approved 15-video experiment')
        if a.prepare_only:
            write_status(run, 'data_ready', dataset=str(a.dataset_dir))
            return
        device = a.device
        if device is None:
            write_status(run, 'waiting_for_idle_gpu', policy='No compute process, <256 MiB, <5% utilization; three consecutive polls')
            stable = set()
            polls = 0
            while device is None:
                current = set(idle_gpus())
                overlap = stable & current
                polls = polls+1 if overlap else 1
                stable = overlap or current
                if stable and polls >= 3:
                    device = sorted(stable, key=int)[0]
                    break
                time.sleep(a.poll_seconds)
        protocol = dict(dataset=str(a.dataset_dir), data_manifest=dataset,
                        device=device, epochs_per_stage=a.epochs, batch=a.batch,
                        p3_initial=str(INITIAL), old_p3=str(old_p3), old_addon=str(old_addon),
                        seed=20260904, input_height_width=[544,960], checkpoint_selection='RGB validation only',
                        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        gpu_allocation='explicit selection' if a.device else 'observed idle; no foreign process stopped')
        (run/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        common = ['--epochs',str(a.epochs),'--batch',str(a.batch),'--workers',str(a.workers),
                  '--device',device,'--seed','20260904','--save-period','1']
        if (run/'training_p3').exists() or (run/'training_addon').exists():
            raise FileExistsError('Training output already exists; refuse silent restart/overwrite')
        execute(run,'train_p3',[python,str(scripts/'train_real_gray_yolo_lovo_fold.py'),
                '--model',str(INITIAL),'--fold-dir',str(a.dataset_dir),'--project',str(run/'training_p3'),
                '--lr0','0.0001','--warmup-epochs','0',*common],env)
        new_p3 = run/'training_p3'/a.dataset_dir.name/'weights/best.pt'
        execute(run,'train_addon',[python,str(scripts/'train_frozen_p3_addon_p2.py'),
                '--p3-model',str(new_p3),'--data',str(a.dataset_dir/'train_rgb_monitor.yaml'),
                '--project',str(run/'training_addon'),'--name','final','--lr0','0.001',
                '--warmup-epochs','1',*common],env)
        freeze = json.loads((run/'training_addon/final/training_manifest.json').read_text())
        if not all(freeze['checkpoint_legacy_regression'][key]['bit_exact'] for key in ('best','last')):
            raise RuntimeError('Frozen legacy branch verification failed')
        models = dict(old_p3=old_p3,new_p3=new_p3,old_addon=old_addon,
                      new_addon=run/'training_addon/final/weights/best.pt')
        for name, model in models.items():
            execute(run,f'evaluate_{name}',[python,str(scripts/'evaluate_real_gray_yolo_lovo_fold.py'),
                    '--model',str(model),'--fold-dir',str(HOLDOUT),
                    '--rgb-data',str(a.dataset_dir/'train_rgb_monitor.yaml'),'--skip-rgb',
                    '--output',str(run/'evaluation'/f'{name}.json'),'--device',device.split(',')[0],
                    '--batch','32','--workers',str(a.workers),
                    '--thresholds','0.01','0.03','0.05','0.10','0.25','0.40','0.45'],env)
        report(run)
        write_status(run,'complete',report=str(run/'COMPARISON.md'))
    except Exception as error:
        previous = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        write_status(run,'failed',failed_stage=previous.get('stage'),error=str(error))
        raise


if __name__ == '__main__':
    main()
