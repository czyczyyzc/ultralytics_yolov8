#!/usr/bin/env python3
"""Run an isolated data-identical loss repair alongside an untouched control."""

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def select_idle_gpu(usage,allowed):
    candidates=[]
    for line in usage.strip().splitlines():
        index,used,free,util=(int(x.strip()) for x in line.split(','))
        if index in allowed and used<=512 and free>=60000 and util<=5:
            candidates.append((-free,util,index))
    return min(candidates)[2] if candidates else None


def main():
    import yaml
    from scripts.anti_uav.prepare_labeler_training_expansion import dump,sha256
    from scripts.anti_uav.run_approved_gray_fixedquota_refresh import compare
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-run',type=Path,required=True);p.add_argument('--native-control-run',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpus',nargs='+',type=int,default=list(range(8)))
    p.add_argument('--adopt-training-pid',type=int,help='Replace only a supervisor, preserving a verified live training process')
    p.add_argument('--adopt-gpu',type=int)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    a.run_dir.mkdir(parents=True,exist_ok=a.adopt_training_pid is not None)
    lock=(a.run_dir/'experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(stage,**extra):
        dump(a.run_dir/'repair_launch_status.json',dict(stage=stage,pid=__import__('os').getpid(),
            time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**extra))
    def finish(gpu,contract):
        while True:
            control_status=json.loads((a.control_run/'status.json').read_text())
            if control_status['stage']=='complete':break
            if control_status['stage']=='failed':raise RuntimeError('Control failed; cannot claim full comparison')
            status('trained_waiting_for_control',gpu=gpu);time.sleep(60)
        compare(SimpleNamespace(run_dir=a.run_dir,source_run=a.control_run,native_control_run=a.native_control_run,
            source_arm_label='approved74_legacy_loss',comparison_arm_label='approved74_regression_repair',
            comparison_intentional_changes=contract['intentional_changes']),gpu)
        status('complete',gpu=gpu,comparison=str(a.run_dir/'comparison/results.json'))
    try:
        if a.adopt_training_pid is not None:
            contract=json.loads((a.run_dir/'experiment_contract.json').read_text())
            launch=json.loads((a.run_dir/'repair_launch.json').read_text())
            protocol=json.loads((a.run_dir/'protocol.json').read_text())
            pid=a.adopt_training_pid
            cmdline=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            if (str(a.run_dir).encode() not in cmdline or b'--regression-repair' not in cmdline
                    or not protocol.get('regression_repair') or a.adopt_gpu!=launch['gpu']
                    or str(a.control_run)!=contract['control_run']):
                raise ValueError('Cannot adopt an unrelated training process')
            for path,expected in contract['pinned'].items():
                if sha256(path)!=expected:raise ValueError('Adopted input identity changed')
            fingerprint=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            dump(a.run_dir/'supervisor_adoption.json',dict(training_pid=pid,gpu=a.adopt_gpu,
                model_process_not_restarted=True,reason='Resolve inherited public holdout config for final comparison'))
            status('training',gpu=a.adopt_gpu,adopted_training_pid=pid)
            while True:
                own=json.loads((a.run_dir/'status.json').read_text())
                if own['stage']=='complete':break
                if own['stage']=='failed':raise RuntimeError('Adopted training failed: '+str(own))
                proc=Path('/proc')/str(pid)/'stat'
                if not proc.exists() or proc.read_text().rsplit(')',1)[1].split()[19]!=fingerprint:
                    raise RuntimeError('Adopted training process exited before a completed status')
                time.sleep(30)
            finish(a.adopt_gpu,contract)
            return
        control=json.loads((a.control_run/'protocol.json').read_text())
        source=Path(control['data']);data=yaml.safe_load(source.read_text())
        preparation=json.loads((a.control_run/'preparation.json').read_text())
        if sha256(source)!=preparation['data_sha256'] or data.get('input_color')!='gray3':
            raise ValueError('Control data identity changed or not gray3')
        if control['training_videos']!=74 or control.get('regression_repair'):
            raise ValueError('Expected unchanged legacy 74-video control')
        cache=Path(data['online_replacement']['cache']);index=json.loads((cache/'index.json').read_text())
        foundation=Path(control['foundation'])
        pinned={str(source):sha256(source),str(cache/'index.json'):preparation['index_sha256'],
            index['catalog']:index['catalog_sha256'],str(foundation):control['foundation_sha256'],
            data['tiny_sampling']['public_manifest']:data['tiny_sampling']['public_manifest_sha256']}
        for path,expected in pinned.items():
            if sha256(path)!=expected:raise ValueError('Pinned input changed: '+str(path))
        contract=dict(control_run=str(a.control_run),same_data=str(source),pinned=pinned,
            training_videos=74,assets=len(index['asset_ids']),input_color='gray3',input_hw=[544,960],
            same_augmentations=control['augmentation'],same_sampling=control['sampling_policy'],
            same_epochs=dict(foundation=30,teacher=12,student=15,localization=0),
            same_batch=64,same_seed=20261015,old_task_weights_not_used=True,
            intentional_changes=['per_GT_per_source_scale_regression_normalization',
                'tiny_width_height_and_edge_supervision','assignment_coverage_audit','three_epoch_loss_ramp'],
            inference_graph_and_thresholds_unchanged=True,control_not_stopped_or_modified=True)
        dump(a.run_dir/'experiment_contract.json',contract)
        while True:
            usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            gpu=select_idle_gpu(usage,a.gpus)
            if gpu is not None:break
            status('waiting_for_idle_gpu',allowed_gpus=a.gpus);time.sleep(60)
        status('preflight',gpu=gpu)
        with (a.run_dir/'preflight.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/anti_uav/preflight_regression_repair.py'),
                '--data',str(source),'--foundation',str(foundation),'--output',str(a.run_dir/'preflight'),
                '--device',str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        acceptance=json.loads((a.run_dir/'preflight/acceptance.json').read_text())
        if not acceptance['complete'] or acceptance['audit']['balance_mix']!=.5:
            raise ValueError('Full-repair real-batch preflight incomplete')
        for path,expected in pinned.items():
            if sha256(path)!=expected:raise ValueError('Input changed during preflight')
        command=[sys.executable,'-u',str(ROOT/'scripts/anti_uav/run_p3_independent_curriculum_training.py'),
            '--data',str(source),'--foundation',str(foundation),'--run-dir',str(a.run_dir),'--device',str(gpu),
            '--batch','64','--workers','16','--foundation-epochs','30','--teacher-epochs','12',
            '--student-epochs','15','--localization-epochs','0','--expected-training-videos','74',
            '--replacement-probability','0.5','--regression-repair']
        dump(a.run_dir/'repair_launch.json',dict(command=command,gpu=gpu,usage_before_preflight=usage,
            preflight_weights_not_used=True,control_run=str(a.control_run)))
        status('training',gpu=gpu)
        with (a.run_dir/'training.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        finish(gpu,contract)
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
