#!/usr/bin/env python3
"""Append exactly requested approved videos, extend safe cutout backgrounds and train."""

from collections import Counter
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import sqlite3

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256, snapshot_tasks


def select_requested(items, requested, baseline):
    if len(requested)!=len(set(requested)) or not requested:
        raise ValueError('Requested videos must be unique')
    blocked=set(baseline['train_video_hashes'])|{baseline['test_sha256'],baseline['validation_sha256']}
    selected=[x for x in items if x['source_sha256'] in set(requested)]
    if len(selected)!=len(requested) or {x['source_sha256'] for x in selected}!=set(requested):
        raise ValueError('A requested video is missing or duplicated')
    if set(requested)&blocked or any(x['status']!='approved' or x['revision']!=x['approved_revision'] for x in selected):
        raise ValueError('Only new currently approved, exact-revision videos may be appended')
    return selected


def choose_shared_gpu(text, allowed, allow_shared):
    candidates=[]
    for line in text.splitlines():
        index,used,free,util=map(int,line.split(','))
        if index not in allowed:continue
        if used<=64 and util==0:return index
        # All permitted workers must retain headroom for the distillation stage.
        if allow_shared and used<=32768 and free>=40000:candidates.append((used,index))
    return min(candidates)[1] if candidates else None


def verify_existing_approval_scope(database, source):
    previous=source.get('current_approved_scope',{})
    tasks={}
    def collect(scope):
        if not isinstance(scope,dict):raise ValueError('Invalid approved scope')
        if scope.get('previous_scope'):collect(scope['previous_scope'])
        for path_key,hash_key in [('snapshot','sha256'),('additions_snapshot','additions_snapshot_sha256')]:
            if not scope.get(path_key):continue
            path=Path(scope[path_key])
            if sha256(path)!=scope[hash_key]:raise ValueError('Previous approval snapshot changed')
            for item in json.loads(path.read_text())['videos']:
                if item['status']=='submitted':
                    continue
                old=tasks.get(item['task_id'])
                if old and (old['revision'],old['approved_revision'])!=(item['revision'],item['approved_revision']):
                    raise ValueError('Conflicting frozen approval revisions')
                tasks[item['task_id']]=item
    collect(previous)
    if not tasks or len(tasks)!=previous.get('approved_training_videos',len(tasks)):
        raise ValueError('Missing audited existing approval scope')
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as con:
        con.execute('BEGIN')
        for item in tasks.values():
            row=con.execute('''SELECT t.status,t.revision,t.approved_revision,v.source_sha256
                FROM tasks t JOIN videos v ON v.id=t.video_id WHERE t.id=?''',(item['task_id'],)).fetchone()
            if row!=('approved',item['revision'],item['approved_revision'],item['source_sha256']):
                raise ValueError('An existing approved task changed; do not silently reuse old labels')
        con.rollback()
    return len(tasks)


def public_manifest_path(source_dir,config):
    sampling=config.get('tiny_sampling',{})
    path=Path(sampling.get('public_manifest',source_dir/'public_accepted.json'))
    expected=sampling.get('public_manifest_sha256')
    if expected and sha256(path)!=expected:raise ValueError('Pinned public manifest changed')
    return path


def training_command(protocol,data,foundation,run,device,videos):
    expected=dict(foundation=30,teacher=12,student=15,localization=0)
    if (protocol['epochs']!=expected or protocol.get('qat') or protocol.get('task_specific_initialization')
            or protocol['input_hw']!=[544,960] or protocol['batch']!=64 or protocol['nbs']!=128):
        raise ValueError('Only the audited public-initialized 30/12/15 FP32 recipe may be refreshed')
    command=[sys.executable,'-u',str(ROOT/'scripts/anti_uav/run_p3_independent_curriculum_training.py'),
        '--data',str(data),'--foundation',str(foundation),'--run-dir',str(run),'--device',str(device),
        '--batch','64','--workers','16','--foundation-epochs','30','--teacher-epochs','12',
        '--student-epochs','15','--localization-epochs','0','--expected-training-videos',str(videos),
        '--replacement-probability','0.5']
    repair=protocol.get('regression_repair')
    if repair:
        options=repair['options']
        if options['tiny_log_size_weight']!=.5:raise ValueError('Unsupported source repair size weight')
        command.extend(['--regression-repair','--repair-balance-mix',str(options['balance_mix']),
                        '--repair-warmup-epochs',str(options['warmup_epochs'])])
    return command


def audit_background_requirement(data_dir,previous_index,output):
    manifest=json.loads((data_dir/'manifest.json').read_text())
    old_hashes={row['source_sha256']:image for image,row in previous_index['images'].items()}
    rows=[]
    for video in manifest['latest_appended_videos']:
        count=covered=duplicates=0;examples=[]
        for label in Path(video['label_directory']).glob('*.txt'):
            if not label.stat().st_size:continue
            image=Path(video['image_directory'])/(label.stem+'.jpg');count+=1
            covered+=str(image) in previous_index['images']
            digest=sha256(image)
            if digest in old_hashes:
                duplicates+=1
                if len(examples)<3:examples.append(dict(new_image=str(image),old_image=old_hashes[digest]))
        if count!=video['positive_samples']:raise ValueError('New positive export coverage changed')
        rows.append(dict(video_sha256=video['sha256'],positive_frames=count,already_cached=covered,
                         exact_image_matches_in_old_ready_cache=duplicates,match_examples=examples))
    result=dict(new_positive_frames=sum(r['positive_frames'] for r in rows),
        already_cached=sum(r['already_cached'] for r in rows),negative_frames_need_no_target_erasure=True,
        old_ready_backgrounds=len(previous_index['images']),old_catalog_reused=True,
        need_new_backgrounds=any(r['already_cached']<r['positive_frames'] for r in rows),videos=rows,
        duplicate_check_scope='Exact frozen JPEG bytes against the old ready background index; not a near-duplicate guarantee')
    dump(output,result)
    if any(r['positive_frames']>=100 and r['exact_image_matches_in_old_ready_cache']>=.95*r['positive_frames'] for r in rows):
        raise ValueError('A new source repeats old positive pixels; reconcile it instead of silently doubling exposure')
    return result


def configure(data_dir, cache_dir, source_config, source_dir, output, visible_box_guard=None):
    import numpy as np
    import yaml
    from scripts.anti_uav.audit_labeler_expansion_regression import AuditDataset
    from scripts.anti_uav.label_pool_sampling import FixedDomainQuotaSampler
    config=yaml.safe_load((data_dir/'train_online_gray_monitor.yaml').read_text())
    public=public_manifest_path(source_dir,source_config)
    public_rows=json.loads(public.read_text());public_paths={r['image'] for r in public_rows}
    source_publics={p for p in Path(config['train']).read_text().splitlines() if p in public_paths}
    if source_publics!=public_paths:raise ValueError('Original public positives were lost')
    scope=output/'native_source_scope.txt'
    scope.write_text('\n'.join(p for p in Path(config['train']).read_text().splitlines() if p not in public_paths)+'\n')
    config['input_color']='gray3'
    if visible_box_guard:
        config['visible_box_guard']=visible_box_guard
    config['online_replacement']=dict(config['online_replacement'],cache=str(cache_dir),replacement_probability=.5)
    base_tiny=dict(source_config['tiny_sampling'])
    reference=json.loads((data_dir/'manifest.json').read_text())['final_entries']
    if base_tiny.get('policy')!='fixed_domain_v1':
        raise ValueError('Refresh requires the existing fixed-domain sampler')
    base_tiny.update(reference_epoch_slots=reference,
        public_manifest=str(public),public_manifest_sha256=sha256(public),
        native_positive_source_list=str(scope),native_positive_source_sha256=sha256(scope))
    config['tiny_sampling']=base_tiny
    labels=np.load(Path(config['train']).with_suffix('.cache'),allow_pickle=True).item()['labels']
    sampler=FixedDomainQuotaSampler(AuditDataset(labels),config['label_sampling'],base_tiny,64,20261015)
    seen=set();reports=[]
    native=[i for k in sampler.GROUPS[:4] for i in sampler.groups[k]]
    for epoch in range(sampler.audit['public_full_coverage_epochs']):
        sampler.set_epoch(epoch);order=list(sampler);counts=Counter(order)
        if any(not 1<=counts[i]<=2 for i in native):raise ValueError('Native exposure floor or replay limit violated')
        for start in range(0,len(order),64):
            batch=order[start:start+64]
            if Counter(sampler.group_by_index[i] for i in batch)!=base_tiny['batch_quotas']:raise ValueError('Batch quota changed')
            if Counter(sampler.public_dataset_by_index[i] for i in batch if sampler.group_by_index[i]=='public')!=base_tiny['public_dataset_quotas']:raise ValueError('Public dataset quota changed')
        selected=[i for i in order if sampler.group_by_index[i]=='public'];seen.update(selected)
        if max(Counter(selected).values())>2:raise ValueError('Public replay bound exceeded')
        reports.append(dict(epoch=epoch,slots=len(order),group_counts=dict(Counter(sampler.group_by_index[i] for i in order)),
            public_counts=dict(Counter(sampler.public_dataset_by_index[i] for i in selected)),public_covered=len(seen)))
    if seen!=set(sampler.groups['public']):raise ValueError('Public rotation did not cover every image')
    config['label_sampling'].update(negatives_per_epoch=sampler.rotating_budget,target_negative_fraction=sampler.audit['negative_fraction'])
    result=data_dir/'train_fixed_gray.yaml';result.write_text(yaml.safe_dump(config,sort_keys=False))
    dump(output/'sampler_acceptance.json',dict(complete=True,sampler=sampler.audit,epochs=reports,
        public_coverage_complete=True,all_native_positive_exposures_preserved=True,input_color='gray3',
        hsv=dict(h=0,s=0,v=.3),data=str(result),data_sha256=sha256(result)))
    manifest=json.loads((data_dir/'manifest.json').read_text())
    previous_scope=manifest.get('current_approved_scope',{})
    latest_scope=json.loads((data_dir/'annotation_snapshot.json').read_text())
    manifest.update(fixed_sampling=sampler.audit,epoch_positive=len(sampler)-sampler.negative_budget,
        epoch_negative=sampler.negative_budget,final_entries=len(sampler),
        additional_negatives_per_epoch=sampler.rotating_budget,final_negative_fraction=sampler.audit['negative_fraction'])
    manifest['current_approved_scope']=dict(previous_scope=previous_scope,
        additions_snapshot=str(data_dir/'annotation_snapshot.json'),additions_snapshot_sha256=sha256(data_dir/'annotation_snapshot.json'),
        captured_at=latest_scope['captured_at'],approved_training_videos=previous_scope['approved_training_videos']+
            sum(x['status']=='approved' for x in latest_scope['videos']),
        historical_sources=previous_scope['historical_sources'])
    manifest['provisional_submitted_training_videos']=manifest.get('provisional_submitted_training_videos',0)+sum(
        x['status']=='submitted' for x in latest_scope['videos'])
    manifest.update(background_cache=str(cache_dir),online_replacement=config['online_replacement'],
                    augmentation_catalog=json.loads((cache_dir/'index.json').read_text())['catalog'])
    dump(data_dir/'manifest.json',manifest)
    state=json.loads((data_dir/'status.json').read_text());cache_state=json.loads((cache_dir/'summary.json').read_text())
    state.update(cache_ready=True,new_ready_backgrounds=cache_state['new_ready_backgrounds'],
                 new_original_only=cache_state['new_original_only'],asset_count=cache_state['asset_count'],
                 actual_sampling_policy='fixed_domain_v1',epoch_slots=len(sampler),
                 negative_fraction=sampler.audit['negative_fraction'],
                 visible_box_guard=config.get('visible_box_guard'),new_old_sampling_weights=False,
                 replacement_policy='p=0.5 on safe cached positives; unsafe positives retained unchanged')
    dump(data_dir/'status.json',state)
    return result


def fixedwing_eval_config(source_yaml):
    import yaml
    direct=source_yaml.parent/'fixedwing_public_holdout.yaml'
    if direct.is_file():return direct
    config=yaml.safe_load(source_yaml.read_text())
    manifest=Path(config['tiny_sampling']['public_manifest'])
    target=manifest.parent/'fixedwing_public_holdout.yaml'
    if not target.is_file():raise FileNotFoundError(target)
    return target


def video00009_acceptance(result, baseline, candidate):
    directions = {"native/c0.03/R": 1, "native/c0.03/P": 1,
                  "native/c0.03/long_lt4px/R": 1, "native/c0.03/long_4to8px/R": 1,
                  "native/mAP50": 1, "native/mAP50-95": 1, "native/c0.03/FP": -1}
    stages = {}
    for stage in ('p3_initial','teacher','student'):
        a,b=(result[name+'_'+stage]['splits']['Video00009'] for name in (baseline,candidate))
        metrics={}
        for key,direction in directions.items():
            before,after=float(a[key]),float(b[key])
            if not math.isfinite(before) or not math.isfinite(after):
                raise ValueError('Missing finite Video00009 acceptance metric')
            metrics[key]=dict(baseline=before,candidate=after,delta=after-before,
                              passed=direction*(after-before)>=-1e-12)
        stages[stage]=dict(passed=all(row['passed'] for row in metrics.values()),metrics=metrics)
    return dict(passed=stages['student']['passed'],release_stage='student',stages=stages,
                baseline=baseline,candidate=candidate,split='Video00009',checkpoint_selection_validation=True,
                automatic_deployment=False)


def comparison_arms(a, baseline_label, new_label):
    reference=getattr(a,'acceptance_baseline_run',None)
    arms=[(baseline_label,reference)] if reference else [('native69',a.native_control_run)]
    source_label=getattr(a,'source_arm_label','public69_before_repair')
    skipped={}
    status=a.source_run/'refresh_status.json'
    if reference and status.is_file() and json.loads(status.read_text()).get('stage')=='stopped_by_user':
        skipped[source_label]=dict(run=str(a.source_run),reason='stopped_by_user')
    else:
        arms.append((source_label,a.source_run))
    arms.append((new_label,a.run_dir))
    if len({label for label,_ in arms})!=len(arms):
        raise ValueError('Duplicate comparison arm labels')
    return arms,skipped


def compare(a, device):
    import numpy as np
    import torch
    import yaml
    from ultralytics import YOLO
    from scripts.anti_uav.evaluate_p3_gray_pair import PairValidator, pool_native
    from scripts.anti_uav.run_native_pool_comparison import HOLDOUT, clean
    output=a.run_dir/'comparison';output.mkdir()
    native_data=Path(json.loads((a.native_control_run/'protocol.json').read_text())['data'])
    source_data=Path(json.loads((a.source_run/'protocol.json').read_text())['data'])
    sources=dict(Video00004=HOLDOUT,Video00009=native_data,fixedwing=fixedwing_eval_config(source_data))
    configs={}
    for key,path in sources.items():
        config=yaml.safe_load(path.read_text());config['input_color']='gray3'
        target=output/(key+'_gray3.yaml');target.write_text(yaml.safe_dump(config,sort_keys=False));configs[key]=target
    expected=dict(Video00004=(2359,448),Video00009=(1421,797),fixedwing=(3451,3451))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={}
    new_label=getattr(a,'comparison_arm_label','approved74_fixedgray')
    reference=getattr(a,'acceptance_baseline_run',None)
    baseline_label='approved74_regression_repair'
    arms,skipped=comparison_arms(a,baseline_label,new_label)
    for arm,run in arms:
        for stage,relative in [('p3_initial','foundation/p3/weights/best.pt'),('teacher','teacher/p2p3/weights/best.pt'),
                               ('student','p3_student_distilled_best_fp32.pt')]:
            name=arm+'_'+stage;weights=run/relative;digest=sha256(weights)
            result[name]=dict(weights=str(weights),sha256=digest,splits={});entries=[]
            for split,data in configs.items():
                dump(a.run_dir/'refresh_status.json',dict(stage='comparing',model=name,split=split))
                model=YOLO(str(weights))
                metrics=model.val(data=str(data),validator=PairValidator,imgsz=[544,960],rect=False,
                    device=str(device),batch=32,workers=4,conf=.001,iou=.45,max_det=100,
                    half=False,plots=False,verbose=False,project=str(output/'val'),name=name+'_'+split)
                native={k:v for k,v in clean(metrics.gray_selection).items() if k.startswith('native/')}
                if (native['native/c0.03/FRAMES'],native['native/c0.03/TP']+native['native/c0.03/FN'])!=expected[split]:
                    raise ValueError('Comparison frame/GT coverage changed')
                result[name]['splits'][split]=native
                if split!='fixedwing':entries.append(dict(metrics=native,arrays=metrics.native_pr_arrays,scales=metrics.native_scale_counts))
                np.savez_compressed(output/(name+'_'+split+'.npz'),**metrics.native_pr_arrays)
                del model;torch.cuda.empty_cache()
            result[name]['splits']['pooled']=pool_native(entries)[0]
            if sha256(weights)!=digest:raise ValueError('Checkpoint changed during comparison')
            dump(output/'results.json',result)
    dump(output/'protocol.json',dict(input_color='gray3',input_hw=[544,960],precision='FP32',tracker=False,
        conf=.03,ap_floor=.001,nms_iou=.45,match_iou=.5,pooled_frames=3780,pooled_gt=1245,
        pooled_contains_selection_validation=True,external_fixedwing_positive_frames_only=True,skipped_arms=skipped,
        intentional_changes=getattr(a,'comparison_intentional_changes',
            ['five_new_approved_videos','fixed_source_size_quotas','gray3_input','brightness_only','new_background_caches'])))
    if reference:
        result=video00009_acceptance(result,baseline_label,new_label)
        dump(output/'video00009_acceptance.json',result)
        return result['passed']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--native-control-run',type=Path,required=True)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--video-root',type=Path,required=True)
    p.add_argument('--previous-cache',type=Path,required=True)
    p.add_argument('--data-output',type=Path,required=True)
    p.add_argument('--cache-output',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    scope=p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--approved-sha256',nargs='+')
    p.add_argument('--gpus',nargs='+',type=int,default=[2,6])
    p.add_argument('--allow-shared-gpu',action='store_true')
    p.add_argument('--background-workers',type=int,default=16)
    p.add_argument('--extraction-workers',type=int,default=5)
    p.add_argument('--expected-training-videos',type=int)
    p.add_argument('--source-arm-label')
    p.add_argument('--comparison-arm-label')
    p.add_argument('--visible-box-guard',choices=['affine_fallback_v1','affine_keep_subpixel_v2'])
    p.add_argument('--approved-frame-assets',type=Path)
    p.add_argument('--bounded-disk',action='store_true')
    scope.add_argument('--all-approved-and-submitted',action='store_true',
                       help='Prepare all additional approved videos and submitted positive frames for a separate experiment')
    p.add_argument('--acceptance-baseline-run',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--train-prepared',action='store_true')
    p.add_argument('--resume-training',action='store_true',
                   help='Resume the interrupted curriculum using unchanged prepared data and optimizer states')
    p.add_argument('--resume-preparation',action='store_true',
                   help='Reuse completed frozen extraction/backgrounds, retry only QA/configuration')
    a=p.parse_args()
    if a.resume_training and (not a.train_prepared or a.prepare_only or a.resume_preparation):
        p.error('--resume-training requires --train-prepared only')
    if a.bounded_disk and not a.approved_frame_assets:
        p.error('--bounded-disk requires --approved-frame-assets')
    if a.all_approved_and_submitted and (not a.prepare_only or not a.approved_frame_assets):
        p.error('--all-approved-and-submitted requires --prepare-only and --approved-frame-assets')
    if not 1<=a.background_workers<=32 or not 1<=a.extraction_workers<=16:
        p.error('Use 1..32 background and 1..16 extraction workers')
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    a.run_dir.mkdir(parents=True,exist_ok=a.train_prepared or a.resume_preparation)
    lock=(a.run_dir/'refresh.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (a.run_dir/'logs').mkdir(exist_ok=True);(a.run_dir/'audit').mkdir(exist_ok=True)
    def status(stage,**more):
        dump(a.run_dir/'refresh_status.json',dict(stage=stage,pid=os.getpid(),
            time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),**more))
    def command(name,cmd):
        status(name,command=[str(x) for x in cmd])
        with (a.run_dir/'logs'/(name+'.log')).open('x') as f:
            subprocess.run([str(x) for x in cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    protocol=json.loads((a.source_run/'protocol.json').read_text())
    import yaml
    source_yaml=Path(protocol['data']);source_dir=source_yaml.parent
    source=json.loads((source_dir/'manifest.json').read_text());source_config=yaml.safe_load(source_yaml.read_text())
    blocked=set(source['train_video_hashes'])|{source['test_sha256'],source['validation_sha256']}
    mixed_items=None
    if a.all_approved_and_submitted:
        mixed_items=snapshot_tasks(a.database,a.video_root,blocked,submitted_positives_only=True)
        if not mixed_items:
            raise ValueError('No additional approved/submitted videos')
        a.approved_sha256=[x['source_sha256'] for x in mixed_items]
    expected_videos=len(source['train_video_hashes'])+len(a.approved_sha256)
    if a.expected_training_videos is not None and a.expected_training_videos!=expected_videos:
        raise ValueError('Requested final video count differs from the frozen addition scope')
    a.source_arm_label=a.source_arm_label or f'approved{len(source["train_video_hashes"])}_source'
    a.comparison_arm_label=a.comparison_arm_label or f'approved{expected_videos}_refreshed'
    additions_scope='approved_and_submitted' if a.all_approved_and_submitted else 'approved'
    a.comparison_intentional_changes=[f'{len(a.approved_sha256)}_new_{additions_scope}_videos',
        'training_only_background_extension','same_gray3_and_per_batch_quotas',
        'same_frozen_cutout_catalog','same_source_regression_loss_and_curriculum']
    if a.visible_box_guard:
        a.comparison_intentional_changes.append('same_visible_box_guard_for_all_sources')
    reference = str(a.acceptance_baseline_run) if a.acceptance_baseline_run else None
    previous_index=json.loads((a.previous_cache/'index.json').read_text())
    if Path(source_config['online_replacement']['cache']).resolve()!=a.previous_cache:
        raise ValueError('Background cache must match the source training configuration')
    requested=a.run_dir/'audit/requested_scope.json'
    try:
        verify_existing_approval_scope(a.database,source)
        if not a.train_prepared and not a.resume_preparation:
            dump(a.run_dir/'refresh_contract.json',dict(source_run=str(a.source_run),
                source_protocol_sha256=sha256(a.source_run/'protocol.json'),
                source_data_sha256=sha256(source_yaml),source_manifest_sha256=sha256(source_dir/'manifest.json'),
                previous_cache=str(a.previous_cache),previous_index_sha256=sha256(a.previous_cache/'index.json'),
                catalog=previous_index['catalog'],catalog_sha256=previous_index['catalog_sha256'],
                asset_count=len(previous_index['asset_ids']),requested_sha256=sorted(a.approved_sha256),
                expected_training_videos=expected_videos,epochs=protocol['epochs'],qat=False,
                input_color=source_config.get('input_color','source'),batch_quotas=source_config['tiny_sampling']['batch_quotas'],
                regression_repair=protocol.get('regression_repair'),old_task_weights_not_used=True,
                visible_box_guard=a.visible_box_guard,acceptance_baseline_run=reference,
                holdouts_unchanged=True,intentional_changes=a.comparison_intentional_changes))
        contract=json.loads((a.run_dir/'refresh_contract.json').read_text())
        for path,key in [(a.source_run/'protocol.json','source_protocol_sha256'),
                         (source_yaml,'source_data_sha256'),(source_dir/'manifest.json','source_manifest_sha256'),
                         (a.previous_cache/'index.json','previous_index_sha256')]:
            if sha256(path)!=contract[key]:raise ValueError('Frozen source changed during extension')
        if contract['requested_sha256']!=sorted(a.approved_sha256):raise ValueError('Requested addition scope changed')
        if contract.get('visible_box_guard')!=a.visible_box_guard or contract.get('acceptance_baseline_run')!=reference:
            raise ValueError('Prepared guard or acceptance baseline changed')
        if not a.train_prepared:
            status('freezing_approved_scope')
            items=mixed_items if mixed_items is not None else select_requested(
                snapshot_tasks(a.database,a.video_root,blocked),a.approved_sha256,source)
            if a.resume_preparation:
                from scripts.anti_uav.run_approved_data_only_refresh import verify_frozen_scope
                frozen=json.loads(requested.read_text())
                verify_frozen_scope(frozen['videos'],items)
                if frozen['source_manifest_sha256']!=sha256(source_dir/'manifest.json'):raise ValueError('Source changed since frozen extraction')
                if (a.run_dir/'preparation.json').exists():raise ValueError('Completed preparation must use --train-prepared')
            else:
                dump(requested,dict(captured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),videos=items,
                    source_manifest_sha256=sha256(source_dir/'manifest.json'),source_database=str(a.database)))
                extract_cmd=[sys.executable,'-u',ROOT/'scripts/anti_uav/prepare_labeler_training_expansion.py',
                    '--source',source_dir,'--source-yaml',source_yaml.name,'--baseline-protocol',a.source_run/'protocol.json',
                    '--database',a.database,'--video-root',a.video_root,'--output',a.data_output,
                    '--workers',a.extraction_workers,'--task-snapshot',requested]
                if a.approved_frame_assets:extract_cmd.extend(['--approved-frame-assets',a.approved_frame_assets])
                if a.bounded_disk:extract_cmd.append('--bounded-disk')
                if a.all_approved_and_submitted:extract_cmd.append('--include-submitted-positives')
                command('extract_approved',extract_cmd)
            data_manifest=json.loads((a.data_output/'manifest.json').read_text())
            if set(data_manifest['train_video_hashes'])!=set(source['train_video_hashes'])|set(a.approved_sha256):raise ValueError('Appended video scope differs from request')
            audit_background_requirement(a.data_output,previous_index,a.run_dir/'audit/background_requirement.json')
            if not a.resume_preparation:
                command('prepare_backgrounds',[sys.executable,'-u',ROOT/'scripts/anti_uav/prepare_snapshot_replacement_cache.py',
                    '--data-dir',a.data_output,'--previous-cache',a.previous_cache,'--output',a.cache_output,
                    '--workers',a.background_workers,'--chunk-size','128','--donors-from-frozen-frames'])
            summary=json.loads((a.cache_output/'summary.json').read_text())
            new_index=json.loads((a.cache_output/'index.json').read_text())
            if (summary['stage']!='complete' or summary['asset_count']!=len(previous_index['asset_ids'])
                    or new_index['asset_ids']!=previous_index['asset_ids']
                    or new_index['catalog_sha256']!=previous_index['catalog_sha256']):
                raise ValueError('New cache incomplete or frozen cutouts changed')
            from scripts.anti_uav.verify_snapshot_replacement_cache import verify_cache
            if summary['new_ready_backgrounds']:
                qa_path=a.run_dir/('replacement_QA_retry' if a.resume_preparation else 'replacement_QA')
                qa=verify_cache(a.cache_output,a.data_output,qa_path)
            else:
                qa=dict(stage='no_safe_replacements',rendered=0,originals_retained=True)
            dump(a.run_dir/'audit/composition_acceptance.json',qa)
            data=configure(a.data_output,a.cache_output,source_config,source_dir,a.run_dir/'audit',a.visible_box_guard)
            dump(a.run_dir/'preparation.json',dict(complete=True,data=str(data),data_sha256=sha256(data),
                requested_sha256=sorted(a.approved_sha256),training_videos=len(data_manifest['train_video_hashes']),
                foundation=protocol['foundation'],foundation_sha256=protocol['foundation_sha256'],
                backgrounds=summary,visual_review_pending=bool(summary['new_ready_backgrounds'])))
            status('prepared_awaiting_visual_review',training_videos=len(data_manifest['train_video_hashes']),cache_summary=summary)
            if a.prepare_only:return
        prepared=json.loads((a.run_dir/'preparation.json').read_text())
        if set(prepared['requested_sha256'])!=set(a.approved_sha256) or sha256(prepared['data'])!=prepared['data_sha256']:raise ValueError('Prepared data differs from requested run')
        cache_summary=json.loads((a.cache_output/'summary.json').read_text())
        if (cache_summary.get('stage')!='complete'
                or cache_summary['index_sha256']!=prepared['backgrounds']['index_sha256']
                or sha256(a.cache_output/'index.json')!=cache_summary['index_sha256']):
            raise ValueError('Replacement cache changed after preparation')
        if prepared['visual_review_pending']:
            review=json.loads((a.run_dir/'audit/visual_review.json').read_text())
            if (review.get('accepted') is not True
                    or review.get('composition_acceptance_sha256')!=sha256(a.run_dir/'audit/composition_acceptance.json')):
                raise ValueError('New replacement previews require matching visual acceptance')
        from scripts.anti_uav.run_approved_data_only_refresh import verify_frozen_scope
        frozen=json.loads(requested.read_text())['videos']
        current=select_requested(snapshot_tasks(a.database,a.video_root,blocked),a.approved_sha256,source)
        verify_frozen_scope(frozen,current)
        if sha256(protocol['foundation'])!=protocol['foundation_sha256']:raise ValueError('Public foundation checkpoint changed')
        while True:
            usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            gpu=choose_shared_gpu(usage,a.gpus,a.allow_shared_gpu)
            if gpu is not None:break
            status('waiting_for_allowed_gpu',allowed_gpus=a.gpus,usage=usage);time.sleep(60)
        if protocol.get('regression_repair'):
            if not (a.resume_training and (a.run_dir/'preflight/acceptance.json').is_file()):
                command('regression_preflight',[sys.executable,'-u',ROOT/'scripts/anti_uav/preflight_regression_repair.py',
                    '--data',prepared['data'],'--foundation',protocol['foundation'],
                    '--output',a.run_dir/'preflight','--device',gpu])
            acceptance=json.loads((a.run_dir/'preflight/acceptance.json').read_text())
            if not acceptance['complete'] or acceptance['audit']['balance_mix']!=protocol['regression_repair']['options']['balance_mix']:
                raise ValueError('New-data full-repair preflight failed')
            while True:
                usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'],text=True)
                gpu=choose_shared_gpu(usage,a.gpus,a.allow_shared_gpu)
                if gpu is not None:break
                status('waiting_for_allowed_gpu_after_preflight',allowed_gpus=a.gpus);time.sleep(60)
        cmd=training_command(protocol,prepared['data'],protocol['foundation'],a.run_dir,gpu,prepared['training_videos'])
        if a.resume_training:
            if not (a.run_dir/'protocol.json').is_file() or not (a.run_dir/'training.log').is_file():
                raise ValueError('No interrupted curriculum exists to resume')
            cmd.append('--resume')
            stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
            dump(a.run_dir/'audit'/('recovery_before_'+stamp+'.json'),dict(
                previous_launch=json.loads((a.run_dir/'launch.json').read_text()),
                previous_status=json.loads((a.run_dir/'status.json').read_text()),
                previous_log_sha256=sha256(a.run_dir/'training.log'),new_command=cmd,gpu=gpu,
                prepared_data_sha256=prepared['data_sha256'],prepared_background_index_sha256=cache_summary['index_sha256']))
        dump(a.run_dir/'launch.json',dict(command=cmd,gpu=gpu,usage_before_launch=usage,allow_shared_gpu=a.allow_shared_gpu,
            shared_memory_limit_mib=32768,minimum_free_mib=40000,previous_run_unchanged=str(a.source_run),
            resuming=a.resume_training))
        manifest=json.loads((a.data_output/'manifest.json').read_text());manifest['training_started']=True
        dump(a.data_output/'manifest.json',manifest)
        status('training',gpu=gpu,training_videos=prepared['training_videos'],resuming=a.resume_training)
        with (a.run_dir/'training.log').open('a' if a.resume_training else 'x') as f:
            if a.resume_training:
                f.write('\nRESUMING ON GPU '+str(gpu)+' AT '+stamp+'; previous failure log retained\n');f.flush()
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        passed=compare(a,gpu)
        status('complete_below_video00009_baseline' if passed is False else 'complete',
               student=str(a.run_dir/'p3_student_distilled_best_fp32.pt'),video00009_acceptance=passed)
    except BaseException as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
