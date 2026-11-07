#!/usr/bin/env python3
"""Restart a frozen curriculum with reviewed extra assets, changing no training slots."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.anti_uav.prepare_labeler_training_expansion import dump, sha256, snapshot_tasks
from scripts.anti_uav.prepare_gray_asset_ablation import assert_asset_only_configs
from scripts.anti_uav.run_approved_gray_fixedquota_refresh import (
    choose_shared_gpu, compare, select_requested, verify_existing_approval_scope,
)
from scripts.anti_uav.run_approved_data_only_refresh import verify_frozen_scope


def validate_extension(old_rows, new_rows, enabled_ids):
    before = {r['id']: r for r in old_rows}
    after = {r['id']: r for r in new_rows}
    if len(before) != len(old_rows) or len(after) != len(new_rows):
        raise ValueError('Duplicate asset ID')
    if not set(before) <= set(after) or not set(enabled_ids) <= set(before):
        raise ValueError('Original assets removed')
    for key, row in before.items():
        if row['cutout_sha256'] != after[key]['cutout_sha256']:
            raise ValueError('Original cutout changed')
    added = [r for r in new_rows if r['id'] not in before]
    if not added or any(r.get('status') != 'visually_screened_compositing_candidate' for r in added):
        raise ValueError('Only visually screened new cutouts may be added')
    return added


def verify_scope(args):
    import yaml
    source = json.loads((args.public_source_run / 'protocol.json').read_text())
    manifest = json.loads((Path(source['data']).parent / 'manifest.json').read_text())
    verify_existing_approval_scope(args.database, manifest)
    frozen = json.loads((args.source_run / 'audit/requested_scope.json').read_text())['videos']
    blocked = set(manifest['train_video_hashes']) | {manifest['test_sha256'], manifest['validation_sha256']}
    current = select_requested(snapshot_tasks(args.database, args.video_root, blocked),
                               [r['source_sha256'] for r in frozen], manifest)
    verify_frozen_scope(frozen, current)
    protocol = json.loads((args.source_run / 'protocol.json').read_text())
    config = yaml.safe_load(Path(protocol['data']).read_text())
    return protocol, config


def prepare(args):
    import yaml
    from scripts.anti_uav.build_gray_replacement_batch import load_assets
    protocol, source = verify_scope(args)
    root = Path(source['online_replacement']['cache'])
    summary = json.loads((root / 'summary.json').read_text())
    if summary['stage'] != 'complete' or summary['is_smoke_subset']:
        raise ValueError('Completed full background cache required')
    if sha256(root / 'index.json') != summary['index_sha256']:
        raise ValueError('Original cache index changed')
    index = json.loads((root / 'index.json').read_text())
    old_catalog = Path(index['catalog'])
    if sha256(old_catalog) != index['catalog_sha256']:
        raise ValueError('Original catalog changed')
    new = json.loads(args.catalog.read_text())
    old = json.loads(old_catalog.read_text())
    added = validate_extension(old['records'], new['records'], index['asset_ids'])
    if any(r.get('video_source', {}).get('source_video_sha256') in index['heldout_sha256'] for r in added):
        raise ValueError('Heldout video used for new assets')
    assets = load_assets(args.catalog, {'24', '25'})
    enabled = [r['id'] for r, _ in assets]
    expected = set(index['asset_ids']) | {r['id'] for r in added}
    if set(enabled) != expected:
        raise ValueError('Asset extension unexpectedly enabled disabled IDs')
    if any(r['video_sha256'] in index['heldout_sha256'] for r in index['images'].values()):
        raise ValueError('Heldout frame in background cache')
    args.cache_output.mkdir(parents=True)
    for pack in {r['pack'] for r in index['images'].values()}:
        if Path(pack).name != pack or not (root / pack).is_file():
            raise ValueError('Invalid background pack')
        (args.cache_output / pack).symlink_to((root / pack).resolve())
    index.update(catalog=str(args.catalog), catalog_sha256=sha256(args.catalog), asset_ids=enabled)
    dump(args.cache_output / 'index.json', index)
    summary.update(asset_count=len(enabled), index_sha256=sha256(args.cache_output / 'index.json'),
                   source_cache=str(root), backgrounds_unchanged=True,
                   added_asset_ids=[r['id'] for r in added], manual_visual_review_pending=True)
    dump(args.cache_output / 'summary.json', summary)
    args.data_output.mkdir(parents=True)
    config = deepcopy(source)
    config['online_replacement']['cache'] = str(args.cache_output)
    assert_asset_only_configs(source, config)
    data = args.data_output / 'train_fixed_gray.yaml'
    data.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = json.loads((Path(protocol['data']).parent / 'manifest.json').read_text())
    manifest.update(asset_refresh_from=str(args.source_run), background_cache=str(args.cache_output),
                    augmentation_catalog=str(args.catalog), training_started=False)
    dump(args.data_output / 'manifest.json', manifest)
    shutil.copy2(args.source_run / 'audit/requested_scope.json', args.run_dir / 'audit/requested_scope.json')
    record = dict(complete=True, data=str(data), data_sha256=sha256(data),
        source_run=str(args.source_run), source_yaml_sha256=sha256(protocol['data']),
        catalog=str(args.catalog), catalog_sha256=sha256(args.catalog),
        cache=str(args.cache_output), index_sha256=summary['index_sha256'],
        asset_count=len(enabled), added_asset_ids=[r['id'] for r in added],
        backgrounds=len(index['images']), backgrounds_unchanged=True,
        foundation=protocol['foundation'], foundation_sha256=protocol['foundation_sha256'],
        training_videos=len(manifest['train_video_hashes']), input_color=config['input_color'],
        training_slots_and_sampling_unchanged=True, old_weights_not_used=True)
    dump(args.run_dir / 'preparation.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source-run', 'public-source-run', 'native-control-run', 'catalog', 'data-output',
                'cache-output', 'run-dir', 'database', 'video-root']:
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--train-prepared', action='store_true')
    p.add_argument('--gpus', nargs='+', type=int, default=[2, 6])
    p.add_argument('--allow-shared-gpu', action='store_true')
    a = p.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve())
    a.run_dir.mkdir(parents=True, exist_ok=a.train_prepared)
    (a.run_dir / 'audit').mkdir(exist_ok=True)
    lock = (a.run_dir / 'asset_refresh.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(stage, **extra):
        dump(a.run_dir / 'refresh_status.json', dict(stage=stage, pid=os.getpid(),
            time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **extra))
    try:
        if not a.train_prepared:
            status('binding_reviewed_assets')
            record = prepare(a)
            status('prepared_awaiting_composition_review', **record)
            if a.prepare_only:
                return
        record = json.loads((a.run_dir / 'preparation.json').read_text())
        verify_scope(a)
        for path, expected in [(record['data'], record['data_sha256']),
                               (record['catalog'], record['catalog_sha256']),
                               (a.cache_output / 'index.json', record['index_sha256']),
                               (record['foundation'], record['foundation_sha256'])]:
            if sha256(path) != expected:
                raise ValueError('Prepared training identity changed')
        review = json.loads((a.run_dir / 'audit/composition_review.json').read_text())
        qa = a.run_dir / 'audit/composition_QA.json'
        if (review.get('accepted') is not True or review.get('catalog_sha256') != record['catalog_sha256']
                or review.get('composition_qa_sha256') != sha256(qa)):
            raise ValueError('Matching cutout/composition visual acceptance required')
        while True:
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True)
            gpu = choose_shared_gpu(usage, a.gpus, a.allow_shared_gpu)
            if gpu is not None:
                break
            status('waiting_for_allowed_gpu', allowed_gpus=a.gpus)
            time.sleep(60)
        command = [sys.executable, '-u', str(ROOT / 'scripts/anti_uav/run_p3_independent_curriculum_training.py'),
            '--data', record['data'], '--foundation', record['foundation'], '--run-dir', str(a.run_dir),
            '--device', str(gpu), '--batch', '64', '--workers', '16', '--foundation-epochs', '30',
            '--teacher-epochs', '12', '--student-epochs', '15', '--localization-epochs', '0',
            '--expected-training-videos', str(record['training_videos']), '--replacement-probability', '0.5']
        dump(a.run_dir / 'launch.json', dict(command=command, gpu=gpu, usage_before_launch=usage,
            restarted_from_public_foundation=True, stopped_source_run=str(a.source_run)))
        status('training', gpu=gpu, training_videos=record['training_videos'], asset_count=record['asset_count'])
        with (a.run_dir / 'training.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        a.source_run = a.public_source_run
        a.comparison_arm_label = 'approved74_fixedgray_target_assets'
        a.comparison_intentional_changes = ['five_new_approved_videos', 'fixed_source_size_quotas',
            'gray3_input', 'brightness_only', 'new_background_caches', 'reviewed_colleague_target_cutouts']
        compare(a, gpu)
        status('complete', student=str(a.run_dir / 'p3_student_distilled_best_fp32.pt'))
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
