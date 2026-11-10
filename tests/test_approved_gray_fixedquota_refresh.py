from pathlib import Path
import json
import sqlite3

import cv2
import numpy as np
import pytest

from scripts.anti_uav.run_approved_gray_fixedquota_refresh import (
    audit_background_requirement,choose_shared_gpu,public_manifest_path,select_requested,
    sha256,training_command,verify_existing_approval_scope,
)
from scripts.anti_uav.prepare_snapshot_replacement_cache import snapshot_temporal
from scripts.anti_uav.gray_temporal_background import BackgroundUnavailable


def test_only_explicit_current_approved_additions_can_be_selected():
    base=dict(train_video_hashes=['old'],test_sha256='test',validation_sha256='val')
    row=dict(source_sha256='new',status='approved',revision=2,approved_revision=2)
    assert select_requested([row,dict(row,source_sha256='unrequested')],['new'],base)==[row]
    for items,requested in [([dict(row,status='submitted')],['new']),([dict(row,revision=3)],['new']),
                            ([row],['new','missing']),([row,row],['new']),([dict(row,source_sha256='test')],['test'])]:
        with pytest.raises(ValueError):select_requested(items,requested,base)


def test_shared_gpu_selection_never_uses_disallowed_or_low_headroom_cards():
    usage='0,0,81000,0\n2,76000,5000,100\n6,22000,59000,99\n7,0,81000,0'
    assert choose_shared_gpu(usage,[2,6],True)==6
    assert choose_shared_gpu(usage,[2,6],False) is None
    assert choose_shared_gpu('6,35000,46000,0',[6],True) is None
    assert choose_shared_gpu('6,10000,20000,0',[6],True) is None


def test_frozen_ts_donors_use_exact_export_indices_without_video_seek(tmp_path):
    frame=np.full((16,24,3),89,np.uint8);assert cv2.imwrite(str(tmp_path/'000000003.jpg'),frame)
    item=dict(video='/never/open/random_seek.ts',frame_width=24,frame_height=16,frame_directory=str(tmp_path),
        source_sha256='train',status='approved')
    temporal=snapshot_temporal(item,{3:[],4:[]},['heldout'],True,'snapshot')
    record=temporal.records['random_seek']
    assert record['capture'] is None
    assert np.array_equal(temporal.read(record,3),np.full((16,24),89,np.uint8))
    with pytest.raises(BackgroundUnavailable,match='missing'):temporal.read(record,4)
    with pytest.raises(BackgroundUnavailable,match='reviewed'):temporal.read(record,2)
    temporal.close()
    with pytest.raises(ValueError,match='Heldout'):snapshot_temporal(item,{3:[]},['train'],True)


def protocol(repair=True):
    result=dict(epochs=dict(foundation=30,teacher=12,student=15,localization=0),qat=False,
                task_specific_initialization=False,input_hw=[544,960],batch=64,nbs=128)
    if repair:result['regression_repair']=dict(options=dict(balance_mix=.5,warmup_epochs=3,tiny_log_size_weight=.5))
    return result


def test_new_data_reuses_repaired_curriculum_not_legacy_loss():
    command=training_command(protocol(),'new87.yaml','public_yolov8n.pt','new87',4,87)
    assert '--regression-repair' in command
    assert float(command[command.index('--repair-balance-mix')+1])==.5
    assert command[command.index('--repair-warmup-epochs')+1]=='3'
    assert command[command.index('--foundation')+1]=='public_yolov8n.pt'
    assert command[command.index('--expected-training-videos')+1]=='87'
    assert '--resume' not in command
    assert '--regression-repair' not in training_command(protocol(False),'new.yaml','public.pt','new',1,87)
    with pytest.raises(ValueError,match='public-initialized'):
        training_command(dict(protocol(),task_specific_initialization=True),'new','old','run',1,87)


def test_asset_only_source_view_resolves_original_pinned_public_manifest(tmp_path):
    public=tmp_path/'public_accepted.json';public.write_text('[]')
    view=tmp_path/'assets361_view';view.mkdir()
    config=dict(tiny_sampling=dict(public_manifest=str(public),public_manifest_sha256=sha256(public)))
    assert public_manifest_path(view,config)==public
    public.write_text('[1]')
    with pytest.raises(ValueError,match='Pinned public'):
        public_manifest_path(view,config)


def test_recursive_approval_scope_checks_inherited_tasks_and_current_sources(tmp_path):
    original=tmp_path/'old.json';added=tmp_path/'added.json'
    a=dict(task_id='t1',revision=2,approved_revision=2,source_sha256='h1')
    b=dict(task_id='t2',revision=3,approved_revision=3,source_sha256='h2')
    original.write_text(json.dumps(dict(videos=[a])));added.write_text(json.dumps(dict(videos=[b])))
    source=dict(current_approved_scope=dict(previous_scope=dict(snapshot=str(original),sha256=sha256(original)),
        additions_snapshot=str(added),additions_snapshot_sha256=sha256(added),approved_training_videos=2))
    db=tmp_path/'tasks.sqlite'
    with sqlite3.connect(db) as con:
        con.executescript('''CREATE TABLE tasks(id TEXT,status TEXT,revision INT,approved_revision INT,video_id TEXT);
            CREATE TABLE videos(id TEXT,source_sha256 TEXT);
            INSERT INTO videos VALUES('v1','h1'),('v2','h2');
            INSERT INTO tasks VALUES('t1','approved',2,2,'v1'),('t2','approved',3,3,'v2');''')
    assert verify_existing_approval_scope(db,source)==2
    with sqlite3.connect(db) as con:con.execute("UPDATE videos SET source_sha256='changed' WHERE id='v1'")
    with pytest.raises(ValueError,match='existing approved task changed'):
        verify_existing_approval_scope(db,source)
    with sqlite3.connect(db) as con:con.execute("UPDATE videos SET source_sha256='h1' WHERE id='v1'")
    original.write_text('{}')
    with pytest.raises(ValueError,match='snapshot changed'):
        verify_existing_approval_scope(db,source)


def test_new_positive_paths_need_backgrounds_while_negatives_do_not(tmp_path):
    data=tmp_path/'data';data.mkdir();images=data/'images';labels=data/'labels'
    images.mkdir();labels.mkdir()
    (images/'000000001.jpg').write_bytes(b'new-positive');(labels/'000000001.txt').write_text('0 .5 .5 .1 .1\n')
    (images/'000000002.jpg').write_bytes(b'negative');(labels/'000000002.txt').write_text('')
    (data/'manifest.json').write_text(json.dumps(dict(latest_appended_videos=[dict(
        sha256='new-video',image_directory=str(images),label_directory=str(labels),positive_samples=1)])))
    old=dict(images={'/old/frame.jpg':dict(source_sha256='unmatched')})
    audit=audit_background_requirement(data,old,tmp_path/'audit.json')
    assert audit['need_new_backgrounds'] is True
    assert audit['new_positive_frames']==1 and audit['already_cached']==0
    assert audit['negative_frames_need_no_target_erasure'] is True
    assert audit['old_catalog_reused'] is True
