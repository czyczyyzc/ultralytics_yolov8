from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.anti_uav.run_approved_gray_fixedquota_refresh import select_requested,choose_shared_gpu
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
