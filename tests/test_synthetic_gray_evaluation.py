import json

import numpy as np
import pytest

from scripts.anti_uav.evaluate_synthetic_gray_pair import frame_entry, groups_for, pool_frames, serializable


def test_raw_frame_arrays_are_json_serializable_without_changing_memory_arrays():
    arrays = dict(tp=np.array([[True, False]]), conf=np.array([.5],dtype=np.float32))
    value = json.loads(json.dumps(serializable([dict(arrays=arrays)])))
    assert value == [dict(arrays=dict(tp=[[True,False]],conf=[.5]))]
    assert isinstance(arrays['tp'],np.ndarray)


def test_size_cohort_uses_original_box_not_changed_rendered_box():
    row = dict(state="replaced", width=1920, height=1080, box=[50,50,12,10],
               metrics=dict(new_box_xywh=[50,50,20,18]))
    assert groups_for(row) == ['all','replaced','replaced_original_4to8px']
    assert groups_for(dict(row,state='original_retained')) == ['all','original_retained']


def test_subset_counts_include_false_positives_without_reusing_full_denominator():
    a=frame_entry(np.empty((0,10),bool),np.array([]),np.array([]),np.array([0]),
                  {c:(0,0,1) for c in (.01,.03,.05)})
    b=frame_entry(np.zeros((1,10),bool),np.array([.8]),np.array([0]),np.array([]),
                  {c:(0,1,0) for c in (.01,.03,.05)})
    rows=[dict(entry=a,groups=['all','replaced']),dict(entry=b,groups=['all','negative_unchanged'])]
    full,_=pool_frames(rows,'all')
    subset,_=pool_frames(rows,'replaced')
    assert full['native/c0.03/FRAMES']==2 and full['native/c0.03/FP']==1
    assert subset['native/c0.03/FRAMES']==1 and subset['native/c0.03/FP']==0
    assert subset['native/c0.03/FN']==1
    with pytest.raises(ValueError):
        pool_frames(rows,'missing')
