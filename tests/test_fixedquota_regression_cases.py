import pytest

from scripts.anti_uav.diagnose_fixedquota_regression import compare_cases, edge_bin


def test_scale_boundaries_match_monitor():
    assert [edge_bin(x) for x in [3.99,4.,8.,8.01,16.,16.01]] == ['lt4','4to8','4to8','8to16','8to16','gt16']


def row(hit, iou=.6, score=.1):
    return dict(image='frame.png',gt=[[10,10,20,20]],gt_classes=[0],gt_long_edge_input_px=[5.],
        matched_gt=[0] if hit else [],gt_best_iou=[iou],gt_best_iou50_score=[score])


def test_changes_are_paired_by_gt_not_aggregate_recall():
    result=compare_cases([row(True)],[row(False,score=.01)])
    assert result['counts']=={'lost':1}
    assert result['by_size']=={'4to8':{'lost':1}}
    assert result['lost_diagnostics']=={'iou50_candidate_below_conf_or_assignment':1}
    assert result['changes'][0]['new_iou50_score']==.01
    assert compare_cases([row(False)],[row(True)])['counts']=={'gained':1}


def test_mismatched_annotations_or_coverage_are_rejected():
    with pytest.raises(ValueError,match='coverage'):
        compare_cases([row(True)],[])
    altered=row(True);altered['gt']=[[11,10,20,20]]
    with pytest.raises(ValueError,match='ground truth'):
        compare_cases([row(True)],[altered])
