from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.anti_uav.balanced_regression_repair import (
    BalancedRegressionDetectionLoss, gt_groups, interior_grid_coverage,
    regression_shares, repair_progress,
)
from scripts.anti_uav.tiny_aware_loss import TinyAwareDetectionLoss
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]


def source(tmp_path):
    path = tmp_path/'public.json'
    content = json.dumps([dict(image='/public/b.jpg', split='train')]).encode()
    path.write_bytes(content)
    return dict(input_color='gray3', tiny_sampling=dict(public_manifest=str(path),
        public_manifest_sha256=hashlib.sha256(content).hexdigest()))


def toy_model():
    model = DetectionModel(ROOT/'ultralytics/cfg/models/v8/yolov8.yaml', nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    return model


def test_multi_object_and_multi_anchor_images_do_not_get_extra_bucket_credit():
    scores = torch.tensor([[.6,.6,0.,0.],[.8,.8,.8,.8]])
    fg = scores > 0
    indices = torch.tensor([[0,0,0,0],[0,0,1,1]])
    groups = torch.tensor([[1,0],[4,4]])
    valid = torch.tensor([[True,False],[True,True]])
    weights, mass, budget = regression_shares(scores, fg, indices, groups, valid)
    torch.testing.assert_close(weights.sum(1), torch.tensor([.5,.5]))
    torch.testing.assert_close(weights[1,:2].sum(), torch.tensor(.25))
    assert mass[0,0] == pytest.approx(1.2)
    assert budget[1] == .5 and budget[4] == .5


def test_unassigned_gt_is_not_fabricated_or_redirected_to_public():
    scores = torch.tensor([[0.,0.],[.8,.8]])
    weights, mass, budget = regression_shares(scores, scores>0, torch.zeros((2,2),dtype=torch.long),
        torch.tensor([[1],[4]]), torch.ones((2,1),dtype=torch.bool))
    assert weights[0].sum() == 0
    assert weights[1].sum() == .5
    assert mass[0,0] == 0 and budget[1] == .5


def test_empty_batch_and_zero_quality_remain_finite():
    for count in [0,1]:
        weights,_,_=regression_shares(torch.zeros(2,3),torch.zeros(2,3,dtype=torch.bool),
            torch.zeros(2,3,dtype=torch.long),torch.zeros(2,count,dtype=torch.long),
            torch.zeros(2,count,dtype=torch.bool))
        assert torch.isfinite(weights).all() and weights.sum()==0


def test_scale_groups_use_augmented_geometry_and_source_priority():
    boxes=torch.tensor([[[0.,0.,3.,2.],[0.,0.,6.,4.],[0.,0.,20.,20.]]]*3)
    groups=gt_groups(boxes,torch.ones(3,3,dtype=torch.bool),torch.tensor([0,3,4]))
    assert groups.tolist()==[[0,1,2],[3,3,3],[4,4,4]]


def test_grid_coverage_uses_strict_interiors_and_actual_output_strides():
    boxes=torch.tensor([[[3.,3.,5.,5.],[7.,7.,9.,9.],[4.,4.,6.,6.]]])
    valid=torch.ones(1,3,dtype=torch.bool)
    assert interior_grid_coverage(boxes,valid,[8.],(64,64)).tolist()==[[True,False,False]]
    assert interior_grid_coverage(boxes,valid,[8.,16.,32.],(64,64)).tolist()==[[True,True,False]]


def test_schedule_is_opt_in_and_starts_with_one_exact_legacy_epoch():
    assert [repair_progress(i,3) for i in [0,1,3,9]]==[0.,1/3,1.,1.]
    with pytest.raises(ValueError):repair_progress(0,0)


def test_idle_gpu_selector_never_uses_another_occupied_card():
    from scripts.anti_uav.run_regression_repair_experiment import select_idle_gpu
    usage='0, 13000, 68000, 0\n1, 0, 81154, 0\n2, 9000, 72000, 94\n3, 0, 81154, 0\n'
    assert select_idle_gpu(usage,list(range(8)))==1
    assert select_idle_gpu(usage,[2,3])==3
    assert select_idle_gpu(usage,[0,2]) is None


def test_epoch_zero_is_exact_legacy_then_repair_backpropagates(tmp_path):
    model=toy_model().train()
    config=source(tmp_path)
    options=dict(tiny_multiplier=1.75,localization_weight=.12,localization_min_px=4.,localization_max_px=64.)
    legacy=TinyAwareDetectionLoss(model,**options)
    repaired=BalancedRegressionDetectionLoss(model,config,**options)
    image=torch.rand(2,3,64,64)
    raw=model(image)
    batch=dict(img=image,im_file=['/gray/a.jpg','/public/b.jpg'],
        batch_idx=torch.tensor([0.,1.,1.]),cls=torch.zeros(3,1),
        bboxes=torch.tensor([[.5625,.5625,6/64,4/64],[.25,.25,.25,.25],[.75,.75,.25,.25]]))
    expected,items=legacy(raw,batch)
    actual,actual_items=repaired(raw,batch)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual_items,items,rtol=0,atol=0)
    repaired.begin_epoch(3)
    loss,items=repaired(raw,batch)
    assert torch.isfinite(loss) and torch.isfinite(items).all()
    loss.backward()
    assert model.model[0].conv.weight.grad is not None
    assert torch.isfinite(model.model[0].conv.weight.grad).all()
    audit=repaired.audit_summary()
    assert audit['balance_mix']==.5
    assert audit['groups']['gray_4to8']['gt']==1
    assert audit['groups']['public']['gt']==2
    clone=deepcopy(repaired)
    assert clone.public_sha256==repaired.public_sha256


def test_empty_detector_batch_keeps_background_classification(tmp_path):
    model=toy_model().train()
    loss=BalancedRegressionDetectionLoss(model,source(tmp_path))
    loss.begin_epoch(3)
    image=torch.rand(2,3,64,64)
    value,items=loss(model(image),dict(img=image,im_file=['/gray/a.jpg','/gray/c.jpg'],
        batch_idx=torch.empty(0),cls=torch.empty(0,1),bboxes=torch.empty(0,4)))
    assert torch.isfinite(value) and items[0]==0 and items[2]==0 and items[1]>0


def test_tiny_width_height_term_penalizes_oversize_without_moving_gt(tmp_path):
    model=toy_model()
    criterion=BalancedRegressionDetectionLoss(model,source(tmp_path),localization_weight=.12)
    target=torch.tensor([[2.,2.,8.,6.]])
    original=target.clone()
    exact=criterion.balanced_edge_loss(target,target,torch.tensor([1.]))
    prediction=torch.tensor([[1.,1.,9.,7.]],requires_grad=True)
    oversized=criterion.balanced_edge_loss(prediction,target,torch.tensor([1.]))
    assert exact==0 and oversized>0
    oversized.backward()
    assert prediction.grad[0,0]<0 and prediction.grad[0,2]>0
    torch.testing.assert_close(target,original,rtol=0,atol=0)


def test_repaired_student_backpropagates_and_exported_p3_remains_exact(tmp_path):
    from scripts.anti_uav.p3_student_distillation import P3StudentDistillationLoss, transfer_student_to_p3
    from scripts.anti_uav.regression_repair_trainers import repaired_loss
    from scripts.anti_uav.train_frozen_p3_addon_p2 import transfer_frozen_p3_weights
    p3=toy_model()
    student=DetectionModel(ROOT/'ultralytics/cfg/models/v8/yolov8-frozen-p3-addon-p2.yaml',nc=1,verbose=False)
    transfer_frozen_p3_weights(p3,student)
    teacher=deepcopy(student).eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    student.args=p3.args
    criterion=P3StudentDistillationLoss(student,teacher,localization_weight=.12)
    config=source(tmp_path);options=dict(balance_mix=.5,warmup_epochs=3,tiny_log_size_weight=.5)
    criterion.primary=repaired_loss(student,config,criterion.primary,options)
    criterion.auxiliary=repaired_loss(student,config,criterion.auxiliary,options)
    criterion.primary.begin_epoch(3);criterion.auxiliary.begin_epoch(3)
    image=torch.rand(2,3,64,64)
    batch=dict(img=image,im_file=['/gray/a.jpg','/public/b.jpg'],batch_idx=torch.tensor([0.,1.]),
        cls=torch.zeros(2,1),bboxes=torch.tensor([[.5625,.5625,6/64,4/64],[.5,.5,.25,.25]]))
    student.train();loss,items=criterion(student(image),batch)
    assert torch.isfinite(loss) and torch.isfinite(items).all()
    loss.backward()
    assert torch.isfinite(student.model[0].conv.weight.grad).all()
    assert all(p.grad is None for p in teacher.parameters())
    student.model[-1].auxiliary_training_only=True;student.eval()
    target=toy_model().eval();transfer_student_to_p3(student,target)
    with torch.inference_mode():
        reference=student(image)[1];actual=target(image)[1]
    assert len(actual)==len(reference)==3
    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,rtol=0,atol=0)
