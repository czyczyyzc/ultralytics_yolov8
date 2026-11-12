import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from scripts.anti_uav.p3_nearest_offset import FALLBACK_SCORE, P3NearestOffsetLoss, reserve_p3_points
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel


def test_offset_initially_exact_and_only_shifts_p3_centers():
    head = Detect(1, (16, 32, 64)).eval()
    head.stride = torch.tensor([8., 16., 32.])
    feats = [torch.randn(1, c, h, h) for c, h in ((16, 8), (32, 4), (64, 2))]
    with torch.no_grad():
        original = head(feats.copy())[0]
        rng = torch.random.get_rng_state()
        head.enable_p3_reference_shift()
        assert torch.equal(torch.random.get_rng_state(), rng)
        shifted = head(feats.copy())[0]
        torch.testing.assert_close(original, shifted, rtol=0, atol=0)
        head.p3_offset.bias.copy_(torch.tensor([.25, -.5]))
        shifted = head(feats.copy())[0]
    torch.testing.assert_close(shifted[:, :2, :64] - original[:, :2, :64],
                               torch.tensor([2., -4.]).view(1, 2, 1).expand(1, 2, 64), atol=1e-5, rtol=0)
    assert torch.equal(original[:, 2:], shifted[:, 2:])
    assert torch.equal(original[:, :, 64:], shifted[:, :, 64:])


def test_rescue_is_p3_specific_preserves_tal_and_never_steals_a_point():
    gt = torch.tensor([[[5., 5., 8., 7.], [3., 3., 5., 5.], [7., 4., 9., 6.], [1., 1., 11., 11.]]])
    classes = torch.zeros(1, 4, 1)
    scores = torch.tensor([[[.9], [0.], [0.], [.8]]])
    fg = scores.squeeze(-1) > 0
    idx = torch.tensor([[1, 0, 0, 0]])
    original_boxes = gt[:, idx[0]].clone()
    assignment = torch.zeros(1, 4, dtype=torch.long), original_boxes.clone(), scores, fg, idx
    forced, audit = reserve_p3_points(assignment, torch.tensor([[4., 4.], [12., 4.], [4., 12.]]),
                                    classes, gt, torch.ones(1, 4, dtype=torch.bool))
    assert forced.tolist() == [[False, True, True, False]]
    assert audit.tolist() == [3, 2, 2]
    torch.testing.assert_close(assignment[1][0, 1:3], gt[0, [0, 2]])
    torch.testing.assert_close(assignment[1][0, [0, 3]], original_boxes[0, [0, 3]])
    torch.testing.assert_close(scores.flatten(), torch.tensor([.9, FALLBACK_SCORE, FALLBACK_SCORE, .8]))


def test_zero_overlap_and_subpixel_short_edge_receive_cls_dfl_and_signed_offset_gradients(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'ultralytics/cfg/models/v8/yolov8.yaml').read_text())
    config.update(scale='n', p3_nearest_offset_v1=True)
    model = DetectionModel(config, nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    manifest = tmp_path / 'public.json'
    manifest.write_text('[]')
    source = dict(input_color='gray3', tiny_sampling=dict(public_manifest=str(manifest),
        public_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    criterion = P3NearestOffsetLoss(model, source)
    criterion.begin_epoch(3)
    raw = [torch.full((1, channels, size, size), -10.) for channels, size in ((67, 8), (65, 4), (65, 2))]
    for feat in raw:
        feat[:, [0, 16, 32, 48]] = 10
    raw[0][:, 64:66] = 0
    for feat in raw:
        feat.requires_grad_()
    # (5, 1.5, 8, 2.25) lies fully off-grid, nearest P3 point at (4, 4).
    batch = dict(im_file=['/gray/a.jpg'], batch_idx=torch.tensor([0.]), cls=torch.zeros(1, 1),
                 bboxes=torch.tensor([[6.5/64, 1.875/64, 3/64, .75/64]]))
    value, parts = criterion(raw, batch)
    assert torch.isfinite(value) and torch.isfinite(parts).all()
    value.backward()
    p3 = raw[0].grad[0, :, 0, 0]
    assert p3[66] < 0
    assert p3[64] < 0 < p3[65]
    assert p3[:64].abs().sum() > 0
    assert raw[1].grad[:, :64].abs().sum() == 0
    assert criterion.audit_summary()['p3_offset']['reserved'] == 1
    batch.update(batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4))
    value, parts = criterion(raw, batch)
    assert torch.isfinite(value) and parts[0] == parts[2] == 0 and parts[1] > 0


def test_equal_fp_curve_uses_ties_and_actual_input_sizes():
    from scripts.anti_uav.evaluate_p3_equal_fp import summarize

    frame = dict(original_hw=[1080, 1920], input_hw=[544, 960], boxes=[[0, 0, 6, 6], [20, 20, 32, 32]], classes=[0, 0],
        preds=[[0, 0, 6, 6, .9, 0], [20, 20, 29, 29, .8, 0], [40, 40, 50, 50, .8, 0]])
    empty = dict(frame, boxes=[], classes=[], preds=[])
    budgets = summarize([frame, empty], (0, 500))
    assert budgets["0"]["sizes"]["long_lt4px"]["recall50"] == 1
    assert budgets["0"]["sizes"]["long_4to8px"]["recall50"] == 0
    assert budgets["500"]["fp"] == 1
    assert budgets["500"]["sizes"]["long_4to8px"]["recall50"] == 1
    assert budgets["500"]["sizes"]["long_4to8px"]["recall75"] == 0
