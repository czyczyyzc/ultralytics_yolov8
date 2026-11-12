import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from scripts.anti_uav.balanced_regression_repair import regression_shares
from scripts.anti_uav.p3_signed_dfl import P3SignedDFLLoss, can_express_at_p3, signed_dfl_loss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.signed_dfl import ltrb


def test_off_grid_original_box_is_representable_and_gets_gradients(tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/v8/yolov8.yaml").read_text())
    config.update(scale="n", p3_signed_dfl_v1=True)
    model = DetectionModel(config, nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    head = model.model[-1]
    assert hasattr(head, "p3_negative_bin") and not hasattr(head, "p3_offset")
    raw = [torch.full((1, channels, size, size), -20.) for channels, size in ((69, 8), (65, 4), (65, 2))]
    target = torch.tensor([5., 5., 8., 7.])
    distances = ltrb(torch.tensor([4., 4.]), target)/8.
    assert can_express_at_p3(torch.tensor([4., 4.]), target)
    assert not can_express_at_p3(torch.tensor([4., 4.]), target + torch.tensor([8., 0., 8., 0.]))
    assert can_express_at_p3(torch.tensor([4., 20.]), torch.tensor([5., 5., 8., 43.]))
    lo = distances.floor().long()
    frac = distances - lo
    for edge in range(4):
        for bin_, prob in ((lo[edge], 1.-frac[edge]), (lo[edge]+1, frac[edge])):
            channel = 64+edge if bin_ == -1 else 16*edge+bin_
            raw[0][0, channel, 0, 0] = prob.log()
    decoded = head._inference(raw)[0, :4, 0]
    torch.testing.assert_close(xywh2xyxy(decoded), target, atol=1e-5, rtol=0)

    manifest = tmp_path/"public.json"
    manifest.write_text("[]")
    source = dict(input_color="gray3", tiny_sampling=dict(public_manifest=str(manifest),
        public_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))
    criterion = P3SignedDFLLoss(model, source)
    criterion.begin_epoch(3)
    # Legacy P3 predicts a point at (4,4), completely outside the original GT.
    for feat in raw:
        feat.fill_(-10.)
        feat[:, [0, 16, 32, 48]] = 10.
        feat.requires_grad_()
    batch = dict(im_file=["/gray/a.jpg"], batch_idx=torch.tensor([0.]), cls=torch.zeros(1, 1),
                 bboxes=torch.tensor([[6.5/64, 6./64, 3./64, 2./64]]))
    loss, parts = criterion(raw, batch)
    assert torch.isfinite(parts).all()
    loss.backward()
    grad = raw[0].grad[0, :, 0, 0]
    assert grad[68] < 0 and grad[64] < 0 and grad[65] < 0 and grad[:64].abs().sum() > 0
    assert raw[1].grad[:, :64].abs().sum() == 0 and raw[2].grad[:, :64].abs().sum() == 0
    assert criterion.audit_summary()["p3_signed_dfl"]["reserved"] == 1

    logits = torch.zeros(1, 4, 17, requires_grad=True)
    signed_dfl_loss(logits, torch.tensor([[-1.1, 15.1, -1., 15.]])).sum().backward()
    assert logits.grad[0, :2].abs().sum() == 0
    assert logits.grad[0, 2, 0] < 0 and logits.grad[0, 3, 16] < 0


def test_unmatched_gt_does_not_change_image_or_bucket_regression_weights():
    scores = torch.tensor([[.8, .6, .25], [.5, 0., 0.], [0., 0., 0.]])
    foreground = torch.tensor([[True, True, True], [True, True, False], [False, False, False]])
    indices = torch.tensor([[0, 0, 1], [0, 2, 2], [0, 0, 0]])
    groups = torch.tensor([[0, 1, 2], [4, 3, 2], [2, 4, 3]])
    valid = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    added = valid.clone()
    added[0, 2] = added[1, 2] = added[2, 0] = True
    base, _, base_budget = regression_shares(scores, foreground, indices, groups, valid, matched_only=True)
    expanded, mass, budget = regression_shares(scores, foreground, indices, groups, added, matched_only=True)
    torch.testing.assert_close(expanded, base, rtol=0, atol=0)
    torch.testing.assert_close(budget, base_budget, rtol=0, atol=0)
    assert expanded[0, 2] > 0 and mass[1, 2] == 0 and budget[2] == 0
    assert expanded[1, 1] == 0 and expanded[2].sum() == 0
