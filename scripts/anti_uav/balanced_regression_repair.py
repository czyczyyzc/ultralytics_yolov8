"""Opt-in training-only repair for source/scale and multi-object regression imbalance."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.anti_uav.tiny_aware_loss import TinyAwareDetectionLoss
from ultralytics.utils.tal import make_anchors


GROUPS = ('gray_sub4', 'gray_4to8', 'gray_other', 'antiuav', 'public')
AUDIT_COLUMNS = ('gt', 'assigned_gt', 'no_interior_grid_gt', 'positive_anchors',
                 'assignment_score_sum', 'normalized_regression_mass')


@lru_cache(maxsize=8)
def public_paths(manifest, expected_sha256):
    content = Path(manifest).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Public source manifest changed')
    records = json.loads(content)
    paths = frozenset(row['image'] for row in records)
    if len(paths) != len(records) or any(row['split'] != 'train' for row in records):
        raise ValueError('Public source manifest is not unique training data')
    return paths


def gt_groups(boxes, valid, domains):
    """Group augmented GT geometry, not the pre-augmentation sampler category."""
    edges = (boxes[..., 2:] - boxes[..., :2]).amax(-1)
    groups = torch.where(edges < 4, 0, torch.where(edges <= 8, 1, 2))
    groups = torch.where(domains[:, None] == 3, 3, groups)
    groups = torch.where(domains[:, None] == 4, 4, groups)
    return groups.masked_fill(~valid, 0)


@torch.no_grad()
def regression_shares(scores, foreground, target_index, groups, valid):
    """One image credit, then one quality-weighted GT credit within each bucket.

    Missing assignments receive zero loss, not fabricated boxes. Their budget is
    not transferred to another source. All reductions use FP32 under AMP.
    """
    scores = scores.detach().float() * foreground
    batch, anchors = scores.shape
    count = valid.shape[1]
    if count == 0:
        empty = scores.new_zeros((batch, 0))
        return scores.new_zeros((batch, anchors)), empty, scores.new_zeros(len(GROUPS))
    indices = target_index.clamp(0, count - 1)
    mass = scores.new_zeros((batch, count)).scatter_add_(1, indices, scores)
    quality = scores.new_zeros((batch, count)).scatter_reduce_(
        1, indices, scores, reduce='amax', include_self=True)
    credit = valid.float() / valid.sum(1, keepdim=True).clamp_min(1)
    budget = scores.new_zeros(len(GROUPS)).scatter_add_(0, groups[valid], credit[valid])
    budget /= valid.any(1).sum().clamp_min(1)
    importance = credit * quality * (mass > 0)
    denominator = scores.new_zeros(len(GROUPS)).scatter_add_(0, groups[valid], importance[valid])
    gt_share = importance * budget[groups] / denominator[groups].clamp_min(1e-8)
    share = scores / mass.gather(1, indices).clamp_min(1e-8) * gt_share.gather(1, indices)
    return share, mass, budget


@torch.no_grad()
def interior_grid_coverage(boxes, valid, strides, image_hw):
    """Count strict in-GT grid eligibility without allocating GT x anchor masks."""
    available = torch.zeros_like(valid)
    for stride in strides:
        s = float(stride)
        lower = torch.floor((boxes[..., :2] + 1e-7) / s - .5) + 1
        upper = torch.ceil((boxes[..., 2:] - 1e-7) / s - .5) - 1
        maximum = boxes.new_tensor([image_hw[1] / s - 1, image_hw[0] / s - 1])
        lower = lower.clamp_min(0)
        upper = torch.minimum(upper, maximum)
        available |= (upper >= lower).all(-1)
    return available & valid


def repair_progress(epoch, warmup_epochs):
    if warmup_epochs < 1:
        raise ValueError('At least one legacy warmup epoch is required')
    return min(1., max(0., epoch / warmup_epochs))


class BalancedRegressionDetectionLoss(TinyAwareDetectionLoss):
    """Leave classification/assignment intact; repair box/DFL and tiny-size supervision."""

    def __init__(self, model, source, balance_mix=.5, warmup_epochs=3,
                 tiny_log_size_weight=.5, first_level_only=False, **legacy_options):
        super().__init__(model, **legacy_options)
        if not 0 < balance_mix <= .75 or not 0 <= tiny_log_size_weight <= 1:
            raise ValueError('Repair weights exceed their bounded range')
        repair_progress(0, warmup_epochs)
        if source.get('input_color') != 'gray3':
            raise ValueError('This controlled repair experiment requires gray3 input')
        config = source['tiny_sampling']
        self.public_manifest = config['public_manifest']
        self.public_sha256 = config['public_manifest_sha256']
        public_paths(self.public_manifest, self.public_sha256)
        self.balance_mix = float(balance_mix)
        self.warmup_epochs = int(warmup_epochs)
        self.tiny_log_size_weight = float(tiny_log_size_weight)
        self.first_level_only = first_level_only
        self.begin_epoch(0)

    def begin_epoch(self, epoch):
        self.epoch = int(epoch)
        self.progress = repair_progress(epoch, self.warmup_epochs)
        self.audit_totals = None
        self.audit_batches = 0

    def image_domains(self, paths, device):
        public = public_paths(self.public_manifest, self.public_sha256)
        domains = [4 if path in public else 3 if '/anti_uav300_yolo/' in path else 0 for path in paths]
        return torch.tensor(domains, device=device, dtype=torch.long)

    @torch.no_grad()
    def record_audit(self, scores, foreground, indices, groups, valid, mass, geometry, weights):
        count = valid.shape[1]
        result = scores.new_zeros((len(GROUPS), len(AUDIT_COLUMNS)), dtype=torch.float32)
        if count:
            anchor_groups = groups.gather(1, indices.clamp(0, count - 1))
            values = [torch.bincount(groups[valid], minlength=len(GROUPS)),
                torch.bincount(groups[valid & (mass > 0)], minlength=len(GROUPS)),
                torch.bincount(groups[valid & ~geometry], minlength=len(GROUPS)),
                torch.bincount(anchor_groups[foreground], minlength=len(GROUPS)),
                torch.bincount(anchor_groups[foreground], weights=scores[foreground].float(), minlength=len(GROUPS)),
                torch.bincount(anchor_groups[foreground], weights=weights[foreground].float(), minlength=len(GROUPS))]
            result = torch.stack(values, dim=1).detach()
        self.audit_totals = result if self.audit_totals is None else self.audit_totals + result
        self.audit_batches += 1

    def audit_summary(self):
        values = self.audit_totals.cpu().tolist() if self.audit_totals is not None else []
        return dict(epoch_zero_based=self.epoch, batches=self.audit_batches,
            balance_mix=self.balance_mix * self.progress, repair_progress=self.progress,
            input_hw=getattr(self, 'last_input_hw', None), batch_size=getattr(self, 'last_batch_size', None),
            legacy_tiny_multiplier=self.tiny_multiplier,
            legacy_localization_range_px=[self.localization_min_px, self.localization_max_px],
            repaired_localization_min_px=2., extra_log_size_range_px=[2., 16.],
            classification_and_assigner_unchanged=True, gradient_norm_not_measured=True,
            groups={key: dict(zip(AUDIT_COLUMNS, row)) for key, row in zip(GROUPS, values)})

    def balanced_edge_loss(self, prediction, target, weights):
        wh = (target[..., 2:] - target[..., :2]).float().clamp_min(1.)
        edge = wh.amax(-1)
        selected = (edge >= 2) & (edge <= self.localization_max_px)
        scale = torch.cat((wh, wh), -1)
        delta = (prediction.float() - target.float()) / scale
        per_anchor = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction='none', beta=.125).mean(-1)
        pred_wh = (prediction[..., 2:] - prediction[..., :2]).float().clamp_min(.1)
        log_ratio = (pred_wh / wh).log()
        size = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), reduction='none', beta=.125).mean(-1)
        per_anchor += self.tiny_log_size_weight * size * (edge <= 16)
        return (per_anchor * weights * selected).sum()

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        if self.first_level_only:
            feats = feats[:1]
        pred_distri, pred_scores = torch.cat(
            [x.view(feats[0].shape[0], self.no, -1) for x in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        bs = pred_scores.shape[0]
        if not hasattr(self, '_stride_values'):
            self._stride_values = tuple(self.stride.detach().cpu().tolist())
        image_hw = (feats[0].shape[2] * self._stride_values[0], feats[0].shape[3] * self._stride_values[0])
        self.last_input_hw, self.last_batch_size = [int(x) for x in image_hw], bs
        imgsz = torch.tensor(image_hw, device=self.device, dtype=pred_scores.dtype)
        anchors, stride = make_anchors(feats, self.stride, .5)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        labels, gt_boxes = targets.split((1, 4), 2)
        valid = gt_boxes.sum(2).gt(0)
        pred_boxes = self.bbox_decode(anchors, pred_distri)
        _, target_boxes, target_scores, fg, indices = self.assigner(
            pred_scores.detach().sigmoid(), (pred_boxes.detach() * stride).type(gt_boxes.dtype),
            anchors * stride, labels, gt_boxes, valid.unsqueeze(-1))
        fg, indices = fg.bool(), indices.long()
        normalizer = max(target_scores.sum(), 1)
        long_edges = (gt_boxes[..., 2:] - gt_boxes[..., :2]).amax(-1)
        tiny_gt = valid & (long_edges >= self.tiny_min_px) & (long_edges <= self.tiny_max_px)
        assigned_tiny = tiny_gt.gather(1, indices.clamp(0, tiny_gt.shape[1]-1)) & fg if tiny_gt.shape[1] else torch.zeros_like(fg)
        multiplier = torch.where(assigned_tiny, pred_scores.new_tensor(self.tiny_multiplier), pred_scores.new_tensor(1.))
        legacy_scores = target_scores * multiplier.unsqueeze(-1)
        domains = self.image_domains(batch['im_file'], gt_boxes.device)
        groups = gt_groups(gt_boxes, valid, domains)
        balanced, mass, _ = regression_shares(target_scores.sum(-1), fg, indices, groups, valid)
        mixed = (1-self.balance_mix*self.progress) * (legacy_scores.sum(-1)/normalizer) + self.balance_mix*self.progress*balanced
        geometry = interior_grid_coverage(gt_boxes.float(), valid, self._stride_values, image_hw)
        self.record_audit(target_scores.sum(-1), fg, indices, groups, valid, mass, geometry, mixed)
        loss = torch.zeros(3, device=self.device)
        loss[1] = (self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum(-1) * multiplier).sum()/normalizer
        if fg.any():
            zero = pred_boxes.sum()*0.
            legacy_edge = self.relative_edge_loss(pred_boxes, target_boxes, target_scores, fg, stride, normalizer) if self.progress < 1 else zero
            balanced_edge = self.balanced_edge_loss((pred_boxes*stride)[fg], target_boxes[fg], balanced[fg]) if self.localization_weight and self.progress else zero
            box_scores, denominator = (legacy_scores, normalizer) if self.progress == 0 else (mixed.unsqueeze(-1), 1.)
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_boxes, anchors, target_boxes/stride,
                                           box_scores, denominator, fg)
            loss[0] += self.localization_weight * ((1-self.progress)*legacy_edge + self.progress*balanced_edge)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum()*bs, loss.detach()
