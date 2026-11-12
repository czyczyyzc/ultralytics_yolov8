"""Opt-in P3-only fallback assignment and signed reference-point supervision."""

import torch
import torch.nn.functional as F

from scripts.anti_uav.balanced_regression_repair import BalancedRegressionDetectionLoss

TINY_MAX_PX = 8.
FALLBACK_SCORE = .25
OFFSET_LOSS_WEIGHT = .5


@torch.no_grad()
def reserve_p3_points(assignment, p3_pixels, labels, gt_boxes, valid, *, eligible=None, can_express=None):
    """Add one nearest free P3 point per unmatched tiny GT without overwriting TAL."""
    target_labels, targets, scores, foreground, indices = assignment
    points = len(p3_pixels)
    forced = torch.zeros_like(foreground, dtype=torch.bool)
    wh = gt_boxes[..., 2:] - gt_boxes[..., :2]
    eligible = (wh.amax(-1) <= TINY_MAX_PX if eligible is None else eligible) & valid & (wh > 0).all(-1)
    if not gt_boxes.shape[1]:
        zero = torch.zeros((), device=gt_boxes.device, dtype=torch.long)
        return forced, torch.stack((zero, zero, zero))
    p3_fg = foreground[:, :points].bool()
    p3_supervised = p3_fg & (scores[:, :points].sum(-1) > 0)
    hits = torch.zeros_like(valid, dtype=torch.long)
    hits.scatter_add_(1, indices[:, :points].long().clamp(0, valid.shape[1] - 1), p3_supervised.long())
    missing = eligible & (hits == 0)
    remaining = missing.clone()
    while remaining.any():
        batch_ids = remaining.any(dim=1).nonzero().flatten()
        gt_ids = remaining[batch_ids].long().argmax(1)
        boxes = gt_boxes[batch_ids, gt_ids]
        centers = (boxes[:, :2] + boxes[:, 2:]) * .5
        distances = (p3_pixels.float()[None] - centers[:, None].float()).square().sum(-1)
        distances.masked_fill_(foreground[batch_ids, :points].bool(), float("inf"))
        if can_express is not None:
            distances.masked_fill_(~can_express(p3_pixels.float()[None], boxes[:, None].float()), float("inf"))
        best, anchor_ids = distances.min(1)
        remaining[batch_ids, gt_ids] = False
        free = best.isfinite()
        b, a, g = batch_ids[free], anchor_ids[free], gt_ids[free]
        cls = labels[b, g, 0].long()
        targets[b, a] = gt_boxes[b, g].to(targets.dtype)
        target_labels[b, a] = cls.to(target_labels.dtype)
        scores[b, a] = 0
        scores[b, a, cls] = FALLBACK_SCORE
        foreground[b, a] = True
        indices[b, a] = g.to(indices.dtype)
        forced[b, a] = True
    return forced, torch.stack((eligible.sum(), missing.sum(), forced.sum()))


class P3NearestOffsetLoss(BalancedRegressionDetectionLoss):
    """Keep the original DFL branch, shifting only the P3 reference point."""

    def begin_epoch(self, epoch):
        super().begin_epoch(epoch)
        self.offset_audit = None

    def audit_summary(self):
        result = super().audit_summary()
        counts = self.offset_audit.tolist() if self.offset_audit is not None else [0, 0, 0]
        result.update(classification_and_assigner_unchanged=False, p3_offset=dict(
            tiny_gt=counts[0], lacked_positive_p3=counts[1], reserved=counts[2],
            no_free_p3=counts[1]-counts[2], fallback_score=FALLBACK_SCORE,
            offset_loss_weight=OFFSET_LOSS_WEIGHT))
        return result

    def prediction_parts(self, feats):
        n = self.reg_max * 4
        self.offsets = feats[0][:, n:n + 2].flatten(2).transpose(1, 2).float()
        p3 = torch.cat((feats[0][:, :n], feats[0][:, n + 2:]), 1)
        return super().prediction_parts([p3, *feats[1:]])

    def decode_predictions(self, anchors, pred_distri):
        boxes = super().decode_predictions(anchors, pred_distri)
        points = self.offsets.shape[1]
        p3 = boxes[:, :points].float() + self.offsets.repeat(1, 1, 2)
        return torch.cat((p3, boxes[:, points:]), 1)

    def assign_predictions(self, scores, boxes, anchors, stride, labels, gt_boxes, valid, pred_distri):
        assignment = super().assign_predictions(scores, boxes, anchors, stride, labels, gt_boxes, valid, pred_distri)
        points = self.offsets.shape[1]
        self.forced, counts = reserve_p3_points(assignment, anchors[:points] * stride[:points], labels, gt_boxes, valid)
        self.offset_audit = counts if self.offset_audit is None else self.offset_audit + counts
        return assignment

    def regression_terms(self, pred_distri, pred_boxes, anchors, targets, scores, denominator, fg):
        centers = (targets[..., :2] + targets[..., 2:]) * .5
        references = torch.where(self.forced[..., None], centers, anchors[None])
        iou, dfl = super().regression_terms(pred_distri, pred_boxes, references, targets, scores, denominator, fg)
        points = self.offsets.shape[1]
        delta = references[:, :points] - anchors[:points]
        weights = scores[:, :points].sum(-1) * fg[:, :points]
        error = F.smooth_l1_loss(self.offsets, delta.float(), beta=.125, reduction="none").sum(-1)
        return iou, dfl + OFFSET_LOSS_WEIGHT * (error * weights).sum() / denominator

    def __call__(self, preds, batch):
        try:
            return super().__call__(preds, batch)
        finally:
            self.offsets = self.forced = None
