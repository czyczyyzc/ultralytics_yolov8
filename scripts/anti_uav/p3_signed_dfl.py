"""Retain standard TAL and supplement off-grid P3 GT with representable signed DFL targets."""

import torch

from scripts.anti_uav.balanced_regression_repair import BalancedRegressionDetectionLoss, interior_grid_coverage
from scripts.anti_uav.p3_nearest_offset import FALLBACK_SCORE, reserve_p3_points
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.signed_dfl import MAX_BIN, MIN_BIN, ltrb, representable, signed_dfl_decode, signed_dfl_logits
from ultralytics.utils.tal import bbox2dist, dist2bbox

P3_STRIDE = 8.


def can_express_at_p3(points_px, boxes_px):
    return representable(ltrb(points_px, boxes_px) / P3_STRIDE)


def signed_dfl_loss(logits, targets):
    """Interpolated CE, with no clamping of out-of-range edges."""
    valid = (targets >= MIN_BIN) & (targets <= MAX_BIN)
    pos = targets[valid].float() - MIN_BIN
    lo = pos.floor().long()
    hi = (lo + 1).clamp_max(MAX_BIN - MIN_BIN)
    frac = pos - lo
    logp = logits[valid].float().log_softmax(-1)
    cost = logp.new_zeros(targets.shape)
    cost[valid] = -logp.gather(-1, lo[:, None]).squeeze(-1) * (1-frac) - logp.gather(-1, hi[:, None]).squeeze(-1) * frac
    return cost.mean(-1)


class P3SignedDFLLoss(BalancedRegressionDetectionLoss):
    matched_regression_budget = True

    def begin_epoch(self, epoch):
        super().begin_epoch(epoch)
        self.signed_audit = None

    def audit_summary(self):
        result = super().audit_summary()
        counts = self.signed_audit.tolist() if self.signed_audit is not None else [0, 0, 0]
        result.update(classification_and_assigner_unchanged=False, normal_tal_unchanged=True,
            regression_budget_matched_gt_only=True, p3_signed_dfl=dict(
                no_interior_p3_gt=counts[0], lacked_positive_p3=counts[1], reserved=counts[2],
                no_free_representable_p3=counts[1]-counts[2], bins=[MIN_BIN, MAX_BIN], fallback_score=FALLBACK_SCORE))
        return result

    def prediction_parts(self, feats):
        n = self.reg_max * 4
        self.negative = feats[0][:, n:n + 4].flatten(2).transpose(1, 2)
        p3 = torch.cat((feats[0][:, :n], feats[0][:, n + 4:]), 1)
        return super().prediction_parts([p3, *feats[1:]])

    def decode_predictions(self, anchors, pred_distri):
        legacy = super().decode_predictions(anchors, pred_distri)
        self.tal_boxes = legacy.detach()
        points = self.negative.shape[1]
        signed = signed_dfl_logits(pred_distri[:, :points], self.negative)
        p3 = dist2bbox(signed_dfl_decode(signed), anchors[:points], xywh=False)
        return torch.cat((p3, legacy[:, points:]), 1)

    def assign_predictions(self, scores, boxes, anchors, stride, labels, gt_boxes, valid, pred_distri):
        # TAL sees exactly the legacy nonnegative projection, before any supplementation.
        legacy = (self.tal_boxes * stride).to(gt_boxes.dtype)
        assignment = super().assign_predictions(scores, legacy, anchors, stride, labels, gt_boxes, valid, pred_distri)
        points = self.negative.shape[1]
        eligible = ~interior_grid_coverage(gt_boxes.float(), valid, (P3_STRIDE,), self.last_input_hw)
        _, counts = reserve_p3_points(assignment, anchors[:points] * stride[:points], labels, gt_boxes, valid,
                                     eligible=eligible, can_express=can_express_at_p3)
        self.signed_audit = counts if self.signed_audit is None else self.signed_audit + counts
        return assignment

    def regression_terms(self, pred_distri, pred_boxes, anchors, targets, scores, denominator, fg):
        weights = scores.sum(-1)
        iou = bbox_iou(pred_boxes[fg], targets[fg], xywh=False, CIoU=True).squeeze(-1)
        iou_loss = ((1. - iou) * weights[fg]).sum() / denominator
        points = self.negative.shape[1]
        p3_fg, coarse_fg = fg[:, :points], fg[:, points:]
        p3_refs = anchors[None, :points].expand(len(fg), -1, -1)[p3_fg]
        p3_targets = ltrb(p3_refs, targets[:, :points][p3_fg])
        p3_logits = signed_dfl_logits(pred_distri[:, :points][p3_fg], self.negative[p3_fg])
        p3_loss = (signed_dfl_loss(p3_logits, p3_targets) * weights[:, :points][p3_fg]).sum()
        coarse_targets = bbox2dist(anchors[points:], targets[:, points:], self.reg_max - 1)[coarse_fg]
        coarse_logits = pred_distri[:, points:][coarse_fg].reshape(-1, self.reg_max)
        coarse_loss = (self.bbox_loss.dfl_loss(coarse_logits, coarse_targets).squeeze(-1) *
                       weights[:, points:][coarse_fg]).sum()
        return iou_loss, (p3_loss + coarse_loss) / denominator

    def __call__(self, preds, batch):
        try:
            return super().__call__(preds, batch)
        finally:
            self.negative = self.tal_boxes = None
