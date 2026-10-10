"""Bounded supervision reweighting for 4-8 px targets."""

from __future__ import annotations

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class TinyAwareDetectionLoss(v8DetectionLoss):
    """Increase positive classification/box gradients for tiny assigned targets only."""

    def __init__(self, model, tiny_multiplier=1.75, tiny_min_px=4.0, tiny_max_px=8.0):
        super().__init__(model)
        self.tiny_multiplier = float(tiny_multiplier)
        self.tiny_min_px = float(tiny_min_px)
        self.tiny_max_px = float(tiny_max_px)
        if not 1.0 <= self.tiny_multiplier <= 2.0:
            raise ValueError("tiny_multiplier must stay in the bounded [1, 2] range")
        if not 0 <= self.tiny_min_px < self.tiny_max_px:
            raise ValueError("Invalid tiny target size range")

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [feature.view(feats[0].shape[0], self.no, -1) for feature in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0).bool()
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        normalizer = max(target_scores.sum(), 1)

        gt_size = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).amax(dim=-1)
        tiny_gt = (
            mask_gt.squeeze(-1)
            & (gt_size >= self.tiny_min_px)
            & (gt_size <= self.tiny_max_px)
        )
        if tiny_gt.shape[1]:
            assigned_tiny = tiny_gt.gather(1, target_gt_idx.clamp(0, tiny_gt.shape[1] - 1)) & fg_mask
        else:
            assigned_tiny = torch.zeros_like(fg_mask)
        anchor_weight = torch.where(
            assigned_tiny,
            pred_scores.new_tensor(self.tiny_multiplier),
            pred_scores.new_tensor(1.0),
        )

        cls = self.bce(pred_scores, target_scores.to(dtype)).sum(dim=-1)
        loss[1] = (cls * anchor_weight).sum() / normalizer
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            weighted_scores = target_scores * anchor_weight.unsqueeze(-1)
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                weighted_scores,
                normalizer,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


class TinyAwareAddOnP2DetectionLoss(TinyAwareDetectionLoss):
    """Apply tiny-aware supervision to only the disposable first/P2 level."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.stride = self.stride[:1]

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        return super().__call__([feats[0]], batch)
