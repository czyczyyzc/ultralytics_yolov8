import torch
from torch import nn


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tgt_left, tgt_top, tgt_right, tgt_bottom = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        tgt_area = (tgt_left + tgt_right) * (tgt_top + tgt_bottom)

        w_intersect = torch.min(pred_left, tgt_left) + torch.min(pred_right, tgt_right)
        gw_intersect = torch.max(pred_left, tgt_left) + torch.max(pred_right, tgt_right)
        h_intersect = torch.min(pred_bottom, tgt_bottom) + torch.min(pred_top, tgt_top)
        gh_intersect = torch.max(pred_bottom, tgt_bottom) + torch.max(pred_top, tgt_top)
        ac_union = gw_intersect * gh_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = tgt_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_union - area_union) / ac_union

        if self.loc_loss_type == "iou":
            losses = -torch.log(ious)
        elif self.loc_loss_type == "linear_iou":
            losses = 1 - ious
        elif self.loc_loss_type == "giou":
            losses = 1 - gious
        else:
            raise NotImplementedError(self.loc_loss_type)

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        assert losses.numel() != 0
        return losses.mean()


linear_iou = IOULoss(loc_loss_type="linear_iou")
