# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from nanotrack.models.iou_loss import linear_iou


def get_cls_loss(pred, label, select):
    if select.numel() == 0:
        return pred.sum() * 0.0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = torch.nonzero(label.eq(1), as_tuple=False).view(-1).to(label.device)
    neg = torch.nonzero(label.eq(0), as_tuple=False).view(-1).to(label.device)
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = torch.nonzero(label_cls.eq(1), as_tuple=False).view(-1).to(label_cls.device)
    if pos.numel() == 0:
        return pred_loc.sum() * 0.0
    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)
    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)
    return linear_iou(pred_loc, label_loc)
