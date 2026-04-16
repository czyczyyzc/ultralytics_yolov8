# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from nanotrack.core.config import cfg
from nanotrack.models.backbone import get_backbone
from nanotrack.models.head import get_ban_head
from nanotrack.models.loss import select_cross_entropy_loss, select_iou_loss
from nanotrack.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = get_backbone(cfg.BACKBONE.TYPE, **cfg.BACKBONE.KWARGS)
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE, **cfg.ADJUST.KWARGS)
        if cfg.BAN.BAN:
            self.ban_head = get_ban_head(cfg.BAN.TYPE, version=cfg.BAN.VERSION, **cfg.BAN.KWARGS)

    def _extract(self, x):
        feat = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            feat = self.neck(feat)
        return feat

    def template(self, z):
        self.zf = self._extract(z)

    def track(self, x):
        xf = self._extract(x)
        cls, loc = self.ban_head(self.zf, xf)
        return {"cls": cls, "loc": loc}

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        template = data["template"]
        search = data["search"]
        label_cls = data["label_cls"]
        label_loc = data["label_loc"]

        zf = self._extract(template)
        xf = self._extract(search)
        cls, loc = self.ban_head(zf, xf)
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = select_iou_loss(loc, label_loc, label_cls)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        return {"total_loss": total_loss, "cls_loss": cls_loss, "loc_loss": loc_loss}
