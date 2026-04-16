# Copyright (c) SenseTime. All Rights Reserved.

from nanotrack.models.backbone.mobile_v3 import mobilenetv3_small, mobilenetv3_small_v3

BACKBONES = {
    "mobilenetv3_small": mobilenetv3_small,
    "mobilenetv3_small_v3": mobilenetv3_small_v3,
}


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
