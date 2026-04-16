# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F


def xcorr_fast(x, kernel):
    """Group conv2d to calculate cross correlation."""
    batch = kernel.size(0)
    pk = kernel.view(-1, x.size(1), kernel.size(2), kernel.size(3))
    px = x.view(1, -1, x.size(2), x.size(3))
    po = F.conv2d(px, pk, groups=batch)
    return po.view(batch, -1, po.size(2), po.size(3))


def xcorr_depthwise(x, kernel):
    """Depthwise cross correlation."""
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    return out.view(batch, channel, out.size(2), out.size(3))


def xcorr_pixelwise(x, kernel):
    """Pixel-wise correlation implemented by matrix multiplication."""
    batch, channels, height, width = x.size()
    kernel_mat = kernel.view(batch, channels, -1).transpose(1, 2)
    x_mat = x.view(batch, channels, -1)
    return torch.matmul(kernel_mat, x_mat).view(batch, -1, height, width)
