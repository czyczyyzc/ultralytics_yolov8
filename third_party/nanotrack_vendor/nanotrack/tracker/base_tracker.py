# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import, division, print_function, unicode_literals

import cv2
import numpy as np
import torch

from nanotrack.core.config import cfg


class BaseTracker(object):
    def init(self, img, bbox):
        raise NotImplementedError

    def track(self, img):
        raise NotImplementedError


class SiameseTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0.0, -context_xmin))
        top_pad = int(max(0.0, -context_ymin))
        right_pad = int(max(0.0, context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0.0, context_ymax - im_sz[0] + 1))

        context_xmin += left_pad
        context_xmax += left_pad
        context_ymin += top_pad
        context_ymax += top_pad

        rows, cols, channels = im.shape
        if any((top_pad, bottom_pad, left_pad, right_pad)):
            padded = np.zeros((rows + top_pad + bottom_pad, cols + left_pad + right_pad, channels), np.uint8)
            padded[top_pad:top_pad + rows, left_pad:left_pad + cols, :] = im
            if top_pad:
                padded[:top_pad, left_pad:left_pad + cols, :] = avg_chans
            if bottom_pad:
                padded[rows + top_pad:, left_pad:left_pad + cols, :] = avg_chans
            if left_pad:
                padded[:, :left_pad, :] = avg_chans
            if right_pad:
                padded[:, cols + left_pad:, :] = avg_chans
            im_patch = padded[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        tensor = torch.from_numpy(im_patch)
        if cfg.CUDA:
            tensor = tensor.cuda()
        return tensor
