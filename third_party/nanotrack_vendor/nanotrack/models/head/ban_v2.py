import math

import torch
import torch.nn as nn

from nanotrack.core.xcorr import xcorr_fast, xcorr_pixelwise


class BAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super().__init__()
        cls_output = cls_out_channels
        loc_output = 4
        self.template_cls_conv = nn.Conv2d(feature_in, feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, feature_in * loc_output, kernel_size=3)
        self.search_cls_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)
        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)
        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class CAModule(nn.Module):
    def __init__(self, channels=64, reduction=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return residual * x


class PixelwiseXCorr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ca_layer = CAModule(channels=64)
        self.conv_kernel = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1), nn.BatchNorm2d(in_channels))
        self.conv_search = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1), nn.BatchNorm2d(in_channels))
        for modules in (self.conv_kernel, self.conv_search):
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_pixelwise(search, kernel)
        return self.ca_layer(feature)


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=64, out_channels=64, weighted=False):
        super().__init__()
        del out_channels, weighted
        self.corr_pw_reg = PixelwiseXCorr(48, 48)
        self.corr_pw_cls = PixelwiseXCorr(48, 48)

        cls_tower, bbox_tower = [], []
        for tower in (cls_tower, bbox_tower):
            tower.extend(
                [
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
                    nn.Conv2d(64, 96, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(96),
                    nn.ReLU6(inplace=True),
                ]
            )
            for _ in range(5):
                tower.extend(
                    [
                        nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=False),
                        nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(96),
                        nn.ReLU6(inplace=True),
                    ]
                )

        self.cls_pw_tower = nn.Sequential(*cls_tower)
        self.bbox_pw_tower = nn.Sequential(*bbox_tower)
        self.cls_pred = nn.Conv2d(96, 2, kernel_size=1, stride=1, padding=0)
        self.bbox_pred = nn.Conv2d(96, 4, kernel_size=1, stride=1, padding=0)

        for modules in (self.cls_pw_tower, self.bbox_pw_tower, self.cls_pred, self.bbox_pred, self.corr_pw_cls, self.corr_pw_reg):
            for m in modules.modules() if isinstance(modules, nn.Module) else []:
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, z_f, x_f):
        pw_reg = self.corr_pw_reg(z_f, x_f)
        pw_cls = self.corr_pw_cls(z_f, x_f)
        logits = self.cls_pred(self.cls_pw_tower(pw_cls))
        bbox_reg = torch.exp(self.bbox_pred(self.bbox_pw_tower(pw_reg)))
        return logits, bbox_reg
