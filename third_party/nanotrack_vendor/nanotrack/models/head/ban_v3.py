import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BAN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


def xcorr_fast(x, kernel):
    batch = kernel.size(0)
    pk = kernel.view(-1, x.size(1), kernel.size(2), kernel.size(3))
    px = x.view(1, -1, x.size(2), x.size(3))
    po = F.conv2d(px, pk, groups=batch)
    return po.view(batch, -1, po.size(2), po.size(3))


def xcorr_depthwise(x, kernel):
    batch = kernel.size(0)
    channels = kernel.size(1)
    x = x.view(1, batch * channels, x.size(2), x.size(3))
    kernel = kernel.view(batch * channels, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, padding=1, groups=batch * channels)
    return out.view(batch, channels, out.size(2), out.size(3))


def xcorr_pixelwise(x, kernel):
    batch, channels, height, width = x.size()
    kernel_mat = kernel.view(batch, channels, -1).transpose(1, 2)
    x_mat = x.view(batch, channels, -1)
    return torch.matmul(kernel_mat, x_mat).view(batch, -1, height, width)


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


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_kernel = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels))
        self.conv_search = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels))

    def forward(self, kernel, search):
        return xcorr_depthwise(self.conv_search(search), self.conv_kernel(kernel))


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
        channels = 64
        self.ca_layer = CAModule(channels)
        self.conv_kernel = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels))
        self.conv_search = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1), nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, kernel, search):
        feature = xcorr_pixelwise(self.conv_search(search), self.conv_kernel(kernel))
        return self.conv(self.ca_layer(feature))


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=96, out_channels=96, weighted=False):
        super().__init__()
        del out_channels, weighted
        self.corr_dw_reg = DepthwiseXCorr(in_channels, in_channels)
        self.corr_pw_reg = PixelwiseXCorr(in_channels, in_channels)
        self.corr_dw_cls = DepthwiseXCorr(in_channels, in_channels)
        self.corr_pw_cls = PixelwiseXCorr(in_channels, in_channels)

        def make_tower():
            layers = []
            for _ in range(6):
                layers.extend(
                    [
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU6(inplace=True),
                        nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(in_channels),
                    ]
                )
            return nn.Sequential(*layers)

        self.cls_tower = make_tower()
        self.bbox_tower = make_tower()
        self.cls_logits = nn.Conv2d(in_channels, 2, kernel_size=1, stride=1, padding=0)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1, padding=0)
        self.down_reg = nn.Conv2d(in_channels + 64, in_channels, kernel_size=1, stride=1, padding=0)
        self.down_cls = nn.Conv2d(in_channels + 64, in_channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def crop(x):
        return x[:, :, 2:6, 2:6] if x.size(3) > 4 else x

    def forward(self, z_f, x_f):
        crop_z_f = self.crop(z_f)
        x_pw_reg = self.corr_pw_reg(z_f, x_f)
        x_pw_cls = self.corr_pw_cls(z_f, x_f)
        x_dw_reg = self.corr_dw_reg(crop_z_f, x_f)
        x_dw_cls = self.corr_dw_cls(crop_z_f, x_f)
        x_reg = self.down_reg(torch.cat((x_pw_reg, x_dw_reg), 1))
        x_cls = self.down_cls(torch.cat((x_pw_cls, x_dw_cls), 1))
        logits = self.cls_logits(self.cls_tower(x_cls))
        bbox_reg = torch.exp(self.bbox_pred(self.bbox_tower(x_reg)))
        return logits, bbox_reg
