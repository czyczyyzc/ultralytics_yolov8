import torch.nn as nn


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.downsample(x)
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for idx in range(self.num):
                self.add_module(f"downsample{idx+2}", AdjustLayer(in_channels[idx], out_channels[idx]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        return [getattr(self, f"downsample{idx+2}")(feat) for idx, feat in enumerate(features)]
