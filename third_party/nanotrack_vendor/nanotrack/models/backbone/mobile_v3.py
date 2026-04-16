import math

import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    min_value = divisor if min_value is None else min_value
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.hard_sigmoid = nn.Hardsigmoid(inplace=inplace)

    def forward(self, x):
        return self.hard_sigmoid(x)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.hard_swish = nn.Hardswish(inplace=inplace)

    def forward(self, x):
        return self.hard_swish(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        hidden = _make_divisible(channel // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel),
            h_sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channels)
        y = self.fc(y).view(batch, channels, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), h_swish())


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super().__init__()
        assert stride in [1, 2]
        activation = h_swish if use_hs else nn.ReLU
        self.identity = stride == 1 and inp == oup
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                activation(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return x + self.conv(x) if self.identity else self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, width_mult=1.0, used_layers=None):
        super().__init__()
        del mode, used_layers
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        for kernel_size, expand, channels, use_se, use_hs, stride in cfgs:
            output_channel = _make_divisible(channels * width_mult, 8)
            exp_size = _make_divisible(input_channel * expand, 8)
            layers.append(InvertedResidual(input_channel, exp_size, output_channel, kernel_size, stride, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.features(x)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()


def mobilenetv3_small(**kwargs):
    cfgs = [
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, mode="small", **kwargs)


def mobilenetv3_small_v3(**kwargs):
    cfgs = [
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]
    return MobileNetV3(cfgs, mode="small", **kwargs)
