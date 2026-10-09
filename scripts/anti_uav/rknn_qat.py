"""Lightweight RKNN-oriented INT8 fake quantization for YOLO convolution layers."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Use quantized values in the forward pass and identity gradients backward."""
    return original + (quantized - original).detach()


def symmetric_weight_fake_quant(weight: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric INT8 fake quantization."""
    reduce_dims = tuple(range(1, weight.ndim))
    scale = weight.detach().float().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / 127.0
    quantized = (weight.float() / scale).round().clamp(-127, 127) * scale
    return _ste(weight, quantized.to(weight.dtype))


class RKNNFakeQuantConv2d(nn.Conv2d):
    """Conv2d with per-channel INT8 weights and EMA-calibrated INT8 input activations."""

    def __init__(self, *args, observer_momentum: float = 0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer_momentum = observer_momentum
        self.qat_enabled = False
        self.observer_enabled = True
        self.register_buffer("activation_min", torch.tensor(0.0))
        self.register_buffer("activation_max", torch.tensor(0.0))
        self.register_buffer("observer_initialized", torch.tensor(False, dtype=torch.bool))

    @classmethod
    def from_conv(cls, source: nn.Conv2d) -> "RKNNFakeQuantConv2d":
        target = cls(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            source.padding,
            source.dilation,
            source.groups,
            source.bias is not None,
            source.padding_mode,
        ).to(device=source.weight.device, dtype=source.weight.dtype)
        target.weight.data.copy_(source.weight.data)
        if source.bias is not None:
            target.bias.data.copy_(source.bias.data)
        target.train(source.training)
        return target

    def _fake_quant_activation(self, value: torch.Tensor) -> torch.Tensor:
        current_min = value.detach().float().amin()
        current_max = value.detach().float().amax()
        if self.training and self.observer_enabled:
            if not bool(self.observer_initialized):
                self.activation_min.copy_(current_min)
                self.activation_max.copy_(current_max)
                self.observer_initialized.fill_(True)
            else:
                momentum = self.observer_momentum
                self.activation_min.mul_(momentum).add_(current_min * (1.0 - momentum))
                self.activation_max.mul_(momentum).add_(current_max * (1.0 - momentum))
        if not bool(self.observer_initialized):
            minimum, maximum = current_min, current_max
        else:
            minimum, maximum = self.activation_min, self.activation_max
        scale = ((maximum - minimum) / 255.0).clamp_min(1e-8)
        zero_point = (-128.0 - minimum / scale).round().clamp(-128, 127)
        quantized = ((value.float() / scale) + zero_point).round().clamp(-128, 127)
        dequantized = (quantized - zero_point) * scale
        return _ste(value, dequantized.to(value.dtype))

    def quantized_weight(self) -> torch.Tensor:
        return symmetric_weight_fake_quant(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.qat_enabled:
            return super().forward(value)
        value = self._fake_quant_activation(value)
        return F.conv2d(
            value,
            self.quantized_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def prepare_rknn_qat(module: nn.Module) -> int:
    """Replace every Conv2d in-place and return the number of wrapped layers."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, RKNNFakeQuantConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, RKNNFakeQuantConv2d.from_conv(child))
            count += 1
        else:
            count += prepare_rknn_qat(child)
    return count


def set_rknn_qat(module: nn.Module, enabled: bool, observer_enabled: bool | None = None) -> int:
    """Toggle fake quantization (and optionally observers) on all wrapped layers."""
    count = 0
    for child in module.modules():
        if isinstance(child, RKNNFakeQuantConv2d):
            child.qat_enabled = enabled
            if observer_enabled is not None:
                child.observer_enabled = observer_enabled
            count += 1
    return count


def sync_rknn_qat_observer_flags(source: nn.Module, target: nn.Module) -> int:
    """Copy non-floating observer state that Ultralytics EMA intentionally skips."""
    source_modules = dict(source.named_modules())
    count = 0
    for name, child in target.named_modules():
        if isinstance(child, RKNNFakeQuantConv2d):
            source_child = source_modules.get(name)
            if not isinstance(source_child, RKNNFakeQuantConv2d):
                raise RuntimeError(f"QAT source/EMA module mismatch at {name}")
            child.observer_initialized.copy_(source_child.observer_initialized)
            count += 1
    return count


def strip_rknn_qat(module: nn.Module, bake_weights: bool = True) -> int:
    """Replace QAT wrappers with ordinary Conv2d layers for portable export."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, RKNNFakeQuantConv2d):
            target = nn.Conv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                child.bias is not None,
                child.padding_mode,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            weight = child.quantized_weight() if bake_weights else child.weight
            target.weight.data.copy_(weight.detach())
            if child.bias is not None:
                target.bias.data.copy_(child.bias.data)
            target.train(child.training)
            setattr(module, name, target)
            count += 1
        else:
            count += strip_rknn_qat(child, bake_weights)
    return count


def qat_copy(module: nn.Module, bake_weights: bool = True) -> nn.Module:
    result = copy.deepcopy(module).float()
    strip_rknn_qat(result, bake_weights=bake_weights)
    return result
