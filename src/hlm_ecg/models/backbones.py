"""Backbone registry for reviewer-defense ECG experiments."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from hlm_ecg.models.resnet1d import ResNet1D


SUPPORTED_ARCHITECTURES = ("resnet1d_tiny", "xresnet1d101_like", "inception_time1d")


def _check_ecg_input(x: torch.Tensor) -> None:
    if x.ndim != 3 or x.shape[1] != 12:
        raise ValueError(f"Expected input shape (batch, 12, 1000), got {tuple(x.shape)}")


def _as_layers(value: object, default: Sequence[int]) -> tuple[int, int, int, int]:
    if value is None:
        return tuple(int(x) for x in default)  # type: ignore[return-value]
    layers = tuple(int(x) for x in value)  # type: ignore[arg-type]
    if len(layers) != 4:
        raise ValueError("layers must contain exactly four integers")
    return layers  # type: ignore[return-value]


class XResNetBottleneck1D(nn.Module):
    """Bottleneck residual block used by the local XResNet1D-like backbone."""

    expansion = 4

    def __init__(self, in_channels: int, planes: int, *, stride: int = 1, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        out_channels = int(planes) * self.expansion
        self.conv1 = nn.Conv1d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class XResNet1DLike(nn.Module):
    """Local XResNet1D-101-like supervised backbone.

    The default depth follows the 3/4/23/3 bottleneck layout used by ResNet-101,
    adapted to 1D ECG signals. Configs may use shallower `layers` for smoke tests.
    """

    def __init__(
        self,
        *,
        in_channels: int = 12,
        num_classes: int = 5,
        base_channels: int = 32,
        layers: tuple[int, int, int, int] = (3, 4, 23, 3),
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.in_channels = base_channels
        planes = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.layer1 = self._make_layer(planes[0], layers[0], stride=1, kernel_size=kernel_size)
        self.layer2 = self._make_layer(planes[1], layers[1], stride=2, kernel_size=kernel_size)
        self.layer3 = self._make_layer(planes[2], layers[2], stride=2, kernel_size=kernel_size)
        self.layer4 = self._make_layer(planes[3], layers[3], stride=2, kernel_size=kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = planes[3] * XResNetBottleneck1D.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, planes: int, blocks: int, *, stride: int, kernel_size: int) -> nn.Sequential:
        layers = [XResNetBottleneck1D(self.in_channels, planes, stride=stride, kernel_size=kernel_size)]
        self.in_channels = int(planes) * XResNetBottleneck1D.expansion
        for _ in range(1, int(blocks)):
            layers.append(XResNetBottleneck1D(self.in_channels, planes, stride=1, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        _check_ecg_input(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encode_features(x))


class InceptionModule1D(nn.Module):
    """InceptionTime-style 1D convolution module."""

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int,
        bottleneck_channels: int,
        kernel_sizes: Sequence[int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        if any(int(k) % 2 == 0 for k in kernel_sizes):
            raise ValueError("Inception kernel sizes must be odd")
        reduced_channels = int(bottleneck_channels)
        self.bottleneck = (
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )
        conv_input = reduced_channels if in_channels > 1 else in_channels
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(conv_input, out_channels, kernel_size=int(k), padding=int(k) // 2, bias=False)
                for k in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU(inplace=True)

    @property
    def out_channels(self) -> int:
        return int(self.bn.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        branches = [conv(z) for conv in self.branches]
        branches.append(self.pool_branch(x))
        return self.relu(self.bn(torch.cat(branches, dim=1)))


class InceptionTime1D(nn.Module):
    """Local InceptionTime-style supervised backbone for 12-lead ECG."""

    def __init__(
        self,
        *,
        in_channels: int = 12,
        num_classes: int = 5,
        base_channels: int = 32,
        inception_depth: int = 6,
        inception_bottleneck_channels: int = 32,
        kernel_sizes: Sequence[int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        if int(inception_depth) <= 0:
            raise ValueError("inception_depth must be positive")
        modules: list[nn.Module] = []
        current_channels = int(in_channels)
        for _ in range(int(inception_depth)):
            module = InceptionModule1D(
                current_channels,
                out_channels=int(base_channels),
                bottleneck_channels=int(inception_bottleneck_channels),
                kernel_sizes=kernel_sizes,
            )
            modules.append(module)
            current_channels = module.out_channels
        self.blocks = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = current_channels
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        _check_ecg_input(x)
        return self.pool(self.blocks(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encode_features(x))


def build_feature_backbone(model_cfg: Mapping[str, object]) -> nn.Module:
    """Build a supported backbone with a stable feature-encoder interface."""

    architecture = str(model_cfg.get("architecture", "resnet1d_tiny"))
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported model.architecture={architecture!r}; expected one of {SUPPORTED_ARCHITECTURES}")
    common = {
        "in_channels": int(model_cfg.get("in_channels", 12)),
        "num_classes": int(model_cfg.get("num_classes", 5)),
        "base_channels": int(model_cfg.get("base_channels", 32)),
    }
    if architecture == "resnet1d_tiny":
        return ResNet1D(
            **common,
            layers=_as_layers(model_cfg.get("layers"), (1, 1, 1, 1)),
            kernel_size=int(model_cfg.get("kernel_size", 7)),
        )
    if architecture == "xresnet1d101_like":
        return XResNet1DLike(
            **common,
            layers=_as_layers(model_cfg.get("layers"), (3, 4, 23, 3)),
            kernel_size=int(model_cfg.get("kernel_size", 7)),
        )
    return InceptionTime1D(
        **common,
        inception_depth=int(model_cfg.get("inception_depth", 6)),
        inception_bottleneck_channels=int(model_cfg.get("inception_bottleneck_channels", common["base_channels"])),
    )
