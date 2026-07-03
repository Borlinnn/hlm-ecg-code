"""Lightweight ResNet1D baseline for PTB-XL full-lead ECG diagnosis."""

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
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
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class ResNet1D(nn.Module):
    """ResNet1D-Wang-style supervised baseline without HLM components."""

    def __init__(
        self,
        *,
        in_channels: int = 12,
        num_classes: int = 5,
        base_channels: int = 32,
        layers: tuple[int, int, int, int] = (1, 1, 1, 1),
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.in_channels = base_channels
        self.layer1 = self._make_layer(channels[0], layers[0], stride=1, kernel_size=kernel_size)
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2, kernel_size=kernel_size)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2, kernel_size=kernel_size)
        self.layer4 = self._make_layer(channels[3], layers[3], stride=2, kernel_size=kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = channels[3]
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, out_channels: int, blocks: int, *, stride: int, kernel_size: int) -> nn.Sequential:
        layers = [BasicBlock1D(self.in_channels, out_channels, stride=stride, kernel_size=kernel_size)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels, stride=1, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError(f"Expected input shape (batch, 12, 1000), got {tuple(x.shape)}")
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.encode_features(x))
