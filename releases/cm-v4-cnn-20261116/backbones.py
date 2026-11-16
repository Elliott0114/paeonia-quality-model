"""Unmodified architecture definitions from the archived four-backbone trainer. No old splitting or trained weights."""
import torch
from torch import nn

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Fire1d(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int, expand_channels: int) -> None:
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv1d(in_channels, squeeze_channels, kernel_size=1),
            nn.GELU(),
        )
        self.expand1 = nn.Conv1d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3 = nn.Conv1d(squeeze_channels, expand_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)

class ChannelShuffle1d(nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        groups = self.groups
        x = x.reshape(batch_size, groups, channels // groups, length)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, channels, length)

class ShuffleBlock1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            ChannelShuffle1d(groups=4),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale

class MBConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            SEBlock1d(hidden),
            nn.Conv1d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.use_residual else out

class SpectralBackbone1D(nn.Module):
    def __init__(self, features: nn.Module, out_dim: int, head_channels: int) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, head_channels, kernel_size=7, padding=3)
        self.features = features
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_channels, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

def build_1d_backbone_model(backbone_name: str, out_dim: int) -> nn.Module:
    name = backbone_name.lower()
    if name == "mobilenet1d":
        head_channels = 32
        features = nn.Sequential(
            nn.BatchNorm1d(head_channels),
            nn.GELU(),
            DepthwiseSeparableConv1d(head_channels, 48, 5, stride=2),
            DepthwiseSeparableConv1d(48, 64, 5, stride=2),
            DepthwiseSeparableConv1d(64, 96, 3, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(96, head_channels, kernel_size=1),
        )
        return SpectralBackbone1D(features, out_dim, head_channels)
    if name == "squeezenet1d":
        head_channels = 64
        features = nn.Sequential(
            nn.BatchNorm1d(head_channels),
            nn.GELU(),
            Fire1d(head_channels, 16, 32),
            nn.MaxPool1d(2),
            Fire1d(64, 16, 32),
            Fire1d(64, 24, 48),
            nn.MaxPool1d(2),
            Fire1d(96, 24, 48),
            nn.Conv1d(96, head_channels, kernel_size=1),
            nn.GELU(),
        )
        return SpectralBackbone1D(features, out_dim, head_channels)
    if name == "shufflenet1d":
        head_channels = 64
        features = nn.Sequential(
            nn.BatchNorm1d(head_channels),
            nn.GELU(),
            nn.Conv1d(head_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ShuffleBlock1d(64),
            nn.MaxPool1d(2),
            ShuffleBlock1d(64),
            nn.MaxPool1d(2),
            nn.Conv1d(64, head_channels, kernel_size=1),
        )
        return SpectralBackbone1D(features, out_dim, head_channels)
    if name == "efficientnet1d":
        head_channels = 48
        features = nn.Sequential(
            nn.BatchNorm1d(head_channels),
            nn.GELU(),
            MBConv1d(head_channels, 48),
            nn.MaxPool1d(2),
            MBConv1d(48, 64),
            nn.MaxPool1d(2),
            MBConv1d(64, 96),
            nn.Conv1d(96, head_channels, kernel_size=1),
            nn.GELU(),
        )
        return SpectralBackbone1D(features, out_dim, head_channels)
    raise ValueError(f"不支持的 backbone: {backbone_name}")
