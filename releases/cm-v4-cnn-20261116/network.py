"""The legacy custom EfficientNet1D architecture, with a six-target head.

No legacy trained weights, splitting, filtering, or evaluation code is reused.
The three class bodies are verified against the archived implementation by AST.
"""
import torch
from torch import nn


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


def build_model(out_dim=6):
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
