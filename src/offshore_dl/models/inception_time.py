"""InceptionTime: Finding AlexNet for Time Series Classification.

Ismail Fawaz et al., Data Mining and Knowledge Discovery, 2020.
Ensemble of 5 Inception networks with multi-scale 1D convolutions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel


def _resolve_kernel_sizes(kernel_sizes: Sequence[int], window_size: int) -> list[int]:
    """Adapt default kernels to the window length.

    * Long raw windows (>=128 timesteps) → [40, 80, 160] (original paper).
    * Short feature windows (<128) → clamp each kernel to ``<= window_size``
      so the convolution spans at most the full sequence, preventing the
      model from collapsing to a majority-class predictor.
    """
    kernels = [int(k) for k in kernel_sizes]
    if kernels == [10, 20, 40] and window_size >= 128:
        return [40, 80, 160]
    return [min(k, max(1, window_size)) for k in kernels]


class InceptionBlock(nn.Module):
    """Single InceptionTime block with multi-scale convolutions."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_sizes: Sequence[int],
        bottleneck_channels: int,
    ) -> None:
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if in_channels > 1
            else nn.Identity()
        )
        branch_in = bottleneck_channels if in_channels > 1 else in_channels
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_in, n_filters, kernel_size=k, padding="same", bias=False
                )
                for k in kernel_sizes
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_x = self.bottleneck(x)
        outputs = [branch(bottleneck_x) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        return self.activation(self.norm(torch.cat(outputs, dim=1)))


class ResidualShortcut(nn.Module):
    """Channel-matching residual projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.proj(residual))


class InceptionTimeModel(BaseModel):
    """InceptionTime classifier for 3W feature or raw windows."""

    def __init__(
        self,
        task: str = "classification",
        n_vars: int = 27,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        n_filters: int = 32,
        kernel_sizes: Sequence[int] = (10, 20, 40),
        depth: int = 6,
        n_classes: int = 10,
        window_size: int = 14,
        lr: float = 1.0e-3,
        weight_decay: float = 0.0,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            task=task,
            n_vars=n_vars,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
        )
        if task != "classification":
            msg = "InceptionTimeModel currently supports classification only."
            raise ValueError(msg)

        self.n_classes = n_classes
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.kernel_sizes = _resolve_kernel_sizes(kernel_sizes, window_size)
        bottleneck_channels = max(1, n_filters)

        blocks: list[nn.Module] = []
        shortcuts: list[nn.Module] = []
        in_channels = n_vars
        out_channels = n_filters * (len(self.kernel_sizes) + 1)

        for block_idx in range(depth):
            blocks.append(
                InceptionBlock(
                    in_channels=in_channels,
                    n_filters=n_filters,
                    kernel_sizes=self.kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                )
            )
            if block_idx % 3 == 2:
                residual_in = n_vars if block_idx == 2 else out_channels
                shortcuts.append(ResidualShortcut(residual_in, out_channels))
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from ``(batch, window, n_vars)`` input."""
        x = x.transpose(1, 2)
        residual = x
        shortcut_idx = 0

        for block_idx, block in enumerate(self.blocks):
            x = block(x)
            if block_idx % 3 == 2:
                x = self.shortcuts[shortcut_idx](x, residual)
                residual = x
                shortcut_idx += 1

        pooled = self.pool(x).squeeze(-1)
        return self.classifier(pooled)

