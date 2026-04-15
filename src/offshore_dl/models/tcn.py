"""TCN model — temporal convolutional network for forecasting.

Dilated causal convolutions with exponential dilation to cover full
input windows. Used for Ganymede gas production forecasting.

Architecture:
  Input (batch, window, n_vars) → Conv1d blocks with residual connections
  → Global average pooling → Linear → (batch, horizon)

The receptive field grows exponentially with dilation:
  RF = 1 + (kernel_size - 1) * sum(d_i) where d_i = 2^i
"""

from __future__ import annotations

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel


class _TCNBlock(nn.Module):
    """Single TCN residual block with dilated causal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Causal padding: pad only on the left
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # 1x1 residual connection if channel dims differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x is (batch, channels, seq_len)."""
        residual = self.residual(x)

        # First conv + causal padding
        out = nn.functional.pad(x, (self.pad, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second conv + causal padding
        out = nn.functional.pad(out, (self.pad, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out + residual


class TCNModel(BaseModel):
    """Temporal Convolutional Network for time series forecasting.

    Uses exponentially dilated causal convolutions to capture temporal
    patterns at multiple scales. Supports forecasting task only.

    Args:
        task: Must be ``"forecasting"``.
        n_vars: Number of input variables (sensor columns).
        n_channels: Hidden channel width for TCN blocks.
        n_layers: Number of TCN blocks (dilation doubles each layer).
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        horizon: Forecast horizon length.
        window_size: Input window size (for receptive field info).
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
    """

    def __init__(
        self,
        task: str = "forecasting",
        n_vars: int = 63,
        n_channels: int = 128,
        n_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        horizon: int = 30,
        window_size: int = 90,
        n_classes: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        **kwargs,
    ) -> None:
        super().__init__(task=task, n_vars=n_vars)

        if task != "forecasting":
            raise ValueError(f"TCNModel only supports forecasting, got {task!r}")

        self.n_channels = n_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.horizon = horizon
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay

        # Build TCN blocks with exponential dilation
        blocks = []
        for i in range(n_layers):
            in_ch = n_vars if i == 0 else n_channels
            dilation = 2 ** i
            blocks.append(_TCNBlock(in_ch, n_channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)

        # Output head: global average pooling → linear → horizon
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, n_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_channels, horizon),
        )

        # Log receptive field
        rf = 1 + 2 * (kernel_size - 1) * (2 ** n_layers - 1)
        self._receptive_field = rf

    @property
    def receptive_field(self) -> int:
        """Total receptive field of the TCN in timesteps."""
        return self._receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor ``(batch, window, n_vars)``.

        Returns:
            ``(batch, horizon)`` predictions.
        """
        # Conv1d expects (batch, channels, seq_len)
        out = x.transpose(1, 2)  # (batch, n_vars, window)
        out = self.tcn(out)      # (batch, n_channels, window)
        out = self.head(out)     # (batch, horizon)
        return out

