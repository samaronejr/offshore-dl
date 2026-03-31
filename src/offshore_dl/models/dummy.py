"""Dummy model for testing the training pipeline.

Minimal architecture (single linear layer) that proves the BaseModel
interface and training engine work for all 3 task types without
needing real model implementations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel


class DummyModel(BaseModel):
    """Dummy model with a single linear layer.

    Used exclusively for testing the training pipeline. Produces valid
    output shapes for classification, forecasting, and anomaly tasks.

    Args:
        task: ``"classification"``, ``"forecasting"``, or ``"anomaly"``.
        n_vars: Number of input variables.
        n_classes: Number of output classes (classification only).
        horizon: Forecast horizon (forecasting only).
        window_size: Input window size (anomaly reconstruction only).
        lr: Learning rate.
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        n_classes: int = 10,
        horizon: int = 30,
        window_size: int = 48,
        lr: float = 0.01,
    ) -> None:
        super().__init__(task=task, n_vars=n_vars)
        self.n_classes = n_classes
        self.horizon = horizon
        self.window_size = window_size
        self.lr = lr

        # Output dimension depends on task
        if task == "classification":
            output_dim = n_classes
        elif task == "forecasting":
            output_dim = horizon
        elif task == "anomaly":
            output_dim = window_size * n_vars
        else:
            msg = f"Unknown task: {task!r}"
            raise ValueError(msg)

        self.linear = nn.Linear(n_vars, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: global average pooling → linear → reshape.

        Args:
            x: ``(batch, window, n_vars)``

        Returns:
            Task-dependent output shape.
        """
        # Global average pooling over time dimension
        pooled = x.mean(dim=1)  # (batch, n_vars)
        out = self.linear(pooled)  # (batch, output_dim)

        if self.task == "anomaly":
            # Reshape to reconstruction: (batch, window, n_vars)
            out = out.view(-1, self.window_size, self.n_vars)

        return out

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        """Create Adam optimizer."""
        lr = self.lr
        if cfg is not None and hasattr(cfg, "training") and hasattr(cfg.training, "lr"):
            lr = cfg.training.lr
        return torch.optim.Adam(self.parameters(), lr=lr)
