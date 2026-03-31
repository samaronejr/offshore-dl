"""MLP model — feedforward baseline for 3W classification.

Flattens (batch, window, n_vars) statistical features to (batch, window*n_vars)
and classifies through fully-connected layers with BatchNorm + GELU + Dropout.

Classification-only — no forecasting or anomaly heads (D001).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel

logger = logging.getLogger(__name__)


class MLPModel(BaseModel):
    """Feedforward MLP for 3W classification on statistical features.

    Architecture: Flatten → [Linear → BatchNorm1d → GELU → Dropout] × N → Linear.
    Input shape ``(batch, window, n_vars)`` is flattened internally to
    ``(batch, window * n_vars)`` before the FC layers.

    Args:
        task: Must be ``"classification"`` — other tasks raise ValueError.
        n_vars: Number of input variables (sensor columns).
        n_classes: Number of output classes.
        window_size: Input window size (used to compute input_dim).
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability after each hidden layer.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        n_classes: int = 10,
        window_size: int = 14,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs,
    ) -> None:
        if task != "classification":
            msg = f"MLPModel only supports classification, got {task!r}"
            raise ValueError(msg)

        super().__init__(task=task, n_vars=n_vars)
        self.n_classes = n_classes
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay

        if hidden_dims is None:
            hidden_dims = [256, 128]

        input_dim = window_size * n_vars

        # Build sequential: Flatten → [Linear → BN → GELU → Dropout] × N → Linear
        layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

        logger.info(
            "MLPModel: input_dim=%d, hidden=%s, n_classes=%d, dropout=%.2f",
            input_dim, hidden_dims, n_classes, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — flatten and classify.

        Args:
            x: Input tensor ``(batch, window, n_vars)``.

        Returns:
            Logits ``(batch, n_classes)``.
        """
        return self.network(x)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        """Create AdamW optimizer with configurable lr and weight decay.

        Args:
            cfg: OmegaConf config with training.lr and training.weight_decay.

        Returns:
            Configured AdamW optimizer.
        """
        lr = self.lr
        wd = self.weight_decay

        if cfg is not None:
            if hasattr(cfg, "model") and hasattr(cfg.model, "training"):
                lr = getattr(cfg.model.training, "lr", lr)
                wd = getattr(cfg.model.training, "weight_decay", wd)
            elif hasattr(cfg, "training"):
                lr = getattr(cfg.training, "lr", lr)
                wd = getattr(cfg.training, "weight_decay", wd)

        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
