"""PatchTST — Transformer foundation model for time series.

Wraps HuggingFace's ``PatchTSTForPrediction`` and ``PatchTSTForClassification``
behind the BaseModel interface. Uses the PatchTST architecture from
``transformers`` — a channel-independent patch-based transformer.

Tasks:
- Classification (3W): PatchTSTForClassification → class logits
- Forecasting (Ganymede): PatchTSTForPrediction → horizon predictions
- Anomaly (CDF): PatchTSTForPrediction in reconstruction mode

Reference: Nie et al. (2023) "A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers."
"""

from __future__ import annotations

import torch
from transformers import (
    PatchTSTConfig,
    PatchTSTForClassification,
    PatchTSTForPrediction,
)

from offshore_dl.models.base import BaseModel


class PatchTSTModel(BaseModel):
    """PatchTST wrapper for all 3 offshore monitoring tasks.

    Args:
        task: ``"classification"``, ``"forecasting"``, or ``"anomaly"``.
        n_vars: Number of input variables (sensor channels).
        patch_len: Patch length in timesteps.
        stride: Patch stride.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feedforward dimension.
        dropout: Dropout rate.
        n_classes: Number of output classes (classification only).
        horizon: Forecast horizon (forecasting only).
        window_size: Input window size (all tasks — context length).
        lr: Learning rate.
        weight_decay: Weight decay.
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        n_classes: int = 10,
        horizon: int = 30,
        window_size: int = 48,
        lr: float = 0.0001,
        weight_decay: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(
            task=task,
            n_vars=n_vars,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )
        self.n_classes = n_classes
        self.horizon = horizon
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.target_channel = kwargs.get("target_channel", 0)

        patch_len = int(patch_len)
        stride = int(stride)
        window_size = int(window_size)
        if window_size <= 0:
            msg = f"window_size must be positive for PatchTST, got {window_size}."
            raise ValueError(msg)
        if patch_len <= 0:
            msg = f"patch_len must be positive for PatchTST, got {patch_len}."
            raise ValueError(msg)
        if stride <= 0:
            msg = f"stride must be positive for PatchTST, got {stride}."
            raise ValueError(msg)
        if patch_len > window_size:
            msg = (
                f"PatchTST patch_len must be <= window_size so at least one patch "
                f"can be extracted, got patch_len={patch_len}, window_size={window_size}."
            )
            raise ValueError(msg)
        n_patches = ((window_size - patch_len) // stride) + 1
        if n_patches < 1:
            msg = (
                f"PatchTST configuration yields no valid patches: "
                f"window_size={window_size}, patch_len={patch_len}, stride={stride}."
            )
            raise ValueError(msg)

        # Build HF PatchTST config
        hf_config = PatchTSTConfig(
            num_input_channels=n_vars,
            context_length=window_size,
            patch_length=patch_len,
            patch_stride=stride,
            d_model=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            ffn_dim=d_ff,
            dropout=dropout,
            head_dropout=dropout,
            # Task-specific
            prediction_length=horizon if task in ("forecasting", "anomaly") else 0,
            num_targets=n_classes if task == "classification" else n_vars,
            # Disable distribution head — we want point predictions
            loss="mse",
        )

        if task == "classification":
            self.backbone = PatchTSTForClassification(hf_config)
        elif task in ("forecasting", "anomaly"):
            # For anomaly: predict the same window length for reconstruction
            if task == "anomaly":
                hf_config.prediction_length = window_size
            self.backbone = PatchTSTForPrediction(hf_config)
        else:
            msg = f"Unknown task: {task!r}"
            raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HF PatchTST.

        Args:
            x: Input tensor ``(batch, window, n_vars)``.

        Returns:
            - classification: ``(batch, n_classes)`` logits
            - forecasting: ``(batch, horizon)`` predictions (target channel)
            - anomaly: ``(batch, window, n_vars)`` reconstruction
        """
        # HF PatchTST expects (batch, seq_len, channels) — same as our format
        if self.task == "classification":
            out = self.backbone(past_values=x)
            return out.prediction_logits  # (batch, n_classes)

        elif self.task == "forecasting":
            out = self.backbone(past_values=x)
            # prediction_outputs: (batch, horizon, n_vars)
            # Return first channel (target) as (batch, horizon)
            return out.prediction_outputs[:, :, self.target_channel]

        elif self.task == "anomaly":
            out = self.backbone(past_values=x)
            # prediction_outputs: (batch, window, n_vars)
            return out.prediction_outputs

        msg = f"Unknown task: {self.task!r}"
        raise ValueError(msg)


