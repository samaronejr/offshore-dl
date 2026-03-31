"""LSTM model — recurrent baseline for all 3 experimental tracks.

Multi-layer LSTM encoder with task-specific output heads:
- Classification (3W): final hidden state → linear → n_classes logits
- Forecasting (Ganymede): final hidden state → linear → horizon values
- Anomaly (CDF): full sequence output → linear → (window, n_vars) reconstruction
"""

from __future__ import annotations

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel


class LSTMModel(BaseModel):
    """Multi-layer LSTM with task-specific output heads.

    Architecture shared across 3W classification, Ganymede forecasting,
    and CDF anomaly detection. The LSTM encoder is identical; only the
    output projection differs by task.

    For classification, uses attention pooling over all timesteps
    instead of just the final hidden state.

    Args:
        task: ``"classification"``, ``"forecasting"``, or ``"anomaly"``.
        n_vars: Number of input variables (sensor columns).
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout between LSTM layers (0 if num_layers==1).
        bidirectional: Use bidirectional LSTM.
        n_classes: Number of output classes (classification only).
        horizon: Forecast horizon length (forecasting only).
        window_size: Input window size (anomaly reconstruction only).
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        n_classes: int = 10,
        horizon: int = 30,
        window_size: int = 48,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
    ) -> None:
        super().__init__(task=task, n_vars=n_vars)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        self.horizon = horizon
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay

        # Direction multiplier
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_size * self.num_directions

        # ── Shared LSTM encoder ──
        # dropout only applies between layers, so 0 when num_layers==1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ── Task-specific output heads ──
        if task == "classification":
            # Attention pooling over time: learn which timesteps matter
            self.attn_w = nn.Linear(self.hidden_dim, 1)
            self.head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim // 2, n_classes),
            )
        elif task == "forecasting":
            self.head = nn.Linear(self.hidden_dim, horizon)
            self.attn_w = None
        elif task == "anomaly":
            # Per-timestep reconstruction: sequence output → n_vars
            self.head = nn.Linear(self.hidden_dim, n_vars)
            self.attn_w = None
        else:
            msg = f"Unknown task: {task!r}"
            raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM encoder + task head.

        Args:
            x: Input tensor ``(batch, window, n_vars)``.

        Returns:
            - classification: ``(batch, n_classes)`` logits
            - forecasting: ``(batch, horizon)`` predictions
            - anomaly: ``(batch, window, n_vars)`` reconstruction
        """
        # LSTM encoder: output (batch, seq_len, hidden_dim), (h_n, c_n)
        output, (h_n, _c_n) = self.lstm(x)
        output = self.layer_norm(output)

        if self.task == "classification":
            # Attention pooling: weighted sum over all timesteps
            # output: (batch, seq_len, hidden_dim)
            attn_scores = self.attn_w(output).squeeze(-1)  # (batch, seq_len)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, seq_len)
            hidden = torch.bmm(
                attn_weights.unsqueeze(1), output,
            ).squeeze(1)  # (batch, hidden_dim)
            hidden = self.dropout(hidden)
            return self.head(hidden)  # (batch, n_classes)

        elif self.task == "forecasting":
            # Use final hidden state for forecasting
            if self.bidirectional:
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                hidden = torch.cat([h_forward, h_backward], dim=-1)
            else:
                hidden = h_n[-1]
            hidden = self.dropout(hidden)
            return self.head(hidden)  # (batch, horizon)

        elif self.task == "anomaly":
            # Use full sequence output for per-timestep reconstruction
            output = self.dropout(output)  # (batch, seq_len, hidden_dim)
            return self.head(output)  # (batch, seq_len, n_vars)

        msg = f"Unknown task: {self.task!r}"
        raise ValueError(msg)

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
