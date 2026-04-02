"""Time-Dependent DeepONet — neural operator for offshore monitoring tasks.

DeepONet decomposes operator learning into:
- **Branch network**: encodes the input function (sensor time series)
- **Trunk network**: encodes query/output locations (time positions or class indices)
- **Output**: bilinear combination of branch and trunk embeddings

Adapted for 3 tasks:
- Classification (3W): Branch encodes window → class logits via linear head
- Forecasting (Ganymede): Branch encodes window, trunk encodes horizon positions → predictions
- Anomaly (CDF): Branch encodes window, trunk encodes reconstruction positions → per-element values

Reference: Lu et al. (2021) "Learning nonlinear operators via DeepONet."
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str = "gelu",
    dropout: float = 0.1,
) -> nn.Sequential:
    """Build an MLP with configurable hidden layers and activation."""
    act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, h_dim),
            act_fn(),
            nn.Dropout(dropout),
        ])
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class SensorConv1dBranch(nn.Module):
    """1D-CNN branch that processes the (14, 27) feature matrix structure.

    Treats 14 descriptors as input channels and convolves across
    the 27-sensor axis.  This preserves the spatial sensor layout
    that a flat MLP would destroy.

    Input:  ``(batch, 14, 27)`` — already ``(batch, channels, seq_len)``
    Output: ``(batch, rank)``
    """

    def __init__(
        self,
        n_features: int = 14,
        n_sensors: int = 27,
        rank: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 3 conv layers with small kernels (27 sensors is short)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — input already in Conv1d layout.

        Args:
            x: ``(batch, n_features, n_sensors)``  e.g. ``(B, 14, 27)``

        Returns:
            Branch embedding ``(batch, rank)``
        """
        # x is already (batch, channels=n_features, seq_len=n_sensors)
        x = self.conv(x)       # → (batch, 128, n_sensors)
        x = self.pool(x)       # → (batch, 128, 1)
        x = x.squeeze(-1)      # → (batch, 128)
        return self.proj(x)     # → (batch, rank)


class SensorAttentionBranch(nn.Module):
    """Transformer-based branch treating each sensor as a token.

    Transposes the ``(14, 27)`` feature matrix so that each of the
    27 sensors becomes a token with a 14-dimensional embedding,
    projects to ``d_model``, adds learnable positional encoding,
    and applies self-attention to capture inter-sensor relationships.

    Input:  ``(batch, 14, 27)``
    Output: ``(batch, rank)``
    """

    def __init__(
        self,
        n_features: int = 14,
        n_sensors: int = 27,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        rank: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors

        # Project 14-dim feature vectors to d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Learnable positional encoding for 27 sensor positions
        self.pos_enc = nn.Parameter(torch.randn(1, n_sensors, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.proj = nn.Linear(d_model, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — transpose to sensor-token format.

        Args:
            x: ``(batch, n_features, n_sensors)``  e.g. ``(B, 14, 27)``

        Returns:
            Branch embedding ``(batch, rank)``
        """
        # Transpose: (batch, 14, 27) → (batch, 27, 14)  (27 tokens of 14 dims)
        x = x.transpose(1, 2)
        x = self.input_proj(x)          # → (batch, 27, d_model)
        x = x + self.pos_enc            # add positional encoding
        x = self.encoder(x)             # → (batch, 27, d_model)
        x = x.mean(dim=1)              # mean pool over 27 tokens → (batch, d_model)
        return self.proj(x)             # → (batch, rank)


class CNNBranch(nn.Module):
    """1D-CNN branch for processing time series in DeepONet.

    Replaces the flat MLP branch to preserve temporal structure.
    Processes ``(batch, window, n_vars)`` via Conv1d layers followed
    by adaptive average pooling → projection to rank dim.
    """

    def __init__(
        self,
        n_vars: int,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        rank: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256]
        kernel_sizes = kernel_sizes or [7, 5, 3]

        layers: list[nn.Module] = []
        in_ch = n_vars
        for out_ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels[-1], rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process time series through CNN branch.

        Args:
            x: ``(batch, window, n_vars)``

        Returns:
            Branch embedding ``(batch, rank)``
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # → (batch, n_vars, window)
        x = self.conv(x)       # → (batch, channels[-1], window)
        x = self.pool(x)       # → (batch, channels[-1], 1)
        x = x.squeeze(-1)      # → (batch, channels[-1])
        return self.proj(x)     # → (batch, rank)


class DeepONetModel(BaseModel):
    """Time-Dependent DeepONet for offshore monitoring tasks.

    The branch network processes the flattened sensor window into a
    rank-dimensional embedding. For forecasting and anomaly tasks,
    the trunk network processes query positions into a rank-dimensional
    embedding, and the output is their inner product. For classification,
    a linear head replaces the trunk.

    Args:
        task: ``"classification"``, ``"forecasting"``, or ``"anomaly"``.
        n_vars: Number of input variables (sensor columns).
        branch_hidden: Hidden layer sizes for branch MLP.
        trunk_hidden: Hidden layer sizes for trunk MLP.
        rank: Embedding dimension for branch/trunk inner product.
        activation: Activation function name.
        dropout: Dropout rate.
        n_classes: Number of output classes (classification only).
        horizon: Forecast horizon length (forecasting only).
        window_size: Input window size (anomaly/reconstruction only).
        branch_type: Branch architecture for short windows (≤30):
            ``"mlp"`` (flat MLP, default), ``"conv1d"`` (sensor Conv1d),
            or ``"attention"`` (sensor Transformer).  Ignored when
            ``window_size > 30`` (always uses CNNBranch).
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        loss_type: Loss function for classification — ``"ce"`` or ``"focal"``.
        focal_gamma: Focusing exponent for focal loss (only used when
            ``loss_type="focal"``).
    """

    _VALID_BRANCH_TYPES = {"mlp", "conv1d", "attention"}

    def __init__(
        self,
        task: str,
        n_vars: int,
        branch_hidden: list[int] | None = None,
        trunk_hidden: list[int] | None = None,
        rank: int = 64,
        activation: str = "gelu",
        dropout: float = 0.1,
        n_classes: int = 10,
        horizon: int = 30,
        window_size: int = 48,
        branch_type: str = "mlp",
        lr: float = 0.0005,
        weight_decay: float = 0.0001,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__(task=task, n_vars=n_vars, loss_type=loss_type, focal_gamma=focal_gamma)
        self.rank = rank
        self.n_classes = n_classes
        self.horizon = horizon
        self.window_size = window_size
        self.branch_type = branch_type
        self.lr = lr
        self.weight_decay = weight_decay

        if branch_type not in self._VALID_BRANCH_TYPES:
            msg = (
                f"Unknown branch_type {branch_type!r}. "
                f"Must be one of {sorted(self._VALID_BRANCH_TYPES)}"
            )
            raise ValueError(msg)

        branch_hidden = branch_hidden or [128, 128]
        trunk_hidden = trunk_hidden or [128, 128]

        # ── Branch network ──
        # For short feature sequences (≤30), branch_type selects between:
        #   "mlp"       — flat MLP (original default)
        #   "conv1d"    — Conv1d across sensor axis, preserving structure
        #   "attention" — Transformer treating each sensor as a token
        # For raw temporal windows (>30), always use CNNBranch regardless
        # of branch_type — temporal conv is the right inductive bias.
        if window_size <= 30:
            if branch_type == "mlp":
                flat_input = window_size * n_vars
                self.branch = nn.Sequential(
                    nn.Flatten(),
                    _build_mlp(
                        flat_input,
                        [256, 256, 128],
                        rank,
                        activation=activation,
                        dropout=dropout,
                    ),
                )
            elif branch_type == "conv1d":
                self.branch = SensorConv1dBranch(
                    n_features=window_size,
                    n_sensors=n_vars,
                    rank=rank,
                    dropout=dropout,
                )
            elif branch_type == "attention":
                self.branch = SensorAttentionBranch(
                    n_features=window_size,
                    n_sensors=n_vars,
                    rank=rank,
                    dropout=dropout,
                )
        else:
            # CNN branch for temporal data — avoids flattening explosion
            self.branch = CNNBranch(
                n_vars=n_vars,
                channels=[64, 128, 256],
                kernel_sizes=[7, 5, 3],
                rank=rank,
                dropout=dropout,
            )

        # ── Task-specific output ──
        if task == "classification":
            # Deeper classification head with residual-style layers
            self.head = nn.Sequential(
                nn.Linear(rank, rank),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(rank, n_classes),
            )
            # Trunk not used for classification
            self.trunk = None
            self._n_queries = n_classes
        elif task == "forecasting":
            # Trunk encodes horizon query positions
            # Query input: normalized time position (1D)
            self.trunk = _build_mlp(
                1, trunk_hidden, rank,
                activation=activation, dropout=dropout,
            )
            # Bias term for each query position
            self.output_bias = nn.Parameter(torch.zeros(horizon))
            self._n_queries = horizon
            self.head = None
        elif task == "anomaly":
            # Trunk encodes (timestep, variable) reconstruction positions
            # Query input: 2D position (normalized_t, normalized_v)
            self.trunk = _build_mlp(
                2, trunk_hidden, rank,
                activation=activation, dropout=dropout,
            )
            self.output_bias = nn.Parameter(torch.zeros(window_size * n_vars))
            self._n_queries = window_size * n_vars
            self.head = None
        else:
            msg = f"Unknown task: {task!r}"
            raise ValueError(msg)

    def _get_query_positions(self, device: torch.device) -> torch.Tensor:
        """Build normalized query position tensor.

        Returns:
            For forecasting: ``(horizon, 1)`` normalized time positions.
            For anomaly: ``(window*n_vars, 2)`` (time, variable) positions.
        """
        if self.task == "forecasting":
            # Normalized positions in [0, 1]
            positions = torch.linspace(0, 1, self.horizon, device=device).unsqueeze(-1)
            return positions
        elif self.task == "anomaly":
            t_pos = torch.linspace(0, 1, self.window_size, device=device)
            v_pos = torch.linspace(0, 1, self.n_vars, device=device)
            # Create grid: (window*n_vars, 2)
            grid_t, grid_v = torch.meshgrid(t_pos, v_pos, indexing="ij")
            positions = torch.stack([grid_t.flatten(), grid_v.flatten()], dim=-1)
            return positions
        msg = "Query positions only for forecasting/anomaly tasks"
        raise RuntimeError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through branch/trunk + inner product.

        Args:
            x: Input tensor ``(batch, window, n_vars)``.

        Returns:
            - classification: ``(batch, n_classes)`` logits
            - forecasting: ``(batch, horizon)`` predictions
            - anomaly: ``(batch, window, n_vars)`` reconstruction
        """
        batch_size = x.shape[0]

        # CNN branch processes (batch, window, n_vars) directly
        branch_emb = self.branch(x)  # (batch, rank)

        if self.task == "classification":
            return self.head(branch_emb)  # (batch, n_classes)

        # Get trunk embeddings for query positions
        query_pos = self._get_query_positions(x.device)  # (n_queries, input_dim)
        trunk_emb = self.trunk(query_pos)  # (n_queries, rank)

        # Inner product: (batch, rank) @ (rank, n_queries) → (batch, n_queries)
        output = torch.matmul(branch_emb, trunk_emb.T) + self.output_bias

        if self.task == "anomaly":
            # Reshape to (batch, window, n_vars)
            output = output.view(batch_size, self.window_size, self.n_vars)

        return output

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
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
