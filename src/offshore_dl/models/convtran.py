"""ConvTran: Improving Position Encoding of Transformers for MTSC.

Foumani et al., Data Mining and Knowledge Discovery, 2024.
Transformer with tAPE (time-Absolute Position Encoding) and eRPE (enhanced Relative PE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel


class TimeAbsolutePositionEncoding(nn.Module):
    """Sinusoidal positions with learnable scaling."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.scale * self.pe[:, :seq_len]


class RelativeMultiheadAttention(nn.Module):
    """Multi-head self-attention with relative position bias."""

    def __init__(
        self, d_model: int, n_heads: int, max_len: int, dropout: float
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            msg = "d_model must be divisible by n_heads."
            raise ValueError(msg)

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_len = max_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_len - 1, n_heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def _relative_position_bias(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        coords = torch.arange(seq_len, device=device)
        rel = coords[:, None] - coords[None, :] + self.max_len - 1
        bias = self.relative_bias[rel]
        return bias.permute(2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + self._relative_position_bias(
            seq_len, x.device
        ).unsqueeze(0)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(attn_output)


class ConvTranEncoderLayer(nn.Module):
    """Transformer encoder block used in ConvTran."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelativeMultiheadAttention(d_model, n_heads, max_len, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class ConvTranModel(BaseModel):
    """ConvTran classifier for multivariate time-series fault diagnosis."""

    def __init__(
        self,
        task: str = "classification",
        n_vars: int = 27,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
        n_classes: int = 10,
        window_size: int = 14,
        lr: float = 1.0e-4,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            task=task,
            n_vars=n_vars,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )
        if task != "classification":
            msg = "ConvTranModel currently supports classification only."
            raise ValueError(msg)

        self.n_classes = n_classes
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.input_projection = nn.Conv1d(n_vars, d_model, kernel_size=1)
        self.position_encoding = TimeAbsolutePositionEncoding(
            d_model=d_model, max_len=window_size
        )
        self.input_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            [
                ConvTranEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    max_len=window_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits from ``(batch, window, n_vars)`` input."""
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_dropout(self.position_encoding(x))

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)

