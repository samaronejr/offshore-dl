"""MambaSL: Single-Layer Mamba with time-variant SSM parameters for time series classification.

Adapted from Jung & Kim (ICLR 2026, OpenReview: YDl4vqQqGP) reference implementation:
``thuml/Time-Series-Library`` → ``models/MambaSingleLayer.py`` + ``layers/MambaBlock.py``
(class ``Mamba_TimeVariant``).

Key architectural differences from standard Mamba (FKMADModel):
    1. Time-variant SSM parameters: Δt, B, C are each individually controlled by flags.
    2. Configurable Conv1d kernel size in token embedding (d_conv).
    3. Optional D skip connection (use_D flag).
    4. Multi-head adaptive attention pooling over per-timestep logits (n_heads).

**CUDA required** — ``selective_scan_fn`` from ``mamba_ssm`` has no CPU backend.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange, repeat
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBASL_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MAMBASL_AVAILABLE = False

from offshore_dl.models.base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-module: Time-Variant Mamba Block
# ---------------------------------------------------------------------------


class MambaTimeVariant(nn.Module):
    """Single Mamba layer with individually-controllable time-variant SSM parameters.

    This is a direct adaptation of ``Mamba_TimeVariant`` from the TSlib reference
    implementation (Jung & Kim 2026). The four binary flags determine whether each
    SSM parameter (Δt, B, C) is computed from the input (time-variant) or kept as a
    learned constant, and whether the D skip connection is active.

    Args:
        d_model: Input/output feature dimension.
        d_state: SSM state dimension.
        d_conv: Depthwise conv kernel size for token mixing.
        expand: Inner-dimension expansion factor.
        dt_rank: Rank for Δt projection. ``"auto"`` uses ``ceil(d_model / 16)``.
        tv_dt: If True, Δt is computed from input (time-variant).
        tv_B: If True, B is computed from input (time-variant).
        tv_C: If True, C is computed from input (time-variant).
        use_D: If True, include the D skip connection in the SSM.
        dt_min: Minimum value for Δt softplus initialisation.
        dt_max: Maximum value for Δt softplus initialisation.
        dt_init_floor: Floor for Δt initialisation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        tv_dt: bool = True,
        tv_B: bool = True,
        tv_C: bool = True,
        use_D: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.tv_dt = tv_dt
        self.tv_B = tv_B
        self.tv_C = tv_C
        self.use_D = use_D

        self.d_inner = int(expand * d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # ── Input projection: (d_model) → (2 * d_inner) ──
        # Projects to x and z gates
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # ── Depthwise conv for local token mixing ──
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # ── SSM parameter projections ──
        # x_proj: (d_inner) → (dt_rank + 2*d_state) — used for Δt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # Δt projection: (dt_rank) → (d_inner)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise Δt bias using the standard Mamba init (log-uniform in [dt_min, dt_max])
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse softplus for initialisation of the bias
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # ── Learnable A matrix (diagonal): real negative values ──
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        # Stored as log for positivity constraint
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # ── Optional D skip connection ──
        if use_D:
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.D._no_weight_decay = True
        else:
            self.D = None

        # ── Output projection ──
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ── Static B/C for non-time-variant path ──
        if not tv_B:
            self.B_const = nn.Parameter(torch.zeros(1, d_state))
        if not tv_C:
            self.C_const = nn.Parameter(torch.zeros(1, d_state))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the time-variant Mamba block.

        Args:
            x: ``(B, L, d_model)``

        Returns:
            ``(B, L, d_model)``
        """
        B, L, _ = x.shape

        # ── Input gating projection ──
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # ── Depthwise conv (causal) ──
        # Conv1d expects (B, C, L)
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal truncation
        x_act = F.silu(x_conv)  # (B, d_inner, L)

        # ── SSM parameter projections ──
        # x_proj operates on (B*L, d_inner) — requires (B, L, d_inner)
        x_for_proj = x_act.transpose(1, 2)  # (B, L, d_inner)
        x_dbc = self.x_proj(x_for_proj)  # (B, L, dt_rank + 2*d_state)

        dt_raw = x_dbc[..., : self.dt_rank]  # (B, L, dt_rank)
        B_raw = x_dbc[..., self.dt_rank : self.dt_rank + self.d_state]  # (B, L, d_state)
        C_raw = x_dbc[..., self.dt_rank + self.d_state :]  # (B, L, d_state)

        # ── Δt (time step) ──
        # Pass the pre-softplus delta to selective_scan_fn along with
        # delta_bias and delta_softplus=True so the scan kernel handles the
        # full transform (bias add + softplus) internally.  Applying softplus
        # here AND passing delta_bias would double-transform and overflow.
        if self.tv_dt:
            dt = self.dt_proj(dt_raw)  # (B, L, d_inner) — pre-softplus
            dt = dt.transpose(1, 2)  # (B, d_inner, L)
            dt_bias = self.dt_proj.bias.float()  # (d_inner,)
        else:
            # Constant Δt: zero-input projection + bias only
            dt = torch.zeros(B, self.d_inner, L, device=x.device, dtype=x.dtype)
            dt_bias = self.dt_proj.bias.float()

        # ── A matrix ──
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # ── B matrix ──
        if self.tv_B:
            # Rearrange to (B, 1, d_state, L) for batch-variant selective_scan_fn
            B_ssm = rearrange(B_raw, "b l n -> b 1 n l")  # (B, 1, d_state, L)
        else:
            # Shape: (1, 1, d_state, 1) → broadcast over B, L
            B_ssm = self.B_const.view(1, 1, self.d_state, 1).expand(B, 1, -1, L)

        # ── C matrix ──
        if self.tv_C:
            C_ssm = rearrange(C_raw, "b l n -> b 1 n l")  # (B, 1, d_state, L)
        else:
            C_ssm = self.C_const.view(1, 1, self.d_state, 1).expand(B, 1, -1, L)

        # ── Selective scan ──
        # selective_scan_fn signature:
        #   u: (B, d_inner, L)   [the convolved features]
        #   delta: (B, d_inner, L)
        #   A: (d_inner, d_state)
        #   B: (B, 1, d_state, L)  [batch-variant]
        #   C: (B, 1, d_state, L)  [batch-variant]
        # Returns: (B, d_inner, L)
        y = selective_scan_fn(
            u=x_act,  # (B, d_inner, L)
            delta=dt,  # (B, d_inner, L) — pre-softplus
            A=A,  # (d_inner, d_state)
            B=B_ssm,  # (B, 1, d_state, L)
            C=C_ssm,  # (B, 1, d_state, L)
            D=self.D.float() if self.D is not None else None,
            z=z.transpose(1, 2),  # (B, d_inner, L)
            delta_bias=dt_bias,  # (d_inner,)  — added before softplus
            delta_softplus=True,  # let scan kernel apply softplus
        )  # (B, d_inner, L)

        y = y.transpose(1, 2)  # (B, L, d_inner)
        out = self.out_proj(y)  # (B, L, d_model)
        return out


# ---------------------------------------------------------------------------
# Token Embedding for classification
# ---------------------------------------------------------------------------


class TokenEmbedding(nn.Module):
    """Project raw input variables to d_model via Conv1d (length-preserving).

    Args:
        n_vars: Number of input variables.
        d_model: Output embedding dimension.
    """

    def __init__(self, n_vars: int, d_model: int) -> None:
        super().__init__()
        # Pad = 1 on each side preserves L when kernel=3
        self.tokenConv = nn.Conv1d(
            in_channels=n_vars,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        # Initialize with kaiming normal (standard for conv)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input.

        Args:
            x: ``(B, L, n_vars)``

        Returns:
            ``(B, L, d_model)``
        """
        # Conv1d expects (B, C, L)
        x = x.transpose(1, 2)  # (B, n_vars, L)
        x = self.tokenConv(x)  # (B, d_model, L)
        return x.transpose(1, 2)  # (B, L, d_model)


# ---------------------------------------------------------------------------
# Multi-Head Adaptive Attention Pooling
# ---------------------------------------------------------------------------


class AdaptiveAttentionPooling(nn.Module):
    """Multi-head adaptive attention pooling over timestep logits.

    For each head, learns an attention query over the sequence dimension,
    producing a pooled feature vector of size ``d_ff``. All head outputs
    are concatenated and projected back to ``d_model``.

    Args:
        d_model: Input feature dimension.
        d_ff: Per-head hidden dimension (attention projection).
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Per-head projections: d_model → d_ff
        self.head_projs = nn.ModuleList([
            nn.Linear(d_model, d_ff) for _ in range(n_heads)
        ])
        # Adaptive max pool over L dimension: (B, L, d_ff) → (B, 1, d_ff)
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Final projection: n_heads * d_ff → d_model
        self.out_proj = nn.Linear(n_heads * d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention-pool the sequence dimension.

        Args:
            x: ``(B, L, d_model)``

        Returns:
            ``(B, d_model)``
        """
        head_outputs = []
        for proj in self.head_projs:
            h = proj(x)  # (B, L, d_ff)
            # pool over L: expects (B, d_ff, L) → (B, d_ff, 1) → (B, d_ff)
            h = self.pool(h.transpose(1, 2)).squeeze(-1)  # (B, d_ff)
            head_outputs.append(h)

        concat = torch.cat(head_outputs, dim=-1)  # (B, n_heads * d_ff)
        return self.out_proj(concat)  # (B, d_model)


# ---------------------------------------------------------------------------
# MambaSLModel
# ---------------------------------------------------------------------------


class MambaSLModel(BaseModel):
    """MambaSL: Single-Layer Mamba with time-variant SSM for time series classification.

    Architecture pipeline:
        Input (B, L, n_vars)
          → Instance Normalization (raw path) or passthrough (feature path)
          → TokenEmbedding → (B, L, d_model)
          → LayerNorm
          → MambaTimeVariant (single layer, time-variant flags = all True)
          → LayerNorm
          → AdaptiveAttentionPooling → (B, d_model)
          → Dropout
          → FFN: d_model → d_ff → GELU → Dropout → n_classes

    **CUDA required** — ``selective_scan_fn`` has no CPU backend.

    Args:
        task: Must be ``"classification"``.
        n_vars: Number of input sensor variables.
        d_model: Token embedding dimension.
        d_state: SSM state dimension.
        d_conv: Conv1d kernel size inside the Mamba block.
        expand: Inner-dim expansion factor for the Mamba block.
        d_ff: Hidden dimension for attention pooling heads.
        n_heads: Number of attention pooling heads.
        n_classes: Number of output classes.
        tv_dt: Time-variant Δt flag (default True per D017).
        tv_B: Time-variant B flag (default True per D017).
        tv_C: Time-variant C flag (default True per D017).
        use_D: Use D skip connection (default True per D017).
        dropout: Dropout probability.
        window_size: Input window length (accepted for production-script compatibility, K001).
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        loss_type: ``"ce"`` or ``"focal"``.
        focal_gamma: Focusing exponent for focal loss.
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_ff: int = 256,
        n_heads: int = 4,
        n_classes: int = 10,
        tv_dt: bool = True,
        tv_B: bool = True,
        tv_C: bool = True,
        use_D: bool = True,
        dropout: float = 0.1,
        window_size: int = 720,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
    ) -> None:
        if task != "classification":
            msg = (
                f"MambaSLModel supports classification only, got task={task!r}. "
                "The single-layer Mamba + attention-pooling pipeline is designed for "
                "sequence classification."
            )
            raise ValueError(msg)

        if not _MAMBASL_AVAILABLE:
            raise ImportError(
                "MambaSLModel requires mamba_ssm and einops. "
                "Install with: pip install mamba-ssm einops"
            )

        super().__init__(
            task=task,
            n_vars=n_vars,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )

        # Store hyperparams for checkpoint / HPO
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.tv_dt = tv_dt
        self.tv_B = tv_B
        self.tv_C = tv_C
        self.use_D = use_D
        self.dropout_p = dropout
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay

        # ── Token embedding ──
        self.embedding = TokenEmbedding(n_vars=n_vars, d_model=d_model)
        self.post_embed_norm = nn.LayerNorm(d_model)

        # ── Single Mamba layer with time-variant SSM ──
        self.mamba = MambaTimeVariant(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            tv_dt=tv_dt,
            tv_B=tv_B,
            tv_C=tv_C,
            use_D=use_D,
        )
        self.post_mamba_norm = nn.LayerNorm(d_model)

        # ── Attention pooling ──
        self.pooling = AdaptiveAttentionPooling(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
        )

        # ── Classification head ──
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, n_vars)`` input tensor.

        Returns:
            ``(B, n_classes)`` classification logits.
        """
        # ── Instance normalization (per-sample z-score) ──
        # Prevents numerical instability from wide-ranging raw sensor values
        # (K017). Feature inputs are already normalized; z-score on top is
        # harmless but also stable.
        eps = 1e-5
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, n_vars)
        std = x.std(dim=1, keepdim=True).clamp(min=eps)
        x = (x - mean) / std
        # Clamp to guard against extreme outliers destabilising SSM layers
        x = x.clamp(-10.0, 10.0)

        # ── Token embedding ──
        h = self.embedding(x)  # (B, L, d_model)
        h = self.post_embed_norm(h)

        # ── Single Mamba layer (pre-norm residual) ──
        residual = h
        h = self.mamba(h)  # (B, L, d_model)
        h = h + residual
        h = self.post_mamba_norm(h)

        # ── Attention pooling ──
        pooled = self.pooling(h)  # (B, d_model)
        pooled = self.dropout(pooled)

        # ── Classification head ──
        return self.head(pooled)  # (B, n_classes)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        """Create AdamW optimizer.

        Args:
            cfg: Optional OmegaConf config with training.lr / training.weight_decay.

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
