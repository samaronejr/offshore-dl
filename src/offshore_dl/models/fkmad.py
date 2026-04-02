"""FKM-AD: Fourier-KAN → Mamba → GatedSharpening → AttentionPooling → ClassificationHead.

Novel architecture from Wang et al. (2025, arXiv:2511.15083) adapted for
supervised 10-class classification of offshore equipment faults.

**CUDA required** — Mamba (mamba_ssm) has no CPU backend.

Sub-modules:
    - FourierKANProjection: learnable Fourier basis expansion with low-rank projection
    - GatedSharpeningTemperature: element-wise gating that sharpens temporal contrast
    - FKMADModel: full pipeline inheriting BaseModel
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from mamba_ssm import Mamba

from offshore_dl.models.base import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-module 1: FourierKAN Projection
# ---------------------------------------------------------------------------


class FourierKANProjection(nn.Module):
    """Learnable Fourier basis expansion with low-rank projection.

    For each input variable, computes sin/cos features at *learnable*
    frequencies, concatenates them, and projects to ``d_model`` via a
    low-rank bottleneck.  A parallel linear branch provides a residual
    path so the model can fall back to a simple linear projection when
    the Fourier features are not informative.

    Args:
        n_vars: Number of input variables.
        d_model: Output embedding dimension.
        n_fourier_freqs: Number of Fourier frequency components per variable.
        fourier_rank: Low-rank bottleneck dimension.
        fourier_scale: Divisor applied to inputs before the Fourier basis.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        n_fourier_freqs: int = 8,
        fourier_rank: int = 32,
        fourier_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_fourier_freqs = n_fourier_freqs
        self.fourier_scale = fourier_scale

        # Linear residual branch
        self.linear_branch = nn.Linear(n_vars, d_model)

        # Learnable frequencies — backprop-enabled
        self.fourier_freqs = nn.Parameter(
            torch.arange(1, n_fourier_freqs + 1, dtype=torch.float32)
        )

        # Low-rank projection: (2 * F * n_vars) → rank → d_model
        fourier_dim = 2 * n_fourier_freqs * n_vars
        self.fourier_to_rank = nn.Linear(fourier_dim, fourier_rank, bias=False)
        self.rank_to_model = nn.Linear(fourier_rank, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input through linear + Fourier branches.

        Args:
            x: ``(B, L, n_vars)``

        Returns:
            ``(B, L, d_model)``
        """
        # Linear path
        h_linear = self.linear_branch(x)

        # Fourier path
        x_scaled = x / self.fourier_scale  # (B, L, n_vars)

        # Broadcasting: freqs (F,) → (1, 1, 1, F)
        freqs = self.fourier_freqs.view(1, 1, 1, -1)
        x_exp = x_scaled.unsqueeze(-1)  # (B, L, n_vars, 1)

        phase = 2.0 * math.pi * freqs * x_exp  # (B, L, n_vars, F)
        sin_feat = torch.sin(phase)  # (B, L, n_vars, F)
        cos_feat = torch.cos(phase)  # (B, L, n_vars, F)

        # Concatenate sin/cos on last dim → (B, L, n_vars, 2F)
        fourier_feat = torch.cat([sin_feat, cos_feat], dim=-1)

        # Flatten to (B, L, 2*F*n_vars)
        B, L = fourier_feat.shape[:2]
        fourier_feat = fourier_feat.reshape(B, L, -1)

        # Low-rank projection → (B, L, d_model)
        h_fourier = self.rank_to_model(self.fourier_to_rank(fourier_feat))

        return h_linear + h_fourier


# ---------------------------------------------------------------------------
# Sub-module 2: Gated Sharpening Temperature
# ---------------------------------------------------------------------------


class GatedSharpeningTemperature(nn.Module):
    """Element-wise gating that sharpens temporal contrast (Wang et al. Eq. 7).

    Computes a deviation signal from the temporal mean (with stop-gradient)
    and uses it to gate the input, amplifying timesteps that deviate from
    the sequence average.

    Args:
        d_model: Feature dimension.
        gamma_z_init: Initial value of the learnable sharpening coefficient.
    """

    def __init__(self, d_model: int, gamma_z_init: float = 1.0) -> None:
        super().__init__()
        self.gamma_z = nn.Parameter(torch.tensor(gamma_z_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated sharpening.

        Args:
            x: ``(B, L, d_model)``

        Returns:
            ``(B, L, d_model)`` — sharpened output.
        """
        # Stop-gradient on temporal mean (paper Eq. 7)
        mean_t = x.mean(dim=1, keepdim=True).detach()
        z_prime = self.gamma_z * (x - mean_t)
        return x * torch.sigmoid(z_prime)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class FKMADModel(BaseModel):
    """FKM-AD: Fourier-KAN Mamba Anomaly/Fault Detector.

    Full pipeline:
        Input (B, L, n_vars)
          → FourierKANProjection → (B, L, d_model)
          → LayerNorm
          → n_mamba_layers × (LayerNorm → Mamba → residual)
          → GatedSharpeningTemperature
          → LayerNorm
          → AttentionPooling → (B, d_model)
          → ClassificationHead → (B, n_classes)

    **CUDA required** — Mamba has no CPU backend.

    Args:
        task: Must be ``"classification"``.
        n_vars: Number of input sensor variables.
        d_model: Model hidden dimension.
        d_state: Mamba SSM state dimension.
        d_conv: Mamba local convolution width.
        expand: Mamba expansion factor.
        n_fourier_freqs: Fourier frequency count.
        fourier_rank: Low-rank bottleneck for Fourier features.
        fourier_scale: Input normalisation divisor for Fourier basis.
        gamma_z_init: Initial sharpening coefficient.
        n_classes: Number of output classes.
        n_mamba_layers: Number of stacked Mamba blocks.
        dropout: Dropout probability.
        window_size: Input window length (accepted for compatibility with
            production scripts, see K001).
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
        n_fourier_freqs: int = 8,
        fourier_rank: int = 32,
        fourier_scale: float = 1.0,
        gamma_z_init: float = 1.0,
        n_classes: int = 10,
        n_mamba_layers: int = 2,
        dropout: float = 0.2,
        window_size: int = 720,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
    ) -> None:
        if task != "classification":
            msg = (
                f"FKMADModel supports classification only, got task={task!r}. "
                "The Mamba+AttentionPooling pipeline is designed for sequence "
                "classification — forecasting and anomaly tasks are not supported."
            )
            raise ValueError(msg)

        super().__init__(
            task=task,
            n_vars=n_vars,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
        )

        # Store all hyperparams for checkpoint / HPO
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_fourier_freqs = n_fourier_freqs
        self.fourier_rank = fourier_rank
        self.fourier_scale = fourier_scale
        self.gamma_z_init = gamma_z_init
        self.n_classes = n_classes
        self.n_mamba_layers = n_mamba_layers
        self.dropout_p = dropout
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay

        # ── Input projection ──
        self.projection = FourierKANProjection(
            n_vars=n_vars,
            d_model=d_model,
            n_fourier_freqs=n_fourier_freqs,
            fourier_rank=fourier_rank,
            fourier_scale=fourier_scale,
        )
        self.post_proj_norm = nn.LayerNorm(d_model)

        # ── Mamba encoder (pre-norm residual blocks) ──
        self.mamba_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_mamba_layers)]
        )
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_mamba_layers)
            ]
        )

        # ── Gated sharpening ──
        self.sharpening = GatedSharpeningTemperature(
            d_model=d_model, gamma_z_init=gamma_z_init,
        )
        self.pre_pool_norm = nn.LayerNorm(d_model)

        # ── Attention pooling ──
        self.attn_w = nn.Linear(d_model, 1)

        # ── Classification head ──
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, n_vars)`` input tensor.

        Returns:
            ``(B, n_classes)`` classification logits.
        """
        # Input projection (linear + Fourier)
        h = self.projection(x)  # (B, L, d_model)
        h = self.post_proj_norm(h)

        # Mamba encoder with pre-norm residual
        for norm, mamba in zip(self.mamba_norms, self.mamba_layers):
            residual = h
            h = norm(h)
            h = mamba(h)
            h = h + residual

        # Gated sharpening
        h = self.sharpening(h)  # (B, L, d_model)
        h = self.pre_pool_norm(h)

        # Attention pooling: learn which timesteps matter
        attn_scores = self.attn_w(h).squeeze(-1)  # (B, L)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, L)
        pooled = torch.bmm(
            attn_weights.unsqueeze(1), h,
        ).squeeze(1)  # (B, d_model)
        pooled = self.dropout(pooled)

        # Classification head
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
