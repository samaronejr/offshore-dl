"""ConvTimeNet: Deformable-patch Conv-based Time Series Classifier.

Architecture from Cheng et al. (WWW 2025) — adapted for offshore fault
classification.  All sub-modules (BoxCoder, OffsetPredictor, DeformablePatch,
SublayerConnection, _ConvEncoderLayer, _ConvEncoder, ConvTimeNet_backbone)
are consolidated in this single file.

Key adaptations from the official repo:
    * D020 — all ``device='cuda:0'`` removed; anchors use ``register_buffer``.
    * ``forward(self, x)`` takes only ``x: (B, seq_len, n_vars)``.
    * Instance normalization (K017) applied as first forward() op.
    * Structural re-parameterisation merges weights *once* on ``eval()``.
"""

from __future__ import annotations

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from offshore_dl.models.base import BaseModel, instance_normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

def _get_activation_fn(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {activation!r}")


# ---------------------------------------------------------------------------
# DeformablePatch sub-modules (from dlutils.py)
# ---------------------------------------------------------------------------


class OffsetPredictor(nn.Module):
    """Conv1d-based offset prediction for deformable patches.

    With ``mod=0`` (default): Conv1d → activation → Conv1d producing 2-D
    offsets (center shift, width adjustment).

    Args:
        in_feats: Number of input channels (n_vars).
        patch_size: Kernel size for the first conv.
        stride: Stride for the first conv.
        act: Activation function name.
        mod: Predictor variant (only 0 implemented).
    """

    def __init__(
        self,
        in_feats: int,
        patch_size: int,
        stride: int,
        act: str = "gelu",
        mod: int = 0,
    ) -> None:
        super().__init__()
        self.mod = mod
        if mod == 0:
            self.offset_predictor = nn.Sequential(
                nn.Conv1d(in_feats, 64, patch_size, stride),
                _get_activation_fn(act),
                nn.Conv1d(64, 2, 1, 1),
            )
        else:
            raise ValueError(f"OffsetPredictor mod={mod} not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict offsets from input.

        Args:
            x: ``(B, n_vars, seq_len)``

        Returns:
            ``(B, 2, patch_count)`` — center and width offsets.
        """
        return self.offset_predictor(x)


class BoxCoder(nn.Module):
    """Anchor-based sampling grid generator for deformable patching.

    Generates evenly-spaced anchors along the temporal axis, then decodes
    predicted offsets into bounding boxes and creates a meshgrid of sampling
    locations via interpolation.

    Args:
        patch_count: Number of patches.
        patch_stride: Stride between patch centres.
        patch_size: Size of each patch.
        seq_len: Padded sequence length.
        channels: Number of input channels (n_vars).
        weights: Scaling for (center, width) offsets.
        tanh: Whether to apply tanh to offsets.
    """

    def __init__(
        self,
        patch_count: int,
        patch_stride: int,
        patch_size: int,
        seq_len: int,
        channels: int,
        weights: tuple[float, float] = (1.0, 1.0),
        tanh: bool = False,
    ) -> None:
        super().__init__()
        self.patch_count = patch_count
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.channels = channels
        self.weights = weights
        self.tanh = tanh

        # Generate anchors and register as buffer (D020: device-agnostic)
        anchors = self._generate_anchors()
        self.register_buffer("anchors", anchors)

    def _generate_anchors(self) -> torch.Tensor:
        """Create evenly-spaced anchor boxes.

        Returns:
            ``(1, 2, patch_count)`` — centres and widths.
        """
        centres = torch.arange(0, self.patch_count, dtype=torch.float32) * self.patch_stride + self.patch_size / 2.0
        widths = torch.full_like(centres, float(self.patch_size))
        # Stack as (1, 2, patch_count): row 0 = centres, row 1 = widths
        anchors = torch.stack([centres, widths], dim=0).unsqueeze(0)
        return anchors

    def _decode(self, offsets: torch.Tensor) -> torch.Tensor:
        """Decode offsets relative to anchors into absolute boxes.

        Args:
            offsets: ``(B, 2, patch_count)``

        Returns:
            ``(B, 2, patch_count)`` — decoded (left, right) boundaries.
        """
        if self.tanh:
            offsets = torch.tanh(offsets)

        w_center, w_width = self.weights
        # offsets[:, 0] = center shift, offsets[:, 1] = width adjustment
        center_offset = offsets[:, 0:1, :] / w_center
        width_offset = offsets[:, 1:2, :] / w_width

        anchor_center = self.anchors[:, 0:1, :]  # (1, 1, patch_count)
        anchor_width = self.anchors[:, 1:2, :]    # (1, 1, patch_count)

        pred_center = anchor_center + center_offset * anchor_width
        # Clamp width_offset to prevent exp() overflow on long sequences (K027)
        width_offset = width_offset.clamp(-4.0, 4.0)
        pred_width = anchor_width * torch.exp(width_offset)

        # Convert to (left, right) boundaries
        left = pred_center - pred_width / 2.0
        right = pred_center + pred_width / 2.0

        return torch.cat([left, right], dim=1)  # (B, 2, patch_count)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """Decode offsets and create sampling grid.

        Args:
            offsets: ``(B, 2, patch_count)``

        Returns:
            ``(B, patch_count * patch_size, channels, 2)`` — grid for
            ``F.grid_sample``.
        """
        boxes = self._decode(offsets)  # (B, 2, patch_count)
        B = boxes.shape[0]

        # For each box, create patch_size evenly-spaced sample points
        # left/right: (B, 1, patch_count)
        left = boxes[:, 0:1, :]   # (B, 1, patch_count)
        right = boxes[:, 1:2, :]  # (B, 1, patch_count)

        # Create sampling locations: patch_size points per patch
        # steps: (1, patch_size, 1) from 0 to 1
        steps = torch.linspace(0, 1, self.patch_size, device=boxes.device).view(1, -1, 1)
        # sample_x: (B, patch_size, patch_count) — absolute positions
        sample_x = left + steps * (right - left)

        # Normalize x to [-1, 1] for grid_sample; clamp for numerical safety
        sample_x_norm = 2.0 * sample_x / (self.seq_len - 1) - 1.0
        sample_x_norm = sample_x_norm.clamp(-1.0, 1.0)

        # Reshape to (B, patch_count, patch_size) then flatten spatial dim
        sample_x_norm = sample_x_norm.permute(0, 2, 1)  # (B, patch_count, patch_size)
        sample_x_norm = sample_x_norm.reshape(B, -1)     # (B, patch_count * patch_size)

        # y-coordinates: each sample maps to all channels (but grid_sample
        # expects 2D input treated as image — channels dim handled separately)
        # Create grid: (B, 1, patch_count*patch_size, 2) for channels=1 height
        # Actually we use F.grid_sample with input (B, channels, 1, seq_len)
        # So grid should be (B, 1, patch_count*patch_size, 2) with y=0
        grid_x = sample_x_norm.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        grid_y = torch.zeros_like(grid_x)  # y=0 since height=1
        grid = torch.cat([grid_x, grid_y], dim=-1)  # (B, 1, N, 2)

        return grid


class DeformablePatch(nn.Module):
    """Deformable patch embedding — the key innovation of ConvTimeNet.

    Pads the input, predicts sampling offsets, uses ``F.grid_sample`` to
    extract deformable patches, then projects via Conv2d to ``d_model``.

    Args:
        in_feats: Number of input channels (n_vars).
        out_feats: Output dimension (d_model).
        seq_len: Original sequence length.
        patch_size: Patch size.
        stride: Patch stride.
        padding_tp: Padding type (default: reflect).
        norm: Normalisation type ('batch' or 'layer').
        act: Activation function name.
        offset_mod: OffsetPredictor variant.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        seq_len: int,
        patch_size: int,
        stride: int,
        padding_tp: str | None = None,
        norm: str = "batch",
        act: str = "gelu",
        offset_mod: int = 0,
    ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride

        # Compute padding to ensure full coverage
        # Pad so that (padded_len - patch_size) / stride + 1 = desired patch_count
        self.pad_len = 0
        if padding_tp is not None:
            # Compute how much padding needed
            remainder = (seq_len - patch_size) % stride
            if remainder != 0:
                self.pad_len = stride - remainder
        else:
            remainder = (seq_len - patch_size) % stride
            if remainder != 0:
                self.pad_len = stride - remainder

        self.padded_len = seq_len + self.pad_len
        self.patch_count = (self.padded_len - patch_size) // stride + 1
        self.new_len = self.patch_count  # exposed for backbone

        # Offset predictor
        self.offset_predictor = OffsetPredictor(in_feats, patch_size, stride, act=act, mod=offset_mod)

        # Box coder (D020: no device= arg)
        self.box_coder = BoxCoder(
            patch_count=self.patch_count,
            patch_stride=stride,
            patch_size=patch_size,
            seq_len=self.padded_len,
            channels=in_feats,
        )

        # Projection: Conv2d to project patches to d_model
        # Input to conv2d: (B, in_feats, 1, patch_count * patch_size) reshaped
        # Actually: project from (B, in_feats, patch_count, patch_size) → (B, out_feats, patch_count, 1)
        self.proj = nn.Conv2d(in_feats, out_feats, kernel_size=(1, patch_size), stride=(1, 1))
        self.act = _get_activation_fn(act)
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_feats)
        else:
            self.norm = nn.LayerNorm(out_feats)
        self.norm_tp = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deformable patches and project.

        Args:
            x: ``(B, seq_len, n_vars)``

        Returns:
            ``(B, d_model, patch_count)``
        """
        B, L, C = x.shape

        # Transpose to (B, C, L) for conv operations
        x_t = x.permute(0, 2, 1)  # (B, n_vars, seq_len)

        # Pad if needed
        if self.pad_len > 0:
            x_t = F.pad(x_t, (0, self.pad_len), mode="replicate")

        # Predict offsets: (B, 2, patch_count)
        offsets = self.offset_predictor(x_t)

        # Generate sampling grid
        grid = self.box_coder(offsets)  # (B, 1, patch_count*patch_size, 2)

        # Reshape input for grid_sample: (B, C, 1, padded_len)
        x_2d = x_t.unsqueeze(2)  # (B, C, 1, padded_len)

        # Sample: (B, C, 1, patch_count*patch_size)
        sampled = F.grid_sample(x_2d, grid, mode="bilinear", padding_mode="border", align_corners=True)

        # Reshape to (B, C, patch_count, patch_size)
        sampled = sampled.squeeze(2)  # (B, C, patch_count*patch_size)
        sampled = sampled.view(B, C, self.patch_count, self.patch_size)

        # Project via Conv2d: (B, C, patch_count, patch_size) → (B, out_feats, patch_count, 1)
        projected = self.proj(sampled)  # (B, out_feats, patch_count, 1)
        projected = projected.squeeze(-1)  # (B, out_feats, patch_count)

        # Activation + Norm
        projected = self.act(projected)
        if self.norm_tp == "batch":
            projected = self.norm(projected)
        else:
            projected = self.norm(projected.permute(0, 2, 1)).permute(0, 2, 1)

        return projected  # (B, d_model, patch_count)


# ---------------------------------------------------------------------------
# ConvTimeNet backbone sub-modules (from ConvTimeNet_backbone.py)
# ---------------------------------------------------------------------------


class SublayerConnection(nn.Module):
    """Residual connection with optional learnable scaling parameter (α).

    When ``enable_res_parameter=True``, computes ``x + dropout(α * out_x)``
    where α is a learnable scalar initialised to 0.5.

    Args:
        enable_res_parameter: Enable learnable residual scaling.
        dropout: Dropout probability.
    """

    def __init__(self, enable_res_parameter: bool, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, out_x: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            return x + self.dropout(self.a * out_x)


class _ConvEncoderLayer(nn.Module):
    """Depthwise conv block with structural re-parameterisation.

    During training, runs parallel large + small depthwise convolutions
    and sums the outputs.  During eval, merges the weights into a single
    conv for faster inference.

    Args:
        kernel_size: Large depthwise conv kernel size.
        d_model: Channel dimension.
        d_ff: Feed-forward expansion dimension.
        dropout: Dropout probability.
        activation: Activation function name.
        enable_res_param: Enable learnable residual parameter.
        norm: 'batch' or 'layer'.
        small_ks: Small depthwise conv kernel size (default 3).
        re_param: Enable structural re-parameterisation.
    """

    def __init__(
        self,
        kernel_size: int,
        d_model: int,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        enable_res_param: bool = True,
        norm: str = "batch",
        small_ks: int = 3,
        re_param: bool = True,
    ) -> None:
        super().__init__()
        self.norm_tp = norm
        self.re_param = re_param
        self._reparam_merged = False

        # Depthwise conv (with optional structural re-parameterisation)
        if self.re_param:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding="same", groups=d_model)
            self.DW_conv_small = nn.Conv1d(d_model, d_model, self.small_ks, stride=1, padding="same", groups=d_model)
            self.DW_infer = nn.Conv1d(d_model, d_model, self.large_ks, stride=1, padding="same", groups=d_model)
        else:
            self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding="same", groups=d_model)

        self.dw_act = _get_activation_fn(activation)
        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model)

        # Position-wise feed-forward
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1, 1),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, 1, 1),
        )

        # Add & Norm
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model) if norm == "batch" else nn.LayerNorm(d_model)

    def _merge_reparam_weights(self) -> None:
        """Merge large + small conv weights into DW_infer (once)."""
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        merged = copy.deepcopy(self.DW_conv_large)
        merged.weight.data += F.pad(self.DW_conv_small.weight.data, (left_pad, right_pad), value=0)
        merged.bias.data += self.DW_conv_small.bias.data
        # Copy to DW_infer
        self.DW_infer.weight.data.copy_(merged.weight.data)
        self.DW_infer.bias.data.copy_(merged.bias.data)
        self._reparam_merged = True

    def train(self, mode: bool = True) -> _ConvEncoderLayer:
        """Override train() to reset merge flag when switching back to train."""
        super().train(mode)
        if mode:
            self._reparam_merged = False
        return self

    def eval(self) -> _ConvEncoderLayer:
        """Override eval() to merge weights once."""
        super().eval()
        if self.re_param and not self._reparam_merged:
            self._merge_reparam_weights()
        return self

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: ``(B, d_model, L)``."""
        # Depthwise conv layer
        if not self.re_param:
            dw_out = self.DW_conv(src)
            src = self.sublayerconnect1(src, self.dw_act(dw_out))
        else:
            if self.training:
                large_out = self.DW_conv_large(src)
                small_out = self.DW_conv_small(src)
                src = self.sublayerconnect1(src, self.dw_act(large_out + small_out))
            else:
                if not self._reparam_merged:
                    self._merge_reparam_weights()
                merge_out = self.DW_infer(src)
                src = self.sublayerconnect1(src, self.dw_act(merge_out))

        # Norm
        if self.norm_tp != "batch":
            src = src.permute(0, 2, 1)
        src = self.dw_norm(src)
        if self.norm_tp != "batch":
            src = src.permute(0, 2, 1)

        # Position-wise feed-forward
        src2 = self.ff(src)

        # Add & Norm
        src2 = self.sublayerconnect2(src, src2)
        if self.norm_tp != "batch":
            src2 = src2.permute(0, 2, 1)
        src2 = self.norm_ffn(src2)
        if self.norm_tp != "batch":
            src2 = src2.permute(0, 2, 1)

        return src2


class _ConvEncoder(nn.Module):
    """Stack of ``_ConvEncoderLayer`` modules with different kernel sizes.

    Args:
        d_model: Channel dimension.
        d_ff: Feed-forward expansion dimension.
        kernel_size: List of kernel sizes (one per layer).
        dropout: Dropout probability.
        activation: Activation function name.
        n_layers: Number of encoder layers.
        enable_res_param: Enable learnable residual parameter.
        norm: 'batch' or 'layer'.
        re_param: Enable structural re-parameterisation.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_size: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        n_layers: int = 3,
        enable_res_param: bool = False,
        norm: str = "batch",
        re_param: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = [19, 19, 29, 29, 37, 37]
        self.layers = nn.ModuleList([
            _ConvEncoderLayer(
                kernel_size[i], d_model, d_ff=d_ff, dropout=dropout,
                activation=activation, enable_res_param=enable_res_param,
                norm=norm, re_param=re_param,
            )
            for i in range(n_layers)
        ])

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class ConvTimeNet_backbone(nn.Module):
    """ConvTimeNet backbone: linear projection → encoder → pooling head.

    Args:
        c_in: Input channels (d_model from DeformablePatch).
        c_out: Number of output classes.
        seq_len: Sequence length (patch_count from DeformablePatch).
        n_layers: Number of encoder layers.
        d_model: Model dimension.
        d_ff: Feed-forward expansion dimension.
        dropout: Dropout probability.
        act: Activation function name.
        pooling_tp: Pooling type ('max', 'mean', or 'cat').
        fc_dropout: Dropout before final linear.
        enable_res_param: Enable learnable residual parameter.
        dw_ks: List of depthwise conv kernel sizes.
        norm: 'batch' or 'layer'.
        use_embed: Whether to use linear embedding projection.
        re_param: Enable structural re-parameterisation.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.1,
        act: str = "gelu",
        pooling_tp: str = "max",
        fc_dropout: float = 0.0,
        enable_res_param: bool = False,
        dw_ks: list[int] | None = None,
        norm: str = "batch",
        use_embed: bool = True,
        re_param: bool = False,
    ) -> None:
        super().__init__()
        if dw_ks is None:
            dw_ks = [7, 13, 19]
        assert n_layers == len(dw_ks), f"dw_ks length ({len(dw_ks)}) must match n_layers ({n_layers})"

        self.c_out = c_out
        self.seq_len = seq_len

        # Input embedding
        self.use_embed = use_embed
        self.W_P = nn.Linear(c_in, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _ConvEncoder(
            d_model, d_ff, kernel_size=dw_ks, dropout=dropout,
            activation=act, n_layers=n_layers, enable_res_param=enable_res_param,
            norm=norm, re_param=re_param,
        )

        self.flatten = nn.Flatten()

        # Head
        self.head_nf = seq_len * d_model if pooling_tp == "cat" else d_model
        self.head = self._create_head(self.head_nf, c_out, act=act, pooling_tp=pooling_tp, fc_dropout=fc_dropout)

    def _create_head(self, nf: int, c_out: int, act: str = "gelu", pooling_tp: str = "max", fc_dropout: float = 0.0) -> nn.Sequential:
        layers: list[nn.Module] = []
        if pooling_tp == "cat":
            layers = [_get_activation_fn(act), self.flatten]
            if fc_dropout:
                layers.append(nn.Dropout(fc_dropout))
        elif pooling_tp == "mean":
            layers = [nn.AdaptiveAvgPool1d(1), self.flatten]
        elif pooling_tp == "max":
            layers = [nn.AdaptiveMaxPool1d(1), self.flatten]

        layers.append(nn.Linear(nf, c_out))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, d_model, patch_count)`` from DeformablePatch.

        Returns:
            ``(B, c_out)`` logits.
        """
        u = x
        if self.use_embed:
            # x comes as (B, d_model, patch_count), transpose for linear
            u = self.W_P(x.transpose(2, 1))  # (B, patch_count, d_model)

        # Encoder expects (B, d_model, L)
        z = self.encoder(u.transpose(2, 1).contiguous())  # (B, d_model, patch_count)

        # Classification head
        return self.head(z)  # (B, c_out)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class ConvTimeNetModel(BaseModel):
    """ConvTimeNet: classification-only model with deformable patching.

    Full pipeline:
        Input (B, seq_len, n_vars)
          → Instance normalisation (K017)
          → DeformablePatch → (B, d_model, patch_count)
          → ConvTimeNet_backbone → (B, n_classes)

    Args:
        task: Must be ``"classification"``.
        n_vars: Number of input sensor variables.
        n_classes: Number of output classes.
        d_model: Model hidden dimension.
        d_ff: Feed-forward expansion dimension.
        patch_size: Patch size for deformable embedding.
        patch_stride: Patch stride for deformable embedding.
        dw_ks: List of depthwise conv kernel sizes (one per encoder layer).
        dropout: Dropout probability.
        pooling_tp: Pooling type ('max', 'mean', 'cat').
        fc_dropout: Dropout before final linear layer.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        window_size: Input window length (accepted for compatibility, K001).
        loss_type: ``"ce"`` or ``"focal"``.
        focal_gamma: Focusing exponent for focal loss.
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        n_classes: int = 10,
        d_model: int = 128,
        d_ff: int = 256,
        patch_size: int = 8,
        patch_stride: int = 4,
        dw_ks: list[int] | None = None,
        dropout: float = 0.1,
        pooling_tp: str = "max",
        fc_dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        window_size: int = 720,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        if task != "classification":
            msg = (
                f"ConvTimeNetModel supports classification only, got task={task!r}. "
                "The DeformablePatch+ConvEncoder pipeline is designed for sequence "
                "classification — forecasting and anomaly tasks are not supported."
            )
            raise ValueError(msg)

        super().__init__(
            task=task,
            n_vars=n_vars,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
        )

        if dw_ks is None:
            dw_ks = [7, 13, 19]

        # Store hyperparams
        self.d_model = d_model
        self.d_ff = d_ff
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.dw_ks = list(dw_ks)
        self.dropout_p = dropout
        self.pooling_tp = pooling_tp
        self.fc_dropout = fc_dropout
        self.n_classes = n_classes
        self.window_size = window_size
        self.lr = lr
        self.weight_decay = weight_decay

        # ── DeformablePatch embedding ──
        self.depatch_embedding = DeformablePatch(
            in_feats=n_vars,
            out_feats=d_model,
            seq_len=window_size,
            patch_size=patch_size,
            stride=patch_stride,
        )

        # ── ConvTimeNet backbone ──
        n_layers = len(dw_ks)
        new_len = self.depatch_embedding.new_len
        self.backbone = ConvTimeNet_backbone(
            c_in=d_model,
            c_out=n_classes,
            seq_len=new_len,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            act="gelu",
            pooling_tp=pooling_tp,
            fc_dropout=fc_dropout,
            enable_res_param=True,
            dw_ks=dw_ks,
            norm="batch",
            use_embed=False,
            re_param=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, seq_len, n_vars)`` input tensor.

        Returns:
            ``(B, n_classes)`` classification logits.
        """
        x = instance_normalize(x)

        # ── DeformablePatch embedding ──
        out_patch = self.depatch_embedding(x)  # (B, d_model, patch_count)

        # ── Backbone (expects (B, C, L), use_embed=False) ──
        output = self.backbone(out_patch.permute(0, 2, 1))  # (B, n_classes)

        return output

