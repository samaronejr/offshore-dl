"""Tests for ConvTimeNetModel — deformable-patch conv-based time series classifier.

ConvTimeNet has NO CUDA-only dependencies (Cheng et al., WWW 2025), so the
main test classes run entirely on CPU.  CUDA tests are added as additional
classes that are *skipped* when CUDA is unavailable.

Test organisation (5 classes):
1. TestConvTimeNetClassificationRaw       — raw (720, 27) windows, CPU
2. TestConvTimeNetClassificationFeatures  — feature (14, 27) inputs, CPU
3. TestConvTimeNetSubModules              — DeformablePatch, BoxCoder,
                                           _ConvEncoderLayer re-param, backbone
4. TestConvTimeNetModelConfig             — constructor / BaseModel contract
5. TestConvTimeNetImport                  — package-level import via __init__
"""

import torch
import pytest

from offshore_dl.models.convtimenet import (
    ConvTimeNetModel,
    DeformablePatch,
    BoxCoder,
    _ConvEncoderLayer,
    ConvTimeNet_backbone,
)
from offshore_dl.models.base import model_summary

CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_SKIP = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Classification on raw (720, 27) windows — CPU
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvTimeNetClassificationRaw:
    """ConvTimeNetModel classification on raw 720-step sensor windows (CPU)."""

    @pytest.fixture
    def model(self):
        return ConvTimeNetModel(
            task="classification",
            n_vars=27,
            n_classes=10,
            window_size=720,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 720, 27)
        y = torch.randint(0, 10, (4,))
        return x, y, [{}] * 4

    def test_forward_shape(self, model, batch):
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 10)

    def test_training_step_returns_scalar_loss(self, model, batch):
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_validation_step_returns_dict(self, model, batch):
        result = model.validation_step(batch)
        assert "loss" in result
        assert "predictions" in result
        assert "targets" in result
        assert result["predictions"].shape == (4,)

    def test_predict_returns_class_indices(self, model, batch):
        preds = model.predict(batch)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64

    def test_forward_values_finite(self, model, batch):
        """Raw sensor windows must produce finite logits."""
        x, _, _ = batch
        out = model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf on raw input"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Classification on statistical-feature (14, 27) inputs — CPU
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvTimeNetClassificationFeatures:
    """ConvTimeNetModel classification on 14-step statistical feature inputs (CPU)."""

    @pytest.fixture
    def model(self):
        return ConvTimeNetModel(
            task="classification",
            n_vars=27,
            n_classes=10,
            window_size=14,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 14, 27)
        y = torch.randint(0, 10, (4,))
        return x, y, [{}] * 4

    def test_forward_shape(self, model, batch):
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 10)

    def test_training_step_returns_scalar_loss(self, model, batch):
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_validation_step_returns_dict(self, model, batch):
        result = model.validation_step(batch)
        assert "loss" in result
        assert "predictions" in result
        assert "targets" in result
        assert result["predictions"].shape == (4,)

    def test_predict_returns_class_indices(self, model, batch):
        preds = model.predict(batch)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64

    def test_forward_values_finite(self, model, batch):
        """Feature inputs (short windows) must also produce finite logits."""
        x, _, _ = batch
        out = model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf on feature input"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sub-module isolation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvTimeNetSubModules:
    """Isolated shape and correctness tests for the core sub-modules."""

    def test_deformable_patch_output_shape_raw(self):
        """DeformablePatch must return (B, d_model, patch_count) for raw input."""
        patch = DeformablePatch(in_feats=27, out_feats=64, seq_len=720, patch_size=8, stride=4)
        x = torch.randn(2, 720, 27)
        out = patch(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 64
        assert out.shape[2] == patch.patch_count

    def test_deformable_patch_output_shape_features(self):
        """DeformablePatch must return correct shape for 14-step feature input."""
        patch = DeformablePatch(in_feats=27, out_feats=64, seq_len=14, patch_size=8, stride=4)
        x = torch.randn(2, 14, 27)
        out = patch(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 64
        assert out.shape[2] == patch.patch_count

    def test_boxcoder_anchor_generation(self):
        """BoxCoder anchors must have shape (1, 2, patch_count) and be finite."""
        coder = BoxCoder(
            patch_count=10,
            patch_stride=4,
            patch_size=8,
            seq_len=40,
            channels=27,
        )
        assert coder.anchors.shape == (1, 2, 10)
        assert torch.isfinite(coder.anchors).all()

    def test_boxcoder_forward_grid_shape(self):
        """BoxCoder.forward must return a grid of shape (B, 1, patch_count*patch_size, 2)."""
        coder = BoxCoder(patch_count=10, patch_stride=4, patch_size=8, seq_len=40, channels=27)
        offsets = torch.zeros(2, 2, 10)  # (B, 2, patch_count)
        grid = coder(offsets)
        assert grid.shape == (2, 1, 80, 2)

    def test_conv_encoder_layer_reparam_train_vs_eval_equal(self):
        """Structural re-parameterisation: DW_infer(x) == DW_conv_large(x) + DW_conv_small(x).

        Tests the convolution merge directly (not the full layer forward) to
        avoid BatchNorm train/eval discrepancies (batch stats vs running stats).
        The invariant is: after eval() triggers the merge, the single merged
        conv produces the same output as summing the two parallel convs.
        """
        torch.manual_seed(42)
        layer = _ConvEncoderLayer(kernel_size=7, d_model=16, d_ff=32, dropout=0.0, re_param=True)
        x = torch.randn(2, 16, 20)

        # Capture parallel-sum result in train mode
        layer.train()
        with torch.no_grad():
            parallel_out = layer.DW_conv_large(x) + layer.DW_conv_small(x)

        # Trigger weight merge by switching to eval
        layer.eval()
        assert layer._reparam_merged is True, "Merge flag must be set after eval()"
        with torch.no_grad():
            merged_out = layer.DW_infer(x)

        assert torch.allclose(parallel_out, merged_out, atol=1e-5), (
            f"Re-param DW conv mismatch: max diff = {(parallel_out - merged_out).abs().max().item():.2e}"
        )

    def test_conv_encoder_layer_reparam_idempotent(self):
        """Calling eval() twice must not corrupt the merged weights."""
        torch.manual_seed(7)
        layer = _ConvEncoderLayer(kernel_size=7, d_model=16, d_ff=32, dropout=0.0, re_param=True)
        x = torch.randn(2, 16, 20)

        layer.eval()
        with torch.no_grad():
            out1 = layer(x)

        layer.eval()  # second eval() call — should be idempotent
        with torch.no_grad():
            out2 = layer(x)

        assert torch.allclose(out1, out2, atol=1e-7)

    def test_conv_encoder_layer_retrain_after_eval(self):
        """After switching back to train(), layer.train() resets the merge flag."""
        torch.manual_seed(0)
        layer = _ConvEncoderLayer(kernel_size=7, d_model=16, d_ff=32, dropout=0.0, re_param=True)
        layer.eval()
        assert layer._reparam_merged is True
        layer.train()
        assert layer._reparam_merged is False

    def test_convtimenet_backbone_output_shape(self):
        """ConvTimeNet_backbone must return (B, c_out) for both seq lengths.

        ConvTimeNetModel passes (B, patch_count, d_model) to backbone — i.e.
        the sequence dimension is first (L), channels second.  With
        use_embed=False the backbone forwards u = x directly then transposes
        for the encoder, so the expected input shape is (B, L, d_model).
        """
        for seq_len in (14, 90):
            backbone = ConvTimeNet_backbone(
                c_in=64, c_out=10, seq_len=seq_len, n_layers=3,
                d_model=64, d_ff=128, dw_ks=[7, 13, 19],
                use_embed=False, re_param=True,
            )
            # Input shape: (B, seq_len, d_model) — matches ConvTimeNetModel convention
            x = torch.randn(2, seq_len, 64)
            out = backbone(x)
            assert out.shape == (2, 10), f"Wrong shape for seq_len={seq_len}: {out.shape}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Model config and BaseModel contract tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvTimeNetModelConfig:
    """Configuration and BaseModel contract tests for ConvTimeNetModel."""

    def test_configure_optimizers_returns_adamw(self):
        """configure_optimizers must return AdamW with the specified learning rate."""
        model = ConvTimeNetModel(
            task="classification", n_vars=27, n_classes=10,
            lr=0.005, window_size=720,
        )
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)

    def test_classification_only_rejection(self):
        """ConvTimeNetModel must reject non-classification tasks with ValueError."""
        with pytest.raises(ValueError, match="classification only"):
            ConvTimeNetModel(task="forecasting", n_vars=27, n_classes=10)

    def test_param_count_above_10k(self):
        """Default model must have > 10K trainable parameters."""
        model = ConvTimeNetModel(
            task="classification", n_vars=27, n_classes=10, window_size=720,
        )
        summary = model_summary(model)
        assert summary["param_count"] > 10_000
        assert summary["trainable_params"] == summary["param_count"]

    def test_window_size_kwarg_accepted(self):
        """Constructor must accept window_size and store it (K001 compatibility)."""
        model = ConvTimeNetModel(
            task="classification", n_vars=27, n_classes=10, window_size=512,
        )
        assert model.window_size == 512

    def test_window_size_14_accepted(self):
        """window_size=14 (feature path) must be accepted without error."""
        model = ConvTimeNetModel(
            task="classification", n_vars=27, n_classes=10, window_size=14,
        )
        assert model.window_size == 14

    def test_custom_dw_ks_accepted(self):
        """Custom dw_ks list must be stored and used correctly."""
        model = ConvTimeNetModel(
            task="classification", n_vars=27, n_classes=10,
            dw_ks=[5, 9, 13], window_size=720,
        )
        assert model.dw_ks == [5, 9, 13]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Package import test
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvTimeNetImport:
    """Verify package-level import works through offshore_dl.models.__init__."""

    def test_import_from_package(self):
        from offshore_dl.models import ConvTimeNetModel as Imported
        assert Imported is ConvTimeNetModel

    def test_convtimenet_in_all(self):
        """ConvTimeNetModel must appear in offshore_dl.models.__all__."""
        import offshore_dl.models as models_pkg
        assert "ConvTimeNetModel" in models_pkg.__all__


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CUDA tests — skipped when CUDA is unavailable
# ═══════════════════════════════════════════════════════════════════════════════


@CUDA_SKIP
class TestConvTimeNetCUDA:
    """ConvTimeNetModel forward pass on GPU — verifies device-agnostic design (D020)."""

    @pytest.fixture
    def model_raw(self):
        return ConvTimeNetModel(
            task="classification", n_vars=27, n_classes=10, window_size=720,
        ).cuda()

    @pytest.fixture
    def model_feat(self):
        return ConvTimeNetModel(
            task="classification", n_vars=27, n_classes=10, window_size=14,
        ).cuda()

    def test_forward_raw_cuda(self, model_raw):
        x = torch.randn(2, 720, 27, device="cuda")
        out = model_raw(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()

    def test_forward_features_cuda(self, model_feat):
        x = torch.randn(2, 14, 27, device="cuda")
        out = model_feat(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()
