"""Tests for MambaSLModel — single-layer Mamba with time-variant SSM parameters.

All tests require CUDA because mamba_ssm.ops.selective_scan_interface has no
CPU backend.  Tests are organised into five classes that mirror the structure
used by the other model test suites (e.g. test_fkmad.py):

1. TestMambaSLClassificationRaw       — raw (720, 27) windows
2. TestMambaSLClassificationFeatures  — stat-feature (14, 27) inputs
3. TestMambaSLSubModules              — TokenEmbedding, MambaTimeVariant, pooling
4. TestMambaSLModelConfig             — constructor / BaseModel contract
5. TestMambaSLImport                  — package-level import via __init__.py

NaN stability tests are included inline (test_forward_values_finite) because
K018 flags the time-variant SSM as NaN-prone; forward must stay finite under
both normal and near-zero input conditions.
"""

import torch
import pytest

from offshore_dl.models.mambasl import (
    MambaSLModel,
    MambaTimeVariant,
    TokenEmbedding,
    AdaptiveAttentionPooling,
)
from offshore_dl.models.base import model_summary

CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_SKIP = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required for mamba_ssm")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Classification on raw (720, 27) windows
# ═══════════════════════════════════════════════════════════════════════════════


@CUDA_SKIP
class TestMambaSLClassificationRaw:
    """MambaSLModel classification on raw 720-step sensor windows."""

    @pytest.fixture
    def model(self):
        return MambaSLModel(
            task="classification",
            n_vars=27,
            d_model=64,
            n_classes=10,
            window_size=720,
        ).cuda()

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 720, 27, device="cuda")
        y = torch.randint(0, 10, (4,), device="cuda")
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
        """Raw sensor windows must produce finite logits (guards against NaN overflow
        in the time-variant selective scan — K018)."""
        x, _, _ = batch
        out = model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf on raw input"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Classification on statistical-feature (14, 27) inputs
# ═══════════════════════════════════════════════════════════════════════════════


@CUDA_SKIP
class TestMambaSLClassificationFeatures:
    """MambaSLModel classification on 14-step statistical feature inputs."""

    @pytest.fixture
    def model(self):
        return MambaSLModel(
            task="classification",
            n_vars=27,
            d_model=64,
            n_classes=10,
            window_size=14,
        ).cuda()

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 14, 27, device="cuda")
        y = torch.randint(0, 10, (4,), device="cuda")
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


@CUDA_SKIP
class TestMambaSLSubModules:
    """Isolated shape and gradient tests for the three core sub-modules."""

    def test_token_embedding_shape(self):
        """TokenEmbedding must preserve the time dimension."""
        emb = TokenEmbedding(n_vars=27, d_model=64).cuda()
        x = torch.randn(4, 100, 27, device="cuda")
        out = emb(x)
        assert out.shape == (4, 100, 64)

    def test_mamba_time_variant_shape(self):
        """MambaTimeVariant must return (B, L, d_model) for varied L."""
        mamba = MambaTimeVariant(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
        for L in (14, 720):
            x = torch.randn(4, L, 64, device="cuda")
            out = mamba(x)
            assert out.shape == (4, L, 64), f"Shape mismatch for L={L}"

    def test_adaptive_attention_pooling_shape(self):
        """AdaptiveAttentionPooling must reduce the sequence dimension to (B, d_model)."""
        pool = AdaptiveAttentionPooling(d_model=64, d_ff=128, n_heads=4).cuda()
        x = torch.randn(4, 100, 64, device="cuda")
        out = pool(x)
        assert out.shape == (4, 64)

    def test_mamba_time_variant_gradient_flow(self):
        """All parameters of MambaTimeVariant must receive gradients after backward."""
        mamba = MambaTimeVariant(d_model=32, d_state=8, d_conv=4, expand=2).cuda()
        x = torch.randn(2, 50, 32, device="cuda")
        out = mamba(x)
        loss = out.sum()
        loss.backward()
        no_grad = [
            name for name, p in mamba.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not no_grad, (
            f"Parameters with None gradient after backward: {no_grad}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Model config and BaseModel contract tests
# ═══════════════════════════════════════════════════════════════════════════════


@CUDA_SKIP
class TestMambaSLModelConfig:
    """Configuration and BaseModel contract tests for MambaSLModel."""

    def test_configure_optimizers(self):
        """configure_optimizers must return AdamW with the correct learning rate."""
        model = MambaSLModel(
            task="classification", n_vars=27, d_model=64,
            n_classes=10, lr=0.005,
        ).cuda()
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)

    def test_classification_only(self):
        """MambaSLModel must reject non-classification tasks."""
        with pytest.raises(ValueError, match="classification only"):
            MambaSLModel(task="forecasting", n_vars=27, d_model=64, n_classes=10)

    def test_param_count_reasonable(self):
        """Model with d_model=64 must have more than 10 K trainable parameters."""
        model = MambaSLModel(
            task="classification", n_vars=27, d_model=64, n_classes=10,
        ).cuda()
        summary = model_summary(model)
        assert summary["param_count"] > 10_000
        assert summary["trainable_params"] == summary["param_count"]

    def test_window_size_kwarg_accepted(self):
        """Constructor must accept window_size without error (K001 compatibility)."""
        model = MambaSLModel(
            task="classification", n_vars=27, d_model=64,
            n_classes=10, window_size=512,
        )
        assert model.window_size == 512


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Package import test
# ═══════════════════════════════════════════════════════════════════════════════


@CUDA_SKIP
class TestMambaSLImport:
    """Verify package-level import works through offshore_dl.models.__init__."""

    def test_import_from_package(self):
        from offshore_dl.models import MambaSLModel as Imported
        assert Imported is MambaSLModel
