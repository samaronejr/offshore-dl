"""Tests for TCN Ganymede forecasting model."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from offshore_dl.models.tcn import TCNModel


class TestForwardShape:
    """Verify TCN forward pass produces correct output shapes."""

    def test_forward_h7(self):
        model = TCNModel(task="forecasting", n_vars=63, horizon=7, window_size=90)
        x = torch.randn(4, 90, 63)
        out = model(x)
        assert out.shape == (4, 7)

    def test_forward_h30(self):
        model = TCNModel(task="forecasting", n_vars=63, horizon=30, window_size=90)
        x = torch.randn(4, 90, 63)
        out = model(x)
        assert out.shape == (4, 30)

    def test_forward_h90(self):
        model = TCNModel(task="forecasting", n_vars=63, horizon=90, window_size=90)
        x = torch.randn(4, 90, 63)
        out = model(x)
        assert out.shape == (4, 90)

    def test_forward_values_finite(self):
        model = TCNModel(task="forecasting", n_vars=63, horizon=7)
        x = torch.randn(4, 90, 63)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_batch_size_1(self):
        model = TCNModel(task="forecasting", n_vars=63, horizon=7)
        x = torch.randn(1, 90, 63)
        out = model(x)
        assert out.shape == (1, 7)


class TestReceptiveField:
    """Verify receptive field covers input window."""

    def test_default_rf_covers_90(self):
        """Default config (n_layers=6, kernel_size=3): RF=253 > 90."""
        model = TCNModel(n_vars=63, n_layers=6, kernel_size=3)
        assert model.receptive_field >= 90

    def test_rf_formula(self):
        """RF = 1 + 2*(k-1)*(2^L - 1)."""
        model = TCNModel(n_vars=63, n_layers=4, kernel_size=3)
        expected = 1 + 2 * (3 - 1) * (2**4 - 1)  # 61
        assert model.receptive_field == expected

    def test_small_rf_warning(self):
        """4 layers with k=3 gives RF=61 < 90."""
        model = TCNModel(n_vars=63, n_layers=4, kernel_size=3)
        assert model.receptive_field < 90  # known limitation for small configs


class TestTaskRestriction:
    """TCN only supports forecasting."""

    def test_classification_rejected(self):
        with pytest.raises(ValueError, match="forecasting"):
            TCNModel(task="classification", n_vars=63)

    def test_anomaly_rejected(self):
        with pytest.raises(ValueError, match="forecasting"):
            TCNModel(task="anomaly", n_vars=63)


class TestCausalConvolution:
    """Verify causal property — output at t only depends on inputs ≤ t."""

    def test_causal_no_future_leakage(self):
        model = TCNModel(n_vars=1, n_channels=16, n_layers=2, kernel_size=3, horizon=1)
        model.eval()

        # Create input where second half is zero
        x = torch.randn(1, 10, 1)
        x_masked = x.clone()
        x_masked[:, 5:, :] = 0.0

        # Get TCN internal representations
        with torch.no_grad():
            # First 5 timesteps should produce same output regardless of
            # what comes after (causal property)
            repr_full = x.transpose(1, 2)
            repr_masked = x_masked.transpose(1, 2)
            for block in model.tcn:
                repr_full = block(repr_full)
                repr_masked = block(repr_masked)

        # First timestep output should be identical
        assert torch.allclose(repr_full[:, :, 0], repr_masked[:, :, 0], atol=1e-5)


class TestOptimizer:
    """Verify configure_optimizers."""

    def test_returns_optimizer(self):
        model = TCNModel(n_vars=63, lr=0.01)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)

    def test_lr_from_init(self):
        model = TCNModel(n_vars=63, lr=0.01)
        opt = model.configure_optimizers()
        assert opt.defaults["lr"] == 0.01


class TestConfig:
    """Verify YAML config loads correctly."""

    def test_config_loads(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/models/tcn.yaml")
        assert cfg.model.name == "tcn"
        assert cfg.model.architecture.n_channels == 128
        assert cfg.model.architecture.n_layers == 6

    def test_optuna_search_space(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/models/tcn.yaml")
        ss = cfg.model.optuna_search_space
        assert "n_channels" in ss
        assert "n_layers" in ss
        assert "kernel_size" in ss
        assert "dropout" in ss
        assert "lr" in ss


class TestParameterCount:
    """Verify model has reasonable parameter count."""

    def test_param_count_reasonable(self):
        model = TCNModel(n_vars=63, n_channels=128, n_layers=6, horizon=30)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be in 100K-5M range for a reasonable TCN
        assert 50_000 < n_params < 5_000_000
