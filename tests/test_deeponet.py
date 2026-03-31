"""Tests for DeepONetModel — all 3 task types."""

import torch
import pytest

from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.base import model_summary


# ═══════════════════════════════════════════════════════════════════
# Classification (3W)
# ═══════════════════════════════════════════════════════════════════

class TestDeepONetClassification:
    """DeepONet for 3W 10-class classification."""

    @pytest.fixture
    def model(self):
        return DeepONetModel(
            task="classification", n_vars=27, n_classes=10,
            branch_hidden=[32, 32], rank=16, window_size=720,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 720, 27)
        y = torch.randint(0, 10, (4,))
        return x, y, [{}] * 4

    def test_forward_shape(self, model, batch) -> None:
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 10)

    def test_training_step_returns_scalar_loss(self, model, batch) -> None:
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_validation_step_returns_dict(self, model, batch) -> None:
        result = model.validation_step(batch)
        assert "loss" in result
        assert "predictions" in result
        assert result["predictions"].shape == (4,)

    def test_predict_returns_class_indices(self, model, batch) -> None:
        preds = model.predict(batch)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64


# ═══════════════════════════════════════════════════════════════════
# Forecasting (Ganymede)
# ═══════════════════════════════════════════════════════════════════

class TestDeepONetForecasting:
    """DeepONet for Ganymede gas production forecasting."""

    @pytest.fixture
    def model(self):
        return DeepONetModel(
            task="forecasting", n_vars=63, horizon=30,
            branch_hidden=[32, 32], trunk_hidden=[32, 32],
            rank=16, window_size=90,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 90, 63)
        y = torch.randn(4, 30)
        return x, y, [{}] * 4

    def test_forward_shape(self, model, batch) -> None:
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 30)

    def test_training_step_returns_scalar_loss(self, model, batch) -> None:
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_predict_returns_horizon(self, model, batch) -> None:
        preds = model.predict(batch)
        assert preds.shape == (4, 30)


# ═══════════════════════════════════════════════════════════════════
# Anomaly Detection (CDF)
# ═══════════════════════════════════════════════════════════════════

class TestDeepONetAnomaly:
    """DeepONet for CDF unsupervised anomaly detection."""

    @pytest.fixture
    def model(self):
        return DeepONetModel(
            task="anomaly", n_vars=11, window_size=48,
            branch_hidden=[32, 32], trunk_hidden=[32, 32], rank=16,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 48, 11)
        return x, x.clone(), [{}] * 4

    def test_forward_shape(self, model, batch) -> None:
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 48, 11)

    def test_training_step_returns_scalar_loss(self, model, batch) -> None:
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_predict_returns_reconstruction(self, model, batch) -> None:
        preds = model.predict(batch)
        assert preds.shape == (4, 48, 11)


# ═══════════════════════════════════════════════════════════════════
# Model Summary + Convergence
# ═══════════════════════════════════════════════════════════════════

class TestDeepONetMisc:
    """Model summary and convergence sanity checks."""

    def test_param_count_reasonable(self) -> None:
        model = DeepONetModel(
            task="forecasting", n_vars=63, horizon=30,
            branch_hidden=[128, 128], trunk_hidden=[128, 128],
            rank=64, window_size=90,
        )
        summary = model_summary(model)
        # MLP-based — should have substantial params from branch input
        assert summary["param_count"] > 100_000

    def test_configure_optimizers(self) -> None:
        model = DeepONetModel(task="classification", n_vars=27, n_classes=10, lr=0.003, window_size=720)
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.003

    def test_convergence_sanity(self) -> None:
        """Loss should decrease over a few gradient steps."""
        model = DeepONetModel(
            task="forecasting", n_vars=10, horizon=5,
            branch_hidden=[16], trunk_hidden=[16], rank=8, window_size=20,
        )
        optimizer = model.configure_optimizers()

        x = torch.randn(8, 20, 10)
        y = torch.randn(8, 5)
        batch = (x, y, [{}] * 8)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (at least final < initial)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ═══════════════════════════════════════════════════════════════════
# Integration: DeepONet through ExperimentRunner
# ═══════════════════════════════════════════════════════════════════

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from offshore_dl.evaluation.cv import TemporalSplitCV
from offshore_dl.training.experiment import ExperimentRunner


class _TinyDataset(Dataset):
    def __init__(self, task="classification", n=40, n_vars=10, window=20, n_classes=3, horizon=5):
        self.X = torch.randn(n, window, n_vars)
        if task == "classification":
            self.y = torch.randint(0, n_classes, (n,))
        elif task == "forecasting":
            self.y = torch.randn(n, horizon)
        elif task == "anomaly":
            self.y = self.X.clone()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], {}


class TestDeepONetIntegration:
    """Integration tests: DeepONet through the full pipeline."""

    def test_forecasting_pipeline(self) -> None:
        ds = _TinyDataset("forecasting", n=40, n_vars=10, horizon=5, window=20)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=DeepONetModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={
                "task": "forecasting", "n_vars": 10, "horizon": 5, "window_size": 20,
                "branch_hidden": [16], "trunk_hidden": [16], "rank": 8,
            },
        )
        results = runner.run(use_mlflow=False)
        assert "mae" in results["fold_results"][0]["metrics"]

    def test_anomaly_pipeline(self) -> None:
        ds = _TinyDataset("anomaly", n=40, n_vars=10, window=20)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=DeepONetModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={
                "task": "anomaly", "n_vars": 10, "window_size": 20,
                "branch_hidden": [16], "trunk_hidden": [16], "rank": 8,
            },
        )
        results = runner.run(use_mlflow=False)
        assert "error_mean" in results["fold_results"][0]["metrics"]
