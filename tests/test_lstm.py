"""Tests for LSTMModel — all 3 task types."""

import torch
import pytest

from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.base import model_summary


# ═══════════════════════════════════════════════════════════════════
# Classification (3W)
# ═══════════════════════════════════════════════════════════════════

class TestLSTMClassification:
    """LSTM for 3W 10-class classification."""

    @pytest.fixture
    def model(self):
        return LSTMModel(task="classification", n_vars=27, n_classes=10, hidden_size=32, num_layers=1)

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

    def test_bidirectional(self, batch) -> None:
        model = LSTMModel(
            task="classification", n_vars=27, n_classes=10,
            hidden_size=32, num_layers=1, bidirectional=True,
        )
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 10)


# ═══════════════════════════════════════════════════════════════════
# Forecasting (Ganymede)
# ═══════════════════════════════════════════════════════════════════

class TestLSTMForecasting:
    """LSTM for Ganymede gas production forecasting."""

    @pytest.fixture
    def model(self):
        return LSTMModel(task="forecasting", n_vars=63, horizon=30, hidden_size=32, num_layers=1)

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

class TestLSTMAnomaly:
    """LSTM for CDF unsupervised anomaly detection."""

    @pytest.fixture
    def model(self):
        return LSTMModel(task="anomaly", n_vars=11, window_size=48, hidden_size=32, num_layers=1)

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 48, 11)
        # Reconstruction target is the input itself
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
# Model Summary
# ═══════════════════════════════════════════════════════════════════

class TestLSTMModelSummary:
    """Model summary statistics for LSTM."""

    def test_param_count_reasonable(self) -> None:
        model = LSTMModel(task="classification", n_vars=27, n_classes=10, hidden_size=128, num_layers=2)
        summary = model_summary(model)
        # 2-layer LSTM with 128 hidden + linear head should have ~200k+ params
        assert summary["param_count"] > 50_000
        assert summary["trainable_params"] == summary["param_count"]

    def test_configure_optimizers(self) -> None:
        model = LSTMModel(task="classification", n_vars=27, n_classes=10, lr=0.005)
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        # Check LR was set
        assert optimizer.param_groups[0]["lr"] == 0.005


# ═══════════════════════════════════════════════════════════════════
# Integration: LSTM through ExperimentRunner
# ═══════════════════════════════════════════════════════════════════

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from offshore_dl.evaluation.cv import TemporalSplitCV
from offshore_dl.training.experiment import ExperimentRunner


class _TinyDataset(Dataset):
    """Tiny synthetic dataset for integration testing."""

    def __init__(self, task="classification", n=40, n_vars=10, window=20, n_classes=3, horizon=5):
        self.X = torch.randn(n, window, n_vars)
        self.task = task
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


class TestLSTMIntegration:
    """Integration tests: LSTM through the full pipeline."""

    def test_classification_pipeline(self) -> None:
        ds = _TinyDataset("classification", n=40, n_vars=10, n_classes=3, window=20)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=LSTMModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3,
                          "hidden_size": 16, "num_layers": 1, "window_size": 20},
        )
        results = runner.run(use_mlflow=False)
        assert "f1_macro" in results["fold_results"][0]["metrics"]

    def test_forecasting_pipeline(self) -> None:
        ds = _TinyDataset("forecasting", n=40, n_vars=10, horizon=5, window=20)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=LSTMModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "forecasting", "n_vars": 10, "horizon": 5,
                          "hidden_size": 16, "num_layers": 1},
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
            model_class=LSTMModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "anomaly", "n_vars": 10, "window_size": 20,
                          "hidden_size": 16, "num_layers": 1},
        )
        results = runner.run(use_mlflow=False)
        assert "error_mean" in results["fold_results"][0]["metrics"]
