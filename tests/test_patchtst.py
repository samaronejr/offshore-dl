"""Tests for PatchTSTModel — all 3 task types."""

import torch
import pytest

from offshore_dl.models.patchtst import PatchTSTModel
from offshore_dl.models.base import model_summary


# Use small architecture for fast tests
SMALL_KWARGS = {"patch_len": 8, "stride": 4, "d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64}


# ═══════════════════════════════════════════════════════════════════
# Classification (3W)
# ═══════════════════════════════════════════════════════════════════

class TestPatchTSTClassification:
    """PatchTST for 3W 10-class classification."""

    @pytest.fixture
    def model(self):
        return PatchTSTModel(
            task="classification", n_vars=27, n_classes=10, window_size=64,
            **SMALL_KWARGS,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 64, 27)
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

    def test_predict_returns_class_indices(self, model, batch) -> None:
        preds = model.predict(batch)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64


# ═══════════════════════════════════════════════════════════════════
# Forecasting (Ganymede)
# ═══════════════════════════════════════════════════════════════════

class TestPatchTSTForecasting:
    """PatchTST for Ganymede gas production forecasting."""

    @pytest.fixture
    def model(self):
        return PatchTSTModel(
            task="forecasting", n_vars=10, horizon=14, window_size=48,
            **SMALL_KWARGS,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 48, 10)
        y = torch.randn(4, 14)
        return x, y, [{}] * 4

    def test_forward_shape(self, model, batch) -> None:
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 14)

    def test_training_step_returns_scalar_loss(self, model, batch) -> None:
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_predict_returns_horizon(self, model, batch) -> None:
        preds = model.predict(batch)
        assert preds.shape == (4, 14)


# ═══════════════════════════════════════════════════════════════════
# Anomaly Detection (CDF)
# ═══════════════════════════════════════════════════════════════════

class TestPatchTSTAnomaly:
    """PatchTST for CDF unsupervised anomaly detection."""

    @pytest.fixture
    def model(self):
        return PatchTSTModel(
            task="anomaly", n_vars=11, window_size=48,
            **SMALL_KWARGS,
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
# Model Summary + Misc
# ═══════════════════════════════════════════════════════════════════

class TestPatchTSTMisc:
    def test_param_count_reasonable(self) -> None:
        model = PatchTSTModel(
            task="forecasting", n_vars=10, horizon=14, window_size=48,
            d_model=64, n_heads=4, n_layers=2, d_ff=128, patch_len=8, stride=4,
        )
        summary = model_summary(model)
        assert summary["param_count"] > 10_000

    def test_configure_optimizers(self) -> None:
        model = PatchTSTModel(task="classification", n_vars=11, n_classes=5, window_size=32, lr=0.002, **SMALL_KWARGS)
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 0.002


# ═══════════════════════════════════════════════════════════════════
# Integration: PatchTST through ExperimentRunner
# ═══════════════════════════════════════════════════════════════════

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from offshore_dl.evaluation.cv import TemporalSplitCV
from offshore_dl.training.experiment import ExperimentRunner


class _TinyDataset(Dataset):
    def __init__(self, task="classification", n=40, n_vars=10, window=32, n_classes=3, horizon=8):
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


class TestPatchTSTIntegration:
    def test_forecasting_pipeline(self) -> None:
        ds = _TinyDataset("forecasting", n=32, n_vars=10, horizon=8, window=32)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=PatchTSTModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={
                "task": "forecasting", "n_vars": 10, "horizon": 8, "window_size": 32,
                **SMALL_KWARGS,
            },
        )
        results = runner.run(use_mlflow=False)
        assert "mae" in results["fold_results"][0]["metrics"]

    def test_anomaly_pipeline(self) -> None:
        ds = _TinyDataset("anomaly", n=32, n_vars=10, window=32)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=PatchTSTModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={
                "task": "anomaly", "n_vars": 10, "window_size": 32,
                **SMALL_KWARGS,
            },
        )
        results = runner.run(use_mlflow=False)
        assert "error_mean" in results["fold_results"][0]["metrics"]
