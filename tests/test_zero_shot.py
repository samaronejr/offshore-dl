"""Tests for zero-shot foundation model wrappers."""

import torch
import pytest

from offshore_dl.models.chronos_wrapper import ChronosWrapper


class _DummyChronosPipeline:
    def predict(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        batch = context.shape[0]
        base = context[:, -1:].repeat(1, prediction_length).to(dtype=torch.float32)
        return base.unsqueeze(1).repeat(1, 3, 1)


def _install_dummy_chronos(model: ChronosWrapper) -> None:
    model._pipeline = _DummyChronosPipeline()



# ═══════════════════════════════════════════════════════════════════
# Chronos
# ═══════════════════════════════════════════════════════════════════

class TestChronosWrapper:
    """Tests for ChronosWrapper zero-shot forecasting."""

    def test_classification_raises(self) -> None:
        with pytest.raises(ValueError, match="classification"):
            ChronosWrapper(task="classification", n_vars=10)

    def test_forecasting_forward_shape(self) -> None:
        model = ChronosWrapper(
            task="forecasting", n_vars=10, horizon=14, window_size=48,
            model_name="amazon/chronos-t5-tiny",
        )
        _install_dummy_chronos(model)
        x = torch.randn(2, 48, 10)
        out = model(x)
        assert out.shape == (2, 14)

    def test_anomaly_forward_shape(self) -> None:
        model = ChronosWrapper(
            task="anomaly", n_vars=3, window_size=20,
            model_name="amazon/chronos-t5-tiny",
        )
        _install_dummy_chronos(model)
        x = torch.randn(2, 20, 3)
        out = model(x)
        assert out.shape == (2, 20, 3)

    def test_training_step_warns(self) -> None:
        model = ChronosWrapper(task="forecasting", n_vars=5, horizon=10)
        batch = (torch.randn(2, 48, 5), torch.randn(2, 10), [{}] * 2)
        with pytest.warns(UserWarning, match="zero-shot"):
            loss = model.training_step(batch)
        assert loss.item() == 0.0

    def test_predict_returns_tensor(self) -> None:
        model = ChronosWrapper(
            task="forecasting", n_vars=5, horizon=10, window_size=30,
            model_name="amazon/chronos-t5-tiny",
        )
        _install_dummy_chronos(model)
        batch = (torch.randn(2, 30, 5), torch.randn(2, 10), [{}] * 2)
        preds = model.predict(batch)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (2, 10)

    def test_configure_optimizers(self) -> None:
        model = ChronosWrapper(task="forecasting", n_vars=5, horizon=10)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.SGD)

    def test_is_zero_shot_and_outputs_follow_input_device(self) -> None:
        model = ChronosWrapper(task="forecasting", n_vars=2, horizon=4, window_size=6)
        _install_dummy_chronos(model)
        x = torch.randn(3, 6, 2)
        out = model(x)
        assert model.is_zero_shot is True
        assert out.device == x.device

    def test_training_step_loss_follows_batch_device(self) -> None:
        model = ChronosWrapper(task="forecasting", n_vars=2, horizon=4, window_size=6)
        x = torch.randn(3, 6, 2)
        with pytest.warns(UserWarning, match="zero-shot"):
            loss = model.training_step((x, torch.randn(3, 4), [{}] * 3))
        assert loss.device == x.device


# ═══════════════════════════════════════════════════════════════════
# TimesFM
# ═══════════════════════════════════════════════════════════════════

from offshore_dl.models.timesfm_wrapper import TimesFMWrapper
from offshore_dl.models.timesfm_wrapper import is_available as timesfm_available


class _DummyTimesFMBackend:
    def __init__(self, horizon: int) -> None:
        self.horizon = horizon

    def forecast(self, inputs):
        rows = []
        for arr in inputs:
            rows.append([float(arr[-1]) if len(arr) else 0.0] * self.horizon)
        return rows, None


def _install_dummy_timesfm(model: "TimesFMWrapper") -> None:
    horizon = model.window_size if model.task == "anomaly" else model.horizon
    model._model = _DummyTimesFMBackend(horizon)



class TestTimesFMWrapper:
    """Tests for TimesFMWrapper zero-shot forecasting."""

    def test_is_zero_shot_marker(self) -> None:
        assert TimesFMWrapper.is_zero_shot is True

    def test_classification_raises(self) -> None:
        with pytest.raises((ValueError, ImportError)):
            TimesFMWrapper(task="classification", n_vars=10)

    @pytest.mark.skipif(not timesfm_available(), reason="TimesFM not installed")
    def test_forecasting_forward_shape(self) -> None:
        model = TimesFMWrapper(
            task="forecasting", n_vars=10, horizon=14, window_size=48,
        )
        _install_dummy_timesfm(model)
        x = torch.randn(2, 48, 10)
        out = model(x)
        assert out.shape == (2, 14)

    @pytest.mark.skipif(not timesfm_available(), reason="TimesFM not installed")
    def test_anomaly_forward_shape(self) -> None:
        model = TimesFMWrapper(
            task="anomaly", n_vars=3, window_size=20,
        )
        _install_dummy_timesfm(model)
        x = torch.randn(2, 20, 3)
        out = model(x)
        assert out.shape == (2, 20, 3)

    def test_training_step_warns(self) -> None:
        if not timesfm_available():
            pytest.skip("TimesFM not installed")
        model = TimesFMWrapper(task="forecasting", n_vars=5, horizon=10)
        batch = (torch.randn(2, 48, 5), torch.randn(2, 10), [{}] * 2)
        with pytest.warns(UserWarning, match="zero-shot"):
            loss = model.training_step(batch)
        assert loss.item() == 0.0

    def test_configure_optimizers(self) -> None:
        if not timesfm_available():
            pytest.skip("TimesFM not installed")
        model = TimesFMWrapper(task="forecasting", n_vars=5, horizon=10)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.SGD)


# ═══════════════════════════════════════════════════════════════════
# TiRex
# ═══════════════════════════════════════════════════════════════════

from offshore_dl.models.tirex_wrapper import TiRexWrapper
from offshore_dl.models.tirex_wrapper import is_available as tirex_available


class _DummyTiRexBackend:
    def forecast(self, context: torch.Tensor, prediction_length: int):
        mean = context[:, -1:].repeat(1, prediction_length).to(dtype=torch.float32)
        return None, mean


def _install_dummy_tirex(model: "TiRexWrapper") -> None:
    model._model = _DummyTiRexBackend()



class TestTiRexWrapper:
    """Tests for TiRexWrapper zero-shot forecasting."""

    def test_is_zero_shot_marker(self) -> None:
        assert TiRexWrapper.is_zero_shot is True

    def test_classification_raises(self) -> None:
        with pytest.raises((ValueError, ImportError)):
            TiRexWrapper(task="classification", n_vars=10)

    @pytest.mark.skipif(not tirex_available(), reason="TiRex not installed")
    def test_forecasting_forward_shape(self) -> None:
        model = TiRexWrapper(
            task="forecasting", n_vars=10, horizon=14, window_size=48,
        )
        _install_dummy_tirex(model)
        x = torch.randn(2, 48, 10)
        out = model(x)
        assert out.shape == (2, 14)

    @pytest.mark.skipif(not tirex_available(), reason="TiRex not installed")
    def test_anomaly_forward_shape(self) -> None:
        model = TiRexWrapper(
            task="anomaly", n_vars=3, window_size=20,
        )
        _install_dummy_tirex(model)
        x = torch.randn(2, 20, 3)
        out = model(x)
        assert out.shape == (2, 20, 3)

    def test_training_step_warns(self) -> None:
        if not tirex_available():
            pytest.skip("TiRex not installed")
        model = TiRexWrapper(task="forecasting", n_vars=5, horizon=10)
        batch = (torch.randn(2, 48, 5), torch.randn(2, 10), [{}] * 2)
        with pytest.warns(UserWarning, match="zero-shot"):
            loss = model.training_step(batch)
        assert loss.item() == 0.0

    def test_configure_optimizers(self) -> None:
        if not tirex_available():
            pytest.skip("TiRex not installed")
        model = TiRexWrapper(task="forecasting", n_vars=5, horizon=10)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.SGD)


# ═══════════════════════════════════════════════════════════════════
# TimesFM / TiRex availability checks
# ═══════════════════════════════════════════════════════════════════

class TestFMAvailability:
    """Test that unavailable FMs raise clear errors."""

    def test_timesfm_unavailable_raises(self) -> None:
        if timesfm_available():
            pytest.skip("TimesFM is installed — can't test unavailability")

        with pytest.raises(ImportError, match="TimesFM is not installed"):
            TimesFMWrapper(task="forecasting", n_vars=10)

    def test_tirex_unavailable_raises(self) -> None:
        if tirex_available():
            pytest.skip("TiRex is installed — can't test unavailability")

        with pytest.raises(ImportError, match="TiRex is not installed"):
            TiRexWrapper(task="forecasting", n_vars=10)

    def test_timesfm_classification_raises(self) -> None:
        """Even if TimesFM were available, classification should be rejected."""
        with pytest.raises((ValueError, ImportError)):
            TimesFMWrapper(task="classification", n_vars=10)

    def test_tirex_classification_raises(self) -> None:
        with pytest.raises((ValueError, ImportError)):
            TiRexWrapper(task="classification", n_vars=10)


# ═══════════════════════════════════════════════════════════════════
# Integration: Chronos through ExperimentRunner
# ═══════════════════════════════════════════════════════════════════

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from offshore_dl.evaluation.cv import TemporalSplitCV
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.training.experiment import ExperimentRunner


class _TinyDataset(Dataset):
    def __init__(self, task="forecasting", n=20, n_vars=3, window=30, horizon=10):
        self.X = torch.randn(n, window, n_vars)
        if task == "forecasting":
            self.y = torch.randn(n, horizon)
        elif task == "anomaly":
            self.y = self.X.clone()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], {}


class TestChronosIntegration:
    """Chronos through the ExperimentRunner pipeline."""

    def test_forecasting_pipeline(self) -> None:
        """Chronos produces forecasting metrics through ExperimentRunner.

        Note: Chronos doesn't train — the Trainer.fit() calls
        training_step which returns 0 loss. The real value is
        in the predict() calls during validation.
        """
        ds = _TinyDataset("forecasting", n=20, n_vars=3, horizon=10, window=30)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 4, "max_epochs": 1, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        original_load = ChronosWrapper._load_pipeline
        def fake_load(self):
            _install_dummy_chronos(self)
        ChronosWrapper._load_pipeline = fake_load
        runner = ExperimentRunner(
            model_class=ChronosWrapper,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={
                "task": "forecasting", "n_vars": 3, "horizon": 10, "window_size": 30,
                "model_name": "amazon/chronos-t5-tiny",
            },
        )
        try:
            results = runner.run(use_mlflow=False)
        finally:
            ChronosWrapper._load_pipeline = original_load
        assert "mae" in results["fold_results"][0]["metrics"]
