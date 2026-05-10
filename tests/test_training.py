"""Tests for training engine: BaseModel, Trainer, ExperimentRunner."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from offshore_dl.evaluation.cv import ExpandingWindowCV, TemporalSplitCV
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.models.base import BaseModel, model_summary
from offshore_dl.models.dummy import DummyModel
from offshore_dl.training.experiment import ExperimentRunner, NormalizedSubset
from offshore_dl.training.optuna_utils import (
    OptunaObjective,
    convergence_callback,
    create_study,
    run_hpo,
)
from offshore_dl.training.trainer import CostTracker, EarlyStopping, Trainer

import time


# ═══════════════════════════════════════════════════════════════════
# DummyModel Tests
# ═══════════════════════════════════════════════════════════════════


class TestDummyModelClassification:
    """DummyModel for classification task."""

    @pytest.fixture
    def model(self) -> DummyModel:
        return DummyModel(task="classification", n_vars=27, n_classes=10)

    def test_forward_shape(self, model: DummyModel) -> None:
        x = torch.randn(4, 100, 27)
        out = model(x)
        assert out.shape == (4, 10)

    def test_training_step_returns_scalar_loss(self, model: DummyModel) -> None:
        batch = (torch.randn(4, 100, 27), torch.randint(0, 10, (4,)), [{}] * 4)
        loss = model.training_step(batch)
        assert loss.dim() == 0  # scalar
        assert loss.requires_grad

    def test_validation_step_returns_dict(self, model: DummyModel) -> None:
        batch = (torch.randn(4, 100, 27), torch.randint(0, 10, (4,)), [{}] * 4)
        result = model.validation_step(batch)
        assert "loss" in result
        assert "predictions" in result
        assert "targets" in result
        assert result["predictions"].shape == (4,)  # argmax class indices

    def test_predict_returns_class_indices(self, model: DummyModel) -> None:
        batch = (torch.randn(4, 100, 27), torch.randint(0, 10, (4,)), [{}] * 4)
        preds = model.predict(batch)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64

    def test_predict_scores_returns_probabilities(self, model: DummyModel) -> None:
        batch = (torch.randn(4, 100, 27), torch.randint(0, 10, (4,)), [{}] * 4)
        scores = model.predict_scores(batch)
        assert scores.shape == (4, 10)
        torch.testing.assert_close(
            scores.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=0.0
        )


class TestDummyModelForecasting:
    """DummyModel for forecasting task."""

    @pytest.fixture
    def model(self) -> DummyModel:
        return DummyModel(task="forecasting", n_vars=63, horizon=30)

    def test_forward_shape(self, model: DummyModel) -> None:
        x = torch.randn(4, 90, 63)
        out = model(x)
        assert out.shape == (4, 30)

    def test_training_step_returns_scalar_loss(self, model: DummyModel) -> None:
        batch = (torch.randn(4, 90, 63), torch.randn(4, 30), [{}] * 4)
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_predict_returns_horizon(self, model: DummyModel) -> None:
        batch = (torch.randn(4, 90, 63), torch.randn(4, 30), [{}] * 4)
        preds = model.predict(batch)
        assert preds.shape == (4, 30)


class TestDummyModelAnomaly:
    """DummyModel for anomaly (reconstruction) task."""

    @pytest.fixture
    def model(self) -> DummyModel:
        return DummyModel(task="anomaly", n_vars=11, window_size=48)

    def test_forward_shape(self, model: DummyModel) -> None:
        x = torch.randn(4, 48, 11)
        out = model(x)
        assert out.shape == (4, 48, 11)

    def test_training_step_returns_scalar_loss(self, model: DummyModel) -> None:
        x = torch.randn(4, 48, 11)
        batch = (x, x.clone(), [{}] * 4)  # reconstruction target = input
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_predict_returns_reconstruction(self, model: DummyModel) -> None:
        x = torch.randn(4, 48, 11)
        batch = (x, x.clone(), [{}] * 4)
        preds = model.predict(batch)
        assert preds.shape == (4, 48, 11)


class TestModelSummary:
    """Test model_summary utility."""

    def test_returns_required_keys(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        summary = model_summary(model)
        assert "param_count" in summary
        assert "trainable_params" in summary
        assert "model_size_mb" in summary

    def test_param_count_positive(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        summary = model_summary(model)
        assert summary["param_count"] > 0
        assert summary["trainable_params"] > 0

    def test_model_size_reasonable(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        summary = model_summary(model)
        assert summary["model_size_mb"] < 1.0  # tiny model


# ═══════════════════════════════════════════════════════════════════
# Trainer Tests
# ═══════════════════════════════════════════════════════════════════


def _make_classification_loaders(
    n_train=40, n_val=20, n_vars=27, window=50, n_classes=10, batch_size=8
):
    """Create tiny train/val DataLoaders for classification testing."""
    X_train = torch.randn(n_train, window, n_vars)
    y_train = torch.randint(0, n_classes, (n_train,))
    X_val = torch.randn(n_val, window, n_vars)
    y_val = torch.randint(0, n_classes, (n_val,))

    # Wrap in DataLoader with metadata (empty dicts)
    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i], {}

    train_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=batch_size)
    val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


class TestTrainer:
    """Tests for the Trainer class."""

    def test_fit_runs_without_error(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        train_loader, val_loader = _make_classification_loaders()
        trainer = Trainer(device="cpu")
        history = trainer.fit(
            model, train_loader, val_loader, max_epochs=3, patience=10
        )
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 3

    def test_history_has_cost(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        train_loader, val_loader = _make_classification_loaders()
        trainer = Trainer(device="cpu")
        history = trainer.fit(model, train_loader, val_loader, max_epochs=2)
        assert "cost" in history
        assert history["cost"]["param_count"] > 0
        assert history["cost"]["wall_time_seconds"] >= 0  # tiny model may round to 0

    def test_early_stopping_triggers(self) -> None:
        es = EarlyStopping(patience=3)
        # Simulate non-improving val_loss
        for i in range(10):
            stopped = es.step(1.0, i)
            if stopped:
                break
        assert stopped
        assert es.counter == 3

    def test_checkpoint_save_load(self, tmp_path) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        optimizer = model.configure_optimizers()

        # Get predictions before save
        x = torch.randn(2, 50, 27)
        batch = (x, torch.zeros(2, dtype=torch.long), [{}] * 2)
        pred_before = model.predict(batch).clone()

        # Save
        ckpt_path = Trainer.save_checkpoint(model, optimizer, epoch=5, path=tmp_path)
        assert ckpt_path.exists()

        # Modify model weights (simulate continued training)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        # Load
        epoch = Trainer.load_checkpoint(model, optimizer, tmp_path)
        assert epoch == 5

        # Predictions should match pre-save
        pred_after = model.predict(batch)
        torch.testing.assert_close(pred_before, pred_after)

    def test_forecasting_trainer(self) -> None:
        model = DummyModel(task="forecasting", n_vars=10, horizon=7)

        class ForecastDS(torch.utils.data.Dataset):
            def __init__(self, n):
                self.X = torch.randn(n, 30, 10)
                self.y = torch.randn(n, 7)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, i):
                return self.X[i], self.y[i], {}

        train_loader = DataLoader(ForecastDS(20), batch_size=4)
        val_loader = DataLoader(ForecastDS(10), batch_size=4)
        trainer = Trainer(device="cpu")
        history = trainer.fit(model, train_loader, val_loader, max_epochs=2)
        assert len(history["train_loss"]) == 2

    def test_classification_checkpoint_metric_tracks_f1_macro(self) -> None:
        model = DummyModel(task="classification", n_vars=10, n_classes=3)
        train_loader, val_loader = _make_classification_loaders(
            n_vars=10, n_classes=3, batch_size=4
        )
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 4,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                    "checkpoint_metric": "f1_macro",
                    "checkpoint_mode": "max",
                }
            }
        )
        trainer = Trainer(cfg=cfg, device="cpu")
        history = trainer.fit(model, train_loader, val_loader)

        assert history["best_metric_name"] == "f1_macro"
        assert history["best_metric_mode"] == "max"
        assert history["best_metric"] is not None
        assert "f1_macro" in history["val_metrics"][0]


class TestCostTracker:
    """Tests for CostTracker."""

    def test_tracks_wall_time(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        with CostTracker(model) as tracker:
            time.sleep(0.01)
        assert tracker.results["wall_time_seconds"] > 0

    def test_tracks_param_count(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        with CostTracker(model) as tracker:
            pass
        assert tracker.results["param_count"] > 0


# ═══════════════════════════════════════════════════════════════════
# ExperimentRunner Tests
# ═══════════════════════════════════════════════════════════════════


def _make_tiny_dataset(
    task="classification", n=40, n_vars=10, window=20, n_classes=3, horizon=5
):
    """Create a tiny in-memory dataset for experiment testing."""

    class TinyDS(torch.utils.data.Dataset):
        def __init__(self):
            self.X = torch.randn(n, window, n_vars)
            if task == "classification":
                self.y = torch.randint(0, n_classes, (n,))
            elif task == "forecasting":
                self.y = torch.randn(n, horizon)
            elif task == "anomaly":
                self.y = self.X.clone()  # reconstruction target

        def __len__(self):
            return n

        def __getitem__(self, i):
            metadata = (
                {"instance_id": f"instance_{i // 2}"}
                if task == "classification"
                else {}
            )
            return self.X[i], self.y[i], metadata

    return TinyDS()


class _ScoreAwareDataset(torch.utils.data.Dataset):
    """Tiny classification dataset with deterministic per-sample logits."""

    def __init__(self) -> None:
        self.codes = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
        self.targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
        self.instance_ids = ["inst_a", "inst_b", "inst_c", "inst_d", "inst_e", "inst_f"]

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, i: int):
        x = self.codes[i].view(1, 1).repeat(2, 1)
        return x, self.targets[i], {"instance_id": self.instance_ids[i]}


class _ScoreAwareModel(BaseModel):
    """Classification model with fixed logits for metric-plumbing tests."""

    def __init__(self) -> None:
        super().__init__(task="classification", n_vars=1)
        self.n_classes = 3
        self.dummy = nn.Parameter(torch.tensor(0.0))
        self.logit_table = torch.tensor(
            [
                [4.0, 1.0, 0.5],
                [3.0, 2.9, 2.8],
                [0.5, 1.0, 4.0],
                [3.5, 1.0, 0.5],
                [1.0, 4.0, 0.5],
                [2.8, 2.9, 2.7],
            ],
            dtype=torch.float32,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes = x[:, 0, 0].long()
        return self.logit_table[codes] + (self.dummy * 0.0)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        return torch.optim.SGD([self.dummy], lr=0.0)


class _ZeroShotForecastModel(BaseModel):
    """Zero-shot model that must be moved to device but not fitted."""

    is_zero_shot = True
    fit_calls = 0
    to_calls = 0

    def __init__(self, task="forecasting", n_vars=10, horizon=5) -> None:
        super().__init__(task=task, n_vars=n_vars)
        self.horizon = horizon
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.horizon, device=x.device) + self.dummy

    def to(self, *args, **kwargs):
        type(self).to_calls += 1
        return super().to(*args, **kwargs)

    def training_step(self, batch):
        type(self).fit_calls += 1
        return torch.tensor(0.0, requires_grad=True)

    def predict(self, batch):
        return self.forward(batch[0])

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        return torch.optim.SGD([self.dummy], lr=0.0)


class _GroupedForecastDataset(torch.utils.data.Dataset):
    """Forecasting dataset with group metadata for MASE plumbing tests."""

    def __init__(self, include_order: bool = True, order_key: str = "target_start") -> None:
        self.X = torch.randn(12, 4, 2)
        self.y = torch.arange(12, dtype=torch.float32).view(12, 1)
        self.groups = np.array(["a"] * 6 + ["b"] * 6)
        self.include_order = include_order
        self.order_key = order_key

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        metadata = {"well_id": self.groups[i]}
        if self.include_order:
            metadata[self.order_key] = i
        return self.X[i], self.y[i], metadata


class TestExperimentRunner:
    """Tests for the ExperimentRunner orchestration."""

    def test_normalized_subset_uses_train_stats(self) -> None:
        """Validation subset must use training mean/std, not its own."""
        data = [
            (torch.tensor([[float(i)]]), torch.tensor(float(i)), {}) for i in range(20)
        ]
        train_idx = list(range(10))
        val_idx = list(range(10, 20))

        train_vals = torch.stack([data[i][0] for i in train_idx])
        mean = train_vals.mean(dim=0)
        std = train_vals.std(dim=0)

        train_subset = NormalizedSubset(data, train_idx, mean, std)
        val_subset = NormalizedSubset(data, val_idx, mean, std)

        train_item = train_subset[0][0]
        val_item = val_subset[0][0]
        expected = (torch.tensor([[10.0]]) - mean) / std

        assert train_item.shape == torch.Size([1, 1])
        assert torch.allclose(val_item, expected, atol=1e-5)

    def test_classification_experiment(self) -> None:
        ds = _make_tiny_dataset("classification", n_classes=3)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
                "mlflow": {"tracking_uri": "mlruns", "experiment_prefix": "test"},
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
        )
        results = runner.run(use_mlflow=False)
        assert len(results["fold_results"]) == 1
        assert "metrics" in results["fold_results"][0]
        assert "f1_macro" in results["fold_results"][0]["metrics"]

    def test_runtime_adjustments_are_in_experiment_results(self) -> None:
        ds = _make_tiny_dataset("classification", n_classes=3)

        class EmptyCV:
            def get_splits(self, _n_samples):
                return []

        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=EmptyCV(),
            cfg=OmegaConf.create({"training": {"batch_size": 8, "max_epochs": 1}}),
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
            runtime_adjustments={"patchtst_short_window": {"patch_len": {"from": 16, "to": 14}}},
        )

        results = runner.run(use_mlflow=False)

        assert results["runtime_adjustments"]["patchtst_short_window"]["patch_len"] == {
            "from": 16,
            "to": 14,
        }

    def test_forecasting_experiment(self) -> None:
        ds = _make_tiny_dataset("forecasting", horizon=5)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "forecasting", "n_vars": 10, "horizon": 5},
        )
        results = runner.run(use_mlflow=False)
        assert "mae" in results["fold_results"][0]["metrics"]

    def test_forecasting_experiment_threads_grouped_mase_metadata(self) -> None:
        ds = _GroupedForecastDataset()
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 4,
                    "max_epochs": 1,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "forecasting", "n_vars": 2, "horizon": 1},
        )

        results = runner.run(use_mlflow=False)
        metrics = results["fold_results"][0]["metrics"]

        assert "mase_flat" in metrics
        assert "mase_group_macro" in metrics
        assert "mase_group_weighted" in metrics
        assert metrics["mase_aggregation"] == "group_weighted"
        assert metrics["mase_denominator_source"] == "grouped_train"

    def test_forecasting_experiment_marks_mase_unavailable_when_order_missing(self) -> None:
        ds = _GroupedForecastDataset(include_order=False)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 4,
                    "max_epochs": 1,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "forecasting", "n_vars": 2, "horizon": 1},
        )

        results = runner.run(use_mlflow=False)
        metrics = results["fold_results"][0]["metrics"]

        assert "mae" in metrics
        assert "rmse" in metrics
        assert np.isnan(metrics["mase"])
        assert metrics["mase_aggregation"] == "unavailable"

    def test_extract_temporal_order_prefers_target_start_over_start_idx(self) -> None:
        metadata = {"target_start": torch.tensor([10, 11]), "start_idx": torch.tensor([1, 2])}

        order = ExperimentRunner._extract_order_ids(metadata)

        np.testing.assert_array_equal(order, np.array([10, 11], dtype=object))

    def test_extract_temporal_order_accepts_start_idx_fallback(self) -> None:
        metadata = {"start_idx": torch.tensor([1, 2, 3])}

        order = ExperimentRunner._extract_order_ids(metadata)

        np.testing.assert_array_equal(order, np.array([1, 2, 3], dtype=object))

    def test_anomaly_experiment(self) -> None:
        ds = _make_tiny_dataset("anomaly")
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "anomaly", "n_vars": 10, "window_size": 20},
        )
        results = runner.run(use_mlflow=False)
        assert "error_mean" in results["fold_results"][0]["metrics"]

    def test_zero_shot_experiment_skips_fit_and_returns_zero_epoch_history(self) -> None:
        _ZeroShotForecastModel.fit_calls = 0
        _ZeroShotForecastModel.to_calls = 0
        ds = _make_tiny_dataset("forecasting", horizon=5)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 3,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=_ZeroShotForecastModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "forecasting", "n_vars": 10, "horizon": 5},
        )

        results = runner.run(use_mlflow=False)
        history = results["fold_results"][0]["history"]

        assert _ZeroShotForecastModel.fit_calls == 0
        assert _ZeroShotForecastModel.to_calls >= 1
        assert history["epochs_run"] == 0
        assert history["best_epoch"] is None
        assert history["final_train_loss"] is None
        assert history["final_val_loss"] is None

    def test_aggregate_metrics(self) -> None:
        ds = _make_tiny_dataset("classification", n_classes=3, n=60)

        cv = ExpandingWindowCV(n_splits=2, min_train_ratio=0.4)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
        )
        results = runner.run(use_mlflow=False)
        assert "f1_macro_mean" in results["aggregate"]
        assert "f1_macro_std" in results["aggregate"]

    def test_mlflow_logging(self, tmp_path) -> None:
        """MLflow runs are logged when enabled."""
        ds = _make_tiny_dataset("classification", n_classes=3)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
                "mlflow": {
                    "tracking_uri": str(tmp_path / "mlruns"),
                    "experiment_prefix": "test",
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
        )
        runner.run(use_mlflow=True)
        # Check MLflow directory was created
        mlruns_dir = tmp_path / "mlruns"
        assert mlruns_dir.exists()

    def test_mlflow_metric_value_accepts_only_finite_real_numbers(self) -> None:
        assert ExperimentRunner._is_mlflow_metric_value(1)
        assert ExperimentRunner._is_mlflow_metric_value(1.5)
        assert ExperimentRunner._is_mlflow_metric_value(np.float64(2.0))

        for value in (True, False, np.nan, np.inf, "x", [1], {"a": 1}):
            assert not ExperimentRunner._is_mlflow_metric_value(value)

    def test_mlflow_metric_logging_sends_provenance_to_params(self) -> None:
        class FakeMLflow:
            def __init__(self) -> None:
                self.metrics = []
                self.params = []

            def log_metric(self, key, value):
                self.metrics.append((key, value))

            def log_param(self, key, value):
                self.params.append((key, value))

        fake = FakeMLflow()
        ExperimentRunner._log_mlflow_values(
            fake,
            "val",
            {
                "mae": 1.25,
                "mase": np.nan,
                "mase_aggregation": "unavailable",
                "mase_denominator_source": "missing_order",
                "stopped_early": True,
                "confusion_matrix": [[1, 0], [0, 1]],
            },
        )

        assert fake.metrics == [("val_mae", 1.25)]
        assert ("val_mase_aggregation", "unavailable") in fake.params
        assert ("val_mase_denominator_source", "missing_order") in fake.params
        assert ("val_stopped_early", "True") in fake.params
        assert not any(key == "val_mase" for key, _ in fake.metrics)
        assert not any(key == "val_confusion_matrix" for key, _ in fake.metrics)

    def test_mlflow_parent_and_nested_logging_helpers_skip_non_numeric(self) -> None:
        class FakeMLflow:
            def __init__(self) -> None:
                self.metrics = []
                self.params = []

            def log_metric(self, key, value):
                self.metrics.append((key, value))

            def log_param(self, key, value):
                self.params.append((key, value))

        fake = FakeMLflow()
        ExperimentRunner._log_mlflow_values(
            fake,
            "",
            {"mae_mean": 1.0, "mase_mean": np.nan, "mase_aggregation": "unavailable"},
        )
        ExperimentRunner._log_mlflow_values(
            fake,
            "test",
            {"rmse": np.float64(2.0), "mase_denominator_source": "missing_order"},
        )

        assert ("mae_mean", 1.0) in fake.metrics
        assert ("test_rmse", 2.0) in fake.metrics
        assert ("mase_aggregation", "unavailable") in fake.params
        assert ("test_mase_denominator_source", "missing_order") in fake.params
        assert not any(key == "mase_mean" for key, _ in fake.metrics)

    def test_nested_classification(self) -> None:
        """run_nested: inner CV + retrain + held-out test for classification."""
        ds = _make_tiny_dataset("classification", n=80, n_classes=3)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
        )

        # Simulate 80/20 holdout
        train_pool = np.arange(0, 64)
        test_indices = np.arange(64, 80)

        results = runner.run_nested(
            train_pool=train_pool,
            test_indices=test_indices,
            use_mlflow=False,
        )

        # Structure checks
        assert "test_metrics" in results
        assert "cv_aggregate" in results
        assert "cv_fold_results" in results
        assert "retrain_history" in results
        assert results["n_train"] == 64
        assert results["n_test"] == 16

        # Test metrics are scalar classification metrics
        tm = results["test_metrics"]
        assert "accuracy" in tm
        assert "f1_macro" in tm
        assert 0.0 <= tm["accuracy"] <= 1.0

    def test_nested_forecasting(self) -> None:
        """run_nested: inner CV + retrain + held-out test for forecasting."""
        ds = _make_tiny_dataset("forecasting", n=80, horizon=5)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "forecasting", "n_vars": 10, "horizon": 5},
        )

        train_pool = np.arange(0, 64)
        test_indices = np.arange(64, 80)

        results = runner.run_nested(
            train_pool=train_pool,
            test_indices=test_indices,
            use_mlflow=False,
        )

        assert "test_metrics" in results
        assert "mae" in results["test_metrics"]
        assert results["n_train"] == 64
        assert results["n_test"] == 16

    def test_nested_retrain_split_is_disjoint(self) -> None:
        train_idx, val_idx = ExperimentRunner._split_retrain_train_val(list(range(20)))
        assert set(train_idx).isdisjoint(set(val_idx))
        assert len(train_idx) == 18
        assert len(val_idx) == 2

    def test_classification_metrics_use_probability_scores_and_instance_ids(
        self,
    ) -> None:
        ds = _ScoreAwareDataset()
        cv = TemporalSplitCV(train_ratio=0.5)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 2,
                    "max_epochs": 1,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
            }
        )
        runner = ExperimentRunner(
            model_class=_ScoreAwareModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={},
        )

        results = runner.run(use_mlflow=False)
        metrics = results["fold_results"][0]["metrics"]

        val_targets = np.array([0, 1, 2])
        mean, std = NormalizedSubset.compute_stats(ds, [0, 1, 2])
        val_features = torch.stack([ds[i][0] for i in [3, 4, 5]])
        val_features = (val_features - mean) / std
        expected_model = _ScoreAwareModel()
        val_logits = expected_model.forward(val_features)
        val_preds = expected_model._extract_predictions(val_logits).detach().numpy()
        val_probs = (
            expected_model._extract_prediction_scores(val_logits).detach().numpy()
        )
        val_instance_ids = np.array(["inst_d", "inst_e", "inst_f"])
        expected = MetricRegistry.compute(
            "classification",
            val_preds,
            val_targets,
            prediction_scores=val_probs,
            instance_ids=val_instance_ids,
        )

        assert metrics["auc_pr"] == pytest.approx(expected["auc_pr"], rel=1e-5)
        assert metrics["edr"] == pytest.approx(expected["edr"], rel=1e-5)


# ═══════════════════════════════════════════════════════════════════
# Optuna Tests
# ═══════════════════════════════════════════════════════════════════


class TestOptunaIntegration:
    """Tests for Optuna HPO integration."""

    def test_create_study(self, tmp_path) -> None:
        cfg = OmegaConf.create(
            {
                "optuna": {
                    "storage": f"sqlite:///{tmp_path}/test.db",
                    "pruner": "median",
                },
            }
        )
        study = create_study(cfg, "test_study")
        assert study.study_name == "test_study"

    def test_convergence_callback_stops(self, tmp_path) -> None:
        import optuna

        # Use optimize() so study.stop() works inside the callback context
        cb = convergence_callback(patience=3, threshold=0.001)
        counter = {"n": 0}

        def objective(trial):
            counter["n"] += 1
            return 1.0  # constant — no improvement

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, callbacks=[cb])

        # Should stop well before 20 trials due to convergence
        assert counter["n"] < 20

    def test_run_hpo(self, tmp_path) -> None:
        ds = _make_tiny_dataset("classification", n=40, n_classes=3)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create(
            {
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "early_stopping_patience": 5,
                    "gradient_clip_val": 1.0,
                },
                "optuna": {
                    "storage": f"sqlite:///{tmp_path}/optuna.db",
                    "pruner": "median",
                    "n_trials_min": 2,
                    "convergence_patience": 20,
                    "convergence_threshold": 0.005,
                },
            }
        )
        result = run_hpo(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
            search_space={
                "lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True}
            },
            n_trials=2,
            study_name="test_hpo",
        )
        assert result["n_trials_completed"] == 2
        assert "best_value" in result
        assert "best_params" in result

    def test_optuna_objective_reports_fold_intermediates(self) -> None:
        class FakeTrial:
            def __init__(self):
                self.reports = []

            def report(self, value, step):
                self.reports.append((value, step))

            def should_prune(self):
                return False

        objective = OptunaObjective(
            model_class=DummyModel,
            dataset=_make_tiny_dataset("classification", n=10, n_classes=3),
            cv_strategy=TemporalSplitCV(train_ratio=0.7),
            base_cfg=OmegaConf.create({}),
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
            primary_metric="f1_macro",
        )
        trial = FakeTrial()
        callback = objective._make_fold_callback(trial)
        callback(0, [{"metrics": {"f1_macro": 0.25}, "history": {}}])

        assert trial.reports == [(0.25, 0)]


# ═══════════════════════════════════════════════════════════════════
# LR Scheduler Tests
# ═══════════════════════════════════════════════════════════════════


class TestLRSchedulers:
    """Verify Trainer._build_scheduler creates all 3 scheduler types."""

    @pytest.fixture
    def trainer(self):
        return Trainer(device="cpu")

    @pytest.fixture
    def optimizer(self):
        model = DummyModel(task="classification", n_vars=10, n_classes=3)
        return torch.optim.Adam(model.parameters(), lr=0.01)

    def test_onecycle_scheduler(self, trainer, optimizer):
        sched = trainer._build_scheduler(
            optimizer, "onecycle", max_epochs=10, steps_per_epoch=5, cfg_t=None
        )
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)

    def test_cosine_scheduler(self, trainer, optimizer):
        sched = trainer._build_scheduler(
            optimizer, "cosine", max_epochs=10, steps_per_epoch=5, cfg_t=None
        )
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_reduce_on_plateau_scheduler(self, trainer, optimizer):
        sched = trainer._build_scheduler(
            optimizer, "reduce_on_plateau", max_epochs=10, steps_per_epoch=5, cfg_t=None
        )
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_none_scheduler(self, trainer, optimizer):
        sched = trainer._build_scheduler(
            optimizer, None, max_epochs=10, steps_per_epoch=5, cfg_t=None
        )
        assert sched is None

    def test_unknown_scheduler_returns_none(self, trainer, optimizer):
        sched = trainer._build_scheduler(
            optimizer, "nonexistent", max_epochs=10, steps_per_epoch=5, cfg_t=None
        )
        assert sched is None


# ═══════════════════════════════════════════════════════════════════
# NaN Loss Handling Test
# ═══════════════════════════════════════════════════════════════════


class TestNaNLossHandling:
    """Verify Trainer skips batches with NaN/Inf loss without crashing."""

    def test_nan_loss_batch_is_skipped(self):
        class NaNModel(DummyModel):
            """DummyModel that returns NaN loss on the first batch."""
            def __init__(self):
                super().__init__(task="classification", n_vars=10, n_classes=3)
                self._call_count = 0

            def training_step(self, batch):
                self._call_count += 1
                if self._call_count == 1:
                    return torch.tensor(float("nan"), requires_grad=True)
                return super().training_step(batch)

        model = NaNModel()
        ds = torch.utils.data.TensorDataset(
            torch.randn(20, 50, 10),
            torch.randint(0, 3, (20,)),
        )
        # Wrap to add metadata tuple element
        class WrappedDS(torch.utils.data.Dataset):
            def __init__(self, tds):
                self.tds = tds
            def __len__(self):
                return len(self.tds)
            def __getitem__(self, idx):
                x, y = self.tds[idx]
                return x, y, {}

        wrapped = WrappedDS(ds)
        train_loader = DataLoader(wrapped, batch_size=4)
        val_loader = DataLoader(wrapped, batch_size=4)

        cfg = OmegaConf.create({"training": {
            "max_epochs": 2, "batch_size": 4,
            "early_stopping_patience": 10, "gradient_clip_val": 1.0,
        }})

        trainer = Trainer(cfg=cfg, device="cpu")
        history = trainer.fit(model, train_loader, val_loader)

        # Training should complete without crashing
        assert history["epochs_run"] == 2
        # At least some batches should have produced finite losses
        assert len(history["train_loss"]) == 2
        assert all(np.isfinite(l) for l in history["train_loss"])
