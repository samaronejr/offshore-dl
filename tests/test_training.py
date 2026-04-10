"""Tests for training engine: BaseModel, Trainer, ExperimentRunner."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel, model_summary
from offshore_dl.models.dummy import DummyModel


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
        torch.testing.assert_close(scores.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=0.0)


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

from torch.utils.data import DataLoader, TensorDataset

from offshore_dl.training.trainer import CostTracker, EarlyStopping, Trainer


def _make_classification_loaders(n_train=40, n_val=20, n_vars=27, window=50, n_classes=10, batch_size=8):
    """Create tiny train/val DataLoaders for classification testing."""
    X_train = torch.randn(n_train, window, n_vars)
    y_train = torch.randint(0, n_classes, (n_train,))
    X_val = torch.randn(n_val, window, n_vars)
    y_val = torch.randint(0, n_classes, (n_val,))

    # Wrap in DataLoader with metadata (empty dicts)
    class SimpleDS(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i], {}

    train_loader = DataLoader(SimpleDS(X_train, y_train), batch_size=batch_size)
    val_loader = DataLoader(SimpleDS(X_val, y_val), batch_size=batch_size)
    return train_loader, val_loader


class TestTrainer:
    """Tests for the Trainer class."""

    def test_fit_runs_without_error(self) -> None:
        model = DummyModel(task="classification", n_vars=27, n_classes=10)
        train_loader, val_loader = _make_classification_loaders()
        trainer = Trainer(device="cpu")
        history = trainer.fit(model, train_loader, val_loader, max_epochs=3, patience=10)
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
            def __len__(self): return len(self.X)
            def __getitem__(self, i): return self.X[i], self.y[i], {}

        train_loader = DataLoader(ForecastDS(20), batch_size=4)
        val_loader = DataLoader(ForecastDS(10), batch_size=4)
        trainer = Trainer(device="cpu")
        history = trainer.fit(model, train_loader, val_loader, max_epochs=2)
        assert len(history["train_loss"]) == 2


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


import time


# ═══════════════════════════════════════════════════════════════════
# ExperimentRunner Tests
# ═══════════════════════════════════════════════════════════════════

from offshore_dl.evaluation.cv import TemporalSplitCV
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.training.experiment import ExperimentRunner, NormalizedSubset


def _make_tiny_dataset(task="classification", n=40, n_vars=10, window=20, n_classes=3, horizon=5):
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
        def __len__(self): return n
        def __getitem__(self, i):
            metadata = {"instance_id": f"instance_{i // 2}"} if task == "classification" else {}
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
        self.logit_table = torch.tensor([
            [4.0, 1.0, 0.5],
            [3.0, 2.9, 2.8],
            [0.5, 1.0, 4.0],
            [3.5, 1.0, 0.5],
            [1.0, 4.0, 0.5],
            [2.8, 2.9, 2.7],
        ], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        codes = x[:, 0, 0].long()
        return self.logit_table[codes] + (self.dummy * 0.0)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        return torch.optim.SGD([self.dummy], lr=0.0)


class TestExperimentRunner:
    """Tests for the ExperimentRunner orchestration."""

    def test_classification_experiment(self) -> None:
        ds = _make_tiny_dataset("classification", n_classes=3)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
            "mlflow": {"tracking_uri": "mlruns", "experiment_prefix": "test"},
        })
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

    def test_forecasting_experiment(self) -> None:
        ds = _make_tiny_dataset("forecasting", horizon=5)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "forecasting", "n_vars": 10, "horizon": 5},
        )
        results = runner.run(use_mlflow=False)
        assert "mae" in results["fold_results"][0]["metrics"]

    def test_anomaly_experiment(self) -> None:
        ds = _make_tiny_dataset("anomaly")
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "anomaly", "n_vars": 10, "window_size": 20},
        )
        results = runner.run(use_mlflow=False)
        assert "error_mean" in results["fold_results"][0]["metrics"]

    def test_aggregate_metrics(self) -> None:
        ds = _make_tiny_dataset("classification", n_classes=3, n=60)
        from offshore_dl.evaluation.cv import ExpandingWindowCV
        cv = ExpandingWindowCV(n_splits=2, min_train_ratio=0.4)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
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
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
            "mlflow": {"tracking_uri": str(tmp_path / "mlruns"), "experiment_prefix": "test"},
        })
        runner = ExperimentRunner(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
        )
        results = runner.run(use_mlflow=True)
        # Check MLflow directory was created
        mlruns_dir = tmp_path / "mlruns"
        assert mlruns_dir.exists()

    def test_nested_classification(self) -> None:
        """run_nested: inner CV + retrain + held-out test for classification."""
        import numpy as np

        ds = _make_tiny_dataset("classification", n=80, n_classes=3)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2,
                         "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
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
        import numpy as np

        ds = _make_tiny_dataset("forecasting", n=80, horizon=5)
        cv = TemporalSplitCV(train_ratio=0.7)
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2,
                         "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
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

    def test_classification_metrics_use_probability_scores_and_instance_ids(self) -> None:
        ds = _ScoreAwareDataset()
        cv = TemporalSplitCV(train_ratio=0.5)
        cfg = OmegaConf.create({
            "training": {"batch_size": 2, "max_epochs": 1,
                         "early_stopping_patience": 5, "gradient_clip_val": 1.0},
        })
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
        val_probs = expected_model._extract_prediction_scores(val_logits).detach().numpy()
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


from omegaconf import OmegaConf


# ═══════════════════════════════════════════════════════════════════
# Optuna Tests
# ═══════════════════════════════════════════════════════════════════

from offshore_dl.training.optuna_utils import convergence_callback, create_study, run_hpo


class TestOptunaIntegration:
    """Tests for Optuna HPO integration."""

    def test_create_study(self, tmp_path) -> None:
        cfg = OmegaConf.create({
            "optuna": {"storage": f"sqlite:///{tmp_path}/test.db", "pruner": "median"},
        })
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
        cfg = OmegaConf.create({
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "gradient_clip_val": 1.0},
            "optuna": {
                "storage": f"sqlite:///{tmp_path}/optuna.db",
                "pruner": "median",
                "n_trials_min": 2,
                "convergence_patience": 20,
                "convergence_threshold": 0.005,
            },
        })
        result = run_hpo(
            model_class=DummyModel,
            dataset=ds,
            cv_strategy=cv,
            cfg=cfg,
            model_kwargs={"task": "classification", "n_vars": 10, "n_classes": 3},
            search_space={"lr": {"type": "float", "low": 0.001, "high": 0.1, "log": True}},
            n_trials=2,
            study_name="test_hpo",
        )
        assert result["n_trials_completed"] == 2
        assert "best_value" in result
        assert "best_params" in result
