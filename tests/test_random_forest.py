"""Tests for Random Forest 3W classification pipeline."""

from __future__ import annotations

import json
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier


class TestFlattenShape:
    """Verify (14, 27) → 378-dim flattening."""

    def test_flatten_14x27(self):
        x = np.random.randn(14, 27).astype(np.float32)
        flat = x.reshape(-1)
        assert flat.shape == (378,)

    def test_flatten_preserves_values(self):
        x = np.arange(14 * 27, dtype=np.float32).reshape(14, 27)
        flat = x.reshape(-1)
        assert flat[0] == 0.0
        assert flat[-1] == 14 * 27 - 1

    def test_batch_flatten(self):
        X = np.random.randn(32, 14, 27).astype(np.float32)
        X_flat = X.reshape(32, -1)
        assert X_flat.shape == (32, 378)


class TestFitPredictShape:
    """Verify RandomForestClassifier works on 378-dim features."""

    @pytest.fixture
    def data(self):
        np.random.seed(42)
        X = np.random.randn(200, 378).astype(np.float32)
        y = np.random.randint(0, 10, size=200)
        return X, y

    def test_predict_shape(self, data):
        X, y = data
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X[:150], y[:150])
        preds = clf.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_proba_shape(self, data):
        X, y = data
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X[:150], y[:150])
        probs = clf.predict_proba(X[150:])
        assert probs.shape[0] == 50
        # n_classes may be < 10 due to random sampling
        assert probs.shape[1] <= 10

    def test_class_weight_balanced(self, data):
        X, y = data
        clf = RandomForestClassifier(
            n_estimators=10, class_weight="balanced", random_state=42
        )
        clf.fit(X[:150], y[:150])
        preds = clf.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_values_valid(self, data):
        X, y = data
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X[:150], y[:150])
        preds = clf.predict(X[150:])
        assert all(0 <= p <= 9 for p in preds)


class TestMetricComputation:
    """Verify MetricRegistry works with RF outputs."""

    def test_classification_keys(self):
        from offshore_dl.evaluation.metrics import MetricRegistry
        preds = np.array([0, 1, 2, 0, 1])
        targets = np.array([0, 1, 1, 0, 2])
        metrics = MetricRegistry.compute("classification", preds, targets)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics

    def test_accuracy_nonnegative(self):
        from offshore_dl.evaluation.metrics import MetricRegistry
        preds = np.array([0, 1, 2, 0, 1])
        targets = np.array([0, 1, 1, 0, 2])
        metrics = MetricRegistry.compute("classification", preds, targets)
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestJSONSchema:
    """Verify expected JSON output structure."""

    @pytest.fixture
    def sample_result(self):
        return {
            "test_metrics": {
                "accuracy": 0.85,
                "f1_macro": 0.82,
                "f1_weighted": 0.84,
                "auc_pr": 0.78,
                "edr": 1.0,
            },
            "cv_aggregate": {
                "accuracy_mean": 0.83,
                "accuracy_std": 0.02,
                "f1_macro_mean": 0.80,
                "f1_macro_std": 0.03,
            },
            "cv_fold_results": [
                {"fold_idx": 0, "metrics": {"accuracy": 0.85, "f1_macro": 0.83}},
                {"fold_idx": 1, "metrics": {"accuracy": 0.81, "f1_macro": 0.77}},
            ],
            "n_train": 100,
            "n_test": 25,
            "n_cv_folds": 5,
        }

    def test_required_top_level_keys(self, sample_result):
        required = {"test_metrics", "cv_aggregate", "cv_fold_results", "n_train", "n_test", "n_cv_folds"}
        assert required.issubset(sample_result.keys())

    def test_json_serializable(self, sample_result):
        s = json.dumps(sample_result)
        parsed = json.loads(s)
        assert parsed["test_metrics"]["accuracy"] == 0.85

    def test_cv_fold_results_structure(self, sample_result):
        for fold in sample_result["cv_fold_results"]:
            assert "fold_idx" in fold
            assert "metrics" in fold
            assert isinstance(fold["metrics"], dict)


class TestConfig:
    """Verify YAML config loads correctly."""

    def test_config_loads(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/models/random_forest.yaml")
        assert cfg.model.name == "random_forest"
        assert cfg.model.architecture.n_estimators == 500
        assert cfg.model.architecture.class_weight == "balanced"

    def test_optuna_search_space(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load("configs/models/random_forest.yaml")
        ss = cfg.model.optuna_search_space
        assert "n_estimators" in ss
        assert "max_depth" in ss
        assert ss.n_estimators.type == "int"
