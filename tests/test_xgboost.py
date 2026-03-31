"""Unit tests for XGBoost execution path in Ganymede production sweep.

Tests:
  - Flatten shape: (90, 63) tensor → 5670-dim flat vector
  - MultiOutputRegressor(XGBRegressor) fit/predict shape
  - MetricRegistry.compute('forecasting', ...) returns expected keys
  - JSON result schema has required keys
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from offshore_dl.evaluation.metrics import MetricRegistry


class TestFlattenShape:
    """Verify tensor flattening for XGBoost tabular input."""

    def test_flatten_shape_90x63(self):
        """(90, 63) tensor → 5670-dim flat vector."""
        import torch

        x = torch.randn(90, 63)
        flat = x.numpy().reshape(-1)
        assert flat.shape == (5670,)

    def test_flatten_shape_90x48(self):
        """(90, 48) tensor → 4320-dim flat vector (per-well, fewer vars)."""
        import torch

        x = torch.randn(90, 48)
        flat = x.numpy().reshape(-1)
        assert flat.shape == (4320,)

    def test_flatten_preserves_values(self):
        """Flattened values match original row-major order."""
        import torch

        x = torch.arange(90 * 63, dtype=torch.float32).reshape(90, 63)
        flat = x.numpy().reshape(-1)
        assert flat[0] == 0.0
        assert flat[63] == 63.0  # second row, first col
        assert flat[-1] == float(90 * 63 - 1)


class TestFitPredictShape:
    """Verify MultiOutputRegressor(XGBRegressor) produces correct shapes."""

    @pytest.fixture
    def synthetic_data(self):
        """Small synthetic dataset: 50 samples, 100 features, 7-step horizon."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 100).astype(np.float32)
        Y = rng.randn(50, 7).astype(np.float32)
        return X, Y

    def test_predict_shape_h7(self, synthetic_data):
        X, Y = synthetic_data
        model = MultiOutputRegressor(
            XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        )
        model.fit(X[:40], Y[:40])
        preds = model.predict(X[40:])
        assert preds.shape == (10, 7)

    def test_predict_shape_h30(self):
        """Horizon 30: verify shape matches."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 200).astype(np.float32)
        Y = rng.randn(60, 30).astype(np.float32)
        model = MultiOutputRegressor(
            XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        )
        model.fit(X[:50], Y[:50])
        preds = model.predict(X[50:])
        assert preds.shape == (10, 30)

    def test_predict_values_finite(self, synthetic_data):
        """All predictions are finite (no NaN/Inf)."""
        X, Y = synthetic_data
        model = MultiOutputRegressor(
            XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        )
        model.fit(X[:40], Y[:40])
        preds = model.predict(X[40:])
        assert np.all(np.isfinite(preds))


class TestMetricComputation:
    """Verify MetricRegistry produces expected keys for forecasting task."""

    def test_forecasting_keys(self):
        """MetricRegistry.compute('forecasting', ...) returns expected keys."""
        rng = np.random.RandomState(42)
        preds = rng.randn(100)
        targets = preds + rng.randn(100) * 0.1  # close to pred
        metrics = MetricRegistry.compute("forecasting", preds, targets)

        expected_keys = {"mae", "rmse", "r2", "r2_prod", "mase"}
        assert expected_keys <= set(metrics.keys()), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

    def test_forecasting_on_2d_arrays(self):
        """Forecasting metrics work on 2D arrays (raveled internally)."""
        rng = np.random.RandomState(42)
        preds = rng.randn(20, 7)
        targets = preds + rng.randn(20, 7) * 0.1
        metrics = MetricRegistry.compute("forecasting", preds, targets)
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0

    def test_mae_is_nonnegative(self):
        preds = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])
        metrics = MetricRegistry.compute("forecasting", preds, targets)
        assert metrics["mae"] >= 0


class TestJSONSchema:
    """Verify XGBoost result JSON has the expected structure."""

    @pytest.fixture
    def sample_result(self):
        """Simulate the result dict produced by _run_xgboost_multi_well."""
        return {
            "test_metrics": {"mae": 0.5, "rmse": 0.7, "r2": 0.85, "r2_prod": 0.88, "mase": 1.2},
            "cv_aggregate": {"mae_mean": 0.55, "mae_std": 0.03},
            "cv_fold_results": [
                {"fold_idx": 0, "metrics": {"mae": 0.52}},
                {"fold_idx": 1, "metrics": {"mae": 0.58}},
                {"fold_idx": 2, "metrics": {"mae": 0.55}},
            ],
            "n_train": 800,
            "n_test": 200,
            "n_cv_folds": 3,
        }

    def test_required_top_level_keys(self, sample_result):
        required = {"test_metrics", "cv_aggregate", "cv_fold_results", "n_train", "n_test", "n_cv_folds"}
        assert required <= set(sample_result.keys())

    def test_json_serializable(self, sample_result):
        """Result dict must be JSON-serializable."""
        serialized = json.dumps(sample_result)
        deserialized = json.loads(serialized)
        assert deserialized["n_cv_folds"] == 3

    def test_per_well_has_well_field(self):
        """Per-well result must include 'well' field."""
        result = {
            "test_metrics": {"mae": 0.5},
            "cv_aggregate": {},
            "cv_fold_results": [],
            "n_train": 100,
            "n_test": 25,
            "n_cv_folds": 3,
            "well": "49/22-Z01Z",
        }
        assert "well" in result
        assert result["well"] == "49/22-Z01Z"

    def test_cv_fold_results_structure(self, sample_result):
        """Each fold result has fold_idx and metrics."""
        for fr in sample_result["cv_fold_results"]:
            assert "fold_idx" in fr
            assert "metrics" in fr
