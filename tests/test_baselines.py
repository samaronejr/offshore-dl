"""Tests for naive baseline models."""

from __future__ import annotations

import numpy as np
import pytest

from offshore_dl.evaluation.baselines import (
    MajorityClassBaseline,
    MeanReconstructionBaseline,
    SeasonalNaiveBaseline,
)


# ═══════════════════════════════════════════════════════════════════
# MajorityClassBaseline Tests
# ═══════════════════════════════════════════════════════════════════


class TestMajorityClassBaseline:
    """Tests for majority class prediction."""

    def test_predicts_majority_class(self) -> None:
        y_train = np.array([0, 0, 0, 1, 1, 2])  # 0 is majority
        baseline = MajorityClassBaseline()
        baseline.fit(np.zeros(len(y_train)), y_train)
        preds = baseline.predict(np.zeros(5))
        assert np.all(preds == 0)

    def test_all_same_class(self) -> None:
        y_train = np.array([3, 3, 3, 3])
        baseline = MajorityClassBaseline()
        baseline.fit(np.zeros(4), y_train)
        preds = baseline.predict(np.zeros(10))
        assert np.all(preds == 3)

    def test_predict_before_fit_raises(self) -> None:
        baseline = MajorityClassBaseline()
        with pytest.raises(RuntimeError):
            baseline.predict(np.zeros(5))

    def test_output_shape(self) -> None:
        baseline = MajorityClassBaseline()
        baseline.fit(np.zeros(10), np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 0]))
        preds = baseline.predict(np.zeros(7))
        assert preds.shape == (7,)


# ═══════════════════════════════════════════════════════════════════
# SeasonalNaiveBaseline Tests
# ═══════════════════════════════════════════════════════════════════


class TestSeasonalNaiveBaseline:
    """Tests for seasonal naive forecasting."""

    def test_repeats_period(self) -> None:
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        baseline = SeasonalNaiveBaseline(period=7)
        baseline.fit(np.zeros(7), y_train)
        preds = baseline.predict(np.zeros(14))
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        np.testing.assert_array_almost_equal(preds, expected)

    def test_uses_last_period(self) -> None:
        y_train = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
        baseline = SeasonalNaiveBaseline(period=3)
        baseline.fit(np.zeros(6), y_train)
        preds = baseline.predict(np.zeros(3))
        np.testing.assert_array_almost_equal(preds, [1.0, 2.0, 3.0])

    def test_predict_before_fit_raises(self) -> None:
        baseline = SeasonalNaiveBaseline()
        with pytest.raises(RuntimeError):
            baseline.predict(np.zeros(5))

    def test_output_length(self) -> None:
        baseline = SeasonalNaiveBaseline(period=7)
        baseline.fit(np.zeros(30), np.arange(30, dtype=float))
        preds = baseline.predict(np.zeros(15))
        assert len(preds) == 15


# ═══════════════════════════════════════════════════════════════════
# MeanReconstructionBaseline Tests
# ═══════════════════════════════════════════════════════════════════


class TestMeanReconstructionBaseline:
    """Tests for mean reconstruction anomaly baseline."""

    def test_predicts_mean(self) -> None:
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        baseline = MeanReconstructionBaseline()
        baseline.fit(X_train)
        preds = baseline.predict(np.zeros((2, 2)))
        expected_mean = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(preds[0], expected_mean)
        np.testing.assert_array_almost_equal(preds[1], expected_mean)

    def test_predict_before_fit_raises(self) -> None:
        baseline = MeanReconstructionBaseline()
        with pytest.raises(RuntimeError):
            baseline.predict(np.zeros((5, 3)))

    def test_output_shape(self) -> None:
        X_train = np.random.randn(100, 48, 11)  # (n_samples, window, n_vars)
        baseline = MeanReconstructionBaseline()
        baseline.fit(X_train)
        X_val = np.random.randn(20, 48, 11)
        preds = baseline.predict(X_val)
        assert preds.shape == X_val.shape

    def test_3d_input(self) -> None:
        """CDF-like 3D input (n_samples, window_size, n_vars)."""
        X_train = np.ones((50, 48, 11)) * 5.0
        baseline = MeanReconstructionBaseline()
        baseline.fit(X_train)
        preds = baseline.predict(np.zeros((10, 48, 11)))
        assert preds.shape == (10, 48, 11)
        # Mean over all samples and timesteps, per var → 5.0
        np.testing.assert_array_almost_equal(preds[0, 0, :], np.full(11, 5.0))
