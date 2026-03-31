"""Tests for pure stateless transform functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from offshore_dl.data.transforms import (
    apply_zscore,
    causal_forward_fill,
    compute_class_weights,
    compute_zscore_stats,
    detect_frozen_values,
    sliding_window_segmentation,
)


class TestDetectFrozenValues:
    """Frozen value detection via rolling variance."""

    def test_constant_series_becomes_nan(self) -> None:
        """A constant column should have frozen values replaced with NaN."""
        df = pd.DataFrame({"sensor": [5.0] * 100})
        result = detect_frozen_values(df, window=10, columns=["sensor"])
        # After rolling window fills (first 9 are NaN from rolling), the rest are frozen
        assert result["sensor"].isna().sum() > 0

    def test_varying_series_unchanged(self) -> None:
        """A column with variance should remain unchanged."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"sensor": rng.randn(100)})
        result = detect_frozen_values(df, window=10, columns=["sensor"])
        # No values should be replaced
        assert result["sensor"].notna().sum() == 100

    def test_mixed_frozen_and_varying(self) -> None:
        """A series with both frozen and varying segments."""
        data = list(np.random.randn(50)) + [3.0] * 50
        df = pd.DataFrame({"sensor": data})
        result = detect_frozen_values(df, window=10, columns=["sensor"])
        # The frozen part should have NaN, the varying part should not
        assert result["sensor"].iloc[:50].notna().sum() == 50
        assert result["sensor"].iloc[50:].isna().sum() > 0

    def test_preserves_other_columns(self) -> None:
        """Non-target columns should not be modified."""
        df = pd.DataFrame({"sensor": [5.0] * 100, "other": range(100)})
        result = detect_frozen_values(df, window=10, columns=["sensor"])
        pd.testing.assert_series_equal(result["other"], df["other"])


class TestCausalForwardFill:
    """Forward fill with temporal limit."""

    def test_short_gap_filled(self) -> None:
        """Gaps within the limit should be filled."""
        data = [1.0, np.nan, np.nan, 4.0]
        df = pd.DataFrame({"sensor": data})
        result = causal_forward_fill(df, limit=3, columns=["sensor"])
        assert result["sensor"].isna().sum() == 0
        assert result["sensor"].iloc[1] == 1.0  # filled with previous value

    def test_long_gap_not_filled(self) -> None:
        """Gaps exceeding the limit should remain NaN."""
        data = [1.0] + [np.nan] * 5 + [7.0]
        df = pd.DataFrame({"sensor": data})
        result = causal_forward_fill(df, limit=3, columns=["sensor"])
        # First 3 NaN filled, last 2 remain NaN
        assert result["sensor"].iloc[1] == 1.0  # filled
        assert result["sensor"].iloc[2] == 1.0  # filled
        assert result["sensor"].iloc[3] == 1.0  # filled
        assert pd.isna(result["sensor"].iloc[4])  # not filled
        assert pd.isna(result["sensor"].iloc[5])  # not filled

    def test_causal_no_backward_fill(self) -> None:
        """Values should only propagate forward, never backward."""
        data = [np.nan, np.nan, 3.0, np.nan]
        df = pd.DataFrame({"sensor": data})
        result = causal_forward_fill(df, limit=10, columns=["sensor"])
        assert pd.isna(result["sensor"].iloc[0])  # leading NaN stays
        assert pd.isna(result["sensor"].iloc[1])  # leading NaN stays
        assert result["sensor"].iloc[2] == 3.0
        assert result["sensor"].iloc[3] == 3.0  # forward filled


class TestZScore:
    """Z-score normalization with separate stats computation."""

    def test_normalized_mean_std(self) -> None:
        """After z-score, mean ≈ 0 and std ≈ 1."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"a": rng.randn(1000) * 10 + 5, "b": rng.randn(1000) * 2 - 3})
        stats = compute_zscore_stats(df, columns=["a", "b"])
        result = apply_zscore(df, stats)
        assert abs(result["a"].mean()) < 0.1
        assert abs(result["a"].std() - 1.0) < 0.1
        assert abs(result["b"].mean()) < 0.1
        assert abs(result["b"].std() - 1.0) < 0.1

    def test_stats_are_tuples(self) -> None:
        """Stats should be (mean, std) tuples."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        stats = compute_zscore_stats(df, columns=["x"])
        assert "x" in stats
        assert len(stats["x"]) == 2
        assert isinstance(stats["x"][0], float)

    def test_constant_column_std_one(self) -> None:
        """A constant column should get std=1 to avoid division by zero."""
        df = pd.DataFrame({"const": [5.0] * 100})
        stats = compute_zscore_stats(df, columns=["const"])
        assert stats["const"][1] == 1.0

    def test_no_leakage_between_partitions(self) -> None:
        """Stats from train should be applied to test without recomputation."""
        rng = np.random.RandomState(42)
        train = pd.DataFrame({"x": rng.randn(100)})
        test = pd.DataFrame({"x": rng.randn(50) * 5 + 10})  # different dist

        stats = compute_zscore_stats(train)
        result = apply_zscore(test, stats)
        # Test data should NOT have mean≈0 std≈1 — it uses train stats
        assert abs(result["x"].mean()) > 1.0  # shifted by train normalization


class TestSlidingWindowSegmentation:
    """Sliding window segmentation."""

    def test_correct_window_count(self) -> None:
        """Number of windows matches the formula."""
        T, n_vars = 100, 5
        w, s = 20, 10
        values = np.random.randn(T, n_vars)
        windows = sliding_window_segmentation(values, w, s)
        expected = (T - w) // s + 1
        assert len(windows) == expected

    def test_start_end_correct(self) -> None:
        """Each window has correct start and end indices."""
        values = np.random.randn(50, 3)
        windows = sliding_window_segmentation(values, window_size=10, stride=5)
        for win in windows:
            assert win["end"] - win["start"] == 10
            assert win["end"] <= 50

    def test_labels_majority_vote(self) -> None:
        """Window label should be the majority class within the window."""
        values = np.zeros((20, 2))
        labels = np.array([0] * 15 + [1] * 5)
        windows = sliding_window_segmentation(values, window_size=20, stride=20, labels=labels)
        assert len(windows) == 1
        assert windows[0]["label"] == 0  # majority is class 0

    def test_too_short_series(self) -> None:
        """Series shorter than window size should return empty list."""
        values = np.random.randn(5, 3)
        windows = sliding_window_segmentation(values, window_size=10, stride=5)
        assert windows == []


class TestComputeClassWeights:
    """Inverse-frequency class weights."""

    def test_balanced_classes_equal_weights(self) -> None:
        """Perfectly balanced classes should have equal weights."""
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
        weights = compute_class_weights(labels)
        assert len(weights) == 3
        for w in weights.values():
            assert abs(w - 1.0) < 0.01

    def test_imbalanced_rare_class_higher(self) -> None:
        """Rare classes should get higher weights."""
        labels = np.array([0] * 900 + [1] * 100)
        weights = compute_class_weights(labels)
        assert weights[1] > weights[0]

    def test_handles_nan_labels(self) -> None:
        """NaN labels should be excluded from weight computation."""
        labels = np.array([0, 0, 1, 1, np.nan, np.nan])
        weights = compute_class_weights(labels)
        assert len(weights) == 2
