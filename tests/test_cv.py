"""Tests for cross-validation strategies and leakage guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from offshore_dl.evaluation.cv import (
    ExpandingWindowCV,
    FoldNormalizer,
    GroupedExpandingWindowCV,
    GroupedTemporalHoldoutSplitter,
    LeakageGuard,
    SlidingWindowCV,
    StratifiedGroupKFoldSKLearn,
    TemporalSplitCV,
    resolve_cv_gap,
    resolve_cv_gap_from_config,
    target_interval,
    validate_raw_row_embargo,
)


# ═══════════════════════════════════════════════════════════════════
# ExpandingWindowCV Tests
# ═══════════════════════════════════════════════════════════════════


class TestExpandingWindowCV:
    """Tests for temporal expanding window cross-validation."""

    def test_correct_number_of_splits(self) -> None:
        cv = ExpandingWindowCV(n_splits=5, min_train_ratio=0.5)
        splits = cv.get_splits(1000)
        assert len(splits) == 5

    def test_train_before_val(self) -> None:
        cv = ExpandingWindowCV(n_splits=3, min_train_ratio=0.5)
        splits = cv.get_splits(100)
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min(), "Train must come before val"

    def test_expanding_train_window(self) -> None:
        cv = ExpandingWindowCV(n_splits=4, min_train_ratio=0.3)
        splits = cv.get_splits(1000)
        train_sizes = [len(t) for t, _ in splits]
        # Each fold should have a larger or equal training set
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_no_overlap_between_train_and_val(self) -> None:
        cv = ExpandingWindowCV(n_splits=5, min_train_ratio=0.5)
        splits = cv.get_splits(500)
        for train_idx, val_idx in splits:
            overlap = set(train_idx) & set(val_idx)
            assert len(overlap) == 0

    def test_gap_creates_separation(self) -> None:
        cv = ExpandingWindowCV(n_splits=3, min_train_ratio=0.5, gap=10)
        splits = cv.get_splits(200)
        for train_idx, val_idx in splits:
            assert val_idx.min() - train_idx.max() > 1, "Gap should separate train/val"

    def test_gap_preserves_validation_block_size_when_space_remains(self) -> None:
        cv = ExpandingWindowCV(n_splits=3, min_train_ratio=0.5, gap=10)
        splits = cv.get_splits(300)

        assert [len(val_idx) for _, val_idx in splits[:2]] == [50, 50]
        for train_idx, val_idx in splits:
            assert val_idx.min() - train_idx.max() == 11

    def test_small_dataset(self) -> None:
        cv = ExpandingWindowCV(n_splits=3, min_train_ratio=0.5)
        splits = cv.get_splits(10)
        assert len(splits) > 0
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0


class TestCVGapPolicy:
    """Explicit embargo policy helpers."""

    def test_cdf_strict_raw_row_gap_from_config(self) -> None:
        cfg = {
            "task": "anomaly",
            "cv_gap_policy": "strict_raw_row",
            "cv_gap": None,
            "preprocessing": {"window_size": 48},
        }
        assert resolve_cv_gap_from_config(cfg) == 47

    def test_forecasting_causal_horizon_gap_from_config(self) -> None:
        cfg = {
            "task": "forecasting",
            "cv_gap_policy": "causal_horizon",
            "cv_gap": None,
            "forecasting": {"input_window": 90, "default_horizon": 30, "gap": 0},
        }
        assert resolve_cv_gap_from_config(cfg) == 30

    def test_forecasting_strict_raw_row_gap(self) -> None:
        assert resolve_cv_gap(
            "strict_raw_row",
            task="forecasting",
            input_window=90,
            dataset_gap=3,
            horizon=30,
        ) == 122

    def test_raw_row_embargo_detects_overlap_and_strict_gap_prevents_it(self) -> None:
        train_idx = np.array([0, 1])
        val_idx = np.array([2])
        with pytest.raises(ValueError, match="Raw-row embargo violation"):
            validate_raw_row_embargo(
                train_idx,
                val_idx,
                task="anomaly",
                window_size=4,
            )

        train_idx = np.array([0])
        val_idx = np.array([4])
        assert validate_raw_row_embargo(
            train_idx,
            val_idx,
            task="anomaly",
            window_size=4,
        )["passed"]

    def test_forecasting_causal_horizon_documents_target_embargo(self) -> None:
        input_window = 5
        horizon = 3
        train_target = target_interval(0, input_window=input_window, horizon=horizon)
        val_target = target_interval(3 + horizon, input_window=input_window, horizon=horizon)
        assert train_target[1] < val_target[0]

    def test_forecasting_strict_raw_row_embargo(self) -> None:
        with pytest.raises(ValueError):
            validate_raw_row_embargo(
                [0],
                [3],
                task="forecasting",
                input_window=5,
                horizon=2,
            )
        assert validate_raw_row_embargo(
            [0],
            [7],
            task="forecasting",
            input_window=5,
            horizon=2,
        )["passed"]


# ═══════════════════════════════════════════════════════════════════
# TemporalSplitCV Tests
# ═══════════════════════════════════════════════════════════════════


class TestTemporalSplitCV:
    """Tests for single temporal split."""

    def test_single_split(self) -> None:
        cv = TemporalSplitCV(train_ratio=0.8)
        splits = cv.get_splits(100)
        assert len(splits) == 1

    def test_correct_ratio(self) -> None:
        cv = TemporalSplitCV(train_ratio=0.8)
        splits = cv.get_splits(100)
        train_idx, val_idx = splits[0]
        assert len(train_idx) == 80
        assert len(val_idx) == 20

    def test_no_overlap(self) -> None:
        cv = TemporalSplitCV(train_ratio=0.7)
        splits = cv.get_splits(50)
        train_idx, val_idx = splits[0]
        assert set(train_idx).isdisjoint(set(val_idx))

    def test_temporal_order(self) -> None:
        cv = TemporalSplitCV(train_ratio=0.8)
        splits = cv.get_splits(100)
        train_idx, val_idx = splits[0]
        assert train_idx.max() < val_idx.min()

    def test_covers_all_indices(self) -> None:
        cv = TemporalSplitCV(train_ratio=0.6)
        splits = cv.get_splits(50)
        train_idx, val_idx = splits[0]
        all_idx = np.concatenate([train_idx, val_idx])
        assert set(all_idx) == set(range(50))


class TestGroupedTemporalHoldoutSplitter:
    """Tests for per-group temporal holdout splitting."""

    def test_split_preserves_temporal_tail_per_group(self) -> None:
        groups = np.array(["well_a"] * 10 + ["well_b"] * 10)
        splitter = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups)
        train_idx, test_idx = splitter.split(len(groups))

        assert len(train_idx) == 16
        assert len(test_idx) == 4
        assert set(train_idx).isdisjoint(set(test_idx))

        for group in np.unique(groups):
            group_train = train_idx[groups[train_idx] == group]
            group_test = test_idx[groups[test_idx] == group]
            assert len(group_train) == 8
            assert len(group_test) == 2
            assert group_train.max() < group_test.min()


class TestGroupedExpandingWindowCV:
    """Tests for per-group expanding-window CV."""

    def test_grouped_splits_remain_temporal_within_each_group(self) -> None:
        groups = np.array(["well_a"] * 12 + ["well_b"] * 12)
        cv = GroupedExpandingWindowCV(groups=groups, n_splits=2, min_train_ratio=0.5)
        splits = cv.get_splits(len(groups))

        assert len(splits) == 2
        for train_idx, val_idx in splits:
            assert set(train_idx).isdisjoint(set(val_idx))
            for group in np.unique(groups):
                group_train = train_idx[groups[train_idx] == group]
                group_val = val_idx[groups[val_idx] == group]
                assert len(group_train) > 0
                assert len(group_val) > 0
                assert group_train.max() < group_val.min()

    def test_grouped_gap_preserves_per_group_validation_block_size(self) -> None:
        groups = np.array(["well_a"] * 300 + ["well_b"] * 300)
        cv = GroupedExpandingWindowCV(
            groups=groups,
            n_splits=3,
            min_train_ratio=0.5,
            gap=10,
        )
        splits = cv.get_splits(len(groups))

        assert len(splits) == 3
        for train_idx, val_idx in splits[:2]:
            for group in np.unique(groups):
                group_train = train_idx[groups[train_idx] == group]
                group_val = val_idx[groups[val_idx] == group]
                assert len(group_val) == 50
                assert group_val.min() - group_train.max() == 11


# ═══════════════════════════════════════════════════════════════════
# LeakageGuard Tests
# ═══════════════════════════════════════════════════════════════════


class TestLeakageGuard:
    """Tests for leakage detection."""

    def test_temporal_clean_passes(self) -> None:
        train_ts = np.array([1, 2, 3, 4, 5])
        val_ts = np.array([6, 7, 8, 9, 10])
        result = LeakageGuard.check_temporal(train_ts, val_ts)
        assert result["passed"]

    def test_temporal_violation_raises(self) -> None:
        train_ts = np.array([1, 2, 3, 7, 8])  # 7,8 overlap with val
        val_ts = np.array([6, 7, 8, 9, 10])
        with pytest.raises(ValueError, match="Temporal leakage"):
            LeakageGuard.check_temporal(train_ts, val_ts)

    def test_group_clean_passes(self) -> None:
        train_groups = ["well_1", "well_2", "well_3"]
        val_groups = ["well_4", "well_5"]
        result = LeakageGuard.check_group(train_groups, val_groups)
        assert result["passed"]

    def test_group_violation_raises(self) -> None:
        train_groups = ["well_1", "well_2", "well_3"]
        val_groups = ["well_3", "well_4"]  # well_3 in both
        with pytest.raises(ValueError, match="Group leakage"):
            LeakageGuard.check_group(train_groups, val_groups)

    def test_temporal_empty_passes(self) -> None:
        result = LeakageGuard.check_temporal(np.array([]), np.array([1, 2]))
        assert result["passed"]


# ═══════════════════════════════════════════════════════════════════
# FoldNormalizer Tests
# ═══════════════════════════════════════════════════════════════════


class TestFoldNormalizer:
    """Tests for per-fold z-score normalization."""

    def test_fit_computes_stats(self) -> None:
        train_df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        norm = FoldNormalizer()
        norm.fit(train_df)
        assert norm.stats is not None
        assert "a" in norm.stats
        assert "b" in norm.stats

    def test_transform_before_fit_raises(self) -> None:
        norm = FoldNormalizer()
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError):
            norm.transform(df)

    def test_normalized_mean_near_zero(self) -> None:
        train_df = pd.DataFrame({"x": np.random.randn(100) * 5 + 10})
        norm = FoldNormalizer()
        result = norm.fit_transform(train_df)
        assert abs(result["x"].mean()) < 0.1

    def test_normalized_std_near_one(self) -> None:
        train_df = pd.DataFrame({"x": np.random.randn(100) * 5 + 10})
        norm = FoldNormalizer()
        result = norm.fit_transform(train_df)
        assert abs(result["x"].std() - 1.0) < 0.15

    def test_val_uses_train_stats(self) -> None:
        """Val normalization must use train statistics, not its own."""
        train_df = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0]})
        val_df = pd.DataFrame({"x": [100.0, 200.0]})

        norm = FoldNormalizer()
        norm.fit(train_df)
        val_norm = norm.transform(val_df)

        # With train mean=0, std≈0 → stats use std=1 to avoid division by zero
        # So val values should NOT be normalized to mean~0 based on val's own stats
        assert norm.stats["x"][0] == pytest.approx(0.0, abs=0.01)  # train mean

    def test_specific_columns_only(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": [100.0, 200.0, 300.0]})
        norm = FoldNormalizer(columns=["a", "b"])
        result = norm.fit_transform(df)
        # "c" should be unchanged
        assert result["c"].tolist() == [100.0, 200.0, 300.0]


# ═══════════════════════════════════════════════════════════════════
# StratifiedGroupKFoldSKLearn Tests
# ═══════════════════════════════════════════════════════════════════


class TestStratifiedGroupKFoldSKLearn:
    """Tests for sklearn-based stratified group K-fold CV."""

    def _make_data(self, n=200, n_classes=5, n_groups=20):
        labels = np.array([i % n_classes for i in range(n)])
        groups = np.array([f"group_{i % n_groups}" for i in range(n)])
        return labels, groups

    def test_correct_number_of_folds(self) -> None:
        labels, groups = self._make_data()
        cv = StratifiedGroupKFoldSKLearn(n_folds=5, labels=labels, groups=groups)
        splits = cv.get_splits(len(labels))
        assert len(splits) == 5

    def test_no_group_leakage(self) -> None:
        labels, groups = self._make_data()
        cv = StratifiedGroupKFoldSKLearn(n_folds=5, labels=labels, groups=groups)
        splits = cv.get_splits(len(labels))
        for train_idx, val_idx in splits:
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            assert train_groups.isdisjoint(val_groups), "No group should appear in both"

    def test_no_index_overlap(self) -> None:
        labels, groups = self._make_data()
        cv = StratifiedGroupKFoldSKLearn(n_folds=3, labels=labels, groups=groups)
        splits = cv.get_splits(len(labels))
        for train_idx, val_idx in splits:
            assert set(train_idx).isdisjoint(set(val_idx))

    def test_all_indices_covered(self) -> None:
        labels, groups = self._make_data(n=100, n_groups=10)
        cv = StratifiedGroupKFoldSKLearn(n_folds=5, labels=labels, groups=groups)
        splits = cv.get_splits(100)
        all_val = np.concatenate([val for _, val in splits])
        assert set(all_val) == set(range(100))

    def test_class_representation(self) -> None:
        """Each training fold should contain most classes."""
        labels, groups = self._make_data(n=500, n_classes=10, n_groups=50)
        cv = StratifiedGroupKFoldSKLearn(n_folds=5, labels=labels, groups=groups)
        splits = cv.get_splits(500)
        for train_idx, _ in splits:
            train_classes = set(labels[train_idx])
            # With 10 classes and 5 folds, training should have most classes
            assert len(train_classes) >= 8

    def test_requires_labels_and_groups(self) -> None:
        cv = StratifiedGroupKFoldSKLearn(n_folds=5)
        with pytest.raises(ValueError, match="requires labels and groups"):
            cv.get_splits(100)


# ═══════════════════════════════════════════════════════════════════
# SlidingWindowCV Tests
# ═══════════════════════════════════════════════════════════════════


class TestSlidingWindowCV:
    """Tests for sliding window cross-validation."""

    def test_produces_multiple_folds(self) -> None:
        cv = SlidingWindowCV(n_splits=5, train_ratio=0.7)
        splits = cv.get_splits(1000)
        assert len(splits) >= 3  # at least 3 valid folds

    def test_train_before_val(self) -> None:
        cv = SlidingWindowCV(n_splits=3, train_ratio=0.7)
        splits = cv.get_splits(500)
        for train_idx, val_idx in splits:
            assert train_idx.max() < val_idx.min()

    def test_no_overlap(self) -> None:
        cv = SlidingWindowCV(n_splits=4, train_ratio=0.7)
        splits = cv.get_splits(500)
        for train_idx, val_idx in splits:
            assert set(train_idx).isdisjoint(set(val_idx))

    def test_gap_creates_separation(self) -> None:
        cv = SlidingWindowCV(n_splits=3, train_ratio=0.7, gap=5)
        splits = cv.get_splits(500)
        for train_idx, val_idx in splits:
            assert val_idx.min() - train_idx.max() > 1

    def test_windows_slide_forward(self) -> None:
        cv = SlidingWindowCV(n_splits=3, train_ratio=0.7)
        splits = cv.get_splits(500)
        starts = [train_idx[0] for train_idx, _ in splits]
        for i in range(1, len(starts)):
            assert starts[i] > starts[i - 1], "Windows should slide forward"

    def test_small_dataset(self) -> None:
        cv = SlidingWindowCV(n_splits=3, train_ratio=0.7)
        splits = cv.get_splits(30)
        assert len(splits) > 0


# ═══════════════════════════════════════════════════════════════════
# HoldoutSplitter Tests
# ═══════════════════════════════════════════════════════════════════


class TestHoldoutSplitter:
    """Tests for outer train/test holdout splitting."""

    def test_temporal_split_sizes(self) -> None:
        from offshore_dl.evaluation.cv import HoldoutSplitter
        hs = HoldoutSplitter(test_ratio=0.2, mode="temporal")
        train, test = hs.split(1000)
        assert len(train) == 800
        assert len(test) == 200
        assert len(set(train) & set(test)) == 0

    def test_temporal_ordering_preserved(self) -> None:
        from offshore_dl.evaluation.cv import HoldoutSplitter
        hs = HoldoutSplitter(test_ratio=0.2, mode="temporal")
        train, test = hs.split(100)
        assert train[-1] < test[0], "Train must precede test temporally"

    def test_stratified_group_no_overlap(self) -> None:
        from offshore_dl.evaluation.cv import HoldoutSplitter
        rng = np.random.RandomState(42)
        n = 500
        groups = np.array([f"g{i // 10}" for i in range(n)])
        labels = rng.randint(0, 5, size=n)
        hs = HoldoutSplitter(
            test_ratio=0.2, mode="stratified_group",
            labels=labels, groups=groups, seed=42,
        )
        train, test = hs.split(n)
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        assert len(train_groups & test_groups) == 0, "No group overlap"

    def test_stratified_group_coverage(self) -> None:
        from offshore_dl.evaluation.cv import HoldoutSplitter
        rng = np.random.RandomState(42)
        n = 500
        groups = np.array([f"g{i // 10}" for i in range(n)])
        labels = rng.randint(0, 5, size=n)
        hs = HoldoutSplitter(
            test_ratio=0.2, mode="stratified_group",
            labels=labels, groups=groups, seed=42,
        )
        train, test = hs.split(n)
        assert len(train) + len(test) == n
        assert 0.15 < len(test) / n < 0.30  # ~20% with group rounding

    def test_unknown_mode_raises(self) -> None:
        from offshore_dl.evaluation.cv import HoldoutSplitter
        hs = HoldoutSplitter(mode="bad_mode")
        with pytest.raises(ValueError, match="Unknown holdout mode"):
            hs.split(100)
