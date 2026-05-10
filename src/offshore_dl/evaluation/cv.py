"""Cross-validation strategies with leakage prevention.

Provides CV split generators for all 3 experimental tracks:
- ``StratifiedGroupKFoldCV`` for 3W (existing benchmark folds or sklearn)
- ``ExpandingWindowCV`` for Ganymede (temporal expanding window)
- ``TemporalSplitCV`` for CDF (single temporal split)
- ``SlidingWindowCV`` for CDF (multiple temporal folds)

Also provides:
- ``HoldoutSplitter``: Creates outer train/test splits respecting groups
  or temporal ordering, for use in nested CV (inner CV on training pool,
  final evaluation on held-out test set).
- ``LeakageGuard`` for validating split integrity
- ``FoldNormalizer`` for per-fold z-score normalization.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from offshore_dl.data.transforms import apply_zscore, compute_zscore_stats

logger = logging.getLogger(__name__)


CV_GAP_POLICY_NONE = "none"
CV_GAP_POLICY_CAUSAL_HORIZON = "causal_horizon"
CV_GAP_POLICY_STRICT_RAW_ROW = "strict_raw_row"
SUPPORTED_CV_GAP_POLICIES = {
    CV_GAP_POLICY_NONE,
    CV_GAP_POLICY_CAUSAL_HORIZON,
    CV_GAP_POLICY_STRICT_RAW_ROW,
}


def resolve_cv_gap(
    policy: str,
    *,
    task: str,
    input_window: int | None = None,
    horizon: int | None = None,
    dataset_gap: int = 0,
    window_size: int | None = None,
    explicit_gap: int | str | None = None,
) -> int:
    """Resolve a temporal CV embargo gap from an explicit policy.

    ``causal_horizon`` prevents forecasting targets from touching training
    windows while allowing historical input context overlap. ``strict_raw_row``
    prevents any raw timestep used by a validation sample from appearing in a
    training sample.
    """
    explicit_gap = normalize_cv_gap(explicit_gap)
    if explicit_gap is not None:
        return explicit_gap

    if policy not in SUPPORTED_CV_GAP_POLICIES:
        msg = (
            f"Unsupported cv_gap_policy {policy!r}; expected one of "
            f"{sorted(SUPPORTED_CV_GAP_POLICIES)}"
        )
        raise ValueError(msg)

    if policy == CV_GAP_POLICY_NONE:
        return 0

    if task == "anomaly":
        if policy != CV_GAP_POLICY_STRICT_RAW_ROW:
            msg = "anomaly CV currently supports only strict_raw_row or explicit cv_gap"
            raise ValueError(msg)
        if window_size is None:
            raise ValueError("strict_raw_row anomaly CV requires window_size")
        return max(0, int(window_size) - 1)

    if task == "forecasting":
        if horizon is None:
            raise ValueError("forecasting CV gap policy requires horizon")
        if policy == CV_GAP_POLICY_CAUSAL_HORIZON:
            return int(horizon)
        if policy == CV_GAP_POLICY_STRICT_RAW_ROW:
            if input_window is None:
                raise ValueError("strict_raw_row forecasting CV requires input_window")
            return max(0, int(input_window) + int(dataset_gap) + int(horizon) - 1)

    msg = f"Unsupported task/policy combination: task={task!r}, policy={policy!r}"
    raise ValueError(msg)


def normalize_cv_gap(explicit_gap: int | str | None) -> int | None:
    """Normalize explicit ``cv_gap`` config values.

    ``None`` and ``"auto"`` mean "defer to policy".  Numeric strings are
    accepted for CLI/dotlist compatibility.  Negative values are invalid in
    every caller.
    """
    if explicit_gap is None or explicit_gap == "auto":
        return None
    gap = int(explicit_gap)
    if gap < 0:
        raise ValueError("cv_gap must be non-negative")
    return gap


def resolve_cv_gap_from_config(data_cfg) -> int:
    """Resolve ``data.cv_gap``/``data.cv_gap_policy`` semantics from config.

    Intended factory wiring:
    ``gap=resolve_cv_gap_from_config(cfg.data)`` when constructing temporal
    forecasting/CDF CV strategies.
    """
    def _get(name: str, default=None):
        if isinstance(data_cfg, dict):
            return data_cfg.get(name, default)
        return getattr(data_cfg, name, default)

    task = _get("task")
    explicit_gap = _get("cv_gap", None)

    forecasting = _get("forecasting", {}) or {}
    preprocessing = _get("preprocessing", {}) or {}

    def _nested(container, name: str, default=None):
        if isinstance(container, dict):
            return container.get(name, default)
        return getattr(container, name, default)

    horizon = _nested(forecasting, "default_horizon", None)
    input_window = _nested(forecasting, "input_window", None)
    dataset_gap = _nested(forecasting, "gap", 0) or 0
    window_size = _nested(preprocessing, "window_size", None)

    policy = _get(
        "cv_gap_policy",
        CV_GAP_POLICY_STRICT_RAW_ROW if task == "anomaly" else CV_GAP_POLICY_CAUSAL_HORIZON,
    )
    return resolve_cv_gap(
        policy,
        task=task,
        input_window=input_window,
        horizon=horizon,
        dataset_gap=dataset_gap,
        window_size=window_size,
        explicit_gap=explicit_gap,
    )


def raw_row_interval(
    sample_index: int,
    *,
    task: str,
    input_window: int | None = None,
    horizon: int | None = None,
    dataset_gap: int = 0,
    window_size: int | None = None,
) -> tuple[int, int]:
    """Return inclusive raw-row interval consumed by a sample."""
    start = int(sample_index)
    if task == "anomaly":
        if window_size is None:
            raise ValueError("anomaly raw-row interval requires window_size")
        return start, start + int(window_size) - 1
    if task == "forecasting":
        if input_window is None or horizon is None:
            raise ValueError("forecasting raw-row interval requires input_window and horizon")
        return start, start + int(input_window) + int(dataset_gap) + int(horizon) - 1
    raise ValueError(f"Unsupported task: {task!r}")


def target_interval(
    sample_index: int,
    *,
    input_window: int,
    horizon: int,
    dataset_gap: int = 0,
) -> tuple[int, int]:
    """Return inclusive forecasting target interval for a sample."""
    start = int(sample_index) + int(input_window) + int(dataset_gap)
    return start, start + int(horizon) - 1


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True when inclusive integer intervals overlap."""
    return max(a[0], b[0]) <= min(a[1], b[1])


def validate_raw_row_embargo(
    train_idx: np.ndarray | list[int],
    val_idx: np.ndarray | list[int],
    *,
    task: str,
    input_window: int | None = None,
    horizon: int | None = None,
    dataset_gap: int = 0,
    window_size: int | None = None,
) -> dict:
    """Validate that train/validation samples share no raw timesteps."""
    train_intervals = [
        raw_row_interval(
            int(i),
            task=task,
            input_window=input_window,
            horizon=horizon,
            dataset_gap=dataset_gap,
            window_size=window_size,
        )
        for i in train_idx
    ]
    val_intervals = [
        raw_row_interval(
            int(i),
            task=task,
            input_window=input_window,
            horizon=horizon,
            dataset_gap=dataset_gap,
            window_size=window_size,
        )
        for i in val_idx
    ]
    violations = sum(
        1
        for train_interval in train_intervals
        for val_interval in val_intervals
        if intervals_overlap(train_interval, val_interval)
    )
    if violations:
        raise ValueError(f"Raw-row embargo violation: {violations} overlapping intervals")
    return {"passed": True, "violations": 0}


class BaseCVStrategy(ABC):
    """Abstract base for cross-validation strategies.

    Subclasses implement ``get_splits()`` which returns a list of
    ``(train_indices, val_indices)`` tuples, each as numpy arrays.
    """

    @abstractmethod
    def get_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return train/val index splits.

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            List of (train_idx_array, val_idx_array) tuples.
        """


class StratifiedGroupKFoldCV(BaseCVStrategy):
    """3W benchmark folds from ``folds_clf_02.csv``.

    Maps instance-level fold assignments to dataset sample indices.
    Each instance maps to multiple sliding windows — all windows from
    the same instance go to the same fold.

    Args:
        folds_path: Path to folds CSV (columns: instancia, fold, is_ova).
        dataset: ThreeWDataset instance for mapping instances to sample indices.
        n_folds: Number of CV folds (default 5, folds 0-4).
    """

    def __init__(
        self,
        folds_path: str | Path,
        dataset,
        n_folds: int = 5,
    ) -> None:
        self.folds_path = Path(folds_path)
        self.dataset = dataset
        self.n_folds = n_folds

        self._folds_df = pd.read_csv(self.folds_path)
        # Parse class_id from instance path (e.g. "2/WELL-00012_xxx.csv" → 2)
        self._folds_df["class_id"] = (
            self._folds_df["instancia"].str.split("/").str[0].astype(int)
        )

    def _build_instance_to_fold_map(self) -> dict[str, int]:
        """Map instance identifiers to fold assignments.

        The folds CSV uses paths like ``2/WELL-00012_20170320143144.csv``.
        The dataset metadata has ``instance_id`` which is the parquet stem
        (e.g., ``WELL-00012_20170320143144``). Match by extracting the stem.
        """
        inst_to_fold = {}
        for _, row in self._folds_df.iterrows():
            fold = int(row["fold"])
            if fold < 0:
                continue  # skip holdout (fold=-1)
            # Extract stem: "2/WELL-00012_20170320143144.csv" → "WELL-00012_20170320143144"
            stem = row["instancia"].split("/")[-1].replace(".csv", "")
            inst_to_fold[stem] = fold
        return inst_to_fold

    def get_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return 5-fold train/val splits mapped from folds_clf_02.csv.

        Samples whose instance_id isn't in the folds file (e.g., classes 4,8,9
        or holdout instances) are excluded from all splits.
        """
        inst_to_fold = self._build_instance_to_fold_map()

        # Map each dataset sample to its fold
        sample_folds = np.full(n_samples, -1, dtype=np.int32)
        for idx in range(n_samples):
            # Fast-path: read instance_id from _windows dict (avoids full tensor I/O)
            if hasattr(self.dataset, "_windows"):
                instance_id = self.dataset._windows[idx].get("instance_id", "")
            else:
                _, _, meta = self.dataset[idx]
                instance_id = meta.get("instance_id", "")
            if instance_id in inst_to_fold:
                sample_folds[idx] = inst_to_fold[instance_id]

        splits = []
        for fold_id in range(self.n_folds):
            # Val = samples in this fold; Train = samples in other folds (excluding -1)
            val_mask = sample_folds == fold_id
            train_mask = (sample_folds >= 0) & (sample_folds != fold_id)

            val_idx = np.where(val_mask)[0]
            train_idx = np.where(train_mask)[0]

            if len(val_idx) == 0 or len(train_idx) == 0:
                logger.warning("Fold %d has empty train or val set, skipping", fold_id)
                continue

            splits.append((train_idx, val_idx))

        logger.info(
            "StratifiedGroupKFoldCV: %d folds, %d mapped samples / %d total",
            len(splits),
            int((sample_folds >= 0).sum()),
            n_samples,
        )
        return splits


class StratifiedGroupKFoldSKLearn(BaseCVStrategy):
    """Well-stratified K-fold CV using sklearn's StratifiedGroupKFold.

    Guarantees:
      1. No ``group`` (well/instance) appears in both train and val.
      2. Class distribution is approximately preserved in each fold.
      3. Works with all 10 classes (unlike ``StratifiedGroupKFoldCV`` which
         depends on the external ``folds_clf_02.csv`` covering only 7 classes).

    Args:
        n_folds: Number of CV folds (default 5).
        labels: Array of class labels (one per sample).
        groups: Array of group identifiers — typically ``instance_id``
            (all windows from the same instance stay together).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_folds: int = 5,
        labels: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.labels = labels
        self.groups = groups
        self.seed = seed

    def get_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return stratified group K-fold splits.

        Requires ``labels`` and ``groups`` to be set (either in __init__
        or before calling this method).
        """
        from sklearn.model_selection import StratifiedGroupKFold

        if self.labels is None or self.groups is None:
            msg = "StratifiedGroupKFoldSKLearn requires labels and groups"
            raise ValueError(msg)

        sgkf = StratifiedGroupKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed,
        )

        X_dummy = np.zeros(n_samples)
        splits = []
        for train_idx, val_idx in sgkf.split(X_dummy, self.labels, self.groups):
            splits.append((train_idx, val_idx))

        # Log class and group distribution per fold
        for i, (train_idx, val_idx) in enumerate(splits):
            train_classes = set(self.labels[train_idx])
            val_classes = set(self.labels[val_idx])
            train_groups = set(self.groups[train_idx])
            val_groups = set(self.groups[val_idx])
            group_overlap = train_groups & val_groups
            logger.info(
                "  Fold %d: train=%d (%d classes, %d groups), "
                "val=%d (%d classes, %d groups), group_overlap=%d",
                i, len(train_idx), len(train_classes), len(train_groups),
                len(val_idx), len(val_classes), len(val_groups),
                len(group_overlap),
            )

        logger.info(
            "StratifiedGroupKFoldSKLearn: %d folds, %d samples, %d groups",
            len(splits), n_samples, len(set(self.groups)),
        )
        return splits


class ExpandingWindowCV(BaseCVStrategy):
    """Expanding window cross-validation for temporal forecasting.

    Training window grows with each fold while validation window stays
    constant. All train indices < all val indices (causal guarantee).

    Args:
        n_splits: Number of CV folds.
        min_train_ratio: Minimum fraction of data for the first training window.
        gap: Number of samples to skip between train and val (prevent leakage
            from overlapping windows).
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_ratio: float = 0.5,
        gap: int = 0,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.gap = gap

    def get_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return expanding-window splits.

        The data is divided into n_splits+1 blocks after the minimum
        training portion. Each fold trains on all blocks up to fold i
        and validates on block i+1.
        """
        min_train = max(1, int(n_samples * self.min_train_ratio))
        remaining = n_samples - min_train
        block_size = max(1, remaining // (self.n_splits))

        splits = []
        for i in range(self.n_splits):
            train_end = min_train + i * block_size
            val_start = train_end + self.gap
            val_end = min(val_start + block_size, n_samples)

            if val_start >= n_samples or val_start >= val_end:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))

        logger.info(
            "ExpandingWindowCV: %d folds, n=%d, min_train=%d, block=%d, gap=%d",
            len(splits), n_samples, min_train, block_size, self.gap,
        )
        return splits


class GroupedExpandingWindowCV(BaseCVStrategy):
    """Expanding-window CV applied independently within each group.

    This is designed for multi-well forecasting datasets whose samples are
    stored as a concatenation of per-well temporal windows. A plain
    ``ExpandingWindowCV`` over the flattened sample index would incorrectly
    interpret well boundaries as temporal boundaries. Instead, this strategy
    performs the expanding split inside each group and then concatenates the
    per-group train/validation indices fold-by-fold.

    Args:
        groups: Group identifier for each sample (for example, ``well_idx``).
        n_splits: Number of expanding folds to request per group.
        min_train_ratio: Minimum fraction of each group reserved for the first
            training window.
        gap: Number of samples to skip between train and validation within each
            group.
    """

    def __init__(
        self,
        groups: np.ndarray | list,
        n_splits: int = 5,
        min_train_ratio: float = 0.5,
        gap: int = 0,
    ) -> None:
        self.groups = np.asarray(groups)
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.gap = gap

    def subset(self, indices: np.ndarray | list[int]) -> "GroupedExpandingWindowCV":
        """Return the same strategy restricted to a subset of samples."""
        indices = np.asarray(indices, dtype=np.int64)
        return GroupedExpandingWindowCV(
            groups=self.groups[indices],
            n_splits=self.n_splits,
            min_train_ratio=self.min_train_ratio,
            gap=self.gap,
        )

    def get_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return grouped expanding-window splits."""
        if len(self.groups) != n_samples:
            msg = (
                f"GroupedExpandingWindowCV groups length mismatch: "
                f"{len(self.groups)} != {n_samples}"
            )
            raise ValueError(msg)

        per_group_splits: list[tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = []
        unique_groups = pd.unique(self.groups)

        for group in unique_groups:
            group_idx = np.where(self.groups == group)[0]
            group_cv = ExpandingWindowCV(
                n_splits=self.n_splits,
                min_train_ratio=self.min_train_ratio,
                gap=self.gap,
            )
            local_splits = group_cv.get_splits(len(group_idx))
            if not local_splits:
                logger.warning(
                    "GroupedExpandingWindowCV: skipping group %r with 0 valid splits",
                    group,
                )
                continue
            per_group_splits.append((group_idx, local_splits))

        if not per_group_splits:
            return []

        n_effective_splits = min(len(local_splits) for _, local_splits in per_group_splits)

        splits = []
        for fold_idx in range(n_effective_splits):
            train_parts = []
            val_parts = []
            for group_idx, local_splits in per_group_splits:
                local_train, local_val = local_splits[fold_idx]
                train_parts.append(group_idx[local_train])
                val_parts.append(group_idx[local_val])

            train_idx = np.concatenate(train_parts)
            val_idx = np.concatenate(val_parts)
            train_idx.sort()
            val_idx.sort()
            splits.append((train_idx, val_idx))

        logger.info(
            "GroupedExpandingWindowCV: %d folds across %d groups, n=%d",
            len(splits), len(per_group_splits), n_samples,
        )
        return splits


class SlidingWindowCV(BaseCVStrategy):
    """Sliding window cross-validation for temporal anomaly detection.

    Fixed-size training and validation windows slide forward through time.
    Unlike ``ExpandingWindowCV`` (growing train), the training window size
    stays constant — appropriate for non-stationary data where older
    observations may be less relevant.

    Args:
        n_splits: Number of CV folds.
        train_ratio: Fraction of the window allocated to training (default 0.7).
        gap: Number of samples to skip between train and val (default 0).
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        gap: int = 0,
    ) -> None:
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap = gap

    def get_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return sliding-window splits.

        Each fold has a fixed-size training window followed by a gap and a
        validation window. The windows slide forward across the dataset.
        """
        # Total size per fold = train + gap + val
        # Distribute the data across n_splits folds with overlap
        fold_total = n_samples // (self.n_splits + 1)
        train_size = max(1, int(fold_total * self.train_ratio / (self.train_ratio + (1 - self.train_ratio))))
        val_size = max(1, fold_total - train_size - self.gap)
        step = max(1, (n_samples - train_size - self.gap - val_size) // max(1, self.n_splits - 1))

        splits = []
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_size
            val_start = train_end + self.gap
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))

        logger.info(
            "SlidingWindowCV: %d folds, n=%d, train_size=%d, val_size=%d, "
            "step=%d, gap=%d",
            len(splits), n_samples, train_size, val_size, step, self.gap,
        )
        return splits


class TemporalSplitCV(BaseCVStrategy):
    """Single temporal train/val split for small datasets (CDF).

    Args:
        train_ratio: Fraction of data for training (default 0.8).
    """

    def __init__(self, train_ratio: float = 0.8) -> None:
        self.train_ratio = train_ratio

    def get_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return a single split: first train_ratio for train, rest for val."""
        split_point = int(n_samples * self.train_ratio)
        split_point = max(1, min(split_point, n_samples - 1))

        train_idx = np.arange(0, split_point)
        val_idx = np.arange(split_point, n_samples)

        logger.info(
            "TemporalSplitCV: train=%d, val=%d, ratio=%.2f",
            len(train_idx), len(val_idx), self.train_ratio,
        )
        return [(train_idx, val_idx)]


# ═══════════════════════════════════════════════════════════════════
# Holdout splitters for nested CV (outer train/test split)
# ═══════════════════════════════════════════════════════════════════


class HoldoutSplitter:
    """Create an outer train/test split for proper nested evaluation.

    Returns global indices into the full dataset:
    - ``train_pool``: indices used for inner CV + final retraining
    - ``test_set``:   held-out indices never seen during training

    Two modes:
    - **Stratified group** (classification): splits at the group level
      so all samples from the same group stay together. Class distribution
      is approximately preserved via stratified group splitting.
    - **Temporal** (forecasting/anomaly): takes the last ``test_ratio``
      fraction of samples as the test set, preserving temporal ordering.

    Args:
        test_ratio: Fraction of data reserved for the held-out test set.
        mode: ``"stratified_group"`` or ``"temporal"``.
        labels: Per-sample class labels (required for stratified_group).
        groups: Per-sample group IDs (required for stratified_group).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        test_ratio: float = 0.2,
        mode: str = "stratified_group",
        labels: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        self.test_ratio = test_ratio
        self.mode = mode
        self.labels = labels
        self.groups = groups
        self.seed = seed

    def split(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_pool_indices, test_indices).

        Args:
            n_samples: Total number of samples in the dataset.

        Returns:
            Tuple of (train_pool, test_set) as numpy index arrays.
        """
        if self.mode == "stratified_group":
            return self._split_stratified_group(n_samples)
        elif self.mode == "temporal":
            return self._split_temporal(n_samples)
        else:
            msg = f"Unknown holdout mode: {self.mode!r}. Use 'stratified_group' or 'temporal'."
            raise ValueError(msg)

    def _split_stratified_group(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Stratified group split: groups stay together, classes balanced."""
        if self.labels is None or self.groups is None:
            msg = "stratified_group mode requires labels and groups"
            raise ValueError(msg)

        from sklearn.model_selection import StratifiedGroupKFold

        # Use StratifiedGroupKFold with n_splits = 1/test_ratio (rounded)
        # and take the first fold's val as the test set.
        n_splits = max(2, round(1.0 / self.test_ratio))
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.seed,
        )
        X_dummy = np.zeros(n_samples)

        # Take the first split: train = training pool, val = test set
        train_idx, test_idx = next(
            sgkf.split(X_dummy, self.labels, self.groups)
        )

        # Verify no group overlap
        train_groups = set(self.groups[train_idx])
        test_groups = set(self.groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            msg = f"Group leakage in holdout: {len(overlap)} groups in both train and test"
            raise ValueError(msg)

        logger.info(
            "HoldoutSplitter (stratified_group): train_pool=%d (%.1f%%), "
            "test=%d (%.1f%%), groups: train=%d, test=%d, overlap=%d",
            len(train_idx), 100 * len(train_idx) / n_samples,
            len(test_idx), 100 * len(test_idx) / n_samples,
            len(train_groups), len(test_groups), len(overlap),
        )
        return train_idx, test_idx

    def _split_temporal(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Temporal split: last test_ratio fraction as test set."""
        split_point = int(n_samples * (1.0 - self.test_ratio))
        split_point = max(1, min(split_point, n_samples - 1))

        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point, n_samples)

        logger.info(
            "HoldoutSplitter (temporal): train_pool=%d (%.1f%%), "
            "test=%d (%.1f%%)",
            len(train_idx), 100 * len(train_idx) / n_samples,
            len(test_idx), 100 * len(test_idx) / n_samples,
        )
        return train_idx, test_idx


class GroupedTemporalHoldoutSplitter:
    """Temporal holdout applied independently within each group.

    Splits each group into an early training segment and a late held-out test
    segment, then concatenates the resulting indices across groups. This keeps
    the holdout temporal for every well instead of for the accidental
    well-concatenated sample order.

    Args:
        test_ratio: Fraction of each group assigned to the held-out test tail.
        groups: Group identifier for each sample.
    """

    def __init__(
        self,
        test_ratio: float = 0.2,
        groups: np.ndarray | list | None = None,
    ) -> None:
        self.test_ratio = test_ratio
        self.groups = None if groups is None else np.asarray(groups)

    def split(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Return grouped temporal (train_pool, test_set) indices."""
        if self.groups is None:
            raise ValueError("GroupedTemporalHoldoutSplitter requires groups")
        if len(self.groups) != n_samples:
            msg = (
                f"GroupedTemporalHoldoutSplitter groups length mismatch: "
                f"{len(self.groups)} != {n_samples}"
            )
            raise ValueError(msg)

        train_parts = []
        test_parts = []
        for group in pd.unique(self.groups):
            group_idx = np.where(self.groups == group)[0]
            if len(group_idx) < 2:
                logger.warning(
                    "GroupedTemporalHoldoutSplitter: skipping group %r with <2 samples",
                    group,
                )
                continue

            split_point = int(len(group_idx) * (1.0 - self.test_ratio))
            split_point = max(1, min(split_point, len(group_idx) - 1))

            train_parts.append(group_idx[:split_point])
            test_parts.append(group_idx[split_point:])

        if not train_parts or not test_parts:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        train_idx.sort()
        test_idx.sort()

        logger.info(
            "GroupedTemporalHoldoutSplitter: train_pool=%d (%.1f%%), test=%d (%.1f%%), groups=%d",
            len(train_idx), 100 * len(train_idx) / n_samples,
            len(test_idx), 100 * len(test_idx) / n_samples,
            len(pd.unique(self.groups)),
        )
        return train_idx, test_idx


class LeakageGuard:
    """Validates cross-validation splits for data leakage.

    Two independent checks:
    - Temporal: no train timestamp ≥ any val timestamp
    - Group: no group (e.g., well/instance) appears in both train and val
    """

    @staticmethod
    def check_temporal(
        train_timestamps: np.ndarray,
        val_timestamps: np.ndarray,
    ) -> dict:
        """Check that no train timestamp is ≥ any val timestamp.

        Returns:
            Dict with 'passed' bool and 'violations' count.

        Raises:
            ValueError: If temporal leakage is detected.
        """
        if len(train_timestamps) == 0 or len(val_timestamps) == 0:
            return {"passed": True, "violations": 0}

        max_train = np.max(train_timestamps)
        min_val = np.min(val_timestamps)

        violations = int(np.sum(train_timestamps >= min_val))
        passed = max_train < min_val

        if not passed:
            msg = (
                f"Temporal leakage: {violations} train timestamps ≥ min val timestamp. "
                f"max_train={max_train}, min_val={min_val}"
            )
            raise ValueError(msg)

        return {"passed": True, "violations": 0}

    @staticmethod
    def check_group(
        train_groups: np.ndarray | list,
        val_groups: np.ndarray | list,
    ) -> dict:
        """Check that no group appears in both train and val.

        Returns:
            Dict with 'passed' bool and 'leaked_groups' set.

        Raises:
            ValueError: If group leakage is detected.
        """
        train_set = set(np.asarray(train_groups).flat)
        val_set = set(np.asarray(val_groups).flat)
        leaked = train_set & val_set

        if leaked:
            msg = f"Group leakage: {len(leaked)} groups in both train and val: {leaked}"
            raise ValueError(msg)

        return {"passed": True, "leaked_groups": set()}


class FoldNormalizer:
    """Per-fold z-score normalization with leakage prevention.

    Fits normalization statistics on training data only, then applies
    the same transform to both train and validation partitions.

    Args:
        columns: Column names to normalize. If None, normalizes all float columns.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self._stats: dict[str, tuple[float, float]] | None = None

    def fit(self, train_df: pd.DataFrame) -> "FoldNormalizer":
        """Compute z-score statistics from training data only.

        Args:
            train_df: Training partition DataFrame.

        Returns:
            self, for chaining.
        """
        self._stats = compute_zscore_stats(train_df, columns=self.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization using stored training statistics.

        Args:
            df: DataFrame to normalize.

        Returns:
            Normalized DataFrame.

        Raises:
            RuntimeError: If ``fit()`` hasn't been called.
        """
        if self._stats is None:
            msg = "FoldNormalizer.fit() must be called before transform()"
            raise RuntimeError(msg)
        return apply_zscore(df, self._stats)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on train_df and return transformed copy."""
        return self.fit(train_df).transform(train_df)

    @property
    def stats(self) -> dict[str, tuple[float, float]] | None:
        """Return computed (mean, std) stats. None if not yet fitted."""
        return self._stats
