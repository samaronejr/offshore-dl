"""Pure stateless transform functions for offshore DL data pipelines.

Every function here is stateless: data in → data out. No side effects,
no internal state, no configuration mutation. Functions that need parameters
(e.g. window sizes) take them as arguments.

Used by preprocessing pipelines (preprocess_3w.py, preprocess_cdf.py,
preprocess_ganymede.py) and at training time for fold-specific normalization.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 3W-specific transforms
# ═══════════════════════════════════════════════════════════════════


def detect_frozen_values(
    df: pd.DataFrame,
    window: int = 60,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Detect frozen sensor values via rolling variance.

    A sensor reading is "frozen" when its rolling variance over ``window``
    consecutive timesteps is exactly zero. Frozen values are replaced with NaN.

    Args:
        df: Input DataFrame with numeric sensor columns.
        window: Rolling window size in timesteps.
        columns: Columns to check. If None, uses all float64 columns.

    Returns:
        DataFrame with frozen values replaced by NaN.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["float64"]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        rolling_var = df[col].rolling(window=window, min_periods=window).var()
        frozen_mask = rolling_var == 0.0
        n_frozen = frozen_mask.sum()
        if n_frozen > 0:
            df.loc[frozen_mask, col] = np.nan
            logger.debug(
                "Column '%s': %d frozen values replaced with NaN", col, n_frozen
            )
    return df


def causal_forward_fill(
    df: pd.DataFrame,
    limit: int = 300,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Causal forward-fill with a hard timestep limit.

    Gaps exceeding ``limit`` consecutive NaN timesteps remain NaN.
    This is strictly causal — only past values propagate forward.

    Args:
        df: Input DataFrame.
        limit: Maximum consecutive NaN timesteps to fill.
        columns: Columns to fill. If None, uses all float64 columns.

    Returns:
        DataFrame with NaN gaps ≤ limit filled forward.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=["float64"]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].ffill(limit=limit)

    return df


def sliding_window_segmentation(
    values: np.ndarray,
    window_size: int,
    stride: int,
    labels: np.ndarray | None = None,
) -> list[dict]:
    """Segment a time series into fixed-size windows.

    Args:
        values: Array of shape ``(T, n_vars)`` — the sensor data.
        window_size: Number of timesteps per window.
        stride: Step size between consecutive windows.
        labels: Optional array of shape ``(T,)`` — per-timestep labels.
            If provided, each window's label is the majority class.

    Returns:
        List of dicts, each with keys ``start``, ``end``, and optionally ``label``.
    """
    n_timesteps = values.shape[0]
    if n_timesteps < window_size:
        return []

    windows = []
    for start in range(0, n_timesteps - window_size + 1, stride):
        end = start + window_size
        entry = {"start": start, "end": end}

        if labels is not None:
            window_labels = labels[start:end]
            # Majority vote — exclude NaN values
            valid = window_labels[~pd.isna(window_labels)]
            if len(valid) > 0:
                counts = Counter(valid.astype(int))
                entry["label"] = counts.most_common(1)[0][0]
            else:
                entry["label"] = -1  # no valid label

        windows.append(entry)

    return windows


# ═══════════════════════════════════════════════════════════════════
# Shared transforms (used across datasets)
# ═══════════════════════════════════════════════════════════════════


def compute_zscore_stats(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute per-variable mean and std from TRAINING DATA ONLY.

    Args:
        df: Training partition DataFrame.
        columns: Columns to compute stats for. If None, all float64.

    Returns:
        Dict mapping column name → (mean, std).
    """
    if columns is None:
        columns = df.select_dtypes(include=["float64"]).columns.tolist()

    stats = {}
    for col in columns:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std()
        # Prevent division by zero — constant columns get std=1
        if std == 0 or pd.isna(std):
            std = 1.0
        stats[col] = (float(mean), float(std))

    return stats


def apply_zscore(
    df: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Apply z-score normalization using precomputed statistics.

    Uses stats from ``compute_zscore_stats`` — NO leakage because
    stats come from the training partition only.

    Args:
        df: DataFrame to normalize.
        stats: Dict from ``compute_zscore_stats``.

    Returns:
        Normalized DataFrame.
    """
    df = df.copy()
    for col, (mean, std) in stats.items():
        if col in df.columns:
            df[col] = (df[col] - mean) / std
    return df


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced classification.

    Args:
        labels: Array of integer class labels.

    Returns:
        Dict mapping class_id → weight (higher weight for rarer classes).
    """
    labels = labels[~pd.isna(labels)].astype(int)
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)

    weights = {}
    for cls, count in counts.items():
        weights[cls] = total / (n_classes * count)

    return weights


# ═══════════════════════════════════════════════════════════════════
# Ganymede-specific transforms (used by T04)
# ═══════════════════════════════════════════════════════════════════


def detect_shutdowns(
    df: pd.DataFrame,
    gas_column: str = "ALLOC_GAS_VOL_SM3",
    zero_days_threshold: int = 2,
) -> pd.DataFrame:
    """Flag periods where gas production is zero for consecutive days.

    Uses a **causal** rolling count: a day is flagged as shutdown only when
    it has been zero for at least ``zero_days_threshold`` consecutive days
    *up to and including* that day.  The original non-causal implementation
    used ``groupby.transform("sum")`` which labels the first day of a zero
    run using the *total* run length (future information).

    Args:
        df: Daily production DataFrame (must be sorted by date ascending).
        gas_column: Column containing gas production values.
        zero_days_threshold: Minimum consecutive zero days to flag.

    Returns:
        DataFrame with added ``is_shutdown`` boolean column.
    """
    df = df.copy()
    is_zero = (df[gas_column] == 0) | df[gas_column].isna()

    # Causal consecutive-zero counter: reset to 0 on any non-zero day.
    # Each day only sees its own history, not future days.
    consecutive = is_zero.astype(int).groupby(
        (~is_zero).cumsum()
    ).cumcount() + is_zero.astype(int)

    df["is_shutdown"] = (consecutive >= zero_days_threshold) & is_zero
    return df


def compute_ema_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute exponential moving averages at multiple window sizes.

    Args:
        df: Input DataFrame.
        columns: Columns to compute EMAs for.
        windows: EMA spans in days. Default ``[7, 14, 30, 90]``.

    Returns:
        DataFrame with added EMA columns named ``{col}_ema_{span}``.
    """
    if windows is None:
        windows = [7, 14, 30, 90]

    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        for span in windows:
            df[f"{col}_ema_{span}"] = df[col].ewm(span=span, adjust=False).mean()
    return df


def compute_rate_of_change(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Compute first-order numerical derivative (diff / dt).

    Args:
        df: Input DataFrame.
        columns: Columns to differentiate.

    Returns:
        DataFrame with added ``{col}_roc`` columns.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[f"{col}_roc"] = df[col].diff()
    return df


def compute_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain-specific ratios for offshore production data.

    Currently computes BHP/WHP ratio when both are available.

    Args:
        df: Production DataFrame.

    Returns:
        DataFrame with added ratio columns.
    """
    df = df.copy()
    if "BHP_BARG" in df.columns and "WHP_BARG" in df.columns:
        # Avoid division by zero
        whp_safe = df["WHP_BARG"].replace(0, np.nan)
        df["bhp_whp_ratio"] = df["BHP_BARG"] / whp_safe
    return df


def log_transform(
    df: pd.DataFrame,
    columns: list[str],
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Apply log(x + eps) transform for right-skewed variables.

    Args:
        df: Input DataFrame.
        columns: Columns to transform.
        eps: Small constant to handle zeros.

    Returns:
        DataFrame with log-transformed columns (in place, same name).
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = np.log(df[col] + eps)
    return df


def gaussian_noise_augment(
    features: np.ndarray,
    sigma_frac: float = 0.02,
) -> np.ndarray:
    """Add Gaussian noise scaled by the feature standard deviation."""
    x = np.asarray(features, dtype=np.float32).copy()
    scale = float(np.std(x))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    noise = np.random.normal(0.0, sigma_frac * scale, size=x.shape).astype(np.float32)
    out = x + noise
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def feature_dropout_augment(
    features: np.ndarray,
    drop_prob: float = 0.1,
) -> np.ndarray:
    """Randomly zero out whole sensor columns."""
    x = np.asarray(features, dtype=np.float32).copy()
    if drop_prob <= 0:
        return x
    drop_mask = np.random.random(x.shape[-1]) < drop_prob
    if np.any(drop_mask):
        x[:, drop_mask] = 0.0
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def time_feature_warp_augment(
    features: np.ndarray,
    scale_range: tuple = (0.9, 1.1),
) -> np.ndarray:
    """Apply per-row scaling across the feature axis."""
    x = np.asarray(features, dtype=np.float32).copy()
    low, high = scale_range
    if low > high:
        low, high = high, low
    scales = np.random.uniform(low, high, size=(x.shape[0], 1)).astype(np.float32)
    out = x * scales
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out
