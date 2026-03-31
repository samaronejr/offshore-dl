"""Statistical feature extraction for multivariate time-series windows.

Compresses a ``(window_size, n_vars)`` raw sensor window into a compact
``(n_features, n_vars)`` matrix of statistical descriptors.  Each column
(sensor) is summarized independently, preserving the multi-variate
structure that downstream models expect.

The feature order per sensor is deterministic, so models can treat the
feature axis as a short "sequence" dimension.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


# ── Feature catalogue ────────────────────────────────────────────
# Each entry: (name, callable(col_1d) → scalar)

def _slope(x: np.ndarray) -> float:
    """Linear-regression slope over the window."""
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    # Fast OLS via closed form: slope = cov(t,x) / var(t)
    t_mean = (n - 1) / 2.0
    x_mean = x.mean()
    num = np.dot(t - t_mean, x - x_mean)
    den = np.dot(t - t_mean, t - t_mean)
    return float(num / den) if den > 0 else 0.0


def _mean_abs_change(x: np.ndarray) -> float:
    """Mean absolute first-difference."""
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x))))


def _count_above_mean(x: np.ndarray) -> float:
    """Fraction of values above the mean."""
    if len(x) == 0:
        return 0.0
    return float(np.mean(x > x.mean()))


def _number_peaks(x: np.ndarray, support: int = 5) -> float:
    """Count local maxima with given support width."""
    if len(x) < 2 * support + 1:
        return 0.0
    count = 0
    for i in range(support, len(x) - support):
        left = x[i - support : i]
        right = x[i + 1 : i + support + 1]
        if np.all(x[i] > left) and np.all(x[i] > right):
            count += 1
    return float(count)


def _rms(x: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(x ** 2)))


def _iqr(x: np.ndarray) -> float:
    """Inter-quartile range."""
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _energy(x: np.ndarray) -> float:
    """Sum of squared values (signal energy)."""
    return float(np.sum(x ** 2))


FEATURE_FUNCTIONS: list[tuple[str, callable]] = [
    ("mean",             lambda x: float(np.mean(x))),
    ("std",              lambda x: float(np.std(x))),
    ("min",              lambda x: float(np.min(x))),
    ("max",              lambda x: float(np.max(x))),
    ("median",           lambda x: float(np.median(x))),
    ("skewness",         lambda x: float(sp_stats.skew(x))),
    ("kurtosis",         lambda x: float(sp_stats.kurtosis(x))),
    ("slope",            _slope),
    ("mean_abs_change",  _mean_abs_change),
    ("count_above_mean", _count_above_mean),
    ("number_peaks",     lambda x: _number_peaks(x, support=5)),
    ("rms",              _rms),
    ("iqr",              _iqr),
    ("energy",           _energy),
]

N_FEATURES = len(FEATURE_FUNCTIONS)  # 14


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract statistical features from a raw sensor window.

    Args:
        window: Array of shape ``(timesteps, n_vars)``.

    Returns:
        Feature matrix of shape ``(n_features, n_vars)`` where
        ``n_features = 14`` (see ``FEATURE_FUNCTIONS``).
        Each column is the per-sensor feature vector.
    """
    timesteps, n_vars = window.shape
    w = window.astype(np.float64)

    # Vectorized statistics (all sensors at once)
    mean = np.mean(w, axis=0)
    std = np.std(w, axis=0)
    mn = np.min(w, axis=0)
    mx = np.max(w, axis=0)
    med = np.median(w, axis=0)
    skw = sp_stats.skew(w, axis=0)
    krt = sp_stats.kurtosis(w, axis=0)

    # Slope via vectorized OLS
    n = timesteps
    t = np.arange(n, dtype=np.float64)
    t_mean = (n - 1) / 2.0
    t_centered = t - t_mean
    t_var = np.dot(t_centered, t_centered)
    if t_var > 0:
        slope = (t_centered @ (w - mean)) / t_var
    else:
        slope = np.zeros(n_vars)

    # Mean absolute change
    mac = np.mean(np.abs(np.diff(w, axis=0)), axis=0) if timesteps > 1 else np.zeros(n_vars)

    # Count above mean (fraction)
    cam = np.mean(w > mean, axis=0)

    # Number of peaks (vectorized per-column with small support)
    peaks = np.zeros(n_vars, dtype=np.float64)
    support = 5
    if timesteps >= 2 * support + 1:
        for i in range(support, timesteps - support):
            left_max = np.max(w[i - support:i], axis=0)
            right_max = np.max(w[i + 1:i + support + 1], axis=0)
            peaks += (w[i] > left_max) & (w[i] > right_max)

    # RMS
    rms = np.sqrt(np.mean(w ** 2, axis=0))

    # IQR
    iqr = np.percentile(w, 75, axis=0) - np.percentile(w, 25, axis=0)

    # Energy
    energy = np.sum(w ** 2, axis=0)

    # Stack: (14, n_vars)
    out = np.stack([
        mean, std, mn, mx, med, skw, krt,
        slope, mac, cam, peaks, rms, iqr, energy,
    ], axis=0).astype(np.float32)

    # Guard NaN / Inf
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def feature_names() -> list[str]:
    """Return ordered list of feature names."""
    return [name for name, _ in FEATURE_FUNCTIONS]
