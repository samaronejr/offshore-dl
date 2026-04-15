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

try:
    import pywt
except ImportError:
    pywt = None

try:
    from scipy import signal as sp_signal
except ImportError:
    sp_signal = None


# ── Feature catalogue ────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    "mean", "std", "min", "max", "median", "skewness", "kurtosis",
    "slope", "mean_abs_change", "count_above_mean", "number_peaks",
    "rms", "iqr", "energy",
]

N_FEATURES = len(FEATURE_NAMES)  # 14


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Extract statistical features from a raw sensor window.

    Args:
        window: Array of shape ``(timesteps, n_vars)``.

    Returns:
        Feature matrix of shape ``(n_features, n_vars)`` where
        ``n_features = 14`` (see ``FEATURE_NAMES``).
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
    mac = (
        np.mean(np.abs(np.diff(w, axis=0)), axis=0)
        if timesteps > 1
        else np.zeros(n_vars)
    )

    # Count above mean (fraction)
    cam = np.mean(w > mean, axis=0)

    # Number of peaks (vectorized per-column with small support)
    peaks = np.zeros(n_vars, dtype=np.float64)
    support = 5
    if timesteps >= 2 * support + 1:
        for i in range(support, timesteps - support):
            left_max = np.max(w[i - support : i], axis=0)
            right_max = np.max(w[i + 1 : i + support + 1], axis=0)
            peaks += (w[i] > left_max) & (w[i] > right_max)

    # RMS
    rms = np.sqrt(np.mean(w**2, axis=0))

    # IQR
    iqr = np.percentile(w, 75, axis=0) - np.percentile(w, 25, axis=0)

    # Energy
    energy = np.sum(w**2, axis=0)

    # Stack: (14, n_vars)
    out = np.stack(
        [
            mean,
            std,
            mn,
            mx,
            med,
            skw,
            krt,
            slope,
            mac,
            cam,
            peaks,
            rms,
            iqr,
            energy,
        ],
        axis=0,
    ).astype(np.float32)

    # Guard NaN / Inf
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


class MultiScaleFeatureExtractor:
    """Extract and stack descriptor banks from multiple trailing scales."""

    def __init__(self, scales: list[int] | tuple[int, ...] | None = None) -> None:
        self.scales = [int(scale) for scale in (scales or [360, 720])]
        if not self.scales:
            msg = "MultiScaleFeatureExtractor requires at least one scale."
            raise ValueError(msg)
        if any(scale <= 0 for scale in self.scales):
            msg = f"All scales must be positive, got {self.scales}."
            raise ValueError(msg)

    def extract(self, window: np.ndarray) -> np.ndarray:
        timesteps, _n_vars = window.shape
        feature_blocks = []
        for scale in self.scales:
            if scale > timesteps:
                msg = f"Scale {scale} exceeds window length {timesteps}."
                raise ValueError(msg)
            feature_blocks.append(extract_window_features(window[-scale:]))
        return np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)


class WaveletFeatureExtractor:
    """Extract per-sensor wavelet energy at fixed temporal scales."""

    def __init__(
        self,
        scales: list[int] | tuple[int, ...] | None = None,
        wavelet: str = "morl",
    ) -> None:
        self.scales = [int(scale) for scale in (scales or [30, 90, 180, 360])]
        if not self.scales:
            msg = "WaveletFeatureExtractor requires at least one scale."
            raise ValueError(msg)
        if any(scale <= 0 for scale in self.scales):
            msg = f"All scales must be positive, got {self.scales}."
            raise ValueError(msg)
        self.wavelet = wavelet

    def _extract_pywt(
        self, signal_1d: np.ndarray, valid_scales: list[int]
    ) -> np.ndarray:
        coefficients, _freqs = pywt.cwt(signal_1d, valid_scales, self.wavelet)
        return np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)

    def _extract_scipy(
        self,
        signal_1d: np.ndarray,
        valid_scales: list[int],
    ) -> np.ndarray:
        if sp_signal is None or not hasattr(sp_signal, "cwt"):
            msg = "Wavelet extraction requires either pywt or scipy.signal.cwt."
            raise ImportError(msg)

        wavelet_fn = getattr(sp_signal, "ricker", None)
        if wavelet_fn is None:
            wavelet_fn = getattr(sp_signal, "morlet2", None)
        if wavelet_fn is None:
            msg = "No compatible scipy wavelet function available."
            raise ImportError(msg)

        coefficients = sp_signal.cwt(signal_1d, wavelet_fn, valid_scales)
        return np.sum(np.abs(coefficients) ** 2, axis=1, dtype=np.float64)

    def extract(self, window: np.ndarray) -> np.ndarray:
        if window.ndim != 2:
            msg = f"Expected 2D window, got shape {window.shape}."
            raise ValueError(msg)

        timesteps, n_vars = window.shape
        cleaned = np.nan_to_num(
            np.asarray(window, dtype=np.float64),
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        out = np.zeros((len(self.scales), n_vars), dtype=np.float32)
        for sensor_idx in range(n_vars):
            signal_1d = cleaned[:, sensor_idx]
            valid_positions = [
                i for i, scale in enumerate(self.scales) if scale <= timesteps
            ]
            if not valid_positions:
                continue

            valid_scales = [self.scales[i] for i in valid_positions]
            if pywt is not None:
                energies = self._extract_pywt(signal_1d, valid_scales)
            else:
                energies = self._extract_scipy(signal_1d, valid_scales)

            out[valid_positions, sensor_idx] = energies.astype(np.float32, copy=False)

        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class PhysicsFeatureExtractor:
    """Extract domain-informed cross-sensor ratio features for 3W windows."""

    DEFAULT_SENSOR_COLUMNS = [
        "ABER-CKGL",
        "ABER-CKP",
        "ESTADO-DHSV",
        "ESTADO-M1",
        "ESTADO-M2",
        "ESTADO-PXO",
        "ESTADO-SDV-GL",
        "ESTADO-SDV-P",
        "ESTADO-W1",
        "ESTADO-W2",
        "ESTADO-XO",
        "P-ANULAR",
        "P-JUS-BS",
        "P-JUS-CKGL",
        "P-JUS-CKP",
        "P-MON-CKGL",
        "P-MON-CKP",
        "P-MON-SDV-P",
        "P-PDG",
        "PT-P",
        "P-TPT",
        "QBS",
        "QGL",
        "T-JUS-CKP",
        "T-MON-CKP",
        "T-PDG",
        "T-TPT",
    ]

    def __init__(self, sensor_columns: list[str] | None = None) -> None:
        self.sensor_columns = list(sensor_columns or self.DEFAULT_SENSOR_COLUMNS)
        self._name_to_index = {
            name: idx for idx, name in enumerate(self.sensor_columns)
        }

    def _index(self, name: str, default: int) -> int:
        return self._name_to_index.get(name, default)

    def extract(self, window: np.ndarray) -> np.ndarray:
        if window.ndim != 2:
            msg = f"Expected 2D window, got shape {window.shape}."
            raise ValueError(msg)

        cleaned = np.nan_to_num(
            np.asarray(window, dtype=np.float64),
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        eps = 1e-6
        pressure_gradient = (
            cleaned[:, self._index("P-PDG", 18)] - cleaned[:, self._index("P-TPT", 20)]
        )
        np.nan_to_num(pressure_gradient, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        temperature_ratio = cleaned[:, self._index("T-JUS-CKP", 23)] / (
            cleaned[:, self._index("T-TPT", 26)] + eps
        )
        np.nan_to_num(temperature_ratio, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        gas_lift_efficiency = cleaned[:, self._index("QGL", 22)] / (
            np.abs(
                cleaned[:, self._index("P-JUS-CKGL", 13)]
                - cleaned[:, self._index("P-ANULAR", 11)]
            )
            + eps
        )
        np.nan_to_num(gas_lift_efficiency, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        choke_coefficient = cleaned[:, self._index("QBS", 21)] / np.sqrt(
            np.abs(
                cleaned[:, self._index("P-MON-CKP", 16)]
                - cleaned[:, self._index("P-JUS-CKP", 14)]
            )
            + eps
        )
        np.nan_to_num(choke_coefficient, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        ratios = np.stack(
            [
                pressure_gradient,
                temperature_ratio,
                gas_lift_efficiency,
                choke_coefficient,
            ],
            axis=1,
        )
        np.nan_to_num(ratios, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return extract_window_features(ratios)
