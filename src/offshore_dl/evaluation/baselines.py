"""Naive baseline models for all experimental tracks.

Sets the performance floor that every DL model must beat.

- ``MajorityClassBaseline`` — 3W classification
- ``SeasonalNaiveBaseline`` — Ganymede forecasting
- ``MeanReconstructionBaseline`` — CDF anomaly detection
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class BaselineModel(Protocol):
    """Protocol for naive baseline models."""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None: ...
    def predict(self, X_val: np.ndarray) -> np.ndarray: ...


class MajorityClassBaseline:
    """Predicts the most frequent class from training data.

    Used for 3W classification baseline.
    """

    def __init__(self) -> None:
        self._majority_class: int | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Record the most frequent class."""
        classes, counts = np.unique(y_train, return_counts=True)
        self._majority_class = int(classes[np.argmax(counts)])
        logger.info("MajorityClassBaseline: majority class = %d", self._majority_class)

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Predict majority class for all samples."""
        if self._majority_class is None:
            msg = "MajorityClassBaseline.fit() must be called before predict()"
            raise RuntimeError(msg)
        return np.full(len(X_val), self._majority_class, dtype=np.int64)


class SeasonalNaiveBaseline:
    """Repeats the last observed seasonal period as forecast.

    For Ganymede (daily data), default period=7 means the forecast
    repeats the last week's pattern.

    Args:
        period: Number of timesteps in one seasonal cycle.
    """

    def __init__(self, period: int = 7) -> None:
        self.period = period
        self._pattern: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Store the last `period` values from training targets."""
        y = np.asarray(y_train).ravel()
        self._pattern = y[-self.period :].copy()
        logger.info(
            "SeasonalNaiveBaseline: period=%d, pattern shape=%s",
            self.period, self._pattern.shape,
        )

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Repeat the seasonal pattern to cover the validation period.

        Args:
            X_val: Validation inputs (used only for length).

        Returns:
            Repeated seasonal pattern matching val length.
        """
        if self._pattern is None:
            msg = "SeasonalNaiveBaseline.fit() must be called before predict()"
            raise RuntimeError(msg)
        n = len(X_val)
        # Tile the pattern to cover n predictions
        repeats = (n // self.period) + 1
        repeated = np.tile(self._pattern, repeats)
        return repeated[:n]


class MeanReconstructionBaseline:
    """Uses the per-variable training mean as reconstruction.

    Anomaly score is the L2 distance between input and mean reconstruction.
    Used for CDF unsupervised anomaly detection baseline.
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None) -> None:
        """Compute per-variable mean from training data.

        Args:
            X_train: Training data of shape (n_samples, ..., n_vars).
            y_train: Ignored (unsupervised).
        """
        X = np.asarray(X_train, dtype=np.float64)
        # Mean over all axes except the last (features)
        axes = tuple(range(X.ndim - 1))
        self._mean = np.mean(X, axis=axes)
        logger.info("MeanReconstructionBaseline: mean shape=%s", self._mean.shape)

    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Return the mean vector as reconstruction for every input.

        Args:
            X_val: Validation data of shape (n_samples, ..., n_vars).

        Returns:
            Array of same shape as X_val, filled with the training mean.
        """
        if self._mean is None:
            msg = "MeanReconstructionBaseline.fit() must be called before predict()"
            raise RuntimeError(msg)
        return np.broadcast_to(self._mean, X_val.shape).copy()
