"""Unified metric registry for all experimental tracks.

Single entry point: ``MetricRegistry.compute(task, predictions, targets)``
returns a dict of all relevant metrics for the given task type.

Supported tasks:
    - ``"classification"`` — F1-macro, AUC-PR, accuracy, EDR, per-class report
    - ``"forecasting"`` — MAE, RMSE, R², MASE
    - ``"anomaly"`` — reconstruction error distribution statistics
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Compute evaluation metrics for any experimental track.

    Usage::

        metrics = MetricRegistry.compute("classification", preds, labels)
        # → {"f1_macro": 0.85, "auc_pr": 0.78, "accuracy": 0.90, "edr": 0.92, ...}
    """

    @staticmethod
    def compute(
        task: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        *,
        class_labels: list[int] | None = None,
        seasonal_period: int = 7,
        instance_ids: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        prediction_scores: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute all metrics for the given task.

        Args:
            task: One of ``"classification"``, ``"forecasting"``, ``"anomaly"``.
            predictions: Model predictions.
            targets: Ground truth targets.
            class_labels: Unique class labels (for classification AUC-PR).
            seasonal_period: Period for MASE naive denominator (forecasting).
            instance_ids: Per-sample instance IDs for EDR computation (classification).
            y_train: Training targets for MASE denominator (forecasting).
            prediction_scores: Probability scores for AUC-PR (classification).

        Returns:
            Dict mapping metric names to float values.
        """
        if task == "classification":
            return MetricRegistry._classification_metrics(
                predictions, targets,
                class_labels=class_labels,
                instance_ids=instance_ids,
                prediction_scores=prediction_scores,
            )
        elif task == "forecasting":
            return MetricRegistry._forecasting_metrics(
                predictions, targets,
                seasonal_period=seasonal_period,
                y_train=y_train,
            )
        elif task == "anomaly":
            return MetricRegistry._anomaly_metrics(predictions, targets)
        else:
            msg = f"Unknown task: {task!r}. Expected 'classification', 'forecasting', or 'anomaly'."
            raise ValueError(msg)

    @staticmethod
    def _classification_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        class_labels: list[int] | None = None,
        instance_ids: np.ndarray | None = None,
        prediction_scores: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute classification metrics.

        Returns:
            Dict with f1_macro, f1_weighted, auc_pr, accuracy, edr.
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        results: dict[str, float] = {}

        results["f1_macro"] = float(f1_score(targets, predictions, average="macro", zero_division=0))
        results["f1_weighted"] = float(f1_score(targets, predictions, average="weighted", zero_division=0))
        results["accuracy"] = float(accuracy_score(targets, predictions))

        # AUC-PR (macro-averaged, one-vs-rest)
        if class_labels is None:
            class_labels = sorted(set(targets))

        if len(class_labels) > 2:
            try:
                targets_bin = label_binarize(targets, classes=class_labels)
                if prediction_scores is not None:
                    # Use probability scores (preferred — gives true AUC-PR curve)
                    scores = np.asarray(prediction_scores)
                    if scores.ndim == 1 or scores.shape[-1] != len(class_labels):
                        # Fallback to binarized hard labels if scores shape doesn't match
                        scores = label_binarize(predictions, classes=class_labels)
                else:
                    # Fallback: binarized hard predictions (degenerate single-point PR)
                    scores = label_binarize(predictions, classes=class_labels)
                auc_pr = float(average_precision_score(
                    targets_bin, scores, average="macro",
                ))
            except (ValueError, IndexError):
                auc_pr = 0.0
        else:
            try:
                if prediction_scores is not None:
                    scores = np.asarray(prediction_scores)
                    if scores.ndim > 1:
                        scores = scores[:, 1]  # binary: use positive class score
                else:
                    scores = predictions
                auc_pr = float(average_precision_score(targets, scores))
            except ValueError:
                auc_pr = 0.0
        results["auc_pr"] = auc_pr

        # Event Detection Rate (EDR)
        if instance_ids is not None:
            results["edr"] = MetricRegistry._compute_edr(predictions, targets, instance_ids)
        else:
            # Without instance IDs, approximate: fraction of unique classes correctly predicted at least once
            unique_classes = set(targets)
            detected = sum(
                1 for cls in unique_classes
                if np.any(predictions[targets == cls] == cls)
            )
            results["edr"] = float(detected / max(len(unique_classes), 1))

        # Confusion matrix (rows=true, cols=predicted)
        cm = confusion_matrix(targets, predictions, labels=class_labels)
        results["confusion_matrix"] = cm.tolist()
        results["class_labels"] = [int(c) for c in class_labels]

        return results

    @staticmethod
    def _compute_edr(
        predictions: np.ndarray,
        targets: np.ndarray,
        instance_ids: np.ndarray,
    ) -> float:
        """Event Detection Rate: fraction of event instances with ≥1 correct prediction."""
        events: dict[str, bool] = defaultdict(lambda: False)
        for pred, target, inst_id in zip(predictions, targets, instance_ids):
            if target != 0:  # non-normal class = event
                key = f"{inst_id}_{target}"
                if pred == target:
                    events[key] = True
                elif key not in events:
                    events[key] = False

        if not events:
            return 1.0  # no events → trivially "detected all"

        return float(sum(events.values()) / len(events))

    @staticmethod
    def _forecasting_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        seasonal_period: int = 7,
        y_train: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute forecasting metrics.

        Returns:
            Dict with mae, rmse, r2, mase.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        targets = np.asarray(targets, dtype=np.float64).ravel()

        results: dict[str, float] = {}

        results["mae"] = float(mean_absolute_error(targets, predictions))
        results["rmse"] = float(np.sqrt(mean_squared_error(targets, predictions)))

        if np.var(targets) > 1e-12:
            results["r2"] = float(r2_score(targets, predictions))
        else:
            results["r2"] = 0.0

        # R² on productive periods only (excludes shutdown zeros)
        productive = np.abs(targets) > 0.01
        if productive.sum() > 10 and np.var(targets[productive]) > 1e-12:
            results["r2_prod"] = float(
                r2_score(targets[productive], predictions[productive])
            )
        else:
            results["r2_prod"] = results["r2"]

        # MASE: Mean Absolute Scaled Error
        # Denominator: MAE of naive seasonal forecast on TRAINING data
        # (Hyndman & Koehler, 2006)
        scale_data = y_train if y_train is not None else targets
        scale_data = np.asarray(scale_data, dtype=np.float64).ravel()
        n_scale = len(scale_data)
        if n_scale > seasonal_period:
            naive_errors = np.abs(scale_data[seasonal_period:] - scale_data[:-seasonal_period])
            naive_mae = np.mean(naive_errors)
            if naive_mae > 1e-12:
                results["mase"] = results["mae"] / naive_mae
            else:
                results["mase"] = 0.0  # perfect naive → 0
        else:
            # Series too short for seasonal naive
            results["mase"] = float("inf")

        return results

    @staticmethod
    def _anomaly_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """Compute anomaly detection metrics from reconstruction/prediction errors.

        Args:
            predictions: Reconstructed/predicted values.
            targets: Original values (ground truth).

        Returns:
            Dict with error_mean, error_std, error_p50, error_p95, error_p99.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        # Per-sample reconstruction error (L2 norm over features)
        if predictions.ndim > 1:
            errors = np.sqrt(np.mean((predictions - targets) ** 2, axis=-1))
        else:
            errors = np.abs(predictions - targets)

        results: dict[str, float] = {
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "error_p50": float(np.percentile(errors, 50)),
            "error_p95": float(np.percentile(errors, 95)),
            "error_p99": float(np.percentile(errors, 99)),
        }

        return results


def format_metrics(results: dict[str, float], task: str = "") -> str:
    """Format a metrics dict as a human-readable string.

    Args:
        results: Dict of metric name → float value.
        task: Optional task name for the header.

    Returns:
        Formatted string with one metric per line.
    """
    lines = []
    if task:
        lines.append(f"  Task: {task}")
    for key, val in sorted(results.items()):
        if isinstance(val, float):
            lines.append(f"  {key:>20s}: {val:.6f}")
        else:
            lines.append(f"  {key:>20s}: {val}")
    return "\n".join(lines)
