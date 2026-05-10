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
        groups: np.ndarray | None = None,
        y_train_groups: np.ndarray | None = None,
        order: np.ndarray | None = None,
        y_train_order: np.ndarray | None = None,
        mase_aggregation: str = "group_weighted",
        prediction_scores: np.ndarray | None = None,
    ) -> dict:
        """Compute all metrics for the given task.

        Args:
            task: One of ``"classification"``, ``"forecasting"``, ``"anomaly"``.
            predictions: Model predictions.
            targets: Ground truth targets.
            class_labels: Unique class labels (for classification AUC-PR).
            seasonal_period: Period for MASE naive denominator (forecasting).
            instance_ids: Per-sample instance IDs for EDR computation (classification).
            y_train: Training targets for MASE denominator (forecasting).
            groups: Per-sample forecasting group IDs aligned to predictions/targets.
            y_train_groups: Training group IDs aligned to y_train samples.
            order: Per-sample temporal order aligned to predictions/targets.
            y_train_order: Temporal order aligned to y_train samples.
            mase_aggregation: Primary MASE aggregation when group data is available:
                ``"group_weighted"``, ``"group_macro"``, or ``"flat"``.
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
                groups=groups,
                y_train_groups=y_train_groups,
                order=order,
                y_train_order=y_train_order,
                mase_aggregation=mase_aggregation,
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

        # AUC-PR (macro-averaged, one-vs-rest).  Class labels must cover the
        # full label space, not just classes seen in this fold's val set —
        # otherwise rare-class folds end up with shape-mismatched scores and
        # silently fall back to auc_pr=0.0.
        if class_labels is None:
            inferred = set(np.asarray(targets).tolist())
            inferred.update(np.asarray(predictions).tolist())
            if prediction_scores is not None:
                scores_arr = np.asarray(prediction_scores)
                if scores_arr.ndim == 2:
                    inferred.update(range(scores_arr.shape[1]))
            class_labels = sorted(inferred)

        if len(class_labels) > 2:
            try:
                targets_bin = label_binarize(targets, classes=class_labels)
                if prediction_scores is not None:
                    # Use probability scores (preferred — gives true AUC-PR curve)
                    scores = np.asarray(prediction_scores, dtype=np.float64)
                    if scores.ndim == 1 or scores.shape[-1] != len(class_labels):
                        # Fallback to binarized hard labels if scores shape doesn't match
                        scores = label_binarize(predictions, classes=class_labels)
                    else:
                        # Sanitize non-finite scores (NaN/Inf from divergent training
                        # batches or sensor outliers) — sklearn's AP raises ValueError
                        # on NaN, which would otherwise drop us into the except clause
                        # and silently emit auc_pr=0.0.
                        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
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
        groups: np.ndarray | None = None,
        y_train_groups: np.ndarray | None = None,
        order: np.ndarray | None = None,
        y_train_order: np.ndarray | None = None,
        mase_aggregation: str = "group_weighted",
    ) -> dict:
        """Compute forecasting metrics.

        Returns:
            Dict with mae, rmse, r2, and explicit MASE aggregation/provenance.
        """
        predictions_raw = np.asarray(predictions, dtype=np.float64)
        targets_raw = np.asarray(targets, dtype=np.float64)
        predictions = predictions_raw.ravel()
        targets = targets_raw.ravel()

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

        mase_flat, flat_source = MetricRegistry._compute_mase_flat(
            predictions,
            targets,
            seasonal_period=seasonal_period,
            y_train=y_train,
            groups=groups,
            y_train_groups=y_train_groups,
            order=order,
            y_train_order=y_train_order,
        )
        results["mase_flat"] = mase_flat
        results["mase_group_macro"] = float("nan")
        results["mase_group_weighted"] = float("nan")
        results["mase_denominator_source"] = flat_source

        group_metrics = MetricRegistry._compute_grouped_mase(
            predictions_raw,
            targets_raw,
            groups=groups,
            y_train=y_train,
            y_train_groups=y_train_groups,
            order=order,
            y_train_order=y_train_order,
            seasonal_period=seasonal_period,
        )
        if group_metrics is not None:
            grouped_source = group_metrics.pop("mase_denominator_source", None)
            results.update(group_metrics)
            if grouped_source is not None:
                results["mase_denominator_source"] = grouped_source
            if mase_aggregation == "group_macro":
                results["mase"] = results["mase_group_macro"]
                results["mase_aggregation"] = "group_macro"
            elif mase_aggregation == "flat":
                results["mase"] = results["mase_flat"]
                results["mase_aggregation"] = "flat"
            elif mase_aggregation == "group_weighted":
                results["mase"] = results["mase_group_weighted"]
                results["mase_aggregation"] = "group_weighted"
            else:
                msg = "mase_aggregation must be 'group_weighted', 'group_macro', or 'flat'"
                raise ValueError(msg)
        else:
            if np.isfinite(results["mase_flat"]) or np.isinf(results["mase_flat"]):
                results["mase"] = results["mase_flat"]
                results["mase_aggregation"] = "flat_fallback"
            else:
                results["mase"] = float("nan")
                results["mase_aggregation"] = "unavailable"

        return results

    @staticmethod
    def _compute_mase_flat(
        predictions: np.ndarray,
        targets: np.ndarray,
        *,
        seasonal_period: int,
        y_train: np.ndarray | None,
        groups: np.ndarray | None,
        y_train_groups: np.ndarray | None,
        order: np.ndarray | None,
        y_train_order: np.ndarray | None,
    ) -> tuple[float, str]:
        """Compute flat MASE and denominator provenance.

        Flat MASE is valid only for a single chronological series.  It must not
        concatenate shuffled multi-well rows or ravel overlapping horizon
        windows into an artificial denominator.
        """
        scale_data = y_train if y_train is not None else targets
        source = "train_flat" if y_train is not None else "eval_flat"
        scale_order = y_train_order if y_train is not None else order

        for candidate_groups in (groups, y_train_groups):
            if candidate_groups is not None and MetricRegistry._has_missing_values(candidate_groups):
                return float("nan"), "missing_group"
            if MetricRegistry._has_multiple_groups(candidate_groups):
                return float("nan"), "multi_group_flat_unavailable"

        if scale_order is None:
            return float("nan"), "missing_order"

        scale_data = MetricRegistry._ordered_denominator_series(scale_data, scale_order)
        if scale_data is None:
            return float("nan"), "missing_order"
        if len(scale_data) <= seasonal_period:
            return float("inf"), source
        naive_errors = np.abs(scale_data[seasonal_period:] - scale_data[:-seasonal_period])
        naive_mae = float(np.mean(naive_errors))
        mae = float(mean_absolute_error(targets, predictions))
        if naive_mae <= 1e-12:
            return (0.0 if mae <= 1e-12 else float("inf")), source
        return mae / naive_mae, source

    @staticmethod
    def _compute_grouped_mase(
        predictions: np.ndarray,
        targets: np.ndarray,
        *,
        groups: np.ndarray | None,
        y_train: np.ndarray | None,
        y_train_groups: np.ndarray | None,
        order: np.ndarray | None,
        y_train_order: np.ndarray | None,
        seasonal_period: int,
    ) -> dict[str, float] | None:
        """Compute macro/weighted group MASE when group metadata is aligned.

        Grouped denominators are sorted inside the metric layer using explicit
        per-sample temporal order.  Missing order makes MASE unavailable rather
        than falling back to arbitrary append order.
        """
        if groups is None:
            return None

        groups = np.asarray(groups)
        n_samples = predictions.shape[0] if predictions.ndim > 0 else len(groups)
        if len(groups) != n_samples:
            logger.warning("Ignoring grouped MASE: groups length does not match samples")
            return None
        if MetricRegistry._has_missing_values(groups):
            logger.warning("Ignoring grouped MASE: groups contain missing values")
            return None

        pred_2d = np.asarray(predictions, dtype=np.float64).reshape(n_samples, -1)
        target_2d = np.asarray(targets, dtype=np.float64).reshape(n_samples, -1)
        order_arr = MetricRegistry._coerce_order(order, n_samples)

        train_by_group: dict[object, np.ndarray] = {}
        train_source_available = False
        if y_train is not None and y_train_groups is not None:
            y_train_arr = np.asarray(y_train, dtype=np.float64)
            y_train_groups_arr = np.asarray(y_train_groups)
            n_train_samples = y_train_arr.shape[0] if y_train_arr.ndim > 0 else len(y_train_groups_arr)
            y_train_order_arr = MetricRegistry._coerce_order(y_train_order, n_train_samples)
            if len(y_train_groups_arr) != n_train_samples:
                logger.warning("Ignoring grouped MASE: y_train_groups length mismatch")
                return None
            if MetricRegistry._has_missing_values(y_train_groups_arr):
                logger.warning("Ignoring grouped MASE: y_train_groups contain missing values")
                return None
            if y_train_order_arr is None:
                logger.warning("Ignoring grouped MASE: missing y_train_order")
                return None
            train_source_available = True
            for group in pd_unique(y_train_groups_arr):
                mask = y_train_groups_arr == group
                series = MetricRegistry._ordered_denominator_series(
                    y_train_arr[mask],
                    y_train_order_arr[mask],
                )
                if series is None:
                    logger.warning(
                        "Ignoring grouped MASE: cannot form ordered training denominator"
                    )
                    return None
                train_by_group[group] = series

        if not train_source_available and order_arr is None:
            logger.warning("Ignoring grouped MASE: missing evaluation order")
            return None

        group_mases = []
        group_weights = []
        denominator_source = "grouped_train" if train_source_available else "grouped_eval"
        for group in pd_unique(groups):
            mask = groups == group
            group_errors = np.abs(pred_2d[mask] - target_2d[mask]).ravel()
            group_mae = float(np.mean(group_errors))
            scale_data = train_by_group.get(group)
            if scale_data is None:
                if order_arr is None:
                    logger.warning(
                        "Ignoring grouped MASE: no ordered denominator for group %r", group
                    )
                    return None
                scale_data = MetricRegistry._ordered_denominator_series(
                    target_2d[mask],
                    order_arr[mask],
                )
                denominator_source = "grouped_eval"
            if scale_data is None:
                logger.warning("Ignoring grouped MASE: invalid ordered denominator")
                return None
            if len(scale_data) <= seasonal_period:
                group_mase = float("inf")
            else:
                naive_mae = float(np.mean(np.abs(scale_data[seasonal_period:] - scale_data[:-seasonal_period])))
                group_mase = (
                    0.0 if naive_mae <= 1e-12 and group_mae <= 1e-12
                    else float("inf") if naive_mae <= 1e-12
                    else group_mae / naive_mae
                )
            group_mases.append(group_mase)
            group_weights.append(int(mask.sum()))

        if not group_mases:
            return None

        weights = np.asarray(group_weights, dtype=np.float64)
        mases = np.asarray(group_mases, dtype=np.float64)
        return {
            "mase_group_macro": float(np.mean(mases)),
            "mase_group_weighted": float(np.average(mases, weights=weights)),
            "mase_denominator_source": denominator_source,
        }

    @staticmethod
    def _has_multiple_groups(groups: np.ndarray | None) -> bool:
        if groups is None:
            return False
        values = np.asarray(groups, dtype=object)
        if values.size == 0:
            return False
        values = values.reshape(-1)
        values = np.asarray(
            [v for v in values if not MetricRegistry._is_missing_value(v)],
            dtype=object,
        )
        return len(pd_unique(values)) > 1

    @staticmethod
    def _has_missing_values(values: np.ndarray) -> bool:
        return any(MetricRegistry._is_missing_value(v) for v in np.asarray(values, dtype=object).reshape(-1))

    @staticmethod
    def _is_missing_value(value: object) -> bool:
        if value is None:
            return True
        try:
            return bool(np.isnan(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _coerce_order(order: np.ndarray | None, n_samples: int) -> np.ndarray | None:
        if order is None:
            return None
        order_arr = np.asarray(order, dtype=object)
        if order_arr.ndim == 0:
            order_arr = order_arr.reshape(1)
        order_arr = order_arr.reshape(-1)
        if len(order_arr) != n_samples:
            logger.warning(
                "Ignoring MASE order: expected %d values, got %d",
                n_samples,
                len(order_arr),
            )
            return None
        if any(MetricRegistry._is_missing_value(v) for v in order_arr):
            logger.warning("Ignoring MASE order: contains missing values")
            return None
        return order_arr

    @staticmethod
    def _ordered_denominator_series(
        values: np.ndarray,
        order: np.ndarray | None,
    ) -> np.ndarray | None:
        """Return one defensible chronological target series for MASE scale.

        For multi-horizon windows, this intentionally uses the first horizon
        value after sorting by sample order instead of raveling all horizons
        into artificial adjacent observations.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        n_samples = arr.shape[0]
        arr_2d = arr.reshape(n_samples, -1)

        order_arr = MetricRegistry._coerce_order(order, n_samples)
        if order is not None and order_arr is None:
            return None
        if order_arr is not None:
            sort_idx = np.argsort(order_arr, kind="mergesort")
            arr_2d = arr_2d[sort_idx]
        elif arr_2d.shape[1] > 1:
            return None

        return arr_2d[:, 0].astype(np.float64, copy=False)

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

        sq_error = (predictions - targets) ** 2
        timestep_errors = None
        if predictions.ndim == 1:
            errors = np.abs(predictions - targets)
        elif predictions.ndim == 2:
            errors = np.sqrt(np.mean(sq_error, axis=1))
        elif predictions.ndim == 3:
            errors = np.sqrt(np.mean(sq_error, axis=(1, 2)))
            timestep_errors = np.sqrt(np.mean(sq_error, axis=2))
        else:
            errors = np.sqrt(np.mean(sq_error.reshape(sq_error.shape[0], -1), axis=1))

        results: dict[str, float] = {
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "error_p50": float(np.percentile(errors, 50)),
            "error_p95": float(np.percentile(errors, 95)),
            "error_p99": float(np.percentile(errors, 99)),
        }
        if timestep_errors is not None:
            results.update(
                {
                    "timestep_error_mean": float(np.mean(timestep_errors)),
                    "timestep_error_std": float(np.std(timestep_errors)),
                    "timestep_error_p50": float(np.percentile(timestep_errors, 50)),
                    "timestep_error_p95": float(np.percentile(timestep_errors, 95)),
                    "timestep_error_p99": float(np.percentile(timestep_errors, 99)),
                }
            )

        return results


def pd_unique(values: np.ndarray) -> np.ndarray:
    """Stable unique values for object or numeric group arrays without pandas."""
    return np.asarray(list(dict.fromkeys(np.asarray(values).tolist())), dtype=object)


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
