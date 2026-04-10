"""Tests for evaluation metrics registry."""

from __future__ import annotations

import numpy as np
import pytest

from offshore_dl.evaluation.metrics import MetricRegistry, format_metrics


# ═══════════════════════════════════════════════════════════════════
# Classification Metrics Tests
# ═══════════════════════════════════════════════════════════════════


class TestClassificationMetrics:
    """Tests for classification metric computation."""

    def test_perfect_predictions(self) -> None:
        targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        preds = targets.copy()
        results = MetricRegistry.compute("classification", preds, targets)
        assert results["f1_macro"] == pytest.approx(1.0)
        assert results["accuracy"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self) -> None:
        targets = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        preds = np.array([1, 1, 1, 2, 2, 2, 0, 0, 0])
        results = MetricRegistry.compute("classification", preds, targets)
        assert results["f1_macro"] == pytest.approx(0.0)
        assert results["accuracy"] == pytest.approx(0.0)

    def test_has_required_keys(self) -> None:
        targets = np.array([0, 1, 2, 0, 1, 2])
        preds = np.array([0, 1, 2, 1, 1, 2])
        results = MetricRegistry.compute("classification", preds, targets)
        assert "f1_macro" in results
        assert "f1_weighted" in results
        assert "auc_pr" in results
        assert "accuracy" in results
        assert "edr" in results

    def test_binary_classification(self) -> None:
        targets = np.array([0, 0, 1, 1, 1])
        preds = np.array([0, 0, 1, 1, 0])
        results = MetricRegistry.compute("classification", preds, targets)
        assert 0.0 < results["f1_macro"] < 1.0

    def test_edr_with_instance_ids(self) -> None:
        targets = np.array([0, 1, 1, 2, 2, 0])
        preds = np.array([0, 1, 0, 0, 0, 0])  # detects event type 1, misses event type 2
        instance_ids = np.array(["i1", "i2", "i2", "i3", "i3", "i4"])
        results = MetricRegistry.compute(
            "classification", preds, targets, instance_ids=instance_ids
        )
        # Event i2 (class 1): detected. Event i3 (class 2): not detected. EDR = 0.5
        assert results["edr"] == pytest.approx(0.5)

    def test_auc_pr_prefers_probability_scores_over_hard_labels(self) -> None:
        targets = np.array([0, 1, 2, 0, 1, 2])
        preds = np.array([0, 0, 2, 0, 1, 1])
        probs = np.array([
            [0.90, 0.05, 0.05],
            [0.35, 0.34, 0.31],
            [0.10, 0.10, 0.80],
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.33, 0.34, 0.33],
        ])

        without_scores = MetricRegistry.compute("classification", preds, targets)
        with_scores = MetricRegistry.compute(
            "classification", preds, targets, prediction_scores=probs,
        )

        assert with_scores["auc_pr"] > without_scores["auc_pr"]


# ═══════════════════════════════════════════════════════════════════
# Forecasting Metrics Tests
# ═══════════════════════════════════════════════════════════════════


class TestForecastingMetrics:
    """Tests for forecasting metric computation."""

    def test_perfect_predictions(self) -> None:
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        preds = targets.copy()
        results = MetricRegistry.compute("forecasting", preds, targets)
        assert results["mae"] == pytest.approx(0.0)
        assert results["rmse"] == pytest.approx(0.0)
        assert results["r2"] == pytest.approx(1.0)

    def test_constant_prediction_r2_zero(self) -> None:
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        preds = np.full_like(targets, targets.mean())
        results = MetricRegistry.compute("forecasting", preds, targets)
        assert results["r2"] == pytest.approx(0.0, abs=0.05)

    def test_mae_known_value(self) -> None:
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        preds = np.array([2.0, 3.0, 4.0, 5.0])  # all off by 1
        results = MetricRegistry.compute("forecasting", preds, targets)
        assert results["mae"] == pytest.approx(1.0)

    def test_has_required_keys(self) -> None:
        targets = np.random.randn(20)
        preds = targets + np.random.randn(20) * 0.1
        results = MetricRegistry.compute("forecasting", preds, targets)
        assert "mae" in results
        assert "rmse" in results
        assert "r2" in results
        assert "mase" in results

    def test_mase_with_seasonal_period(self) -> None:
        # Create data with perfect seasonal pattern
        targets = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 5)  # 35 values
        # Perfect seasonal naive prediction would match exactly
        preds = targets.copy()
        results = MetricRegistry.compute("forecasting", preds, targets, seasonal_period=7)
        assert results["mase"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════
# Anomaly Metrics Tests
# ═══════════════════════════════════════════════════════════════════


class TestAnomalyMetrics:
    """Tests for anomaly detection metric computation."""

    def test_perfect_reconstruction(self) -> None:
        targets = np.random.randn(50, 11)
        preds = targets.copy()
        results = MetricRegistry.compute("anomaly", preds, targets)
        assert results["error_mean"] == pytest.approx(0.0)
        assert results["error_std"] == pytest.approx(0.0)

    def test_has_required_keys(self) -> None:
        targets = np.random.randn(50, 11)
        preds = targets + np.random.randn(50, 11) * 0.1
        results = MetricRegistry.compute("anomaly", preds, targets)
        assert "error_mean" in results
        assert "error_std" in results
        assert "error_p50" in results
        assert "error_p95" in results
        assert "error_p99" in results

    def test_percentiles_order(self) -> None:
        targets = np.random.randn(100, 11)
        preds = targets + np.random.randn(100, 11) * 0.5
        results = MetricRegistry.compute("anomaly", preds, targets)
        assert results["error_p50"] <= results["error_p95"]
        assert results["error_p95"] <= results["error_p99"]

    def test_1d_errors(self) -> None:
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        results = MetricRegistry.compute("anomaly", preds, targets)
        assert results["error_mean"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# Format Tests
# ═══════════════════════════════════════════════════════════════════


class TestFormatMetrics:
    """Tests for metric formatting."""

    def test_format_produces_string(self) -> None:
        results = {"mae": 0.5, "rmse": 0.7}
        output = format_metrics(results, task="forecasting")
        assert isinstance(output, str)
        assert "mae" in output

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task"):
            MetricRegistry.compute("unknown", np.array([1]), np.array([1]))


# ═══════════════════════════════════════════════════════════════════
# Confusion Matrix Tests
# ═══════════════════════════════════════════════════════════════════


class TestConfusionMatrix:
    """Tests for confusion matrix in classification metrics."""

    def test_confusion_matrix_present(self) -> None:
        preds = np.array([0, 1, 2, 0, 1, 2])
        targets = np.array([0, 1, 2, 0, 1, 2])
        results = MetricRegistry.compute("classification", preds, targets)
        assert "confusion_matrix" in results
        assert "class_labels" in results

    def test_perfect_confusion_matrix(self) -> None:
        preds = np.array([0, 0, 1, 1, 2, 2])
        targets = np.array([0, 0, 1, 1, 2, 2])
        results = MetricRegistry.compute("classification", preds, targets)
        cm = np.array(results["confusion_matrix"])
        # Perfect: diagonal only
        assert np.trace(cm) == len(preds)
        assert cm.sum() - np.trace(cm) == 0

    def test_confusion_matrix_shape(self) -> None:
        preds = np.array([0, 1, 2, 3, 4])
        targets = np.array([0, 1, 2, 3, 4])
        results = MetricRegistry.compute("classification", preds, targets)
        cm = results["confusion_matrix"]
        assert len(cm) == 5
        assert len(cm[0]) == 5

    def test_confusion_matrix_serializable(self) -> None:
        """Confusion matrix should be JSON-serializable (list of lists)."""
        import json
        preds = np.array([0, 1, 0, 1])
        targets = np.array([0, 0, 1, 1])
        results = MetricRegistry.compute("classification", preds, targets)
        # Should not raise
        json.dumps(results["confusion_matrix"])
        json.dumps(results["class_labels"])
