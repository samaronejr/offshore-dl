"""Run TimesFM zero-shot baseline on Ganymede and CDF.

No training — just inference. Classification not supported.
Uses small subsets for feasible verification.

Usage:
    python scripts/run_timesfm_baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from offshore_dl.data.datasets import CDFDataset, GanymedeDataset
from offshore_dl.evaluation.cv import ExpandingWindowCV, SlidingWindowCV, TemporalSplitCV
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.models.timesfm_wrapper import TimesFMWrapper
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _zero_shot_evaluate(model, dataset, val_idx, task, batch_size=32, max_samples=200):
    """Run zero-shot inference on a subset of the val set."""
    # Limit samples for feasibility
    if len(val_idx) > max_samples:
        val_idx = val_idx[:max_samples]

    all_preds = []
    all_targets = []

    for i in range(0, len(val_idx), batch_size):
        batch_idx = val_idx[i:i + batch_size]
        batch_x = torch.stack([dataset[j][0] for j in batch_idx])
        batch_y = torch.stack([dataset[j][1] for j in batch_idx])
        batch = (batch_x, batch_y, [{}] * len(batch_idx))

        preds = model.predict(batch)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())

    predictions = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    return MetricRegistry.compute(task, predictions, targets)


def run_ganymede() -> dict:
    """TimesFM zero-shot on Ganymede forecasting."""
    set_global_seed(42)
    dataset = GanymedeDataset("configs/data/ganymede.yaml")
    cv = ExpandingWindowCV(n_splits=3, min_train_ratio=0.5)

    model = TimesFMWrapper(
        task="forecasting", n_vars=dataset[0][0].shape[-1],
        horizon=30, window_size=90,
    )

    splits = cv.get_splits(len(dataset))
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info("═══ Ganymede Fold %d/%d ═══", fold_idx + 1, len(splits))
        start = time.time()
        metrics = _zero_shot_evaluate(model, dataset, val_idx, "forecasting", max_samples=200)
        elapsed = time.time() - start
        logger.info("  MAE=%.4f, RMSE=%.4f (%.1fs)", metrics["mae"], metrics["rmse"], elapsed)
        fold_results.append({"fold_idx": fold_idx, "metrics": metrics})

    agg = _aggregate(fold_results)
    return {"fold_results": fold_results, "aggregate": agg, "n_folds": len(splits)}


def run_cdf() -> dict:
    """TimesFM zero-shot on CDF anomaly detection."""
    set_global_seed(42)
    dataset = CDFDataset("configs/data/cdf.yaml")
    cv = SlidingWindowCV(n_splits=5, train_ratio=0.7)

    model = TimesFMWrapper(
        task="anomaly", n_vars=11, window_size=48,
    )

    splits = cv.get_splits(len(dataset))
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info("═══ CDF Fold %d/%d ═══", fold_idx + 1, len(splits))
        start = time.time()
        # Very small subset for CDF anomaly — per-channel inference is slow
        metrics = _zero_shot_evaluate(model, dataset, val_idx, "anomaly", batch_size=8, max_samples=50)
        elapsed = time.time() - start
        logger.info("  error_mean=%.4f (%.1fs)", metrics["error_mean"], elapsed)
        fold_results.append({"fold_idx": fold_idx, "metrics": metrics})

    agg = _aggregate(fold_results)
    return {"fold_results": fold_results, "aggregate": agg, "n_folds": len(splits)}


def _aggregate(fold_results):
    agg = {}
    for key in fold_results[0]["metrics"]:
        values = [fr["metrics"][key] for fr in fold_results if np.isfinite(fr["metrics"].get(key, float("nan")))]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def main() -> None:
    results_dir = Path("results/timesfm")
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("cdf", run_cdf),
        ("ganymede", run_ganymede),
    ]

    summary = {}
    for name, fn in runs:
        logger.info("=" * 60)
        logger.info("Starting TimesFM on %s", name)
        logger.info("=" * 60)

        start = time.time()
        try:
            results = fn()
            elapsed = time.time() - start

            out_path = results_dir / f"{name}.json"
            out_path.write_text(json.dumps(_make_serializable(results), indent=2))

            agg = results.get("aggregate", {})
            summary[name] = {"status": "ok", "elapsed": round(elapsed, 1), "aggregate": agg}

            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(agg.items()) if "_mean" in k)
            logger.info("✓ %s: %s (%.1fs)", name, metric_str, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            summary[name] = {"status": "error", "elapsed": round(elapsed, 1), "error": str(e)}
            logger.error("✗ %s failed: %s (%.1fs)", name, e, elapsed)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  TIMESFM ZERO-SHOT RESULTS")
    print("=" * 60)
    for name, s in summary.items():
        if s["status"] == "ok":
            agg = s["aggregate"]
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(agg.items()) if "_mean" in k)
            print(f"  {name:12s} ✓ {s['elapsed']:6.1f}s  {metric_str}")
        else:
            print(f"  {name:12s} ✗ {s['elapsed']:6.1f}s  ERROR: {s['error']}")
    print("=" * 60)

    (results_dir / "summary.json").write_text(json.dumps(_make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
