"""Diagnostic script to run naive baselines through CV and print results.

Usage::

    python -m offshore_dl.evaluation.check

Runs all 3 naive baselines, computes per-fold metrics, saves JSON results,
and prints formatted summary tables.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _run_3w_baseline(results_dir: Path) -> dict:
    """Run majority class baseline on 3W through stratified group K-fold.

    Uses instance-level classification (one prediction per instance) rather
    than per-window, which is faster and more aligned with the benchmark.
    """
    import pandas as pd

    from offshore_dl.evaluation.baselines import MajorityClassBaseline
    from offshore_dl.evaluation.metrics import MetricRegistry

    print("\n" + "=" * 60)
    print("  3W: Majority Class Baseline")
    print("=" * 60)

    t0 = time.time()

    # Load folds directly — no need to instantiate the full dataset for majority class
    folds_df = pd.read_csv("data/raw/3w/folds/folds_clf_02.csv")
    folds_df["class_id"] = folds_df["instancia"].str.split("/").str[0].astype(int)

    # Filter out holdout (fold=-1)
    cv_df = folds_df[folds_df["fold"] >= 0].copy()
    n_folds = 5

    fold_results = []
    for fold_id in range(n_folds):
        val_mask = cv_df["fold"] == fold_id
        train_mask = cv_df["fold"] != fold_id

        y_train = cv_df.loc[train_mask, "class_id"].values
        y_val = cv_df.loc[val_mask, "class_id"].values

        if len(y_val) == 0 or len(y_train) == 0:
            continue

        baseline = MajorityClassBaseline()
        baseline.fit(np.zeros(len(y_train)), y_train)
        preds = baseline.predict(np.zeros(len(y_val)))

        metrics = MetricRegistry.compute("classification", preds, y_val)
        fold_results.append(metrics)
        print(f"  Fold {fold_id}: F1-macro={metrics['f1_macro']:.4f}, acc={metrics['accuracy']:.4f}")

    # Aggregate
    agg = _aggregate_folds(fold_results)
    print(f"  Mean±Std: F1-macro={agg['f1_macro_mean']:.4f}±{agg['f1_macro_std']:.4f}")

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    result = {"folds": fold_results, "aggregate": agg, "elapsed": elapsed}
    _save_json(result, results_dir / "3w_majority_baseline.json")
    return result


def _run_ganymede_baseline(results_dir: Path) -> dict:
    """Run seasonal naive baseline on Ganymede through expanding window CV."""
    from offshore_dl.data.datasets import GanymedeDataset
    from offshore_dl.evaluation.baselines import SeasonalNaiveBaseline
    from offshore_dl.evaluation.cv import ExpandingWindowCV
    from offshore_dl.evaluation.metrics import MetricRegistry

    print("\n" + "=" * 60)
    print("  Ganymede: Seasonal Naive Baseline (period=7)")
    print("=" * 60)

    t0 = time.time()

    ds = GanymedeDataset(
        "configs/data/ganymede.yaml",
        mode="multi_well",
        horizon=30,
        input_window=90,
    )

    cv = ExpandingWindowCV(n_splits=5, min_train_ratio=0.5)
    splits = cv.get_splits(len(ds))

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Extract targets (1D gas production)
        y_train = np.array([ds[i][1].numpy() for i in train_idx[:500]])  # subsample for speed
        y_val = np.array([ds[i][1].numpy() for i in val_idx[:200]])

        # Flatten targets for metric computation
        y_train_flat = y_train.ravel()
        y_val_flat = y_val.ravel()

        baseline = SeasonalNaiveBaseline(period=7)
        baseline.fit(np.zeros(len(y_train_flat)), y_train_flat)
        preds = baseline.predict(np.zeros(len(y_val_flat)))

        metrics = MetricRegistry.compute("forecasting", preds, y_val_flat, seasonal_period=7)
        fold_results.append(metrics)
        print(f"  Fold {fold_idx}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}, MASE={metrics['mase']:.4f}")

    agg = _aggregate_folds(fold_results)
    elapsed = time.time() - t0
    print(f"  Mean±Std: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    result = {"folds": fold_results, "aggregate": agg, "elapsed": elapsed}
    _save_json(result, results_dir / "ganymede_seasonal_naive_baseline.json")
    return result


def _run_cdf_baseline(results_dir: Path) -> dict:
    """Run mean reconstruction baseline on CDF through temporal split."""
    from offshore_dl.data.datasets import CDFDataset
    from offshore_dl.evaluation.baselines import MeanReconstructionBaseline
    from offshore_dl.evaluation.cv import TemporalSplitCV
    from offshore_dl.evaluation.metrics import MetricRegistry

    print("\n" + "=" * 60)
    print("  CDF: Mean Reconstruction Baseline")
    print("=" * 60)

    t0 = time.time()

    ds = CDFDataset(
        "configs/data/cdf.yaml",
        mode="reconstruction",
        window_stride=1,
    )

    cv = TemporalSplitCV(train_ratio=0.8)
    splits = cv.get_splits(len(ds))

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Extract windows
        X_train = np.stack([ds[i][0].numpy() for i in train_idx])
        X_val = np.stack([ds[i][0].numpy() for i in val_idx])

        baseline = MeanReconstructionBaseline()
        baseline.fit(X_train)
        preds = baseline.predict(X_val)

        metrics = MetricRegistry.compute("anomaly", preds, X_val)
        fold_results.append(metrics)
        print(f"  Split {fold_idx}: error_mean={metrics['error_mean']:.4f}, p95={metrics['error_p95']:.4f}")

    agg = _aggregate_folds(fold_results)
    elapsed = time.time() - t0
    print(f"  Mean±Std: error_mean={agg['error_mean_mean']:.4f}±{agg['error_mean_std']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    result = {"folds": fold_results, "aggregate": agg, "elapsed": elapsed}
    _save_json(result, results_dir / "cdf_mean_reconstruction_baseline.json")
    return result


def _aggregate_folds(fold_results: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean±std."""
    if not fold_results:
        return {}

    keys = fold_results[0].keys()
    agg = {}
    for key in keys:
        values = [f[key] for f in fold_results if isinstance(f.get(key), (int, float)) and np.isfinite(f[key])]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg


def _save_json(data: dict, path: Path) -> None:
    """Save results dict as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


def main() -> int:
    """Run all baseline evaluations."""
    results_dir = Path("results/baselines")
    results_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for runner_name, runner in [
        ("3W", _run_3w_baseline),
        ("Ganymede", _run_ganymede_baseline),
        ("CDF", _run_cdf_baseline),
    ]:
        try:
            runner(results_dir)
        except Exception as e:
            print(f"\n  ✗ {runner_name} FAILED: {e}")
            logger.exception("Baseline failed: %s", runner_name)
            success = False

    print("\n" + "=" * 60)
    if success:
        print("  All 3 baselines complete ✓")
        return 0
    else:
        print("  Some baselines FAILED ✗")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
