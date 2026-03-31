"""Run LSTM baseline on all 3 datasets and save results.

Quick CPU-feasible verification run with small epoch counts.
Full training runs use more epochs on GPU with proper CV.

Usage:
    python scripts/run_lstm_baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from offshore_dl.data.datasets import CDFDataset, GanymedeDataset, ThreeWDataset
from offshore_dl.evaluation.cv import ExpandingWindowCV, SlidingWindowCV, TemporalSplitCV
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.training.trainer import Trainer
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_cdf_baseline() -> dict:
    """Run LSTM on CDF anomaly detection."""
    set_global_seed(42)
    dataset = CDFDataset("configs/data/cdf.yaml")
    cv = SlidingWindowCV(n_splits=5, train_ratio=0.7)
    cfg = OmegaConf.create({
        "training": {
            "batch_size": 32, "max_epochs": 5,
            "early_stopping_patience": 10, "gradient_clip_val": 1.0,
        },
    })
    runner = ExperimentRunner(
        model_class=LSTMModel,
        dataset=dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs={
            "task": "anomaly", "n_vars": 11, "window_size": 48,
            "hidden_size": 64, "num_layers": 1, "dropout": 0.1,
        },
    )
    return runner.run(use_mlflow=False)


def run_ganymede_baseline() -> dict:
    """Run LSTM on Ganymede gas production forecasting."""
    set_global_seed(42)
    dataset = GanymedeDataset("configs/data/ganymede.yaml")
    cv = ExpandingWindowCV(n_splits=3, min_train_ratio=0.5)
    cfg = OmegaConf.create({
        "training": {
            "batch_size": 64, "max_epochs": 3,
            "early_stopping_patience": 10, "gradient_clip_val": 1.0,
        },
    })
    n_vars = dataset[0][0].shape[-1]
    runner = ExperimentRunner(
        model_class=LSTMModel,
        dataset=dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs={
            "task": "forecasting", "n_vars": n_vars, "horizon": 30,
            "hidden_size": 64, "num_layers": 1, "dropout": 0.1,
        },
    )
    return runner.run(use_mlflow=False)


def run_3w_baseline() -> dict:
    """Run LSTM on 3W classification with limited instances.

    Uses max_instances_per_class=3 for CPU feasibility and TemporalSplitCV
    since the small random subset won't overlap with benchmark folds.
    Full runs use StratifiedGroupKFoldCV on GPU with all instances.
    """
    set_global_seed(42)
    dataset = ThreeWDataset(
        "configs/data/3w.yaml",
        max_instances_per_class=3,
    )
    cv = TemporalSplitCV(train_ratio=0.7)
    cfg = OmegaConf.create({
        "training": {
            "batch_size": 32, "max_epochs": 3,
            "early_stopping_patience": 10, "gradient_clip_val": 1.0,
        },
    })
    runner = ExperimentRunner(
        model_class=LSTMModel,
        dataset=dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs={
            "task": "classification", "n_vars": 27, "n_classes": 10,
            "window_size": 720, "hidden_size": 64, "num_layers": 1, "dropout": 0.1,
        },
    )
    return runner.run(use_mlflow=False)


def _make_serializable(obj):
    """Convert for JSON output."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if k != "study"}
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
    """Run all 3 baselines."""
    results_dir = Path("results/lstm")
    results_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("cdf", run_cdf_baseline),
        ("ganymede", run_ganymede_baseline),
        ("3w", run_3w_baseline),
    ]

    summary = {}
    for name, fn in runs:
        logger.info("=" * 60)
        logger.info("Starting LSTM on %s", name)
        logger.info("=" * 60)

        start = time.time()
        try:
            results = fn()
            elapsed = time.time() - start

            # Save results
            out_path = results_dir / f"{name}.json"
            out_path.write_text(json.dumps(_make_serializable(results), indent=2))

            agg = results.get("aggregate", {})
            summary[name] = {"status": "ok", "elapsed": round(elapsed, 1), "aggregate": agg}

            # Print key metrics
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(agg.items()) if "_mean" in k)
            logger.info("✓ %s: %s (%.1fs)", name, metric_str, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            summary[name] = {"status": "error", "elapsed": round(elapsed, 1), "error": str(e)}
            logger.error("✗ %s failed: %s (%.1fs)", name, e, elapsed)
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("  LSTM BASELINE RESULTS")
    print("=" * 60)
    for name, s in summary.items():
        if s["status"] == "ok":
            agg = s["aggregate"]
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(agg.items()) if "_mean" in k)
            print(f"  {name:12s} ✓ {s['elapsed']:6.1f}s  {metric_str}")
        else:
            print(f"  {name:12s} ✗ {s['elapsed']:6.1f}s  ERROR: {s['error']}")
    print("=" * 60)

    # Save summary
    (results_dir / "summary.json").write_text(json.dumps(_make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
