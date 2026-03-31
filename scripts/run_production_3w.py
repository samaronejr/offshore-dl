"""Production training: LSTM, DeepONet, PatchTST on full 3W dataset.

Trains all three models on the complete 3W dataset (no
``max_instances_per_class`` limit) using ``TemporalSplitCV(train_ratio=0.7)``
— NOT StratifiedGroupKFoldCV (D#41: folds_clf_02.csv incompatible with
3W v2.0.0).

Uses direct ``ExperimentRunner`` construction (approach c from the plan)
to avoid modifying DATASET_REGISTRY defaults.

Usage::

    # Full production training (GPU)
    python scripts/run_production_3w.py

    # Smoke test (CPU, 1 epoch)
    python scripts/run_production_3w.py --max-epochs 1 --device cpu

    # Docker invocation
    docker_run.sh python scripts/run_production_3w.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Allow invocation from project root or via docker_run.sh
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from offshore_dl.data.datasets import ThreeWDataset
from offshore_dl.evaluation.cv import TemporalSplitCV
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.patchtst import PatchTSTModel
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Model registry (class + config path + task-specific kwargs)
# ═══════════════════════════════════════════════════════════════════

MODELS: dict[str, dict] = {
    "lstm": {
        "class": LSTMModel,
        "config": "configs/models/lstm.yaml",
    },
    "deeponet": {
        "class": DeepONetModel,
        "config": "configs/models/deeponet.yaml",
    },
    "patchtst": {
        "class": PatchTSTModel,
        "config": "configs/models/patchtst.yaml",
    },
}

RESULTS_DIR = Path("results")


def _make_serializable(obj):
    """Convert non-serializable types for JSON output."""
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


def _run_model(
    model_name: str,
    dataset: ThreeWDataset,
    max_epochs: int,
    batch_size: int,
    device: str,
) -> dict:
    """Train one model on the full 3W dataset with TemporalSplitCV.

    Follows approach (c): direct ExperimentRunner construction, bypassing
    DATASET_REGISTRY (whose cv_factory defaults to StratifiedGroupKFoldCV).
    """
    set_global_seed(42)

    entry = MODELS[model_name]
    model_class = entry["class"]

    # Load merged config: base + data + model
    cfg = load_merged_config(
        "configs/base.yaml",
        "configs/data/3w.yaml",
        entry["config"],
    )

    # Apply CLI overrides
    cfg.training.max_epochs = max_epochs
    cfg.training.batch_size = batch_size
    cfg.device = device

    # TemporalSplitCV — NOT StratifiedGroupKFoldCV
    cv = TemporalSplitCV(train_ratio=0.7)

    # Build model kwargs: task-specific + architecture from config
    model_kwargs = {
        "task": "classification",
        "n_vars": 27,
        "n_classes": cfg.data.n_classes,
        "window_size": cfg.data.preprocessing.window_size,
    }

    # Merge architecture params from model config
    if hasattr(cfg, "model") and hasattr(cfg.model, "architecture"):
        arch = OmegaConf.to_container(cfg.model.architecture, resolve=True)
        model_kwargs.update(arch)

    # Merge training LR/weight_decay from model config
    if hasattr(cfg, "model") and hasattr(cfg.model, "training"):
        model_kwargs["lr"] = cfg.model.training.lr
        model_kwargs["weight_decay"] = cfg.model.training.weight_decay

    runner = ExperimentRunner(
        model_class=model_class,
        dataset=dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=model_kwargs,
    )

    return runner.run(use_mlflow=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production training: 3 models on full 3W (TemporalSplitCV)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")

    args = parser.parse_args()

    logger.info("═" * 70)
    logger.info("3W PRODUCTION TRAINING — 3 models on full dataset")
    logger.info("  device=%s  max_epochs=%d  batch_size=%d", args.device, args.max_epochs, args.batch_size)
    logger.info("═" * 70)

    # Load dataset once — shared across all models
    logger.info("Loading full 3W dataset (no max_instances_per_class) …")
    ds_start = time.time()
    dataset = ThreeWDataset("configs/data/3w.yaml")
    logger.info("  3W loaded: %d samples (%.1fs)", len(dataset), time.time() - ds_start)

    sweep_start = time.time()
    summary: dict[str, dict] = {}

    for model_name in MODELS:
        logger.info("─" * 60)
        logger.info("TRAINING: %s on 3W (full dataset)", model_name.upper())
        logger.info("─" * 60)

        start = time.time()
        try:
            results = _run_model(
                model_name=model_name,
                dataset=dataset,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                device=args.device,
            )
            elapsed = time.time() - start

            # Save results — overwrites baseline (production supersedes)
            out_path = RESULTS_DIR / model_name / "3w.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(_make_serializable(results), indent=2))
            logger.info("  Results saved: %s", out_path)

            agg = results.get("aggregate", {})
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(agg.items()) if "_mean" in k
            )
            summary[model_name] = {
                "status": "ok",
                "elapsed": round(elapsed, 1),
                "aggregate": agg,
                "n_folds": results.get("n_folds", 0),
            }
            logger.info("✓ %s: %s (%.1fs)", model_name, metric_str, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            summary[model_name] = {
                "status": "error",
                "elapsed": round(elapsed, 1),
                "error": str(e),
            }
            logger.error("✗ %s failed: %s (%.1fs)", model_name, e, elapsed)
            traceback.print_exc()

    # ── Final report ─────────────────────────────────────────────
    total_elapsed = time.time() - sweep_start

    print(f"\n{'═'*70}")
    print(f"  3W PRODUCTION TRAINING COMPLETE")
    print(f"{'═'*70}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    for model_name, s in summary.items():
        if s["status"] == "ok":
            agg = s["aggregate"]
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(agg.items()) if "_mean" in k
            )
            print(f"    {model_name:12s} ✓ {s['elapsed']:8.1f}s  {metric_str}")
        else:
            print(f"    {model_name:12s} ✗ {s['elapsed']:8.1f}s  ERROR: {s['error']}")
    print(f"{'═'*70}\n")

    # Save summary
    summary_path = RESULTS_DIR / "summary_production_3w.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    logger.info("Summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
