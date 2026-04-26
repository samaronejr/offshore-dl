"""Production training: all 6 models on CDF anomaly detection with SlidingWindowCV.

Usage::

    python scripts/run_production_cdf.py --device cuda
    python scripts/run_production_cdf.py --max-epochs 5 --device cpu  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from offshore_dl.data.datasets import CDFDataset
from offshore_dl.evaluation.cv import SlidingWindowCV
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.patchtst import PatchTSTModel
from offshore_dl.training.experiment import ExperimentRunner, NormalizedSubset
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

# ── Trained model configs ──

TRAINED_MODELS: dict[str, dict] = {
    "lstm": {
        "class": LSTMModel,
        "config": "configs/models/lstm.yaml",
        "kwargs": {
            "task": "anomaly",
            "n_vars": 11,
            "window_size": 48,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": True,
            "lr": 1e-3,
        },
    },
    "deeponet": {
        "class": DeepONetModel,
        "config": "configs/models/deeponet.yaml",
        "kwargs": {
            "task": "anomaly",
            "n_vars": 11,
            "window_size": 48,
            "branch_hidden": [64, 64],
            "trunk_hidden": [64, 64],
            "rank": 64,
            "dropout": 0.15,
            "lr": 5e-4,
        },
    },
    "patchtst": {
        "class": PatchTSTModel,
        "config": "configs/models/patchtst.yaml",
        "kwargs": {
            "task": "anomaly",
            "n_vars": 11,
            "window_size": 48,
            "patch_len": 8,
            "stride": 4,
            "d_model": 64,
            "d_ff": 128,
            "n_heads": 4,
            "n_layers": 2,
            "lr": 5e-4,
        },
    },
}


from offshore_dl.utils.serialization import make_serializable as _make_serializable


def run_trained_model(
    model_name: str,
    dataset: CDFDataset,
    max_epochs: int,
    device: str,
) -> dict:
    """Train one model on CDF with nested CV (holdout + inner SlidingWindowCV)."""
    set_global_seed(42)

    entry = TRAINED_MODELS[model_name]
    cfg = load_merged_config(
        "configs/base.yaml",
        "configs/data/cdf.yaml",
        entry["config"],
    )
    cfg.training.max_epochs = max_epochs
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    cv = SlidingWindowCV(n_splits=3, train_ratio=0.7)

    # Temporal holdout: last 20% as test
    from offshore_dl.evaluation.cv import HoldoutSplitter

    holdout = HoldoutSplitter(test_ratio=0.2, mode="temporal")
    train_pool, test_indices = holdout.split(len(dataset))
    logger.info(
        "  CDF holdout: train_pool=%d, test=%d", len(train_pool), len(test_indices)
    )

    runner = ExperimentRunner(
        model_class=entry["class"],
        dataset=dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=entry["kwargs"],
    )
    return runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=True,
    )


def run_fm_cdf(model_name: str, dataset: CDFDataset, device: str) -> dict:
    """Run a foundation model on CDF with nested protocol (holdout + inner SlidingWindowCV)."""
    set_global_seed(42)

    fm_cfg = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=False))
    fm_cfg.data.preprocessing.mode = "prediction"
    fm_cfg.data.preprocessing.prediction_horizon = dataset.window_size
    fm_dataset = CDFDataset(fm_cfg)

    from offshore_dl.evaluation.cv import HoldoutSplitter

    holdout = HoldoutSplitter(test_ratio=0.2, mode="temporal")
    train_pool, test_indices = holdout.split(len(fm_dataset))
    logger.info(
        "  CDF FM holdout: train_pool=%d, test=%d", len(train_pool), len(test_indices)
    )

    cv = SlidingWindowCV(n_splits=3, train_ratio=0.7)

    from offshore_dl.evaluation.metrics import MetricRegistry

    def _normalize_windows(
        windows: np.ndarray, mean: torch.Tensor, std: torch.Tensor
    ) -> np.ndarray:
        return ((windows - mean.cpu().numpy()) / std.cpu().numpy()).astype(np.float32)

    def _rename_fm_metrics(metrics: dict) -> dict:
        renamed = dict(metrics)
        if "error_mean" in renamed:
            renamed["forecast_error_mean"] = renamed.pop("error_mean")
        return renamed

    def _predict_fm(model_name, inputs):
        """Generate FM predictions for a batch of windows. inputs: np.ndarray (N, 48, 11)."""
        import torch as _torch

        if model_name == "chronos":
            from offshore_dl.models.chronos_wrapper import ChronosWrapper

            model = ChronosWrapper(task="anomaly", n_vars=11, window_size=48)
        elif model_name == "timesfm":
            from offshore_dl.models.timesfm_wrapper import TimesFMWrapper

            model = TimesFMWrapper(task="anomaly", n_vars=11, window_size=48)
        elif model_name == "tirex":
            from offshore_dl.models.tirex_wrapper import TiRexWrapper

            model = TiRexWrapper(task="anomaly", n_vars=11, window_size=48)
        else:
            raise ValueError(f"Unknown FM: {model_name}")

        # Process in batches to avoid OOM
        batch_size = 32
        all_preds = []
        for i in range(0, len(inputs), batch_size):
            batch = _torch.tensor(inputs[i : i + batch_size], dtype=_torch.float32)
            with _torch.no_grad():
                preds = model.forward(batch)
            all_preds.append(preds.numpy())
        return np.concatenate(all_preds, axis=0)

    # ── Inner CV on training pool ──
    inner_splits = cv.get_splits(len(train_pool))
    fold_results = []
    for fold_idx, (tr_rel, val_rel) in enumerate(inner_splits):
        # Map relative indices → absolute dataset indices
        val_idx = train_pool[val_rel]
        tr_idx = train_pool[tr_rel]

        logger.info(
            "  %s inner fold %d/%d (val=%d)",
            model_name,
            fold_idx + 1,
            len(inner_splits),
            len(val_idx),
        )

        mean, std = NormalizedSubset.compute_stats(fm_dataset, tr_idx)

        val_inputs = np.stack([fm_dataset[i][0].numpy() for i in val_idx])
        val_targets = np.stack([fm_dataset[i][1].numpy() for i in val_idx])
        val_inputs = _normalize_windows(val_inputs, mean, std)
        val_targets = _normalize_windows(val_targets, mean, std)

        predictions = _predict_fm(model_name, val_inputs)

        metrics = _rename_fm_metrics(
            MetricRegistry.compute("anomaly", predictions, val_targets)
        )
        fold_results.append({"fold_idx": fold_idx, "metrics": metrics})

    # CV aggregate
    cv_agg = {}
    all_keys = set()
    for fr in fold_results:
        all_keys.update(fr["metrics"].keys())
    for key in sorted(all_keys):
        vals = [
            fr["metrics"].get(key, 0)
            for fr in fold_results
            if isinstance(fr["metrics"].get(key, 0), (int, float))
        ]
        if vals:
            cv_agg[f"{key}_mean"] = float(np.mean(vals))
            cv_agg[f"{key}_std"] = float(np.std(vals))

    # ── Evaluate on held-out test set ──
    logger.info(
        "  %s evaluating on held-out test (%d samples)", model_name, len(test_indices)
    )

    test_mean, test_std = NormalizedSubset.compute_stats(fm_dataset, train_pool)
    test_inputs = np.stack([fm_dataset[i][0].numpy() for i in test_indices])
    test_targets = np.stack([fm_dataset[i][1].numpy() for i in test_indices])
    test_inputs = _normalize_windows(test_inputs, test_mean, test_std)
    test_targets = _normalize_windows(test_targets, test_mean, test_std)

    test_preds = _predict_fm(model_name, test_inputs)

    test_metrics = _rename_fm_metrics(
        MetricRegistry.compute("anomaly", test_preds, test_targets)
    )

    return {
        "test_metrics": test_metrics,
        "cv_aggregate": cv_agg,
        "cv_fold_results": fold_results,
        "metric_note": "FM error is one-step-ahead forecasting error on normalized inputs; trained model error is multi-step reconstruction error.",
        "n_train": len(train_pool),
        "n_test": len(test_indices),
        "n_cv_folds": len(inner_splits),
    }


def main():
    set_global_seed(42)

    parser = argparse.ArgumentParser(
        description="CDF anomaly detection — all models, 3-fold sliding window CV"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run (default: all). Options: lstm deeponet patchtst chronos timesfm tirex",
    )
    args = parser.parse_args()

    all_models = ["lstm", "deeponet", "patchtst", "chronos", "timesfm", "tirex"]
    models = args.models or all_models

    logger.info("═" * 70)
    logger.info("CDF ANOMALY DETECTION — 3-fold Sliding Window CV")
    logger.info(
        "  device=%s  max_epochs=%d  models=%s", args.device, args.max_epochs, models
    )
    logger.info("═" * 70)

    dataset = CDFDataset("configs/data/cdf.yaml")
    logger.info("CDF loaded: %d samples, %d vars", len(dataset), dataset.n_vars)

    summary = {}
    for model_name in models:
        logger.info("─" * 60)
        logger.info("RUN: %s", model_name)
        logger.info("─" * 60)
        start = time.time()
        try:
            if model_name in TRAINED_MODELS:
                result = run_trained_model(
                    model_name, dataset, args.max_epochs, args.device
                )
            else:
                result = run_fm_cdf(model_name, dataset, args.device)

            elapsed = time.time() - start
            out_path = RESULTS_DIR / model_name / "cdf.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(_make_serializable(result), indent=2))

            # Extract metrics for summary — nested uses test_metrics, old CV uses aggregate
            agg = result.get("test_metrics", result.get("aggregate", {}))
            metric_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(agg.items())
                if isinstance(v, (int, float))
            )
            logger.info("✓ %s: %s (%.1fs)", model_name, metric_str, elapsed)
            summary[model_name] = {
                "status": "ok",
                "elapsed": round(elapsed, 1),
                "test_metrics": agg,
            }

        except Exception as e:
            elapsed = time.time() - start
            logger.error("✗ %s failed: %s (%.1fs)", model_name, e, elapsed)
            traceback.print_exc()
            summary[model_name] = {
                "status": "error",
                "error": str(e),
                "elapsed": round(elapsed, 1),
            }

    # Save summary
    summary_path = RESULTS_DIR / "summary_production_cdf.json"
    summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    logger.info("Summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
