"""Optuna HPO for trained models on 3W and Ganymede.

Runs hyperparameter optimization using search spaces from YAML configs.
Each trial runs inner K-fold CV; the best params are then used for a
final nested evaluation (retrain on full pool → held-out test).

Usage::

    # 3W classification (all models, 30 trials each)
    python scripts/run_optuna_hpo.py --dataset 3w --n-trials 30 --device cuda

    # Ganymede forecasting (specific model, specific horizon)
    python scripts/run_optuna_hpo.py --dataset ganymede --models patchtst --horizon h7 --device cuda

    # Smoke test
    python scripts/run_optuna_hpo.py --dataset 3w --models lstm --n-trials 2 --device cpu
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

from offshore_dl.utils.config import load_merged_config
from offshore_dl.data.datasets import ThreeWFeatureDataset, GanymedeDataset
from offshore_dl.evaluation.cv import (
    HoldoutSplitter,
    StratifiedGroupKFoldSKLearn,
    ExpandingWindowCV,
)
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.mlp import MLPModel
from offshore_dl.models.patchtst import PatchTSTModel
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.training.optuna_utils import run_hpo
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


# ═══════════════════════════════════════════════════════════════════
# 3W CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

THREE_W_MODELS = {
    "lstm": {
        "class": LSTMModel,
        "config": "configs/models/lstm.yaml",
        "kwargs": {
            "task": "classification",
            "n_vars": 27,
            "window_size": 14,
            "n_classes": 10,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
        },
    },
    "deeponet": {
        "class": DeepONetModel,
        "config": "configs/models/deeponet.yaml",
        "kwargs": {
            "task": "classification",
            "n_vars": 27,
            "window_size": 14,
            "n_classes": 10,
            "rank": 128,
            "branch_hidden": [128, 128],
            "dropout": 0.2,
        },
    },
    "patchtst": {
        "class": PatchTSTModel,
        "config": "configs/models/patchtst.yaml",
        "kwargs": {
            "task": "classification",
            "n_vars": 27,
            "window_size": 14,
            "n_classes": 10,
            "pretrained": False,
            "d_model": 256,
            "d_ff": 512,
            "n_heads": 8,
            "n_layers": 3,
            "patch_len": 7,
            "stride": 4,
            "dropout": 0.15,
        },
    },
    "mlp": {
        "class": MLPModel,
        "config": "configs/models/mlp.yaml",
        "kwargs": {
            "task": "classification",
            "n_vars": 27,
            "window_size": 14,
            "n_classes": 10,
            "hidden_dims": [256, 128],
            "dropout": 0.3,
        },
    },
}


def run_3w_hpo(model_name: str, n_trials: int, device: str) -> dict:
    """Run HPO for a 3W model."""
    set_global_seed(42)

    entry = THREE_W_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )
    logger.info("  Search space: %s", list(search_space.keys()))

    cfg = load_merged_config("configs/base.yaml", "configs/data/3w.yaml", entry["config"])
    cfg.training.max_epochs = 50  # shorter per trial for HPO
    cfg.training.batch_size = 64
    cfg.device = device
    cfg.training.scheduler = "cosine"

    dataset = ThreeWFeatureDataset("configs/data/3w.yaml")
    logger.info("  3W loaded: %d samples", len(dataset))

    # Holdout split for final evaluation
    labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])
    groups = np.array([dataset[i][2].get("instance_id", i) for i in range(len(dataset))])
    holdout = HoldoutSplitter(
        test_ratio=0.2, mode="stratified_group",
        labels=labels, groups=groups,
    )
    train_pool, test_indices = holdout.split(len(dataset))
    logger.info("  Holdout: train=%d, test=%d", len(train_pool), len(test_indices))

    # Inner CV for HPO trials — needs labels/groups for stratified split
    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    cv = StratifiedGroupKFoldSKLearn(
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
    )

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=entry["kwargs"],
        primary_metric="f1_macro",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"3w_{model_name}",
        direction="maximize",
    )

    logger.info("  Best params: %s (value=%.4f, trials=%d)",
                hpo_result["best_params"], hpo_result["best_value"],
                hpo_result["n_trials_completed"])

    # ── Final evaluation with best params ──
    best_kwargs = dict(entry["kwargs"])
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/3w.yaml", entry["config"])
    best_cfg.training.max_epochs = 50  # same as HPO trials — lr schedule must match
    best_cfg.training.batch_size = 64
    best_cfg.device = device
    best_cfg.training.scheduler = "cosine"

    from offshore_dl.training.optuna_utils import OptunaObjective
    for param_name, value in hpo_result["best_params"].items():
        if param_name in OptunaObjective.TRAINING_PARAMS:
            OmegaConf.update(best_cfg, f"training.{param_name}", value)
        else:
            best_kwargs[param_name] = value

    # Translate branch_width → branch_hidden (list of 2)
    if "branch_width" in best_kwargs:
        w = best_kwargs.pop("branch_width")
        best_kwargs["branch_hidden"] = [w, w]

    final_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
    )
    runner = ExperimentRunner(
        model_class=entry["class"],
        dataset=dataset,
        cv_strategy=final_cv,
        cfg=best_cfg,
        model_kwargs=best_kwargs,
    )
    nested_result = runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=False,
    )

    return {
        "hpo": {
            "best_params": hpo_result["best_params"],
            "best_value": hpo_result["best_value"],
            "n_trials": hpo_result["n_trials_completed"],
        },
        "test_metrics": nested_result["test_metrics"],
        "cv_aggregate": nested_result["cv_aggregate"],
        "baseline_kwargs": entry["kwargs"],
        "best_kwargs": best_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════
# GANYMEDE FORECASTING
# ═══════════════════════════════════════════════════════════════════

GANYMEDE_MODELS = {
    "lstm": {
        "class": LSTMModel,
        "config": "configs/models/lstm.yaml",
        "kwargs": {
            "task": "forecasting",
            "n_vars": 63,
            "window_size": 30,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
        },
    },
    "deeponet": {
        "class": DeepONetModel,
        "config": "configs/models/deeponet.yaml",
        "kwargs": {
            "task": "forecasting",
            "n_vars": 63,
            "window_size": 30,
            "rank": 128,
            "branch_hidden": [128, 128],
            "dropout": 0.2,
        },
    },
    "patchtst": {
        "class": PatchTSTModel,
        "config": "configs/models/patchtst.yaml",
        "kwargs": {
            "task": "forecasting",
            "n_vars": 63,
            "window_size": 30,
            "pretrained": False,
            "d_model": 256,
            "d_ff": 512,
            "n_heads": 8,
            "n_layers": 3,
            "patch_len": 8,
            "stride": 4,
            "dropout": 0.15,
        },
    },
}


def run_ganymede_hpo(
    model_name: str, horizon: str, n_trials: int, device: str,
) -> dict:
    """Run HPO for a Ganymede model at a specific horizon."""
    set_global_seed(42)

    entry = GANYMEDE_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )

    cfg = load_merged_config("configs/base.yaml", "configs/data/ganymede.yaml", entry["config"])
    cfg.training.max_epochs = 50
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    horizon_days = int(horizon.replace("h", ""))
    dataset = GanymedeDataset("configs/data/ganymede.yaml", horizon=horizon_days)
    logger.info("  Ganymede %s loaded: %d samples", horizon, len(dataset))

    # Temporal holdout
    holdout = HoldoutSplitter(test_ratio=0.2, mode="temporal")
    train_pool, test_indices = holdout.split(len(dataset))

    # Inner CV for HPO
    cv = ExpandingWindowCV(n_splits=3)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs=entry["kwargs"],
        primary_metric="mae",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"ganymede_{model_name}_{horizon}",
    )

    logger.info("  Best params: %s (value=%.4f)", hpo_result["best_params"], hpo_result["best_value"])

    # Final evaluation with best params
    best_kwargs = dict(entry["kwargs"])
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/ganymede.yaml", entry["config"])
    best_cfg.training.max_epochs = 50  # same as HPO trials — lr schedule must match
    best_cfg.training.batch_size = 32
    best_cfg.device = device
    best_cfg.training.scheduler = "cosine"

    from offshore_dl.training.optuna_utils import OptunaObjective
    for param_name, value in hpo_result["best_params"].items():
        if param_name in OptunaObjective.TRAINING_PARAMS:
            OmegaConf.update(best_cfg, f"training.{param_name}", value)
        else:
            best_kwargs[param_name] = value

    # Translate branch_width → branch_hidden (list of 2)
    if "branch_width" in best_kwargs:
        w = best_kwargs.pop("branch_width")
        best_kwargs["branch_hidden"] = [w, w]

    final_cv = ExpandingWindowCV(n_splits=3)
    runner = ExperimentRunner(
        model_class=entry["class"],
        dataset=dataset,
        cv_strategy=final_cv,
        cfg=best_cfg,
        model_kwargs=best_kwargs,
    )
    nested_result = runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=False,
    )

    return {
        "hpo": {
            "best_params": hpo_result["best_params"],
            "best_value": hpo_result["best_value"],
            "n_trials": hpo_result["n_trials_completed"],
        },
        "test_metrics": nested_result["test_metrics"],
        "cv_aggregate": nested_result["cv_aggregate"],
        "baseline_kwargs": entry["kwargs"],
        "best_kwargs": best_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════

def _make_serializable(obj):
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


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for trained models")
    parser.add_argument("--dataset", required=True, choices=["3w", "ganymede"],
                        help="Dataset to optimize on")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to optimize (default: all trained)")
    parser.add_argument("--horizon", default="h7",
                        help="Ganymede horizon (h7/h14/h30/h90)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials per model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.dataset == "3w":
        model_registry = THREE_W_MODELS
        default_models = ["lstm", "deeponet", "patchtst", "mlp"]
    else:
        model_registry = GANYMEDE_MODELS
        default_models = ["lstm", "deeponet", "patchtst"]

    models = args.models or default_models

    logger.info("═" * 70)
    logger.info("OPTUNA HPO — %s", args.dataset.upper())
    logger.info("  models=%s  n_trials=%d  device=%s", models, args.n_trials, args.device)
    logger.info("═" * 70)

    summary = {}
    for model_name in models:
        if model_name not in model_registry:
            logger.warning("Unknown model: %s (skipping)", model_name)
            continue

        logger.info("─" * 60)
        logger.info("HPO: %s", model_name)
        logger.info("─" * 60)
        start = time.time()
        try:
            if args.dataset == "3w":
                result = run_3w_hpo(model_name, args.n_trials, args.device)
            else:
                result = run_ganymede_hpo(model_name, args.horizon, args.n_trials, args.device)

            elapsed = time.time() - start

            # Save per-model result
            out_dir = RESULTS_DIR / "hpo" / args.dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{args.horizon}" if args.dataset == "ganymede" else ""
            out_path = out_dir / f"{model_name}{suffix}.json"
            out_path.write_text(json.dumps(_make_serializable(result), indent=2))

            tm = result.get("test_metrics", {})
            if args.dataset == "3w":
                logger.info("✓ %s: test acc=%.4f, F1m=%.4f (best_trial_value=%.4f, %d trials, %.0fs)",
                            model_name, tm.get("accuracy", 0), tm.get("f1_macro", 0),
                            result["hpo"]["best_value"], result["hpo"]["n_trials"], elapsed)
            else:
                logger.info("✓ %s: test MAE=%.4f, R2p=%.4f (best_trial_value=%.4f, %d trials, %.0fs)",
                            model_name, tm.get("mae", 0), tm.get("r2_prod", 0),
                            result["hpo"]["best_value"], result["hpo"]["n_trials"], elapsed)

            summary[model_name] = {
                "status": "ok", "elapsed": round(elapsed, 1),
                "best_params": result["hpo"]["best_params"],
                "test_metrics": tm,
            }

        except Exception as e:
            elapsed = time.time() - start
            logger.error("✗ %s failed: %s (%.1fs)", model_name, e, elapsed)
            traceback.print_exc()
            summary[model_name] = {"status": "error", "error": str(e), "elapsed": round(elapsed, 1)}

    # Save summary
    summary_path = RESULTS_DIR / "hpo" / f"summary_{args.dataset}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    logger.info("HPO summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
