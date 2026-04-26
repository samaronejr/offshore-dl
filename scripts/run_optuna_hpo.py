"""Optuna HPO for trained models on 3W, Ganymede, and SPE BERG.

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
import optuna
import torch
from omegaconf import OmegaConf

from offshore_dl.utils.config import load_merged_config
from offshore_dl.data.datasets import ThreeWFeatureDataset, GanymedeDataset, SPEBergDataset, VolveDataset
from offshore_dl.evaluation.cv import (
    GroupedExpandingWindowCV,
    GroupedTemporalHoldoutSplitter,
    HoldoutSplitter,
    StratifiedGroupKFoldSKLearn,
)
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def _load_pytorch_deps():
    """Lazy-load PyTorch model classes and training utilities.

    Deferred to avoid importing transformers/torchvision at module level,
    which can fail when torchvision version is incompatible.
    """
    from offshore_dl.models.deeponet import DeepONetModel
    from offshore_dl.models.lstm import LSTMModel
    from offshore_dl.models.patchtst import PatchTSTModel
    from offshore_dl.models.tcn import TCNModel
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo
    try:
        from offshore_dl.models.fkmad import FKMADModel
    except (ImportError, ModuleNotFoundError, RuntimeError):
        FKMADModel = None
    try:
        from offshore_dl.models.mambasl import MambaSLModel
    except (ImportError, ModuleNotFoundError, RuntimeError):
        MambaSLModel = None
    try:
        from offshore_dl.models.convtimenet import ConvTimeNetModel
    except (ImportError, ModuleNotFoundError, RuntimeError):
        ConvTimeNetModel = None
    return {
        "LSTMModel": LSTMModel,
        "DeepONetModel": DeepONetModel,
        "PatchTSTModel": PatchTSTModel,
        "TCNModel": TCNModel,
        "FKMADModel": FKMADModel,
        "MambaSLModel": MambaSLModel,
        "ConvTimeNetModel": ConvTimeNetModel,
        "ExperimentRunner": ExperimentRunner,
        "run_hpo": run_hpo,
    }


def _get_3w_models():
    """Build 3W model registry (lazy, imports PyTorch models)."""
    deps = _load_pytorch_deps()
    models = {
        "lstm": {
            "class": deps["LSTMModel"],
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
            "class": deps["DeepONetModel"],
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
            "class": deps["PatchTSTModel"],
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
    }
    if deps["FKMADModel"] is not None:
        models["fkmad"] = {
            "class": deps["FKMADModel"],
            "config": "configs/models/fkmad.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "d_model": 128,
                "n_mamba_layers": 2,
                "dropout": 0.2,
            },
        }
    if deps["MambaSLModel"] is not None:
        models["mambasl"] = {
            "class": deps["MambaSLModel"],
            "config": "configs/models/mambasl.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "d_model": 128,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "d_ff": 256,
                "n_heads": 4,
                "dropout": 0.1,
            },
        }
    if deps["ConvTimeNetModel"] is not None:
        models["convtimenet"] = {
            "class": deps["ConvTimeNetModel"],
            "config": "configs/models/convtimenet.yaml",
            "kwargs": {
                "task": "classification",
                "n_vars": 27,
                "window_size": 14,
                "n_classes": 10,
                "d_model": 128,
                "d_ff": 256,
                "patch_size": 8,
                "patch_stride": 4,
                "dw_ks": [7, 13, 19],
                "dropout": 0.1,
                "pooling_tp": "max",
            },
        }
    return models


def _get_ganymede_models():
    """Build Ganymede model registry (lazy, imports PyTorch models)."""
    deps = _load_pytorch_deps()
    return {
        "lstm": {
            "class": deps["LSTMModel"],
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
            "class": deps["DeepONetModel"],
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
            "class": deps["PatchTSTModel"],
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
        "tcn": {
            "class": deps["TCNModel"],
            "config": "configs/models/tcn.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 63,
                "window_size": 90,
                "n_channels": 128,
                "n_layers": 6,
                "kernel_size": 3,
                "dropout": 0.2,
            },
        },
    }


def _get_spe_berg_models():
    """Build SPE BERG model registry (lazy, imports PyTorch models)."""
    deps = _load_pytorch_deps()
    return {
        "lstm": {
            "class": deps["LSTMModel"],
            "config": "configs/models/lstm.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
            },
        },
        "deeponet": {
            "class": deps["DeepONetModel"],
            "config": "configs/models/deeponet.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
                "rank": 128,
                "branch_hidden": [128, 128],
                "dropout": 0.2,
            },
        },
        "patchtst": {
            "class": deps["PatchTSTModel"],
            "config": "configs/models/patchtst.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
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
        "tcn": {
            "class": deps["TCNModel"],
            "config": "configs/models/tcn.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 67,
                "window_size": 90,
                "n_channels": 128,
                "n_layers": 6,
                "kernel_size": 3,
                "dropout": 0.2,
            },
        },
    }


def _get_volve_models():
    """Build Volve model registry (lazy, imports PyTorch models).

    n_vars=73 (Volve has 73 feature columns), window_size=90.
    PatchTST requires target_channel=48 because BORE_OIL_VOL is at column index 48.
    """
    deps = _load_pytorch_deps()
    return {
        "lstm": {
            "class": deps["LSTMModel"],
            "config": "configs/models/lstm.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "hidden_size": 256,
                "num_layers": 2,
                "dropout": 0.3,
                "bidirectional": True,
            },
        },
        "deeponet": {
            "class": deps["DeepONetModel"],
            "config": "configs/models/deeponet.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "rank": 128,
                "branch_hidden": [128, 128],
                "dropout": 0.2,
            },
        },
        "patchtst": {
            "class": deps["PatchTSTModel"],
            "config": "configs/models/patchtst.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "pretrained": False,
                "d_model": 256,
                "d_ff": 512,
                "n_heads": 8,
                "n_layers": 3,
                "patch_len": 8,
                "stride": 4,
                "dropout": 0.15,
                "target_channel": 48,
            },
        },
        "tcn": {
            "class": deps["TCNModel"],
            "config": "configs/models/tcn.yaml",
            "kwargs": {
                "task": "forecasting",
                "n_vars": 73,
                "window_size": 90,
                "n_channels": 128,
                "n_layers": 6,
                "kernel_size": 3,
                "dropout": 0.2,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════
# 3W CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════


def run_3w_hpo(model_name: str, n_trials: int, device: str) -> dict:
    """Run HPO for a 3W model."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    THREE_W_MODELS = _get_3w_models()
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
        use_mlflow=True,
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
# 3W CLASSIFICATION — SKLEARN MODELS (Random Forest, etc.)
# ═══════════════════════════════════════════════════════════════════


def _suggest_param(trial, name: str, spec: dict):
    """Sample a hyperparameter from its YAML spec (replicates OptunaObjective._suggest)."""
    suggest_type = spec.get("type", "float")
    if suggest_type == "float":
        return trial.suggest_float(
            name, spec.get("low", 0.0001), spec.get("high", 0.1),
            log=spec.get("log", False),
        )
    elif suggest_type == "int":
        return trial.suggest_int(name, spec.get("low", 1), spec.get("high", 100))
    elif suggest_type == "categorical":
        return trial.suggest_categorical(name, spec.get("choices", []))
    else:
        msg = f"Unknown suggest type: {suggest_type!r}"
        raise ValueError(msg)


def run_3w_rf_hpo(n_trials: int, device: str) -> dict:
    """Run sklearn Random Forest HPO for 3W classification.

    Custom Optuna objective: each trial samples RF hyperparameters from
    the YAML search space, runs 5-fold StratifiedGroupKFoldSKLearn inner CV,
    and maximizes mean f1_macro.  After the study completes, the best params
    are used to retrain on the full training pool and evaluate on held-out test.
    """
    from sklearn.ensemble import RandomForestClassifier

    set_global_seed(42)

    model_cfg = OmegaConf.load("configs/models/random_forest.yaml")
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True,
    )
    base_arch = OmegaConf.to_container(
        model_cfg.model.architecture, resolve=True,
    )
    logger.info("  RF search space: %s", list(search_space.keys()))

    # Load dataset and extract numpy arrays
    dataset = ThreeWFeatureDataset("configs/data/3w.yaml")
    n = len(dataset)
    logger.info("  3W loaded: %d samples", n)

    n_features = dataset[0][0].shape[0]  # window_size (14)
    n_sensors = dataset[0][0].shape[1]   # 27
    flat_dim = n_features * n_sensors

    X_all = np.empty((n, flat_dim), dtype=np.float32)
    Y_all = np.empty(n, dtype=np.int64)
    for i in range(n):
        x, y, _ = dataset[i]
        X_all[i] = x.numpy().reshape(-1)
        Y_all[i] = int(y)

    labels = np.array([int(dataset[i][1]) for i in range(n)])
    groups = np.array([dataset[i][2].get("instance_id", i) for i in range(n)])

    # Holdout split
    holdout = HoldoutSplitter(
        test_ratio=0.2, mode="stratified_group",
        labels=labels, groups=groups,
    )
    train_pool, test_indices = holdout.split(n)
    logger.info("  Holdout: train=%d, test=%d", len(train_pool), len(test_indices))

    X_train_pool, Y_train_pool = X_all[train_pool], Y_all[train_pool]
    X_test, Y_test = X_all[test_indices], Y_all[test_indices]

    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]

    # Inner CV for trials
    inner_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
    )
    inner_splits = inner_cv.get_splits(len(train_pool))

    # Optuna objective
    def objective(trial):
        params = {name: _suggest_param(trial, name, spec)
                  for name, spec in search_space.items()}
        # Merge sampled params with base architecture defaults
        rf_kwargs = dict(base_arch)
        rf_kwargs.update(params)

        fold_f1s = []
        for local_train, local_val in inner_splits:
            X_tr, Y_tr = X_train_pool[local_train], Y_train_pool[local_train]
            X_va, Y_va = X_train_pool[local_val], Y_train_pool[local_val]

            clf = RandomForestClassifier(**rf_kwargs)
            clf.fit(X_tr, Y_tr)
            preds = clf.predict(X_va)
            probs = clf.predict_proba(X_va)

            metrics = MetricRegistry.compute(
                "classification", preds, Y_va, prediction_scores=probs,
            )
            fold_f1s.append(metrics["f1_macro"])

        return float(np.mean(fold_f1s))

    # Create study and run optimization
    cfg = load_merged_config(
        "configs/base.yaml", "configs/data/3w.yaml",
        "configs/models/random_forest.yaml",
    )
    study = optuna.create_study(
        study_name="3w_random_forest",
        direction="maximize",
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    best_value = study.best_value
    n_completed = len(study.trials)
    logger.info("  Best params: %s (f1_macro=%.4f, trials=%d)",
                best_params, best_value, n_completed)

    # ── Final evaluation with best params ──
    final_kwargs = dict(base_arch)
    final_kwargs.update(best_params)

    # Setup MLflow
    mlflow = None
    try:
        import mlflow as _mlflow
        _mlflow.set_tracking_uri("mlruns")
        _mlflow.set_experiment("3w-random-forest-hpo")
        mlflow = _mlflow
    except ImportError:
        pass

    if mlflow:
        mlflow.start_run(run_name="rf_hpo_best_retrain")
        mlflow.log_params({k: str(v) for k, v in best_params.items()})
        mlflow.log_metric("hpo_best_f1_macro", best_value)

    # Inner CV with best params for cv_aggregate
    cv_fold_results = []
    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
        X_tr, Y_tr = X_train_pool[local_train], Y_train_pool[local_train]
        X_va, Y_va = X_train_pool[local_val], Y_train_pool[local_val]

        clf = RandomForestClassifier(**final_kwargs)
        clf.fit(X_tr, Y_tr)
        preds = clf.predict(X_va)
        probs = clf.predict_proba(X_va)
        metrics = MetricRegistry.compute(
            "classification", preds, Y_va, prediction_scores=probs,
        )
        cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})
        if mlflow:
            mlflow.log_metric(f"cv_fold_{fold_idx}_f1_macro", metrics["f1_macro"])

    cv_agg = {}
    if cv_fold_results:
        metric_keys = [
            k for k in cv_fold_results[0]["metrics"]
            if isinstance(cv_fold_results[0]["metrics"][k], (int, float))
        ]
        for k in metric_keys:
            vals = [f["metrics"][k] for f in cv_fold_results]
            cv_agg[f"{k}_mean"] = float(np.mean(vals))
            cv_agg[f"{k}_std"] = float(np.std(vals))

    # Retrain on full pool → held-out test
    final_clf = RandomForestClassifier(**final_kwargs)
    final_clf.fit(X_train_pool, Y_train_pool)
    test_preds = final_clf.predict(X_test)
    test_probs = final_clf.predict_proba(X_test)
    test_metrics = MetricRegistry.compute(
        "classification", test_preds, Y_test, prediction_scores=test_probs,
    )

    if mlflow:
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)
        mlflow.end_run()

    metric_str = ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(test_metrics.items())
        if isinstance(v, (int, float))
    )
    logger.info("  RF HPO final — TEST: %s", metric_str)

    return {
        "hpo": {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_completed,
        },
        "test_metrics": test_metrics,
        "cv_aggregate": cv_agg,
        "cv_fold_results": cv_fold_results,
        "baseline_kwargs": base_arch,
        "best_kwargs": final_kwargs,
    }


# ═══════════════════════════════════════════════════════════════════
# GANYMEDE FORECASTING (PyTorch models)
# ═══════════════════════════════════════════════════════════════════


def run_ganymede_hpo(
    model_name: str, horizon: str, n_trials: int, device: str,
) -> dict:
    """Run HPO for a Ganymede model at a specific horizon."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    GANYMEDE_MODELS = _get_ganymede_models()
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
    dataset = GanymedeDataset("configs/data/ganymede.yaml", horizon=horizon_days, filter_shutdowns=False)
    logger.info("  Ganymede %s loaded: %d samples", horizon, len(dataset))
    groups = np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)

    # Temporal holdout
    holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups)
    train_pool, test_indices = holdout.split(len(dataset))

    # Inner CV for HPO
    cv = GroupedExpandingWindowCV(groups=groups[train_pool], n_splits=3)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs={**entry["kwargs"], "horizon": horizon_days},
        primary_metric="mae",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"ganymede_{model_name}_{horizon}",
    )

    logger.info("  Best params: %s (value=%.4f)", hpo_result["best_params"], hpo_result["best_value"])

    # Final evaluation with best params
    best_kwargs = {**entry["kwargs"], "horizon": horizon_days}
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

    final_cv = GroupedExpandingWindowCV(groups=groups, n_splits=3)
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
        use_mlflow=True,
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
# SPE BERG FORECASTING (PyTorch models)
# ═══════════════════════════════════════════════════════════════════


def run_spe_berg_hpo(
    model_name: str, horizon: str, n_trials: int, device: str,
) -> dict:
    """Run HPO for a SPE BERG model at a specific horizon."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    SPE_BERG_MODELS = _get_spe_berg_models()
    entry = SPE_BERG_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )

    cfg = load_merged_config("configs/base.yaml", "configs/data/spe_berg.yaml", entry["config"])
    cfg.training.max_epochs = 50
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    horizon_days = int(horizon.replace("h", ""))
    dataset = SPEBergDataset("configs/data/spe_berg.yaml", horizon=horizon_days, filter_shutdowns=False)
    logger.info("  SPE BERG %s loaded: %d samples", horizon, len(dataset))
    groups = np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)

    # Temporal holdout
    holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups)
    train_pool, test_indices = holdout.split(len(dataset))

    # Inner CV for HPO
    cv = GroupedExpandingWindowCV(groups=groups[train_pool], n_splits=3)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs={**entry["kwargs"], "horizon": horizon_days},
        primary_metric="mae",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"spe_berg_{model_name}_{horizon}",
    )

    logger.info("  Best params: %s (value=%.4f)", hpo_result["best_params"], hpo_result["best_value"])

    # Final evaluation with best params
    best_kwargs = {**entry["kwargs"], "horizon": horizon_days}
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/spe_berg.yaml", entry["config"])
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

    final_cv = GroupedExpandingWindowCV(groups=groups, n_splits=3)
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
        use_mlflow=True,
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
# VOLVE FORECASTING (PyTorch models)
# ═══════════════════════════════════════════════════════════════════


def run_volve_hpo(
    model_name: str, horizon: str, n_trials: int, device: str,
) -> dict:
    """Run HPO for a Volve model at a specific horizon."""
    from offshore_dl.training.experiment import ExperimentRunner
    from offshore_dl.training.optuna_utils import run_hpo

    set_global_seed(42)

    VOLVE_MODELS = _get_volve_models()
    entry = VOLVE_MODELS[model_name]
    model_cfg = OmegaConf.load(entry["config"])
    search_space = OmegaConf.to_container(
        model_cfg.model.optuna_search_space, resolve=True
    )

    cfg = load_merged_config("configs/base.yaml", "configs/data/volve.yaml", entry["config"])
    cfg.training.max_epochs = 50
    cfg.training.batch_size = 32
    cfg.device = device
    cfg.training.scheduler = "cosine"

    horizon_days = int(horizon.replace("h", ""))
    dataset = VolveDataset("configs/data/volve.yaml", horizon=horizon_days, filter_shutdowns=False)
    logger.info("  Volve %s loaded: %d samples", horizon, len(dataset))
    groups = np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)

    # Temporal holdout
    holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups)
    train_pool, test_indices = holdout.split(len(dataset))

    # Inner CV for HPO
    cv = GroupedExpandingWindowCV(groups=groups[train_pool], n_splits=3)

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_pool.tolist())

    hpo_result = run_hpo(
        model_class=entry["class"],
        dataset=train_dataset,
        cv_strategy=cv,
        cfg=cfg,
        model_kwargs={**entry["kwargs"], "horizon": horizon_days},
        primary_metric="mae",
        search_space=search_space,
        n_trials=n_trials,
        study_name=f"volve_{model_name}_{horizon}",
    )

    logger.info("  Best params: %s (value=%.4f)", hpo_result["best_params"], hpo_result["best_value"])

    # Final evaluation with best params
    best_kwargs = {**entry["kwargs"], "horizon": horizon_days}
    best_cfg = load_merged_config("configs/base.yaml", "configs/data/volve.yaml", entry["config"])
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

    final_cv = GroupedExpandingWindowCV(groups=groups, n_splits=3)
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
        use_mlflow=True,
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

from offshore_dl.utils.serialization import make_serializable as _make_serializable


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for trained models")
    parser.add_argument("--dataset", required=True, choices=["3w", "ganymede", "spe_berg", "volve"],
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
        # Valid 3W models — names only for validation (lazy import of PyTorch deps)
        valid_models = {"lstm", "deeponet", "patchtst", "random_forest", "fkmad", "mambasl", "convtimenet"}
        default_models = ["lstm", "deeponet", "patchtst", "random_forest", "fkmad", "mambasl", "convtimenet"]
    elif args.dataset == "spe_berg":
        # Valid SPE BERG models
        valid_models = {"lstm", "deeponet", "patchtst", "tcn"}
        default_models = ["lstm", "deeponet", "patchtst", "tcn"]
    elif args.dataset == "volve":
        # Valid Volve models
        valid_models = {"lstm", "deeponet", "patchtst", "tcn"}
        default_models = ["lstm", "deeponet", "patchtst", "tcn"]
    else:
        # Valid Ganymede models
        valid_models = {"lstm", "deeponet", "patchtst", "tcn"}
        default_models = ["lstm", "deeponet", "patchtst", "tcn"]

    models = args.models or default_models

    logger.info("═" * 70)
    logger.info("OPTUNA HPO — %s", args.dataset.upper())
    logger.info("  models=%s  n_trials=%d  device=%s", models, args.n_trials, args.device)
    logger.info("═" * 70)

    summary = {}
    for model_name in models:
        if model_name not in valid_models:
            logger.warning("Unknown model: %s (skipping)", model_name)
            continue

        logger.info("─" * 60)
        logger.info("HPO: %s", model_name)
        logger.info("─" * 60)
        start = time.time()
        try:
            if args.dataset == "3w":
                if model_name == "random_forest":
                    result = run_3w_rf_hpo(args.n_trials, args.device)
                else:
                    result = run_3w_hpo(model_name, args.n_trials, args.device)
            elif args.dataset == "spe_berg":
                result = run_spe_berg_hpo(model_name, args.horizon, args.n_trials, args.device)
            elif args.dataset == "volve":
                result = run_volve_hpo(model_name, args.horizon, args.n_trials, args.device)
            else:
                result = run_ganymede_hpo(model_name, args.horizon, args.n_trials, args.device)

            elapsed = time.time() - start

            # Save per-model result
            out_dir = RESULTS_DIR / "hpo" / args.dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{args.horizon}" if args.dataset in ("ganymede", "spe_berg", "volve") else ""
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
