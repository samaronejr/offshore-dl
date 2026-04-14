"""Production training: LSTM, DeepONet, PatchTST on 3W with feature extraction.

Uses ``ThreeWFeatureDataset`` which compresses each ``(720, 27)`` raw
window into ``(14, 27)`` statistical features before feeding to models.
This follows published 3W literature where feature extraction +
classification achieves 87–95 % F1 (Fernandes Junior et al., 2024).

Usage::

    python scripts/run_production_3w_features.py --device cuda
    python scripts/run_production_3w_features.py --max-epochs 1 --device cpu  # smoke
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
from sklearn.ensemble import RandomForestClassifier

from offshore_dl.data.datasets import (
    ThreeWFeatureDataset,
    ThreeWPhysicsDataset,
    ThreeWWindowDataset,
    ThreeWWaveletDataset,
)
from offshore_dl.data.feature_extractor import N_FEATURES
from offshore_dl.evaluation.cv import HoldoutSplitter, StratifiedGroupKFoldSKLearn
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.patchtst import PatchTSTModel

# MLflow functionality entirely commented/remains filed below
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed

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

try:
    from offshore_dl.models.hydra_rocket import HydraRocketModel
except ImportError:
    HydraRocketModel = None

try:
    from offshore_dl.models.convtran import ConvTranModel
except ImportError:
    ConvTranModel = None

try:
    from offshore_dl.models.inception_time import InceptionTimeModel
except ImportError:
    InceptionTimeModel = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Model registry — same models, tuned for the shorter (14, 27) input
# ═══════════════════════════════════════════════════════════════════

MODELS: dict[str, dict] = {
    "lstm": {
        "class": LSTMModel,
        "config": "configs/models/lstm.yaml",
        "overrides": {
            # Smaller net — only 14 timesteps, not 720
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            "lr": 1e-3,
        },
    },
    "deeponet": {
        "class": DeepONetModel,
        "config": "configs/models/deeponet.yaml",
        "overrides": {
            "rank": 64,
            "dropout": 0.077,
            "lr": 0.00135,
            "branch_hidden": [64, 64],
        },
    },
    "patchtst": {
        "class": PatchTSTModel,
        "config": "configs/models/patchtst.yaml",
        "overrides": {
            # Patch size ≤ seq_len (14); stride ≤ patch_size
            "patch_len": 7,
            "stride": 4,
            "d_model": 128,
            "d_ff": 256,
            "n_heads": 4,
            "n_layers": 2,
            "lr": 5e-4,
        },
    },
    "wavelet_rf": {
        "class": None,
        "config": "configs/models/random_forest.yaml",
        "overrides": {},
        "dataset_config": "configs/data/3w_wavelet.yaml",
        "window_size": 18,
    },
    "wavelet_deeponet": {
        "class": DeepONetModel,
        "config": "configs/models/deeponet.yaml",
        "overrides": {
            "rank": 64,
            "dropout": 0.077,
            "lr": 0.00135,
            "branch_hidden": [64, 64],
        },
        "dataset_config": "configs/data/3w_wavelet.yaml",
        "window_size": 18,
    },
    "physics_rf": {
        "class": None,
        "config": "configs/models/random_forest.yaml",
        "overrides": {},
        "dataset_config": "configs/data/3w_physics.yaml",
        "window_size": 14,
    },
    "physics_deeponet": {
        "class": DeepONetModel,
        "config": "configs/models/deeponet.yaml",
        "overrides": {
            "rank": 64,
            "dropout": 0.077,
            "lr": 0.00135,
            "branch_hidden": [64, 64],
        },
        "dataset_config": "configs/data/3w_physics.yaml",
        "window_size": 14,
    },
    "window360_rf": {
        "class": None,
        "config": "configs/models/random_forest.yaml",
        "overrides": {},
        "dataset_config": "configs/data/3w_window_360.yaml",
        "window_size": 360,
    },
    "window1440_rf": {
        "class": None,
        "config": "configs/models/random_forest.yaml",
        "overrides": {},
        "dataset_config": "configs/data/3w_window_1440.yaml",
        "window_size": 1440,
    },
}

if FKMADModel is not None:
    MODELS["fkmad"] = {
        "class": FKMADModel,
        "config": "configs/models/fkmad.yaml",
        "overrides": {
            "d_model": 256,
            "n_mamba_layers": 2,
            "dropout": 0.39228800977342004,
            "d_state": 32,
            "n_fourier_freqs": 15,
            "fourier_rank": 32,
            "gamma_z_init": 2.6634386107304193,
        },
    }

if MambaSLModel is not None:
    MODELS["mambasl"] = {
        "class": MambaSLModel,
        "config": "configs/models/mambasl.yaml",
        "overrides": {
            "d_model": 64,
            "d_state": 16,
            "d_ff": 256,
            "d_conv": 8,
            "expand": 4,
            "n_heads": 1,
            "dropout": 0.05441608345844953,
        },
    }

if ConvTimeNetModel is not None:
    MODELS["convtimenet"] = {
        "class": ConvTimeNetModel,
        "config": "configs/models/convtimenet.yaml",
        "overrides": {
            "d_model": 128,
            "d_ff": 512,
            "patch_size": 8,
            "patch_stride": 4,
            "dw_ks": [7, 13, 19],
            "dropout": 0.1234,
            "pooling_tp": "max",
        },
    }

if ConvTranModel is not None:
    MODELS["convtran"] = {
        "class": ConvTranModel,
        "config": "configs/models/convtran.yaml",
        "overrides": {},
    }

if InceptionTimeModel is not None:
    MODELS["inception_time"] = {
        "class": InceptionTimeModel,
        "config": "configs/models/inception_time.yaml",
        "overrides": {},
    }

TREE_MODELS = ["random_forest"]
HYDRA_MODELS = ["hydra_rocket"]
RAW_MODELS = [
    "fkmad_raw",
    "mambasl_raw",
    "convtimenet_raw",
    "convtran_raw",
    "inception_time_raw",
]
# ALL_MODELS includes all known model names (even optional ones) so --models
# validation accepts them regardless of whether CUDA imports succeeded.
_OPTIONAL_FEATURE_MODELS = [
    m for m in ["fkmad", "mambasl", "convtimenet"] if m not in MODELS
]
MULTISCALE_MODELS = ["multiscale_rf", "multiscale_deeponet"]
WAVELET_MODELS = ["wavelet_rf", "wavelet_deeponet"]
PHYSICS_MODELS = ["physics_rf", "physics_deeponet"]
WINDOW_MODELS = ["window360_rf", "window1440_rf"]
ALL_MODELS = list(
    dict.fromkeys(
        list(MODELS.keys())
        + _OPTIONAL_FEATURE_MODELS
        + TREE_MODELS
        + HYDRA_MODELS
        + RAW_MODELS
        + MULTISCALE_MODELS
        + WAVELET_MODELS
        + PHYSICS_MODELS
        + WINDOW_MODELS
    )
)

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


def _compute_class_weights(y_train: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train.astype(np.int64), minlength=n_classes)
    counts = np.maximum(counts, 1)
    weights = len(y_train) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _stack_raw_windows(dataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(indices), dataset.window_size, dataset.n_vars), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.int64)
    for out_idx, ds_idx in enumerate(indices):
        x_i, y_i, _ = dataset[int(ds_idx)]
        X[out_idx] = x_i.numpy()
        y[out_idx] = int(y_i)
    return X, y


def _run_rf_model(
    dataset: ThreeWFeatureDataset,
    labels: np.ndarray,
    groups: np.ndarray,
    train_pool: np.ndarray,
    test_indices: np.ndarray,
    use_mlflow: bool = True,
) -> dict:
    """Run Random Forest with nested CV: inner stratified-group CV → retrain → test.

    Flattens (14, 27) → 378-dim feature vector for sklearn RandomForestClassifier.
    """
    set_global_seed(42)

    model_cfg = OmegaConf.load("configs/models/random_forest.yaml")
    arch = OmegaConf.to_container(model_cfg.model.architecture, resolve=True)

    n = len(dataset)
    sample_shape = dataset[0][0].numpy().reshape(-1).shape[0]
    X_all = np.empty((n, sample_shape), dtype=np.float32)
    Y_all = np.empty(n, dtype=np.int64)
    for i in range(n):
        x, y, _ = dataset[i]
        X_all[i] = x.numpy().reshape(-1)
        Y_all[i] = int(y)

    X_train_pool, Y_train_pool = X_all[train_pool], Y_all[train_pool]
    X_test, Y_test = X_all[test_indices], Y_all[test_indices]

    # Inner 5-fold stratified-group CV
    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]
    inner_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5,
        labels=pool_labels,
        groups=pool_groups,
        seed=42,
    )
    inner_splits = inner_cv.get_splits(len(train_pool))
    cv_fold_results = []

    # Setup MLflow
    mlflow = None
    if use_mlflow:
        try:
            import mlflow as _mlflow

            _mlflow.set_tracking_uri("mlruns")
            _mlflow.set_experiment("3w-random-forest")
            mlflow = _mlflow
        except ImportError:
            pass

    if False:  # Temporarily disabling MLFlow start run loop specifically improper continuous occurred locally repeat next
        mlflow.log_params({k: str(v) for k, v in arch.items()})

    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
        logger.info(
            "  ── random_forest inner fold %d/%d", fold_idx + 1, len(inner_splits)
        )
        X_tr, Y_tr = X_train_pool[local_train], Y_train_pool[local_train]
        X_va, Y_va = X_train_pool[local_val], Y_train_pool[local_val]

        clf = RandomForestClassifier(**arch)
        clf.fit(X_tr, Y_tr)
        preds = clf.predict(X_va)
        probs = clf.predict_proba(X_va)

        metrics = MetricRegistry.compute(
            "classification", preds, Y_va, prediction_scores=probs
        )
        cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})
        logger.info(
            "    fold %d: accuracy=%.4f, f1_macro=%.4f",
            fold_idx,
            metrics["accuracy"],
            metrics["f1_macro"],
        )
        if mlflow:
            mlflow.log_metric(f"cv_fold_{fold_idx}_accuracy", metrics["accuracy"])
            mlflow.log_metric(f"cv_fold_{fold_idx}_f1_macro", metrics["f1_macro"])

    # Aggregate CV metrics
    cv_agg = {}
    if cv_fold_results:
        metric_keys = [
            k
            for k in cv_fold_results[0]["metrics"]
            if isinstance(cv_fold_results[0]["metrics"][k], (int, float))
        ]
        for k in metric_keys:
            vals = [f["metrics"][k] for f in cv_fold_results]
            cv_agg[f"{k}_mean"] = float(np.mean(vals))
            cv_agg[f"{k}_std"] = float(np.std(vals))

    # Retrain on full training pool, evaluate on held-out test
    logger.info(
        "  ── random_forest: retrain on full train pool (%d samples)", len(train_pool)
    )
    final_clf = RandomForestClassifier(**arch)
    final_clf.fit(X_train_pool, Y_train_pool)
    test_preds = final_clf.predict(X_test)
    test_probs = final_clf.predict_proba(X_test)
    test_metrics = MetricRegistry.compute(
        "classification", test_preds, Y_test, prediction_scores=test_probs
    )

    if mlflow:
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)
        mlflow.end_run()

    result = {
        "test_metrics": test_metrics,
        "test_predictions": test_preds.tolist(),
        "test_probabilities": test_probs.tolist(),
        "test_targets": Y_test.tolist(),
        "cv_aggregate": cv_agg,
        "cv_fold_results": cv_fold_results,
        "n_train": len(train_pool),
        "n_test": len(test_indices),
        "n_cv_folds": len(inner_splits),
    }

    metric_str = ", ".join(
        f"{k}={v:.4f}"
        for k, v in sorted(test_metrics.items())
        if isinstance(v, (int, float))
    )
    print("\n  RANDOM FOREST on 3W features")
    print(f"  TEST: {metric_str}")

    return result


def _run_model(
    model_name: str,
    dataset: ThreeWFeatureDataset,
    max_epochs: int,
    batch_size: int,
    device: str,
    labels: np.ndarray,
    groups: np.ndarray,
    train_pool: np.ndarray,
    test_indices: np.ndarray,
    use_mlflow: bool = True,
) -> dict:
    """Train one model with nested CV: inner CV on train pool → retrain → test."""
    set_global_seed(42)

    entry = MODELS[model_name]
    model_class = entry["class"]

    cfg = load_merged_config(
        "configs/base.yaml",
        entry.get("dataset_config", "configs/data/3w.yaml"),
        entry["config"],
    )

    cfg.training.max_epochs = max_epochs
    cfg.training.batch_size = batch_size
    cfg.device = device

    # Use cosine scheduler — onecycle warmup is too slow for 14-step features
    cfg.training.scheduler = "cosine"

    # ── Inner CV strategy (applied within train_pool only) ──
    # Labels/groups are remapped to train_pool indices inside run_nested.
    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]

    inner_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5,
        labels=pool_labels,
        groups=pool_groups,
        seed=42,
    )

    # window_size = N_FEATURES (14) — the feature sequence length
    model_kwargs = {
        "task": "classification",
        "n_vars": dataset.n_vars,
        "n_classes": cfg.data.n_classes,
        "window_size": dataset.window_size,
    }

    train_class_weights = _compute_class_weights(pool_labels, cfg.data.n_classes)
    model_kwargs["class_weights"] = train_class_weights

    # Merge architecture params from model config
    if hasattr(cfg, "model") and hasattr(cfg.model, "architecture"):
        arch = OmegaConf.to_container(cfg.model.architecture, resolve=True)
        model_kwargs.update(arch)

    for key in ("loss_type", "focal_gamma"):
        if hasattr(cfg, "model") and hasattr(cfg.model, key):
            model_kwargs[key] = getattr(cfg.model, key)

    # Merge training LR/weight_decay from model config first
    if hasattr(cfg, "model") and hasattr(cfg.model, "training"):
        model_kwargs["lr"] = cfg.model.training.lr
        model_kwargs["weight_decay"] = cfg.model.training.weight_decay

    # Apply per-model overrides for feature-based training
    overrides = entry.get("overrides", {})
    model_kwargs.update(overrides)

    runner = ExperimentRunner(
        model_class=model_class,
        dataset=dataset,
        cv_strategy=inner_cv,
        cfg=cfg,
        model_kwargs=model_kwargs,
    )

    return runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=use_mlflow,
    )


def _run_hydra_rocket_model(
    dataset,
    labels: np.ndarray,
    groups: np.ndarray,
    train_pool: np.ndarray,
    test_indices: np.ndarray,
) -> dict:
    """Run Hydra+MultiROCKET with nested CV on raw 3W windows."""
    if HydraRocketModel is None:
        msg = "hydra_rocket is unavailable. Install aeon with `pip install aeon`."
        raise ImportError(msg)

    set_global_seed(42)
    n_classes = int(labels.max()) + 1

    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]
    inner_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5,
        labels=pool_labels,
        groups=pool_groups,
        seed=42,
    )
    inner_splits = inner_cv.get_splits(len(train_pool))
    cv_fold_results = []

    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
        logger.info(
            "  ── hydra_rocket inner fold %d/%d", fold_idx + 1, len(inner_splits)
        )
        fold_train_idx = train_pool[local_train]
        fold_val_idx = train_pool[local_val]
        X_tr, y_tr = _stack_raw_windows(dataset, fold_train_idx)
        X_va, y_va = _stack_raw_windows(dataset, fold_val_idx)

        clf = HydraRocketModel(
            task="classification",
            n_vars=dataset.n_vars,
            n_classes=n_classes,
            n_kernels=8192,
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)
        preds = np.argmax(probs, axis=1)
        metrics = MetricRegistry.compute(
            "classification",
            preds,
            y_va,
            prediction_scores=probs,
        )
        cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})

    cv_agg = {}
    if cv_fold_results:
        metric_keys = [
            k
            for k, v in cv_fold_results[0]["metrics"].items()
            if isinstance(v, (int, float))
        ]
        for key in metric_keys:
            vals = [fold["metrics"][key] for fold in cv_fold_results]
            cv_agg[f"{key}_mean"] = float(np.mean(vals))
            cv_agg[f"{key}_std"] = float(np.std(vals))

    logger.info(
        "  ── hydra_rocket: retrain on full train pool (%d samples)", len(train_pool)
    )
    X_train_pool, y_train_pool = _stack_raw_windows(dataset, train_pool)
    X_test, y_test = _stack_raw_windows(dataset, test_indices)
    final_clf = HydraRocketModel(
        task="classification",
        n_vars=dataset.n_vars,
        n_classes=n_classes,
        n_kernels=8192,
    )
    final_clf.fit(X_train_pool, y_train_pool)
    test_probs = final_clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = MetricRegistry.compute(
        "classification",
        test_preds,
        y_test,
        prediction_scores=test_probs,
    )

    return {
        "test_metrics": test_metrics,
        "cv_aggregate": cv_agg,
        "cv_fold_results": cv_fold_results,
        "n_train": len(train_pool),
        "n_test": len(test_indices),
        "n_cv_folds": len(inner_splits),
    }


def main() -> None:
    # Do NOT add set_global_seed() here: each model function resets it independently
    # so that model weight initialisation is unaffected by prior dataset loading.
    parser = argparse.ArgumentParser(
        description="Production training: 3 models on 3W feature-extracted data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="Max training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow tracking"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run (default: all). Choices: " + ", ".join(ALL_MODELS),
    )

    args = parser.parse_args()

    # Validate --models filter
    valid_all = set(ALL_MODELS)
    if args.models:
        unknown = [m for m in args.models if m not in valid_all]
        if unknown:
            logger.error("Unknown model(s): %s. Available: %s", unknown, ALL_MODELS)
            sys.exit(1)
        models_to_run = args.models
    else:
        models_to_run = ALL_MODELS

    # ── Detect which dataset variants are required ──
    non_feature_models = set(RAW_MODELS + HYDRA_MODELS + MULTISCALE_MODELS)
    non_feature_models.update(WAVELET_MODELS)
    non_feature_models.update(WINDOW_MODELS)
    needs_feature_dataset = any(
        (m not in non_feature_models) and (m not in PHYSICS_MODELS)
        for m in models_to_run
    )
    needs_physics_dataset = any(m in PHYSICS_MODELS for m in models_to_run)

    logger.info("═" * 70)
    logger.info("3W FEATURE-BASED TRAINING — nested CV (inner 5-fold + held-out test)")
    logger.info(
        "  device=%s  max_epochs=%d  batch_size=%d",
        args.device,
        args.max_epochs,
        args.batch_size,
    )
    logger.info("  Features: (720, 27) → (%d, 27) statistical descriptors", N_FEATURES)
    logger.info("═" * 70)

    feature_dataset = None
    physics_dataset = None
    labels = None
    groups = None
    if needs_feature_dataset:
        logger.info("Loading 3W dataset with feature extraction …")
        ds_start = time.time()
        feature_dataset = ThreeWFeatureDataset("configs/data/3w.yaml")
        logger.info(
            "  3W loaded: %d samples (%.1fs)",
            len(feature_dataset),
            time.time() - ds_start,
        )

        # ── Compute labels and groups once for all feature models ──
        labels = np.array([feature_dataset[i][1] for i in range(len(feature_dataset))])
        groups = np.array(
            [feature_dataset[i][2]["instance_id"] for i in range(len(feature_dataset))]
        )
        dataset = feature_dataset
        _raw_ds_shared = None  # no pre-loaded raw dataset

    if needs_physics_dataset:
        logger.info("Loading 3W dataset with physics features …")
        ds_start = time.time()
        physics_dataset = ThreeWPhysicsDataset("configs/data/3w_physics.yaml")
        logger.info(
            "  3W physics loaded: %d samples (%.1fs)",
            len(physics_dataset),
            time.time() - ds_start,
        )
        if labels is None or groups is None:
            labels = np.array(
                [physics_dataset[i][1] for i in range(len(physics_dataset))]
            )
            groups = np.array(
                [
                    physics_dataset[i][2]["instance_id"]
                    for i in range(len(physics_dataset))
                ]
            )
        if feature_dataset is None:
            dataset = physics_dataset

    if labels is None or groups is None:
        # Raw-only mode: load ThreeWDataset with cache_in_memory=False
        # (caches feature arrays ~8 GB but NOT DataFrames ~17 GB).
        # This dataset is reused directly by the fkmad_raw block below.
        logger.info("Raw-only mode — loading 3W dataset for holdout + training …")
        ds_start = time.time()
        from offshore_dl.data.datasets import ThreeWDataset as _ThreeWDataset

        _raw_ds_shared = _ThreeWDataset("configs/data/3w.yaml", cache_in_memory=False)
        logger.info(
            "  3W loaded: %d samples (%.1fs)",
            len(_raw_ds_shared),
            time.time() - ds_start,
        )

        labels = np.array([w["label"] for w in _raw_ds_shared._windows])
        groups = np.array([w["instance_id"] for w in _raw_ds_shared._windows])
        dataset = None  # no feature dataset

    # ── Outer holdout split: 80% train pool, 20% held-out test ──
    holdout = HoldoutSplitter(
        test_ratio=0.2,
        mode="stratified_group",
        labels=labels,
        groups=groups,
        seed=42,
    )
    train_pool, test_indices = holdout.split(
        len(labels)
    )  # works for both feature and raw-only
    logger.info(
        "Holdout split: train_pool=%d, test=%d",
        len(train_pool),
        len(test_indices),
    )

    sweep_start = time.time()
    summary: dict[str, dict] = {}

    for model_name in models_to_run:
        if (
            model_name in RAW_MODELS
            or model_name in HYDRA_MODELS
            or model_name in MULTISCALE_MODELS
            or model_name in WINDOW_MODELS
        ):
            continue

        dataset_for_model = physics_dataset if model_name in PHYSICS_MODELS else dataset

        logger.info("─" * 60)
        logger.info(
            "TRAINING: %s on 3W %s",
            model_name.upper(),
            "physics features" if model_name in PHYSICS_MODELS else "features",
        )
        logger.info("─" * 60)

        start = time.time()
        try:
            is_tree = model_name in TREE_MODELS
            if is_tree:
                results = _run_rf_model(
                    dataset=dataset_for_model,
                    labels=labels,
                    groups=groups,
                    train_pool=train_pool,
                    test_indices=test_indices,
                    use_mlflow=not args.no_mlflow,
                )
            else:
                results = _run_model(
                    model_name=model_name,
                    dataset=dataset_for_model,
                    max_epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    device=args.device,
                    labels=labels,
                    groups=groups,
                    train_pool=train_pool,
                    test_indices=test_indices,
                    use_mlflow=not args.no_mlflow,
                )
            elapsed = time.time() - start

            out_path = RESULTS_DIR / model_name / "3w.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(_make_serializable(results), indent=2))
            logger.info("  Results saved: %s", out_path)

            agg = results.get("test_metrics", results.get("aggregate", {}))
            metric_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(agg.items())
                if isinstance(v, (int, float))
            )
            summary[model_name] = {
                "status": "ok",
                "elapsed": round(elapsed, 1),
                "test_metrics": results.get("test_metrics", {}),
                "cv_aggregate": results.get("cv_aggregate", {}),
                "n_train": results.get("n_train", 0),
                "n_test": results.get("n_test", 0),
                "n_cv_folds": results.get("n_cv_folds", 0),
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

    total_elapsed = time.time() - sweep_start

    print(f"\n{'═' * 70}")
    print("  3W FEATURE-BASED TRAINING COMPLETE (nested CV + held-out test)")
    print(f"{'═' * 70}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Train pool: {len(train_pool)}, Held-out test: {len(test_indices)}")
    for model_name, s in summary.items():
        if s["status"] == "ok":
            tm = s.get("test_metrics", {})
            metric_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(tm.items())
                if isinstance(v, (int, float))
            )
            print(f"    {model_name:12s} ✓ {s['elapsed']:8.1f}s  TEST: {metric_str}")
        else:
            print(
                f"    {model_name:12s} ✗ {s['elapsed']:8.1f}s  ERROR: {s.get('error', 'unknown')}"
            )
    print(f"{'═' * 70}\n")

    summary_path = RESULTS_DIR / "summary_production_3w_features.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    logger.info("Summary saved: %s", summary_path)

    run_multiscale = any(
        (args.models is None) or (m in args.models) for m in MULTISCALE_MODELS
    )
    if run_multiscale:
        logger.info("─" * 60)
        logger.info("MULTI-SCALE FEATURE CLASSIFICATION (360+720 → 28×27)")
        logger.info("─" * 60)
        try:
            if dataset is not None:
                del dataset
                dataset = None
            if _raw_ds_shared is not None:
                del _raw_ds_shared
                _raw_ds_shared = None
            import gc

            gc.collect()
            from offshore_dl.data.datasets import ThreeWMultiScaleDataset

            ms_start = time.time()
            ms_dataset = ThreeWMultiScaleDataset("configs/data/3w_multiscale.yaml")
            logger.info(
                "  MultiScale loaded: %d samples (%.1fs)",
                len(ms_dataset),
                time.time() - ms_start,
            )
            ms_labels = np.array([ms_dataset[i][1] for i in range(len(ms_dataset))])
            ms_groups = np.array(
                [ms_dataset[i][2]["instance_id"] for i in range(len(ms_dataset))]
            )
            pool_labels_ms = ms_labels[train_pool]
            pool_groups_ms = ms_groups[train_pool]
            inner_cv_ms = StratifiedGroupKFoldSKLearn(
                n_folds=5,
                labels=pool_labels_ms,
                groups=pool_groups_ms,
                seed=42,
            )
            ms_models_to_run = [
                m
                for m in MULTISCALE_MODELS
                if (args.models is None) or (m in args.models)
            ]
            for ms_name in ms_models_to_run:
                if ms_name == "multiscale_rf":
                    ms_model_cls = None
                    ms_cfg_path = "configs/models/random_forest.yaml"
                elif ms_name == "multiscale_deeponet":
                    ms_model_cls = DeepONetModel
                    ms_cfg_path = "configs/models/deeponet.yaml"
                else:
                    continue
                ms_item_start = time.time()
                if ms_name == "multiscale_rf":
                    ms_results = _run_rf_model(
                        dataset=ms_dataset,
                        labels=ms_labels,
                        groups=ms_groups,
                        train_pool=train_pool,
                        test_indices=test_indices,
                        use_mlflow=False,
                    )
                else:
                    ms_cfg = load_merged_config(
                        "configs/base.yaml",
                        "configs/data/3w_multiscale.yaml",
                        ms_cfg_path,
                    )
                    ms_cfg.training.max_epochs = args.max_epochs
                    ms_cfg.training.batch_size = args.batch_size
                    ms_cfg.device = args.device
                    ms_cfg.training.scheduler = "cosine"
                    ms_mkw = {
                        "task": "classification",
                        "n_vars": 27,
                        "n_classes": ms_cfg.data.n_classes,
                        "window_size": ms_dataset.n_features,
                    }
                    if hasattr(ms_cfg, "model") and hasattr(
                        ms_cfg.model, "architecture"
                    ):
                        ms_mkw.update(
                            OmegaConf.to_container(
                                ms_cfg.model.architecture, resolve=True
                            )
                        )
                    if hasattr(ms_cfg, "model") and hasattr(ms_cfg.model, "training"):
                        ms_mkw["lr"] = ms_cfg.model.training.lr
                        if hasattr(ms_cfg.model.training, "weight_decay"):
                            ms_mkw["weight_decay"] = ms_cfg.model.training.weight_decay
                    ms_runner = ExperimentRunner(
                        model_class=ms_model_cls,
                        dataset=ms_dataset,
                        cv_strategy=inner_cv_ms,
                        cfg=ms_cfg,
                        model_kwargs=ms_mkw,
                    )
                    ms_results = ms_runner.run_nested(
                        train_pool=train_pool,
                        test_indices=test_indices,
                        use_mlflow=False,
                    )
                ms_elapsed = time.time() - ms_item_start
                ms_out = RESULTS_DIR / ms_name / "3w.json"
                ms_out.parent.mkdir(parents=True, exist_ok=True)
                ms_out.write_text(json.dumps(_make_serializable(ms_results), indent=2))
                ms_agg = ms_results.get("test_metrics", {})
                ms_str = ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in sorted(ms_agg.items())
                    if isinstance(v, (int, float))
                )
                summary[ms_name] = {
                    "status": "ok",
                    "elapsed": round(ms_elapsed, 1),
                    "test_metrics": ms_agg,
                }
                logger.info("✓ %s: %s (%.1fs)", ms_name, ms_str, ms_elapsed)
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
        except Exception as e:
            logger.error("✗ multi-scale failed: %s", e)
            traceback.print_exc()
            for ms_name in MULTISCALE_MODELS:
                summary[ms_name] = {"status": "error", "error": str(e)}
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

    run_wavelet = any(
        (args.models is None) or (m in args.models) for m in WAVELET_MODELS
    )
    if run_wavelet:
        logger.info("─" * 60)
        logger.info("WAVELET FEATURE CLASSIFICATION (14 stats + 4 wavelet → 18×27)")
        logger.info("─" * 60)
        try:
            if dataset is not None:
                del dataset
                dataset = None
            if _raw_ds_shared is not None:
                del _raw_ds_shared
                _raw_ds_shared = None
            import gc

            gc.collect()

            wavelet_start = time.time()
            wavelet_dataset = ThreeWWaveletDataset("configs/data/3w_wavelet.yaml")
            logger.info(
                "  Wavelet loaded: %d samples (%.1fs)",
                len(wavelet_dataset),
                time.time() - wavelet_start,
            )
            wavelet_labels = np.array(
                [wavelet_dataset[i][1] for i in range(len(wavelet_dataset))]
            )
            wavelet_groups = np.array(
                [
                    wavelet_dataset[i][2]["instance_id"]
                    for i in range(len(wavelet_dataset))
                ]
            )
            wavelet_models_to_run = [
                m for m in WAVELET_MODELS if (args.models is None) or (m in args.models)
            ]
            for wavelet_name in wavelet_models_to_run:
                wavelet_item_start = time.time()
                if wavelet_name == "wavelet_rf":
                    wavelet_results = _run_rf_model(
                        dataset=wavelet_dataset,
                        labels=wavelet_labels,
                        groups=wavelet_groups,
                        train_pool=train_pool,
                        test_indices=test_indices,
                        use_mlflow=False,
                    )
                elif wavelet_name == "wavelet_deeponet":
                    set_global_seed(42)
                    wavelet_entry = MODELS[wavelet_name]
                    wavelet_cfg = load_merged_config(
                        "configs/base.yaml",
                        wavelet_entry["dataset_config"],
                        wavelet_entry["config"],
                    )
                    wavelet_cfg.training.max_epochs = args.max_epochs
                    wavelet_cfg.training.batch_size = args.batch_size
                    wavelet_cfg.device = args.device
                    wavelet_cfg.training.scheduler = "cosine"
                    wavelet_kwargs = {
                        "task": "classification",
                        "n_vars": 27,
                        "n_classes": wavelet_cfg.data.n_classes,
                        "window_size": wavelet_entry["window_size"],
                    }
                    if hasattr(wavelet_cfg, "model") and hasattr(
                        wavelet_cfg.model, "architecture"
                    ):
                        wavelet_kwargs.update(
                            OmegaConf.to_container(
                                wavelet_cfg.model.architecture,
                                resolve=True,
                            )
                        )
                    if hasattr(wavelet_cfg, "model") and hasattr(
                        wavelet_cfg.model, "training"
                    ):
                        wavelet_kwargs["lr"] = wavelet_cfg.model.training.lr
                        if hasattr(wavelet_cfg.model.training, "weight_decay"):
                            wavelet_kwargs["weight_decay"] = (
                                wavelet_cfg.model.training.weight_decay
                            )
                    wavelet_kwargs["class_weights"] = _compute_class_weights(
                        wavelet_labels[train_pool],
                        wavelet_cfg.data.n_classes,
                    )
                    wavelet_kwargs.update(wavelet_entry.get("overrides", {}))
                    wavelet_inner_cv = StratifiedGroupKFoldSKLearn(
                        n_folds=5,
                        labels=wavelet_labels[train_pool],
                        groups=wavelet_groups[train_pool],
                        seed=42,
                    )
                    wavelet_runner = ExperimentRunner(
                        model_class=wavelet_entry["class"],
                        dataset=wavelet_dataset,
                        cv_strategy=wavelet_inner_cv,
                        cfg=wavelet_cfg,
                        model_kwargs=wavelet_kwargs,
                    )
                    wavelet_results = wavelet_runner.run_nested(
                        train_pool=train_pool,
                        test_indices=test_indices,
                        use_mlflow=False,
                    )
                else:
                    continue
                wavelet_elapsed = time.time() - wavelet_item_start
                wavelet_out = RESULTS_DIR / wavelet_name / "3w.json"
                wavelet_out.parent.mkdir(parents=True, exist_ok=True)
                wavelet_out.write_text(
                    json.dumps(_make_serializable(wavelet_results), indent=2)
                )
                wavelet_agg = wavelet_results.get("test_metrics", {})
                summary[wavelet_name] = {
                    "status": "ok",
                    "elapsed": round(wavelet_elapsed, 1),
                    "test_metrics": wavelet_agg,
                }
                logger.info("✓ %s completed (%.1fs)", wavelet_name, wavelet_elapsed)
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
        except Exception as e:
            logger.error("✗ wavelet failed: %s", e)
            traceback.print_exc()
            for wavelet_name in WAVELET_MODELS:
                summary[wavelet_name] = {"status": "error", "error": str(e)}
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

    run_window = any((args.models is None) or (m in args.models) for m in WINDOW_MODELS)
    if run_window:
        logger.info("─" * 60)
        logger.info("WINDOW-SIZE FEATURE CLASSIFICATION (RF on 3W feature matrices)")
        logger.info("─" * 60)
        try:
            for window_name in WINDOW_MODELS:
                if (args.models is not None) and (window_name not in args.models):
                    continue
                window_entry = MODELS[window_name]
                window_start = time.time()
                window_dataset = ThreeWWindowDataset(
                    window_entry["dataset_config"],
                    window_size=window_entry["window_size"],
                )
                window_labels = np.array(
                    [window_dataset[i][1] for i in range(len(window_dataset))]
                )
                window_groups = np.array(
                    [
                        window_dataset[i][2]["instance_id"]
                        for i in range(len(window_dataset))
                    ]
                )
                window_holdout = HoldoutSplitter(
                    test_ratio=0.2,
                    mode="stratified_group",
                    labels=window_labels,
                    groups=window_groups,
                    seed=42,
                )
                window_train_pool, window_test_indices = window_holdout.split(
                    len(window_labels)
                )
                window_results = _run_rf_model(
                    dataset=window_dataset,
                    labels=window_labels,
                    groups=window_groups,
                    train_pool=window_train_pool,
                    test_indices=window_test_indices,
                    use_mlflow=False,
                )
                window_elapsed = time.time() - window_start
                window_out = RESULTS_DIR / window_name / "3w.json"
                window_out.parent.mkdir(parents=True, exist_ok=True)
                window_out.write_text(
                    json.dumps(_make_serializable(window_results), indent=2)
                )
                window_agg = window_results.get("test_metrics", {})
                summary[window_name] = {
                    "status": "ok",
                    "elapsed": round(window_elapsed, 1),
                    "test_metrics": window_agg,
                }
                logger.info("✓ %s completed (%.1fs)", window_name, window_elapsed)
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
        except Exception as e:
            logger.error("✗ window-size sweep failed: %s", e)
            traceback.print_exc()
            for window_name in WINDOW_MODELS:
                summary[window_name] = {"status": "error", "error": str(e)}
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

    run_hydra_rocket = (args.models is None) or ("hydra_rocket" in args.models)
    if run_hydra_rocket:
        logger.info("─" * 60)
        logger.info("HYDRA_ROCKET CLASSIFICATION (720×27 raw windows)")
        logger.info("─" * 60)
        hydra_start = time.time()
        try:
            from offshore_dl.data.datasets import ThreeWDataset

            if _raw_ds_shared is not None:
                hydra_dataset = _raw_ds_shared
                logger.info(
                    "  Reusing raw dataset from holdout phase: %d samples",
                    len(hydra_dataset),
                )
            else:
                hydra_dataset = ThreeWDataset(
                    "configs/data/3w.yaml", cache_in_memory=False
                )
                logger.info(
                    "  Raw 3W loaded for hydra_rocket: %d samples", len(hydra_dataset)
                )

            hydra_results = _run_hydra_rocket_model(
                dataset=hydra_dataset,
                labels=labels,
                groups=groups,
                train_pool=train_pool,
                test_indices=test_indices,
            )
            hydra_elapsed = time.time() - hydra_start
            hydra_path = RESULTS_DIR / "hydra_rocket" / "3w.json"
            hydra_path.parent.mkdir(parents=True, exist_ok=True)
            hydra_path.write_text(
                json.dumps(_make_serializable(hydra_results), indent=2)
            )
            summary["hydra_rocket"] = {
                "status": "ok",
                "elapsed": round(hydra_elapsed, 1),
                "test_metrics": hydra_results.get("test_metrics", {}),
                "cv_aggregate": hydra_results.get("cv_aggregate", {}),
                "n_train": hydra_results.get("n_train", 0),
                "n_test": hydra_results.get("n_test", 0),
                "n_cv_folds": hydra_results.get("n_cv_folds", 0),
            }
            logger.info("✓ hydra_rocket completed (%.1fs)", hydra_elapsed)
        except Exception as e:
            hydra_elapsed = time.time() - hydra_start
            summary["hydra_rocket"] = {
                "status": "error",
                "elapsed": round(hydra_elapsed, 1),
                "error": str(e),
            }
            logger.error("✗ hydra_rocket failed: %s (%.1fs)", e, hydra_elapsed)
            traceback.print_exc()

    # ── FKM-AD Raw Classification (720×27 windows, Mamba backbone) ──
    # Uses raw 3W windows instead of feature-extracted data.
    # Reuses the same holdout split (train_pool/test_indices) from the
    # feature dataset — sample ordering is identical.
    run_fkmad_raw = (args.models is None) or ("fkmad_raw" in args.models)
    if run_fkmad_raw and FKMADModel is not None:
        logger.info("─" * 60)
        logger.info("FKMAD_RAW CLASSIFICATION (720×27 raw windows)")
        logger.info("─" * 60)
        try:
            # ── Free feature dataset to reclaim memory before loading raw ──
            if dataset is not None:
                del dataset
            import gc

            gc.collect()

            from offshore_dl.data.datasets import ThreeWDataset

            # Reuse raw dataset from raw-only mode if available,
            # otherwise load fresh with cache_in_memory=False to avoid
            # the ~17 GB DataFrame cache that caused OOM on 30 GB RAM.
            if _raw_ds_shared is not None:
                raw_dataset = _raw_ds_shared
                logger.info(
                    "  Reusing raw dataset from holdout phase: %d samples",
                    len(raw_dataset),
                )
            else:
                raw_dataset = ThreeWDataset(
                    "configs/data/3w.yaml", cache_in_memory=False
                )
                logger.info(
                    "  Raw 3W loaded: %d samples (cache_in_memory=False)",
                    len(raw_dataset),
                )

            # Extract labels/groups from _windows metadata (zero-copy, no __getitem__)
            raw_labels = np.array([w["label"] for w in raw_dataset._windows])
            raw_groups = np.array([w["instance_id"] for w in raw_dataset._windows])

            # Inner CV within train_pool (reuse same holdout split)
            pool_labels_raw = raw_labels[train_pool]
            pool_groups_raw = raw_groups[train_pool]
            inner_cv_raw = StratifiedGroupKFoldSKLearn(
                n_folds=5,
                labels=pool_labels_raw,
                groups=pool_groups_raw,
                seed=42,
            )

            cfg_raw = load_merged_config(
                "configs/base.yaml",
                "configs/data/3w.yaml",
                "configs/models/fkmad.yaml",
            )
            cfg_raw.training.max_epochs = args.max_epochs
            cfg_raw.training.batch_size = (
                32  # reduced for 720-length sequences — VRAM safety
            )
            cfg_raw.device = args.device
            cfg_raw.training.scheduler = "cosine"

            model_kwargs_raw = {
                "task": "classification",
                "n_vars": 27,
                "n_classes": cfg_raw.data.n_classes,
                "window_size": 720,
            }

            # Merge architecture params from model config
            if hasattr(cfg_raw, "model") and hasattr(cfg_raw.model, "architecture"):
                arch_raw = OmegaConf.to_container(
                    cfg_raw.model.architecture, resolve=True
                )
                model_kwargs_raw.update(arch_raw)

            # Merge training LR/weight_decay
            if hasattr(cfg_raw, "model") and hasattr(cfg_raw.model, "training"):
                model_kwargs_raw["lr"] = cfg_raw.model.training.lr
                model_kwargs_raw["weight_decay"] = cfg_raw.model.training.weight_decay

            fkmad_raw_start = time.time()
            runner_raw = ExperimentRunner(
                model_class=FKMADModel,
                dataset=raw_dataset,
                cv_strategy=inner_cv_raw,
                cfg=cfg_raw,
                model_kwargs=model_kwargs_raw,
            )

            fkmad_raw_results = runner_raw.run_nested(
                train_pool=train_pool,
                test_indices=test_indices,
                use_mlflow=not args.no_mlflow,
            )
            fkmad_raw_elapsed = time.time() - fkmad_raw_start

            out_path_raw = RESULTS_DIR / "fkmad_raw" / "3w.json"
            out_path_raw.parent.mkdir(parents=True, exist_ok=True)
            out_path_raw.write_text(
                json.dumps(_make_serializable(fkmad_raw_results), indent=2)
            )
            logger.info("  Results saved: %s", out_path_raw)

            agg_raw = fkmad_raw_results.get("test_metrics", {})
            metric_str_raw = ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(agg_raw.items())
                if isinstance(v, (int, float))
            )
            summary["fkmad_raw"] = {
                "status": "ok",
                "elapsed": round(fkmad_raw_elapsed, 1),
                "test_metrics": fkmad_raw_results.get("test_metrics", {}),
                "cv_aggregate": fkmad_raw_results.get("cv_aggregate", {}),
                "n_train": fkmad_raw_results.get("n_train", 0),
                "n_test": fkmad_raw_results.get("n_test", 0),
                "n_cv_folds": fkmad_raw_results.get("n_cv_folds", 0),
            }
            logger.info("✓ fkmad_raw: %s (%.1fs)", metric_str_raw, fkmad_raw_elapsed)

            # Re-save summary with fkmad_raw
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

        except Exception as e:
            logger.error("✗ fkmad_raw failed: %s", e)
            traceback.print_exc()
            summary["fkmad_raw"] = {"status": "error", "error": str(e)}
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    elif run_fkmad_raw and FKMADModel is None:
        logger.warning("FKMADModel not available (CUDA required) — skipping fkmad_raw")
        summary["fkmad_raw"] = {
            "status": "skipped",
            "error": "FKMADModel import failed (CUDA required)",
        }
        summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

    # ── MambaSL Raw Classification (720×27 windows, single-layer Mamba) ──
    # Uses raw 3W windows instead of feature-extracted data.
    # Reuses the same holdout split (train_pool/test_indices) from the
    # feature dataset — sample ordering is identical.
    run_mambasl_raw = (args.models is None) or ("mambasl_raw" in args.models)
    if run_mambasl_raw and MambaSLModel is not None:
        logger.info("─" * 60)
        logger.info("MAMBASL_RAW CLASSIFICATION (720×27 raw windows)")
        logger.info("─" * 60)
        try:
            # ── Free feature dataset to reclaim memory before loading raw ──
            if dataset is not None:
                del dataset
                dataset = None
            import gc

            gc.collect()

            from offshore_dl.data.datasets import ThreeWDataset

            # Reuse raw dataset from holdout phase or fkmad_raw if available,
            # otherwise load fresh with cache_in_memory=False to avoid
            # the ~17 GB DataFrame cache that caused OOM on 30 GB RAM.
            if _raw_ds_shared is not None:
                raw_dataset_sl = _raw_ds_shared
                logger.info(
                    "  Reusing raw dataset from holdout phase: %d samples",
                    len(raw_dataset_sl),
                )
            else:
                # Try to reuse the ThreeWDataset that fkmad_raw may have created
                try:
                    raw_dataset_sl = raw_dataset  # type: ignore[name-defined]
                    logger.info(
                        "  Reusing raw dataset from fkmad_raw: %d samples",
                        len(raw_dataset_sl),
                    )
                except NameError:
                    raw_dataset_sl = ThreeWDataset(
                        "configs/data/3w.yaml", cache_in_memory=False
                    )
                    logger.info(
                        "  Raw 3W loaded: %d samples (cache_in_memory=False)",
                        len(raw_dataset_sl),
                    )

            # Extract labels/groups from _windows metadata (zero-copy)
            sl_raw_labels = np.array([w["label"] for w in raw_dataset_sl._windows])
            sl_raw_groups = np.array(
                [w["instance_id"] for w in raw_dataset_sl._windows]
            )

            # Inner CV within train_pool (reuse same holdout split)
            pool_labels_sl = sl_raw_labels[train_pool]
            pool_groups_sl = sl_raw_groups[train_pool]
            inner_cv_sl = StratifiedGroupKFoldSKLearn(
                n_folds=5,
                labels=pool_labels_sl,
                groups=pool_groups_sl,
                seed=42,
            )

            cfg_sl = load_merged_config(
                "configs/base.yaml",
                "configs/data/3w.yaml",
                "configs/models/mambasl.yaml",
            )
            cfg_sl.training.max_epochs = args.max_epochs
            cfg_sl.training.batch_size = (
                32  # reduced for 720-length sequences — VRAM safety
            )
            cfg_sl.device = args.device
            cfg_sl.training.scheduler = "cosine"

            model_kwargs_sl = {
                "task": "classification",
                "n_vars": 27,
                "n_classes": cfg_sl.data.n_classes,
                "window_size": 720,
            }

            # Merge architecture params from model config
            if hasattr(cfg_sl, "model") and hasattr(cfg_sl.model, "architecture"):
                arch_sl = OmegaConf.to_container(
                    cfg_sl.model.architecture, resolve=True
                )
                model_kwargs_sl.update(arch_sl)

            # Merge training LR/weight_decay
            if hasattr(cfg_sl, "model") and hasattr(cfg_sl.model, "training"):
                model_kwargs_sl["lr"] = cfg_sl.model.training.lr
                model_kwargs_sl["weight_decay"] = cfg_sl.model.training.weight_decay

            mambasl_raw_start = time.time()
            runner_sl_raw = ExperimentRunner(
                model_class=MambaSLModel,
                dataset=raw_dataset_sl,
                cv_strategy=inner_cv_sl,
                cfg=cfg_sl,
                model_kwargs=model_kwargs_sl,
            )

            mambasl_raw_results = runner_sl_raw.run_nested(
                train_pool=train_pool,
                test_indices=test_indices,
                use_mlflow=not args.no_mlflow,
            )
            mambasl_raw_elapsed = time.time() - mambasl_raw_start

            out_path_sl_raw = RESULTS_DIR / "mambasl_raw" / "3w.json"
            out_path_sl_raw.parent.mkdir(parents=True, exist_ok=True)
            out_path_sl_raw.write_text(
                json.dumps(_make_serializable(mambasl_raw_results), indent=2)
            )
            logger.info("  Results saved: %s", out_path_sl_raw)

            agg_sl_raw = mambasl_raw_results.get("test_metrics", {})
            metric_str_sl_raw = ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(agg_sl_raw.items())
                if isinstance(v, (int, float))
            )
            summary["mambasl_raw"] = {
                "status": "ok",
                "elapsed": round(mambasl_raw_elapsed, 1),
                "test_metrics": mambasl_raw_results.get("test_metrics", {}),
                "cv_aggregate": mambasl_raw_results.get("cv_aggregate", {}),
                "n_train": mambasl_raw_results.get("n_train", 0),
                "n_test": mambasl_raw_results.get("n_test", 0),
                "n_cv_folds": mambasl_raw_results.get("n_cv_folds", 0),
            }
            logger.info(
                "✓ mambasl_raw: %s (%.1fs)", metric_str_sl_raw, mambasl_raw_elapsed
            )

            # Re-save summary with mambasl_raw
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

        except Exception as e:
            logger.error("✗ mambasl_raw failed: %s", e)
            traceback.print_exc()
            summary["mambasl_raw"] = {"status": "error", "error": str(e)}
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    elif run_mambasl_raw and MambaSLModel is None:
        logger.warning(
            "MambaSLModel not available (CUDA required) — skipping mambasl_raw"
        )
        summary["mambasl_raw"] = {
            "status": "skipped",
            "error": "MambaSLModel import failed (CUDA required)",
        }
        summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

    # ── ConvTimeNet Raw Classification (720×27 windows, CPU-native) ──
    # Uses raw 3W windows instead of feature-extracted data.
    # ConvTimeNet is CPU-native (no CUDA-only deps) — runs on any device.
    # Reuses the same holdout split (train_pool/test_indices) from the
    # feature dataset — sample ordering is identical.
    run_convtimenet_raw = (args.models is None) or ("convtimenet_raw" in args.models)
    if run_convtimenet_raw and ConvTimeNetModel is not None:
        logger.info("─" * 60)
        logger.info("CONVTIMENET_RAW CLASSIFICATION (720×27 raw windows)")
        logger.info("─" * 60)
        try:
            # ── Free feature dataset to reclaim memory before loading raw ──
            if dataset is not None:
                del dataset
                dataset = None
            import gc

            gc.collect()

            from offshore_dl.data.datasets import ThreeWDataset

            # Reuse raw dataset from holdout phase or prior raw blocks if available,
            # otherwise load fresh with cache_in_memory=False to avoid OOM.
            if _raw_ds_shared is not None:
                raw_dataset_ct = _raw_ds_shared
                logger.info(
                    "  Reusing raw dataset from holdout phase: %d samples",
                    len(raw_dataset_ct),
                )
            else:
                try:
                    raw_dataset_ct = raw_dataset_sl  # type: ignore[name-defined]
                    logger.info(
                        "  Reusing raw dataset from mambasl_raw: %d samples",
                        len(raw_dataset_ct),
                    )
                except NameError:
                    try:
                        raw_dataset_ct = raw_dataset  # type: ignore[name-defined]
                        logger.info(
                            "  Reusing raw dataset from fkmad_raw: %d samples",
                            len(raw_dataset_ct),
                        )
                    except NameError:
                        raw_dataset_ct = ThreeWDataset(
                            "configs/data/3w.yaml", cache_in_memory=False
                        )
                        logger.info(
                            "  Raw 3W loaded: %d samples (cache_in_memory=False)",
                            len(raw_dataset_ct),
                        )

            # Extract labels/groups from _windows metadata (zero-copy)
            ct_raw_labels = np.array([w["label"] for w in raw_dataset_ct._windows])
            ct_raw_groups = np.array(
                [w["instance_id"] for w in raw_dataset_ct._windows]
            )

            # Inner CV within train_pool (reuse same holdout split)
            pool_labels_ct = ct_raw_labels[train_pool]
            pool_groups_ct = ct_raw_groups[train_pool]
            inner_cv_ct = StratifiedGroupKFoldSKLearn(
                n_folds=5,
                labels=pool_labels_ct,
                groups=pool_groups_ct,
                seed=42,
            )

            cfg_ct = load_merged_config(
                "configs/base.yaml",
                "configs/data/3w.yaml",
                "configs/models/convtimenet.yaml",
            )
            cfg_ct.training.max_epochs = args.max_epochs
            cfg_ct.training.batch_size = (
                32  # reduced for 720-length sequences — VRAM safety
            )
            cfg_ct.device = args.device
            cfg_ct.training.scheduler = "cosine"

            model_kwargs_ct = {
                "task": "classification",
                "n_vars": 27,
                "n_classes": cfg_ct.data.n_classes,
                "window_size": 720,
            }

            # Merge architecture params from model config
            if hasattr(cfg_ct, "model") and hasattr(cfg_ct.model, "architecture"):
                arch_ct = OmegaConf.to_container(
                    cfg_ct.model.architecture, resolve=True
                )
                model_kwargs_ct.update(arch_ct)

            # Merge training LR/weight_decay
            if hasattr(cfg_ct, "model") and hasattr(cfg_ct.model, "training"):
                model_kwargs_ct["lr"] = cfg_ct.model.training.lr
                model_kwargs_ct["weight_decay"] = cfg_ct.model.training.weight_decay

            convtimenet_raw_start = time.time()
            runner_ct_raw = ExperimentRunner(
                model_class=ConvTimeNetModel,
                dataset=raw_dataset_ct,
                cv_strategy=inner_cv_ct,
                cfg=cfg_ct,
                model_kwargs=model_kwargs_ct,
            )

            convtimenet_raw_results = runner_ct_raw.run_nested(
                train_pool=train_pool,
                test_indices=test_indices,
                use_mlflow=not args.no_mlflow,
            )
            convtimenet_raw_elapsed = time.time() - convtimenet_raw_start

            out_path_ct_raw = RESULTS_DIR / "convtimenet_raw" / "3w.json"
            out_path_ct_raw.parent.mkdir(parents=True, exist_ok=True)
            out_path_ct_raw.write_text(
                json.dumps(_make_serializable(convtimenet_raw_results), indent=2)
            )
            logger.info("  Results saved: %s", out_path_ct_raw)

            agg_ct_raw = convtimenet_raw_results.get("test_metrics", {})
            metric_str_ct_raw = ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(agg_ct_raw.items())
                if isinstance(v, (int, float))
            )
            summary["convtimenet_raw"] = {
                "status": "ok",
                "elapsed": round(convtimenet_raw_elapsed, 1),
                "test_metrics": convtimenet_raw_results.get("test_metrics", {}),
                "cv_aggregate": convtimenet_raw_results.get("cv_aggregate", {}),
                "n_train": convtimenet_raw_results.get("n_train", 0),
                "n_test": convtimenet_raw_results.get("n_test", 0),
                "n_cv_folds": convtimenet_raw_results.get("n_cv_folds", 0),
            }
            logger.info(
                "✓ convtimenet_raw: %s (%.1fs)",
                metric_str_ct_raw,
                convtimenet_raw_elapsed,
            )

            # Re-save summary with convtimenet_raw
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

        except Exception as e:
            logger.error("✗ convtimenet_raw failed: %s", e)
            traceback.print_exc()
            summary["convtimenet_raw"] = {"status": "error", "error": str(e)}
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    elif run_convtimenet_raw and ConvTimeNetModel is None:
        logger.warning("ConvTimeNetModel not available — skipping convtimenet_raw")
        summary["convtimenet_raw"] = {
            "status": "skipped",
            "error": "ConvTimeNetModel import failed",
        }
        summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

    RAW_DL_MODELS = {
        "convtran_raw": (ConvTranModel, "configs/models/convtran.yaml"),
        "inception_time_raw": (
            InceptionTimeModel,
            "configs/models/inception_time.yaml",
        ),
    }
    for raw_name, (raw_cls, raw_cfg_path) in RAW_DL_MODELS.items():
        run_this = (args.models is None) or (raw_name in args.models)
        if not run_this or raw_cls is None:
            continue
        logger.info("─" * 60)
        logger.info("%s CLASSIFICATION (720×27 raw windows)", raw_name.upper())
        logger.info("─" * 60)
        try:
            if dataset is not None:
                del dataset
                dataset = None
            import gc

            gc.collect()
            from offshore_dl.data.datasets import ThreeWDataset

            if _raw_ds_shared is not None:
                _raw_ds = _raw_ds_shared
            else:
                try:
                    _raw_ds = raw_dataset  # type: ignore[name-defined]
                except NameError:
                    _raw_ds = ThreeWDataset(
                        "configs/data/3w.yaml", cache_in_memory=False
                    )
            _raw_labels = np.array([w["label"] for w in _raw_ds._windows])
            _raw_groups = np.array([w["instance_id"] for w in _raw_ds._windows])
            _pool_labels = _raw_labels[train_pool]
            _pool_groups = _raw_groups[train_pool]
            _inner_cv = StratifiedGroupKFoldSKLearn(
                n_folds=5,
                labels=_pool_labels,
                groups=_pool_groups,
                seed=42,
            )
            _cfg = load_merged_config(
                "configs/base.yaml", "configs/data/3w.yaml", raw_cfg_path
            )
            _cfg.training.max_epochs = args.max_epochs
            _cfg.training.batch_size = 64
            _cfg.device = args.device
            _cfg.training.scheduler = "cosine"
            _mkw = {
                "task": "classification",
                "n_vars": 27,
                "n_classes": _cfg.data.n_classes,
                "window_size": 720,
            }
            if hasattr(_cfg, "model") and hasattr(_cfg.model, "architecture"):
                _mkw.update(
                    OmegaConf.to_container(_cfg.model.architecture, resolve=True)
                )
            if hasattr(_cfg, "model") and hasattr(_cfg.model, "training"):
                _mkw["lr"] = _cfg.model.training.lr
                if hasattr(_cfg.model.training, "weight_decay"):
                    _mkw["weight_decay"] = _cfg.model.training.weight_decay
            _start = time.time()
            _runner = ExperimentRunner(
                model_class=raw_cls,
                dataset=_raw_ds,
                cv_strategy=_inner_cv,
                cfg=_cfg,
                model_kwargs=_mkw,
            )
            _results = _runner.run_nested(
                train_pool=train_pool,
                test_indices=test_indices,
                use_mlflow=not args.no_mlflow,
            )
            _elapsed = time.time() - _start
            _out = RESULTS_DIR / raw_name / "3w.json"
            _out.parent.mkdir(parents=True, exist_ok=True)
            _out.write_text(json.dumps(_make_serializable(_results), indent=2))
            logger.info("  Results saved: %s", _out)
            _agg = _results.get("test_metrics", {})
            _ms = ", ".join(
                f"{k}={v:.4f}"
                for k, v in sorted(_agg.items())
                if isinstance(v, (int, float))
            )
            summary[raw_name] = {
                "status": "ok",
                "elapsed": round(_elapsed, 1),
                "test_metrics": _results.get("test_metrics", {}),
            }
            logger.info("✓ %s: %s (%.1fs)", raw_name, _ms, _elapsed)
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
        except Exception as e:
            logger.error("✗ %s failed: %s", raw_name, e)
            traceback.print_exc()
            summary[raw_name] = {"status": "error", "error": str(e)}
            summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))

    # ── TiRex Classification (embedding + RF) ──
    # TiRex uses raw windows, not feature-extracted — loads its own dataset.
    # Uses the SAME holdout split (train_pool/test_indices) as trained models
    # to ensure comparable evaluation on identical test samples.
    # Inner CV within train_pool, then retrain RF on full train_pool, test on held-out.
    # Skip if --models filter is active and tirex is not in the list.
    run_tirex = (args.models is None) or ("tirex" in args.models)
    if run_tirex:
        logger.info("─" * 60)
        logger.info("TIREX CLASSIFICATION (nested: inner CV + held-out test)")
        logger.info("─" * 60)
        try:
            from offshore_dl.models.tirex_classifier import (
                TiRexClassifier,
                is_available,
            )

            if is_available():
                from offshore_dl.data.datasets import ThreeWDataset
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import (
                    accuracy_score,
                    f1_score,
                    average_precision_score,
                    confusion_matrix as sk_confusion_matrix,
                )

                raw_dataset = ThreeWDataset(
                    "configs/data/3w.yaml", cache_in_memory=False
                )
                logger.info(
                    "  Raw 3W loaded: %d samples (cache_in_memory=False)",
                    len(raw_dataset),
                )

                # Extract ALL embeddings once using a single TiRex instance
                n = len(raw_dataset)
                clf = TiRexClassifier(
                    n_vars=27,
                    n_classes=10,
                    device=args.device,
                    batch_size=32,
                )
                all_indices = np.arange(n)
                all_embeddings, all_labels = clf.extract_all_embeddings(
                    raw_dataset, all_indices
                )
                logger.info("  Embeddings extracted: %s", all_embeddings.shape)

                # Free GPU memory — TiRex model no longer needed
                del clf
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Get groups from _windows metadata (zero-copy, no __getitem__)
                tirex_groups = np.array(
                    [w["instance_id"] for w in raw_dataset._windows]
                )

                # ── Use same holdout split as trained models ──
                # The holdout was computed on the feature dataset with the same
                # labels and groups. Since both datasets have identical sample
                # ordering (same underlying 3W instances), indices are directly
                # reusable.
                tirex_train_pool = train_pool
                tirex_test_indices = test_indices
                tirex_test_groups = tirex_groups[tirex_test_indices]

                # Inner CV within train_pool
                pool_labels_tirex = all_labels[tirex_train_pool]
                pool_groups_tirex = tirex_groups[tirex_train_pool]
                tirex_inner_cv = StratifiedGroupKFoldSKLearn(
                    n_folds=5,
                    labels=pool_labels_tirex,
                    groups=pool_groups_tirex,
                    seed=42,
                )
                tirex_inner_splits = tirex_inner_cv.get_splits(len(tirex_train_pool))

                tirex_start = time.time()
                tirex_cv_fold_results = []
                classes = np.unique(all_labels)

                # ── Inner CV folds (for variance estimates) ──
                for fold_idx, (local_train, local_val) in enumerate(tirex_inner_splits):
                    global_train = tirex_train_pool[local_train]
                    global_val = tirex_train_pool[local_val]
                    logger.info(
                        "  TiRex inner fold %d/%d (train=%d, val=%d)",
                        fold_idx + 1,
                        len(tirex_inner_splits),
                        len(global_train),
                        len(global_val),
                    )

                    X_train = all_embeddings[global_train]
                    y_train = all_labels[global_train]
                    X_val = all_embeddings[global_val]
                    y_val = all_labels[global_val]

                    rf = RandomForestClassifier(
                        n_estimators=500,
                        n_jobs=-1,
                        random_state=42,
                        class_weight="balanced",
                    )
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_val)
                    y_proba = rf.predict_proba(X_val)

                    acc = float(accuracy_score(y_val, y_pred))
                    f1_macro = float(
                        f1_score(y_val, y_pred, average="macro", zero_division=0)
                    )

                    from sklearn.preprocessing import label_binarize

                    y_val_bin = label_binarize(y_val, classes=classes)
                    auc_pr_scores = []
                    for c_idx in range(len(classes)):
                        if y_val_bin[:, c_idx].sum() > 0:
                            auc_pr_scores.append(
                                float(
                                    average_precision_score(
                                        y_val_bin[:, c_idx], y_proba[:, c_idx]
                                    )
                                )
                            )
                    auc_pr = float(np.mean(auc_pr_scores)) if auc_pr_scores else 0.0

                    tirex_cv_fold_results.append(
                        {
                            "fold_idx": fold_idx,
                            "metrics": {
                                "accuracy": acc,
                                "f1_macro": f1_macro,
                                "auc_pr": auc_pr,
                            },
                        }
                    )
                    logger.info(
                        "    inner: acc=%.4f  f1m=%.4f  auc=%.4f", acc, f1_macro, auc_pr
                    )

                # ── Retrain RF on full train_pool ──
                logger.info(
                    "  Retraining RF on full train pool (%d samples)",
                    len(tirex_train_pool),
                )
                X_train_full = all_embeddings[tirex_train_pool]
                y_train_full = all_labels[tirex_train_pool]

                rf_final = RandomForestClassifier(
                    n_estimators=500,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                )
                rf_final.fit(X_train_full, y_train_full)

                # ── Evaluate on held-out test ──
                X_test = all_embeddings[tirex_test_indices]
                y_test = all_labels[tirex_test_indices]
                y_pred_test = rf_final.predict(X_test)
                y_proba_test = rf_final.predict_proba(X_test)

                test_acc = float(accuracy_score(y_test, y_pred_test))
                test_f1_macro = float(
                    f1_score(y_test, y_pred_test, average="macro", zero_division=0)
                )
                test_f1_weighted = float(
                    f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
                )

                from sklearn.preprocessing import label_binarize

                y_test_bin = label_binarize(y_test, classes=classes)
                test_auc_pr_scores = []
                for c_idx in range(len(classes)):
                    if y_test_bin[:, c_idx].sum() > 0:
                        test_auc_pr_scores.append(
                            float(
                                average_precision_score(
                                    y_test_bin[:, c_idx], y_proba_test[:, c_idx]
                                )
                            )
                        )
                test_auc_pr = (
                    float(np.mean(test_auc_pr_scores)) if test_auc_pr_scores else 0.0
                )

                metrics = MetricRegistry.compute(
                    "classification",
                    y_pred_test,
                    y_test,
                    instance_ids=tirex_test_groups
                    if len(tirex_test_groups) > 0
                    else None,
                    prediction_scores=y_proba_test if len(y_proba_test) > 0 else None,
                )
                test_edr = metrics["edr"]

                cm = sk_confusion_matrix(y_test, y_pred_test, labels=classes)

                test_metrics = {
                    "accuracy": test_acc,
                    "f1_macro": test_f1_macro,
                    "f1_weighted": test_f1_weighted,
                    "auc_pr": test_auc_pr,
                    "edr": test_edr,
                    "confusion_matrix": cm.tolist(),
                    "class_labels": [int(c) for c in classes],
                }

                # Aggregate inner CV
                tirex_cv_agg = {}
                for key in ["accuracy", "f1_macro", "auc_pr"]:
                    vals = [fr["metrics"][key] for fr in tirex_cv_fold_results]
                    tirex_cv_agg[f"{key}_mean"] = float(np.mean(vals))
                    tirex_cv_agg[f"{key}_std"] = float(np.std(vals))

                tirex_elapsed = time.time() - tirex_start

                tirex_results = {
                    "test_metrics": test_metrics,
                    "cv_aggregate": tirex_cv_agg,
                    "cv_fold_results": tirex_cv_fold_results,
                    "n_train": len(tirex_train_pool),
                    "n_test": len(tirex_test_indices),
                    "n_cv_folds": len(tirex_inner_splits),
                    "embedding_dim": all_embeddings.shape[1],
                    "n_estimators": 500,
                }

                out_path = RESULTS_DIR / "tirex" / "3w.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(_make_serializable(tirex_results), indent=2)
                )

                metric_str = ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in sorted(test_metrics.items())
                    if isinstance(v, (int, float))
                )
                logger.info("✓ tirex TEST: %s (%.1fs)", metric_str, tirex_elapsed)
                summary["tirex"] = {
                    "status": "ok",
                    "elapsed": round(tirex_elapsed, 1),
                    "test_metrics": test_metrics,
                    "cv_aggregate": tirex_cv_agg,
                    "n_train": len(tirex_train_pool),
                    "n_test": len(tirex_test_indices),
                }
            else:
                logger.warning("TiRex not available — skipping classification")
                summary["tirex"] = {"status": "skipped", "error": "TiRex not installed"}
        except Exception as e:
            logger.error("✗ tirex failed: %s", e)
            traceback.print_exc()
            summary["tirex"] = {"status": "error", "error": str(e)}

        # Re-save summary with TiRex
        summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
        logger.info("Updated summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
