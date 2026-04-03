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

from offshore_dl.data.datasets import ThreeWFeatureDataset
from offshore_dl.data.feature_extractor import N_FEATURES
from offshore_dl.evaluation.cv import HoldoutSplitter, StratifiedGroupKFoldSKLearn
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.models.deeponet import DeepONetModel
from offshore_dl.models.lstm import LSTMModel
from offshore_dl.models.patchtst import PatchTSTModel
from offshore_dl.training.experiment import ExperimentRunner
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed

try:
    from offshore_dl.models.fkmad import FKMADModel
except (ImportError, ModuleNotFoundError, RuntimeError):
    FKMADModel = None

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
}

if FKMADModel is not None:
    MODELS["fkmad"] = {
        "class": FKMADModel,
        "config": "configs/models/fkmad.yaml",
        "overrides": {
            "d_model": 128,
            "n_mamba_layers": 2,
            "dropout": 0.2,
        },
    }

TREE_MODELS = ["random_forest"]
RAW_MODELS = ["fkmad_raw"]
ALL_MODELS = list(MODELS.keys()) + TREE_MODELS + RAW_MODELS

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

    # Extract numpy arrays: flatten (14, 27) → (378,)
    n = len(dataset)
    X_all = np.empty((n, N_FEATURES * 27), dtype=np.float32)
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
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
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

    if mlflow:
        mlflow.start_run(run_name="random_forest_nested_cv")
        mlflow.log_params({k: str(v) for k, v in arch.items()})

    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
        logger.info("  ── random_forest inner fold %d/%d", fold_idx + 1, len(inner_splits))
        X_tr, Y_tr = X_train_pool[local_train], Y_train_pool[local_train]
        X_va, Y_va = X_train_pool[local_val], Y_train_pool[local_val]

        clf = RandomForestClassifier(**arch)
        clf.fit(X_tr, Y_tr)
        preds = clf.predict(X_va)
        probs = clf.predict_proba(X_va)

        metrics = MetricRegistry.compute("classification", preds, Y_va, prediction_scores=probs)
        cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})
        logger.info("    fold %d: accuracy=%.4f, f1_macro=%.4f",
                     fold_idx, metrics["accuracy"], metrics["f1_macro"])
        if mlflow:
            mlflow.log_metric(f"cv_fold_{fold_idx}_accuracy", metrics["accuracy"])
            mlflow.log_metric(f"cv_fold_{fold_idx}_f1_macro", metrics["f1_macro"])

    # Aggregate CV metrics
    cv_agg = {}
    if cv_fold_results:
        metric_keys = [k for k in cv_fold_results[0]["metrics"] if isinstance(cv_fold_results[0]["metrics"][k], (int, float))]
        for k in metric_keys:
            vals = [f["metrics"][k] for f in cv_fold_results]
            cv_agg[f"{k}_mean"] = float(np.mean(vals))
            cv_agg[f"{k}_std"] = float(np.std(vals))

    # Retrain on full training pool, evaluate on held-out test
    logger.info("  ── random_forest: retrain on full train pool (%d samples)", len(train_pool))
    final_clf = RandomForestClassifier(**arch)
    final_clf.fit(X_train_pool, Y_train_pool)
    test_preds = final_clf.predict(X_test)
    test_probs = final_clf.predict_proba(X_test)
    test_metrics = MetricRegistry.compute("classification", test_preds, Y_test, prediction_scores=test_probs)

    if mlflow:
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)
        mlflow.end_run()

    result = {
        "test_metrics": test_metrics,
        "cv_aggregate": cv_agg,
        "cv_fold_results": cv_fold_results,
        "n_train": len(train_pool),
        "n_test": len(test_indices),
        "n_cv_folds": len(inner_splits),
    }

    metric_str = ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(test_metrics.items())
        if isinstance(v, (int, float))
    )
    print(f"\n  RANDOM FOREST on 3W features")
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
        "configs/data/3w.yaml",
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
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
    )

    # window_size = N_FEATURES (14) — the feature sequence length
    model_kwargs = {
        "task": "classification",
        "n_vars": 27,
        "n_classes": cfg.data.n_classes,
        "window_size": N_FEATURES,  # 14, not 720
    }

    # Merge architecture params from model config
    if hasattr(cfg, "model") and hasattr(cfg.model, "architecture"):
        arch = OmegaConf.to_container(cfg.model.architecture, resolve=True)
        model_kwargs.update(arch)

    # Apply per-model overrides for feature-based training
    overrides = entry.get("overrides", {})
    model_kwargs.update(overrides)

    # Merge training LR/weight_decay from model config
    if hasattr(cfg, "model") and hasattr(cfg.model, "training"):
        model_kwargs["lr"] = cfg.model.training.lr
        model_kwargs["weight_decay"] = cfg.model.training.weight_decay

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production training: 3 models on 3W feature-extracted data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to run (default: all). Choices: " + ", ".join(ALL_MODELS))

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

    # ── Detect whether any non-raw models are requested ──
    needs_feature_dataset = any(m not in RAW_MODELS for m in models_to_run)

    logger.info("═" * 70)
    logger.info("3W FEATURE-BASED TRAINING — nested CV (inner 5-fold + held-out test)")
    logger.info("  device=%s  max_epochs=%d  batch_size=%d", args.device, args.max_epochs, args.batch_size)
    logger.info("  Features: (720, 27) → (%d, 27) statistical descriptors", N_FEATURES)
    logger.info("═" * 70)

    if needs_feature_dataset:
        logger.info("Loading 3W dataset with feature extraction …")
        ds_start = time.time()
        dataset = ThreeWFeatureDataset("configs/data/3w.yaml")
        logger.info("  3W loaded: %d samples (%.1fs)", len(dataset), time.time() - ds_start)

        # ── Compute labels and groups once for all feature models ──
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        groups = np.array([dataset[i][2]["instance_id"] for i in range(len(dataset))])
        _raw_ds_shared = None  # no pre-loaded raw dataset
    else:
        # Raw-only mode: load ThreeWDataset with cache_in_memory=False
        # (caches feature arrays ~8 GB but NOT DataFrames ~17 GB).
        # This dataset is reused directly by the fkmad_raw block below.
        logger.info("Raw-only mode — loading 3W dataset for holdout + training …")
        ds_start = time.time()
        from offshore_dl.data.datasets import ThreeWDataset as _ThreeWDataset
        _raw_ds_shared = _ThreeWDataset("configs/data/3w.yaml", cache_in_memory=False)
        logger.info("  3W loaded: %d samples (%.1fs)", len(_raw_ds_shared), time.time() - ds_start)

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
    train_pool, test_indices = holdout.split(len(labels))  # works for both feature and raw-only
    logger.info(
        "Holdout split: train_pool=%d, test=%d",
        len(train_pool), len(test_indices),
    )

    sweep_start = time.time()
    summary: dict[str, dict] = {}

    for model_name in models_to_run:
        if model_name in RAW_MODELS:
            continue  # handled in dedicated block below

        logger.info("─" * 60)
        logger.info("TRAINING: %s on 3W features", model_name.upper())
        logger.info("─" * 60)

        start = time.time()
        try:
            is_tree = model_name in TREE_MODELS
            if is_tree:
                results = _run_rf_model(
                    dataset=dataset,
                    labels=labels,
                    groups=groups,
                    train_pool=train_pool,
                    test_indices=test_indices,
                    use_mlflow=not args.no_mlflow,
                )
            else:
                results = _run_model(
                    model_name=model_name,
                    dataset=dataset,
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
                f"{k}={v:.4f}" for k, v in sorted(agg.items())
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

    print(f"\n{'═'*70}")
    print(f"  3W FEATURE-BASED TRAINING COMPLETE (nested CV + held-out test)")
    print(f"{'═'*70}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Train pool: {len(train_pool)}, Held-out test: {len(test_indices)}")
    for model_name, s in summary.items():
        if s["status"] == "ok":
            tm = s.get("test_metrics", {})
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(tm.items())
                if isinstance(v, (int, float))
            )
            print(f"    {model_name:12s} ✓ {s['elapsed']:8.1f}s  TEST: {metric_str}")
        else:
            print(f"    {model_name:12s} ✗ {s['elapsed']:8.1f}s  ERROR: {s.get('error', 'unknown')}")
    print(f"{'═'*70}\n")

    summary_path = RESULTS_DIR / "summary_production_3w_features.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_make_serializable(summary), indent=2))
    logger.info("Summary saved: %s", summary_path)

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
            import gc; gc.collect()

            from offshore_dl.data.datasets import ThreeWDataset

            # Reuse raw dataset from raw-only mode if available,
            # otherwise load fresh with cache_in_memory=False to avoid
            # the ~17 GB DataFrame cache that caused OOM on 30 GB RAM.
            if _raw_ds_shared is not None:
                raw_dataset = _raw_ds_shared
                logger.info("  Reusing raw dataset from holdout phase: %d samples", len(raw_dataset))
            else:
                raw_dataset = ThreeWDataset("configs/data/3w.yaml", cache_in_memory=False)
                logger.info("  Raw 3W loaded: %d samples (cache_in_memory=False)", len(raw_dataset))

            # Extract labels/groups from _windows metadata (zero-copy, no __getitem__)
            raw_labels = np.array([w["label"] for w in raw_dataset._windows])
            raw_groups = np.array([w["instance_id"] for w in raw_dataset._windows])

            # Inner CV within train_pool (reuse same holdout split)
            pool_labels_raw = raw_labels[train_pool]
            pool_groups_raw = raw_groups[train_pool]
            inner_cv_raw = StratifiedGroupKFoldSKLearn(
                n_folds=5, labels=pool_labels_raw, groups=pool_groups_raw, seed=42,
            )

            cfg_raw = load_merged_config(
                "configs/base.yaml", "configs/data/3w.yaml", "configs/models/fkmad.yaml",
            )
            cfg_raw.training.max_epochs = args.max_epochs
            cfg_raw.training.batch_size = 32  # reduced for 720-length sequences — VRAM safety
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
                arch_raw = OmegaConf.to_container(cfg_raw.model.architecture, resolve=True)
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
            out_path_raw.write_text(json.dumps(_make_serializable(fkmad_raw_results), indent=2))
            logger.info("  Results saved: %s", out_path_raw)

            agg_raw = fkmad_raw_results.get("test_metrics", {})
            metric_str_raw = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(agg_raw.items())
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
        summary["fkmad_raw"] = {"status": "skipped", "error": "FKMADModel import failed (CUDA required)"}
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
            from offshore_dl.models.tirex_classifier import TiRexClassifier, is_available
    
            if is_available():
                from offshore_dl.data.datasets import ThreeWDataset
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import (
                    accuracy_score, f1_score, average_precision_score,
                    confusion_matrix as sk_confusion_matrix,
                )
    
                raw_dataset = ThreeWDataset("configs/data/3w.yaml", cache_in_memory=False)
                logger.info("  Raw 3W loaded: %d samples (cache_in_memory=False)", len(raw_dataset))
    
                # Extract ALL embeddings once using a single TiRex instance
                n = len(raw_dataset)
                clf = TiRexClassifier(
                    n_vars=27, n_classes=10, device=args.device,
                    batch_size=32,
                )
                all_indices = np.arange(n)
                all_embeddings, all_labels = clf.extract_all_embeddings(raw_dataset, all_indices)
                logger.info("  Embeddings extracted: %s", all_embeddings.shape)
    
                # Free GPU memory — TiRex model no longer needed
                del clf
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
                # Get groups from _windows metadata (zero-copy, no __getitem__)
                tirex_groups = np.array([w["instance_id"] for w in raw_dataset._windows])
    
                # ── Use same holdout split as trained models ──
                # The holdout was computed on the feature dataset with the same
                # labels and groups. Since both datasets have identical sample
                # ordering (same underlying 3W instances), indices are directly
                # reusable.
                tirex_train_pool = train_pool
                tirex_test_indices = test_indices
    
                # Inner CV within train_pool
                pool_labels_tirex = all_labels[tirex_train_pool]
                pool_groups_tirex = tirex_groups[tirex_train_pool]
                tirex_inner_cv = StratifiedGroupKFoldSKLearn(
                    n_folds=5, labels=pool_labels_tirex,
                    groups=pool_groups_tirex, seed=42,
                )
                tirex_inner_splits = tirex_inner_cv.get_splits(len(tirex_train_pool))
    
                tirex_start = time.time()
                tirex_cv_fold_results = []
                classes = np.unique(all_labels)
    
                # ── Inner CV folds (for variance estimates) ──
                for fold_idx, (local_train, local_val) in enumerate(tirex_inner_splits):
                    global_train = tirex_train_pool[local_train]
                    global_val = tirex_train_pool[local_val]
                    logger.info("  TiRex inner fold %d/%d (train=%d, val=%d)",
                               fold_idx + 1, len(tirex_inner_splits),
                               len(global_train), len(global_val))
    
                    X_train = all_embeddings[global_train]
                    y_train = all_labels[global_train]
                    X_val = all_embeddings[global_val]
                    y_val = all_labels[global_val]
    
                    rf = RandomForestClassifier(
                        n_estimators=500, n_jobs=-1, random_state=42,
                        class_weight="balanced",
                    )
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_val)
                    y_proba = rf.predict_proba(X_val)
    
                    acc = float(accuracy_score(y_val, y_pred))
                    f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    
                    from sklearn.preprocessing import label_binarize
                    y_val_bin = label_binarize(y_val, classes=classes)
                    auc_pr_scores = []
                    for c_idx in range(len(classes)):
                        if y_val_bin[:, c_idx].sum() > 0:
                            auc_pr_scores.append(
                                float(average_precision_score(y_val_bin[:, c_idx], y_proba[:, c_idx]))
                            )
                    auc_pr = float(np.mean(auc_pr_scores)) if auc_pr_scores else 0.0
    
                    tirex_cv_fold_results.append({
                        "fold_idx": fold_idx,
                        "metrics": {"accuracy": acc, "f1_macro": f1_macro, "auc_pr": auc_pr},
                    })
                    logger.info("    inner: acc=%.4f  f1m=%.4f  auc=%.4f", acc, f1_macro, auc_pr)
    
                # ── Retrain RF on full train_pool ──
                logger.info("  Retraining RF on full train pool (%d samples)", len(tirex_train_pool))
                X_train_full = all_embeddings[tirex_train_pool]
                y_train_full = all_labels[tirex_train_pool]
    
                rf_final = RandomForestClassifier(
                    n_estimators=500, n_jobs=-1, random_state=42,
                    class_weight="balanced",
                )
                rf_final.fit(X_train_full, y_train_full)
    
                # ── Evaluate on held-out test ──
                X_test = all_embeddings[tirex_test_indices]
                y_test = all_labels[tirex_test_indices]
                y_pred_test = rf_final.predict(X_test)
                y_proba_test = rf_final.predict_proba(X_test)
    
                test_acc = float(accuracy_score(y_test, y_pred_test))
                test_f1_macro = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0))
                test_f1_weighted = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))
    
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=classes)
                test_auc_pr_scores = []
                for c_idx in range(len(classes)):
                    if y_test_bin[:, c_idx].sum() > 0:
                        test_auc_pr_scores.append(
                            float(average_precision_score(y_test_bin[:, c_idx], y_proba_test[:, c_idx]))
                        )
                test_auc_pr = float(np.mean(test_auc_pr_scores)) if test_auc_pr_scores else 0.0
    
                per_class_f1 = f1_score(y_test, y_pred_test, average=None, zero_division=0)
                # NOTE: This EDR approximation (fraction of classes with F1>0) differs from
                # MetricRegistry.compute_edr (fraction of event instances with ≥1 correct prediction).
                # Both are reported for completeness.
                edr_class_fraction = float(np.mean(per_class_f1 > 0))
                test_edr = edr_class_fraction
    
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
                out_path.write_text(json.dumps(_make_serializable(tirex_results), indent=2))
    
                metric_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in sorted(test_metrics.items())
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
