"""Production sweep: 7 models × 4 horizons × 2 modes on Volve.

Orchestrates all production runs for the Volve Norwegian North Sea oil
production forecasting benchmark. Trained models (LSTM, DeepONet, PatchTST,
TCN) use ``run_and_save()``; zero-shot FMs (Chronos, TimesFM, TiRex) use
direct instantiation with manual CV evaluation.

Usage::

    # Full production sweep (GPU)
    python scripts/run_production_volve.py

    # Smoke test (CPU, 1 epoch, single model)
    python scripts/run_production_volve.py --max-epochs 1 --device cpu --models lstm

    # Dry run — print plan without executing
    python scripts/run_production_volve.py --dry-run

    # Docker invocation
    docker_run.sh python scripts/run_production_volve.py --device cuda
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

from offshore_dl.data.datasets import VolveDataset
from offshore_dl.evaluation.cv import (
    GroupedExpandingWindowCV,
    GroupedTemporalHoldoutSplitter,
)
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Sweep dimensions
# ═══════════════════════════════════════════════════════════════════

HORIZONS = [7, 14, 30, 90]
MODES = ["multi_well", "per_well"]
TRAINED_MODELS = ["lstm", "deeponet", "patchtst", "tcn"]
FM_MODELS = ["chronos", "timesfm", "tirex"]
TREE_MODELS: list[str] = []
ALL_MODELS = TRAINED_MODELS + FM_MODELS + TREE_MODELS

# Volve: 6 wells (Norwegian North Sea)
WELLS = ["NO_15_9-F-1_C", "NO_15_9-F-5_AH", "NO_15_9-F-11_H", "NO_15_9-F-12_H", "NO_15_9-F-14_H", "NO_15_9-F-15_D"]

FM_WRAPPER_MAP = {
    "chronos": ("offshore_dl.models.chronos_wrapper", "ChronosWrapper"),
    "timesfm": ("offshore_dl.models.timesfm_wrapper", "TimesFMWrapper"),
    "tirex": ("offshore_dl.models.tirex_wrapper", "TiRexWrapper"),
}

RESULTS_DIR = Path("results")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _safe_well(name: str) -> str:
    return name.replace("/", "_")


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


def _sample_groups(dataset: VolveDataset) -> np.ndarray:
    """Return per-sample well-group IDs for grouped temporal splitting."""
    return np.array([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)


def _make_holdout(dataset: VolveDataset) -> GroupedTemporalHoldoutSplitter:
    """Create a per-well temporal holdout splitter for this dataset."""
    return GroupedTemporalHoldoutSplitter(
        test_ratio=0.2,
        groups=_sample_groups(dataset),
    )


def _make_inner_cv(
    dataset: VolveDataset,
    indices: np.ndarray,
) -> GroupedExpandingWindowCV:
    """Create grouped expanding CV restricted to a subset of samples."""
    return GroupedExpandingWindowCV(
        groups=_sample_groups(dataset)[indices],
        n_splits=3,
        min_train_ratio=0.5,
    )


def _aggregate(fold_results: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean/std summaries."""
    agg: dict[str, float] = {}
    if not fold_results:
        return agg
    for key in fold_results[0]["metrics"]:
        values = [
            fr["metrics"][key]
            for fr in fold_results
            if np.isfinite(fr["metrics"].get(key, float("nan")))
        ]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg


def _zero_shot_evaluate(model, dataset, val_idx, batch_size=32, max_samples=None):
    """Run zero-shot FM inference on a validation set."""
    if max_samples and len(val_idx) > max_samples:
        val_idx = val_idx[:max_samples]

    all_preds = []
    all_targets = []

    for i in range(0, len(val_idx), batch_size):
        batch_idx = val_idx[i : i + batch_size]
        batch_x = torch.stack([dataset[j][0] for j in batch_idx])
        batch_y = torch.stack([dataset[j][1] for j in batch_idx])
        batch = (batch_x, batch_y, [{}] * len(batch_idx))

        preds = model.predict(batch)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())

    predictions = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    return MetricRegistry.compute("forecasting", predictions, targets)


def _load_fm_class(model_name: str):
    """Dynamically import an FM wrapper class."""
    module_path, class_name = FM_WRAPPER_MAP[model_name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ═══════════════════════════════════════════════════════════════════
# Trained-model sweep (uses run_and_save)
# ═══════════════════════════════════════════════════════════════════


def _run_trained_model(
    model_name: str,
    horizon: int,
    mode: str,
    well: str | None,
    max_epochs: int | None,
    device: str,
    use_mlflow: bool = True,
) -> dict:
    """Run a trained model with nested CV: temporal holdout + inner CV.

    Protocol:
      1. Temporal holdout: last 20% of samples → test set
      2. Inner 3-fold ExpandingWindowCV within the 80% training pool
      3. Retrain on full training pool
      4. Evaluate on held-out test set
    """
    # Lazy import — pulls in transformers/torch heavy dependencies only when
    # a trained model is actually requested (not needed for tree/FM paths).
    from offshore_dl.run_experiment import build_experiment  # noqa: F811

    ds_kwargs: dict = {"horizon": horizon, "mode": mode, "filter_shutdowns": False}
    if well:
        ds_kwargs["well_name"] = well

    runner, cfg = build_experiment(
        model_name=model_name,
        dataset_name="volve",
        max_epochs=max_epochs,
        device=device,
        dataset_kwargs=ds_kwargs,
    )

    # Temporal holdout: last 20% as test
    n = len(runner.dataset)
    holdout = _make_holdout(runner.dataset)
    train_pool, test_indices = holdout.split(n)

    results = runner.run_nested(
        train_pool=train_pool,
        test_indices=test_indices,
        use_mlflow=use_mlflow,
    )

    # Save results
    if well:
        safe_name = _safe_well(well)
        out_path = RESULTS_DIR / model_name / f"volve_h{horizon}_{mode}_{safe_name}.json"
    else:
        out_path = RESULTS_DIR / model_name / f"volve_h{horizon}_{mode}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_make_serializable(results), indent=2))
    logger.info("  Saved %s", out_path)

    # Print summary
    tm = results.get("test_metrics", {})
    metric_str = ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(tm.items())
        if isinstance(v, (int, float))
    )
    print(f"\n  {model_name.upper()} on VOLVE h{horizon} {mode}"
          f"{(' ' + well) if well else ''}")
    print(f"  TEST: {metric_str}")
    print(f"  Results saved: {out_path}\n")

    return results


# ═══════════════════════════════════════════════════════════════════
# FM zero-shot sweep (direct instantiation)
# ═══════════════════════════════════════════════════════════════════


def _run_fm_multi_well(
    model_name: str,
    horizon: int,
    max_samples: int | None = None,
) -> dict:
    """Run zero-shot FM on Volve multi_well with temporal holdout.

    FMs don't train, so the protocol is simpler:
      1. Temporal holdout: last 20% → test set
      2. Evaluate FM on test set only
      3. Also run inner CV on train pool for variance estimates

    D028: target_channel must be passed because Volve's target
    (BORE_OIL_VOL) does NOT sort to index 0 in the common columns.
    """
    set_global_seed(42)
    dataset = VolveDataset("configs/data/volve.yaml", horizon=horizon, mode="multi_well", filter_shutdowns=False)
    n_vars = dataset.n_vars

    fm_class = _load_fm_class(model_name)
    # D028: pass target_channel so FM predicts the correct channel
    model = fm_class(task="forecasting", n_vars=n_vars, horizon=horizon, window_size=90,
                     target_channel=dataset._target_col_idx)

    # Temporal holdout: last 20%
    n = len(dataset)
    holdout = _make_holdout(dataset)
    train_pool, test_indices = holdout.split(n)

    # ── Evaluate on held-out test set (primary metric) ──
    test_metrics = _zero_shot_evaluate(model, dataset, test_indices, max_samples=max_samples)

    # ── Inner CV on train pool (for variance estimates) ──
    cv = _make_inner_cv(dataset, train_pool)
    inner_splits = cv.get_splits(len(train_pool))
    cv_fold_results = []
    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
        global_val = train_pool[local_val]
        logger.info("  ── %s h%d multi_well inner fold %d/%d", model_name, horizon, fold_idx + 1, len(inner_splits))
        metrics = _zero_shot_evaluate(model, dataset, global_val, max_samples=max_samples)
        cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})

    cv_agg = _aggregate(cv_fold_results)

    result = {
        "test_metrics": test_metrics,
        "cv_aggregate": cv_agg,
        "cv_fold_results": cv_fold_results,
        "n_train": len(train_pool),
        "n_test": len(test_indices),
        "n_cv_folds": len(inner_splits),
    }

    # Save
    out_path = RESULTS_DIR / model_name / f"volve_h{horizon}_multi_well.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_make_serializable(result), indent=2))
    logger.info("  Saved %s", out_path)

    return result


def _run_fm_per_well(
    model_name: str,
    horizon: int,
    max_samples: int | None = None,
) -> list[dict]:
    """Run zero-shot FM on each Volve well for one horizon.

    Batches by model: creates FM once per well (n_vars can vary), iterates wells.

    D028: target_channel must be passed — BORE_OIL_VOL is not at index 0.
    """
    fm_class = _load_fm_class(model_name)
    per_well_results = []

    for well in WELLS:
        set_global_seed(42)
        safe = _safe_well(well)
        dataset = VolveDataset(
            "configs/data/volve.yaml",
            horizon=horizon,
            mode="per_well",
            well_name=well,
            filter_shutdowns=False,
        )

        if len(dataset) == 0:
            logger.warning("  ── %s h%d per_well %s: empty dataset, skipping", model_name, horizon, well)
            per_well_results.append({
                "well": well, "status": "skipped", "reason": "empty dataset",
            })
            continue

        n_vars = dataset.n_vars  # varies per well
        # D028: pass target_channel so FM predicts the correct channel
        model = fm_class(task="forecasting", n_vars=n_vars, horizon=horizon, window_size=90,
                         target_channel=dataset._target_col_idx)

        # Temporal holdout: last 20%
        n = len(dataset)
        holdout = _make_holdout(dataset)
        train_pool, test_idx = holdout.split(n)

        # Evaluate on held-out test
        test_metrics = _zero_shot_evaluate(model, dataset, test_idx, max_samples=max_samples)

        # Inner CV for variance
        cv = _make_inner_cv(dataset, train_pool)
        inner_splits = cv.get_splits(len(train_pool))
        cv_fold_results = []
        for fold_idx, (local_train, local_val) in enumerate(inner_splits):
            global_val = train_pool[local_val]
            logger.info("  ── %s h%d per_well %s inner fold %d/%d", model_name, horizon, well, fold_idx + 1, len(inner_splits))
            metrics = _zero_shot_evaluate(model, dataset, global_val, max_samples=max_samples)
            cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})

        cv_agg = _aggregate(cv_fold_results)
        result = {
            "test_metrics": test_metrics,
            "cv_aggregate": cv_agg,
            "cv_fold_results": cv_fold_results,
            "n_train": len(train_pool),
            "n_test": len(test_idx),
            "n_cv_folds": len(inner_splits),
            "well": well,
        }

        out_path = RESULTS_DIR / model_name / f"volve_h{horizon}_per_well_{safe}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_make_serializable(result), indent=2))
        logger.info("  Saved %s", out_path)

        per_well_results.append({"well": well, "status": "ok", "test_metrics": test_metrics, "cv_aggregate": cv_agg})

    return per_well_results


# ═══════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════


def _build_plan(models: list[str], multi_well_only: bool = False) -> list[dict]:
    """Build the ordered list of runs for the sweep."""
    plan: list[dict] = []
    for model in models:
        is_fm = model in FM_MODELS
        is_tree = model in TREE_MODELS
        for horizon in HORIZONS:
            # multi_well
            plan.append({
                "model": model,
                "horizon": horizon,
                "mode": "multi_well",
                "well": None,
                "is_fm": is_fm,
                "is_tree": is_tree,
            })
            # per_well: individual wells
            if not multi_well_only:
                for well in WELLS:
                    plan.append({
                        "model": model,
                        "horizon": horizon,
                        "mode": "per_well",
                        "well": well,
                        "is_fm": is_fm,
                        "is_tree": is_tree,
                    })
    return plan


def _print_plan(plan: list[dict]) -> None:
    """Print sweep plan without executing."""
    print(f"\n{'═'*70}")
    print(f"  VOLVE PRODUCTION SWEEP PLAN — {len(plan)} runs")
    print(f"{'═'*70}")
    for i, run in enumerate(plan, 1):
        well_str = f" well={run['well']}" if run["well"] else ""
        if run["is_fm"]:
            tag = " [zero-shot]"
        elif run.get("is_tree"):
            tag = " [tree]"
        else:
            tag = " [trained]"
        print(f"  {i:4d}. {run['model']:10s} h={run['horizon']:2d} {run['mode']:12s}{well_str}{tag}")
    print(f"{'═'*70}")
    print(f"  Total runs: {len(plan)}")
    n_trained = sum(1 for r in plan if not r["is_fm"] and not r.get("is_tree"))
    n_fm = sum(1 for r in plan if r["is_fm"])
    n_tree = sum(1 for r in plan if r.get("is_tree"))
    print(f"  Trained: {n_trained}, Zero-shot FM: {n_fm}, Tree: {n_tree}")
    print(f"{'═'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production sweep: Volve Norwegian North Sea oil forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max training epochs (None = use config)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS, help="Models to run")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap val samples per FM fold (for smoke tests)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose result JSON already exists")
    parser.add_argument("--multi-well-only", action="store_true", help="Run only multi_well mode, skip per_well")

    args = parser.parse_args()

    plan = _build_plan(args.models, multi_well_only=args.multi_well_only)

    if args.dry_run:
        _print_plan(plan)
        return

    logger.info("═" * 70)
    logger.info("VOLVE PRODUCTION SWEEP — %d runs", len(plan))
    logger.info("  device=%s  max_epochs=%s  models=%s", args.device, args.max_epochs, args.models)
    logger.info("═" * 70)

    sweep_start = time.time()
    all_status: dict[str, list[dict]] = {}  # model → list of run statuses

    # Group plan by model for efficient FM batching
    current_model = None
    for run_spec in plan:
        model = run_spec["model"]
        horizon = run_spec["horizon"]
        mode = run_spec["mode"]
        well = run_spec["well"]
        is_fm = run_spec["is_fm"]
        is_tree = run_spec.get("is_tree", False)

        if model not in all_status:
            all_status[model] = []

        run_label = f"{model} h{horizon} {mode}"
        if well:
            run_label += f" {well}"

        logger.info("─" * 60)
        logger.info("RUN: %s", run_label)
        logger.info("─" * 60)

        # Compute expected output path for skip check
        if well:
            _skip_safe = _safe_well(well)
            out_path = RESULTS_DIR / model / f"volve_h{horizon}_per_well_{_skip_safe}.json"
        elif mode == "multi_well" and is_fm:
            # FM multi_well uses a different naming convention (no mode prefix)
            out_path = RESULTS_DIR / model / f"volve_h{horizon}_multi_well.json"
        else:
            out_path = RESULTS_DIR / model / f"volve_h{horizon}_{mode}.json"

        if args.skip_existing and out_path.exists():
            logger.info("  SKIP (exists): %s", out_path)
            all_status[model].append({"run": run_label, "status": "skipped", "reason": "exists"})
            continue

        start = time.time()
        try:
            if is_fm:
                if mode == "multi_well":
                    result = _run_fm_multi_well(model, horizon, max_samples=args.max_samples)
                    agg = result.get("aggregate", {})
                else:
                    # per_well individual run
                    set_global_seed(42)
                    safe = _safe_well(well)
                    dataset = VolveDataset(
                        "configs/data/volve.yaml",
                        horizon=horizon,
                        mode="per_well",
                        well_name=well,
                        filter_shutdowns=False,
                    )
                    if len(dataset) == 0:
                        elapsed = time.time() - start
                        all_status[model].append({
                            "run": run_label, "status": "skipped",
                            "reason": "empty dataset", "elapsed": round(elapsed, 1),
                        })
                        logger.warning("  Skipped: empty dataset")
                        continue

                    n_vars = dataset.n_vars
                    fm_class = _load_fm_class(model)
                    # D028: pass target_channel so FM predicts the correct channel
                    fm_model = fm_class(task="forecasting", n_vars=n_vars, horizon=horizon, window_size=90,
                                        target_channel=dataset._target_col_idx)

                    # Temporal holdout
                    n_ds = len(dataset)
                    holdout = _make_holdout(dataset)
                    pw_train_pool, pw_test_idx = holdout.split(n_ds)

                    # Evaluate on held-out test
                    test_metrics = _zero_shot_evaluate(fm_model, dataset, pw_test_idx, max_samples=args.max_samples)

                    # Inner CV for variance
                    cv = _make_inner_cv(dataset, pw_train_pool)
                    inner_splits = cv.get_splits(len(pw_train_pool))
                    fold_results = []
                    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
                        global_val = pw_train_pool[local_val]
                        metrics = _zero_shot_evaluate(fm_model, dataset, global_val, max_samples=args.max_samples)
                        fold_results.append({"fold_idx": fold_idx, "metrics": metrics})

                    cv_agg = _aggregate(fold_results)
                    result = {
                        "test_metrics": test_metrics,
                        "cv_aggregate": cv_agg,
                        "cv_fold_results": fold_results,
                        "n_train": len(pw_train_pool),
                        "n_test": len(pw_test_idx),
                        "n_cv_folds": len(inner_splits),
                        "well": well,
                    }

                    out_path = RESULTS_DIR / model / f"volve_h{horizon}_per_well_{safe}.json"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(_make_serializable(result), indent=2))
                    logger.info("  Saved %s", out_path)
            elif is_tree:
                # Tree model — sklearn-style fit/predict (placeholder for future models)
                raise NotImplementedError(f"No tree-model runner for {model}")
            else:
                # Trained model — uses nested CV
                result = _run_trained_model(model, horizon, mode, well, args.max_epochs, args.device, use_mlflow=not args.no_mlflow)

            # Extract primary metrics (test_metrics for nested, fall back to aggregate)
            tm = result.get("test_metrics", result.get("aggregate", {}))
            elapsed = time.time() - start
            metric_str = ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(tm.items())
                if isinstance(v, (int, float))
            )
            all_status[model].append({
                "run": run_label, "status": "ok",
                "elapsed": round(elapsed, 1),
                "test_metrics": tm,
            })
            logger.info("✓ %s: %s (%.1fs)", run_label, metric_str, elapsed)

        except ImportError as e:
            # FM model dependency not available (e.g. TimesFM needs Python <3.12, TiRex not installed)
            # Write a graceful unavailability stub so downstream code can filter rather than crash.
            elapsed = time.time() - start
            stub = {
                "test_metrics": {},
                "cv_aggregate": {},
                "cv_fold_results": [],
                "status": "unavailable",
                "reason": str(e),
                "n_train": 0,
                "n_test": 0,
                "n_cv_folds": 0,
            }
            if is_fm:
                try:
                    stub_path = out_path
                    stub_path.parent.mkdir(parents=True, exist_ok=True)
                    stub_path.write_text(json.dumps(stub, indent=2))
                    logger.warning("  UNAVAILABLE %s → stub written: %s", run_label, stub_path)
                except Exception:
                    pass
            all_status[model].append({
                "run": run_label, "status": "unavailable",
                "elapsed": round(elapsed, 1), "error": str(e),
            })
            logger.error("✗ %s unavailable: %s (%.1fs)", run_label, e, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            all_status[model].append({
                "run": run_label, "status": "error",
                "elapsed": round(elapsed, 1), "error": str(e),
            })
            logger.error("✗ %s failed: %s (%.1fs)", run_label, e, elapsed)
            traceback.print_exc()

    # ── Per-model summary files ──────────────────────────────────
    for model, statuses in all_status.items():
        summary_path = RESULTS_DIR / model / "summary_production_volve.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_make_serializable(statuses), indent=2))
        logger.info("Summary saved: %s", summary_path)

    # ── Final report ─────────────────────────────────────────────
    total_elapsed = time.time() - sweep_start
    n_ok = sum(1 for sts in all_status.values() for s in sts if s["status"] == "ok")
    n_err = sum(1 for sts in all_status.values() for s in sts if s["status"] == "error")
    n_skip = sum(1 for sts in all_status.values() for s in sts if s["status"] == "skipped")

    print(f"\n{'═'*70}")
    print(f"  VOLVE PRODUCTION SWEEP COMPLETE")
    print(f"{'═'*70}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  OK: {n_ok}  Errors: {n_err}  Skipped: {n_skip}")
    for model, statuses in all_status.items():
        ok = sum(1 for s in statuses if s["status"] == "ok")
        err = sum(1 for s in statuses if s["status"] == "error")
        skip = sum(1 for s in statuses if s["status"] == "skipped")
        print(f"    {model:12s} — ok={ok}, err={err}, skip={skip}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
