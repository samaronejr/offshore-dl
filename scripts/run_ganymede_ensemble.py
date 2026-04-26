"""Ensemble stacking for Ganymede gas production forecasting.

Combines zero-shot foundation models with trained forecasters using a
linear regression stacker trained on inner-CV out-of-fold predictions and
evaluated on the same grouped temporal holdout used by the base runs.

Method (Modi & Pan 2025-inspired):
1. Collect per-sample validation/test predictions from each base model.
2. Build stacked features from base-model forecasts.
3. Fit a linear regression meta-learner on validation-fold predictions.
4. Evaluate the stacker on the held-out test set.

Notes
-----
- Existing result JSONs in this repo currently contain aggregate metrics only.
  When sample-level predictions are missing, this script can regenerate
  zero-shot FM predictions directly.
- Trained-model regeneration requires re-running training because checkpoints
  are not stored in the repository results. By default those models are
  skipped unless ``--retrain-trained-models`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

# Allow invocation from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from offshore_dl.data.datasets import GanymedeDataset
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.run_experiment import build_experiment
from offshore_dl.utils.reproducibility import set_global_seed
from offshore_dl.utils.serialization import make_serializable as _make_serializable
from sweep_utils import (
    FM_WRAPPER_MAP,
    load_fm_class as _load_fm_class,
    make_holdout as _make_holdout,
    make_inner_cv as _make_inner_cv,
    sample_groups as _sample_groups,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "ensemble_stack"
HORIZONS = [7, 14, 30, 90]
FM_MODELS = ["timesfm", "tirex", "chronos"]
TRAINED_MODELS = ["lstm", "deeponet", "patchtst"]
DEFAULT_MODELS = FM_MODELS + TRAINED_MODELS


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _result_path(model_name: str, horizon: int) -> Path:
    return RESULTS_DIR / model_name / f"ganymede_h{horizon}_multi_well.json"


def _stack_fold_payloads(
    cv_fold_results: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    blocks = []
    for fold in cv_fold_results:
        if not {"sample_indices", "predictions", "targets"}.issubset(fold):
            return None
        blocks.append(
            (
                np.asarray(fold["sample_indices"], dtype=np.int64),
                np.asarray(fold["predictions"], dtype=np.float32),
                np.asarray(fold["targets"], dtype=np.float32),
            )
        )

    if not blocks:
        return None

    indices = np.concatenate([b[0] for b in blocks])
    predictions = np.concatenate([b[1] for b in blocks], axis=0)
    targets = np.concatenate([b[2] for b in blocks], axis=0)
    order = np.argsort(indices)
    return indices[order], predictions[order], targets[order]


def _bundle_from_result(model_name: str, result: dict) -> dict | None:
    train_payload = _stack_fold_payloads(result.get("cv_fold_results", []))
    if train_payload is None:
        return None
    if not {"test_indices", "test_predictions", "test_targets"}.issubset(result):
        return None

    train_indices, train_predictions, train_targets = train_payload
    return {
        "model": model_name,
        "source": "stored_result",
        "train_indices": train_indices,
        "train_predictions": train_predictions,
        "train_targets": train_targets,
        "test_indices": np.asarray(result["test_indices"], dtype=np.int64),
        "test_predictions": np.asarray(result["test_predictions"], dtype=np.float32),
        "test_targets": np.asarray(result["test_targets"], dtype=np.float32),
    }


def _predict_fm_indices(
    model,
    dataset: GanymedeDataset,
    indices: np.ndarray,
    batch_size: int,
) -> dict:
    all_preds = []
    all_targets = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_x = torch.stack([dataset[int(i)][0] for i in batch_idx])
        batch_y = torch.stack([dataset[int(i)][1] for i in batch_idx])
        batch = (batch_x, batch_y, [{}] * len(batch_idx))
        preds = model.predict(batch)
        all_preds.append(preds.detach().cpu())
        all_targets.append(batch_y.cpu())

    predictions = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return {
        "sample_indices": np.asarray(indices, dtype=np.int64),
        "predictions": predictions,
        "targets": targets,
        "metrics": MetricRegistry.compute("forecasting", predictions, targets),
    }


def _regenerate_fm_bundle(
    model_name: str,
    horizon: int,
    batch_size: int,
) -> dict:
    set_global_seed(42)
    dataset = GanymedeDataset(
        "configs/data/ganymede.yaml",
        horizon=horizon,
        mode="multi_well",
        filter_shutdowns=False,
    )
    fm_class = _load_fm_class(model_name)
    model = fm_class(
        task="forecasting",
        n_vars=dataset.n_vars,
        horizon=horizon,
        window_size=dataset.input_window,
        target_channel=dataset._target_col_idx,
    )

    holdout = _make_holdout(dataset)
    train_pool, test_indices = holdout.split(len(dataset))

    inner_cv = _make_inner_cv(dataset, train_pool)
    inner_splits = inner_cv.get_splits(len(train_pool))
    fold_payloads = []
    for local_train, local_val in inner_splits:
        del local_train
        global_val = train_pool[local_val]
        fold_payloads.append(
            _predict_fm_indices(model, dataset, global_val, batch_size)
        )

    train_indices, train_predictions, train_targets = _stack_fold_payloads(
        fold_payloads
    )
    test_payload = _predict_fm_indices(model, dataset, test_indices, batch_size)
    return {
        "model": model_name,
        "source": "regenerated_fm",
        "train_indices": train_indices,
        "train_predictions": train_predictions,
        "train_targets": train_targets,
        "test_indices": test_payload["sample_indices"],
        "test_predictions": test_payload["predictions"],
        "test_targets": test_payload["targets"],
    }


def _regenerate_trained_bundle(
    model_name: str,
    horizon: int,
    device: str,
    max_epochs: int | None,
) -> dict:
    ds_kwargs = {"horizon": horizon, "mode": "multi_well", "filter_shutdowns": False}
    runner, _ = build_experiment(
        model_name=model_name,
        dataset_name="ganymede",
        max_epochs=max_epochs,
        device=device,
        dataset_kwargs=ds_kwargs,
    )
    holdout = _make_holdout(runner.dataset)
    train_pool, test_indices = holdout.split(len(runner.dataset))
    result = runner.run_nested(
        train_pool=train_pool, test_indices=test_indices, use_mlflow=False
    )
    bundle = _bundle_from_result(model_name, _make_serializable(result))
    if bundle is None:
        raise RuntimeError(
            f"{model_name}: trained regeneration did not produce sample-level predictions"
        )
    bundle["source"] = "retrained_nested"
    return bundle


def _resolve_model_bundle(
    model_name: str,
    horizon: int,
    batch_size: int,
    device: str,
    max_epochs: int | None,
    retrain_trained_models: bool,
) -> tuple[dict | None, str | None]:
    result = _load_json(_result_path(model_name, horizon))
    if result is not None:
        bundle = _bundle_from_result(model_name, result)
        if bundle is not None:
            return bundle, None

    if model_name in FM_MODELS:
        try:
            return _regenerate_fm_bundle(model_name, horizon, batch_size), None
        except Exception as exc:
            return None, f"FM regeneration failed: {exc}"

    if retrain_trained_models:
        logger.warning(
            "%s h%d: no stored sample-level predictions/checkpoints; re-running deterministic nested training",
            model_name,
            horizon,
        )
        try:
            return _regenerate_trained_bundle(
                model_name, horizon, device, max_epochs
            ), None
        except Exception as exc:
            return None, f"trained regeneration failed: {exc}"

    return (
        None,
        "missing sample-level predictions and no checkpoint-free inference path",
    )


def _align_indices(
    bundles: list[dict], split: str
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    index_key = f"{split}_indices"
    pred_key = f"{split}_predictions"
    target_key = f"{split}_targets"

    common = bundles[0][index_key]
    for bundle in bundles[1:]:
        common = np.intersect1d(common, bundle[index_key], assume_unique=False)
    common = np.asarray(common, dtype=np.int64)
    if len(common) == 0:
        raise ValueError(f"No shared {split} sample indices across selected models")

    aligned_predictions = []
    reference_targets = None
    for bundle in bundles:
        positions = {
            int(idx): pos for pos, idx in enumerate(bundle[index_key].tolist())
        }
        order = [positions[int(idx)] for idx in common]
        preds = np.asarray(bundle[pred_key][order], dtype=np.float32)
        targets = np.asarray(bundle[target_key][order], dtype=np.float32)
        aligned_predictions.append(preds)
        if reference_targets is None:
            reference_targets = targets
        elif not np.allclose(reference_targets, targets, atol=1e-5, rtol=1e-5):
            raise ValueError(f"Target mismatch while aligning {split} predictions")

    return common, aligned_predictions, reference_targets


def _summarize_weights(
    model_names: list[str], coef: np.ndarray, horizon: int
) -> dict[str, dict[str, float]]:
    if coef.ndim == 1:
        coef = coef[None, :]
    summary = {}
    for i, model_name in enumerate(model_names):
        block = coef[:, i * horizon : (i + 1) * horizon]
        summary[model_name] = {
            "mean_abs_weight": float(np.mean(np.abs(block))),
            "mean_signed_weight": float(np.mean(block)),
            "max_abs_weight": float(np.max(np.abs(block))),
        }
    return summary


def _run_horizon(
    horizon: int,
    models: list[str],
    batch_size: int,
    device: str,
    max_epochs: int | None,
    retrain_trained_models: bool,
) -> dict:
    included = []
    skipped = []

    for model_name in models:
        logger.info("Resolving %s h%d predictions", model_name, horizon)
        bundle, reason = _resolve_model_bundle(
            model_name=model_name,
            horizon=horizon,
            batch_size=batch_size,
            device=device,
            max_epochs=max_epochs,
            retrain_trained_models=retrain_trained_models,
        )
        if bundle is None:
            skipped.append({"model": model_name, "reason": reason})
            logger.warning("Skipping %s h%d: %s", model_name, horizon, reason)
            continue
        included.append(bundle)

    if len(included) < 2:
        raise RuntimeError(f"Need at least 2 models for stacking; got {len(included)}")

    train_indices, train_pred_blocks, y_train = _align_indices(included, "train")
    test_indices, test_pred_blocks, y_test = _align_indices(included, "test")
    x_train = np.concatenate(train_pred_blocks, axis=1)
    x_test = np.concatenate(test_pred_blocks, axis=1)

    stacker = LinearRegression()
    stacker.fit(x_train, y_train)
    y_pred = stacker.predict(x_test)
    test_metrics = MetricRegistry.compute("forecasting", y_pred, y_test)

    model_names = [bundle["model"] for bundle in included]
    result = {
        "ensemble_type": "linear_regression_stacking",
        "dataset": "ganymede",
        "horizon": horizon,
        "models_used": model_names,
        "model_sources": {bundle["model"]: bundle["source"] for bundle in included},
        "skipped_models": skipped,
        "stack_train_samples": int(len(train_indices)),
        "stack_test_samples": int(len(test_indices)),
        "test_indices": test_indices,
        "test_predictions": y_pred,
        "test_targets": y_test,
        "test_metrics": test_metrics,
        "stacker": {
            "intercept": stacker.intercept_,
            "coef": stacker.coef_,
            "per_model_weight_summary": _summarize_weights(
                model_names, stacker.coef_, horizon
            ),
        },
    }

    out_path = OUTPUT_DIR / f"ganymede_h{horizon}_multi_well.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_make_serializable(result), indent=2))
    logger.info("Saved ensemble result to %s", out_path)
    logger.info(
        "h%d ensemble: r2_prod=%.4f mae=%.4f rmse=%.4f",
        horizon,
        test_metrics.get("r2_prod", float("nan")),
        test_metrics.get("mae", float("nan")),
        test_metrics.get("rmse", float("nan")),
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--retrain-trained-models",
        action="store_true",
        help="Regenerate trained-model predictions by re-running nested training when sample-level predictions are unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for horizon in args.horizons:
        _run_horizon(
            horizon=horizon,
            models=args.models,
            batch_size=args.batch_size,
            device=args.device,
            max_epochs=args.max_epochs,
            retrain_trained_models=args.retrain_trained_models,
        )


if __name__ == "__main__":
    main()
