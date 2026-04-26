"""Standalone LightGBM baseline for Ganymede forecasting.

Loads preprocessed per-well parquet files, adds enhanced autoregressive /
seasonal features, reproduces the benchmark's grouped temporal holdout + inner
GroupedExpandingWindowCV protocol, and writes benchmark-compatible JSON files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from sklearn.ensemble import HistGradientBoostingRegressor

from offshore_dl.evaluation.cv import (
    GroupedExpandingWindowCV,
    GroupedTemporalHoldoutSplitter,
)
from offshore_dl.evaluation.metrics import MetricRegistry
from offshore_dl.utils.config import load_merged_config
from offshore_dl.utils.reproducibility import set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/lightgbm")
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "max_depth": 7,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "n_estimators": 200,
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass
class SupervisedData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    sample_meta: list[dict]


from offshore_dl.utils.serialization import make_serializable as _make_serializable


def _safe_well(name: str) -> str:
    return name.replace("/", "_")


def _load_well_frames(processed_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    wells: list[tuple[str, pd.DataFrame]] = []
    for path in sorted(processed_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "DAYTIME" in df.columns:
                df["DAYTIME"] = pd.to_datetime(df["DAYTIME"])
                df = df.set_index("DAYTIME")
            else:
                df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        well_name = path.stem.replace("_", "/", 1)
        wells.append((well_name, df))
    if not wells:
        raise FileNotFoundError(f"No parquet files found in {processed_dir}")
    return wells


def _engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    for lag in [1, 3, 7, 14, 30]:
        df[f"target_lag_{lag}"] = df[target_col].shift(lag)

    shifted = df[target_col].shift(1)
    for window in [7, 14, 30]:
        df[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        df[f"roll_std_{window}"] = shifted.rolling(window).std()

    df["target_pct_change_7"] = df[target_col].pct_change(7)
    df["target_pct_change_30"] = df[target_col].pct_change(30)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _flatten_window(window: np.ndarray) -> np.ndarray:
    return window.reshape(-1).astype(np.float32, copy=False)


def _build_supervised_data(
    wells: list[tuple[str, pd.DataFrame]],
    *,
    target_col: str,
    input_window: int,
    horizon: int,
    gap: int,
    max_samples: int | None,
) -> SupervisedData:
    engineered = [(name, _engineer_features(df, target_col)) for name, df in wells]
    all_columns = sorted(set.union(*(set(df.columns) for _, df in engineered)))
    total_needed = input_window + gap + horizon

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    groups: list[int] = []
    sample_meta: list[dict] = []

    feature_names = [
        f"t-{input_window - 1 - step:02d}__{col}"
        for step in range(input_window)
        for col in all_columns
    ]

    for well_idx, (well_name, df) in enumerate(engineered):
        aligned = df.reindex(columns=all_columns, fill_value=0.0)
        arr = aligned.to_numpy(dtype=np.float32)
        target = aligned[target_col].to_numpy(dtype=np.float32)

        for start in range(0, max(0, len(aligned) - total_needed + 1)):
            input_end = start + input_window
            target_start = input_end + gap
            target_end = target_start + horizon

            X_rows.append(_flatten_window(arr[start:input_end]))
            y_rows.append(target[target_start:target_end].copy())
            groups.append(well_idx)
            sample_meta.append(
                {
                    "well_name": well_name,
                    "well_idx": well_idx,
                    "start_idx": start,
                    "input_end": input_end,
                    "target_start": target_start,
                    "target_end": target_end,
                }
            )

            if max_samples is not None and len(X_rows) >= max_samples:
                return SupervisedData(
                    X=np.asarray(X_rows, dtype=np.float32),
                    y=np.asarray(y_rows, dtype=np.float32),
                    groups=np.asarray(groups, dtype=np.int32),
                    feature_names=feature_names,
                    sample_meta=sample_meta,
                )

    return SupervisedData(
        X=np.asarray(X_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.float32),
        groups=np.asarray(groups, dtype=np.int32),
        feature_names=feature_names,
        sample_meta=sample_meta,
    )


def _make_regressor(params: dict):
    """Create LGBMRegressor if available, else HistGradientBoostingRegressor."""
    if LGBMRegressor is not None:
        return LGBMRegressor(**params)
    hgb_params = {
        "max_iter": params.get("n_estimators", 1000),
        "max_leaf_nodes": params.get("num_leaves", 31),
        "learning_rate": params.get("learning_rate", 0.05),
        "max_depth": params.get("max_depth", 7),
        "min_samples_leaf": params.get("min_child_samples", 20),
        "l2_regularization": params.get("reg_lambda", 0.1),
        "early_stopping": True,
        "n_iter_no_change": 50,
        "validation_fraction": 0.1,
        "random_state": 42,
    }
    return HistGradientBoostingRegressor(**hgb_params)


def _fit_multioutput_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
) -> list:
    models = []
    for step in range(y.shape[1]):
        model = _make_regressor(params)
        model.fit(X, y[:, step])
        models.append(model)
    return models


def _predict_multioutput(models: list, X: np.ndarray) -> np.ndarray:
    preds = [model.predict(X) for model in models]
    return np.stack(preds, axis=1).astype(np.float32, copy=False)


def _aggregate(fold_results: list[dict]) -> dict:
    if not fold_results:
        return {}
    agg: dict[str, float] = {}
    metric_keys = fold_results[0]["metrics"].keys()
    for key in metric_keys:
        values = [
            fr["metrics"][key] for fr in fold_results if np.isfinite(fr["metrics"][key])
        ]
        if values:
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
    return agg


def _evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    params: dict,
) -> tuple[dict, list[LGBMRegressor]]:
    models = _fit_multioutput_lgbm(X[train_idx], y[train_idx], params)
    preds = _predict_multioutput(models, X[eval_idx])
    metrics = MetricRegistry.compute(
        "forecasting",
        preds,
        y[eval_idx],
        y_train=y[train_idx],
    )
    return metrics, models


def run_horizon(
    cfg_path: str,
    horizon: int,
    max_samples: int | None,
    output_dir: Path,
) -> dict:
    cfg = load_merged_config("configs/base.yaml", cfg_path)
    processed_dir = Path(cfg.data.paths.processed)
    wells = _load_well_frames(processed_dir)
    dataset = _build_supervised_data(
        wells,
        target_col=cfg.data.target_column,
        input_window=int(cfg.data.forecasting.input_window),
        horizon=horizon,
        gap=int(cfg.data.forecasting.get("gap", 0)),
        max_samples=max_samples,
    )

    if len(dataset.X) < 2:
        raise ValueError("Not enough samples to run LightGBM baseline")

    holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=dataset.groups)
    train_pool, test_idx = holdout.split(len(dataset.X))

    cv = GroupedExpandingWindowCV(
        groups=dataset.groups[train_pool], n_splits=3, min_train_ratio=0.5
    )
    inner_splits = cv.get_splits(len(train_pool))

    cv_fold_results = []
    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
        global_train = train_pool[local_train]
        global_val = train_pool[local_val]
        metrics, _ = _evaluate_split(
            dataset.X, dataset.y, global_train, global_val, DEFAULT_PARAMS
        )
        cv_fold_results.append(
            {
                "fold_idx": fold_idx,
                "metrics": metrics,
                "n_train": len(global_train),
                "n_val": len(global_val),
            }
        )

    test_metrics, models = _evaluate_split(
        dataset.X, dataset.y, train_pool, test_idx, DEFAULT_PARAMS
    )
    try:
        feature_importance = np.mean(
            np.stack([model.feature_importances_ for model in models], axis=0),
            axis=0,
        )
        top_k = np.argsort(feature_importance)[-20:][::-1]
    except AttributeError:
        feature_importance = np.zeros(dataset.X.shape[1])
        top_k = np.arange(min(20, dataset.X.shape[1]))

    result = {
        "test_metrics": test_metrics,
        "cv_aggregate": _aggregate(cv_fold_results),
        "cv_fold_results": cv_fold_results,
        "n_train": len(train_pool),
        "n_test": len(test_idx),
        "n_cv_folds": len(inner_splits),
        "model_name": "lightgbm",
        "dataset": "ganymede",
        "mode": "multi_well",
        "horizon": horizon,
        "input_window": int(cfg.data.forecasting.input_window),
        "feature_count": int(dataset.X.shape[1]),
        "enhanced_feature_columns": [
            "target_lag_1",
            "target_lag_3",
            "target_lag_7",
            "target_lag_14",
            "target_lag_30",
            "roll_mean_7",
            "roll_std_7",
            "roll_mean_14",
            "roll_std_14",
            "roll_mean_30",
            "roll_std_30",
            "target_pct_change_7",
            "target_pct_change_30",
            "month_sin",
            "month_cos",
        ],
        "params": DEFAULT_PARAMS,
        "top_feature_importance": [
            {
                "feature": dataset.feature_names[int(idx)],
                "importance": float(feature_importance[int(idx)]),
            }
            for idx in top_k
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"ganymede_h{horizon}_multi_well.json"
    out_path.write_text(json.dumps(_make_serializable(result), indent=2))
    logger.info("Saved %s", out_path)
    logger.info("h=%d test metrics: %s", horizon, result["test_metrics"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[7, 14, 30, 90],
        help="Forecast horizons to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total supervised samples for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Accepted for compatibility; LightGBM runs on CPU in this script.",
    )
    parser.add_argument(
        "--config",
        default="configs/data/ganymede.yaml",
        help="Dataset config path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(42)
    logger.info("Running Ganymede LightGBM baseline on device=%s", args.device)
    for horizon in args.horizons:
        run_horizon(
            cfg_path=args.config,
            horizon=horizon,
            max_samples=args.max_samples,
            output_dir=RESULTS_DIR,
        )


if __name__ == "__main__":
    main()
