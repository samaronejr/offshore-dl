"""Forecasting-result audit utilities for MASE-first reporting.

The audit is intentionally protocol-preserving: it reads existing result JSONs,
classifies metric provenance, and writes evidence artifacts before any model or
experiment protocol changes are considered.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

DATASETS = ("ganymede", "spe_berg", "inner_mongolia", "volve")
HORIZONS = (7, 14, 30, 90)
METRICS = ("mae", "rmse", "r2", "r2_prod", "mase")
METRIC_METADATA = (
    "mase_denominator_source",
    "mase_naive_mae",
    "mase_scale_n",
    "mase_valid",
    "mase_warning",
    "mase_denominator_sample_size",
    "mase_flat",
    "mase_group_macro",
    "mase_group_weighted",
    "mase_aggregation",
)
TRAINED_MODELS = frozenset({"lstm", "deeponet", "patchtst", "tcn"})
FM_MODELS = frozenset({"chronos", "timesfm", "tirex"})
AD_HOC_MODELS = frozenset({"chronos_finetuned", "ensemble_stack"})
DEFAULT_FORECAST_MODELS = tuple(sorted(TRAINED_MODELS | FM_MODELS))

RESULT_FILENAME_RE = re.compile(
    r"^(?P<dataset>" + "|".join(DATASETS) + r")_h(?P<horizon>\d+)_"
    r"(?P<mode>multi_well|per_well)(?:_(?P<well>.+))?\.json$"
)

CONFIG_PATHS = {
    "ganymede": Path("configs/data/ganymede.yaml"),
    "spe_berg": Path("configs/data/spe_berg.yaml"),
    "inner_mongolia": Path("configs/data/inner_mongolia.yaml"),
    "volve": Path("configs/data/volve.yaml"),
}

AD_HOC_PROVENANCE_ROWS = (
    {
        "model": "chronos_finetuned",
        "dataset": "ganymede",
        "horizon": np.nan,
        "mode": "multi_well",
        "well": "",
        "fold_or_split": "writer_path",
        "row_type": "writer_provenance",
        "denominator_source": "ad_hoc_validation_fallback",
        "suspected_failure_mode": "ad_hoc_writer_no_y_train;eval_target_fallback;missing_current_artifact",
        "classification": "path_audited_no_current_artifact",
        "evidence_path": "scripts/run_ganymede_chronos_finetune.py:35,337,367-369",
    },
    {
        "model": "ensemble_stack",
        "dataset": "ganymede",
        "horizon": np.nan,
        "mode": "multi_well",
        "well": "",
        "fold_or_split": "writer_path",
        "row_type": "writer_provenance",
        "denominator_source": "ad_hoc_validation_fallback",
        "suspected_failure_mode": "ad_hoc_writer_no_y_train;eval_target_fallback;missing_current_artifact",
        "classification": "path_audited_no_current_artifact",
        "evidence_path": "scripts/run_ganymede_ensemble.py:58-59,351,376-378",
    },
)


class RawDenominatorComputer:
    """Compute protocol-matching raw-training MASE denominators on demand.

    The current stored metrics do not retain denominator provenance. This helper
    reconstructs the grouped temporal holdout and inner-CV train indices used by
    the production scripts, then computes the seasonal-naive MAE from raw target
    values without applying a second target denormalization.
    """

    def __init__(self, seasonal_period: int = 7) -> None:
        self.seasonal_period = seasonal_period
        self._dataset_cache: dict[tuple[str, int, str, str], Any] = {}
        self._denominator_cache: dict[tuple[str, int, str, str, str, int | None], float] = {}
        self.errors: dict[tuple[str, int, str, str, str, int | None], str] = {}
        self._well_maps: dict[str, dict[str, str]] = {}

    def naive_mae(
        self,
        dataset_name: str,
        horizon: int,
        mode: str,
        well: str,
        fold_or_split: str,
    ) -> float:
        """Return raw-training seasonal-naive MAE for a test or CV split."""
        fold_idx = _parse_fold_idx(fold_or_split)
        split_kind = "cv_fold" if fold_idx is not None else "test"
        cache_key = (dataset_name, int(horizon), mode, well or "", split_kind, fold_idx)
        if cache_key in self._denominator_cache:
            return self._denominator_cache[cache_key]

        try:
            dataset = self._dataset(dataset_name, int(horizon), mode, well or "")
            train_indices = self._train_indices(dataset, split_kind, fold_idx)
            values = _collect_targets(dataset, train_indices)
            naive_mae = seasonal_naive_mae(values, self.seasonal_period)
            self._denominator_cache[cache_key] = naive_mae
            return naive_mae
        except Exception as exc:  # pragma: no cover - exercised by artifact runs.
            self.errors[cache_key] = f"{type(exc).__name__}: {exc}"
            raise

    def _dataset(self, dataset_name: str, horizon: int, mode: str, well: str):
        restored_well = self.restore_well(dataset_name, well) if well else ""
        key = (dataset_name, horizon, mode, restored_well)
        if key in self._dataset_cache:
            return self._dataset_cache[key]

        from offshore_dl.data.datasets import (  # noqa: PLC0415 - lazy heavy import.
            GanymedeDataset,
            InnerMongoliaDataset,
            SPEBergDataset,
            VolveDataset,
        )

        cls_by_dataset = {
            "ganymede": GanymedeDataset,
            "spe_berg": SPEBergDataset,
            "inner_mongolia": InnerMongoliaDataset,
            "volve": VolveDataset,
        }
        cls = cls_by_dataset[dataset_name]
        kwargs: dict[str, Any] = {
            "horizon": horizon,
            "mode": mode,
            "filter_shutdowns": False,
        }
        if mode == "per_well" and restored_well:
            kwargs["well_name"] = restored_well
        dataset = cls(CONFIG_PATHS[dataset_name], **kwargs)
        self._dataset_cache[key] = dataset
        return dataset

    def restore_well(self, dataset_name: str, safe_well_name: str) -> str:
        """Map result-file well names back to config well names when needed."""
        if not safe_well_name:
            return ""
        if dataset_name not in self._well_maps:
            cfg = OmegaConf.load(CONFIG_PATHS[dataset_name])
            wells = list(cfg.data.get("wells", []))
            self._well_maps[dataset_name] = {_safe_well(str(w)): str(w) for w in wells}
        return self._well_maps[dataset_name].get(safe_well_name, safe_well_name)

    @staticmethod
    def _train_indices(dataset, split_kind: str, fold_idx: int | None) -> np.ndarray:
        from offshore_dl.evaluation.cv import (  # noqa: PLC0415 - lazy import.
            GroupedExpandingWindowCV,
            GroupedTemporalHoldoutSplitter,
        )

        groups = np.asarray([well_idx for well_idx, _ in dataset._samples], dtype=np.int32)
        holdout = GroupedTemporalHoldoutSplitter(test_ratio=0.2, groups=groups)
        train_pool, _test_indices = holdout.split(len(dataset))
        if split_kind == "test":
            return np.asarray(train_pool, dtype=np.int64)

        if fold_idx is None:
            msg = "fold_idx is required for cv_fold denominator reconstruction"
            raise ValueError(msg)
        cv = GroupedExpandingWindowCV(
            groups=groups[train_pool],
            n_splits=3,
            min_train_ratio=0.5,
        )
        splits = cv.get_splits(len(train_pool))
        if fold_idx >= len(splits):
            msg = f"Requested fold {fold_idx}, but only {len(splits)} folds exist"
            raise IndexError(msg)
        local_train, _local_val = splits[fold_idx]
        return np.asarray(train_pool[local_train], dtype=np.int64)


def _safe_well(name: str) -> str:
    return str(name).replace("/", "_")


def _parse_fold_idx(fold_or_split: str) -> int | None:
    match = re.match(r"cv_fold_(\d+)$", str(fold_or_split))
    return int(match.group(1)) if match else None


def _collect_targets(dataset, indices: np.ndarray) -> np.ndarray:
    targets = []
    for idx in np.asarray(indices, dtype=np.int64):
        _features, target, _metadata = dataset[int(idx)]
        targets.append(np.asarray(target, dtype=np.float64).ravel())
    if not targets:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(targets).astype(np.float64, copy=False)


def seasonal_naive_mae(values: np.ndarray, seasonal_period: int = 7) -> float:
    """Compute the MASE denominator used by ``MetricRegistry``."""
    scale_data = np.asarray(values, dtype=np.float64).ravel()
    if len(scale_data) <= seasonal_period:
        return float("inf")
    return float(np.mean(np.abs(scale_data[seasonal_period:] - scale_data[:-seasonal_period])))


def parse_result_filename(path: Path) -> dict[str, Any] | None:
    match = RESULT_FILENAME_RE.match(path.name)
    if not match:
        return None
    return {
        "dataset": match.group("dataset"),
        "horizon": int(match.group("horizon")),
        "mode": match.group("mode"),
        "well": match.group("well") or "",
    }


def load_expected_scenarios(config_dir: Path = Path("configs/data")) -> pd.DataFrame:
    """Build the dataset × horizon × mode × well scenario grid from configs."""
    rows: list[dict[str, Any]] = []
    for dataset_name in DATASETS:
        cfg_path = config_dir / f"{dataset_name}.yaml"
        cfg = OmegaConf.load(cfg_path)
        horizons = [int(h) for h in cfg.data.forecasting.get("horizons", HORIZONS)]
        modes = [str(m) for m in cfg.data.get("modes", ["multi_well"])]
        wells = [_safe_well(str(w)) for w in cfg.data.get("wells", [])]
        for horizon in horizons:
            for mode in modes:
                if mode == "per_well":
                    for well in wells:
                        rows.append(
                            {
                                "dataset": dataset_name,
                                "horizon": horizon,
                                "mode": mode,
                                "well": well,
                            }
                        )
                else:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "horizon": horizon,
                            "mode": mode,
                            "well": "",
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates().sort_values(
        ["dataset", "horizon", "mode", "well"]
    )


def _flat_metrics(payload: Any) -> tuple[dict[str, Any], bool]:
    """Return flat numeric test metrics and whether the payload is malformed."""
    if not isinstance(payload, dict):
        return {}, payload is not None
    metrics: dict[str, Any] = {}
    malformed = False
    for metric in METRICS:
        value = payload.get(metric, np.nan)
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            metrics[metric] = float(value)
        else:
            metrics[metric] = np.nan
            if metric in payload:
                malformed = True
    for key in METRIC_METADATA:
        metrics[key] = payload.get(key, "")
    if "metrics" in payload:
        malformed = True
    return metrics, malformed


def _contains_model_result_jsons(root: Path) -> bool:
    """Return whether ``root`` looks like a result root with model subdirs."""
    if not root.exists() or not root.is_dir():
        return False
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        if any(path.is_file() and parse_result_filename(path) for path in model_dir.iterdir()):
            return True
    return False


def resolve_results_dir(results_dir: Path | None = None) -> Path:
    """Resolve the result root for the current prefix/postfix layout.

    Historical artifacts live under ``results/pre_fix`` and repaired reruns
    should land under ``results/post_fix``.  When no explicit root is supplied,
    prefer non-empty repaired outputs, then historical pre-fix outputs, then the
    legacy top-level ``results`` root.
    """
    if results_dir is not None:
        root = Path(results_dir)
        if _contains_model_result_jsons(root):
            return root
        if root.name == "results":
            post_fix = root / "post_fix"
            pre_fix = root / "pre_fix"
            if _contains_model_result_jsons(post_fix):
                return post_fix
            if _contains_model_result_jsons(pre_fix):
                return pre_fix
        return root

    root = Path("results")
    post_fix = root / "post_fix"
    pre_fix = root / "pre_fix"
    if _contains_model_result_jsons(post_fix):
        return post_fix
    if _contains_model_result_jsons(pre_fix):
        return pre_fix
    return root


def collect_result_rows(results_dir: Path | None = None) -> pd.DataFrame:
    """Collect aggregate-ingestable forecasting result rows from JSON files."""
    results_dir = resolve_results_dir(results_dir)
    rows: list[dict[str, Any]] = []
    if not results_dir.exists():
        return pd.DataFrame()

    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        for path in sorted(model_dir.iterdir()):
            parsed = parse_result_filename(path)
            if parsed is None:
                continue
            row = {"model": model, **parsed, "file_path": str(path)}
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                row.update(
                    {
                        "status": "malformed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                for metric in METRICS:
                    row[metric] = np.nan
                rows.append(row)
                continue

            raw_status = str(data.get("status", "ok"))
            metrics, malformed_metrics = _flat_metrics(data.get("test_metrics", {}))
            status = raw_status
            if raw_status == "ok" and malformed_metrics:
                status = "malformed"
            elif raw_status == "ok" and not any(np.isfinite(metrics[m]) for m in METRICS):
                status = "partial"

            row.update(
                {
                    "status": status,
                    "raw_status": raw_status,
                    "reason": data.get("reason", ""),
                    "n_train": data.get("n_train", np.nan),
                    "n_test": data.get("n_test", np.nan),
                    "n_cv_folds": data.get("n_cv_folds", np.nan),
                    "has_test_predictions": "test_predictions" in data,
                    "has_test_targets": "test_targets" in data,
                    "cv_fold_count": len(data.get("cv_fold_results", []) or []),
                }
            )
            row.update(metrics)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "mode", "well", "model"])


def build_coverage_audit(
    results_dir: Path | None = None,
    summary_csv: Path = Path("results/forecasting_summary.csv"),
) -> pd.DataFrame:
    """Classify expected forecasting artifacts as ok/unavailable/missing/etc."""
    results_dir = resolve_results_dir(results_dir)
    scenarios = load_expected_scenarios()
    actual = collect_result_rows(results_dir)
    discovered_models = set(actual["model"].dropna().astype(str)) if not actual.empty else set()
    expected_models = sorted(set(DEFAULT_FORECAST_MODELS) | discovered_models)

    expected_rows = []
    for model in expected_models:
        for scenario in scenarios.to_dict("records"):
            expected_rows.append({"model": model, **scenario, "expected": True})
    expected = pd.DataFrame(expected_rows)

    merge_keys = ["model", "dataset", "horizon", "mode", "well"]
    coverage = expected.merge(actual, on=merge_keys, how="outer", suffixes=("", "_actual"))
    coverage["expected"] = coverage["expected"].fillna(False).astype(bool)
    coverage["status"] = coverage["status"].fillna("missing")
    coverage["file_path"] = coverage.get("file_path", pd.Series(dtype=str)).fillna("")

    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        summary["well"] = summary["well"].fillna("").astype(str)
        summary = summary[[*merge_keys, "status"]].rename(columns={"status": "summary_status"})
        coverage = coverage.merge(summary, on=merge_keys, how="left")
        coverage["in_forecasting_summary_csv"] = coverage["summary_status"].notna()
    else:
        coverage["summary_status"] = ""
        coverage["in_forecasting_summary_csv"] = False

    status_order = {
        "ok": 0,
        "unavailable": 1,
        "skipped": 2,
        "partial": 3,
        "malformed": 4,
        "missing": 5,
    }
    coverage["status_sort"] = coverage["status"].map(status_order).fillna(9).astype(int)
    columns = [
        "model",
        "dataset",
        "horizon",
        "mode",
        "well",
        "expected",
        "status",
        "raw_status",
        "summary_status",
        "in_forecasting_summary_csv",
        "file_path",
        "reason",
        "n_train",
        "n_test",
        "n_cv_folds",
        "has_test_predictions",
        "has_test_targets",
        "cv_fold_count",
        *METRICS,
    ]
    for column in columns:
        if column not in coverage.columns:
            coverage[column] = np.nan if column in METRICS else ""
    return coverage[columns].sort_values(["dataset", "horizon", "mode", "well", "model"])


def denominator_source_for(
    model: str,
    status: str,
    stored_source: str | None = None,
) -> str:
    if status not in {"ok", "partial", "malformed"}:
        return "unavailable"
    if stored_source in {"raw_train", "y_train", "train_flat", "grouped_train"}:
        return "raw_train"
    if stored_source in {"evaluation_targets_fallback", "eval_flat", "grouped_eval"}:
        if model in AD_HOC_MODELS:
            return "ad_hoc_validation_fallback"
        return "validation_target_fallback"
    if stored_source in {
        "missing_order",
        "missing_group",
        "multi_group_flat_unavailable",
        "unavailable",
    }:
        return "unavailable"
    if model in TRAINED_MODELS:
        return "double_denorm_suspect"
    if model in FM_MODELS:
        return "validation_target_fallback"
    if model in AD_HOC_MODELS:
        return "ad_hoc_validation_fallback"
    return "unknown"


def select_mase_audit_candidates(result_rows: pd.DataFrame, worst_n: int = 25) -> pd.DataFrame:
    """Select ranking-critical rows for Gate 2 denominator auditing."""
    if result_rows.empty:
        return result_rows.assign(row_type=pd.Series(dtype=str))

    selected: dict[tuple[Any, ...], set[str]] = {}
    key_cols = ["model", "dataset", "horizon", "mode", "well"]

    def mark(frame: pd.DataFrame, reason: str) -> None:
        for row in frame[key_cols].itertuples(index=False, name=None):
            selected.setdefault(row, set()).add(reason)

    ok = result_rows[result_rows["status"].eq("ok")].copy()
    finite = ok[np.isfinite(ok["mase"].astype(float))]
    if not finite.empty:
        mark(finite.sort_values("mase", ascending=False).head(worst_n), "worst_finite_mase")

    zero_nonzero = ok[
        ok["mase"].fillna(np.nan).eq(0)
        & ok["mae"].notna()
        & ok["mae"].astype(float).gt(1e-12)
    ]
    mark(zero_nonzero, "zero_mase_nonzero_mae")

    z06 = result_rows[
        result_rows["dataset"].eq("ganymede")
        & result_rows["well"].astype(str).str.contains("Z06", na=False)
    ]
    mark(z06, "ganymede_z06")

    controls = []
    for _dataset, group in finite[(finite["mase"] > 0) & finite["mase"].lt(100)].groupby("dataset"):
        median = group["mase"].median()
        controls.append(group.iloc[(group["mase"] - median).abs().argsort()[:3]])
    if controls:
        mark(pd.concat(controls), "normal_control")

    unavailable = result_rows[~result_rows["status"].eq("ok")]
    mark(unavailable, "unavailable_or_non_ok")

    rows = []
    indexed = result_rows.set_index(key_cols, drop=False)
    for key, reasons in selected.items():
        if key not in indexed.index:
            continue
        hit = indexed.loc[key]
        if isinstance(hit, pd.DataFrame):
            hit = hit.iloc[0]
        row = hit.to_dict()
        row["row_type"] = ";".join(sorted(reasons))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "mode", "well", "model"])


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, np.nan)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    return float("nan")


def _stored_naive_from_metrics(mae: float, mase: float) -> float:
    if not np.isfinite(mae) or not np.isfinite(mase):
        return float("nan")
    if mase > 1e-12:
        return float(mae / mase)
    if abs(mae) <= 1e-12 and mase == 0:
        return 0.0
    return 0.0


def _failure_flags(
    denominator_source: str,
    mae: float,
    stored_mase: float,
    inferred_naive_mae: float,
    recompute_error: str = "",
) -> list[str]:
    flags: list[str] = []
    if denominator_source == "double_denorm_suspect":
        flags.extend(
            [
                "double_denormalized_raw_train",
                "random_subsample_denominator",
                "non_chronological_flattened_scale",
            ]
        )
    elif denominator_source == "validation_target_fallback":
        flags.extend(["eval_target_fallback", "non_chronological_flattened_scale"])
    elif denominator_source == "ad_hoc_validation_fallback":
        flags.extend(["ad_hoc_writer_no_y_train", "eval_target_fallback"])
    elif denominator_source == "unknown":
        flags.append("unknown_writer")
    elif denominator_source == "unavailable":
        flags.append("unavailable_result")

    if np.isfinite(mae) and np.isfinite(stored_mase) and stored_mase == 0 and mae > 1e-12:
        flags.append("zero_mase_nonzero_mae")
    if np.isfinite(inferred_naive_mae) and 0 < inferred_naive_mae < 1e-4:
        flags.append("tiny_inferred_naive_mae")
    if math.isinf(stored_mase):
        flags.append("inf_mase_short_scale")
    if any(not np.isfinite(v) for v in (mae, stored_mase)) and not math.isinf(stored_mase):
        flags.append("nonfinite_metric")
    if recompute_error:
        flags.append("missing_recompute_artifacts")
    return sorted(set(flags))


def _classification(
    denominator_source: str,
    flags: list[str],
    recomputed_mase: float,
) -> str:
    if denominator_source == "unavailable":
        return "unavailable"
    if denominator_source == "unknown":
        return "unknown_blocker"
    severe = {"zero_mase_nonzero_mae", "inf_mase_short_scale", "nonfinite_metric"}
    if severe & set(flags):
        return "invalid_stored_recomputable" if np.isfinite(recomputed_mase) else "invalid_stored"
    if denominator_source in {
        "double_denorm_suspect",
        "validation_target_fallback",
        "ad_hoc_validation_fallback",
    }:
        return "suspect_stored_recomputable" if np.isfinite(recomputed_mase) else "suspect_stored"
    if denominator_source == "raw_train":
        return "valid_stored"
    return "valid_recomputed" if np.isfinite(recomputed_mase) else "unknown_blocker"


def audit_mase_denominators(
    result_rows: pd.DataFrame,
    denominator_computer: RawDenominatorComputer | None = None,
    recompute_raw: bool = True,
) -> pd.DataFrame:
    """Create the Gate 2 MASE denominator audit table."""
    denominator_computer = denominator_computer or RawDenominatorComputer()
    candidates = select_mase_audit_candidates(result_rows)
    records: list[dict[str, Any]] = []

    for candidate in candidates.to_dict("records"):
        path = Path(str(candidate.get("file_path", "")))
        model = str(candidate.get("model", ""))
        status = str(candidate.get("status", "missing"))
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
        except (json.JSONDecodeError, OSError):
            data = {}

        metric_payloads: list[tuple[str, dict[str, Any]]] = []
        if isinstance(data.get("test_metrics"), dict):
            metric_payloads.append(("test", data.get("test_metrics", {})))
        elif status == "ok":
            metric_payloads.append(("test", {}))

        for fold in data.get("cv_fold_results", []) or []:
            if not isinstance(fold, dict):
                continue
            fold_idx = fold.get("fold_idx", len(metric_payloads))
            metrics = fold.get("metrics", {})
            if isinstance(metrics, dict):
                metric_payloads.append((f"cv_fold_{fold_idx}", metrics))

        if not metric_payloads:
            metric_payloads.append(("test", {}))

        for fold_or_split, metrics in metric_payloads:
            stored_source = str(
                metrics.get(
                    "mase_denominator_source",
                    candidate.get("mase_denominator_source", ""),
                )
                or ""
            )
            source = denominator_source_for(model, status, stored_source)
            mae = _metric_value(metrics, "mae")
            rmse = _metric_value(metrics, "rmse")
            stored_mase = _metric_value(metrics, "mase")
            inferred_naive_mae = _stored_naive_from_metrics(mae, stored_mase)
            raw_naive_mae = float("nan")
            recomputed_mase = float("nan")
            recompute_error = ""
            if recompute_raw and status == "ok" and np.isfinite(mae):
                try:
                    raw_naive_mae = denominator_computer.naive_mae(
                        str(candidate["dataset"]),
                        int(candidate["horizon"]),
                        str(candidate["mode"]),
                        str(candidate.get("well", "") or ""),
                        fold_or_split,
                    )
                    if raw_naive_mae > 1e-12 and np.isfinite(raw_naive_mae):
                        recomputed_mase = float(mae / raw_naive_mae)
                    elif abs(mae) <= 1e-12:
                        recomputed_mase = 0.0
                    else:
                        recomputed_mase = float("inf")
                except Exception as exc:  # pragma: no cover - depends on local data.
                    recompute_error = f"{type(exc).__name__}: {exc}"

            flags = _failure_flags(
                source,
                mae,
                stored_mase,
                inferred_naive_mae,
                recompute_error,
            )
            classification = _classification(source, flags, recomputed_mase)
            delta = (
                recomputed_mase - stored_mase
                if np.isfinite(recomputed_mase) and np.isfinite(stored_mase)
                else np.nan
            )
            records.append(
                {
                    "dataset": candidate.get("dataset"),
                    "model": model,
                    "horizon": candidate.get("horizon"),
                    "mode": candidate.get("mode"),
                    "well_or_group": candidate.get("well", "") or "multi_well",
                    "fold_or_split": fold_or_split,
                    "row_type": candidate.get("row_type", "selected"),
                    "mae": mae,
                    "rmse": rmse,
                    "naive_mae": raw_naive_mae,
                    "stored_naive_mae_inferred": inferred_naive_mae,
                    "stored_mase": stored_mase,
                    "recomputed_mase": recomputed_mase,
                    "delta": delta,
                    "denominator_source": source,
                    "suspected_failure_mode": ";".join(flags),
                    "classification": classification,
                    "ranking_delta_if_repaired": np.nan,
                    "evidence_path": candidate.get("file_path", ""),
                    "recompute_error": recompute_error,
                }
            )

    for row in AD_HOC_PROVENANCE_ROWS:
        complete = {
            "mae": np.nan,
            "rmse": np.nan,
            "naive_mae": np.nan,
            "stored_naive_mae_inferred": np.nan,
            "stored_mase": np.nan,
            "recomputed_mase": np.nan,
            "delta": np.nan,
            "ranking_delta_if_repaired": np.nan,
            "recompute_error": "",
            **row,
        }
        complete["well_or_group"] = complete.pop("well") or "multi_well"
        records.append(complete)

    columns = [
        "dataset",
        "model",
        "horizon",
        "mode",
        "well_or_group",
        "fold_or_split",
        "row_type",
        "mae",
        "rmse",
        "naive_mae",
        "stored_naive_mae_inferred",
        "stored_mase",
        "recomputed_mase",
        "delta",
        "denominator_source",
        "suspected_failure_mode",
        "classification",
        "ranking_delta_if_repaired",
        "evidence_path",
        "recompute_error",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        ["dataset", "horizon", "mode", "well_or_group", "model", "fold_or_split"],
        na_position="last",
    )


def add_recomputed_test_mase(
    result_rows: pd.DataFrame,
    denominator_computer: RawDenominatorComputer,
) -> pd.DataFrame:
    """Add raw-train MASE columns to final-test result rows where possible."""
    rows = []
    for row in result_rows.to_dict("records"):
        row = dict(row)
        mae = float(row.get("mae", np.nan)) if row.get("mae", np.nan) == row.get("mae", np.nan) else np.nan
        row["raw_train_naive_mae"] = np.nan
        row["recomputed_mase"] = np.nan
        row["effective_mase"] = row.get("mase", np.nan)
        row["effective_mase_source"] = "stored_mase"
        row["recompute_error"] = ""
        if row.get("status") == "ok" and np.isfinite(mae):
            try:
                naive = denominator_computer.naive_mae(
                    str(row["dataset"]),
                    int(row["horizon"]),
                    str(row["mode"]),
                    str(row.get("well", "") or ""),
                    "test",
                )
                row["raw_train_naive_mae"] = naive
                if naive > 1e-12 and np.isfinite(naive):
                    row["recomputed_mase"] = float(mae / naive)
                elif abs(mae) <= 1e-12:
                    row["recomputed_mase"] = 0.0
                else:
                    row["recomputed_mase"] = float("inf")
                row["effective_mase"] = row["recomputed_mase"]
                row["effective_mase_source"] = "raw_train_recomputed"
            except Exception as exc:  # pragma: no cover - depends on local data.
                row["recompute_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def compute_weighted_mase_summary(
    result_rows: pd.DataFrame,
    metric_col: str = "effective_mase",
) -> pd.DataFrame:
    """Compute MASE summary with per-well rows collapsed inside scenarios.

    Primary scenario key is dataset × horizon × mode. For ``per_well`` mode,
    each model receives one scenario value: the mean MASE across wells available
    for that model in that scenario. Cross-scenario summaries therefore do not
    let datasets with many wells dominate the aggregate.
    """
    if result_rows.empty:
        return pd.DataFrame(
            columns=["model", "scenario_count", "mean_mase", "median_mase", "mean_rank"]
        )

    work = result_rows.copy()
    work = work[work["status"].eq("ok")]
    work = work[np.isfinite(work[metric_col].astype(float))]
    if work.empty:
        return pd.DataFrame(
            columns=["model", "scenario_count", "mean_mase", "median_mase", "mean_rank"]
        )

    scenario_values = (
        work.groupby(["dataset", "horizon", "mode", "model"], dropna=False)[metric_col]
        .mean()
        .reset_index(name="scenario_mase")
    )

    ranks = []
    for scenario, group in scenario_values.groupby(["dataset", "horizon", "mode"]):
        valid = group[np.isfinite(group["scenario_mase"].astype(float))].copy()
        if len(valid) < 2:
            continue
        valid["scenario_rank"] = valid["scenario_mase"].rank(ascending=True, method="average")
        valid["scenario_key"] = "|".join(map(str, scenario))
        ranks.append(valid)
    ranked = pd.concat(ranks, ignore_index=True) if ranks else scenario_values.assign(
        scenario_rank=np.nan,
        scenario_key="",
    )

    summary = (
        ranked.groupby("model")
        .agg(
            scenario_count=("scenario_mase", "count"),
            mean_mase=("scenario_mase", "mean"),
            median_mase=("scenario_mase", "median"),
            mean_rank=("scenario_rank", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["mean_rank", "mean_mase", "model"], na_position="last")


def scenario_level_mase(result_rows: pd.DataFrame, metric_col: str = "effective_mase") -> pd.DataFrame:
    work = result_rows[result_rows["status"].eq("ok")].copy()
    work = work[np.isfinite(work[metric_col].astype(float))]
    if work.empty:
        return pd.DataFrame()
    return (
        work.groupby(["dataset", "horizon", "mode", "model"], dropna=False)[metric_col]
        .mean()
        .reset_index(name="scenario_mase")
        .sort_values(["dataset", "horizon", "mode", "model"])
    )


def _markdown_table(df: pd.DataFrame, max_rows: int = 20, float_digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.select_dtypes(include=["float"]).columns:
        view[col] = view[col].map(
            lambda x: "" if pd.isna(x) else ("inf" if math.isinf(x) else f"{x:.{float_digits}g}")
        )
    text = view.fillna("").astype(str)
    headers = list(text.columns)
    rows = text.values.tolist()
    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def write_coverage_report(coverage: pd.DataFrame, out_path: Path) -> None:
    status_counts = coverage["status"].value_counts(dropna=False).reset_index()
    status_counts.columns = ["status", "count"]
    grouped = (
        coverage.groupby(["dataset", "status"]).size().reset_index(name="count")
        .sort_values(["dataset", "status"])
    )
    unavailable = coverage[coverage["status"].eq("unavailable")]
    unavailable_sample = unavailable[
        ["model", "dataset", "horizon", "mode", "well", "reason"]
    ].drop_duplicates()

    content = f"""# Forecasting Coverage Audit

## Global status distribution

{_markdown_table(status_counts, max_rows=50)}

## Status by dataset

{_markdown_table(grouped, max_rows=100)}

## Unavailable reasons (sample)

    {_markdown_table(unavailable_sample, max_rows=30)}

## Notes

- Expected grid is built from the four forecasting data configs and the discovered/default forecasting model set.
- `missing` means a model × dataset × horizon × mode × well/scenario artifact is expected by config but no aggregate-ingestable JSON currently exists.
- `unavailable` means an artifact exists but records dependency/runtime unavailability rather than metrics.
"""
    out_path.write_text(content)


def write_mase_table_proposal(
    original_summary: pd.DataFrame,
    effective_summary: pd.DataFrame,
    audited_rows: pd.DataFrame,
    out_path: Path,
) -> None:
    deltas = audited_rows[np.isfinite(audited_rows["delta"].astype(float))].copy()
    if not deltas.empty:
        deltas["abs_delta"] = deltas["delta"].abs()
        deltas = deltas.sort_values("abs_delta", ascending=False)
    largest_delta_cols = [
        "model",
        "dataset",
        "horizon",
        "mode",
        "well_or_group",
        "fold_or_split",
        "stored_mase",
        "recomputed_mase",
        "delta",
        "denominator_source",
        "classification",
    ]

    content = f"""# MASE-First Table Proposal

## Policy

- Primary metric: MASE.
- Primary scenario key: dataset × horizon × mode.
- Per-well rows are averaged inside each dataset × horizon × mode scenario before cross-scenario aggregation.
- Stored MASE and recomputed raw-train MASE are both retained; `effective_mase` uses recomputed raw-train denominators where available.

## Cross-scenario summary using stored MASE

{_markdown_table(original_summary, max_rows=30)}

## Cross-scenario summary using audited/effective MASE

{_markdown_table(effective_summary, max_rows=30)}

## Largest audited MASE changes

    {_markdown_table(deltas[largest_delta_cols], max_rows=25)}

## Draft table outputs

- `forecasting_mase_first_raw_wide_original.csv`
- `forecasting_mase_first_raw_wide_effective.csv`
- `forecasting_mase_weighted_summary_original.csv`
- `forecasting_mase_weighted_summary_effective.csv`
"""
    out_path.write_text(content)


def write_comparability_ledger(out_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "proposal": "Fix MASE denominator metadata and zero-denominator handling",
                "tier": "table-preserving",
                "rationale": "Changes metric validity without changing data splits, windows, horizons, or training protocol.",
            },
            {
                "proposal": "Recompute MASE from stored MAE plus raw-train denominator",
                "tier": "recompute-only comparable",
                "rationale": "Comparable if predictions/targets/MAE remain from the same original protocol and only the denominator is repaired.",
            },
            {
                "proposal": "Regenerate MASE-first weighted tables",
                "tier": "table-preserving",
                "rationale": "Changes reporting aggregation, not experiment protocol; original row-level metrics remain visible.",
            },
            {
                "proposal": "Rerun trained models after MASE fix only",
                "tier": "recompute-only comparable",
                "rationale": "Comparable if seeds, splits, windows, data filters, and model configs are unchanged.",
            },
            {
                "proposal": "Enable shutdown filtering",
                "tier": "protocol-changing/not directly comparable",
                "rationale": "Changes sample population; must be dual-reported beside unfiltered benchmark.",
            },
            {
                "proposal": "Change target transforms, input windows, horizons, modes, or splits",
                "tier": "protocol-changing/not directly comparable",
                "rationale": "Changes task definition or evaluation protocol; cannot be mixed into original ranking.",
            },
            {
                "proposal": "Add ensembles or new baselines",
                "tier": "protocol-changing/not directly comparable",
                "rationale": "Can be reported as an additional model family only after denominator provenance is fixed and labeled.",
            },
        ]
    )
    content = f"""# Forecasting Comparability Ledger

{_markdown_table(rows, max_rows=20)}

## Rule

Any shutdown-filtered, target-protocol-altered, split-altered, mode-altered, feature-population-altered, or horizon-altered result is dual-reported and never mixed into the original-protocol ranking.
"""
    out_path.write_text(content)


def write_pivot_recommendation(
    coverage: pd.DataFrame,
    audited_rows: pd.DataFrame,
    original_summary: pd.DataFrame,
    effective_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    status_counts = coverage["status"].value_counts(dropna=False).to_dict()
    source_counts = (
        audited_rows["denominator_source"].value_counts(dropna=False).to_dict()
    )
    class_counts = audited_rows["classification"].value_counts(dropna=False).to_dict()
    zero_nonzero = (
        audited_rows["suspected_failure_mode"]
        .fillna("")
        .str.contains("zero_mase_nonzero_mae")
        .sum()
    )
    recomputed = audited_rows[np.isfinite(audited_rows["recomputed_mase"].astype(float))]
    big_delta = recomputed[
        np.isfinite(recomputed["stored_mase"].astype(float))
        & (recomputed["delta"].abs() > 0.25)
    ]

    merged = original_summary[["model", "mean_rank"]].rename(
        columns={"mean_rank": "stored_mean_rank"}
    ).merge(
        effective_summary[["model", "mean_rank"]].rename(
            columns={"mean_rank": "effective_mean_rank"}
        ),
        on="model",
        how="outer",
    )
    merged["rank_delta"] = merged["effective_mean_rank"] - merged["stored_mean_rank"]
    merged = merged.sort_values(
        "rank_delta",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last",
    )

    content = f"""# Forecasting Pivot Recommendation

## Recommendation

Proceed next with **metric repair + MASE-first table regeneration before model/objective/HPO work**.

## Evidence

- Coverage status counts: `{status_counts}`.
- Denominator source counts in the Gate 2 audit: `{source_counts}`.
- Audit classification counts: `{class_counts}`.
- Zero-MASE/nonzero-MAE audited records: `{int(zero_nonzero)}`.
- Audited records with absolute recomputed-vs-stored MASE delta > 0.25: `{len(big_delta)}`.

## Rank sensitivity snapshot

{_markdown_table(merged, max_rows=20)}

## Ranked next steps

1. **Repair metric implementation and metadata**: make zero denominator with nonzero MAE invalid/inf instead of silently `0.0`; store denominator source, raw naive MAE, and whether `y_train` was used.
2. **Repair trained-model denominator collection**: stop denormalizing raw dataset targets a second time; preserve chronological/group provenance for MASE scale data.
3. **Regenerate MASE-first tables from repaired metrics**: use the weighted dataset × horizon × mode policy generated in this audit.
4. **Rerun only where artifacts are insufficient**: trained JSONs usually lack predictions/targets, so controlled reruns may be needed after metric repair if stored MAE + raw denominator is not accepted as sufficient.
5. **Only then start model/objective/HPO improvements**: tune against validated MASE and keep any shutdown-filter/window/protocol sensitivity dual-reported.

## Pivot lane

Default lane: `metric repair + table regeneration without broad production reruns`.

Escalate to controlled reruns only for rows where repaired MASE cannot be recomputed from trustworthy existing artifacts.
"""
    out_path.write_text(content)


def write_wide_mase(result_rows: pd.DataFrame, metric_col: str, out_path: Path) -> None:
    pivot = result_rows.pivot_table(
        index=["dataset", "horizon", "mode", "well"],
        columns="model",
        values=metric_col,
        aggfunc="first",
    )
    pivot.to_csv(out_path)


def run_audit(
    output_dir: Path = Path("reports/forecasting_performance_audit"),
    results_dir: Path | None = None,
    recompute_raw: bool = True,
) -> dict[str, Path]:
    """Run all five audit gates and return artifact paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    branch = _current_git_branch()
    resolved_results_dir = resolve_results_dir(results_dir)

    result_rows = collect_result_rows(resolved_results_dir)
    coverage = build_coverage_audit(resolved_results_dir)
    denominator_computer = RawDenominatorComputer()
    audited_rows = audit_mase_denominators(
        result_rows,
        denominator_computer=denominator_computer,
        recompute_raw=recompute_raw,
    )
    enriched = add_recomputed_test_mase(result_rows, denominator_computer) if recompute_raw else result_rows.assign(
        raw_train_naive_mae=np.nan,
        recomputed_mase=np.nan,
        effective_mase=result_rows["mase"],
        effective_mase_source="stored_mase",
    )

    original_summary = compute_weighted_mase_summary(enriched.assign(effective_mase=enriched["mase"]), "effective_mase")
    effective_summary = compute_weighted_mase_summary(enriched, "effective_mase")

    artifacts = {
        "coverage_csv": output_dir / "forecasting_coverage_audit.csv",
        "coverage_report": output_dir / "forecasting_coverage_report.md",
        "mase_denominator_csv": output_dir / "forecasting_mase_denominator_audit.csv",
        "raw_rows_csv": output_dir / "forecasting_result_rows_with_effective_mase.csv",
        "wide_original_csv": output_dir / "forecasting_mase_first_raw_wide_original.csv",
        "wide_effective_csv": output_dir / "forecasting_mase_first_raw_wide_effective.csv",
        "scenario_effective_csv": output_dir / "forecasting_mase_scenario_values_effective.csv",
        "weighted_original_csv": output_dir / "forecasting_mase_weighted_summary_original.csv",
        "weighted_effective_csv": output_dir / "forecasting_mase_weighted_summary_effective.csv",
        "table_proposal": output_dir / "forecasting_mase_first_table_proposal.md",
        "comparability_ledger": output_dir / "forecasting_comparability_ledger.md",
        "pivot_recommendation": output_dir / "forecasting_pivot_recommendation.md",
        "manifest": output_dir / "forecasting_audit_manifest.json",
    }

    coverage.to_csv(artifacts["coverage_csv"], index=False)
    write_coverage_report(coverage, artifacts["coverage_report"])
    audited_rows.to_csv(artifacts["mase_denominator_csv"], index=False)
    enriched.to_csv(artifacts["raw_rows_csv"], index=False)
    write_wide_mase(enriched.assign(original_mase=enriched["mase"]), "original_mase", artifacts["wide_original_csv"])
    write_wide_mase(enriched, "effective_mase", artifacts["wide_effective_csv"])
    scenario_level_mase(enriched, "effective_mase").to_csv(artifacts["scenario_effective_csv"], index=False)
    original_summary.to_csv(artifacts["weighted_original_csv"], index=False)
    effective_summary.to_csv(artifacts["weighted_effective_csv"], index=False)
    write_mase_table_proposal(
        original_summary,
        effective_summary,
        audited_rows,
        artifacts["table_proposal"],
    )
    write_comparability_ledger(artifacts["comparability_ledger"])
    write_pivot_recommendation(
        coverage,
        audited_rows,
        original_summary,
        effective_summary,
        artifacts["pivot_recommendation"],
    )

    manifest = {
        "branch": branch,
        "results_dir": str(resolved_results_dir),
        "output_dir": str(output_dir),
        "result_rows": int(len(result_rows)),
        "coverage_rows": int(len(coverage)),
        "mase_audit_rows": int(len(audited_rows)),
        "raw_recompute_errors": len(denominator_computer.errors),
        "artifacts": {k: str(v) for k, v in artifacts.items() if k != "manifest"},
    }
    artifacts["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return artifacts


def _current_git_branch() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run forecasting MASE-first audit gates.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Result root to audit. Defaults to non-empty results/post_fix, then "
            "results/pre_fix, then legacy results/."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/forecasting_performance_audit"))
    parser.add_argument("--skip-raw-recompute", action="store_true")
    args = parser.parse_args(argv)

    artifacts = run_audit(
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        recompute_raw=not args.skip_raw_recompute,
    )
    print("Forecasting audit artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
