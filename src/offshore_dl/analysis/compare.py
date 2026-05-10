"""Comparative analysis across all models and tracks.

Loads results from ``results/{model}/{track}.json``, builds ranked
comparison tables, and runs statistical significance tests where
fold-level data is available.

Usage::

    python -m offshore_dl.analysis.compare
    python -m offshore_dl.analysis.compare --results-dir results --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─── Primary metrics per track ───
TRACK_PRIMARY_METRIC = {
    "3w": ("f1_macro", "max"),  # higher is better
    "ganymede": ("mae", "min"),  # lower is better
    "cdf": ("error_mean", "min"),  # lower is better
}

TRACK_DISPLAY_NAME = {
    "3w": "3W Classification",
    "ganymede": "Ganymede Forecasting",
    "cdf": "CDF Anomaly Detection",
}

ALL_MODELS = [
    "lstm",
    "deeponet",
    "patchtst",
    "tcn",
    "chronos",
    "timesfm",
    "tirex",
    "random_forest",
    "fkmad",
    "mambasl",
    "convtimenet",
]
ALL_TRACKS = ["3w", "ganymede", "cdf"]


def _contains_result_jsons(root: Path) -> bool:
    """Return True when ``root`` looks like a benchmark result directory."""
    if not root.exists():
        return False
    if (root / "tirex_3w_nested.json").exists():
        return True
    for model in [*ALL_MODELS, "baselines"]:
        model_dir = root / model
        if model_dir.is_dir() and any(model_dir.glob("*.json")):
            return True
    return False


def resolve_results_dir(results_dir: str | Path = "results") -> Path:
    """Resolve legacy ``results`` roots after pre/post-fix partitioning.

    Existing analysis entry points historically accepted ``results/`` directly.
    After benchmark outputs were partitioned into ``results/pre_fix`` and
    ``results/post_fix``, callers should still be able to pass the parent root.
    Prefer repaired outputs when present, otherwise fall back to the archived
    pre-fix snapshot.
    """
    root = Path(results_dir)
    if _contains_result_jsons(root):
        return root

    for child_name in ("post_fix", "pre_fix"):
        child = root / child_name
        if _contains_result_jsons(child):
            return child
    return root


def load_all_results(results_dir: str | Path = "results") -> dict[str, dict[str, dict]]:
    """Load all model results from disk.

    Returns:
        Nested dict: ``results[model][track] = result_dict``.
    """
    results_dir = resolve_results_dir(results_dir)
    all_results: dict[str, dict[str, dict]] = {}

    for model in ALL_MODELS:
        all_results[model] = {}
        for track in ALL_TRACKS:
            path = results_dir / model / f"{track}.json"
            # TiRex 3W has a non-standard result path
            if model == "tirex" and track == "3w":
                alt_path = results_dir / "tirex_3w_nested.json"
                if not path.exists() and alt_path.exists():
                    path = alt_path
            if path.exists():
                with open(path) as f:
                    all_results[model][track] = json.load(f)

    # Load naive baselines
    all_results["naive"] = {}
    baseline_map = {
        "3w": "3w_majority_baseline.json",
        "ganymede": "ganymede_seasonal_naive_baseline.json",
        "cdf": "cdf_mean_reconstruction_baseline.json",
    }
    for track, filename in baseline_map.items():
        path = results_dir / "baselines" / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Normalize baseline format to match model results
            all_results["naive"][track] = {"aggregate": data, "n_folds": 0}

    return all_results


def get_metric_value(result: dict, metric: str) -> float | None:
    """Extract a metric value from a result dict."""
    agg = result.get("aggregate", {})

    # Try aggregate format first (mean suffix)
    key = f"{metric}_mean"
    if key in agg:
        return agg[key]

    # Try direct key (baselines)
    if metric in agg:
        return agg[metric]

    # Try test_metrics (holdout test results — preferred over CV aggregate)
    test_metrics = result.get("test_metrics", {})
    if metric in test_metrics:
        return test_metrics[metric]

    # Try fold_results (cv_fold_results is the canonical key)
    fold_results = result.get("cv_fold_results", result.get("fold_results", []))
    if fold_results:
        values = []
        for fr in fold_results:
            v = fr.get("metrics", {}).get(metric)
            if v is not None:
                values.append(v)
        if values:
            return float(np.mean(values))

    return None


def get_fold_values(result: dict, metric: str) -> list[float]:
    """Extract per-fold metric values for statistical testing."""
    fold_results = result.get("cv_fold_results", result.get("fold_results", []))
    values = []
    for fr in fold_results:
        v = fr.get("metrics", {}).get(metric)
        if v is not None:
            values.append(float(v))
    return values


def build_comparison_table(all_results: dict) -> dict[str, list[dict]]:
    """Build ranked comparison tables per track.

    Returns:
        Dict mapping track name to list of model entries, sorted by primary metric.
    """
    tables: dict[str, list[dict]] = {}

    for track in ALL_TRACKS:
        metric_name, direction = TRACK_PRIMARY_METRIC[track]
        entries = []

        for model in ALL_MODELS + ["naive"]:
            if track not in all_results.get(model, {}):
                continue

            result = all_results[model][track]
            value = get_metric_value(result, metric_name)

            if value is None:
                continue

            # Collect all metrics for display
            all_metrics = {}
            agg = result.get("aggregate", {})
            for k, v in agg.items():
                if k.endswith("_mean") and isinstance(v, (int, float)):
                    clean_key = k[:-5]  # strip trailing "_mean" only
                    all_metrics[clean_key] = v
                elif isinstance(v, (int, float)) and not k.endswith("_std"):
                    all_metrics[k] = v

            test_metrics = result.get("test_metrics", {})
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)) and k not in all_metrics:
                    all_metrics[k] = v

            entries.append(
                {
                    "model": model,
                    "primary_value": value,
                    "all_metrics": all_metrics,
                    "n_folds": result.get("n_folds", 0),
                }
            )

        # Sort: min → ascending, max → descending
        reverse = direction == "max"
        entries.sort(key=lambda e: e["primary_value"], reverse=reverse)

        # Add rank
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1

        tables[track] = entries

    return tables


def run_statistical_tests(all_results: dict) -> dict[str, dict]:
    """Run Wilcoxon and Friedman tests where fold data is available.

    Returns:
        Dict with test results per track.
    """
    from scipy import stats

    test_results: dict[str, dict] = {}

    for track in ALL_TRACKS:
        metric_name, direction = TRACK_PRIMARY_METRIC[track]

        # Collect fold-level values for each model
        model_folds: dict[str, list[float]] = {}
        for model in ALL_MODELS:
            if track in all_results.get(model, {}):
                folds = get_fold_values(all_results[model][track], metric_name)
                if len(folds) >= 2:
                    model_folds[model] = folds

        if len(model_folds) < 2:
            test_results[track] = {
                "status": "insufficient_folds",
                "n_models": len(model_folds),
            }
            continue

        track_tests: dict[str, Any] = {"n_models": len(model_folds)}

        # ── Pairwise Wilcoxon signed-rank tests ──
        model_names = sorted(model_folds.keys())
        pairwise: list[dict] = []
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i + 1 :]:
                v1 = model_folds[m1]
                v2 = model_folds[m2]
                n = min(len(v1), len(v2))
                if n < 2:
                    continue
                try:
                    stat, p_value = stats.wilcoxon(v1[:n], v2[:n])
                    pairwise.append(
                        {
                            "model_a": m1,
                            "model_b": m2,
                            "statistic": float(stat),
                            "p_value": float(p_value),
                            "significant": bool(p_value < 0.05),
                        }
                    )
                except ValueError:
                    # All differences are zero
                    pairwise.append(
                        {
                            "model_a": m1,
                            "model_b": m2,
                            "statistic": 0.0,
                            "p_value": 1.0,
                            "significant": False,
                            "note": "identical values",
                        }
                    )

        # Apply Holm correction for multiple comparisons
        if pairwise:
            raw_pvals = [r["p_value"] for r in pairwise]
            try:
                from statsmodels.stats.multitest import multipletests

                _, corrected_pvals, _, _ = multipletests(raw_pvals, method="holm")
                for r, cp in zip(pairwise, corrected_pvals):
                    r["p_value_uncorrected"] = r["p_value"]
                    r["p_value"] = float(cp)
                    r["significant"] = bool(cp < 0.05)
                    r["correction"] = "holm"
            except ImportError:
                for r in pairwise:
                    r["correction"] = "none"

        track_tests["wilcoxon_pairwise"] = pairwise

        # ── Sample-size warnings ──
        n_folds = min(len(v) for v in model_folds.values())
        n_models = len(model_folds)

        # Warn if n_folds < n_models (Friedman chi-sq approximation unreliable)
        if n_folds < n_models:
            track_tests["friedman_warning"] = (
                f"n_folds={n_folds} < n_models={n_models}: "
                "chi-squared approximation may be unreliable (Iman & Davenport 1980)"
            )

        # Warn if Wilcoxon cannot reach significance
        min_wilcoxon_p = 2.0 / (2**n_folds) if n_folds > 0 else 1.0
        if min_wilcoxon_p > 0.05:
            track_tests["wilcoxon_warning"] = (
                f"With n={n_folds} paired observations, minimum achievable "
                f"Wilcoxon p-value is {min_wilcoxon_p:.4f} > 0.05"
            )

        # ── Friedman test (multi-model) ──
        if len(model_folds) >= 3:
            # Align fold counts
            min_folds = min(len(v) for v in model_folds.values())
            if min_folds >= 2:
                aligned = [v[:min_folds] for v in model_folds.values()]
                try:
                    stat, p_value = stats.friedmanchisquare(*aligned)
                    # Kendall's W (coefficient of concordance) = chi2 / (n * (k - 1))
                    kendalls_w = (
                        stat / (min_folds * (n_models - 1))
                        if min_folds > 0 and n_models > 1
                        else 0.0
                    )
                    track_tests["friedman"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                        "models": model_names,
                        "kendalls_w": float(kendalls_w),
                    }
                except ValueError:
                    track_tests["friedman"] = {"status": "insufficient_data"}

        test_results[track] = track_tests

    return test_results


def format_comparison_table(track: str, entries: list[dict]) -> str:
    """Format a comparison table as a readable string."""
    metric_name, direction = TRACK_PRIMARY_METRIC[track]
    display = TRACK_DISPLAY_NAME[track]
    arrow = "↑" if direction == "max" else "↓"

    lines = [
        f"\n{'=' * 70}",
        f"  {display} — Primary metric: {metric_name} ({arrow} better)",
        f"{'=' * 70}",
        f"  {'Rank':>4s}  {'Model':<12s}  {metric_name:<15s}  {'Folds':>5s}  Other metrics",
        f"  {'-' * 4}  {'-' * 12}  {'-' * 15}  {'-' * 5}  {'-' * 30}",
    ]

    for entry in entries:
        model = entry["model"]
        val = entry["primary_value"]
        rank = entry["rank"]
        n_folds = entry.get("n_folds", "—")

        # Select a few other metrics to show
        other = entry.get("all_metrics", {})
        other_str = ", ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(other.items())
            if k != metric_name and isinstance(v, (int, float))
        )[:50]

        lines.append(
            f"  {rank:>4d}  {model:<12s}  {val:<15.4f}  {n_folds:>5}  {other_str}"
        )

    lines.append(f"{'=' * 70}")
    return "\n".join(lines)


def format_test_results(track: str, test_result: dict) -> str:
    """Format statistical test results."""
    display = TRACK_DISPLAY_NAME[track]
    lines = [f"\n  Statistical Tests — {display}"]

    if test_result.get("status") == "insufficient_folds":
        lines.append("  ⚠ Insufficient fold data for statistical testing")
        return "\n".join(lines)

    # Wilcoxon pairwise
    pairwise = test_result.get("wilcoxon_pairwise", [])
    if pairwise:
        lines.append("  Wilcoxon signed-rank (pairwise):")
        for pw in pairwise:
            sig = "✓" if pw["significant"] else "✗"
            lines.append(
                f"    {pw['model_a']:>10s} vs {pw['model_b']:<10s}  p={pw['p_value']:.4f} {sig}"
            )

    # Friedman
    friedman = test_result.get("friedman")
    if friedman and "statistic" in friedman:
        sig = "✓" if friedman["significant"] else "✗"
        lines.append(
            f"  Friedman χ²={friedman['statistic']:.4f}, p={friedman['p_value']:.4f} {sig}"
        )

    return "\n".join(lines)


def generate_latex_table(track: str, entries: list[dict]) -> str:
    """Generate a LaTeX table for COPPE thesis format."""
    metric_name, direction = TRACK_PRIMARY_METRIC[track]
    display = TRACK_DISPLAY_NAME[track]
    arrow = r"$\uparrow$" if direction == "max" else r"$\downarrow$"

    # Collect all unique metrics across entries
    all_metric_keys = set()
    for e in entries:
        all_metric_keys.update(e.get("all_metrics", {}).keys())

    # Select top metrics to include
    if track == "3w":
        show_metrics = ["f1_macro", "accuracy", "auc_pr"]
    elif track == "ganymede":
        show_metrics = ["mae", "rmse", "r2", "mase"]
    elif track == "cdf":
        show_metrics = ["error_mean", "error_p50", "error_p95"]
    else:
        show_metrics = list(all_metric_keys)[:4]

    col_spec = "l" + "r" * len(show_metrics)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{display} — Model Comparison}}",
        f"  \\label{{tab:{track}_comparison}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    # Header
    header = "    Model"
    for m in show_metrics:
        header += f" & {m.replace('_', ' ').title()}"
        if m == metric_name:
            header += f" {arrow}"
    header += r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    # Best values per metric (for bolding)
    best_vals: dict[str, float] = {}
    for m in show_metrics:
        vals = []
        for e in entries:
            v = e.get("all_metrics", {}).get(m)
            if v is not None:
                vals.append(v)
        if vals:
            _, dir_m = TRACK_PRIMARY_METRIC.get(track, (None, "min"))
            # Use same direction logic for primary; default min for others
            if m == metric_name:
                best_vals[m] = max(vals) if direction == "max" else min(vals)
            else:
                best_vals[m] = min(vals)  # lower is better for error metrics

    # Data rows
    for entry in entries:
        model = entry["model"].replace("_", r"\_")
        row = f"    {model}"
        for m in show_metrics:
            v = entry.get("all_metrics", {}).get(m)
            if v is not None:
                val_str = f"{v:.4f}"
                if m in best_vals and abs(v - best_vals[m]) < 1e-6:
                    val_str = r"\textbf{" + val_str + "}"
                row += f" & {val_str}"
            else:
                row += " & —"
        row += r" \\"
        lines.append(row)

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


HORIZONS = [7, 14, 30, 90]


def load_multihorizon_results(
    results_dir: Path,
) -> dict[str, dict[int, dict]]:
    """Load multi-horizon result files for all models.

    Returns:
        ``{model: {horizon: result_dict}}`` — only populated for models
        that have ``ganymede_h{h}_multi_well.json`` files.
    """
    results_dir = resolve_results_dir(results_dir)
    mh: dict[str, dict[int, dict]] = {}
    for model in ALL_MODELS:
        model_horizons: dict[int, dict] = {}
        for h in HORIZONS:
            path = results_dir / model / f"ganymede_h{h}_multi_well.json"
            if path.exists():
                with open(path) as f:
                    model_horizons[h] = json.load(f)
            else:
                logger.warning("Missing multi-horizon file: %s", path)
        if model_horizons:
            mh[model] = model_horizons
    return mh


def load_baseline_mae(results_dir: Path) -> float | None:
    """Load seasonal naive baseline MAE for table footer."""
    results_dir = resolve_results_dir(results_dir)
    path = results_dir / "baselines" / "ganymede_seasonal_naive_baseline.json"
    if not path.exists():
        logger.warning("Missing baseline file: %s", path)
        return None
    with open(path) as f:
        data = json.load(f)
    agg = data.get("aggregate", {})
    return agg.get("mae_mean") or agg.get("mae")


def generate_multihorizon_table(
    mh_results: dict[str, dict[int, dict]],
    baseline_mae: float | None,
) -> str:
    """Generate a COPPE LaTeX table comparing MAE across forecast horizons.

    Rows = models, columns = 7d / 14d / 30d / 90d.
    """
    # Extract MAE per (model, horizon)
    cells: dict[str, dict[int, float | None]] = {}
    for model in ALL_MODELS:
        if model not in mh_results:
            continue
        row: dict[int, float | None] = {}
        for h in HORIZONS:
            r = mh_results[model].get(h)
            row[h] = get_metric_value(r, "mae") if r else None
        cells[model] = row

    # Best (min) MAE per horizon column — only over real models, not baseline
    best: dict[int, float] = {}
    for h in HORIZONS:
        vals = [cells[m][h] for m in cells if cells[m].get(h) is not None]
        if vals:
            best[h] = min(vals)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Ganymede Forecasting --- MAE by Forecast Horizon}",
        r"  \label{tab:ganymede_multihorizon}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Model & 7d & 14d & 30d & 90d \\",
        r"    \midrule",
    ]

    for model in ALL_MODELS:
        if model not in cells:
            continue
        row_str = f"    {model}"
        for h in HORIZONS:
            v = cells[model].get(h)
            if v is not None:
                val_str = f"{v:.4f}"
                if h in best and abs(v - best[h]) < 1e-6:
                    val_str = r"\textbf{" + val_str + "}"
                row_str += f" & {val_str}"
            else:
                row_str += r" & ---"
        row_str += r" \\"
        lines.append(row_str)

    # Baseline row
    if baseline_mae is not None:
        lines.append(r"    \midrule")
        bl_val = f"{baseline_mae:.4f}"
        lines.append(
            f"    Seasonal Naive & {bl_val} & {bl_val} & {bl_val} & {bl_val}" + r" \\"
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def load_perwell_results(
    results_dir: Path,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """Load per-well and multi-well results for horizon 30.

    Returns:
        ``(perwell_results, multiwell_results)`` where
        ``perwell_results = {model: {well_name: result_dict}}`` and
        ``multiwell_results = {model: result_dict}``.
    """
    results_dir = resolve_results_dir(results_dir)
    perwell: dict[str, dict[str, dict]] = {}
    multiwell: dict[str, dict] = {}

    for model in ALL_MODELS:
        model_dir = results_dir / model

        # Multi-well: prefer h30 file, fall back to flat ganymede.json
        mw_path = model_dir / "ganymede_h30_multi_well.json"
        if not mw_path.exists():
            mw_path = model_dir / "ganymede.json"
        if mw_path.exists():
            with open(mw_path) as f:
                multiwell[model] = json.load(f)

        # Per-well: discover via glob
        pw_files = sorted(model_dir.glob("ganymede_h30_per_well_*.json"))
        if not pw_files:
            logger.warning("No per-well files for model %s", model)
            continue
        wells: dict[str, dict] = {}
        for pw_path in pw_files:
            # Extract well name from filename
            # ganymede_h30_per_well_49_22-Z01Z.json → 49_22-Z01Z
            stem = pw_path.stem  # ganymede_h30_per_well_49_22-Z01Z
            prefix = "ganymede_h30_per_well_"
            well_name = stem[len(prefix) :]
            with open(pw_path) as f:
                wells[well_name] = json.load(f)
        perwell[model] = wells

    return perwell, multiwell


def generate_perwell_table(
    perwell_results: dict[str, dict[str, dict]],
    multiwell_results: dict[str, dict],
) -> str:
    """Generate a COPPE LaTeX table comparing per-well vs multi-well MAE.

    Rows = wells (sorted), columns depend on how many models have per-well data.
    """
    models_with_pw = sorted(perwell_results.keys())
    if not models_with_pw:
        return "% No per-well results available.\n"

    # Collect all well names across models
    all_wells: set[str] = set()
    for model in models_with_pw:
        all_wells.update(perwell_results[model].keys())
    wells_sorted = sorted(all_wells)

    # Simplified layout when only one model has per-well data
    single_model = len(models_with_pw) == 1

    if single_model:
        model = models_with_pw[0]
        mw_result = multiwell_results.get(model)
        mw_mae = get_metric_value(mw_result, "mae") if mw_result else None

        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            f"  \\caption{{Per-Well vs Multi-Well MAE --- {model.upper()} (h=30d)}}",
            r"  \label{tab:ganymede_perwell}",
            r"  \begin{tabular}{lrr}",
            r"    \toprule",
            r"    Well & Multi-well MAE & Per-well MAE \\",
            r"    \midrule",
        ]

        for well in wells_sorted:
            pw_result = perwell_results[model].get(well)
            pw_mae = get_metric_value(pw_result, "mae") if pw_result else None

            mw_str = f"{mw_mae:.4f}" if mw_mae is not None else "---"
            pw_str = f"{pw_mae:.4f}" if pw_mae is not None else "---"

            # Bold the better value
            if mw_mae is not None and pw_mae is not None:
                if pw_mae < mw_mae:
                    pw_str = r"\textbf{" + pw_str + "}"
                elif mw_mae < pw_mae:
                    mw_str = r"\textbf{" + mw_str + "}"
                # equal → no bold

            safe_well = well.replace("_", r"\_")
            lines.append(f"    {safe_well} & {mw_str} & {pw_str}" + r" \\")

    else:
        # Multi-model layout
        col_spec = "l" + "rr" * len(models_with_pw)
        header_top = "    Well"
        header_sub = "    "
        for m in models_with_pw:
            header_top += f" & \\multicolumn{{2}}{{c}}{{{m.upper()}}}"
            header_sub += " & MW & PW"
        header_top += r" \\"
        header_sub += r" \\"

        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Per-Well vs Multi-Well MAE Comparison (h=30d)}",
            r"  \label{tab:ganymede_perwell}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            header_top,
            header_sub,
            r"    \midrule",
        ]

        for well in wells_sorted:
            safe_well = well.replace("_", r"\_")
            row = f"    {safe_well}"
            for m in models_with_pw:
                mw_result = multiwell_results.get(m)
                mw_mae = get_metric_value(mw_result, "mae") if mw_result else None
                pw_result = perwell_results[m].get(well)
                pw_mae = get_metric_value(pw_result, "mae") if pw_result else None

                mw_str = f"{mw_mae:.4f}" if mw_mae is not None else "---"
                pw_str = f"{pw_mae:.4f}" if pw_mae is not None else "---"

                if mw_mae is not None and pw_mae is not None:
                    if pw_mae < mw_mae:
                        pw_str = r"\textbf{" + pw_str + "}"
                    elif mw_mae < pw_mae:
                        mw_str = r"\textbf{" + mw_str + "}"

                row += f" & {mw_str} & {pw_str}"
            row += r" \\"
            lines.append(row)

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the full comparison pipeline."""
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # Load all results
    all_results = load_all_results(args.results_dir)
    logger.info("Loaded results for %d models", len(all_results))

    # Build comparison tables
    tables = build_comparison_table(all_results)

    # Print comparison tables
    for track, entries in tables.items():
        print(format_comparison_table(track, entries))

    # Run statistical tests
    test_results = run_statistical_tests(all_results)
    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)
    for track, result in test_results.items():
        print(format_test_results(track, result))

    # Generate LaTeX tables
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for track, entries in tables.items():
        latex = generate_latex_table(track, entries)
        latex_path = output_dir / f"{track}_comparison.tex"
        latex_path.write_text(latex)
        logger.info("LaTeX table saved: %s", latex_path)

    # ── Multi-horizon comparison ──
    results_path = Path(args.results_dir)
    mh_results = load_multihorizon_results(results_path)
    baseline_mae = load_baseline_mae(results_path)
    mh_latex = generate_multihorizon_table(mh_results, baseline_mae)
    mh_path = output_dir / "ganymede_multihorizon_comparison.tex"
    mh_path.write_text(mh_latex)
    logger.info("Multi-horizon table saved: %s", mh_path)

    # ── Per-well comparison ──
    pw_results, mw_results = load_perwell_results(results_path)
    pw_latex = generate_perwell_table(pw_results, mw_results)
    pw_path = output_dir / "ganymede_perwell_comparison.tex"
    pw_path.write_text(pw_latex)
    logger.info("Per-well table saved: %s", pw_path)

    # Save summary markdown
    summary_lines = ["# Model Comparison Summary\n"]
    for track, entries in tables.items():
        summary_lines.append(format_comparison_table(track, entries))
    summary_lines.append("\n## Statistical Tests\n")
    for track, result in test_results.items():
        summary_lines.append(format_test_results(track, result))

    # Multi-horizon summary section
    summary_lines.append("\n## Multi-Horizon Degradation (Ganymede)\n")
    if mh_results:
        summary_lines.append("| Model | 7d MAE | 14d MAE | 30d MAE | 90d MAE |")
        summary_lines.append("|-------|--------|---------|---------|---------|")
        for model in ALL_MODELS:
            if model not in mh_results:
                continue
            row = f"| {model}"
            for h in HORIZONS:
                r = mh_results[model].get(h)
                v = get_metric_value(r, "mae") if r else None
                row += f" | {v:.4f}" if v is not None else " | —"
            row += " |"
            summary_lines.append(row)
        if baseline_mae is not None:
            summary_lines.append(
                f"| Seasonal Naive | {baseline_mae:.4f} | {baseline_mae:.4f}"
                f" | {baseline_mae:.4f} | {baseline_mae:.4f} |"
            )
    else:
        summary_lines.append("No multi-horizon results available.\n")

    # Per-well summary section
    summary_lines.append("\n## Per-Well vs Multi-Well (h=30d)\n")
    if pw_results:
        for model in sorted(pw_results.keys()):
            mw_result = mw_results.get(model)
            mw_mae = get_metric_value(mw_result, "mae") if mw_result else None
            summary_lines.append(f"### {model.upper()}")
            summary_lines.append("| Well | Multi-well MAE | Per-well MAE |")
            summary_lines.append("|------|---------------|-------------|")
            for well in sorted(pw_results[model].keys()):
                pw_r = pw_results[model][well]
                pw_mae = get_metric_value(pw_r, "mae")
                mw_str = f"{mw_mae:.4f}" if mw_mae is not None else "—"
                pw_str = f"{pw_mae:.4f}" if pw_mae is not None else "—"
                summary_lines.append(f"| {well} | {mw_str} | {pw_str} |")
    else:
        summary_lines.append("No per-well results available.\n")

    summary_path = output_dir / "comparison_summary.md"
    summary_path.write_text("\n".join(summary_lines))
    logger.info("Summary saved: %s", summary_path)

    # Save test results JSON
    json_path = output_dir / "statistical_tests.json"
    json_path.write_text(json.dumps(test_results, indent=2))
    logger.info("Test results saved: %s", json_path)


if __name__ == "__main__":
    main()
