"""Aggregate per-model forecasting result JSONs into manuscript-ready tables.

Walks ``results/<model>/<dataset>_h<H>_<mode>[_<well>].json`` files, collects
``test_metrics`` for each combination, and writes:

- ``results/forecasting_summary.csv``  — long format, one row per
  (dataset, horizon, mode, well, model).
- ``results/forecasting_summary_wide_<metric>.csv`` — wide pivots for MAE
  and R²_prod (rows = scenarios, columns = models).
- ``reports/forecasting_borda.json`` — Borda rank-aggregate per dataset and
  cross-dataset, identifying the model that wins the most scenarios.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
REPORT_DIR = Path("reports")

DATASETS = ["ganymede", "spe_berg", "inner_mongolia", "volve"]
METRICS = ["mae", "rmse", "r2", "r2_prod", "mase"]

FILENAME_RE = re.compile(
    r"^(?P<dataset>" + "|".join(DATASETS) + r")_h(?P<horizon>\d+)_"
    r"(?P<mode>multi_well|per_well)(?:_(?P<well>.+))?\.json$"
)


def collect_rows() -> pd.DataFrame:
    rows = []
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for path in sorted(model_dir.iterdir()):
            m = FILENAME_RE.match(path.name)
            if not m:
                continue
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
            tm = data.get("test_metrics", {}) or {}
            row = {
                "model": model,
                "dataset": m.group("dataset"),
                "horizon": int(m.group("horizon")),
                "mode": m.group("mode"),
                "well": m.group("well") or "",
                "status": data.get("status", "ok"),
            }
            for k in METRICS:
                row[k] = tm.get(k, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def write_long(df: pd.DataFrame, out: Path) -> None:
    df.sort_values(["dataset", "horizon", "mode", "well", "model"]).to_csv(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(df))


def write_wide(df: pd.DataFrame, metric: str, out: Path) -> None:
    pivot = df.pivot_table(
        index=["dataset", "horizon", "mode", "well"],
        columns="model",
        values=metric,
        aggfunc="first",
    )
    pivot.to_csv(out)
    logger.info("Wrote %s (shape %s)", out, pivot.shape)


def borda_rankings(df: pd.DataFrame) -> dict:
    """Borda count: for each scenario, rank models by metric (lower=better for
    error metrics, higher=better for r2/r2_prod). Sum ranks across scenarios.
    """
    higher_better = {"r2", "r2_prod"}
    out: dict = {"per_dataset": {}, "cross_dataset": {}}

    for metric in METRICS:
        per_dataset: dict[str, dict[str, float]] = {}
        global_ranks: dict[str, list[float]] = defaultdict(list)

        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds].copy()
            scenario_keys = ["horizon", "mode", "well"]
            ranks_per_model: dict[str, list[float]] = defaultdict(list)
            for _, group in sub.groupby(scenario_keys):
                vals = group.dropna(subset=[metric])
                if len(vals) < 2:
                    continue
                ascending = metric not in higher_better
                ranked = vals[metric].rank(ascending=ascending, method="average")
                for model, r in zip(vals["model"], ranked, strict=False):
                    ranks_per_model[model].append(float(r))
                    global_ranks[model].append(float(r))
            per_dataset[ds] = {
                m: round(float(np.mean(rs)), 3)
                for m, rs in ranks_per_model.items()
                if rs
            }

        cross = {
            m: round(float(np.mean(rs)), 3)
            for m, rs in global_ranks.items()
            if rs
        }
        out["per_dataset"][metric] = per_dataset
        out["cross_dataset"][metric] = cross
    return out


def main() -> None:
    df = collect_rows()
    if df.empty:
        logger.warning("No forecasting result files matched. Nothing to aggregate.")
        return

    logger.info(
        "Collected %d rows: %d models × %d datasets × scenarios",
        len(df),
        df["model"].nunique(),
        df["dataset"].nunique(),
    )

    write_long(df, RESULTS_DIR / "forecasting_summary.csv")
    write_wide(df, "mae", RESULTS_DIR / "forecasting_summary_wide_mae.csv")
    write_wide(df, "r2_prod", RESULTS_DIR / "forecasting_summary_wide_r2_prod.csv")

    borda = borda_rankings(df)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "forecasting_borda.json").write_text(json.dumps(borda, indent=2))
    logger.info("Wrote %s", REPORT_DIR / "forecasting_borda.json")

    logger.info("Cross-dataset Borda (lower = better) — MAE:")
    for m, r in sorted(borda["cross_dataset"]["mae"].items(), key=lambda kv: kv[1]):
        logger.info("  %-12s %.3f", m, r)
    logger.info("Cross-dataset Borda (lower = better) — R²_prod:")
    for m, r in sorted(borda["cross_dataset"]["r2_prod"].items(), key=lambda kv: kv[1]):
        logger.info("  %-12s %.3f", m, r)


if __name__ == "__main__":
    main()
