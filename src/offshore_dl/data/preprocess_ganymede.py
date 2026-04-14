"""Ganymede (NSTA) gas production forecasting preprocessing pipeline.

Reads cleaned Ganymede CSV, applies transforms per well, and saves
processed parquets to ``data/processed/ganymede/``.

Normalization is NOT applied here — done per-fold at training time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from offshore_dl.data.transforms import (
    compute_derived_ratios,
    compute_ema_features,
    compute_rate_of_change,
    detect_shutdowns,
    log_transform,
)
from offshore_dl.utils.config import load_merged_config

logger = logging.getLogger(__name__)


def preprocess_ganymede(
    cfg: DictConfig | None = None,
    base_config: str = "configs/base.yaml",
    data_config: str = "configs/data/ganymede.yaml",
) -> dict:
    """Run the Ganymede preprocessing pipeline.

    Steps per well:
        1. Parse dates, sort chronologically
        2. Detect shutdown periods
        3. Compute EMA features
        4. Compute rate-of-change features
        5. Compute derived ratios (BHP/WHP)
        6. Log-transform gas production
        7. Drop 100%-missing columns
        8. Save per-well parquet

    Returns:
        Dict with per-well stats.
    """
    if cfg is None:
        cfg = load_merged_config(base_config, data_config)

    raw_path = Path(cfg.data.paths.raw)
    out_dir = Path(cfg.data.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    df_all = pd.read_csv(raw_path)
    logger.info(
        "Ganymede raw data: %d rows, %d columns", len(df_all), len(df_all.columns)
    )

    time_col = cfg.data.time_column
    well_col = cfg.data.well_column
    target_col = cfg.data.target_column

    df_all[time_col] = pd.to_datetime(df_all[time_col])

    # Feature columns to keep
    feature_cols = list(cfg.data.feature_columns)

    # Preprocessing params
    shutdown_days = cfg.data.preprocessing.shutdown_zero_days
    ema_windows = list(cfg.data.preprocessing.ema_windows)
    log_cols = list(cfg.data.preprocessing.log_transform_columns)

    stats = {"wells": {}}

    for well_name in sorted(df_all[well_col].unique()):
        df = df_all[df_all[well_col] == well_name].copy()
        df = df.set_index(time_col).sort_index()

        # Drop metadata columns
        drop_cols = [c for c in ["WELL_UWI", "FIELD", "WELLNAME"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # Drop 100%-missing columns
        all_nan = [c for c in df.columns if df[c].isna().all()]
        if all_nan:
            df = df.drop(columns=all_nan)

        # Keep only feature columns that exist
        avail_features = [c for c in feature_cols if c in df.columns]

        # Apply transforms
        if target_col in df.columns:
            df = detect_shutdowns(
                df, gas_column=target_col, zero_days_threshold=shutdown_days
            )

        ema_cols = [c for c in avail_features if c in df.columns]
        df = compute_ema_features(df, columns=ema_cols, windows=ema_windows)
        df = compute_rate_of_change(df, columns=ema_cols)
        df = compute_derived_ratios(df)

        log_avail = [c for c in log_cols if c in df.columns]
        if log_avail:
            df = log_transform(
                df, columns=log_avail, eps=cfg.data.preprocessing.log_eps
            )

        # Forward-fill remaining NaN, then replace any leading NaN with 0.0
        df = df.ffill().fillna(0.0)

        # Save
        safe_name = well_name.replace("/", "_")
        out_path = out_dir / f"{safe_name}.parquet"
        df.to_parquet(out_path)

        stats["wells"][well_name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": (str(df.index.min()), str(df.index.max())),
            "dropped_columns": all_nan,
        }

        logger.info(
            "  Well %s: %d rows, %d cols, %s → %s",
            well_name,
            len(df),
            len(df.columns),
            df.index.min().date(),
            df.index.max().date(),
        )

    logger.info("Ganymede preprocessing complete: %d wells", len(stats["wells"]))
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_ganymede()
