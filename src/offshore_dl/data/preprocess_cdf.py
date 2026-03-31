"""CDF (Cognite Data Fusion) preprocessing pipeline.

Reads raw compressor sensor CSV, drops all-NaN columns, renames columns
to clean short names, and saves processed parquet.

No normalization applied here — done per-fold at training time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from offshore_dl.utils.config import load_merged_config

logger = logging.getLogger(__name__)


def preprocess_cdf(
    cfg: DictConfig | None = None,
    base_config: str = "configs/base.yaml",
    data_config: str = "configs/data/cdf.yaml",
) -> dict:
    """Run the CDF preprocessing pipeline.

    Steps:
        1. Read raw CSV
        2. Parse timestamps as DatetimeIndex
        3. Drop 100%-missing columns
        4. Rename to clean short names
        5. Save as parquet

    Returns:
        Dict with preprocessing stats.
    """
    if cfg is None:
        cfg = load_merged_config(base_config, data_config)

    raw_path = Path(cfg.data.paths.raw)
    out_dir = Path(cfg.data.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    df = pd.read_csv(raw_path)
    logger.info("CDF raw data: %d rows, %d columns", len(df), len(df.columns))

    # Parse timestamps
    ts_col = cfg.data.timestamp_column
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()

    # Drop well_id column (single well, captured in metadata)
    well_col = cfg.data.get("well_column", "well_id")
    well_id = df[well_col].iloc[0] if well_col in df.columns else "unknown"
    if well_col in df.columns:
        df = df.drop(columns=[well_col])

    # Drop all-NaN columns
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        logger.info("Dropping %d all-NaN columns: %s", len(all_nan_cols), all_nan_cols)
        df = df.drop(columns=all_nan_cols)

    # Keep only configured sensor columns (those not dropped)
    sensor_cols = [c for c in cfg.data.sensor_columns if c in df.columns]
    df = df[sensor_cols]

    # Rename to clean short names
    renames = dict(cfg.data.get("column_renames", {}))
    actual_renames = {k: v for k, v in renames.items() if k in df.columns}
    df = df.rename(columns=actual_renames)

    # Fill remaining NaN with forward fill (causal, small dataset)
    df = df.ffill().bfill()  # bfill for leading NaN only

    # Save
    out_path = out_dir / "cdf_processed.parquet"
    df.to_parquet(out_path)

    stats = {
        "rows": len(df),
        "columns": list(df.columns),
        "n_columns": len(df.columns),
        "date_range": (str(df.index.min()), str(df.index.max())),
        "well_id": str(well_id),
        "dropped_columns": all_nan_cols,
        "missing_pct": {c: float(df[c].isna().mean() * 100) for c in df.columns},
    }

    logger.info(
        "CDF preprocessing complete: %d rows, %d columns, %s to %s",
        stats["rows"], stats["n_columns"],
        stats["date_range"][0], stats["date_range"][1],
    )

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_cdf()
