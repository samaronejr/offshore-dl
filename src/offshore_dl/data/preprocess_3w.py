"""3W v2.0.0 preprocessing pipeline.

Reads raw parquet instances from ``data/raw/3w/``, applies cleaning transforms
(frozen value detection, causal forward fill), and saves processed parquets
to ``data/processed/3w/``.

The pipeline is deterministic: same input → same output, every time.
Normalization (z-score) is NOT applied here — it's done per-fold at training
time to prevent data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from offshore_dl.data.transforms import causal_forward_fill, detect_frozen_values
from offshore_dl.utils.config import load_merged_config

logger = logging.getLogger(__name__)

# 3W v2.0.0 sensor columns (27 variables)
SENSOR_COLUMNS = [
    "ABER-CKGL", "ABER-CKP",
    "ESTADO-DHSV", "ESTADO-M1", "ESTADO-M2", "ESTADO-PXO",
    "ESTADO-SDV-GL", "ESTADO-SDV-P", "ESTADO-W1", "ESTADO-W2", "ESTADO-XO",
    "P-ANULAR", "P-JUS-BS", "P-JUS-CKGL", "P-JUS-CKP",
    "P-MON-CKGL", "P-MON-CKP", "P-MON-SDV-P",
    "P-PDG", "PT-P", "P-TPT",
    "QBS", "QGL",
    "T-JUS-CKP", "T-MON-CKP", "T-PDG", "T-TPT",
]


def preprocess_instance(
    parquet_path: Path,
    frozen_window: int = 60,
    ffill_limit: int = 300,
) -> pd.DataFrame:
    """Preprocess a single 3W instance.

    Steps:
        1. Read parquet (timestamp is index)
        2. Detect and replace frozen sensor values
        3. Causal forward-fill within limit
        4. Keep sensor columns + class + state

    Args:
        parquet_path: Path to the raw parquet file.
        frozen_window: Rolling window for frozen detection.
        ffill_limit: Max consecutive timesteps for forward fill.

    Returns:
        Cleaned DataFrame with timestamp index.
    """
    df = pd.read_parquet(parquet_path)

    # Ensure timestamp is index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    # Identify which sensor columns are actually present and not all-NaN
    available_sensors = [
        c for c in SENSOR_COLUMNS
        if c in df.columns and not df[c].isna().all()
    ]

    # Apply transforms only to available sensor columns
    df = detect_frozen_values(df, window=frozen_window, columns=available_sensors)
    df = causal_forward_fill(df, limit=ffill_limit, columns=available_sensors)

    return df


def preprocess_3w(
    cfg: DictConfig | None = None,
    base_config: str = "configs/base.yaml",
    data_config: str = "configs/data/3w.yaml",
) -> dict:
    """Run the full 3W preprocessing pipeline.

    Args:
        cfg: Pre-loaded config. If None, loads from file paths.
        base_config: Path to base config (ignored if cfg provided).
        data_config: Path to 3W data config (ignored if cfg provided).

    Returns:
        Dict with preprocessing stats: class_counts, total_instances, etc.
    """
    if cfg is None:
        cfg = load_merged_config(base_config, data_config)

    raw_dir = Path(cfg.data.paths.raw)
    out_dir = Path(cfg.data.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_window = cfg.data.preprocessing.frozen_value_window
    ffill_limit = cfg.data.preprocessing.forward_fill_limit

    stats = {"class_counts": {}, "total_instances": 0, "errors": []}

    # Process each class folder (0-9)
    for class_id in range(10):
        class_dir = raw_dir / str(class_id)
        if not class_dir.exists():
            logger.warning("Class directory not found: %s", class_dir)
            continue

        class_out_dir = out_dir / str(class_id)
        class_out_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(class_dir.glob("*.parquet"))
        stats["class_counts"][class_id] = len(parquet_files)

        for pf in parquet_files:
            try:
                df = preprocess_instance(pf, frozen_window, ffill_limit)
                out_path = class_out_dir / pf.name
                df.to_parquet(out_path)
                stats["total_instances"] += 1
            except Exception as e:
                logger.error("Error processing %s: %s", pf, e)
                stats["errors"].append(str(pf))

    logger.info(
        "3W preprocessing complete: %d instances across %d classes",
        stats["total_instances"],
        len(stats["class_counts"]),
    )
    for cls, count in sorted(stats["class_counts"].items()):
        logger.info("  Class %d: %d instances", cls, count)

    if stats["errors"]:
        logger.warning("  %d errors encountered", len(stats["errors"]))

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_3w()
