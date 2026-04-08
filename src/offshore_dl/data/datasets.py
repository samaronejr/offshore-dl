"""PyTorch Dataset implementations for all three experimental tracks.

- ``ThreeWDataset`` — 3W v2.0.0 anomaly classification (T02)
- ``ThreeWFeatureDataset`` — 3W with statistical feature extraction (T02-v2)
- ``CDFDataset`` — CDF unsupervised anomaly discovery (T03)
- ``GanymedeDataset`` — NSTA Ganymede gas production forecasting (T04)
- ``SPEBergDataset`` — SPE Bergen Eagle Ford shale gas production forecasting (T05)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from offshore_dl.data.base import BaseDataset
from offshore_dl.data.preprocess_3w import SENSOR_COLUMNS
from offshore_dl.data.transforms import sliding_window_segmentation
from offshore_dl.utils.config import load_merged_config

logger = logging.getLogger(__name__)


class ThreeWDataset(BaseDataset):
    """3W v2.0.0 anomaly detection/classification dataset.

    Loads preprocessed parquet instances, applies sliding window segmentation,
    and returns ``(tensor[w, n_vars], label: int, metadata: dict)`` tuples.

    Handles variable sensor availability across wells by using a fixed column
    set and filling missing columns with zeros.

    Args:
        config: Path to 3W YAML config or pre-loaded DictConfig.
        base_config: Path to base config (for resolving interpolations).
        window_size: Override window size from config.
        window_stride: Override window stride from config.
        cache_in_memory: Whether to cache all data in RAM.
    """

    def __init__(
        self,
        config: str | Path | DictConfig,
        base_config: str | Path = "configs/base.yaml",
        window_size: int | None = None,
        window_stride: int | None = None,
        cache_in_memory: bool | None = None,
        class_ids: list[int] | None = None,
        max_instances_per_class: int | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.cfg = load_merged_config(base_config, config)
        else:
            self.cfg = config

        self.window_size = window_size or self.cfg.data.preprocessing.window_size
        self.window_stride = window_stride or self.cfg.data.preprocessing.window_stride

        # Use sensor columns from config or default
        self.sensor_columns = list(
            self.cfg.data.get("sensor_columns", SENSOR_COLUMNS)
        )
        self.n_vars = len(self.sensor_columns)

        should_cache = cache_in_memory
        if should_cache is None:
            should_cache = self.cfg.data.preprocessing.get("cache_in_memory", True)
        self._cache_in_memory = should_cache
        self._class_ids = class_ids
        self._max_instances_per_class = max_instances_per_class

        # Build instance index
        self._instances: list[dict] = []
        self._windows: list[dict] = []
        self._data_cache: dict[str, pd.DataFrame] = {}
        self._feature_cache: dict[str, np.ndarray] = {}

        self._build_index()

    def _build_index(self) -> None:
        """Scan processed parquet directory and build window index."""
        processed_dir = Path(self.cfg.data.paths.processed)

        if not processed_dir.exists():
            # Fall back to raw directory if processed doesn't exist
            processed_dir = Path(self.cfg.data.paths.raw)
            logger.info("Processed dir not found, using raw: %s", processed_dir)

        class_range = self._class_ids if self._class_ids else range(self.cfg.data.n_classes)

        for class_id in class_range:
            class_dir = processed_dir / str(class_id)
            if not class_dir.exists():
                continue

            parquet_files_all = sorted(class_dir.glob("*.parquet"))
            if self._max_instances_per_class is not None:
                parquet_files_all = parquet_files_all[:self._max_instances_per_class]

            for pf in parquet_files_all:
                instance_id = pf.stem
                well_id = instance_id.split("_")[0] if "_" in instance_id else instance_id
                source_type = (
                    "simulated" if instance_id.startswith("SIMULATED")
                    else "drawn" if instance_id.startswith("DRAWN")
                    else "real"
                )

                # Read to compute window count
                df = self._load_instance(pf)
                if df is None:
                    continue

                values = self._extract_features(df)

                # Use folder class_id as ground truth label (not per-timestep
                # class column which has transient states like 101+).
                windows = sliding_window_segmentation(
                    values, self.window_size, self.window_stride, labels=None
                )

                for w in windows:
                    w["file"] = str(pf)
                    w["class_id"] = class_id
                    w["label"] = class_id  # ground truth from folder
                    w["well_id"] = well_id
                    w["source_type"] = source_type
                    w["instance_id"] = instance_id

                self._windows.extend(windows)

                # Cache strategy: feature arrays are always needed for
                # __getitem__ window slicing.
                # - cache_in_memory=True: cache DataFrames (legacy) — feature
                #   cache populated lazily on first __getitem__ hit.
                # - cache_in_memory=False: cache feature arrays eagerly here
                #   (~8 GB vs ~17 GB DataFrames), skip DataFrame cache.
                if self._cache_in_memory:
                    self._data_cache[str(pf)] = df
                else:
                    self._feature_cache[str(pf)] = values

        logger.info(
            "ThreeWDataset: %d windows from %d class directories, w=%d, s=%d",
            len(self._windows), self.cfg.data.n_classes,
            self.window_size, self.window_stride,
        )

    def _load_instance(self, path: Path) -> pd.DataFrame | None:
        """Load a single parquet instance."""
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            return df
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            return None

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract sensor columns into a numpy array.

        Missing columns are filled with zeros. Remaining NaNs/Infs filled with 0.
        Nullable integer columns (e.g. pd.Int16) are handled via pd.to_numeric.
        """
        features = np.zeros((len(df), self.n_vars), dtype=np.float32)
        for i, col in enumerate(self.sensor_columns):
            if col in df.columns:
                vals = df[col]
                # Handle nullable integer types (Int8/16/32/64) which have pd.NA
                if hasattr(vals.dtype, "numpy_dtype"):
                    vals = vals.to_numpy(dtype="float64", na_value=np.nan)
                else:
                    vals = vals.to_numpy(dtype="float64")
                # Convert to float64 first to avoid float32 overflow during
                # feature extraction (e.g. energy = sum(x²) on Pa-scale
                # pressures ~1e7 would overflow float32).
                features[:, i] = vals.astype(np.float32)

        # Replace NaN/Inf with 0
        np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict]:
        """Return a single windowed sample.

        Returns:
            Tuple of (features[w, n_vars], label, metadata).
        """
        w = self._windows[index]

        # Load data — use feature cache to avoid re-extracting every call
        file_path = w["file"]
        if file_path in self._feature_cache:
            features = self._feature_cache[file_path]
        else:
            if file_path in self._data_cache:
                df = self._data_cache[file_path]
            else:
                df = self._load_instance(Path(file_path))
                if self._cache_in_memory:
                    self._data_cache[file_path] = df
            features = self._extract_features(df)
            if self._cache_in_memory:
                self._feature_cache[file_path] = features
        window_data = features[w["start"]:w["end"]]

        tensor = torch.from_numpy(window_data)  # [w, n_vars]
        label = int(w.get("label", w["class_id"]))

        metadata = {
            "well_id": w["well_id"],
            "class_id": w["class_id"],
            "source_type": w["source_type"],
            "instance_id": w["instance_id"],
            "window_start": w["start"],
            "window_end": w["end"],
        }

        return tensor, label, metadata

    def __len__(self) -> int:
        return len(self._windows)

    def get_metadata(self) -> dict:
        """Return dataset-level metadata."""
        class_counts = {}
        for w in self._windows:
            cls = w["class_id"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        return {
            "class": "ThreeWDataset",
            "length": len(self),
            "n_vars": self.n_vars,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "class_counts": class_counts,
            "sensor_columns": self.sensor_columns,
        }


class ThreeWFeatureDataset(BaseDataset):
    """3W dataset with statistical feature extraction.

    Wraps ``ThreeWDataset`` and replaces each ``(720, 27)`` raw window
    with a ``(14, 27)`` statistical feature matrix.  Each of the 14 rows
    is a statistical descriptor (mean, std, slope, …) computed per-sensor,
    compressing 720 timesteps into 14 features while keeping the
    ``(seq, n_vars)`` layout that LSTM / DeepONet / PatchTST expect.

    The feature extraction follows published 3W literature (Fernandes
    Junior et al., 2024) which achieved F1 ≈ 0.95 with statistical
    features + classical ML.

    Args:
        config: Forwarded to ``ThreeWDataset``.
        base_config: Forwarded to ``ThreeWDataset``.
        **kwargs: Extra keyword arguments forwarded to ``ThreeWDataset``.
    """

    def __init__(
        self,
        config: str | Path | DictConfig,
        base_config: str | Path = "configs/base.yaml",
        **kwargs,
    ) -> None:
        from offshore_dl.data.feature_extractor import N_FEATURES, extract_window_features

        self._inner = ThreeWDataset(config, base_config=base_config, **kwargs)
        self.n_features = N_FEATURES  # 14
        self.n_vars = self._inner.n_vars
        self.sensor_columns = self._inner.sensor_columns
        self.cfg = self._inner.cfg

        # Override window_size so models use n_features as sequence length
        self.window_size = N_FEATURES

        # Pre-compute ALL features at init — avoids per-getitem scipy calls
        n = len(self._inner)
        logger.info(
            "ThreeWFeatureDataset: pre-computing features for %d windows "
            "(720, %d) → (%d, %d) …",
            n, self.n_vars, self.n_features, self.n_vars,
        )
        self._feature_cache: list[np.ndarray] = [None] * n
        self._labels: list[int] = [0] * n
        self._metadata: list[dict] = [{}] * n

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(n):
                raw_tensor, label, meta = self._inner[i]
                self._feature_cache[i] = extract_window_features(raw_tensor.numpy())
                self._labels[i] = label
                self._metadata[i] = meta
                if (i + 1) % 50000 == 0:
                    logger.info("  … extracted %d / %d (%.0f%%)", i + 1, n, 100 * (i + 1) / n)

        logger.info(
            "ThreeWFeatureDataset: %d windows pre-computed", n,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict]:
        features = self._feature_cache[index]  # (14, n_vars) pre-computed
        return torch.from_numpy(features), self._labels[index], self._metadata[index]

    def __len__(self) -> int:
        return len(self._inner)

    def get_metadata(self) -> dict:
        meta = self._inner.get_metadata()
        meta["class"] = "ThreeWFeatureDataset"
        meta["n_features"] = self.n_features
        meta["original_window_size"] = self._inner.window_size
        return meta


class CDFDataset(BaseDataset):
    """CDF unsupervised anomaly discovery dataset.

    Single compressor sensor data with no labels. Supports two modes:

    - **reconstruction**: input and target are the same window (autoencoder).
    - **prediction**: input is a window, target is the next ``h`` timesteps.

    Returns ``(input_tensor, target_tensor, metadata)`` — no label int.

    Args:
        config: Path to CDF YAML config or pre-loaded DictConfig.
        base_config: Path to base config.
        mode: ``"reconstruction"`` or ``"prediction"``. Overrides config.
        window_size: Override window size from config.
        window_stride: Override window stride from config.
    """

    def __init__(
        self,
        config: str | Path | DictConfig,
        base_config: str | Path = "configs/base.yaml",
        mode: str | None = None,
        window_size: int | None = None,
        window_stride: int | None = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.cfg = load_merged_config(base_config, config)
        else:
            self.cfg = config

        self.mode = mode or self.cfg.data.preprocessing.get("mode", "reconstruction")
        self.window_size = window_size or self.cfg.data.preprocessing.window_size
        self.window_stride = window_stride or self.cfg.data.preprocessing.window_stride
        self.prediction_horizon = self.cfg.data.preprocessing.get("prediction_horizon", 12)

        self._load_data()

    def _load_data(self) -> None:
        """Load the processed CDF parquet and build window index."""
        processed_dir = Path(self.cfg.data.paths.processed)
        processed_path = processed_dir / "cdf_processed.parquet"

        if not processed_path.exists():
            # Run preprocessing if not done yet
            from offshore_dl.data.preprocess_cdf import preprocess_cdf
            preprocess_cdf(self.cfg)

        self._df = pd.read_parquet(processed_path)
        self._values = self._df.values.astype(np.float32)
        self.n_vars = self._values.shape[1]
        self.columns = list(self._df.columns)

        # Replace any remaining NaN with 0
        np.nan_to_num(self._values, copy=False, nan=0.0)

        # Build window index
        n = len(self._values)
        if self.mode == "prediction":
            max_start = n - self.window_size - self.prediction_horizon
        else:
            max_start = n - self.window_size

        self._starts = list(range(0, max(0, max_start) + 1, self.window_stride))

        logger.info(
            "CDFDataset (%s): %d windows, w=%d, stride=%d, n_vars=%d",
            self.mode, len(self._starts), self.window_size,
            self.window_stride, self.n_vars,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return a single sample as ``(input, target, metadata)``."""
        start = self._starts[index]
        end = start + self.window_size

        input_data = self._values[start:end]
        input_tensor = torch.from_numpy(input_data.copy())

        if self.mode == "reconstruction":
            target_tensor = input_tensor.clone()
        else:  # prediction
            target_end = end + self.prediction_horizon
            target_data = self._values[end:target_end]
            target_tensor = torch.from_numpy(target_data.copy())

        metadata = {
            "well_id": "23-KA-9101",
            "window_start": start,
            "window_end": end,
            "mode": self.mode,
            "timestamp_start": str(self._df.index[start]),
            "timestamp_end": str(self._df.index[end - 1]),
        }

        return input_tensor, target_tensor, metadata

    def __len__(self) -> int:
        return len(self._starts)

    def get_metadata(self) -> dict:
        return {
            "class": "CDFDataset",
            "length": len(self),
            "n_vars": self.n_vars,
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "mode": self.mode,
            "columns": self.columns,
            "prediction_horizon": self.prediction_horizon if self.mode == "prediction" else None,
        }


class GanymedeDataset(BaseDataset):
    """NSTA Ganymede gas production forecasting dataset.

    Supports per-well and multi-well modes with configurable forecast horizons.

    Returns ``(input_tensor[w, n_vars], target_tensor[h], metadata)`` where
    the target is the gas production column over the forecast horizon.

    Args:
        config: Path to Ganymede YAML config or pre-loaded DictConfig.
        base_config: Path to base config.
        mode: ``"per_well"`` or ``"multi_well"``. Overrides config.
        well_name: Specific well to load (per_well mode only).
        horizon: Override forecast horizon from config.
        input_window: Override input window size from config.
    """

    def __init__(
        self,
        config: str | Path | DictConfig,
        base_config: str | Path = "configs/base.yaml",
        mode: str = "multi_well",
        well_name: str | None = None,
        horizon: int | None = None,
        input_window: int | None = None,
        filter_shutdowns: bool = False,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.cfg = load_merged_config(base_config, config)
        else:
            self.cfg = config

        self.mode = mode
        self.well_name = well_name
        self.horizon = horizon or self.cfg.data.forecasting.default_horizon
        self.input_window = input_window or self.cfg.data.forecasting.input_window
        self.target_col = self.cfg.data.target_column
        self.filter_shutdowns = filter_shutdowns

        self._load_data()

    def _load_data(self) -> None:
        """Load processed Ganymede parquets and build sample index."""
        processed_dir = Path(self.cfg.data.paths.processed)

        if not processed_dir.exists() or not any(processed_dir.glob("*.parquet")):
            from offshore_dl.data.preprocess_ganymede import preprocess_ganymede
            preprocess_ganymede(self.cfg)

        # Load wells
        self._well_data: list[tuple[str, pd.DataFrame]] = []

        if self.mode == "per_well" and self.well_name:
            safe_name = self.well_name.replace("/", "_")
            path = processed_dir / f"{safe_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                self._well_data.append((self.well_name, df))
        else:
            # Load all wells
            for path in sorted(processed_dir.glob("*.parquet")):
                df = pd.read_parquet(path)
                name = path.stem.replace("_", "/", 1)  # restore well name
                self._well_data.append((name, df))

        # Build sample index: (well_idx, start_idx)
        self._samples: list[tuple[int, int]] = []

        for well_idx, (name, df) in enumerate(self._well_data):
            n = len(df)
            total_needed = self.input_window + self.horizon

            for start in range(0, max(0, n - total_needed + 1)):
                self._samples.append((well_idx, start))

        # Align all wells to common column set (union, fill missing with 0)
        if self._well_data:
            all_cols: list[set[str]] = [set(df.columns) for _, df in self._well_data]
            self._common_columns = sorted(set.union(*all_cols))
        else:
            self._common_columns = []

        # Compute target column index in the ALIGNED column order
        if self.target_col in self._common_columns:
            self._target_col_idx = self._common_columns.index(self.target_col)
        else:
            self._target_col_idx = 0
            logger.warning(
                "Target column '%s' not found in aligned columns, using col 0",
                self.target_col,
            )

        # Precompute numpy arrays with aligned columns
        self._arrays: list[np.ndarray] = []
        for _, df in self._well_data:
            aligned = df.reindex(columns=self._common_columns, fill_value=0.0)
            arr = aligned.values.astype(np.float32)
            np.nan_to_num(arr, copy=False, nan=0.0)
            self._arrays.append(arr)

        self.n_vars = len(self._common_columns)

        # ── Filter shutdown windows ──
        # If requested, drop windows where the target is all-zero
        # (well offline). These add no forecasting signal and poison R².
        if self.filter_shutdowns and self._arrays:
            original_n = len(self._samples)
            filtered = []
            for well_idx, start in self._samples:
                arr = self._arrays[well_idx]
                input_end = start + self.input_window
                target_end = input_end + self.horizon
                target_vals = arr[input_end:target_end, self._target_col_idx]
                if np.any(target_vals > 0.01):
                    filtered.append((well_idx, start))
            self._samples = filtered
            logger.info(
                "  Filtered shutdowns: %d → %d samples (%.0f%% removed)",
                original_n, len(filtered),
                100 * (1 - len(filtered) / original_n) if original_n > 0 else 0,
            )

        logger.info(
            "GanymedeDataset (%s): %d samples, %d wells, input=%d, horizon=%d, n_vars=%d",
            self.mode, len(self._samples), len(self._well_data),
            self.input_window, self.horizon, self.n_vars,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return a single forecasting sample."""
        well_idx, start = self._samples[index]
        well_name, _ = self._well_data[well_idx]
        arr = self._arrays[well_idx]
        input_end = start + self.input_window
        target_end = input_end + self.horizon

        # Input: all features over the input window
        input_data = arr[start:input_end]
        input_tensor = torch.from_numpy(input_data.copy())

        # Target: only the gas production column over the horizon
        target_data = arr[input_end:target_end, self._target_col_idx]
        target_tensor = torch.from_numpy(target_data.copy())

        metadata = {
            "well_name": well_name,
            "well_idx": well_idx,
            "start_idx": start,
            "input_end": input_end,
            "target_end": target_end,
            "horizon": self.horizon,
            "mode": self.mode,
        }

        return input_tensor, target_tensor, metadata

    def __len__(self) -> int:
        return len(self._samples)

    def get_metadata(self) -> dict:
        well_counts = {}
        for well_idx, _ in self._samples:
            name = self._well_data[well_idx][0]
            well_counts[name] = well_counts.get(name, 0) + 1

        return {
            "class": "GanymedeDataset",
            "length": len(self),
            "n_vars": self.n_vars,
            "input_window": self.input_window,
            "horizon": self.horizon,
            "mode": self.mode,
            "well_counts": well_counts,
        }


class SPEBergDataset(BaseDataset):
    """SPE Bergen Eagle Ford shale gas production forecasting dataset.

    Supports per-well and multi-well modes with configurable forecast horizons.
    Reads preprocessed per-well parquets from ``data/processed/spe_berg/``
    (produced by ``preprocess_spe_berg``).

    Returns ``(input_tensor[w, n_vars], target_tensor[h], metadata)`` where
    the target is the ``Gas_Volume_MMscf`` column over the forecast horizon.

    If the preprocessed parquets are missing, preprocessing is run automatically.

    Args:
        config: Path to SPE BERG YAML config or pre-loaded DictConfig.
        base_config: Path to base config.
        mode: ``"per_well"`` or ``"multi_well"``. Overrides config default.
        well_name: Specific well filename stem to load (per_well mode only,
            e.g. ``"well_1"``).
        horizon: Override forecast horizon from config.
        input_window: Override input window size from config.
        filter_shutdowns: If True, drop samples whose target window is all-zero
            (well offline / shutdown).
    """

    def __init__(
        self,
        config: str | Path | DictConfig,
        base_config: str | Path = "configs/base.yaml",
        mode: str = "multi_well",
        well_name: str | None = None,
        horizon: int | None = None,
        input_window: int | None = None,
        filter_shutdowns: bool = False,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.cfg = load_merged_config(base_config, config)
        else:
            self.cfg = config

        self.mode = mode
        self.well_name = well_name
        self.horizon = horizon or self.cfg.data.forecasting.default_horizon
        self.input_window = input_window or self.cfg.data.forecasting.input_window
        self.target_col = self.cfg.data.target_column
        self.filter_shutdowns = filter_shutdowns

        self._load_data()

    def _load_data(self) -> None:
        """Load processed SPE BERG parquets and build sample index."""
        processed_dir = Path(self.cfg.data.paths.processed)

        if not processed_dir.exists() or not any(processed_dir.glob("*.parquet")):
            from offshore_dl.data.preprocess_spe_berg import preprocess_spe_berg
            preprocess_spe_berg(self.cfg)

        # Load wells
        self._well_data: list[tuple[str, pd.DataFrame]] = []

        if self.mode == "per_well" and self.well_name:
            path = processed_dir / f"{self.well_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                self._well_data.append((self.well_name, df))
            else:
                logger.warning(
                    "Well parquet not found: %s — loading all wells instead", path
                )
                for p in sorted(processed_dir.glob("*.parquet")):
                    df = pd.read_parquet(p)
                    self._well_data.append((p.stem, df))
        else:
            # Multi-well: load all wells sorted by filename stem
            for path in sorted(processed_dir.glob("*.parquet")):
                df = pd.read_parquet(path)
                self._well_data.append((path.stem, df))

        # Build sample index: (well_idx, start_idx)
        self._samples: list[tuple[int, int]] = []
        total_needed = self.input_window + self.horizon

        for well_idx, (name, df) in enumerate(self._well_data):
            n = len(df)
            for start in range(0, max(0, n - total_needed + 1)):
                self._samples.append((well_idx, start))

        # Align all wells to a common sorted column set (union, fill missing with 0)
        if self._well_data:
            all_cols: list[set[str]] = [set(df.columns) for _, df in self._well_data]
            self._common_columns: list[str] = sorted(set.union(*all_cols))
        else:
            self._common_columns = []

        # Compute target column index in the ALIGNED (sorted) column order
        if self.target_col in self._common_columns:
            self._target_col_idx = self._common_columns.index(self.target_col)
        else:
            self._target_col_idx = 0
            logger.warning(
                "Target column '%s' not found in aligned columns, using col 0",
                self.target_col,
            )

        # Precompute numpy arrays with aligned columns (avoids per-getitem reindex)
        self._arrays: list[np.ndarray] = []
        for _, df in self._well_data:
            aligned = df.reindex(columns=self._common_columns, fill_value=0.0)
            arr = aligned.values.astype(np.float32)
            np.nan_to_num(arr, copy=False, nan=0.0)
            self._arrays.append(arr)

        self.n_vars = len(self._common_columns)

        # ── Filter shutdown windows ──
        # Drop samples where the entire target horizon is zero (well offline).
        if self.filter_shutdowns and self._arrays:
            original_n = len(self._samples)
            filtered = []
            for well_idx, start in self._samples:
                arr = self._arrays[well_idx]
                input_end = start + self.input_window
                target_end = input_end + self.horizon
                target_vals = arr[input_end:target_end, self._target_col_idx]
                if np.any(target_vals > 0.01):
                    filtered.append((well_idx, start))
            self._samples = filtered
            logger.info(
                "  Filtered shutdowns: %d → %d samples (%.0f%% removed)",
                original_n, len(filtered),
                100 * (1 - len(filtered) / original_n) if original_n > 0 else 0,
            )

        logger.info(
            "SPEBergDataset (%s): %d samples, %d wells, input=%d, horizon=%d, n_vars=%d",
            self.mode, len(self._samples), len(self._well_data),
            self.input_window, self.horizon, self.n_vars,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return a single forecasting sample.

        Returns:
            Tuple of ``(input_tensor[input_window, n_vars], target_tensor[horizon], metadata)``.
        """
        well_idx, start = self._samples[index]
        well_name, _ = self._well_data[well_idx]
        arr = self._arrays[well_idx]
        input_end = start + self.input_window
        target_end = input_end + self.horizon

        # Input: all features over the input window
        input_data = arr[start:input_end]
        input_tensor = torch.from_numpy(input_data.copy())

        # Target: only Gas_Volume_MMscf over the forecast horizon
        target_data = arr[input_end:target_end, self._target_col_idx]
        target_tensor = torch.from_numpy(target_data.copy())

        metadata = {
            "well_name": well_name,
            "well_idx": well_idx,
            "start_idx": start,
            "input_end": input_end,
            "target_end": target_end,
            "horizon": self.horizon,
            "mode": self.mode,
        }

        return input_tensor, target_tensor, metadata

    def __len__(self) -> int:
        return len(self._samples)

    def get_metadata(self) -> dict:
        """Return dataset-level metadata."""
        well_counts: dict[str, int] = {}
        for well_idx, _ in self._samples:
            name = self._well_data[well_idx][0]
            well_counts[name] = well_counts.get(name, 0) + 1

        return {
            "class": "SPEBergDataset",
            "length": len(self),
            "n_vars": self.n_vars,
            "input_window": self.input_window,
            "horizon": self.horizon,
            "mode": self.mode,
            "well_counts": well_counts,
            "target_col": self.target_col,
            "target_col_idx": self._target_col_idx,
        }


class VolveDataset(BaseDataset):
    """Volve Norwegian North Sea oil production forecasting dataset.

    Supports per-well and multi-well modes with configurable forecast horizons.
    Reads preprocessed per-well parquets from ``data/processed/volve/``
    (produced by ``preprocess_volve``).

    Returns ``(input_tensor[w, n_vars], target_tensor[h], metadata)`` where
    the target is the ``BORE_OIL_VOL`` column over the forecast horizon.

    If the preprocessed parquets are missing, preprocessing is run automatically.

    6 production wells from the Equinor Volve field (F-4 AH excluded —
    injection-only well with zero oil production).

    Args:
        config: Path to Volve YAML config or pre-loaded DictConfig.
        base_config: Path to base config.
        mode: ``"per_well"`` or ``"multi_well"``. Overrides config default.
        well_name: Specific well filename stem to load (per_well mode only,
            e.g. ``"NO_15_9-F-1_C"``).
        horizon: Override forecast horizon from config.
        input_window: Override input window size from config.
        filter_shutdowns: If True, drop samples whose target window is all-zero
            (well offline / shutdown).
    """

    def __init__(
        self,
        config: str | Path | DictConfig,
        base_config: str | Path = "configs/base.yaml",
        mode: str = "multi_well",
        well_name: str | None = None,
        horizon: int | None = None,
        input_window: int | None = None,
        filter_shutdowns: bool = False,
    ) -> None:
        if isinstance(config, (str, Path)):
            self.cfg = load_merged_config(base_config, config)
        else:
            self.cfg = config

        self.mode = mode
        self.well_name = well_name
        self.horizon = horizon or self.cfg.data.forecasting.default_horizon
        self.input_window = input_window or self.cfg.data.forecasting.input_window
        self.target_col = self.cfg.data.target_column
        self.filter_shutdowns = filter_shutdowns

        self._load_data()

    def _load_data(self) -> None:
        """Load processed Volve parquets and build sample index."""
        processed_dir = Path(self.cfg.data.paths.processed)

        if not processed_dir.exists() or not any(processed_dir.glob("*.parquet")):
            from offshore_dl.data.preprocess_volve import preprocess_volve
            preprocess_volve(self.cfg)

        # Load wells
        self._well_data: list[tuple[str, pd.DataFrame]] = []

        if self.mode == "per_well" and self.well_name:
            path = processed_dir / f"{self.well_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                self._well_data.append((self.well_name, df))
            else:
                logger.warning(
                    "VolveDataset: well parquet not found: %s — loading all wells instead", path
                )
                for p in sorted(processed_dir.glob("*.parquet")):
                    df = pd.read_parquet(p)
                    self._well_data.append((p.stem, df))
        else:
            # Multi-well: load all wells sorted by filename stem
            for path in sorted(processed_dir.glob("*.parquet")):
                df = pd.read_parquet(path)
                self._well_data.append((path.stem, df))

        # Build sample index: (well_idx, start_idx)
        self._samples: list[tuple[int, int]] = []
        total_needed = self.input_window + self.horizon

        for well_idx, (name, df) in enumerate(self._well_data):
            n = len(df)
            for start in range(0, max(0, n - total_needed + 1)):
                self._samples.append((well_idx, start))

        # Align all wells to a common sorted column set (union, fill missing with 0)
        if self._well_data:
            all_cols: list[set[str]] = [set(df.columns) for _, df in self._well_data]
            self._common_columns: list[str] = sorted(set.union(*all_cols))
        else:
            self._common_columns = []

        # Compute target column index in the ALIGNED (sorted) column order
        if self.target_col in self._common_columns:
            self._target_col_idx = self._common_columns.index(self.target_col)
        else:
            self._target_col_idx = 0
            logger.warning(
                "VolveDataset: target column '%s' not found in aligned columns, using col 0",
                self.target_col,
            )

        # Precompute numpy arrays with aligned columns (avoids per-getitem reindex)
        self._arrays: list[np.ndarray] = []
        for _, df in self._well_data:
            aligned = df.reindex(columns=self._common_columns, fill_value=0.0)
            arr = aligned.values.astype(np.float32)
            np.nan_to_num(arr, copy=False, nan=0.0)
            self._arrays.append(arr)

        self.n_vars = len(self._common_columns)

        # ── Filter shutdown windows ──
        # Drop samples where the entire target horizon is zero (well offline).
        if self.filter_shutdowns and self._arrays:
            original_n = len(self._samples)
            filtered = []
            for well_idx, start in self._samples:
                arr = self._arrays[well_idx]
                input_end = start + self.input_window
                target_end = input_end + self.horizon
                target_vals = arr[input_end:target_end, self._target_col_idx]
                if np.any(target_vals > 0.01):
                    filtered.append((well_idx, start))
            self._samples = filtered
            logger.info(
                "VolveDataset: filtered shutdowns: %d → %d samples (%.0f%% removed)",
                original_n, len(filtered),
                100 * (1 - len(filtered) / original_n) if original_n > 0 else 0,
            )

        logger.info(
            "VolveDataset (%s): %d samples, %d wells, input=%d, horizon=%d, n_vars=%d",
            self.mode, len(self._samples), len(self._well_data),
            self.input_window, self.horizon, self.n_vars,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Return a single forecasting sample.

        Returns:
            Tuple of ``(input_tensor[input_window, n_vars], target_tensor[horizon], metadata)``.
        """
        well_idx, start = self._samples[index]
        well_name, _ = self._well_data[well_idx]
        arr = self._arrays[well_idx]
        input_end = start + self.input_window
        target_end = input_end + self.horizon

        # Input: all features over the input window
        input_data = arr[start:input_end]
        input_tensor = torch.from_numpy(input_data.copy())

        # Target: only BORE_OIL_VOL over the forecast horizon
        target_data = arr[input_end:target_end, self._target_col_idx]
        target_tensor = torch.from_numpy(target_data.copy())

        metadata = {
            "well_name": well_name,
            "well_idx": well_idx,
            "start_idx": start,
            "input_end": input_end,
            "target_end": target_end,
            "horizon": self.horizon,
            "mode": self.mode,
        }

        return input_tensor, target_tensor, metadata

    def __len__(self) -> int:
        return len(self._samples)

    def get_metadata(self) -> dict:
        """Return dataset-level metadata."""
        well_counts: dict[str, int] = {}
        for well_idx, _ in self._samples:
            name = self._well_data[well_idx][0]
            well_counts[name] = well_counts.get(name, 0) + 1

        return {
            "class": "VolveDataset",
            "length": len(self),
            "n_vars": self.n_vars,
            "input_window": self.input_window,
            "horizon": self.horizon,
            "mode": self.mode,
            "well_counts": well_counts,
            "target_col": self.target_col,
            "target_col_idx": self._target_col_idx,
        }
