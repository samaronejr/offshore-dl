"""Tests for the Volve data pipeline and VolveDataset.

Covers:
- Preprocessing output: 6 per-well parquets, schema, no NaN, EMA/ROC columns
- VolveDataset multi-well mode: shapes, types, target column, NaN-free
- VolveDataset per-well mode: single-well isolation, horizon shapes
- Edge cases: F-5 AH boundary (160 rows), non-existent well fallback, multiple horizons
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pytest
import torch

from offshore_dl.data.datasets import VolveDataset


# ════════════════════════════════════════════════════════════════════
# Module-level singletons — loaded once, reused across all tests
# ════════════════════════════════════════════════════════════════════

_volve_multi_cache: VolveDataset | None = None
_volve_per_well_cache: VolveDataset | None = None


def _get_multi_well(project_root: Path) -> VolveDataset:
    """Singleton multi-well dataset — loads once, reused across all tests."""
    global _volve_multi_cache
    if _volve_multi_cache is None:
        config_path = project_root / "configs" / "data" / "volve.yaml"
        base_path = project_root / "configs" / "base.yaml"
        _volve_multi_cache = VolveDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="multi_well",
            horizon=30,
            input_window=90,
        )
    return _volve_multi_cache


def _get_per_well(project_root: Path) -> VolveDataset:
    """Singleton per-well dataset for NO_15_9-F-12_H — loads once, reused."""
    global _volve_per_well_cache
    if _volve_per_well_cache is None:
        config_path = project_root / "configs" / "data" / "volve.yaml"
        base_path = project_root / "configs" / "base.yaml"
        _volve_per_well_cache = VolveDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="per_well",
            well_name="NO_15_9-F-12_H",
            horizon=7,
            input_window=90,
        )
    return _volve_per_well_cache


# ════════════════════════════════════════════════════════════════════
# TestVolvePreprocessing
# ════════════════════════════════════════════════════════════════════


class TestVolvePreprocessing:
    """Verify the preprocessing pipeline output: 6 parquets, schema, quality."""

    @pytest.fixture
    def volve_dir(self, project_root: Path) -> Path:
        return project_root / "data" / "processed" / "volve"

    @pytest.fixture
    def sample_df(self, volve_dir: Path) -> pd.DataFrame:
        """Load NO_15_9-F-12_H.parquet as a representative sample."""
        return pd.read_parquet(volve_dir / "NO_15_9-F-12_H.parquet")

    # ── Existence / count ──────────────────────────────────────────

    def test_6_parquets_exist(self, volve_dir: Path) -> None:
        """Preprocessing must produce exactly 6 per-well parquets."""
        files = list(volve_dir.glob("*.parquet"))
        assert len(files) == 6, f"Expected 6 parquets, got {len(files)}"

    def test_f4_excluded(self, volve_dir: Path) -> None:
        """F-4 AH injection well must NOT appear in the output (filtered by FLOW_KIND)."""
        f4_path = volve_dir / "NO_15_9-F-4_AH.parquet"
        assert not f4_path.exists(), (
            "NO_15_9-F-4_AH.parquet must be excluded (injection-only well)"
        )

    # ── Schema / column quality ────────────────────────────────────

    def test_parquet_has_no_nan(self, sample_df: pd.DataFrame) -> None:
        """No NaN values allowed — preprocessing fills all gaps."""
        total_nan = sample_df.isna().sum().sum()
        assert total_nan == 0, (
            f"Found {total_nan} NaN values in NO_15_9-F-12_H.parquet"
        )

    def test_ema_columns_exist(self, sample_df: pd.DataFrame) -> None:
        """EMA-smoothed columns must be generated for BORE_OIL_VOL."""
        assert "BORE_OIL_VOL_ema_7" in sample_df.columns, "Missing BORE_OIL_VOL_ema_7"
        assert "BORE_OIL_VOL_ema_14" in sample_df.columns, "Missing BORE_OIL_VOL_ema_14"
        assert "BORE_OIL_VOL_ema_30" in sample_df.columns, "Missing BORE_OIL_VOL_ema_30"
        assert "BORE_OIL_VOL_ema_90" in sample_df.columns, "Missing BORE_OIL_VOL_ema_90"

    def test_roc_columns_exist(self, sample_df: pd.DataFrame) -> None:
        """ROC columns must be generated for dynamic series."""
        assert "BORE_OIL_VOL_roc" in sample_df.columns, "Missing BORE_OIL_VOL_roc"

    def test_target_column_exists(self, sample_df: pd.DataFrame) -> None:
        """BORE_OIL_VOL (the forecast target) must be in every parquet."""
        assert "BORE_OIL_VOL" in sample_df.columns

    def test_datetime_index_monotonic(self, sample_df: pd.DataFrame) -> None:
        """DatetimeIndex must be monotonically increasing."""
        assert isinstance(sample_df.index, pd.DatetimeIndex), (
            f"Index type is {type(sample_df.index)}, expected DatetimeIndex"
        )
        assert sample_df.index.is_monotonic_increasing, (
            "DatetimeIndex is not monotonically increasing"
        )

    def test_all_wells_same_column_count(self, volve_dir: Path) -> None:
        """All 6 wells should expose the same 73-column schema."""
        files = sorted(volve_dir.glob("*.parquet"))
        col_counts = {p.stem: len(pd.read_parquet(p).columns) for p in files}
        unique_counts = set(col_counts.values())
        assert len(unique_counts) == 1, (
            f"Wells have inconsistent column counts: {col_counts}"
        )
        (n_cols,) = unique_counts
        assert n_cols == 73, f"Expected 73 columns, got {n_cols}"

    def test_shutdown_flag_exists(self, sample_df: pd.DataFrame) -> None:
        """is_shutdown column must be present (marks well-offline days)."""
        assert "is_shutdown" in sample_df.columns, "Missing is_shutdown column"


# ════════════════════════════════════════════════════════════════════
# TestVolveDatasetMultiWell
# ════════════════════════════════════════════════════════════════════


class TestVolveDatasetMultiWell:
    """Tests for VolveDataset in multi_well mode (horizon=30, window=90)."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> VolveDataset:
        return _get_multi_well(project_root)

    # ── Length / existence ─────────────────────────────────────────

    def test_dataset_not_empty(self, dataset: VolveDataset) -> None:
        assert len(dataset) > 0

    def test_6_wells_loaded(self, dataset: VolveDataset) -> None:
        """All 6 production wells must be loaded in multi_well mode."""
        assert len(dataset._well_data) == 6, (
            f"Expected 6 wells, got {len(dataset._well_data)}"
        )

    # ── Return type contract ───────────────────────────────────────

    def test_returns_correct_types(self, dataset: VolveDataset) -> None:
        """__getitem__ must return (Tensor, Tensor, dict)."""
        inp, tgt, meta = dataset[0]
        assert isinstance(inp, torch.Tensor)
        assert isinstance(tgt, torch.Tensor)
        assert isinstance(meta, dict)

    # ── Tensor shapes ──────────────────────────────────────────────

    def test_input_shape(self, dataset: VolveDataset) -> None:
        """Input tensor must be [input_window=90, n_vars]."""
        inp, _, _ = dataset[0]
        assert inp.shape[0] == 90, f"Expected 90 time steps, got {inp.shape[0]}"
        assert inp.shape[1] == dataset.n_vars

    def test_target_shape(self, dataset: VolveDataset) -> None:
        """Target tensor must be [horizon=30] for h30."""
        _, tgt, _ = dataset[0]
        assert tgt.shape == (30,), f"Expected shape (30,), got {tgt.shape}"

    def test_target_is_1d(self, dataset: VolveDataset) -> None:
        """Target must be a 1-D tensor (one scalar per forecast step)."""
        _, tgt, _ = dataset[0]
        assert tgt.dim() == 1

    # ── Feature dimensionality ─────────────────────────────────────

    def test_n_vars_is_73(self, dataset: VolveDataset) -> None:
        """Volve preprocessing produces exactly 73 feature columns per well."""
        assert dataset.n_vars == 73, f"Expected 73, got {dataset.n_vars}"

    # ── Target column index ────────────────────────────────────────

    def test_target_col_idx_is_bore_oil_vol(self, dataset: VolveDataset) -> None:
        """_common_columns[_target_col_idx] must resolve to BORE_OIL_VOL."""
        assert dataset._common_columns[dataset._target_col_idx] == "BORE_OIL_VOL", (
            f"Expected 'BORE_OIL_VOL', got "
            f"'{dataset._common_columns[dataset._target_col_idx]}'"
        )

    def test_target_col_idx_is_48(self, dataset: VolveDataset) -> None:
        """BORE_OIL_VOL lands at sorted index 48 in the Volve column set."""
        assert dataset._target_col_idx == 48, (
            f"Expected index 48, got {dataset._target_col_idx}"
        )

    # ── NaN / data quality ─────────────────────────────────────────

    def test_no_nan_in_samples(self, dataset: VolveDataset) -> None:
        """First 10 samples must contain no NaN in either input or target."""
        for i in range(min(10, len(dataset))):
            inp, tgt, _ = dataset[i]
            assert not torch.isnan(inp).any(), f"NaN in input tensor at index {i}"
            assert not torch.isnan(tgt).any(), f"NaN in target tensor at index {i}"

    # ── Metadata contract ─────────────────────────────────────────

    def test_metadata_has_required_keys(self, dataset: VolveDataset) -> None:
        """Sample metadata dict must include all required keys."""
        _, _, meta = dataset[0]
        for key in ("well_name", "well_idx", "start_idx", "horizon", "mode"):
            assert key in meta, f"Missing metadata key: '{key}'"

    def test_get_metadata_summary(self, dataset: VolveDataset) -> None:
        """Dataset-level get_metadata() must reflect class, length, and n_vars."""
        meta = dataset.get_metadata()
        assert meta["class"] == "VolveDataset"
        assert meta["length"] == len(dataset)
        assert meta["n_vars"] == dataset.n_vars
        assert meta["target_col"] == "BORE_OIL_VOL"


# ════════════════════════════════════════════════════════════════════
# TestVolveDatasetPerWell
# ════════════════════════════════════════════════════════════════════


class TestVolveDatasetPerWell:
    """Tests for VolveDataset in per_well mode (NO_15_9-F-12_H, horizon=7)."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> VolveDataset:
        return _get_per_well(project_root)

    def test_single_well_only(self, dataset: VolveDataset) -> None:
        """Per-well mode with a valid well name must expose only that one well."""
        meta = dataset.get_metadata()
        assert len(meta["well_counts"]) == 1, (
            f"Expected 1 well, got {len(meta['well_counts'])}: {meta['well_counts']}"
        )

    def test_correct_horizon_h7(self, dataset: VolveDataset) -> None:
        """Target shape must be (7,) when horizon=7."""
        _, tgt, _ = dataset[0]
        assert tgt.shape == (7,), f"Expected shape (7,), got {tgt.shape}"

    def test_dataset_not_empty(self, dataset: VolveDataset) -> None:
        assert len(dataset) > 0

    def test_correct_well_name_in_metadata(self, dataset: VolveDataset) -> None:
        """Sample metadata well_name must match the requested well."""
        _, _, meta = dataset[0]
        assert meta["well_name"] == "NO_15_9-F-12_H"


# ════════════════════════════════════════════════════════════════════
# TestVolveDatasetEdgeCases
# ════════════════════════════════════════════════════════════════════


class TestVolveDatasetEdgeCases:
    """Edge cases: F-5 AH boundary, non-existent well fallback, multiple horizons."""

    @pytest.fixture
    def config_path(self, project_root: Path) -> str:
        return str(project_root / "configs" / "data" / "volve.yaml")

    @pytest.fixture
    def base_path(self, project_root: Path) -> str:
        return str(project_root / "configs" / "base.yaml")

    def test_f5_ah_h90_empty(self, config_path: str, base_path: str) -> None:
        """F-5 AH (160 rows) must yield 0 samples for h90 (needs 180 rows minimum).

        total_needed = input_window(90) + horizon(90) = 180 > 160 rows.
        """
        ds = VolveDataset(
            config=config_path,
            base_config=base_path,
            mode="per_well",
            well_name="NO_15_9-F-5_AH",
            horizon=90,
            input_window=90,
        )
        assert len(ds) == 0, (
            f"F-5 AH (160 rows) with h90+w90 must yield 0 samples, got {len(ds)}"
        )

    def test_f5_ah_h7_nonempty(self, config_path: str, base_path: str) -> None:
        """F-5 AH (160 rows) must yield > 0 samples for h7 (needs 97 rows minimum).

        total_needed = input_window(90) + horizon(7) = 97 <= 160 rows.
        """
        ds = VolveDataset(
            config=config_path,
            base_config=base_path,
            mode="per_well",
            well_name="NO_15_9-F-5_AH",
            horizon=7,
            input_window=90,
        )
        assert len(ds) > 0, (
            f"F-5 AH (160 rows) with h7+w90 should yield > 0 samples, got {len(ds)}"
        )

    def test_multiple_horizons_correct_shapes(
        self, config_path: str, base_path: str
    ) -> None:
        """Each horizon value must produce a correctly sized target tensor."""
        horizons = [7, 14, 30, 90]
        for h in horizons:
            ds = VolveDataset(
                config=config_path,
                base_config=base_path,
                mode="multi_well",
                horizon=h,
                input_window=90,
            )
            assert len(ds) > 0, f"Dataset empty for horizon={h}"
            _, tgt, _ = ds[0]
            assert tgt.shape == (h,), (
                f"horizon={h}: expected shape ({h},), got {tgt.shape}"
            )

    def test_nonexistent_well_fallback(self, config_path: str, base_path: str) -> None:
        """Per-well mode with a non-existent well name falls back to all 6 wells.

        VolveDataset logs a warning and loads all 6 wells rather than returning
        an empty dataset (mirrors SPEBergDataset fallback behaviour).
        """
        ds = VolveDataset(
            config=config_path,
            base_config=base_path,
            mode="per_well",
            well_name="nonexistent_well_xyz",
            horizon=30,
            input_window=90,
        )
        # Fallback: loads all 6 wells → dataset is NOT empty
        assert len(ds) > 0, "Fallback to all wells should produce non-empty dataset"
        assert len(ds._well_data) == 6, (
            f"Expected fallback to 6 wells, got {len(ds._well_data)}"
        )
