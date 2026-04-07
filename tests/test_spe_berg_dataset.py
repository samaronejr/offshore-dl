"""Tests for the SPE BERG data pipeline and SPEBergDataset.

Covers:
- Preprocessing output: 53 per-well parquets, schema, no NaN, static/EMA columns
- SPEBergDataset multi-well mode: shapes, types, target column, NaN-free
- SPEBergDataset per-well mode: single-well isolation, horizon shapes
- Edge cases: non-existent well fallback, multiple horizons
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
import pytest
import torch

from offshore_dl.data.datasets import SPEBergDataset


# ════════════════════════════════════════════════════════════════════
# Module-level singletons — loaded once, reused across all tests
# ════════════════════════════════════════════════════════════════════

_spe_multi_cache: SPEBergDataset | None = None
_spe_per_well_cache: SPEBergDataset | None = None


def _get_multi_well(project_root: Path) -> SPEBergDataset:
    """Singleton multi-well dataset — loads once, reused across all tests."""
    global _spe_multi_cache
    if _spe_multi_cache is None:
        config_path = project_root / "configs" / "data" / "spe_berg.yaml"
        base_path = project_root / "configs" / "base.yaml"
        _spe_multi_cache = SPEBergDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="multi_well",
            horizon=30,
            input_window=90,
        )
    return _spe_multi_cache


def _get_per_well(project_root: Path) -> SPEBergDataset:
    """Singleton per-well dataset for well_1 — loads once, reused."""
    global _spe_per_well_cache
    if _spe_per_well_cache is None:
        config_path = project_root / "configs" / "data" / "spe_berg.yaml"
        base_path = project_root / "configs" / "base.yaml"
        _spe_per_well_cache = SPEBergDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="per_well",
            well_name="well_1",
            horizon=7,
            input_window=90,
        )
    return _spe_per_well_cache


# ════════════════════════════════════════════════════════════════════
# TestSPEBergPreprocessing
# ════════════════════════════════════════════════════════════════════


class TestSPEBergPreprocessing:
    """Verify the preprocessing pipeline output: 53 parquets, schema, quality."""

    @pytest.fixture
    def spe_dir(self, project_root: Path) -> Path:
        return project_root / "data" / "processed" / "spe_berg"

    @pytest.fixture
    def sample_df(self, spe_dir: Path) -> pd.DataFrame:
        """Load well_1.parquet as a representative sample."""
        return pd.read_parquet(spe_dir / "well_1.parquet")

    # ── Existence / count ──────────────────────────────────────────

    def test_all_53_parquets_exist(self, spe_dir: Path) -> None:
        """Preprocessing must produce exactly 53 per-well parquets."""
        files = list(spe_dir.glob("*.parquet"))
        assert len(files) == 53, f"Expected 53 parquets, got {len(files)}"

    # ── Schema / column quality ────────────────────────────────────

    def test_parquet_has_no_nan(self, sample_df: pd.DataFrame) -> None:
        """No NaN values allowed — preprocessing fills all gaps."""
        total_nan = sample_df.isna().sum().sum()
        assert total_nan == 0, f"Found {total_nan} NaN values in well_1.parquet"

    def test_static_features_present(self, sample_df: pd.DataFrame) -> None:
        """All six static geological / completion features must be present."""
        required_static = [
            "static_Stages",
            "static_Lateral_Length_ft",
            "static_TVD_ft",
            "static_Porosity",
            "static_Gas_Saturation",
            "static_Initial_Pressure_psi",
        ]
        for col in required_static:
            assert col in sample_df.columns, f"Missing static column: {col}"

    def test_static_features_constant(self, sample_df: pd.DataFrame) -> None:
        """Static features must be constant per well (std ≈ 0, within float32 noise).

        Note: floating-point broadcast of integer static values can introduce
        sub-1e-10 rounding noise; we allow a generous tolerance.
        """
        static_cols = [c for c in sample_df.columns if c.startswith("static_")]
        assert len(static_cols) > 0, "No static_ columns found"
        for col in static_cols:
            std = sample_df[col].std()
            assert std < 1e-6, (
                f"Static column '{col}' has non-negligible variation (std={std})"
            )

    def test_ema_columns_exist(self, sample_df: pd.DataFrame) -> None:
        """EMA-smoothed columns must be generated for Gas_Volume_MMscf."""
        assert "Gas_Volume_MMscf_ema_7" in sample_df.columns
        assert "Gas_Volume_MMscf_ema_14" in sample_df.columns
        assert "Gas_Volume_MMscf_ema_30" in sample_df.columns
        assert "Gas_Volume_MMscf_ema_90" in sample_df.columns

    def test_roc_columns_exist(self, sample_df: pd.DataFrame) -> None:
        """ROC columns must be generated for dynamic series."""
        assert "Gas_Volume_MMscf_roc" in sample_df.columns

    def test_target_column_exists(self, sample_df: pd.DataFrame) -> None:
        """Gas_Volume_MMscf (the forecast target) must be in every parquet."""
        assert "Gas_Volume_MMscf" in sample_df.columns

    def test_day_index_monotonic(self, sample_df: pd.DataFrame) -> None:
        """DatetimeIndex must be monotonically increasing."""
        assert isinstance(sample_df.index, pd.DatetimeIndex), (
            f"Index type is {type(sample_df.index)}, expected DatetimeIndex"
        )
        assert sample_df.index.is_monotonic_increasing, (
            "DatetimeIndex is not monotonically increasing"
        )

    def test_all_wells_have_same_column_count(self, spe_dir: Path) -> None:
        """All 53 wells should expose the same 67-column schema."""
        files = sorted(spe_dir.glob("*.parquet"))
        col_counts = {p.stem: len(pd.read_parquet(p).columns) for p in files}
        unique_counts = set(col_counts.values())
        assert len(unique_counts) == 1, (
            f"Wells have inconsistent column counts: {unique_counts}"
        )
        (n_cols,) = unique_counts
        assert n_cols == 67, f"Expected 67 columns, got {n_cols}"


# ════════════════════════════════════════════════════════════════════
# TestSPEBergDatasetMultiWell
# ════════════════════════════════════════════════════════════════════


class TestSPEBergDatasetMultiWell:
    """Tests for SPEBergDataset in multi_well mode (horizon=30, window=90)."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> SPEBergDataset:
        return _get_multi_well(project_root)

    # ── Length / existence ─────────────────────────────────────────

    def test_dataset_not_empty(self, dataset: SPEBergDataset) -> None:
        assert len(dataset) > 0

    def test_all_53_wells_loaded(self, dataset: SPEBergDataset) -> None:
        """All 53 wells must be loaded in multi_well mode."""
        assert len(dataset._well_data) == 53

    # ── Return type contract ───────────────────────────────────────

    def test_returns_correct_types(self, dataset: SPEBergDataset) -> None:
        """__getitem__ must return (Tensor, Tensor, dict)."""
        inp, tgt, meta = dataset[0]
        assert isinstance(inp, torch.Tensor)
        assert isinstance(tgt, torch.Tensor)
        assert isinstance(meta, dict)

    # ── Tensor shapes ──────────────────────────────────────────────

    def test_input_shape(self, dataset: SPEBergDataset) -> None:
        """Input tensor must be [input_window=90, n_vars]."""
        inp, _, _ = dataset[0]
        assert inp.shape[0] == 90, f"Expected 90 time steps, got {inp.shape[0]}"
        assert inp.shape[1] == dataset.n_vars

    def test_target_shape(self, dataset: SPEBergDataset) -> None:
        """Target tensor must be [horizon=30] for h30."""
        _, tgt, _ = dataset[0]
        assert tgt.shape == (30,), f"Expected shape (30,), got {tgt.shape}"

    def test_target_is_1d(self, dataset: SPEBergDataset) -> None:
        """Target must be a 1-D tensor (one scalar per forecast step)."""
        _, tgt, _ = dataset[0]
        assert tgt.dim() == 1

    # ── Feature dimensionality ────────────────────────────────────

    def test_n_vars_includes_static(self, dataset: SPEBergDataset) -> None:
        """n_vars must exceed 40 — includes static, EMA, and ROC columns."""
        assert dataset.n_vars > 40, f"n_vars={dataset.n_vars} seems too low"

    def test_n_vars_is_67(self, dataset: SPEBergDataset) -> None:
        """SPE BERG produces exactly 67 feature columns per well."""
        assert dataset.n_vars == 67, f"Expected 67, got {dataset.n_vars}"

    # ── Target column index ───────────────────────────────────────

    def test_target_col_idx_is_gas_volume(self, dataset: SPEBergDataset) -> None:
        """_common_columns[_target_col_idx] must resolve to Gas_Volume_MMscf."""
        assert dataset._common_columns[dataset._target_col_idx] == "Gas_Volume_MMscf", (
            f"Expected 'Gas_Volume_MMscf', got "
            f"'{dataset._common_columns[dataset._target_col_idx]}'"
        )

    def test_target_col_idx_not_zero(self, dataset: SPEBergDataset) -> None:
        """Per D028: Gas_Volume_MMscf is NOT alphabetically first (idx must be > 0)."""
        assert dataset._target_col_idx > 0, (
            f"_target_col_idx={dataset._target_col_idx} — Gas_Volume_MMscf cannot "
            f"be at index 0 in alphabetical sort"
        )

    def test_target_col_idx_is_30(self, dataset: SPEBergDataset) -> None:
        """Gas_Volume_MMscf lands at sorted index 30 in the SPE BERG column union."""
        assert dataset._target_col_idx == 30, (
            f"Expected index 30, got {dataset._target_col_idx}"
        )

    # ── NaN / data quality ─────────────────────────────────────────

    def test_no_nan_in_samples(self, dataset: SPEBergDataset) -> None:
        """First 10 samples must contain no NaN in either input or target."""
        for i in range(min(10, len(dataset))):
            inp, tgt, _ = dataset[i]
            assert not torch.isnan(inp).any(), f"NaN in input tensor at index {i}"
            assert not torch.isnan(tgt).any(), f"NaN in target tensor at index {i}"

    # ── Metadata contract ─────────────────────────────────────────

    def test_metadata_has_required_keys(self, dataset: SPEBergDataset) -> None:
        """Sample metadata dict must include all required keys."""
        _, _, meta = dataset[0]
        for key in ("well_name", "well_idx", "start_idx", "horizon", "mode"):
            assert key in meta, f"Missing metadata key: '{key}'"

    def test_get_metadata_summary(self, dataset: SPEBergDataset) -> None:
        """Dataset-level get_metadata() must reflect class, length, and n_vars."""
        meta = dataset.get_metadata()
        assert meta["class"] == "SPEBergDataset"
        assert meta["length"] == len(dataset)
        assert meta["n_vars"] == dataset.n_vars
        assert meta["target_col"] == "Gas_Volume_MMscf"

    def test_metadata_well_counts_covers_all_wells(self, dataset: SPEBergDataset) -> None:
        """well_counts in get_metadata must list all 53 loaded wells."""
        meta = dataset.get_metadata()
        assert len(meta["well_counts"]) == 53


# ════════════════════════════════════════════════════════════════════
# TestSPEBergDatasetPerWell
# ════════════════════════════════════════════════════════════════════


class TestSPEBergDatasetPerWell:
    """Tests for SPEBergDataset in per_well mode (well_1, horizon=7)."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> SPEBergDataset:
        return _get_per_well(project_root)

    def test_single_well_only(self, dataset: SPEBergDataset) -> None:
        """Per-well mode with a valid well name must expose only that one well."""
        meta = dataset.get_metadata()
        assert len(meta["well_counts"]) == 1, (
            f"Expected 1 well, got {len(meta['well_counts'])}: {meta['well_counts']}"
        )

    def test_correct_horizon_h7(self, dataset: SPEBergDataset) -> None:
        """Target shape must be (7,) when horizon=7."""
        _, tgt, _ = dataset[0]
        assert tgt.shape == (7,), f"Expected shape (7,), got {tgt.shape}"

    def test_dataset_not_empty(self, dataset: SPEBergDataset) -> None:
        assert len(dataset) > 0

    def test_correct_well_name_in_metadata(self, dataset: SPEBergDataset) -> None:
        """Sample metadata well_name must match the requested well."""
        _, _, meta = dataset[0]
        assert meta["well_name"] == "well_1"


# ════════════════════════════════════════════════════════════════════
# TestSPEBergDatasetEdgeCases
# ════════════════════════════════════════════════════════════════════


class TestSPEBergDatasetEdgeCases:
    """Edge cases: non-existent well fallback, multiple horizons, boundary conditions."""

    @pytest.fixture
    def config_path(self, project_root: Path) -> str:
        return str(project_root / "configs" / "data" / "spe_berg.yaml")

    @pytest.fixture
    def base_path(self, project_root: Path) -> str:
        return str(project_root / "configs" / "base.yaml")

    def test_nonexistent_well_falls_back_to_all_wells(
        self, config_path: str, base_path: str
    ) -> None:
        """Per-well mode with a non-existent well name falls back to all 53 wells.

        The implementation intentionally loads all wells rather than returning an
        empty dataset, as documented in T02-SUMMARY (D029 / fallback design).
        """
        ds = SPEBergDataset(
            config=config_path,
            base_config=base_path,
            mode="per_well",
            well_name="nonexistent_well_xyz",
            horizon=30,
            input_window=90,
        )
        # Fallback: loads all 53 wells → dataset is NOT empty
        assert len(ds) > 0, "Fallback to all wells should produce non-empty dataset"
        assert len(ds._well_data) == 53, (
            f"Expected fallback to 53 wells, got {len(ds._well_data)}"
        )

    def test_multiple_horizons_correct_target_sizes(
        self, config_path: str, base_path: str
    ) -> None:
        """Each horizon value must produce a correctly sized target tensor."""
        horizons = [7, 14, 30, 90]
        for h in horizons:
            ds = SPEBergDataset(
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

    def test_h90_boundary_shortest_well(
        self, config_path: str, base_path: str
    ) -> None:
        """Shortest well (249 rows) must still yield samples with h90 + window=90."""
        ds = SPEBergDataset(
            config=config_path,
            base_config=base_path,
            mode="per_well",
            well_name="well_11",  # 249 rows — shortest well
            horizon=90,
            input_window=90,
        )
        # 249 >= 90+90 = 180, so at least 70 samples should exist
        assert len(ds) > 0, "Shortest well (249 rows) should yield samples for h90"

    def test_h7_target_size(self, config_path: str, base_path: str) -> None:
        """h7 target must have shape (7,)."""
        ds = SPEBergDataset(
            config=config_path,
            base_config=base_path,
            mode="multi_well",
            horizon=7,
            input_window=90,
        )
        _, tgt, _ = ds[0]
        assert tgt.shape == (7,)

    def test_h90_target_size(self, config_path: str, base_path: str) -> None:
        """h90 target must have shape (90,)."""
        ds = SPEBergDataset(
            config=config_path,
            base_config=base_path,
            mode="multi_well",
            horizon=90,
            input_window=90,
        )
        _, tgt, _ = ds[0]
        assert tgt.shape == (90,)

    def test_input_shape_consistent_across_horizons(
        self, config_path: str, base_path: str
    ) -> None:
        """Input shape must be (90, n_vars) regardless of horizon."""
        for h in [7, 14, 30, 90]:
            ds = SPEBergDataset(
                config=config_path,
                base_config=base_path,
                mode="multi_well",
                horizon=h,
                input_window=90,
            )
            inp, _, _ = ds[0]
            assert inp.shape[0] == 90, (
                f"horizon={h}: input window should be 90, got {inp.shape[0]}"
            )
            assert inp.shape[1] == 67, (
                f"horizon={h}: n_vars should be 67, got {inp.shape[1]}"
            )
