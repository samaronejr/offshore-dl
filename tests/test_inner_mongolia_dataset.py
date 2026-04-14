"""Tests for the Inner Mongolia data pipeline and InnerMongoliaDataset.

Covers:
- Preprocessing output: 29 per-well parquets, schema, no NaN, EMA/ROC columns
- InnerMongoliaDataset multi-well mode: shapes, types, target column, NaN-free
- InnerMongoliaDataset per-well mode: single-well isolation, horizon shapes
- Edge cases: short well (57-15X at 181 days), filtered well (56-14X), multiple horizons
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from offshore_dl.data.datasets import InnerMongoliaDataset


# ════════════════════════════════════════════════════════════════════
# Module-level singletons — loaded once, reused across all tests
# ════════════════════════════════════════════════════════════════════

_im_multi_cache: InnerMongoliaDataset | None = None
_im_per_well_cache: InnerMongoliaDataset | None = None


def _get_multi_well(project_root: Path) -> InnerMongoliaDataset:
    """Singleton multi-well dataset — loads once, reused across all tests."""
    global _im_multi_cache
    if _im_multi_cache is None:
        config_path = project_root / "configs" / "data" / "inner_mongolia.yaml"
        base_path = project_root / "configs" / "base.yaml"
        _im_multi_cache = InnerMongoliaDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="multi_well",
            horizon=30,
            input_window=90,
        )
    return _im_multi_cache


def _get_per_well(project_root: Path) -> InnerMongoliaDataset:
    """Singleton per-well dataset for 58-25 — loads once, reused."""
    global _im_per_well_cache
    if _im_per_well_cache is None:
        config_path = project_root / "configs" / "data" / "inner_mongolia.yaml"
        base_path = project_root / "configs" / "base.yaml"
        _im_per_well_cache = InnerMongoliaDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="per_well",
            well_name="58-25",
            horizon=7,
            input_window=90,
        )
    return _im_per_well_cache


# ════════════════════════════════════════════════════════════════════
# TestInnerMongoliaPreprocessing
# ════════════════════════════════════════════════════════════════════


class TestInnerMongoliaPreprocessing:
    """Verify the preprocessing pipeline output: 29 parquets, schema, quality."""

    @pytest.fixture
    def im_dir(self, project_root: Path) -> Path:
        return project_root / "data" / "processed" / "inner_mongolia"

    @pytest.fixture
    def sample_df(self, im_dir: Path) -> pd.DataFrame:
        """Load 58-25.parquet as a representative sample (longest well, 5240 days)."""
        return pd.read_parquet(im_dir / "58-25.parquet")

    # ── Existence / count ──────────────────────────────────────────

    def test_29_parquets_exist(self, im_dir: Path) -> None:
        """Preprocessing must produce exactly 29 per-well parquets."""
        files = list(im_dir.glob("*.parquet"))
        assert len(files) == 29, f"Expected 29 parquets, got {len(files)}"

    def test_56_14x_excluded(self, im_dir: Path) -> None:
        """56-14X must NOT appear in the output (only 177 days, below 180-day min)."""
        path = im_dir / "56-14X.parquet"
        assert not path.exists(), (
            "56-14X.parquet must be excluded (<180 production days)"
        )

    # ── Schema / column quality ────────────────────────────────────

    def test_parquet_has_no_nan(self, sample_df: pd.DataFrame) -> None:
        """No NaN values allowed — preprocessing fills all gaps."""
        total_nan = sample_df.isna().sum().sum()
        assert total_nan == 0, (
            f"Found {total_nan} NaN values in 58-25.parquet"
        )

    def test_target_column_present(self, sample_df: pd.DataFrame) -> None:
        """daily_gas_volume_1e4m3 must be present in the parquet."""
        assert "daily_gas_volume_1e4m3" in sample_df.columns

    def test_liquid_level_severity_present(self, sample_df: pd.DataFrame) -> None:
        """liquid_level_severity must be present as an auxiliary feature."""
        assert "liquid_level_severity" in sample_df.columns

    def test_ema_columns_exist(self, sample_df: pd.DataFrame) -> None:
        """EMA columns must be generated for dynamic features."""
        ema_cols = [c for c in sample_df.columns if "_ema_" in c]
        assert len(ema_cols) > 0, "No EMA columns found"
        # 9 dynamic cols × 4 windows = 36 EMA columns
        assert len(ema_cols) == 36, f"Expected 36 EMA columns, got {len(ema_cols)}"

    def test_roc_columns_exist(self, sample_df: pd.DataFrame) -> None:
        """ROC (rate of change) columns must be generated."""
        roc_cols = [c for c in sample_df.columns if "_roc" in c]
        assert len(roc_cols) > 0, "No ROC columns found"

    def test_shutdown_column_present(self, sample_df: pd.DataFrame) -> None:
        """is_shutdown column must be present."""
        assert "is_shutdown" in sample_df.columns

    def test_all_columns_numeric(self, sample_df: pd.DataFrame) -> None:
        """All columns must be numeric (float or bool for is_shutdown)."""
        for col in sample_df.columns:
            assert sample_df[col].dtype.kind in ("f", "b"), (
                f"Column {col} has non-numeric dtype: {sample_df[col].dtype}"
            )

    def test_expected_column_count(self, sample_df: pd.DataFrame) -> None:
        """56 columns: 10 base + 36 EMA + 9 ROC + 1 shutdown_flag."""
        assert len(sample_df.columns) == 56, (
            f"Expected 56 columns, got {len(sample_df.columns)}"
        )

    def test_date_index_is_datetime(self, sample_df: pd.DataFrame) -> None:
        """Index must be a DatetimeIndex."""
        assert isinstance(sample_df.index, pd.DatetimeIndex), (
            f"Expected DatetimeIndex, got {type(sample_df.index)}"
        )

    def test_shortest_included_well(self, im_dir: Path) -> None:
        """57-14X (180 days, exactly at threshold) must be included."""
        path = im_dir / "57-14X.parquet"
        assert path.exists(), "57-14X should be included (180 days >= 180 min)"
        df = pd.read_parquet(path)
        assert len(df) == 180


# ════════════════════════════════════════════════════════════════════
# TestInnerMongoliaMultiWell
# ════════════════════════════════════════════════════════════════════


class TestInnerMongoliaMultiWell:
    """Multi-well mode tests: shapes, types, target, NaN-free."""

    @pytest.fixture
    def ds(self, project_root: Path) -> InnerMongoliaDataset:
        return _get_multi_well(project_root)

    def test_positive_length(self, ds: InnerMongoliaDataset) -> None:
        """Dataset must have a positive number of samples."""
        assert len(ds) > 0

    def test_29_wells_loaded(self, ds: InnerMongoliaDataset) -> None:
        """Multi-well mode must load all 29 eligible wells."""
        assert len(ds._well_data) == 29

    def test_input_shape(self, ds: InnerMongoliaDataset) -> None:
        """Input tensor shape must be [90, n_vars]."""
        x, y, meta = ds[0]
        assert x.shape[0] == 90  # input_window
        assert x.shape[1] == ds.n_vars

    def test_target_shape(self, ds: InnerMongoliaDataset) -> None:
        """Target tensor shape must be [30] for h30."""
        x, y, meta = ds[0]
        assert y.shape == (30,)

    def test_tensors_are_float(self, ds: InnerMongoliaDataset) -> None:
        """Input and target must be float32 tensors."""
        x, y, meta = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_no_nan_in_input(self, ds: InnerMongoliaDataset) -> None:
        """Input tensor must contain no NaN."""
        x, y, meta = ds[0]
        assert not torch.isnan(x).any()

    def test_no_nan_in_target(self, ds: InnerMongoliaDataset) -> None:
        """Target tensor must contain no NaN."""
        x, y, meta = ds[0]
        assert not torch.isnan(y).any()

    def test_metadata_keys(self, ds: InnerMongoliaDataset) -> None:
        """Metadata must contain required keys."""
        _, _, meta = ds[0]
        required = {"well_name", "well_idx", "start_idx", "input_end", "target_end", "horizon", "mode"}
        assert required.issubset(meta.keys())

    def test_get_metadata(self, ds: InnerMongoliaDataset) -> None:
        """get_metadata() must return InnerMongoliaDataset class name."""
        md = ds.get_metadata()
        assert md["class"] == "InnerMongoliaDataset"
        assert md["n_vars"] == 56


# ════════════════════════════════════════════════════════════════════
# TestInnerMongoliaPerWell
# ════════════════════════════════════════════════════════════════════


class TestInnerMongoliaPerWell:
    """Per-well mode tests: single-well isolation, horizon shapes."""

    @pytest.fixture
    def ds(self, project_root: Path) -> InnerMongoliaDataset:
        return _get_per_well(project_root)

    def test_single_well_loaded(self, ds: InnerMongoliaDataset) -> None:
        """Per-well mode must load exactly 1 well."""
        assert len(ds._well_data) == 1
        assert ds._well_data[0][0] == "58-25"

    def test_input_shape(self, ds: InnerMongoliaDataset) -> None:
        """Input tensor shape must be [90, n_vars]."""
        x, y, meta = ds[0]
        assert x.shape[0] == 90

    def test_target_shape_h7(self, ds: InnerMongoliaDataset) -> None:
        """Target tensor shape must be [7] for h7."""
        x, y, meta = ds[0]
        assert y.shape == (7,)

    def test_well_name_in_metadata(self, ds: InnerMongoliaDataset) -> None:
        """Metadata must contain correct well name."""
        _, _, meta = ds[0]
        assert meta["well_name"] == "58-25"


# ════════════════════════════════════════════════════════════════════
# TestInnerMongoliaEdgeCases
# ════════════════════════════════════════════════════════════════════


class TestInnerMongoliaEdgeCases:
    """Edge cases: short wells, horizons, nonexistent wells."""

    def test_h90_reduces_samples(self, project_root: Path) -> None:
        """h90 must produce fewer samples than h30 (more days consumed)."""
        ds_h30 = _get_multi_well(project_root)
        config_path = project_root / "configs" / "data" / "inner_mongolia.yaml"
        base_path = project_root / "configs" / "base.yaml"
        ds_h90 = InnerMongoliaDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="multi_well",
            horizon=90,
            input_window=90,
        )
        assert len(ds_h90) < len(ds_h30)

    def test_nonexistent_well_fallback(self, project_root: Path) -> None:
        """Nonexistent well name must fall back to loading all wells."""
        config_path = project_root / "configs" / "data" / "inner_mongolia.yaml"
        base_path = project_root / "configs" / "base.yaml"
        ds = InnerMongoliaDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="per_well",
            well_name="NONEXISTENT",
            horizon=7,
        )
        # Should fall back to all wells
        assert len(ds._well_data) == 29

    def test_shortest_well_has_samples_h7(self, project_root: Path) -> None:
        """57-15X (181 days) must produce samples for h7 (needs 90+7=97 days)."""
        config_path = project_root / "configs" / "data" / "inner_mongolia.yaml"
        base_path = project_root / "configs" / "base.yaml"
        ds = InnerMongoliaDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="per_well",
            well_name="57-15X",
            horizon=7,
        )
        assert len(ds) > 0, "57-15X with h7 should produce samples (181 days > 97 needed)"

    def test_shortest_well_has_samples_h90(self, project_root: Path) -> None:
        """57-15X (181 days) must produce samples for h90 (needs 90+90=180 days)."""
        config_path = project_root / "configs" / "data" / "inner_mongolia.yaml"
        base_path = project_root / "configs" / "base.yaml"
        ds = InnerMongoliaDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="per_well",
            well_name="57-15X",
            horizon=90,
        )
        # 181 days - 180 needed = 1 sample at minimum
        assert len(ds) > 0, "57-15X with h90 should produce at least 1 sample"
