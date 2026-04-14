"""Tests for dataset implementations.

Tests are organized by dataset class. Each section runs against real data
to verify the full pipeline from raw parquet to PyTorch tensor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from offshore_dl.data.datasets import (
    CDFDataset,
    GanymedeDataset,
    InnerMongoliaDataset,
    SPEBergDataset,
    ThreeWDataset,
    VolveDataset,
)

# ═══════════════════════════════════════════════════════════════════
# 3W Dataset Tests
# ═══════════════════════════════════════════════════════════════════


_threew_dataset_cache: ThreeWDataset | None = None


def _get_threew_dataset(project_root: Path) -> ThreeWDataset:
    """Singleton ThreeWDataset — loads once, reused across all tests."""
    global _threew_dataset_cache
    if _threew_dataset_cache is None:
        config_path = project_root / "configs" / "data" / "3w.yaml"
        base_path = project_root / "configs" / "base.yaml"
        _threew_dataset_cache = ThreeWDataset(
            config=str(config_path),
            base_config=str(base_path),
            window_size=100,
            window_stride=5000,  # large stride to limit window count for speed
            cache_in_memory=False,  # don't cache all 1.8GB in RAM for tests
            class_ids=[0, 1, 7],  # subset for fast testing (real, simulated, drawn)
            max_instances_per_class=3,
        )
    return _threew_dataset_cache


class TestThreeWDataset:
    """Tests for the 3W anomaly classification dataset."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> ThreeWDataset:
        """Get the singleton ThreeWDataset."""
        return _get_threew_dataset(project_root)

    def test_dataset_not_empty(self, dataset: ThreeWDataset) -> None:
        """Dataset should have > 0 samples."""
        assert len(dataset) > 0

    def test_getitem_returns_correct_types(self, dataset: ThreeWDataset) -> None:
        """__getitem__ should return (Tensor, int, dict)."""
        tensor, label, metadata = dataset[0]
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(metadata, dict)

    def test_tensor_shape(self, dataset: ThreeWDataset) -> None:
        """Feature tensor should be [window_size, n_vars]."""
        tensor, _, _ = dataset[0]
        assert tensor.shape == (100, 27)  # window_size=100, 27 sensors

    def test_tensor_dtype(self, dataset: ThreeWDataset) -> None:
        """Feature tensor should be float32."""
        tensor, _, _ = dataset[0]
        assert tensor.dtype == torch.float32

    def test_label_in_valid_range(self, dataset: ThreeWDataset) -> None:
        """Labels should be integers in [0, 9]."""
        for i in range(min(50, len(dataset))):
            _, label, _ = dataset[i]
            assert 0 <= label <= 9, f"Label {label} out of range at index {i}"

    def test_no_nan_in_tensor(self, dataset: ThreeWDataset) -> None:
        """Output tensors should have no NaN values."""
        for i in range(min(20, len(dataset))):
            tensor, _, _ = dataset[i]
            assert not torch.isnan(tensor).any(), f"NaN found at index {i}"

    def test_metadata_has_required_keys(self, dataset: ThreeWDataset) -> None:
        """Metadata should include well_id, class_id, source_type."""
        _, _, metadata = dataset[0]
        assert "well_id" in metadata
        assert "class_id" in metadata
        assert "source_type" in metadata
        assert "instance_id" in metadata

    def test_get_metadata_returns_summary(self, dataset: ThreeWDataset) -> None:
        """Dataset-level metadata should include class counts."""
        meta = dataset.get_metadata()
        assert meta["class"] == "ThreeWDataset"
        assert meta["length"] == len(dataset)
        assert meta["n_vars"] == 27
        assert "class_counts" in meta


# ═══════════════════════════════════════════════════════════════════
# CDF Dataset Tests
# ═══════════════════════════════════════════════════════════════════


class TestCDFDatasetReconstruction:
    """Tests for CDF dataset in reconstruction mode."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> CDFDataset:
        config_path = project_root / "configs" / "data" / "cdf.yaml"
        base_path = project_root / "configs" / "base.yaml"
        return CDFDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="reconstruction",
            window_stride=100,  # fast testing
        )

    def test_dataset_not_empty(self, dataset: CDFDataset) -> None:
        assert len(dataset) > 0

    def test_returns_two_tensors_and_dict(self, dataset: CDFDataset) -> None:
        inp, target, meta = dataset[0]
        assert isinstance(inp, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(meta, dict)

    def test_reconstruction_input_equals_target(self, dataset: CDFDataset) -> None:
        inp, target, _ = dataset[0]
        assert torch.equal(inp, target)

    def test_tensor_shape(self, dataset: CDFDataset) -> None:
        inp, target, _ = dataset[0]
        assert inp.shape[0] == 48  # window_size from config
        assert inp.shape[1] == 11  # 11 sensor columns (dropped 1 all-NaN)
        assert inp.shape == target.shape

    def test_no_nan_in_tensors(self, dataset: CDFDataset) -> None:
        for i in range(min(10, len(dataset))):
            inp, target, _ = dataset[i]
            assert not torch.isnan(inp).any()
            assert not torch.isnan(target).any()

    def test_dropped_column_absent(self, dataset: CDFDataset) -> None:
        meta = dataset.get_metadata()
        # The 100%-missing column should not appear
        for col in meta["columns"]:
            assert "Poly Eff Dev" not in col


class TestCDFDatasetPrediction:
    """Tests for CDF dataset in prediction mode."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> CDFDataset:
        config_path = project_root / "configs" / "data" / "cdf.yaml"
        base_path = project_root / "configs" / "base.yaml"
        return CDFDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="prediction",
            window_stride=100,
        )

    def test_prediction_shapes(self, dataset: CDFDataset) -> None:
        inp, target, _ = dataset[0]
        assert inp.shape[0] == 48  # window_size
        assert target.shape[0] == 12  # prediction_horizon
        assert inp.shape[1] == target.shape[1]  # same n_vars

    def test_input_target_not_equal(self, dataset: CDFDataset) -> None:
        inp, target, _ = dataset[0]
        # Different sizes means they can't be equal
        assert inp.shape[0] != target.shape[0]


# ═══════════════════════════════════════════════════════════════════
# Ganymede Dataset Tests
# ═══════════════════════════════════════════════════════════════════


class TestGanymedeDatasetMultiWell:
    """Tests for Ganymede multi-well forecasting dataset."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> GanymedeDataset:
        config_path = project_root / "configs" / "data" / "ganymede.yaml"
        base_path = project_root / "configs" / "base.yaml"
        return GanymedeDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="multi_well",
            horizon=30,
            input_window=90,
        )

    def test_dataset_not_empty(self, dataset: GanymedeDataset) -> None:
        assert len(dataset) > 0

    def test_returns_correct_types(self, dataset: GanymedeDataset) -> None:
        inp, target, meta = dataset[0]
        assert isinstance(inp, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(meta, dict)

    def test_input_shape(self, dataset: GanymedeDataset) -> None:
        inp, _, _ = dataset[0]
        assert inp.shape[0] == 90  # input_window

    def test_target_shape(self, dataset: GanymedeDataset) -> None:
        _, target, _ = dataset[0]
        assert target.shape[0] == 30  # horizon

    def test_target_is_1d(self, dataset: GanymedeDataset) -> None:
        _, target, _ = dataset[0]
        assert target.dim() == 1  # scalar target per timestep

    def test_no_temporal_overlap(self, dataset: GanymedeDataset) -> None:
        """Input window must end strictly before target window starts."""
        _, _, meta = dataset[0]
        assert meta["input_end"] <= meta["target_end"] - meta["horizon"]

    def test_multiple_wells_in_multi_mode(self, dataset: GanymedeDataset) -> None:
        meta = dataset.get_metadata()
        assert len(meta["well_counts"]) > 1  # multiple wells

    def test_no_nan(self, dataset: GanymedeDataset) -> None:
        for i in range(min(10, len(dataset))):
            inp, target, _ = dataset[i]
            assert not torch.isnan(inp).any()
            assert not torch.isnan(target).any()


@pytest.mark.parametrize(
    ("dataset_cls", "well_name"),
    [
        (GanymedeDataset, "well-a"),
        (SPEBergDataset, "well_1"),
        (VolveDataset, "NO_15_9-F-1_C"),
        (InnerMongoliaDataset, "58-25"),
    ],
)
def test_forecasting_datasets_honor_gap(dataset_cls, well_name: str) -> None:
    dataset = dataset_cls.__new__(dataset_cls)
    dataset._samples = [(0, 0)]
    dataset._well_data = [(well_name, None)]
    dataset._arrays = [np.arange(30, dtype=np.float32).reshape(10, 3)]
    dataset.input_window = 3
    dataset.gap = 1
    dataset.horizon = 2
    dataset._target_col_idx = 1
    dataset.mode = "multi_well"

    inp, target, meta = dataset[0]

    assert torch.equal(
        inp,
        torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.float32),
    )
    assert torch.equal(target, torch.tensor([13, 16], dtype=torch.float32))
    assert meta["target_start"] == 4
    assert meta["target_end"] == 6
    assert meta["gap"] == 1


class TestGanymedeDatasetPerWell:
    """Tests for Ganymede per-well mode."""

    @pytest.fixture
    def dataset(self, project_root: Path) -> GanymedeDataset:
        config_path = project_root / "configs" / "data" / "ganymede.yaml"
        base_path = project_root / "configs" / "base.yaml"
        return GanymedeDataset(
            config=str(config_path),
            base_config=str(base_path),
            mode="per_well",
            well_name="49/22-Z01Z",
            horizon=7,
            input_window=30,
        )

    def test_single_well_only(self, dataset: GanymedeDataset) -> None:
        meta = dataset.get_metadata()
        assert len(meta["well_counts"]) == 1

    def test_correct_horizon(self, dataset: GanymedeDataset) -> None:
        _, target, _ = dataset[0]
        assert target.shape[0] == 7
