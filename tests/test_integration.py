"""Integration tests — DataLoader factory with all 3 datasets.

Verifies that datasets work through the DataLoader pipeline,
producing correctly shaped and typed batched tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from offshore_dl.data.dataloaders import BalancedBatchSampler, create_dataloader
from offshore_dl.data.datasets import CDFDataset, GanymedeDataset, ThreeWDataset


class TestBalancedBatchSampler:
    """Test class-balanced sampling."""

    def test_produces_all_indices(self) -> None:
        labels = [0] * 10 + [1] * 10 + [2] * 10
        sampler = BalancedBatchSampler(labels, batch_size=6)
        indices = list(sampler)
        assert len(indices) == 30

    def test_batch_has_class_diversity(self) -> None:
        labels = [0] * 50 + [1] * 50 + [2] * 50
        sampler = BalancedBatchSampler(labels, batch_size=6, drop_last=True)
        indices = list(sampler)
        # Check first batch has multiple classes
        first_batch = indices[:6]
        batch_labels = [labels[i] for i in first_batch]
        unique = set(batch_labels)
        assert len(unique) >= 2, f"First batch has only classes: {unique}"


class TestDataLoaderFactory:
    """Test create_dataloader with real datasets."""

    def test_threew_dataloader(self, project_root: Path) -> None:
        ds = ThreeWDataset(
            str(project_root / "configs" / "data" / "3w.yaml"),
            window_size=100,
            window_stride=5000,
            cache_in_memory=False,
            class_ids=[0, 1],
            max_instances_per_class=2,
        )
        dl = create_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)
        batch = next(iter(dl))
        features, labels, metadata = batch

        assert features.shape == (4, 100, 27)
        assert features.dtype == torch.float32
        assert len(labels) == 4

    def test_cdf_dataloader(self, project_root: Path) -> None:
        ds = CDFDataset(
            str(project_root / "configs" / "data" / "cdf.yaml"),
            mode="reconstruction",
            window_stride=100,
        )
        dl = create_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)
        batch = next(iter(dl))
        inputs, targets, metadata = batch

        assert inputs.shape[0] == 4
        assert inputs.shape[1] == 48  # window_size
        assert inputs.shape == targets.shape
        assert inputs.dtype == torch.float32

    def test_ganymede_dataloader(self, project_root: Path) -> None:
        ds = GanymedeDataset(
            str(project_root / "configs" / "data" / "ganymede.yaml"),
            mode="multi_well",
            horizon=7,
            input_window=30,
        )
        dl = create_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)
        batch = next(iter(dl))
        inputs, targets, metadata = batch

        assert inputs.shape[0] == 4
        assert inputs.shape[1] == 30  # input_window
        assert targets.shape == (4, 7)  # horizon
        assert inputs.dtype == torch.float32

    def test_balanced_dataloader(self, project_root: Path) -> None:
        ds = ThreeWDataset(
            str(project_root / "configs" / "data" / "3w.yaml"),
            window_size=100,
            window_stride=5000,
            cache_in_memory=False,
            class_ids=[0, 1, 7],
            max_instances_per_class=3,
        )
        # Extract labels for balanced sampling
        labels = [ds[i][1] for i in range(len(ds))]
        dl = create_dataloader(
            ds, batch_size=4, balanced=True, labels=labels,
            num_workers=0, pin_memory=False,
        )
        batch = next(iter(dl))
        features, batch_labels, metadata = batch
        assert features.shape[0] == 4
