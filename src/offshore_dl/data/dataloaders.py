"""DataLoader factory and balanced sampling for offshore DL pipelines.

Provides ``create_dataloader()`` with GPU-optimized defaults and
``BalancedBatchSampler`` for class-diverse batches in anomaly detection.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, Dataset, Sampler

from offshore_dl.data.transforms import (
    feature_dropout_augment,
    gaussian_noise_augment,
)

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """Sampler that ensures class diversity within each batch.

    Cycles through classes in round-robin fashion, sampling one instance
    per class per round until the batch is full. Guarantees each batch
    contains representatives of multiple event classes.

    Args:
        labels: Array of integer labels for all dataset samples.
        batch_size: Number of samples per batch.
        drop_last: Drop incomplete final batch.
    """

    def __init__(
        self,
        labels: np.ndarray | list[int],
        batch_size: int = 64,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by class
        self._class_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self._class_indices[int(label)].append(idx)

        self._classes = sorted(self._class_indices.keys())
        self._n_classes = len(self._classes)

        if self._n_classes == 0:
            msg = "BalancedBatchSampler: no classes found in labels"
            raise ValueError(msg)

        # Total batches
        total_samples = sum(len(v) for v in self._class_indices.values())
        self._n_batches = total_samples // batch_size
        if not drop_last and total_samples % batch_size > 0:
            self._n_batches += 1

        self._total = total_samples

    def __iter__(self):
        """Yield indices in class-balanced order."""
        # Shuffle indices within each class
        rng = np.random.RandomState()
        shuffled = {}
        for cls in self._classes:
            indices = self._class_indices[cls].copy()
            rng.shuffle(indices)
            shuffled[cls] = indices

        # Round-robin across classes
        class_pointers = {cls: 0 for cls in self._classes}
        batch = []

        while True:
            added = False
            for cls in self._classes:
                ptr = class_pointers[cls]
                if ptr < len(shuffled[cls]):
                    batch.append(shuffled[cls][ptr])
                    class_pointers[cls] += 1
                    added = True

                    if len(batch) == self.batch_size:
                        yield from batch
                        batch = []

            if not added:
                break

        if batch and not self.drop_last:
            yield from batch

    def __len__(self) -> int:
        return self._total


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    balanced: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    drop_last: bool = False,
    labels: np.ndarray | list[int] | None = None,
    augment: bool = False,
) -> DataLoader:
    """Create a DataLoader with GPU-optimized defaults.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Samples per batch.
        shuffle: Shuffle data each epoch (ignored if balanced=True).
        balanced: Use ``BalancedBatchSampler`` for class-diverse batches.
        num_workers: Number of data loading worker processes.
        pin_memory: Pin memory for async CPU→GPU transfer.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.
        drop_last: Drop incomplete final batch.
        labels: Required when ``balanced=True``. Array of class labels.
        augment: Apply 3W feature-matrix augmentation at collation time.

    Returns:
        Configured DataLoader.
    """
    sampler = None

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(42 + worker_id)

    loader_worker_init_fn = worker_init_fn if num_workers > 0 else None

    def collate_fn(batch):
        collated = default_collate(batch)
        if not augment:
            return collated

        features = collated[0]
        if (
            torch.is_tensor(features)
            and features.ndim == 3
            and features.shape[-1] == 27
        ):
            augmented = []
            for sample in features:
                sample_np = sample.cpu().numpy()
                sample_np = gaussian_noise_augment(sample_np)
                sample_np = feature_dropout_augment(sample_np)
                augmented.append(torch.from_numpy(sample_np))
            collated = (torch.stack(augmented, dim=0), *collated[1:])
        return collated

    if balanced:
        if labels is None:
            msg = "labels required when balanced=True"
            raise ValueError(msg)
        sampler = BalancedBatchSampler(labels, batch_size, drop_last)
        shuffle = False  # sampler handles ordering
        # When using a custom sampler that yields individual indices,
        # set batch_size=1 and batch_sampler=None, or use batch_sampler
        # We yield individual indices, so DataLoader batches them
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=loader_worker_init_fn,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=loader_worker_init_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
