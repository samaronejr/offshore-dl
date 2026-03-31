"""Abstract base dataset for all offshore DL datasets.

All concrete datasets (ThreeW, CDF, Ganymede) inherit from ``BaseDataset``
and implement ``__getitem__`` and ``__len__``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

logger = logging.getLogger(__name__)


class BaseDataset(ABC, torch.utils.data.Dataset):
    """Base class for all offshore DL datasets.

    Subclasses must implement:
        - ``__getitem__(index)`` → ``(features: Tensor, target: Tensor | int, metadata: dict)``
        - ``__len__()`` → ``int``

    The return format convention:
        - **Classification** (3W): ``(Tensor[w, n_vars], int, dict)``
        - **Unsupervised** (CDF): ``(Tensor[w, n_vars], Tensor[w_or_h, n_vars], dict)``
        - **Forecasting** (Ganymede): ``(Tensor[w, n_vars], Tensor[h], dict)``
    """

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any, dict]:
        """Return a single sample as ``(features, target, metadata)``."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples."""
        ...

    def get_metadata(self) -> dict:
        """Return dataset-level metadata (name, size, column info, etc.).

        Override in subclasses for dataset-specific information.
        """
        return {
            "class": type(self).__name__,
            "length": len(self),
        }
