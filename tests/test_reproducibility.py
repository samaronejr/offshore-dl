"""Tests for deterministic seed management."""

from __future__ import annotations

import numpy as np
import torch

from offshore_dl.utils.reproducibility import set_global_seed


class TestSetGlobalSeed:
    """Verify seed determinism across random number generators."""

    def test_torch_determinism(self) -> None:
        """Two calls with the same seed produce identical torch tensors."""
        set_global_seed(42)
        t1 = torch.randn(10)

        set_global_seed(42)
        t2 = torch.randn(10)

        assert torch.equal(t1, t2), "Torch output not deterministic with same seed"

    def test_numpy_determinism(self) -> None:
        """Two calls with the same seed produce identical numpy arrays."""
        set_global_seed(42)
        a1 = np.random.randn(10)

        set_global_seed(42)
        a2 = np.random.randn(10)

        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different output."""
        set_global_seed(42)
        t1 = torch.randn(10)

        set_global_seed(99)
        t2 = torch.randn(10)

        assert not torch.equal(t1, t2), "Different seeds should produce different output"

    def test_python_random_determinism(self) -> None:
        """Python stdlib random is also seeded."""
        import random

        set_global_seed(42)
        v1 = [random.random() for _ in range(10)]

        set_global_seed(42)
        v2 = [random.random() for _ in range(10)]

        assert v1 == v2

    def test_cublas_env_set(self) -> None:
        """CUBLAS_WORKSPACE_CONFIG env var is set after seeding."""
        import os

        set_global_seed(42)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
