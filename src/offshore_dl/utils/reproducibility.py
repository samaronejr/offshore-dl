"""Deterministic seed management for reproducible experiments.

Sets seeds across all random number generators used in the pipeline:
Python stdlib, NumPy, PyTorch (CPU + CUDA), and cuDNN. CUDA/CUBLAS
reproducibility is best-effort unless strict mode is enabled before CUDA
initialization.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_global_seed(seed: int = 42, *, strict: bool = False) -> None:
    """Configure deterministic RNG state for experiment reproducibility.

    Covers:
        - Python stdlib ``random``
        - NumPy
        - PyTorch CPU + all CUDA devices
        - cuDNN deterministic mode
        - CUBLAS workspace config when set before CUDA context initialization

    Args:
        seed: Integer seed value. Default 42.
        strict: When ``True``, request PyTorch deterministic algorithms with
            ``warn_only=False`` and raise if CUDA is already initialized before
            ``CUBLAS_WORKSPACE_CONFIG`` has the required value. When ``False``,
            deterministic algorithms use warning-only mode for compatibility.
    """
    cuda_initialized = torch.cuda.is_initialized()
    cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if strict and cuda_initialized and cublas_config != _CUBLAS_WORKSPACE_CONFIG:
        msg = (
            "strict reproducibility requires CUBLAS_WORKSPACE_CONFIG="
            f"{_CUBLAS_WORKSPACE_CONFIG!r} before CUDA is initialized"
        )
        raise RuntimeError(msg)

    # CUBLAS reads this at CUDA context creation time; set it before CUDA seeding
    # so entrypoints that call set_global_seed early get deterministic reductions.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism (trades speed for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=not strict)

    mode = "strict" if strict else "warn-only"
    logger.info(
        "Global seed set to %d (random, numpy, torch, CUDA, cuDNN, CUBLAS; %s)",
        seed,
        mode,
    )
