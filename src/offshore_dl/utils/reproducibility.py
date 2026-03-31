"""Deterministic seed management for reproducible experiments.

Sets seeds across all random number generators used in the pipeline:
Python stdlib, NumPy, PyTorch (CPU + CUDA), and cuDNN.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Lock down all sources of randomness for deterministic execution.

    Covers:
        - Python stdlib ``random``
        - NumPy
        - PyTorch CPU + all CUDA devices
        - cuDNN deterministic mode
        - CUBLAS workspace config (prevents non-deterministic reductions)

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism (trades speed for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS workspace — prevents non-deterministic reductions on GPU
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # PyTorch deterministic algorithms — raises error on non-deterministic ops
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info("Global seed set to %d (random, numpy, torch, CUDA, cuDNN)", seed)
