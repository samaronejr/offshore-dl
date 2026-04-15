"""JSON serialization helpers for experiment results."""

from __future__ import annotations

import numpy as np
import torch


def make_serializable(obj):
    """Convert non-serializable types for JSON output.

    Recursively walks dicts/lists and converts NumPy/PyTorch types
    to native Python types.  Filters out ``"study"`` keys (Optuna
    study objects are not serializable).
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items() if k != "study"}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj
