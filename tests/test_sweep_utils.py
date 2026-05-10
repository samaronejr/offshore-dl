"""Tests for shared production sweep helpers."""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from scripts.sweep_utils import _unwrap_eval_result, resolve_dataset_cv_gap, split_metadata


class FakeForecastDataset:
    horizon = 7
    input_window = 90
    gap = 0
    window_size = 90
    _samples = [(0, i) for i in range(100)]

    def __init__(self) -> None:
        self.cfg = OmegaConf.create(
            {
                "data": {
                    "task": "forecasting",
                    "cv_gap_policy": "causal_horizon",
                    "cv_gap": None,
                    "forecasting": {
                        "input_window": 90,
                        "default_horizon": 30,
                        "gap": 0,
                    },
                }
            }
        )



def test_unwrap_eval_result_keeps_metrics_scalar_schema() -> None:
    eval_result = {
        "metrics": {"mae": 1.0, "mase_valid": True},
        "sample_indices": np.array([1, 2]),
        "predictions": np.array([[1.0], [2.0]]),
        "targets": np.array([[1.5], [2.5]]),
    }

    stored = _unwrap_eval_result(eval_result, fold_idx=3)

    assert stored["fold_idx"] == 3
    assert stored["metrics"] == {"mae": 1.0, "mase_valid": True}
    assert "predictions" not in stored["metrics"]
    assert "targets" not in stored["metrics"]
    assert stored["sample_indices"].tolist() == [1, 2]



def test_split_metadata_records_active_forecasting_gap() -> None:
    dataset = FakeForecastDataset()

    assert resolve_dataset_cv_gap(dataset) == 7
    meta = split_metadata(dataset, n_train=80, n_test=20, n_cv_folds=3)

    assert meta["cv_gap_policy"] == "causal_horizon"
    assert meta["cv_gap"] == 7
    assert meta["outer_gap"] == 7
    assert meta["inner_gap"] == 7
    assert meta["raw_row_embargo_mode"] == "target_rows_only"
