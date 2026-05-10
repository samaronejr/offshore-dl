"""Tests for forecasting HPO shape/gap helpers."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.run_optuna_hpo import (
    _forecast_cv_gap,
    _forecast_model_kwargs,
    _forecast_split_metadata,
)


class TinyForecastDataset:
    horizon = 14
    input_window = 90
    gap = 0
    _target_col_idx = 5

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

    def __getitem__(self, _idx: int):
        return torch.zeros(90, 17), torch.zeros(self.horizon), {"well_idx": 0, "target_start": 90}



def test_forecast_hpo_kwargs_use_actual_dataset_shape() -> None:
    dataset = TinyForecastDataset()
    entry = {
        "kwargs": {
            "task": "forecasting",
            "n_vars": 999,
            "window_size": 30,
            "horizon": 30,
        }
    }

    kwargs = _forecast_model_kwargs(entry, dataset, horizon_days=dataset.horizon)

    assert kwargs["n_vars"] == 17
    assert kwargs["window_size"] == 90
    assert kwargs["horizon"] == 14
    assert kwargs["target_channel"] == 5



def test_forecast_hpo_gap_uses_active_horizon_not_config_default() -> None:
    dataset = TinyForecastDataset()

    assert _forecast_cv_gap(dataset.cfg, dataset) == 14

    meta = _forecast_split_metadata(
        dataset.cfg,
        dataset,
        n_train=100,
        n_test=20,
        n_cv_folds=3,
        n_trial_folds=2,
    )
    assert meta["cv_gap"] == 14
    assert meta["outer_gap"] == 14
    assert meta["inner_gap"] == 14
    assert meta["horizon"] == 14
    assert meta["n_trial_folds"] == 2
