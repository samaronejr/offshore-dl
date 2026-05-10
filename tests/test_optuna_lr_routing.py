"""Regression tests for Optuna optimizer/trainer parameter routing."""

from __future__ import annotations

from omegaconf import OmegaConf

from offshore_dl.models.lstm import LSTMModel
from offshore_dl.training.optuna_utils import OptunaObjective


class _FixedTrial:
    def suggest_float(self, name, low, high, log=False):
        values = {"lr": 0.0042, "weight_decay": 0.0123}
        return values[name]

    def suggest_categorical(self, name, choices):
        values = {"scheduler": "cosine", "batch_size": 16}
        return values[name]

    def suggest_int(self, name, low, high):
        return low


def test_optuna_optimizer_params_reach_kwargs_and_both_config_paths(monkeypatch) -> None:
    captured: dict = {}

    class _Runner:
        def __init__(self, model_class, dataset, cv_strategy, cfg, model_kwargs):
            captured["model_class"] = model_class
            captured["cfg"] = cfg
            captured["model_kwargs"] = model_kwargs

        def run(self, use_mlflow=False):
            return {"aggregate": {"val_loss_mean": 0.5}}

    monkeypatch.setattr("offshore_dl.training.experiment.ExperimentRunner", _Runner)

    cfg = OmegaConf.create(
        {
            "training": {"lr": 0.001, "weight_decay": 0.0001, "batch_size": 64},
            "model": {"training": {"lr": 0.001, "weight_decay": 0.0001}},
        }
    )
    objective = OptunaObjective(
        model_class=LSTMModel,
        dataset=object(),
        cv_strategy=object(),
        base_cfg=cfg,
        model_kwargs={"task": "classification", "n_vars": 3},
        search_space={
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
            "scheduler": {"type": "categorical", "choices": ["cosine"]},
            "batch_size": {"type": "categorical", "choices": [16]},
        },
    )

    assert objective(_FixedTrial()) == 0.5

    assert captured["model_kwargs"]["lr"] == 0.0042
    assert captured["model_kwargs"]["weight_decay"] == 0.0123
    assert captured["cfg"].training.lr == 0.0042
    assert captured["cfg"].training.weight_decay == 0.0123
    assert captured["cfg"].model.training.lr == 0.0042
    assert captured["cfg"].model.training.weight_decay == 0.0123
    assert captured["cfg"].training.scheduler == "cosine"
    assert captured["cfg"].model.training.scheduler == "cosine"
    assert captured["cfg"].training.batch_size == 16
