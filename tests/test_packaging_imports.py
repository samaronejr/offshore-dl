"""Package metadata and optional import wiring tests."""

from __future__ import annotations

import builtins
import importlib
import sys
import tomllib
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _load_pyproject() -> dict:
    with Path("pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_license_and_optional_extras_match_documented_install_paths() -> None:
    project = _load_pyproject()["project"]
    assert project["license"]["text"] == "Apache-2.0"

    extras = project["optional-dependencies"]
    for key in ("dev", "fm", "mamba", "aeon", "stats"):
        assert key in extras
    assert any(req.startswith("mamba-ssm") for req in extras["mamba"])
    assert any(req.startswith("aeon") for req in extras["aeon"])
    assert any(req.startswith("statsmodels") for req in extras["stats"])


def test_timesfm_extra_is_limited_to_supported_python_versions() -> None:
    extras = _load_pyproject()["project"]["optional-dependencies"]
    timesfm_entries = [req for req in extras["fm"] if req.startswith("timesfm")]

    assert timesfm_entries == ["timesfm>=1.3.0; python_version < '3.12'"]


def test_run_experiment_imports_without_patchtst(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "offshore_dl.models.patchtst":
            raise ImportError("simulated missing transformers/PatchTST")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("offshore_dl.run_experiment", None)
    sys.modules.pop("offshore_dl.models.patchtst", None)

    try:
        module = importlib.import_module("offshore_dl.run_experiment")
        assert "patchtst" not in module.MODEL_REGISTRY
        assert "lstm" in module.MODEL_REGISTRY
    finally:
        monkeypatch.undo()
        sys.modules.pop("offshore_dl.run_experiment", None)
        importlib.import_module("offshore_dl.run_experiment")


def test_build_experiment_forwards_label_smoothing(monkeypatch: pytest.MonkeyPatch) -> None:
    import offshore_dl.run_experiment as run_experiment

    class DummyDataset:
        n_vars = 2
        horizon = 3
        _target_col_idx = 0
        _samples = [(0, 0), (0, 1), (0, 2), (0, 3)]

        def __init__(self, *_args, **_kwargs):
            pass

        def __getitem__(self, _idx):
            import torch
            return torch.zeros(5, 2), torch.zeros(3), {}

        def __len__(self):
            return len(self._samples)

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "device": "cpu",
            "training": {"batch_size": 2, "max_epochs": 1},
            "data": {},
            "model": {
                "name": "lstm",
                "label_smoothing": 0.15,
                "training": {"lr": 0.01, "weight_decay": 0.02},
            },
        }
    )
    monkeypatch.setattr(run_experiment, "load_merged_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setitem(
        run_experiment.DATASET_REGISTRY,
        "dummy_forecast",
        {
            "class": DummyDataset,
            "config": "unused.yaml",
            "task": "forecasting",
            "cv_factory": lambda cfg, ds: object(),
            "model_kwargs": lambda ds, cfg: {"task": "forecasting", "n_vars": 2, "horizon": 3, "window_size": 5},
        },
    )

    runner, _ = run_experiment.build_experiment("lstm", "dummy_forecast")
    assert runner.model_kwargs["label_smoothing"] == 0.15
    assert runner.model_kwargs["lr"] == 0.01
    assert runner.model_kwargs["weight_decay"] == 0.02


def test_cv_gap_policy_helpers() -> None:
    from offshore_dl.run_experiment import _cdf_cv_gap, _forecast_cv_gap

    class ForecastDataset:
        horizon = 7
        input_window = 30
        gap = 2

        def __getitem__(self, _idx):
            import torch
            return torch.zeros(30, 4), torch.zeros(7), {}

    assert _forecast_cv_gap(OmegaConf.create({"data": {}}), ForecastDataset()) == 7
    assert (
        _forecast_cv_gap(
            OmegaConf.create({"data": {"cv_gap_policy": "strict_raw_row"}}),
            ForecastDataset(),
        )
        == 38
    )
    assert (
        _forecast_cv_gap(
            OmegaConf.create({"data": {"cv_gap": 5, "cv_gap_policy": "strict_raw_row"}}),
            ForecastDataset(),
        )
        == 5
    )
    assert _cdf_cv_gap(OmegaConf.create({"data": {"preprocessing": {"window_size": 12}}})) == 11
    assert _cdf_cv_gap(OmegaConf.create({"data": {"cv_gap": 3, "preprocessing": {"window_size": 12}}})) == 3


def test_patchtst_build_experiment_sanitizes_short_window_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import offshore_dl.run_experiment as run_experiment

    class DummyDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def __getitem__(self, _idx):
            import torch

            return torch.zeros(14, 2), torch.zeros(1), {}

    class DummyRunner:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyPatchTST:
        pass

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "device": "cpu",
            "training": {"batch_size": 2, "max_epochs": 1},
            "data": {},
            "model": {
                "name": "patchtst",
                "architecture": {"patch_len": 16, "stride": 32, "d_model": 16},
                "training": {"lr": 0.01, "weight_decay": 0.02},
            },
        }
    )
    monkeypatch.setattr(run_experiment, "load_merged_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(run_experiment, "ExperimentRunner", DummyRunner)
    monkeypatch.setitem(run_experiment.MODEL_REGISTRY, "patchtst", DummyPatchTST)
    monkeypatch.setitem(
        run_experiment.DATASET_REGISTRY,
        "dummy_short_window",
        {
            "class": DummyDataset,
            "config": "unused.yaml",
            "task": "forecasting",
            "cv_factory": lambda cfg, ds: object(),
            "model_kwargs": lambda ds, cfg: {
                "task": "forecasting",
                "n_vars": 2,
                "horizon": 1,
                "window_size": 14,
            },
        },
    )

    runner, _ = run_experiment.build_experiment("patchtst", "dummy_short_window")

    assert runner.model_kwargs["patch_len"] == 14
    assert runner.model_kwargs["stride"] == 14
    assert runner.model_kwargs["d_model"] == 16
    assert runner.model_kwargs["lr"] == 0.01
    assert runner.model_kwargs["weight_decay"] == 0.02


def test_patchtst_short_window_sanitizer_preserves_valid_kwargs() -> None:
    import offshore_dl.run_experiment as run_experiment

    kwargs = {"window_size": 32, "patch_len": 16, "stride": 8}
    run_experiment._sanitize_patchtst_short_window(kwargs)

    assert kwargs == {"window_size": 32, "patch_len": 16, "stride": 8}
