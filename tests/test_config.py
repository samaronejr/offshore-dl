"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from offshore_dl.utils.config import (
    config_to_flat_dict,
    load_config,
    load_merged_config,
)


class TestLoadConfig:
    """Test basic config loading."""

    def test_load_base_config(self, configs_dir: Path) -> None:
        cfg = load_config(configs_dir / "base.yaml")
        assert isinstance(cfg, DictConfig)
        assert cfg.seed == 42
        assert cfg.device == "cuda"

    def test_base_config_has_paths(self, configs_dir: Path) -> None:
        cfg = load_config(configs_dir / "base.yaml")
        assert cfg.paths.raw_data == "data/raw"
        assert cfg.paths.processed_data == "data/processed"
        assert cfg.paths.splits == "data/splits"

    def test_base_config_has_mlflow(self, configs_dir: Path) -> None:
        cfg = load_config(configs_dir / "base.yaml")
        assert cfg.mlflow.tracking_uri == "http://localhost:5000"
        assert cfg.mlflow.experiment_prefix == "offshore-dl"

    def test_base_config_has_optuna(self, configs_dir: Path) -> None:
        cfg = load_config(configs_dir / "base.yaml")
        assert cfg.optuna.n_trials_min == 50
        assert cfg.optuna.pruner == "median"

    def test_base_config_has_checkpoint_defaults(self, configs_dir: Path) -> None:
        cfg = load_config(configs_dir / "base.yaml")
        assert cfg.training.checkpoint_metric == "val_loss"
        assert cfg.training.checkpoint_mode == "min"

    def test_cli_override(self, configs_dir: Path) -> None:
        cfg = load_config(
            configs_dir / "base.yaml",
            overrides=["seed=123", "device=cpu"],
        )
        assert cfg.seed == 123
        assert cfg.device == "cpu"

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")


class TestLoadMergedConfig:
    """Test config merging."""

    def test_merge_base_with_data(self, configs_dir: Path) -> None:
        cfg = load_merged_config(
            configs_dir / "base.yaml",
            configs_dir / "data" / "3w.yaml",
        )
        # Base values preserved
        assert cfg.seed == 42
        # Data values merged
        assert cfg.data.name == "3w"
        assert cfg.data.task == "classification"
        assert cfg.data.n_classes == 10

    def test_merge_with_cli_override(self, configs_dir: Path) -> None:
        cfg = load_merged_config(
            configs_dir / "base.yaml",
            configs_dir / "data" / "ganymede.yaml",
            overrides=["data.forecasting.default_horizon=7"],
        )
        assert cfg.data.forecasting.default_horizon == 7

    def test_hpo_scheduler_choices_are_supported(self, configs_dir: Path) -> None:
        supported = {"cosine", "onecycle", "reduce_on_plateau"}
        for model_name in ("lstm", "deeponet"):
            cfg = load_config(configs_dir / "models" / f"{model_name}.yaml")
            choices = set(cfg.model.optuna_search_space.scheduler.choices)
            assert choices <= supported


class TestConfigToFlatDict:
    """Test config flattening for MLflow logging."""

    def test_flat_dict(self, configs_dir: Path) -> None:
        cfg = load_config(configs_dir / "base.yaml")
        flat = config_to_flat_dict(cfg)
        assert isinstance(flat, dict)
        assert flat["seed"] == "42"
        assert flat["paths.raw_data"] == "data/raw"
        assert flat["mlflow.tracking_uri"] == "http://localhost:5000"

    def test_all_values_are_strings(self, configs_dir: Path) -> None:
        cfg = load_config(configs_dir / "base.yaml")
        flat = config_to_flat_dict(cfg)
        for k, v in flat.items():
            assert isinstance(v, str), f"Key {k} has non-string value: {type(v)}"


def test_hpo_scheduler_choices_are_supported(configs_dir: Path) -> None:
    """Model HPO configs must not search schedulers unsupported by Trainer."""
    supported = {"onecycle", "cosine", "reduce_on_plateau"}
    offenders: dict[str, list[str]] = {}

    for path in sorted((configs_dir / "models").glob("*.yaml")):
        cfg = load_config(path)
        search_space = cfg.get("model", {}).get("optuna_search_space", {})
        scheduler_spec = search_space.get("scheduler")
        if not scheduler_spec:
            continue
        choices = list(scheduler_spec.get("choices", []))
        unsupported = [choice for choice in choices if choice not in supported]
        if unsupported:
            offenders[path.name] = unsupported

    assert offenders == {}
