"""Hierarchical YAML configuration with OmegaConf.

Supports:
    - Base config (``configs/base.yaml``) with global settings
    - Dataset-specific configs merged on top (``configs/data/*.yaml``)
    - Model-specific configs merged on top (``configs/models/*.yaml``)
    - CLI overrides via dotlist (``key=value`` pairs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def load_config(
    base_path: str | Path,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load and merge a YAML config with optional CLI overrides.

    Args:
        base_path: Path to the primary YAML config file.
        overrides: Optional list of ``"key=value"`` strings for CLI overrides.

    Returns:
        Merged OmegaConf DictConfig.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        msg = f"Config file not found: {base_path}"
        raise FileNotFoundError(msg)

    cfg = OmegaConf.load(base_path)
    logger.debug("Loaded base config from %s", base_path)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cli_cfg)
        logger.debug("Applied %d CLI overrides", len(overrides))

    return cfg


def load_merged_config(
    base_path: str | Path,
    *extra_paths: str | Path,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load a base config and merge additional configs on top.

    Useful for composing base + data + model configs::

        cfg = load_merged_config(
            "configs/base.yaml",
            "configs/data/3w.yaml",
            "configs/models/lstm.yaml",
        )

    Args:
        base_path: Path to the base YAML config.
        *extra_paths: Additional YAML configs to merge (in order).
        overrides: Optional CLI override dotlist applied last.

    Returns:
        Fully merged DictConfig.
    """
    cfg = load_config(base_path)

    for path in extra_paths:
        path = Path(path)
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)
        extra = OmegaConf.load(path)
        cfg = OmegaConf.merge(cfg, extra)
        logger.debug("Merged config from %s", path)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cli_cfg)
        logger.debug("Applied %d CLI overrides", len(overrides))

    return cfg


def resolve_config(cfg: DictConfig) -> DictConfig:
    """Resolve all interpolations in the config and return a frozen copy.

    Useful before logging to MLflow — ensures all values are concrete.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return OmegaConf.create(resolved)


def config_to_flat_dict(cfg: DictConfig) -> dict[str, str]:
    """Flatten a nested config into dot-separated key-value pairs.

    Useful for logging to MLflow params::

        mlflow.log_params(config_to_flat_dict(cfg))
    """
    flat: dict[str, str] = {}

    def _flatten(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            else:
                flat[key] = str(v)

    container = OmegaConf.to_container(cfg, resolve=True)
    _flatten(container)
    return flat
