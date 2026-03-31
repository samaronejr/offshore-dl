"""Shared test fixtures for offshore-dl tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# Project root (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def configs_dir(project_root: Path) -> Path:
    """Return the configs/ directory."""
    return project_root / "configs"


@pytest.fixture
def raw_data_dir(project_root: Path) -> Path:
    """Return the data/raw/ directory."""
    return project_root / "data" / "raw"


@pytest.fixture
def processed_data_dir(project_root: Path) -> Path:
    """Return the data/processed/ directory."""
    d = project_root / "data" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d
