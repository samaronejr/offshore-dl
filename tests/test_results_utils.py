"""Tests for result-root resolution."""

from __future__ import annotations

from pathlib import Path

from offshore_dl.utils.results import resolve_results_dir


def test_write_results_default_to_post_fix(monkeypatch) -> None:
    monkeypatch.delenv("OFFSHORE_DL_RESULTS_DIR", raising=False)

    assert resolve_results_dir(for_write=True) == Path("results/post_fix")


def test_results_env_overrides_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OFFSHORE_DL_RESULTS_DIR", str(tmp_path))

    assert resolve_results_dir(for_write=True) == tmp_path


def test_explicit_results_dir_has_priority(monkeypatch, tmp_path) -> None:
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("OFFSHORE_DL_RESULTS_DIR", str(tmp_path / "env"))

    assert resolve_results_dir(explicit, for_write=True) == explicit


def test_read_results_prefers_nonempty_post_fix_then_pre_fix(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OFFSHORE_DL_RESULTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "post_fix").mkdir(parents=True)
    pre = tmp_path / "results" / "pre_fix"
    (pre / "lstm").mkdir(parents=True)
    (pre / "lstm" / "ganymede.json").write_text("{}")

    assert resolve_results_dir(for_write=False) == Path("results/pre_fix")

    post = tmp_path / "results" / "post_fix"
    (post / "lstm").mkdir(parents=True)
    (post / "lstm" / "ganymede.json").write_text("{}")

    assert resolve_results_dir(for_write=False) == Path("results/post_fix")


def test_hpo_validator_default_stays_results_hpo(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OFFSHORE_DL_RESULTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "pre_fix" / "lstm").mkdir(parents=True)
    (tmp_path / "results" / "pre_fix" / "lstm" / "3w.json").write_text("{}")

    import importlib
    import scripts.validate_hpo_3w_results as validator

    validator = importlib.reload(validator)
    assert validator.DEFAULT_OUTPUT_DIR == Path("results/hpo")
