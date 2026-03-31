"""Tests for comparative analysis module."""

import json
from pathlib import Path

import numpy as np
import pytest

from offshore_dl.analysis.compare import (
    build_comparison_table,
    format_comparison_table,
    generate_latex_table,
    generate_multihorizon_table,
    generate_perwell_table,
    get_fold_values,
    get_metric_value,
    load_all_results,
    load_baseline_mae,
    load_multihorizon_results,
    load_perwell_results,
    run_statistical_tests,
)


class TestLoadResults:
    """Test result loading from disk."""

    def test_load_all_results(self) -> None:
        results = load_all_results("results")
        assert "lstm" in results
        assert "deeponet" in results
        assert "patchtst" in results
        assert "chronos" in results
        assert "timesfm" in results
        assert "tirex" in results

    def test_lstm_has_all_tracks(self) -> None:
        results = load_all_results("results")
        assert "3w" in results["lstm"]
        assert "ganymede" in results["lstm"]
        assert "cdf" in results["lstm"]

    def test_chronos_missing_3w(self) -> None:
        results = load_all_results("results")
        assert "3w" not in results.get("chronos", {})

    def test_timesfm_missing_3w(self) -> None:
        results = load_all_results("results")
        assert "3w" not in results.get("timesfm", {})

    def test_tirex_has_3w(self) -> None:
        results = load_all_results("results")
        assert "3w" in results.get("tirex", {})


class TestMetricExtraction:
    """Test metric value extraction."""

    def test_get_metric_from_aggregate(self) -> None:
        result = {"aggregate": {"mae_mean": 2.5, "mae_std": 0.3}}
        assert get_metric_value(result, "mae") == 2.5

    def test_get_metric_from_fold_results(self) -> None:
        result = {"fold_results": [
            {"metrics": {"mae": 1.0}},
            {"metrics": {"mae": 3.0}},
        ]}
        assert get_metric_value(result, "mae") == 2.0

    def test_get_fold_values(self) -> None:
        result = {"fold_results": [
            {"metrics": {"mae": 1.0}},
            {"metrics": {"mae": 2.0}},
            {"metrics": {"mae": 3.0}},
        ]}
        values = get_fold_values(result, "mae")
        assert values == [1.0, 2.0, 3.0]

    def test_missing_metric_returns_none(self) -> None:
        result = {"aggregate": {"f1": 0.5}}
        assert get_metric_value(result, "nonexistent") is None


class TestComparisonTable:
    """Test comparison table building."""

    def test_build_tables(self) -> None:
        results = load_all_results("results")
        tables = build_comparison_table(results)
        assert "3w" in tables
        assert "ganymede" in tables
        assert "cdf" in tables

    def test_entries_are_ranked(self) -> None:
        results = load_all_results("results")
        tables = build_comparison_table(results)
        ganymede = tables["ganymede"]
        # Should be sorted by MAE ascending (lower is better)
        for i in range(len(ganymede) - 1):
            assert ganymede[i]["primary_value"] <= ganymede[i + 1]["primary_value"]

    def test_3w_sorted_descending(self) -> None:
        results = load_all_results("results")
        tables = build_comparison_table(results)
        entries_3w = tables["3w"]
        # F1-macro: higher is better → descending
        for i in range(len(entries_3w) - 1):
            assert entries_3w[i]["primary_value"] >= entries_3w[i + 1]["primary_value"]

    def test_format_table_string(self) -> None:
        results = load_all_results("results")
        tables = build_comparison_table(results)
        table_str = format_comparison_table("ganymede", tables["ganymede"])
        assert "Ganymede" in table_str
        assert "mae" in table_str


class TestStatisticalTests:
    """Test Wilcoxon and Friedman statistical tests."""

    def test_ganymede_has_tests(self) -> None:
        results = load_all_results("results")
        test_results = run_statistical_tests(results)
        # Ganymede has 3 folds for trained models
        ganymede_tests = test_results.get("ganymede", {})
        # Should have pairwise tests
        assert "wilcoxon_pairwise" in ganymede_tests or ganymede_tests.get("status") == "insufficient_folds"

    def test_cdf_insufficient_folds(self) -> None:
        results = load_all_results("results")
        test_results = run_statistical_tests(results)
        # CDF has only 1 fold — should report insufficient
        cdf_tests = test_results.get("cdf", {})
        assert cdf_tests.get("status") == "insufficient_folds" or "wilcoxon_pairwise" in cdf_tests

    def test_significant_is_bool(self) -> None:
        """Verify that 'significant' fields survive JSON round-trip as bool, not str."""
        results = load_all_results("results")
        test_results = run_statistical_tests(results)
        # Round-trip through JSON (no default=str)
        serialized = json.dumps(test_results, indent=2)
        deserialized = json.loads(serialized)
        for track, track_tests in deserialized.items():
            for pw in track_tests.get("wilcoxon_pairwise", []):
                assert isinstance(pw["significant"], bool), (
                    f"{track} wilcoxon {pw['model_a']} vs {pw['model_b']}: "
                    f"significant is {type(pw['significant'])}, expected bool"
                )
            friedman = track_tests.get("friedman", {})
            if "significant" in friedman:
                assert isinstance(friedman["significant"], bool), (
                    f"{track} friedman: significant is {type(friedman['significant'])}, expected bool"
                )


class TestLatexGeneration:
    """Test LaTeX table generation."""

    def test_ganymede_latex(self) -> None:
        results = load_all_results("results")
        tables = build_comparison_table(results)
        latex = generate_latex_table("ganymede", tables["ganymede"])
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "mae" in latex.lower() or "MAE" in latex

    def test_3w_latex(self) -> None:
        results = load_all_results("results")
        tables = build_comparison_table(results)
        latex = generate_latex_table("3w", tables["3w"])
        assert r"\textbf{" in latex  # Best value bolded

    def test_cdf_latex(self) -> None:
        results = load_all_results("results")
        tables = build_comparison_table(results)
        latex = generate_latex_table("cdf", tables["cdf"])
        assert "error" in latex.lower()


class TestMultiHorizon:
    """Test multi-horizon loading and table generation."""

    def test_load_multihorizon_results(self) -> None:
        mh = load_multihorizon_results(Path("results"))
        assert "lstm" in mh
        # LSTM has all 4 horizons from the smoke test
        assert set(mh["lstm"].keys()) == {7, 14, 30, 90}

    def test_load_multihorizon_graceful_missing(self) -> None:
        """Baselines without h* files return no entries (not crash)."""
        mh = load_multihorizon_results(Path("results"))
        # Baselines don't have h* files — they should simply be absent
        assert "baselines" not in mh

    def test_generate_multihorizon_table(self) -> None:
        mh = load_multihorizon_results(Path("results"))
        baseline_mae = load_baseline_mae(Path("results"))
        latex = generate_multihorizon_table(mh, baseline_mae)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "7d" in latex and "14d" in latex and "30d" in latex and "90d" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert "lstm" in latex

    def test_generate_multihorizon_with_baseline(self) -> None:
        mh = load_multihorizon_results(Path("results"))
        baseline_mae = load_baseline_mae(Path("results"))
        latex = generate_multihorizon_table(mh, baseline_mae)
        assert "Seasonal Naive" in latex

    def test_generate_multihorizon_no_baseline(self) -> None:
        mh = load_multihorizon_results(Path("results"))
        latex = generate_multihorizon_table(mh, None)
        assert "Seasonal Naive" not in latex

    def test_load_baseline_mae(self) -> None:
        mae = load_baseline_mae(Path("results"))
        assert mae is not None
        assert mae > 0


class TestPerWell:
    """Test per-well loading and table generation."""

    def test_load_perwell_results(self) -> None:
        pw, mw = load_perwell_results(Path("results"))
        assert "lstm" in pw
        # LSTM has 7 wells
        assert len(pw["lstm"]) == 7

    def test_load_perwell_graceful_missing(self) -> None:
        """Baselines without per_well files return no entries."""
        pw, mw = load_perwell_results(Path("results"))
        assert "baselines" not in pw

    def test_load_perwell_multiwell_fallback(self) -> None:
        """Models without h30_multi_well.json fall back to ganymede.json."""
        _, mw = load_perwell_results(Path("results"))
        # deeponet has ganymede.json but not h30_multi_well.json
        assert "deeponet" in mw

    def test_generate_perwell_table(self) -> None:
        pw, mw = load_perwell_results(Path("results"))
        latex = generate_perwell_table(pw, mw)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\toprule" in latex
        assert "Well" in latex
        assert "MW" in latex
        assert "PW" in latex

    def test_generate_perwell_table_has_wells(self) -> None:
        pw, mw = load_perwell_results(Path("results"))
        latex = generate_perwell_table(pw, mw)
        # Well names contain Z0x patterns
        assert "Z01Z" in latex or "Z02Z" in latex

    def test_generate_perwell_table_empty(self) -> None:
        latex = generate_perwell_table({}, {})
        assert "No per-well results" in latex
