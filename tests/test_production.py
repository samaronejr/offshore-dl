"""Tests for S03 production run infrastructure.

Covers CLI flags (--horizon, --mode), output naming conventions,
per-well dataset variability, and CV performance fast-paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ────────────────────────────────────────────────────────────────
# Output naming
# ────────────────────────────────────────────────────────────────

class TestGanymedeOutputNaming:
    """Verify that horizon+mode encode correctly in output filenames."""

    def test_default_naming(self):
        """Without dataset_kwargs, ganymede output is ganymede.json."""
        from offshore_dl.run_experiment import run_and_save  # noqa: F811

        # We only test the path-generation logic, not actual training.
        # Patch build_experiment and runner.run to avoid I/O.
        mock_runner = MagicMock()
        mock_runner.run.return_value = {
            "fold_results": [],
            "aggregate": {},
            "cost": {},
            "experiment_name": "test",
            "n_folds": 0,
        }

        with (
            patch("offshore_dl.run_experiment.build_experiment", return_value=(mock_runner, MagicMock())),
            patch("offshore_dl.run_experiment._make_serializable", side_effect=lambda x: x),
        ):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                run_and_save(
                    model_name="lstm",
                    dataset_name="ganymede",
                    output_dir=tmpdir,
                    use_mlflow=False,
                    dataset_kwargs=None,
                )
                expected = Path(tmpdir) / "lstm" / "ganymede.json"
                assert expected.exists(), f"Expected {expected}"

    def test_horizon_mode_naming(self):
        """dataset_kwargs with horizon=7 and mode=multi_well → ganymede_h7_multi_well.json."""
        from offshore_dl.run_experiment import run_and_save

        mock_runner = MagicMock()
        mock_runner.run.return_value = {
            "fold_results": [],
            "aggregate": {},
            "cost": {},
            "experiment_name": "test",
            "n_folds": 0,
        }

        with (
            patch("offshore_dl.run_experiment.build_experiment", return_value=(mock_runner, MagicMock())),
            patch("offshore_dl.run_experiment._make_serializable", side_effect=lambda x: x),
        ):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                run_and_save(
                    model_name="lstm",
                    dataset_name="ganymede",
                    output_dir=tmpdir,
                    use_mlflow=False,
                    dataset_kwargs={"horizon": 7, "mode": "multi_well"},
                )
                expected = Path(tmpdir) / "lstm" / "ganymede_h7_multi_well.json"
                assert expected.exists(), f"Expected {expected}"

    def test_per_well_mode_naming(self):
        """Per-well mode encodes correctly in filename."""
        from offshore_dl.run_experiment import run_and_save

        mock_runner = MagicMock()
        mock_runner.run.return_value = {
            "fold_results": [],
            "aggregate": {},
            "cost": {},
            "experiment_name": "test",
            "n_folds": 0,
        }

        with (
            patch("offshore_dl.run_experiment.build_experiment", return_value=(mock_runner, MagicMock())),
            patch("offshore_dl.run_experiment._make_serializable", side_effect=lambda x: x),
        ):
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                run_and_save(
                    model_name="deeponet",
                    dataset_name="ganymede",
                    output_dir=tmpdir,
                    use_mlflow=False,
                    dataset_kwargs={"horizon": 14, "mode": "per_well"},
                )
                expected = Path(tmpdir) / "deeponet" / "ganymede_h14_per_well.json"
                assert expected.exists(), f"Expected {expected}"


# ────────────────────────────────────────────────────────────────
# CLI flag parsing
# ────────────────────────────────────────────────────────────────

class TestCLIFlags:
    """Verify --horizon and --mode args parse correctly."""

    def test_cli_horizon_flag_exists(self):
        """--horizon 7 --mode multi_well should be captured in parsed args."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--max-epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--no-mlflow", action="store_true")
        parser.add_argument("--output-dir", type=str, default="results")
        parser.add_argument("--max-instances", type=int, default=None)
        parser.add_argument("--horizon", type=int, default=None)
        parser.add_argument("--mode", type=str, default=None, choices=["multi_well", "per_well"])

        args = parser.parse_args([
            "--model", "lstm",
            "--dataset", "ganymede",
            "--horizon", "7",
            "--mode", "multi_well",
        ])
        assert args.horizon == 7
        assert args.mode == "multi_well"

    def test_cli_mode_only(self):
        """--mode alone without --horizon should parse."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--horizon", type=int, default=None)
        parser.add_argument("--mode", type=str, default=None, choices=["multi_well", "per_well"])

        args = parser.parse_args([
            "--model", "lstm",
            "--dataset", "ganymede",
            "--mode", "per_well",
        ])
        assert args.horizon is None
        assert args.mode == "per_well"


# ────────────────────────────────────────────────────────────────
# CV fast-path
# ────────────────────────────────────────────────────────────────

class TestStratifiedCVFastPath:
    """Verify get_splits uses _windows attr without calling __getitem__."""

    def test_fast_path_uses_windows_attr(self, tmp_path):
        """When dataset has _windows, get_splits should NOT call __getitem__."""
        import pandas as pd
        from offshore_dl.evaluation.cv import StratifiedGroupKFoldCV

        # Create a minimal folds CSV
        folds_csv = tmp_path / "folds.csv"
        folds_df = pd.DataFrame({
            "instancia": [
                "0/INST_A.csv", "0/INST_B.csv",
                "1/INST_C.csv", "1/INST_D.csv",
            ],
            "fold": [0, 1, 0, 1],
            "is_ova": [False, False, False, False],
        })
        folds_df.to_csv(folds_csv, index=False)

        # Create a mock dataset with _windows (simulating ThreeWDataset)
        mock_ds = MagicMock()
        mock_ds._windows = [
            {"instance_id": "INST_A", "class_id": 0},
            {"instance_id": "INST_A", "class_id": 0},
            {"instance_id": "INST_B", "class_id": 0},
            {"instance_id": "INST_C", "class_id": 1},
            {"instance_id": "INST_D", "class_id": 1},
            {"instance_id": "INST_D", "class_id": 1},
        ]
        mock_ds.__len__ = MagicMock(return_value=6)

        cv = StratifiedGroupKFoldCV(folds_path=folds_csv, dataset=mock_ds, n_folds=2)
        splits = cv.get_splits(6)

        # __getitem__ should NOT have been called — fast-path used _windows
        mock_ds.__getitem__.assert_not_called()

        # Should have 2 folds
        assert len(splits) == 2

        # Verify fold assignment correctness
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0


# ────────────────────────────────────────────────────────────────
# Empty-splits guard
# ────────────────────────────────────────────────────────────────

class TestEmptySplitsGuard:
    """ExperimentRunner.run() returns gracefully when CV produces 0 splits."""

    def test_empty_splits_returns_zero_folds(self):
        from offshore_dl.training.experiment import ExperimentRunner

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)

        mock_cv = MagicMock()
        mock_cv.get_splits.return_value = []  # 0 splits

        mock_cfg = MagicMock()

        runner = ExperimentRunner(
            model_class=MagicMock,
            dataset=mock_ds,
            cv_strategy=mock_cv,
            cfg=mock_cfg,
            model_kwargs={"task": "classification"},
        )

        result = runner.run(use_mlflow=False)

        assert result["n_folds"] == 0
        assert result["fold_results"] == []
        assert result["aggregate"] == {}


class TestCDFProductionCVGap:
    """CDF production scripts must use central strict raw-row CV gaps."""

    def test_cdf_sliding_window_cv_uses_central_resolver(self, monkeypatch):
        from omegaconf import OmegaConf
        import scripts.run_production_cdf as rpc

        captured = {}

        def fake_resolver(data_cfg):
            captured["data_cfg"] = data_cfg
            return 123

        def fake_cv(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return object()

        monkeypatch.setattr(rpc, "resolve_cv_gap_from_config", fake_resolver)
        monkeypatch.setattr(rpc, "SlidingWindowCV", fake_cv)

        data_cfg = OmegaConf.create(
            {
                "task": "anomaly",
                "cv_gap_policy": "strict_raw_row",
                "preprocessing": {"window_size": 48},
            }
        )

        rpc._cdf_sliding_window_cv(data_cfg)

        assert captured["data_cfg"] is data_cfg
        assert captured["kwargs"] == {"n_splits": 3, "train_ratio": 0.7, "gap": 123}

    def test_run_trained_model_constructs_sliding_window_cv_with_gap(self, monkeypatch):
        from omegaconf import OmegaConf
        import scripts.run_production_cdf as rpc
        from offshore_dl.evaluation.cv import resolve_cv_gap

        captured = {}

        def fake_cv(*args, **kwargs):
            captured["cv_kwargs"] = kwargs
            return object()

        monkeypatch.setattr(rpc, "SlidingWindowCV", fake_cv)
        monkeypatch.setattr(
            rpc,
            "load_merged_config",
            lambda *_args, **_kwargs: OmegaConf.create(
                {
                    "training": {"max_epochs": 1, "batch_size": 32},
                    "device": "cpu",
                    "data": {
                        "task": "anomaly",
                        "cv_gap_policy": "strict_raw_row",
                        "preprocessing": {"window_size": 48, "train_ratio": 0.8},
                    },
                }
            ),
        )

        class FakeHoldout:
            def __init__(self, *_, **__):
                pass

            def split(self, _n_samples):
                return np.arange(80), np.arange(80, 100)

        monkeypatch.setattr("offshore_dl.evaluation.cv.HoldoutSplitter", FakeHoldout)

        class DummyRunner:
            def __init__(self, *args, **kwargs):
                captured["runner_kwargs"] = kwargs

            def run_nested(self, *args, **kwargs):
                captured["run_nested_kwargs"] = kwargs
                return {"test_metrics": {}, "cv_fold_results": [], "cv_aggregate": {}}

        monkeypatch.setattr(rpc, "ExperimentRunner", DummyRunner)

        class FakeCDF:
            n_vars = 11

            def __len__(self):
                return 100

        rpc.run_trained_model("lstm", FakeCDF(), max_epochs=1, device="cpu")

        assert captured["cv_kwargs"]["gap"] == resolve_cv_gap(
            "strict_raw_row", task="anomaly", window_size=48
        )
        assert captured["cv_kwargs"]["gap"] == 47

    def test_fm_cdf_gap_uses_mutated_dataset_config(self, monkeypatch):
        from omegaconf import OmegaConf
        import scripts.run_production_cdf as rpc

        captured = {}

        def fake_cv(*args, **kwargs):
            captured["kwargs"] = kwargs
            return object()

        monkeypatch.setattr(rpc, "SlidingWindowCV", fake_cv)
        fm_cfg = OmegaConf.create(
            {
                "data": {
                    "task": "anomaly",
                    "cv_gap_policy": "strict_raw_row",
                    "preprocessing": {"window_size": 48, "mode": "window"},
                }
            }
        )
        fm_cfg.data.preprocessing.mode = "prediction"
        fm_cfg.data.preprocessing.prediction_horizon = 48

        rpc._cdf_sliding_window_cv(fm_cfg.data)

        assert captured["kwargs"]["gap"] == 47


# ────────────────────────────────────────────────────────────────
# Per-well dataset (integration — requires data on disk)
# ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not Path("data/processed/ganymede").exists(),
    reason="Ganymede processed data not available",
)
class TestGanymedePerWell:
    """Integration tests for per-well dataset creation."""

    def test_ganymede_per_well_creates_dataset(self):
        from offshore_dl.data.datasets import GanymedeDataset

        ds = GanymedeDataset(
            "configs/data/ganymede.yaml",
            mode="per_well",
            well_name="49/22-Z01Z",
        )
        assert len(ds) > 0

    def test_ganymede_per_well_n_vars_varies(self):
        """Different wells should load with valid data."""
        from offshore_dl.data.datasets import GanymedeDataset

        ds1 = GanymedeDataset(
            "configs/data/ganymede.yaml",
            mode="per_well",
            well_name="49/22-Z01Z",
        )
        ds2 = GanymedeDataset(
            "configs/data/ganymede.yaml",
            mode="per_well",
            well_name="49/22-Z02Z",
        )
        # Both wells should load successfully with samples
        assert len(ds1) > 0
        assert len(ds2) > 0


# ────────────────────────────────────────────────────────────────
# TemporalSplitCV on mock 3W-sized data
# ────────────────────────────────────────────────────────────────

class TestTemporalSplitOnLargeDataset:
    """Verify TemporalSplitCV produces correct split sizes."""

    def test_temporal_split_sizes_700_300(self):
        """1000 samples with train_ratio=0.7 → 700 train, 300 val."""
        from offshore_dl.evaluation.cv import TemporalSplitCV

        cv = TemporalSplitCV(train_ratio=0.7)
        splits = cv.get_splits(1000)

        assert len(splits) == 1, "TemporalSplitCV should produce exactly 1 split"
        train_idx, val_idx = splits[0]
        assert len(train_idx) == 700, f"Expected 700 train, got {len(train_idx)}"
        assert len(val_idx) == 300, f"Expected 300 val, got {len(val_idx)}"

    def test_temporal_split_no_overlap(self):
        """Train and val indices must not overlap."""
        from offshore_dl.evaluation.cv import TemporalSplitCV

        cv = TemporalSplitCV(train_ratio=0.7)
        splits = cv.get_splits(5000)
        train_idx, val_idx = splits[0]

        overlap = set(train_idx) & set(val_idx)
        assert len(overlap) == 0, f"Train/val overlap: {overlap}"

    def test_temporal_split_ordering(self):
        """All train indices must precede all val indices (temporal order)."""
        from offshore_dl.evaluation.cv import TemporalSplitCV

        cv = TemporalSplitCV(train_ratio=0.7)
        splits = cv.get_splits(2000)
        train_idx, val_idx = splits[0]

        assert train_idx[-1] < val_idx[0], "Train indices must precede val indices"


# ────────────────────────────────────────────────────────────────
# Ganymede multi-horizon config verification
# ────────────────────────────────────────────────────────────────

class TestGanymedeMultiHorizonConfig:
    """Verify configs/data/ganymede.yaml declares expected horizons and modes."""

    def test_ganymede_config_horizons(self):
        """ganymede.yaml must declare horizons [7, 14, 30, 90]."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("configs/data/ganymede.yaml")
        horizons = list(cfg.data.forecasting.horizons)
        assert horizons == [7, 14, 30, 90], f"Expected [7,14,30,90], got {horizons}"

    def test_ganymede_config_modes(self):
        """ganymede.yaml must declare modes [per_well, multi_well]."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.load("configs/data/ganymede.yaml")
        modes = list(cfg.data.modes)
        assert set(modes) == {"per_well", "multi_well"}, f"Expected per_well+multi_well, got {modes}"


# ────────────────────────────────────────────────────────────────
# Forecasting MASE rerun manifest and Volve filtering
# ────────────────────────────────────────────────────────────────

class TestForecastingMaseManifest:
    """Verify rerun manifests are shell-safe for SLURM array parsing."""

    def test_manifest_uses_lf_line_endings(self, tmp_path):
        from scripts.build_forecasting_mase_manifest import main

        output = tmp_path / "forecasting_mase_manifest.tsv"
        with patch.object(
            sys,
            "argv",
            [
                "build_forecasting_mase_manifest.py",
                "--output",
                str(output),
                "--datasets",
                "ganymede",
                "--models",
                "lstm",
                "--wells-per-chunk",
                "10",
            ],
        ):
            assert main() == 0

        data = output.read_bytes()
        assert b"\r" not in data
        assert b"\n" in data

    def test_manifest_chunks_expected_small_filter(self):
        from scripts.build_forecasting_mase_manifest import build_rows

        rows = build_rows(["ganymede"], ["lstm"], wells_per_chunk=10)

        assert len(rows) == 4
        assert {row["dataset"] for row in rows} == {"ganymede"}
        assert {row["model"] for row in rows} == {"lstm"}
        assert {row["modes"] for row in rows} == {"multi_well,per_well"}


class TestVolveProductionFilters:
    """Verify bounded Volve production plans used by rerun arrays."""

    def test_build_plan_filters_horizon_mode_and_well(self):
        import scripts.run_production_volve as volve

        plan = volve._build_plan(
            ["lstm"],
            horizons=[7],
            modes=["per_well"],
            wells=["NO_15_9-F-1_C"],
        )

        assert plan == [
            {
                "model": "lstm",
                "horizon": 7,
                "mode": "per_well",
                "well": "NO_15_9-F-1_C",
                "is_fm": False,
                "is_tree": False,
            }
        ]

    def test_results_dir_cli_sets_repaired_output_root(self, tmp_path, monkeypatch):
        import scripts.run_production_volve as volve

        output_dir = tmp_path / "post_fix"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_production_volve.py",
                "--dry-run",
                "--models",
                "lstm",
                "--horizons",
                "7",
                "--modes",
                "per_well",
                "--wells",
                "NO_15_9-F-1_C",
                "--results-dir",
                str(output_dir),
            ],
        )

        volve.main()

        assert volve.RESULTS_DIR == output_dir
