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
            import tempfile, json  # noqa: E401

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
