from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from offshore_dl.analysis import forecasting_audit as audit


def test_collect_result_rows_reads_mase_and_unavailable_status(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    ok_dir = results_dir / "lstm"
    unavail_dir = results_dir / "timesfm"
    ok_dir.mkdir(parents=True)
    unavail_dir.mkdir(parents=True)
    (ok_dir / "ganymede_h7_multi_well.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "test_metrics": {
                    "mae": 1.0,
                    "rmse": 2.0,
                    "r2": 0.5,
                    "r2_prod": 0.6,
                    "mase": 0.7,
                },
            }
        )
    )
    (unavail_dir / "spe_berg_h7_multi_well.json").write_text(
        json.dumps({"status": "unavailable", "reason": "missing dependency", "test_metrics": {}})
    )

    df = audit.collect_result_rows(results_dir)

    assert len(df) == 2
    ok = df[df["model"].eq("lstm")].iloc[0]
    assert ok["dataset"] == "ganymede"
    assert ok["horizon"] == 7
    assert ok["mode"] == "multi_well"
    assert ok["mase"] == pytest.approx(0.7)
    unavailable = df[df["model"].eq("timesfm")].iloc[0]
    assert unavailable["status"] == "unavailable"
    assert np.isnan(unavailable["mase"])


def test_collect_result_rows_resolves_prefix_postfix_layout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    pre_dir = results_root / "pre_fix" / "lstm"
    post_dir = results_root / "post_fix" / "tcn"
    pre_dir.mkdir(parents=True)
    post_dir.mkdir(parents=True)
    (pre_dir / "ganymede_h7_multi_well.json").write_text(
        json.dumps({"status": "ok", "test_metrics": {"mae": 1.0, "mase": 1.0}})
    )
    (post_dir / "ganymede_h7_multi_well.json").write_text(
        json.dumps({"status": "ok", "test_metrics": {"mae": 0.5, "mase": 0.5}})
    )

    df = audit.collect_result_rows(results_root)

    assert set(df["model"]) == {"tcn"}
    assert df.iloc[0]["file_path"].startswith(str(results_root / "post_fix"))


def test_collect_result_rows_keeps_current_mase_metadata(tmp_path: Path) -> None:
    model_dir = tmp_path / "results" / "lstm"
    model_dir.mkdir(parents=True)
    (model_dir / "ganymede_h7_multi_well.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "test_metrics": {
                    "mae": 1.0,
                    "mase": 0.9,
                    "mase_flat": 1.2,
                    "mase_group_weighted": 0.9,
                    "mase_group_macro": 1.1,
                    "mase_aggregation": "group_weighted",
                    "mase_denominator_source": "grouped_train",
                },
            }
        )
    )

    df = audit.collect_result_rows(tmp_path / "results")
    row = df.iloc[0]

    assert row["mase_group_weighted"] == pytest.approx(0.9)
    assert row["mase_aggregation"] == "group_weighted"
    assert row["mase_denominator_source"] == "grouped_train"


def test_select_mase_audit_candidates_includes_zero_mase_nonzero_mae() -> None:
    rows = pd.DataFrame(
        [
            {
                "model": "lstm",
                "dataset": "ganymede",
                "horizon": 7,
                "mode": "per_well",
                "well": "49_22-Z01Z",
                "status": "ok",
                "mae": 0.2,
                "mase": 0.0,
                "file_path": "results/lstm/ganymede_h7_per_well_49_22-Z01Z.json",
            },
            {
                "model": "chronos",
                "dataset": "ganymede",
                "horizon": 7,
                "mode": "multi_well",
                "well": "",
                "status": "ok",
                "mae": 0.1,
                "mase": 1.0,
                "file_path": "results/chronos/ganymede_h7_multi_well.json",
            },
        ]
    )

    selected = audit.select_mase_audit_candidates(rows)

    hit = selected[selected["model"].eq("lstm")].iloc[0]
    assert "zero_mase_nonzero_mae" in hit["row_type"]


def test_weighted_mase_summary_collapses_per_well_rows_before_cross_dataset_rank() -> None:
    rows = []
    # Ten per-well rows in one scenario should become one scenario value.
    for i in range(10):
        rows.append(
            {
                "dataset": "ganymede",
                "horizon": 7,
                "mode": "per_well",
                "well": f"well_{i}",
                "model": "model_a",
                "status": "ok",
                "effective_mase": 1.0,
            }
        )
        rows.append(
            {
                "dataset": "ganymede",
                "horizon": 7,
                "mode": "per_well",
                "well": f"well_{i}",
                "model": "model_b",
                "status": "ok",
                "effective_mase": 3.0,
            }
        )
    # A second scenario with the opposite winner gets equal cross-scenario weight.
    rows.extend(
        [
            {
                "dataset": "volve",
                "horizon": 7,
                "mode": "multi_well",
                "well": "",
                "model": "model_a",
                "status": "ok",
                "effective_mase": 4.0,
            },
            {
                "dataset": "volve",
                "horizon": 7,
                "mode": "multi_well",
                "well": "",
                "model": "model_b",
                "status": "ok",
                "effective_mase": 2.0,
            },
        ]
    )

    summary = audit.compute_weighted_mase_summary(pd.DataFrame(rows))

    a = summary.set_index("model").loc["model_a"]
    b = summary.set_index("model").loc["model_b"]
    assert a["scenario_count"] == 2
    assert b["scenario_count"] == 2
    assert a["mean_mase"] == pytest.approx((1.0 + 4.0) / 2)
    assert b["mean_mase"] == pytest.approx((3.0 + 2.0) / 2)
    assert a["mean_rank"] == pytest.approx(1.5)
    assert b["mean_rank"] == pytest.approx(1.5)


def test_denominator_source_rules_cover_known_writer_paths() -> None:
    assert audit.denominator_source_for("lstm", "ok") == "double_denorm_suspect"
    assert audit.denominator_source_for("chronos", "ok") == "validation_target_fallback"
    assert audit.denominator_source_for("chronos_finetuned", "ok") == "ad_hoc_validation_fallback"
    assert audit.denominator_source_for("lstm", "ok", "raw_train") == "raw_train"
    assert audit.denominator_source_for("lstm", "ok", "grouped_train") == "raw_train"
    assert (
        audit.denominator_source_for("chronos", "ok", "evaluation_targets_fallback")
        == "validation_target_fallback"
    )
    assert (
        audit.denominator_source_for("chronos", "ok", "grouped_eval")
        == "validation_target_fallback"
    )
    assert audit.denominator_source_for("lstm", "ok", "missing_order") == "unavailable"
    assert audit.denominator_source_for("unknown_model", "ok") == "unknown"
    assert audit.denominator_source_for("timesfm", "unavailable") == "unavailable"
