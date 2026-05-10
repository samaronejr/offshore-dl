#!/usr/bin/env python3
"""Validate and summarize Stage 1 3W classification HPO campaign outputs.

A result is accepted only when it contains final benchmark evidence: HPO best
params/value/trial count, non-empty CV aggregate, held-out test metrics with
``f1_macro``, and split metadata. This intentionally rejects historical partial
JSONs that contain tuned params but no final evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from offshore_dl.utils.serialization import make_serializable

DEFAULT_OUTPUT_DIR = Path("results/hpo")
DEFAULT_MANIFEST = Path("scripts/hpo_3w_models.txt")
REQUIRED_SPLIT_KEYS = ("n_train", "n_test", "n_cv_folds")


def load_manifest(path: Path) -> list[str]:
    """Load model names from a newline manifest, ignoring comments/blanks."""
    models: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            models.append(line)
    return models


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def nonempty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def split_metadata(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("split_metadata")
    if nonempty_mapping(meta):
        return dict(meta)
    return {k: data.get(k) for k in REQUIRED_SPLIT_KEYS if data.get(k) is not None}


def validate_result(data: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Return (valid, reasons)."""
    reasons: list[str] = []
    if data is None:
        return False, ["missing_or_invalid_json"]

    hpo = data.get("hpo")
    if not nonempty_mapping(hpo):
        reasons.append("missing_hpo")
    else:
        if not nonempty_mapping(hpo.get("best_params")):
            reasons.append("missing_hpo.best_params")
        if hpo.get("best_value") is None:
            reasons.append("missing_hpo.best_value")
        if hpo.get("n_trials") is None:
            reasons.append("missing_hpo.n_trials")

    test_metrics = data.get("test_metrics")
    if not nonempty_mapping(test_metrics):
        reasons.append("missing_test_metrics")
    elif test_metrics.get("f1_macro") is None:
        reasons.append("missing_test_metrics.f1_macro")

    if not nonempty_mapping(data.get("cv_aggregate")):
        reasons.append("missing_cv_aggregate")

    meta = split_metadata(data)
    for key in REQUIRED_SPLIT_KEYS:
        if meta.get(key) is None:
            reasons.append(f"missing_split_metadata.{key}")

    if data.get("status") == "error":
        reasons.append("status_error")

    return not reasons, reasons


def result_path(output_dir: Path, campaign_id: str, model: str) -> Path:
    return output_dir / "3w" / campaign_id / f"{model}.json"


def build_summary(
    *,
    output_dir: Path,
    campaign_id: str,
    models: list[str],
    allow_incomplete: bool,
) -> tuple[dict[str, Any], bool]:
    summary: dict[str, Any] = {}
    all_valid = True
    for model in models:
        path = result_path(output_dir, campaign_id, model)
        data = load_json(path)
        valid, reasons = validate_result(data)
        if not valid and not allow_incomplete:
            all_valid = False
        hpo = data.get("hpo", {}) if isinstance(data, dict) else {}
        test_metrics = data.get("test_metrics", {}) if isinstance(data, dict) else {}
        summary[model] = {
            "status": "ok" if valid else "invalid",
            "path": str(path),
            "reasons": reasons,
            "best_value": hpo.get("best_value"),
            "n_trials": hpo.get("n_trials"),
            "test_f1_macro": test_metrics.get("f1_macro"),
            "test_accuracy": test_metrics.get("accuracy"),
            "hpo": hpo,
            "test_metrics": test_metrics,
            "cv_aggregate": data.get("cv_aggregate", {}) if isinstance(data, dict) else {},
            "split_metadata": split_metadata(data) if isinstance(data, dict) else {},
        }
    return summary, all_valid


def print_table(summary: dict[str, Any]) -> None:
    print("model             status   best_value   test_f1    trials  path/reasons")
    print("-" * 100)
    for model, row in summary.items():
        best = row.get("best_value")
        f1 = row.get("test_f1_macro")
        trials = row.get("n_trials")
        detail = row["path"] if row["status"] == "ok" else ",".join(row["reasons"])
        print(
            f"{model:<17} {row['status']:<8} "
            f"{best if best is not None else '-':>10} "
            f"{f1 if f1 is not None else '-':>9} "
            f"{trials if trials is not None else '-':>7}  {detail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True, help="Campaign id under results/hpo/3w/")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="HPO output root")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Model manifest path")
    parser.add_argument("--models", nargs="+", default=None, help="Validate only these models")
    parser.add_argument("--write-summary", action="store_true", help="Write summary.json if validation passes")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        default=os.environ.get("ALLOW_INCOMPLETE") == "1",
        help="Exit 0 even if some outputs are incomplete (summary marks them invalid)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    models = args.models or load_manifest(Path(args.manifest))
    if not models:
        parser.error("no models supplied and manifest is empty/missing")

    summary, all_valid = build_summary(
        output_dir=output_dir,
        campaign_id=args.campaign_id,
        models=models,
        allow_incomplete=args.allow_incomplete,
    )
    print_table(summary)

    if args.write_summary:
        if not all_valid and not args.allow_incomplete:
            print("\nRefusing to write final summary: validation failed", file=sys.stderr)
        else:
            summary_path = output_dir / "3w" / args.campaign_id / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(make_serializable(summary), indent=2))
            print(f"\nSummary written: {summary_path}")

    if all_valid or args.allow_incomplete:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
