#!/usr/bin/env python3
"""
Validate SPE BERG production sweep results.

Checks all expected result JSONs exist and contain valid metrics.
Handles gracefully:
  - status=unavailable stubs (TimesFM/TiRex Python-incompatible)
  - 42 active wells (well_12..well_53; well_1..11 have 0 samples after shutdown filter)

Exit codes:
  0 — All expected files found and valid (or valid stubs for unavailable models)
  1 — Missing files, invalid metrics, or malformed JSON detected
"""

import json
import math
import sys
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")

MODELS = ["lstm", "deeponet", "patchtst", "tcn", "chronos", "timesfm", "tirex"]
HORIZONS = [7, 14, 30, 90]

# Wells 1-11 produce 0 samples after shutdown filtering (no production data).
# The SPE BERG dataset has wells well_1..well_53 but 1-11 are empty.
# Only well_12..well_53 (42 wells) have data and produce per_well files.
ACTIVE_WELL_IDS = list(range(12, 54))  # [12, 13, ..., 53]
N_ACTIVE_WELLS = len(ACTIVE_WELL_IDS)  # 42

# Models known to be unavailable in Python 3.13 — produce status=unavailable stubs.
UNAVAILABLE_MODELS = {"timesfm", "tirex"}

# ─── Expected file counts ─────────────────────────────────────────────────────
# multi_well: 7 models × 4 horizons = 28 files
# per_well:   7 models × 4 horizons × 42 active wells = 1,176 files
# Total:      1,204 files
EXPECTED_MULTI_WELL = len(MODELS) * len(HORIZONS)
EXPECTED_PER_WELL = len(MODELS) * len(HORIZONS) * N_ACTIVE_WELLS
EXPECTED_TOTAL = EXPECTED_MULTI_WELL + EXPECTED_PER_WELL


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_finite_float(value) -> bool:
    """Return True iff value is a finite Python float/int (not NaN, not Inf)."""
    try:
        f = float(value)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _validate_metrics(data: dict, path: Path) -> list[str]:
    """
    Validate that data contains a metrics dict with finite mae and r2.

    Returns a list of error strings (empty = valid).
    Accepts both test_metrics and cv_aggregate as sources of truth.
    For status=unavailable stubs, skips metric checks.
    """
    errors: list[str] = []

    # Stubs from unavailable models: status=unavailable, metrics={}
    if data.get("status") == "unavailable":
        # Validate stub structure — should have the expected stub keys
        required_stub_keys = {"test_metrics", "cv_aggregate", "status", "reason"}
        missing = required_stub_keys - set(data.keys())
        if missing:
            errors.append(f"  INVALID_STUB: missing keys {missing}")
        return errors

    # Real result: must have test_metrics or cv_aggregate
    metrics = data.get("test_metrics") or data.get("cv_aggregate") or {}
    if not metrics:
        errors.append("  NO_METRICS: test_metrics and cv_aggregate both empty or absent")
        return errors

    # For cv_aggregate, metrics are stored as {key_mean: v, key_std: v}
    # For test_metrics, metrics are stored as {key: v}
    # Determine which format we have
    is_aggregate = "mae_mean" in metrics or "r2_mean" in metrics

    if is_aggregate:
        mae_key, r2_key = "mae_mean", "r2_mean"
    else:
        mae_key, r2_key = "mae", "r2"

    # MAE: must be a positive finite float
    if mae_key not in metrics:
        errors.append(f"  MISSING_MAE: key '{mae_key}' not in metrics")
    else:
        mae = metrics[mae_key]
        if not _is_finite_float(mae):
            errors.append(f"  INVALID_MAE: {mae_key}={mae!r} is not finite")
        elif float(mae) < 0:
            errors.append(f"  NEGATIVE_MAE: {mae_key}={mae!r} must be >= 0")

    # R²: must be a finite float (can be negative for bad models)
    if r2_key not in metrics:
        errors.append(f"  MISSING_R2: key '{r2_key}' not in metrics")
    else:
        r2 = metrics[r2_key]
        if not _is_finite_float(r2):
            errors.append(f"  INVALID_R2: {r2_key}={r2!r} is not finite")

    return errors


def _load_json(path: Path):
    """Load JSON from path. Returns (data, error_str). On failure error_str is non-empty."""
    if not path.exists():
        return None, "MISSING"
    if path.stat().st_size == 0:
        return None, "EMPTY_FILE"
    try:
        with path.open() as f:
            return json.load(f), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"MALFORMED_JSON: {exc}"


# ─── Main validation ──────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print(f"SPE BERG PRODUCTION RESULTS VALIDATION")
    print(f"  Expected: {EXPECTED_MULTI_WELL} multi_well + {EXPECTED_PER_WELL} per_well = {EXPECTED_TOTAL} total")
    print(f"  Active wells: {N_ACTIVE_WELLS} (well_12..well_53; well_1..11 are empty datasets)")
    print(f"  Unavailable model stubs: {sorted(UNAVAILABLE_MODELS)} (Python 3.13 incompatible)")
    print("=" * 70)

    n_found = 0
    n_valid = 0
    n_missing = 0
    n_invalid = 0

    missing_files: list[str] = []
    invalid_files: list[tuple[str, list[str]]] = []

    # ── Validate multi_well files ──────────────────────────────────────────────
    print("\n[1/2] Checking multi_well files ({} expected)...".format(EXPECTED_MULTI_WELL))
    for model in MODELS:
        for h in HORIZONS:
            path = RESULTS_DIR / model / f"spe_berg_h{h}_multi_well.json"
            data, err = _load_json(path)

            if err:
                n_missing += 1
                missing_files.append(str(path))
                print(f"  ✗ MISSING  {path.relative_to(RESULTS_DIR)}")
                continue

            n_found += 1
            metric_errors = _validate_metrics(data, path)
            if metric_errors:
                n_invalid += 1
                invalid_files.append((str(path), metric_errors))
                print(f"  ✗ INVALID  {path.relative_to(RESULTS_DIR)}: {metric_errors[0].strip()}")
            else:
                n_valid += 1
                status = data.get("status", "ok")
                if status == "unavailable":
                    label = f"  ✓ STUB     {path.relative_to(RESULTS_DIR)} (unavailable)"
                else:
                    tm = data.get("test_metrics") or {}
                    mae = tm.get("mae", "?")
                    r2 = tm.get("r2", "?")
                    label = f"  ✓ OK       {path.relative_to(RESULTS_DIR)}  mae={mae:.4f}  r2={r2:.4f}" if isinstance(mae, float) else f"  ✓ OK       {path.relative_to(RESULTS_DIR)}"
                print(label)

    # ── Validate per_well files ────────────────────────────────────────────────
    print(f"\n[2/2] Checking per_well files ({EXPECTED_PER_WELL} expected across {N_ACTIVE_WELLS} wells)...")
    per_well_ok = 0
    per_well_missing = 0
    per_well_invalid = 0

    for model in MODELS:
        for h in HORIZONS:
            for well_id in ACTIVE_WELL_IDS:
                well_name = f"well_{well_id}"
                path = RESULTS_DIR / model / f"spe_berg_h{h}_per_well_{well_name}.json"
                data, err = _load_json(path)

                if err:
                    n_missing += 1
                    per_well_missing += 1
                    missing_files.append(str(path))
                    continue

                n_found += 1
                metric_errors = _validate_metrics(data, path)
                if metric_errors:
                    n_invalid += 1
                    per_well_invalid += 1
                    invalid_files.append((str(path), metric_errors))
                else:
                    n_valid += 1
                    per_well_ok += 1

    print(f"  per_well results: {per_well_ok} valid, {per_well_missing} missing, {per_well_invalid} invalid")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Files expected:   {EXPECTED_TOTAL}")
    print(f"  Files found:      {n_found}")
    print(f"  Files valid:      {n_valid}")
    print(f"  Files missing:    {n_missing}")
    print(f"  Files invalid:    {n_invalid}")
    print("=" * 70)

    if missing_files:
        print(f"\nMISSING ({len(missing_files)}):")
        # Show first 20 missing files to avoid flooding output
        for p in missing_files[:20]:
            print(f"  - {p}")
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more")

    if invalid_files:
        print(f"\nINVALID ({len(invalid_files)}):")
        for p, errs in invalid_files[:20]:
            print(f"  - {p}: {errs[0].strip()}")
        if len(invalid_files) > 20:
            print(f"  ... and {len(invalid_files) - 20} more")

    # Exit 0 only when all expected files are present and valid
    if n_missing == 0 and n_invalid == 0 and n_found == EXPECTED_TOTAL:
        print("\n✅ ALL RESULTS VALID — sweep complete")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED — {n_missing} missing, {n_invalid} invalid")
        return 1


if __name__ == "__main__":
    sys.exit(main())
