#!/usr/bin/env python3
"""Apply ConvTimeNet HPO best params to YAML config + production script, then re-run production.

Usage:
    python scripts/_apply_convtimenet_hpo_and_rerun.py [--dry-run] [--device cuda]

Reads results/hpo/3w/convtimenet.json, updates:
  1. configs/models/convtimenet.yaml — architecture + training.lr (K015)
  2. scripts/run_production_3w_features.py — MODELS["convtimenet"]["overrides"]
Then runs: python scripts/run_production_3w_features.py --models convtimenet convtimenet_raw --device <device>

Special params (unique to ConvTimeNet):
  - dw_ks : list of ints (e.g. [7, 13, 19]) — YAML list syntax + Python list literal in overrides
  - pooling_tp : string categorical ("max" / "mean") — needs quotes in YAML + string in overrides

Per D019: convtimenet_raw inherits architecture params from MODELS["convtimenet"]["overrides"]
at runtime, so the overrides dict MUST be updated here.
Per K015: lr MUST be updated in YAML (training.lr), not just overrides.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Apply ConvTimeNet HPO results and re-run production")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying")
    parser.add_argument("--device", default="cuda", help="Device for production run")
    parser.add_argument("--skip-rerun", action="store_true", help="Only apply config, skip production run")
    args = parser.parse_args()

    hpo_path = Path("results/hpo/3w/convtimenet.json")
    yaml_path = Path("configs/models/convtimenet.yaml")
    script_path = Path("scripts/run_production_3w_features.py")

    # ── 1. Read HPO results ──
    if not hpo_path.exists():
        print(f"ERROR: {hpo_path} does not exist. HPO must complete first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(hpo_path.read_text())
    bp = data["hpo"]["best_params"]
    print(f"HPO best params: {json.dumps(bp, indent=2)}")
    print(f"HPO best value (f1_macro): {data['hpo']['best_value']:.4f}")
    print(f"HPO trials completed: {data['hpo']['n_trials']}")

    # ── 2. Update YAML config ──
    yaml_text = yaml_path.read_text()

    # Scope all regex substitutions to ONLY the architecture+training section, never the
    # optuna_search_space block (which also contains the same param names as YAML keys).
    # Splitting at the optuna_search_space line prevents false matches.
    search_space_marker = "  optuna_search_space:"
    if search_space_marker in yaml_text:
        split_idx = yaml_text.index(search_space_marker)
        yaml_prefix = yaml_text[:split_idx]
        yaml_suffix = yaml_text[split_idx:]
    else:
        yaml_prefix = yaml_text
        yaml_suffix = ""

    # Scalar architecture params to update in YAML (ConvTimeNet search space per D019)
    # dw_ks and pooling_tp are handled separately below (list and string types)
    # Use [ \t]* (not \s*) to avoid crossing newlines in the capture group.
    yaml_scalar_params = {
        "d_model": int,
        "d_ff": int,
        "patch_size": int,
        "patch_stride": int,
        "dropout": float,
    }

    new_prefix = yaml_prefix
    for param, cast in yaml_scalar_params.items():
        if param in bp:
            val = cast(bp[param])
            pattern = rf"^(\s+{param}:[ \t]*).*$"
            new_prefix = re.sub(pattern, rf"\g<1>{val}", new_prefix, flags=re.MULTILINE)
            print(f"  YAML {param}: {val}")

    # dw_ks — list param: replace "dw_ks: [...]" with updated list
    if "dw_ks" in bp:
        dw_ks_val = bp["dw_ks"]
        # Ensure it's a list of ints
        if not isinstance(dw_ks_val, list):
            dw_ks_val = list(dw_ks_val)
        dw_ks_str = "[" + ", ".join(str(int(k)) for k in dw_ks_val) + "]"
        pattern = r"^(\s+dw_ks:[ \t]*).*$"
        new_prefix = re.sub(pattern, rf"\g<1>{dw_ks_str}", new_prefix, flags=re.MULTILINE)
        print(f"  YAML dw_ks: {dw_ks_str}")

    # pooling_tp — string param: must be quoted in YAML (e.g. pooling_tp: "max")
    if "pooling_tp" in bp:
        pooling_val = str(bp["pooling_tp"])
        pattern = r'^(\s+pooling_tp:[ \t]*).*$'
        new_prefix = re.sub(pattern, rf'\g<1>"{pooling_val}"', new_prefix, flags=re.MULTILINE)
        print(f'  YAML pooling_tp: "{pooling_val}"')

    # Training lr — MUST be in YAML per K015
    if "lr" in bp:
        lr_val = bp["lr"]
        pattern = r"^(\s+lr:[ \t]*).*$"
        # Use scientific notation for small values
        if lr_val < 0.001:
            lr_str = f"{lr_val:.10e}"
        else:
            lr_str = f"{lr_val}"
        new_prefix = re.sub(pattern, rf"\g<1>{lr_str}", new_prefix, count=1, flags=re.MULTILINE)
        print(f"  YAML training.lr: {lr_str} (K015 — MUST be in YAML)")

    new_yaml = new_prefix + yaml_suffix

    if args.dry_run:
        print("\n--- YAML (dry-run) ---")
        print(new_yaml)
    else:
        yaml_path.write_text(new_yaml)
        print(f"\n✓ Updated {yaml_path}")

    # ── 3. Update production script overrides ──
    script_text = script_path.read_text()

    # Build overrides dict for all ConvTimeNet arch params (D019: convtimenet_raw inherits these)
    override_params_order = ["d_model", "d_ff", "patch_size", "patch_stride", "dw_ks", "dropout", "pooling_tp"]

    # Format the overrides dict lines — special handling for dw_ks (list) and pooling_tp (str)
    override_lines = []
    for p in override_params_order:
        if p not in bp:
            continue
        v = bp[p]
        if p == "dw_ks":
            # Must be a Python list literal of ints
            if not isinstance(v, list):
                v = list(v)
            list_lit = "[" + ", ".join(str(int(k)) for k in v) + "]"
            override_lines.append(f'            "{p}": {list_lit},')
        elif p == "pooling_tp":
            # Must be a quoted string in the Python dict
            override_lines.append(f'            "{p}": "{v}",')
        elif isinstance(v, float):
            override_lines.append(f'            "{p}": {v},')
        else:
            override_lines.append(f'            "{p}": {v},')

    overrides_block = "{\n" + "\n".join(override_lines) + "\n        }"

    # Replace the overrides dict in MODELS["convtimenet"]
    # Handles both empty `{}` and already-populated `{...}` dicts
    ct_section = re.search(
        r'(MODELS\["convtimenet"\]\s*=\s*\{.*?"overrides":\s*)\{[^}]*\}',
        script_text,
        re.DOTALL,
    )
    if ct_section:
        new_script = (
            script_text[:ct_section.start(1)]
            + ct_section.group(1)
            + overrides_block
            + script_text[ct_section.end():]
        )
        if args.dry_run:
            # Show just the changed region for visibility
            new_section = re.search(
                r'MODELS\["convtimenet"\]\s*=\s*\{.*?\}.*?\}',
                new_script,
                re.DOTALL,
            )
            if new_section:
                print(f"\n--- Script overrides (dry-run) ---")
                print(new_section.group(0))
        else:
            script_path.write_text(new_script)
            print(f"✓ Updated {script_path} MODELS['convtimenet']['overrides']")
    else:
        print("WARNING: Could not find MODELS['convtimenet'] overrides block in script", file=sys.stderr)

    # ── 4. Run production ──
    if args.skip_rerun:
        print("\n--skip-rerun: skipping production run")
        return

    if args.dry_run:
        print(f"\nDry-run: would run: python scripts/run_production_3w_features.py --models convtimenet convtimenet_raw --device {args.device}")
        return

    print(f"\n{'='*60}")
    print(f"Running production: convtimenet + convtimenet_raw on {args.device}")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "scripts/run_production_3w_features.py",
        "--models", "convtimenet", "convtimenet_raw",
        "--device", args.device,
    ]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
