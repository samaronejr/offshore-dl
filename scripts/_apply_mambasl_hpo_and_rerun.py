#!/usr/bin/env python3
"""Apply MambaSL HPO best params to YAML config + production script, then re-run production.

Usage:
    python scripts/_apply_mambasl_hpo_and_rerun.py [--dry-run] [--device cuda]

Reads results/hpo/3w/mambasl.json, updates:
  1. configs/models/mambasl.yaml — architecture + training.lr (K015)
  2. scripts/run_production_3w_features.py — MODELS["mambasl"]["overrides"]
Then runs: python scripts/run_production_3w_features.py --models mambasl mambasl_raw --device <device>
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Apply MambaSL HPO results and re-run production")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without applying")
    parser.add_argument("--device", default="cuda", help="Device for production run")
    parser.add_argument("--skip-rerun", action="store_true", help="Only apply config, skip production run")
    args = parser.parse_args()

    hpo_path = Path("results/hpo/3w/mambasl.json")
    yaml_path = Path("configs/models/mambasl.yaml")
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

    # Architecture params to update in YAML (MambaSL search space per D018)
    yaml_arch_params = {
        "d_model": int,
        "d_state": int,
        "d_ff": int,
        "d_conv": int,
        "expand": int,
        "n_heads": int,
        "dropout": float,
    }

    new_yaml = yaml_text
    for param, cast in yaml_arch_params.items():
        if param in bp:
            val = cast(bp[param])
            # Match "    param: value" pattern in YAML
            pattern = rf"^(\s+{param}:\s*).*$"
            if isinstance(val, float):
                replacement = rf"\g<1>{val}"
            else:
                replacement = rf"\g<1>{val}"
            new_yaml = re.sub(pattern, replacement, new_yaml, flags=re.MULTILINE)
            print(f"  YAML {param}: {val}")

    # Training lr — MUST be in YAML per K015
    if "lr" in bp:
        lr_val = bp["lr"]
        pattern = r"^(\s+lr:\s*).*$"
        # Use scientific notation for small values
        if lr_val < 0.001:
            lr_str = f"{lr_val:.10e}"
        else:
            lr_str = f"{lr_val}"
        new_yaml = re.sub(pattern, rf"\g<1>{lr_str}", new_yaml, count=1, flags=re.MULTILINE)
        print(f"  YAML training.lr: {lr_str} (K015 — MUST be in YAML)")

    if args.dry_run:
        print("\n--- YAML (dry-run) ---")
        print(new_yaml)
    else:
        yaml_path.write_text(new_yaml)
        print(f"\n✓ Updated {yaml_path}")

    # ── 3. Update production script overrides ──
    script_text = script_path.read_text()

    # Build the new overrides dict string — all MambaSL arch params
    override_params = {}
    for p in ["d_model", "d_state", "d_ff", "d_conv", "expand", "n_heads", "dropout"]:
        if p in bp:
            override_params[p] = bp[p]

    # Format the overrides dict
    override_lines = []
    for k, v in override_params.items():
        if isinstance(v, float):
            override_lines.append(f'            "{k}": {v},')
        else:
            override_lines.append(f'            "{k}": {v},')

    overrides_block = "{\n" + "\n".join(override_lines) + "\n        }"

    # Replace the overrides dict in MODELS["mambasl"]
    # Handles both empty `{}` and already-populated `{...}` dicts
    mambasl_section = re.search(
        r'(MODELS\["mambasl"\]\s*=\s*\{.*?"overrides":\s*)\{[^}]*\}',
        script_text,
        re.DOTALL,
    )
    if mambasl_section:
        new_script = (
            script_text[:mambasl_section.start(1)]
            + mambasl_section.group(1)
            + overrides_block
            + script_text[mambasl_section.end():]
        )
        if args.dry_run:
            # Show just the changed region
            new_section = re.search(
                r'MODELS\["mambasl"\]\s*=\s*\{.*?\}.*?\}',
                new_script,
                re.DOTALL,
            )
            if new_section:
                print(f"\n--- Script overrides (dry-run) ---")
                print(new_section.group(0))
        else:
            script_path.write_text(new_script)
            print(f"✓ Updated {script_path} MODELS['mambasl']['overrides']")
    else:
        print("WARNING: Could not find MODELS['mambasl'] overrides block in script", file=sys.stderr)

    # ── 4. Run production ──
    if args.skip_rerun:
        print("\n--skip-rerun: skipping production run")
        return

    if args.dry_run:
        print(f"\nDry-run: would run: python scripts/run_production_3w_features.py --models mambasl mambasl_raw --device {args.device}")
        return

    print(f"\n{'='*60}")
    print(f"Running production: mambasl + mambasl_raw on {args.device}")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "scripts/run_production_3w_features.py",
        "--models", "mambasl", "mambasl_raw",
        "--device", args.device,
    ]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
