"""Targeted TCN run: only multi_well for h14, h30, h90 (h7 already done)."""
import sys, importlib.util
from pathlib import Path

# Direct import of run_production_ganymede module
spec = importlib.util.spec_from_file_location(
    "run_production_ganymede",
    Path(__file__).resolve().parent / "run_production_ganymede.py",
)
rpg = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
spec.loader.exec_module(rpg)

HORIZONS = [14, 30, 90]

for h in HORIZONS:
    out = rpg.RESULTS_DIR / "tcn" / f"ganymede_h{h}_multi_well.json"
    if out.exists():
        print(f"SKIP h={h}: {out} already exists")
        continue
    print(f"\n{'='*60}")
    print(f"  Running TCN h={h} multi_well")
    print(f"{'='*60}", flush=True)
    rpg._run_trained_model("tcn", h, "multi_well", None, None, "cuda", use_mlflow=False)
    print(f"  DONE: {out}", flush=True)

print("\nAll multi_well runs complete.")
