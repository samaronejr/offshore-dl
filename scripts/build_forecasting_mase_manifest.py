#!/usr/bin/env python3
"""Build a SLURM-array manifest for forecasting MASE repair reruns.

Each manifest row is one bounded production-script invocation:
``dataset, model, horizon, modes, wells``.
The first well chunk for a horizon includes both ``multi_well`` and that
chunk's ``per_well`` wells; later chunks are ``per_well`` only. This prevents
multi-well duplicates while keeping array tasks small enough for HPC limits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf

DATASETS = ("ganymede", "spe_berg", "volve", "inner_mongolia")
MODELS = ("lstm", "deeponet", "patchtst", "tcn", "chronos", "timesfm", "tirex")
CONFIG_DIR = Path("configs/data")


def chunks(values: list[str], size: int) -> list[list[str]]:
    if size < 1:
        msg = "chunk size must be >= 1"
        raise ValueError(msg)
    return [values[i : i + size] for i in range(0, len(values), size)]


def build_rows(
    datasets: list[str],
    models: list[str],
    wells_per_chunk: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        cfg = OmegaConf.load(CONFIG_DIR / f"{dataset}.yaml")
        horizons = [int(h) for h in cfg.data.forecasting.horizons]
        wells = [str(w) for w in cfg.data.get("wells", [])]
        well_chunks = chunks(wells, wells_per_chunk) if wells else [[]]
        for model in models:
            for horizon in horizons:
                if not wells:
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "horizon": str(horizon),
                            "modes": "multi_well",
                            "wells": "",
                        }
                    )
                    continue
                for chunk_idx, well_chunk in enumerate(well_chunks):
                    modes = "multi_well,per_well" if chunk_idx == 0 else "per_well"
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "horizon": str(horizon),
                            "modes": modes,
                            "wells": ",".join(well_chunk),
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wells-per-chunk", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()

    rows = build_rows(args.datasets, args.models, args.wells_per_chunk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "horizon", "modes", "wells"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rerun tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
