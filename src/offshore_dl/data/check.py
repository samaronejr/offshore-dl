"""Diagnostic script to validate all datasets load correctly.

Usage::

    python -m offshore_dl.data.check

Loads each dataset with minimal parameters, prints summary stats,
and exits 0 on success or 1 on any failure.
"""

from __future__ import annotations

import logging
import sys
import time

logger = logging.getLogger(__name__)


def check_dataset(name: str, create_fn) -> bool:
    """Load a dataset and print summary stats."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    try:
        t0 = time.time()
        ds = create_fn()
        elapsed = time.time() - t0

        print(f"  Length:    {len(ds):,}")
        print(f"  Load time: {elapsed:.2f}s")

        # Get a sample
        sample = ds[0]
        if len(sample) == 3:
            feat, target, meta = sample
            print(f"  Input:    shape={tuple(feat.shape)}, dtype={feat.dtype}")
            if hasattr(target, "shape"):
                print(f"  Target:   shape={tuple(target.shape)}, dtype={target.dtype}")
            else:
                print(f"  Target:   {target} (type={type(target).__name__})")
            print(f"  Metadata: {list(meta.keys())}")

            # NaN check
            import torch
            nan_input = torch.isnan(feat).sum().item()
            print(f"  NaN (input):  {nan_input}")
            if hasattr(target, "shape"):
                nan_target = torch.isnan(target).sum().item()
                print(f"  NaN (target): {nan_target}")

        # Dataset-level metadata
        meta = ds.get_metadata()
        for k, v in meta.items():
            if k not in ("class", "length"):
                print(f"  {k}: {v}")

        print(f"  ✓ OK")
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        logger.exception("Dataset check failed: %s", name)
        return False


def main() -> int:
    """Run all dataset checks."""
    from offshore_dl.data.datasets import CDFDataset, GanymedeDataset, ThreeWDataset

    results = []

    # ThreeW — use subset for speed
    results.append(check_dataset(
        "ThreeWDataset (3 classes, 2 instances each)",
        lambda: ThreeWDataset(
            "configs/data/3w.yaml",
            window_size=100,
            window_stride=5000,
            cache_in_memory=False,
            class_ids=[0, 1, 7],
            max_instances_per_class=2,
        ),
    ))

    # CDF
    results.append(check_dataset(
        "CDFDataset (reconstruction mode)",
        lambda: CDFDataset("configs/data/cdf.yaml", mode="reconstruction", window_stride=100),
    ))

    # Ganymede
    results.append(check_dataset(
        "GanymedeDataset (multi-well, horizon=30)",
        lambda: GanymedeDataset("configs/data/ganymede.yaml", mode="multi_well", horizon=30),
    ))

    print(f"\n{'='*60}")
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"  All {total} datasets OK")
        return 0
    else:
        print(f"  {total - passed}/{total} datasets FAILED")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
