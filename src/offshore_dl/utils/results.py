"""Helpers for benchmark result roots.

Post-repair benchmark runs must not overwrite historical artifacts.  Writers
therefore default to ``results/post_fix`` unless an explicit output directory or
``OFFSHORE_DL_RESULTS_DIR`` is provided.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("results")
DEFAULT_WRITE_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "post_fix"


def resolve_results_dir(
    output_dir: str | Path | None = None,
    *,
    for_write: bool = True,
) -> Path:
    """Resolve the benchmark result root.

    Priority: explicit argument, ``OFFSHORE_DL_RESULTS_DIR``, then repaired
    write default (``results/post_fix``).  Read callers may pass
    ``for_write=False`` to prefer an existing post-fix tree while remaining
    compatible with legacy flat ``results`` layouts.
    """
    if output_dir is not None:
        return Path(output_dir)

    env_dir = os.environ.get("OFFSHORE_DL_RESULTS_DIR")
    if env_dir:
        return Path(env_dir)

    if for_write:
        return DEFAULT_WRITE_RESULTS_DIR

    if _contains_result_jsons(DEFAULT_WRITE_RESULTS_DIR):
        return DEFAULT_WRITE_RESULTS_DIR

    pre_fix = DEFAULT_RESULTS_ROOT / "pre_fix"
    if _contains_result_jsons(pre_fix):
        return pre_fix

    return DEFAULT_RESULTS_ROOT


def _contains_result_jsons(path: Path) -> bool:
    """Return True when a result root contains JSON artifacts."""
    try:
        return path.exists() and any(path.rglob("*.json"))
    except OSError:
        return False
