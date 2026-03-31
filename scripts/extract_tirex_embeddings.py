"""Extract TiRex embeddings and save to disk in chunks.

Avoids OOM by writing embeddings to a memory-mapped file incrementally
instead of accumulating a 5 GB array in RAM.

Usage:
    scripts/docker_run.sh python scripts/extract_tirex_embeddings.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from offshore_dl.data.datasets import ThreeWDataset
    from offshore_dl.models.tirex_classifier import TiRexClassifier, is_available

    if not is_available():
        logger.error("TiRex not available")
        return

    raw_dataset = ThreeWDataset("configs/data/3w.yaml")
    n = len(raw_dataset)
    logger.info("Raw 3W: %d samples", n)

    clf = TiRexClassifier(n_vars=27, n_classes=10, device="cuda", batch_size=32)

    # First batch to determine embedding dimension
    first_feat, _, _ = raw_dataset[0]
    first_batch = first_feat.unsqueeze(0)
    first_emb = clf._extract_embeddings_batch(first_batch)
    emb_dim = first_emb.shape[1]
    logger.info("Embedding dim: %d", emb_dim)

    # Create output directory
    out_dir = Path("results/tirex")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create memory-mapped file for embeddings
    emb_path = out_dir / "3w_embeddings.dat"
    fp = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n, emb_dim))
    labels = np.zeros(n, dtype=np.int64)

    batch_size = clf.batch_size
    start = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)

        windows = []
        for idx in range(batch_start, batch_end):
            feat, label, _ = raw_dataset[idx]
            windows.append(feat)
            labels[idx] = label

        batch_tensor = torch.stack(windows)
        emb = clf._extract_embeddings_batch(batch_tensor)

        # Write directly to memmap — no accumulation in RAM
        fp[batch_start:batch_end] = emb.astype(np.float32)

        if (batch_start // batch_size) % 50 == 0:
            elapsed = time.time() - start
            pct = 100 * batch_end / n
            logger.info("  … %d / %d (%.0f%%) — %.1fs", batch_end, n, pct, elapsed)

    fp.flush()
    elapsed = time.time() - start
    logger.info("Embeddings written to %s: (%d, %d) in %.1fs", emb_path, n, emb_dim, elapsed)

    # Save labels and groups
    np.save(out_dir / "3w_labels.npy", labels)

    # Extract groups (instance_id)
    groups = np.array([raw_dataset[i][2]["instance_id"] for i in range(n)])
    np.save(out_dir / "3w_groups.npy", groups)
    logger.info("Saved labels and groups")


if __name__ == "__main__":
    main()
