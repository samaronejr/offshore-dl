"""Run TiRex 5-fold RF classification from pre-extracted embeddings.

Reads embeddings from results/tirex/3w_embeddings.dat (memmap),
trains RF per fold on host (no Docker/GPU needed).

Usage:
    PYTHONPATH=src python scripts/run_tirex_rf_folds.py
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

EMB_DIM = 6144  # TiRex: 12 layers × 512 hidden


def main():
    from offshore_dl.evaluation.cv import StratifiedGroupKFoldSKLearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, average_precision_score,
        confusion_matrix as sk_cm,
    )
    from sklearn.preprocessing import label_binarize

    out_dir = Path("results/tirex")

    # Load from memmap — stays on disk, doesn't consume RAM until accessed
    emb_path = out_dir / "3w_embeddings.dat"
    labels = np.load(out_dir / "3w_labels.npy")
    groups = np.load(out_dir / "3w_groups.npy")
    n = len(labels)
    logger.info("Loaded labels (%d) and groups (%d unique)", n, len(np.unique(groups)))

    embeddings = np.memmap(emb_path, dtype="float32", mode="r", shape=(n, EMB_DIM))
    logger.info("Embeddings memmap: %s", embeddings.shape)

    cv = StratifiedGroupKFoldSKLearn(n_folds=5, labels=labels, groups=groups, seed=42)
    splits = cv.get_splits(n)
    classes = np.unique(labels)

    fold_results = []
    t0 = time.time()

    for fi, (tr, va) in enumerate(splits):
        logger.info("RF fold %d/5 (train=%d, val=%d)", fi, len(tr), len(va))

        # Copy only what RF needs into contiguous arrays
        X_train = np.array(embeddings[tr])
        y_train = labels[tr]
        X_val = np.array(embeddings[va])
        y_val = labels[va]

        rf = RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=42,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)
        yp = rf.predict(X_val)
        yprob = rf.predict_proba(X_val)

        acc = float(accuracy_score(y_val, yp))
        f1m = float(f1_score(y_val, yp, average="macro", zero_division=0))
        f1w = float(f1_score(y_val, yp, average="weighted", zero_division=0))

        ybin = label_binarize(y_val, classes=classes)
        auc_scores = [
            float(average_precision_score(ybin[:, c], yprob[:, c]))
            for c in range(len(classes)) if ybin[:, c].sum() > 0
        ]
        auc_pr = float(np.mean(auc_scores)) if auc_scores else 0.0
        pf1 = f1_score(y_val, yp, average=None, zero_division=0)
        edr = float(np.mean(pf1 > 0))
        cm = sk_cm(y_val, yp, labels=classes)

        fold_results.append({
            "fold_idx": fi,
            "metrics": {
                "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
                "auc_pr": auc_pr, "edr": edr,
                "confusion_matrix": cm.tolist(),
                "class_labels": [int(c) for c in classes],
            },
        })
        logger.info("  acc=%.4f f1m=%.4f auc=%.4f", acc, f1m, auc_pr)

        # Free memory
        del rf, X_train, X_val, yp, yprob
        gc.collect()

    elapsed = time.time() - t0

    agg = {}
    for k in ["accuracy", "f1_macro", "f1_weighted", "auc_pr", "edr"]:
        vs = [fr["metrics"][k] for fr in fold_results]
        agg[f"{k}_mean"] = float(np.mean(vs))
        agg[f"{k}_std"] = float(np.std(vs))

    result = {
        "fold_results": fold_results,
        "aggregate": agg,
        "n_folds": 5,
        "embedding_dim": EMB_DIM,
        "n_estimators": 500,
    }

    # Handle Docker root-owned directories
    out_path = Path("results") / "tirex_3w_5fold.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("DONE → %s (%.1fs)", out_path, elapsed)
    logger.info("Aggregate: %s", {k: f"{v:.4f}" for k, v in agg.items()})


if __name__ == "__main__":
    main()
