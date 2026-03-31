"""TiRex RF with nested CV: holdout test + inner 5-fold CV.

Uses pre-extracted embeddings from results/tirex/3w_embeddings.dat
(memmap, 208K × 6144 float32). Runs entirely on CPU — no GPU needed.

Protocol:
  1. Load cached embeddings + labels + groups
  2. Stratified group holdout: 80% train pool, 20% test
  3. Inner 5-fold SGKF within train pool (variance estimates)
  4. Retrain RF on full train pool
  5. Evaluate on held-out test set
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import label_binarize

from offshore_dl.evaluation.cv import HoldoutSplitter, StratifiedGroupKFoldSKLearn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
EMBED_PATH = RESULTS_DIR / "tirex" / "3w_embeddings.dat"
LABELS_PATH = RESULTS_DIR / "tirex" / "3w_labels.npy"
GROUPS_PATH = RESULTS_DIR / "tirex" / "3w_groups.npy"

EMB_DIM = 6144
N_ESTIMATORS = 500


def _compute_metrics(y_true, y_pred, y_proba, classes):
    """Compute classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    y_bin = label_binarize(y_true, classes=classes)
    auc_scores = []
    for c_idx in range(len(classes)):
        if y_bin[:, c_idx].sum() > 0:
            auc_scores.append(
                float(average_precision_score(y_bin[:, c_idx], y_proba[:, c_idx]))
            )
    auc_pr = float(np.mean(auc_scores)) if auc_scores else 0.0

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    edr = float(np.mean(per_class_f1 > 0))

    cm = sk_confusion_matrix(y_true, y_pred, labels=classes)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_pr": auc_pr,
        "edr": edr,
        "confusion_matrix": cm.tolist(),
        "class_labels": [int(c) for c in classes],
    }


def main():
    logger.info("Loading cached TiRex embeddings …")
    n_samples = len(np.load(LABELS_PATH))
    embeddings = np.memmap(EMBED_PATH, dtype=np.float32, mode="r", shape=(n_samples, EMB_DIM))
    labels = np.load(LABELS_PATH)
    groups = np.load(GROUPS_PATH)
    classes = np.unique(labels)

    logger.info("  Embeddings: %s, Labels: %s, Groups: %d unique",
                embeddings.shape, labels.shape, len(np.unique(groups)))

    # ── Outer holdout: 80/20 stratified group split ──
    holdout = HoldoutSplitter(
        test_ratio=0.2,
        mode="stratified_group",
        labels=labels,
        groups=groups,
        seed=42,
    )
    train_pool, test_indices = holdout.split(n_samples)

    # ── Inner 5-fold CV within train pool ──
    pool_labels = labels[train_pool]
    pool_groups = groups[train_pool]
    inner_cv = StratifiedGroupKFoldSKLearn(
        n_folds=5, labels=pool_labels, groups=pool_groups, seed=42,
    )
    inner_splits = inner_cv.get_splits(len(train_pool))

    logger.info("═══ Nested CV: %d inner folds on %d train, %d test ═══",
                len(inner_splits), len(train_pool), len(test_indices))

    cv_fold_results = []
    for fold_idx, (local_train, local_val) in enumerate(inner_splits):
        global_train = train_pool[local_train]
        global_val = train_pool[local_val]

        logger.info("── Inner fold %d/%d (train=%d, val=%d) ──",
                     fold_idx + 1, len(inner_splits),
                     len(global_train), len(global_val))

        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, n_jobs=-1,
            random_state=42, class_weight="balanced",
        )
        rf.fit(embeddings[global_train], labels[global_train])
        y_pred = rf.predict(embeddings[global_val])
        y_proba = rf.predict_proba(embeddings[global_val])

        metrics = _compute_metrics(labels[global_val], y_pred, y_proba, classes)
        cv_fold_results.append({"fold_idx": fold_idx, "metrics": metrics})
        logger.info("  acc=%.4f  f1m=%.4f  auc=%.4f  (%.1fs)",
                     metrics["accuracy"], metrics["f1_macro"],
                     metrics["auc_pr"], time.time() - t0)

    # CV aggregate
    cv_agg = {}
    for key in ["accuracy", "f1_macro", "f1_weighted", "auc_pr", "edr"]:
        vals = [fr["metrics"][key] for fr in cv_fold_results]
        cv_agg[f"{key}_mean"] = float(np.mean(vals))
        cv_agg[f"{key}_std"] = float(np.std(vals))

    logger.info("Inner CV aggregate: %s",
                {k: f"{v:.4f}" for k, v in cv_agg.items() if k.endswith("_mean")})

    # ── Retrain RF on full train pool ──
    logger.info("═══ Retraining RF on full train pool (%d samples) ═══", len(train_pool))
    t0 = time.time()
    rf_final = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, n_jobs=-1,
        random_state=42, class_weight="balanced",
    )
    rf_final.fit(embeddings[train_pool], labels[train_pool])
    logger.info("  RF trained in %.1fs", time.time() - t0)

    # ── Evaluate on held-out test set ──
    logger.info("═══ Evaluating on held-out test (%d samples) ═══", len(test_indices))
    y_pred_test = rf_final.predict(embeddings[test_indices])
    y_proba_test = rf_final.predict_proba(embeddings[test_indices])

    test_metrics = _compute_metrics(labels[test_indices], y_pred_test, y_proba_test, classes)
    logger.info("TEST: acc=%.4f  f1m=%.4f  auc=%.4f",
                test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["auc_pr"])

    # ── Save results ──
    results = {
        "test_metrics": test_metrics,
        "cv_aggregate": cv_agg,
        "cv_fold_results": cv_fold_results,
        "n_train": len(train_pool),
        "n_test": len(test_indices),
        "n_cv_folds": len(inner_splits),
        "embedding_dim": EMB_DIM,
        "n_estimators": N_ESTIMATORS,
    }

    out_path = RESULTS_DIR / "tirex_3w_nested.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _ser(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_ser(v) for v in obj]
        return obj

    out_path.write_text(json.dumps(_ser(results), indent=2))
    logger.info("Results saved: %s", out_path)


if __name__ == "__main__":
    main()
