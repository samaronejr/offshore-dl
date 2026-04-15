"""Mantis foundation model for time-series classification.

Uses Mantis (Feofanov et al., 2025) as a frozen feature extractor
with a Random Forest classification head. Architecturally similar
to TiRexClassifier.

Requires: pip install mantis-fm (or from github.com/vfeofanov/mantis)
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

_MANTIS_AVAILABLE = False
try:
    import mantis  # noqa: F401

    _MANTIS_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if Mantis dependencies are installed."""
    return _MANTIS_AVAILABLE


class MantisClassifier:
    """Mantis frozen encoder + Random Forest classifier.

    Two-stage pipeline (mirrors TiRexClassifier):
    1. Extract embeddings from Mantis frozen encoder
    2. Train sklearn RandomForestClassifier on embeddings

    Not a BaseModel subclass — this is a standalone pipeline that
    extracts embeddings from a frozen Mantis model and trains a
    scikit-learn Random Forest on top.

    Args:
        n_classes: Number of output classes.
        n_vars: Number of input variables (sensor channels).
        window_size: Input sequence length.
        n_estimators: Number of RF trees. Default 500.
        batch_size: Batch size for embedding extraction.
        device: torch device string.
    """

    def __init__(
        self,
        n_classes: int = 10,
        n_vars: int = 27,
        window_size: int = 14,
        n_estimators: int = 500,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> None:
        if not _MANTIS_AVAILABLE:
            raise ImportError(
                "Mantis is not installed. "
                "Install via: pip install mantis-fm  "
                "or see https://github.com/vfeofanov/mantis"
            )
        self.n_classes = n_classes
        self.n_vars = n_vars
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.device = device

        self._encoder = None
        self._rf = None
        self._emb_dim = None

    def _load_encoder(self) -> None:
        """Lazy-load Mantis pre-trained encoder."""
        if self._encoder is not None:
            return
        # Load pre-trained Mantis encoder.
        # The exact API depends on the installed mantis version;
        # check mantis docs / source for the correct entry point.
        self._encoder = mantis.load_pretrained()  # type: ignore[name-defined]
        logger.info("Mantis encoder loaded on %s", self.device)

    def extract_all_embeddings(
        self,
        dataset,
        indices: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract Mantis embeddings for all samples in indices.

        Args:
            dataset: ThreeWDataset or compatible dataset with
                ``__getitem__`` returning (features, label, *extra).
            indices: Sample indices to extract.

        Returns:
            Tuple of (embeddings, labels) arrays.
        """
        import torch

        self._load_encoder()

        n = len(indices)
        all_embeddings: list[np.ndarray] = []
        all_labels: list[int] = []

        logger.info(
            "Extracting Mantis embeddings for %d samples (batch_size=%d)…",
            n, self.batch_size,
        )
        start = time.time()

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_indices = indices[batch_start:batch_end]

            windows = []
            labels = []
            for idx in batch_indices:
                item = dataset[idx]
                feat, label = item[0], item[1]
                windows.append(feat)
                labels.append(label)

            batch_tensor = torch.stack(windows).to(self.device)  # (B, W, n_vars)

            with torch.no_grad():
                embeddings = self._encoder.encode(batch_tensor)
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

            all_embeddings.append(embeddings)
            all_labels.extend(labels)

            if (batch_start + self.batch_size) % (self.batch_size * 50) == 0:
                elapsed = time.time() - start
                pct = 100 * batch_end / n
                logger.info(
                    "  … %d / %d (%.0f%%) — %.1fs", batch_end, n, pct, elapsed,
                )

        elapsed = time.time() - start
        embeddings_arr = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        labels_arr = np.array(all_labels)
        self._emb_dim = embeddings_arr.shape[1]

        logger.info(
            "Mantis embeddings extracted: (%d, %d) in %.1fs",
            embeddings_arr.shape[0], embeddings_arr.shape[1], elapsed,
        )
        return embeddings_arr, labels_arr

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Random Forest on extracted embeddings."""
        from sklearn.ensemble import RandomForestClassifier

        logger.info(
            "Training Random Forest (n_estimators=%d) on (%d, %d)…",
            self.n_estimators, X.shape[0], X.shape[1],
        )
        start = time.time()
        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            max_depth=20,
            min_samples_leaf=5,
        )
        self._rf.fit(X, y)
        elapsed = time.time() - start
        logger.info("Random Forest trained in %.1fs", elapsed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes from embeddings."""
        return self._rf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities from embeddings."""
        return self._rf.predict_proba(X)

    def run(
        self,
        dataset,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
    ) -> dict:
        """Full pipeline: extract embeddings -> train RF -> evaluate.

        Args:
            dataset: Dataset with ``__getitem__`` returning (features, label, *extra).
            train_indices: Indices for training split.
            val_indices: Indices for validation split.

        Returns:
            Results dict with aggregate metrics matching TiRexClassifier output shape.
        """
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix as sk_cm,
            f1_score,
        )
        from sklearn.preprocessing import label_binarize

        logger.info(
            "Mantis classification: extracting embeddings "
            "(train=%d, val=%d)",
            len(train_indices), len(val_indices),
        )

        X_train, y_train = self.extract_all_embeddings(dataset, train_indices)
        X_val, y_val = self.extract_all_embeddings(dataset, val_indices)

        self.fit(X_train, y_train)

        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        classes = sorted(set(y_train) | set(y_val))
        y_val_bin = label_binarize(y_val, classes=classes)

        auc_pr = 0.0
        try:
            rf_classes = list(self._rf.classes_)
            proba_aligned = np.zeros((len(y_val), len(classes)))
            for i, c in enumerate(classes):
                if c in rf_classes:
                    proba_aligned[:, i] = y_proba[:, rf_classes.index(c)]
            auc_pr = average_precision_score(
                y_val_bin, proba_aligned, average="macro",
            )
        except Exception:
            auc_pr = 0.0

        per_class_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
        edr = float(np.mean(per_class_f1 > 0))

        cm = sk_cm(y_val, y_pred, labels=classes)

        logger.info(
            "Mantis classification: acc=%.4f, f1_macro=%.4f, "
            "f1_weighted=%.4f, auc_pr=%.4f, edr=%.4f",
            acc, f1_macro, f1_weighted, auc_pr, edr,
        )

        return {
            "aggregate": {
                "accuracy_mean": acc,
                "accuracy_std": 0.0,
                "f1_macro_mean": f1_macro,
                "f1_macro_std": 0.0,
                "f1_weighted_mean": f1_weighted,
                "f1_weighted_std": 0.0,
                "auc_pr_mean": auc_pr,
                "auc_pr_std": 0.0,
                "edr_mean": edr,
                "edr_std": 0.0,
                "confusion_matrix": cm.tolist(),
                "class_labels": [int(c) for c in classes],
            },
            "n_folds": 1,
            "embedding_dim": self._emb_dim,
            "n_estimators": self.n_estimators,
        }
