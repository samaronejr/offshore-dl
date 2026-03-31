"""TiRex zero-shot classification via embedding extraction.

Uses TiRex as a frozen feature extractor following Auer et al. (2025):
"Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors
for Time Series Classification."

Pipeline:
1. Feed each sensor channel through TiRex → hidden states (12 layers × 512 dim)
2. Mean-pool over sequence (tokens) dimension
3. Concatenate all layers → per-channel embedding (6144D)
4. Mean-pool across sensor channels → final embedding (6144D)
5. Train a Random Forest on the embeddings

This is NOT a neural network classifier — it uses sklearn RF on frozen
TiRex embeddings.  The ``run()`` method handles the full pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_TIREX_AVAILABLE = False
try:
    from tirex import load_model as _tirex_load_model

    _TIREX_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if TiRex dependencies are installed."""
    return _TIREX_AVAILABLE


class TiRexClassifier:
    """TiRex embedding + Random Forest classifier for 3W.

    Not a BaseModel subclass — this is a standalone pipeline that
    extracts embeddings from a frozen TiRex model and trains a
    scikit-learn Random Forest on top.

    Args:
        n_vars: Number of input sensors (channels).
        n_classes: Number of output classes.
        max_context: Max context length for TiRex.
        n_estimators: Number of RF trees.
        batch_size: Batch size for embedding extraction.
        device: torch device.
    """

    def __init__(
        self,
        n_vars: int = 27,
        n_classes: int = 10,
        max_context: int = 512,
        n_estimators: int = 500,
        batch_size: int = 64,
        device: str = "cuda",
    ) -> None:
        if not _TIREX_AVAILABLE:
            msg = (
                "TiRex is not installed. Requires GPU with CUDA ≥8.0. "
                "Install via: pip install git+https://github.com/NX-AI/tirex"
            )
            raise ImportError(msg)

        self.n_vars = n_vars
        self.n_classes = n_classes
        self.max_context = max_context
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.device = device

        self._tirex = None
        self._rf = None
        self._emb_dim = None

    def _load_tirex(self) -> None:
        """Lazy-load TiRex model."""
        if self._tirex is None:
            self._tirex = _tirex_load_model("NX-AI/TiRex")
            self._tirex.eval()
            self._tirex = self._tirex.to(self.device)
            logger.info("TiRex model loaded on %s", self.device)

    @torch.no_grad()
    def _extract_embeddings_batch(
        self, windows: torch.Tensor,
    ) -> np.ndarray:
        """Extract TiRex embeddings for a batch of multivariate windows.

        Args:
            windows: ``(batch, window_size, n_vars)`` tensor.

        Returns:
            Embeddings array ``(batch, emb_dim)`` — mean-pooled across
            channels and layers.
        """
        self._load_tirex()
        batch_size, window_size, n_vars = windows.shape

        # Reshape all channels into one big batch for a single _embed_context call
        # (batch, window, n_vars) → (batch * n_vars, window)
        all_channels = windows.permute(0, 2, 1).reshape(-1, window_size).to(self.device)

        # Truncate to max context
        if all_channels.shape[1] > self.max_context:
            all_channels = all_channels[:, -self.max_context:]

        # Process all channels at once in sub-batches to avoid OOM
        # TiRex handles batches well — use larger sub-batches
        ch_batch_size = min(256, all_channels.shape[0])
        all_hidden = []
        for start in range(0, all_channels.shape[0], ch_batch_size):
            end = min(start + ch_batch_size, all_channels.shape[0])
            hidden = self._tirex._embed_context(all_channels[start:end])
            # hidden: (sub_batch, tokens, layers, hidden_dim)
            # Mean pool over tokens → (sub_batch, layers, hidden_dim)
            pooled = hidden.mean(dim=1)
            # L2 normalize per layer
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            # Concat layers → (sub_batch, layers*hidden_dim)
            flat = pooled.reshape(end - start, -1)
            all_hidden.append(flat)

        # (batch * n_vars, emb_dim_per_channel)
        all_flat = torch.cat(all_hidden, dim=0)

        # Reshape back: (batch, n_vars, emb_dim_per_channel)
        emb_per_ch = all_flat.reshape(batch_size, n_vars, -1)

        # Mean pool across channels → (batch, emb_dim)
        embedding = emb_per_ch.mean(dim=1)

        return embedding.cpu().numpy()

    def extract_all_embeddings(
        self,
        dataset,
        indices: list[int] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings for all samples in indices.

        Args:
            dataset: ThreeWDataset or ThreeWFeatureDataset.
            indices: Sample indices to extract.

        Returns:
            Tuple of (embeddings, labels) arrays.
        """
        n = len(indices)
        all_embeddings = []
        all_labels = []

        logger.info(
            "Extracting TiRex embeddings for %d samples (batch_size=%d)…",
            n, self.batch_size,
        )
        start = time.time()

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_indices = indices[batch_start:batch_end]

            windows = []
            labels = []
            for idx in batch_indices:
                feat, label, _ = dataset[idx]
                windows.append(feat)
                labels.append(label)

            batch_tensor = torch.stack(windows)  # (B, W, n_vars)
            emb = self._extract_embeddings_batch(batch_tensor)
            all_embeddings.append(emb)
            all_labels.extend(labels)

            if (batch_start + self.batch_size) % (self.batch_size * 50) == 0:
                elapsed = time.time() - start
                pct = 100 * batch_end / n
                logger.info(
                    "  … %d / %d (%.0f%%) — %.1fs", batch_end, n, pct, elapsed,
                )

        elapsed = time.time() - start
        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        labels = np.array(all_labels)
        self._emb_dim = embeddings.shape[1]

        logger.info(
            "Embeddings extracted: (%d, %d) in %.1fs",
            embeddings.shape[0], embeddings.shape[1], elapsed,
        )
        return embeddings, labels

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
        """Full pipeline: extract embeddings → train RF → evaluate.

        Returns:
            Results dict with aggregate metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            average_precision_score,
        )
        from sklearn.preprocessing import label_binarize

        # Extract embeddings
        X_train, y_train = self.extract_all_embeddings(dataset, train_indices)
        X_val, y_val = self.extract_all_embeddings(dataset, val_indices)

        # Train RF
        self.fit(X_train, y_train)

        # Predict
        y_pred = self.predict(X_val)
        y_proba = self.predict_proba(X_val)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        # AUC-PR (macro average)
        classes = sorted(set(y_train) | set(y_val))
        y_val_bin = label_binarize(y_val, classes=classes)
        # Align proba columns to classes
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

        # EDR (event detection rate) — fraction of classes with F1 > 0
        per_class_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
        edr = float(np.mean(per_class_f1 > 0))

        # Confusion matrix
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_val, y_pred, labels=classes)

        logger.info(
            "TiRex classification: acc=%.4f, f1_macro=%.4f, "
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
