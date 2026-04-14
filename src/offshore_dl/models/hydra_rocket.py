"""Hydra+MultiROCKET: Competing Convolutional Kernels for Fast and Accurate TSC.

Dempster et al., Data Mining and Knowledge Discovery, 2023.
Uses aeon-toolkit's MultiRocketHydraClassifier wrapped in BaseModel interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel


class HydraRocketModel(BaseModel):
    """BaseModel-compatible wrapper around aeon's MultiRocketHydraClassifier."""

    def __init__(
        self,
        task: str = "classification",
        n_vars: int = 1,
        n_classes: int = 10,
        n_kernels: int = 8192,
        random_state: int = 42,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        if task != "classification":
            msg = "HydraRocketModel only supports classification tasks."
            raise ValueError(msg)

        super().__init__(
            task=task,
            n_vars=n_vars,
            loss_type="ce",
            class_weights=class_weights,
        )

        try:
            from aeon.classification.convolution_based import MultiRocketHydraClassifier
        except ImportError as exc:
            msg = (
                "aeon is required for HydraROCKET. Install it with `pip install aeon` "
                "or add it to your offshore-dl environment extras."
            )
            raise ImportError(msg) from exc

        self.n_classes = n_classes
        self.n_kernels = n_kernels
        self.random_state = random_state
        self.classifier = MultiRocketHydraClassifier(
            n_kernels=n_kernels,
            random_state=random_state,
        )
        self._is_fitted = False
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _to_numpy(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def fit(
        self,
        X_train: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
    ) -> "HydraRocketModel":
        X_np = self._to_numpy(X_train)
        y_np = self._to_numpy(y_train).astype(np.int64)
        self.classifier.fit(X_np, y_np)
        self._is_fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_fitted:
            msg = "HydraRocketModel must be fitted before calling forward()."
            raise RuntimeError(msg)
        probs = self.classifier.predict_proba(self._to_numpy(x))
        return torch.as_tensor(probs, dtype=torch.float32)

    def training_step(self, batch: tuple) -> torch.Tensor:
        raise NotImplementedError(
            "HydraROCKET uses sklearn fit(), not training_step(). Use run_production_3w_features.py --models hydra_rocket"
        )

    def predict(self, batch: tuple) -> torch.Tensor:
        features, _targets, _metadata = batch
        if not self._is_fitted:
            msg = "HydraRocketModel must be fitted before calling predict()."
            raise RuntimeError(msg)
        preds = self.classifier.predict(self._to_numpy(features))
        return torch.as_tensor(preds, dtype=torch.int64)

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            msg = "HydraRocketModel must be fitted before calling predict_proba()."
            raise RuntimeError(msg)
        return np.asarray(
            self.classifier.predict_proba(self._to_numpy(X)), dtype=np.float32
        )

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        raise NotImplementedError(
            "HydraROCKET uses sklearn fit() and has no optimizer."
        )
