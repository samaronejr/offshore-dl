"""Abstract base model for all offshore DL architectures.

Every model (LSTM, DeepONet, PatchTST, etc.) inherits ``BaseModel``
and implements the required abstract methods. The training engine
calls these methods — models never touch the training loop directly.

Return format contract:
    - ``training_step(batch)`` → scalar loss tensor
    - ``validation_step(batch)`` → dict with 'loss', 'predictions', 'targets'
    - ``predict(batch)`` → predictions tensor
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """Abstract base for all offshore DL models.

    Args:
        task: One of ``"classification"``, ``"forecasting"``, ``"anomaly"``.
        n_vars: Number of input variables (sensor columns).
        **kwargs: Passed to nn.Module.
    """

    def __init__(self, task: str, n_vars: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.n_vars = n_vars
        self._loss_fn = self._build_loss_fn()

    def _build_loss_fn(self, class_weights: torch.Tensor | None = None) -> nn.Module:
        """Create the appropriate loss function for this task.

        Args:
            class_weights: Optional class weights for classification.

        Returns:
            Loss module.
        """
        if self.task == "classification":
            return nn.CrossEntropyLoss(weight=class_weights)
        elif self.task in ("forecasting", "anomaly"):
            return nn.MSELoss()
        else:
            msg = f"Unknown task: {self.task!r}"
            raise ValueError(msg)

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """Update classification loss with class weights.

        Args:
            weights: Tensor of per-class weights.
        """
        if self.task == "classification":
            self._loss_fn = nn.CrossEntropyLoss(weight=weights.to(next(self.parameters()).device))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, window, n_vars)``.

        Returns:
            Model output. Shape depends on task:
                - classification: ``(batch, n_classes)`` logits
                - forecasting: ``(batch, horizon)`` predictions
                - anomaly: ``(batch, window, n_vars)`` reconstruction
        """

    def training_step(self, batch: tuple) -> torch.Tensor:
        """Compute training loss for one batch.

        Args:
            batch: Tuple of ``(features, targets, metadata)`` from DataLoader.

        Returns:
            Scalar loss tensor (differentiable).
        """
        features, targets, _metadata = batch
        outputs = self.forward(features)
        return self._compute_loss(outputs, targets)

    def validation_step(self, batch: tuple) -> dict:
        """Compute validation metrics for one batch.

        Args:
            batch: Tuple of ``(features, targets, metadata)`` from DataLoader.

        Returns:
            Dict with 'loss' (scalar), 'predictions' (tensor), 'targets' (tensor).
        """
        features, targets, _metadata = batch
        with torch.no_grad():
            outputs = self.forward(features)
            loss = self._compute_loss(outputs, targets)

        predictions = self._extract_predictions(outputs)
        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
        }

    def predict(self, batch: tuple) -> torch.Tensor:
        """Generate predictions for a batch (no loss computation).

        Args:
            batch: Tuple of ``(features, targets, metadata)`` from DataLoader.

        Returns:
            Predictions tensor.
        """
        features, _targets, _metadata = batch
        with torch.no_grad():
            outputs = self.forward(features)
        return self._extract_predictions(outputs)

    @abstractmethod
    def configure_optimizers(self, cfg) -> torch.optim.Optimizer:
        """Create and return the optimizer.

        Args:
            cfg: OmegaConf config (contains lr, weight_decay, etc.).

        Returns:
            Configured optimizer.
        """

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with task-aware handling."""
        if self.task == "classification":
            # targets are int class labels
            if targets.dtype in (torch.float32, torch.float64):
                targets = targets.long()
            return self._loss_fn(outputs, targets)
        elif self.task == "forecasting":
            return self._loss_fn(outputs, targets.float())
        elif self.task == "anomaly":
            # Reconstruction: output should match input features
            return self._loss_fn(outputs, targets.float())
        else:
            msg = f"Unknown task: {self.task!r}"
            raise ValueError(msg)

    def _extract_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Extract predictions from raw model output."""
        if self.task == "classification":
            return outputs.argmax(dim=-1)
        else:
            return outputs


def model_summary(model: nn.Module) -> dict:
    """Compute model summary statistics.

    Args:
        model: Any nn.Module.

    Returns:
        Dict with param_count, trainable_params, model_size_mb.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Size in MB (float32 = 4 bytes per param)
    size_mb = total * 4 / (1024 * 1024)

    return {
        "param_count": total,
        "trainable_params": trainable,
        "model_size_mb": round(size_mb, 4),
    }
