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
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Down-weights well-classified examples so the model focuses on hard ones.
    When ``gamma=0`` this reduces exactly to standard cross-entropy.

    Args:
        gamma: Focusing exponent.  Higher values increase focus on hard examples.
        weight: Optional per-class weight tensor (same semantics as
            :class:`torch.nn.CrossEntropyLoss` *weight*).
        reduction: ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Register as buffer so it moves with .to(device) automatically
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(batch, n_classes)`` raw logits.
            targets: ``(batch,)`` integer class labels.

        Returns:
            Scalar loss.
        """
        # p_t is the true-class probability and must be computed from
        # unweighted CE. Class weights scale the CE term only; folding them
        # into p_t would make the focal modulation depend on class weighting.
        unweighted_ce = F.cross_entropy(logits, targets, reduction="none")
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-unweighted_ce)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BaseModel(nn.Module, ABC):
    """Abstract base for all offshore DL models.

    Args:
        task: One of ``"classification"``, ``"forecasting"``, ``"anomaly"``.
        n_vars: Number of input variables (sensor columns).
        loss_type: Loss function for classification — ``"ce"`` (cross-entropy,
            default) or ``"focal"`` (focal loss).  Ignored for other tasks.
        focal_gamma: Focusing exponent for focal loss.  Only used when
            ``loss_type="focal"``.  Default ``2.0``.
        **kwargs: Passed to nn.Module.
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.n_vars = n_vars
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self._loss_fn = self._build_loss_fn(class_weights=class_weights)

        # Defaults for configure_optimizers; subclasses override in their __init__
        if not hasattr(self, "lr"):
            self.lr = 1e-3
        if not hasattr(self, "weight_decay"):
            self.weight_decay = 1e-4

    def _build_loss_fn(self, class_weights: torch.Tensor | None = None) -> nn.Module:
        """Create the appropriate loss function for this task.

        Args:
            class_weights: Optional class weights for classification.

        Returns:
            Loss module.
        """
        if self.task == "classification":
            if self.loss_type == "ce":
                return nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=self.label_smoothing,
                )
            elif self.loss_type == "focal":
                return FocalLoss(gamma=self.focal_gamma, weight=class_weights)
            else:
                msg = f"Unknown loss_type: {self.loss_type!r}. Must be 'ce' or 'focal'."
                raise ValueError(msg)
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
            param = next(self.parameters(), None)
            if param is not None:
                weights = weights.to(param.device)
            self._loss_fn = self._build_loss_fn(class_weights=weights)

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

    def predict_scores(self, batch: tuple) -> torch.Tensor:
        """Generate prediction scores for a batch.

        For classification this returns softmax probabilities. For other tasks
        it returns the raw model outputs unchanged.
        """
        features, _targets, _metadata = batch
        with torch.no_grad():
            outputs = self.forward(features)
        return self._extract_prediction_scores(outputs)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        """Create AdamW optimizer with configurable lr and weight decay.

        Args:
            cfg: OmegaConf config (contains lr, weight_decay, etc.).

        Returns:
            Configured AdamW optimizer.
        """
        lr = self.lr
        wd = self.weight_decay

        if cfg is not None:
            if hasattr(cfg, "model") and hasattr(cfg.model, "training"):
                lr = getattr(cfg.model.training, "lr", lr)
                wd = getattr(cfg.model.training, "weight_decay", wd)
            elif hasattr(cfg, "training"):
                lr = getattr(cfg.training, "lr", lr)
                wd = getattr(cfg.training, "weight_decay", wd)

        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

    def _compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
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

    def _extract_prediction_scores(self, outputs: torch.Tensor) -> torch.Tensor:
        """Extract score-like outputs for metric computation."""
        if self.task == "classification":
            return torch.softmax(outputs, dim=-1)
        return outputs


def instance_normalize(
    x: torch.Tensor, eps: float = 1e-5, clamp_val: float = 10.0
) -> torch.Tensor:
    """Per-sample z-score normalization along the sequence dimension.

    Prevents numerical instability from wide-ranging raw sensor values.
    Model-internal so it doesn't leak across CV folds.

    Args:
        x: ``(B, L, n_vars)`` input tensor.
        eps: Minimum std to avoid division by zero.
        clamp_val: Clamp normalized values to ``[-clamp_val, clamp_val]``.

    Returns:
        Normalized tensor of same shape.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=eps)
    x = (x - mean) / std
    return x.clamp(-clamp_val, clamp_val)


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
