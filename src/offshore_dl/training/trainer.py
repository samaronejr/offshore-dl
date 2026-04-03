"""Training engine: epoch-based train/val loop with early stopping and checkpointing.

Every model trains through ``Trainer.fit()`` — models never touch the
training loop directly.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from offshore_dl.models.base import model_summary

logger = logging.getLogger(__name__)


class CostTracker:
    """Track computational costs during training.

    Usage as a context manager::

        with CostTracker(model) as tracker:
            # ... training ...
        costs = tracker.results
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._start_time: float = 0.0
        self._results: dict | None = None

    def __enter__(self) -> "CostTracker":
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc) -> None:
        wall_time = time.time() - self._start_time
        summary = model_summary(self.model)

        gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except RuntimeError:
                pass

        self._results = {
            "wall_time_seconds": round(wall_time, 2),
            "gpu_memory_peak_mb": round(gpu_mem_mb, 2),
            "param_count": summary["param_count"],
            "trainable_params": summary["trainable_params"],
            "model_size_mb": summary["model_size_mb"],
        }

    @property
    def results(self) -> dict:
        """Return cost tracking results after context exit."""
        if self._results is None:
            return {"wall_time_seconds": 0, "gpu_memory_peak_mb": 0, "param_count": 0}
        return self._results


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_epoch: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current epoch's validation loss.
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1

        self.should_stop = self.counter >= self.patience
        return self.should_stop


class Trainer:
    """Epoch-based training engine.

    Handles the train/val loop, gradient clipping, early stopping,
    checkpointing, and cost tracking. Models plug in via the
    ``BaseModel`` interface.

    Args:
        cfg: OmegaConf configuration.
        device: Override device (default: from config or CPU).
    """

    def __init__(self, cfg: DictConfig | None = None, device: str | None = None) -> None:
        self.cfg = cfg
        if device:
            self.device = torch.device(device)
        elif cfg and hasattr(cfg, "device"):
            dev_str = cfg.device
            if dev_str == "cuda" and not torch.cuda.is_available():
                dev_str = "cpu"
            self.device = torch.device(dev_str)
        else:
            self.device = torch.device("cpu")

    def fit(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        max_epochs: int | None = None,
        patience: int | None = None,
        gradient_clip_val: float | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> dict:
        """Run the complete training loop.

        Args:
            model: BaseModel instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            max_epochs: Override from config.
            patience: Override early stopping patience.
            gradient_clip_val: Override gradient clipping.
            checkpoint_dir: Directory for saving checkpoints.

        Returns:
            Training history dict with per-epoch losses and metrics,
            plus cost tracker results.
        """
        cfg_t = self.cfg.training if self.cfg and hasattr(self.cfg, "training") else None
        max_epochs = max_epochs or (cfg_t.max_epochs if cfg_t else 10)
        patience = patience or (cfg_t.early_stopping_patience if cfg_t else 10)
        gradient_clip_val = gradient_clip_val or (cfg_t.gradient_clip_val if cfg_t else 1.0)

        model = model.to(self.device)
        optimizer = model.configure_optimizers(self.cfg)
        early_stopping = EarlyStopping(patience=patience)
        best_state = None

        # ── LR scheduler ──
        scheduler_name = None
        if cfg_t and hasattr(cfg_t, "scheduler"):
            scheduler_name = cfg_t.scheduler

        scheduler = self._build_scheduler(
            optimizer, scheduler_name, max_epochs, len(train_loader), cfg_t,
        )
        scheduler_per_batch = scheduler_name == "onecycle"

        history = {
            "train_loss": [],
            "val_loss": [],
            "epochs_run": 0,
            "best_epoch": 0,
            "stopped_early": False,
        }

        with CostTracker(model) as cost_tracker:
            for epoch in range(max_epochs):
                # ── Train ──
                model.train()
                train_losses = []
                for batch in train_loader:
                    batch = self._to_device(batch)
                    optimizer.zero_grad()
                    loss = model.training_step(batch)

                    # Skip NaN batches — Mamba layers can produce rare
                    # NaN losses from specific input/weight combinations.
                    # Skipping prevents corrupting the entire model.
                    if torch.isnan(loss) or torch.isinf(loss):
                        optimizer.zero_grad()
                        continue

                    loss.backward()

                    if gradient_clip_val > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                    optimizer.step()

                    # Step per-batch schedulers (OneCycleLR)
                    if scheduler and scheduler_per_batch:
                        scheduler.step()

                    train_losses.append(loss.item())

                avg_train_loss = float(np.mean(train_losses)) if train_losses else float('nan')

                # ── Validate ──
                model.eval()
                val_losses = []
                for batch in val_loader:
                    batch = self._to_device(batch)
                    result = model.validation_step(batch)
                    val_loss_item = result["loss"].item()
                    if not (np.isnan(val_loss_item) or np.isinf(val_loss_item)):
                        val_losses.append(val_loss_item)

                avg_val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

                # Step per-epoch schedulers
                if scheduler and not scheduler_per_batch:
                    if scheduler_name == "reduce_on_plateau":
                        if not np.isnan(avg_val_loss):
                            scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)

                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f, lr=%.2e",
                    epoch + 1, max_epochs, avg_train_loss, avg_val_loss, current_lr,
                )

                # ── Checkpoint best (skip if val loss is NaN) ──
                if not np.isnan(avg_val_loss) and avg_val_loss <= early_stopping.best_loss:
                    best_state = copy.deepcopy(model.state_dict())
                    if checkpoint_dir:
                        self.save_checkpoint(model, optimizer, epoch, checkpoint_dir)

                # ── Early stopping ──
                if early_stopping.step(avg_val_loss, epoch):
                    logger.info("Early stopping at epoch %d (best: %d)", epoch + 1, early_stopping.best_epoch + 1)
                    history["stopped_early"] = True
                    break

            # Restore best model weights
            if best_state is not None:
                model.load_state_dict(best_state)
                logger.info("Restored best model from epoch %d", early_stopping.best_epoch + 1)

            history["epochs_run"] = epoch + 1
            history["best_epoch"] = early_stopping.best_epoch

        history["cost"] = cost_tracker.results
        return history

    def _to_device(self, batch: tuple) -> tuple:
        """Move batch tensors to the training device."""
        features, targets, metadata = batch

        features = features.to(self.device, dtype=torch.float32)

        if isinstance(targets, torch.Tensor):
            if targets.is_floating_point():
                targets = targets.to(self.device, dtype=torch.float32)
            else:
                targets = targets.to(self.device)
        elif isinstance(targets, (list, np.ndarray)):
            targets = torch.tensor(targets, device=self.device)

        return features, targets, metadata

    @staticmethod
    def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        name: str | None,
        max_epochs: int,
        steps_per_epoch: int,
        cfg_t: Any = None,
    ) -> Any:
        """Build an LR scheduler from config.

        Supported schedulers:
        - ``"onecycle"``: OneCycleLR with per-batch stepping.
        - ``"cosine"``: CosineAnnealingLR with per-epoch stepping.
        - ``"reduce_on_plateau"``: ReduceLROnPlateau (per-epoch, on val_loss).

        Args:
            optimizer: The optimizer to schedule.
            name: Scheduler name string from config.
            max_epochs: Total training epochs.
            steps_per_epoch: Number of batches per epoch.
            cfg_t: Training config subtree for scheduler params.

        Returns:
            Scheduler instance, or None if name is None/unknown.
        """
        if not name:
            return None

        name = name.lower().replace("_", "").replace("-", "")

        if name == "onecycle":
            max_lr = optimizer.defaults["lr"]
            pct_start = 0.3
            if cfg_t and hasattr(cfg_t, "scheduler_pct_start"):
                pct_start = cfg_t.scheduler_pct_start
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start,
            )

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs,
            )

        if name in ("reduceonplateau", "reduce_on_plateau"):
            patience = 5
            factor = 0.5
            if cfg_t and hasattr(cfg_t, "scheduler_patience"):
                patience = cfg_t.scheduler_patience
            if cfg_t and hasattr(cfg_t, "scheduler_factor"):
                factor = cfg_t.scheduler_factor
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=patience, factor=factor,
            )

        logger.warning("Unknown scheduler %r — training without LR schedule", name)
        return None

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        path: str | Path,
    ) -> Path:
        """Save model + optimizer state.

        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            path: Directory or file path.

        Returns:
            Path to saved checkpoint.
        """
        path = Path(path)
        if path.is_dir() or not path.suffix:
            path.mkdir(parents=True, exist_ok=True)
            path = path / "checkpoint.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }, path)

        logger.info("Checkpoint saved: %s", path)
        return path

    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        path: str | Path,
    ) -> int:
        """Load model + optimizer state.

        Args:
            model: Model to load into.
            optimizer: Optimizer to load into (optional).
            path: Path to checkpoint file.

        Returns:
            Epoch number from checkpoint.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "checkpoint.pt"

        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info("Checkpoint loaded: %s (epoch %d)", path, checkpoint["epoch"])
        return checkpoint["epoch"]
