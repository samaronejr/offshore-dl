"""MOMENT foundation model for time-series classification.

Uses MOMENT (Goswami et al., 2024) as a frozen feature extractor
with optional LoRA fine-tuning, plus a classification head.

Requires: pip install momentfm
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MOMENT_AVAILABLE = False
try:
    from momentfm import MOMENTPipeline

    _MOMENT_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if momentfm dependencies are installed."""
    return _MOMENT_AVAILABLE


class MomentClassifier:
    """MOMENT embeddings + classification head.

    Three-stage fine-tuning (Goswami et al.):
    1. Linear probe: freeze encoder, train classification head only
    2. LoRA fine-tuning: add LoRA adapters (rank=4), train with lower LR
    3. Full fine-tuning: unfreeze all (optional, usually not needed)

    Args:
        n_classes: Number of output classes.
        n_vars: Number of input variables (channels).
        window_size: Input sequence length.
        model_name: HuggingFace model ID. Default "AutonLab/MOMENT-1-large".
        mode: "linear_probe", "lora", or "full". Default "linear_probe".
        lora_rank: LoRA rank. Default 4.
    """

    def __init__(
        self,
        n_classes: int = 10,
        n_vars: int = 27,
        window_size: int = 14,
        model_name: str = "AutonLab/MOMENT-1-large",
        mode: str = "linear_probe",
        lora_rank: int = 4,
    ) -> None:
        if not _MOMENT_AVAILABLE:
            raise ImportError("MOMENT not installed. Run: pip install momentfm")
        self.n_classes = n_classes
        self.n_vars = n_vars
        self.window_size = window_size
        self.model_name = model_name
        self.mode = mode
        self.lora_rank = lora_rank
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load MOMENT model."""
        if self._model is not None:
            return

        self._model = MOMENTPipeline.from_pretrained(
            self.model_name,
            model_kwargs={
                "task_name": "classification",
                "n_channels": self.n_vars,
                "num_class": self.n_classes,
            },
        )
        self._model.init()

        if self.mode == "linear_probe":
            # Freeze encoder, only train head
            for name, param in self._model.named_parameters():
                if "head" not in name.lower() and "classifier" not in name.lower():
                    param.requires_grad = False

        logger.info("MOMENT loaded: %s (mode=%s)", self.model_name, self.mode)

    def run(
        self,
        dataset,
        train_indices,
        val_indices,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-4,
        device: str = "cpu",
    ) -> dict:
        """Train and evaluate MOMENT classifier.

        Returns dict with predictions, targets, metrics.
        """
        import torch
        from torch.utils.data import DataLoader, Subset

        from offshore_dl.evaluation.metrics import MetricRegistry

        self._load_model()
        model = self._model.to(device)

        # Create data loaders
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Train
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for features, targets, _meta in train_loader:
                features = features.to(device)
                targets = targets.to(device).long()

                optimizer.zero_grad()
                output = model(features)
                # MOMENT returns a ModelOutput object with logits
                logits = output.logits if hasattr(output, "logits") else output
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for features, targets, _meta in val_loader:
                features = features.to(device)
                output = model(features)
                logits = output.logits if hasattr(output, "logits") else output
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_targets.append(targets)

        predictions = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        metrics = MetricRegistry.compute("classification", predictions, targets)
        return {"predictions": predictions, "targets": targets, "metrics": metrics}
