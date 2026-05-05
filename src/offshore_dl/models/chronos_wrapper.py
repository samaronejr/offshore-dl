"""Chronos zero-shot forecasting wrapper.

Wraps Amazon's Chronos (T5-based) time series foundation model for
zero-shot forecasting and anomaly detection. No training — inference only.

Chronos operates on univariate series. For multivariate input, we
forecast the target channel independently.

For anomaly detection: predict the next window and compute
reconstruction error against actual values.
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel

logger = logging.getLogger(__name__)


class ChronosWrapper(BaseModel):
    """Zero-shot forecasting via Amazon Chronos.

    Args:
        task: ``"forecasting"`` or ``"anomaly"``. Classification not supported.
        n_vars: Number of input variables.
        model_name: HuggingFace model ID.
        horizon: Forecast horizon (forecasting) or window size (anomaly).
        window_size: Input context length.
        n_samples: Number of probabilistic samples from Chronos.
        target_channel: Which channel to forecast (default: 0).
        lr: Unused (zero-shot), kept for BaseModel compatibility.
    """

    is_zero_shot = True

    def __init__(
        self,
        task: str,
        n_vars: int,
        model_name: str = "amazon/chronos-t5-tiny",
        horizon: int = 30,
        window_size: int = 48,
        n_samples: int = 20,
        target_channel: int = 0,
        lr: float = 0.001,
        **kwargs,
    ) -> None:
        if task == "classification":
            msg = "Chronos does not support classification — use forecasting or anomaly"
            raise ValueError(msg)

        super().__init__(task=task, n_vars=n_vars)
        self.model_name = model_name
        self.horizon = horizon
        self.window_size = window_size
        self.n_samples = n_samples
        self.target_channel = target_channel
        self.lr = lr

        self._pipeline = None
        # Dummy parameter so PyTorch treats this as a module with parameters
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _load_pipeline(self):
        """Lazy-load the Chronos pipeline."""
        if self._pipeline is None:
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                dtype=torch.float32,
            )
            logger.info("Chronos pipeline loaded: %s", self.model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-shot forecast via Chronos pipeline.

        Args:
            x: Input ``(batch, window, n_vars)``.

        Returns:
            - forecasting: ``(batch, horizon)`` median predictions
            - anomaly: ``(batch, window, n_vars)`` — predicts each channel
        """
        self._load_pipeline()
        device = x.device

        if self.task == "forecasting":
            # Extract target channel on CPU because the Chronos pipeline is
            # loaded with ``device_map="cpu"``.
            context = x[:, :, self.target_channel].detach().cpu()
            pred_length = self.horizon

            # Chronos predict: returns (batch, n_samples, pred_length)
            with torch.no_grad():
                samples = self._pipeline.predict(context, prediction_length=pred_length)

            # Median of samples: (batch, pred_length)
            median = samples.median(dim=1).values
            return median.to(device=device)

        elif self.task == "anomaly":
            # NOTE: FMs forecast the NEXT window of values, not reconstruct the
            # current window.  This means the "reconstruction error" is actually
            # a one-step-ahead forecasting error per channel.  Cross-sensor
            # correlations are not captured (channels processed independently).
            # See audit finding H9/H11.
            # Per-channel prediction for reconstruction
            all_preds = []
            for ch in range(self.n_vars):
                context = x[:, :, ch].detach().cpu()  # (batch, window)
                with torch.no_grad():
                    samples = self._pipeline.predict(context, prediction_length=self.window_size)
                median = samples.median(dim=1).values  # (batch, window)
                all_preds.append(median)

            # Stack: (batch, window, n_vars)
            return torch.stack(all_preds, dim=-1).to(device=device)

        msg = f"Unsupported task: {self.task}"
        raise ValueError(msg)

    def training_step(self, batch: tuple) -> torch.Tensor:
        """No-op — zero-shot model doesn't train. Returns dummy loss."""
        warnings.warn("Chronos is zero-shot — training_step is a no-op", stacklevel=2)
        device = batch[0].device if batch and isinstance(batch[0], torch.Tensor) else self._dummy.device
        return torch.tensor(0.0, device=device, requires_grad=True)

    def predict(self, batch: tuple) -> torch.Tensor:
        """Generate zero-shot predictions."""
        features, _targets, _metadata = batch
        return self.forward(features)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        """Return a dummy optimizer (zero-shot, no training)."""
        return torch.optim.SGD([self._dummy], lr=self.lr)
