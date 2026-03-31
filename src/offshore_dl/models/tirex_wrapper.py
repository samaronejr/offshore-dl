"""TiRex zero-shot forecasting wrapper.

TiRex is a 35M-parameter xLSTM-based time series foundation model
from NX-AI. Requires GPU with CUDA compute capability ≥8.0 and
installation from the GitHub repository (not PyPI).

This wrapper provides the full BaseModel interface but raises
ImportError when TiRex is not available.
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel

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


class TiRexWrapper(BaseModel):
    """Zero-shot forecasting via TiRex (xLSTM backbone).

    Requires: GPU with CUDA ≥8.0 and ``pip install git+https://github.com/NX-AI/tirex``.

    Args:
        task: ``"forecasting"`` or ``"anomaly"``. Classification not supported.
        n_vars: Number of input variables.
        context_length: Max context length.
        horizon: Forecast horizon.
        window_size: Input window size.
        target_channel: Which channel to forecast (default: 0).
        lr: Unused (zero-shot).
    """

    def __init__(
        self,
        task: str,
        n_vars: int,
        context_length: int = 512,
        horizon: int = 30,
        window_size: int = 48,
        lr: float = 0.001,
        **kwargs,
    ) -> None:
        if task == "classification":
            msg = "TiRex does not support classification"
            raise ValueError(msg)

        if not _TIREX_AVAILABLE:
            msg = (
                "TiRex is not installed. Requires GPU with CUDA ≥8.0. "
                "Install via: pip install git+https://github.com/NX-AI/tirex"
            )
            raise ImportError(msg)

        super().__init__(task=task, n_vars=n_vars)
        self.context_length = context_length
        self.horizon = horizon
        self.window_size = window_size
        self.target_channel = kwargs.get("target_channel", 0)
        self.lr = lr

        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._model = None

    def _load_model(self):
        """Lazy-load TiRex model from NX-AI."""
        if self._model is None:
            self._model = _tirex_load_model("NX-AI/TiRex")
            logger.info("TiRex model loaded (NX-AI/TiRex)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-shot forecast via TiRex.

        Args:
            x: Input ``(batch, window, n_vars)``.

        Returns:
            - forecasting: ``(batch, horizon)`` mean predictions
            - anomaly: ``(batch, window, n_vars)`` — predicts each channel
        """
        self._load_model()

        if self.task == "forecasting":
            # Extract target channel: (batch, window)
            context = x[:, :, self.target_channel]
            # TiRex forecast returns (quantiles, mean)
            quantiles, mean = self._model.forecast(
                context=context, prediction_length=self.horizon
            )
            # mean shape: (batch, horizon)
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean, dtype=torch.float32)
            return mean

        elif self.task == "anomaly":
            # NOTE: FMs forecast the NEXT window of values, not reconstruct the
            # current window.  This means the "reconstruction error" is actually
            # a one-step-ahead forecasting error per channel.  Cross-sensor
            # correlations are not captured (channels processed independently).
            # See audit finding H9/H11.
            # Per-channel prediction for reconstruction
            all_preds = []
            for ch in range(self.n_vars):
                context = x[:, :, ch]  # (batch, window)
                quantiles, mean = self._model.forecast(
                    context=context, prediction_length=self.window_size
                )
                if not isinstance(mean, torch.Tensor):
                    mean = torch.tensor(mean, dtype=torch.float32)
                all_preds.append(mean)

            # Stack: (batch, window, n_vars)
            return torch.stack(all_preds, dim=-1)

        msg = f"Unsupported task: {self.task}"
        raise ValueError(msg)

    def training_step(self, batch: tuple) -> torch.Tensor:
        warnings.warn("TiRex is zero-shot — training_step is a no-op", stacklevel=2)
        return torch.tensor(0.0, requires_grad=True)

    def predict(self, batch: tuple) -> torch.Tensor:
        features, _targets, _metadata = batch
        return self.forward(features)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        return torch.optim.SGD([self._dummy], lr=self.lr)
