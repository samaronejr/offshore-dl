"""TimesFM 2.5 zero-shot forecasting wrapper.

Google's TimesFM requires Python <3.12 and JAX. This wrapper provides
the full BaseModel interface but raises ImportError with a clear message
when the dependency is unavailable.

Run in Docker with Python 3.11 for TimesFM support.
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn

from offshore_dl.models.base import BaseModel

logger = logging.getLogger(__name__)

_TIMESFM_AVAILABLE = False
try:
    import timesfm as _timesfm_lib

    _TIMESFM_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Check if TimesFM dependencies are installed."""
    return _TIMESFM_AVAILABLE


class TimesFMWrapper(BaseModel):
    """Zero-shot forecasting via Google TimesFM 2.5.

    Requires: ``pip install timesfm`` (Python <3.12, JAX backend).

    Args:
        task: ``"forecasting"`` or ``"anomaly"``. Classification not supported.
        n_vars: Number of input variables.
        context_length: Input context length for TimesFM.
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
            msg = "TimesFM does not support classification"
            raise ValueError(msg)

        if not _TIMESFM_AVAILABLE:
            msg = (
                "TimesFM is not installed. Requires Python <3.12 and JAX. "
                "Install via: pip install timesfm (in a Python 3.11 environment)"
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
        """Lazy-load TimesFM model with PyTorch backend."""
        if self._model is None:
            backend = "gpu" if torch.cuda.is_available() else "cpu"
            # For anomaly task, horizon_len must match window_size
            # so reconstruction covers the full window
            horizon_len = (
                self.window_size if self.task == "anomaly" else self.horizon
            )
            self._model = _timesfm_lib.TimesFm(
                hparams=_timesfm_lib.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=32,
                    horizon_len=horizon_len,
                    context_len=self.context_length,
                ),
                checkpoint=_timesfm_lib.TimesFmCheckpoint(
                    version="pytorch",
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
                ),
            )
            logger.info(
                "TimesFM model loaded (backend=%s, horizon_len=%d)",
                backend,
                horizon_len,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-shot forecast via TimesFM.

        Args:
            x: Input ``(batch, window, n_vars)``.

        Returns:
            - forecasting: ``(batch, horizon)`` mean predictions
            - anomaly: ``(batch, window, n_vars)`` — predicts each channel
        """
        self._load_model()

        if self.task == "forecasting":
            # Extract target channel: (batch, window) → numpy
            context = x[:, :, self.target_channel]
            x_np = context.detach().cpu().numpy()
            # TimesFM expects a list of 1-D arrays
            inputs = [arr for arr in x_np]
            # TimesFM forecast returns (mean_forecast, full_forecast)
            mean_forecast, full_forecast = self._model.forecast(inputs)
            return torch.tensor(mean_forecast, dtype=torch.float32)

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
                x_np = context.detach().cpu().numpy()
                inputs = [arr for arr in x_np]
                mean_forecast, full_forecast = self._model.forecast(inputs)
                pred = torch.tensor(mean_forecast, dtype=torch.float32)
                all_preds.append(pred)

            # Stack: (batch, window, n_vars)
            return torch.stack(all_preds, dim=-1)

        msg = f"Unsupported task: {self.task}"
        raise ValueError(msg)

    def training_step(self, batch: tuple) -> torch.Tensor:
        warnings.warn("TimesFM is zero-shot — training_step is a no-op", stacklevel=2)
        return torch.tensor(0.0, requires_grad=True)

    def predict(self, batch: tuple) -> torch.Tensor:
        features, _targets, _metadata = batch
        return self.forward(features)

    def configure_optimizers(self, cfg=None) -> torch.optim.Optimizer:
        return torch.optim.SGD([self._dummy], lr=self.lr)
