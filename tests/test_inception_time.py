"""Tests for InceptionTimeModel — classification only."""

from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from offshore_dl.models.inception_time import InceptionTimeModel


class TestInceptionTimeClassification:
    """InceptionTime for 3W 10-class classification."""

    @pytest.fixture
    def model(self):
        return InceptionTimeModel(
            task="classification",
            n_vars=27,
            n_classes=10,
            window_size=14,
            n_filters=32,
            depth=3,
            lr=0.001,
            weight_decay=1e-4,
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 14, 27)
        y = torch.randint(0, 10, (4,))
        return x, y, [{}] * 4

    def test_forward_shape(self, model, batch) -> None:
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 10)

    def test_training_step_returns_scalar_loss(self, model, batch) -> None:
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_validation_step_returns_dict(self, model, batch) -> None:
        result = model.validation_step(batch)
        assert "loss" in result
        assert "predictions" in result
        assert "targets" in result

    def test_predict_returns_class_indices(self, model, batch) -> None:
        preds = model.predict(batch)
        assert preds.shape == (4,)

    def test_configure_optimizers(self, model) -> None:
        opt = model.configure_optimizers()
        assert isinstance(opt, AdamW)

    def test_non_classification_task_rejected(self) -> None:
        with pytest.raises(ValueError, match="classification"):
            InceptionTimeModel(task="forecasting", n_vars=27)
