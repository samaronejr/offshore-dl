"""Tests for MLPModel — classification only."""

import torch
import pytest

from offshore_dl.models.mlp import MLPModel


class TestMLPClassification:
    """MLP for 3W 10-class classification on statistical features."""

    @pytest.fixture
    def model(self):
        return MLPModel(task="classification", n_vars=27, n_classes=10, window_size=14, hidden_dims=[32, 16])

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
        assert loss.requires_grad

    def test_validation_step_returns_dict(self, model, batch) -> None:
        result = model.validation_step(batch)
        assert "loss" in result
        assert "predictions" in result
        assert result["predictions"].shape == (4,)

    def test_predict_returns_class_indices(self, model, batch) -> None:
        preds = model.predict(batch)
        assert preds.shape == (4,)
        assert preds.dtype == torch.int64

    def test_class_weights_update(self, model, batch) -> None:
        model.set_class_weights(torch.ones(10))
        x, _, _ = batch
        out = model(x)
        assert out.shape == (4, 10)

    def test_rejects_non_classification_task(self) -> None:
        with pytest.raises(ValueError, match="only supports classification"):
            MLPModel(task="forecasting", n_vars=27, n_classes=10)

    def test_configure_optimizers_returns_adamw(self, model) -> None:
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)
