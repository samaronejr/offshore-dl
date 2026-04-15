"""Tests for trunk-based DeepONet classification."""

import math
import pytest
import torch
from offshore_dl.models.deeponet import DeepONetModel, _simplex_etf


class TestSimplexETF:
    def test_shape(self):
        M = _simplex_etf(10, 64)
        assert M.shape == (10, 64)

    def test_equiangular(self):
        M = _simplex_etf(10, 64)
        M_norm = torch.nn.functional.normalize(M, dim=-1)
        cosines = M_norm @ M_norm.T
        # Off-diagonal should be approximately -1/(K-1) = -1/9
        off_diag = cosines[~torch.eye(10, dtype=bool)]
        expected = -1.0 / 9.0
        assert torch.allclose(off_diag, torch.tensor(expected), atol=0.01)


class TestTrunkClfDirect:
    """Direct mode: class_embed_dim == rank."""

    @pytest.fixture
    def model(self):
        return DeepONetModel(
            task="classification", n_vars=27, n_classes=10,
            window_size=14, rank=64, trunk_clf=True,
            class_embed_dim=64, etf_init=True,
            branch_type="conv1d",
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 14, 27)
        y = torch.randint(0, 10, (4,))
        return (x, y, [{} for _ in range(4)])

    def test_forward_shape(self, model, batch):
        out = model(batch[0])
        assert out.shape == (4, 10)

    def test_training_step_scalar(self, model, batch):
        loss = model.training_step(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_predict_shape(self, model, batch):
        preds = model.predict(batch)
        assert preds.shape == (4,)

    def test_class_embeddings_are_learnable(self, model):
        assert model.class_embeddings.requires_grad

    def test_temperature_is_learnable(self, model):
        assert model.log_temperature.requires_grad

    def test_gradient_flows_to_class_embeddings(self, model, batch):
        loss = model.training_step(batch)
        loss.backward()
        assert model.class_embeddings.grad is not None
        assert model.class_embeddings.grad.abs().sum() > 0


class TestTrunkClfMediated:
    """Trunk-mediated mode: class_embed_dim < rank."""

    @pytest.fixture
    def model(self):
        return DeepONetModel(
            task="classification", n_vars=27, n_classes=10,
            window_size=14, rank=64, trunk_clf=True,
            class_embed_dim=8, trunk_hidden=[64, 64],
            branch_type="conv1d",
        )

    @pytest.fixture
    def batch(self):
        x = torch.randn(4, 14, 27)
        y = torch.randint(0, 10, (4,))
        return (x, y, [{} for _ in range(4)])

    def test_forward_shape(self, model, batch):
        out = model(batch[0])
        assert out.shape == (4, 10)

    def test_trunk_exists(self, model):
        assert model.trunk is not None

    def test_gradient_flows_through_trunk(self, model, batch):
        loss = model.training_step(batch)
        loss.backward()
        # Check trunk MLP has gradients
        for name, param in model.trunk.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for trunk.{name}"


class TestFixedETF:
    """Fixed (frozen) ETF mode."""

    @pytest.fixture
    def model(self):
        return DeepONetModel(
            task="classification", n_vars=27, n_classes=10,
            window_size=14, rank=64, trunk_clf=True,
            class_embed_dim=64, etf_init=True, fixed_etf=True,
            branch_type="conv1d",
        )

    def test_class_embeddings_frozen(self, model):
        assert not model.class_embeddings.requires_grad

    def test_forward_still_works(self, model):
        x = torch.randn(4, 14, 27)
        out = model(x)
        assert out.shape == (4, 10)


class TestBackwardCompatibility:
    """Ensure trunk_clf=False preserves original behavior."""

    @pytest.fixture
    def model(self):
        return DeepONetModel(
            task="classification", n_vars=27, n_classes=10,
            window_size=14, rank=64, trunk_clf=False,
            branch_type="conv1d",
        )

    def test_has_mlp_head(self, model):
        assert model.head is not None

    def test_no_trunk(self, model):
        assert model.trunk is None

    def test_no_class_embeddings(self, model):
        assert not hasattr(model, 'class_embeddings') or model.class_embeddings is None

    def test_forward_shape(self, model):
        x = torch.randn(4, 14, 27)
        out = model(x)
        assert out.shape == (4, 10)
