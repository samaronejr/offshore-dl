"""Tests for HydraRocketModel — sklearn-style classifier, aeon required."""

from __future__ import annotations

import importlib

import pytest

aeon_available = importlib.util.find_spec("aeon") is not None

pytestmark = pytest.mark.skipif(
    not aeon_available, reason="aeon is not installed"
)


class TestHydraRocketModel:
    """HydraRocket uses fit/predict, not training_step/configure_optimizers."""

    @pytest.fixture
    def model(self):
        from offshore_dl.models.hydra_rocket import HydraRocketModel

        return HydraRocketModel(
            task="classification",
            n_vars=27,
            n_classes=10,
        )

    def test_instantiation(self, model) -> None:
        from offshore_dl.models.hydra_rocket import HydraRocketModel

        assert isinstance(model, HydraRocketModel)

    def test_training_step_raises_not_implemented(self, model) -> None:
        with pytest.raises(NotImplementedError):
            model.training_step(())

    def test_configure_optimizers_raises_not_implemented(self, model) -> None:
        with pytest.raises(NotImplementedError):
            model.configure_optimizers()
