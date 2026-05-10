"""Tests for physics-informed 3W feature extraction."""

from __future__ import annotations

import numpy as np
import torch

from offshore_dl.data.datasets import ThreeWFeatureDataset, ThreeWPhysicsDataset
from offshore_dl.data.feature_extractor import PhysicsFeatureExtractor


def test_physics_feature_extractor_output_shape() -> None:
    extractor = PhysicsFeatureExtractor()
    window = np.random.default_rng(42).standard_normal((720, 27)).astype(np.float32)

    features = extractor.extract(window)

    assert features.shape == (14, 4)


def test_physics_feature_extractor_handles_nan_input() -> None:
    extractor = PhysicsFeatureExtractor()
    window = np.full((720, 27), np.nan, dtype=np.float32)

    features = extractor.extract(window)

    assert np.array_equal(features, np.zeros((14, 4), dtype=np.float32))
    assert np.isfinite(features).all()


def test_physics_feature_extractor_is_deterministic() -> None:
    extractor = PhysicsFeatureExtractor()
    window = np.random.default_rng(7).standard_normal((720, 27)).astype(np.float32)

    first = extractor.extract(window)
    second = extractor.extract(window)

    assert np.array_equal(first, second)


def test_threew_feature_dataset_release_inner_cache() -> None:
    """Feature HPO can drop raw caches after descriptor pre-computation."""

    class DummyInner:
        def __init__(self) -> None:
            self._data_cache = {"instance": object()}
            self._feature_cache = {"instance": np.zeros((4, 2), dtype=np.float32)}

    dataset = ThreeWFeatureDataset.__new__(ThreeWFeatureDataset)
    dataset._inner = DummyInner()

    dataset.release_inner_cache()

    assert dataset._inner._data_cache == {}
    assert dataset._inner._feature_cache == {}


def test_threew_physics_dataset_concatenates_features(monkeypatch) -> None:
    stat_features = torch.zeros((14, 27), dtype=torch.float32)
    raw_tensor = torch.ones((720, 27), dtype=torch.float32)

    def fake_super_getitem(self, index: int):
        return stat_features, 3, {"instance_id": "well-1"}

    monkeypatch.setattr(ThreeWFeatureDataset, "__getitem__", fake_super_getitem)

    dataset = ThreeWPhysicsDataset.__new__(ThreeWPhysicsDataset)
    dataset._inner = [(raw_tensor, 3, {"instance_id": "well-1"})]
    dataset._physics_cache = [None]
    dataset.physics_extractor = PhysicsFeatureExtractor()

    combined, label, metadata = ThreeWPhysicsDataset.__getitem__(dataset, 0)

    assert combined.shape == (14, 31)
    assert label == 3
    assert metadata["instance_id"] == "well-1"
    assert torch.isfinite(combined).all()
