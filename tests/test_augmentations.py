from __future__ import annotations

import numpy as np

from offshore_dl.data.transforms import (
    feature_dropout_augment,
    gaussian_noise_augment,
    time_feature_warp_augment,
)


def test_gaussian_noise_augment_preserves_shape_and_changes_values() -> None:
    x = np.arange(14 * 27, dtype=np.float32).reshape(14, 27)
    y = gaussian_noise_augment(x, sigma_frac=0.5)

    assert y.shape == x.shape
    assert not np.array_equal(x, y)
    assert np.isfinite(y).all()


def test_feature_dropout_augment_preserves_shape_and_changes_values() -> None:
    x = np.arange(14 * 27, dtype=np.float32).reshape(14, 27)
    y = feature_dropout_augment(x, drop_prob=1.0)

    assert y.shape == x.shape
    assert not np.array_equal(x, y)
    assert np.isfinite(y).all()


def test_time_feature_warp_augment_preserves_shape_and_changes_values() -> None:
    x = np.arange(14 * 27, dtype=np.float32).reshape(14, 27)
    y = time_feature_warp_augment(x, scale_range=(1.5, 1.5))

    assert y.shape == x.shape
    assert not np.array_equal(x, y)
    assert np.isfinite(y).all()
