"""Tests for wavelet feature extraction."""

from __future__ import annotations

import numpy as np

from offshore_dl.data.feature_extractor import WaveletFeatureExtractor


def test_wavelet_feature_extractor_output_shape() -> None:
    extractor = WaveletFeatureExtractor(scales=[30, 90, 180, 360])
    window = np.random.default_rng(42).standard_normal((720, 27)).astype(np.float32)

    features = extractor.extract(window)

    assert features.shape == (4, 27)


def test_wavelet_feature_extractor_handles_nan_input() -> None:
    extractor = WaveletFeatureExtractor(scales=[30, 90, 180, 360])
    window = np.full((720, 27), np.nan, dtype=np.float32)

    features = extractor.extract(window)

    assert np.array_equal(features, np.zeros((4, 27), dtype=np.float32))
    assert np.isfinite(features).all()


def test_wavelet_feature_extractor_is_deterministic() -> None:
    extractor = WaveletFeatureExtractor(scales=[30, 90, 180, 360])
    window = np.random.default_rng(7).standard_normal((720, 27)).astype(np.float32)

    first = extractor.extract(window)
    second = extractor.extract(window)

    assert np.array_equal(first, second)
