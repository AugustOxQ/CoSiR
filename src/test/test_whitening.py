"""PCA whitening should remove correlation and equalize component variance."""

import numpy as np

from src.model.whitening import pca_whiten, pca_whiten_truncated


def test_pca_whiten_decorrelates_and_scales_full_rank_features():
    rng = np.random.default_rng(42)
    independent = rng.normal(size=(2000, 4)) * [5.0, 2.0, 0.5, 0.1]
    features = independent @ np.array([
        [1.0, 0.7, 0.0, 0.0],
        [0.0, 1.0, 0.8, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.2, 0.0, 0.0, 1.0],
    ])

    whitened, pca = pca_whiten(features)
    covariance = np.cov(whitened, rowvar=False)

    assert whitened.shape == features.shape
    np.testing.assert_allclose(pca.transform(features), whitened)
    np.testing.assert_allclose(np.diag(covariance), 1.0, atol=1e-8)
    np.testing.assert_allclose(covariance - np.diag(np.diag(covariance)), 0.0, atol=1e-8)


def test_pca_whiten_preserves_tiny_full_rank_component_from_float32_input():
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(2000, 4)) * [5.0, 2.0, 0.5, 1e-5]
    features = (latent @ np.array([
        [1.0, 0.7, 0.0, 0.0],
        [0.0, 1.0, 0.8, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.2, 0.0, 0.0, 1.0],
    ])).astype(np.float32)

    whitened, pca = pca_whiten(features)

    assert pca.n_components_ == features.shape[1]
    np.testing.assert_allclose(np.var(whitened, axis=0, ddof=1), 1.0, atol=0.02)


def test_pca_whiten_truncated_drops_originally_low_variance_directions():
    rng = np.random.default_rng(13)
    latent = rng.normal(size=(4000, 12)) * np.sqrt(
        [100.0, 25.0, 9.0, 4.0] + [1e-8] * 8
    )
    rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    features = (latent @ rotation).astype(np.float32)

    whitened, pca = pca_whiten_truncated(features, variance_threshold=0.99)

    assert whitened.shape == (len(features), 4)
    assert pca.n_components_ == features.shape[1]
    dropped = (features.astype(np.float64) - features.mean(axis=0)) @ pca.components_[4:].T
    assert np.var(dropped, axis=0, ddof=1).max() < 1e-6
    covariance = np.cov(whitened, rowvar=False)
    np.testing.assert_allclose(np.diag(covariance), 1.0, atol=1e-8)
    np.testing.assert_allclose(covariance - np.diag(np.diag(covariance)), 0.0, atol=1e-8)
