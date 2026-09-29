"""Full-rank and variance-truncated PCA whitening for frozen features."""

import numpy as np
from sklearn.decomposition import PCA


def pca_whiten(features: np.ndarray, seed: int = 42) -> tuple[np.ndarray, PCA]:
    """Fit full-rank PCA in float64 so small-variance components remain stable."""
    features = np.asarray(features, dtype=np.float64)
    pca = PCA(n_components=features.shape[1], whiten=True, random_state=seed)
    pca.fit(features)
    return pca.transform(features), pca


def select_pca_rank(pca: PCA, variance_threshold: float) -> int:
    """Return the smallest PCA rank reaching the requested original variance."""
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1]")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return min(int(np.searchsorted(cumulative, variance_threshold) + 1), len(cumulative))


def pca_whiten_truncated(
    features: np.ndarray, variance_threshold: float, seed: int = 42
) -> tuple[np.ndarray, PCA]:
    """Fit float64 full-rank PCA, then return only the top-K whitened columns.

    K is the smallest rank explaining at least ``variance_threshold`` of the
    original variance. The returned PCA retains its full-rank fit; transform
    new rows with ``pca.transform(rows)[:, :K]``.
    """
    whitened, pca = pca_whiten(features, seed=seed)
    return whitened[:, :select_pca_rank(pca, variance_threshold)], pca
