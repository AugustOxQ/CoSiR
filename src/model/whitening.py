"""Full-rank PCA whitening for frozen feature arrays."""

import numpy as np
from sklearn.decomposition import PCA


def pca_whiten(features: np.ndarray, seed: int = 42) -> tuple[np.ndarray, PCA]:
    """Fit full-rank PCA in float64 so small-variance components remain stable."""
    features = np.asarray(features, dtype=np.float64)
    pca = PCA(n_components=features.shape[1], whiten=True, random_state=seed)
    pca.fit(features)
    return pca.transform(features), pca
