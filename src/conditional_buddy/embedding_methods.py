"""
Embedding + normalization for the conditional-buddies pipeline (Steps 5-6).

    spectral_embedding   — Laplacian Eigenmaps on the sparse affinity graph
                           (no dense N×N allocation; scales to large N).
    normalise_embedding  — per-dimension rank → [-1, 1] (default) or z-score.

Why rank by default: Laplacian Eigenmaps localizes its low eigenvectors on a few
hub nodes, so a z-score (mean/std) normalization divides by an outlier-dominated
std and collapses ~all samples to one point — useless as an initialization.
Per-dimension rank normalization guarantees spread while preserving neighbourhood
ordering (buddies stay close), which is exactly what an init needs.

(SMACOF was considered but dropped: sklearn 1.8 removed the weight= argument its
weighted, missing-edge-aware variant required, and spectral + rank normalization
covers the use case and scales to MS-COCO / RedCaps.)
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.manifold import SpectralEmbedding


def spectral_embedding(D_mixed: csr_matrix, n_dim: int, seed: int = 42) -> np.ndarray:
    """
    Laplacian Eigenmaps on the sparse affinity graph derived from mixed distances.

    Affinity = 1 - distance (both rank-normalised, values in [0, 1]). The graph is
    symmetrised and fed to SpectralEmbedding with a precomputed affinity, so no
    dense N×N matrix is ever allocated.
    """
    A_mixed = D_mixed.copy().tocsr()
    A_mixed.data = 1.0 - A_mixed.data            # invert: closer → higher weight
    A_mixed = (A_mixed + A_mixed.T) * 0.5         # symmetrise numerical noise

    se = SpectralEmbedding(
        n_components=n_dim,
        affinity="precomputed",
        eigen_solver="arpack",
        random_state=seed,
    )
    return se.fit_transform(A_mixed).astype(np.float32)  # (N, n_dim)


def normalise_embedding(emb: np.ndarray, method: str = "rank") -> np.ndarray:
    """
    Normalise an embedding to ~[-1, 1] before assigning to conditions.

    method="rank"   (default): per-dimension rank → uniform [-1, 1]. Robust to the
                    eigenvector localization of Laplacian Eigenmaps; guarantees
                    spread while preserving neighbourhood ordering.
    method="zscore" (legacy):  zero-mean unit-variance per dim, clip ±3, /3. Kept
                    for comparison; collapses when a few dims are outlier-dominated.
    """
    if method == "rank":
        out = np.empty_like(emb, dtype=np.float32)
        n = emb.shape[0]
        for j in range(emb.shape[1]):
            ranks = rankdata(emb[:, j])           # 1..n, ties averaged
            u = (ranks - 0.5) / n                 # → (0, 1)
            out[:, j] = (u * 2.0 - 1.0).astype(np.float32)  # → (-1, 1)
        return out
    elif method == "zscore":
        from sklearn.preprocessing import StandardScaler

        emb = StandardScaler().fit_transform(emb)
        emb = np.clip(emb, -3.0, 3.0) / 3.0
        return emb.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalise method '{method}'. Use 'rank' or 'zscore'.")
