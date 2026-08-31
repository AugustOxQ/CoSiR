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

import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.manifold import SpectralEmbedding

# arpack is exact and well-validated for small graphs, but it relies on a sparse
# shift-invert factorization that does not scale (memory/time) and stalls on
# disconnected graphs, whose Laplacian has one near-zero eigenvalue per component
# clustered at 0 — exactly arpack's worst case. Above this many nodes we switch to a
# matrix-free multigrid solver (amg) that scales to 1e5+ and tolerates disconnection.
ARPACK_MAX_N = 20000


def _amg_spectral_embedding(adjacency: csr_matrix, n_dim: int, seed: int) -> np.ndarray:
    """``SpectralEmbedding(eigen_solver="amg")``, with one dtype leak patched out.

    Inside sklearn's amg path (``sklearn.manifold._spectral_embedding``), the
    diagonal shift is built as ``diag_shift = 1e-5 * sparse.eye(N)``.
    ``scipy.sparse.eye`` defaults to float64 regardless of the Laplacian's own
    dtype, so ``laplacian += diag_shift`` silently promotes the whole matrix to
    float64. That's harmless at N~150k (~1GB extra) but at N~3M the doubling
    propagates through every level of pyamg's multigrid hierarchy plus the
    LOBPCG working vectors, adding tens of GB and crashing the process
    (SIGKILL, no clean Python exception — the OOM kill happens below Python's
    exception machinery).

    An earlier version of this function hand-reimplemented sklearn's whole amg
    algorithm to fix this, but a transcription difference somewhere broke
    eigenvector correctness (verified against sklearn's own arpack path on a
    synthetic clustered graph: reimplementation scored ~0 ARI vs ~0.75 for both
    sklearn's original amg and arpack). Far lower-risk: monkeypatch just the
    one leaking call for the duration of sklearn's own (already-validated) code
    path, so every other line runs exactly as it always has.

    Forcing float32 alone *also* broke correctness the same way (~0 ARI) — a
    second, independent bug: scipy's ``lobpcg(tol=None)`` defaults to
    ``sqrt(eps(dtype)) * n``. At N~3M that's ~1069 in float32 (eps~1.19e-7) vs
    ~46 in float64 (eps~2.22e-16) — a nonsensically loose tolerance that makes
    LOBPCG report "converged" almost immediately on an unconverged, effectively
    random result. Fixed by passing an explicit tolerance computed from
    float64's epsilon regardless of the array's actual (float32) dtype, via
    ``eigen_tol=`` — sklearn's own auto-tolerance formula, just no longer tied
    to whatever dtype the array happens to be in.

    Third change: pyamg's default ``smooth=('jacobi', ...)`` prolongation
    smoothing adds substantial fill-in per multigrid level. On this project's
    graphs (mutual-kNN union, not a mesh/PDE operator smoothed_aggregation was
    designed for) it buys little — measured on a real 1.5M-node subset,
    ``smooth=None`` (plain tentative-prolongation aggregation, i.e. classical
    AMG-style coarsening without the smoothing step) cut peak memory roughly
    3x with a clean 6-level hierarchy ([1.5M, 114k, 4k, 38, 11, 2]) vs. the
    smoothed default, and produced numerically indistinguishable embeddings on
    the same clustered-graph correctness check used to validate the dtype fix
    above. sklearn's ``SpectralEmbedding`` doesn't expose this pyamg parameter
    either, so it's injected the same way: monkeypatching the one call site.
    """
    import pyamg
    import sklearn.manifold._spectral_embedding as _se_mod

    orig_eye = _se_mod.sparse.eye
    orig_solver = pyamg.smoothed_aggregation_solver

    def _eye_no_promote(n, *args, **kwargs):
        kwargs.setdefault("dtype", np.float32)
        return orig_eye(n, *args, **kwargs)

    def _solver_unsmoothed(A, *args, **kwargs):
        kwargs.setdefault("smooth", None)
        return orig_solver(A, *args, **kwargs)

    tol = np.sqrt(np.finfo(np.float64).eps) * adjacency.shape[0]

    _se_mod.sparse.eye = _eye_no_promote
    pyamg.smoothed_aggregation_solver = _solver_unsmoothed
    try:
        se = SpectralEmbedding(
            n_components=n_dim,
            affinity="precomputed",
            eigen_solver="amg",
            eigen_tol=tol,
            random_state=seed,
        )
        return se.fit_transform(adjacency).astype(np.float32)
    finally:
        _se_mod.sparse.eye = orig_eye
        pyamg.smoothed_aggregation_solver = orig_solver


def spectral_embedding(
    D_mixed: csr_matrix,
    n_dim: int,
    seed: int = 42,
    eigen_solver: str = "auto",
    b_edges: csr_matrix = None,
    b_weight: float = 1.0,
) -> np.ndarray:
    """
    Laplacian Eigenmaps on the sparse affinity graph derived from mixed distances.

    Affinity = 1 - distance (both rank-normalised, values in [0, 1]). The graph is
    symmetrised and fed to SpectralEmbedding with a precomputed affinity, so no
    dense N×N matrix is ever allocated.

    eigen_solver:
        "auto"   — arpack for N ≤ ARPACK_MAX_N (exact, preserves prior results),
                   amg for larger N (matrix-free, scales / tolerates disconnection).
        "arpack" | "amg" | "lobpcg" — force a specific sklearn solver.
    amg requires pyamg; if it is missing we fall back to lobpcg with a warning.

    b_edges, b_weight: optional "B-lean". When b_edges (a symmetric binary matrix of
        strict-intersection buddy edges, a subset of D_mixed's sparsity) is given and
        b_weight != 1.0, the affinity of those edges is multiplied by b_weight before
        the eigendecomposition — pulling strict buddies tighter while the rest of the
        graph still provides connectivity. b_weight=1.0 (default) is a no-op.
    """
    A_mixed = D_mixed.copy().tocsr()
    A_mixed.data = 1.0 - A_mixed.data            # invert: closer → higher weight
    A_mixed = (A_mixed + A_mixed.T) * 0.5         # symmetrise numerical noise

    if b_edges is not None and b_weight != 1.0:
        Bm = b_edges.tocsr().copy()
        Bm.data[:] = 1.0
        Bm = ((Bm + Bm.T) * 0.5).tocsr()          # ensure symmetric binary support
        Bm.data[:] = 1.0
        # affinity on B edges scaled by b_weight (weights may exceed 1 — fine for a
        # precomputed affinity). B ⊆ E so A_mixed.multiply(Bm) selects the B affinities.
        A_mixed = (A_mixed + (b_weight - 1.0) * A_mixed.multiply(Bm)).tocsr()

    n = A_mixed.shape[0]
    if eigen_solver == "auto":
        eigen_solver = "arpack" if n <= ARPACK_MAX_N else "amg"

    if eigen_solver == "amg":
        try:
            import pyamg  # noqa: F401
        except ImportError:
            warnings.warn(
                "pyamg not installed; falling back to lobpcg for spectral embedding. "
                "Install pyamg (`pip install pyamg`) for the faster, more robust amg "
                "solver on large graphs.",
                RuntimeWarning,
            )
            eigen_solver = "lobpcg"
        else:
            # Bypass sklearn's SpectralEmbedding here — its amg path has a dtype
            # leak (float64 promotion via sparse.eye's default dtype) that's
            # catastrophic at N~3M; see _amg_spectral_embedding's docstring.
            return _amg_spectral_embedding(A_mixed, n_dim, seed)

    se = SpectralEmbedding(
        n_components=n_dim,
        affinity="precomputed",
        eigen_solver=eigen_solver,
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
