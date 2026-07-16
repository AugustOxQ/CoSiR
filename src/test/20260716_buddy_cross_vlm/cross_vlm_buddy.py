"""
Cross-VLM buddy survival: rebuild the cross-modal buddy graph for every
(vision encoder x text encoder) cell of a 4x4 grid and measure how much the
resulting buddy edge sets agree. Pure-function library; see run_grid.py for the
CLI driver.

Edge sets are sorted np.int64 arrays of keys `i*N + j` (i < j), N = #nodes.

Design: docs/superpowers/specs/2026-07-16-buddy-cross-vlm-survival-design.md
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, triu

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT,
          os.path.join(ROOT, "src", "test", "20260623_redcaps_buddy"),
          os.path.join(ROOT, "src", "test", "20260708_heldout_grid")):
    if p not in sys.path:
        sys.path.insert(0, p)

VISION = ["clip_img", "dinov2", "siglip_v", "vit_sup"]
TEXT = ["clip_txt", "minilm", "bge", "e5"]
CELLS = [(v, t) for v in VISION for t in TEXT]  # 16 cells


# ── edge-set representation ──────────────────────────────────────────────────

def adj_to_keys(A: csr_matrix) -> np.ndarray:
    """Upper-triangular (i<j) edges of a symmetric adjacency as sorted int64 keys i*N+j."""
    N = A.shape[0]
    U = triu(A, k=1).tocoo()
    keys = U.row.astype(np.int64) * N + U.col.astype(np.int64)
    keys.sort()
    return keys


# ── pairwise agreement ───────────────────────────────────────────────────────

def jaccard(a: np.ndarray, b: np.ndarray):
    """(jaccard, overlap_coef, intersection_size) for two sorted-unique key arrays."""
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    union = int(a.size + b.size - inter)
    jac = inter / union if union else 0.0
    denom = min(int(a.size), int(b.size))
    ov = inter / denom if denom else 0.0
    return jac, ov, inter


def perm_null_jaccard(a: np.ndarray, b: np.ndarray, N: int, n_perm: int = 200, seed: int = 42):
    """
    Chance-correct Jaccard(a, b) by node-relabeling `b` under random permutations
    (preserves b's exact degree sequence, destroys alignment). Returns observed,
    null mean, lift = observed/null_mean, and percentile of observed in the null.
    """
    obs, _, _ = jaccard(a, b)
    bi, bj = b // N, b % N
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        perm = rng.permutation(N)
        pi, pj = perm[bi], perm[bj]
        lo = np.minimum(pi, pj).astype(np.int64)
        hi = np.maximum(pi, pj).astype(np.int64)
        bk = np.unique(lo * N + hi)
        nulls[k], _, _ = jaccard(a, bk)
    null_mean = float(nulls.mean())
    lift = obs / null_mean if null_mean > 0 else float("inf")
    percentile = float((nulls <= obs).mean())
    return {"observed": obs, "null_mean": null_mean, "lift": lift, "percentile": percentile}


def agreement_matrix(cell_keys: dict, N: int, n_perm: int = 200, seed: int = 42):
    """Full pairwise Jaccard / overlap / chance-lift across all cells."""
    names = list(cell_keys.keys())
    n = len(names)
    jac = np.eye(n, dtype=np.float64)
    ov = np.eye(n, dtype=np.float64)
    lift = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cell_keys[names[i]], cell_keys[names[j]]
            jj, oo, _ = jaccard(a, b)
            jac[i, j] = jac[j, i] = jj
            ov[i, j] = ov[j, i] = oo
            res = perm_null_jaccard(a, b, N, n_perm=n_perm, seed=seed)
            lift[i, j] = lift[j, i] = res["lift"]
    off = ~np.eye(n, dtype=bool)
    return {
        "names": names,
        "jaccard": jac,
        "overlap": ov,
        "lift": lift,
        "median_offdiag_jaccard": float(np.median(jac[off])),
        "median_offdiag_lift": float(np.nanmedian(lift[off])),
    }
