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
    mask = U.data != 0  # guard against explicitly-stored zero entries
    keys = U.row[mask].astype(np.int64) * N + U.col[mask].astype(np.int64)
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


# ── common node set ──────────────────────────────────────────────────────────

def valid_vision_mask(feats: dict) -> np.ndarray:
    """Rows where EVERY vision encoder has a nonzero feature (missing images -> zero rows)."""
    N = next(iter(feats.values())).shape[0]
    mask = np.ones(N, dtype=bool)
    for v in VISION:
        mask &= (np.abs(feats[v]).sum(axis=1) > 0)
    return mask


# ── consensus core ───────────────────────────────────────────────────────────

def consensus_counts(cell_keys_list: list):
    """(unique_keys, counts) where counts[k] = #cells containing unique_keys[k]."""
    allk = np.concatenate(cell_keys_list) if cell_keys_list else np.empty(0, np.int64)
    uniq, counts = np.unique(allk, return_counts=True)
    return uniq, counts


def survival_curve(counts: np.ndarray, n_cells: int) -> np.ndarray:
    """Length-n_cells array; index t-1 = #edges present in >= t cells."""
    return np.array([int((counts >= t).sum()) for t in range(1, n_cells + 1)], dtype=np.int64)


def core_edges(unique_keys: np.ndarray, counts: np.ndarray, t: int, N: int) -> np.ndarray:
    """(M, 2) node-index edge list for edges present in >= t cells."""
    keys = unique_keys[counts >= t]
    return np.stack([keys // N, keys % N], axis=1).astype(np.int64)


class _SubShim:
    """Minimal stand-in for redcaps_buddy.Data: only .sub_id and .sub_names are used."""
    def __init__(self, sub_id, sub_names):
        self.sub_id = np.asarray(sub_id)
        self.sub_names = list(sub_names)


def core_subreddit_lift(unique_keys, counts, N, sub_id, sub_names, n_cells: int):
    """Same-subreddit lift of the >= t consensus core, for t = 1..n_cells."""
    import redcaps_buddy as rb
    shim = _SubShim(sub_id, sub_names)
    out = []
    for t in range(1, n_cells + 1):
        e = core_edges(unique_keys, counts, t, N)
        if len(e) == 0:
            out.append({"t": t, "n_edges": 0, "lift": float("nan")})
            continue
        res = rb.subreddit_lift(shim, e)
        out.append({"t": t, "n_edges": int(len(e)), "lift": float(res["overall_lift"])})
    return out


# ── feature loading + per-cell graph building ────────────────────────────────

def load_grid_features(smoke: int = 0):
    """
    Load all 8 grid feature matrices in redcaps row order, slice to the common
    node set (rows valid across every vision encoder). Returns
    (feats, sub_id, sub_names, vmask) with feats sliced to the common nodes.
    smoke>0 keeps only the first `smoke` valid rows (pipeline sanity, not interpreted).
    """
    import redcaps_buddy as rb
    from extract_heldout import cache_path

    data = rb.load_data()
    feats = {"clip_img": np.ascontiguousarray(data.img, dtype=np.float32),
             "clip_txt": np.ascontiguousarray(data.txt, dtype=np.float32)}
    for m in ["dinov2", "siglip_v", "vit_sup", "minilm", "bge", "e5"]:
        p = cache_path("redcaps", m, 0)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing held-out cache {p}; run:\n"
                f"  python src/test/20260708_heldout_grid/extract_heldout.py "
                f"--dataset redcaps --model {m}")
        feats[m] = np.load(p).astype(np.float32)

    vmask = valid_vision_mask(feats)
    if smoke:
        idx = np.where(vmask)[0][:smoke]
        keep = np.zeros(data.n, dtype=bool)
        keep[idx] = True
        vmask = keep

    feats = {k: v[vmask] for k, v in feats.items()}
    sub_id = data.sub_id[vmask]
    print(f"[cross-vlm] common nodes: {int(vmask.sum())}/{data.n} "
          f"(dropped {int((~vmask).sum())})")
    return feats, sub_id, data.sub_names, vmask


def build_cell_graphs(feats: dict, K: int = 30, device: str = "cuda", use_half: bool = True):
    """
    Build one mutual-kNN graph per distinct feature matrix (8 total), then the 16
    cells' B (intersection) and E (union) edge sets. Returns (cell_B, cell_E, N).
    """
    from src.conditional_buddy.buddy_graph import mutual_knn, union_graph

    N = next(iter(feats.values())).shape[0]
    A = {name: mutual_knn(feats[name], K=K, device=device, use_half=use_half)
         for name in feats}
    cell_B, cell_E = {}, {}
    for v, t in CELLS:
        Aimg, Atxt = A[v], A[t]
        B = Aimg.multiply(Atxt)
        B.data[:] = 1.0
        B = B.tocsr()
        B.eliminate_zeros()
        E = union_graph(Aimg, Atxt)
        key = f"{v}x{t}"
        cell_B[key] = adj_to_keys(B)
        cell_E[key] = adj_to_keys(E)
    return cell_B, cell_E, N
