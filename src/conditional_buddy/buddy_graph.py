"""
Cross-modal mutual-KNN ("buddy") graph construction.

Steps 1–4 of the conditional-buddies initialization pipeline:
    1. mutual_knn          — per-modality mutual K-nearest-neighbour graph
    2. union_graph         — broad union graph for initialization
    3. sparse_cosine_distance / ensure_min_degree
    4. rank_normalise_sparse / mix_distances

Nearest-neighbour search runs behind ``mutual_knn(..., backend=...)``:
"torch" is exact GPU brute-force topk (mathematically identical to
faiss.IndexFlatIP) — O(N^2), the dominant cost of the whole pipeline at
N~3M (~360s/modality measured). "cuvs" is approximate (CAGRA), effectively
O(N log N) — measured 2.2x faster than exact already at N=1.5M (37s vs 84s
per modality), widening as N grows, at ~98-99% recall@K on real embeddings
and equivalent downstream buddy-init quality. "auto" (the mutual_knn
default) picks by N: brute-force still wins through at least N=500k (ANN
index-build overhead isn't paid off yet — measured, both modalities via
mutual_knn end-to-end on real data: 17.9s exact vs 31.1s cuvs at N=500k, on
top of 3.2s vs 7.9s single-modality at N=300k), so there is no reason to
prefer cuvs below the threshold.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

# Below this N, exact brute-force topk is faster in practice (cuVS CAGRA's
# index-build overhead dominates at small N); above it, exact's O(N^2) cost
# starts to exceed cuvs's O(N log N)-ish cost. Measured on real embeddings,
# both modalities via mutual_knn end-to-end: exact still wins at N=500k
# (17.9s vs 31.1s cuvs); cuvs wins clearly by N=1.5M (~167s vs ~75s,
# extrapolated x2 from the single-modality 83.7s/37.3s measurement). The
# true crossover sits somewhere in between; 1M is a conservative middle
# ground backed by data on both sides, not the exact crossover point.
CUVS_MIN_N = 1_000_000


# ── Nearest-neighbour search (GPU brute-force) ───────────────────────────────


def _to_gpu_normalized(features: np.ndarray, device: str, use_half: bool) -> torch.Tensor:
    """L2-normalize features and move to the compute device (optionally fp16)."""
    feats = torch.from_numpy(np.ascontiguousarray(features)).float().to(device)
    feats = F.normalize(feats, dim=1)
    if use_half:
        feats = feats.half()
    return feats


def _knn_indices_torch(
    feats_gpu: torch.Tensor,
    K: int,
    batch_size: int,
) -> np.ndarray:
    """
    Exact cosine top-K over all rows, self excluded.

    feats_gpu: (N, D) L2-normalized tensor on the compute device.
    Returns:   (N, K) int64 array of neighbour indices.
    """
    N = feats_gpu.shape[0]
    feats_t = feats_gpu.t().contiguous()
    out = np.empty((N, K), dtype=np.int64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sims = feats_gpu[start:end] @ feats_t  # [b, N]
        # Exclude self by driving the diagonal entry to -inf before topk.
        rows = torch.arange(end - start, device=feats_gpu.device)
        cols = torch.arange(start, end, device=feats_gpu.device)
        sims[rows, cols] = float("-inf")
        out[start:end] = sims.topk(K, dim=1).indices.cpu().numpy()
    return out


def _top1_indices_torch(
    query_feats_gpu: torch.Tensor,
    db_feats_gpu: torch.Tensor,
    query_global_idx: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Top-1 cosine neighbour of each query row against the full database, self excluded.

    Batched over the query dimension like ``_knn_indices_torch`` above — an
    unbatched [n_q, N] matmul is fine at N~150k but allocates hundreds of GB at
    N~3M (a 25k-row isolated set against a 3.1M-row database is a 305GB fp32
    matrix), even though n_q is tiny relative to N.
    """
    n_q = query_feats_gpu.shape[0]
    db_t = db_feats_gpu.t().contiguous()
    out = np.empty(n_q, dtype=np.int64)
    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        sims = query_feats_gpu[start:end] @ db_t  # [b, N]
        rows = torch.arange(end - start, device=query_feats_gpu.device)
        cols = torch.from_numpy(query_global_idx[start:end]).to(query_feats_gpu.device)
        sims[rows, cols] = float("-inf")
        out[start:end] = sims.argmax(dim=1).cpu().numpy()
    return out


def _knn_indices_cuvs(
    feats_gpu: torch.Tensor,
    K: int,
    itopk_size: int = 128,
) -> np.ndarray:
    """
    Approximate cosine top-K over all rows via cuVS CAGRA, self excluded.

    Same contract as ``_knn_indices_torch``: feats_gpu is (N, D) L2-normalized on
    the compute device, returns (N, K) int64 neighbour indices. CAGRA's distance
    metric is inner-product on normalized vectors (== cosine similarity, matching
    the exact backend's ranking exactly).

    CAGRA requires float32 (fp16 not supported); the caller may have moved
    features to fp16 for the exact backend's memory budget, so this always
    upcasts its own copy regardless of feats_gpu's dtype.
    """
    from cuvs.neighbors import cagra

    N = feats_gpu.shape[0]
    feats_f32 = feats_gpu.float()
    index = cagra.build(cagra.IndexParams(metric="inner_product"), feats_f32)
    _, neighbors = cagra.search(
        cagra.SearchParams(itopk_size=itopk_size), index, feats_f32, K + 1
    )
    neighbors = torch.as_tensor(neighbors, device=feats_gpu.device).long()  # [N, K+1]

    # Self is almost always column 0 (a normalized vector's closest match is
    # itself) but isn't guaranteed to be — and on rare approximate-search misses
    # may not appear at all. Push any self-match to the end via a stable sort on
    # "is this column self", then take the first K columns; if self never
    # appears, this is equivalent to just dropping the K+1'th (least similar)
    # neighbour, which is the correct fallback.
    row_ids = torch.arange(N, device=feats_gpu.device).unsqueeze(1)
    is_self = (neighbors == row_ids).float()
    order = is_self.argsort(dim=1, stable=True)
    gathered = torch.gather(neighbors, 1, order)
    return gathered[:, :K].cpu().numpy()


def mutual_knn(
    features: np.ndarray,
    K: int,
    device: str = "cuda",
    batch_size: int = 1024,
    backend: str = "auto",
    use_half: bool = True,
    itopk_size: int = 128,
) -> csr_matrix:
    """
    Build a mutual KNN adjacency matrix.

    features: (N, D) float32 (L2-normalisation applied internally).
    Returns:  (N, N) sparse binary matrix A_mut where A_mut[i, j] = 1 iff j is in
              i's top-K AND i is in j's top-K.

    backend: "auto" (default) picks "torch" (exact) below CUVS_MIN_N and "cuvs"
        (approximate, CAGRA) above it — see module docstring for the measured
        crossover. "torch" or "cuvs" force one explicitly. cuvs falls back to
        torch with a warning if the package isn't importable (e.g. a lighter
        install without the RAPIDS stack).
    """
    N = features.shape[0]
    if backend == "auto":
        backend = "cuvs" if N >= CUVS_MIN_N else "torch"
    if backend == "cuvs":
        try:
            import cuvs  # noqa: F401
        except ImportError:
            import warnings

            warnings.warn(
                "mutual_knn backend='cuvs' requested but cuvs is not installed; "
                "falling back to exact 'torch' (slower at large N, but correct).",
                RuntimeWarning,
            )
            backend = "torch"
    if backend not in ("torch", "cuvs"):
        raise NotImplementedError(
            f"mutual_knn backend '{backend}' not implemented; use 'torch', 'cuvs', or 'auto'."
        )

    feats_gpu = _to_gpu_normalized(features, device, use_half)
    if backend == "cuvs":
        indices = _knn_indices_cuvs(feats_gpu, K, itopk_size)  # (N, K)
    else:
        indices = _knn_indices_torch(feats_gpu, K, batch_size)  # (N, K)

    rows = np.repeat(np.arange(N), K)
    cols = indices.reshape(-1)
    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    A_mut = A.multiply(A.T)  # keep edge only if mutual
    A_mut.data[:] = 1.0
    return A_mut.tocsr()


# ── Graph combination ────────────────────────────────────────────────────────


def union_graph(A_img: csr_matrix, A_txt: csr_matrix) -> csr_matrix:
    """Union of the two mutual graphs (edge exists in either modality), binarised."""
    E = (A_img + A_txt).tocsr()
    E.data[:] = 1.0
    return E


def ensure_min_degree(
    E: csr_matrix,
    img_feats: np.ndarray,
    txt_feats: np.ndarray,
    device: str = "cuda",
    use_half: bool = True,
) -> Tuple[csr_matrix, Dict[str, int]]:
    """
    Guarantee every node has degree >= 1.

    Mutual-KNN (even unioned) can leave isolated samples; every sample still needs
    a condition vector, so each isolated node is connected to its top-1 nearest
    neighbour in whichever modality gives the higher similarity. Edges are added
    symmetrically.

    Returns (E_fixed, stats) where stats reports the number of isolated nodes fixed.
    """
    E = E.tocsr()
    degrees = np.diff(E.indptr)
    isolated = np.where(degrees == 0)[0]
    stats = {"num_isolated": int(len(isolated))}
    if len(isolated) == 0:
        return E, stats

    img_gpu = _to_gpu_normalized(img_feats, device, use_half)
    txt_gpu = _to_gpu_normalized(txt_feats, device, use_half)
    q_img = img_gpu[torch.from_numpy(isolated).to(device)]
    q_txt = txt_gpu[torch.from_numpy(isolated).to(device)]

    nn_img = _top1_indices_torch(q_img, img_gpu, isolated)
    nn_txt = _top1_indices_torch(q_txt, txt_gpu, isolated)

    # Pick the modality whose top-1 neighbour is more similar for each isolated node.
    sim_img = (q_img.float() * img_gpu[torch.from_numpy(nn_img).to(device)].float()).sum(1).cpu().numpy()
    sim_txt = (q_txt.float() * txt_gpu[torch.from_numpy(nn_txt).to(device)].float()).sum(1).cpu().numpy()
    chosen = np.where(sim_img >= sim_txt, nn_img, nn_txt)

    add_rows = np.concatenate([isolated, chosen])
    add_cols = np.concatenate([chosen, isolated])
    add = coo_matrix(
        (np.ones(len(add_rows), dtype=np.float32), (add_rows, add_cols)),
        shape=E.shape,
    )
    E_fixed = (E + add).tocsr()
    E_fixed.data[:] = 1.0
    return E_fixed, stats


def ensure_connected(
    E: csr_matrix,
    img_feats: np.ndarray,
    txt_feats: np.ndarray,
    alpha: float = 0.5,
    device: str = "cuda",  # noqa: ARG001 — kept for signature parity; runs on CPU
    use_half: bool = True,  # noqa: ARG001
) -> Tuple[csr_matrix, Dict[str, int]]:
    """
    Make E a single connected component with minimal, content-aware bridge edges.

    ``img_feats``/``txt_feats`` must already be L2-normalized — the sole caller
    (``build_buddy_graphs``) always passes its own ``img_n``/``txt_n``, so
    re-normalizing here would be a second full-size copy of dead-redundant work
    (harmless at N~150k, ~6GB of pure waste per call at N~3M).

    A disconnected graph's Laplacian has one near-zero eigenvalue per connected
    component, so a low-dimensional spectral embedding is consumed by component
    indicators and carries no within-graph structure. ``ensure_min_degree`` fixes
    isolated *nodes* but not disconnected *components*.

    We pick a medoid per component (node nearest the component centroid in the
    mix-weighted concat feature ``[√α·img, √(1-α)·txt]`` — whose cosine equals
    ``α·cos_img + (1-α)·cos_txt``, matching the pipeline's mixed similarity), build a
    minimum spanning tree over the medoids, and add the C-1 MST edges to E
    (binary, symmetric). Bridge distances are filled in downstream by
    ``sparse_cosine_distance`` and are naturally weak (cross-component pairs are far).

    No-op when E is already connected. Returns (E_connected, stats).
    """
    E = E.tocsr()
    n_comp, labels = connected_components(E, directed=False, return_labels=True)
    stats = {"n_components": int(n_comp), "bridges_added": 0}
    if n_comp <= 1:
        return E, stats

    # mix-weighted concat feature; cos(concat) = α·cos_img + (1-α)·cos_txt
    # np.sqrt(alpha) is a numpy float64 scalar; under NEP 50 promotion rules
    # (numpy>=2.0) multiplying a float32 array by it silently upcasts to float64,
    # transiently doubling — then, across two terms plus the concat output, nearly
    # quadrupling — memory here. At N~3M that's a ~51GB transient spike (fine at
    # N~150k, ~2GB). Force float32 scalars so no promotion happens.
    sqrt_a = np.float32(np.sqrt(alpha))
    sqrt_1ma = np.float32(np.sqrt(1.0 - alpha))
    concat = np.concatenate([sqrt_a * img_feats, sqrt_1ma * txt_feats], axis=1)

    n = E.shape[0]
    onehot = csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), labels)), shape=(n, n_comp)
    )
    sums = onehot.T @ concat                       # (C, D) component feature sums
    counts = np.asarray(onehot.sum(0)).ravel()     # (C,)
    centroids = sums / counts[:, None]

    # medoid = node within each component most aligned with its centroid
    score = np.einsum("nd,nd->n", concat, centroids[labels])
    medoids = np.array(
        [np.where(labels == c)[0][np.argmax(score[labels == c])] for c in range(n_comp)]
    )

    # MST over medoids in concat-cosine distance (+ε so no real edge is a dropped 0)
    med = concat[medoids]
    dist = (1.0 - med @ med.T) + 1e-6
    np.fill_diagonal(dist, 0.0)
    mst = minimum_spanning_tree(dist).tocoo()
    ea, eb = medoids[mst.row], medoids[mst.col]

    add_rows = np.concatenate([ea, eb])
    add_cols = np.concatenate([eb, ea])
    add = coo_matrix(
        (np.ones(len(add_rows), dtype=np.float32), (add_rows, add_cols)), shape=E.shape
    )
    E_conn = (E + add).tocsr()
    E_conn.data[:] = 1.0
    stats["bridges_added"] = int(len(ea))
    return E_conn, stats


# ── Distances (Steps 3–4) ────────────────────────────────────────────────────


def sparse_cosine_distance(
    feats: np.ndarray, E: csr_matrix, batch_size: int = 1_000_000
) -> csr_matrix:
    """
    Cosine distance (1 - cosine similarity) on the existing edges of E only.

    feats: (N, D) float32, L2-normalised. Returns a sparse matrix with E's sparsity.

    Batched over edges: ``feats[rows]``/``feats[cols]`` gather one full feature row
    per edge endpoint, materializing an (nnz, D) array. Fine at nnz~150k (~0.3GB),
    but at nnz~54M (a real union-graph size at N~3M) that's a single ~110GB
    allocation. Chunking bounds peak memory to batch_size rows regardless of nnz.
    """
    E_coo = E.tocoo()
    rows, cols = E_coo.row, E_coo.col
    nnz = len(rows)
    sim = np.empty(nnz, dtype=np.float32)
    for start in range(0, nnz, batch_size):
        end = min(start + batch_size, nnz)
        sim[start:end] = np.einsum(
            "nd,nd->n", feats[rows[start:end]], feats[cols[start:end]]
        )
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    return csr_matrix((dist, (rows, cols)), shape=E.shape)


def rank_normalise_sparse(D: csr_matrix) -> csr_matrix:
    """Replace each stored value with its global rank / nnz, so values are in (0, 1]."""
    D = D.tocoo()
    order = np.argsort(D.data)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(D.data) + 1)
    normed = ranks.astype(np.float32) / len(D.data)
    return csr_matrix((normed, (D.row, D.col)), shape=D.shape)


def mix_distances(D_img_n: csr_matrix, D_txt_n: csr_matrix, alpha: float) -> csr_matrix:
    """Convex combination of the two rank-normalised distance matrices."""
    return (alpha * D_img_n + (1.0 - alpha) * D_txt_n).tocsr()


def mix_distances_typed(
    D_img_n: csr_matrix, D_txt_n: csr_matrix, A_img: csr_matrix, A_txt: csr_matrix,
    alpha: float,
) -> csr_matrix:
    """
    Modality-provenance-aware distance mixing. The fixed-alpha mix_distances() blends
    BOTH modalities' distance on EVERY edge of E, regardless of which modality(ies)
    originally justified that edge -- a diagnostic (src/test/20260824_buddy_graph_disagreement/)
    found this collapses ~98% of a real buddy graph's edges (single-modality-only) from a
    good rank (median 0.2-0.3) to statistical noise (median ~0.50) on real RedCaps data.

    This function instead uses each edge's OWN supporting modality's rank-normalised
    distance alone for edges supported by only one modality's mutual-kNN graph, and the
    existing fixed-alpha blend for edges supported by BOTH (no disagreement to correct)
    or by NEITHER (added by ensure_min_degree/ensure_connected -- not owned by either
    modality, so there is no single supporting distance to prefer).

    D_img_n, D_txt_n: rank-normalised distances on E's edges (same sparsity as each
        other, i.e. both built via sparse_cosine_distance(feats, E) then
        rank_normalise_sparse -- E's edges, not A_img's or A_txt's).
    A_img, A_txt: the ORIGINAL per-modality mutual-kNN graphs (pre-union, pre-repair) --
        used only to classify each edge of E, not to source any distance values.
    """
    N = D_img_n.shape[0]
    coo = D_img_n.tocoo()
    rows, cols = coo.row, coo.col
    d_img = coo.data
    # Index-based (not position-based) lookup -- do not assume D_txt_n's internal
    # storage order matches D_img_n's; scipy does not guarantee this across independently
    # rank-normalised matrices even when both share the same sparsity pattern.
    d_txt = np.asarray(D_txt_n.tocsr()[rows, cols]).ravel()

    def _keys(A: csr_matrix) -> np.ndarray:
        A_coo = A.tocoo()
        mask = A_coo.data != 0
        k = A_coo.row[mask].astype(np.int64) * N + A_coo.col[mask].astype(np.int64)
        k.sort()
        return k

    keys = rows.astype(np.int64) * N + cols.astype(np.int64)
    in_img = np.isin(keys, _keys(A_img))
    in_txt = np.isin(keys, _keys(A_txt))
    img_only = in_img & ~in_txt
    txt_only = ~in_img & in_txt

    mixed = alpha * d_img + (1.0 - alpha) * d_txt  # default: "both" and "repair" edges
    mixed = np.where(img_only, d_img, mixed)
    mixed = np.where(txt_only, d_txt, mixed)

    return csr_matrix((mixed, (rows, cols)), shape=D_img_n.shape)
