"""
Cross-modal mutual-KNN ("buddy") graph construction.

Steps 1–4 of the conditional-buddies initialization pipeline:
    1. mutual_knn          — per-modality mutual K-nearest-neighbour graph
    2. union_graph         — broad union graph for initialization
    3. sparse_cosine_distance / ensure_min_degree
    4. rank_normalise_sparse / mix_distances

All heavy nearest-neighbour search runs on the GPU via exact brute-force topk
(mathematically identical to faiss.IndexFlatIP). The search is isolated behind
``mutual_knn(..., backend=...)`` so an approximate cuvs backend can be swapped in
for very large N without touching callers.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


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


def mutual_knn(
    features: np.ndarray,
    K: int,
    device: str = "cuda",
    batch_size: int = 1024,
    backend: str = "torch",
    use_half: bool = True,
) -> csr_matrix:
    """
    Build a mutual KNN adjacency matrix.

    features: (N, D) float32 (L2-normalisation applied internally).
    Returns:  (N, N) sparse binary matrix A_mut where A_mut[i, j] = 1 iff j is in
              i's top-K AND i is in j's top-K.
    """
    if backend != "torch":
        raise NotImplementedError(
            f"mutual_knn backend '{backend}' not implemented; use 'torch'. "
            "(A cuvs ANN backend can be added here for million-scale N.)"
        )
    N = features.shape[0]
    feats_gpu = _to_gpu_normalized(features, device, use_half)
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
    img_n = img_feats / (np.linalg.norm(img_feats, axis=1, keepdims=True) + 1e-12)
    txt_n = txt_feats / (np.linalg.norm(txt_feats, axis=1, keepdims=True) + 1e-12)
    concat = np.concatenate(
        [np.sqrt(alpha) * img_n, np.sqrt(1.0 - alpha) * txt_n], axis=1
    ).astype(np.float32)

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


def sparse_cosine_distance(feats: np.ndarray, E: csr_matrix) -> csr_matrix:
    """
    Cosine distance (1 - cosine similarity) on the existing edges of E only.

    feats: (N, D) float32, L2-normalised. Returns a sparse matrix with E's sparsity.
    """
    E_coo = E.tocoo()
    rows, cols = E_coo.row, E_coo.col
    sim = np.einsum("nd,nd->n", feats[rows], feats[cols])
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
