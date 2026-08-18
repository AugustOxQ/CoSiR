"""
End-to-end conditional-buddies initialization (Steps 1–6).

`compute_buddy_init` is a pure function over in-memory feature arrays: it builds
the cross-modal buddy graph, embeds it, normalises the result, and (optionally)
reorders rows to a target sample-id order. It has no FeatureManager dependency,
which keeps it easy to unit-test.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csgraph, csr_matrix

from .buddy_graph import (
    ensure_connected,
    ensure_min_degree,
    mix_distances,
    mutual_knn,
    rank_normalise_sparse,
    sparse_cosine_distance,
    union_graph,
)
from .embedding_methods import normalise_embedding, spectral_embedding


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows in place (no callers need the pre-normalization values back —
    a second full-size copy here is fine at N~150k but adds ~6GB of dead RAM per call
    at N~3M, tripling peak usage across the shard-list -> raw -> normalized chain)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x


def build_buddy_graphs(
    img_feats: np.ndarray,
    txt_feats: np.ndarray,
    K: int = 30,
    alpha: float = 0.5,
    device: str = "cuda",
    knn_batch_size: int = 1024,
    use_half: bool = True,
    connect_components: bool = True,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Return (A_img, A_txt, E) — per-modality mutual graphs and their fixed union."""
    img_n = _l2_normalize(img_feats)
    txt_n = _l2_normalize(txt_feats)

    A_img = mutual_knn(img_n, K, device, knn_batch_size, use_half=use_half)
    A_txt = mutual_knn(txt_n, K, device, knn_batch_size, use_half=use_half)
    print(
        f"[buddies] Step 1: mutual KNN (K={K}) — "
        f"A_img nnz={A_img.nnz:,} (avg deg {A_img.nnz / A_img.shape[0]:.2f}), "
        f"A_txt nnz={A_txt.nnz:,} (avg deg {A_txt.nnz / A_txt.shape[0]:.2f})"
    )

    E = union_graph(A_img, A_txt)
    E, deg_stats = ensure_min_degree(E, img_n, txt_n, device, use_half=use_half)
    n_comp = csgraph.connected_components(E, directed=False, return_labels=False)
    print(
        f"[buddies] Step 2: union graph E — nnz={E.nnz:,}, "
        f"avg degree {E.nnz / E.shape[0]:.2f}, isolated fixed={deg_stats['num_isolated']}, "
        f"connected components={n_comp}"
    )

    if connect_components:
        E, conn_stats = ensure_connected(E, img_n, txt_n, alpha=alpha, device=device,
                                         use_half=use_half)
        n_comp = csgraph.connected_components(E, directed=False, return_labels=False)
        print(
            f"[buddies] Step 2b: connected E — added {conn_stats['bridges_added']} "
            f"bridge(s) over {conn_stats['n_components']} components → "
            f"connected components={n_comp}"
        )
    return A_img, A_txt, E


def compute_buddy_init(
    img_feats: np.ndarray,
    txt_feats: np.ndarray,
    n_dim: int,
    method: str = "spectral",
    K: int = 30,
    alpha: float = 0.5,
    device: str = "cuda",
    knn_batch_size: int = 1024,
    normalize_method: str = "rank",
    seed: int = 42,
    use_half: bool = True,
    eigen_solver: str = "auto",
    connect_components: bool = True,
    return_edges: bool = False,
    b_weight: float = 1.0,
    input_sample_ids: Optional[List[int]] = None,
    output_sample_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute conditional-buddies init vectors from CLIP features.

    img_feats, txt_feats: (N, D) float32, same row order.
    method:            embedding method ('spectral'; only spectral is supported).
    normalize_method:  'rank' (default) or 'zscore' — see normalise_embedding.
    eigen_solver:      'auto' (default; arpack for small N, amg for large N),
                       or force 'arpack' | 'amg' | 'lobpcg' — see spectral_embedding.
    connect_components: bridge disconnected components of E with a minimal MST of
                       cross-component edges (default True; no-op when E is already
                       connected). Required for a usable spectral init on fragmented
                       graphs — see ensure_connected.
    return_edges:      if True, return (emb, edges); if False (default) return emb only.
    Returns: (N, n_dim) float32 in ~[-1, 1], or if return_edges=True, a tuple (emb, edges)
             where edges is np.int64 [2, M] undirected edge list (i < j), with endpoints
             expressed as table positions in output_sample_ids order if reordering is
             requested, else input-row order.

    Rows are returned in input order unless both ``input_sample_ids`` and
    ``output_sample_ids`` are given, in which case rows are reordered so row i
    corresponds to output_sample_ids[i] (the framework's sample-id order).
    """
    img_n = _l2_normalize(img_feats)
    txt_n = _l2_normalize(txt_feats)

    A_img, A_txt, E = build_buddy_graphs(
        img_n, txt_n, K=K, alpha=alpha, device=device, knn_batch_size=knn_batch_size,
        use_half=use_half, connect_components=connect_components,
    )

    # B-lean: strict-intersection buddies to upweight in the spectral affinity.
    b_edges = None
    if b_weight != 1.0:
        b_edges = A_img.multiply(A_txt).tocsr()
        b_edges.data[:] = 1.0
        b_edges.eliminate_zeros()
        print(f"[buddies] B-lean: b_weight={b_weight}, strict-B edges nnz={b_edges.nnz:,} "
              f"(of E nnz={E.nnz:,})")

    # Step 3: sparse per-modality distances on E's edges
    D_img = sparse_cosine_distance(img_n, E)
    D_txt = sparse_cosine_distance(txt_n, E)

    # Step 4: rank-normalise and mix
    D_mixed = mix_distances(
        rank_normalise_sparse(D_img), rank_normalise_sparse(D_txt), alpha
    )
    print(f"[buddies] Step 4: mixed distance matrix nnz={D_mixed.nnz:,} (alpha={alpha})")

    # Step 5: embed
    if method != "spectral":
        raise ValueError(f"Unknown method '{method}'. Only 'spectral' is supported.")
    emb = spectral_embedding(D_mixed, n_dim, seed=seed, eigen_solver=eigen_solver,
                             b_edges=b_edges, b_weight=b_weight)
    print(f"[buddies] Step 5: {method} embedding shape={emb.shape} "
          f"(eigen_solver={eigen_solver})")

    # Step 6: normalise (rank by default — robust to eigenvector localization)
    emb = normalise_embedding(emb, method=normalize_method)
    print(
        f"[buddies] Step 6: normalised init — shape={emb.shape}, "
        f"range=[{emb.min():.3f}, {emb.max():.3f}]"
    )

    if output_sample_ids is not None:
        if input_sample_ids is None:
            raise ValueError("output_sample_ids given but input_sample_ids is None.")
        pos = {sid: i for i, sid in enumerate(input_sample_ids)}
        reorder = [pos[sid] for sid in output_sample_ids]
        emb = emb[reorder]

    if not return_edges:
        return emb

    coo = E.tocoo()
    upper = coo.row < coo.col
    edges = np.stack([coo.row[upper], coo.col[upper]]).astype(np.int64)  # input-row positions
    if output_sample_ids is not None:
        inv = np.empty(len(reorder), dtype=np.int64)
        inv[np.asarray(reorder, dtype=np.int64)] = np.arange(len(reorder), dtype=np.int64)
        edges = inv[edges]  # remap input positions -> output positions
        edges = np.sort(edges, axis=0)  # re-enforce i < j after remap
    return emb, edges
