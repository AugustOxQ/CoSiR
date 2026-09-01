"""Regularization functions used by LabelContrastiveLoss_enhance."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from src.conditional_buddy.buddy_graph import mutual_knn
from src.conditional_buddy.compute_buddies import (
    EDGE_TYPE_BOTH,
    EDGE_TYPE_IMG_ONLY,
    EDGE_TYPE_REPAIR,
    EDGE_TYPE_TXT_ONLY,
)


_BUDDY_EDGE_TYPE_CODES = {
    "img_only": int(EDGE_TYPE_IMG_ONLY),
    "txt_only": int(EDGE_TYPE_TXT_ONLY),
    "both": int(EDGE_TYPE_BOTH),
    "repair": int(EDGE_TYPE_REPAIR),
}
_DEFAULT_TYPE_WEIGHTS = {"img_only": 1.0, "txt_only": 1.0, "both": 1.0, "repair": 0.0}


def boundary_penalty(embeddings, radius=1.0, alpha=0.1):
    """Penalise embeddings whose L2 norm exceeds `radius`."""
    norms = torch.norm(embeddings, p=2, dim=1)
    penalty = torch.where(
        norms > radius, (norms - radius) ** 2, torch.zeros_like(norms)
    )
    return alpha * torch.mean(penalty)


def manifold_smoothness_loss_sparse(
    conditions, text_emb, conditional_text_pos, k=3, model=None, alpha=0.1
):
    """Penalise inconsistent modulation between k-nearest neighbours in condition space."""
    batch_size = len(conditions)

    dist_matrix = torch.cdist(conditions, conditions)

    text_emb_normalized = F.normalize(text_emb, p=2, dim=1)

    mask = torch.eye(batch_size, device=dist_matrix.device).bool()
    dist_matrix = dist_matrix.masked_fill(mask, float("inf"))

    _, neighbor_indices = torch.topk(dist_matrix, k, largest=False, dim=1)

    device = conditions.device
    random_neighbor_idx = torch.randint(0, k, (batch_size,), device=device)
    selected_neighbors = neighbor_indices[torch.arange(batch_size, device=device), random_neighbor_idx]

    neighbor_conditions = conditions[selected_neighbors]
    conditional_text_from_neighbor = model.combine(text_emb, None, neighbor_conditions)

    delta_current = conditional_text_pos - text_emb_normalized
    delta_neighbor = conditional_text_from_neighbor - text_emb_normalized

    smoothness = F.cosine_similarity(delta_current, delta_neighbor, dim=-1)

    distances = dist_matrix.gather(1, selected_neighbors.unsqueeze(1)).squeeze(1)
    distances = torch.clamp(distances, min=1e-8, max=10.0)
    weights = torch.exp(-distances + 1e-8)

    L_smooth_weighted = ((1 - smoothness) * weights).mean()

    return alpha * L_smooth_weighted


def build_neighbor_csr(edge_index: torch.Tensor, num_nodes: int):
    """Build a symmetric CSR neighbour structure from an undirected edge list.

    edge_index: LongTensor [2, M], one direction per edge (endpoints in any order).
    Returns (indptr [num_nodes+1], indices [2M]) LongTensors on edge_index.device.
    Self-loops are dropped; the graph is symmetrised (each edge stored both ways).
    """
    device = edge_index.device
    if edge_index.numel() == 0:
        indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        indices = torch.empty(0, dtype=torch.long, device=device)
        return indptr, indices
    src = torch.cat([edge_index[0], edge_index[1]])
    dst = torch.cat([edge_index[1], edge_index[0]])
    keep = src != dst
    src, dst = src[keep], dst[keep]
    order = torch.argsort(src)
    src, dst = src[order], dst[order]
    counts = torch.bincount(src, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    return indptr, dst.contiguous()


def _build_weighted_neighbor_csr(
    edge_index: torch.Tensor, edge_weights: torch.Tensor, num_nodes: int,
):
    """Build a symmetric CSR and setup-time CDF metadata for weighted draws."""
    device = edge_index.device
    if edge_index.numel() == 0:
        indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        empty = torch.empty(0, dtype=torch.long, device=device)
        return indptr, empty, torch.empty(0, device=device), torch.zeros(num_nodes, device=device)
    src = torch.cat([edge_index[0], edge_index[1]])
    dst = torch.cat([edge_index[1], edge_index[0]])
    weights = torch.cat([edge_weights, edge_weights])
    keep = src != dst
    src, dst, weights = src[keep], dst[keep], weights[keep]
    order = torch.argsort(src)
    src, dst, weights = src[order], dst[order], weights[order]
    counts = torch.bincount(src, minlength=num_nodes)
    indptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    edge_cdf = torch.cumsum(weights, dim=0)
    row_weight_sums = torch.zeros(num_nodes, dtype=weights.dtype, device=device)
    starts, ends = indptr[:-1], indptr[1:]
    nonempty = ends > starts
    end_values = edge_cdf[ends[nonempty] - 1]
    start_values = torch.zeros_like(end_values)
    has_prior = starts[nonempty] > 0
    start_values[has_prior] = edge_cdf[starts[nonempty][has_prior] - 1]
    row_weight_sums[nonempty] = end_values - start_values
    return indptr, dst.contiguous(), edge_cdf, row_weight_sums


def build_semantic_buddy_csr(
    edge_index: torch.Tensor,
    edge_types,
    num_nodes: int,
    *,
    exclude_repair: bool = False,
    type_weights: Optional[dict] = None,
):
    """Build the optional semantic buddy graph and weighted-sampling metadata.

    ``edge_types`` is the uint8 provenance array aligned with ``edge_index``:
    ``img_only``, ``txt_only``, ``both``, and ``repair``.  If either option is
    active, missing provenance fails loudly.  ``type_weights`` accepts any
    subset of those string keys; unspecified semantic types default to 1.0 and
    ``repair`` defaults to 0.0.  A ``None`` mapping preserves uniform sampling.

    Returns ``(indptr, indices, edge_cdf, row_weight_sums)``.  The latter two
    are ``None`` when no type weights were requested, retaining the old loss
    sampling path exactly.  Otherwise they are CSR-aligned setup-time metadata
    for vectorized weighted-with-replacement sampling.
    """
    type_aware = exclude_repair or type_weights is not None
    if type_aware and edge_types is None:
        raise ValueError(
            "Type-aware buddy supervision requires buddy_edge_types.npy, but the "
            "loaded buddy template has no edge provenance. Rebuild the template "
            "with Experiment 15.1 provenance persistence first."
        )
    if edge_types is None:
        indptr, indices = build_neighbor_csr(edge_index, num_nodes)
        return indptr, indices, None, None

    types = torch.as_tensor(edge_types, dtype=torch.uint8, device=edge_index.device).reshape(-1)
    if types.numel() != edge_index.shape[1]:
        raise ValueError(
            "buddy_edge_types.npy must have one entry per buddy_edges.npy column "
            f"({types.numel()} types for {edge_index.shape[1]} edges)."
        )
    if torch.any((types < min(_BUDDY_EDGE_TYPE_CODES.values())) | (types > max(_BUDDY_EDGE_TYPE_CODES.values()))):
        raise ValueError("buddy_edge_types.npy contains an unknown edge-type code.")

    if exclude_repair:
        keep = types != int(EDGE_TYPE_REPAIR)
        edge_index, types = edge_index[:, keep], types[keep]

    if type_weights is None:
        indptr, indices = build_neighbor_csr(edge_index, num_nodes)
        return indptr, indices, None, None

    unknown = set(type_weights) - set(_BUDDY_EDGE_TYPE_CODES)
    if unknown:
        raise ValueError(f"Unknown buddy_type_weights keys: {sorted(unknown)}")
    weights_by_name = dict(_DEFAULT_TYPE_WEIGHTS)
    for name, value in type_weights.items():
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"buddy_type_weights[{name!r}] must be a finite non-negative number.")
        weights_by_name[name] = value
    edge_weights = torch.empty(types.shape[0], dtype=torch.float32, device=edge_index.device)
    for name, code in _BUDDY_EDGE_TYPE_CODES.items():
        edge_weights[types == code] = weights_by_name[name]
    return _build_weighted_neighbor_csr(edge_index, edge_weights, num_nodes)


def sample_buddy_neighbors(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    num_samples: int,
    *,
    edge_cdf: Optional[torch.Tensor] = None,
    row_weight_sums: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """Return active input rows and sampled CSR neighbours with replacement.

    Without CDF metadata this is intentionally the exact historical uniform
    ``rand * degree`` implementation.  With setup-time CDF metadata, one
    global ``searchsorted`` samples each anchor's local CSR row by its weights.
    """
    deg = indptr[anchor_positions + 1] - indptr[anchor_positions]
    mask = deg > 0
    if (edge_cdf is None) != (row_weight_sums is None):
        raise ValueError("edge_cdf and row_weight_sums must be supplied together.")
    if row_weight_sums is not None:
        mask = mask & (row_weight_sums[anchor_positions] > 0)
    active = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if active.numel() == 0:
        return active, torch.empty((0, num_samples), dtype=torch.long, device=indices.device)

    anchors = anchor_positions[active]
    starts = indptr[anchors].unsqueeze(1)
    A = anchors.shape[0]
    rand = torch.rand(A, num_samples, device=indices.device, generator=generator)
    if edge_cdf is None:
        degm = deg[active].unsqueeze(1)
        offsets = torch.clamp((rand * degm).long(), max=degm - 1)
        return active, indices[starts + offsets]

    totals = row_weight_sums[anchors].unsqueeze(1)
    bases = torch.zeros_like(totals)
    has_prior = starts > 0
    bases[has_prior] = edge_cdf[(starts[has_prior] - 1)]
    flat_positions = torch.searchsorted(edge_cdf, (bases + rand * totals).reshape(-1), right=True)
    return active, indices[flat_positions.reshape(A, num_samples)]


def buddy_graph_smoothness_loss(
    embeddings: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    num_samples: int = 4,
    edge_cdf: Optional[torch.Tensor] = None,
    row_weight_sums: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Mean squared distance between batch anchors and sampled buddies along E.

    This is the Laplacian-Eigenmaps energy that produced the buddy init, kept alive
    during training. Anchors with no neighbour are skipped; if none have a neighbour
    the term is exactly 0 (with no gradient).

    embeddings:       [N, D] differentiable table (embedding_manager.embeddings).
    indptr, indices:  CSR neighbour structure from build_neighbor_csr.
    anchor_positions: [A] long, table positions of the batch's samples.
    num_samples:      buddies sampled per anchor (with replacement).
    """
    device = embeddings.device
    active, nbr_pos = sample_buddy_neighbors(
        indptr, indices, anchor_positions, num_samples,
        edge_cdf=edge_cdf, row_weight_sums=row_weight_sums, generator=generator,
    )
    if active.numel() == 0:
        return embeddings.sum() * 0.0
    anchors = anchor_positions[active]
    z_a = embeddings[anchors].unsqueeze(1)                         # [A', 1, D]
    z_n = embeddings[nbr_pos]                                      # [A', num_samples, D]
    return (z_a - z_n).pow(2).sum(-1).mean()


def buddy_contrastive_loss(
    comb_emb: torch.Tensor,
    anchor_positions: torch.Tensor,
    other_feat_table: torch.Tensor,
    project_other,
    other_emb_neg: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    num_pos: int = 4,
    temperature: float = 0.07,
    edge_cdf: Optional[torch.Tensor] = None,
    row_weight_sums: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """Multi-positive InfoNCE pulling anchors toward their buddies along E.

    For each batch anchor, sample up to ``num_pos`` buddies from the CSR graph,
    look up their (frozen) other-side features, project them into retrieval space
    with ``project_other``, and treat them as positives. Negatives are the batch's
    own ``other_emb_neg`` rows, with each anchor's own row masked out. Anchors with
    no buddy are skipped; if none have a buddy the loss is exactly 0 (grad-safe).

    comb_emb:         [B, Dp]   anchor combined features (differentiable).
    anchor_positions: [B]       z-table positions of the batch (== batch_indices).
    other_feat_table: [N, Dfeat] frozen other-side features in z-table order.
    project_other:    callable [*, Dfeat] -> [*, Dp].
    other_emb_neg:    [B, Dp]   in-batch negatives (the batch's other_emb).
    indptr, indices:  CSR neighbour structure from build_neighbor_csr (graph E).
    num_pos:          buddies sampled per anchor (with replacement).
    temperature:      softmax temperature.
    Returns (loss scalar, alignment scalar detached = mean cos(anchor, buddy)).
    """
    device = comb_emb.device
    active, buddy_pos = sample_buddy_neighbors(
        indptr, indices, anchor_positions, num_pos,
        edge_cdf=edge_cdf, row_weight_sums=row_weight_sums, generator=generator,
    )
    if active.numel() == 0:
        return comb_emb.sum() * 0.0, torch.zeros((), device=device)

    A = active.shape[0]

    q = F.normalize(comb_emb[active], dim=-1)                    # [A, Dp]
    pos = F.normalize(project_other(other_feat_table[buddy_pos]), dim=-1)  # [A, num_pos, Dp]
    neg = F.normalize(other_emb_neg, dim=-1)                     # [B, Dp]

    logits_pos = torch.einsum("ad,akd->ak", q, pos) / temperature   # [A, num_pos]
    logits_neg = (q @ neg.t()) / temperature                        # [A, B]
    self_mask = torch.zeros(A, neg.shape[0], dtype=torch.bool, device=device)
    self_mask[torch.arange(A, device=device), active] = True
    logits_neg = logits_neg.masked_fill(self_mask, float("-inf"))

    logits_all = torch.cat([logits_pos, logits_neg], dim=1)         # [A, num_pos + B]
    log_z = torch.logsumexp(logits_all, dim=1, keepdim=True)        # [A, 1]
    log_prob_pos = logits_pos - log_z                              # [A, num_pos]
    loss = -log_prob_pos.mean(dim=1).mean()
    alignment = (logits_pos * temperature).mean().detach()
    return loss, alignment


def reorder_features_to_z(feat: torch.Tensor, feat_ids, z_ids) -> torch.Tensor:
    """Reindex a feature-store tensor into embedding-manager (z-table) order.

    feat:     [M, Dfeat] features in feature-store order.
    feat_ids: list[int] sample ids in feature-store order (len M).
    z_ids:    list[int] sample ids in z-table order (embedding_manager.sample_ids).
    Returns [len(z_ids), Dfeat] where row p is the feature for sample z_ids[p].
    """
    fpos = {int(sid): i for i, sid in enumerate(feat_ids)}
    order = torch.tensor([fpos[int(sid)] for sid in z_ids], dtype=torch.long)
    return feat[order].contiguous()


def _undirected_edge_set(edge_index: torch.Tensor):
    """Set of frozenset{u, v} for each column; self-loops dropped."""
    s = set()
    ei = edge_index.detach().cpu()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u != v:
            s.add(frozenset((u, v)))
    return s


def edge_jaccard(ei_a: torch.Tensor, ei_b: torch.Tensor) -> float:
    """Undirected Jaccard between two edge_index tensors (both empty -> 1.0)."""
    sa, sb = _undirected_edge_set(ei_a), _undirected_edge_set(ei_b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def refresh_buddy_graph(
    model,
    combine_feat_table: torch.Tensor,
    z_table: torch.Tensor,
    clip_edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 30,
    blend: float = 1.0,
    chunk: int = 4096,
    knn_device: str = "cuda",
    generator: Optional[torch.Generator] = None,
):
    """Rebuild the buddy CSR from the model's current combined features.

    Under no_grad: run a chunked full-N ``model.combine`` pass to get
    ``comb_all`` [N, Dp], build a comb-space mutual-KNN, keep a ``blend``
    fraction of its edges, union them with the always-kept frozen CLIP edges,
    and rebuild the CSR. Returns (indptr, indices, comb_edges[2, Mc], stats).
    ``blend == 0`` never calls mutual_knn and reproduces the CLIP-only CSR.
    """
    device = clip_edge_index.device
    with torch.no_grad():
        if blend > 0:
            comb_all = compute_comb_all_eval(model, combine_feat_table, z_table, chunk=chunk)

            A = mutual_knn(comb_all.float().cpu().numpy(), k, device=knn_device)  # scipy csr
            coo = A.tocoo()
            keep_dir = coo.row < coo.col  # one direction per edge, drop self-loops
            src = torch.from_numpy(coo.row[keep_dir].astype(np.int64))
            dst = torch.from_numpy(coo.col[keep_dir].astype(np.int64))
            comb_edges = torch.stack([src, dst], dim=0).to(device)  # [2, Mc]

            m = comb_edges.shape[1]
            if blend < 1.0 and m > 0:
                keep = int(round(blend * m))
                perm = torch.randperm(m, generator=generator)[:keep].to(device)
                comb_edges = comb_edges[:, perm]
        else:
            comb_edges = torch.empty(2, 0, dtype=torch.long, device=device)

        edge_index = torch.cat([clip_edge_index, comb_edges], dim=1)
        indptr, indices = build_neighbor_csr(edge_index, num_nodes)

        clip_set = _undirected_edge_set(clip_edge_index)
        comb_set = _undirected_edge_set(comb_edges)
        new_frac = len(comb_set - clip_set) / max(len(comb_set), 1)
        stats = {
            "graph_n_comb_edges": float(comb_edges.shape[1]),
            "graph_new_edge_frac": float(new_frac),
            "graph_avg_degree": float(indptr[-1].item()) / max(num_nodes, 1),
        }
    return indptr, indices, comb_edges, stats


def compute_comb_all_eval(model, combine_feat_table, z_table, chunk: int = 4096):
    """Combined features for all N under eval-mode + no_grad.

    Eval mode disables dropout so the representation is the deterministic inference
    one (what retrieval sees); ``no_grad`` keeps it off the autograd tape. The model's
    train/eval state is restored on exit. Returns [N, Dp] detached, on the same
    device as combine_feat_table (GPU, in practice).

    Writes each chunk directly into a pre-allocated [N, Dp] output tensor instead of
    appending to a list and torch.cat-ing at the end: the list-then-cat pattern holds
    the *entire* table twice at the moment cat allocates its result (once piecemeal
    across the list, once contiguous) — a transient GPU memory spike large enough to
    OOM at N~3M. Pre-allocating once avoids that doubling entirely, with no need to
    move off GPU. (An earlier version of this function moved chunks to CPU instead —
    that fixed the memory spike but made its only consumer, buddy_knn_preservation,
    silently ~1000x slower: that function's similarity matmul is O(N^2 * D), and a
    CPU matmul is ~50-100x slower per-FLOP than the GPU it was validated on, on ~44x
    more work at N=1M vs the N~150k it was designed for. No exception, no OOM — just
    a computation that looked identical to a hang for hours. Staying on GPU keeps
    both the memory fix and the speed the algorithm actually needs.)
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            n = combine_feat_table.shape[0]
            out = None
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                c = model.combine(combine_feat_table[s:e], None, z_table[s:e].detach()).detach()
                if out is None:
                    out = torch.empty((n,) + c.shape[1:], device=c.device, dtype=c.dtype)
                out[s:e] = c
            return out
    finally:
        model.train(was_training)


def buddy_knn_preservation(
    comb_all: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    k: int = 10,
    chunk: int = None,
) -> float:
    """Fraction of each node's CLIP-graph buddies that stay in its top-k comb-space NN.

    For every node with at least one buddy, take its ``k`` nearest neighbours by cosine
    in ``comb_all`` (self excluded) and measure the fraction of its buddies (from the
    symmetric CLIP CSR ``indptr``/``indices``) that land in that top-k. Averaged over
    nodes with degree > 0 (isolated nodes are skipped). Higher = training kept the buddy
    neighbourhood alive where retrieval lives. Pure diagnostic; no gradient.

    chunk: rows of the [chunk, N] similarity matrix computed per iteration. Default
    (None) scales inversely with N, capped at the original fixed default (2048) so
    behavior at small/medium N is unchanged: at N~150k, 2048 rows is ~1.2GB fp32 —
    fine. At N~3M it's ~25GB in one allocation, which alone exceeds the GPU when
    training has already committed a large chunk of it — torch.OutOfMemoryError,
    even though this loop was already chunked, just not aggressively enough at this
    scale. Budgeted to keep each chunk's matmul under ~1.5GB regardless of N.

    indptr/indices are moved to comb_all's device once, up front, in case they ever
    differ (e.g. a caller passing a CPU comb_all) — cheap and a no-op in the common
    case where both are already on the same GPU.

    The per-node fraction is computed in edge-space (vectorized), not with a Python
    loop over each node: at N~150k a `for` loop over every node was slow but
    tolerable; at N~3M it's 3.1 million individual Python-level iterations —
    tens of minutes of pure interpreter/dispatch overhead regardless of device,
    fast enough to look "stuck" (no error, no progress, GPU idle) rather than
    fast enough to look "slow." CSR rows [s, e) are contiguous in `indices` by
    construction (see build_neighbor_csr), so each chunk's whole buddy-edge list
    is one slice; every edge is checked against its owning node's top-k in one
    batched comparison, then summed per node with scatter_add.
    """
    device = comb_all.device
    N = comb_all.shape[0]
    q = F.normalize(comb_all, dim=-1)
    if chunk is None:
        chunk = max(1, min(2048, int(1.5e9 // (N * q.element_size()))))
    indptr = indptr.to(device)
    indices = indices.to(device)
    deg = indptr[1:] - indptr[:-1]
    total, count = 0.0, 0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = e - s
        sims = q[s:e] @ q.t()                                    # [b, N]
        sims[torch.arange(b, device=device), torch.arange(s, e, device=device)] = float("-inf")
        topk = sims.topk(k, dim=1).indices                      # [b, k]

        chunk_deg = deg[s:e]                                     # [b]
        has_buddies = chunk_deg > 0
        if not bool(has_buddies.any()):
            continue

        edge_lo = int(indptr[s].item())
        edge_hi = int(indptr[e].item())
        local_indices = indices[edge_lo:edge_hi]                 # [n_edges], CSR-contiguous over [s,e)
        row_of_edge = torch.repeat_interleave(
            torch.arange(b, device=device), chunk_deg
        )                                                          # [n_edges] -> local row in [0, b)

        edge_topk = topk[row_of_edge]                            # [n_edges, k]
        matched = (edge_topk == local_indices.unsqueeze(1)).any(dim=1).float()  # [n_edges]

        preserved = torch.zeros(b, device=device, dtype=torch.float32)
        preserved.scatter_add_(0, row_of_edge, matched)
        frac = preserved[has_buddies] / chunk_deg[has_buddies].float()
        total += frac.sum().item()
        count += int(has_buddies.sum().item())
    return total / max(count, 1)
