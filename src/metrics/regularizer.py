"""Regularization functions used by LabelContrastiveLoss_enhance."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from src.conditional_buddy.buddy_graph import mutual_knn


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


def buddy_graph_smoothness_loss(
    embeddings: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    num_samples: int = 4,
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
    deg = indptr[anchor_positions + 1] - indptr[anchor_positions]   # [A]
    mask = deg > 0
    if not torch.any(mask):
        return embeddings.sum() * 0.0
    anchors = anchor_positions[mask]
    deg = deg[mask].unsqueeze(1)                                    # [A', 1]
    starts = indptr[anchors].unsqueeze(1)                          # [A', 1]
    A = anchors.shape[0]
    rand = torch.rand(A, num_samples, device=device, generator=generator)
    offsets = torch.clamp((rand * deg).long(), max=deg - 1)        # [A', num_samples] in [0, deg)
    nbr_pos = indices[starts + offsets]                            # [A', num_samples] table positions
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
    deg = indptr[anchor_positions + 1] - indptr[anchor_positions]   # [B]
    mask = deg > 0
    if not torch.any(mask):
        return comb_emb.sum() * 0.0, torch.zeros((), device=device)

    active = torch.nonzero(mask, as_tuple=False).squeeze(1)        # [A] batch rows
    anchors = anchor_positions[active]                            # [A] z-positions
    degm = deg[mask].unsqueeze(1)                                # [A, 1]
    starts = indptr[anchors].unsqueeze(1)                        # [A, 1]
    A = anchors.shape[0]
    rand = torch.rand(A, num_pos, device=device, generator=generator)
    offsets = torch.clamp((rand * degm).long(), max=degm - 1)     # [A, num_pos] in [0, deg)
    buddy_pos = indices[starts + offsets]                        # [A, num_pos] z-positions

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
            was_training = model.training
            model.eval()
            try:
                combs = []
                for s in range(0, num_nodes, chunk):
                    e = min(s + chunk, num_nodes)
                    c = model.combine(combine_feat_table[s:e], None, z_table[s:e].detach())
                    combs.append(c.detach())
                comb_all = torch.cat(combs, dim=0)  # [N, Dp]
            finally:
                model.train(was_training)

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
