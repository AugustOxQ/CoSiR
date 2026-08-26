"""
Cheap, no-training diagnostic for a specific buddy-graph construction concern: the union
graph E = A_img u A_txt is binarised immediately (union_graph, buddy_graph.py:207-211),
discarding which modality(ies) support each edge. compute_buddy_init then computes BOTH
D_img and D_txt on every edge of E and blends them with a fixed alpha regardless of an
edge's origin (mix_distances, buddy_graph.py:379-382) -- diluting a single-modality-only
edge's effective distance with the OTHER, disagreeing modality's (large) distance for that
same pair. A separate, related concern: a node bridging otherwise-unrelated neighbors via
two different modalities (img-only to one neighbor, txt-only to another) could induce
false transitivity in the downstream spectral embedding.

This module answers, on real cached features, with no training and no new graph
construction beyond what compute_buddy_init already does:
  1. What fraction of E's edges are img_only / txt_only / both / repair-added
     (ensure_min_degree / ensure_connected can add edges absent from BOTH A_img and A_txt)?
  2. How much does the fixed-alpha mix move a single-modality-only edge's rank away from
     its supporting modality's own (good) rank?
  3. How many nodes are "bridges" -- connected to at least one neighbor via an img_only
     edge AND at least one neighbor via a txt_only edge?

Usage
-----
  python diagnose_disagreement.py                  # full run against cached RedCaps-150k
  python diagnose_disagreement.py --scale 300k      # or 500k (reuses Experiment 9's diverse samples)

Requires: numpy, scipy, torch (for mutual_knn's GPU path; falls back to CPU).
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, triu
from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))


def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Same semantics as buddy_graph.rank_normalise_sparse: smallest value -> rank
    1/n (best/closest), largest -> rank n/n (worst/farthest)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks / len(x)


def diagnose(img_n: np.ndarray, txt_n: np.ndarray, A_img: csr_matrix, A_txt: csr_matrix,
             E: csr_matrix, alpha: float = 0.5):
    """
    Returns (typed, r_img, r_txt, r_mixed): typed is classify_edges's output; r_img/r_txt
    are E's edges' rank-normalised image/text cosine distances (same construction as
    compute_buddy_init's Step 3-4, computed directly here rather than round-tripping
    through sparse_cosine_distance/mix_distances, so the row/col order is guaranteed to
    match typed["keys"] exactly); r_mixed is the fixed-alpha blend actually used downstream.
    """
    N = img_n.shape[0]
    typed = classify_edges(A_img, A_txt, E, N)
    keys = typed["keys"]
    i = (keys // N).astype(np.int64)
    j = (keys % N).astype(np.int64)

    d_img = 1.0 - np.clip(np.einsum("nd,nd->n", img_n[i], img_n[j]), -1.0, 1.0)
    d_txt = 1.0 - np.clip(np.einsum("nd,nd->n", txt_n[i], txt_n[j]), -1.0, 1.0)

    r_img = rank_normalize(d_img)
    r_txt = rank_normalize(d_txt)
    r_mixed = alpha * r_img + (1.0 - alpha) * r_txt
    return typed, r_img, r_txt, r_mixed


SCALES = {
    "150k": (None, None),
    "300k": ("/data/SSD2/pre_extract/redcaps_300k_diverse/features",
             "/data/PDD/redcaps/redcaps_plus/redcaps_300k_diverse.json"),
    "500k": ("/data/SSD2/pre_extract/redcaps_500k_diverse/features",
             "/data/PDD/redcaps/redcaps_plus/redcaps_500k_diverse.json"),
}


def run(scale: str = "150k", K: int = 30, alpha: float = 0.5):
    import redcaps_buddy as rb
    from src.conditional_buddy.compute_buddies import build_buddy_graphs

    storage_dir, annotation_path = SCALES[scale]
    print(f"Loading RedCaps data ({scale})...")
    data = rb.load_data() if storage_dir is None else rb.load_data(
        storage_dir=storage_dir, annotation_path=annotation_path)
    N = data.n
    print(f"  {N:,} rows")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()

    print(f"Building buddy graphs (K={K}, alpha={alpha})...")
    A_img, A_txt, E = build_buddy_graphs(data.img, data.txt, K=K, alpha=alpha,
                                          device=device, use_half=use_half)

    typed, r_img, r_txt, r_mixed = diagnose(data.img, data.txt, A_img, A_txt, E, alpha=alpha)
    n_edges = len(typed["keys"])
    n_img_only = int(typed["img_only"].sum())
    n_txt_only = int(typed["txt_only"].sum())
    n_both = int(typed["both"].sum())
    n_repair = int(typed["repair"].sum())

    print(f"\n=== Edge-type breakdown ({n_edges:,} edges of E) ===")
    for name, n in (("img_only", n_img_only), ("txt_only", n_txt_only),
                     ("both", n_both), ("repair", n_repair)):
        print(f"  {name:>10}: {n:>10,}  ({100*n/n_edges:5.1f}%)")

    print(f"\n=== Dilution check (fixed alpha={alpha} blend vs. supporting-modality-only rank) ===")
    print("  Rank in [0,1]; 0 = best/closest of all E's edges in that modality, 1 = worst/farthest.")
    if n_img_only > 0:
        m = typed["img_only"]
        print(f"  img_only edges (n={n_img_only:,}): "
              f"median r_img={np.median(r_img[m]):.3f} (its own modality's rank) "
              f"-> median r_mixed={np.median(r_mixed[m]):.3f} "
              f"(diluted by median r_txt={np.median(r_txt[m]):.3f} of the disagreeing modality)")
    if n_txt_only > 0:
        m = typed["txt_only"]
        print(f"  txt_only edges (n={n_txt_only:,}): "
              f"median r_txt={np.median(r_txt[m]):.3f} (its own modality's rank) "
              f"-> median r_mixed={np.median(r_mixed[m]):.3f} "
              f"(diluted by median r_img={np.median(r_img[m]):.3f} of the disagreeing modality)")
    if n_both > 0:
        m = typed["both"]
        print(f"  both edges     (n={n_both:,}): median r_img={np.median(r_img[m]):.3f}, "
              f"median r_txt={np.median(r_txt[m]):.3f}, median r_mixed={np.median(r_mixed[m]):.3f} "
              f"(no disagreement expected here -- sanity baseline)")

    bstats = bridge_node_stats(typed, N)
    print(f"\n=== Bridge-node check ===")
    print(f"  {bstats['n_bridge_nodes']:,} / {N:,} nodes ({100*bstats['frac_bridge_nodes']:.2f}%) "
          f"have >=1 img_only edge AND >=1 txt_only edge (candidate bridges)")
    if bstats["n_bridge_nodes"] > 0:
        deg_total_bridge = (bstats["deg_img_only"][bstats["is_bridge"]]
                             + bstats["deg_txt_only"][bstats["is_bridge"]])
        print(f"  median (img_only + txt_only) degree among bridge nodes: {np.median(deg_total_bridge):.1f}")

    return typed, r_img, r_txt, r_mixed, bstats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES.keys()), default="150k")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    run(scale=args.scale, K=args.K, alpha=args.alpha)
