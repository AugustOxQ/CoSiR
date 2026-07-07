"""
Buddy-analysis library — are "condition buddies" (cross-modal mutual KNN) meaningful?

Design: docs/superpowers/specs/2026-06-22-buddy-analysis-design.md

Key facts established before writing this:
  * Impressions feature store: img/txt (12123, 512), `sample_ids` SHUFFLED.
  * Join to metadata is POSITIONAL: meta[sample_id] is the record. (The `image_id`
    JSON field is non-unique, 0..2123 — do NOT join on it.)
  * Only 814 unique SOURCE IMAGES (ImgId `{hash}_cap{N}_{M}` -> `{hash}`); each
    photo reused 7-56x across caption types. Same-source img cos 0.822 vs 0.416
    random -> source-image identity is a real, finer-than-type grouping.

Two buddy graphs:
  * B = A_img ∩ A_txt  (strict: mutual NN in BOTH modalities — the "true" buddy)
  * E = A_img ∪ A_txt  (union: what actually initializes the condition vectors)
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, triu

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utils import FeatureManager
from src.conditional_buddy.buddy_graph import mutual_knn, union_graph

STORAGE = "/data/SSD2/pre_extract/impressions/features"
ANNOT = "/project/Impressions/metadata/impressions_train.json"
IMG_ROOT = "/project/Impressions/media"
TYPE_MAP = {"caption": 0, "description": 1, "impression": 2, "aesthetic": 3}
TYPE_NAMES = ["caption", "description", "impression", "aesthetic"]


def _base(imgid: str) -> str:
    """Source-image hash from ImgId `{hash}_cap{N}_{M}`."""
    return re.sub(r"_cap\d+.*$", "", imgid)


@dataclass
class Data:
    img: np.ndarray          # (N, D) L2-normalized CLIP image features, row order
    txt: np.ndarray          # (N, D) L2-normalized CLIP text features, row order
    sample_ids: list         # metadata list-index for each row
    types: np.ndarray        # (N,) int type id per row
    source_id: np.ndarray    # (N,) int source-image id per row (0..813)
    source_names: list       # source hash per source id
    records: list            # meta[sample_id] per row (caption, image filename, ...)

    @property
    def n(self):
        return self.img.shape[0]


def load_data() -> Data:
    """Load features + metadata with the correct positional join, in feature-row order."""
    fm = FeatureManager(STORAGE)
    d = fm.load_all_to_ram(["img_features", "txt_features"])
    img = F.normalize(d["img_features"].float(), dim=1).numpy().astype(np.float32)
    txt = F.normalize(d["txt_features"].float(), dim=1).numpy().astype(np.float32)
    sample_ids = [int(x) for x in d["sample_ids"]]
    meta = json.load(open(ANNOT))
    records = [meta[s] for s in sample_ids]
    types = np.array([TYPE_MAP[r["caption_type"]] for r in records], dtype=np.int64)
    bases = [_base(r["ImgId"]) for r in records]
    name2id = {b: i for i, b in enumerate(sorted(set(bases)))}
    source_id = np.array([name2id[b] for b in bases], dtype=np.int64)
    source_names = sorted(set(bases))
    return Data(img, txt, sample_ids, types, source_id, source_names, records)


def build_graphs(data: Data, K: int = 30, device: str = "cuda"):
    """Return dict with A_img, A_txt, and buddy graphs B (intersection) and E (union)."""
    A_img = mutual_knn(data.img, K=K, device=device)
    A_txt = mutual_knn(data.txt, K=K, device=device)
    E = union_graph(A_img, A_txt)
    # strict intersection: edge in BOTH modalities' mutual graphs
    B = A_img.multiply(A_txt)
    B.data[:] = 1
    B = B.tocsr()
    B.eliminate_zeros()
    return {"A_img": A_img, "A_txt": A_txt, "B": B, "E": E}


def edges(graph: csr_matrix) -> np.ndarray:
    """Undirected edge list (i<j) as (M, 2) int array from a symmetric adjacency."""
    U = triu(graph, k=1).tocoo()
    return np.stack([U.row, U.col], axis=1).astype(np.int64)


# ── Step 1: source-image identity ───────────────────────────────────────────

def identity_stats(data: Data, E_edges: np.ndarray) -> dict:
    """Within- vs cross-source-image split of a graph's edges."""
    same = data.source_id[E_edges[:, 0]] == data.source_id[E_edges[:, 1]]
    deg = np.bincount(E_edges.ravel(), minlength=data.n)
    return {
        "n_edges": int(len(E_edges)),
        "avg_degree": float(2 * len(E_edges) / data.n),
        "frac_within_image": float(same.mean()) if len(E_edges) else float("nan"),
        "n_within": int(same.sum()),
        "n_cross": int((~same).sum()),
        "isolated_nodes": int((deg == 0).sum()),
        "within_mask": same,
    }


# ── Step 2: type confusion matrix as lift over base rate ─────────────────────

def type_confusion(data: Data, E_edges: np.ndarray):
    """
    Symmetric type co-occurrence over edges.
    Returns (counts, lift, chi2, p) where lift[i,j] = observed / expected-if-random.
    Expected uses the type marginal of edge endpoints (so it controls for type imbalance).
    """
    from scipy.stats import chi2_contingency
    T = len(TYPE_NAMES)
    ti, tj = data.types[E_edges[:, 0]], data.types[E_edges[:, 1]]
    counts = np.zeros((T, T), dtype=np.float64)
    for a, b in zip(ti, tj):
        counts[a, b] += 1
        counts[b, a] += 1  # symmetrize
    total = counts.sum()
    p = counts.sum(axis=1) / total                       # endpoint type marginal
    expected = total * np.outer(p, p)
    lift = np.divide(counts, expected, out=np.full_like(counts, np.nan), where=expected > 0)
    chi2, pval, _, _ = chi2_contingency(counts + 1e-9)
    return counts, lift, float(chi2), float(pval)


# ── Step 3: independent held-out signal NN test ──────────────────────────────

def heldout_distance_test(heldout: np.ndarray, data: Data, E_edges: np.ndarray,
                          within_mask: np.ndarray, n_random: int = 50000, seed: int = 42):
    """
    Buddy-vs-random mean cosine DISTANCE in a held-out representation `heldout` (N, d),
    split into within-image and cross-image edges. Lower buddy/random ratio = buddies
    are closer = more meaningful in a signal NOT used to build the graph.
    """
    h = heldout / (np.linalg.norm(heldout, axis=1, keepdims=True) + 1e-12)

    def mean_dist(pairs):
        if len(pairs) == 0:
            return float("nan")
        sims = (h[pairs[:, 0]] * h[pairs[:, 1]]).sum(1)
        return float((1 - sims).mean())

    rng = np.random.default_rng(seed)
    ri, rj = rng.integers(0, data.n, n_random), rng.integers(0, data.n, n_random)
    rand_pairs = np.stack([ri, rj], 1)[ri != rj]
    rand_same = data.source_id[rand_pairs[:, 0]] == data.source_id[rand_pairs[:, 1]]

    return {
        "buddy_within": mean_dist(E_edges[within_mask]),
        "buddy_cross": mean_dist(E_edges[~within_mask]),
        "random_cross": mean_dist(rand_pairs[~rand_same]),
        "random_all": mean_dist(rand_pairs),
    }
