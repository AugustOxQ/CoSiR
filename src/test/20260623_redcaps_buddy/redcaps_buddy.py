"""
RedCaps buddy-analysis library — does the condition-buddy signal generalize to a
clean 1:1 dataset (no near-duplicate scaffolding)?

Counterpart to src/test/20260622_buddy_analysis/buddy_analysis.py, adapted for
RedCaps:
  * RedCaps is genuinely 1 image : 1 caption (every image_id unique), so there is
    NO same-photo identity probe — every buddy edge is cross-content by design.
  * Free ground truth is the SUBREDDIT (350 of them), parsed from the image path
    `redcaps/imagesYYYY/<subreddit>/id.jpg` (replaces Impressions' 4 caption types).

Two buddy graphs (same as before), K=30:
  * B = A_img ∩ A_txt  (strict: mutual NN in BOTH modalities)
  * E = A_img ∪ A_txt  (union: what actually initializes the condition vectors)
"""
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, triu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.utils import FeatureManager
from src.conditional_buddy.buddy_graph import mutual_knn, union_graph

STORAGE = "/data/SSD2/pre_extract/redcaps_150k/features"
ANNOT = "/data/PDD/redcaps/redcaps_plus/redcaps_150k.json"
IMG_ROOT = "/data/PDD"


def subreddit_of(rec) -> str:
    parts = rec["image"].split("/")
    return parts[2] if len(parts) > 2 else "?"


@dataclass
class Data:
    img: np.ndarray          # (N, D) L2-normalized CLIP image features, row order
    txt: np.ndarray          # (N, D) L2-normalized CLIP text features, row order
    sample_ids: list         # metadata list-index for each row
    sub_id: np.ndarray       # (N,) int subreddit id per row
    sub_names: list          # subreddit name per id
    records: list            # meta[sample_id] per row

    @property
    def n(self):
        return self.img.shape[0]


def load_data() -> Data:
    """Load features + metadata with the positional join, in feature-row order."""
    fm = FeatureManager(STORAGE)
    d = fm.load_all_to_ram(["img_features", "txt_features"])
    img = F.normalize(d["img_features"].float(), dim=1).numpy().astype(np.float32)
    txt = F.normalize(d["txt_features"].float(), dim=1).numpy().astype(np.float32)
    sample_ids = [int(x) for x in d["sample_ids"]]
    meta = json.load(open(ANNOT))
    records = [meta[s] for s in sample_ids]
    subs = [subreddit_of(r) for r in records]
    name2id = {b: i for i, b in enumerate(sorted(set(subs)))}
    sub_id = np.array([name2id[b] for b in subs], dtype=np.int64)
    sub_names = sorted(set(subs))
    return Data(img, txt, sample_ids, sub_id, sub_names, records)


def build_graphs(data: Data, K: int = 30, device: str = "cuda"):
    """A_img, A_txt, and buddy graphs B (intersection) and E (union)."""
    A_img = mutual_knn(data.img, K=K, device=device)
    A_txt = mutual_knn(data.txt, K=K, device=device)
    E = union_graph(A_img, A_txt)
    B = A_img.multiply(A_txt)
    B.data[:] = 1
    B = B.tocsr()
    B.eliminate_zeros()
    return {"A_img": A_img, "A_txt": A_txt, "B": B, "E": E}


def edges(graph: csr_matrix) -> np.ndarray:
    """Undirected edge list (i<j) as (M, 2) int array from a symmetric adjacency."""
    U = triu(graph, k=1).tocoo()
    return np.stack([U.row, U.col], axis=1).astype(np.int64)


def graph_stats(data: Data, e: np.ndarray) -> dict:
    deg = np.bincount(e.ravel(), minlength=data.n)
    return {
        "n_edges": int(len(e)),
        "avg_degree": float(2 * len(e) / data.n),
        "isolated_nodes": int((deg == 0).sum()),
    }


# ── Subreddit lift (replaces type-confusion; no same-photo split exists) ──────

def subreddit_lift(data: Data, e: np.ndarray, top_k: int = 15):
    """
    Same-subreddit enrichment over a graph's edges.

    overall_lift = P(edge endpoints share subreddit) / expected-if-random, where
    'expected' uses the subreddit marginal of edge ENDPOINTS (controls for the
    degree/subreddit imbalance — heavily-connected subreddits don't inflate it).

    Also returns per-subreddit within-subreddit lift for the most enriched ones.
    top_k: number of top subreddits to return (default 15). Set to None to return all
    subreddits passing the exp_s > 5 reliability filter, sorted by lift descending.
    """
    si, sj = data.sub_id[e[:, 0]], data.sub_id[e[:, 1]]
    same = si == sj
    n_sub = len(data.sub_names)

    # endpoint marginal over subreddits (each edge contributes both endpoints)
    endpoints = np.concatenate([si, sj])
    p = np.bincount(endpoints, minlength=n_sub).astype(np.float64)
    p /= p.sum()
    exp_same = float((p ** 2).sum())                 # P(two independent endpoints match)
    obs_same = float(same.mean())
    overall_lift = obs_same / exp_same if exp_same > 0 else float("nan")

    # per-subreddit: observed same-sub edges vs expected for that subreddit
    obs_s = np.bincount(si[same], minlength=n_sub) + np.bincount(sj[same], minlength=n_sub)
    deg_s = np.bincount(endpoints, minlength=n_sub).astype(np.float64)
    exp_s = deg_s * p                                # expected same-sub endpoint count
    lift_s = np.divide(obs_s, exp_s, out=np.full(n_sub, np.nan), where=exp_s > 5)
    order = np.argsort(-np.nan_to_num(lift_s))
    per_sub = [(data.sub_names[i], float(lift_s[i]), int(deg_s[i]))
               for i in order[:top_k] if exp_s[i] > 5]

    return {
        "obs_same_frac": obs_same,
        "exp_same_frac": exp_same,
        "overall_lift": overall_lift,
        "n_subreddits": n_sub,
        "top_enriched": per_sub,
    }


# ── Held-out independent-signal NN test (every edge is cross-content) ─────────

def heldout_distance_test(heldout: np.ndarray, data: Data, e: np.ndarray,
                          n_random: int = 50000, seed: int = 42):
    """
    Buddy-vs-random mean cosine DISTANCE in a held-out representation (N, d) that
    the graph never saw (DINOv2). Lower buddy distance = buddies are closer =
    meaningful in a signal not used to build the graph. All edges are cross-content.
    Rows with all-zero held-out features (missing images) are skipped.
    """
    h = heldout / (np.linalg.norm(heldout, axis=1, keepdims=True) + 1e-12)
    valid = np.abs(heldout).sum(1) > 0

    def mean_dist(pairs):
        if len(pairs) == 0:
            return float("nan")
        ok = valid[pairs[:, 0]] & valid[pairs[:, 1]]
        pairs = pairs[ok]
        if len(pairs) == 0:
            return float("nan")
        sims = (h[pairs[:, 0]] * h[pairs[:, 1]]).sum(1)
        return float((1 - sims).mean())

    rng = np.random.default_rng(seed)
    ri, rj = rng.integers(0, data.n, n_random), rng.integers(0, data.n, n_random)
    rand_pairs = np.stack([ri, rj], 1)[ri != rj]

    return {
        "buddy": mean_dist(e),
        "random": mean_dist(rand_pairs),
        "n_buddy_edges": int(len(e)),
    }
