"""
Experiment 12 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md,
Experiment 12 subsection): cross-modal polysemy bridge-node diagnostic.

Does the existing buddy-graph construction already implicitly reflect cross-modal
polysemy -- a bridge node A connected to B via an image-only mutual-kNN edge and to C via
a text-only one -- and does the buddy-init spectral embedding place B and C closer
together than a degree-matched random baseline? If so, is that pull graded by real
shared-neighbor structure (legitimate signal), or flat/arbitrary ("false transitivity",
the risk Experiment 10's own diagnostic flagged but never measured)? Separately: does a
per-node polysemy label predict anything about per-sample retrieval rank / condition
drift (reusing Experiment 11.2's per-sample outputs)?

No new training, no new graph-construction mechanism -- reuses classify_edges/
bridge_node_stats (src/conditional_buddy/buddy_graph.py) and an already-completed
buddy-init template.

Usage
-----
  python scripts/analyze_polysemy_bridges.py --selftest
  python scripts/analyze_polysemy_bridges.py \\
      --storage-dir /data/SSD2/pre_extract/redcaps_150k/features \\
      --template-dir res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings \\
      --n-bridge-sample 5000

Requires: numpy, scipy (all already deps).
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analyze_condition_retrieval_correlation import spearman_correlate


def degree_deciles(E: csr_matrix, n_buckets: int = 10) -> np.ndarray:
    """Per-node degree-decile bucket id (0 = lowest degree, n_buckets-1 = highest),
    used to sample a degree-matched baseline node for the false-transitivity check."""
    degree = np.diff(E.indptr)
    ranks = rankdata(degree, method="average") / len(degree)  # in (0, 1]
    buckets = np.minimum((ranks * n_buckets).astype(np.int64), n_buckets - 1)
    return buckets


def sample_baselines(
    pairs: np.ndarray,
    E: csr_matrix,
    buckets: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 50,
) -> np.ndarray:
    """For each (A, B, C) row, sample a degree-bucket-matched C' that is NOT a direct
    E-neighbor of B and not equal to B or C -- the "is B pulled toward C specifically, or
    just toward any similarly-connected node" baseline. -1 where no candidate was found
    within max_tries (excluded from downstream stats by the caller)."""
    node_ids_by_bucket = {k: np.where(buckets == k)[0] for k in np.unique(buckets)}
    out = np.full(len(pairs), -1, dtype=np.int64)
    for row in range(len(pairs)):
        a, b, c = pairs[row]
        candidates = node_ids_by_bucket[buckets[c]]
        b_neighbors = set(E.indices[E.indptr[b]:E.indptr[b + 1]].tolist())
        for _ in range(max_tries):
            c_prime = int(rng.choice(candidates))
            if c_prime != b and c_prime != c and c_prime not in b_neighbors:
                out[row] = c_prime
                break
    return out


def shared_neighbor_jaccard(E: csr_matrix, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Jaccard overlap of each (b[i], c[i]) pair's neighbor sets in E -- the LINE/GraRep-
    style second-order-proximity score used to test whether any embedded pull is graded
    by real shared-neighbor structure. Loop-based: fine at the capped sample size
    (~thousands of pairs) this experiment uses."""
    out = np.empty(len(b), dtype=np.float64)
    for idx in range(len(b)):
        nb = set(E.indices[E.indptr[b[idx]]:E.indptr[b[idx] + 1]].tolist())
        nc = set(E.indices[E.indptr[c[idx]]:E.indptr[c[idx] + 1]].tolist())
        union = len(nb | nc)
        out[idx] = len(nb & nc) / union if union > 0 else 0.0
    return out


def embedded_l2_distance(emb: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Euclidean distance between rows i and j of the buddy-init embedding -- matches
    this project's existing condition_drift L2 convention."""
    return np.linalg.norm(emb[i] - emb[j], axis=1)


def paired_pull_summary(dist_bc: np.ndarray, dist_bc_baseline: np.ndarray) -> dict:
    """Paired difference (baseline - bridge_pair): positive means the bridge-derived
    (B, C) pair sits CLOSER together in the embedding than its degree-matched baseline
    pair, i.e. a 'pull'. Same mean/std/sem/z convention as this project's other
    paired-delta analysis scripts."""
    pull = dist_bc_baseline - dist_bc
    n = len(pull)
    mean = float(pull.mean())
    std = float(pull.std(ddof=1)) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 and std == std else float("nan")
    z = mean / sem if sem == sem and sem > 0 else float("nan")
    return {
        "n": n, "mean": mean, "std": std, "sem": sem, "z": z,
        "frac_pulled_closer": float((pull > 0).mean()),
    }


def build_typed_adjacency(typed: dict, N: int) -> Tuple[csr_matrix, csr_matrix]:
    """Symmetric binary adjacency for just the img-only and just the txt-only edges of
    the union graph, so per-node neighbor lists can be sliced by modality-provenance
    type without re-scanning classify_edges's flat edge list each time."""
    keys = typed["keys"]
    i = (keys // N).astype(np.int64)
    j = (keys % N).astype(np.int64)

    def _sym(mask):
        ii, jj = i[mask], j[mask]
        rows = np.concatenate([ii, jj])
        cols = np.concatenate([jj, ii])
        data = np.ones(len(rows), dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(N, N))

    return _sym(typed["img_only"]), _sym(typed["txt_only"])


def extract_bridge_pairs(
    bridge_stats: dict,
    E_img_only: csr_matrix,
    E_txt_only: csr_matrix,
    n_sample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """For up to n_sample bridge nodes, pick one random img-only neighbor (B) and one
    random txt-only neighbor (C). Returns int64 array of shape (M, 3): columns [A, B, C].
    B != C always -- a node's img-only and txt-only neighbor sets are disjoint by
    construction (classify_edges assigns each edge to exactly one bucket)."""
    bridge_ids = np.where(bridge_stats["is_bridge"])[0]
    if len(bridge_ids) > n_sample:
        bridge_ids = rng.choice(bridge_ids, size=n_sample, replace=False)

    triples = []
    for a in bridge_ids:
        b_candidates = E_img_only.indices[E_img_only.indptr[a]:E_img_only.indptr[a + 1]]
        c_candidates = E_txt_only.indices[E_txt_only.indptr[a]:E_txt_only.indptr[a + 1]]
        if len(b_candidates) == 0 or len(c_candidates) == 0:
            continue
        b = int(rng.choice(b_candidates))
        c = int(rng.choice(c_candidates))
        triples.append((int(a), b, c))
    if not triples:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(triples, dtype=np.int64)


def label_nodes(bridge_stats: dict) -> np.ndarray:
    """Per-node polysemy label from classify_edges/bridge_node_stats's own per-node
    degree counts: 'bridge' (has both an img-only AND a txt-only edge -- the "A" node in
    Experiment 12's A/B/C example), 'img_only_only' / 'txt_only_only' (has edges of only
    one such type), or 'neither' (only 'both'/'repair' edges, or no edges of these types
    at all)."""
    deg_img = bridge_stats["deg_img_only"]
    deg_txt = bridge_stats["deg_txt_only"]
    is_bridge = bridge_stats["is_bridge"]
    labels = np.full(len(deg_img), "neither", dtype="<U16")
    labels[(deg_img > 0) & ~is_bridge] = "img_only_only"
    labels[(deg_txt > 0) & ~is_bridge] = "txt_only_only"
    labels[is_bridge] = "bridge"
    return labels


def _selftest():
    # label_nodes: 4 nodes -- one bridge, one img-only-only, one txt-only-only, one bare.
    bridge_stats = {
        "deg_img_only": np.array([2, 1, 0, 0]),
        "deg_txt_only": np.array([1, 0, 3, 0]),
        "is_bridge": np.array([True, False, False, False]),
    }
    labels = label_nodes(bridge_stats)
    assert list(labels) == ["bridge", "img_only_only", "txt_only_only", "neither"], labels

    # build_typed_adjacency + extract_bridge_pairs, on the same synthetic graph as
    # buddy_graph's own bridge-node test: node 1 is a bridge (img-only to 0, txt-only to 4).
    from scipy.sparse import csr_matrix as _csr
    from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges

    def _sym(n, edges):
        rows, cols = [], []
        for i, j in edges:
            rows += [i, j]; cols += [j, i]
        return _csr((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

    n = 5
    A_img = _sym(n, [(0, 1), (2, 3)])
    A_txt = _sym(n, [(1, 4), (2, 3)])
    E = _sym(n, [(0, 1), (1, 4), (2, 3)])
    typed = classify_edges(A_img, A_txt, E, n)
    bstats = bridge_node_stats(typed, n)

    E_img_only, E_txt_only = build_typed_adjacency(typed, n)
    assert E_img_only[0, 1] == 1 and E_img_only[1, 0] == 1
    assert E_txt_only[1, 4] == 1 and E_txt_only[4, 1] == 1
    assert E_img_only[2, 3] == 0 and E_txt_only[2, 3] == 0  # (2,3) is a "both" edge

    rng = np.random.default_rng(0)
    pairs = extract_bridge_pairs(bstats, E_img_only, E_txt_only, n_sample=10, rng=rng)
    assert pairs.shape == (1, 3), pairs.shape
    a, b, c = pairs[0]
    assert a == 1 and b == 0 and c == 4, pairs

    # degree_deciles: 10 nodes with distinct degrees -> deciles 0..9 in order.
    n10 = 10
    edges10 = [(0, k) for k in range(1, 10)]  # node 0 has degree 9; nodes 1-9 have degree 1
    E10 = _sym(n10, edges10)
    buckets = degree_deciles(E10, n_buckets=10)
    assert buckets[0] == 9, buckets  # highest degree -> top decile
    assert buckets[1] < buckets[0], buckets

    # sample_baselines: node 5 (degree 1, bucket low) should never return node 0 (a direct
    # neighbor of b=1) or nodes with a very different degree.
    pairs10 = np.array([[0, 1, 5]])  # a=0, b=1 (neighbor of 0), c=5
    rng10 = np.random.default_rng(1)
    baselines = sample_baselines(pairs10, E10, buckets, rng10)
    assert baselines.shape == (1,)
    assert baselines[0] not in (1, 5, -1), baselines  # not b, not c, and a candidate WAS found

    # shared_neighbor_jaccard: b and c share exactly node 0 as a neighbor (of 9 total).
    jac = shared_neighbor_jaccard(E10, np.array([1]), np.array([2]))
    assert abs(jac[0] - (1 / 1)) < 1e-9, jac  # N(1)={0}, N(2)={0} -> intersection=union=1

    # embedded_l2_distance: known Euclidean distances.
    emb = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    d = embedded_l2_distance(emb, np.array([0, 1]), np.array([1, 2]))
    assert np.allclose(d, [5.0, 5.0]), d

    # paired_pull_summary: baseline consistently 1.0 farther than the bridge pair ->
    # mean pull exactly 1.0, all wins.
    dist_bc = np.array([1.0, 1.0, 1.0])
    dist_baseline = np.array([2.0, 2.0, 2.0])
    summary = paired_pull_summary(dist_bc, dist_baseline)
    assert summary["n"] == 3
    assert abs(summary["mean"] - 1.0) < 1e-9, summary
    assert summary["frac_pulled_closer"] == 1.0, summary
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
