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
