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
