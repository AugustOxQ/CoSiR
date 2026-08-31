# Cross-Modal Polysemy Bridge-Node Diagnostic (Experiment 12) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer Experiment 12 from `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (Experiment 12 subsection) — does the existing buddy-graph construction already implicitly reflect cross-modal polysemy (a bridge node A connected to B via an image-only edge and to C via a text-only edge), does the resulting spectral embedding place B and C closer together than chance, and — if so — is that pull graded by real shared-neighbor structure (legitimate signal) or flat/arbitrary ("false transitivity", the risk Experiment 10's own diagnostic flagged but never measured)? Also: does a per-node polysemy label predict anything about per-sample retrieval rank or condition drift (reusing Experiment 11.2's machinery)?

**Architecture:** No new training, no new graph-construction mechanism. `classify_edges()`/`bridge_node_stats()` (currently a one-off diagnostic in `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py`) move into `src/conditional_buddy/buddy_graph.py` as reusable public functions (Task 1). `scripts/analyze_condition_retrieval_correlation.py`'s `analyze_pair()` gets a small opt-in extension to persist the per-sample arrays it already computes internally but currently only aggregates (Task 2). A new script, `scripts/analyze_polysemy_bridges.py`, is built incrementally (Tasks 3–7) as a set of small pure functions (each with its own selftest coverage) wired together by one `run()`/`main()` (Task 7), then run once against the existing RedCaps-150k buddy-init template and cached features to produce the actual finding (Task 8).

**Tech Stack:** Python 3.10, numpy, scipy.sparse, scipy.stats (spearmanr, rankdata). No new dependencies. No PyTorch/training/wandb needed except to load one existing checkpoint snapshot in Task 8 for the retrieval-correlation cross-reference.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, Experiment 12 subsection (added 2026-08-26).

## Global Constraints

- Always run Python commands with `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` first (per `.claude/CLAUDE.md`).
- Zero new training runs, zero new feature extraction. Every task in this plan is a code/analysis change against already-cached features and an already-completed buddy-init template.
- RedCaps-150k operating point, matching Experiments 9–11: `K=30`, `alpha=0.5`. Feature store: `/data/SSD2/pre_extract/redcaps_150k/features` (`configs/dataset/redcaps_150k.yaml`). Shared buddy-init template: `res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings/` (built once by Experiment 11.1's first run against that `results_dir`; every seed/arm sharing that `results_dir`, including 11.1/11.2/11.3, loads this exact same template — there is only **one** buddy-init graph/embedding to audit here, not one per seed).
- This diagnostic's own statistical convention is **pair-count-based**, not the ≥3-training-seed convention from spec §5 (which applies to training-outcome comparisons) — matches the precedent set by Experiment 9's and Experiment 10's own diagnostic scripts (`diagnose_disagreement.py`), which report population-level statistics (medians, fractions) over many edges/nodes from one deterministic graph, no seed loop.
- `classify_edges`/`bridge_node_stats`'s existing invariant, verified by their current tests and preserved here: every edge of the union graph `E` falls into exactly one of `{img_only, txt_only, both, repair}`; a node's img-only and txt-only neighbor sets are therefore always disjoint (an edge cannot be both types at once).
- One existing `src/` file (`src/conditional_buddy/buddy_graph.py`) and one existing test-diagnostic file (`src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py`) are modified in Task 1; one existing `scripts/` file (`scripts/analyze_condition_retrieval_correlation.py`) is modified in Task 2. Per CLAUDE.md, log both changes in `.claude/20260826_log.md` (one `# <path>` section per file; append to the existing Task-1-created file rather than creating a second one).

---

### Task 1: Promote `classify_edges`/`bridge_node_stats` into `buddy_graph.py`

**Files:**
- Modify: `src/conditional_buddy/buddy_graph.py` (append after `mix_distances_typed`, i.e. after its closing line ~433)
- Modify: `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py` (remove the local `classify_edges`/`bridge_node_stats`/`_adj_to_keys` definitions at lines 38–129; import them instead)
- Create: `src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py`
- Create: `.claude/20260826_log.md`

**Interfaces:**
- Produces: `classify_edges(A_img: csr_matrix, A_txt: csr_matrix, E: csr_matrix, N: int) -> dict` (keys: `"keys"`, `"img_only"`, `"txt_only"`, `"both"`, `"repair"`, exactly as today) and `bridge_node_stats(typed: dict, N: int) -> dict` (keys: `"n_bridge_nodes"`, `"frac_bridge_nodes"`, `"deg_img_only"`, `"deg_txt_only"`, `"is_bridge"`, exactly as today), both from `src.conditional_buddy.buddy_graph`. Consumed by Task 3 onward.

- [ ] **Step 1: Write the failing test**

Create `src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py`:

```python
"""Tests for classify_edges/bridge_node_stats (Experiment 12,
docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md), promoted from the
one-off src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py diagnostic
into src/conditional_buddy/buddy_graph.py as reusable public functions.

Run:
    python src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from scipy.sparse import csr_matrix

from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges


def _csr(n, edges):
    rows, cols = [], []
    for i, j in edges:
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def test_classify_edges_buckets_correctly():
    n = 6
    A_img = _csr(n, [(0, 1), (2, 3)])
    A_txt = _csr(n, [(0, 1), (4, 5)])
    E = _csr(n, [(0, 1), (2, 3), (4, 5), (1, 2)])

    typed = classify_edges(A_img, A_txt, E, n)
    keys = typed["keys"]
    pairs = {(int(k // n), int(k % n)) for k in keys}
    assert pairs == {(0, 1), (2, 3), (4, 5), (1, 2)}, pairs

    def _idx(pair):
        return list(keys).index(pair[0] * n + pair[1])

    assert typed["both"][_idx((0, 1))]
    assert typed["img_only"][_idx((2, 3))]
    assert typed["txt_only"][_idx((4, 5))]
    assert typed["repair"][_idx((1, 2))]
    stacked = np.stack([typed["img_only"], typed["txt_only"], typed["both"], typed["repair"]])
    assert np.all(stacked.sum(axis=0) == 1), "every edge must be classified into exactly one bucket"
    print("PASS test_classify_edges_buckets_correctly")


def test_bridge_node_detection():
    n = 5
    # Node 1 connects to 0 via img_only, and to 4 via txt_only -> node 1 is a bridge.
    A_img = _csr(n, [(0, 1), (2, 3)])
    A_txt = _csr(n, [(1, 4), (2, 3)])
    E = _csr(n, [(0, 1), (1, 4), (2, 3)])
    typed = classify_edges(A_img, A_txt, E, n)
    stats = bridge_node_stats(typed, n)
    assert stats["is_bridge"][1] == True, stats["is_bridge"]
    assert stats["is_bridge"][0] == False
    assert stats["is_bridge"][2] == False
    assert stats["n_bridge_nodes"] == 1
    print(f"PASS test_bridge_node_detection (n_bridge_nodes={stats['n_bridge_nodes']})")


def test_img_only_and_txt_only_neighbor_sets_are_disjoint():
    """A node's img-only and txt-only neighbor sets can never overlap -- each edge is
    classified into exactly one bucket, so no neighbor can appear via both bucket types
    for the same node. This is the invariant Task 4's extract_bridge_pairs relies on to
    guarantee B != C without an explicit check."""
    n = 5
    A_img = _csr(n, [(0, 1), (2, 3)])
    A_txt = _csr(n, [(1, 4), (2, 3)])
    E = _csr(n, [(0, 1), (1, 4), (2, 3)])
    typed = classify_edges(A_img, A_txt, E, n)
    stats = bridge_node_stats(typed, n)
    bridge_id = 1
    keys = typed["keys"]
    i = (keys // n).astype(np.int64)
    j = (keys % n).astype(np.int64)
    img_only_neighbors = set(j[(i == bridge_id) & typed["img_only"]]) | set(i[(j == bridge_id) & typed["img_only"]])
    txt_only_neighbors = set(j[(i == bridge_id) & typed["txt_only"]]) | set(i[(j == bridge_id) & typed["txt_only"]])
    assert img_only_neighbors.isdisjoint(txt_only_neighbors), (img_only_neighbors, txt_only_neighbors)
    print("PASS test_img_only_and_txt_only_neighbor_sets_are_disjoint")


if __name__ == "__main__":
    test_classify_edges_buckets_correctly()
    test_bridge_node_detection()
    test_img_only_and_txt_only_neighbor_sets_are_disjoint()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py
```

Expected: `ImportError: cannot import name 'classify_edges' from 'src.conditional_buddy.buddy_graph'`.

- [ ] **Step 3: Move the functions into `buddy_graph.py`**

In `src/conditional_buddy/buddy_graph.py`, change the import line near the top:

```python
from scipy.sparse import csr_matrix, coo_matrix
```

to:

```python
from scipy.sparse import csr_matrix, coo_matrix, triu
```

Then append, at the end of the file (after `mix_distances_typed`'s closing `return csr_matrix(...)` line):

```python


# ── Edge/node classification (Experiment 12) ─────────────────────────────────


def _adj_to_keys(A: csr_matrix, N: int) -> np.ndarray:
    """Upper-triangular (i<j) edges of a symmetric adjacency as sorted int64 keys
    i*N+j -- fast set-membership via np.isin on sorted unique arrays."""
    U = triu(A, k=1).tocoo()
    mask = U.data != 0
    keys = (U.row[mask].astype(np.int64) * N + U.col[mask].astype(np.int64))
    keys.sort()
    return keys


def classify_edges(A_img: csr_matrix, A_txt: csr_matrix, E: csr_matrix, N: int) -> dict:
    """
    Classify every edge of the union graph E by which modality(ies) support it.

    Returns {"keys": sorted int64 edge keys (i*N+j, i<j) for E,
             "img_only"/"txt_only"/"both"/"repair": boolean masks aligned to "keys"}.
    Every edge falls into exactly one bucket -- an edge cannot be both img_only and
    txt_only, so a node's img-only and txt-only neighbor sets are always disjoint.
    """
    keys_img = _adj_to_keys(A_img, N)
    keys_txt = _adj_to_keys(A_txt, N)
    keys_E = _adj_to_keys(E, N)

    in_img = np.isin(keys_E, keys_img, assume_unique=True)
    in_txt = np.isin(keys_E, keys_txt, assume_unique=True)

    return {
        "keys": keys_E,
        "img_only": in_img & ~in_txt,
        "txt_only": ~in_img & in_txt,
        "both": in_img & in_txt,
        "repair": ~in_img & ~in_txt,
    }


def bridge_node_stats(typed: dict, N: int) -> dict:
    """A node is a 'bridge' if it has at least one img_only edge AND at least one
    txt_only edge -- i.e. it connects to different neighbors via different,
    non-overlapping modality evidence (Experiment 12's node "A")."""
    keys = typed["keys"]
    i = (keys // N).astype(np.int64)
    j = (keys % N).astype(np.int64)

    deg_img_only = np.zeros(N, dtype=np.int64)
    deg_txt_only = np.zeros(N, dtype=np.int64)
    for mask, deg in ((typed["img_only"], deg_img_only), (typed["txt_only"], deg_txt_only)):
        ii, jj = i[mask], j[mask]
        np.add.at(deg, ii, 1)
        np.add.at(deg, jj, 1)

    is_bridge = (deg_img_only > 0) & (deg_txt_only > 0)
    return {
        "n_bridge_nodes": int(is_bridge.sum()),
        "frac_bridge_nodes": float(is_bridge.mean()),
        "deg_img_only": deg_img_only,
        "deg_txt_only": deg_txt_only,
        "is_bridge": is_bridge,
    }
```

- [ ] **Step 4: Update the diagnostic to import instead of redefine**

In `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py`, delete two blocks by content (not by line number — removing the first block shifts the second): the entire `def _adj_to_keys(...):` function body through the entire `def classify_edges(...):` function body (everything from `def _adj_to_keys` up to, but not including, the `def rank_normalize(` line that follows them), and separately the entire `def bridge_node_stats(...):` function body (from `def bridge_node_stats` up to, but not including, the `SCALES = {` line that follows it). Then add near the top (after the existing `from scipy.sparse import csr_matrix, triu` import):

```python
from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges
```

Leave `rank_normalize`, `diagnose`, `bridge_node_stats` call sites, `run()`, and the CLI unchanged — they only ever called these two functions by name, so the diagnostic script's own behavior and its own existing test (`test_diagnose_disagreement.py`, which imports `bridge_node_stats, classify_edges, diagnose, rank_normalize` from this module) keep working unchanged via the re-export.

- [ ] **Step 5: Run both test files to verify everything passes**

```bash
python src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py
python src/test/20260824_buddy_graph_disagreement/test_diagnose_disagreement.py
```

Expected: `ALL TESTS PASSED` from the new file; `ALL TESTS PASSED` from the pre-existing one too (confirms the re-export didn't break the original diagnostic's own test suite).

- [ ] **Step 6: Log the change**

Create `.claude/20260826_log.md`:

```markdown
# /src/conditional_buddy/buddy_graph.py

## Added `classify_edges`/`bridge_node_stats` (promoted from a one-off diagnostic)

**Before:** These lived only in `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py`,
a one-off diagnostic script, not importable as reusable infrastructure.

**After:** Moved into `src/conditional_buddy/buddy_graph.py` as public functions, byte-identical
logic. `diagnose_disagreement.py` now imports them instead of redefining them.

**Why:** Experiment 12 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`)
depends on them as core logic, not a throwaway diagnostic — see
`docs/superpowers/plans/2026-08-26-polysemy-bridge-audit.md` Task 1.

# /src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py

## Now imports classify_edges/bridge_node_stats instead of defining them locally

**Why:** DRY — see Task 1 above. No behavior change; `test_diagnose_disagreement.py`
continues to pass unchanged via the re-export.
```

- [ ] **Step 7: Commit**

```bash
git add src/conditional_buddy/buddy_graph.py src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py .claude/20260826_log.md
git commit -m "refactor: promote classify_edges/bridge_node_stats into buddy_graph.py (Experiment 12)"
```

---

### Task 2: Persist 11.2's per-sample arrays as an opt-in `.npz` dump

**Files:**
- Modify: `scripts/analyze_condition_retrieval_correlation.py`

**Interfaces:**
- Produces: `build_per_sample_dump(sample_ids: List[int], delta_rank: np.ndarray, delta_rank_swap: np.ndarray, condition_drift: np.ndarray, embedding_shift: np.ndarray) -> Dict[str, np.ndarray]`. `analyze_pair(..., dump_per_sample: bool = False)` — when `True`, writes `condition_geometry/per_sample_retrieval_correlation.npz` under `trained_dir` with these five arrays. Consumed by Task 6/8.

- [ ] **Step 1: Write the failing selftest addition**

In `scripts/analyze_condition_retrieval_correlation.py`'s `_selftest()`, add before the final `print("SELFTEST OK")` line:

```python
    # build_per_sample_dump: packs aligned arrays, keyed by sample_id.
    dump = build_per_sample_dump(
        sample_ids=[10, 11, 12],
        delta_rank=np.array([1, -2, 0]),
        delta_rank_swap=np.array([2, -1, 0]),
        condition_drift=np.array([0.1, 0.2, 0.3]),
        embedding_shift=np.array([0.01, 0.02, 0.03]),
    )
    assert list(dump["sample_ids"]) == [10, 11, 12], dump
    assert list(dump["delta_rank"]) == [1, -2, 0], dump
    assert list(dump["embedding_shift"]) == [0.01, 0.02, 0.03], dump

    try:
        build_per_sample_dump(
            sample_ids=[10, 11],
            delta_rank=np.array([1, -2, 0]),
            delta_rank_swap=np.array([2, -1, 0]),
            condition_drift=np.array([0.1, 0.2, 0.3]),
            embedding_shift=np.array([0.01, 0.02, 0.03]),
        )
        raise AssertionError("expected a length-mismatch error")
    except AssertionError as e:
        assert "length" in str(e), e
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_retrieval_correlation.py --selftest
```

Expected: `NameError: name 'build_per_sample_dump' is not defined`.

- [ ] **Step 3: Implement `build_per_sample_dump` and wire it into `analyze_pair`**

Add this function above `analyze_pair` (after `_project_other`, before `def analyze_pair(`):

```python
def build_per_sample_dump(
    sample_ids: List[int],
    delta_rank: np.ndarray,
    delta_rank_swap: np.ndarray,
    condition_drift: np.ndarray,
    embedding_shift: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pack the per-sample arrays analyze_pair already computes internally into one dict
    keyed by sample_id, for persistence (analyze_pair only ever aggregated these into
    summary statistics before; Experiment 12 needs the raw per-sample values)."""
    n = len(sample_ids)
    for name, arr in (
        ("delta_rank", delta_rank), ("delta_rank_swap", delta_rank_swap),
        ("condition_drift", condition_drift), ("embedding_shift", embedding_shift),
    ):
        assert len(arr) == n, f"length mismatch: sample_ids has {n} entries, {name} has {len(arr)}"
    return {
        "sample_ids": np.asarray(sample_ids, dtype=np.int64),
        "delta_rank": np.asarray(delta_rank),
        "delta_rank_swap": np.asarray(delta_rank_swap),
        "condition_drift": np.asarray(condition_drift),
        "embedding_shift": np.asarray(embedding_shift),
    }
```

Change `analyze_pair`'s signature (currently `rank_chunk: int = 200,`) to add one more parameter:

```python
def analyze_pair(
    frozen_dir: str,
    trained_dir: str,
    n_query_sample: int = 3000,
    seed: int = 0,
    k_extremes: int = 20,
    rank_chunk: int = 200,
    dump_per_sample: bool = False,
) -> dict:
```

Then, right after the existing `out_path = Path(trained_dir) / "condition_geometry" / "retrieval_correlation_vs_frozen.json"` block finishes writing (i.e., right after the `json.dump(result, f, indent=2)` line, still before the `print(f"\n{'='*78}...` block), add:

```python
    if dump_per_sample:
        dump = build_per_sample_dump(
            sample_ids=query_sample_ids,
            delta_rank=delta_rank,
            delta_rank_swap=delta_rank_swap,
            condition_drift=drift_query,
            embedding_shift=shift_trained,
        )
        dump_path = Path(trained_dir) / "condition_geometry" / "per_sample_retrieval_correlation.npz"
        np.savez(dump_path, **dump)
        print(f"  Wrote per-sample dump: {dump_path}")
```

- [ ] **Step 4: Add the CLI flag**

In `main()`, add next to the other `ap.add_argument` calls:

```python
    ap.add_argument("--dump-per-sample", action="store_true",
                    help="also write condition_geometry/per_sample_retrieval_correlation.npz "
                         "with the raw per-sample delta_rank/condition_drift/embedding_shift "
                         "arrays (Experiment 12)")
```

And change the `analyze_pair(...)` call inside `if args.pair:` to:

```python
        analyze_pair(
            args.pair[0], args.pair[1],
            n_query_sample=args.n_query_sample, seed=args.seed, k_extremes=args.k_extremes,
            rank_chunk=max(1, args.rank_chunk), dump_per_sample=args.dump_per_sample,
        )
```

- [ ] **Step 5: Run the selftest again to verify it passes**

```bash
python scripts/analyze_condition_retrieval_correlation.py --selftest
```

Expected: `SELFTEST OK`, exit code 0.

- [ ] **Step 6: Log the change**

Append to `.claude/20260826_log.md`:

```markdown
# /scripts/analyze_condition_retrieval_correlation.py

## Added opt-in per-sample `.npz` dump to `analyze_pair`

**Before:** `analyze_pair` computed `delta_rank`/`delta_rank_swap`/`condition_drift`/
`embedding_shift` per query sample internally, but only ever persisted their aggregates
and top-k extremes to `condition_geometry/retrieval_correlation_vs_frozen.json`.

**After:** New `dump_per_sample: bool = False` parameter (and `--dump-per-sample` CLI flag);
when set, also writes `condition_geometry/per_sample_retrieval_correlation.npz` with the
full per-sample arrays, keyed by `sample_ids`, via the new pure helper
`build_per_sample_dump`. Default-off, so existing callers/output are unaffected.

**Why:** Experiment 12 needs to join per-sample retrieval-rank/drift against a per-node
polysemy label — see `docs/superpowers/plans/2026-08-26-polysemy-bridge-audit.md` Task 2.
```

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_condition_retrieval_correlation.py .claude/20260826_log.md
git commit -m "feat: add opt-in per-sample retrieval-correlation dump (Experiment 12)"
```

---

### Task 3: New script skeleton — node labeling

**Files:**
- Create: `scripts/analyze_polysemy_bridges.py`

**Interfaces:**
- Produces: `label_nodes(bridge_stats: dict) -> np.ndarray` (dtype `"<U16"`, one of `"neither"`/`"img_only_only"`/`"txt_only_only"`/`"bridge"` per node, aligned to `bridge_stats`'s arrays). Consumed by Task 6/7/8.

- [ ] **Step 1: Write the script with a failing selftest**

Create `scripts/analyze_polysemy_bridges.py`:

```python
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
```

- [ ] **Step 2: Run it to verify the selftest passes right away**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `SELFTEST OK` (this task's function is simple enough that the first write already passes — still run it to confirm before committing).

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add analyze_polysemy_bridges.py skeleton with label_nodes (Experiment 12)"
```

---

### Task 4: Typed adjacency + bridge-pair extraction

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py`

**Interfaces:**
- Consumes: `classify_edges`/`bridge_node_stats` output shapes (Task 1).
- Produces: `build_typed_adjacency(typed: dict, N: int) -> Tuple[csr_matrix, csr_matrix]` (returns `(E_img_only, E_txt_only)`, both symmetric binary); `extract_bridge_pairs(bridge_stats: dict, E_img_only: csr_matrix, E_txt_only: csr_matrix, n_sample: int, rng: np.random.Generator) -> np.ndarray` (int64, shape `(M, 3)`, columns `[A, B, C]`). Consumed by Task 5/7.

- [ ] **Step 1: Add the failing selftest**

In `_selftest()`, add before `print("SELFTEST OK")`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `NameError: name 'build_typed_adjacency' is not defined`.

- [ ] **Step 3: Implement both functions**

Add above `label_nodes`:

```python
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
```

- [ ] **Step 4: Run the selftest again to verify it passes**

```bash
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `SELFTEST OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add typed-adjacency bridge-pair extraction (Experiment 12)"
```

---

### Task 5: Degree-matched baseline sampling + distance/proximity metrics

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py`

**Interfaces:**
- Produces: `degree_deciles(E: csr_matrix, n_buckets: int = 10) -> np.ndarray`; `sample_baselines(pairs: np.ndarray, E: csr_matrix, buckets: np.ndarray, rng: np.random.Generator, max_tries: int = 50) -> np.ndarray` (int64, shape `(M,)`, `-1` where no candidate found); `shared_neighbor_jaccard(E: csr_matrix, b: np.ndarray, c: np.ndarray) -> np.ndarray`; `embedded_l2_distance(emb: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray`; `paired_pull_summary(dist_bc: np.ndarray, dist_bc_baseline: np.ndarray) -> dict`. Consumed by Task 7/8.

- [ ] **Step 1: Add the failing selftest**

In `_selftest()`, add before `print("SELFTEST OK")`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `NameError: name 'degree_deciles' is not defined`.

- [ ] **Step 3: Implement all four functions**

Add above `build_typed_adjacency`:

```python
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
```

- [ ] **Step 4: Run the selftest again to verify it passes**

```bash
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `SELFTEST OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add baseline sampling + distance/proximity metrics (Experiment 12)"
```

---

### Task 6: Cross-reference against 11.2's per-sample retrieval-rank/drift dump

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py`

**Interfaces:**
- Consumes: the `.npz` produced by Task 2's `dump_per_sample=True`; `spearman_correlate` (already imported at module top from `analyze_condition_retrieval_correlation`).
- Produces: `correlate_polysemy_with_retrieval(labels: np.ndarray, sample_ids: List[int], npz_path: str) -> dict`. Consumed by Task 7.

- [ ] **Step 1: Add the failing selftest**

In `_selftest()`, add before `print("SELFTEST OK")`:

```python
    # correlate_polysemy_with_retrieval: 4 nodes, sample_ids in FeatureManager order;
    # dump covers only 3 of them (id 103 missing -> must be excluded, not crash).
    import tempfile
    labels4 = np.array(["bridge", "neither", "img_only_only", "neither"])
    sample_ids4 = [100, 101, 102, 103]
    with tempfile.TemporaryDirectory() as tmp:
        npz_path = os.path.join(tmp, "dump.npz")
        np.savez(
            npz_path,
            sample_ids=np.array([100, 101, 102], dtype=np.int64),
            delta_rank=np.array([10, 0, -5]),
            delta_rank_swap=np.array([8, 0, -3]),
            condition_drift=np.array([0.5, 0.1, 0.2]),
            embedding_shift=np.array([0.05, 0.01, 0.02]),
        )
        result = correlate_polysemy_with_retrieval(labels4, sample_ids4, npz_path)
    assert result["n_joined"] == 3, result
    assert result["bridge"]["n"] == 1 and result["bridge"]["median_abs_delta_rank"] == 10.0, result
    assert result["neither"]["n"] == 1  # only sample 101 (id 103 was excluded, not in the dump)
    assert "corr_is_polysemic_vs_abs_delta_rank" in result, result
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `NameError: name 'correlate_polysemy_with_retrieval' is not defined`.

- [ ] **Step 3: Implement the function**

Add above `main()`:

```python
def correlate_polysemy_with_retrieval(
    labels: np.ndarray, sample_ids: List[int], npz_path: str
) -> dict:
    """Join the per-node polysemy label (row-aligned to sample_ids, this script's own
    FeatureManager-order labeling) against Experiment 11.2's per-sample retrieval-rank/
    drift dump (Task 2's .npz, keyed by actual sample id -- a different population, the
    training rows of one specific trained run, so only the intersection is used). Reports
    per-label median |delta_rank|/condition_drift/embedding_shift, plus Spearman
    correlations between "is this sample polysemic at all" and each retrieval-side metric.
    """
    data = np.load(npz_path)
    dump_ids = data["sample_ids"].tolist()
    id_to_row = {sid: row for row, sid in enumerate(sample_ids)}
    keep = [i for i, sid in enumerate(dump_ids) if sid in id_to_row]
    rows = np.array([id_to_row[dump_ids[i]] for i in keep], dtype=np.int64)

    label_kept = labels[rows]
    delta_rank = data["delta_rank"][keep].astype(float)
    condition_drift = data["condition_drift"][keep].astype(float)
    embedding_shift = data["embedding_shift"][keep].astype(float)
    is_polysemic = (label_kept != "neither").astype(float)

    result: dict = {"n_joined": len(keep)}
    for lbl in ("neither", "img_only_only", "txt_only_only", "bridge"):
        mask = label_kept == lbl
        if mask.sum() == 0:
            continue
        result[lbl] = {
            "n": int(mask.sum()),
            "median_abs_delta_rank": float(np.median(np.abs(delta_rank[mask]))),
            "median_condition_drift": float(np.median(condition_drift[mask])),
            "median_embedding_shift": float(np.median(embedding_shift[mask])),
        }
    result["corr_is_polysemic_vs_abs_delta_rank"] = spearman_correlate(is_polysemic, np.abs(delta_rank))
    result["corr_is_polysemic_vs_condition_drift"] = spearman_correlate(is_polysemic, condition_drift)
    result["corr_is_polysemic_vs_embedding_shift"] = spearman_correlate(is_polysemic, embedding_shift)
    return result
```

- [ ] **Step 4: Run the selftest again to verify it passes**

```bash
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `SELFTEST OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add polysemy-label vs retrieval-rank/drift correlation (Experiment 12)"
```

---

### Task 7: Wire up `run()`/`main()` end-to-end, smoke-test against real data

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py`

**Interfaces:**
- Consumes: `build_buddy_graphs` (`src.conditional_buddy.buddy_graph`), `classify_edges`/`bridge_node_stats` (Task 1), every pure function from Tasks 3–6, `FeatureManager` (`src.utils`).
- Produces: `run(storage_dir: str, template_dir: str, K: int = 30, alpha: float = 0.5, n_bridge_sample: int = 5000, seed: int = 0, device: str = "cuda", per_sample_npz: str = None) -> dict`, and a `main()` CLI. Consumed by Task 8.

- [ ] **Step 1: Implement `run()` and the real CLI**

Replace `main()`'s body (the `ap.print_help()` branch) is kept, but first add, above `main()`:

```python
def _load_features(storage_dir: str):
    from src.utils import FeatureManager

    fm = FeatureManager(storage_dir)
    data = fm.load_all_to_ram(["img_features", "txt_features"])
    img = data["img_features"].numpy().astype(np.float32)
    txt = data["txt_features"].numpy().astype(np.float32)
    sample_ids = [int(s) for s in data["sample_ids"].tolist()]
    return img, txt, sample_ids


def run(
    storage_dir: str,
    template_dir: str,
    K: int = 30,
    alpha: float = 0.5,
    n_bridge_sample: int = 5000,
    seed: int = 0,
    device: str = "cuda",
    per_sample_npz: str = None,
) -> dict:
    """End-to-end Experiment 12 pass: rebuild the buddy graph from cached features,
    classify its edges, sample bridge-node (A, B, C) triples, measure whether the
    ALREADY-SAVED buddy-init embedding (template_dir) pulls B and C together vs. a
    degree-matched baseline, check whether that pull is graded by shared-neighbor
    structure, and (if per_sample_npz is given) cross-reference the per-node polysemy
    label against Experiment 11.2's per-sample retrieval-rank/drift dump."""
    from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges
    from src.conditional_buddy.compute_buddies import _l2_normalize, build_buddy_graphs

    img, txt, sample_ids = _load_features(storage_dir)
    img_n, txt_n = _l2_normalize(img), _l2_normalize(txt)
    A_img, A_txt, E = build_buddy_graphs(img_n, txt_n, K=K, alpha=alpha, device=device)

    template_ids = np.load(Path(template_dir) / "sample_ids.npy").tolist()
    assert template_ids == sample_ids, (
        "template_dir's sample_ids.npy must match the freshly-loaded feature store's "
        "sample order exactly (CLAUDE.md's sample-id-consistency rule) -- do not proceed "
        "past this assertion if it fires; it means the wrong template/feature-store pair "
        "was passed"
    )
    emb = np.load(Path(template_dir) / "embeddings.npy")

    N = len(sample_ids)
    typed = classify_edges(A_img, A_txt, E, N)
    bstats = bridge_node_stats(typed, N)
    labels = label_nodes(bstats)
    E_img_only, E_txt_only = build_typed_adjacency(typed, N)

    rng = np.random.default_rng(seed)
    pairs = extract_bridge_pairs(bstats, E_img_only, E_txt_only, n_bridge_sample, rng)
    buckets = degree_deciles(E)
    baselines = sample_baselines(pairs, E, buckets, rng)
    valid = baselines >= 0
    pairs, baselines = pairs[valid], baselines[valid]

    a_idx, b_idx, c_idx = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    dist_bc = embedded_l2_distance(emb, b_idx, c_idx)
    dist_bc_baseline = embedded_l2_distance(emb, b_idx, baselines)
    jaccard = shared_neighbor_jaccard(E, b_idx, c_idx)
    pull_summary = paired_pull_summary(dist_bc, dist_bc_baseline)
    grading_corr = spearman_correlate(jaccard, dist_bc_baseline - dist_bc)

    result = {
        "n_bridge_nodes": bstats["n_bridge_nodes"],
        "frac_bridge_nodes": bstats["frac_bridge_nodes"],
        "n_pairs_sampled": int(len(pairs)),
        "pull_summary": pull_summary,
        "grading_corr_jaccard_vs_pull": grading_corr,
        "label_counts": {lbl: int((labels == lbl).sum())
                         for lbl in ("neither", "img_only_only", "txt_only_only", "bridge")},
    }
    if per_sample_npz is not None:
        result["retrieval_correlation"] = correlate_polysemy_with_retrieval(
            labels, sample_ids, per_sample_npz
        )
    return result
```

Then replace `main()` entirely with:

```python
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage-dir", default="/data/SSD2/pre_extract/redcaps_150k/features")
    ap.add_argument("--template-dir",
                    default="res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n-bridge-sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-sample-npz", default=None,
                    help="path to a Task 2 --dump-per-sample .npz for the retrieval-rank/"
                         "drift cross-reference (optional)")
    ap.add_argument("--out", default=None, help="write the JSON result here (optional)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    result = run(
        storage_dir=args.storage_dir, template_dir=args.template_dir, K=args.K,
        alpha=args.alpha, n_bridge_sample=args.n_bridge_sample, seed=args.seed,
        device=args.device, per_sample_npz=args.per_sample_npz,
    )

    print(f"\n{'='*78}\nExperiment 12 - cross-modal polysemy bridge-node diagnostic\n{'='*78}")
    print(f"  bridge nodes: {result['n_bridge_nodes']:,} ({100*result['frac_bridge_nodes']:.1f}% of nodes)")
    print(f"  label counts: {result['label_counts']}")
    print(f"  sampled bridge pairs: {result['n_pairs_sampled']:,}")
    ps = result["pull_summary"]
    sig = f"  mean/SEM={ps['z']:+.1f}{' *' if ps['z'] == ps['z'] and abs(ps['z']) >= 2 else ''}" if ps["n"] > 1 else ""
    print(f"  pull (baseline_dist - bc_dist): mean={ps['mean']:+.4f} (n={ps['n']}, "
          f"frac_pulled_closer={ps['frac_pulled_closer']:.3f}){sig}")
    gc = result["grading_corr_jaccard_vs_pull"]
    print(f"  grading check: corr(shared_neighbor_jaccard, pull) rho={gc['rho']:+.3f} p={gc['p']:.3e}")
    if "retrieval_correlation" in result:
        rc = result["retrieval_correlation"]
        print(f"  retrieval cross-reference (n_joined={rc['n_joined']}):")
        for lbl in ("neither", "img_only_only", "txt_only_only", "bridge"):
            if lbl in rc:
                print(f"    {lbl}: n={rc[lbl]['n']} median|delta_rank|={rc[lbl]['median_abs_delta_rank']:.1f} "
                      f"median_drift={rc[lbl]['median_condition_drift']:.4f}")
        c1 = rc["corr_is_polysemic_vs_abs_delta_rank"]
        print(f"    corr(is_polysemic, |delta_rank|): rho={c1['rho']:+.3f} p={c1['p']:.3e}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the selftest once more to confirm nothing broke**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_polysemy_bridges.py --selftest
```

Expected: `SELFTEST OK`.

- [ ] **Step 3: Smoke-test against real cached data at small scale**

```bash
python scripts/analyze_polysemy_bridges.py --n-bridge-sample 50 --device cuda
```

Expected: no traceback; prints bridge-node count/fraction (should roughly match the ~80% figure Experiment 10's diagnostic already found on this same dataset), `n_pairs_sampled` at or near 50, and a `pull_summary`/`grading_corr` line. If the `template_ids == sample_ids` assertion in `run()` fires, stop — it means `--storage-dir`/`--template-dir` don't point at the same underlying dataset/config the template was built from; do not proceed to Task 8 until this passes cleanly. **Use `--device cuda` (the CLI default) here, not `cpu`** — `n_bridge_sample` only caps how many bridge *pairs* get sampled after the graph is built; it does NOT shrink the mutual-kNN graph-construction cost, which is a brute-force O(N²) pass over the *full* 150k feature set regardless of this flag (`mutual_knn`'s module docstring, `src/conditional_buddy/buddy_graph.py`) — confirmed impractically slow on CPU (42+ minutes with no output) when this was actually run without a GPU.

- [ ] **Step 4: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: wire up analyze_polysemy_bridges.py end-to-end CLI (Experiment 12)"
```

---

### Task 8: Run the real analysis and write up results

**Files:**
- Create: `docs/reports/2026-08-26_polysemy_bridge_diagnostic.md`

**Interfaces:**
- Consumes: Task 7's `run()`/CLI, Task 2's `--dump-per-sample` flag.
- Produces: a results report other tasks/readers can cite, following this project's existing `docs/reports/YYYY-MM-DD_*.md` convention.

- [ ] **Step 1: Produce the per-sample retrieval-correlation dump from an existing 11.1 pair**

Pick one already-completed same-seed frozen/trained pair from Experiment 11.1 (see `docs/reports/2026-08-25_condition_freeze_ablation.md`'s run manifest for the exact `run dir`s) and re-run 11.2's analysis with the new flag:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_retrieval_correlation.py --pair <11.1_frozen_run_dir> <11.1_trained_run_dir> --dump-per-sample
```

Note the printed `Wrote per-sample dump: <trained_run_dir>/condition_geometry/per_sample_retrieval_correlation.npz` path — pass it as `--per-sample-npz` below.

- [ ] **Step 2: Run the real Experiment 12 analysis and capture its output**

```bash
python scripts/analyze_polysemy_bridges.py \
  --n-bridge-sample 5000 --device cuda \
  --per-sample-npz <trained_run_dir>/condition_geometry/per_sample_retrieval_correlation.npz \
  --out res/CoSiR_condition_freeze_ablation/redcaps_150k/polysemy_bridges_summary.json
```

Save the full printed output — it is the source of every number in Step 3.

- [ ] **Step 3: Write the results report**

Create `docs/reports/2026-08-26_polysemy_bridge_diagnostic.md`:

```markdown
# Cross-modal polysemy: does the buddy graph already reflect it, and is any B-C pull real signal or a "false transitivity" artifact?

**Date:** 2026-08-26 · **Dataset:** RedCaps, 150,000 rows (matches C5/Exp 9-11's scale) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/analyze_polysemy_bridges.py`, `src/conditional_buddy/buddy_graph.py` (`classify_edges`/`bridge_node_stats`)
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 12
**Precursor:** `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py` (found ~80% of nodes are "bridge" nodes; this experiment measures what that structurally implies for the resulting embedding)

---

## TL;DR

[One paragraph, filled from Step 2's actual output: state the bridge-node fraction found, the
pull_summary's mean/SEM (seed-replicated language does NOT apply here -- this is a
pair-count-based statistic over N sampled bridge-pairs from one deterministic graph, not a
training-seed comparison), whether the grading correlation (shared-neighbor Jaccard vs. pull)
is significant and in the expected direction, and which of the spec's three named branches
applies: real+graded / real+ungraded ("false transitivity" confirmed) / null.]

## Method

[1-2 sentences: reused already-completed buddy-init template
(`res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings/`), rebuilt A_img/A_txt/E
from cached features (K=30, alpha=0.5) to classify edges, sampled up to 5000 bridge-node (A,B,C)
triples, compared embedded B-C distance against a degree-matched baseline, and checked whether
the pull correlates with shared-neighbor Jaccard.]

## Results

### Bridge structure

[n_bridge_nodes, frac_bridge_nodes, label_counts from Step 2's output -- compare frac_bridge_nodes
against Experiment 10's own ~80% figure as a cross-check that the rebuilt graph matches.]

### False-transitivity audit

[pull_summary's n/mean/std/sem/z, frac_pulled_closer, and the grading_corr_jaccard_vs_pull
rho/p -- state plainly whether B-C are pulled together, and whether that pull tracks real
shared-neighbor overlap or looks arbitrary.]

### Retrieval/drift cross-reference

[Per-label median |delta_rank|/condition_drift/embedding_shift table, and the
corr_is_polysemic_vs_* rho/p values, from Step 2's "retrieval cross-reference" section --
state whether being polysemic (of either type) predicts anything.]

## Interpretation

[Which of the spec's three named branches applies (real+graded / real+ungraded / null), and what
it means for the paper's account of buddy-init: does this reinforce "buddy-init geometry alone is
already fine" (11.1-11.3's throughline), or does it identify a genuine limitation worth a
follow-up modality-aware representation experiment (not committed here, per the spec)?]

## Caveats

- This is a single-graph diagnostic (one buddy-init template, not repeated across training
  seeds) — statistical significance here comes from the number of sampled bridge-pairs, not
  from seed replication. This matches the precedent of Experiment 9's and Experiment 10's own
  diagnostic scripts, which are likewise single-graph analyses.
- The retrieval/drift cross-reference uses one specific 11.1 trained run's per-sample dump
  (Step 1); it is in-sample and own-condition, same scoping caveat 11.2's own report already
  states for that population.

## Reproduction

\`\`\`bash
python scripts/analyze_condition_retrieval_correlation.py --pair <frozen_dir> <trained_dir> --dump-per-sample
python scripts/analyze_polysemy_bridges.py --n-bridge-sample 5000 --device cuda \
  --per-sample-npz <trained_dir>/condition_geometry/per_sample_retrieval_correlation.npz
\`\`\`
```

Fill in every bracketed section using Step 2's actual output — do not leave any bracket placeholder in the committed file.

- [ ] **Step 4: Commit**

```bash
git add docs/reports/2026-08-26_polysemy_bridge_diagnostic.md res/CoSiR_condition_freeze_ablation/redcaps_150k/polysemy_bridges_summary.json
git commit -m "results: cross-modal polysemy bridge-node diagnostic (Experiment 12)"
```

---

## Self-review

- **Spec coverage:** Task 1 implements the spec's promotion of `classify_edges`/`bridge_node_stats`. Task 3 implements the per-node polysemy label. Tasks 4-5 implement the bridge-pair extraction, degree-matched baseline, and shared-neighbor-proximity grading check (Q1/Q3, the false-transitivity audit). Task 2/6 implement the retrieval/drift cross-reference (Q2). Task 7 wires it end-to-end with a real-data smoke test. Task 8 runs the real analysis and produces the report deliverable the spec's §7 "Results reports" item requires.
- **Placeholder scan:** every code block is complete and runnable as written; Task 8's report template brackets are explicitly called out as required-to-fill, not left-in placeholders.
- **Type consistency:** `label_nodes(bridge_stats) -> np.ndarray`, `build_typed_adjacency(typed, N) -> Tuple[csr_matrix, csr_matrix]`, `extract_bridge_pairs(bridge_stats, E_img_only, E_txt_only, n_sample, rng) -> np.ndarray`, `sample_baselines(pairs, E, buckets, rng, max_tries) -> np.ndarray`, `shared_neighbor_jaccard`, `embedded_l2_distance`, `paired_pull_summary`, `correlate_polysemy_with_retrieval`, and `run(...)` all use identical names/argument order everywhere they're called across Tasks 3-8.
