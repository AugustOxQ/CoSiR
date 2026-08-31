# Experiment 14: Closed-Triangle Bridge Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Experiment 12/C10's bridge-node "false transitivity" pull statistic a positive control by sampling **closed triangles** (a hub node's two text-only neighbors that are *also* directly connected by a real image-only edge) alongside the existing **open** hub pairs (no such direct edge — structurally identical to Experiment 12's B/C), and compare pull magnitude between them.

**Architecture:** Purely additive extensions to two existing, already-tested modules — `src/conditional_buddy/buddy_graph.py` (graph-topology functions) and `scripts/analyze_polysemy_bridges.py` (sampling/statistics/CLI). No existing function's behavior or return shape changes; every new parameter defaults to preserving current behavior exactly. Zero new training or feature extraction — everything operates on the already-built RedCaps-150k buddy graph, one already-completed buddy-init template, and the 6 already-on-disk per-sample `.npz` dumps from Experiment 12.3.

**Tech Stack:** Python, NumPy, SciPy sparse (`csr_matrix`), the project's existing `FeatureManager`/`build_buddy_graphs` pipeline. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, "### Experiment 14" section (commit `ab565d9`).

## Global Constraints

- No new training or eval runs anywhere in this experiment (spec: Scope/Tooling).
- No worktree isolation — implement directly on `experiment/condition_drift_retrieval_correlation` (spec: this is graph/script-only work, not a model/training change).
- Every new function/parameter is additive: `classify_edges`, `bridge_node_stats`, `label_nodes`, `correlate_polysemy_with_retrieval`, and `pool_cross_references`'s existing behavior and existing tests must keep passing byte-for-byte unchanged (spec: Tooling).
- RedCaps-150k first; escalate to the already-extracted RedCaps-300k feature store only if Task 6's incidence count is too sparse (spec: Scope).
- Activate the project's conda env before running any Python: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` (CLAUDE.md).
- Judge the primary success criterion on pull **magnitude**, not just `mean/SEM` significance (spec: Success criteria — this project's own standing caution that mean/SEM inflates with sample size).

---

### Task 1: `hub_neighbor_pairs()` in `buddy_graph.py`

**Files:**
- Modify: `src/conditional_buddy/buddy_graph.py:24` (import line), append new function after line 496 (end of file)
- Modify: `src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py:17` (import line), insert new test before line 88 (`if __name__ == "__main__":` block), and add the new test's call into that block

**Interfaces:**
- Consumes: `classify_edges(A_img, A_txt, E, N) -> dict` (existing, `buddy_graph.py:449`) and `bridge_node_stats(typed, N) -> dict` (existing, `buddy_graph.py:467`), specifically its `"deg_txt_only": np.ndarray[int64, N]` field.
- Produces: `hub_neighbor_pairs(typed: dict, bridge_stats: dict, N: int) -> dict` returning `{"hub": np.ndarray[int64, M], "c": np.ndarray[int64, M], "d": np.ndarray[int64, M], "is_closed": np.ndarray[bool, M]}` — one row per unordered pair of a hub node's (`deg_txt_only >= 2`) text-only neighbors. `is_closed[i]` is `True` iff `(c[i], d[i])` is also a real `img_only` edge. Tasks 2, 4, and 5 consume this directly.

- [ ] **Step 1: Write the failing test**

In `src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py`, change line 17's import to:

```python
from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges, hub_neighbor_pairs
```

Then insert this new test function after `test_img_only_and_txt_only_neighbor_sets_are_disjoint` (i.e. after line 85, before the blank line and `if __name__ == "__main__":` block):

```python
def test_hub_neighbor_pairs_closed_vs_open():
    """A hub node (>=2 txt_only neighbors) whose neighbor-pair is ALSO connected by a
    real img_only edge is a 'closed triangle' (Experiment 14's positive-control case);
    a hub neighbor-pair with no such direct edge is 'open' (structurally identical to
    Experiment 12's B/C bridge pair, just sourced from a >=2-neighbor hub). A node with
    only 1 txt_only neighbor is a bridge but not a hub and contributes no pairs."""
    n = 8
    # Node 1: hub, txt_only neighbors 2 and 5; 2-5 is ALSO a real img_only edge -> closed.
    # Node 6: hub, txt_only neighbors 3 and 4; no edge between 3-4 -> open.
    # Node 0: bridge but NOT a hub (only 1 txt_only neighbor, to node 7) -> contributes no pairs.
    A_img = _csr(n, [(0, 1), (2, 5)])
    A_txt = _csr(n, [(1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])
    E = _csr(n, [(0, 1), (2, 5), (1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])

    typed = classify_edges(A_img, A_txt, E, n)
    stats = bridge_node_stats(typed, n)
    assert stats["deg_txt_only"][1] == 2
    assert stats["deg_txt_only"][6] == 2
    assert stats["deg_txt_only"][0] == 1  # bridge, but not a hub

    result = hub_neighbor_pairs(typed, stats, n)
    found = {
        (int(h), frozenset((int(c), int(d)))): bool(closed)
        for h, c, d, closed in zip(result["hub"], result["c"], result["d"], result["is_closed"])
    }
    assert found[(1, frozenset((2, 5)))] is True, found  # closed triangle
    assert found[(6, frozenset((3, 4)))] is False, found  # open pair
    assert len(found) == 2, found  # node 0 (deg_txt_only=1) contributes nothing
    print("PASS test_hub_neighbor_pairs_closed_vs_open")
```

And update the `if __name__ == "__main__":` block (lines 88-93) to:

```python
if __name__ == "__main__":
    test_classify_edges_buckets_correctly()
    test_bridge_node_detection()
    test_img_only_and_txt_only_neighbor_sets_are_disjoint()
    test_hub_neighbor_pairs_closed_vs_open()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py`
Expected: FAIL with `ImportError: cannot import name 'hub_neighbor_pairs'`

- [ ] **Step 3: Write minimal implementation**

In `src/conditional_buddy/buddy_graph.py`, change line 24's import to:

```python
from typing import Dict, List, Tuple
```

Then append this function at the end of the file (after `bridge_node_stats`, line 496):

```python


def hub_neighbor_pairs(typed: dict, bridge_stats: dict, N: int) -> dict:
    """For every 'hub' node (>=2 txt_only neighbors, Experiment 14), enumerate all
    unordered pairs of its txt_only neighbors and label each pair 'closed' (the pair is
    ALSO connected by a real img_only edge -- a closed triangle: hub--C txt_only,
    hub--D txt_only, C--D img_only) or 'open' (no such direct edge -- structurally
    identical to Experiment 12's B/C bridge pair, but sourced from a >=2-txt-neighbor
    hub instead of a single-txt-neighbor bridge). A node can contribute to multiple
    pairs if it has 3+ txt_only neighbors.

    Returns {"hub": int64 (M,), "c": int64 (M,), "d": int64 (M,), "is_closed": bool (M,)}."""
    keys = typed["keys"]
    i = (keys // N).astype(np.int64)
    j = (keys % N).astype(np.int64)
    txt_i, txt_j = i[typed["txt_only"]], j[typed["txt_only"]]
    img_keys = np.sort(keys[typed["img_only"]])

    neighbors: Dict[int, List[int]] = {}
    for a, b in zip(txt_i.tolist(), txt_j.tolist()):
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    hub_ids = np.where(bridge_stats["deg_txt_only"] >= 2)[0]
    hubs, cs, ds = [], [], []
    for hub in hub_ids.tolist():
        nbrs = neighbors.get(hub, [])
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                hubs.append(hub)
                cs.append(nbrs[x])
                ds.append(nbrs[y])

    if not hubs:
        empty = np.zeros(0, dtype=np.int64)
        return {"hub": empty, "c": empty, "d": empty, "is_closed": np.zeros(0, dtype=bool)}

    hub_arr = np.array(hubs, dtype=np.int64)
    c_arr = np.array(cs, dtype=np.int64)
    d_arr = np.array(ds, dtype=np.int64)
    lo = np.minimum(c_arr, d_arr)
    hi = np.maximum(c_arr, d_arr)
    pair_keys = lo * N + hi
    is_closed = np.isin(pair_keys, img_keys)

    return {"hub": hub_arr, "c": c_arr, "d": d_arr, "is_closed": is_closed}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py`
Expected: `PASS test_hub_neighbor_pairs_closed_vs_open` then `ALL TESTS PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conditional_buddy/buddy_graph.py src/test/20260826_polysemy_bridges/test_buddy_graph_bridge_functions.py
git commit -m "feat: add hub_neighbor_pairs for Experiment 14 closed-triangle detection"
```

---

### Task 2: `extract_hub_pairs()` and `closed_triangle_membership()` in `analyze_polysemy_bridges.py`

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py` — insert `extract_hub_pairs()` after `extract_bridge_pairs()` (after line 165), insert `closed_triangle_membership()` after `label_nodes()` (after line 181), extend `_selftest()` (before `print("SELFTEST OK")`, line 313), extend the import inside `_selftest()` at line 197.

**Interfaces:**
- Consumes: `hub_neighbor_pairs()` from Task 1.
- Produces: `extract_hub_pairs(typed: dict, bridge_stats: dict, N: int, n_sample_per_group: int, rng: np.random.Generator) -> dict` returning `{"pairs": np.ndarray[int64, (M, 3)]` (columns `[hub, c, d]`)`, "is_closed": np.ndarray[bool, M]}`. `closed_triangle_membership(hub_pairs: dict, N: int) -> Tuple[np.ndarray, np.ndarray]` returning `(in_closed_triangle: bool[N], in_open_hub_pair: bool[N])`, taking the *raw* (unsampled) output of `hub_neighbor_pairs`. Task 5 consumes both.

- [ ] **Step 1: Write the failing test**

In `scripts/analyze_polysemy_bridges.py`, change line 197's import (inside `_selftest()`) to:

```python
    from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges, hub_neighbor_pairs
```

Then insert this block into `_selftest()`, right before the final `print("SELFTEST OK")` (line 313):

```python
    # extract_hub_pairs + closed_triangle_membership: reuse the same synthetic
    # closed/open hub structure as buddy_graph's own hub_neighbor_pairs test.
    n8 = 8
    A_img8 = _csr(n8, [(0, 1), (2, 5)])
    A_txt8 = _csr(n8, [(1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])
    E8 = _csr(n8, [(0, 1), (2, 5), (1, 2), (1, 5), (6, 3), (6, 4), (7, 0)])
    typed8 = classify_edges(A_img8, A_txt8, E8, n8)
    bstats8 = bridge_node_stats(typed8, n8)
    raw_hub_pairs = hub_neighbor_pairs(typed8, bstats8, n8)
    assert len(raw_hub_pairs["hub"]) == 2, raw_hub_pairs  # one closed pair, one open pair

    rng8 = np.random.default_rng(2)
    sampled = extract_hub_pairs(typed8, bstats8, n8, n_sample_per_group=10, rng=rng8)
    assert sampled["pairs"].shape == (2, 3), sampled["pairs"].shape
    assert set(sampled["is_closed"].tolist()) == {True, False}, sampled["is_closed"]

    in_closed, in_open = closed_triangle_membership(raw_hub_pairs, n8)
    assert in_closed[2] and in_closed[5], in_closed  # the closed triangle's C/D
    assert not in_closed[3] and not in_closed[4], in_closed  # the open pair's C/D
    assert in_open[3] and in_open[4], in_open
    assert not in_open[2] and not in_open[5], in_open
    print("PASS extract_hub_pairs + closed_triangle_membership")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --selftest`
Expected: FAIL with `NameError: name 'extract_hub_pairs' is not defined`

- [ ] **Step 3: Write minimal implementation**

Insert after `extract_bridge_pairs` (after line 165, before `def label_nodes`):

```python


def extract_hub_pairs(
    typed: dict,
    bridge_stats: dict,
    N: int,
    n_sample_per_group: int,
    rng: np.random.Generator,
) -> dict:
    """Experiment 14: sample up to n_sample_per_group closed-triangle pairs and up to
    n_sample_per_group open hub pairs INDEPENDENTLY (see buddy_graph.hub_neighbor_pairs
    for the closed/open definition), so a sparse closed-triangle population isn't
    crowded out by open pairs if both groups exist in very different numbers.

    Returns {"pairs": int64 (M, 3) array of columns [hub, c, d], "is_closed": bool (M,)}."""
    from src.conditional_buddy.buddy_graph import hub_neighbor_pairs

    raw = hub_neighbor_pairs(typed, bridge_stats, N)
    if len(raw["hub"]) == 0:
        return {"pairs": np.zeros((0, 3), dtype=np.int64), "is_closed": np.zeros(0, dtype=bool)}

    pairs_all = np.stack([raw["hub"], raw["c"], raw["d"]], axis=1)
    is_closed_all = raw["is_closed"]

    kept_pairs, kept_closed = [], []
    for want_closed in (True, False):
        idx = np.where(is_closed_all == want_closed)[0]
        if len(idx) > n_sample_per_group:
            idx = rng.choice(idx, size=n_sample_per_group, replace=False)
        kept_pairs.append(pairs_all[idx])
        kept_closed.append(np.full(len(idx), want_closed, dtype=bool))

    return {
        "pairs": np.concatenate(kept_pairs, axis=0),
        "is_closed": np.concatenate(kept_closed, axis=0),
    }
```

Insert after `label_nodes` (after line 181, before `def _selftest():`):

```python


def closed_triangle_membership(hub_pairs: dict, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node boolean flags derived from hub_neighbor_pairs' RAW (unsampled) output:
    in_closed_triangle is True for any node appearing as the c/d endpoint of at least
    one closed pair; in_open_hub_pair is True for any node appearing as the c/d endpoint
    of at least one open pair. NOT mutually exclusive with each other, or with
    label_nodes' categories -- a node with 3+ txt_only neighbors can be in both groups."""
    in_closed = np.zeros(N, dtype=bool)
    in_open = np.zeros(N, dtype=bool)
    closed_mask = hub_pairs["is_closed"]
    for endpoints in (hub_pairs["c"], hub_pairs["d"]):
        in_closed[endpoints[closed_mask]] = True
        in_open[endpoints[~closed_mask]] = True
    return in_closed, in_open
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `PASS extract_hub_pairs + closed_triangle_membership` then `SELFTEST OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add extract_hub_pairs and closed_triangle_membership for Experiment 14"
```

---

### Task 3: `extra_flags` support in `correlate_polysemy_with_retrieval()` and `pool_cross_references()`

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py:316-356` (`correlate_polysemy_with_retrieval`), `scripts/analyze_polysemy_bridges.py:359-376` (`pool_cross_references`), and `_selftest()` (before `print("SELFTEST OK")`, line 313 — insert before Task 2's new block so both additions coexist; order between them doesn't matter).

**Interfaces:**
- Consumes: nothing new from other tasks — this task is independently testable with synthetic flags.
- Produces: `correlate_polysemy_with_retrieval(labels, sample_ids, npz_path, return_raw=False, extra_flags: dict = None)` — when `extra_flags` (a `{name: np.ndarray[bool or 0/1, N_full_population]}` dict) is given, adds `result[f"corr_{name}_vs_abs_delta_rank"]`, `result[f"corr_{name}_vs_delta_rank"]`, and `result[f"{name}_true"]` (group stats) per flag, on top of the existing unchanged output. `pool_cross_references(results, tags, extra_flag_names: List[str] = None)` pools those same extra keys across runs the same way it already pools `corr_is_polysemic_vs_*`. Task 5 wires both into `run()`.

- [ ] **Step 1: Write the failing test**

Insert into `_selftest()`, before `print("SELFTEST OK")` (line 313):

```python
    # correlate_polysemy_with_retrieval + pool_cross_references: extra_flags param
    # (Experiment 14's is_hub/in_closed_triangle/in_open_hub_pair cross-references).
    extra_flags4 = {"is_hub": np.array([True, False, True, False])}
    with tempfile.TemporaryDirectory() as tmp:
        npz_path4 = os.path.join(tmp, "dump4.npz")
        np.savez(
            npz_path4,
            sample_ids=np.array([100, 101, 102], dtype=np.int64),
            delta_rank=np.array([10, 0, -5]),
            delta_rank_swap=np.array([8, 0, -3]),
            condition_drift=np.array([0.5, 0.1, 0.2]),
            embedding_shift=np.array([0.05, 0.01, 0.02]),
        )
        result_flagged = correlate_polysemy_with_retrieval(
            labels4, sample_ids4, npz_path4, extra_flags=extra_flags4
        )
    assert "corr_is_hub_vs_delta_rank" in result_flagged, result_flagged
    assert "corr_is_hub_vs_abs_delta_rank" in result_flagged, result_flagged
    assert result_flagged["is_hub_true"]["n"] == 2, result_flagged  # sample ids 100, 102

    r1f = {**r1, "corr_is_hub_vs_abs_delta_rank": {"rho": 0.30, "p": 0.01},
           "corr_is_hub_vs_delta_rank": {"rho": 0.05, "p": 0.5}}
    r2f = {**r2, "corr_is_hub_vs_abs_delta_rank": {"rho": 0.50, "p": 0.01},
           "corr_is_hub_vs_delta_rank": {"rho": -0.05, "p": 0.5}}
    pooled_flagged = pool_cross_references(
        [r1f, r2f], tags=["trained/seed1", "trained/seed2"], extra_flag_names=["is_hub"]
    )
    assert "corr_is_hub_vs_abs_delta_rank" in pooled_flagged["pooled"], pooled_flagged
    assert abs(pooled_flagged["pooled"]["corr_is_hub_vs_abs_delta_rank"]["mean"] - 0.40) < 1e-9, pooled_flagged
    print("PASS correlate_polysemy_with_retrieval/pool_cross_references extra_flags")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --selftest`
Expected: FAIL with `TypeError: correlate_polysemy_with_retrieval() got an unexpected keyword argument 'extra_flags'`

- [ ] **Step 3: Write minimal implementation**

Change the `correlate_polysemy_with_retrieval` signature (line 316-318) to:

```python
def correlate_polysemy_with_retrieval(
    labels: np.ndarray, sample_ids: List[int], npz_path: str, return_raw: bool = False,
    extra_flags: dict = None,
) -> dict:
```

Insert this block right after the existing `result["corr_is_polysemic_vs_embedding_shift"] = ...` line (354) and before `if return_raw:` (355):

```python
    if extra_flags:
        for flag_name, flag_arr in extra_flags.items():
            flag_kept = flag_arr[rows].astype(float)
            result[f"corr_{flag_name}_vs_abs_delta_rank"] = spearman_correlate(flag_kept, np.abs(delta_rank))
            result[f"corr_{flag_name}_vs_delta_rank"] = spearman_correlate(flag_kept, delta_rank)
            mask_true = flag_kept > 0
            if mask_true.sum() > 0:
                result[f"{flag_name}_true"] = {
                    "n": int(mask_true.sum()),
                    "median_abs_delta_rank": float(np.median(np.abs(delta_rank[mask_true]))),
                    "median_delta_rank": float(np.median(delta_rank[mask_true])),
                }
```

Change `pool_cross_references`'s signature and loop (lines 359-376) to:

```python
def pool_cross_references(results: List[dict], tags: List[str], extra_flag_names: List[str] = None) -> dict:
    """Pool multiple already-computed correlate_polysemy_with_retrieval() results (one
    per run/seed) into a per-run table plus mean/std/sem/z of each run's own rho, across
    runs -- this project's standard multi-seed synthesis convention (see summarize() in
    scripts/analyze_condition_freeze_ablation.py). extra_flag_names pools the same
    corr_{name}_vs_{abs_}delta_rank keys correlate_polysemy_with_retrieval's extra_flags
    param produces (Experiment 14), on top of the always-present is_polysemic keys. Does
    not re-touch any per-sample data; purely aggregates already-computed per-run dicts."""
    assert len(results) == len(tags), (len(results), len(tags))
    per_run = {tag: r for tag, r in zip(tags, results)}
    corr_keys = ["corr_is_polysemic_vs_abs_delta_rank", "corr_is_polysemic_vs_delta_rank"]
    for flag in (extra_flag_names or []):
        corr_keys += [f"corr_{flag}_vs_abs_delta_rank", f"corr_{flag}_vs_delta_rank"]
    pooled = {}
    for corr_key in corr_keys:
        rhos = np.array([r[corr_key]["rho"] for r in results], dtype=float)
        n = len(rhos)
        mean = float(rhos.mean())
        std = float(rhos.std(ddof=1)) if n > 1 else float("nan")
        sem = std / np.sqrt(n) if n > 1 and std == std else float("nan")
        z = mean / sem if sem == sem and sem > 0 else float("nan")
        pooled[corr_key] = {"n": n, "mean": mean, "std": std, "sem": sem, "z": z}
    return {"n_runs": len(results), "per_run": per_run, "pooled": pooled}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `PASS correlate_polysemy_with_retrieval/pool_cross_references extra_flags` then `SELFTEST OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add extra_flags support to the retrieval cross-reference for Experiment 14"
```

---

### Task 4: `_build_typed_graph()` refactor, `count_hub_pairs()`, and `--counts-only` CLI (Task 0 spike tooling)

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py:414-496` (`run`, factor out its graph-construction prefix), append `count_hub_pairs()` after the new `_build_typed_graph()`, modify `main()` (lines 499-570).

**Interfaces:**
- Consumes: `hub_neighbor_pairs()` (Task 1), existing `_load_features`, `build_buddy_graphs`, `classify_edges`, `bridge_node_stats`.
- Produces: `_build_typed_graph(storage_dir: str, K: int, alpha: float, device: str) -> Tuple[dict, dict, List[int], csr_matrix]` returning `(typed, bridge_stats, sample_ids, E)`. `count_hub_pairs(storage_dir: str, K: int = 30, alpha: float = 0.5, device: str = "cuda") -> dict` returning `{"n_hub_nodes": int, "n_pairs_total": int, "n_closed": int, "n_open": int}`. Task 5 consumes `_build_typed_graph`.

- [ ] **Step 1: Write the failing test**

This task is CLI/integration-shaped (it needs the real cached RedCaps-150k feature store, not a synthetic graph), so its "test" is a manual dry run rather than a `_selftest()` addition — matching this script's own existing precedent (`run()`/`main()` have no selftest coverage either; they're exercised for real in Task 6). Confirm the current CLI has no `--counts-only` flag yet:

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --counts-only`
Expected: FAIL with `error: unrecognized arguments: --counts-only`

- [ ] **Step 2: Write the implementation**

Replace the graph-construction prefix of `run()` (lines 434-449, from `img, txt, sample_ids = _load_features(storage_dir)` through `bstats = bridge_node_stats(typed, N)`) by extracting it into a new function, inserted right before `def run(` (line 414):

```python
def _build_typed_graph(
    storage_dir: str, K: int, alpha: float, device: str,
) -> Tuple[dict, dict, List[int], csr_matrix]:
    """Rebuild the buddy graph from cached features and classify its edges -- the
    shared prefix of run() (Experiment 12) and count_hub_pairs() (Experiment 14's
    Task 0 incidence spike). Returns (typed, bridge_stats, sample_ids, E)."""
    from src.conditional_buddy.buddy_graph import bridge_node_stats, classify_edges
    from src.conditional_buddy.compute_buddies import _l2_normalize, build_buddy_graphs

    img, txt, sample_ids = _load_features(storage_dir)
    img_n, txt_n = _l2_normalize(img), _l2_normalize(txt)
    A_img, A_txt, E = build_buddy_graphs(img_n, txt_n, K=K, alpha=alpha, device=device)

    N = len(sample_ids)
    typed = classify_edges(A_img, A_txt, E, N)
    bstats = bridge_node_stats(typed, N)
    return typed, bstats, sample_ids, E


def count_hub_pairs(storage_dir: str, K: int = 30, alpha: float = 0.5, device: str = "cuda") -> dict:
    """Experiment 14, Task 0: cheap incidence count of hub nodes and closed-vs-open
    hub-neighbor pairs on the already-built buddy graph -- zero embedding/template
    loading, just the graph. Run this BEFORE committing to any real sampling; escalate
    to RedCaps-300k only if n_closed here is too sparse (see the plan's Task 6)."""
    from src.conditional_buddy.buddy_graph import hub_neighbor_pairs

    typed, bstats, _, _ = _build_typed_graph(storage_dir, K, alpha, device)
    raw = hub_neighbor_pairs(typed, bstats, len(bstats["deg_txt_only"]))
    return {
        "n_hub_nodes": int((bstats["deg_txt_only"] >= 2).sum()),
        "n_pairs_total": int(len(raw["hub"])),
        "n_closed": int(raw["is_closed"].sum()),
        "n_open": int((~raw["is_closed"]).sum()),
    }
```

Then replace `run()`'s old prefix (the block being extracted) with a call to the new helper — this is completed as part of Task 5's edit to `run()`, since `run()`'s body is touched again there; for this task, only `_build_typed_graph`/`count_hub_pairs` need to exist and be correct, `run()` itself is edited in Task 5.

Add the CLI flag in `main()` — insert after the `--K`/`--alpha` args (around line 505-506) and dispatch before building the rest of `run()`'s arguments (right after the existing `--selftest` dispatch block, lines 517-519):

```python
    ap.add_argument("--counts-only", action="store_true",
                     help="Experiment 14 Task 0: print hub/closed/open pair counts on the "
                          "already-built graph and exit -- no template/embedding loading, "
                          "no sampling. Run this before choosing --n-hub-sample or escalating "
                          "to a larger feature store.")
```

(add this line among the other `ap.add_argument(...)` calls, anywhere before `args = ap.parse_args()`), and right after the existing:

```python
    if args.selftest:
        _selftest()
        return
```

add:

```python
    if args.counts_only:
        counts = count_hub_pairs(storage_dir=args.storage_dir, K=args.K, alpha=args.alpha, device=args.device)
        print(f"\n{'='*78}\nExperiment 14 - hub/closed-triangle incidence count (Task 0)\n{'='*78}")
        print(f"  hub nodes (deg_txt_only >= 2): {counts['n_hub_nodes']:,}")
        print(f"  hub neighbor-pairs total: {counts['n_pairs_total']:,}")
        print(f"  closed (real img_only edge): {counts['n_closed']:,}")
        print(f"  open (no direct edge): {counts['n_open']:,}")
        return
```

- [ ] **Step 3: Run to verify the new flag is recognized**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --counts-only --storage-dir /data/SSD2/pre_extract/redcaps_150k/features`
Expected: prints the four count lines above (real numbers — this already IS Task 6's real Task-0 run; if this succeeds here, Task 6 can just reuse this exact command's output rather than re-running it)

- [ ] **Step 4: Run the full existing selftest to confirm no regressions**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `SELFTEST OK` (unchanged from Task 3)

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: add count_hub_pairs Task-0 incidence spike and --counts-only CLI flag"
```

---

### Task 5: Wire the closed-vs-open pull comparison into `run()`/`main()`

**Files:**
- Modify: `scripts/analyze_polysemy_bridges.py:414-496` (`run`, using `_build_typed_graph` from Task 4 and `extract_hub_pairs`/`closed_triangle_membership` from Tasks 2-3), `scripts/analyze_polysemy_bridges.py:499-570` (`main`, new `--n-hub-sample` arg and print output).

**Interfaces:**
- Consumes: `_build_typed_graph` (Task 4), `extract_hub_pairs`, `closed_triangle_membership` (Task 2), `correlate_polysemy_with_retrieval`'s `extra_flags` param, `pool_cross_references`'s `extra_flag_names` param (Task 3), `hub_neighbor_pairs` (Task 1).
- Produces: `run()`'s result dict gains `"hub_pair_counts"`, `"closed_triangle_pull"`, `"open_hub_pull"` keys; when `per_sample_npz` is given, its `"retrieval_correlation"` now also includes the `is_hub`/`in_closed_triangle`/`in_open_hub_pair` cross-reference keys. This is the final integration point — no downstream task consumes it, Task 6 runs it directly.

- [ ] **Step 1: Write the failing check**

This task's correctness is checked by the full self-test plus a real dry run against the smoke-scale (already-cached) 150k feature store, not a new synthetic unit test — `run()` is an integration function that already has no dedicated selftest of its own (only its component functions do, all covered in Tasks 1-3). Confirm the current `run()` does not yet expose the new result keys:

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -c "import scripts.analyze_polysemy_bridges as m; import inspect; print('n_hub_sample' in inspect.signature(m.run).parameters)"`
Expected: `False`

- [ ] **Step 2: Write the implementation**

Replace `run()`'s signature and body (lines 414-496) with:

```python
def run(
    storage_dir: str,
    template_dir: str,
    K: int = 30,
    alpha: float = 0.5,
    n_bridge_sample: int = 5000,
    n_hub_sample: int = 5000,
    seed: int = 0,
    device: str = "cuda",
    per_sample_npz=None,
    save_raw: str = None,
) -> dict:
    """End-to-end Experiment 12 + Experiment 14 pass: rebuild the buddy graph from
    cached features, classify its edges, sample bridge-node (A, B, C) triples AND
    hub-node closed/open (hub, C, D) pairs, measure whether the ALREADY-SAVED
    buddy-init embedding pulls each kind of pair together vs. a degree-matched
    baseline, check whether the bridge pull is graded by shared-neighbor structure,
    compare closed-triangle pull magnitude against open-hub pull magnitude (Experiment
    14's primary question), and (if per_sample_npz is given) cross-reference the
    per-node polysemy label AND the new is_hub/in_closed_triangle/in_open_hub_pair
    flags against Experiment 11.2/12.3's per-sample retrieval-rank/drift dump(s)."""
    from src.conditional_buddy.buddy_graph import hub_neighbor_pairs

    typed, bstats, sample_ids, E = _build_typed_graph(storage_dir, K, alpha, device)
    N = len(sample_ids)

    template_ids = np.load(Path(template_dir) / "sample_ids.npy").tolist()
    assert template_ids == sample_ids, (
        "template_dir's sample_ids.npy must match the freshly-loaded feature store's "
        "sample order exactly (CLAUDE.md's sample-id-consistency rule) -- do not proceed "
        "past this assertion if it fires; it means the wrong template/feature-store pair "
        "was passed"
    )
    emb = np.load(Path(template_dir) / "embeddings.npy")

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

    # Experiment 14: closed-triangle vs. open-hub-pair pull comparison.
    hub_pairs_raw = hub_neighbor_pairs(typed, bstats, N)
    hub_sample = extract_hub_pairs(typed, bstats, N, n_hub_sample, rng)
    hub_triples, hub_is_closed = hub_sample["pairs"], hub_sample["is_closed"]
    hub_baselines = sample_baselines(hub_triples, E, buckets, rng)
    hub_valid = hub_baselines >= 0
    hub_triples, hub_is_closed = hub_triples[hub_valid], hub_is_closed[hub_valid]
    hub_baselines = hub_baselines[hub_valid]
    hub_c, hub_d = hub_triples[:, 1], hub_triples[:, 2]
    dist_cd = embedded_l2_distance(emb, hub_c, hub_d)
    dist_cd_baseline = embedded_l2_distance(emb, hub_c, hub_baselines)
    closed_pull = paired_pull_summary(dist_cd[hub_is_closed], dist_cd_baseline[hub_is_closed])
    open_pull = paired_pull_summary(dist_cd[~hub_is_closed], dist_cd_baseline[~hub_is_closed])

    result = {
        "n_bridge_nodes": bstats["n_bridge_nodes"],
        "frac_bridge_nodes": bstats["frac_bridge_nodes"],
        "n_pairs_sampled": int(len(pairs)),
        "pull_summary": pull_summary,
        "grading_corr_jaccard_vs_pull": grading_corr,
        "label_counts": {lbl: int((labels == lbl).sum())
                         for lbl in ("neither", "img_only_only", "txt_only_only", "bridge")},
        "hub_pair_counts": {
            "n_hub_nodes": int((bstats["deg_txt_only"] >= 2).sum()),
            "n_pairs_total": int(len(hub_pairs_raw["hub"])),
            "n_closed_total": int(hub_pairs_raw["is_closed"].sum()),
            "n_open_total": int((~hub_pairs_raw["is_closed"]).sum()),
        },
        "closed_triangle_pull": closed_pull,
        "open_hub_pull": open_pull,
    }
    retrieval_raw = None
    if per_sample_npz is not None:
        in_closed, in_open = closed_triangle_membership(hub_pairs_raw, N)
        extra_flags = {
            "is_hub": bstats["deg_txt_only"] >= 2,
            "in_closed_triangle": in_closed,
            "in_open_hub_pair": in_open,
        }
        npz_list = [per_sample_npz] if isinstance(per_sample_npz, str) else list(per_sample_npz)
        if len(npz_list) == 1:
            if save_raw is not None:
                result["retrieval_correlation"], retrieval_raw = correlate_polysemy_with_retrieval(
                    labels, sample_ids, npz_list[0], return_raw=True, extra_flags=extra_flags,
                )
            else:
                result["retrieval_correlation"] = correlate_polysemy_with_retrieval(
                    labels, sample_ids, npz_list[0], extra_flags=extra_flags,
                )
        else:
            per_run_results = [
                correlate_polysemy_with_retrieval(labels, sample_ids, p, extra_flags=extra_flags)
                for p in npz_list
            ]
            tags = [Path(p).parent.parent.name for p in npz_list]
            result["retrieval_correlation"] = pool_cross_references(
                per_run_results, tags, extra_flag_names=list(extra_flags.keys()),
            )
    if save_raw is not None:
        save_raw_arrays(
            save_raw, a_idx, b_idx, c_idx, dist_bc, dist_bc_baseline, jaccard, retrieval_raw
        )
    return result
```

In `main()`, add the new CLI arg right after `--n-bridge-sample` (around line 506):

```python
    ap.add_argument("--n-hub-sample", type=int, default=5000,
                    help="max closed-triangle pairs AND max open hub pairs to sample "
                         "independently (Experiment 14) -- see --counts-only first")
```

and pass it through the `run(...)` call in `main()` (around line 521-524):

```python
    result = run(
        storage_dir=args.storage_dir, template_dir=args.template_dir, K=args.K,
        alpha=args.alpha, n_bridge_sample=args.n_bridge_sample, n_hub_sample=args.n_hub_sample,
        seed=args.seed, device=args.device, per_sample_npz=args.per_sample_npz, save_raw=args.save_raw,
    )
```

Add print output for the new result keys, right after the existing `grading check` print line (536) and before the `if "retrieval_correlation" in result:` block (537):

```python
    hc = result["hub_pair_counts"]
    print(f"  hub nodes: {hc['n_hub_nodes']:,}; hub pairs: {hc['n_pairs_total']:,} "
          f"({hc['n_closed_total']:,} closed, {hc['n_open_total']:,} open)")
    for name, ps in (("closed-triangle", result["closed_triangle_pull"]), ("open-hub", result["open_hub_pull"])):
        if ps["n"] == 0:
            print(f"  {name} pull: n=0 (none sampled)")
            continue
        sig = f" mean/SEM={ps['z']:+.1f}{' *' if ps['z'] == ps['z'] and abs(ps['z']) >= 2 else ''}" if ps["n"] > 1 else ""
        print(f"  {name} pull (baseline_dist - pair_dist): mean={ps['mean']:+.4f} "
              f"(n={ps['n']}, frac_pulled_closer={ps['frac_pulled_closer']:.3f}){sig}")
```

- [ ] **Step 3: Run the full selftest to confirm no regressions**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python scripts/analyze_polysemy_bridges.py --selftest`
Expected: `SELFTEST OK`

- [ ] **Step 4: Run the check from Step 1 again to confirm the new parameter exists**

Run: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && python -c "import scripts.analyze_polysemy_bridges as m; import inspect; print('n_hub_sample' in inspect.signature(m.run).parameters)"`
Expected: `True`

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_polysemy_bridges.py
git commit -m "feat: wire closed-triangle vs open-hub pull comparison into run()/main()"
```

---

### Task 6: Run for real — Task 0 incidence count, scope decision, full comparison, and report

**Files:**
- Create: `docs/reports/YYYY-MM-DD_closed_triangle_bridge_diagnostic.md` (date = the day this task is actually run)

This task has no pre-written code changes — it is the actual empirical run the prior five tasks built the tooling for. Follow it in order; the exact numbers below are unknown until run, but the decision rule and reporting structure are fixed now so the result is unambiguous to write up.

- [ ] **Step 1: Run Task 0's incidence count**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_polysemy_bridges.py --counts-only \
  --storage-dir /data/SSD2/pre_extract/redcaps_150k/features
```

Record the four printed numbers (`n_hub_nodes`, `n_pairs_total`, `n_closed`, `n_open`).

- [ ] **Step 2: Apply the 150k-vs-300k decision rule**

If `n_closed >= 30`: proceed with RedCaps-150k for the rest of this task (30 is chosen as a floor consistent with this plan's smallest existing precedent for a population-level correlation statistic, C8's n=15-16 encoder pairs — see the spec's Success Criteria). If `n_closed < 30`: re-run Step 1 against the RedCaps-300k feature store (`/data/SSD2/pre_extract/redcaps_300k/features` — confirm the exact path matches what Experiment 6/C6 used) instead, and use that store for Step 3 onward. Either way, state in the report which store was used and why.

- [ ] **Step 3: Run the full closed-vs-open pull comparison, with the retrieval cross-reference**

```bash
python scripts/analyze_polysemy_bridges.py \
  --storage-dir /data/SSD2/pre_extract/redcaps_150k/features \
  --template-dir res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings \
  --n-hub-sample 5000 --device cuda \
  --per-sample-npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/<seed1_trained_dir>/per_sample.npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/<seed2_trained_dir>/per_sample.npz \
    res/CoSiR_condition_freeze_ablation/redcaps_150k/<seed3_trained_dir>/per_sample.npz \
    res/CoSiR_pred_stopgrad_ablation/redcaps_150k/<seed1_pred_coupled_dir>/per_sample.npz \
    res/CoSiR_pred_stopgrad_ablation/redcaps_150k/<seed2_pred_coupled_dir>/per_sample.npz \
    res/CoSiR_pred_stopgrad_ablation/redcaps_150k/<seed3_pred_coupled_dir>/per_sample.npz \
  --out docs/reports/assets/2026-XX-XX_closed_triangle_bridge_diagnostic_result.json
```

(Substitute the exact 6 per-sample `.npz` paths Experiment 12.3 already produced/used — find them via `find res -name 'per_sample.npz'` or by checking `docs/reports/2026-08-26_polysemy_bridge_diagnostic.md`'s Experiment 12.3 section for the exact run directory names; if `--template-dir` uses a different graph snapshot than 300k required per Step 2, substitute that path throughout.)

- [ ] **Step 4: Apply the pre-registered success criteria and write the report**

Using the printed `closed-triangle pull` vs. `open-hub pull` lines from Step 3 (compare `mean` magnitude, not just significance, per the spec):
- **Discriminates**: closed-triangle mean pull is substantially larger than open-hub mean pull.
- **Does not discriminate**: the two are comparable in magnitude.
- **Ambiguous**: closed-triangle pull is smaller.

Write `docs/reports/YYYY-MM-DD_closed_triangle_bridge_diagnostic.md` following this project's existing report structure (see `docs/reports/2026-08-26_polysemy_bridge_diagnostic.md` for the template: Date/Dataset/Code/Motivated-by/Compute header, TL;DR, Method, Results table, Interpretation numbered list applying the three-way criteria above, Caveats, Reproduction commands). Include: the Task 0 counts and which feature store was used (Step 2), the closed vs. open pull comparison (primary, #2), the hub-vs-plain-bridge pull comparison read off `label_counts`/`hub_pair_counts` context (secondary, #1, descriptive), and the `is_hub`/`in_closed_triangle`/`in_open_hub_pair` retrieval cross-reference results (secondary, #3, descriptive).

- [ ] **Step 5: Fold the result into the spec's claims table**

Following this plan's established pattern (see how Experiment 12.3's result was folded into C10, and Experiment 13's into C11): add a new claim row (or extend C10/C12 if the result is best read as a direct qualifier of the existing false-transitivity claim rather than a standalone one — judgment call at write time) to `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, and a "**Result (YYYY-MM-DD):**" line to the Experiment 14 plan-section entry itself, matching Experiment 12.5's pattern (spec line with `**Result:**` inline, `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md:242`).

- [ ] **Step 6: Commit**

```bash
git add docs/reports/YYYY-MM-DD_closed_triangle_bridge_diagnostic.md docs/reports/assets/ \
  docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
git commit -m "results: run Experiment 14 closed-triangle bridge diagnostic"
```

---

## Self-Review Notes

- **Spec coverage:** Task 0 (spike) → Task 4/6. Tooling's `buddy_graph.py` addition → Task 1. `analyze_polysemy_bridges.py` additions (`extract_hub_pairs`, `label_nodes`-adjacent extension, `correlate_polysemy_with_retrieval` extension) → Tasks 2-3 (the plan implements the taxonomy extension as additive boolean flags rather than new mutually-exclusive `label_nodes` category strings, to avoid silently changing the existing 4-way partition's historical bridge/Experiment-12/C10 counts — a deliberate deviation from the spec's literal wording, noted here since the spec said "hub/closed_triangle labels" without specifying mechanism). Primary/secondary success criteria → Task 6. Zero new training/eval → verified no task adds a training invocation.
- **Placeholder scan:** no TBD/TODO; Task 6's exact numbers are necessarily unknown pre-run (an empirical result, not a placeholder) but its decision rule, commands, and report structure are fully specified.
- **Type consistency:** `hub_neighbor_pairs` (Task 1) → `extract_hub_pairs`/`closed_triangle_membership` (Task 2) → `run()` (Task 5) all agree on the `{"hub"/"c"/"d"/"is_closed"}` dict shape and the `(M, 3)` `[hub, c, d]` pairs-array convention.
