# Cross-VLM Buddy Survival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Post-plan deviation (2026-07-17, implemented):** The chance-correction below is written as a Monte-Carlo permutation null (`perm_null_jaccard`, `--n_perm`/`--seed`). That was intractable at full scale (N=150k) and was replaced — with user approval — by the closed-form **analytic** null it was only estimating: `chance_null_jaccard(a, b, N)` with `E[inter] = |a|·|b| / C(N,2)`. The `n_perm`/`seed` knobs were removed. The spec (`…specs/2026-07-16-…`) is the source of truth; Task 2 / Task 5 code below reflects the pre-deviation API.

**Goal:** Rebuild the cross-modal buddy graph for every (vision × text) encoder cell of a 4×4 grid on RedCaps and measure how much the resulting buddy edge sets agree, plus extract a consensus "core" of surviving buddies.

**Architecture:** A pure-function library (`cross_vlm_buddy.py`) computes per-cell buddy edge sets, chance-corrected pairwise agreement, and consensus core; a thin CLI driver (`run_grid.py`) loads features, runs the library, and writes JSON + plots. All heavy encoders are reused from the held-out grid (cached features) and the existing buddy-graph builders — no new model plumbing.

**Tech Stack:** Python 3.10, numpy, scipy.sparse, matplotlib, pytest; conda env `CoSiR`; existing `src/conditional_buddy/buddy_graph.py` and `src/test/20260623_redcaps_buddy/redcaps_buddy.py`.

## Global Constraints

- Conda env for every command: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR`.
- New code folder: `src/test/20260716_buddy_cross_vlm/` (repo dated-folder convention).
- Dataset: RedCaps only. No Impressions, no training, no spectral embedding (YAGNI per spec).
- Grid is fixed: `VISION = ["clip_img", "dinov2", "siglip_v", "vit_sup"]`, `TEXT = ["clip_txt", "minilm", "bge", "e5"]`, cell key format `f"{v}x{t}"`, 16 cells in `[(v, t) for v in VISION for t in TEXT]` order.
- Buddy defs computed for BOTH `B = A_img ∩ A_txt` and `E = A_img ∪ A_txt`.
- Edge sets are represented as **sorted `np.int64` arrays of keys `key = i * N + j` with `i < j`** throughout (N = number of common nodes).
- K = 30, `n_perm` = 200, `seed` = 42 defaults.
- Artifacts dir: `docs/reports/assets/buddy_cross_vlm/`.

---

### Task 1: Scaffold folder and ensure RedCaps held-out feature caches exist

**Files:**
- Create: `src/test/20260716_buddy_cross_vlm/20260716_buddy_cross_vlm_log.md` (stub)
- Depends on (read-only): `src/test/20260708_heldout_grid/extract_heldout.py`, `heldout_feats/redcaps/*.npy`

**Interfaces:**
- Consumes: nothing.
- Produces: the six cache files `src/test/20260708_heldout_grid/heldout_feats/redcaps/{dinov2,siglip_v,vit_sup,minilm,bge,e5}.npy`, each shape `(data.n, dim)` aligned to `redcaps_buddy.load_data()` row order. Later tasks load these via `extract_heldout.cache_path("redcaps", <model>, 0)`.

- [ ] **Step 1: Create the folder and log stub**

Create `src/test/20260716_buddy_cross_vlm/20260716_buddy_cross_vlm_log.md`:

```markdown
# Cross-VLM Buddy Survival — Log

Spec: docs/superpowers/specs/2026-07-16-buddy-cross-vlm-survival-design.md
Plan: docs/superpowers/plans/2026-07-16-buddy-cross-vlm-survival.md

## Prerequisite: held-out RedCaps caches
(to be filled: which caches existed, which were extracted)

## Results
(to be filled after the real run)
```

- [ ] **Step 2: Check which RedCaps caches already exist**

Run:
```bash
ls -1 src/test/20260708_heldout_grid/heldout_feats/redcaps/ 2>/dev/null
```
Expected: some subset of `dinov2.npy siglip_v.npy vit_sup.npy minilm.npy bge.npy e5.npy`. Note which are missing.

- [ ] **Step 3: Extract any missing caches (GPU)**

For each missing `<model>` from `{dinov2, siglip_v, vit_sup, minilm, bge, e5}`, run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python src/test/20260708_heldout_grid/extract_heldout.py --dataset redcaps --model <model>
```
Expected per model: `saved .../redcaps/<model>.npy shape=(<n>, <dim>) ...` (or `reused DINOv2 cache` for dinov2). This needs a GPU and downloads HF weights on first use.

- [ ] **Step 4: Verify all six exist and record in the log**

Run:
```bash
ls -1 src/test/20260708_heldout_grid/heldout_feats/redcaps/{dinov2,siglip_v,vit_sup,minilm,bge,e5}.npy
```
Expected: all six paths listed, no "No such file" error. Fill the log's prerequisite section with which existed vs were extracted.

- [ ] **Step 5: Commit**

```bash
git add src/test/20260716_buddy_cross_vlm/20260716_buddy_cross_vlm_log.md
git commit -m "scaffold cross-vlm buddy folder; verify redcaps heldout caches"
```

---

### Task 2: Agreement metrics core (`cross_vlm_buddy.py`)

**Files:**
- Create: `src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py`
- Test: `src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py`

**Interfaces:**
- Consumes: nothing (pure functions over `np.int64` key arrays).
- Produces:
  - `VISION: list[str]`, `TEXT: list[str]`, `CELLS: list[tuple[str,str]]`
  - `adj_to_keys(A: csr_matrix) -> np.ndarray` — sorted int64 keys `i*N+j`, `i<j`
  - `jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]` → `(jaccard, overlap_coef, intersection_size)`
  - `perm_null_jaccard(a, b, N, n_perm=200, seed=42) -> dict` with keys `observed, null_mean, lift, percentile`
  - `agreement_matrix(cell_keys: dict[str, np.ndarray], N: int, n_perm=200, seed=42) -> dict` with keys `names, jaccard, overlap, lift, median_offdiag_jaccard, median_offdiag_lift`

- [ ] **Step 1: Write failing tests for jaccard, perm null, agreement**

Create `src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py`:

```python
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cross_vlm_buddy as cvb


def test_jaccard_identical():
    a = np.array([1, 2, 5], dtype=np.int64)
    jac, ov, inter = cvb.jaccard(a, a)
    assert jac == 1.0 and ov == 1.0 and inter == 3


def test_jaccard_disjoint():
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([3, 4], dtype=np.int64)
    jac, ov, inter = cvb.jaccard(a, b)
    assert jac == 0.0 and inter == 0


def test_perm_null_identical_has_high_lift():
    N = 50
    # edges among distinct node pairs, encoded i*N+j (i<j)
    a = np.sort(np.array([0 * N + 1, 2 * N + 3, 4 * N + 5], dtype=np.int64))
    res = cvb.perm_null_jaccard(a, a, N, n_perm=100, seed=0)
    assert res["observed"] == 1.0
    assert res["lift"] > 5.0            # identical sets crush the permuted null
    assert res["percentile"] == 1.0


def test_perm_null_random_lift_near_one():
    N = 200
    rng = np.random.default_rng(3)

    def rand_keys(m):
        i = rng.integers(0, N, m); j = rng.integers(0, N, m)
        ok = i != j
        lo = np.minimum(i[ok], j[ok]); hi = np.maximum(i[ok], j[ok])
        return np.unique(lo.astype(np.int64) * N + hi.astype(np.int64))

    a, b = rand_keys(300), rand_keys(300)
    res = cvb.perm_null_jaccard(a, b, N, n_perm=100, seed=1)
    assert 0.3 < res["lift"] < 3.0      # independent graphs: no real agreement


def test_agreement_matrix_shape_and_diag():
    N = 20
    cells = {"a": np.array([1, 2], dtype=np.int64),
             "b": np.array([1, 3], dtype=np.int64),
             "c": np.array([7, 8], dtype=np.int64)}
    out = cvb.agreement_matrix(cells, N, n_perm=20, seed=0)
    assert out["jaccard"].shape == (3, 3)
    assert np.allclose(np.diag(out["jaccard"]), 1.0)
    assert out["jaccard"][0, 1] > 0.0 and out["jaccard"][0, 2] == 0.0
    assert np.isfinite(out["median_offdiag_jaccard"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python -m pytest src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'cross_vlm_buddy'` (module not created yet).

- [ ] **Step 3: Write `cross_vlm_buddy.py` metrics core**

Create `src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py`:

```python
"""
Cross-VLM buddy survival: rebuild the cross-modal buddy graph for every
(vision encoder x text encoder) cell of a 4x4 grid and measure how much the
resulting buddy edge sets agree. Pure-function library; see run_grid.py for the
CLI driver.

Edge sets are sorted np.int64 arrays of keys `i*N + j` (i < j), N = #nodes.

Design: docs/superpowers/specs/2026-07-16-buddy-cross-vlm-survival-design.md
"""
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix, triu

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT,
          os.path.join(ROOT, "src", "test", "20260623_redcaps_buddy"),
          os.path.join(ROOT, "src", "test", "20260708_heldout_grid")):
    if p not in sys.path:
        sys.path.insert(0, p)

VISION = ["clip_img", "dinov2", "siglip_v", "vit_sup"]
TEXT = ["clip_txt", "minilm", "bge", "e5"]
CELLS = [(v, t) for v in VISION for t in TEXT]  # 16 cells


# ── edge-set representation ──────────────────────────────────────────────────

def adj_to_keys(A: csr_matrix) -> np.ndarray:
    """Upper-triangular (i<j) edges of a symmetric adjacency as sorted int64 keys i*N+j."""
    N = A.shape[0]
    U = triu(A, k=1).tocoo()
    keys = U.row.astype(np.int64) * N + U.col.astype(np.int64)
    keys.sort()
    return keys


# ── pairwise agreement ───────────────────────────────────────────────────────

def jaccard(a: np.ndarray, b: np.ndarray):
    """(jaccard, overlap_coef, intersection_size) for two sorted-unique key arrays."""
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    union = int(a.size + b.size - inter)
    jac = inter / union if union else 0.0
    denom = min(int(a.size), int(b.size))
    ov = inter / denom if denom else 0.0
    return jac, ov, inter


def perm_null_jaccard(a: np.ndarray, b: np.ndarray, N: int, n_perm: int = 200, seed: int = 42):
    """
    Chance-correct Jaccard(a, b) by node-relabeling `b` under random permutations
    (preserves b's exact degree sequence, destroys alignment). Returns observed,
    null mean, lift = observed/null_mean, and percentile of observed in the null.
    """
    obs, _, _ = jaccard(a, b)
    bi, bj = b // N, b % N
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        perm = rng.permutation(N)
        pi, pj = perm[bi], perm[bj]
        lo = np.minimum(pi, pj).astype(np.int64)
        hi = np.maximum(pi, pj).astype(np.int64)
        bk = np.unique(lo * N + hi)
        nulls[k], _, _ = jaccard(a, bk)
    null_mean = float(nulls.mean())
    lift = obs / null_mean if null_mean > 0 else float("inf")
    percentile = float((nulls <= obs).mean())
    return {"observed": obs, "null_mean": null_mean, "lift": lift, "percentile": percentile}


def agreement_matrix(cell_keys: dict, N: int, n_perm: int = 200, seed: int = 42):
    """Full pairwise Jaccard / overlap / chance-lift across all cells."""
    names = list(cell_keys.keys())
    n = len(names)
    jac = np.eye(n, dtype=np.float64)
    ov = np.eye(n, dtype=np.float64)
    lift = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = cell_keys[names[i]], cell_keys[names[j]]
            jj, oo, _ = jaccard(a, b)
            jac[i, j] = jac[j, i] = jj
            ov[i, j] = ov[j, i] = oo
            res = perm_null_jaccard(a, b, N, n_perm=n_perm, seed=seed)
            lift[i, j] = lift[j, i] = res["lift"]
    off = ~np.eye(n, dtype=bool)
    return {
        "names": names,
        "jaccard": jac,
        "overlap": ov,
        "lift": lift,
        "median_offdiag_jaccard": float(np.median(jac[off])),
        "median_offdiag_lift": float(np.nanmedian(lift[off])),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python -m pytest src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py \
        src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py
git commit -m "feat: cross-vlm buddy agreement metrics (jaccard, perm null, matrix)"
```

---

### Task 3: Consensus core, subreddit-lift validation, and node mask (`cross_vlm_buddy.py`)

**Files:**
- Modify: `src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py` (append functions)
- Modify: `src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py` (append tests)

**Interfaces:**
- Consumes: `VISION` (from Task 2); `redcaps_buddy.subreddit_lift(data, e)` where `data` needs only `.sub_id` (int array) and `.sub_names` (list), and `e` is an `(M, 2)` int array of node indices.
- Produces:
  - `valid_vision_mask(feats: dict[str, np.ndarray]) -> np.ndarray[bool]`
  - `consensus_counts(cell_keys_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]` → `(unique_keys, counts)`
  - `survival_curve(counts: np.ndarray, n_cells: int) -> np.ndarray` (length `n_cells`, index t-1 = #edges in ≥ t cells)
  - `core_edges(unique_keys, counts, t, N) -> np.ndarray` shape `(M, 2)`
  - `core_subreddit_lift(unique_keys, counts, N, sub_id, sub_names, n_cells) -> list[dict]` (one dict per t: `{t, n_edges, lift}`)

- [ ] **Step 1: Write failing tests for mask, consensus, survival, core lift**

Append to `src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py`:

```python
def test_valid_vision_mask_drops_zero_rows():
    # 4 rows; row 2 is zero in one vision encoder -> dropped. Text ignored by mask.
    feats = {
        "clip_img": np.ones((4, 3), np.float32),
        "dinov2": np.array([[1, 1], [1, 1], [0, 0], [1, 1]], np.float32),
        "siglip_v": np.ones((4, 2), np.float32),
        "vit_sup": np.ones((4, 5), np.float32),
        "clip_txt": np.ones((4, 3), np.float32),  # present but irrelevant to mask
    }
    mask = cvb.valid_vision_mask(feats)
    assert mask.tolist() == [True, True, False, True]


def test_consensus_counts_and_survival():
    cells = [np.array([1, 2, 3], np.int64),
             np.array([1, 2], np.int64),
             np.array([1], np.int64)]
    uniq, counts = cvb.consensus_counts(cells)
    assert uniq.tolist() == [1, 2, 3]
    assert counts.tolist() == [3, 2, 1]           # key 1 in all 3 cells, key 3 in one
    surv = cvb.survival_curve(counts, n_cells=3)
    assert surv.tolist() == [3, 2, 1]             # >=1:3 edges, >=2:2, >=3:1


def test_core_edges_decode():
    N = 10
    uniq = np.array([0 * N + 1, 2 * N + 3], np.int64)  # edges (0,1) and (2,3)
    counts = np.array([3, 1], np.int64)
    e = cvb.core_edges(uniq, counts, t=2, N=N)
    assert e.tolist() == [[0, 1]]                  # only the count>=2 edge survives


def test_core_subreddit_lift_monotone_when_core_is_coherent():
    # 6 nodes, 2 subreddits: {0,1,2} sub 0, {3,4,5} sub 1.
    # High-consensus edges are within-subreddit; low-consensus edges cross.
    N = 6
    sub_id = np.array([0, 0, 0, 1, 1, 1])
    sub_names = ["A", "B"]
    within = [0 * N + 1, 1 * N + 2, 3 * N + 4]     # same-sub (should be coherent core)
    cross = [0 * N + 3, 1 * N + 4]                  # cross-sub (noise, low consensus)
    cells = [np.array(sorted(within + cross), np.int64) for _ in range(5)] \
        + [np.array(sorted(within), np.int64) for _ in range(5)]
    uniq, counts = cvb.consensus_counts(cells)
    curve = cvb.core_subreddit_lift(uniq, counts, N, sub_id, sub_names, n_cells=10)
    lift_low = next(c["lift"] for c in curve if c["t"] == 1)
    lift_high = next(c["lift"] for c in curve if c["t"] == 10)
    assert lift_high >= lift_low                    # purer core -> higher same-sub lift
```

- [ ] **Step 2: Run new tests to verify they fail**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python -m pytest src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py -k "mask or consensus or core" -v
```
Expected: FAIL — `AttributeError: module 'cross_vlm_buddy' has no attribute 'valid_vision_mask'`.

- [ ] **Step 3: Append consensus/core/mask functions to `cross_vlm_buddy.py`**

Append to `src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py`:

```python
# ── common node set ──────────────────────────────────────────────────────────

def valid_vision_mask(feats: dict) -> np.ndarray:
    """Rows where EVERY vision encoder has a nonzero feature (missing images -> zero rows)."""
    N = next(iter(feats.values())).shape[0]
    mask = np.ones(N, dtype=bool)
    for v in VISION:
        mask &= (np.abs(feats[v]).sum(axis=1) > 0)
    return mask


# ── consensus core ───────────────────────────────────────────────────────────

def consensus_counts(cell_keys_list: list):
    """(unique_keys, counts) where counts[k] = #cells containing unique_keys[k]."""
    allk = np.concatenate(cell_keys_list) if cell_keys_list else np.empty(0, np.int64)
    uniq, counts = np.unique(allk, return_counts=True)
    return uniq, counts


def survival_curve(counts: np.ndarray, n_cells: int) -> np.ndarray:
    """Length-n_cells array; index t-1 = #edges present in >= t cells."""
    return np.array([int((counts >= t).sum()) for t in range(1, n_cells + 1)], dtype=np.int64)


def core_edges(unique_keys: np.ndarray, counts: np.ndarray, t: int, N: int) -> np.ndarray:
    """(M, 2) node-index edge list for edges present in >= t cells."""
    keys = unique_keys[counts >= t]
    return np.stack([keys // N, keys % N], axis=1).astype(np.int64)


class _SubShim:
    """Minimal stand-in for redcaps_buddy.Data: only .sub_id and .sub_names are used."""
    def __init__(self, sub_id, sub_names):
        self.sub_id = np.asarray(sub_id)
        self.sub_names = list(sub_names)


def core_subreddit_lift(unique_keys, counts, N, sub_id, sub_names, n_cells: int):
    """Same-subreddit lift of the >= t consensus core, for t = 1..n_cells."""
    import redcaps_buddy as rb
    shim = _SubShim(sub_id, sub_names)
    out = []
    for t in range(1, n_cells + 1):
        e = core_edges(unique_keys, counts, t, N)
        if len(e) == 0:
            out.append({"t": t, "n_edges": 0, "lift": float("nan")})
            continue
        res = rb.subreddit_lift(shim, e)
        out.append({"t": t, "n_edges": int(len(e)), "lift": float(res["overall_lift"])})
    return out
```

- [ ] **Step 4: Run all tests to verify they pass**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python -m pytest src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py \
        src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py
git commit -m "feat: consensus core, subreddit-lift validation, vision node mask"
```

---

### Task 4: Feature loading and per-cell graph building (`cross_vlm_buddy.py`)

**Files:**
- Modify: `src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py` (append functions)

**Interfaces:**
- Consumes: `redcaps_buddy.load_data()` → object with `.img (N,D) float32 L2-normed`, `.txt (N,D)`, `.n`, `.sub_id (N,)`, `.sub_names list`; `extract_heldout.cache_path(dataset, model, smoke)`; `src.conditional_buddy.buddy_graph.mutual_knn`, `union_graph`; `adj_to_keys`, `valid_vision_mask`, `VISION`, `TEXT`, `CELLS`.
- Produces:
  - `load_grid_features(smoke=0) -> tuple[dict[str,np.ndarray], np.ndarray, list, np.ndarray]` → `(feats, sub_id, sub_names, vmask)`; `feats` has all 8 keys sliced to the common node set.
  - `build_cell_graphs(feats, K=30, device="cuda", use_half=True) -> tuple[dict, dict, int]` → `(cell_B, cell_E, N)`; dict keys are `f"{v}x{t}"` in `CELLS` order, values are sorted int64 key arrays.

There is no unit test here (GPU + real caches); Task 5's smoke run exercises it end to end.

- [ ] **Step 1: Append loading + graph-building functions**

Append to `src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py`:

```python
# ── feature loading + per-cell graph building ────────────────────────────────

def load_grid_features(smoke: int = 0):
    """
    Load all 8 grid feature matrices in redcaps row order, slice to the common
    node set (rows valid across every vision encoder). Returns
    (feats, sub_id, sub_names, vmask) with feats sliced to the common nodes.
    smoke>0 keeps only the first `smoke` valid rows (pipeline sanity, not interpreted).
    """
    import redcaps_buddy as rb
    from extract_heldout import cache_path

    data = rb.load_data()
    feats = {"clip_img": np.ascontiguousarray(data.img, dtype=np.float32),
             "clip_txt": np.ascontiguousarray(data.txt, dtype=np.float32)}
    for m in ["dinov2", "siglip_v", "vit_sup", "minilm", "bge", "e5"]:
        p = cache_path("redcaps", m, 0)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing held-out cache {p}; run:\n"
                f"  python src/test/20260708_heldout_grid/extract_heldout.py "
                f"--dataset redcaps --model {m}")
        feats[m] = np.load(p).astype(np.float32)

    vmask = valid_vision_mask(feats)
    if smoke:
        idx = np.where(vmask)[0][:smoke]
        keep = np.zeros(data.n, dtype=bool)
        keep[idx] = True
        vmask = keep

    feats = {k: v[vmask] for k, v in feats.items()}
    sub_id = data.sub_id[vmask]
    print(f"[cross-vlm] common nodes: {int(vmask.sum())}/{data.n} "
          f"(dropped {int((~vmask).sum())})")
    return feats, sub_id, data.sub_names, vmask


def build_cell_graphs(feats: dict, K: int = 30, device: str = "cuda", use_half: bool = True):
    """
    Build one mutual-kNN graph per distinct feature matrix (8 total), then the 16
    cells' B (intersection) and E (union) edge sets. Returns (cell_B, cell_E, N).
    """
    from src.conditional_buddy.buddy_graph import mutual_knn, union_graph

    N = next(iter(feats.values())).shape[0]
    A = {name: mutual_knn(feats[name], K=K, device=device, use_half=use_half)
         for name in feats}
    cell_B, cell_E = {}, {}
    for v, t in CELLS:
        Aimg, Atxt = A[v], A[t]
        B = Aimg.multiply(Atxt)
        B.data[:] = 1.0
        B = B.tocsr()
        B.eliminate_zeros()
        E = union_graph(Aimg, Atxt)
        key = f"{v}x{t}"
        cell_B[key] = adj_to_keys(B)
        cell_E[key] = adj_to_keys(E)
    return cell_B, cell_E, N
```

- [ ] **Step 2: Import-smoke the new functions (no full run yet)**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python -c "import sys; sys.path.insert(0,'src/test/20260716_buddy_cross_vlm'); \
import cross_vlm_buddy as c; print(len(c.CELLS), c.CELLS[0], c.CELLS[-1]); \
print(callable(c.load_grid_features), callable(c.build_cell_graphs))"
```
Expected: `16 ('clip_img', 'clip_txt') ('vit_sup', 'e5')` then `True True`.

- [ ] **Step 3: Commit**

```bash
git add src/test/20260716_buddy_cross_vlm/cross_vlm_buddy.py
git commit -m "feat: grid feature loading + per-cell buddy graph building"
```

---

### Task 5: CLI driver, plots, and smoke run (`run_grid.py`)

**Files:**
- Create: `src/test/20260716_buddy_cross_vlm/run_grid.py`

**Interfaces:**
- Consumes: everything from `cross_vlm_buddy.py` (`load_grid_features`, `build_cell_graphs`, `agreement_matrix`, `consensus_counts`, `survival_curve`, `core_subreddit_lift`, `core_edges`, `CELLS`).
- Produces: artifacts in `docs/reports/assets/buddy_cross_vlm/` — `grid_agreement.json`, `agreement_B.png`, `agreement_E.png`, `survival_curves.png`, `core_lift.png`, `core_edges_B.npy`, `core_edges_E.npy`.

- [ ] **Step 1: Write `run_grid.py`**

Create `src/test/20260716_buddy_cross_vlm/run_grid.py`:

```python
"""
Cross-VLM buddy survival driver: build the 4x4 (vision x text) buddy grid on
RedCaps, compute chance-corrected pairwise agreement + consensus core (for B and
E), validate the core with subreddit lift, and write JSON + plots.

Usage:
  python run_grid.py --smoke 512      # fast pipeline sanity (magnitudes not interpreted)
  python run_grid.py                  # full RedCaps run

Design: docs/superpowers/specs/2026-07-16-buddy-cross-vlm-survival-design.md
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import cross_vlm_buddy as cvb

ASSETS = os.path.join(ROOT, "docs", "reports", "assets", "buddy_cross_vlm")


def plot_heatmap(mat, names, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_survival(survB, survE, path):
    t = np.arange(1, len(survB) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, survB, "o-", label="B (intersection)")
    ax.plot(t, survE, "s-", label="E (union)")
    ax.set_xlabel("consensus level t (edge present in >= t of 16 cells)")
    ax.set_ylabel("# surviving buddy edges")
    ax.set_yscale("log")
    ax.set_title("Buddy survival curve across the VLM grid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_core_lift(liftB, liftE, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve, lab, mk in ((liftB, "B (intersection)", "o-"),
                           (liftE, "E (union)", "s-")):
        t = [c["t"] for c in curve]
        lift = [c["lift"] for c in curve]
        ax.plot(t, lift, mk, label=lab)
    ax.axhline(1.0, color="grey", ls="--", lw=1, label="chance (random pairs)")
    ax.set_xlabel("consensus level t")
    ax.set_ylabel("same-subreddit lift of the >= t core")
    ax.set_title("Are surviving buddies semantically coherent?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--core_t", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)

    feats, sub_id, sub_names, vmask = cvb.load_grid_features(smoke=args.smoke)
    cell_B, cell_E, N = cvb.build_cell_graphs(feats, K=args.K, device=args.device)
    n_cells = len(cvb.CELLS)

    aggB = cvb.agreement_matrix(cell_B, N, n_perm=args.n_perm, seed=args.seed)
    aggE = cvb.agreement_matrix(cell_E, N, n_perm=args.n_perm, seed=args.seed)

    uB, cB = cvb.consensus_counts(list(cell_B.values()))
    uE, cE = cvb.consensus_counts(list(cell_E.values()))
    survB = cvb.survival_curve(cB, n_cells)
    survE = cvb.survival_curve(cE, n_cells)
    liftB = cvb.core_subreddit_lift(uB, cB, N, sub_id, sub_names, n_cells)
    liftE = cvb.core_subreddit_lift(uE, cE, N, sub_id, sub_names, n_cells)

    # smoke: assert the pipeline produced finite, well-shaped output; do not interpret.
    if args.smoke:
        assert aggB["jaccard"].shape == (n_cells, n_cells)
        assert np.isfinite(aggB["median_offdiag_jaccard"])
        assert survB.shape == (n_cells,)
        assert len(liftB) == n_cells
        print(f"[smoke] OK  N={N}  medianJ(B)={aggB['median_offdiag_jaccard']:.4f} "
              f"medianLift(B)={aggB['median_offdiag_lift']:.2f}")
        return

    summary = {
        "n_nodes": int(N),
        "K": args.K,
        "cells": [f"{v}x{t}" for v, t in cvb.CELLS],
        "B": {"median_offdiag_jaccard": aggB["median_offdiag_jaccard"],
              "median_offdiag_lift": aggB["median_offdiag_lift"],
              "jaccard": aggB["jaccard"].tolist(),
              "overlap": aggB["overlap"].tolist(),
              "lift": aggB["lift"].tolist(),
              "survival": survB.tolist(),
              "core_lift": liftB},
        "E": {"median_offdiag_jaccard": aggE["median_offdiag_jaccard"],
              "median_offdiag_lift": aggE["median_offdiag_lift"],
              "jaccard": aggE["jaccard"].tolist(),
              "overlap": aggE["overlap"].tolist(),
              "lift": aggE["lift"].tolist(),
              "survival": survE.tolist(),
              "core_lift": liftE},
    }
    with open(os.path.join(ASSETS, "grid_agreement.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(ASSETS, 'grid_agreement.json')}")

    plot_heatmap(aggB["jaccard"], aggB["names"], "B buddy-graph agreement (Jaccard)",
                 os.path.join(ASSETS, "agreement_B.png"))
    plot_heatmap(aggE["jaccard"], aggE["names"], "E buddy-graph agreement (Jaccard)",
                 os.path.join(ASSETS, "agreement_E.png"))
    plot_survival(survB, survE, os.path.join(ASSETS, "survival_curves.png"))
    plot_core_lift(liftB, liftE, os.path.join(ASSETS, "core_lift.png"))

    np.save(os.path.join(ASSETS, "core_edges_B.npy"),
            cvb.core_edges(uB, cB, args.core_t, N))
    np.save(os.path.join(ASSETS, "core_edges_E.npy"),
            cvb.core_edges(uE, cE, args.core_t, N))
    print(f"wrote core_edges_{{B,E}}.npy (t>={args.core_t})")
    print(f"[done] B: medianJ={aggB['median_offdiag_jaccard']:.4f} "
          f"medianLift={aggB['median_offdiag_lift']:.2f} | "
          f"E: medianJ={aggE['median_offdiag_jaccard']:.4f} "
          f"medianLift={aggE['median_offdiag_lift']:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke pipeline**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python src/test/20260716_buddy_cross_vlm/run_grid.py --smoke 512 --n_perm 20
```
Expected: prints `[cross-vlm] common nodes: ...`, then `[smoke] OK N=512 medianJ(B)=... medianLift(B)=...`, exit 0, no traceback.

- [ ] **Step 3: Commit**

```bash
git add src/test/20260716_buddy_cross_vlm/run_grid.py
git commit -m "feat: cross-vlm buddy grid driver + plots + smoke run"
```

---

### Task 6: Full RedCaps run and results write-up

**Files:**
- Modify: `src/test/20260716_buddy_cross_vlm/20260716_buddy_cross_vlm_log.md`
- Produces: the real artifacts in `docs/reports/assets/buddy_cross_vlm/`

**Interfaces:**
- Consumes: `run_grid.py` (full run). Produces: no new code, only results + interpretation.

- [ ] **Step 1: Run the full grid**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR && \
python src/test/20260716_buddy_cross_vlm/run_grid.py
```
Expected: writes `grid_agreement.json`, two `agreement_*.png`, `survival_curves.png`, `core_lift.png`, `core_edges_{B,E}.npy`; final `[done]` line with median Jaccard + median lift for B and E.

- [ ] **Step 2: Record results in the log**

Fill `20260716_buddy_cross_vlm_log.md` Results section with: N (common nodes), median off-diagonal Jaccard and lift for B and E, the shape of the survival curve (does a core persist at t≥8?), and whether subreddit lift rises with t (core coherence). Note the CLIP×CLIP row as the reference-anchored slice. State the headline answer: **do buddies survive across VL models, beyond chance?**

- [ ] **Step 3: Commit**

```bash
git add src/test/20260716_buddy_cross_vlm/20260716_buddy_cross_vlm_log.md \
        docs/reports/assets/buddy_cross_vlm/
git commit -m "results: cross-vlm buddy survival full run + write-up"
```

---

## Self-Review

**Spec coverage:**
- Grid (4×4, reuse held-out 6 + CLIP) → Task 4 `load_grid_features`, `VISION`/`TEXT`. ✓
- Common node set → Task 3 `valid_vision_mask` + Task 4 slicing. ✓
- Both B and E → Task 4 `build_cell_graphs`; all metrics run for both. ✓
- Pairwise agreement (Jaccard + overlap) → Task 2 `agreement_matrix`. ✓
- Chance correction (node-relabel null, lift) → Task 2 `perm_null_jaccard`. ✓
- Consensus core + survival curve → Task 3 `consensus_counts`/`survival_curve`/`core_edges`. ✓
- Subreddit-lift core validation (independent GT) → Task 3 `core_subreddit_lift`. ✓
- Artifacts (json, 2 heatmaps, survival, core-lift, core edge npy) → Task 5. ✓
- Testing (5 unit checks, smoke) → Tasks 2/3 units + Task 5 smoke. ✓
- Prereq caches → Task 1. ✓
- RedCaps only, no training/spectral/Impressions → respected; no such tasks. ✓

**Placeholder scan:** no TBD/TODO in code; the log stub's "to be filled" is intentional and completed in Task 6. ✓

**Type consistency:** edge sets are sorted int64 key arrays everywhere; `agreement_matrix` consumes `dict[str, keys]` produced by `build_cell_graphs`; `core_subreddit_lift` consumes `(unique_keys, counts)` from `consensus_counts`; `_SubShim` exposes exactly the `.sub_id`/`.sub_names` that `redcaps_buddy.subreddit_lift` reads. ✓
