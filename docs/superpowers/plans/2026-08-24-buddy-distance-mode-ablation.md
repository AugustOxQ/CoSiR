# Buddy Distance-Mode Ablation (Experiment 10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and test the "typed" distance-mixing fix identified by the buddy-graph modality-disagreement diagnostic (`src/test/20260824_buddy_graph_disagreement/`): does using each edge's own supporting modality's rank alone (instead of the current fixed-alpha blend that dilutes ~98% of edges toward statistical noise) improve retrieval, and specifically does it narrow C6's still-open gap to raw CLIP (`test_pre_diff`)?

**Architecture:** `compute_buddy_init` (`src/conditional_buddy/compute_buddies.py`) currently computes both image and text cosine distance on every edge of the union graph `E` and blends them with a fixed `alpha*D_img + (1-alpha)*D_txt`, regardless of which modality(ies) actually support that edge. This plan adds an opt-in `distance_mode: "blend" | "typed"` parameter (default `"blend"`, exactly reproducing current behavior — backward-compatible for every existing caller): `"typed"` uses each edge's own supporting modality's rank-normalized distance alone for img-only/txt-only edges, and the existing blend only for `both` (cross-modally-confirmed, no disagreement) and `repair` (added by `ensure_min_degree`/`ensure_connected`, owned by neither modality) edges. This is threaded through exactly the same integration points as Experiment 8's `encoder_pair` (a new `train.buddies.distance_mode` config key, added to the buddies template-compatibility `_extra` dict), then swept via the same bash-loop-over-template-key-axis / Hydra-multirun-over-seed pattern as `scripts/run_init_ablation.sh` / `scripts/run_buddy_init_encoder_ablation.sh`.

**Tech Stack:** Python 3.10, NumPy, SciPy (sparse), Hydra/OmegaConf, PyTorch, wandb, pandas. Existing CoSiR training entrypoint `main_cosir.py`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 10 (added 2026-08-24).

## Global Constraints

- Always run Python/bash commands with `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` first (per `.claude/CLAUDE.md`).
- Statistical standard (spec §5): ≥3 seeds, paired-within-seed Δ, report mean ± std and mean/SEM, compare against the noise floor (~0.1–0.7 R1), never against zero.
- **Non-finished-run filter, from the start.** A prior task in this session's work (Experiment 8's Task 5) shipped an analysis script without this filter and it had to be fixed in review — do not repeat that: any new `fetch()`-style wandb-reading function in this plan must exclude `run.state != "finished"` rows from the start, not as a follow-up fix.
- Fixed operating point (same as C5/C6/Experiment 1/8): `lr=1e-3`, `lr_label=1e-4`, `embedding_dim=16`, `alpha=0.5`, `initialization_strategy=buddies` fixed, RedCaps-150k. Every training-time buddy term stays OFF (never pass `+loss.lambda_buddy`, `+loss.lambda_buddy_con`, `+loss.buddy_refresh*`).
- `train.buddies.distance_mode` is a **template-compatibility key**, exactly like `initialization_strategy`/`encoder_pair`: give every mode its own `results_dir` — never share one `results_dir` across `blend` and `typed`.
- `distance_mode="blend"` (the default, whether passed explicitly or omitted) must be **numerically identical** to `compute_buddy_init`'s current behavior — this is the core backward-compatibility property the whole plan depends on, and every task touching `compute_buddy_init` must verify it, not assume it.
- `test_raw`/`test_diff`/`test_pre_diff` are already automatically logged by every eval call (`src/eval/pipeline.py`) — no new eval code is needed anywhere in this plan.
- Two existing `src/` files are modified in this plan (`src/conditional_buddy/buddy_graph.py`, `src/conditional_buddy/compute_buddies.py`, `src/utils/embedding_manager_nocache.py`, `src/hook/train_cosir.py` — four, not two). Per CLAUDE.md, log each in `.claude/20260824_log.md` (the file already exists from earlier work this session; append new sections, don't overwrite).

---

### Task 1: `mix_distances_typed` — `src/conditional_buddy/buddy_graph.py`

**Files:**
- Modify: `src/conditional_buddy/buddy_graph.py` (add one new function; no existing function changes)
- Test: `src/test/20260824_buddy_distance_mode/test_mix_distances_typed.py`

**Interfaces:**
- Consumes: nothing new — operates on the same `csr_matrix` types `rank_normalise_sparse`/`mix_distances` already produce/consume.
- Produces: `mix_distances_typed(D_img_n: csr_matrix, D_txt_n: csr_matrix, A_img: csr_matrix, A_txt: csr_matrix, alpha: float) -> csr_matrix`. Consumed by Task 2's modified `compute_buddy_init`.

- [ ] **Step 1: Write the failing test**

Create `src/test/20260824_buddy_distance_mode/test_mix_distances_typed.py`:

```python
"""
Test: buddy_graph.mix_distances_typed -- modality-provenance-aware distance mixing.
Uses each edge's own supporting modality's rank alone for img-only/txt-only edges;
keeps the existing fixed-alpha blend for "both" (cross-modally-confirmed) and "repair"
(neither modality) edges, where there is no disagreement to correct.

Run:
    python src/test/20260824_buddy_distance_mode/test_mix_distances_typed.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
from scipy.sparse import csr_matrix

from src.conditional_buddy.buddy_graph import mix_distances, mix_distances_typed


def test_both_and_repair_edges_keep_the_blend():
    """Edges present in both A_img and A_txt ("both"), or in neither ("repair"), must
    get the EXACT SAME value as the existing fixed-alpha mix_distances -- only
    single-modality edges should differ."""
    n = 4
    D_img = np.zeros((n, n))
    D_txt = np.zeros((n, n))
    D_img[0, 1] = D_img[1, 0] = 0.3   # (0,1): "both" edge
    D_txt[0, 1] = D_txt[1, 0] = 0.4
    D_img[2, 3] = D_img[3, 2] = 0.6   # (2,3): "repair" edge (in neither A_img nor A_txt)
    D_txt[2, 3] = D_txt[3, 2] = 0.7
    D_img_n, D_txt_n = csr_matrix(D_img), csr_matrix(D_txt)

    A_img = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    A_txt = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))

    alpha = 0.5
    mixed_typed = mix_distances_typed(D_img_n, D_txt_n, A_img, A_txt, alpha)
    mixed_blend = mix_distances(D_img_n, D_txt_n, alpha)

    assert abs(mixed_typed[0, 1] - mixed_blend[0, 1]) < 1e-9, (mixed_typed[0, 1], mixed_blend[0, 1])
    assert abs(mixed_typed[2, 3] - mixed_blend[2, 3]) < 1e-9, (mixed_typed[2, 3], mixed_blend[2, 3])
    print("PASS test_both_and_repair_edges_keep_the_blend")


def test_single_modality_edges_use_their_own_distance_alone():
    n = 4
    D_img = np.zeros((n, n))
    D_txt = np.zeros((n, n))
    D_img[0, 1] = D_img[1, 0] = 0.2   # img_only edge: good image rank...
    D_txt[0, 1] = D_txt[1, 0] = 0.9   # ...but bad (disagreeing) text rank
    D_img[2, 3] = D_img[3, 2] = 0.85  # txt_only edge: bad image rank...
    D_txt[2, 3] = D_txt[3, 2] = 0.15  # ...but good text rank
    D_img_n, D_txt_n = csr_matrix(D_img), csr_matrix(D_txt)

    A_img = csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))  # only (0,1)
    A_txt = csr_matrix(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))  # only (2,3)

    mixed = mix_distances_typed(D_img_n, D_txt_n, A_img, A_txt, alpha=0.5)
    assert abs(mixed[0, 1] - 0.2) < 1e-9, (
        f"img_only edge should use its own (image) distance alone, got {mixed[0, 1]}"
    )
    assert abs(mixed[2, 3] - 0.15) < 1e-9, (
        f"txt_only edge should use its own (text) distance alone, got {mixed[2, 3]}"
    )
    print("PASS test_single_modality_edges_use_their_own_distance_alone")


if __name__ == "__main__":
    test_both_and_repair_edges_keep_the_blend()
    test_single_modality_edges_use_their_own_distance_alone()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_buddy_distance_mode/test_mix_distances_typed.py
```
Expected: `ImportError: cannot import name 'mix_distances_typed' from 'src.conditional_buddy.buddy_graph'`.

- [ ] **Step 3: Implement `mix_distances_typed`**

Add the following to `src/conditional_buddy/buddy_graph.py`, directly after the existing `mix_distances` function:

```python
def mix_distances_typed(
    D_img_n: csr_matrix, D_txt_n: csr_matrix, A_img: csr_matrix, A_txt: csr_matrix,
    alpha: float,
) -> csr_matrix:
    """
    Modality-provenance-aware distance mixing. The fixed-alpha mix_distances() blends
    BOTH modalities' distance on EVERY edge of E, regardless of which modality(ies)
    originally justified that edge -- a diagnostic (src/test/20260824_buddy_graph_disagreement/)
    found this collapses ~98% of a real buddy graph's edges (single-modality-only) from a
    good rank (median 0.2-0.3) to statistical noise (median ~0.50) on real RedCaps data.

    This function instead uses each edge's OWN supporting modality's rank-normalised
    distance alone for edges supported by only one modality's mutual-kNN graph, and the
    existing fixed-alpha blend for edges supported by BOTH (no disagreement to correct)
    or by NEITHER (added by ensure_min_degree/ensure_connected -- not owned by either
    modality, so there is no single supporting distance to prefer).

    D_img_n, D_txt_n: rank-normalised distances on E's edges (same sparsity as each
        other, i.e. both built via sparse_cosine_distance(feats, E) then
        rank_normalise_sparse -- E's edges, not A_img's or A_txt's).
    A_img, A_txt: the ORIGINAL per-modality mutual-kNN graphs (pre-union, pre-repair) --
        used only to classify each edge of E, not to source any distance values.
    """
    N = D_img_n.shape[0]
    coo = D_img_n.tocoo()
    rows, cols = coo.row, coo.col
    d_img = coo.data
    # Index-based (not position-based) lookup -- do not assume D_txt_n's internal
    # storage order matches D_img_n's; scipy does not guarantee this across independently
    # rank-normalised matrices even when both share the same sparsity pattern.
    d_txt = np.asarray(D_txt_n.tocsr()[rows, cols]).ravel()

    def _keys(A: csr_matrix) -> np.ndarray:
        A_coo = A.tocoo()
        mask = A_coo.data != 0
        k = A_coo.row[mask].astype(np.int64) * N + A_coo.col[mask].astype(np.int64)
        k.sort()
        return k

    keys = rows.astype(np.int64) * N + cols.astype(np.int64)
    in_img = np.isin(keys, _keys(A_img))
    in_txt = np.isin(keys, _keys(A_txt))
    img_only = in_img & ~in_txt
    txt_only = ~in_img & in_txt

    mixed = alpha * d_img + (1.0 - alpha) * d_txt  # default: "both" and "repair" edges
    mixed = np.where(img_only, d_img, mixed)
    mixed = np.where(txt_only, d_txt, mixed)

    return csr_matrix((mixed, (rows, cols)), shape=D_img_n.shape)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python src/test/20260824_buddy_distance_mode/test_mix_distances_typed.py
```
Expected: `ALL TESTS PASSED`.

- [ ] **Step 5: Log the change**

Append to `.claude/20260824_log.md` (create only the header if the file doesn't exist yet in your checkout — it should already exist from earlier work this session):

```markdown
# /src/conditional_buddy/buddy_graph.py

## New function: `mix_distances_typed`

Modality-provenance-aware alternative to `mix_distances`: uses each edge's own
supporting modality's rank alone for img-only/txt-only edges instead of always blending
both modalities. Purely additive -- `mix_distances` is untouched, no existing caller
affected.

**Why:** Experiment 10 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
§4), following up on the modality-disagreement diagnostic
(`src/test/20260824_buddy_graph_disagreement/`).
```

- [ ] **Step 6: Commit**

```bash
git add src/conditional_buddy/buddy_graph.py src/test/20260824_buddy_distance_mode/test_mix_distances_typed.py .claude/20260824_log.md
git commit -m "feat: add mix_distances_typed for modality-provenance-aware distance mixing (Experiment 10)"
```

---

### Task 2: `distance_mode` parameter on `compute_buddy_init`

**Files:**
- Modify: `src/conditional_buddy/compute_buddies.py`
- Test: `src/test/20260824_buddy_distance_mode/test_compute_buddy_init_distance_mode.py`

**Interfaces:**
- Consumes: Task 1's `mix_distances_typed`.
- Produces: `compute_buddy_init(..., distance_mode: str = "blend", ...)`. Consumed by Task 3's `_buddy_init` wiring.

- [ ] **Step 1: Write the failing tests**

Create `src/test/20260824_buddy_distance_mode/test_compute_buddy_init_distance_mode.py`:

```python
"""
Test: compute_buddy_init's distance_mode parameter. "blend" (the default) must
reproduce the EXACT original fixed-alpha behavior for full backward compatibility;
"typed" must route through Task 1's mix_distances_typed and produce a different,
hand-verifiable result on a graph engineered to have real modality disagreement.

Run:
    python src/test/20260824_buddy_distance_mode/test_compute_buddy_init_distance_mode.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import torch

from src.conditional_buddy.compute_buddies import compute_buddy_init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()


def _two_cluster_features(n_per=60, dim=32, seed=1):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(0, 1, dim)
    c1 = rng.normal(5, 1, dim)
    labels = np.array([0] * n_per + [1] * n_per)
    centers = np.stack([c0, c1])
    img = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    txt = centers[labels] + rng.normal(0, 0.5, (2 * n_per, dim))
    return img.astype(np.float32), txt.astype(np.float32)


def test_blend_default_matches_no_arg_call():
    """distance_mode='blend' (explicit) must be numerically identical to calling
    compute_buddy_init with no distance_mode argument at all -- the core backward-
    compatibility property this whole plan depends on."""
    img, txt = _two_cluster_features(seed=2)
    emb_no_arg = compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF, seed=42)
    emb_explicit_blend = compute_buddy_init(
        img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF, seed=42, distance_mode="blend"
    )
    np.testing.assert_allclose(emb_no_arg, emb_explicit_blend, atol=1e-6)
    print("PASS test_blend_default_matches_no_arg_call")


def test_invalid_distance_mode_raises():
    img, txt = _two_cluster_features(seed=3)
    try:
        compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF,
                            distance_mode="bogus")
        raise AssertionError("expected ValueError for invalid distance_mode")
    except ValueError as e:
        assert "distance_mode" in str(e), str(e)
        print("PASS test_invalid_distance_mode_raises")


def test_typed_mode_changes_output_on_engineered_disagreement():
    """On a graph engineered to have real image/text disagreement (scramble some text
    rows relative to their images), 'typed' must produce a DIFFERENT embedding than
    'blend' -- if identical, the new code path was not actually exercised."""
    img, txt = _two_cluster_features(seed=4)
    rng = np.random.default_rng(5)
    scramble = rng.permutation(len(txt))[:20]
    txt = txt.copy()
    txt[scramble] = txt[scramble][::-1]

    emb_blend = compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF,
                                    seed=42, distance_mode="blend")
    emb_typed = compute_buddy_init(img, txt, n_dim=8, K=15, device=DEVICE, use_half=USE_HALF,
                                    seed=42, distance_mode="typed")
    assert not np.allclose(emb_blend, emb_typed), (
        "typed mode produced identical output to blend -- new code path not exercised"
    )
    print("PASS test_typed_mode_changes_output_on_engineered_disagreement")


if __name__ == "__main__":
    test_blend_default_matches_no_arg_call()
    test_invalid_distance_mode_raises()
    test_typed_mode_changes_output_on_engineered_disagreement()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_buddy_distance_mode/test_compute_buddy_init_distance_mode.py
```
Expected: `TypeError: compute_buddy_init() got an unexpected keyword argument 'distance_mode'`.

- [ ] **Step 3: Implement `distance_mode` in `compute_buddy_init`**

In `src/conditional_buddy/compute_buddies.py`, add `distance_mode: str = "blend",` to `compute_buddy_init`'s parameter list (immediately after the existing `b_weight: float = 1.0,` line), and add one sentence to its docstring documenting the new parameter:

```python
    distance_mode:     'blend' (default) uses the existing fixed-alpha
                       alpha*D_img+(1-alpha)*D_txt on every edge of E; 'typed' uses each
                       edge's own supporting modality's rank alone for img-only/txt-only
                       edges (see mix_distances_typed) -- 'blend' exactly reproduces the
                       function's pre-2026-08-24 behavior; only pass 'typed' explicitly.
```

Then replace the current Step 3/4 block:

```python
    # Step 3: sparse per-modality distances on E's edges
    D_img = sparse_cosine_distance(img_n, E)
    D_txt = sparse_cosine_distance(txt_n, E)

    # Step 4: rank-normalise and mix
    D_mixed = mix_distances(
        rank_normalise_sparse(D_img), rank_normalise_sparse(D_txt), alpha
    )
    print(f"[buddies] Step 4: mixed distance matrix nnz={D_mixed.nnz:,} (alpha={alpha})")
```

with:

```python
    # Step 3: sparse per-modality distances on E's edges
    D_img = sparse_cosine_distance(img_n, E)
    D_txt = sparse_cosine_distance(txt_n, E)

    # Step 4: rank-normalise and mix
    if distance_mode not in ("blend", "typed"):
        raise ValueError(
            f"Unknown distance_mode '{distance_mode}'. Use 'blend' (default) or 'typed'."
        )
    D_img_n = rank_normalise_sparse(D_img)
    D_txt_n = rank_normalise_sparse(D_txt)
    if distance_mode == "typed":
        D_mixed = mix_distances_typed(D_img_n, D_txt_n, A_img, A_txt, alpha)
    else:
        D_mixed = mix_distances(D_img_n, D_txt_n, alpha)
    print(f"[buddies] Step 4: mixed distance matrix nnz={D_mixed.nnz:,} "
          f"(alpha={alpha}, distance_mode={distance_mode})")
```

Finally, add `mix_distances_typed` to the existing `from .buddy_graph import (...)` block at the top of the file (alongside `mix_distances`, `rank_normalise_sparse`, etc. — keep the import list alphabetically consistent with how it's already ordered).

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python src/test/20260824_buddy_distance_mode/test_compute_buddy_init_distance_mode.py
```
Expected: `ALL TESTS PASSED`.

- [ ] **Step 5: Regression-check the existing synthetic test suite**

```bash
python src/test/20260609_conditional_buddy/test_compute_buddies.py
```
Expected: still `PASS` on both existing tests (`test_shape_and_range`, `test_buddies_closer_than_random`) — confirms the `distance_mode` addition didn't disturb the default path for any pre-existing caller.

- [ ] **Step 6: Log the change**

Append to `.claude/20260824_log.md`:

```markdown
# /src/conditional_buddy/compute_buddies.py

## `compute_buddy_init`: added `distance_mode` parameter

**Before:** always used the fixed-alpha `mix_distances` blend on every edge of E.

**After:** `distance_mode='blend'` (default) is byte-for-byte the same as before;
`distance_mode='typed'` routes through `mix_distances_typed` (Task 1) instead. Verified
backward-compatible via `test_blend_default_matches_no_arg_call` and the pre-existing
`test_compute_buddies.py` synthetic suite.

**Why:** Experiment 10 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
§4). See `docs/superpowers/plans/2026-08-24-buddy-distance-mode-ablation.md` Task 2.
```

- [ ] **Step 7: Commit**

```bash
git add src/conditional_buddy/compute_buddies.py src/test/20260824_buddy_distance_mode/test_compute_buddy_init_distance_mode.py .claude/20260824_log.md
git commit -m "feat: add distance_mode parameter to compute_buddy_init (Experiment 10)"
```

---

### Task 3: Wire `train.buddies.distance_mode` into training

**Files:**
- Modify: `src/utils/embedding_manager_nocache.py:345-415` (`_buddy_init`)
- Modify: `src/hook/train_cosir.py:253-286` (`_buddy_kwargs`/`_extra` block)

**Interfaces:**
- Consumes: Task 2's `compute_buddy_init(..., distance_mode=...)`.
- Produces: a new Hydra override `+train.buddies.distance_mode=typed` that a training run can set; absent by default (preserves current behavior exactly). Consumed by Task 4's sweep script.

- [ ] **Step 1: Modify `_buddy_init`**

In `src/utils/embedding_manager_nocache.py`, add `distance_mode: str = "blend",` to `_buddy_init`'s parameter list (immediately after the existing `feature_override: Optional[...] = None,` line), and add `distance_mode=distance_mode,` to its call to `compute_buddy_init(...)` (immediately after the existing `b_weight=b_weight,` line). Add one line to the docstring:

```python
        distance_mode: 'blend' (default) or 'typed' -- forwarded to compute_buddy_init
                  unchanged; see that function's docstring.
```

- [ ] **Step 2: Modify `train_cosir.py`'s `_buddy_kwargs`/`_extra` block**

The block currently reads (around line 253):

```python
    _bud = getattr(cfg.train, "buddies", None)
    _buddy_kwargs = {
        "k": int(getattr(_bud, "k", 30)) if _bud is not None else 30,
        "alpha": float(getattr(_bud, "alpha", 0.5)) if _bud is not None else 0.5,
        "method": str(getattr(_bud, "method", "spectral")) if _bud is not None else "spectral",
        "knn_batch_size": int(getattr(_bud, "knn_batch_size", 1024)) if _bud is not None else 1024,
        "normalize_method": str(getattr(_bud, "normalize_method", "rank")) if _bud is not None else "rank",
        "seed": int(cfg.seed),
        "b_weight": float(getattr(_bud, "b_weight", 1.0)) if _bud is not None else 1.0,
    }
```

Add one new key to that dict (after `"b_weight": ...,`):

```python
        "distance_mode": str(getattr(_bud, "distance_mode", "blend")) if _bud is not None else "blend",
```

Then, in the `_extra` block further down (after the existing `if _buddy_kwargs["b_weight"] != 1.0: _extra["b_weight"] = ...` lines, and after the existing `if _encoder_pair: _extra["encoder_pair"] = _encoder_pair` line), add:

```python
        # Only add distance_mode to the template key when it departs from the default,
        # so existing (pre-distance_mode) templates stay compatible for standard runs
        # while a changed mode still forces a rebuild (no silent template reuse across
        # blend/typed).
        if _buddy_kwargs["distance_mode"] != "blend":
            _extra["distance_mode"] = _buddy_kwargs["distance_mode"]
```

- [ ] **Step 3: Smoke-test the new override, and its equivalence to the default path**

Same equivalence-check pattern as Experiment 8's Task 3 (`+train.buddies.encoder_pair=clip_img:clip_txt` vs. no override) — here, confirm `distance_mode=blend` (explicit) reproduces the no-override path, and that `distance_mode=typed` visibly takes the new branch:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python main_cosir.py \
  dataset=redcaps_150k eval.evaluation_interval=1 eval.oracle_aggregation=max \
  eval.test_ratio=0.2 model=clip_base model.num_layers=6 model.embedding_dim=16 \
  optimizer.lr=1e-3 optimizer.lr_label=1e-4 seed=1 \
  train.initialization_strategy=buddies train.buddies.alpha=0.5 \
  +train.buddies.distance_mode=blend train.epochs=2 \
  experiment.results_dir=res/CoSiR_buddy_distance_mode_ablation/_smoke/blend \
  wandb.group="buddy-distance-mode-ablation-smoke"

python main_cosir.py \
  dataset=redcaps_150k eval.evaluation_interval=1 eval.oracle_aggregation=max \
  eval.test_ratio=0.2 model=clip_base model.num_layers=6 model.embedding_dim=16 \
  optimizer.lr=1e-3 optimizer.lr_label=1e-4 seed=1 \
  train.initialization_strategy=buddies train.buddies.alpha=0.5 \
  +train.buddies.distance_mode=typed train.epochs=2 \
  experiment.results_dir=res/CoSiR_buddy_distance_mode_ablation/_smoke/typed \
  wandb.group="buddy-distance-mode-ablation-smoke"
```
Expected: both runs complete 2 epochs with no traceback. The `distance_mode=blend` run's log line `[buddies] Step 4: mixed distance matrix nnz=... (alpha=0.5, distance_mode=blend)` should show `distance_mode=blend`; the `typed` run's equivalent line should show `distance_mode=typed`. Check both runs' `test_oracle/t2i_R1`/`i2t_R1` via wandb — they are NOT expected to match each other here (unlike Experiment 8's CLIP-pair equivalence check, `blend` and `typed` are genuinely different constructions by design) — this step is a pipeline-sanity smoke test, not an equivalence check. Do confirm `test_raw/t2i_R1` and `test_raw/i2t_R1` (frozen CLIP, no conditioning) ARE identical between the two runs (same backbone, same test set) — if they differ, something unrelated to this change is broken and must be investigated before proceeding.

- [ ] **Step 4: Log the change**

Append to `.claude/20260824_log.md`:

```markdown
# /src/utils/embedding_manager_nocache.py (second entry)

## `_buddy_init`: added `distance_mode` parameter (forwarded, no new logic)

Added `distance_mode: str = "blend"`, forwarded unchanged to `compute_buddy_init`. No
new logic in this file -- see compute_buddies.py's log entry (Task 2) for the actual
behavior change.

# /src/hook/train_cosir.py (second entry)

## `_init_embedding_manager`: added `train.buddies.distance_mode` override

**Before:** the `buddies` strategy always used the fixed-alpha blend (no way to select
`typed` distance mixing).

**After:** an optional `+train.buddies.distance_mode=typed` Hydra override selects the
new construction (Task 1/2); absent by default -- behavior unchanged for every existing
config/sweep. Also added to the buddies template-compatibility `extra` dict so a stale
template from one mode is never silently reused under the other.

**Why:** Experiment 10 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
§4). See `docs/superpowers/plans/2026-08-24-buddy-distance-mode-ablation.md` Task 3.
```

- [ ] **Step 5: Commit**

```bash
git add src/utils/embedding_manager_nocache.py src/hook/train_cosir.py .claude/20260824_log.md
git commit -m "feat: wire train.buddies.distance_mode override into training (Experiment 10)"
```

---

### Task 4: Sweep script — `scripts/run_buddy_distance_mode_ablation.sh`

**Files:**
- Create: `scripts/run_buddy_distance_mode_ablation.sh`

**Interfaces:**
- Consumes: Task 3's `+train.buddies.distance_mode` override.
- Produces: per-mode experiment directories under `${BASE_RESULTS_DIR}/mode_{blend,typed}/`, wandb runs tagged `${WANDB_TAG}` in group `buddy distance-mode ablation` — consumed by Task 5's analysis script.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
set -euo pipefail
# Experiment 10 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md §4):
# does modality-provenance-aware distance mixing ("typed") beat the current fixed-alpha
# blend ("blend") on retrieval, and does it narrow C6's still-open gap to raw CLIP
# (test_pre_diff)? Same operating point and isolation discipline as Experiment 1:
# initialization_strategy=buddies fixed, all training-time buddy terms OFF, only the
# init-construction's distance_mode varies.
#
# train.buddies.distance_mode is a TEMPLATE-COMPATIBILITY key, exactly like
# initialization_strategy/encoder_pair: each mode gets its OWN results_dir so its own
# template_embeddings/, avoiding template-reuse races.
#
#   SMOKE=1 bash scripts/run_buddy_distance_mode_ablation.sh   # 2 epochs, seed=1, both modes
#   bash scripts/run_buddy_distance_mode_ablation.sh           # full sweep (6 runs)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DISTANCE_MODE_SWEEP="${DISTANCE_MODE_SWEEP:-blend typed}"
SEED_SWEEP="${SEED_SWEEP:-1,2,3}"

LR_SWEEP="${LR_SWEEP:-1e-3}"
LR_LABEL_SWEEP="${LR_LABEL_SWEEP:-1e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-16}"
ALPHA="${ALPHA:-0.5}"

DATASET="${DATASET:-redcaps_150k}"
TEST_RATIO="${TEST_RATIO:-0.2}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-res/CoSiR_buddy_distance_mode_ablation/${DATASET}}"
WANDB_TAG="${WANDB_TAG:-buddy-distance-mode-ablation-${DATASET}}"
WANDB_GROUP="${WANDB_GROUP:-buddy distance-mode ablation}"

if [ -n "${SMOKE:-}" ]; then
  EPOCHS="${EPOCHS:-2}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-1}"
  SEED_SWEEP="${SEED_SWEEP_SMOKE:-1}"
  WANDB_TAG="${WANDB_TAG}-smoke"
  echo ">>> SMOKE: 2 epochs, seed=1, both modes — template-build + pipeline sanity"
else
  EPOCHS="${EPOCHS:-100}"
  EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
fi

echo "==================================================================="
echo "Buddy distance-mode ablation ($DATASET): {$DISTANCE_MODE_SWEEP} x seeds={$SEED_SWEEP}"
echo "  fixed: lr=$LR_SWEEP lr_label=$LR_LABEL_SWEEP dim=$EMBEDDING_DIM alpha=$ALPHA, initialization_strategy=buddies"
echo "  EPOCHS=$EPOCHS EVAL_INTERVAL=$EVAL_INTERVAL tag=$WANDB_TAG group=$WANDB_GROUP"
echo "  training-time buddy terms: OFF (not passed -> default lambda_buddy=0, lambda_buddy_con=0, buddy_refresh=False)"
echo "==================================================================="

for MODE in $DISTANCE_MODE_SWEEP; do
  RD="${BASE_RESULTS_DIR}/mode_${MODE}"
  echo ">>> distance_mode=${MODE}  ->  results_dir=${RD}"
  python main_cosir.py -m \
    dataset="$DATASET" \
    eval.evaluation_interval="$EVAL_INTERVAL" \
    eval.oracle_aggregation=max \
    eval.test_ratio="$TEST_RATIO" \
    model=clip_base \
    model.num_layers=6 \
    model.embedding_dim="$EMBEDDING_DIM" \
    optimizer.lr="$LR_SWEEP" \
    optimizer.lr_label="$LR_LABEL_SWEEP" \
    seed="$SEED_SWEEP" \
    train.initialization_strategy=buddies \
    train.buddies.alpha="$ALPHA" \
    +train.buddies.distance_mode="$MODE" \
    train.epochs="$EPOCHS" \
    experiment.results_dir="$RD" \
    wandb.group="$WANDB_GROUP" \
    +loss.log_buddy_preservation=true \
    ${WANDB_TAG:+++wandb.tags=[$WANDB_TAG]} \
    ${EXTRA_OVERRIDES:-}
done

echo "==================================================================="
echo "Done. Analyse (paired, typed vs. blend, mean delta +/- std, plus test_pre_diff gap-to-CLIP) with:"
echo "  python scripts/analyze_buddy_distance_mode_ablation.py --tag $WANDB_TAG"
echo "==================================================================="
```

- [ ] **Step 2: Smoke-test it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 bash scripts/run_buddy_distance_mode_ablation.sh
```
Expected: two short runs (`mode_blend`, `mode_typed`), 2 epochs each, no traceback, ending with the `Done. Analyse ...` banner. Confirm `mode_typed`'s log shows `distance_mode=typed` in the `[buddies] Step 4` line (matching Task 3 Step 3's manual check, now via the actual sweep script).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_buddy_distance_mode_ablation.sh
git commit -m "feat: add buddy distance-mode ablation sweep runner (Experiment 10)"
```

---

### Task 5: Analysis script — `scripts/analyze_buddy_distance_mode_ablation.py`

**Files:**
- Create: `scripts/analyze_buddy_distance_mode_ablation.py`

**Interfaces:**
- Consumes: wandb runs from Task 4 (group `buddy distance-mode ablation`).
- Produces: printed paired Δ tables (`typed` vs. `blend`, mean ± std, mean/SEM) for `test_oracle` (retrieval) and `test_pre_diff` (gap to CLIP) — read directly by Task 7's report-writing step.

Same testing approach as `scripts/analyze_init_ablation.py`/`scripts/analyze_buddy_init_encoder_ablation.py`: one pure, offline-testable core, TDD'd via `--selftest`. **Unlike Experiment 8's Task 5, this script's `fetch()` includes the non-finished-run filter from the start** (per this plan's Global Constraints) — do not write the version without it.

- [ ] **Step 1: Write the failing selftest**

Create `scripts/analyze_buddy_distance_mode_ablation.py` with only the imports, constants, and `_selftest()`:

```python
"""
Paired analysis for Experiment 10 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
S4): does modality-provenance-aware distance mixing ("typed") beat the current fixed-alpha
blend ("blend") on retrieval, and does it narrow C6's gap to raw CLIP (test_pre_diff)?

Reads wandb runs from the 'buddy distance-mode ablation' group
(scripts/run_buddy_distance_mode_ablation.sh), pairs typed vs blend WITHIN each seed, and
reports mean delta +/- std and mean/SEM (spec S5) for test_oracle (retrieval, higher is
better) AND test_pre_diff (ours - CLIP, already logged automatically by every eval call
in src/eval/pipeline.py -- LESS NEGATIVE / higher delta = narrower gap to CLIP).

Usage
-----
  python scripts/analyze_buddy_distance_mode_ablation.py --tag buddy-distance-mode-ablation-redcaps_150k
  python scripts/analyze_buddy_distance_mode_ablation.py --selftest   # offline check, no wandb call

Requires: wandb, pandas, numpy (all already deps).
"""
import argparse

import numpy as np
import pandas as pd

METRICS = [
    "test_oracle/t2i_R1", "test_oracle/i2t_R1",
    "test_pre_diff/t2i_R1", "test_pre_diff/i2t_R1",
    "test_raw/t2i_R1", "test_raw/i2t_R1",
]
BASELINE = "blend"
TREATMENT = "typed"
CELL = [("distance_mode", ("train", "buddies", "distance_mode")), ("seed", ("seed",))]


def _selftest():
    """Offline arithmetic check - no wandb call. Verifies compute_paired_deltas/summarize
    against hand-computed numbers, for both a retrieval metric and the gap-to-CLIP metric,
    before ever touching real run data."""
    df = pd.DataFrame([
        {"distance_mode": "blend", "seed": 1, "test_oracle/t2i_R1": 50.0, "test_pre_diff/t2i_R1": -10.0},
        {"distance_mode": "typed", "seed": 1, "test_oracle/t2i_R1": 52.0, "test_pre_diff/t2i_R1": -8.0},
        {"distance_mode": "blend", "seed": 2, "test_oracle/t2i_R1": 48.0, "test_pre_diff/t2i_R1": -11.0},
        {"distance_mode": "typed", "seed": 2, "test_oracle/t2i_R1": 51.0, "test_pre_diff/t2i_R1": -9.0},
        {"distance_mode": "blend", "seed": 3, "test_oracle/t2i_R1": 49.0, "test_pre_diff/t2i_R1": -10.5},
        {"distance_mode": "typed", "seed": 3, "test_oracle/t2i_R1": 49.0, "test_pre_diff/t2i_R1": -9.5},
    ])
    deltas = compute_paired_deltas(df, "test_oracle/t2i_R1")
    assert len(deltas) == 3, f"expected 3 paired cells, got {len(deltas)}"
    got = sorted(d for _, d in deltas)
    want = [0.0, 2.0, 3.0]
    assert got == want, f"expected deltas {want}, got {got}"
    s = summarize(deltas)
    assert s["n"] == 3
    assert abs(s["mean"] - (5.0 / 3)) < 1e-9

    pre_deltas = compute_paired_deltas(df, "test_pre_diff/t2i_R1")
    pre_s = summarize(pre_deltas)
    assert pre_s["n"] == 3
    assert pre_s["mean"] > 0, (
        "expected a positive test_pre_diff delta in this toy example "
        "(typed less negative than blend -> narrower gap to CLIP)"
    )
    print("SELFTEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_buddy_distance_mode_ablation.py --selftest
```
Expected: `NameError: name 'compute_paired_deltas' is not defined`.

- [ ] **Step 3: Implement the core functions and CLI**

Add the following above `_selftest()` (after the `CELL` constant):

```python
def cget(cfg, path, default=None):
    d = cfg
    for p in path:
        if d is None:
            return default
        try:
            d = d.get(p) if hasattr(d, "get") else getattr(d, p, None)
        except Exception:
            return default
    return default if d is None else d


def sget(summ, key, default=np.nan):
    try:
        v = summ.get(key, default)
    except Exception:
        v = getattr(summ, key, default)
    return default if v is None else v


def fetch(entity, project, group, tag=None):
    import wandb
    api = wandb.Api()
    rows = []
    skipped_unfinished = 0
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        if tag and tag not in (run.tags or []):
            continue
        if run.state != "finished":
            skipped_unfinished += 1
            continue
        cfg, summ = run.config, run.summary
        mode = cget(cfg, ("train", "buddies", "distance_mode"))
        if mode not in (BASELINE, TREATMENT):
            continue
        row = {"run_id": run.id, "distance_mode": mode}
        for metric in METRICS:
            v = sget(summ, metric)
            row[metric] = float(v) if not (isinstance(v, float) and np.isnan(v)) else np.nan
        for cname, cpath in CELL:
            cv = cget(cfg, cpath)
            row[cname] = cv
        rows.append(row)
    if skipped_unfinished:
        print(f"  ({skipped_unfinished} non-finished run(s) under this group excluded from analysis)")
    return pd.DataFrame(rows)


def compute_paired_deltas(df, metric):
    """Pair BASELINE vs TREATMENT within each seed. Returns list of (seed, delta) where
    delta = treatment - baseline. Pure function - no wandb, no I/O."""
    deltas = []
    for seed, cell in df.groupby("seed"):
        by_mode = cell.groupby("distance_mode")[metric].max()
        if BASELINE not in by_mode.index or TREATMENT not in by_mode.index:
            continue
        b, t = by_mode[BASELINE], by_mode[TREATMENT]
        if np.isnan(b) or np.isnan(t):
            continue
        deltas.append((seed, t - b))
    return deltas


def summarize(deltas):
    """mean, std, sem, z=mean/sem, win-rate - matches this project's existing convention."""
    n = len(deltas)
    if n == 0:
        return {"n": 0}
    arr = np.asarray([d for _, d in deltas], dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    z = mean / sem if (n > 1 and sem > 0) else float("nan")
    wins = int((arr > 0).sum())
    return {"n": n, "mean": mean, "std": std, "sem": sem, "z": z, "wins": wins}


def analyze(entity, project, group, tag=None):
    print(f"\n{'='*78}\nExperiment 10 - buddy distance-mode ablation  group='{group}'"
          + (f"  tag='{tag}'" if tag else "") + f"\n{'='*78}")
    df = fetch(entity, project, group, tag=tag)
    if df.empty:
        print("  (no runs found - check --entity/--project/--tag)")
        return
    modes = sorted(df["distance_mode"].unique())
    print(f"  {len(df)} finished run(s); modes present: {modes}")

    for label, metric in [
        ("retrieval t2i R1", "test_oracle/t2i_R1"),
        ("retrieval i2t R1", "test_oracle/i2t_R1"),
        ("gap-to-CLIP t2i (ours-CLIP)", "test_pre_diff/t2i_R1"),
        ("gap-to-CLIP i2t (ours-CLIP)", "test_pre_diff/i2t_R1"),
    ]:
        deltas = compute_paired_deltas(df, metric)
        s = summarize(deltas)
        if s["n"] == 0:
            print(f"\n  {label}: (no paired cells with both {BASELINE} and {TREATMENT} present)")
            continue
        sig = f"  mean/SEM={s['z']:+.1f}{' *' if not np.isnan(s['z']) and abs(s['z']) >= 2 else ''}" if s["n"] > 1 else ""
        spread = f" +/- {s['std']:.2f}" if s["n"] > 1 else ""
        print(f"\n  {label}: typed - blend, over {s['n']} seed(s), "
              f"{TREATMENT} wins {s['wins']}/{s['n']} "
              f"(mean delta = {s['mean']:+.2f}{spread}){sig}")

    for metric in ("test_raw/t2i_R1", "test_raw/i2t_R1"):
        vals = df[metric].dropna().unique()
        print(f"\n  {metric} distinct values across all runs: {sorted(vals)} "
              f"(sanity check - should be a single value, same frozen backbone/test set)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="augustoxq")
    ap.add_argument("--project", default="cosir_image")
    ap.add_argument("--group", default="buddy distance-mode ablation")
    ap.add_argument("--tag", default=None, help="only include runs carrying this wandb tag")
    ap.add_argument("--selftest", action="store_true", help="offline arithmetic check, no wandb call")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    analyze(args.entity, args.project, args.group, tag=args.tag)


if __name__ == "__main__":
    main()
```

Replace the Step 1 placeholder `if __name__ == "__main__":` block with the one above (it supersedes it).

- [ ] **Step 4: Run the selftest again to verify it passes**

```bash
python scripts/analyze_buddy_distance_mode_ablation.py --selftest
```
Expected: `SELFTEST OK`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_buddy_distance_mode_ablation.py
git commit -m "feat: add paired analysis script for buddy distance-mode ablation (Experiment 10)"
```

---

### Task 6: Launch the full sweep

**Files:** none (execution only).

**Interfaces:**
- Consumes: Task 4's sweep script.
- Produces: 6 finished wandb runs (2 modes × 3 seeds) feeding Task 7's analysis.

- [ ] **Step 1: Confirm smoke tests and equivalence checks passed**

Verify Task 3 Step 3's manual smoke pair and Task 4 Step 2's sweep-script smoke test both completed without error, and that `test_raw` matched between the two smoke runs, before spending full compute. Clean up any leftover `res/CoSiR_buddy_distance_mode_ablation/_smoke/` or smoke-tagged sweep-script output directories first (same hygiene lesson as Experiment 8's Task 4/6).

- [ ] **Step 2: Launch the full sweep**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_buddy_distance_mode_ablation.sh
```
This runs 2 modes × 3 seeds = 6 runs at 100 epochs each — the same cost as the cheapest cell of Experiment 1. Launch with `run_in_background` or `nohup ... &` if executing interactively.

- [ ] **Step 3: Verify all 6 runs finished**

Check the wandb UI (project `cosir_image`, group `buddy distance-mode ablation`, tag `buddy-distance-mode-ablation-redcaps_150k`) or query via `wandb.Api()` that all 6 runs show `state == "finished"` with `test_oracle/*`, `test_pre_diff/*`, and `test_raw/*` present in their summary.

---

### Task 7: Analyze results and write the report

**Files:**
- Create: `docs/reports/2026-08-24_buddy_distance_mode_ablation.md` (adjust date to when Task 6 actually completes)
- Modify: `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (add the Experiment 10 outcome, per its success criteria in §4)

**Interfaces:**
- Consumes: `scripts/analyze_buddy_distance_mode_ablation.py` (Task 5) output.
- Produces: a direct answer to C6's open "does anything close the gap to CLIP" question.

- [ ] **Step 1: Run the analysis**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_buddy_distance_mode_ablation.py --tag buddy-distance-mode-ablation-redcaps_150k
```
Capture the full printed output (retrieval deltas, gap-to-CLIP deltas, the `test_raw` sanity check).

- [ ] **Step 2: Apply the spec's decision rule**

Per spec §4 Experiment 10's success criteria: **positive** (retrieval win, seed-replicated, mean/SEM ≥ 2, AND narrower `test_pre_diff` gap) → genuine methodological improvement, candidate new default; **null** (no reliable retrieval difference) → the dilution is real but doesn't propagate to a measurable training effect, still a legitimate negative result; **negative** (`typed` worse) → report as found, brief investigation only if time allows.

- [ ] **Step 3: Write the results report**

Create `docs/reports/2026-08-24_buddy_distance_mode_ablation.md` following the structure of `docs/reports/2026-08-19_buddy_init_ablation_redcaps_300k.md` (which already established the `test_raw`/`test_pre_diff` reporting convention): method, the diagnostic's motivating numbers (cite `src/test/20260824_buddy_graph_disagreement/`), results tables (retrieval AND gap-to-CLIP), the decision-rule outcome, caveats, reproduction commands.

- [ ] **Step 4: Update the spec**

In `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`, add the Experiment 10 outcome as a new row in §2's claims table (next available letter), citing the new report, and update §3.3's framing paragraph if the result is positive enough to change the "does anything close the gap to CLIP" narrative established by C6.

- [ ] **Step 5: Commit**

```bash
git add docs/reports/2026-08-24_buddy_distance_mode_ablation.md docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
git commit -m "results: buddy distance-mode ablation (Experiment 10)"
```

---

## Self-Review

- **Spec coverage:** Tasks 1–3 build the (previously nonexistent) `distance_mode` capability end-to-end (graph-mixing function → `compute_buddy_init` parameter → training config wiring). Tasks 4–7 cover the sweep, analysis (including the `test_pre_diff` gap-to-CLIP comparison that is the actual motivating question), execution, and report/spec update.
- **Placeholder scan:** every code block is complete and runnable as written; no bracketed placeholders.
- **Type/interface consistency:** `mix_distances_typed(D_img_n, D_txt_n, A_img, A_txt, alpha) -> csr_matrix` (Task 1) is called identically in Task 2's `compute_buddy_init` modification. `distance_mode: str = "blend"` threads through `compute_buddy_init` (Task 2) → `_buddy_init` (Task 3) → `_buddy_kwargs`/`+train.buddies.distance_mode` (Task 3) → the sweep script's `+train.buddies.distance_mode="$MODE"` (Task 4) with the same string values (`"blend"`/`"typed"`) at every hop. `compute_paired_deltas(df, metric) -> list[(seed, float)]` and `summarize(deltas) -> dict` (Task 5) are defined once and used identically in `analyze()`/`_selftest()`.
- **Scope check:** RedCaps-150k, 2 modes, 3 seeds (6 runs) only — matches the spec's stated scope exactly, no 300k extension bundled into this plan.
- **Backward compatibility (the plan's core risk):** explicitly tested at three levels — Task 2's `test_blend_default_matches_no_arg_call` (unit), Task 2 Step 5's regression run of the pre-existing `test_compute_buddies.py` suite (integration), and Task 3 Step 3's `test_raw` cross-check between the two smoke runs (end-to-end, real training). A failure at any level blocks proceeding to Task 4, per each task's own steps.
