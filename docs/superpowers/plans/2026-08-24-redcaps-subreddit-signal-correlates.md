# RedCaps Subreddit Signal-Strength Correlates (Experiment 9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer Experiment 9 from `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 — deepen C1's aggregate ~20× RedCaps subreddit-lift number into a per-subreddit breakdown, and check whether subreddit size, caption diversity, and visual homogeneity predict where the buddy signal is strongest.

**Architecture:** `subreddit_lift()` (`src/test/20260623_redcaps_buddy/redcaps_buddy.py`) already computes a per-subreddit lift array internally but only returns the top-15 most-enriched. This plan extends it to optionally return every qualifying subreddit (Task 1), then adds a small, self-contained analysis script that pulls three per-subreddit properties (sample count, caption diversity, visual homogeneity — the latter two via a closed-form mean-pairwise-cosine-similarity identity, avoiding an O(n²) pass) from the same already-loaded `Data` object, and correlates them against the extended lift array (Task 2). No training, no new features, no graph rebuilding — this is a pure analysis pass over data the pipeline already has cached.

**Tech Stack:** Python 3.10, numpy, matplotlib (`Agg` backend, matching this codebase's existing convention). No new dependencies, no GPU required.

**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 9 (added 2026-08-24).

## Global Constraints

- Always run Python commands with `source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR` first (per `.claude/CLAUDE.md`).
- This deepens the C1 signal-validation claim (spec §2) — it is not a re-measurement of the downstream training-use claims (C5/C6) and must not be conflated with them in the eventual report.
- `subreddit_lift`'s existing default behavior (`top_k=15`) must not change — every existing caller (`cross_vlm_buddy.py`'s core validation, any prior report quoting "top 15") keeps working unmodified.
- `src/test/20260623_redcaps_buddy/redcaps_buddy.py` is modified by this plan (an existing, shared, multiply-imported module — used by `cross_vlm_buddy.py`, `phase2_vlm.py`, `run_phase1.py`, `run_structure.py`). Per CLAUDE.md, log the change in `.claude/20260824_log.md` (reuse the file created by the Experiment 8 plan if it already exists in this session; otherwise create it).
- No training compute — this whole plan is CPU-only analysis over already-cached RedCaps CLIP features (`redcaps_buddy.load_data()`).

---

### Task 1: Extend `subreddit_lift` to return every qualifying subreddit

**Files:**
- Modify: `src/test/20260623_redcaps_buddy/redcaps_buddy.py:99-136` (`subreddit_lift`)
- Test: `src/test/20260824_redcaps_subreddit_correlates/test_subreddit_lift_all.py`
- Modify: `.claude/20260824_log.md` (append, or create if Experiment 8's plan hasn't run yet)

**Interfaces:**
- Consumes: nothing new — same `Data` and edge-array inputs `subreddit_lift` already takes.
- Produces: `subreddit_lift(data, e, top_k=15)` — `top_k=None` now returns every subreddit passing the existing `exp_s > 5` reliability filter (previously always truncated to 15), sorted by lift descending, same as before. Consumed by Task 2's correlation script.

- [ ] **Step 1: Write the failing test**

Create `src/test/20260824_redcaps_subreddit_correlates/test_subreddit_lift_all.py`:

```python
"""
Test: subreddit_lift(..., top_k=None) returns every subreddit passing the exp_s > 5
reliability filter, not just the top 15 — and top_k=15 (the existing default) is unchanged.

Run:
    python src/test/20260824_redcaps_subreddit_correlates/test_subreddit_lift_all.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

import numpy as np

from redcaps_buddy import Data, subreddit_lift


def _synthetic_data(n_subs=20, per_sub=30, seed=1):
    """n_subs subreddits x per_sub samples each; every subreddit gets some same-sub edges
    so all pass the exp_s > 5 filter."""
    rng = np.random.default_rng(seed)
    n = n_subs * per_sub
    sub_id = np.repeat(np.arange(n_subs), per_sub)
    img = np.zeros((n, 4), dtype=np.float32)  # unused by subreddit_lift
    txt = np.zeros((n, 4), dtype=np.float32)
    sample_ids = list(range(n))
    sub_names = [f"sub{i}" for i in range(n_subs)]
    records = [{} for _ in range(n)]
    data = Data(img, txt, sample_ids, sub_id, sub_names, records)

    # Build enough same-subreddit edges per subreddit to clear exp_s > 5, with varying
    # density per subreddit so lift genuinely differs across subreddits.
    edges = []
    for s in range(n_subs):
        idx = np.where(sub_id == s)[0]
        n_edges = 10 + s  # increasing density -> increasing lift by construction
        pairs = rng.choice(idx, size=(n_edges, 2))
        edges.extend(pairs.tolist())
    e = np.array(edges, dtype=np.int64)
    return data, e


def test_top_k_none_returns_all_qualifying():
    data, e = _synthetic_data(n_subs=20)
    result_all = subreddit_lift(data, e, top_k=None)
    result_15 = subreddit_lift(data, e, top_k=15)
    assert len(result_15["top_enriched"]) == 15
    assert len(result_all["top_enriched"]) >= 15, (
        f"expected >=15 qualifying subreddits, got {len(result_all['top_enriched'])}"
    )
    # The top-15 (by lift, descending) from the top_k=None result must exactly match the
    # top_k=15 result — same ranking, just not truncated.
    names_all_top15 = [name for name, _, _ in result_all["top_enriched"][:15]]
    names_15 = [name for name, _, _ in result_15["top_enriched"]]
    assert names_all_top15 == names_15, (names_all_top15, names_15)
    print(f"PASS test_top_k_none_returns_all_qualifying "
          f"({len(result_all['top_enriched'])} qualifying subreddits)")


def test_default_top_k_unchanged():
    data, e = _synthetic_data(n_subs=20)
    result = subreddit_lift(data, e)  # no top_k passed -> must still default to 15
    assert len(result["top_enriched"]) == 15
    print("PASS test_default_top_k_unchanged")


if __name__ == "__main__":
    test_top_k_none_returns_all_qualifying()
    test_default_top_k_unchanged()
    print("ALL TESTS PASSED")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_redcaps_subreddit_correlates/test_subreddit_lift_all.py
```
Expected: `TypeError: subreddit_lift() got an unexpected keyword argument 'top_k'` is NOT the error (top_k already exists) — instead expect an `AssertionError` from `test_top_k_none_returns_all_qualifying` (currently `top_k=None` would be passed straight into `order[:None]`, which in Python slicing means "no truncation" already — **check this first**: `order[:top_k]` with `top_k=None` is actually already a no-op slice in Python. Run the test as-is; if it already passes, skip to Step 4 and note in the commit message that `top_k=None` worked without modification. If it fails (e.g. because the `if exp_s[i] > 5` filter combined with `order[:top_k]` behaves differently than expected, or `top_k=None` was never exercised before and something else breaks), proceed to Step 3.

- [ ] **Step 3: Implement (only if Step 2 showed a real failure)**

In `src/test/20260623_redcaps_buddy/redcaps_buddy.py`, update the `subreddit_lift` signature's docstring to document the `top_k=None` behavior explicitly (even if the slicing already works, the contract should be written down):

```python
def subreddit_lift(data: Data, e: np.ndarray, top_k: int = 15):
    """
    Same-subreddit enrichment over a graph's edges.

    overall_lift = P(edge endpoints share subreddit) / expected-if-random, where
    'expected' uses the subreddit marginal of edge ENDPOINTS (controls for the
    degree/subreddit imbalance — heavily-connected subreddits don't inflate it).

    Also returns per-subreddit within-subreddit lift for the most enriched ones.
    top_k=None returns every subreddit passing the exp_s > 5 reliability filter, sorted
    by lift descending, instead of truncating to the top 15 (used by Experiment 9's
    subreddit-signal-correlates analysis, docs/superpowers/specs/
    2026-08-04-buddy-publication-plan-design.md).
    """
```
If the slicing did need a real code change (e.g. `order[:top_k]` didn't behave as expected for `None`), replace that line with an explicit branch:
```python
    per_sub_idx = order if top_k is None else order[:top_k]
    per_sub = [(data.sub_names[i], float(lift_s[i]), int(deg_s[i]))
               for i in per_sub_idx if exp_s[i] > 5]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python src/test/20260824_redcaps_subreddit_correlates/test_subreddit_lift_all.py
```
Expected: `ALL TESTS PASSED`.

- [ ] **Step 5: Log the change (only if Step 3's code edit was needed)**

If Step 3 required an actual code change, append to `.claude/20260824_log.md` (create it if the Experiment 8 plan hasn't run yet in this session):

```markdown
# /src/test/20260623_redcaps_buddy/redcaps_buddy.py

## `subreddit_lift`: documented/enabled `top_k=None` (all qualifying subreddits)

**Before:** always truncated to the top `top_k` (default 15) most-enriched subreddits.

**After:** `top_k=None` returns every subreddit passing the existing `exp_s > 5`
reliability filter, sorted by lift descending. Default behavior (`top_k=15`) unchanged.

**Why:** Experiment 9 (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
§4) needs the full per-subreddit lift distribution, not just the top 15, to correlate
against subreddit properties. See `docs/superpowers/plans/
2026-08-24-redcaps-subreddit-signal-correlates.md` Task 1.
```

If Step 2 showed `top_k=None` already worked with zero code changes, skip this step — a docstring-only clarification doesn't need a change-log entry per CLAUDE.md's rule (behavior didn't change).

- [ ] **Step 6: Commit**

```bash
git add src/test/20260623_redcaps_buddy/redcaps_buddy.py src/test/20260824_redcaps_subreddit_correlates/test_subreddit_lift_all.py
git add .claude/20260824_log.md 2>/dev/null || true
git commit -m "feat: support top_k=None in subreddit_lift for full per-subreddit breakdown (Experiment 9)"
```

---

### Task 2: Correlation analysis script

**Files:**
- Create: `src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py`
- Test: covered by the script's own `--selftest` (Step 1 below), matching this codebase's convention for analysis scripts with no pytest infra.

**Interfaces:**
- Consumes: `redcaps_buddy.load_data()`, `redcaps_buddy.subreddit_lift(..., top_k=None)` (Task 1), and the RedCaps buddy graph edge array `E` (already built and cached by the existing RedCaps buddy pipeline — see Step 3 for exactly where this plan gets it).
- Produces: a printed per-subreddit table, three Pearson correlations (lift vs. size / diversity / homogeneity), and a scatter figure — consumed directly by Task 3's report.

- [ ] **Step 1: Write the failing selftest**

Create `src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py` with only the imports and `_selftest()`:

```python
"""
Experiment 9 (spec docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md S4):
what predicts buddy-signal strength across RedCaps subreddits? Correlates per-subreddit
buddy-edge lift (redcaps_buddy.subreddit_lift, full breakdown) against three properties:
sample count, caption diversity (1 - mean pairwise CLIP-text cosine similarity within the
subreddit), and visual homogeneity (mean pairwise CLIP-image cosine similarity within the
subreddit) — both computed via a closed-form identity (no O(n^2) pairwise loop).

Usage
-----
  python analyze_subreddit_correlates.py            # full run against cached RedCaps data
  python analyze_subreddit_correlates.py --selftest # offline arithmetic check

Requires: numpy, matplotlib (Agg backend). Run from anywhere; sys.path is fixed up below.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "20260623_redcaps_buddy")))

MIN_SUBREDDIT_SIZE = 20  # below this, pairwise-similarity estimates are too noisy to trust


def _selftest():
    # Orthogonal unit vectors -> mean pairwise cosine similarity = 0.
    orth = np.eye(5, dtype=np.float32)
    assert abs(mean_pairwise_cosine_sim(orth) - 0.0) < 1e-6, mean_pairwise_cosine_sim(orth)

    # Identical unit vectors -> mean pairwise cosine similarity = 1.
    same = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (6, 1))
    assert abs(mean_pairwise_cosine_sim(same) - 1.0) < 1e-6, mean_pairwise_cosine_sim(same)

    # Single row -> undefined (nan), not a crash.
    assert np.isnan(mean_pairwise_cosine_sim(np.array([[1.0, 0.0]], dtype=np.float32)))

    # Cross-check the closed form against a brute-force O(n^2) loop on random unit vectors.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    fast = mean_pairwise_cosine_sim(X)
    pairs = [(i, j) for i in range(12) for j in range(i + 1, 12)]
    brute = np.mean([X[i] @ X[j] for i, j in pairs])
    assert abs(fast - brute) < 1e-4, (fast, brute)

    print("SELFTEST OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py --selftest
```
Expected: `NameError: name 'mean_pairwise_cosine_sim' is not defined`.

- [ ] **Step 3: Implement `mean_pairwise_cosine_sim` and the full analysis**

Add the following above `_selftest()` (after the `MIN_SUBREDDIT_SIZE` constant):

```python
def mean_pairwise_cosine_sim(X: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity over unit-norm row vectors, via the closed-form
    identity ||sum(X)||^2 = n + 2*sum_{i<j}(x_i . x_j) (exact for unit-norm rows) —
    O(n*d) instead of O(n^2*d). Returns nan for n < 2.
    """
    n = X.shape[0]
    if n < 2:
        return float("nan")
    S = X.sum(axis=0)
    total = float(S @ S) - n  # = 2 * sum_{i<j} x_i . x_j
    return total / (n * (n - 1))


def subreddit_properties(data, sub_id_filter=None):
    """
    Per-subreddit (size, caption_diversity, visual_homogeneity), keyed by subreddit name.
    caption_diversity = 1 - mean_pairwise_cosine_sim(txt rows); visual_homogeneity =
    mean_pairwise_cosine_sim(img rows). Skips subreddits with < MIN_SUBREDDIT_SIZE samples.
    """
    props = {}
    n_sub = len(data.sub_names)
    for s in range(n_sub):
        idx = np.where(data.sub_id == s)[0]
        if len(idx) < MIN_SUBREDDIT_SIZE:
            continue
        diversity = 1.0 - mean_pairwise_cosine_sim(data.txt[idx])
        homogeneity = mean_pairwise_cosine_sim(data.img[idx])
        props[data.sub_names[s]] = {
            "size": len(idx), "caption_diversity": diversity, "visual_homogeneity": homogeneity,
        }
    return props


def correlate(lift_by_sub: dict, props_by_sub: dict):
    """Pearson r between per-subreddit lift and each of the three properties, over
    subreddits present in both dicts. Returns {property_name: (r, n)}."""
    names = [n for n in lift_by_sub if n in props_by_sub]
    lifts = np.array([lift_by_sub[n] for n in names])
    out = {}
    for prop in ("size", "caption_diversity", "visual_homogeneity"):
        vals = np.array([props_by_sub[n][prop] for n in names])
        mask = ~np.isnan(lifts) & ~np.isnan(vals)
        if mask.sum() < 3:
            out[prop] = (float("nan"), int(mask.sum()))
            continue
        r = float(np.corrcoef(lifts[mask], vals[mask])[0, 1])
        out[prop] = (r, int(mask.sum()))
    return out


def run():
    import redcaps_buddy as rb

    print("Loading RedCaps data + buddy graph...")
    data = rb.load_data()
    # Reuses the same buddy-graph construction path as the rest of the RedCaps buddy
    # analysis (K=30, alpha=0.5 — the project-wide default, configs/train/default.yaml).
    from src.conditional_buddy.buddy_graph import mutual_knn, union_graph
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()
    A_img = mutual_knn(data.img, 30, device, 1024, use_half=use_half)
    A_txt = mutual_knn(data.txt, 30, device, 1024, use_half=use_half)
    E = union_graph(A_img, A_txt)
    e = np.stack(E.nonzero(), axis=1)
    e = e[e[:, 0] < e[:, 1]]  # upper triangle only, matches subreddit_lift's expected input

    print(f"Computing full per-subreddit lift ({len(data.sub_names)} subreddits)...")
    lift_result = rb.subreddit_lift(data, e, top_k=None)
    lift_by_sub = {name: lift for name, lift, _deg in lift_result["top_enriched"]}
    print(f"  overall_lift={lift_result['overall_lift']:.2f}x over "
          f"{len(lift_result['top_enriched'])} qualifying subreddits")

    print("Computing per-subreddit properties (size, caption diversity, visual homogeneity)...")
    props_by_sub = subreddit_properties(data)

    corr = correlate(lift_by_sub, props_by_sub)
    print("\nCorrelation(subreddit lift, property):")
    for prop, (r, n) in corr.items():
        print(f"  {prop:>20}: r={r:+.3f}  (n={n} subreddits)")

    _write_figure(lift_by_sub, props_by_sub)
    return lift_by_sub, props_by_sub, corr


def _write_figure(lift_by_sub, props_by_sub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in lift_by_sub if n in props_by_sub]
    lifts = [lift_by_sub[n] for n in names]
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
        "docs", "reports", "assets", "redcaps_subreddit_correlates")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, prop in zip(axes, ("size", "caption_diversity", "visual_homogeneity")):
        vals = [props_by_sub[n][prop] for n in names]
        ax.scatter(vals, lifts, s=14, alpha=0.6)
        ax.set_xlabel(prop)
        ax.set_ylabel("subreddit lift")
        if prop == "size":
            ax.set_xscale("log")
    fig.tight_layout()
    path = os.path.join(out_dir, "lift_vs_properties.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        run()
```

- [ ] **Step 4: Run the selftest to verify it passes**

```bash
python src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py --selftest
```
Expected: `SELFTEST OK`.

- [ ] **Step 5: Commit**

```bash
git add src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py
git commit -m "feat: add RedCaps subreddit signal-strength correlation analysis (Experiment 9)"
```

---

### Task 3: Run the full analysis and write the report

**Files:**
- Create: `docs/reports/2026-08-24_redcaps_subreddit_signal_correlates.md` (adjust date to when this actually runs)
- Modify: `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` (extend C1's evidence note in §2)

**Interfaces:**
- Consumes: Task 2's `run()` output.
- Produces: a citable deepening of C1 for the paper's signal-validation section.

- [ ] **Step 1: Run the full analysis**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py
```
Capture: the overall lift and subreddit count, the three correlation coefficients with sample sizes, and confirm `docs/reports/assets/redcaps_subreddit_correlates/lift_vs_properties.png` was written.

- [ ] **Step 2: Write the results report**

Create `docs/reports/2026-08-24_redcaps_subreddit_signal_correlates.md` with: method (full per-subreddit lift, the three properties and how each is computed, the closed-form identity used and why), the full results table (or top/bottom-N if the full table is large — state the total count either way), the three correlation coefficients with interpretation, the scatter figure, caveats (e.g. `MIN_SUBREDDIT_SIZE` threshold's effect on which subreddits are included), and reproduction command.

- [ ] **Step 3: Update the spec**

In `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §2, extend C1's row (or add a short note beneath the claims table, following the same pattern C6 used to extend C5) citing the new report and stating whether any property meaningfully predicts subreddit-level signal strength.

- [ ] **Step 4: Commit**

```bash
git add docs/reports/2026-08-24_redcaps_subreddit_signal_correlates.md docs/reports/assets/redcaps_subreddit_correlates/ docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md
git commit -m "results: RedCaps subreddit signal-strength correlates (Experiment 9)"
```

---

## Self-Review

- **Spec coverage:** Task 1 gives the full per-subreddit breakdown Experiment 9 needs beyond the existing top-15. Task 2 covers all three named properties (size, caption diversity, visual homogeneity) and the correlation itself. Task 3 covers execution, the report deliverable, and the spec update back into C1.
- **Placeholder scan:** every code block is complete and runnable; Task 1's Step 2/Step 3 branch (conditional on whether `order[:None]` already worked) is an explicit, resolvable investigation step, not a vague placeholder — either outcome has a fully-specified next action.
- **Type/interface consistency:** `mean_pairwise_cosine_sim(X: np.ndarray) -> float`, `subreddit_properties(data, ...) -> dict`, and `correlate(lift_by_sub, props_by_sub) -> dict[str, (float, int)]` (Task 2) are defined once and used identically in `run()`/`_selftest()`. `subreddit_lift(..., top_k=None)`'s return shape (`top_enriched: list[(name, lift, degree)]`) matches what Task 2's `run()` consumes.
- **Scope check:** this plan is fully independent of Experiment 8 and every other experiment in the spec — no training, no shared compute, can run any time. It deepens C1 only; it explicitly does not touch the C5/C6 downstream-use claims (stated in Global Constraints to prevent scope drift during report-writing).
