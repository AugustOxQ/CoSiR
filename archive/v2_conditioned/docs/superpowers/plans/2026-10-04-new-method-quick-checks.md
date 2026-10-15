# Quick checks D0, N1, N2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the scorers and the decision-rule functions of the three CPU development checks (D0 label-probe aspect reading, N1 centered agreement rule, N2 find-then-select cascade) and a runner that scores them on the seed-42 development episodes and applies the committed decision table.

**Architecture:** One new pure-numpy module `src/eval/aspect_quick_checks.py` holds the scorers (each returns the project's `scores[cond][dir] -> (E, 13)` dict, so `per_anchor` scores it) and the decision functions of `DECISION_RULE.md`. A runner in `src/test/20261108_new_method_quick_checks/run_checks.py` loads the seed-42 episodes through E3's `EvalContext`, fits the label probes, encodes the factor codes, reuses the A′ nested cross-fit (`crossfit_nested`) unchanged, and writes results plus `decision.json`. No existing source file is edited.

**Tech Stack:** Python 3.10 (conda env `CoSiR`), numpy, torch (CPU, only through existing helpers), scikit-learn `LogisticRegression`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md` (approved 2026-10-04) and the committed decision rule `src/test/20261108_new_method_quick_checks/DECISION_RULE.md` (commit 7e50f18). Read both before starting a task.

## Global Constraints

- Python: `/root/miniconda3/envs/CoSiR/bin/python`. Never install anything into the env.
- CPU only. Prefix every command with `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. The local GPU is reserved for other projects.
- Tests: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q` from `/project/CoSiR`.
- Implementers run only the **smoke** mode of the runner (`--smoke`). The real seed-42 run is launched by the controller after review; do not run `run_checks.py` without `--smoke`.
- Git: branch `main`; stage files by explicit path only (never `git add -A` / `git add .`); `bin/`, `docs/paper/` and `docs/*.DS_Store` stay untracked. Commit messages end with the two lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1`.
- Do not edit any existing file under `src/`. The folder `src/test/20261108_new_method_quick_checks/` already holds `DECISION_RULE.md` (committed; never edit it) and a `.gitignore` that ignores `*.npz`, `*.json`, `*.log`, `results/`.
- Metric conventions are the project's: `src.eval.aspect_metrics.per_anchor` (a tie or a non-finite row is a miss), condition gain = R@1 − other-aspect rate, either rate = R@1 + other-aspect rate, painting-clustered bootstrap (5,000 resamples, seed 42).
- Episode layout (`src.eval.aspect_episodes.AspectEpisodes`): `candidates[:, 0]` = p_a, `candidates[:, 1]` = p_b, then 11 negatives; `ep.condition("a")` returns `(support_img, support_txt, contrast_img, contrast_txt, 0)` with supports = the aspect-a pairs; condition `"b"` swaps the roles.

## Review Focus

1. **Probe class alignment.** The image and caption probes of one aspect must have identical `classes_` (genre has −1 rows that are dropped); a mismatch would make cross-modal dot products meaningless while every number still looks plausible. Pinned by an assertion in Task 3's `fit_probes`, exercised by the smoke run.
2. **Episode / aspect misalignment.** Told needs the per-episode conditioned aspect from `ctx.pair_index`; a wrong mapping silently gives a low Told gain and a wrong D0 reading. Pinned by Task 1's `test_told_uses_the_per_episode_aspect` and by Task 3's reproduction assertions (cosine equals E1's arrays; A3, C0, SE nested and control equal the A′ pilot's stored arrays in the real run).
3. **Silent misses from NaN.** A NaN code or posterior on an episode row turns that ranking into a miss and depresses every number. Pinned by Task 1's non-finite tests and by Task 3's `assert_finite_scores` on every score dict.
4. **Flat rows counted as misses.** A rule whose weights are all zero on an episode gives a constant score row, which `per_anchor` counts as a miss in term-only scoring. Not a bug, but it must be visible: Task 3 reports `flat_share` for every term.
5. **Re-running after reading numbers.** The real run must refuse to overwrite its results. Pinned by the overwrite guard in Task 3's `main`.

---

### Task 1: Scorers for N1, diagonal KISSME on codes, the N2 cascade and D0

**Files:**
- Create: `src/eval/aspect_quick_checks.py`
- Test: `src/test/test_aspect_quick_checks.py`

**Interfaces:**
- Consumes: `src.eval.aspect_metrics.CONDITIONS` (`("a", "b")`), `DIRECTIONS` (`("i2t", "t2i")`), `first_place`, `per_anchor`; `src.eval.aspect_scorers.EvalInputs` (fields `img`, `txt`, `img_codes`, `txt_codes`; codes indexed by global row); `src.eval.aspect_episodes.AspectEpisodes`; `src.model.aspect_rule.agreement_weights` (tests only).
- Produces (all score functions return `{cond: {dir: np.ndarray (E, K)}}`):
  - `ASPECTS = ("emotion", "style", "genre")`
  - `centered_agreement_weights(sup_img, sup_txt, con_img, con_txt) -> np.ndarray (E, F) float32`
  - `centered_term(inputs: EvalInputs, ep, uniform: bool = False) -> dict`
  - `code_scale(img_codes_train, txt_codes_train) -> np.ndarray (F,) float64`
  - `kissme_diag_term(inputs: EvalInputs, ep, scale: np.ndarray) -> dict`
  - `cascade_scores(control: dict, rerank: dict, k: int) -> dict`
  - `both_in_topk(control: dict, k: int) -> np.ndarray (E,)`
  - `probe_dots(post: dict, ep, aspects=ASPECTS) -> {aspect: {dir: (E, K)}}` where `post[aspect][modality]` is an `(N_rows, C_aspect)` posterior array, modality in `("img", "txt")`
  - `aspect_deltas(post: dict, ep, cond: str, aspects=ASPECTS) -> np.ndarray (E, H)`
  - `told_scores(post: dict, ep, aspect_a: np.ndarray, aspect_b: np.ndarray, aspects=ASPECTS) -> dict` (`aspect_a`, `aspect_b` are `(E,)` integer indices into `aspects`)
  - `inferred_weights(post: dict, ep, cond: str, mode: str, aspects=ASPECTS) -> tuple[np.ndarray (E, H), np.ndarray (E,) bool]`
  - `inferred_scores(post: dict, ep, mode: str, aspects=ASPECTS) -> tuple[dict, dict]` (second value: `{cond: {"weights": (E, H), "fallback": (E,) bool}}`)

- [ ] **Step 1: Write the failing tests**

Create `src/test/test_aspect_quick_checks.py`:

```python
import numpy as np
import pytest
import torch

from src.eval.aspect_episodes import AspectEpisodes
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, first_place, per_anchor
from src.eval.aspect_quick_checks import (
    aspect_deltas, both_in_topk, cascade_scores, centered_agreement_weights, centered_term, code_scale,
    inferred_scores, inferred_weights, kissme_diag_term, told_scores,
)
from src.eval.aspect_scorers import EvalInputs
from src.model.aspect_rule import agreement_weights


def _ep(n_ep=1):
    """Hand-built episodes on 30 rows each: anchor 0, candidates 1..13 (p_a = row 1, p_b = row 2), aspect-a pairs
    (images 14..17, captions 18..21), aspect-b pairs (images 22..25, captions 26..29)."""
    rows = np.tile(np.arange(30), (n_ep, 1)) + 30 * np.arange(n_ep)[:, None]
    return AspectEpisodes("a", "b", rows[:, 0], rows[:, 1:14], rows[:, 14:18], rows[:, 18:22], rows[:, 22:26],
                          rows[:, 26:30])


def _inputs(img_codes, txt_codes):
    feats = np.random.default_rng(0).normal(size=(len(img_codes), 4)).astype(np.float32)
    return EvalInputs(feats, feats.copy(), np.asarray(img_codes, np.float32), np.asarray(txt_codes, np.float32))


def _sc(a):
    """The same score matrix under both conditions and both directions."""
    a = np.asarray(a, dtype=np.float64)
    return {c: {d: a.copy() for d in DIRECTIONS} for c in CONDITIONS}


# ---------------------------------------------------------------- N1: centered agreement rule

def test_centered_weights_reward_covariation_not_mean_activity():
    # factor 0: image and caption codes rise together across the 4 support pairs; factor 1: both high and constant
    sup = np.array([[[0.1, 0.9], [0.2, 0.9], [0.3, 0.9], [0.4, 0.9]]])
    zero = np.zeros((1, 4, 2))
    np.testing.assert_allclose(centered_agreement_weights(sup, sup, zero, zero), [[1.0, 0.0]], atol=1e-6)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
    old = agreement_weights(t(sup), t(sup), t(zero), t(zero)).numpy()
    assert old[0, 1] > old[0, 0]          # the uncentered rule prefers the generally active factor


def test_centered_weights_subtract_contrast_covariance():
    v = np.array([0.1, 0.2, 0.3, 0.4])
    sup = np.stack([v, v], axis=-1)[None]                       # both factors covary in the supports
    con = np.stack([v, np.full(4, 0.5)], axis=-1)[None]         # only factor 0 covaries in the contrasts
    np.testing.assert_allclose(centered_agreement_weights(sup, sup, con, con), [[0.0, 1.0]], atol=1e-6)


def test_centered_weights_zero_and_nonfinite_rows():
    flat = np.full((1, 4, 3), 0.5)
    assert (centered_agreement_weights(flat, flat, flat, flat) == 0).all()
    bad = flat.copy()
    bad[0, 2, 1] = np.nan
    assert np.isnan(centered_agreement_weights(bad, flat, flat, flat)).all()


def test_centered_term_centres_the_query_on_its_own_modality_examples():
    img, txt = np.zeros((30, 2)), np.zeros((30, 2))
    v = np.array([0.2, 0.4, 0.6, 0.8])
    img[14:18, 0], txt[18:22, 0] = v, v - 0.5        # condition a's support pairs covary on factor 0
    img[22:26, 0] = 0.5                               # contrast images constant; the 8 example images average 0.5
    img[0, 0] = 0.2                                   # the query image lies below the example images' mean
    txt[1:14, 0] = np.linspace(0.0, 1.0, 13)          # candidate captions; column 0 has the lowest code
    s = centered_term(_inputs(img, txt), _ep())["a"]["i2t"]
    assert s.shape == (1, 13)
    # (q - mu_q) < 0, so the lowest candidate wins; without centring, or centring on the captions (mean 0), the
    # highest candidate (column 12) would win
    assert np.argmax(s[0]) == 0


def test_centered_uniform_term_ignores_the_condition():
    rng = np.random.default_rng(1)
    t = centered_term(_inputs(rng.random((60, 3)), rng.random((60, 3))), _ep(2), uniform=True)
    for d in DIRECTIONS:
        np.testing.assert_array_equal(t["a"][d], t["b"][d])
    assert (per_anchor(t)["gain"] == 0).all()


# ---------------------------------------------------------------- diagonal KISSME on codes

def _kissme_world():
    img, txt = np.zeros((30, 2)), np.zeros((30, 2))
    img[14:18], txt[18:22] = [1.0, 2.0], [1.0, 0.0]    # condition a supports: difference (0, 2)
    img[22:26], txt[26:30] = [2.0, 1.0], [0.0, 1.0]    # condition a contrasts: difference (2, 0)
    img[0] = [1.0, 1.0]
    txt[1] = [1.0, 3.0]                                 # column 0 matches the query on factor 0
    txt[2:14] = [3.0, 1.0]                              # the others match it on factor 1
    return img, txt


def test_kissme_diag_prefers_closeness_where_supports_agree():
    img, txt = _kissme_world()
    t = kissme_diag_term(_inputs(img, txt), _ep(), scale=np.ones(2))
    assert np.argmax(t["a"]["i2t"][0]) == 0             # m = (0.8, -0.8): column 0 scores 3.2, the others -3.2
    assert np.argmax(t["b"]["i2t"][0]) != 0             # roles swap under condition b


def test_kissme_diag_divides_codes_by_the_scale():
    img, txt = _kissme_world()
    s = np.array([2.0, 4.0])
    a = kissme_diag_term(_inputs(img, txt), _ep(), scale=s)["a"]["i2t"]
    b = kissme_diag_term(_inputs(img / s, txt / s), _ep(), scale=np.ones(2))["a"]["i2t"]
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_kissme_diag_is_flat_when_supports_and_contrasts_vary_alike():
    img, txt = np.zeros((30, 2)), np.zeros((30, 2))
    img[14:18], txt[18:22] = [1.0, 2.0], [0.0, 0.0]
    img[22:26], txt[26:30] = [1.0, 2.0], [0.0, 0.0]
    txt[1:14] = np.random.default_rng(2).random((13, 2))
    t = kissme_diag_term(_inputs(img, txt), _ep(), scale=np.ones(2))
    assert np.ptp(t["a"]["i2t"][0]) == 0


def test_code_scale_pools_modalities_and_guards_dead_factors():
    tr = np.array([[0.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(code_scale(tr, tr), [2.0, 1.0])     # std of (0, 4, 0, 4) = 2; dead factor -> 1
    with pytest.raises(ValueError):
        code_scale(np.array([[np.nan, 0.0]]), tr)


# ---------------------------------------------------------------- N2: find-then-select cascade

def test_cascade_reorders_only_the_top_k():
    control = _sc([[0.5, 0.8, 0.9, 0.1, 0.7]])          # control order: 2, 1, 4, 0, 3
    rerank = _sc([[9.0, 1.0, 0.0, 5.0, 2.0]])
    out = cascade_scores(control, rerank, k=3)["a"]["i2t"][0]
    # top 3 by control = {2, 1, 4}, reordered by rerank to 4, 1, 2; columns 0 and 3 stay below in control order,
    # although column 0 has the highest rerank score
    assert list(np.argsort(-out)) == [4, 1, 2, 0, 3]
    assert list(np.argsort(-cascade_scores(control, rerank, k=1)["a"]["i2t"][0])) == [2, 1, 4, 0, 3]


def test_cascade_breaks_rerank_ties_by_control():
    control, rerank = _sc([[0.8, 0.9, 0.1]]), _sc([[0.0, 0.0, 5.0]])
    out = cascade_scores(control, rerank, k=2)["a"]["i2t"]
    # columns 0 and 1 tie on rerank; the control prefers column 1, so column 1 leads (not the lower column index)
    assert list(np.argsort(-out[0])) == [1, 0, 2]
    assert first_place(out, 1)[0] == 1.0                # no tie is created, so the leader is a strict hit


def test_cascade_nonfinite_rows_become_misses():
    control = _sc([[0.9, np.nan, 0.1], [0.9, 0.8, 0.1], [0.9, 0.8, 0.1]])
    rerank = _sc([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [0.0, 1.0, np.nan]])
    out = cascade_scores(control, rerank, k=2)["a"]["i2t"]
    assert np.isnan(out[0]).all() and np.isnan(out[1]).all()
    assert np.isfinite(out[2]).all()                    # a NaN rerank score outside the top k does not matter
    with pytest.raises(ValueError):
        cascade_scores(control, rerank, k=0)
    with pytest.raises(ValueError):
        cascade_scores(control, rerank, k=4)


def test_both_in_topk_counts_rankings_with_both_aspect_candidates():
    s = np.array([[0.9, 0.8, 0.1, 0.0], [0.9, 0.1, 0.8, 0.0]])
    np.testing.assert_array_equal(both_in_topk(_sc(s), 2), [1.0, 0.0])
    np.testing.assert_array_equal(both_in_topk(_sc(s), 3), [1.0, 1.0])


# ---------------------------------------------------------------- D0: told and inferred label-probe scorers

ASP = ("a", "b", "c")


def _d0_world(pairs_agree=True):
    """One episode, aspects (a, b, c) with 5 values, one-hot posteriors (the same probe output for both modalities).
    Anchor: a=0, b=0, c=0. p_a (row 1) shares a only, p_b (row 2) shares b only, negatives share neither."""
    lab = {h: np.full(30, 4) for h in ASP}
    for h in ASP:
        lab[h][0] = 0
    lab["a"][1], lab["b"][2] = 0, 0
    vals = np.array([1, 2, 3, 4])
    lab["a"][14:18], lab["a"][18:22] = vals, (vals if pairs_agree else np.roll(vals, 1))   # aspect-a pairs share a
    lab["b"][14:18], lab["b"][18:22] = 1, 2                                                 # ... and differ on b
    lab["b"][22:26], lab["b"][26:30] = vals, (vals if pairs_agree else np.roll(vals, 1))   # aspect-b pairs share b
    lab["a"][22:26], lab["a"][26:30] = 1, 2                                                 # ... and differ on a
    lab["c"][[*range(14, 18), *range(22, 26)]], lab["c"][[*range(18, 22), *range(26, 30)]] = 1, 2
    post = {}
    for h in ASP:
        p = np.eye(5)[lab[h]]
        post[h] = {"img": p, "txt": p.copy()}
    return post


def test_aspect_deltas_point_at_the_conditioned_aspect():
    post, ep = _d0_world(), _ep()
    np.testing.assert_allclose(aspect_deltas(post, ep, "a", ASP), [[1.0, -1.0, 0.0]])
    np.testing.assert_allclose(aspect_deltas(post, ep, "b", ASP), [[-1.0, 1.0, 0.0]])


def test_told_uses_the_per_episode_aspect():
    post, ep = _d0_world(), _ep()
    told = per_anchor(told_scores(post, ep, np.array([0]), np.array([1]), ASP))
    assert told["r1"][0] == 1.0 and told["gain"][0] == 1.0
    swapped = per_anchor(told_scores(post, ep, np.array([1]), np.array([0]), ASP))
    assert swapped["gain"][0] == -1.0


def test_inferred_hard_and_soft_recover_the_aspect():
    post, ep = _d0_world(), _ep()
    for mode in ("hard", "soft"):
        scores, info = inferred_scores(post, ep, mode, ASP)
        assert per_anchor(scores)["gain"][0] == 1.0
        assert info["a"]["weights"].argmax(axis=1)[0] == 0 and info["b"]["weights"].argmax(axis=1)[0] == 1
        assert not info["a"]["fallback"].any()


def test_soft_falls_back_to_uniform_when_no_aspect_is_shown():
    post, ep = _d0_world(pairs_agree=False), _ep()
    w, fallback = inferred_weights(post, ep, "a", "soft", ASP)
    assert fallback.all()
    np.testing.assert_allclose(w, [[1 / 3, 1 / 3, 1 / 3]])
    with pytest.raises(ValueError):
        inferred_weights(post, ep, "a", "median", ASP)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'src.eval.aspect_quick_checks'`.

- [ ] **Step 3: Write the module**

Create `src/eval/aspect_quick_checks.py`:

```python
"""Quick checks after the A′ repair (spec docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md §4):
N1's centered agreement rule, diagonal KISSME on factor codes, N2's find-then-select cascade and D0's label-probe
told / inferred scorers. Every score function returns scores[cond][dir] -> (E, K), like src.eval.aspect_scorers."""

import numpy as np

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS
from src.eval.aspect_scorers import EvalInputs

ASPECTS = ("emotion", "style", "genre")


def _finite_rows(*arrays) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(np.asarray(a).reshape(len(a), -1)).all(axis=1)
    return ok


# ---------------------------------------------------------------- N1 (spec §4.2)

def centered_agreement_weights(sup_img, sup_txt, con_img, con_txt) -> np.ndarray:
    """(E,S,F) x4 -> (E,F): ReLU(cov_S - cov_C), L1-normalized, where cov_S(l) is the covariance across the S support
    pairs between the image code and the caption code of factor l (divided by S). An all-zero row stays zero; an
    episode with any non-finite code gets an all-NaN row (a miss, never a silent hit)."""
    def cov(x, y):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        return ((x - x.mean(axis=1, keepdims=True)) * (y - y.mean(axis=1, keepdims=True))).mean(axis=1)

    w = np.maximum(cov(sup_img, sup_txt) - cov(con_img, con_txt), 0.0)
    total = w.sum(axis=1, keepdims=True)
    w = np.where(total > 0, w / np.where(total > 0, total, 1.0), 0.0)
    w[~_finite_rows(sup_img, sup_txt, con_img, con_txt)] = np.nan
    return w.astype(np.float32)


def _code_sides(inputs: EvalInputs, ep, d: str):
    """(query codes (E,F), candidate codes (E,K,F), the 8 example codes of the query's modality (E,8,F))."""
    if d == "i2t":
        q, c = inputs.img_codes[ep.anchor], inputs.txt_codes[ep.candidates]
        ex = np.concatenate([inputs.img_codes[ep.pairs_a_img], inputs.img_codes[ep.pairs_b_img]], axis=1)
    else:
        q, c = inputs.txt_codes[ep.anchor], inputs.img_codes[ep.candidates]
        ex = np.concatenate([inputs.txt_codes[ep.pairs_a_txt], inputs.txt_codes[ep.pairs_b_txt]], axis=1)
    return q.astype(np.float64), c.astype(np.float64), ex.astype(np.float64)


def centered_term(inputs: EvalInputs, ep, uniform: bool = False) -> dict:
    """N1: sum_l w_l (q_l - mu^q_l)(c_l - mu^c_l), with w from centered_agreement_weights (or 1/F: the condition
    removed), mu^c the mean over the candidates and mu^q the mean over the 8 example items of the query's modality
    (the same 8 items under both conditions)."""
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        si, st, ci, ct, _ = ep.condition(cond)
        if uniform:
            f = inputs.img_codes.shape[1]
            w = np.full((len(ep.anchor), f), 1.0 / f)
        else:
            w = centered_agreement_weights(inputs.img_codes[si], inputs.txt_codes[st], inputs.img_codes[ci],
                                           inputs.txt_codes[ct]).astype(np.float64)
        for d in DIRECTIONS:
            q, c, ex = _code_sides(inputs, ep, d)
            out[cond][d] = np.einsum("nf,nf,nkf->nk", w, q - ex.mean(axis=1),
                                     c - c.mean(axis=1, keepdims=True)).astype(np.float32)
    return out


# ---------------------------------------------------------------- diagonal KISSME on codes (comparator)

def code_scale(img_codes_train, txt_codes_train) -> np.ndarray:
    """(F,) per-factor std of the pooled image and caption codes of training rows; a dead factor (std 0) gets 1."""
    pooled = np.concatenate([np.asarray(img_codes_train), np.asarray(txt_codes_train)]).astype(np.float64)
    if not np.isfinite(pooled).all():
        raise ValueError("training codes must be finite")
    std = pooled.std(axis=0)
    return np.where(std > 0, std, 1.0)


def kissme_diag_term(inputs: EvalInputs, ep, scale: np.ndarray) -> dict:
    """Diagonal KISSME (Köstinger et al. 2012) on codes divided by ``scale``: per factor, v = mean over the 4 pairs of
    the squared image-minus-caption difference, m_l = 1/(v_S,l + 1) - 1/(v_C,l + 1) (ridge 1, as E1's KISSME on
    unit-variance coordinates), score = -sum_l m_l (q_l - c_l)^2."""
    s = np.asarray(scale, dtype=np.float64)
    ic, tc = inputs.img_codes / s, inputs.txt_codes / s
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        si, st, ci, ct, _ = ep.condition(cond)
        v_s = ((ic[si] - tc[st]) ** 2).mean(axis=1)
        v_c = ((ic[ci] - tc[ct]) ** 2).mean(axis=1)
        m = 1.0 / (v_s + 1.0) - 1.0 / (v_c + 1.0)
        for d in DIRECTIONS:
            q, c = (ic[ep.anchor], tc[ep.candidates]) if d == "i2t" else (tc[ep.anchor], ic[ep.candidates])
            out[cond][d] = (-np.einsum("nf,nkf->nk", m, (q[:, None, :] - c) ** 2)).astype(np.float32)
    return out


# ---------------------------------------------------------------- N2 (spec §4.3)

def _topk(s: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first (exact ties: the lower column first)."""
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


def cascade_scores(control: dict, rerank: dict, k: int) -> dict:
    """Keep the order of ``control`` outside its top k; reorder the top k by ``rerank`` (ties broken by the control
    score) and keep them above every other candidate. A row with a non-finite control score, or a non-finite rerank
    score inside its top k, becomes all NaN (a miss)."""
    out = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        for d in DIRECTIONS:
            s = np.asarray(control[cond][d], dtype=np.float64)
            r = np.asarray(rerank[cond][d], dtype=np.float64)
            if not 1 <= k <= s.shape[1]:
                raise ValueError(f"k must be in 1..{s.shape[1]}, got {k}")
            rows = np.arange(len(s))[:, None]
            top = _topk(np.where(np.isfinite(s), s, -np.inf), k)
            r_top, s_top = r[rows, top], s[rows, top]
            order = np.lexsort((-s_top, -r_top), axis=-1)          # primary: rerank desc; secondary: control desc
            ranked = np.take_along_axis(top, order, axis=1)
            o = s.copy()
            base = np.max(np.where(np.isfinite(s), s, -np.inf), axis=1)
            o[rows, ranked] = base[:, None] + 1.0 + (k - np.arange(k))[None, :]
            o[~(np.isfinite(s).all(axis=1) & np.isfinite(r_top).all(axis=1))] = np.nan
            out[cond][d] = o.astype(np.float32)
    return out


def both_in_topk(control: dict, k: int) -> np.ndarray:
    """Per anchor, the share of its four rankings (2 conditions x 2 directions) whose top k under ``control`` holds
    both p_a (column 0) and p_b (column 1); a ranking with a non-finite score counts as 0."""
    vals = []
    for cond in CONDITIONS:
        for d in DIRECTIONS:
            s = np.asarray(control[cond][d], dtype=np.float64)
            top = _topk(np.where(np.isfinite(s), s, -np.inf), k)
            hit = (top == 0).any(axis=1) & (top == 1).any(axis=1) & np.isfinite(s).all(axis=1)
            vals.append(hit.astype(np.float64))
    return np.mean(vals, axis=0)


# ---------------------------------------------------------------- D0 (spec §4.1; diagnostic only)

def probe_dots(post: dict, ep, aspects=ASPECTS) -> dict:
    """{aspect: {dir: (E,K) p_h(query) . p_h(candidate)}}, each item scored with its own modality's posterior."""
    return {h: {"i2t": np.einsum("nc,nkc->nk", post[h]["img"][ep.anchor], post[h]["txt"][ep.candidates]),
                "t2i": np.einsum("nc,nkc->nk", post[h]["txt"][ep.anchor], post[h]["img"][ep.candidates])}
            for h in aspects}


def aspect_deltas(post: dict, ep, cond: str, aspects=ASPECTS) -> np.ndarray:
    """(E,H): Delta_h = S_h - C_h, the mean within-pair agreement p_h(image) . p_h(caption) over the 4 support pairs of
    condition ``cond`` minus the same over its 4 contrast pairs."""
    si, st, ci, ct, _ = ep.condition(cond)
    cols = []
    for h in aspects:
        pi, pt = post[h]["img"], post[h]["txt"]
        cols.append(np.einsum("nsc,nsc->ns", pi[si], pt[st]).mean(axis=1)
                    - np.einsum("nsc,nsc->ns", pi[ci], pt[ct]).mean(axis=1))
    return np.stack(cols, axis=1)


def _stacked_dots(post, ep, aspects):
    dots = probe_dots(post, ep, aspects)
    return {d: np.stack([dots[h][d] for h in aspects], axis=1) for d in DIRECTIONS}      # (E, H, K)


def told_scores(post: dict, ep, aspect_a, aspect_b, aspects=ASPECTS) -> dict:
    """D0 Told: the conditioned aspect's probe dot product. ``aspect_a`` / ``aspect_b`` are (E,) indices into
    ``aspects``: the aspect that condition a (resp. b) is about, per episode."""
    stack = _stacked_dots(post, ep, aspects)
    out = {}
    for cond, idx in (("a", np.asarray(aspect_a)), ("b", np.asarray(aspect_b))):
        out[cond] = {d: stack[d][np.arange(len(idx)), idx] for d in DIRECTIONS}
    return out


def inferred_weights(post: dict, ep, cond: str, mode: str, aspects=ASPECTS) -> tuple:
    """(E,H) aspect weights from Delta and the (E,) soft-fallback flag. 'hard': one-hot argmax (ties to the first
    aspect). 'soft': max(Delta, 0) normalised to sum 1, uniform 1/H where every Delta <= 0 (flag set)."""
    delta = aspect_deltas(post, ep, cond, aspects)
    if mode == "hard":
        w = np.zeros_like(delta)
        w[np.arange(len(delta)), delta.argmax(axis=1)] = 1.0
        return w, np.zeros(len(delta), dtype=bool)
    if mode == "soft":
        pos = np.maximum(delta, 0.0)
        total = pos.sum(axis=1, keepdims=True)
        fallback = total[:, 0] <= 0
        w = np.where(fallback[:, None], 1.0 / delta.shape[1], pos / np.where(total > 0, total, 1.0))
        return w, fallback
    raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")


def inferred_scores(post: dict, ep, mode: str, aspects=ASPECTS) -> tuple:
    """D0 Inferred: the Told scores of each aspect weighted by inferred_weights; also returns, per condition, the
    weights and the soft-fallback flags."""
    stack = _stacked_dots(post, ep, aspects)
    out, info = {}, {}
    for cond in CONDITIONS:
        w, fallback = inferred_weights(post, ep, cond, mode, aspects)
        out[cond] = {d: np.einsum("nh,nhk->nk", w, stack[d]) for d in DIRECTIONS}
        info[cond] = {"weights": w, "fallback": fallback}
    return out, info
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q`
Expected: all tests pass (17 passed).

- [ ] **Step 5: Check that each rule's test fails when the rule is removed**

For each line below, make the change temporarily, run the named test, confirm it FAILS, then revert (`git diff src/eval/aspect_quick_checks.py` must be empty afterwards):

| Temporary change in `aspect_quick_checks.py` | Test that must fail |
|---|---|
| In `centered_agreement_weights.cov`, drop both mean subtractions (`return (x * y).mean(axis=1)`) | `test_centered_weights_reward_covariation_not_mean_activity` |
| In `centered_agreement_weights`, use `cov(sup_img, sup_txt)` without `- cov(con_img, con_txt)` | `test_centered_weights_subtract_contrast_covariance` |
| In `centered_term`, replace `q - ex.mean(axis=1)` by `q` | `test_centered_term_centres_the_query_on_its_own_modality_examples` |
| In `_code_sides`, for `"i2t"` build `ex` from `txt_codes[ep.pairs_a_txt]` and `txt_codes[ep.pairs_b_txt]` | `test_centered_term_centres_the_query_on_its_own_modality_examples` |
| In `kissme_diag_term`, replace `ic, tc = inputs.img_codes / s, inputs.txt_codes / s` by `ic, tc = inputs.img_codes, inputs.txt_codes` | `test_kissme_diag_divides_codes_by_the_scale` |
| In `cascade_scores`, set `top = _topk(..., s.shape[1])` (rerank all candidates) | `test_cascade_reorders_only_the_top_k` |
| In `cascade_scores`, use `order = np.lexsort((top, -r_top), axis=-1)` (rerank ties broken by column index instead of the control) | `test_cascade_breaks_rerank_ties_by_control` |
| In `told_scores`, use `idx = np.asarray(aspect_a)` for both conditions | `test_told_uses_the_per_episode_aspect` |
| In `inferred_weights` soft branch, drop the fallback (`w = pos / np.where(total > 0, total, 1.0)`) | `test_soft_falls_back_to_uniform_when_no_aspect_is_shown` |

Record the outcome of each line (fails as expected / did not fail) in the task report. If a line did not fail, strengthen that test until it does, and say so.

- [ ] **Step 6: Commit**

```bash
git add src/eval/aspect_quick_checks.py src/test/test_aspect_quick_checks.py
git commit -m "feat(v2): quick-check scorers: N1 centered agreement rule, diagonal KISSME on codes, N2 cascade, D0 told/inferred

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1"
```

---

### Task 2: Decision-rule functions of DECISION_RULE.md

**Files:**
- Modify: `src/eval/aspect_quick_checks.py` (append; created in Task 1 of this plan, so no change log is needed)
- Test: `src/test/test_aspect_quick_checks.py` (append)

**Interfaces:**
- Consumes: result dicts shaped like `src.eval.aspect_metrics.compare` / `summarize` entries: `{"point": float, "ci95": [lo, hi], ...}` in percentage points.
- Produces:
  - `D0_FRACTION = 0.5`, `N2_RARE_SHARE = 10.0`
  - `CONFIG_ORDER = ("N1-nested-A3", "N1-nested-C0", "N1-nested-SE", "N2-2-agree", "N2-3-agree", "N2-5-agree", "N2-2-N1", "N2-3-N1", "N2-5-N1")`
  - `d0_reading(told_gain: dict, gains: dict) -> dict` with keys `variant` (`"hard"`/`"soft"`), `gain`, `told_gain`, `threshold`, `reading` (`"close"`/`"far"`/`"unreadable"`)
  - `config_passes(r1_vs_control: dict, gain_vs_control: dict) -> bool`
  - `n1_stop(gain_minus_current: float, either_n1: float, either_cos: float) -> dict` with keys `stop` (bool), `reasons` (list of str)
  - `n2_reading(share_both: float, gain_vs_control: dict) -> dict` with keys `rare`, `no_gain` (bools)
  - `decision_row(configs: list, d0: str) -> dict` with keys `row` (1, 2, 3 or None), `config` (str or None), `next` (str)

- [ ] **Step 1: Write the failing tests**

Append to `src/test/test_aspect_quick_checks.py`:

```python
# ---------------------------------------------------------------- decision rule (DECISION_RULE.md)

from src.eval.aspect_quick_checks import (  # noqa: E402
    CONFIG_ORDER, config_passes, d0_reading, decision_row, n1_stop, n2_reading,
)


def _r(point, lo, hi):
    return {"point": point, "ci95": [lo, hi]}


def test_d0_reading_threshold_is_half_of_told_gain():
    told = _r(10.0, 9.0, 11.0)
    assert d0_reading(told, {"hard": 5.0, "soft": 4.0})["reading"] == "close"       # exactly half counts as close
    assert d0_reading(told, {"hard": 4.99, "soft": 4.0})["reading"] == "far"
    r = d0_reading(told, {"hard": 3.0, "soft": 6.0})
    assert r["variant"] == "soft" and r["gain"] == 6.0 and r["threshold"] == 5.0 and r["reading"] == "close"
    assert d0_reading(told, {"hard": 4.0, "soft": 4.0})["variant"] == "hard"        # ties go to hard


def test_d0_reading_is_unreadable_when_told_has_no_reliable_gain():
    assert d0_reading(_r(1.0, -0.2, 2.0), {"hard": 0.9, "soft": 0.8})["reading"] == "unreadable"
    assert d0_reading(_r(1.0, 0.0, 2.0), {"hard": 0.9, "soft": 0.8})["reading"] == "unreadable"


def test_config_passes_needs_both_lower_bounds_above_zero():
    assert config_passes(_r(0.5, 0.1, 0.9), _r(0.4, 0.05, 0.8))
    assert not config_passes(_r(0.5, 0.1, 0.9), _r(0.4, 0.0, 0.8))
    assert not config_passes(_r(0.5, -0.1, 0.9), _r(0.4, 0.05, 0.8))


def test_n1_stop_rules():
    assert n1_stop(0.3, 26.0, 25.92) == {"stop": False, "reasons": []}
    assert n1_stop(0.0, 26.0, 25.92)["stop"]                     # gain no more than the current rule
    assert n1_stop(0.3, 25.0, 25.92)["stop"]                     # either rate below cosine's
    assert len(n1_stop(-0.1, 20.0, 25.92)["reasons"]) == 2


def test_n2_reading():
    assert n2_reading(9.99, _r(0.2, 0.01, 0.4)) == {"rare": True, "no_gain": False}
    assert n2_reading(10.0, _r(0.2, 0.0, 0.4)) == {"rare": False, "no_gain": True}


def _cfg(name, passes, m_r=0.0, m_g=0.0):
    return {"name": name, "passes": passes, "m_r": m_r, "m_g": m_g}


def test_decision_row_order_and_pick():
    configs = [_cfg("N1-nested-A3", True, 0.3, 0.2), _cfg("N1-nested-C0", True, 0.5, 0.2),
               _cfg("N2-2-agree", True, 0.1, 0.9)]
    d = decision_row(configs, "far")                              # row 1 wins over any D0 reading
    assert d["row"] == 1 and d["config"] == "N1-nested-A3"        # min margins 0.2, 0.2, 0.1: tie -> earlier
    none = [_cfg("N1-nested-A3", False), _cfg("N2-2-agree", False)]
    assert decision_row(none, "close")["row"] == 2
    assert decision_row(none, "far")["row"] == 3
    assert decision_row(none, "unreadable")["row"] is None
    with pytest.raises(ValueError):
        decision_row(none, "maybe")
    with pytest.raises(ValueError):
        decision_row([_cfg("N2-3-agree", False), _cfg("N1-nested-A3", False)], "far")   # out of CONFIG_ORDER
    with pytest.raises(ValueError):
        decision_row([_cfg("N3-nested-A3", False)], "far")                               # unknown name
    assert CONFIG_ORDER[0] == "N1-nested-A3" and len(CONFIG_ORDER) == 9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q`
Expected: collection error, `ImportError: cannot import name 'CONFIG_ORDER'`.

- [ ] **Step 3: Append the decision functions**

Append to `src/eval/aspect_quick_checks.py`:

```python
# ---------------------------------------------------------------- decision rule
# src/test/20261108_new_method_quick_checks/DECISION_RULE.md, committed before any check ran.

D0_FRACTION = 0.5          # §2: "close to Told" = Inferred keeps at least half of Told's condition gain
N2_RARE_SHARE = 10.0       # §5: both aspect candidates "rarely" in the top k = a share below 10%
CONFIG_ORDER = ("N1-nested-A3", "N1-nested-C0", "N1-nested-SE", "N2-2-agree", "N2-3-agree", "N2-5-agree",
                "N2-2-N1", "N2-3-N1", "N2-5-N1")
NEXT = {1: "fix the passing configuration and test it on fresh episode seeds 45, 47, 48 (GO rule, DECISION_RULE.md §6)",
        2: "build N6 (cross-modal heads on the label-free k-means partitions, aspect picked by D0's decision variant), "
           "run the same checks, then the same test",
        3: "stop method work; move to branch 3 with D0, N1 and N2 reported as analysis results"}


def d0_reading(told_gain: dict, gains: dict) -> dict:
    """§2. ``told_gain``: Told's pooled condition gain {'point', 'ci95'} (points); ``gains``: {'hard': point,
    'soft': point}. The decision variant has the larger gain (ties to 'hard'). 'close' if its gain >= 0.5 x Told's,
    'far' otherwise, 'unreadable' if Told's gain has a lower bound <= 0."""
    variant = "hard" if gains["hard"] >= gains["soft"] else "soft"
    threshold = D0_FRACTION * told_gain["point"]
    if told_gain["ci95"][0] <= 0:
        reading = "unreadable"
    else:
        reading = "close" if gains[variant] >= threshold else "far"
    return {"variant": variant, "gain": float(gains[variant]), "told_gain": float(told_gain["point"]),
            "threshold": float(threshold), "reading": reading}


def config_passes(r1_vs_control: dict, gain_vs_control: dict) -> bool:
    """§4: both paired differences against the configuration's own condition-free control have lower bounds > 0."""
    return bool(r1_vs_control["ci95"][0] > 0 and gain_vs_control["ci95"][0] > 0)


def n1_stop(gain_minus_current: float, either_n1: float, either_cos: float) -> dict:
    """§5: N1 stops on a checkpoint if its term-only gain is not above the current rule's (paired point) or its
    term-only either rate is below cosine's (points)."""
    reasons = []
    if gain_minus_current <= 0:
        reasons.append("term-only gain not above the current rule")
    if either_n1 < either_cos:
        reasons.append("term-only either rate below cosine")
    return {"stop": bool(reasons), "reasons": reasons}


def n2_reading(share_both: float, gain_vs_control: dict) -> dict:
    """§5 readings at one k: both aspect candidates rarely in the top k; the reordering adds no gain."""
    return {"rare": bool(share_both < N2_RARE_SHARE), "no_gain": bool(gain_vs_control["ci95"][0] <= 0)}


def decision_row(configs: list, d0: str) -> dict:
    """§4, first matching row wins. ``configs``: dicts with 'name', 'passes', 'm_r', 'm_g' (points), in
    CONFIG_ORDER (configurations that were not computed are left out). Several passing: the largest
    min(m_r, m_g), ties to the earlier one."""
    names = [c["name"] for c in configs]
    if any(n not in CONFIG_ORDER for n in names) or names != sorted(names, key=CONFIG_ORDER.index):
        raise ValueError(f"configs must be named from CONFIG_ORDER and kept in its order, got {names}")
    if d0 not in ("close", "far", "unreadable"):
        raise ValueError(f"unknown D0 reading {d0!r}")
    passing = [c for c in configs if c["passes"]]
    if passing:
        best = max(passing, key=lambda c: min(c["m_r"], c["m_g"]))      # max keeps the first of equal maxima
        return {"row": 1, "config": best["name"], "next": NEXT[1]}
    if d0 == "unreadable":
        return {"row": None, "config": None, "next": "D0 cannot be read (Told's gain is not reliably positive); "
                                                      "report to the user without applying a row"}
    row = 2 if d0 == "close" else 3
    return {"row": row, "config": None, "next": NEXT[row]}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_quick_checks.py -q`
Expected: all tests pass (23 passed).

- [ ] **Step 5: Commit**

```bash
git add src/eval/aspect_quick_checks.py src/test/test_aspect_quick_checks.py
git commit -m "feat(v2): decision-rule functions for the quick checks (D0 threshold, pass rule, stop rules, table)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1"
```

---

### Task 3: Runner on the seed-42 episodes (smoke run only)

**Files:**
- Create: `src/test/20261108_new_method_quick_checks/run_checks.py`

**Interfaces:**
- Consumes: everything Task 1 and Task 2 produce; `run_gonogo` as `rg` (`rg.EvalContext(seed, smoke)` with `.data`, `.groups`, `.selection`, `.in_sel`, `.img`, `.txt`, `.masked(arr)`, `.encode(ckpt) -> (img_codes, txt_codes)`, `.pooled`, `.n`, `.pair_index`, `.anchor_group`, `.parity`, `.cos`, `.shas`, `.summary(pa)`, `.per_pair(pa)`; `rg.PAIRS`, `rg.POOLED_ORDER`, `rg.E1`, `rg.HERE`, `rg.folders`, `rg.checkpoint_path`, `rg.sha_file`, `rg.sha_array`, `rg.assert_finite_tree`); `src.eval.aspect_nested.crossfit_nested(cos, t_u, t_a, parity) -> (nested, control, picks)`; `src.eval.aspect_scorers.agreement_term`; `src.train.train_factors.encode_rows`, `load_factor_checkpoint`; `src.data.artelingo_splits.artelingo_splits`, `artelingo_aspect_labels`.
- Produces: `results/checks_seed42.json`, `results/per_anchor_checks_seed42.npz`, `results/checks_seed42.txt`, `results/decision.json` (smoke: under `results/smoke/`).

- [ ] **Step 1: Write the runner**

Create `src/test/20261108_new_method_quick_checks/run_checks.py`:

```python
"""Quick checks D0, N1 and N2 on the seed-42 development episodes. The rules are DECISION_RULE.md in this folder
(committed in 7e50f18 before any scorer existed); this script applies them mechanically. CPU only.

Real run:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
           python src/test/20261108_new_method_quick_checks/run_checks.py         -> results/ (refuses to overwrite)
Smoke run: ... run_checks.py --smoke    -> results/smoke/ (E1's smoke episodes; A1's smoke checkpoint stands in for
           A3, L3 and LT; probes fitted on 3,000 rows)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src/test/20261101_aspect_factor_gonogo"))
import run_gonogo as rg  # noqa: E402  (puts the repo root on sys.path)

from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import METRICS, cluster_bootstrap, compare, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (ASPECTS, CONFIG_ORDER, both_in_topk, cascade_scores,  # noqa: E402
                                          centered_term, code_scale, config_passes, d0_reading, decision_row,
                                          inferred_scores, kissme_diag_term, n1_stop, n2_reading, told_scores)
from src.eval.aspect_scorers import EvalInputs, agreement_term  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

SEED = 42
KS = (2, 3, 5)
PROBE_ROWS, SMOKE_PROBE_ROWS, PROBE_SEED = 60_000, 3_000, 0
DECIDING = ("A3", "C0", "SE")                 # may pass DECISION_RULE.md §4
DIAGNOSTIC = ("L3", "LT")                     # label-trained: readings only
NESTED = ("agree", "N1", "kissme")            # conditioned terms inside the nested score
A_PRIME = ROOT / "src/test/20261105_method_repair_diagnostics"
A3_SHA = "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2"
CODE_SHA = {"C0": "71559d058314a45b863d0f47500bd06ee3ff56623142806c02841099e3d7c43c",
            "SE": "845d6cd330f84adfdacd0ab98db40d54289302e9e39f827248f1df441e2a78e1"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rel(p):
    return str(Path(p).relative_to(ROOT))


def point_ci(values, clusters):
    """Pooled mean (pp) and painting-clustered 95% interval of one per-anchor array."""
    r = cluster_bootstrap(values, clusters)
    return {"point": 100 * r["point"], "ci95": [100 * c for c in r["ci95"]]}


def assert_finite_scores(scores, what):
    for c in scores:
        for d in scores[c]:
            bad = int((~np.isfinite(np.asarray(scores[c][d]))).any(axis=1).sum())
            if bad:
                raise AssertionError(f"{what}: {bad} episodes with a non-finite score ({c}, {d})")


def flat_share(scores):
    """Share (pp) of rankings whose score row is constant (every candidate tied, which per_anchor counts as a miss)."""
    return 100 * float(np.mean([np.ptp(np.asarray(scores[c][d], np.float64), axis=1) == 0
                                for c in scores for d in scores[c]]))


class Report:
    """Summaries of per-anchor arrays against the comparators shared by every scorer of one episode set."""

    def __init__(self, ctx, cos_pa, rca_pa):
        self.ctx, self.cl, self.cos, self.rca = ctx, ctx.anchor_group, cos_pa, rca_pa

    def describe(self, pa):
        return {"summary": self.ctx.summary(pa), "either": point_ci(pa["r1"] + pa["other"], self.cl)}

    def paired(self, pa, pb):
        return {m: compare(pa, pb, self.cl, m) for m in ("r1", "gain")}

    def full(self, pa, control=None):
        out = {**self.describe(pa), "vs_cosine": self.paired(pa, self.cos), "vs_rca": self.paired(pa, self.rca)}
        if control is not None:
            out["vs_control"] = self.paired(pa, control)
        return out


def fit_probes(ctx, labels, scorer_train, n_rows):
    """D0's representation (spec §4.1): one logistic regression per aspect and modality on unit-normalised CLIP
    features of n_rows scorer-train rows with their labels (rows labelled -1 left out); posteriors on selection rows,
    NaN elsewhere."""
    tr = np.random.default_rng(PROBE_SEED).choice(scorer_train, n_rows, replace=False)
    sel = ctx.selection
    feats = {"img": ctx.data.img_features, "txt": ctx.data.txt_features}
    post, prov = {}, {"draw_rows_sha256": rg.sha_array(np.sort(tr)), "n_draw": int(n_rows), "seed": PROBE_SEED}
    for h in ASPECTS:
        lab = np.asarray(labels[h])
        fit = tr[lab[tr] >= 0]
        clfs = {m: LogisticRegression(C=1.0, max_iter=300).fit(unit(F[fit]), lab[fit]) for m, F in feats.items()}
        if not np.array_equal(clfs["img"].classes_, clfs["txt"].classes_):
            raise AssertionError(f"{h}: image and caption probes have different classes")
        labelled = sel[lab[sel] >= 0]
        post[h] = {}
        for m, F in feats.items():
            full = np.full((len(ctx.groups), len(clfs[m].classes_)), np.nan, dtype=np.float32)
            full[sel] = clfs[m].predict_proba(unit(F[sel]))
            if not (np.isfinite(full[sel]).all() and np.isnan(full[~ctx.in_sel]).all()):
                raise AssertionError(f"{h}/{m}: posteriors must be finite on selection rows and NaN elsewhere")
            post[h][m] = full
        prov[h] = {"n_fit": int(len(fit)), "classes": clfs["img"].classes_.tolist(),
                   "selection_accuracy": {m: 100 * float(clfs[m].score(unit(F[labelled]), lab[labelled]))
                                          for m, F in feats.items()}}
        log(f"probe {h}: fit on {len(fit)} rows")
    return post, prov


def model_inputs(ctx, name, scorer_train, smoke):
    """(EvalInputs with selection-masked codes, per-factor scale from scorer-train codes, provenance)."""
    if name in CODE_SHA:
        path = rg.E1 / f"codes_{name}.npz"
        sha = rg.sha_file(path)
        if sha != CODE_SHA[name]:
            raise AssertionError(f"{path}: SHA-256 {sha} differs from DECISION_RULE.md")
        z = np.load(path)
        ic, tc = ctx.masked(z["img"]), ctx.masked(z["txt"])
        train_img, train_txt = z["img"][scorer_train], z["txt"][scorer_train]
        prov = {"codes": rel(path), "sha256": sha}
    else:
        if smoke:
            ckpt = rg.checkpoint_path(name, SEED, True)            # A1's smoke checkpoint stands in
        elif name == "A3":
            ckpt = rg.checkpoint_path("A3", SEED, False)
        else:
            ckpt = A_PRIME / "checkpoints" / f"{name}_seed{SEED}.pt"
        sha = rg.sha_file(ckpt)
        if not smoke and name == "A3":
            pick = json.loads((rg.HERE / "results" / "picked.json").read_text())
            if not (pick["run"] == "A3" and sha == pick["checkpoint_sha256"] == A3_SHA):
                raise AssertionError("A3 checkpoint is not E3's pick")
        elif not smoke:
            listed = json.loads((A_PRIME / "results" / "label_checkpoints.json").read_text())
            if listed.get(f"{name}_seed{SEED}") != sha:
                raise AssertionError(f"{name}: SHA-256 not in label_checkpoints.json")
        ic, tc = ctx.encode(ckpt)
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        train_img, train_txt = encode_rows(model, ctx.data.img_features, ctx.data.txt_features, rows=scorer_train,
                                           device="cpu")
        prov = {"checkpoint": rel(ckpt), "sha256": sha}
    return EvalInputs(ctx.img, ctx.txt, ic, tc), code_scale(train_img, train_txt), prov


def summary_text(r):
    def c(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:6.2f}, {x['ci95'][1]:6.2f}]"

    p = r["provenance"]
    lines = [f"Quick checks, seed {SEED}, n={p['n_episodes']}{' (SMOKE)' if p['smoke'] else ''}",
             f"cosine R@1 {c(r['cosine']['summary']['r1'])} either {r['cosine']['either']['point']:.2f}; "
             f"RCA R@1 {c(r['rca']['summary']['r1'])} gain {c(r['rca']['summary']['gain'])}", "D0 (label probes):"]
    for v in ("told", "hard", "soft"):
        e = r["d0"][v]
        lines.append(f"  {v:5s} R@1 {c(e['summary']['r1'])} gain {c(e['summary']['gain'])} "
                     f"either {e['either']['point']:6.2f}")
    d = r["d0"]["reading"]
    lines.append(f"  hard-pick accuracy {r['d0']['hard_pick_accuracy']['pooled']:.1f}%; soft fallback "
                 f"{r['d0']['soft_fallback_share']:.2f}%; variant {d['variant']} gain {d['gain']:.2f} vs threshold "
                 f"{d['threshold']:.2f} -> {d['reading']}")
    lines.append("Factor codes: term only (R@1, gain, either, flat %), control, nested vs control:")
    for name, e in r["factors"].items():
        for t, x in e["term_only"].items():
            lines.append(f"  {name:3s} term {t:10s} R@1 {x['summary']['r1']['point']:6.2f} gain "
                         f"{x['summary']['gain']['point']:6.2f} either {x['either']['point']:6.2f} "
                         f"flat {x['flat_share']:5.2f}")
        lines.append(f"  {name:3s} N1 stop: {e['n1_stop']}")
        lines.append(f"  {name:3s} control R@1 {e['control']['summary']['r1']['point']:6.2f} "
                     f"either {e['control']['either']['point']:6.2f}")
        for t, x in e["nested"].items():
            lines.append(f"  {name:3s} nested {t:7s} R@1 {c(x['summary']['r1'])} gain {c(x['summary']['gain'])} "
                         f"vs control R@1 {c(x['vs_control']['r1'])} picks {x['picks']}")
    lines.append("N2 on A3:")
    for k in KS:
        lines.append(f"  k={k} both-in-top-k {c(r['n2'][f'both_in_top{k}'])}")
        for rr in r["n2"]["rerankers"]:
            x = r["n2"][f"{k}-{rr}"]
            lines.append(f"    {rr:5s} R@1 {c(x['summary']['r1'])} gain {c(x['summary']['gain'])} either "
                         f"{x['either']['point']:6.2f} vs control R@1 {c(x['vs_control']['r1'])} {x['reading']}")
    lines.append("Configurations (pass = R@1 and gain lower bounds against the own control above 0):")
    for x in r["configs"]:
        lines.append(f"  {x['name']:14s} m_R {x['m_r']:6.2f} {x['r1_ci95']}  m_g {x['m_g']:6.2f} {x['gain_ci95']}  "
                     f"pass {x['passes']}")
    dec = r["decision"]
    lines.append(f"DECISION: row {dec['row']} ({dec['config']}): {dec['next']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    res = HERE / "results" / ("smoke" if smoke else "")
    names = ("checks_seed42.json", "per_anchor_checks_seed42.npz", "checks_seed42.txt", "decision.json")
    if not smoke and any((res / n).exists() for n in names):
        raise SystemExit(f"results exist in {res}; the checks run once. Refusing to overwrite.")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ctx = rg.EvalContext(SEED, smoke)
    ep, cl = ctx.pooled, ctx.anchor_group
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    labels = artelingo_aspect_labels(ctx.data)
    t9 = np.load(rg.folders(smoke)["e1"] / "per_anchor_seed42.npz")
    cos_pa = per_anchor(ctx.cos)
    if not all(np.array_equal(cos_pa[m], t9[f"cosine__{m}"]) for m in METRICS):
        raise AssertionError("episodes misaligned with E1's seed-42 arrays")
    rca_pa = {m: t9[f"rca__{m}"].astype(np.float64) for m in METRICS}
    rep = Report(ctx, cos_pa, rca_pa)
    arrays = {"anchor_group": cl, "pair_index": ctx.pair_index}

    def keep(prefix, pa):
        for m in METRICS:
            arrays[f"{prefix}__{m}"] = np.asarray(pa[m])

    keep("cosine", cos_pa)
    rule = HERE / "DECISION_RULE.md"
    result = {"provenance": {"decision_rule": {"path": rel(rule), "sha256": rg.sha_file(rule), "commit": "7e50f18"},
                             "script_sha256": rg.sha_file(Path(__file__)),
                             "module_sha256": rg.sha_file(ROOT / "src/eval/aspect_quick_checks.py"),
                             "episodes_sha256": ctx.shas, "n_episodes": int(ctx.n),
                             "n_clusters": int(len(np.unique(cl))), "sklearn": sklearn.__version__, "smoke": smoke},
              "cosine": rep.describe(cos_pa), "rca": rep.describe(rca_pa)}
    either_cos = result["cosine"]["either"]["point"]
    log(f"context ready: {ctx.n} episodes")

    # ---------------------------------------------------------------- D0
    post, probe_prov = fit_probes(ctx, labels, scorer_train, SMOKE_PROBE_ROWS if smoke else PROBE_ROWS)
    aspect_a = np.array([ASPECTS.index(rg.PAIRS[i][0]) for i in ctx.pair_index])
    aspect_b = np.array([ASPECTS.index(rg.PAIRS[i][1]) for i in ctx.pair_index])
    hard, hard_info = inferred_scores(post, ep, "hard")
    soft, soft_info = inferred_scores(post, ep, "soft")
    d0_pa = {}
    for name, s in (("told", told_scores(post, ep, aspect_a, aspect_b)), ("hard", hard), ("soft", soft)):
        assert_finite_scores(s, f"D0 {name}")
        d0_pa[name] = per_anchor(s)
        keep(f"d0_{name}", d0_pa[name])
    d0 = {name: {**rep.full(pa), "per_pair": ctx.per_pair(pa)} for name, pa in d0_pa.items()}
    truth = {"a": aspect_a, "b": aspect_b}
    picked = {c: hard_info[c]["weights"].argmax(axis=1) for c in truth}
    acc = {c: 100 * float(np.mean(picked[c] == truth[c])) for c in truth}
    acc["pooled"] = 0.5 * (acc["a"] + acc["b"])
    d0["hard_pick_accuracy"] = acc
    d0["hard_pick_accuracy_per_pair"] = {
        rg.POOLED_ORDER[i]: 100 * float(np.mean([np.mean(picked[c][ctx.pair_index == i] == truth[c][ctx.pair_index == i])
                                                 for c in truth])) for i in range(len(rg.PAIRS))}
    d0["soft_fallback_share"] = 100 * float(np.mean([soft_info[c]["fallback"] for c in truth]))
    reading = d0_reading(d0["told"]["summary"]["gain"], {v: d0[v]["summary"]["gain"]["point"] for v in ("hard", "soft")})
    var = d0_pa[reading["variant"]]
    d0["gain_minus_half_told"] = point_ci(var["gain"] - 0.5 * d0_pa["told"]["gain"], cl)
    d0["r1_analogue"] = point_ci((var["r1"] - cos_pa["r1"]) - 0.5 * (d0_pa["told"]["r1"] - cos_pa["r1"]), cl)
    d0["reading"], d0["probes"] = reading, probe_prov
    result["d0"] = d0
    log(f"D0 done: {reading['reading']}")

    # ---------------------------------------------------------------- N1, the current rule, diagonal KISSME
    factors, n1_stops, controls = {}, {}, {}
    pilot = None if smoke else np.load(A_PRIME / "results" / "per_anchor_pilot_seed42.npz")
    terms_a3 = None
    for name in (*DECIDING, *DIAGNOSTIC):
        inp, scale, prov = model_inputs(ctx, name, scorer_train, smoke)
        terms = {"agree": agreement_term(inp, ep), "N1": centered_term(inp, ep),
                 "N1_uniform": centered_term(inp, ep, uniform=True), "kissme": kissme_diag_term(inp, ep, scale),
                 "uniform": agreement_term(inp, ep, uniform=True)}
        entry = {"provenance": prov, "term_only": {}, "nested": {}}
        tpa = {}
        for t, s in terms.items():
            assert_finite_scores(s, f"{name} {t}")
            tpa[t] = per_anchor(s)
            keep(f"{name}__term_{t}", tpa[t])
            entry["term_only"][t] = {**rep.describe(tpa[t]), "flat_share": flat_share(s)}
        entry["n1_minus_current_gain"] = compare(tpa["N1"], tpa["agree"], cl, "gain")
        entry["n1_stop"] = n1_stop(entry["n1_minus_current_gain"]["point"], entry["term_only"]["N1"]["either"]["point"],
                                   either_cos)
        n1_stops[name] = entry["n1_stop"]
        for t in NESTED:
            nested, control, picks = crossfit_nested(ctx.cos, terms["uniform"], terms[t], ctx.parity)
            pn, pc = per_anchor(nested), per_anchor(control)
            if name not in controls:
                controls[name] = (control, pc)
                entry["control"] = rep.describe(pc)
                keep(f"{name}__control", pc)
            elif not all(np.array_equal(pc[m], controls[name][1][m]) for m in METRICS):
                raise AssertionError(f"{name}: the nested uniform control must not depend on the conditioned term")
            keep(f"{name}__nested_{t}", pn)
            entry["nested"][t] = {**rep.full(pn, control=pc), "picks": picks}
            if pilot is not None and t == "agree" and name in DECIDING:
                for kind, pa in (("nested", pn), ("control", pc)):
                    if not all(np.array_equal(pa[m], pilot[f"{name}__{kind}__{m}"]) for m in METRICS):
                        raise AssertionError(f"{name} {kind}: differs from the A′ pilot's stored arrays")
        if name == "A3":
            terms_a3 = terms
        factors[name] = entry
        log(f"{name} done")
    result["factors"] = factors

    # ---------------------------------------------------------------- N2 on A3
    control_a3, pc_a3 = controls["A3"]
    rerankers = {"agree": terms_a3["agree"]}
    if not n1_stops["A3"]["stop"]:
        rerankers["N1"] = terms_a3["N1"]
    one = per_anchor(cascade_scores(control_a3, terms_a3["agree"], 1))
    if not all(np.array_equal(one[m], pc_a3[m]) for m in METRICS):
        raise AssertionError("the k = 1 cascade must equal its control")
    n2 = {"rerankers": list(rerankers)}
    for k in KS:
        both = both_in_topk(control_a3, k)
        arrays[f"n2_both_top{k}"] = both
        n2[f"both_in_top{k}"] = point_ci(both, cl)
        for rr, term in rerankers.items():
            casc = cascade_scores(control_a3, term, k)
            assert_finite_scores(casc, f"N2 k={k} {rr}")
            pa = per_anchor(casc)
            keep(f"n2_{k}_{rr}", pa)
            entry = rep.full(pa, control=pc_a3)
            entry["reading"] = n2_reading(n2[f"both_in_top{k}"]["point"], entry["vs_control"]["gain"])
            n2[f"{k}-{rr}"] = entry
    result["n2"] = n2
    log("N2 done")

    # ---------------------------------------------------------------- decision (DECISION_RULE.md §4)
    configs = []
    for cname in CONFIG_ORDER:
        kind, a, b = cname.split("-")
        if kind == "N1":
            vc = factors[b]["nested"]["N1"]["vs_control"]
        elif f"{a}-{b}" in n2:
            vc = n2[f"{a}-{b}"]["vs_control"]
        else:
            continue                                   # N2 with N1's term: not computed because N1 stopped on A3
        configs.append({"name": cname, "passes": config_passes(vc["r1"], vc["gain"]), "m_r": vc["r1"]["point"],
                        "m_g": vc["gain"]["point"], "r1_ci95": vc["r1"]["ci95"], "gain_ci95": vc["gain"]["ci95"]})
    decision = decision_row(configs, reading["reading"])
    result["configs"], result["decision"] = configs, decision
    rg.assert_finite_tree(result)
    record = {"d0": reading, "configs": configs, "n1_stop": n1_stops,
              "n2_readings": {k: v["reading"] for k, v in n2.items() if isinstance(v, dict) and "reading" in v},
              "decision": decision, "smoke": smoke}
    (res / "checks_seed42.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(res / "per_anchor_checks_seed42.npz", **arrays)
    (res / "decision.json").write_text(json.dumps(record, indent=1))
    text = summary_text(result)
    (res / "checks_seed42.txt").write_text(text + "\n")
    print(text)
    log(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke run**

Run from `/project/CoSiR`:
`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261108_new_method_quick_checks/run_checks.py --smoke 2>&1 | tee src/test/20261108_new_method_quick_checks/results/smoke_run.log`
(create `results/` first with `mkdir -p src/test/20261108_new_method_quick_checks/results`).
Expected: the run ends with `DECISION: row ...` and `runtime ...s`; `results/smoke/` holds the four output files. `ConvergenceWarning` lines from scikit-learn are acceptable (the spike's probe used the same settings). Any `AssertionError` is a defect to fix in the runner, not to silence. Smoke numbers mean nothing (A1's 50-step smoke checkpoint, 3,000 probe rows); do not tune anything on them.

- [ ] **Step 3: Check the smoke outputs**

Run:
```bash
cd /project/CoSiR && /root/miniconda3/envs/CoSiR/bin/python -c "
import json, numpy as np
r = json.load(open('src/test/20261108_new_method_quick_checks/results/smoke/checks_seed42.json'))
d = json.load(open('src/test/20261108_new_method_quick_checks/results/smoke/decision.json'))
z = np.load('src/test/20261108_new_method_quick_checks/results/smoke/per_anchor_checks_seed42.npz')
assert set(r['factors']) == {'A3', 'C0', 'SE', 'L3', 'LT'}
assert [c['name'] for c in d['configs']][:6] == ['N1-nested-A3', 'N1-nested-C0', 'N1-nested-SE', 'N2-2-agree', 'N2-3-agree', 'N2-5-agree']
assert len(z['anchor_group']) == r['provenance']['n_episodes']
assert all(np.isfinite(z[k]).all() for k in z.files)
print('smoke outputs OK', d['decision'])
"
```
Expected: `smoke outputs OK {...}`.

- [ ] **Step 4: Commit**

```bash
git add src/test/20261108_new_method_quick_checks/run_checks.py
git commit -m "feat(v2): runner for the quick checks D0, N1, N2 on the seed-42 episodes (smoke-tested)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01VzPGkSgZ3UrCEi5JYqTDw1"
```

---

## After the tasks (controller, not an implementer task)

1. Run the real checks once on CPU (check `uptime` and `free -g` first), in the background:
   `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261108_new_method_quick_checks/run_checks.py > src/test/20261108_new_method_quick_checks/results/run.log 2>&1`
2. Read `results/decision.json` and `results/checks_seed42.txt`; write `src/test/20261108_new_method_quick_checks/20261108_new_method_quick_checks_log.md`; record seed 42's new use in `docs/superpowers/episode_seed_ledger.md`.
3. Final whole-branch review on the most capable model, re-deriving the load-bearing numbers (D0 Told and Inferred gains, the threshold reading, every configuration's paired margins against its control, the both-in-top-k shares) from the stored per-anchor arrays with independent code; one fix wave; a scoped re-review.
4. Write the report `docs/reports/auto/v2/2026-11-08_new_method_quick_checks.md` plus its `reports_sum.md` row and run `scripts/check_reports_sum.py`.
5. Report the decision-table outcome to the user before any further step.
