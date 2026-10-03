# Method-repair diagnostics (H1 nested-score pilot + H3 learnability) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pre-registered diagnostics stage of method A′ (the H1 nested-score pilot on existing checkpoints and the H3 label-trained learnability diagnostic, in parallel) and apply its joint decision table, stopping before any A′ pre-registration or seed-45 scoring.

**Architecture:** A small library module `src/eval/aspect_nested.py` holds the nested score, its uniform control, the spec-§15 cross-fitting and every pre-registered reading, all unit-tested. One config flag (`aspect_tau_fixed`) is added to factor training. The experiment folder `src/test/20261105_method_repair_diagnostics/` holds the pre-registration (committed first), a bank builder (LAB from scorer-train labels, MK from label-free k-means at matched granularity), a training runner, two CPU scorers (H1 pilot, H3) and a decision script. Seed-42 selection episodes are read through E3's `run_gonogo.EvalContext`.

**Tech Stack:** Python 3.11 (`/root/miniconda3/envs/CoSiR/bin/python`), numpy, torch, scikit-learn (MiniBatchKMeans), pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §15 (revision 3) and §6, §10; the ARS review that set the rules: `src/test/20261104_ars_repair_order_review/editorial_decision.md` (required changes R1 to R12) and `phase2_methodology.md` (Q1 to Q5); the handoff `docs/superpowers/handoffs/2026-10-03-method-repair-handoff.md`.

## Global Constraints

- **Environment.** Python is always `/root/miniconda3/envs/CoSiR/bin/python`. Tests: `/root/miniconda3/envs/CoSiR/bin/python -m pytest <file> -q` from `/project/CoSiR`. Never install into the CoSiR env.
- **Shared machine.** Before GPU work run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`; wrap every GPU command in `flock -n -o -E 75 /tmp/gpu0.lock <cmd>` (exit 75: another session holds the GPU, stop and report). CPU work sets `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. Never kill a process you did not start.
- **Row scope.** Training and every H3 in-distribution episode read **scorer-train** rows only (183,694 local rows, `artelingo_splits(data).scorer_train` order). Development scoring reads **selection** rows only, through `rg.EvalContext` (NaN outside selection, asserted). Val and held rows are never read.
- **Episode seeds.** Only seed-42 selection episodes (E1's `episodes_seed42.npz`, SHA-256 checked by `EvalContext`) are scored in this plan. Seeds 43 (spent), 44 (MLLM) and 45 (reserved for the single A′ test) are never built, loaded or scored: no script of this plan creates or reads an episode or result file for seeds 43 to 45.
- **Labels.** Evaluation labels (emotion, style, genre) enter training **only** in the LAB bank of the H3 diagnostic. LAB checkpoints (`L3`, `L5`, `LT`, and any seed-43 rerun) are never A′ candidates; their SHA-256s are listed in `results/label_checkpoints.json`. No setting is chosen from LAB results.
- **Fixed numbers** (spec §15, PREREGISTRATION.md): nested grid λ_u ∈ {0, 0.5, 1, 2, 4, 8, 16}, λ_a ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 16}; control over the 30 distinct sums; parity `np.arange(n) % 2`; 5,000 bootstrap resamples, bootstrap seed 42, painting clusters; K_SE = 2.80; bank size 65,536; fresh episodes 4,096 per pair; training 2,000 steps, 32 episodes per step, L = 32, model seed 42 (43 only under the inconclusive rule).
- **Seeds of new randomness:** LAB bank 1042, MK bank 2042, fresh label episodes 3042 + pair index, MK k-means `random_state` 42.
- **Commits.** One commit per task step marked Commit, on main, files staged by explicit path (`bin/` stays untracked; other sessions share main). End each commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01G5pcVF4Bg8shSQjiExASSM`. Do not push.
- **Change log.** Edits to existing source files get an entry in `.claude/20261003_log.md` (a `# <path>` header, before/after snippets, why); it is gitignored but tracked, add with `git add -f`.
- **Experiment folder.** `src/test/20261105_method_repair_diagnostics/` with `.gitignore` copied from `src/test/20261023_aspect_episode_spike/.gitignore`; ends with `20261105_method_repair_diagnostics_log.md`.
- **Reports.** `docs/reports/auto/v2/2026-11-04_ars_repair_order_review.md` and `docs/reports/auto/v2/2026-11-05_method_repair_diagnostics.md`, one row each in `docs/reports/reports_sum.md` (v2 table, after the last row), then `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py` prints OK. Paper-draft style, a real baseline beside every number, figures, no dashes as punctuation.
- **Long jobs** (the four trainings) are launched by the main session with Bash `run_in_background: true`; subagents write and smoke-test scripts only.
- **Pre-registration first.** `PREREGISTRATION.md` and spec §15 are committed (Task 1) before any script of this plan runs on real data, smoke runs included. A later change is a dated addendum committed before the result it affects.

## Review Focus

1. **Non-finite rows through the nested score.** A NaN row in cos, T_u or T_a that is *used* (weight > 0) must make the fused row NaN (a miss, never a silent hit); a NaN row in a term with weight 0 must not. Owner: Task 2 (`test_nested_nonfinite_rows_are_misses_only_when_used`).
2. **Tie order of the pick.** Equal criteria must resolve to the first cell in row-major order (λ_u outer, λ_a inner, ascending) and to the smallest σ for the control; a reordered grid would silently change picks. Owner: Task 2 (`test_crossfit_nested_ties_go_to_first_cell_and_smallest_sigma`).
3. **Cross-fit leakage.** A half's pick must be decided on that half only and applied only to the other half. Owner: Task 2 (`test_crossfit_nested_halves_are_independent`).
4. **Readings at their boundaries.** Margin exactly 0, exactly 2.80·SE, a CI lower bound exactly 0, an unresolved "inconclusive" reaching the joint table. Owner: Task 2 (`test_readings_at_boundaries`, `test_h3_reading_refuses_unresolved_fit`).
5. **Fixed τ really fixed, default unchanged.** `aspect_tau_fixed=True` must leave τ at its step-0 value for every logged step, and the default config must reproduce the existing golden codes. Owner: Task 3 (`test_fixed_tau_stays_at_step0_value`, existing `test_default_config_codes_match_pre_change_golden_values`).

---

### Task 1: Pre-registration, seed ledger and folder scaffold (controller)

**Files:**
- Create: `src/test/20261105_method_repair_diagnostics/PREREGISTRATION.md`
- Create: `src/test/20261105_method_repair_diagnostics/.gitignore` (copy)
- Create: `docs/superpowers/episode_seed_ledger.md`
- Modify (already edited, commit here): `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` (§15, status line)
- Commit also: `docs/superpowers/plans/2026-10-03-method-repair-diagnostics.md` (this plan) and the ARS record `src/test/20261104_ars_repair_order_review/*.md`

- [ ] **Step 1: Copy the .gitignore**

```bash
mkdir -p src/test/20261105_method_repair_diagnostics
cp src/test/20261023_aspect_episode_spike/.gitignore src/test/20261105_method_repair_diagnostics/.gitignore
```

- [ ] **Step 2: Write PREREGISTRATION.md** with exactly these sections (the controller writes the text; it must state every rule below verbatim in substance):
  1. Binding authority: spec §15 (revision 3) at the commit of this task, with the spec file's SHA-256; the ARS record.
  2. Data, rows, seeds (as Global Constraints).
  3. Banks: LAB = `build_episode_bank({"emotion","genre","style"} labels of scorer-train rows, local_groups, arange(n), n_per_pair=21846, seed=1042, min_paintings=30)` truncated to 65,536 as in E2, validated on 1,000 sampled episodes; MK = the same builder on `affect8` (MiniBatchKMeans k=8, random_state 42, n_init 3, batch 4096 on the cached GoEmotions `affect_probs`), `image23` (`kmeans_partition` of CLIP image features, k=23, seed 42), `caption10` (`kmeans_partition` of CLIP caption features, k=10, seed 42), bank seed 2042. MK uses no evaluation label.
  4. Runs: C0 base config (E3's `run_config` base) with L3 = LAB + {λ_aspect 3, λ_swap 1}; L5 = LAB + {λ_aspect 1, λ_swap 1, β 0}; LT = L3 + `aspect_tau_fixed=True`; MK3 = MK + {λ_aspect 3, λ_swap 1}; all 2,000 steps, model seed 42.
  5. Scorers: term-only agreement rule (per-anchor metrics of `agreement_term` itself, equivalent to λ = ∞); the nested score and nested control with spec-§15 cross-fitting (`crossfit_nested`); paired `compare`, painting clusters, 5,000 resamples, seed 42; SE := (upper − lower) / (2 × 1.959964) of the paired 95% interval.
  6. **H1 pilot rule (primary model A3, checkpoint SHA-256 = `picked.json`).** m_R = R@1(nested) − R@1(control), m_g = gain(nested) − gain(control), both paired on the seed-42 pooled episodes. *Not promising* if m_R ≤ 0 or m_g ≤ 0; *promising* if m_R ≥ 2.80·SE_R and m_g ≥ 2.80·SE_g; *inconclusive* otherwise. A1, A2, A4, A5, A6, C0 and SE are descriptive rows. Descriptive also: A3 − C0 under the nested score; nested and control against cosine and RCA; the 56-cell fixed profile of A3; predicted seed-45 power Φ(m/(2·SE) − 1.96) per metric.
  7. **H3 rules.** Fit of a LAB run X: paired term-only gain X − A3 on fresh label episodes (scorer-train rows, seed 3042 + pair index, 4,096 per pair, third aspect controlled, validated): *fits* if the lower bound > 0; *no fit* if the point ≤ 0; *inconclusive* otherwise, then X is retrained once at model seed 43 and decided on the per-anchor mean of its two seeds against A3: *fits* if the lower bound > 0, else *no fit*. H3 *no fit*: none of L3, L5, LT fits. Ceiling (when at least one fits): the fitting LAB run with the largest seed-42 cross-fitted nested gain; *ceiling sufficient* if that gain point ≥ g* = max(5.6·SE_R, 2.8·SE_g) with SE_R, SE_g from A3's pilot; else *ceiling too low*. Matched-k (descriptive unless the H2 path runs): MK3 − A3 term-only gain on seed 42, paired; lower bound > 0 means the H2 grid runs on MK banks, otherwise on AIC. Loss against its constant-score value (3.258 with swap, 2.565 without) and τ are descriptive.
  8. **Joint decision table.** H1 promising (any H3): pre-register A′ = the nested score on A3 (H4 replaces it only if H3's ceiling is sufficient and an H2 model passes its fit gate and beats A3's pilot margins by Oct 9, decided before the A′ pre-registration commit). H1 inconclusive or not promising with ceiling sufficient: run the H2 grid (§9) behind its gate, re-pilot each gate-passer with the H1 rule, pre-register A′ by Oct 9 on the promising model with the largest min(m_R/SE_R, m_g/SE_g), else branch 3. H1 inconclusive or not promising with H3 no fit or ceiling too low: stop, branch 3. Any path: seed 45 is scored once for one A′ by Oct 12; a failure ends the repair.
  9. **H2 grid (pre-registered now, run only if §8 sends the project there):** bank B (MK if the matched-k reading is positive, else AIC), model seed 42, base A3 settings: G1 fixed τ; G2 β 0; G3 fixed τ + β 0; G4 128 episodes per step; G5 6,000 steps; G6 fixed τ + 128 episodes per step; G7 lr 3e-4 + 6,000 steps; G8 λ_aspect 1 + fixed τ + β 0 (plus MK3 as G0 when B = MK). Gate: paired term-only gain over A3 on fresh pseudo-aspect episodes of B's partitions (scorer-train rows, seed 4042 + pair index, 4,096 per pair), lower bound > 0.
  10. Provenance: SHA-256 of every bank, partition file, checkpoint and episode file in the results JSONs; LAB checkpoint hash list.
  11. Disclosures: H1 came from a post-hoc profile that included seed 43; seed 42 has been looked at by E1, E3's pick and post-hoc profile, and now by this stage; seed 45 shares paintings with seeds 42 and 43.

- [ ] **Step 3: Write `docs/superpowers/episode_seed_ledger.md`**

```markdown
# Episode-seed ledger (ArtELingo aspect episodes, selection rows)

Every scoring of an aspect-episode seed on selection rows is one row. Spec §15: seed 45 is scored once, for one
pre-registered A′.

| Seed | Status | Scored by (date, purpose) |
|---|---|---|
| 42 | development and picks | E1 baselines (2026-10-03); E3 grid pick (2026-10-03); E3 post-hoc λ profile (2026-10-03); method-repair diagnostics: H1 pilot and H3 transfer (2026-10-03/04, `src/test/20261105_method_repair_diagnostics/`) |
| 43 | spent | E1 baselines; E3 GO test, K8, ablation (2026-10-03); E3 post-hoc λ profile and bootstrap-seed sensitivity |
| 44 | MLLM probe | Qwen3-VL-2B v1 and v2 (2026-10-03) |
| 45 | reserved: the single A′ GO test | none |
| 46+ | free; any 8B MLLM probe uses 46 or later | none |
```

- [ ] **Step 4: Commit (before any script runs)**

```bash
git add docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md \
  docs/superpowers/plans/2026-10-03-method-repair-diagnostics.md docs/superpowers/episode_seed_ledger.md \
  src/test/20261105_method_repair_diagnostics/PREREGISTRATION.md src/test/20261105_method_repair_diagnostics/.gitignore \
  src/test/20261104_ars_repair_order_review/*.md src/test/20261104_ars_repair_order_review/.gitignore
git commit -m "docs(v2): spec rev 3 (method A′), ARS repair-order review, diagnostics pre-registration and seed ledger"
```

---

### Task 2: `src/eval/aspect_nested.py`: nested score, control, cross-fitting and readings

**Files:**
- Create: `src/eval/aspect_nested.py`
- Test: `src/test/test_aspect_nested.py`

**Interfaces:**
- Consumes: `src.eval.aspect_metrics.{CONDITIONS, DIRECTIONS, per_anchor}`, `src.model.aspect_rule.zscore_rows`.
- Produces:
  - `NESTED_U: tuple[float, ...]`, `NESTED_A: tuple[float, ...]`, `K_SE = 2.80`, `Z975 = 1.959963984540054`
  - `nested_cells() -> list[tuple[float, float]]` (56, row-major), `control_sums() -> list[float]` (30, ascending)
  - `nested_scores(cos: dict, t_u: dict, t_a: dict, lam_u: float, lam_a: float) -> dict` (scores[cond][dir] arrays, float32)
  - `control_scores(cos: dict, t_u: dict, sigma: float) -> dict`
  - `crossfit_nested(cos: dict, t_u: dict, t_a: dict, parity) -> tuple[dict, dict, dict]` returning `(nested, control, picks)` with `picks = {0: {"sigma": float, "cell": [float, float]}, 1: {...}}`
  - `se_from_ci(ci95) -> float`, `margin_reading(m_r, se_r, m_g, se_g, k=K_SE) -> str` in {"promising", "inconclusive", "not_promising"}, `ceiling_threshold(se_r, se_g) -> float`, `predicted_power(margin, se) -> float`, `fit_reading(result: dict) -> str` in {"fits", "inconclusive", "no_fit"}, `h3_reading(fit: dict[str, str], best_nested_gain: float | None, g_star: float) -> str` in {"no_fit", "ceiling_too_low", "ceiling_sufficient"}, `joint_decision(h1: str, h3: str) -> str` in {"preregister_A3_nested", "h2_grid", "branch_3"}

- [ ] **Step 1: Write the failing tests** in `src/test/test_aspect_nested.py`:

```python
import math

import numpy as np
import pytest

from src.eval.aspect_episodes import build_aspect_episodes
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_nested import (
    K_SE, NESTED_A, NESTED_U, ceiling_threshold, control_scores, control_sums, crossfit_nested, fit_reading,
    h3_reading, joint_decision, margin_reading, nested_cells, nested_scores, predicted_power, se_from_ci,
)
from src.eval.aspect_scorers import EvalInputs, agreement_term, cosine_scores, fused_scores


def _world(n_paintings=1200, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_paintings), 2)
    a = rng.integers(0, 6, 2 * n_paintings)
    b = np.repeat(rng.integers(0, 6, n_paintings), 2)
    pattern_a, pattern_b = rng.random((6, 6)) ** 3, rng.random((6, 6)) ** 3
    code = np.concatenate([pattern_a[a], pattern_b[b]], axis=1).astype(np.float32)
    img = code + 0.3 * rng.normal(size=code.shape).astype(np.float32)
    txt = code + 0.3 * rng.normal(size=code.shape).astype(np.float32)
    ep = build_aspect_episodes({"a": a, "b": b}, groups, np.arange(len(groups)), "a", "b", 400, seed=1,
                               min_paintings=5)
    inputs = EvalInputs(img, txt, code, code)
    return cosine_scores(inputs, ep), agreement_term(inputs, ep, uniform=True), agreement_term(inputs, ep)


def test_grids_are_the_preregistered_ones():
    assert NESTED_U == (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    assert NESTED_A == (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    cells = nested_cells()
    assert len(cells) == 56 and cells[0] == (0.0, 0.0) and cells[1] == (0.0, 0.25) and cells[8] == (0.5, 0.0)
    sums = control_sums()
    assert len(sums) == 30 and sums[0] == 0.0 and sums[-1] == 32.0 and sums == sorted(sums)


def test_nested_reduces_to_cosine_and_to_1d_fusion():
    cos, tu, ta = _world()
    for c in CONDITIONS:
        for d in DIRECTIONS:
            base = nested_scores(cos, tu, ta, 0.0, 0.0)[c][d]
            assert np.allclose(base, fused_scores(cos, ta, 0.0)[c][d])
            for lam in (0.25, 2.0, 16.0):
                assert np.allclose(nested_scores(cos, tu, ta, 0.0, lam)[c][d], fused_scores(cos, ta, lam)[c][d],
                                   atol=1e-5)
                assert np.allclose(control_scores(cos, tu, lam)[c][d], fused_scores(cos, tu, lam)[c][d], atol=1e-5)


def test_control_is_condition_blind():
    cos, tu, _ = _world()
    for s in (0.5, 4.0, 32.0):
        assert np.allclose(per_anchor(control_scores(cos, tu, s))["gain"], 0.0)


def test_nested_rejects_bad_weights():
    cos, tu, ta = _world()
    for bad in ((-1.0, 0.0), (0.0, float("inf")), (float("nan"), 1.0)):
        with pytest.raises(ValueError):
            nested_scores(cos, tu, ta, *bad)


def test_nested_nonfinite_rows_are_misses_only_when_used():
    cos, tu, ta = _world()
    ta_bad = {c: {d: ta[c][d].copy() for d in DIRECTIONS} for c in CONDITIONS}
    for c in CONDITIONS:
        for d in DIRECTIONS:
            ta_bad[c][d][3] = np.nan
    used = nested_scores(cos, tu, ta_bad, 1.0, 2.0)
    unused = nested_scores(cos, tu, ta_bad, 1.0, 0.0)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert np.isnan(used[c][d][3]).all() and np.isfinite(used[c][d][4]).all()
            assert np.isfinite(unused[c][d][3]).all()
    assert per_anchor(used)["r1"][3] == 0.0


def test_crossfit_nested_returns_grid_picks_and_finite_scores():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    nested, control, picks = crossfit_nested(cos, tu, ta, np.arange(n) % 2)
    assert set(picks) == {0, 1}
    for half in (0, 1):
        assert picks[half]["sigma"] in control_sums()
        assert tuple(picks[half]["cell"]) in nested_cells()
    for c in CONDITIONS:
        for d in DIRECTIONS:
            assert np.isfinite(nested[c][d]).all() and np.isfinite(control[c][d]).all()
    assert np.allclose(per_anchor(control)["gain"], 0.0)
    assert per_anchor(nested)["gain"].mean() > 0.05          # the world has aspect-block codes


def test_crossfit_nested_halves_are_independent():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    parity = np.arange(n) % 2
    nested, control, picks = crossfit_nested(cos, tu, ta, parity)
    for half in (0, 1):
        apply = parity != half
        cell, sigma = tuple(picks[half]["cell"]), picks[half]["sigma"]
        for c in CONDITIONS:
            for d in DIRECTIONS:
                assert np.allclose(nested[c][d][apply], nested_scores(cos, tu, ta, *cell)[c][d][apply])
                assert np.allclose(control[c][d][apply], control_scores(cos, tu, sigma)[c][d][apply])
    # changing only half 1's rows must not change half 0's pick (picks are decided on the tuning half alone)
    shuffled = {k: {c: {d: v[c][d].copy() for d in DIRECTIONS} for c in CONDITIONS}
                for k, v in (("cos", cos), ("tu", tu), ("ta", ta))}
    rng = np.random.default_rng(5)
    for c in CONDITIONS:
        for d in DIRECTIONS:
            for key in shuffled:
                rows = shuffled[key][c][d][parity == 1]
                shuffled[key][c][d][parity == 1] = rows + rng.normal(size=rows.shape).astype(np.float32)
    _, _, picks2 = crossfit_nested(shuffled["cos"], shuffled["tu"], shuffled["ta"], parity)
    assert picks2[0] == picks[0]


def test_crossfit_nested_ties_go_to_first_cell_and_smallest_sigma():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    flat = {c: {d: np.zeros_like(cos[c][d]) for d in DIRECTIONS} for c in CONDITIONS}  # every cell and sum ties
    _, _, picks = crossfit_nested(flat, flat, flat, np.arange(n) % 2)
    for half in (0, 1):
        assert picks[half]["sigma"] == 0.0 and picks[half]["cell"] == [0.0, 0.0]


def test_crossfit_nested_validates_parity():
    cos, tu, ta = _world()
    n = len(cos["a"]["i2t"])
    for bad in (np.zeros(n, int), np.arange(n) % 3, np.arange(n - 1) % 2):
        with pytest.raises(ValueError):
            crossfit_nested(cos, tu, ta, bad)


def test_readings_at_boundaries():
    assert se_from_ci([-1.959963984540054, 1.959963984540054]) == pytest.approx(1.0)
    assert margin_reading(0.0, 0.1, 5.0, 0.1) == "not_promising"
    assert margin_reading(5.0, 0.1, -0.01, 0.1) == "not_promising"
    assert margin_reading(K_SE * 0.1, 0.1, K_SE * 0.2, 0.2) == "promising"
    assert margin_reading(K_SE * 0.1 - 1e-9, 0.1, 1.0, 0.1) == "inconclusive"
    assert ceiling_threshold(0.168, 0.2) == pytest.approx(max(5.6 * 0.168, 2.8 * 0.2))
    assert predicted_power(2 * 1.959963984540054, 1.0) == pytest.approx(0.5)
    assert fit_reading({"point": 0.5, "ci95": [0.001, 1.0]}) == "fits"
    assert fit_reading({"point": 0.5, "ci95": [0.0, 1.0]}) == "inconclusive"
    assert fit_reading({"point": 0.0, "ci95": [-1.0, 1.0]}) == "no_fit"


def test_h3_reading_refuses_unresolved_fit():
    with pytest.raises(ValueError):
        h3_reading({"L3": "inconclusive", "L5": "no_fit", "LT": "no_fit"}, None, 1.0)
    assert h3_reading({"L3": "no_fit", "L5": "no_fit", "LT": "no_fit"}, None, 1.0) == "no_fit"
    assert h3_reading({"L3": "fits", "L5": "no_fit", "LT": "no_fit"}, 0.99, 1.0) == "ceiling_too_low"
    assert h3_reading({"L3": "fits", "L5": "no_fit", "LT": "no_fit"}, 1.0, 1.0) == "ceiling_sufficient"
    with pytest.raises(ValueError):
        h3_reading({"L3": "fits", "L5": "no_fit", "LT": "no_fit"}, None, 1.0)


def test_joint_decision_table():
    for h3 in ("no_fit", "ceiling_too_low", "ceiling_sufficient"):
        assert joint_decision("promising", h3) == "preregister_A3_nested"
    for h1 in ("inconclusive", "not_promising"):
        assert joint_decision(h1, "ceiling_sufficient") == "h2_grid"
        assert joint_decision(h1, "no_fit") == "branch_3"
        assert joint_decision(h1, "ceiling_too_low") == "branch_3"
    with pytest.raises(ValueError):
        joint_decision("maybe", "no_fit")
    with pytest.raises(ValueError):
        joint_decision("promising", "fits")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_nested.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.aspect_nested'`.

- [ ] **Step 3: Write the implementation** `src/eval/aspect_nested.py`:

```python
"""Method A′ (CVPR plan spec §15): the nested test-time score z(cos) + λ_u·z(T_u) + λ_a·z(T_a), its nested uniform
control z(cos) + σ·z(T_u), parity cross-fitting with the min-margin pick rule, and the pre-registered readings of the
method-repair diagnostics stage (src/test/20261105_method_repair_diagnostics/PREREGISTRATION.md)."""

import math

import numpy as np
import torch

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.model.aspect_rule import zscore_rows

NESTED_U = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NESTED_A = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
K_SE = 2.80
Z975 = 1.959963984540054
H1_READINGS = ("promising", "inconclusive", "not_promising")
H3_READINGS = ("no_fit", "ceiling_too_low", "ceiling_sufficient")


def nested_cells() -> list:
    """The 56 (λ_u, λ_a) cells in row-major order (λ_u outer, λ_a inner, ascending); ties go to the first."""
    return [(u, a) for u in NESTED_U for a in NESTED_A]


def control_sums() -> list:
    """The 30 distinct σ = λ_u + λ_a, ascending; the control's ties go to the smallest."""
    return sorted({u + a for u, a in nested_cells()})


def _zdict(x: dict) -> dict:
    return {c: {d: zscore_rows(torch.as_tensor(np.asarray(x[c][d]), dtype=torch.float32)) for d in DIRECTIONS}
            for c in CONDITIONS}


def _check(lam_u: float, lam_a: float) -> None:
    if not (math.isfinite(lam_u) and math.isfinite(lam_a)) or lam_u < 0 or lam_a < 0:
        raise ValueError(f"nested weights must be finite and >= 0, got ({lam_u}, {lam_a})")


def _combine(zc: dict, zu: dict, za: dict, lam_u: float, lam_a: float) -> dict:
    """Terms with weight 0 are left out, so a non-finite row in an unused term cannot turn a row into a miss."""
    out = {c: {} for c in CONDITIONS}
    for c in CONDITIONS:
        for d in DIRECTIONS:
            s = zc[c][d]
            if lam_u > 0:
                s = s + lam_u * zu[c][d]
            if lam_a > 0:
                s = s + lam_a * za[c][d]
            out[c][d] = s.numpy().astype(np.float32)
    return out


def nested_scores(cos: dict, t_u: dict, t_a: dict, lam_u: float, lam_a: float) -> dict:
    _check(lam_u, lam_a)
    return _combine(_zdict(cos), _zdict(t_u), _zdict(t_a), lam_u, lam_a)


def control_scores(cos: dict, t_u: dict, sigma: float) -> dict:
    return nested_scores(cos, t_u, t_u, sigma, 0.0)


def _means(scores: dict, rows: np.ndarray) -> tuple:
    m = per_anchor({c: {d: scores[c][d][rows] for d in DIRECTIONS} for c in CONDITIONS})
    return float(m["r1"].mean()), float(m["gain"].mean())


def crossfit_nested(cos: dict, t_u: dict, t_a: dict, parity) -> tuple:
    """Spec §15. On each tuning half the control picks σ by R@1; the nested score then picks the cell maximising
    min(R@1 − that control's R@1 on the same half, condition gain). Each half's picks score the other half."""
    n = len(cos["a"]["i2t"])
    parity = np.asarray(parity)
    if parity.shape != (n,) or not np.isin(parity, (0, 1)).all() or not ((parity == 0).any() and (parity == 1).any()):
        raise ValueError(f"parity must be a length-{n} array of 0/1 with both halves non-empty")
    zc, zu, za = _zdict(cos), _zdict(t_u), _zdict(t_a)
    ctrl_cache, nest_cache = {}, {}

    def ctrl(sigma):
        if sigma not in ctrl_cache:
            ctrl_cache[sigma] = _combine(zc, zu, zu, sigma, 0.0)
        return ctrl_cache[sigma]

    def nest(cell):
        if cell not in nest_cache:
            nest_cache[cell] = _combine(zc, zu, za, *cell)
        return nest_cache[cell]

    shape = {c: {d: np.asarray(cos[c][d]).shape for d in DIRECTIONS} for c in CONDITIONS}
    nested = {c: {d: np.empty(shape[c][d], np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    control = {c: {d: np.empty(shape[c][d], np.float32) for d in DIRECTIONS} for c in CONDITIONS}
    picks = {}
    for half in (0, 1):
        tune, apply = parity == half, parity != half
        sigma = max(control_sums(), key=lambda s: _means(ctrl(s), tune)[0])
        r_ctrl = _means(ctrl(sigma), tune)[0]

        def criterion(cell):
            r1, gain = _means(nest(cell), tune)
            return min(r1 - r_ctrl, gain)

        cell = max(nested_cells(), key=criterion)
        picks[half] = {"sigma": float(sigma), "cell": [float(cell[0]), float(cell[1])]}
        for c in CONDITIONS:
            for d in DIRECTIONS:
                nested[c][d][apply] = nest(cell)[c][d][apply]
                control[c][d][apply] = ctrl(sigma)[c][d][apply]
    return nested, control, picks


def se_from_ci(ci95) -> float:
    """SE of a paired difference from its 95% percentile interval (pre-registered approximation)."""
    lo, hi = ci95
    return (hi - lo) / (2 * Z975)


def margin_reading(m_r: float, se_r: float, m_g: float, se_g: float, k: float = K_SE) -> str:
    """H1 pilot: not promising if either margin <= 0; promising if both >= k SE; inconclusive otherwise."""
    if m_r <= 0 or m_g <= 0:
        return "not_promising"
    if m_r >= k * se_r and m_g >= k * se_g:
        return "promising"
    return "inconclusive"


def ceiling_threshold(se_r: float, se_g: float) -> float:
    """g* = max(2·K_SE·SE_R, K_SE·SE_g): the R@1 margin against the control is about gain / 2."""
    return max(2 * K_SE * se_r, K_SE * se_g)


def predicted_power(margin: float, se: float) -> float:
    """Descriptive: P(lower bound > 0) on a fresh draw if the true margin is half the observed one."""
    return 0.5 * (1.0 + math.erf((margin / (2 * se) - Z975) / math.sqrt(2.0)))


def fit_reading(result: dict) -> str:
    """H3 fit from a paired compare(X, A3, ..., 'gain') result: fits / inconclusive / no_fit."""
    if result["ci95"][0] > 0:
        return "fits"
    if result["point"] <= 0:
        return "no_fit"
    return "inconclusive"


def h3_reading(fit: dict, best_nested_gain, g_star: float) -> str:
    if any(v not in ("fits", "no_fit") for v in fit.values()):
        raise ValueError(f"every LAB fit must be resolved to fits/no_fit first: {fit}")
    if not any(v == "fits" for v in fit.values()):
        return "no_fit"
    if best_nested_gain is None:
        raise ValueError("a fitting LAB run needs its seed-42 nested gain")
    return "ceiling_sufficient" if best_nested_gain >= g_star else "ceiling_too_low"


def joint_decision(h1: str, h3: str) -> str:
    if h1 not in H1_READINGS or h3 not in H3_READINGS:
        raise ValueError(f"unknown readings: h1={h1!r}, h3={h3!r}")
    if h1 == "promising":
        return "preregister_A3_nested"
    return "h2_grid" if h3 == "ceiling_sufficient" else "branch_3"
```

- [ ] **Step 4: Run the tests to verify they pass, plus the neighbours**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_nested.py src/test/test_aspect_scorers.py src/test/test_aspect_rule.py src/test/test_aspect_metrics.py -q`
Expected: all PASS. If `test_crossfit_nested_returns_grid_picks_and_finite_scores` fails only on the gain bound, print the gain and report it rather than loosening the bound silently.

- [ ] **Step 5: Commit**

```bash
git add src/eval/aspect_nested.py src/test/test_aspect_nested.py
git commit -m "feat(v2): nested score, its uniform control, min-margin cross-fit and pre-registered readings (spec §15)"
```

---

### Task 3: `aspect_tau_fixed` in factor training

**Files:**
- Modify: `src/train/train_factors.py` (the `FactorTrainingConfig` fields near line 80; the aspect τ initialisation near lines 270 to 279)
- Test: `src/test/test_aspect_loss.py`
- Change log: `.claude/20261003_log.md`

**Interfaces:**
- Produces: `FactorTrainingConfig.aspect_tau_fixed: bool = False`. With `True`, the aspect temperature is set at step 0 exactly as now (std of the step-0 scores) and never updated; with `False`, behaviour is unchanged. `True` with `lambda_aspect == 0` raises `ValueError`.

- [ ] **Step 1: Write the failing tests** (append to `src/test/test_aspect_loss.py`):

```python
def test_fixed_tau_stays_at_step0_value():
    import dataclasses
    img, txt, groups, bank, graph = _tiny()
    cfg = dataclasses.replace(R3_CONFIG, epochs=6, batch_size=64, lambda_aspect=1.0, aspect_episodes_per_step=8,
                              num_factors=8, painting_batches=True, aspect_tau_fixed=True)
    history = {}
    train_factors(img, txt, graph, cfg, device="cpu", group_ids=groups, aspect_bank=bank, history=history,
                  log_every=1)
    taus = history["tau"]
    assert len(taus) == 6 and all(t == taus[0] for t in taus)


def test_learned_tau_moves_by_default():
    import dataclasses
    img, txt, groups, bank, graph = _tiny()
    cfg = dataclasses.replace(R3_CONFIG, epochs=6, batch_size=64, lambda_aspect=1.0, aspect_episodes_per_step=8,
                              num_factors=8, painting_batches=True)
    assert cfg.aspect_tau_fixed is False
    history = {}
    train_factors(img, txt, graph, cfg, device="cpu", group_ids=groups, aspect_bank=bank, history=history,
                  log_every=1)
    assert len(set(history["tau"])) > 1


def test_fixed_tau_needs_the_aspect_loss():
    import dataclasses, pytest
    img, txt, groups, bank, graph = _tiny()
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, dataclasses.replace(R3_CONFIG, epochs=1, aspect_tau_fixed=True),
                      device="cpu", group_ids=groups)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_loss.py -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'aspect_tau_fixed'` from `dataclasses.replace`.

- [ ] **Step 3: Implement.** In `FactorTrainingConfig`, after `lambda_swap: float = 1.0`, add:

```python
    aspect_tau_fixed: bool = False          # True: keep the aspect-loss temperature at its step-0 value (spec §15)
```

Next to the existing `lambda_aspect` validation (`if config.lambda_aspect < 0: ...`), add:

```python
    if config.aspect_tau_fixed and config.lambda_aspect <= 0:
        raise ValueError("aspect_tau_fixed needs lambda_aspect > 0")
```

In the `if config.lambda_aspect > 0:` initialisation block, replace

```python
        log_tau = torch.nn.Parameter(torch.tensor(math.log(max(std, 1e-6)), device=selected_device))
        optimizer.add_param_group({"params": [log_tau]})
```

(the second occurrence, inside the aspect block only; the condition-loss block above it stays unchanged) with

```python
        if config.aspect_tau_fixed:
            log_tau = torch.tensor(math.log(max(std, 1e-6)), device=selected_device)   # constant, not optimised
        else:
            log_tau = torch.nn.Parameter(torch.tensor(math.log(max(std, 1e-6)), device=selected_device))
            optimizer.add_param_group({"params": [log_tau]})
```

- [ ] **Step 4: Run the tests, including the golden-codes test**

Run: `/root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/test_aspect_loss.py src/test/test_train_factors.py -q`
Expected: all PASS (in particular `test_default_config_codes_match_pre_change_golden_values` and `test_checkpoint_round_trip_gives_identical_codes`).

- [ ] **Step 5: Write the change-log entry** in `.claude/20261003_log.md` (create the file if missing, append otherwise): header `# /src/train/train_factors.py`, the before and after snippets of Step 3, and why ("spec §15 diagnostics: run LT and H2 cells G1, G3, G6, G8 need a fixed aspect temperature; default False keeps every existing run and checkpoint unchanged").

- [ ] **Step 6: Commit**

```bash
git add src/train/train_factors.py src/test/test_aspect_loss.py
git add -f .claude/20261003_log.md
git commit -m "feat(v2): aspect_tau_fixed option for factor training (default off; spec §15 run LT)"
```

---

### Task 4: Shared constants and the LAB and MK banks

**Files:**
- Create: `src/test/20261105_method_repair_diagnostics/common.py`
- Create: `src/test/20261105_method_repair_diagnostics/build_banks.py`

**Interfaces:**
- Consumes: `run_gonogo` (E3 folder) as `rg`: `rg.ROOT`, `rg.E1`, `rg.E2`, `rg.FIELDS`, `rg.C0_CELL`, `rg.FULL_STEPS`, `rg.SMOKE_STEPS`, `rg.sha_file`, `rg.sha_array`; `src.train.pseudo_partitions.{build_episode_bank, kmeans_partition}`; `src.eval.aspect_episodes.{AspectEpisodes, PaintingValueIndex, validate_aspect_episodes}`; `src.data.artelingo_splits.{artelingo_splits, artelingo_aspect_labels}`.
- Produces (`common.py`): `HERE`, `ROOT`, `rg`, `E3 = ROOT / "src/test/20261101_aspect_factor_gonogo"`, `AFFECT_CACHE = ROOT / "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz"`, `C0_CKPT = ROOT / "src/test/20261016_factor_learning_grid/checkpoints/C0_seed42.pt"`, `BANK_SIZE = 65_536`, `BANK_SEEDS = {"LAB": 1042, "MK": 2042}`, `FRESH_LABEL_SEED = 3042`, `N_FRESH = 4096`, `LAB_PARTS = ("emotion", "genre", "style")`, `MK_K = {"affect8": 8, "caption10": 10, "image23": 23}`, `A3 = {"lambda_aspect": 3.0, "lambda_swap": 1.0}`, `RUNS = {"L3": ("LAB", A3), "L5": ("LAB", {"lambda_aspect": 1.0, "lambda_swap": 1.0, "aspect_beta": 0.0}), "LT": ("LAB", {**A3, "aspect_tau_fixed": True}), "MK3": ("MK", A3)}`, `LABEL_RUNS = ("L3", "L5", "LT")`, `folders(smoke) -> dict` (keys `ckpt`, `res`), `run_config(run, seed, steps) -> FactorTrainingConfig`, `load_bank(name, smoke) -> tuple[AspectEpisodes, str]`, `local_rows(data) -> tuple[np.ndarray, np.ndarray]` returning `(st, local_groups)` with the E2 alignment asserts.
- Produces (files): `results/bank_LAB.npz`, `results/bank_MK.npz` (E2's key layout: `aspect_a`, `aspect_b`, `block_sizes`, `block_pairs`, and the six `FIELDS`), `results/partitions_LAB.npz`, `results/partitions_MK.npz`, `results/build_record.json` (SHA-256 of each file, block sizes, eligible value counts per partition, validation counts, timings, and for MK the AMI of each partition with each evaluation aspect as a descriptive diagnostic).

- [ ] **Step 1: Write `common.py`**

```python
"""Shared constants and helpers of the method-repair diagnostics (PREREGISTRATION.md in this folder)."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E3 = ROOT / "src/test/20261101_aspect_factor_gonogo"
sys.path.insert(0, str(E3))
import run_gonogo as rg  # noqa: E402  (puts the repo root on sys.path)

from src.eval.aspect_episodes import AspectEpisodes  # noqa: E402
from src.train.train_factors import R3_CONFIG  # noqa: E402

AFFECT_CACHE = ROOT / "src/test/20261018_affect_factor_learning/cache/affect_prepare.npz"
C0_CKPT = ROOT / "src/test/20261016_factor_learning_grid/checkpoints/C0_seed42.pt"
BANK_SIZE = 65_536
BANK_SEEDS = {"LAB": 1042, "MK": 2042}
FRESH_LABEL_SEED = 3042
N_FRESH = 4096
LAB_PARTS = ("emotion", "genre", "style")
MK_K = {"affect8": 8, "caption10": 10, "image23": 23}
A3 = {"lambda_aspect": 3.0, "lambda_swap": 1.0}
RUNS = {
    "L3": ("LAB", A3),
    "L5": ("LAB", {"lambda_aspect": 1.0, "lambda_swap": 1.0, "aspect_beta": 0.0}),
    "LT": ("LAB", {**A3, "aspect_tau_fixed": True}),
    "MK3": ("MK", A3),
}
LABEL_RUNS = ("L3", "L5", "LT")


def folders(smoke: bool) -> dict:
    sub = "smoke" if smoke else ""
    return {"ckpt": HERE / "checkpoints" / sub, "res": HERE / "results" / sub}


def run_config(run: str, seed: int, steps: int = rg.FULL_STEPS):
    """E3's C0 base recipe (rg.run_config without its RUNS table) plus this stage's run fields."""
    base = dataclasses.replace(R3_CONFIG, painting_batches=True, seed=seed, epochs=steps, **rg.C0_CELL)
    return dataclasses.replace(base, **RUNS[run][1])


def local_rows(data, splits):
    """(scorer-train rows, local painting groups), aligned with E2's partitions.npz (asserted)."""
    st = np.asarray(splits.scorer_train)
    local_groups = np.unique(splits.groups[st], return_inverse=True)[1].astype(np.int64)
    e2 = np.load(rg.E2 / "partitions.npz")
    if not np.array_equal(e2["local_groups"], local_groups):
        raise AssertionError("local_groups differ from E2's partitions.npz")
    return st, local_groups


def load_bank(name: str, smoke: bool) -> tuple:
    res = folders(smoke)["res"]
    path = res / f"bank_{name}.npz"
    record = json.loads((res / "build_record.json").read_text())
    sha = rg.sha_file(path)
    if sha != record["sha256"][f"bank_{name}.npz"]:
        raise AssertionError(f"{path}: SHA-256 differs from build_record.json")
    z = np.load(path)
    return AspectEpisodes(str(z["aspect_a"]), str(z["aspect_b"]), *(z[f].astype(np.int64) for f in rg.FIELDS)), sha
```

- [ ] **Step 2: Write `build_banks.py`** (CPU). Structure, following `src/test/20261031_pseudo_partitions/build_partitions.py`:

```python
"""Method-repair diagnostics: the LAB bank (evaluation labels of scorer-train rows; H3 diagnostic only) and the MK
bank (label-free k-means at matched granularity). Rules: PREREGISTRATION.md §3. Run from the repo root:

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python \
      src/test/20261105_method_repair_diagnostics/build_banks.py [--smoke]
"""
import argparse
import json
from dataclasses import fields
from itertools import combinations
from time import perf_counter

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

from common import AFFECT_CACHE, BANK_SEEDS, BANK_SIZE, LAB_PARTS, MK_K, folders, local_rows, rg
from src.data.artelingo import load_artelingo
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits
from src.eval.aspect_episodes import AspectEpisodes, PaintingValueIndex, eligible_values, validate_aspect_episodes
from src.train.pseudo_partitions import build_episode_bank, kmeans_partition

VALIDATE_N = 1000


def block_slices(sizes):
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(len(sizes))]


def subset(ep, name_a, name_b, idx):
    return AspectEpisodes(name_a, name_b, *(getattr(ep, f.name)[idx] for f in fields(ep)[2:]))


def build(name, parts, groups, n, smoke, rng):
    names = sorted(parts)
    pairs = list(combinations(names, 2))
    per_pair = 256 if smoke else -(-BANK_SIZE // len(pairs))
    total = len(pairs) * per_pair if smoke else BANK_SIZE
    t0 = perf_counter()
    bank = build_episode_bank(parts, groups, np.arange(n), n_per_pair=per_pair, seed=BANK_SEEDS[name],
                              min_paintings=30)
    sizes = [per_pair] * len(pairs)
    if len(bank.anchor) > total:                     # first BANK_SIZE rows of the concatenation, as in E2
        keep = np.arange(total)
        bank = AspectEpisodes(bank.aspect_a, bank.aspect_b, *(getattr(bank, f.name)[keep] for f in fields(bank)[2:]))
        sizes[-1] -= sum(sizes) - total
    assert len(bank.anchor) == sum(sizes) == total
    assert bank.rows().min() >= 0 and bank.rows().max() < n
    build_s = perf_counter() - t0
    index = PaintingValueIndex(parts, groups)
    sample = np.sort(rng.choice(len(bank.anchor), size=min(VALIDATE_N, len(bank.anchor)), replace=False))
    checked = {}
    for sl, (a, b) in zip(block_slices(sizes), pairs):
        idx = sample[(sample >= sl.start) & (sample < sl.stop)]
        third = [x for x in names if x not in (a, b)][0]
        validate_aspect_episodes(subset(bank, a, b, idx), parts, groups, index, third=third)
        checked[f"{a}__{b}"] = int(len(idx))
    return bank, sizes, pairs, checked, build_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="256 episodes per pair, writes to results/smoke/")
    args = ap.parse_args()
    out = folders(args.smoke)["res"]
    out.mkdir(parents=True, exist_ok=True)
    for f in ("bank_LAB.npz", "bank_MK.npz", "build_record.json"):
        if not args.smoke and (out / f).exists():
            raise FileExistsError(f"{out / f} exists; refusing to overwrite pre-registered inputs")
    data = load_artelingo()
    splits = artelingo_splits(data)
    st, groups = local_rows(data, splits)
    n = len(st)
    labels = artelingo_aspect_labels(data)
    lab = {k: labels[k][st].astype(np.int64) for k in LAB_PARTS}
    probs = np.load(AFFECT_CACHE)["affect_probs"]
    assert probs.shape == (n, 28), probs.shape
    mk = {
        "affect8": MiniBatchKMeans(n_clusters=MK_K["affect8"], random_state=42, n_init=3, batch_size=4096)
        .fit_predict(probs).astype(np.int64),
        "caption10": kmeans_partition(data.txt_features[st], np.arange(n), k=MK_K["caption10"], seed=42),
        "image23": kmeans_partition(data.img_features[st], np.arange(n), k=MK_K["image23"], seed=42),
    }
    rec = {"smoke": args.smoke, "n_rows": n, "sha256": {}, "banks": {}, "eligible_values": {}, "mk_ami": {}}
    rng = np.random.default_rng(42)
    for name, parts in (("LAB", lab), ("MK", mk)):
        rec["eligible_values"][name] = {k: len(eligible_values(v, groups, np.arange(n)[v >= 0], 30))
                                        for k, v in parts.items()}
        bank, sizes, pairs, checked, build_s = build(name, parts, groups, n, args.smoke, rng)
        arrays = {f.name: getattr(bank, f.name) for f in fields(bank)[2:]}
        np.savez(out / f"bank_{name}.npz", aspect_a=np.array(bank.aspect_a), aspect_b=np.array(bank.aspect_b),
                 block_sizes=np.array(sizes), block_pairs=np.array([f"{a}__{b}" for a, b in pairs]), **arrays)
        np.savez(out / f"partitions_{name}.npz", **parts, local_groups=groups)
        for f in (f"bank_{name}.npz", f"partitions_{name}.npz"):
            rec["sha256"][f] = rg.sha_file(out / f)
        rec["banks"][name] = {"partitions": sorted(parts), "episodes": int(len(bank.anchor)), "block_sizes": sizes,
                              "block_pairs": [f"{a}__{b}" for a, b in pairs], "validated_per_block": checked,
                              "validation": "passed", "build_s": build_s, "seed": BANK_SEEDS[name]}
        print(f"bank {name}: {rec['banks'][name]}", flush=True)
    for p, v in mk.items():                          # descriptive diagnostic; labels used only here
        rec["mk_ami"][p] = {a: float(adjusted_mutual_info_score(lab[a][lab[a] >= 0], v[lab[a] >= 0]))
                            for a in LAB_PARTS}
    (out / "build_record.json").write_text(json.dumps(rec, indent=1))
    print("build_record:", json.dumps(rec["mk_ami"]), flush=True)


if __name__ == "__main__":
    main()
```

The script must be run with the folder on `sys.path` (`common` import): add `sys.path.insert(0, str(Path(__file__).resolve().parent))` before `from common import ...` (and `import sys` / `from pathlib import Path`).

- [ ] **Step 3: Smoke run**

Run: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261105_method_repair_diagnostics/build_banks.py --smoke`
Expected: two `bank ...` lines with `validation: passed`, `results/smoke/build_record.json` written; LAB eligible values about 8 (emotion), 10 (genre), 23 or more (style); MK 8, 10, 23.

- [ ] **Step 4: Real build** (CPU, about 10 minutes)

Run: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261105_method_repair_diagnostics/build_banks.py 2>&1 | tee src/test/20261105_method_repair_diagnostics/build.log`
Expected: both banks 65,536 episodes, validation passed.

- [ ] **Step 5: Commit** (scripts only; results are gitignored)

```bash
git add src/test/20261105_method_repair_diagnostics/common.py src/test/20261105_method_repair_diagnostics/build_banks.py
git commit -m "feat(v2): LAB (scorer-train labels, H3 only) and MK (matched-k, label-free) episode banks"
```

---

### Task 5: Training runner and the day-1 launcher

**Files:**
- Create: `src/test/20261105_method_repair_diagnostics/train_runs.py`
- Create: `src/test/20261105_method_repair_diagnostics/run_day1.sh`

**Interfaces:**
- Consumes: `common.{RUNS, LABEL_RUNS, folders, run_config, load_bank, local_rows, rg}`; `src.train.train_factors.{train_factors, save_factor_checkpoint}`; E2's `graph.npz` (SHA-256 against E2's `build_record.json`).
- Produces: `checkpoints/<RUN>_seed<seed>.pt`, `results/history_<RUN>_seed<seed>.json` (keys as E3's history files: `run`, `seed`, `bank`, `bank_sha256`, `graph_sha256`, `config`, `train_s`, `peak_gpu_gib`, `history`, `checkpoint`, `checkpoint_sha256`, `code_stats_scorer_train`), and after every LAB run `results/label_checkpoints.json` updated with `{name: sha256}`.

- [ ] **Step 1: Write `train_runs.py`**: `--train RUN` (choices `sorted(RUNS)`), `--seed` (default 42, allowed {42, 43}), `--smoke` (50 steps on the smoke bank). Body = E3's `rg.train` with three changes: the bank comes from `load_bank(RUNS[run][0], smoke)`; the config from `run_config(run, seed, steps)`; after saving, LAB runs append their checkpoint SHA-256 to `results/label_checkpoints.json`. Keep E3's refusals: never overwrite a checkpoint, history or failed record; non-finite codes write `failed_<name>.json` and exit. Copy `rg.train`'s graph checks (`load_npz(rg.E2 / "graph.npz")`, SHA-256 against `json.loads((rg.E2 / "build_record.json").read_text())["sha256"]["graph.npz"]`, shape `(n, n)`) and its `group_ids=local_groups` call:

```python
model, img_codes, txt_codes = train_factors(
    data.img_features[st], data.txt_features[st], graph, config, device=rg.DEVICE, group_ids=local_groups,
    aspect_bank=bank, history=history, log_every=50)
```

Smoke mode uses the smoke banks (`results/smoke/`) and E2's smoke graph (`rg.E2 / "smoke" / "graph.npz"` against the smoke build record), writes to `checkpoints/smoke/` and `results/smoke/`, and may overwrite smoke outputs.

- [ ] **Step 2: Write `run_day1.sh`**

```bash
#!/usr/bin/env bash
# Day-1 trainings of the method-repair diagnostics (PREREGISTRATION.md §4). The CONTROLLER runs this under the GPU lock:
#   flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261105_method_repair_diagnostics/run_day1.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=/root/miniconda3/envs/CoSiR/bin/python
DIR=src/test/20261105_method_repair_diagnostics
mkdir -p "$DIR/results"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4      # four processes share the 32 cores with other sessions
for run in L3 L5 LT MK3; do
  $PY "$DIR/train_runs.py" --train "$run" --seed 42 > "$DIR/results/train_${run}_seed42.log" 2>&1 &
done
wait
echo "day-1 trainings done"
```

- [ ] **Step 3: Smoke test** (GPU check first; smoke uses the GPU briefly)

Run: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` (must be empty or yours), then
`flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261105_method_repair_diagnostics/train_runs.py --train LT --smoke`
Expected: 50 steps, a smoke checkpoint, `history_LT_seed42.json` whose `history["tau"]` values are all equal; then the same for `--train L3 --smoke` with τ values that change.

- [ ] **Step 4: Commit**

```bash
git add src/test/20261105_method_repair_diagnostics/train_runs.py src/test/20261105_method_repair_diagnostics/run_day1.sh
git commit -m "feat(v2): diagnostics training runner (L3, L5, LT, MK3) and day-1 launcher"
```

- [ ] **Step 5: CONTROLLER launches the real day-1 trainings** (not a subagent): check `nvidia-smi`, then Bash with `run_in_background: true`:
`flock -n -o -E 75 /tmp/gpu0.lock bash src/test/20261105_method_repair_diagnostics/run_day1.sh`
Expected: four checkpoints in about 10 to 15 minutes (4 × about 3.9 GiB on the 24 GiB GPU).

---

### Task 6: H1 pilot scorer (seed 42, CPU)

**Files:**
- Create: `src/test/20261105_method_repair_diagnostics/score_pilot.py`

**Interfaces:**
- Consumes: `common.{rg, ROOT}`; `rg.EvalContext(42, smoke)` (attributes `img`, `txt`, `cos`, `pooled`, `parity`, `anchor_group`, `n`, `shas`, methods `encode(ckpt)`, `masked(arr)`, `summary(pa)`); `src.eval.aspect_scorers.{EvalInputs, agreement_term}`; `src.eval.aspect_nested.*`; `src.eval.aspect_metrics.{per_anchor, compare, METRICS}`; E3's `results/picked.json`; E1's `codes_C0.npz`, `codes_SE.npz`, `per_anchor_seed42.npz` (keys `<scorer>__<metric>`, e.g. `rca__r1`).
- Produces: `results/pilot_seed42.json` with, per model in (A3, A1, A2, A4, A5, A6, C0, SE): `nested` and `control` summaries (`ctx.summary`), `either` points (R@1 + other, for nested, control and cosine), `picks`, paired `vs_control` (r1, gain), `vs_cosine`, `vs_rca` (r1, gain), and for A3 also `m_R`, `SE_R`, `m_g`, `SE_g`, `reading`, `g_star`, `predicted_power` {r1, gain, joint}; `A3_minus_C0_nested` (r1, gain); `profile_A3` (all 56 cells: r1, gain, either; and the 30 control sums: r1); provenance (`episodes_sha256`, checkpoint and code SHA-256s, script SHA-256). Also `results/per_anchor_pilot_seed42.npz` (`<model>__<nested|control>__<metric>`) and `results/pilot_seed42.txt`. Refuses to overwrite a real `pilot_seed42.json`.

- [ ] **Step 1: Write `score_pilot.py`.** Model codes:

```python
def model_codes(ctx, name, smoke):
    if name in ("C0", "SE"):
        path = rg.E1 / f"codes_{name}.npz"
        z = np.load(path)
        return ctx.masked(z["img"]), ctx.masked(z["txt"]), {"codes": str(path.relative_to(ROOT)),
                                                            "sha256": rg.sha_file(path)}
    ckpt = rg.checkpoint_path(name, 42, smoke)          # smoke: A1's smoke checkpoint stands in
    sha = rg.sha_file(ckpt)
    if name == "A3" and not smoke:
        pick = json.loads((rg.HERE / "results" / "picked.json").read_text())
        assert pick["run"] == "A3" and sha == pick["checkpoint_sha256"], "A3 checkpoint != E3's pick"
    ic, tc = ctx.encode(ckpt)
    return ic, tc, {"checkpoint": str(ckpt.relative_to(ROOT)), "sha256": sha}
```

Per model: `inp = EvalInputs(ctx.img, ctx.txt, ic, tc)`; `tu = agreement_term(inp, ctx.pooled, uniform=True)`; `ta = agreement_term(inp, ctx.pooled)`; `nested, control, picks = crossfit_nested(ctx.cos, tu, ta, ctx.parity)`; `pn, pc = per_anchor(nested), per_anchor(control)`; `cl = ctx.anchor_group`. A3 reading:

```python
vr, vg = compare(pn, pc, cl, "r1"), compare(pn, pc, cl, "gain")
m_r, se_r, m_g, se_g = vr["point"], se_from_ci(vr["ci95"]), vg["point"], se_from_ci(vg["ci95"])
reading = margin_reading(m_r, se_r, m_g, se_g)
g_star = ceiling_threshold(se_r, se_g)
power = {"r1": predicted_power(m_r, se_r), "gain": predicted_power(m_g, se_g)}
power["joint_if_independent"] = power["r1"] * power["gain"]
```

RCA per-anchor: `t9 = np.load((rg.E1 / "smoke" if smoke else rg.E1) / "per_anchor_seed42.npz")`, `rca = {m: t9[f"rca__{m}"] for m in METRICS}` (assert the keys exist). Profile: for each of `nested_cells()` the uncross-fitted `per_anchor(nested_scores(ctx.cos, tu, ta, u, a))` pooled means (×100) of r1, gain and r1 + other; for each of `control_sums()` the control's r1. `assert_finite_tree` from `rg` on the result before writing. Print a short table and the A3 reading.

- [ ] **Step 2: Smoke run**

Run: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261105_method_repair_diagnostics/score_pilot.py --smoke`
Expected: `results/smoke/pilot_seed42.json`, a reading string for A3, every number finite.

- [ ] **Step 3: Commit**

```bash
git add src/test/20261105_method_repair_diagnostics/score_pilot.py
git commit -m "feat(v2): H1 nested-score pilot scorer (seed 42, A3 primary, pre-registered reading)"
```

- [ ] **Step 4: CONTROLLER runs the real pilot** (CPU, minutes; may run while Task 5's trainings run):
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261105_method_repair_diagnostics/score_pilot.py 2>&1 | tee src/test/20261105_method_repair_diagnostics/results/pilot.log`

---

### Task 7: H3 scorer (fresh label episodes on scorer-train rows, transfer on seed 42)

**Files:**
- Create: `src/test/20261105_method_repair_diagnostics/score_h3.py`

**Interfaces:**
- Consumes: `common.*`; `results/partitions_LAB.npz`; checkpoints `L3`, `L5`, `LT`, `MK3` (seed 42; seed 43 reruns if present), E3's `A3_seed42.pt`, `common.C0_CKPT`; `results/pilot_seed42.json` (for `g_star` and A3's SHA, asserted equal); `src.eval.aspect_episodes.{build_aspect_episodes, validate_aspect_episodes, PaintingValueIndex, concat_episodes}`; `src.train.train_factors.{encode_rows, load_factor_checkpoint}`; `src.eval.aspect_nested.*`.
- Produces: `results/h3.json` with `fresh` (episode SHA-256s, seeds, counts), per LAB run `fit` {compare vs A3 on gain, reading, seed-43 resolution if used}, `X_minus_C0` (descriptive), `loss_vs_constant` (last 10 logged aspect losses: mean and percent below 3.258, or 2.565 when λ_swap = 0), `tau` {first, last}; per LAB run and MK3 `transfer_seed42` {term-only summary and either, nested summary, nested gain point, picks}; `best_fitting_run`, `best_nested_gain`, `g_star`, `h3_reading`; `matched_k` {compare MK3 − A3 term-only gain on seed 42, `granularity_lever`: bool}; `needs_seed43`: list of runs whose fit is inconclusive without a seed-43 checkpoint (then `h3_reading` is `null` and the script exits with code 3). Also `results/per_anchor_h3.npz`, `results/h3.txt`.

- [ ] **Step 1: Write `score_h3.py`.** Fresh label episodes (scorer-train rows):

```python
PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))
lab = {k: v for k, v in np.load(res / "partitions_LAB.npz").items() if k in LAB_PARTS}
groups = np.load(res / "partitions_LAB.npz")["local_groups"]
index = PaintingValueIndex(lab, groups)
parts = []
for i, (a, b, third) in enumerate(PAIRS):
    ep = build_aspect_episodes(lab, groups, np.arange(n), a, b, n_fresh, FRESH_LABEL_SEED + i, third=third,
                               min_paintings=30, index=index)
    validate_aspect_episodes(ep, lab, groups, index, third=third)
    parts.append(ep)
fresh = concat_episodes(parts)
clusters = groups[fresh.anchor]
```

(`n_fresh = 256` in smoke, else `N_FRESH`.) Term-only per-anchor on scorer-train rows for a checkpoint:

```python
def term_only(ckpt, ep):
    model, _ = load_factor_checkpoint(ckpt, device="cpu")
    ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=st, device="cpu")
    assert np.isfinite(ic).all() and np.isfinite(tc).all()
    return per_anchor(agreement_term(EvalInputs(data.img_features[st], data.txt_features[st], ic, tc), ep))
```

Fit per LAB run: `r = compare(pa[X], pa["A3"], clusters, "gain")`, `reading = fit_reading(r)`; if `inconclusive` and `checkpoints/X_seed43.pt` exists, average the two seeds' per-anchor arrays (`{m: 0.5 * (pa42[m] + pa43[m])}`) and decide `fits` if the paired lower bound against A3 is above 0, else `no_fit`; if it does not exist, record X in `needs_seed43`. Transfer on seed 42: `ctx = rg.EvalContext(42, smoke)`, `ic, tc = ctx.encode(ckpt)`, term-only `per_anchor(agreement_term(inp, ctx.pooled))`, nested via `crossfit_nested`. `best_nested_gain` = the largest `ctx.summary(per_anchor(nested))["gain"]["point"]` among fitting LAB runs. `g_star` from `pilot_seed42.json["A3"]["g_star"]`. `h3_reading(fits_resolved, best_nested_gain, g_star)`. Matched-k: `compare(term-only MK3, term-only A3, ctx.anchor_group, "gain")`, `granularity_lever = ci95[0] > 0`. In smoke mode the A1 smoke checkpoint stands in for A3 and C0, and the smoke checkpoints of Task 5 for L3, L5, LT, MK3.

- [ ] **Step 2: Smoke run** (needs Task 5's smoke checkpoints and Task 6's smoke pilot)

Run: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python src/test/20261105_method_repair_diagnostics/score_h3.py --smoke`
Expected: `results/smoke/h3.json`, every number finite, an `h3_reading` or a `needs_seed43` list.

- [ ] **Step 3: Commit**

```bash
git add src/test/20261105_method_repair_diagnostics/score_h3.py
git commit -m "feat(v2): H3 learnability scorer (paired fit vs A3 on fresh label episodes, seed-42 ceiling, matched-k)"
```

- [ ] **Step 4: CONTROLLER runs the real H3 scoring** after Task 5's trainings and Task 6's pilot. If it exits 3 (`needs_seed43`), launch `flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/CoSiR/bin/python src/test/20261105_method_repair_diagnostics/train_runs.py --train <X> --seed 43` (background), then rerun `score_h3.py` once (it may overwrite `h3.json` only when the previous one recorded `needs_seed43`).

---

### Task 8: Joint decision

**Files:**
- Create: `src/test/20261105_method_repair_diagnostics/decide.py`

**Interfaces:**
- Consumes: `results/pilot_seed42.json` (`A3.reading`), `results/h3.json` (`h3_reading`, `matched_k.granularity_lever`); `src.eval.aspect_nested.joint_decision`.
- Produces: `results/decision.json` `{h1, h3, decision, next_step, h2_bank, inputs_sha256, decided_at}` and a printed next step. Refuses to overwrite.

- [ ] **Step 1: Write `decide.py`** with the pre-registered wording:

```python
NEXT = {
    "preregister_A3_nested": ("Write the A′ pre-registration: the nested score on A3, spec-§15 cross-fitting on "
                              "seed 42, the GO test once on fresh seed-45 episodes by Oct 12 (H4 only if H3's ceiling "
                              "is sufficient and an H2 model passes its gate and beats A3's pilot by Oct 9)."),
    "h2_grid": ("Run the pre-registered H2 grid (PREREGISTRATION.md §9) on bank {bank} behind its fit gate, re-pilot "
                "each gate-passer with the H1 rule, pre-register A′ by Oct 9 if one is promising, else branch 3."),
    "branch_3": "Stop the repair: branch 3 (analysis paper), per spec §4 and §15.",
}
```

`h2_bank = "MK" if h3["matched_k"]["granularity_lever"] else "AIC"`; `decision = joint_decision(pilot["A3"]["reading"], h3["h3_reading"])` (raises if `h3_reading` is null).

- [ ] **Step 2: Smoke run** against the smoke JSONs (`--smoke`), then commit.

```bash
git add src/test/20261105_method_repair_diagnostics/decide.py
git commit -m "feat(v2): joint decision of the method-repair diagnostics (pre-registered table)"
```

- [ ] **Step 3: CONTROLLER runs the real decision** once Tasks 6 and 7 have real results.

---

### Task 9: Reports, logs and index

**Files:**
- Create: `docs/reports/auto/v2/2026-11-04_ars_repair_order_review.md`
- Create: `docs/reports/auto/v2/2026-11-05_method_repair_diagnostics.md`
- Create: `docs/reports/assets/2026-11-05_method_repair_diagnostics/build_figures.py` and its figures
- Create: `src/test/20261105_method_repair_diagnostics/20261105_method_repair_diagnostics_log.md`
- Modify: `docs/reports/reports_sum.md` (two rows), `docs/superpowers/episode_seed_ledger.md` (if anything changed)

- [ ] **Step 1: ARS review report** (from `src/test/20261104_ars_repair_order_review/`): the question, the memo, the two-seat panel and its provenance, the verdict (major revision, D1 block repairable, D2 warn), the order recommendation, R1 to R12 and how spec §15 and the pre-registration carry each, the memo error the panel caught (AMI row source), and what the reduced panel did not assess.
- [ ] **Step 2: Diagnostics report**: chain (E3 NO-GO → ARS order review → this stage); H1 pilot (A3 nested against its nested control, against cosine 12.96 and RCA 13.38 on seed 42, and E3's cross-fitted A3 13.39 / 0.52 as the baseline of the old score; the 56-cell profile figure; the reading with m and SE; predicted seed-45 power); H3 (fit table L3, L5, LT against A3 and C0 with loss and τ; transfer and ceiling against g*; matched-k); the joint decision; disclosures (development looks, H3 label use, H1 provenance, seed-45 painting overlap). `build_figures.py` re-derives every number from the per-anchor npz files and asserts equality with the JSONs.
- [ ] **Step 3: Folder log**, the two `reports_sum.md` rows, then run `/root/miniconda3/envs/CoSiR/bin/python scripts/check_reports_sum.py` (must print OK).
- [ ] **Step 4: Commit**

```bash
git add docs/reports/auto/v2/2026-11-04_ars_repair_order_review.md docs/reports/auto/v2/2026-11-05_method_repair_diagnostics.md \
  docs/reports/assets/2026-11-05_method_repair_diagnostics/ docs/reports/reports_sum.md \
  src/test/20261105_method_repair_diagnostics/20261105_method_repair_diagnostics_log.md docs/superpowers/episode_seed_ledger.md
git commit -m "docs(v2): ARS repair-order review and method-repair diagnostics reports"
```

---

### Task 10: Final whole-branch review, fix wave, scoped re-review

- [ ] **Step 1:** One reviewer on the most capable model re-derives every load-bearing number (the A3 pilot margins, SEs and reading; each LAB fit comparison; the ceiling against g*; the matched-k comparison; the decision) from the stored per-anchor arrays with independent code, checks the pre-registration commit precedes every result file's creation, checks no seed-43/44/45 episode was built or scored and no LAB setting entered a candidate, and checks the tests would fail with their guards removed.
- [ ] **Step 2:** One fix wave for confirmed findings (report and code), then a scoped re-review of the fixes.
- [ ] **Step 3:** Hand the decision to the user. Do not write the A′ pre-registration or touch seed 45.
