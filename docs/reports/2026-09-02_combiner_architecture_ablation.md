# What actually matters in the buddy-fusion combiner: family, not dimension — Combiner Architecture Ablation

**Date:** 2026-09-02 · **Dataset:** RedCaps 150k (`redcaps_150k`) · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `src/model/combiner.py` (new classes), `src/model/cosirmodel.py` (`combiner_type` switch), `scripts/run_combiner_architecture_smoke.sh`, `scripts/analyze_combiner_architecture_smoke.py`, `scripts/run_combiner_buddydim_sweep.sh`, `scripts/analyze_combiner_buddydim_sweep.py`, `src/test/20260902_combiner_architecture/test_identity_init.py`
**Plan / working notes:** `.planning/2026-09-01-combiner-architecture-ablation/{task_plan,findings,progress}.md`
**Source brainstorm:** `docs/reports/2026-09-01_combiner_architecture_brainstorm.md` (literature search + candidate shortlist, dispatched to Codex, grounded in the actual repo code)

---

## TL;DR

The production fusion module (`Combiner_new`: two MLP towers → concat → MLP decode → learned scalar gate) was never chosen by ablation — combine mechanism, depth, and buddy dimension were all set ad hoc. A literature-grounded brainstorm (composed-image-retrieval combiners, FiLM, LoRA, Highway/ResNet, SE-Net, HyperNetworks) produced a ranked shortlist of alternatives, which a staged smoke ablation then tested empirically.

**Fusion family is the axis that matters, and it isn't close.** A rank-16 low-rank residual adapter (`x + B·a(z)`, zero-init `B` for exact identity at init) beats the current architecture on oracle i2t retrieval (+3.1 R1, mean/SEM +15, confirmed at 3 seeds) while staying comparatively robust on the deployment tier. Two other literature-motivated candidates — a zero-init full-width residual MLP and a FiLM residual — also win on oracle i2t (+2.7 to +3.0) but **collapse on predictor-conditioned (deployment) retrieval**, regressing 14–28 R1 points versus the current architecture's -1.6 to -3.6, a reproducible effect (mean/SEM up to -95), not noise. Restricting the fusion to a small subspace is what makes conditioning on a tiny buddy vector deployment-safe; letting it touch the full frozen embedding is not.

**Buddy dimension, by contrast, is a non-factor at this scale.** Sweeping `{8,16,32}` for the low-rank winner moved nothing — every metric landed within noise of the dim=16 anchor. The existing default (`embedding_dim: 16` in `configs/model/*.yaml`) has no reason to change on retrieval grounds, and it's already the cheaper/better-justified choice on storage and graph-geometry grounds from prior work.

**A real bug was also found and fixed along the way**, independent of the ablation's headline result: the current production `Combiner_new` is not actually identity-init at t=0 (its gate zero-inits to s=0.5, but the two MLP towers feeding it are randomly initialized), contradicting the codebase's own identity-init convention elsewhere and confounding any prior read of "does conditioning help" with "does a random perturbation at init help."

**Scale caveat:** all of the above is `redcaps_150k`, 30 epochs, 3 seeds — a smoke-test resolution meant to narrow the search space, not paper numbers. `lowrank` should be validated at the project's real training scale/duration before it's adopted as the default combiner for the buddy publication plan's own experiments.

---

## Method

### Phase 1–2: Brainstorm and plan

Dispatched to Codex (`codeagent-wrapper --backend codex`) with instructions to read the actual `src/model/{cosirmodel,combiner,condition_predictor}.py` and `configs/model/*.yaml` — not to invent from a generic prompt — then do a literature search and rank candidate fusion architectures for the specific shape here: a frozen CLIP/SigLIP embedding (512–1024D) fused with a tiny trainable per-sample buddy vector (2–16D, a spectral coordinate from a cross-modal mutual-kNN graph), under the project's identity-init convention. Full shortlist, parameter budgets, and literature refs (LoRA, FiLM, Highway, SE-Net, HyperNetworks, GeneCIS/CLIP4Cir, Deep VIB) in `docs/reports/2026-09-01_combiner_architecture_brainstorm.md`.

Converted into a 7-phase plan (`task_plan.md`): brainstorm → plan → implement 3 candidate combiners + identity-init unit tests → smoke sweep across families → buddy_dim sweep for the winner → rank/depth sweep (deferred, see below) → delivery.

### Phase 3: Implementation

Added three new fusion classes to `src/model/combiner.py`, gated behind a new `combiner_type` constructor arg on `CoSiRModel` (default `"legacy"` = today's unchanged `Combiner_new`; every existing config/call site is bit-identical to before):

- **`residual_control`** — `y = normalize(x + F(x,z))`, one 128-wide residual block, zero-init final projection. The fair "does topology matter, independent of init" control.
- **`lowrank`** — `y = normalize(x + B·a(z))`, rank-16 learned basis `B` (zero-init for exact identity), 1-layer coefficient MLP `a(z)`.
- **`film`** — `y = normalize(x + γ(z)·x + β(z))`, zero-init γ/β projection, **γ bounded to `0.1·tanh(γ_raw)`** after the zeroed projection (added in a follow-up code-review fix — the first implementation shipped γ unbounded, which would have let the multiplicative gate grow without limit during training).

Verified via `src/test/20260902_combiner_architecture/test_identity_init.py`: each new class's forward pass equals `F.normalize(x)` at init (atol=1e-5) across `feature_dim∈{512,768}`, `label_dim∈{2,16}`, and its zero-initialized terminal parameters receive a nonzero gradient after one backward step (proof the branch isn't permanently stuck at identity). Re-run independently by the reviewer (not just trusted from the implementer's report) both before and after the γ-bound fix.

### Phase 4–5: Smoke sweep and buddy_dim sweep

Fixed operating point throughout: `redcaps_150k`, `K=30`, `alpha=0.5`, `lr=1e-3`/`lr_label=1e-4`, 30 epochs, `initialization_strategy=buddies`. Both sweep scripts were `SMOKE=1`-tested (2 epochs, one arm) before committing to the full run, per this project's standard discipline. Both scripts fixed a copy-paste bug from the K-ablation template on their first `SMOKE=1` attempt (`+++wandb.tags=...` — three literal pluses instead of Hydra's `++` override operator) before passing.

Metrics: `test_oracle`/`test_pre_diff` t2i/i2t R1, mirroring the project's existing ablation-analysis convention. `test_pre_diff` is not a scale/naming variant of `test_oracle` — confirmed by reading `src/eval/pipeline.py:328-351` — it is retrieval conditioned on `model.condition_predictor`'s *predicted* buddy vector (reconstructed from the CLIP embedding alone, no graph/table lookup) minus the raw baseline. This is the deployment tier: a family that only helps when the true buddy vector is available is not automatically a deployment win.

---

## Results

**Commands run:**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SMOKE=1 bash scripts/run_combiner_architecture_smoke.sh                 # pipeline sanity (lowrank, 2 epochs)
bash scripts/run_combiner_architecture_smoke.sh                         # 4 arms x seed=1
SEED_SWEEP="2,3" bash scripts/run_combiner_architecture_smoke.sh        # +seeds 2,3 for all 4 arms
python scripts/analyze_combiner_architecture_smoke.py

SMOKE=1 bash scripts/run_combiner_buddydim_sweep.sh                     # pipeline sanity (dim=16, 2 epochs)
bash scripts/run_combiner_buddydim_sweep.sh                             # lowrank, dim in {8,16,32} x 3 seeds
python scripts/analyze_combiner_buddydim_sweep.py
```

### Fusion family (Phase 4), all cells n=3, deltas vs. `legacy`

| combiner_type | oracle t2i R1 Δ | oracle i2t R1 Δ | pre_diff t2i R1 Δ | pre_diff i2t R1 Δ |
|---|---:|---:|---:|---:|
| residual_control | +0.17 (n.s.) | +0.63 (n.s.) | **-13.57** (mean/SEM -62) | **-23.87** (mean/SEM -84) |
| **lowrank (r=16)** | -0.80 (mean/SEM -3.8) | **+3.10** (mean/SEM +15) | -3.60 (mean/SEM -31) | -3.63 (mean/SEM -11) |
| film | -0.33 (n.s.) | **+2.97** (mean/SEM +20) | **-17.63** (mean/SEM -95) | **-24.63** (mean/SEM -78) |

n.s. = not distinguishable from 0 at n=3. Raw per-arm means and the n=1 first pass (later found contaminated by a stray `SMOKE=1` run sharing the same wandb tag/seed as the real `lowrank` run — caught by cross-checking `wandb` run `created_at`/`epoch` directly, fixed in the analyzer by keeping only the max-epoch run per cell) are in `findings.md`.

**Reading:** `residual_control` and `film` are neutral-to-good under oracle conditioning but collapse once the fusion has to work from the predicted (not table) buddy vector — a reproducible fragility, not single-seed noise (confirmed identically at n=1 and n=3). `lowrank`'s contained rank-16 residual is the only family with both a real oracle gain and a comparatively small deployment-tier cost (about 1/5th to 1/7th the regression of the full-width alternatives). This is a clean empirical confirmation of the report's central thesis: restricting the fusion to a small subspace, rather than letting it touch the full frozen embedding, is what makes conditioning on a tiny buddy vector deployment-safe.

### buddy_dim (Phase 5), `lowrank` only, rank=16, all cells n=3

| buddy_dim | oracle t2i R1 | oracle i2t R1 | pre_diff t2i R1 | pre_diff i2t R1 |
|---:|---:|---:|---:|---:|
| 8 | 26.33 | 25.73 | -5.97 | -7.70 |
| 16 (anchor) | 26.33 | 25.87 | -5.37 | -7.30 |
| 32 | 26.10 | 25.60 | -5.00 | -8.00 |

All paired deltas from the dim=16 anchor are within ±0.7 R1 and not distinguishable from 0 — a clean null. No retrieval reason to move off `embedding_dim=16`, which is also favored on storage grounds (≈32MB/1M examples at 8-D vs. 128MB at 32-D) and graph-geometry grounds (72% vs. 63% local-kNN preservation at 16-D vs. 8-D, `docs/reports/2026-06-09_buddies_dim_hparam_study.md`).

---

## What wasn't done, and why

- **Rank/depth sweep (plan's Phase 6)**: deferred by explicit user decision after Phase 5's null result — the headline finding (family matters, dimension doesn't) was already in hand, and rank is a secondary, family-coupled question the report itself only recommended sweeping after dimension. Revisit if a full-scale run of `lowrank` looks capacity-limited.
- **Residual norm / cosine shift / gate value ranges**: the plan's Phase 4 diagnostic wishlist asked for these as sanity checks; none of the four combiner classes or the training loop currently log them, and adding that instrumentation was out of scope for a smoke pass. Would be worth adding before any full-scale validation run.
- **Hypernetwork and token cross-attention candidates** (report's candidates 6–7): never in scope for this pass — explicitly reserved for a clear failure signal (linear conditional coefficients insufficient, or pooled features discarding token-local information), neither of which appeared.

## Recommendation

Adopt `lowrank` (rank=16) as the candidate default combiner for future buddy experiments, but **validate at the project's real training scale/duration first** — everything here is `redcaps_150k` at 30 epochs, not the 300k/500k, 100-epoch regime the publication plan's actual experiments run at. This report does not modify `configs/model/*.yaml`'s default (`combiner_type: "legacy"` stays as-is) or the publication plan spec; that's a separate decision once `lowrank` is validated at scale.
