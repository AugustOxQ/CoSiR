# Findings: Combiner architecture ablation

## 2026-09-01 — Brainstorm complete

Full literature-backed brainstorm (dispatched to Codex, grounded in the actual repo code): `docs/reports/2026-09-01_combiner_architecture_brainstorm.md`.

Headline results:
- **Bug found, not just a gap:** production `Combiner_new` is not identity-init at t=0 (zero-init gate lands at s=0.5, but the two MLP towers feeding it are randomly initialized) — contradicts the codebase's own identity-init convention elsewhere (`other_proj`, `OtherProjMLP`). Must be isolated as its own control before comparing fusion topologies, or any topology comparison is confounded.
- **7 ranked candidate fusion families**, from cheapest/most-restrictive (low-rank residual dictionary, ~10-25K params) to most expensive/least-justified-first (token cross-attention, ~100-250K params). Full detail, identity-init recipes, and literature refs (LoRA, FiLM, Highway/ResNet, SE-Net, HyperNetworks, GeneCIS/CLIP4Cir, Deep VIB) in the report.
- **Axis priority for the ablation:** fusion family/topology first, buddy_dim second, adapter rank/width third, depth last — current `num_layers` knob is the least informative axis because it changes 3 MLP towers simultaneously.
- **buddy_dim guidance:** sweep `{8,16,32}`, not `{2,...}` by default — prior intrinsic study (`docs/reports/2026-06-09_buddies_dim_hparam_study.md`) found 2-D preserves only 6.7% of local kNN structure vs. 72% at 16-D; 2-D should only appear as a deliberately-lossy diagnostic on the eventual best family, not as a default sweep point for wide/SigLIP backbones.

See `task_plan.md` for the resulting phased ablation plan.

## 2026-09-02 — Phase 4 smoke sweep results

4-way comparison at fixed buddy_dim=16, K=30, alpha=0.5, `redcaps_150k`, 30 epochs, **1 seed per arm** (indicative only, not paper numbers — this is exactly the coarse smoke test the plan called for, meant to narrow to a winner/pair before the real Phase 5 sweep). `scripts/run_combiner_architecture_smoke.sh` + `scripts/analyze_combiner_architecture_smoke.py`.

One data-quality catch during analysis: an earlier ad-hoc `SMOKE=1` pipeline-sanity run (2 epochs) shared the same W&B tag/seed as the real 30-epoch `lowrank` run, and the analyzer's `.max()`-per-seed aggregation silently blended the two, distorting `lowrank`'s `test_pre_diff` numbers. Fixed by having the analyzer keep only the max-epoch run per (combiner_type, seed) — see script diff. All numbers below are post-fix.

| combiner_type | oracle t2i R1 | oracle i2t R1 | Δ oracle t2i vs legacy | Δ oracle i2t vs legacy | pre_diff t2i R1 | pre_diff i2t R1 |
|---|---:|---:|---:|---:|---:|---:|
| legacy | 27.50 | 22.80 | — | — | -1.60 | -3.60 |
| residual_control | 27.40 | 23.50 | -0.10 | +0.70 | -15.60 | -27.80 |
| lowrank (r=16) | 26.30 | 25.80 | -1.20 | **+3.00** | -5.00 | -7.20 |
| film | 26.80 | 25.50 | -0.70 | +2.70 | -19.60 | -28.80 |

(`film`'s number reflects the already-bounded `gamma_scale=0.1·tanh` gate — the unbounded-gamma bug found in Phase 3 code review was fixed before this sweep ran, so this result is not confounded by that.)

**Reading:** t2i oracle retrieval is roughly flat across all four (within single-seed noise). i2t oracle retrieval favors both new low-parameter families (`lowrank` +3.0, `film` +2.7) over `legacy`, with `residual_control` closer to flat (+0.7). `test_pre_diff` (a different, harsher comparison tier) is where the families sharply diverge: `residual_control` and `film` both regress heavily (-14 to -28 vs legacy's -1.6/-3.6), while `lowrank` regresses far less (-3.4 to -3.6 delta) — closest to `legacy` on this tier while still leading on oracle i2t.

**Not collected this pass:** residual norm `||delta||`, cosine shift from `x`, and gate/gamma/beta value ranges (the plan's Phase 4 spec asked for these as sanity diagnostics). None of the four combiner classes or the training loop currently log them; adding that instrumentation was out of scope for this smoke pass and would need a small logging addition before Phase 5 if we want it going forward.

**Leaning conclusion (single-seed, treat as directional):** `lowrank` is the standout — only new family with a clean win on oracle i2t R1 while staying closest to `legacy` on the harsher `pre_diff` tier. `residual_control` and `film` both show a large `pre_diff` regression that's large enough to look real even at n=1, not just noise — worth understanding before ruling them out entirely, since `residual_control` in particular is the fairest "does topology matter" control and a big regression there is a genuine, unexplained finding, not just "the new family is worse."

### Confirmed at 3 seeds (2026-09-02)

Added seeds 2,3 for all four arms (`SEED_SWEEP="2,3" bash scripts/run_combiner_architecture_smoke.sh`, 8 more runs, same operating point). The pattern held tightly — small spreads, large mean/SEM ratios (tens to hundreds), not single-seed noise:

| combiner_type | oracle t2i R1 Δ | oracle i2t R1 Δ | pre_diff t2i R1 Δ | pre_diff i2t R1 Δ |
|---|---:|---:|---:|---:|
| residual_control | +0.17 (n.s.) | +0.63 (n.s.) | **-13.57** (mean/SEM -62) | **-23.87** (mean/SEM -84) |
| lowrank (r=16) | -0.80 (mean/SEM -3.8) | **+3.10** (mean/SEM +15) | -3.60 (mean/SEM -31) | -3.63 (mean/SEM -11) |
| film | -0.33 (n.s.) | **+2.97** (mean/SEM +20) | **-17.63** (mean/SEM -95) | **-24.63** (mean/SEM -78) |

(all deltas vs. `legacy`; n.s. = not distinguishable from 0 at n=3)

**What `test_pre_diff` actually measures** (confirmed by reading `src/eval/pipeline.py:328-351`): it's the predictor-conditioned retrieval delta vs. the raw baseline — i.e. it uses `model.condition_predictor`'s *predicted* buddy vector (reconstructed from the CLIP embedding alone, no table/graph lookup), not the true oracle vector. This is exactly the deployment tier the brainstorm report flagged as a distinct concern from oracle retrieval (§ "Buddy dimensionality: guidance", predictor-compatibility note).

**Sharper conclusion:** `residual_control` and `film` are not simply "worse" — they're *fragile to an imperfect predicted condition*. Both look neutral-to-good under oracle conditioning (the true buddy vector) but collapse hard once the fusion has to work from `ConditionPredictor`'s reconstruction instead. `lowrank`'s contained rank-16 residual is far more robust to that same imperfection — it still regresses somewhat vs. `legacy` on `pre_diff`, but by roughly 1/5th to 1/7th as much as the full-width alternatives, while being the only family with a real, reproducible oracle-i2t gain. This is a clean empirical confirmation of the report's central thesis: restricting the fusion to a small subspace, rather than letting it touch the full frozen embedding, is what makes conditioning on a tiny buddy vector deployment-safe.

**Decision:** carry `lowrank` alone into Phase 5 (`buddy_dim` sweep). Drop `residual_control` and `film` from the main track — not because they're uninteresting, but because both show a specific, reproducible failure mode (oracle-vs-predicted fragility) that a buddy_dim or rank sweep wouldn't fix; that failure is a topology property, already isolated. Worth one sentence in the eventual paper-facing writeup as a negative result explaining why full-width fusion was rejected in favor of low-rank.

## 2026-09-02 — Phase 5 buddy_dim sweep results (lowrank, rank=16, 3 seeds each)

`scripts/run_combiner_buddydim_sweep.sh` + `scripts/analyze_combiner_buddydim_sweep.py`, same operating point (K=30, alpha=0.5, `redcaps_150k`, 30 epochs), `buddy_dim in {8,16,32}`, rank fixed at 16.

**Clean null result — buddy_dim doesn't move any of the four metrics** at this scale:

| buddy_dim | oracle t2i R1 | oracle i2t R1 | pre_diff t2i R1 | pre_diff i2t R1 |
|---:|---:|---:|---:|---:|
| 8 | 26.33 | 25.73 | -5.97 | -7.70 |
| 16 (anchor) | 26.33 | 25.87 | -5.37 | -7.30 |
| 32 | 26.10 | 25.60 | -5.00 | -8.00 |

All paired deltas from the dim=16 anchor are within ±0.7 and not distinguishable from 0 at n=3 (mean/SEM mostly under 2; the one exception, buddy_dim=8's t2i `pre_diff` delta showing an enormous mean/SEM ratio, is a `std≈0` display artifact from 3 nearly-identical deltas, not a real effect — the mean itself, -0.60, is tiny).

**Reading:** at this smoke scale, `buddy_dim` is not the lever — `lowrank`'s rank-16 adapter performs the same whether the buddy vector is 8, 16, or 32 dimensions, on both the oracle and predictor-conditioned (deployment) tiers. This matches the report's own axis-priority ranking (fusion family first, buddy_dim second-but-likely-marginal). Given no retrieval difference and buddy_dim's real per-sample storage cost (documented in the brainstorm report: ~32MB/1M examples at 8-D vs. 128MB at 32-D), and the existing graph-geometry evidence favoring 16-D over 8-D (72% vs. 63% local-kNN preservation, `2026-06-09_buddies_dim_hparam_study.md`), there's no reason from this sweep to move off the existing default (`embedding_dim=16`, already used across `configs/model/*.yaml`).

**Caveat:** this is one scale (150k), one epoch budget (30), one seed count (3) — a null here doesn't rule out buddy_dim mattering at the full training scale/duration used in the real experiments; it says the smoke-test resolution isn't the place this axis would show up, consistent with the report's guidance to prioritize fusion family over dimension.
