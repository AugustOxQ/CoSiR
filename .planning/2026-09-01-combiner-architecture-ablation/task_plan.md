# Task Plan: Combiner architecture ablation — fusion family, depth, buddy_dim

## Goal
The current fusion module (`Combiner_new`: two `GeLUNetGradual` MLP towers → concat → MLP decode → learned scalar gate) was never chosen by ablation or literature — combine mechanism, layer count, and buddy dimension are all ad hoc. Decide what to actually try, grounded in a literature-backed brainstorm, and run a focused, staged ablation (cheap identity/unit checks → small smoke sweep → real training sweep) rather than a blind grid search.

## Next Step
User decision (2026-09-01): hold Phase 3 entirely until Experiment 16.2's full 18-run sweep (300k + 500k legs) finishes — not just Phase 4. Reason: the sweep is a bash loop that launches a **fresh `python main_cosir.py` subprocess per K value**, from this same working directory (`/project/CoSiR`); the 500k leg hasn't started its subprocesses yet, so any source edit landing on disk before it starts would be picked up by those runs, silently running the 500k K-cells on different model code than the already-completed 300k K-cells and breaking the ablation's internal consistency. (The already-running process itself would be unaffected either way — Python doesn't re-read imported modules from disk — but the not-yet-launched subprocesses are the real exposure.) A monitor is watching `stage_b_run.log` for the completion marker or a failure signature.

## Current Phase
Complete (Phase 6 deferred by user decision). Delivered: `docs/reports/2026-09-02_combiner_architecture_ablation.md`.

## Background: brainstorm outcome

Dispatched to Codex (via `codeagent-wrapper --backend codex`, per project convention of routing research/implementation work through Codex) to read the actual `src/model/{cosirmodel,combiner,condition_predictor}.py` and `configs/model/*.yaml`, do a literature search, and produce a ranked shortlist of fusion-architecture candidates for this specific shape: a frozen CLIP/SigLIP embedding (512–1024D) fused with a tiny trainable per-sample "buddy" condition vector (2–16D, sourced from a cross-modal mutual-kNN graph — see `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §1–2), under an identity-init constraint the rest of the codebase follows religiously.

Full report: `docs/reports/2026-09-01_combiner_architecture_brainstorm.md`.

**Key finding (bug, not just a design gap):** the production `Combiner_new` is *not* identity-init at t=0 despite the project's own convention — `dynamic_scalar`'s last linear is zero-init (gate starts at s=0.5), but the two MLP towers feeding it are randomly initialized, so the initial output is `normalize(0.5*x + 0.5*delta_random)`, not `normalize(x)`. This confounds any prior read of "does conditioning help" with "does starting from a randomly-perturbed embedding help" — worth isolating before touching topology.

**Ranked shortlist** (full detail + literature refs in the report):
1. Low-rank residual dictionary / conditional adapter (`x + g·B·a(z)`) — first choice, cleanest identity-init and rank ablation, ~10–25K params.
2. FiLM residual modulation (`x + g·(γ(z)⊙x + β(z))`) — best simple baseline, ~35–70K params.
3. Content-aware low-rank (LoRA-style) adapter (`x + g·U·diag(a(z))·V^T·x`) — lets condition modulate a feature-dependent direction, ~20–40K params.
4. Zero-init highway/gated residual MLP — fairest apples-to-apples control against current topology, ~150–400K params.
5. SE-style channel gate + low-rank additive shift — most conservative, tests "selection vs. translation" separately.
6. Restricted hypernetwork over a shared low-rank adapter — second-wave expressiveness test.
7. Token cross-attention over `general_full` (already computed, currently unused) — only if a token-locality hypothesis is needed; most expensive, lowest priority.

**Axis priority** (why this ablation is ordered the way it is): (1) fusion family/residual topology > (2) buddy_dim > (3) adapter rank/width > (4) depth. The current 4–6 "num_layers" knob touches three towers at once and is the *least* informative first question.

## Phases

### Phase 1: Brainstorm & literature grounding
- [x] Read current architecture in full (`cosirmodel.py`, `combiner.py`, `condition_predictor.py`, `configs/model/*.yaml`)
- [x] Dispatch Codex for literature search + candidate generation, grounded in the actual code (not a generic dump)
- [x] Report written: `docs/reports/2026-09-01_combiner_architecture_brainstorm.md`
- **Status:** complete

### Phase 2: Convert shortlist into an executable plan
- [x] Pick the minimal, interpretable ablation order (below) from the report's recommendation
- [x] Write this task_plan.md
- **Status:** complete

### Phase 3: Implementation — identity-init control + 2 new combiner families
- [x] Fix/isolate the identity-init gap: added `CombinerResidualControl` — `y = normalize(x + F(x,z))`, one 128-wide block, zero-init final projection so `F(x,z)=0` at t=0. The fair control for "does topology matter" vs. "does starting near identity matter."
- [x] Added `CombinerLowRankAdapter` (candidate 1): rank 16 (default), `z -> a(z)` via 1-layer MLP, `y = normalize(x + B·a(z))`, zero-init `B` for exact identity (report's "zero-initialize B, leave a(z) nonzero" variant — no separate gate).
- [x] Added `CombinerFiLMResidual` (candidate 2): `z -> (γ,β)` via 2-linear MLP (hidden 64), `y = normalize(x + γ(z)⊙x + β(z))`, zero-init last γ/β projection, **γ additionally bounded to `0.1·tanh(γ_raw)`** (added in a follow-up fix after code review caught the first implementation shipped γ unbounded, contradicting the report's candidate-2 recommendation).
- [x] Wired a `combiner_type` config switch into `CoSiRModel.__init__` (default `"legacy"` = unchanged `Combiner_new`) selecting among `{legacy, residual_control, lowrank, film}`, threaded through `train_cosir.py` and `configs/model/clip_base.yaml`.
- [x] Unit tests: `src/test/20260902_combiner_architecture/test_identity_init.py` — asserts each new combiner's forward pass equals `F.normalize(x)` at init (atol=1e-5) across `feature_dim∈{512,768}`, `label_dim∈{2,16}`; asserts the zero-initialized terminal parameters get a nonzero gradient after one backward pass; smoke-tests `CoSiRModel` construction + `combine()` for both `combine_side` values under all three new types. Passing (re-verified independently by Claude, twice — after initial implementation and after the γ-bound fix).
- **Status:** complete
- **Execution mode:** Codex-backed subagent wrote the initial implementation (per project convention). First attempt stalled in its own internal "dual-model analysis" step and produced only the test skeleton before a session restart interrupted it; re-dispatched with an explicit instruction to implement directly against the existing test contract, which completed cleanly. The γ-bound fix found in code review was small and fully specified, so Claude applied it directly rather than re-delegating.

### Phase 4: Small-scale training smoke sweep (fusion family only, buddy_dim=16 fixed)
- [x] 4-way comparison at fixed `buddy_dim=16`, `combine_side=img`: `{legacy, residual_control, lowrank (r=16), film (γ_scale=0.1)}`, `redcaps_150k`, 30 epochs, K=30/alpha=0.5, 1 seed
- [x] `scripts/run_combiner_architecture_smoke.sh` (SMOKE=1-tested first, per project convention) + `scripts/analyze_combiner_architecture_smoke.py`
- [x] Report per family: oracle/pre_diff t2i+i2t R1 — see `findings.md` 2026-09-02 entry for the table and reading
- [ ] Residual norm / cosine shift / gate value ranges — **not collected**, no existing logging hook; would need a small instrumentation addition if wanted before Phase 5
- **Status:** complete — confirmed at 3 seeds (added seed=2,3 for all 4 arms, 8 more runs). `lowrank` wins: real oracle i2t R1 gain (+3.1, mean/SEM +15) with the smallest `pre_diff` (predictor-conditioned deployment tier, confirmed via `src/eval/pipeline.py:328-351`) regression. `residual_control`/`film` are neutral-to-good on oracle but collapse on `pre_diff` (-14 to -28, mean/SEM up to -95) — a reproducible oracle-vs-predicted-condition fragility, not noise. Decision: drop `residual_control`/`film` from the main track, carry `lowrank` alone into Phase 5. Full detail in `findings.md`.
- **Execution mode:** Claude wrote the sweep/analysis scripts directly (small, mechanical adaptations of the existing K-ablation scripts — not delegated to Codex, which had already stalled twice this session on smaller tasks) and launched/held the GPU runs itself, per project convention.

### Phase 5: buddy_dim sweep for the winner(s)
- [x] `buddy_dim in {8,16,32}` for `lowrank` (rank=16), 3 seeds each, same operating point as Phase 4
- [x] Reported both tiers: oracle and `ConditionPredictor`-predicted (`test_pre_diff`)
- **Status:** complete — clean null result, buddy_dim doesn't move any metric at this scale (all deltas from the dim=16 anchor within noise). No reason from this sweep to move off the existing `embedding_dim=16` default. Full table in `findings.md`.

### Phase 6: Rank/depth sweep (only if Phase 4/5 don't already answer capacity)
- **Status:** deferred by user decision (2026-09-02) after Phase 5's clean null — headline result (family matters, dimension doesn't) already in hand; revisit if a full-scale `lowrank` run looks capacity-limited.

### Phase 7: Delivery
- [x] Write findings report: `docs/reports/2026-09-02_combiner_architecture_ablation.md`
- [ ] Update `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` — **deliberately not done**: report's own recommendation is to validate `lowrank` at the real training scale/duration (300k/500k, 100 epochs) before treating it as a default-combiner decision; this smoke-scale result alone shouldn't move the publication plan's claims.
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Brainstorm delegated to Codex, grounded in the actual repo code (not asked to invent from scratch) | User's explicit instruction this session; Codex caught a real bug (current combiner not identity-init) by reading the code rather than trusting a paraphrase. |
| First ablation axis = fusion family/topology, not depth or buddy_dim | Report's structural argument: the current `num_layers` knob changes 3 towers at once (poor diagnostic), and the identity-init gap means "does conditioning help" has been confounded with "does a random perturbation at init help" — topology-with-exact-identity-init has to be isolated first. |
| `residual_control` (zero-init residual around the current topology) added as its own candidate, not just a footnote | Needed as the fair control: separates "identity init matters" from "the low-rank/FiLM topology itself matters," otherwise a win by candidate 1/2 could be wrongly attributed to topology alone. |
| Phase 4 (training) explicitly gated on the GPU being free of the running K-scaling sweep | Single local RTX 3090 (confirmed via `nvidia-smi`, currently ~3.3GB/24GB used by the buddy-k-scaling-stage-b process); this project's convention (see Experiment 16.2) is sequential single-GPU runs, not concurrent contention. |
| Hypernetwork and token cross-attention candidates deliberately deferred, not scheduled | Report's own recommendation: they're second-wave/failure-mode tests, not part of the minimal first sweep — running them speculatively would just be a wider blind grid search, which is exactly what this plan is trying to avoid. |
| Phase 3 (code implementation) held until Experiment 16.2's full sweep finishes, rather than started now in an isolated worktree | User's explicit call (2026-09-01) when offered both options. The K-scaling sweep launches a fresh `python main_cosir.py` subprocess per K from `/project/CoSiR` itself (bash loop, not one long-lived process); the 500k leg's subprocesses haven't started yet, so on-disk edits before then would land in a not-yet-launched run and break within-sweep code consistency between the 300k and 500k K-cells. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
