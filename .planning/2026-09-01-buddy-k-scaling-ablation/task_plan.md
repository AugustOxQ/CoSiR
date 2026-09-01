# Task Plan: Experiment 16 — Buddy K ablation & K(N) scaling law

## Goal
Determine (1) how varying mutual-KNN K changes strict-buddy vs. union-buddy graph composition and downstream training result at a fixed RedCaps scale (150k/300k/500k, freeze vs. trainable), and (2) whether K should scale with dataset size N, per `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 16.

## Next Step
Experiment 16 (Stages A and B) delivered. Headline: 500k/K=39 (16.1's K(N) prediction) moves i2t R1 by +0.97, mean/SEM=+14.5 vs. K=30 — no further action planned unless the user wants the 500k/K=50 cell completed to 3 seeds, or a structural cross-reference against 16.1's stats.

## Current Phase
Phase 5 (complete)

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent (K ablation, strict vs. union buddy, freeze vs. trainable held constant, K(N) scaling question)
- [x] Identify constraints (staged: cheap diagnostic gates expensive training, per project's Exp 11/15 discipline)
- [x] Document in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Design reviewed and approved via superpowers:brainstorming (bounded path)
- [x] Written into publication plan spec as Experiment 16 (16.1 Stage A, 16.2 Stage B)
- **Status:** complete

### Phase 3: Implementation — 16.1 Stage A (graph diagnostic, no training)
- [x] Codex-backed subagent implements sweep driver: `mutual_knn`/`union_graph`/`ensure_min_degree`/`classify_edges` over (N, K) grid, dumps 18-row table
- [x] Reuse C1a's `subreddit_lift`/`subreddit_enrichment_zscore` machinery, not reimplemented
- [x] Run sweep (150k/300k/500k × K∈{10,20,30,50,75,100})
- [x] Derive candidate K(N) scaling rule (hold avg strict-buddy degree constant, solve for K)
- [x] Short-list 2–3 K values per size for Stage B
- **Status:** complete

### Phase 4: Implementation — 16.2 Stage B (training ablation, gated on 16.1)
- [x] User confirmed 16.1 shortlists; scoped final 16.2 design with user: trained-arm only (freeze/trainable dropped), C6's 300k K=30 reuse dropped (plain-store mismatch), 18 new runs total
- [x] Codex-backed subagent writes `scripts/run_buddy_k_ablation.sh` (K as bash-loop template axis, `_diverse` stores) + 300k/500k wrappers + `scripts/analyze_buddy_k_ablation.py`; Claude reviewed the diff and confirmed correctness (correct paths, correct template-key reasoning)
- [x] SMOKE=1 pipeline sanity check on both wrappers (2 epochs, seed=1, K=30) — both exit 0, correct `_diverse` results_dir/sample counts confirmed
- [x] Launched full 18-run sweep locally (300k `{30,35,50}`×3 seeds, then 500k `{30,39,50}`×3 seeds, sequential on single GPU) — log at `.planning/2026-09-01-buddy-k-scaling-ablation/stage_b_run.log`
- [x] User decision (2026-09-02): stop after 16 runs instead of the full 18 — "we have enough on that one". Watcher killed the sweep the instant the 16th run's `Training Complete!` landed, catching it before Hydra's multirun started seed=2 of 500k/K=50 (only got as far as re-downloading the CLIP model; no results_dir, near-empty wandb run that `analyze_buddy_k_ablation.py`'s finished-state filter already excludes). Final actual scope: 300k K∈{30,35,50} full 3 seeds (9 runs), 500k K∈{30,39} full 3 seeds (6 runs), 500k K=50 **single seed only** (1 run) — 16 total.
- [x] Collect `test_oracle`/`test_pre_diff` t2i/i2t R1 per (N,K) cell (`scripts/analyze_buddy_k_ablation.py`). Headline: **500k/K=39 (the K(N) prediction) moves i2t R1 by +0.97, mean/SEM=+14.5** vs K=30 — the clearest signal in the sweep, past both the significance bar and the noise floor. 300k shows a smaller borderline t2i-only signal (K=35 +0.20 mean/SEM=+2.0, K=50 +0.33 mean/SEM=+2.8); 500k/t2i and 300k/i2t are flat/noise. 500k/K=50 (n=1) not usable as evidence.
- **Status:** complete

### Phase 5: Delivery
- [x] Write findings up as a report (`docs/reports/2026-09-02_buddy_k_ablation_stage_b.md`) + update Experiment 16.2 Result in publication plan spec
- [x] Update findings.md/task_plan.md with final results
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Staged (Stage A cheap diagnostic gates Stage B expensive training) over full grid up front | User confirmed via AskUserQuestion; matches this project's Exp 11/15 discipline of gating expensive ablations on cheap diagnostics. |
| Fixed absolute K grid `{10,20,30,50,75,100}` at every N (not a scaled grid) | Only way to directly read N's effect at fixed K without confounding it with a shifting grid — needed to answer the K(N) scaling question honestly. |
| Structural invariant for the scaling-law candidate: average strict-buddy degree | Directly targets the user's stated concern (strict buddy is the real signal but sparse; union is noisier compensation). |
| Codex (via CCG skill, codex backend) writes 16.1's sweep-driver code as a subagent | User's explicit instruction this session; matches project convention of routing implementation work through Codex where it fits. |
| Claude (not Codex) launches/holds any GPU training runs in 16.2 | Per prior-session feedback: Codex session is unstable for long-running GPU work; Codex writes code, host launches/monitors training. |
| 16.2 drops the freeze-vs-trainable crossing — trained arm only | User's explicit call (2026-09-01): keep 16.2 focused on whether K matters for the training result; the freeze-vs-trainable-across-K question is dropped from scope, not deferred. |
| 16.2 does NOT reuse C6's 300k K=30 trained cell | Checked during scoping: C6 was run via `run_init_ablation_redcaps_300k.sh`, which points at the plain `redcaps_300k` store (15/28 subreddit coverage), not `redcaps_300k_diverse` — the same store-variant bug 16.1 caught, recurring at the reuse layer. All 300k Stage-B cells are new runs on `_diverse`. |
| Run all 18 Stage-B training runs locally (single RTX 3090), sequentially, not on the user's SLURM cluster | User explored cluster automation (1 node, 3×A6000) but found no easy way to automate/semi-automate it this session and decided to drop it and run locally instead (2026-09-01). Both new launcher scripts were SMOKE-tested locally first (300k and 500k wrappers, 2 epochs each, exit 0) before committing to the full sequential sweep. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
