# Brief: PercepT Stage 1 — faithful paper recipe (noise injection + cosine LR schedule, native K=100/67)

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_faithful_recipe_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: testing whether the paper's own mechanisms replace our invented fix

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`,
`run_percept_stage1_balance_sweep_v2_pilot.py`, and
`docs/reports/2026-09-23_artelingo_percept_stage1_report.md` in full first.

This project's standing Stage 1 result (K=60/40, `LAMBDA_BALANCE=1000`) fixes
DEC collapse with a balance regularizer that is **not in the source paper**.
We have now independently verified (full-text read of
https://arxiv.org/html/2606.03345v1, findings saved at
`/tmp/claude-0/-project-CoSiR/a2df7b39-29a4-4d87-82ac-18432fa323d1/scratchpad/gemini_percept_paper_dec_findings.md`)
that the paper's actual recipe at its native K=100/67 differs from our base
pilot (`run_percept_stage1_pilot.py`) in exactly two mechanisms, neither of
which is noise-free-lunch speculation — both are read directly from the
paper's §4.1/Appendix E:

1. **Latent-space Gaussian noise injection during pretraining.** Not input
   corruption, not label smoothing. The equations (§4.1, Pretraining) are:
   `z = G_E(h)`, `z_hat = z + lambda_noise * N(0,1)`, `h_hat = G_D(z_hat)`,
   loss reconstructs the CLEAN `h` from the corrupted `z_hat`. `lambda_noise
   = 0.1` (std-dev 0.1 per latent component, i.e. variance 0.01 additive
   Gaussian noise on the 128-D latent). Confirmed for pretraining; the paper
   does not say whether it persists into joint DEC training — that is a real
   ambiguity, not implementation laziness, so this pilot must test both
   readings (see Phase 1 / Phase 1b below), not silently pick one.
2. **Learning rate schedule.** Both pretraining and joint DEC training use
   Adam, initial LR 1e-3, with cosine annealing down to a floor of 1e-5.
   Pretraining: 100 epochs (unchanged from our base pilot). Joint DEC
   training: the paper's own run is 200 epochs; Appendix E says joint
   training uses "the same Cosine Annealing scheduler" as pretraining, which
   is read as sharing the 1e-5 floor. Our base pilot instead uses a flat
   `DEC_LEARNING_RATE = 1e-4` with no schedule at all.

Everything else the paper specifies already matches our base pilot exactly:
`LAMBDA_RECONSTRUCTION = 1` held constant (Table 1's main row is literally
labelled "PercepT (λ_R=1)"; no schedule is described for it), 100 initial
K-means clusters pruned to 67 by cluster-center norm, encoder/decoder
architecture. **Do not change any of those.** Do not add or reuse this
project's own `LAMBDA_BALANCE` term anywhere in this pilot — the entire
point is to isolate whether the paper's own two mechanisms above are
sufficient to avoid collapse at the paper's own K=100/67, without our
invented fix.

## Implementation

Reuse `run_percept_stage1_pilot.py`'s fused embedding construction, encoder/
decoder architecture, K-means init (100 clusters, `SEED=42`), Student's-t
soft assignment and self-sharpened target, convergence criterion
(`fraction_changed < 0.001`, ceiling `MAX_DEC_EPOCHS = 500`), and 67-of-100
norm-threshold pruning — unchanged, via the same `load_module` sibling-import
pattern `run_percept_stage1_idec_faithful_pilot.py` already uses.

**Change 1 — pretraining loop:** after computing `z = encoder(batch)`, add
`z_hat = z + 0.1 * torch.randn_like(z)`, decode `z_hat` instead of `z`, and
compute reconstruction loss (MSE, matching Appendix E's stated objective;
note in a code comment that the displayed §4.1 equation uses an unsquared L2
norm instead — this is a genuine discrepancy in the paper, MSE is the
implementable convention and what Appendix E calls it) against the clean
input `h`. Optimizer: `Adam(lr=1e-3)` with
`torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,
eta_min=1e-5)`, stepped once per epoch, for all 100 pretraining epochs.

**Change 2 — DEC joint-training loop:** switch the optimizer to
`Adam(lr=1e-3)` with `CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)`,
stepped once per epoch. Keep the existing `fraction_changed`-based early-stop
check and the 500-epoch ceiling exactly as-is — if training stops before
epoch 200 the schedule simply never reaches its floor; if it runs past 200
(possible since the ceiling is 500), clamp the scheduler's step count at 200
so LR stays at the 1e-5 floor rather than erroring or cycling. Loss stays
`total_loss = kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss` with
`LAMBDA_RECONSTRUCTION = 1.0`, exactly as the base pilot — no balance term.

**Change 3 — cluster count:** use `N_INITIAL_CLUSTERS = 100`,
`N_SURVIVING_CLUSTERS = 67` (the base pilot's own defaults — i.e. do NOT
import the K=60/40 override from the balance-sweep pilots).

**Two noise-persistence variants, both run at seed 42 in the same script
(cheap — same architecture, same data, only the joint-phase noise term
differs):**
- **Variant A ("pretrain-only noise")**: the noise injection above applies
  only during the pretraining phase; joint DEC training uses the clean
  `z` (no noise added) when computing reconstruction loss. This is the most
  literal reading of where the paper's equations are written.
- **Variant B ("noise persists")**: identical to A, except the same
  `z_hat = z + 0.1 * randn_like(z)` corruption is also applied during joint
  DEC training's reconstruction term (the KL/clustering term still uses the
  clean `z`/`centers` distance as normal — only reweight what feeds
  reconstruction).

## Two-phase design (same discipline as prior pilots in this directory)

**Phase 1 — single seed=42 run of BOTH variants A and B.** Evaluate train +
held-out exactly as established (`evaluate_assignments`, `verdict`, held-out
Pareto bar: emotion AMI > 0.1236 AND genre AMI > 0.1954). Also compute and
report **silhouette score (SI)** on the 128-D latent `Z` for both the train
and held-out splits at the final epoch, using the same clustering assignment
used for the AMI evaluation — this is the paper's own headline metric
(Table 1 reports 0.97 for the full method vs 0.02 for fused-h-without-DEC on
this exact latent space) and has not been computed by any prior pilot in
this directory; report it even though our existing Pareto bars are AMI-based.
Report the full all-100-center cluster-size collapse diagnostic
(min/max/median/below-1%-count) at the same checkpoint cadence as other
pilots, for both variants.

**Phase 2 — if and only if a variant clears the held-out Pareto bar in
Phase 1**, stress-test that variant (run both, if both clear) with the same
3 additional seeds used elsewhere in this investigation, `SEEDS = (7, 123,
2024)`, using the exact multi-seed methodology from
`run_percept_stage1_seed_stress_pilot.py` (fresh `torch.manual_seed`/CUDA
seed before a fresh autoencoder build, fresh 100-epoch pretrain per seed,
`KMeans(random_state=seed)`, DEC with that variant's loss/noise/schedule
fixed). If neither variant clears the bar in Phase 1, skip Phase 2 entirely
and report both as plain misses — do not silently sweep `lambda_noise` or
the schedule's `T_max`/floor beyond what is specified above; that is new
scope outside this brief.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_pilot_report.md`
with:
- The training trajectory (reconstruction/KL/total loss, LR at each logged
  epoch, convergence epoch and reason) and the cluster-size diagnostic
  trajectory, for both variants.
- Phase 1 results table (seed 42, both variants, both splits): emotion AMI,
  genre AMI, silhouette score, verdict, held-out Pareto bar clearance, and
  explicit collapse-check numbers.
- If Phase 2 ran: the 4-seed results table and summary statistics in the
  same format as `percept_stage1_seed_stress_pilot_report.md`, plus an
  explicit comparison against this project's standing K=60/40
  `LAMBDA_BALANCE=1000` result (held-out emotion AMI mean 0.1252, genre AMI
  mean 0.2486, 4/4 seeds clearing both bars) — does the paper's own faithful
  recipe, at the paper's own K=100/67, match or beat that without any
  balance term?
- A final, honest verdict using this investigation's established language.
  If both variants miss the bar in Phase 1, say so plainly and do not
  reach for an untested explanation — leave that judgment to the calling
  session, which has the full cross-pilot context this script does not.
