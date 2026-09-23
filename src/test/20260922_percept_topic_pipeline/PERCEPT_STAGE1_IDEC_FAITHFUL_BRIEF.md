# Brief: PercepT Stage 1 — faithful IDEC reformulation (literature-standard alternative)

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_idec_faithful_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: a structurally different fix, not a variant of the balance trick

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`,
`run_percept_stage1_balance_sweep_v2_pilot.py`, and
`percept_stage1_seed_stress_pilot_report.md` in full first.

This project's own balance-regularizer fix (add a term pulling the average
soft assignment toward uniform, scaled UP to compete with KL) reduced
collapse but the winning configuration turned out seed-fragile (1/4 seeds
cleared the held-out Pareto bar in a stress test). This pilot tests the
literature-standard alternative: Guo et al. 2017's Improved Deep Embedded
Clustering (IDEC), which fixes plain DEC's instability/collapse tendency by
a different mechanism entirely — keeping the reconstruction loss ACTIVE
throughout joint training (not just during pretraining) with NO down-
weighting, while instead scaling the CLUSTERING loss DOWN by a small
coefficient `gamma` (IDEC's paper uses `gamma=0.1`). This is the mirror
image of what this project tried (which scaled a NEW balance term up rather
than scaling the existing clustering term down), and has published,
independently-replicated evidence behind it for exactly this collapse
failure mode. Implement it as its own clean pilot — do not combine it with
the balance term; the point is to test IDEC's own mechanism in isolation.

## Implementation

Reuse `run_percept_stage1_pilot.py`'s fused embedding construction,
autoencoder architecture and pretraining (unchanged — 100 epochs,
reconstruction-only, `LAMBDA_RECONSTRUCTION` irrelevant here since Phase
pretrain has no clustering term regardless), K-means init (100 clusters,
`SEED=42`), Student's-t soft assignment and self-sharpened target
(unchanged), convergence criterion (`fraction_changed < 0.001`, ceiling
500), and 67-of-100 norm-threshold pruning (unchanged). Import and reuse
these exactly as `run_percept_stage1_balance_sweep_v2_pilot.py` already
does, via the same `load_module`/sibling-import pattern.

**The only change is the joint DEC loss formula.** Replace
`total_loss = kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss +
LAMBDA_BALANCE * balance_loss` (the balance-sweep pilot's formula) with
IDEC's own formula:

```python
GAMMA = 0.1  # IDEC's published default
total_loss = reconstruction_loss + GAMMA * kl_loss
```

No balance term at all — this is deliberately the literature's own fix, not
a hybrid with this project's balance trick. Log `reconstruction_loss`,
`kl_loss`, and `total_loss` at the same cadence the base script already
uses. Also log the same all-100-center cluster-size diagnostic
(min/max/median/below-1%-count) at the same checkpoints, including the
epoch-0 K-means-init snapshot — this pilot needs the same collapse
visibility as every DEC variant tried so far, since a structurally
different loss could plausibly fail in a different way (e.g. reconstruction
now dominating so heavily that clustering barely moves at all, producing
close-to-initial, uninformative assignments rather than a collapsed one —
watch for and explicitly discuss this alternate failure mode too, not just
collapse).

## Two-phase design (same discipline as the reconstruction-weight sweep)

**Phase 1 — single seed=42 run** with the IDEC loss formula above. Evaluate
train + held-out exactly as established (`evaluate_assignments`, `verdict`,
same held-out Pareto bar: emotion AMI > 0.1236 AND genre AMI > 0.1954).

**Phase 2 — if and only if Phase 1 clears the held-out Pareto bar**, stress-
test with the SAME 3 additional seeds already used elsewhere in this
investigation, `SEEDS = (7, 123, 2024)`, using the exact multi-seed
methodology from `run_percept_stage1_seed_stress_pilot.py` (fresh
`torch.manual_seed`/CUDA seed before a fresh autoencoder build, fresh
100-epoch pretrain per seed, `KMeans(random_state=seed)`, DEC with the IDEC
loss formula fixed). If Phase 1 does NOT clear the bar, skip Phase 2 and
report the Phase-1 result plainly as a miss — do not stress-test a
configuration that already failed its first test, that would not be a
useful use of GPU time.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_idec_faithful_pilot_report.md`
with:
- The training trajectory (reconstruction/KL/total loss, convergence epoch
  and reason) and the cluster-size diagnostic trajectory (same format as
  prior pilots in this directory).
- Phase 1 results table (seed 42, 2 splits): emotion AMI, genre AMI,
  verdict, held-out Pareto bar clearance, and explicit collapse-check numbers
  (min/max/median cluster size, fraction below 1%).
- If Phase 2 ran: the 4-seed results table and summary statistics in the
  same format as `percept_stage1_seed_stress_pilot_report.md`, plus an
  explicit comparison against that report's `LAMBDA_RECONSTRUCTION=1,
  LAMBDA_BALANCE=1000` result (mean/min/max held-out AMI, fraction of seeds
  clearing both bars) — is IDEC's mechanism more stable than this project's
  own balance-term fix?
- A final, honest verdict using this investigation's established language.
  If Phase 1 alone missed the bar, say plainly that IDEC's literature-
  standard fix did not outperform this project's own balance approach on
  this task at `gamma=0.1`, without further sweeping gamma unless explicitly
  asked — that would be new scope beyond what this brief covers.
