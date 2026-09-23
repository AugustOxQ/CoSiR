# Brief: PercepT Stage 1 — reconstruction-weight sweep

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_recon_weight_sweep_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: why reconstruction weight, not more balance tuning

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`,
`run_percept_stage1_balance_sweep_v2_pilot.py`,
`percept_stage1_balance_sweep_v2_pilot_report.md`, and
`percept_stage1_seed_stress_pilot_report.md` in full first.

`LAMBDA_BALANCE=1000` (with `LAMBDA_RECONSTRUCTION` left at its original 1.0)
cleared the held-out Pareto bar at seed 42, but a 4-seed stress test showed
this was a fragile, close-to-the-boundary result: mean held-out emotion AMI
across 4 seeds was 0.1235, essentially sitting on top of the 0.1236 bar, and
only 1/4 seeds cleared both thresholds simultaneously.

The likely root cause, diagnosed earlier in this investigation: the raw
reconstruction loss value is tiny (~0.00003) throughout DEC training, so even
at `LAMBDA_RECONSTRUCTION=1.0` its contribution to the total loss is
negligible next to KL (~0.007-0.09). The balance term compensates for
collapse, but nothing anchors the latent space to the actual input data with
real weight — DEC's self-sharpening dynamics are free to rearrange the space
almost entirely on their own terms, which is a plausible source of the
seed-to-seed instability (different random starts settle into different
self-consistent-but-arbitrary partitions, since nothing pulls them back
toward a shared, input-grounded structure).

This pilot tests whether substantially increasing `LAMBDA_RECONSTRUCTION` —
giving the autoencoder's own reconstruction objective a real say throughout
DEC, not just during pretraining — produces a more stable, not just a
higher-average, result. Keep `LAMBDA_BALANCE=1000` fixed (the best value
found so far) and vary only `LAMBDA_RECONSTRUCTION`.

## Two-phase design (mirror the balance sweep's already-validated approach)

**Phase 1 — broad sweep at seed 42 only** (fast, like the original balance
sweep): test `LAMBDA_RECONSTRUCTION ∈ {100, 300, 1000}` (raw reconstruction
values are ~0.00003, so these values bring its contribution to the total
loss into the same order of magnitude as KL's ~0.007-0.09 — this is the
same "make the terms numerically comparable" reasoning already used
successfully for the balance term). Share ONE pretrained autoencoder across
all 3 sweep points at this phase (same technique as
`run_percept_stage1_balance_sweep_v2_pilot.py`: pretrain once, deep-copy
state per sweep point, fresh K-means init per point at `SEED=42`, isolated
DEC run per point). Evaluate train + held-out for each point exactly as
established (same `evaluate_assignments`, `verdict`, Pareto bar).

**Phase 2 — seed-stress the single best Phase-1 point.** Whichever
`LAMBDA_RECONSTRUCTION` value in Phase 1 has the best held-out result
(prioritize: clears the Pareto bar AND has the largest margin over the
emotion bar, since that was the fragile axis last time), immediately run 3
NEW seeds on that exact configuration — reuse
`run_percept_stage1_seed_stress_pilot.py`'s exact multi-seed methodology
(fresh `torch.manual_seed`/CUDA seed before a fresh autoencoder build, fresh
100-epoch pretrain per seed, `KMeans(random_state=seed)`, DEC with the
winning `LAMBDA_RECONSTRUCTION` and `LAMBDA_BALANCE=1000` fixed). Use
`SEEDS = (7, 123, 2024)` — the exact same 3 seeds already used in the prior
stress test, so results are directly comparable point-for-point against
`percept_stage1_seed_stress_pilot_report.md`'s existing rows for
`LAMBDA_RECONSTRUCTION=1` (i.e. this becomes a like-for-like before/after
stability comparison at the SAME 4 seeds, not a different random sample).

Do not skip Phase 2 or make it optional — the entire point of this pilot is
whether reconstruction weighting fixes the seed-instability problem, which
Phase 1 alone (single-seed) cannot answer; Phase 1 only screens for which
value is worth stress-testing.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_recon_weight_sweep_pilot_report.md`
with:
- Phase 1 results table (3 `LAMBDA_RECONSTRUCTION` values x 2 splits): emotion
  AMI, genre AMI, verdict, held-out Pareto bar clearance.
- An explicit statement of which value was selected for Phase 2 stress-testing
  and why.
- Phase 2 results table (4 seeds x 2 splits, same format as
  `percept_stage1_seed_stress_pilot_report.md`, citing seed 42's Phase-1
  number as the first row rather than recomputing it), plus the same summary
  statistics (mean/min/max held-out emotion and genre AMI across the 4 seeds,
  fraction of seeds clearing each bar and both simultaneously).
- A direct, explicit before/after comparison against
  `percept_stage1_seed_stress_pilot_report.md`'s 4-seed results at
  `LAMBDA_RECONSTRUCTION=1` (mean, min, max, and the both-bars-clear seed
  count) — is the new configuration measurably MORE stable (tighter spread,
  more seeds clearing), not just possibly higher on average?
- A final, honest verdict using this investigation's established language:
  is this a "robust result" (comparable bar to the seed-stress report: how
  many of 4 seeds clear both bars) or still "seed-dependent"? Do not round up
  partial improvement to "solved."
