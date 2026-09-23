# Brief: PercepT Stage 1 — intermediate cluster-count sweep (find the crossover)

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_cluster_count_sweep_v2_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: a non-monotonic tradeoff, not a simple stability fix

Read `run_percept_stage1_cluster_count_sweep_pilot.py` and
`percept_stage1_cluster_count_sweep_pilot_report.md` in full first (both in
this directory).

The first cluster-count sweep tested `(N_INITIAL, N_SURVIVING) ∈ {(20,13),
(30,20), (50,33)}` at seed 42, then stress-tested the winner, K=30/20, across
4 seeds. Result: coarsening clusters genuinely fixed the SEED-INSTABILITY
problem (held-out genre AMI range shrank from 0.0617 to 0.0169, emotion
range from 0.0036 to 0.0016, and genre now clears its own bar in 4/4 seeds
reliably) — but the resulting stable point sits reliably just UNDER the
emotion bar (4/4 seeds land at 0.1204-0.1220, versus the 0.1236 threshold),
not over it. The single-seed screen also showed emotion AMI is NOT monotonic
in cluster count across the tested range: 0.1152 (K=20) -> 0.1220 (K=30,
peak) -> 0.1202 (K=50) -> ~0.1234 mean but noisy (K=100, original). This
pilot searches the gap between the stable-but-low K=30 point and the
noisy-but-higher-average K=100 point, to see whether some intermediate K
combines acceptable stability with an emotion AMI that actually clears the
bar.

## Two-phase design (same discipline as every prior sweep in this directory)

**Phase 1 — broad screen at seed 42.** Test `(N_INITIAL, N_SURVIVING) ∈
{(40, 27), (60, 40), (80, 53)}` (same ~67% retention ratio as every other
cluster-count pair tried). Keep `LAMBDA_BALANCE=1000` and
`LAMBDA_RECONSTRUCTION=1` fixed — only cluster count varies, exactly as in
the first cluster-count sweep. Reuse that script's structure directly
(shared pretrained autoencoder, per-point K-means and DEC, evaluation with
the correct per-point surviving-cluster denominator for the collapse
check) — it is the closest existing template, read it in full and follow
its exact conventions rather than re-deriving them.

**Selection for Phase 2:** prefer a point that clears the held-out Pareto
bar (emotion AMI > 0.1236 AND genre AMI > 0.1954) with the largest emotion
margin; if none clear, fall back to the largest emotion margin regardless
(same rule as the first cluster-count sweep).

**Phase 2 — seed-stress the selected point.** Same `SEEDS = (7, 123, 2024)`
used throughout this directory, same full-pipeline re-randomization
methodology (fresh seeding, fresh pretrain, `KMeans(random_state=seed)`, DEC
with the selected cluster counts and the unchanged loss weights).

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_cluster_count_sweep_v2_pilot_report.md`
with:
- Phase 1 results table for this sweep's 3 new points (same columns as the
  first cluster-count sweep's Phase 1 table).
- A single consolidated table showing the FULL cluster-count picture tested
  across both sweeps so far — cite (do not recompute) the prior sweep's
  seed-42 single-run results for K=20/13, K=30/20, K=50/33, and this
  investigation's original K=100/67 seed-42 single-run result (held-out
  emotion 0.1242, genre 0.2466, from `percept_stage1_balance_sweep_v2_pilot_report.md`
  at lambda=1000), alongside this sweep's new K=40/27, K=60/40, K=80/53
  points — one row per K value, held-out emotion AMI and genre AMI only, so
  the non-monotonic trend across the whole range is visible in one place.
- Which point was selected for Phase 2 and why.
- Phase 2 four-seed results table and summary statistics, in the same
  format as the first cluster-count sweep's report.
- An explicit three-way stability/performance comparison table: original
  K=100/67 (mean=0.1234 emotion, 0.2089 genre, 1/4 clear — cite from
  `percept_stage1_seed_stress_pilot_report.md`), K=30/20 (mean=0.1212
  emotion, 0.2616 genre, 0/4 clear — cite from the first cluster-count sweep
  report), and this pilot's new selected point.
- A final, honest verdict using this investigation's established language.
  If this sweep finds a point that both clears the bar in a majority or all
  of the 4 seeds AND keeps most of K=30's stability gain (spread
  meaningfully tighter than K=100's), say so plainly as the new standing
  PercepT Stage 1 configuration. If not, say so plainly and state whether
  the evidence suggests the emotion/genre tradeoff has no crossing point in
  the tested range, or whether a further-refined search might still find
  one.
