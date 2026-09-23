# Brief: PercepT Stage 1 — fewer initial/surviving clusters

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_cluster_count_sweep_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: why cluster count, after two failed stabilization attempts

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`,
`run_percept_stage1_balance_sweep_v2_pilot.py`,
`run_percept_stage1_recon_weight_sweep_pilot.py`,
`percept_stage1_recon_weight_sweep_pilot_report.md`, and
`percept_stage1_idec_faithful_pilot_report.md` in full first.

Two independent, well-motivated stabilization attempts (raising
`LAMBDA_RECONSTRUCTION`, and a faithful IDEC reformulation) both failed to
improve on the original `LAMBDA_BALANCE=1000` result's seed stability
(1/4 seeds clearing the held-out Pareto bar) — reconstruction reweighting
scored 0/4, and IDEC collapsed outright. This pilot tests a structurally
different lever: `N_INITIAL_CLUSTERS=100` (67 surviving) puts the average
cluster size (614 nodes, at N=61,402) almost exactly at the 1%-of-N collapse
threshold (614) — meaning even a well-balanced K=100 clustering sits right
at that boundary, and small per-seed differences in which side of a cluster
boundary a point falls on can easily flip a borderline point's assignment.
Coarser clusters (fewer of them, each larger) should be less sensitive to
this kind of per-seed boundary noise, independent of the loss-formula
questions already tested.

## Two-phase design (same discipline as prior sweeps)

**Phase 1 — broad screen at seed 42.** Test three initial/surviving cluster
count pairs, keeping the paper's own ~67% retention ratio: `(N_INITIAL=20,
N_SURVIVING=13)`, `(N_INITIAL=30, N_SURVIVING=20)`, `(N_INITIAL=50,
N_SURVIVING=33)`. Keep `LAMBDA_BALANCE=1000` and `LAMBDA_RECONSTRUCTION=1`
(the original, unmodified value — both prior pilots showed raising it did
not help, so do not reintroduce that as a second variable here). Share ONE
pretrained autoencoder across all 3 points (same technique as
`run_percept_stage1_balance_sweep_v2_pilot.py`: pretrain once, deep-copy
state per point, fresh K-means init per point at `SEED=42`, isolated DEC run
per point — note `K-means(n_clusters=N_INITIAL, ...)` now varies per sweep
point, unlike the earlier sweeps where cluster count was fixed). The
1%-of-N collapse threshold in `evaluate_assignments`/collapse detection must
be computed against whatever `N_SURVIVING` is active for that sweep point
(not hardcoded to 67) — check this carefully, since it is the exact
mechanism under test.

**Phase 2 — seed-stress the single best Phase-1 point.** Same selection
rule as the reconstruction-weight sweep: prefer a point that clears the
held-out Pareto bar with the largest emotion-bar margin; if none clear,
prefer the largest emotion-bar margin regardless. Run `SEEDS = (7, 123,
2024)` — the same 3 seeds used in every other stress test in this
directory, for direct comparability — using the same full-pipeline
re-randomization methodology (fresh `torch.manual_seed`/CUDA seed, fresh
100-epoch pretrain, `KMeans(random_state=seed)`, DEC at the selected cluster
counts with `LAMBDA_BALANCE=1000`, `LAMBDA_RECONSTRUCTION=1`).

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_cluster_count_sweep_pilot_report.md`
with:
- Phase 1 results table (3 cluster-count pairs x 2 splits): N_INITIAL,
  N_SURVIVING, emotion AMI, genre AMI, verdict, held-out Pareto bar
  clearance, final max cluster size, fraction below 1% (with the correct
  per-point N_SURVIVING denominator).
- Which point was selected for Phase 2 and why.
- Phase 2 four-seed results table and summary statistics (mean/min/max
  held-out emotion and genre AMI, fraction of seeds clearing each bar and
  both simultaneously) in the same format as
  `percept_stage1_seed_stress_pilot_report.md`.
- An explicit before/after comparison against the ORIGINAL K=100/67 seed
  stress result (`percept_stage1_seed_stress_pilot_report.md`: mean=0.1235
  emotion / mean=0.2089 genre, 1/4 both-clear) using the same seed set, so
  it is a like-for-like stability comparison, not just an average-quality
  comparison.
- A final, honest verdict using this investigation's established language.
  If this also fails to reach 4/4 (or a clear improvement over 1/4), say so
  plainly rather than framing a marginal change as a fix.
