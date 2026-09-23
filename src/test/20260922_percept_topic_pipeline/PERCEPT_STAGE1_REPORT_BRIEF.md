# Brief: PercepT Stage 1 — consolidated stage report

Write `docs/reports/2026-09-23_artelingo_percept_stage1_report.md`. This is a
narrative markdown report, not code — write it directly, do not write or
run any script. Do not pause for confirmation, just write the file.

## Audience and purpose

This closes out PercepT Stage 1 (P-Topic Formation) for the ArtELingo
buddy-graph investigation on branch `experiment/percept_topic_pipeline`. The
reader has NOT seen this branch's work before — they know the earlier
fusion-mechanism investigation (`docs/reports/2026-09-22_artelingo_fusion_mechanism_investigation.md`,
which found Attention-h1 as the standing best result: held-out emotion
AMI=0.1249, genre AMI=0.2404) but is now picking up this separate, later
PercepT-replication thread for the first time. Write so it is fully
self-contained: explain what PercepT is, why this branch exists, and walk
through the full arc chronologically. Be concrete and number-precise
throughout — this report will be checked against its source reports
number-by-number, so do not round, approximate, or paraphrase a metric from
memory; read every source file's actual table before citing a number from
it.

## Required source reading (read every one of these in full before writing)

In `src/test/20260922_percept_topic_pipeline/`:
1. `PERCEPT_STAGE1_BRIEF.md` — original literal-replication spec and the
   documented deviations (concat-not-sum fusion, convergence-controlled DEC,
   embedding-level GoEmotions-RoBERTa affect signal instead of label
   probabilities).
2. `percept_stage1_pilot_report.md` — the original, uncollapsed-loss-term
   run: collapsed (median cluster size 0, 65/67 below 1%).
3. `PERCEPT_STAGE1_STABILIZED_BRIEF.md` + `percept_stage1_stabilized_pilot_report.md`
   — first stabilization attempt (balance regularizer, lambda=1): still
   collapsed, but held-out emotion AMI nearly tripled (0.0363 -> 0.0925)
   versus the base pilot.
4. `PERCEPT_STAGE1_BALANCE_SWEEP_BRIEF.md` + `percept_stage1_balance_sweep_pilot_report.md`
   — lambda in {10,50,100,500}: all held-out collapsed, but lambda=500 was
   the best point and still improving at the top of the range.
5. `percept_stage1_balance_sweep_v2_pilot_report.md` — lambda in
   {1000,2500,5000}: ALL THREE cleared the held-out Pareto bar at seed 42 for
   the first time (lambda=1000: emotion 0.1242, genre 0.2466).
6. `PERCEPT_STAGE1_SEED_STRESS_BRIEF.md` + `percept_stage1_seed_stress_pilot_report.md`
   — 4-seed stress test at lambda=1000: only 1/4 seeds cleared both bars
   (mean held-out emotion 0.1235, sitting almost exactly on the 0.1236 bar)
   — a fragile, seed-dependent result, not a robust one.
7. `PERCEPT_STAGE1_RECON_WEIGHT_SWEEP_BRIEF.md` + `percept_stage1_recon_weight_sweep_pilot_report.md`
   — raising LAMBDA_RECONSTRUCTION (100/300/1000) to fix the fragility:
   FAILED, went to 0/4 seeds clearing, no more stable than before.
8. `PERCEPT_STAGE1_IDEC_FAITHFUL_BRIEF.md` + `percept_stage1_idec_faithful_pilot_report.md`
   — literature-standard IDEC reformulation (gamma=0.1, no balance term):
   FAILED, collapsed even harder than the original unregularized pilot
   (97/100 below 1%), and the diagnosis why: even gamma=0.1 leaves KL's raw
   magnitude ~10-100x larger than reconstruction's raw magnitude in this
   setup, so IDEC's published default did not transfer here.
9. `PERCEPT_STAGE1_CLUSTER_COUNT_SWEEP_BRIEF.md` + `percept_stage1_cluster_count_sweep_pilot_report.md`
   — first cluster-count sweep: K=30/20 fixed the SEED-STABILITY problem
   (spreads shrank sharply) but converged to a stable value just UNDER the
   emotion bar (0/4 clearing) — proof the mechanism was real but had not yet
   found the right operating point.
10. `PERCEPT_STAGE1_CLUSTER_COUNT_SWEEP_V2_BRIEF.md` + `percept_stage1_cluster_count_sweep_v2_pilot_report.md`
    — intermediate K search (40/27, 60/40, 80/53): **K=60/40 clears the
    held-out Pareto bar in 4/4 stress-test seeds** (mean emotion AMI=0.1252,
    min=0.1238; mean genre AMI=0.2486, min=0.2328), with spreads meaningfully
    tighter than the original K=100/67 (emotion spread 0.0034 vs 0.0036,
    genre spread 0.0289 vs 0.0617). This is the standing Stage 1 result.
11. The raw training log evidence for K=60/40's health across all 4 seeds
    (already reviewed by the orchestrating session, summarized here for you
    to state accurately): every seed converged fast and consistently (82-88
    epochs via the stability criterion, none hit the 500-epoch ceiling), the
    balance-loss term shrank toward ~0 by convergence in every seed
    (meaning the regularizer settled rather than fighting hard throughout),
    and fraction_changed decayed smoothly and monotonically in every run
    with no oscillation or late-training instability. This is qualitatively
    healthier and more uniform across seeds than the original K=100 runs.

## What the report should cover

1. **Context**: what PercepT/the Beyond Semantics paper is, why this branch
   exists (a literal-as-possible replication test), and the relationship to
   the earlier, separate fusion-mechanism investigation (different branch,
   different mechanism, same underlying ArtELingo dataset and Pareto bar for
   comparability).
2. **The predeclared held-out Pareto bar** stated once, clearly: emotion AMI
   > 0.1236 AND genre AMI > 0.1954, simultaneously, on held-out (val+test,
   zero train overlap) data — inherited from the fusion-mechanism
   investigation for direct comparability.
3. **The full chronological arc** through all 10 source items above,
   including the two fixes that FAILED (reconstruction reweighting, IDEC)
   and why they failed, not just the ones that worked — this investigation's
   established norm is to report negative results plainly, not silently
   drop them.
4. **A single cross-method summary table** (mirroring
   `docs/reports/2026-09-22_artelingo_fusion_mechanism_investigation.md`'s
   cross-method table format) covering every named configuration tried:
   base (collapsed), stabilized lambda=1, balance sweep lambda=500/1000/2500/5000,
   seed-stress lambda=1000 (mean), recon-weight fix (mean), IDEC, cluster
   K=20/30/40/50/60/80/100 (seed-42 single-run for the ones only screened
   once, 4-seed mean for K=30, K=60, and K=100 where stress-tested) — held-out
   emotion AMI, held-out genre AMI, collapse/stability verdict.
5. **Why K=60/40 is trustworthy, not just numerically lucky**: state the
   4-seed robustness explicitly (this is the ONLY configuration in this
   entire Stage-1 effort that clears the bar in 4/4 seeds, not 1/4), AND the
   qualitative trajectory-health evidence from source #11 above.
6. **Comparison to Attention-h1** (the other investigation's standing
   result): both axes are close (K=60/40 held-out: emotion 0.1252 vs
   Attention-h1's 0.1249; genre 0.2486 vs 0.2404) but K=60/40 has something
   Attention-h1 has never been tested for — 4-seed robustness. State this
   difference plainly as a genuine advantage of this result's evidentiary
   strength, not just a tie.
7. **Open items / honest limitations**: this used the embedding-level
   GoEmotions-RoBERTa affect signal (a deviation from the paper's literal
   ModernBERT choice); fusion is concatenation-based, not the paper's
   elementwise sum (documented necessary deviation due to CLIP dimension
   mismatch); DEC training uses a convergence-controlled schedule, not the
   paper's fixed 200 epochs; no comparison has yet been made against a
   literal, un-adapted DEC/IDEC baseline at K=60 scale (the IDEC test was
   only run at K=100).
8. **What Stage 2 will consume**: state that Stage 2 (P-Topic Mapping) will
   freeze K=60/40's cluster assignments (fit on the FULL train split at a
   single representative seed, to be specified in the Stage 2 brief) as
   fixed multi-label pseudo-targets, and train an image-only classifier to
   predict them — do not go into Stage 2's own results here, this report is
   Stage 1 only.

Match the tone and rigor of
`docs/reports/2026-09-22_artelingo_fusion_mechanism_investigation.md` —
evidence-first, plain statement of both successes and failures, no
overselling a thin margin as decisive (even though K=60/40 IS the strongest
result in either branch by the robustness criterion, still state the actual
margins plainly, e.g. the 4-seed minimum emotion AMI 0.1238 clears the bar
0.1236 by only 0.0002 — a thin margin even though it is now a CONSISTENT
one).
