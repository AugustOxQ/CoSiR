# Brief: PercepT Stage 2 — consolidated stage report

Write `docs/reports/2026-09-23_artelingo_percept_stage2_report.md`. This is
a narrative markdown report, not code — write it directly, do not write or
run any script. Do not pause for confirmation, just write the file.

## Audience and purpose

This closes out PercepT Stage 2 (P-Topic Mapping) on branch
`experiment/percept_topic_pipeline`, following
`docs/reports/2026-09-23_artelingo_percept_stage1_report.md` (read in full
first — Stage 2 depends entirely on Stage 1's frozen K=60/40 seed-42
result: held-out emotion AMI=0.1238, genre AMI=0.2617, established there as
this investigation's standing, 4-seed-validated result). The reader has
seen Stage 1's report but not this one. Be concrete and number-precise
throughout — read every source table yourself before citing a number, do
not round or recall from memory.

## Required source reading (read every one of these in full before writing)

In `src/test/20260922_percept_topic_pipeline/`:
1. `PERCEPT_STAGE2_BRIEF.md` — the original design brief (assumed no patch
   features were available; superseded).
2. `PERCEPT_STAGE2_PATCH_BRIEF.md` — the corrected design, once genuine
   CLIP patch-token features were confirmed working in this environment
   (`src/model/cosirmodel.py`'s `encode_img()` computes both pooled and
   full-sequence embeddings; verified directly against real ArtELingo
   images: `[batch, 50, 512]`, no NaNs). Explains why patch attention
   pooling was used instead of the global-embedding fallback, and why this
   is not a downgrade — PercepT's own ablation found this simpler design
   (attention-pool patches -> linear -> sigmoid) beats a more complex
   cross-attention "topic query" mapper.
3. `run_percept_patch_feature_extraction.py` — the one-time patch-token
   extraction and caching step (train: 61,402 paintings, held-out: 9,365,
   both `[N, 50, 512]` float32).
4. `run_percept_stage2_pilot.py` + `percept_stage2_pilot_report.md` — the
   smoke test. Reproduced Stage 1's K=60/40 seed-42 numbers exactly (0.0000
   difference both AMIs), trained cleanly, held-out macro AUC 0.5690 versus
   a 0.5000 train-marginal-frequency baseline. Found that the original
   multi-label threshold (`q > 2.0/40`) never fired: 0% of paintings got a
   second label, so this ran as an effectively single-label classifier, not
   genuinely multi-label as intended.
5. `PERCEPT_STAGE2_SWEEP_BRIEF.md` + `run_percept_stage2_sweep_pilot.py` +
   `percept_stage2_sweep_pilot_report.md` — three-part sweep, ALL sharing
   one Stage-1 re-fit (reproduced exactly again):
   - Part A: 4 mapper-init seeds at the original threshold/lr — held-out
     macro AUC mean 0.5709, range 0.5644-0.5760, all beating baseline
     (seed-robust).
   - Part B: threshold comparison — `q > 1.2/40` was the first to produce
     genuine multi-label targets (76.89% of train paintings multi-labeled,
     mean 2.905 labels/painting, versus 0.00% at 2.0/40 and 0.02% at
     1.5/40), and its macro AUC jumped to 0.6894 at the default lr=1e-3.
   - Part C: learning-rate mini-sweep at the selected `q > 1.2/40`
     threshold — `lr=3e-3` reached macro AUC 0.8256 (seed 42 only at this
     point), a large jump the orchestrating session verified was genuine
     convergence, not instability, by reading the raw per-epoch loss log:
     `lr=3e-4` was still steeply underfit at epoch 100 (loss still falling
     from 0.638 to 0.341), while `lr=3e-3` converged smoothly and
     monotonically to 0.211 with no oscillation.
6. `PERCEPT_STAGE2_BEST_CONFIG_STRESS_BRIEF.md` +
   `run_percept_stage2_best_config_stress_pilot.py` +
   `percept_stage2_best_config_stress_pilot_report.md` — 4-seed stress test
   of the sweep's winning point (`q > 1.2/40`, `lr=3e-3`). **All four seeds
   landed in an extremely tight band: 0.8248-0.8272 (width 0.0025)**, each
   individually exceeding the original threshold's own 4-seed maximum
   (0.5760) — the improvement is robust and reproducible, not mapper-init
   luck.

## What the report should cover

1. **Context**: what PercepT Stage 2 (P-Topic Mapping) is — an image-only
   classifier trained to predict Stage 1's frozen topic assignments, no
   text at inference — and its dependency on Stage 1's K=60/40 result.
2. **The patch-feature availability correction**: state plainly that this
   session initially assumed no patch-level CLIP features were available
   (only global pooled embeddings, per `FeatureManager`'s cached schema),
   designed Stage 2 around that fallback, then the user corrected this
   from memory of the codebase, the orchestrating session verified
   `cosirmodel.py`'s dual pooled/full-sequence output actually works in
   this environment via a direct smoke test on real images, and the design
   was revised before any GPU training happened on the wrong assumption.
   This is a real, worthwhile correction to record, not a footnote.
3. **The full chronological arc** through all 6 source items above,
   including the negative/degenerate finding (original threshold never
   produced real multi-label targets) as plainly as the positive results —
   this investigation's established norm.
4. **A single results table** across every named configuration: threshold,
   learning rate, seed count (1 or 4), held-out macro AUC (single value or
   mean+range), verdict (seed-robust / single-seed-only / baseline).
5. **Why the winning configuration is trustworthy**: the reproducibility
   gate (Stage 1 re-fit matched exactly every time it was checked, 3
   separate times across the smoke test, sweep, and best-config stress
   test), the loss-trajectory sanity check for the large lr jump, and the
   4-seed robustness test's tight spread.
6. **Interpretation of the multi-label threshold finding**: `q > 1.2/40`
   makes most paintings multi-labeled (mean ~2.9 of 40 topics, up to 8) —
   state both readings plainly: this could reflect genuinely rich,
   overlapping topic structure that a stricter threshold was suppressing,
   or it could mean the threshold is now loose enough to be capturing weak
   affinity rather than real membership. Do not resolve this in the
   report's own voice as settled; say what evidence exists (the AUC
   improvement, the loss-convergence check) and what remains genuinely
   open (no qualitative inspection of which images/topics this threshold
   actually pairs together has been done).
7. **Limits inherited from Stage 1**: this report should not re-litigate
   Stage 1's own documented deviations (affect encoder substitution, fusion
   formula adaptation, DEC schedule) — reference Stage 1's report for those
   rather than repeating them, and focus this report's own limits section
   on Stage-2-specific open items (image-only inference not yet validated
   against the paper's own AUC numbers, since those come from a different
   dataset/topic count and were already flagged as non-comparable in the
   Stage 2 briefs; the attention-pooling architecture was not compared
   against a plain-linear-on-pooled-CLIP-embedding baseline, so it is not
   yet established that the patch-attention design specifically, rather
   than any image-only classifier, is responsible for the result).
8. **Standing result statement**: `q > 1.2/40`, `lr=3e-3`, 100 epochs is
   PercepT Stage 2's standing configuration on this branch — held-out macro
   AUC 0.8256 (4-seed mean, range 0.8248-0.8272), versus 0.5000 baseline.

Match the tone and rigor of
`docs/reports/2026-09-23_artelingo_percept_stage1_report.md` — evidence-
first, state both what worked and what didn't, no overselling.
