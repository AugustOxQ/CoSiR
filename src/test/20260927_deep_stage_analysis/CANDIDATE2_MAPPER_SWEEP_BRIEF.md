# Brief: candidate 2 — buddy Stage 2 mapper LR/capacity sweep

## Context

Candidate 1 (`candidate1_variant_c_stress_report.md`, this directory) raised
buddy's Stage 2 macro AUC from 0.5978 to a 4-seed mean of **0.6334**
(std 0.0026) via merging its 3 smallest topics (K=19→16) and inverse-
frequency loss reweighting ("Variant C"). This is now the adopted
configuration and the new baseline to improve further. Candidate 2 is a
mapper hyperparameter/capacity sweep — never tried for buddy's own Stage 2
mapper — mirroring PercepT's own successful LR sweep (which found 3e-3
materially beat the default 1e-3 in its own Stage 2 work,
`percept_stage2_sweep_pilot_report.md` Part C).

## What to build

New script `src/test/20260927_deep_stage_analysis/run_candidate2_mapper_sweep_pilot.py`.
Read `run_candidate1_min_occupancy_pilot.py` and
`run_candidate1_stress_pilot.py` (this directory) in full first — reuse
their functions (`merge_small_communities`, `train_and_eval_seeded`,
`load_module`, the snapshot-loading pattern, patch-feature loading) by
import rather than reimplementing. The exact "Variant C" configuration
(merge the 3 smallest train communities per `SMALL_COMMUNITIES = (16, 17,
18)`, then class-balanced per-sample loss weighting) is the fixed starting
point for every sweep point below — only the learning rate and/or mapper
capacity change.

Sweep at seed 42 first (screening), only stress-test (4 seeds:
42/7/123/2024, matching candidate 1's convention) whatever sweep point(s)
beat the current 0.6334 4-seed-mean baseline by the same +0.005 practical
margin used in candidate 1.

### Part A — learning rate sweep

Try `MAPPER_LEARNING_RATE` in `{3e-4, 1e-3 (current), 3e-3, 1e-2}`, same
100 epochs, same architecture (`AttentionPoolingMapper`, single learned
query, linear head), same Variant C targets/weighting. Report macro AUC
for each.

### Part B — epoch count

For whichever LR wins Part A, also try `MAPPER_EPOCHS` in
`{100 (current), 200, 400}` to check whether the mapper is under-trained
at the current fixed 100 epochs (full-batch training, so this is cheap).

### Part C — capacity (only if A/B show headroom; use judgement)

`AttentionPoolingMapper` currently uses one learned query for attention
pooling over 50 patch tokens, then a single linear head. If Parts A/B
show real, non-trivial gains (i.e. the mapper was clearly under-fit),
also try a minimal capacity increase: either (i) 2-4 learned queries
concatenated before the linear head, or (ii) a one-hidden-layer MLP head
instead of a single linear layer. Keep this minimal and well-justified in
the report — this project's own recipe favors simplicity, so only pursue
this if A/B give a concrete signal that capacity (not just optimization)
is the bottleneck. If A/B show no real headroom, skip Part C entirely and
say so plainly rather than trying capacity changes speculatively.

## Evaluation and output

For every configuration tested at seed 42: macro AUC, min/median/max
per-topic AUC, top-1 accuracy. For whichever configuration(s) beat the
0.6334 baseline by +0.005, run the full 4-seed stress and report mean/min/
max/std, exactly like candidate 1's reports did.

Output: `run_candidate2_mapper_sweep_pilot.py` and
`candidate2_mapper_sweep_pilot_report.md` (this directory) with a results
table for every sweep point and a plain verdict — does any configuration
meaningfully beat 0.6334? If yes, name it and its 4-seed-stressed number.
If no, say so plainly and close candidate 2 as tested; do not keep
expanding the sweep beyond what's specified above.

## Constraints

- Local GPU only, no DAS6 (this is a cheap Stage 2 mapper sweep, not a
  Stage 1 refit).
- Do not modify any existing file — read-only imports only, all new code
  in `src/test/20260927_deep_stage_analysis/`.
- Do not touch PercepT's side of anything.
