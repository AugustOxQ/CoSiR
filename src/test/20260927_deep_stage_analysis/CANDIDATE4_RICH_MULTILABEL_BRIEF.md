# Brief: candidate 4 — richer multi-label Stage 2 targets for buddy

## Context

Buddy's Stage 2 currently uses strict single-label targets (one train
community per painting, from a hard Leiden partition). PercepT's own
Stage 2 experiments found that a richer multi-label threshold (soft
assignment score above a cutoff, allowing multiple positive topics per
painting) changed its Stage 2 AUC substantially
(`percept_stage2_sweep_pilot_report.md` Part B/C). This candidate tests
whether an analogous richer multi-label target helps buddy's own Stage 2,
on top of the already-adopted candidate 1+2 configuration (merge 3
smallest topics to K=16, class-balanced loss, lr=1e-2, epochs=400 —
4-seed mean macro AUC 0.8461, see `candidate2_mapper_sweep_pilot_report.md`
and the master report's §6d).

## What to build

New script `src/test/20260927_deep_stage_analysis/run_candidate4_rich_multilabel_pilot.py`
in this same directory. Read `run_candidate1_min_occupancy_pilot.py`,
`run_candidate1_stress_pilot.py`, and `run_candidate2_mapper_sweep_pilot.py`
(this directory) in full first — reuse their functions
(`merge_small_communities`, `one_hot`, `load_module`, the snapshot-loading
pattern, patch-feature loading, the adopted lr=1e-2/epochs=400/class-
balanced-loss recipe) by import rather than reimplementing.

Buddy's held-out label transfer currently uses
`run_heldout_label_transfer_pilot.py::assign_to_train_communities`, which
returns a single hard label via k-NN majority vote. To get a genuine
multi-label target instead, you need each painting's SOFT assignment
distribution over the (merged, K=16) train communities, not just the hard
argmax. Two reasonable ways to derive this without inventing a new
mechanism from scratch (pick whichever is more faithful to what's already
in the codebase, and say clearly which you picked and why):

1. For TRAIN paintings: their own hard Leiden community label already
   exists; to get soft membership scores, use the same k-NN vote
   fractions `assign_to_train_communities` computes internally (read that
   function — it likely already computes a vote-fraction per neighbor
   community before taking the argmax; if so, expose/reuse that fraction
   rather than only the final hard label).
2. For HELD-OUT paintings: same k-NN vote-fraction mechanism, applied to
   held-out embeddings against the train vocabulary.

Threshold: mark topic `t` positive for a painting if its vote fraction for
`t` exceeds a cutoff. Sweep cutoffs at roughly `{0.5, 0.3, 0.15}` of the
maximum per-painting vote fraction (i.e. relative to that painting's own
top score, not an absolute constant), report the resulting mean/median/max
labels-per-painting and fraction multi-labeled for train and held-out at
each cutoff (matching the label-statistics table convention every Stage 2
report in this project already uses), and pick the cutoff that produces a
genuinely multi-label target set (similar in spirit to PercepT's own
Part B threshold comparison) without degenerating to "every painting gets
all 16 topics" or "no painting gets more than 1."

Train the same `AttentionPoolingMapper` architecture with `BCEWithLogitsLoss`
against these richer multi-hot targets (drop the single-sample class-
balanced weighting from candidate 1 for this experiment, since multi-hot
targets change what "class balance" even means — a straightforward
unweighted BCE against the multi-hot targets is the right baseline here;
note this explicitly in the report rather than silently changing it), at
the adopted lr=1e-2/epochs=400. Screen at seed 42 first.

## Evaluation

Report, for each threshold tested: label statistics (mean/median/max
labels, fraction multi-labeled, train and held-out), Stage 2 macro AUC,
min/median/max per-topic AUC, skipped-topic count. Compare against the
single-label baseline (0.8461, candidate 2's 4-seed mean). Only 4-seed
stress whichever threshold beats 0.8461 by the established +0.005
practical margin.

## Output

`run_candidate4_rich_multilabel_pilot.py` and
`candidate4_rich_multilabel_pilot_report.md` (this directory), with a
results table and a plain verdict: does any richer multi-label threshold
meaningfully beat the single-label 0.8461 baseline? Name the winning
threshold if so, or close candidate 4 as tested with a negative result if
not — do not keep expanding the threshold grid beyond what's specified.

## Constraints

- Local GPU only, no DAS6.
- Do not modify any existing file — read-only imports only, all new code
  in `src/test/20260927_deep_stage_analysis/`.
- Do not touch PercepT's side of anything.
- If deriving soft vote fractions turns out to require modifying
  `assign_to_train_communities` itself (rather than just reading an
  internal value it already computes), do NOT edit that shared file —
  instead write a small local standalone function in the new script that
  reimplements just the k-NN vote-fraction computation (the same k=20
  cosine-nearest-neighbor mechanism), and state plainly in the report that
  this was necessary and why.
