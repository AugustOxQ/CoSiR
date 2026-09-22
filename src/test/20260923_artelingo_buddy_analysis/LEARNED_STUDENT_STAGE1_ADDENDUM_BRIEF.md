# Brief: add epoch-0 baseline and held-out AMI check to the Stage 1 pilot

Edit the existing script
`src/test/20260923_artelingo_buddy_analysis/run_learned_student_stage1_pilot.py`
in place. Do NOT run it — execution happens separately, on GPU, outside this
task. Do not change the architecture, loss, training loop, batch sampling,
or any predeclared constant — this is an additive patch only.

## Why this matters

The just-completed run cleared both predeclared Pareto targets on the TRAIN
split (emotion AMI=0.1284 > 0.1236, genre AMI=0.2799 > 0.1954) — the first
method in this investigation to do so — but the automatic collapse
determination called it "Collapsed" because held-out content recall fell
from 0.1812 (epoch 5, the FIRST checkpoint) to 0.1314 (epoch 200, final).
Two things are missing before this can be trusted either way:

1. The "did this teacher develop" check compares against the epoch-5
   checkpoint, not a true pre-training (random-init, zero gradient steps)
   baseline. Content recall may have started unusually high purely from
   random initialization (a random linear projection of a 50-d PCA space
   that already has real structure can preserve some neighborhood
   information by chance), in which case a decline toward a lower but still
   real, balanced steady state is expected two-teacher rebalancing, not
   collapse. Or epoch 5 may reflect genuine rapid early learning that then
   eroded, which would be more concerning. An epoch-0 checkpoint
   distinguishes these.
2. The Pareto-clearing AMI numbers are computed on the TRAIN split only.
   This investigation already found once (the BERT ceiling pilot) that
   train-split numbers can be substantially inflated relative to genuinely
   held-out performance. The same check has not yet been done here.

## Change 1: epoch-0 baseline checkpoint

Immediately after constructing `model = LearnedStudent().to(device)` and
before the training loop starts, call `evaluate_checkpoint(...)` once with
the freshly initialized (untrained) model, using the exact same arguments
the in-loop checkpoint calls use. Record this as a checkpoint dict with
`"epoch": 0`, `"content_loss": float("nan")`, `"affect_loss": float("nan")`,
`"content_gradient_share": float("nan")` (no gradient exists yet), and
insert it as `trajectory[0]` (i.e., initialize `trajectory = [epoch_0_checkpoint]`
before the loop, rather than starting from an empty list). Log this
checkpoint the same way in-loop checkpoints are logged.

Update every place in the script that currently uses `trajectory[0]` as the
"first checkpoint" baseline for relative-improvement and collapse
determination (the plateau-detection logic's `if len(trajectory) > 1`
branch, and the final `content_developed`/`affect_developed` computation in
`main()`) — these should still reference `trajectory[0]`, which now
correctly means "epoch 0, before any training" instead of "epoch 5, after
one checkpoint interval of training." Verify the plateau-detection loop
still behaves correctly with this extra leading entry (it compares
consecutive trajectory entries pairwise, so an extra entry at the front is
fine and requires no other logic change — confirm this rather than assuming
it).

Keep the existing epoch-5..200 checkpoints and their reporting exactly as
they are; only the definition of "first checkpoint" used for the
development/collapse determination changes, and the epoch-0 row is added to
the reported trajectory table.

## Change 2: held-out AMI evaluation at the final checkpoint

Currently `main()` discards the held-out majority-vote emotion labels:
```python
heldout_majority_emotion = [heldout_pipeline.majority(counts) for counts in heldout_emotion_counts]
del heldout_majority_emotion  # Labels are loaded for split parity; CCA itself is label-free.
```
Wait — check the actual current code before assuming this exact line exists;
read the file first. The held-out emotion counts may currently be loaded
under a different variable name (e.g. `_heldout_counts`) and never
converted to majority labels at all. Whichever is the case, RESTORE/ADD the
held-out majority-emotion label computation and USE it as described below;
do not leave it discarded.

After the existing TRAIN-split final evaluation (the `final_graph` /
`communities` / `emotion_metrics` / `genre_metrics` block), ADD a parallel
held-out evaluation:

1. Run the final trained model (already in `.eval()` mode from the last
   checkpoint call, but call it again explicitly for clarity) on
   `heldout_content`/`heldout_affect` to get final held-out student
   embeddings.
2. Build a held-out student graph via `single_modality.
   build_single_modality_graph("final-heldout-learned-student", ...)`, same
   convention as every other held-out graph in this script.
3. Run `detect_communities(seed=42)` on it.
4. Compute held-out emotion AMI/V-measure against the held-out majority-vote
   emotion labels (`heldout_majority_emotion`, from Change 2's restored
   computation — one label per held-out painting, same convention as train).
5. Compute held-out genre AMI/V-measure the same way the train evaluation
   does — reuse `pipeline.load_genre_map()` (the TRAIN pipeline module, not
   `heldout_pipeline`; this matches the established precedent in
   `run_bert_heldout_pilot.py`, since the genre-labelled diagnostic subset
   file is not split-specific) to find which held-out paintings overlap the
   genre-labelled set, exactly mirroring the existing train genre-evaluation
   block's structure.

## Report changes

In `write_report()`:
- Add the epoch-0 row to the checkpoint trajectory table (it will show
  `NaN` for the loss/gradient-share columns — render `NaN` as `"—"` in the
  table for those three columns specifically, not the literal string
  "nan").
- In the "Stopping and collapse determination" section, make clear that
  "first" now means epoch 0 (pre-training), not epoch 5, and restate the
  four collapse-check numbers using this corrected baseline.
- Add a new "## Held-out generalization" section, structured exactly like
  the existing "## Final comparison" table but for held-out: one row per
  reference method is NOT needed here (the existing references are all
  train-split numbers, not held-out, so a direct table merge would be
  misleading) — instead, present a small two-row comparison: "Learned
  student — TRAIN" (reuse the already-computed train AMI/V-measure numbers)
  and "Learned student — HELD-OUT" (the new numbers from this addendum),
  plus a plainly stated generalization readout: the absolute point drop in
  emotion AMI and genre AMI from train to held-out, and an explicit
  statement of whether the held-out numbers ALSO clear the Pareto bar
  (`emotion_AMI > 0.1236 and genre_AMI > 0.1954`) — using the same bar, since
  it was defined in terms of other methods' train-split numbers throughout
  this investigation, so comparing this run's held-out number against that
  same bar is a conservative (harder) test, not an apples-to-oranges one.
  State explicitly: if held-out fails to clear the bar while train did, this
  is evidence of the same train/held-out generalization gap seen in the BERT
  pilot, not a new failure mode.

Print clear timestamped progress logs matching the rest of the script's
`log()` format for both new pieces of work (epoch-0 checkpoint, held-out
final evaluation).
