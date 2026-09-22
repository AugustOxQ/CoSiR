# Brief: loss-weight sweep on the Stage 1 learned-student architecture

Read `run_learned_student_stage1_pilot.py` in full first (the current,
addendum-updated version). Copy it to a new file
`src/test/20260923_artelingo_buddy_analysis/run_learned_student_weight_sweep_pilot.py`
and make ONLY the changes below. Do NOT run it — execution happens
separately, on GPU, outside this task.

## Context and why this run matters

Stage 1 (linear heads, equally weighted losses) is the best result in this
investigation (`learned_student_stage1_pilot_report.md`): train emotion
AMI=0.1284/genre AMI=0.2799 clear both Pareto targets, but held-out emotion
AMI=0.1095 misses the same bar (0.1236) by about 11%. Stage 2 (more
capacity, same equal weighting) made things worse
(`learned_student_stage2_pilot_report.md`), ruling out capacity as the
lever. This is the one remaining well-motivated, still-"simple" lever from
the stage report's open items: Stage 1's own checkpoint trajectory showed
content gradient share starting at 0.7356 (epoch 5) and settling toward
~0.50 by epoch 200 — the two losses are not naturally balanced early on
under equal weighting. Test whether weighting the CONTENT loss higher
finds a strictly better (or at least held-out-Pareto-clearing) compromise,
using the exact same architecture, data, and diagnostics as Stage 1 — only
the loss weight changes. This is a predeclared, evidence-motivated sweep,
not a search for whichever weight happens to score best after the fact.

## The only change: a predeclared content-loss weight sweep

```python
CONTENT_LOSS_WEIGHTS = (1.0, 1.5, 2.0, 3.0)
```

Change `total_loss = content_loss + affect_loss` to
`total_loss = content_weight * content_loss + affect_loss` (affect stays
at implicit weight 1.0 throughout; only content's weight varies). Wrap the
ENTIRE existing training loop (model construction, epoch-0 baseline,
training epochs, stopping/collapse determination, train and held-out final
evaluation) in an outer loop over `CONTENT_LOSS_WEIGHTS`, re-initializing a
fresh `LearnedStudent()` and fresh `Adam` optimizer for each weight (do not
carry over trained weights between sweep points), and re-seeding
`torch.manual_seed(SEED)` / `np.random.seed(SEED)` before each weight's run
so every sweep point starts from the same architecture-level random
initialization distribution (not literally the same weights, since
`content_weight=1.0`'s run should still exactly reproduce Stage 1's own
result as an in-sweep sanity check — confirm this reproduction explicitly
in the report, see below).

Keep every other constant, the model class, the training/loss/diagnostic
functions, and the stopping/collapse rule IDENTICAL to Stage 1 — copy them
verbatim.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/learned_student_weight_sweep_pilot_report.md`:

- One paragraph of context (why this sweep, referencing Stage 1's gradient-
  share trajectory as the motivation).
- **Sanity check, stated explicitly**: confirm the `content_weight=1.0` run
  reproduces Stage 1's exact train and held-out AMI numbers (0.1284/0.2799
  train, 0.1095/0.2901 held-out) — if it does not reproduce closely (say,
  within 0.005 AMI), state this plainly as a discrepancy needing
  investigation rather than silently reporting different numbers.
- A results table, one row per weight, both train and held-out AMI (8 rows
  total: 4 weights x train/held-out), plus each run's collapse verdict
  (Collapsed / Merely a compromise / Real success) and final gradient
  share, gate mean, and gate saturated fraction — reuse the same
  vocabulary and thresholds as Stage 1's own collapse determination.
- State plainly which weight (if any) is the best point found: does ANY
  weight clear the HELD-OUT Pareto bar (`emotion AMI > 0.1236 AND genre AMI
  > 0.1954`), not just the train bar (every prior method's Pareto bar was
  defined and checked against TRAIN numbers except Stage 1/2's own
  held-out checks — apply the identical bar to held-out here specifically,
  since that is the metric this sweep exists to try to clear).
- Report the full held-out AMI trend across the 4 weights (does emotion AMI
  rise monotonically as content weight increases, fall, or peak in the
  middle — describe the actual shape, do not assume monotonicity).
- A short, honest concluding paragraph: if the best weight beats Stage 1's
  held-out numbers on both axes simultaneously, say so plainly as the new
  best result in this investigation. If no weight clears the held-out
  Pareto bar, say so plainly and note whether this sweep at least narrowed
  the gap versus Stage 1's unweighted held-out emotion AMI shortfall, or
  made no material difference — do not overstate a small numeric wiggle as
  a meaningful finding.

Print clear timestamped progress logs matching the Stage 1 script's
`log()` format throughout, clearly labeling which sweep weight is running
at each stage (e.g. prefix every log line for a given weight's run with
`[weight=X.X]`).
