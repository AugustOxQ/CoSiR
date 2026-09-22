# Brief: Stage 2 — small MLP projection heads (controlled capacity increase)

Read `run_learned_student_stage1_pilot.py` in full first (the CURRENT
version, already includes the epoch-0-baseline and held-out-evaluation
addendum — read the actual file, do not assume its earlier form from any
other document). Copy it to a new file
`src/test/20260923_artelingo_buddy_analysis/run_learned_student_stage2_pilot.py`
and make ONLY the changes below. Do NOT run it — execution happens
separately, on GPU, outside this task.

## Context and why this run matters

Stage 1's linear-projection-heads student (`learned_student_stage1_pilot_report.md`)
was the first method in this investigation to clear both predeclared Pareto
targets on the train split (emotion AMI=0.1284 > 0.1236, genre AMI=0.2799 >
0.1954), and held up reasonably on held-out (genre AMI even improved
slightly to 0.2901; emotion AMI dropped to 0.1095, missing the bar by
~11%). Its trajectory showed a genuine two-teacher trade-off (content
recall declining from its free epoch-0 baseline of 0.1690 to 0.1314 while
affect recall grew substantially, 0.0325 to 0.1152) with healthy balance
metrics throughout (gradient share stayed near 50/50, gate saturation
stayed at 0%) — this is a promising, non-collapsed compromise that may be
capacity-limited by its purely linear projection heads, which is exactly
the condition both brainstorms said licenses trying more capacity next.

**This is a controlled comparison. Change ONLY the projection head
capacity — nothing else** (same PCA dimensionality, same shared embedding
dimension, same loss, same training data/sampling, same learning rate,
epoch budget, checkpoint cadence, diagnostics, and Pareto bar). This isolates
one variable, matching this investigation's established practice.

## The only change: MLP projection heads

Replace `LearnedStudent.__init__`'s two `nn.Linear` projection heads with
small 2-layer MLPs:

```python
HIDDEN_DIM = 64

self.proj_content = nn.Sequential(
    nn.Linear(CONTENT_PCA_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, D_SHARED)
)
self.proj_affect = nn.Sequential(
    nn.Linear(28, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, D_SHARED)
)
```

`D_SHARED` (32), the gate network, `forward()`, and every training/loss/
diagnostic/stopping/collapse function are otherwise UNCHANGED — copy them
verbatim from the Stage 1 script.

## Required renames and additions (do not skip these)

- All `REPORT_PATH`, log messages, docstrings, and report text referring to
  "Stage 1" become "Stage 2", and mentions of "linear" projection heads
  become "small 2-layer MLP (64-unit hidden layer)" projection heads.
  `REPORT_PATH` should point to
  `learned_student_stage2_pilot_report.md`.
- In `write_report()`'s "## Final comparison" and "## Held-out
  generalization" tables, ADD Stage 1's own results as an additional
  reference row (hardcode, do not recompute): train emotion AMI=0.1284,
  V-measure=0.1289, genre AMI=0.2799, V-measure=0.3014; held-out emotion
  AMI=0.1095, V-measure≈(compute proportionally consistent value is not
  available — hardcode 0.1123, the value from Stage 1's own held-out row),
  genre AMI=0.2901, V-measure=0.4240. Label this row "Learned student,
  Stage 1 — linear heads (reference)" in both tables, positioned
  immediately before this run's own row, so the reader can directly compare
  Stage 1 vs. Stage 2 at a glance alongside the existing content-only/
  affect-only/late-union/hierarchical reference rows.
- Add one sentence to the report's opening explaining this is a controlled
  capacity-only comparison against Stage 1, restating what stayed identical
  and what changed.

Print clear timestamped progress logs matching the Stage 1 script's `log()`
format throughout, with "Stage 2" in place of "Stage 1" wherever the
original script names the pilot.
