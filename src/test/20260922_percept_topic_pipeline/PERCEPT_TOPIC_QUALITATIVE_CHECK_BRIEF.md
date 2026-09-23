# Brief: PercepT — qualitative check of what q > 1.2/40 actually pairs together

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_topic_qualitative_check.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.
This is a read-only diagnostic, not a training pilot — no model training,
just inspection.

## Context: the open interpretive question

Read `docs/reports/2026-09-23_artelingo_percept_stage2_report.md` section V
("What the threshold result does, and does not, establish") first. Both the
Stage 2 report and the arch-ablation follow-up flag one remaining open
question: `q > 1.2/40` makes ~77% of train paintings multi-labeled (mean
2.9 of 40 topics), and the AUC evidence shows this is a real, learnable,
non-degenerate target space — but no one has actually looked at WHAT the
extra labels mean. Is a painting's second/third topic genuinely related
content, or is the threshold just loose enough to tag weak affinity as
membership?

## What to build

Reuse `run_percept_stage1_pilot.py` and
`run_percept_stage1_cluster_count_sweep_pilot.py`'s exact Stage-1 re-fit
(K=60/40, seed 42, same reproduction check against 0.1238/0.2617 — stop if
it fails). Derive the same `q > 1.2/40` multi-label targets on the TRAIN
split (reuse `multi_hot_targets()` — import it, do not reimplement; find it
in `run_percept_stage2_pilot.py` or `run_percept_stage2_sweep_pilot.py`,
whichever already defines it).

Pick **5 topics at random** (fixed `random.seed(42)` for reproducibility)
from the 40 surviving topics. For each chosen topic:
- Find every train painting where that topic is the painting's ONLY label
  (single-labeled — its argmax topic, with no other topic crossing
  threshold). List up to 5 such paintings' captions (use the same
  `caption_text()` convention already used throughout this investigation
  — reuse it, don't reimplement — one caption per painting is enough, the
  first English caption row for that painting).
- Find every train painting where that topic is a SECOND OR LATER label
  (i.e. NOT its argmax, but still crosses the `q > 1.2/40` threshold — this
  is exactly the "extra" membership the open question is about). List up to
  5 such paintings' captions, AND for each, also print that same painting's
  PRIMARY (argmax) topic's up-to-2 single-labeled example captions from
  the list above, so a reader can directly compare "what topic X's core
  members look like" against "what got pulled into topic X as a secondary
  label from topic Y."
- Also print each topic's total single-labeled count and total
  secondary-membership count, so the reader knows whether the topic's
  primary identity is well-populated or itself sparse.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_topic_qualitative_check_report.md`
with:
- The Stage-1 reproduction check.
- One section per sampled topic: its single-labeled example captions, its
  secondary-membership example captions (each paired with its own primary
  topic's examples for comparison), and the population counts.
- A closing paragraph giving your own honest read, stated as an
  observation from the printed examples, not a statistical claim (this is
  qualitative, n=5 topics, not a rigorous test): do the secondary labels
  look like genuine thematic overlap (e.g. shared subject matter, style, or
  mood language in the captions) or do they look more like noise/loose
  affinity? Say plainly if the 5 sampled topics give a mixed or unclear
  picture rather than forcing a single verdict — this is meant to inform
  follow-up work, not settle the question definitively.
