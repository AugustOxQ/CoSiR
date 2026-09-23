# Brief: PercepT Stage 2 — P-Topic Mapping (image-only classifier)

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage2_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: what Stage 2 is and what Stage 1 gave it

Read `src/test/20260922_percept_topic_pipeline/PERCEPT_STAGE1_REPORT_BRIEF.md`
lines 41-60 (the PercepT Stage 2 mechanism summary) and
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_cluster_count_sweep_v2_pilot.py`
and `run_percept_stage1_cluster_count_sweep_pilot.py` and
`run_percept_stage1_pilot.py` in full first, for the exact fused-embedding
construction, autoencoder, K-means, DEC, and pruning helpers you must reuse
— do not reimplement any of Stage 1's mechanics, only load and reuse them.

Stage 1's winning, 4-seed-validated configuration is `N_INITIAL_CLUSTERS=60`,
`N_SURVIVING_CLUSTERS=40`, `LAMBDA_BALANCE=1000`, `LAMBDA_RECONSTRUCTION=1`,
`SEED=42` (this specific seed — pick it as the single representative,
reproducible Stage-1 fit for freezing pseudo-labels; it is one of the 4
seeds that cleared the held-out Pareto bar, held-out emotion AMI=0.1238,
genre AMI=0.2617). Re-fit this EXACT configuration from scratch at the start
of this script (same pretrain, same K-means seed, same DEC training) to
obtain a frozen encoder and 40 surviving centers — do not try to load
weights from a previous run's process; there is no saved checkpoint, so
re-fitting deterministically at the same seed is the correct and only way to
reproduce it.

PercepT's own Stage 2 (P-Topic Mapping) is supervised, on FROZEN Stage-1
pseudo-labels: cluster assignments become fixed multi-label targets O ∈
{0,1}^k per image (an image can belong to zero, one, or several topics). A
separate small network predicts topic membership from the image alone (no
text at inference). The paper's own ablation found a SIMPLE mapper —
attention-pool patches to one vector, then a linear multi-label classifier —
beat a more complex cross-attention "topic query" mapper (AUC 0.94 vs 0.91).

**Adaptation required, and why:** this codebase's cached CLIP features are
global pooled image/text embeddings (`FeatureManager` stores only
`img_features`/`txt_features`, no patch-level embeddings — confirmed by the
orchestrating session; extracting patch embeddings fresh for 308k+ images is
out of scope here). Use the existing global CLIP image embedding directly as
input to a plain linear multi-label classifier — this is not a downgrade of
the paper's intent, it already matches the paper's OWN better-performing
mapper variant (a single pooled vector -> linear classifier), just with the
pooling step already done for us by CLIP's own architecture rather than
learned via attention. State this plainly in the report as a deliberate,
paper-consistent simplification, not an ad hoc shortcut.

## Deriving frozen multi-label targets (train and held-out)

Stage 1's own evaluation always used a single hard `argmax` topic per
painting. PercepT's target space is multi-label. Reconstruct genuine
multi-label targets from the SOFT assignment distribution Stage 1 already
computes (`soft_assignments()`, reused from the base pilot module — do not
recompute this differently):

For each painting (train and held-out separately, using the SAME frozen
Stage-1 encoder and the 40 surviving centers): compute its soft assignment
`q` (length 40, row-normalized as `soft_assignments()` already does). The
multi-label target for that painting is the set of topic indices where
EITHER (a) it is that painting's argmax topic (always included — guarantees
every painting has at least one positive label, avoiding an all-zero target
row that BCE cannot learn from), OR (b) `q[topic] > 2.0 / 40` (twice the
uniform probability `1/N_SURVIVING_CLUSTERS` — a simple, explainable
multi-label threshold letting genuinely ambiguous paintings carry more than
one label). Log the resulting label-count distribution (mean, median, max
labels per painting; fraction of paintings with more than one label) for
both splits in the report — this number matters for interpreting Stage 2's
results and must not be omitted.

Held-out targets use the exact same frozen Stage-1 encoder and centers
(projected forward, not refit) — this tests whether the image-only mapper
can recover what the FULL content+affect fused Stage-1 pipeline would have
assigned to that same painting, which is precisely what PercepT's Stage 2
evaluates.

## Stage 2 model and training

- Input: the existing global CLIP image embedding for each painting (reuse
  `load_dedup_features()`'s `img_nodes` array — the SAME array already used
  as Stage 1's content-embedding input — L2-normalize it, since Stage 1's
  own `content_features()` convention does the same).
- Model: `nn.Linear(D_img, 40)` producing raw logits (sigmoid applied only
  inside the loss/at inference, not as a separate module layer) — the
  paper's own simpler, better-performing mapper design.
- Loss: `nn.BCEWithLogitsLoss()` against the multi-hot target matrix
  (train split only).
- Training: Adam, a reasonable fixed learning rate (`1e-3`) and epoch
  budget (`100` epochs, full-batch — train split is 61,402 rows, comparable
  scale to Stage 1's own full-batch DEC training, so the same full-batch
  approach is appropriate and consistent with this investigation's
  conventions). Log mean epoch BCE loss every 10 epochs via `log()`.
- No validation-based early stopping needed — this is a small linear model
  on a fixed target; log the loss trajectory and let it run the full fixed
  budget, consistent with the pretraining convention already used
  throughout this investigation (fixed epoch count, not convergence-based,
  for pieces that are not DEC's own self-referential dynamics).

## Evaluation

On the held-out split (image embeddings + the held-out multi-label targets
derived above — the classifier never sees held-out images during training):
compute per-topic AUC (`sklearn.metrics.roc_auc_score`) for every one of the
40 topics, then report macro-averaged AUC (mean across topics) as the
headline number, plus the min/median/max per-topic AUC (some topics will be
easier than others — report the spread, do not hide it behind one macro
number). Skip a topic's AUC computation (and note it explicitly, do not
silently drop it or crash) if that topic has zero positive or zero negative
held-out examples, since AUC is undefined in that case.

**Baseline for context (no external bar exists here — PercepT's own
0.94/0.91 numbers come from a different dataset/topic count and are not
directly comparable):** compute a trivial baseline's macro AUC where every
held-out painting's predicted score for each topic is that topic's TRAIN-set
marginal frequency (a constant per topic, ignoring the image entirely).
Report both the trained model's macro AUC and this baseline's macro AUC
side by side, so the reader can see how much the image actually contributes
beyond the topic prior.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage2_pilot_report.md`
with:
- Which Stage-1 configuration and seed was re-fit and frozen, and a
  one-line confirmation that its re-derived held-out emotion/genre AMI
  matches the already-established K=60/40 seed-42 numbers (0.1238 / 0.2617)
  as a sanity check that the re-fit reproduced the same result — if it does
  NOT match closely, stop and report this as a reproducibility problem
  rather than proceeding to train Stage 2 on a different clustering than
  intended.
- The multi-label target statistics (mean/median/max labels per painting,
  fraction multi-labeled) for both splits.
- The training loss trajectory.
- The held-out per-topic AUC table (or at minimum macro/min/median/max
  summary plus a note of how many topics were skipped for degenerate
  positive/negative counts), and the baseline comparison.
- A plain, honest closing statement: does the image-only mapper meaningfully
  beat the marginal-frequency baseline? This is the actual test of whether
  Stage 2 learned something real from the image, not just memorized topic
  base rates.
