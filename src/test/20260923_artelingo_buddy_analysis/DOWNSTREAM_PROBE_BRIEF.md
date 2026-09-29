# Brief: shared downstream probe — does either topic bottleneck help predict held-out human labels? (candidate 1's second half)

Write one new script. Do NOT run it — execution happens separately, on
GPU, outside this task.

## Context

Read, in full, before writing any code:
- `docs/reports/2026-09-26_buddy_silhouette_gap_brainstorm.md`, candidate 1
  — the shared downstream-probe idea this pilot implements.
- `src/test/20260923_artelingo_buddy_analysis/buddy_percept_matched_silhouette_audit_pilot_report.md`
  and `run_buddy_percept_matched_silhouette_audit_pilot.py` — the pilot
  this one directly follows up on. It established that PercepT's held-out
  centers are severely occupancy-collapsed (21/67 zero-occupancy, 50/67
  below 1%) while buddy's 19 communities are not (0 empty, 3/19 below 1%),
  alongside a real (not sampling-artifact) silhouette gap. This pilot asks
  the more direct question the audit explicitly left open: does that
  occupancy difference translate into a difference in how *useful* each
  system's frozen topic labels are for predicting real human judgments,
  when the only thing available at prediction time is the raw image.
- `../20260922_percept_topic_pipeline/run_percept_stage2_pilot.py`, in
  full — specifically `AttentionPoolingMapper`, `load_patch_features`,
  `PATCH_FEATURE_DIR`/`TRAIN_PATCH_FEATURE_PATH`/`HELDOUT_PATCH_FEATURE_PATH`,
  and how its `main()` obtains the canonical painting order via
  `pipeline.load_dedup_features()` that the patch-feature cache's rows are
  aligned to. Reuse `AttentionPoolingMapper` and `load_patch_features` and
  the three path constants **unmodified** — import them via the standard
  sibling `load_module` pattern used throughout this directory. Note that
  this existing pilot's own evaluation (`evaluate_auc`) scores the mapper
  against its **own** frozen topic pseudo-labels, not against human labels
  — this pilot adds the human-label evaluation that does not yet exist
  anywhere in this project.
- `attention_h1_embedding_snapshot.npz` (buddy's snapshot: `train_paintings`,
  `train_community_post` — confirmed 19 contiguous integer labels 0–18)
  and `../20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_snapshot.npz`
  (PercepT's snapshot from tonight: `train_paintings`, `train_topic` —
  67-wide label space, argmax over the 67 surviving centers, confirmed
  contiguous 0–66 though only 48 values ever appear on the train split;
  this is expected given the same occupancy collapse the matched audit
  measured, not a bug).
- `run_buddy_percept_matched_silhouette_audit_pilot.py`'s own
  `assert_matching_paintings` and reindex-by-painting-id pattern — reuse
  the same alignment approach (a painting-id → row-index dict, then
  reindex) to align each snapshot's train/held-out label arrays to the
  patch-feature cache's own canonical painting order (obtained fresh via
  `pipeline.load_dedup_features()`/`heldout_pipeline.load_dedup_features()`,
  exactly as `run_percept_stage2_pilot.py`'s `main()` does it). Do not
  assume the snapshots' painting order already matches the patch cache's
  order — verify by explicit reindexing, the same discipline the matched
  audit used.

## What to build

Write
`src/test/20260923_artelingo_buddy_analysis/run_buddy_percept_downstream_probe_pilot.py`.

### Part 1: train one image-only mapper per system

Train **two** `AttentionPoolingMapper` instances on the full 61,402-painting
train patch-feature cache — one with `n_topics=19` against buddy's frozen
train hard labels, one with `n_topics=67` against PercepT's frozen train
hard labels (both aligned to patch-cache order per above). Unlike the
original Stage 2 pilot's multi-hot `BCEWithLogitsLoss`, use plain
`nn.CrossEntropyLoss` against each system's single hard integer label per
painting — state this explicitly as a deliberate design choice in the
script's docstring and the report's Method section (both systems here
supply one hard label per painting, not DEC's soft multi-hot target, so a
single-label cross-entropy objective is the fair, symmetric choice for
this comparison, not a silent deviation from the original pilot). Reuse
`MAPPER_LEARNING_RATE = 1e-3` and `MAPPER_EPOCHS = 100` from the Stage 2
sibling for both mappers, Adam, full-batch (the existing Stage 2 pilot
already trains this same architecture on the same cache size this way, so
this is a known-feasible compute budget — do not change it). Log train
loss every 10 epochs for both.

After training, compute each mapper's held-out softmax topic-probability
predictions (`torch.softmax(mapper(heldout_patch_features), dim=-1)`) —
this is each system's held-out prediction of "what topic does this image
belong to," made from the image alone, no text/affect features involved.

### Part 2: a training-free pooled-feature control

Compute a **mean-pooled** 512-D vector per held-out painting
(`patch_tokens.mean(dim=1)`, no learned query, no training, no topic
bottleneck at all) as the control feature set. This isolates how much
downstream signal is available directly from the image before any
topic-bottleneck compression, independent of either system's chosen
number of topics or occupancy pattern.

### Part 3: the shared downstream probe

For **held-out emotion** (all 9,365 paintings have a majority-emotion
label): split into a stratified 50/50 probe-train/probe-eval split by
majority emotion (`sklearn.model_selection.train_test_split(...,
stratify=majority_emotion, random_state=42)`). Fit
`sklearn.linear_model.LogisticRegression(max_iter=2000)` (default
multinomial handling, no extra regularization tuning — a genuinely
fixed-capacity probe, not a model-selection exercise) three times on the
probe-train split, once per feature set (buddy topic-softmax [19-D],
PercepT topic-softmax [67-D], mean-pooled control [512-D]), each
predicting majority emotion. Evaluate each fitted probe's predictions on
the **held-out** probe-eval split with both AMI (`pipeline.external_metrics`,
consistent with every other metric in this investigation) and plain
accuracy.

For **held-out genre** (only the ~159 genre-annotated paintings overlap):
the sample is too small for a single split to be reliable. Use
`sklearn.model_selection.StratifiedKFold` (try `n_splits=5`; if any genre
class has fewer members than the requested fold count, reduce
`n_splits` to the largest value that works and state the actual value
used in the report — do not crash, do not silently drop rare classes)
with `cross_val_predict` to get out-of-fold genre predictions for each of
the three feature sets, then compute pooled AMI and accuracy across all
out-of-fold predictions combined. Use the same `LogisticRegression`
configuration as the emotion probe.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/buddy_percept_downstream_probe_pilot_report.md`
with: a Method section stating the cross-entropy deliberate-design-choice
note above, the actual `n_splits` used for genre, and the probe-train/
probe-eval split sizes for emotion; a combined results table (rows: buddy
topic softmax, PercepT topic softmax, mean-pooled control; columns:
emotion AMI, emotion accuracy, genre AMI, genre accuracy); the two
mappers' final train losses; and a final verdict section that states
plainly, referencing the matched audit's occupancy numbers directly:
does either topic bottleneck retain most of the control's achievable
signal, does one system's bottleneck retain reliably more than the
other's, and is the answer consistent between emotion and genre. This
investigation's established convention is a blunt, numeric verdict, not a
hedge — if one system is clearly better or worse here, or if the topic
bottleneck loses most of the control's signal for both systems, say so
plainly.

Do not touch git, do not modify any other file in the repository.
