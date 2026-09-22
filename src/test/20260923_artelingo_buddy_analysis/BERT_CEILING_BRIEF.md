# Brief: ArtELingo-native BERT emotion-classifier ceiling pilot

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_bert_ceiling_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context

Prior pilots used an off-the-shelf, out-of-domain GoEmotions classifier
(Reddit-comment-trained) as an affect signal, and its single-modality
buddy-graph (Leiden) ceiling was emotion AMI=0.1180 (genre AMI=0.0396).
We just discovered the ArtELingo authors shipped their OWN fine-tuned
emotion classifier, trained directly on this dataset's captions and its
real 9-way ArtEmis emotion taxonomy:

`/data/PDD/artelingo/ArtELingo/ArtELingo/saved_models/Emotion Prediction/single_head/bert_english/`

This is a standard HuggingFace `BertForSequenceClassification` checkpoint
(`config.json` + `pytorch_model.bin`, `_name_or_path: "bert-base-uncased"`,
9 output labels named `LABEL_0`..`LABEL_8` — anonymized, but the count matches
the real taxonomy). Load it with `AutoModelForSequenceClassification.
from_pretrained(MODEL_PATH)`; the tokenizer was NOT saved alongside it, so
load `AutoTokenizer.from_pretrained("bert-base-uncased")` separately (matches
`_name_or_path`).

**Important interpretive caveat, to be stated prominently in the report, not
just in comments:** this checkpoint was almost certainly trained on
ArtELingo's official train split, which is the same split
`artelingo_train.json` was built from. Any "ceiling" measured on these exact
painting nodes may be partly memorization, not generalization. Still
valuable — it answers "given the best-available in-domain emotion signal,
what AMI ceiling does buddy-graph/DEC clustering hit at all?" — but must not
be presented as a fair apples-to-apples generalization comparison against
the GoEmotions result.

Read `src/test/20260923_artelingo_buddy_analysis/run_single_modality_pilot.py`
and `run_pipeline.py` in full first, for conventions and REUSE, not
reimplementation: `log()`, `load_sibling_module()`, `pipeline.
load_dedup_features()`, `pipeline.external_metrics()`, `pipeline.majority()`,
`pipeline.load_genre_map()`, `pipeline.K`, `pipeline.ALPHA`, `pipeline.
TRAIN_JSON`, `pipeline.assert_extraction_complete()`, and — most importantly
— `run_single_modality_pilot.py`'s own `build_single_modality_graph()`
function (import and call it directly as a library function; do not
re-derive the mutual-kNN + `ensure_min_degree` + `ensure_connected`
"same-array-for-both-modality-slots" graph-repair logic a second time).

## Implementation

1. Write `extract_bert_ceiling_nodes(train_json_path, paintings, device)` —
   closely mirror `run_affect_pilot.py`'s `extract_affect_nodes()` (same
   row-to-painting grouping via each row's `painting` field, same
   `caption_text()` string-vs-list handling, same batching pattern, same
   `np.add.at` accumulation and mean-pooling by painting index), but:
   - Load the model from `MODEL_PATH = "/data/PDD/artelingo/ArtELingo/
     ArtELingo/saved_models/Emotion Prediction/single_head/bert_english"`
     via `AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)`,
     tokenizer via `AutoTokenizer.from_pretrained("bert-base-uncased")`.
   - Use `torch.softmax(logits, dim=-1)` (NOT sigmoid — this is a single-
     label 9-way classification head, softmax is correct here, unlike
     GoEmotions' multi-label sigmoid).
   - batch_size=256, max_length=64 (captions are short), same as the
     GoEmotions extraction convention.
   - Output shape must be (num_paintings, 9).

2. **Per-caption classification-accuracy sanity check — do this FIRST,
   before any pooling or graph-building, and report it prominently.** For
   every English caption row (not yet pooled by painting), compute
   `argmax(softmax(logits))` and compare against that row's own ground-truth
   `emotion` string label. Since the checkpoint's label indices are
   anonymized (`LABEL_0`..`LABEL_8`), you cannot directly compare an index to
   a string — instead, do this: collect `(predicted_index, true_emotion_string)`
   pairs for all rows, then find the BEST index-to-emotion-string mapping via
   a linear assignment / greedy majority-vote approach: for each of the 9
   predicted indices, find which true emotion string is most common among
   rows predicted with that index (this recovers the implicit label mapping
   empirically, matching how the earlier community-vs-label AMI comparisons
   never assumed known cluster semantics either). Report: (a) the recovered
   index→emotion mapping and how many rows support each mapped pair, (b) the
   overall top-1 accuracy under that recovered mapping. This is a real,
   informative number on its own — report it clearly before moving on to the
   graph-based ceiling test, and do NOT skip it even though it isn't part of
   the AMI comparison — it's the direct validation that the checkpoint loads
   and predicts correctly on this data.

3. Build the single-modality graph on the mean-pooled 9-dim softmax vectors,
   via `run_single_modality_pilot.py`'s `build_single_modality_graph()`
   (reused, not reimplemented) — same K, alpha, device-detection convention
   as the rest of this investigation.

4. Run `detect_communities` (from `src.conditional_buddy.prototype_seed`,
   same import as the other scripts) on the resulting graph, seed=42.

5. Compute community-vs-emotion AMI/V-measure (full graph, via `pipeline.
   external_metrics()` and `pipeline.majority()`/`pipeline.
   load_dedup_features()` for `majority_emotion`) and community-vs-genre
   AMI/V-measure (genre subset, via `pipeline.load_genre_map()`) — same
   pattern as every other pilot in this directory.

## Report

Write `src/test/20260923_artelingo_buddy_analysis/bert_ceiling_pilot_report.md`:

- State the memorization/leakage caveat from above prominently, near the top
  of the report, not buried at the end.
- The per-caption classification accuracy check: recovered label mapping,
  support counts, overall accuracy.
- The graph-based ceiling result: a comparison table with two rows —
  "GoEmotions-only (off-the-shelf, out-of-domain) Leiden ceiling": emotion
  AMI 0.1180, genre AMI 0.0396 (hardcode, don't recompute) — and
  "ArtELingo-native BERT (in-domain, train-set, likely leaky) Leiden
  ceiling": the actual numbers from this run.
- One paragraph interpreting the result under BOTH possible outcomes
  (explained in the context section above): if this ceiling is also modest
  (similar to 0.12-0.15) despite high per-caption accuracy and in-domain
  training, that points at the clustering METHOD as the bottleneck, not
  signal quality. If it's substantially higher, that points at domain
  mismatch/signal quality as the main limiter of the GoEmotions result.
  State which of these two readings this specific run's numbers support.

Print clear timestamped progress logs matching the other scripts' `log()`
format throughout (model loading, per-caption accuracy check, extraction
progress, graph building, final evaluation).
