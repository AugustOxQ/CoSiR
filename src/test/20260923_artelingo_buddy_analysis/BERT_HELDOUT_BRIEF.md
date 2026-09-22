# Brief: ArtELingo-native BERT ceiling on genuinely held-out data

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_bert_heldout_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context and why this run matters

`run_bert_ceiling_pilot.py` (already committed) measured the ArtELingo
authors' own fine-tuned emotion classifier (`single_head/bert_english`) on
the TRAIN split's painting nodes — the same split it was almost certainly
trained on. Result: 93.67% per-caption accuracy, emotion AMI=0.2897 (vs.
GoEmotions-only's 0.1180). That result likely mixes real signal with
train-set memorization and is not a fair read of the model's true
generalization.

This script reruns the identical measurement on a genuinely held-out set:
`/data/PDD/artelingo/artelingo_val_test.json` (val+test combined, 46,813
rows, 9,365 unique paintings, verified ZERO painting-level overlap with the
train split). Its CLIP features have been (or will have been, by the time
this runs) extracted into
`/data/SSD2/pre_extract/artelingo_heldout/features` using the same
extraction script and process as the train store.

**Why this matters beyond just "is the number smaller":** a supervised
classifier trained on human emotion labels being accurate on held-out data
is a fair, standard generalization test — but it still isn't evidence that
buddy-graph clustering can discover affect structure *unsupervised* the way
it discovers content/genre structure from raw CLIP features. It would only
show that a well-matched supervised signal, once you have one, survives
being routed through clustering. State this distinction explicitly in the
report's conclusion, not just the leakage caveat — both points matter for
how this result should be used going forward.

Read `run_pipeline.py`, `run_single_modality_pilot.py`, `run_affect_pilot.py`,
and `run_bert_ceiling_pilot.py` in full first, for conventions and reuse.

## Implementation

1. Import `run_pipeline.py`, `run_single_modality_pilot.py`,
   `run_affect_pilot.py`, and `run_bert_ceiling_pilot.py` as sibling library
   modules via the same `load_sibling_module()` /
   `importlib.util.spec_from_file_location` pattern used throughout this
   directory (none of them execute anything at import time beyond defining
   functions/constants — safe to import without triggering their own
   `main()`).

2. **Point the reused helpers at the held-out store, not train**, by
   reassigning the imported `run_pipeline` module's constants BEFORE calling
   its functions:
   ```
   pipeline.STORAGE_DIR = "/data/SSD2/pre_extract/artelingo_heldout/features"
   pipeline.TRAIN_JSON = "/data/PDD/artelingo/artelingo_val_test.json"
   ```
   `pipeline.assert_extraction_complete()` and `pipeline.
   load_dedup_features()` both read these two module-level constants
   directly, so reassigning them redirects both functions to the held-out
   data with no other changes needed. Do this reassignment right after
   importing `run_pipeline.py`, before any calls into it.

3. Call `pipeline.assert_extraction_complete()`, then `pipeline.
   load_dedup_features()` to get `(paintings, img_nodes, txt_nodes,
   emotion_counts)` for the held-out set (img_nodes/txt_nodes are unused
   here beyond the function's own internal image-consistency sanity check —
   this run only needs the BERT/affect signal, not CLIP features, but
   `load_dedup_features()` requires the feature store to exist regardless).
   Get `majority_emotion` via `pipeline.majority()` on `emotion_counts`, same
   as every other script.

4. Extract the BERT affect signal via `run_bert_ceiling_pilot.py`'s
   `extract_bert_ceiling_nodes()` — this function already takes
   `train_json_path` as a parameter (not a hardcoded constant), so call it
   directly with `"/data/PDD/artelingo/artelingo_val_test.json"` as that
   argument, passing a fresh `caption_accuracy` dict to receive the
   recovered-mapping accuracy stats — no need to reimplement or duplicate
   this function. Do NOT reassign `run_bert_ceiling_pilot`'s own module
   constants; it doesn't need them, since its function takes the JSON path
   directly.

5. Build the single-modality graph via `run_single_modality_pilot.py`'s
   `build_single_modality_graph()`, same K/alpha/device convention as every
   other pilot. Run `detect_communities` (from
   `src.conditional_buddy.prototype_seed`), seed=42.

6. Compute community-vs-emotion AMI/V-measure (full held-out graph) via
   `pipeline.external_metrics()`, and community-vs-genre AMI/V-measure via
   `pipeline.load_genre_map()` on whatever subset of held-out paintings
   overlaps the genre-labelled diagnostic set (this overlap will be much
   smaller than the train run's 1,144 — likely under 200 paintings, since
   most of the genre diagnostic set's paintings are in train. Report the
   exact overlap count and flag explicitly in the report that this genre
   check has low sample size / low statistical power on this run, unlike
   the emotion check which uses the full held-out node set).

## Report

Write `src/test/20260923_artelingo_buddy_analysis/bert_heldout_pilot_report.md`:

- State the held-out setup clearly: 46,813 rows, 9,365 unique paintings,
  verified zero painting-level overlap with train.
- Per-caption classification accuracy on held-out captions, using the SAME
  recovered label mapping approach as the train run (empirical per-index
  majority vote — recompute it fresh on this held-out data's own
  predictions, don't reuse the train run's mapping, since the mapping
  should be recoverable independently and comparing the two recovered
  mappings is itself a sanity check — note in the report whether the
  recovered mapping matches the train run's mapping label-for-label).
- A three-row comparison table: "GoEmotions-only (off-the-shelf,
  out-of-domain)": emotion AMI 0.1180, genre AMI 0.0396 (hardcode) —
  "ArtELingo-native BERT, TRAIN split (likely leaky)": accuracy 93.67%,
  emotion AMI 0.2897, genre AMI 0.0582 (hardcode) — "ArtELingo-native BERT,
  HELD-OUT val+test (this run)": the actual numbers just computed. Note the
  genre-subset sample size explicitly next to the held-out genre AMI number.
- A conclusion paragraph addressing BOTH of these, explicitly:
  (a) Memorization vs. generalization: how much did the held-out accuracy
  and AMI drop from the train-split numbers? State plainly whether this
  supports "mostly genuine generalization" (small drop) or "mostly
  memorization" (large drop, held-out numbers close to or below the
  GoEmotions baseline).
  (b) Supervision-dependency scope limit: regardless of (a)'s answer,
  restate clearly that this result — even if held-out numbers stay strong —
  is evidence about a *supervised* signal surviving clustering, not evidence
  that buddy-graph/DEC can discover affect *unsupervised*. This distinction
  matters because RedCaps (CoSiR's other main dataset) has no emotion labels
  to supervise a comparable classifier with, so this specific result
  wouldn't transfer there even if it holds up perfectly on ArtELingo.

Print clear timestamped progress logs matching the other scripts' `log()`
format throughout.
