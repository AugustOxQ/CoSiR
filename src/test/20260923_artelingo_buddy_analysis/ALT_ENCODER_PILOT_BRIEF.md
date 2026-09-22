# Brief: alternate off-the-shelf affect encoder, single-modality ceiling

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_alt_encoder_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context and why this run matters

The affect-fusion pilot (`run_affect_pilot.py`) and single-modality pilot
(`run_single_modality_pilot.py`) both used `SamLowe/roberta-base-go_emotions`
("GoEmotions"), a 28-category sigmoid multi-label classifier trained purely on
Reddit comments — out-of-domain for ArtELingo's descriptive art captions. Its
single-modality Leiden ceiling was **emotion AMI = 0.1180** (genre AMI =
0.0396), committed in `single_modality_pilot_report.md`.

Later, `run_dec_pilot_v2.py` showed that changing the *clustering method*
(Leiden → convergence-controlled DEC) on that same GoEmotions signal raised
emotion AMI to 0.1492 (+26.4% relative) — a real but modest gain, still far
below the in-domain-supervised ceilings (ArtELingo-native BERT: 0.2897 on the
train split, likely leaky; 0.1693 on genuinely held-out data).

This pilot tests the *other* lever: does a better-matched **off-the-shelf**
affect encoder (still requiring zero ArtELingo-specific training) raise the
ceiling on its own, holding the clustering method (Leiden, K=20, seed=42)
fixed exactly as in the GoEmotions single-modality run? Isolate this one
variable — do not also rerun DEC in this pilot; that is an intentional scope
decision to avoid conflating two independent levers in one result.

**Model choice:** `j-hartmann/emotion-english-distilroberta-base` — a 7-way
softmax classifier over Ekman's six basic emotions plus neutral (anger,
disgust, fear, joy, neutral, sadness, surprise). It is trained on a mix of
several text-domain datasets (including Crowdflower, MELD TV-dialogue
transcripts, ISEAR personal narrative self-reports, SemEval-2018, and
GoEmotions itself), which is broader and includes more narrative/self-report
text than GoEmotions' single Reddit source.

**Important caveat — state this explicitly in the report's conclusion:**
because GoEmotions is one of this model's training-mix ingredients, this is
NOT a fully independent comparison; any improvement should be read cautiously
as "a broader-domain-trained encoder" rather than "a completely unrelated
encoder." Do not overclaim independence.

Read `run_pipeline.py`, `run_affect_pilot.py`, and `run_single_modality_pilot.py`
in full first, for conventions and reuse. Match their `log()` timestamped
progress-printing convention throughout.

## Implementation

1. Import `run_pipeline.py` and `run_single_modality_pilot.py` as sibling
   library modules via the same `load_sibling_module()` /
   `importlib.util.spec_from_file_location` pattern used throughout this
   directory (neither executes anything at import time beyond defining
   functions/constants).

2. **Write a new extraction function `extract_alt_affect_nodes(train_json,
   paintings, device)`** in the new script. Do NOT try to reuse or
   monkey-patch `run_affect_pilot.py`'s `extract_affect_nodes()` for this —
   that function hardcodes `torch.sigmoid` inline, which is wrong for this
   model (a single-label softmax classifier, not GoEmotions' multi-label
   sigmoid setup). Instead, copy-adapt its structure directly into the new
   function with these changes:
   - `MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"`
   - Use `AutoModelForSequenceClassification` / `AutoTokenizer` exactly as
     the existing function does.
   - Replace `torch.sigmoid(model(**encoded).logits)` with
     `torch.softmax(model(**encoded).logits, dim=-1)`.
   - Output dimensionality is 7, not 28 — do not hardcode 28 anywhere in the
     new function.
   - Keep the same mean-pooling-by-painting logic, the same English-language
     guard (`record.get("language", "english").lower() != "english"`), the
     same batch size (256) and max length (64), and the same
     missing-painting `RuntimeError` guards as the existing function — these
     are not model-specific and should not change.
   - You may import and reuse `run_affect_pilot.py`'s standalone helpers
     `caption_text()` and `l2_normalize()` directly (these have no
     GoEmotions-specific logic in them).

3. Call `pipeline.assert_extraction_complete()`, then `pipeline.
   load_dedup_features()` to get `(paintings, img_nodes, txt_nodes,
   emotion_counts)`. Get `majority_emotion` via `pipeline.majority()`, same
   as every other script. `img_nodes`/`txt_nodes` are unused beyond the
   function's own internal consistency check.

4. Call `extract_alt_affect_nodes()` to get the new encoder's per-painting
   node features (shape `(len(paintings), 7)`), then L2-normalize them via
   `run_affect_pilot.py`'s reused `l2_normalize()`.

5. Build the single-modality graph via `run_single_modality_pilot.py`'s
   `build_single_modality_graph()` — pass the new 7-d node array directly,
   same call convention as its existing `"GoEmotions-affect-only"` row (K=20,
   from `pipeline.K`). Run `detect_communities` (from
   `src.conditional_buddy.prototype_seed`), seed=42, same as every other
   pilot.

6. Compute community-vs-emotion and community-vs-genre AMI/V-measure via
   `pipeline.external_metrics()` and `pipeline.load_genre_map()`, same
   pattern as `run_single_modality_pilot.py`'s `evaluate_graph()` (reuse that
   function directly by importing it from the loaded
   `run_single_modality_pilot.py` module — do not reimplement it).

## Report

Write `src/test/20260923_artelingo_buddy_analysis/alt_encoder_pilot_report.md`:

- State the setup: model name, its 7-way Ekman+neutral softmax scheme, the
  broader/mixed training-domain rationale, and the explicit non-independence
  caveat (GoEmotions is part of its training mix) — this caveat must appear
  in the setup section, not buried only in the conclusion.
- A comparison table with these exact reference rows (hardcode the known
  values) plus this run's new row:

  | signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
  |---|---:|---:|---:|---:|
  | GoEmotions-affect-only (Leiden, reference) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |
  | GoEmotions-affect-only (DEC v2, reference — different clustering method, not directly comparable to this pilot's Leiden-only rows) | 0.1492 | 0.1498 | 0.0554 | 0.0930 |
  | ArtELingo-native BERT, held-out (reference — in-domain supervised, upper bound) | 0.1693 | 0.1730 | 0.0138 | 0.2518 |
  | j-hartmann-affect-only (Leiden, this run) | *computed* | *computed* | *computed* | *computed* |

- A conclusion paragraph stating plainly whether the new encoder's Leiden
  emotion AMI clears the predeclared bar of **AMI > 0.177** (a 50% relative
  improvement over GoEmotions' 0.1180 single-modality ceiling — the same bar
  `run_dec_pilot_v2.py` used for its own "real win" criterion, chosen here so
  the two independent levers, clustering method and encoder choice, are
  judged on a comparable standard). State the result as: "yes, real
  improvement" / "smaller improvement, does not clear the bar" / "no
  improvement or worse" as appropriate — do not round up an ambiguous result.
  Then restate the non-independence caveat once more, explicitly, regardless
  of which way the result goes.
- Also report genre AMI's direction of change (did it hold, drop, or
  improve alongside any emotion gain) for the same trade-off visibility as
  the other pilots' tables — no bar required for this one, descriptive only.

Print clear timestamped progress logs matching the other scripts' `log()`
format throughout.
