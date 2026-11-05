# r22 explore: text and cross-modal retrieval (small loop, development data only)

Round folder of the r22 exploration. Module prefix `tx_`. Bulk outputs: `/data/SSD2/cosir_r22/20261105_text_retrieval/`.
Written before any score existed (2026-11-05 2026-11-05 17:20). The user's decisions are in the handoff
`docs/superpowers/handoffs/2026-11-05-r22-explore-handoff.md`, section 2; the choices below are agent defaults unless marked.

## Design

**Data.** ArtELingo validation: 6,152 paintings, 30,872 captions. c* seed 0. No held row is read.

**Rows.**

| Row | Text | Image | Note |
|---|---|---|---|
| c* | caption side of the head | image side of the head | the model as saved; the rule reads this row |
| c* free variant | image side of the head | image side of the head | Qwen puts text and images in one space |
| plain Qwen | plain | plain | main comparator (the user's "zero-shot model") |
| teacher's text | instructed | instructed | main comparator |
| CLIP L/14, SigLIP 2, SigLIP | plain | plain | the literature note's floor |
| GoEmotions RoBERTa | stored 768-d | none | text to text, emotion only |

**Directions.** Text to image, image to text, text to text; image to image is kept as a reference and as a metric parity check.

**Label hits.** A caption's emotion is its own label; a painting's emotion is the plurality of its captions (`rs_vh.emotion_labels`), a single
label, so the "at least one label" rule is not used. Style and genre belong to the painting. Unlabelled genre rows leave both query and gallery.
mAP over queries, macro mAP over classes, and a chance row (mean share of relevant gallery items). The gallery excludes the query itself only.
Two sensitivity rows: no captions that name their label (leak check), and no gallery item from the query's own painting.

**Pair hits.** Text to image: the own painting. Image to text: any own caption. Text to text: any other caption of the same painting. R@1/5/10,
MedR and MRR, with the rank of the first hit.

**Read-outs.** Raw cosine, centering per modality, centering plus expansion k = 10. Fit on the validation rows, with no labels. Each row takes its
best read-out per aspect and direction on label mAP. So the pick and the score use the same rows: slightly optimistic for every row alike, and an
exploration number only.

**Leak check.** Emotion uses `ewm_masks`' label word list; style and genre use the fixed patterns in `tx_data.py`.

**Diagnostic.** Each painting's caption mean as the query, the input the caption side trained on, for c*, plain Qwen and the teacher.

**Extras.** p: SemArt validation (1,069), known pair, both directions; our stored SemArt text has author, title and date scrubbed, so a raw-text row
is added for the published comparison. q: MMArt validation (5,769), 4 perspectives plus the unified text by 3 heads, pair hits and label mAP on
MMArt's own labels. r: word queries, bare word and one template per aspect, one query per class, mAP averaged over classes, a separate table.

## The rule, read literally (the user approved its words in the r22 decide chat; these limits are an agent default)

The cells are label-hit mAP, 3 aspects by 3 directions: 9 cells. Each row uses its picked read-out. The rule reads the c* row.

- **Above in a cell:** the 95% paired bootstrap interval (1,000 resamples of paintings, seed 0) of c* minus plain Qwen and of c* minus the teacher's text
  both lie above 0.
- **Outcome 1, "above in most cells":** 5 or more of 9 cells. A medium claim loop follows, on held ArtELingo and the MMArt test rows.
- **Outcome 2, level or below:** fewer than 5 cells. Try option t (caption-side retrain), then read again.
- **Outcome 3:** after t, still fewer than 5 cells. The work goes to the supplementary.

The free variant is read beside the rule. If it alone would change the outcome, that goes to the user. If c* wins on label hits and falls below plain
text on pair hits, the user decides whether the paper states it as a limit.

## Run (2026-11-05, 17:39 to 17:53)

Gates passed: the c* forward pass reproduces the saved validation outputs (largest difference 6.1e-5), so the caption side's input at export is each
painting's L2-normalized caption mean (`input_cap: default`); the metric equals `rt_metrics` on image-to-image (largest difference 6.9e-8); re-encoded
captions match the stored emotion and plain text (mean cosine 0.9999). Results: `results/tables.md`, `results/summary.json`; checks: `results/checks.json`.

**Outcome 1:** c* above plain Qwen and the teacher's text in 6 of 9 label cells; the free variant in 3 of 9 (beside the rule). `tx_checks.py` was added after
the first read, as a check only: with one fixed read-out for every row the count is 5 of 9 at each read-out, and cross-fit picks give 6 of 9.

Deviations from the design: CLIP B/32 joins the SemArt floor, since CLIP L/14 and SigLIP have no SemArt text there; image-to-image is left out of the pair table; word queries use raw cosine and centering only; the teacher is not run on MMArt, by scope.

Caption input in training: one row per painting, the caption mean (`ct2_d2.py` line 286 without `per_caption`; `ms_train.py` indexes it by painting).

Reports: `docs/reports/auto/percept/2027-01-09_text_retrieval.md`, `docs/user_read/2026-11-05_text_retrieval.md`. Claim review by a Sonnet worker: no number wrong, 18 wording and coverage fixes applied.

## Follow-up: two short checks (r23 decide chat, 2026-11-05)

The user accepted two short checks before the claim loop. Validation only, seed 0, no training, no held rows. Written before any score of these
checks. The limits are agent defaults. Code: `tx_followup.py`.

**Check 1, the own-caption term.** Does a term that pulls a painting to its own caption raise pair hits? Models: recipes R0-5 (term on) and R6-5 (term off) of
`20261025_recipe_backbones`, seed 0, sample scale, same teacher share. Both are older than c*. The same rankings as Table 2: each head picks its read-out on
MRR, and each recipe's best head by R@10 is read. The interval is the paired painting bootstrap of R0-5 minus R6-5 on per-query R@10.

- **Supported:** in both text to image and image to text, the interval lies above 0 and the rise is at least half of R6-5's gap to plain Qwen.
- **Partly:** the interval lies above 0 in at least one of the two, without the first case.
- **Not supported:** neither interval lies above 0.

Label mAP of both recipes is shown too, since the term may cost label hits. The check cannot show what the term would do to c* or to clustering.

**Check 2, joined scores.** A joined vector is (sqrt(1 - a) c*, sqrt(a) plain Qwen), a from 0 to 1 by 0.1, per head. Its cosine is (1 - a) c* cosine + a plain
cosine. Label cells take the best read-out per cell, as Table 1. Pair hits take the read-out picked on MRR and the best head by R@10, as Table 2. Comparators are
Table 1's plain Qwen and teacher values.

- **Joined scores enter the claim loop as one extra row** only if one a has pair R@10 within 0.02 of plain Qwen in both text to image and image to text, and
  5 or more of 9 label cells above both comparators by mean score. Its weight is then picked on validation.
- **Otherwise** the curve is a note for the pair-hit point. The claim loop keeps c* alone.

The rule's main read stays on c*; these checks change neither Outcome 1 nor the cell count.
