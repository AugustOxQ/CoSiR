# PercepT on its own encoders: 0.064 / 0.262 / 0.474 test AMI at the class count, all 40-topic seeds kept, strict cut collapsed

> 2026-11-11 · Spec: [r26 build handoff](../../../superpowers/handoffs/2026-11-11-r26-percept-original-handoff.md#2-decided-by-the-user-not-to-be-reopened) and [run handoff](../../../superpowers/handoffs/2026-11-11-r26-run-handoff.md#2-decided-by-the-user-not-to-be-reopened) · Decision rule: none (descriptive, a paper row) · [Run log](../../../../src/test/20261111_percept_original/20261111_percept_original_log.md)

## Results brief

**What we tried.** We trained PercepT at paper settings with 40 topics on PercepT's own encoders: CLIP ViT-L/14 for
image and text, and ModernBERT-base tuned on GoEmotions for the emotion vector. Training used the 42,969 ArtELingo train
paintings, seeds 0 to 4. We scored the 12,281 test paintings with r17's scorer. This fills Table 2's "PercepT, CLIP
L/14" row.

**The level reached.** No claim ladder: this is a descriptive row. Test AMI, k-means at each aspect's class count
(9 / 27 / 10), mean of 5 seeds:

| Row (Table 2) | Emotion | Style | Genre |
|---|---:|---:|---:|
| PercepT, CLIP L/14 + ModernBERT (this run) | 0.064 | 0.262 | 0.474 |
| PercepT, Qwen3 (r17's core version) | 0.068 | 0.192 | 0.424 |
| CoSiR (c*, Qwen3) | 0.102 | 0.411 | 0.510 |

- All 5 seeds of the 40-topic cut count; none collapsed.
- The strict topic rule of the paper (Algorithm 1) collapsed on 5 of 5 seeds, as on CLIP B/32 and Qwen before.

**What it does not show.** Our data differ from PercepT's: 42,969 English paintings, not its 1.2M translated pairs,
and 40 topics, not its own rule. So this row is not a faithful PercepT. Five seeds only. On genre, c*'s mean (0.510)
lies inside this row's seed range (0.457 to 0.512). No seed-level test against c* was run.

**Options for next.**
1. Nothing more: the row is in Table 2 (the paper tab filled it at 20:15).
2. Score the row under r18's WIN / LEVEL / BELOW rule against c*, as r25 did for its baselines. Cost: CPU minutes if
   c*'s saved test partitions are at hand. It tells the paper whether "above PercepT on genre" holds at the seed level.
3. Rerun Stage 2 on the CLIP L/14 global vector, not the patch tokens. Cost: a small code change and about 15 GPU
   minutes. It shows how much the row depends on that agent choice.

**Your reading.** <left for the user>

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** PercepT on its own encoders is a stronger comparator than our Qwen3 PercepT in style (+0.070) and
genre (+0.050), and level in emotion (-0.004). c* stays clearly above it in emotion (+0.038) and style (+0.149). The
genre gap (+0.036) is the same size as this row's seed spread, so "c* beats PercepT on genre" needs a seed-level test.

**Choices that could change a verdict.**
- The class-count reading (the user's choice) against k = 20 (recomputed): 0.068 / 0.256 / 0.460. The order of the
  three rows is unchanged at k = 20 (PercepT Qwen3 0.080 / 0.188 / 0.423; c* 0.107 / 0.407 / 0.500).
- Three agent choices could move this row (not recomputed): Stage 2 on patch tokens, the mean pooling of ModernBERT,
  and no centring of the emotion vector. The size of each effect is not known; a shift of 0.04 in genre would close
  the genre gap.

**Our recommendation.** Option 2, if the paper text says c* beats PercepT in all three aspects. Otherwise option 1:
the table already shows the numbers, and bold marks only the best per aspect.

</details>

## What ran

- **Chosen by:** the user, 2026-11-11, in the paper chat, after seeing the recommendation (build handoff, section 2).
- **Stages and types:** a small loop (r26) from a paper chat: one run that fills a paper row. Stage 1 (Eq. 2 fusion,
  autoencoder and DEC, 100 centres), both topic cuts (40 topics; strict Algorithm 1 with gamma 2), Stage 2 (attention
  pooling on the 257 CLIP L/14 patch tokens, multi-hot targets), held inference, scoring.
- **Cuts and collapse:** the 40-topic cut (code tag `percept_paper_k40`) keeps the 40 of 100 topics with the most
  fit pairs. The strict cut (`percept_paper_strict`) is PercepT's Algorithm 1 with gamma 2. A run collapses when one
  topic holds over half the fit pairs; a strict run also when it keeps under 10 or over 90 topics (GLOSSARY rows 9
  and 10).
- **Data and seeds:** fit on the 42,969 train paintings, label-free early stopping on 6,152 validation paintings; test on
  12,281 paintings (genre on 9,949). Seeds 0 to 4. k-means states 0 to 2.
- **Code:** branch `r26-percept-original`, merged into main at 9b932547; round folder
  `src/test/20261111_percept_original/`. Local RTX 3090, 19:00 to 20:09 (heavy run, about 260 W, so after the weekday
  rule's 19:00).

## Results

| ID | Measured | Number | Compared against | Levels 1 / 2 / 3 | Level reached |
|---|---|---|---|---|---|
| row | test AMI, class count (9 / 27 / 10), mean of 5 seeds | 0.064 / 0.262 / 0.474 | PercepT Qwen3 0.068 / 0.192 / 0.424; c* 0.102 / 0.411 / 0.510 | descriptive | n/a |
| supp | test AMI, k = 20 | 0.068 / 0.256 / 0.460 | PercepT Qwen3 0.080 / 0.188 / 0.423; c* 0.107 / 0.407 / 0.500 | descriptive | n/a |
| own | test AMI, top topic (argmax) | 0.076 / 0.251 / 0.419 | none | reference | n/a |
| collapse | seeds collapsed, 40 topics / strict | 0 of 5 / 5 of 5 | claim test: strict 10 of 10 collapsed | GLOSSARY rows 9 and 10 | n/a |

Per seed, class count: emotion 0.060 to 0.067, style 0.250 to 0.276, genre 0.457 to 0.512. Stage 2 mean validation
AUC 0.965. Top-topic share of the fit pairs 0.051 to 0.072 (collapse bar 0.5).

**Level reached under the rule:** none; a descriptive row with its comparators.

## Agent choices checked

Lines marked "yes" or "maybe" in `decisions.md`, ranked by how much they could move the row:

1. **Stage 2 input** (maybe): attention pooling on patch tokens, PercepT's main version, against the global vector.
   Not recomputed (needs a new Stage 2 path); the size of its effect is not known.
2. **ModernBERT pooling** (maybe): attention-mask mean against CLS. Not recomputed (new extraction and Stage 1).
3. **No centring of the emotion vector** (maybe): B4's PCA map centred it; PercepT's fusion does not. Not recomputed.
4. **The readings** (yes): the user's class-count reading against k = 20, recomputed above. It changes no comparison.

## Analysis

**Our reading.** PercepT on CLIP L/14 clusters style and genre better than our Qwen3 PercepT. Two differences can
explain it. The L/14 row trains Stage 2 on patch tokens with attention pooling, while the Qwen3 row uses an MLP on one
vector. CLIP L/14 also separates style well on its own: CLAY and CRL on CLIP L/14 reach 0.406 and 0.390 in style.
Emotion stays near 0.06 in every PercepT version we ran (B4 on CLIP B/32: 0.064 at the class count).

**The strict cut.** Algorithm 1 kept 96 to 99 of 100 centres on four seeds and one centre on seed 2. Our reading: the
centre norms on our data do not split into a clear kept group. This matches the claim test (10 of 10 runs collapsed on CLIP B/32
and Qwen), so the paper's 40-topic choice stands on three encoders.

## Deviations

- The full run started at 09:40 as a power probe. It drew 255 to 264 W, so it was stopped at 2.4 minutes (no output)
  and queued for 19:00.
- Stage 2 took 11 minutes, under the 25 to 40 minute estimate.
- The paper tab for Table 3 moved twice (sec4 tables 5, 6, 7); the files went to "sec4 tables 7". "Paper
  experiments 2" filled the Table 2 row and scored the supplementary cluster-count grid from our held files; its six
  gate cells equal ours exactly (main 902e0539).

## Claim review

1. Recomputed from the exported test files (`po_recheck.py`; the claim test's labels; scikit-learn directly), against
   `records/po_scores.json`:

   | Number in the brief | Recomputed | File |
   |---|---|---|
   | emotion 0.064 | 0.06358430 (equal at full precision) | `records/po_recheck.log` |
   | style 0.262 | 0.26217891 (equal) | same |
   | genre 0.474 | 0.47443702 (equal) | same |
   | k = 20: 0.068 / 0.256 / 0.460 | equal at full precision | same |

2. Collapse counts read from `po_scores.json`: 40 topics 0 of 5, strict 5 of 5 (kept 96, 99, 1, 99, 98).
3. The one "yes" line (the readings) is in the brief's folded block.
4. Comparator rows read from the paper's `Tables/main_clustering.tex` (class count) and the run handoff (k = 20).

## Files and storage

- **Outputs:** `/data/SSD2/cosir_r26/20261111_percept_original/outputs/` (709 MB): Stage 1 and 2 artifacts, held
  outputs, `test/<cut>/L1__s<seed>.npz` and `val/<cut>/` with names (Table 3 reads `test/k40/`), `score/`, `logs/`.
  Records in git: `records/po_scores.json`, `po_table.md`, `po_recheck.log`, `b4_scorer_check.json`.
- **Feature stores:** `/data/SSD2/pre_extract/artelingo_full/clip_l14_patch/` (23 GB) and `modernbert_go_emotions/`
  (1.1 GB), kept for Table 3 and the supplementary.
- **Storage:** see the disk ledger rows dated 2026-11-11.
