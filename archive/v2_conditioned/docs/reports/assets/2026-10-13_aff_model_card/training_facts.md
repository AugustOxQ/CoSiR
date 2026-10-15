# AFF training: fact sheet for the model card and its training figure

Written 2026-10-13 from the files named per item. The model card
(`docs/reports/stage/2026-10-13_aff_model_card.md`) and the figure `aff_training.png` (built by `build_aff_training.py`
in this folder) both use these facts. Where a source and this sheet differ, the source wins; report the difference. Corrected the
same day after the model card's source check (cosine on development episodes, the factor term's centring, the pick's
comparator, the Leiden resolution, the fresh test's per-seed picks).

Short names: R3 = `src/test/20261121_round3_affect_gate/DECISION_RULE.md`; R6 = `src/test/20261125_artelingo_held_test/DECISION_RULE.md`;
STAGE = `docs/reports/stage/2026-10-12_affect_steering_line.md`; RB = `docs/reports/auto/v2/2026-11-18_reader_fix_csd.md`.

## 0. Frozen inputs (never trained by us)

- **Data:** ArtELingo (WikiArt paintings, each with several captions written by viewers). A row = one painting with
  one caption. Rows are split by painting: **scorer-train** (183,694 rows; every fitted part), **selection**
  (development episodes, seed 42) and **held** (12,281 paintings; the final test). STAGE §4.1, GLOSSARY.md.
- **CLIP ViT-B/32**, frozen: one 512-d feature per image and per caption, unit-normalised before every head.
- **GoEmotions**: a public RoBERTa classifier fine-tuned on Reddit comments, never trained on ArtELingo; it gives each
  caption 28 emotion probabilities. Used only to build the affect groupings, offline. STAGE:417-419.

## 1. Groupings: label-free partitions of the scorer-train rows (seed 42)

| Grouping | Built how | Groups | Used in |
|---|---|---|---|
| affect (Leiden) | Leiden communities (modularity, resolution 1.0) on a 20-nearest-neighbour union graph of the 28 GoEmotions probabilities; groups under 200 rows merged | 41 | AFF's reader and steering term; B′(A0) |
| affect-km | k-means on the same 28 GoEmotions probabilities | 64 | B only |
| image | k-means on CLIP image features | 64 | B, B′(A0), reader, steering term |
| caption | k-means on CLIP caption features | 64 | B, B′(A0), reader, steering term |

Sources: R3 D1; `src/test/20261031_pseudo_partitions/build_partitions.py`; `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`.

## 2. Heads: place any image or caption in a grouping

- One head per grouping and modality: 4 groupings × 2 modalities = **8 heads** (6 for AFF's three groupings, 2 more
  for affect-km inside B).
- Each head: `LogisticRegression(C=1, max_iter=300)` (scikit-learn, lbfgs) on unit-normalised CLIP features,
  fitted on a 60,000-row random draw of scorer-train rows (draw seed 0). Target = the row's group id in that grouping
  (no human labels). Output: a posterior p_h(x) over the groups of grouping h.
- Held-out placement accuracy of the affect (Leiden) heads: image head 9.81%, caption head 35.72% (round 5 report,
  `docs/reports/auto/v2/2026-11-23_idea3_goemotions.md`:41).
- Two scores come from the heads: **grouping score** s_h(q, k) = p_h(q)·p_h(k) for query q and candidate k; **pair
  agreement** a_h(i, t) = p_h(i)·p_h(t) for an example pair's image i and caption t. R3 D2 to D4.

## 3. Factor model (from the earlier "method A" line)

- Checkpoint A3 (`src/test/20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt`): a factor-learning model trained
  2,000 steps, model seed 42, on scorer-train CLIP features only, with a bank of 65,536 pseudo-aspect episodes built
  from the three k-means groupings (affect-km, image, caption). PREREGISTRATION.md of that folder, §3.
- B uses its **centred uniform factor term** (T_N1u): a condition-free score; the query is centred on the episode's 8
  example items and each candidate on the mean of the 13 candidates (`src/eval/aspect_quick_checks.py:49-66`).
  On its own it reached R@1 17.95 on development episodes, against 12.96 for cosine (STAGE:418;
  `docs/reports/auto/v2/2026-11-08_new_method_quick_checks.md`:122-124). The model card explains what it computes.

## 4. Reader: which grouping do the example pairs share?

- **Practice episodes**, no labels: the scorer-train rows are split into two painting halves; on each half the heads
  are re-fitted and a bank of pseudo-aspect episodes is drawn, whose 4 supports share a group of one grouping and
  whose 4 contrasts share a group of another (16,384 per grouping pair; 49,152 per half for AFF's three groupings).
  The answer (which grouping) is known by construction. RB:108-115.
- **Input: 18 numbers** per episode and condition, 6 per grouping: S (mean support-pair agreement), C (mean
  contrast-pair agreement), Δ = S − C, the spread of the 4 support agreements, the spread of the 4 contrast
  agreements, the share of support pairs whose image and caption fall in the same top group. R3 D5.
- **Model:** StandardScaler + multinomial logistic regression per half; C by 5-fold cross-validation on the bank
  (1.0 on half 0, 100.0 on half 1). The two half-readers' probabilities are averaged into P(h).
- **Steering term:** T = Σ_h P(h)·s_h. **Top-two margin** m = the largest minus the second-largest P(h).
- Swapping supports and contrasts flips every Δ (Δ_b = −Δ_a), so evidence for a grouping in condition a is evidence
  against it in condition b.

## 5. Picks on the development episodes (selection rows, episode seed 42, 12,288 episodes = 4,096 per aspect pair)

Every pick is **cross-fitted**: the episodes are split into two parity halves by index; the setting picked on one half
scores the other half. No held row is read.

- **B** (condition-free base): z(cos) + λ_u·z(factor term) + λ_a·z(head agreement averaged over affect-km, image,
  caption), 56 settings (7 × 8 weights); each half picks the setting with the highest R@1. R3 D10;
  `src/eval/aspect_quick_checks.py:297`.
- **B′(A0)** (the bar): the same, with the head agreement averaged over AFF's own three groupings. R3 D11.
- **Thresholds τ:** the 0th, 25th, 50th and 75th percentiles of R1's seed-42 top-two margins: τ_0 ≈ 0.00004 (open
  almost always), τ_1 = 0.217, τ_2 = 0.480, τ_3 = 0.750. R3 D6.
- **Gate:** R1 opens when m ≥ τ; **AFF** opens when m ≥ τ **and** the reader's top grouping is affect (the
  one-sided rule). R3 D6.
- **Fused score:** S = (1 + λ_u)·z(B) + λ_a·gate·z(T); z = z-score over the 13 candidates of one ranking. Gate closed:
  the ranking is B's. R3 D8.
- **224 settings:** 4 τ × 7 λ_u (0, 0.5, 1, 2, 4, 8, 16) × 8 λ_a (0, 0.25, 0.5, 1, 2, 4, 8, 16). Each half picks the
  setting with the largest min(R@1 gain over B, condition gain); the "control" in R3 is B rescaled, (1 + σ)·z(B),
  so the R@1 gain is over B's own ranking. R3 D8 item 5.
- **Matched control:** the same 224 settings with the gated term replaced by its average over the two conditions
  (so it cannot follow the condition); each half picks by R@1. R3 D9.
- **Why affect (reason recorded for the paper, stated after affect was seen to pay):** affect is the grouping least
  redundant with B; its grouping score correlates 0.35 to 0.38 with B, against 0.62 to 0.71 for image and caption.
  R3 D7.

## 6. Frozen and tested

- **AFF's frozen picks** (R6 §4): tune half 0 → (τ_0, λ_u = 4, λ_a = 16), scores parity 1; tune half 1 →
  (τ_2, λ_u = 0, λ_a = 16), scores parity 0. Ratio of steering weight to base weight when the gate is open: 16/5 and
  16/1.
- **Fresh test** (round 3): new episode seeds 49 to 51 on selection rows, 36,864 episodes. The readers, heads, τ and
  the 224-setting grid were frozen; B, B′(A0) and the fusion weights were re-picked by halves on each seed (round 3
  report §3.2). AFF 18.88 R@1, B′(A0) 18.29, +0.591 [+0.462, +0.729]; with the seed-42 settings frozen instead,
  +0.59 [+0.46, +0.72] (descriptive).
- **Held read** (round 6): held paintings, seeds 52 to 54, 36,864 episodes: AFF 19.44 R@1, B′(A0) 18.84,
  margin +0.60 [+0.46, +0.73]; cosine 13.25; R1 19.41 (AFF minus R1 +0.02 [−0.06, +0.11]). STAGE:250-261.
