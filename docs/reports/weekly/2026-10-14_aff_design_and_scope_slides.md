## Method of one-sided affect steering: design

![One-sided affect steering: a base score plus an emotion bonus used only sometimes](../assets/2026-10-14_weekly/slides/aff_design.png)

- One-sided affect steering (AFF in tables and figures). One episode: a query painting, 4 example pairs that share one aspect, such as emotion, never named; 4 contrast pairs that share another; 13 captions to rank
- Grey: a base score that always ranks the candidates
- Blue: an emotion bonus, added only when the gate (block 5) opens
- Blocks 1 to 6: next slide

Notes: This deck follows last week's deck (docs/reports/weekly/2026-10-07_aspect_task_to_affect_steering_slides.md), which gives the task, the data and the background. It has three parts: the method and its current gain; where the gain comes from; the experiments now running. This week has no written weekly report. The numbers come from the AFF model card (docs/reports/stage/2026-10-13_aff_model_card.md), the outside-view card on where the gain comes from (docs/reports/stage/2026-10-14_aff_scope_limits.html), the held-test report (docs/reports/auto/v2/2026-11-25_artelingo_held_test.md), the k-shot report (docs/reports/auto/v2/2026-12-03_kshot_supports.md) and the round 9 run log (src/test/20261202_r9_scope/). Abbreviations are avoided in the text; the figures use the short names AFF (one-sided affect steering), B, B′(A0), B′(A1) and R1. All features are frozen CLIP ViT-B/32 features. The query can be a painting or a caption; the figure shows a painting. Each example pair joins a painting with a caption written for another painting, so a pair carries the shared aspect, not an identity. The examples never show the query's own value: if the query is sad, the example pairs share another feeling. No evaluation label enters a fitted part (groupings, heads, factor model, reader); the labels only build and score the episodes, and the development episodes pick the settings.

---

## Method of one-sided affect steering: the blocks

- **1 Groupings:** label-free clusters, each a stand-in for an unknown aspect: emotion (41 groups of GoEmotions scores), image and caption (64 CLIP clusters each)
- **2 Heads:** small classifiers that place any new image or caption in the groups of each grouping
- **3 Base score B:** CLIP similarity + factor match (32 factors from an earlier model of ours) + group match. It uses the examples, but not which aspect they show
- **4 Reader:** in which grouping do the example pairs agree more than the contrast pairs? A probability per grouping
- **5 Gate, the one-sided rule:** opens only when the reader's top grouping is emotion, with a clear margin over the second
- **6 Emotion bonus:** query and candidate group match, weighted by the reader, so mostly emotion. Large weight; with the gate closed, B ranks alone

Notes: (1) The emotion groups are Leiden communities of GoEmotions' 28 emotion scores over the training captions; GoEmotions is a public text classifier trained on Reddit comments, never on our data. A fourth grouping, k-means on the same emotion scores (64 groups), is used only inside B. (2) Each head is a logistic regression; its output is a probability over the groups, and two items match in a grouping when the dot product of their group probabilities is high. (3) B = z(cosine) + β_f·z(factor match) + β_h·z(group match), each z-scored over the 13 candidates. The factor match subtracts the mean code of the episode's 8 example items (4 example and 4 contrast pairs, the query's modality) from the query's code, and the mean code of the 13 candidates from each candidate's code, then takes the mean product over the 32 factors. Because examples and contrasts enter alike, it is condition-free: swapping them leaves it unchanged. On development episodes the factor match alone lifts R@1 from 12.96 (cosine) to 17.95, and B reaches 18.34. (4) Example pairs join a painting and a caption of different paintings, so the image and caption of a pair are chosen to share a value of the shown aspect, not an identity. Six numbers per grouping (support agreement, contrast agreement, their difference, two spreads, a same-group share) give 18 inputs. (5) Swapping examples and contrasts flips the reader's evidence, so the gate opens mostly on one side of an episode: on held episodes in 79% of first conditions and 30% of second ones at the lower threshold. (6) The fused score is S = (1 + λu)·z(B) + λa·gate·z(T), T = Σ P(grouping)·group match; when the gate opens, the bonus weighs 3.2 or 16 times B, depending on the fixed setting.

---

## Method of one-sided affect steering: training

| Part | Fitted on | Loss or objective |
|---|---|---|
| 4 groupings | training rows | clustering without labels (Leiden, k-means) |
| 8 heads | training rows | log loss on the row's group |
| factor model, 32 factors | practice episodes | ranking loss both ways (weight 3), swap loss (weight 1), pair-agreement and decorrelation terms |
| reader | practice episodes | log loss on the grouping the example pairs share |
| B's two weights, gate threshold, two bonus weights | development episodes | B: best R@1. Gate and bonus: the best setting on both R@1 gain over B and condition gain |

- **Practice episodes** come from the groupings: example pairs share a group of one grouping, contrast pairs a group of another. The answer is known, so no human label is needed
- After this, everything is fixed for the test on held paintings

Notes: Sizes: 183,694 training rows; the heads use a draw of 60,000 of them; the factor model trains for 2,000 steps on 65,536 practice episodes; the reader is fitted twice, once per painting half of the training rows, on 49,152 practice episodes each, and the two are averaged; 12,288 development episodes; B's weights from 56 settings, AFF's from 224. The reader's 18 numbers are 6 per grouping: mean support-pair agreement, mean contrast-pair agreement, their difference, the spread of each, and the share of support pairs whose image and caption fall in the same group. It recovers the practice episodes' grouping about 80% of the time (chance 33%), but on development episodes it picks the right aspect's grouping only 51% of the time. The factor model comes from an earlier line of ours; as a conditioned model it failed (2.96 R@1 points below its own condition-free control on fresh episodes), so only its centred term stays inside B: alone it scores 17.95 R@1 against 12.96 for cosine. The two fixed settings: half 0 opens the gate on almost every emotion pick and weights steering 3.2 times B; half 1 opens it only on confident picks (top-two margin at least 0.48) and weights steering 16 times B. The selection rule takes the minimum of the two gains, so a setting cannot win on one measure by losing the other.

---

## Setting of the comparison: true baselines and our own scores

| Kind | Score | What it is |
|---|---|---|
| True baselines (published) | CLIP cosine | the backbone alone |
| | RCA | a classic metric learned from the episode's example pairs |
| | fine-tuned CLIP | CLIP tuned on ArtELingo pairs, no labels (probe, last block, adapters) |
| | in-context MLLM | Qwen3-VL-8B sees the whole episode in one prompt |
| | describe-then-score | Qwen3-VL-8B names what the examples share; CLIP scores by it |
| Condition-free (ours): same ranking whichever aspect the pairs show | B | AFF's base score |
| | B′(A0) | B on AFF's own three groupings: the bar AFF had to beat, fixed in advance |
| | B′(A1) | B′(A0) plus a style grouping from CSD, a style encoder: our strongest |
| | matched control | AFF with only the condition removed |
| Reads the condition (ours) | R1 | AFF without the one-sided rule: steers on any grouping the reader picks |

- Condition gain: how much more often the shown aspect's candidate comes first than the other aspect's. It is exactly 0 for every condition-free score

Notes: B′(A0) is a strong bar we built ourselves, not a published method: it uses the examples and AFF's groupings, so it takes credit for everything AFF gains without reading the condition. The matched control follows the project rule that a conditioned score must beat the same score with only the condition removed. R@1 is the share of rankings in which the right candidate comes first; differences are in R@1 points.

---

## Method of one-sided affect steering: one example against B

| Query: a sad Baroque painting | Condition a | Condition b: example and contrast pairs swapped |
|---|---|---|
| Target | X: sad, not Baroque | Y: Baroque, not sad |
| Example pairs agree in | emotion (no pair is sad) | style (no pair is Baroque) |
| Contrast pairs agree in | style | emotion |
| B | the same scores as in condition b | the same scores as in condition a |
| Reader's top grouping | emotion | not emotion |
| Gate | opens | stays closed |
| One-sided affect steering | B plus a large emotion bonus: X moves to first | B alone |

- B is right in at most one of the two conditions, so its condition gain is 0
- One-sided affect steering changes only condition a: where B put Y first, it is now right in both
- The risk: the reader picks emotion when the pairs show style or genre, and the bonus lifts the wrong candidate

Notes: An illustration, not data: the painting, the captions and the values are made up. The example pairs never show the query's own value, so in condition a no pair is sad, and in condition b no pair is Baroque. The reader compares, per grouping, how often the image and caption of each example pair fall in the same group with the same share for the contrast pairs; in condition b the contrast pairs agree in emotion, so the emotion evidence is negative and the gate stays closed. Condition gain compares the same target under both conditions: for X, how often X comes first in condition a minus how often it comes first in condition b. The matched control averages the emotion bonus over the two conditions, so it gives X the same lift in both: it is also condition-free, so it too is right in at most one of the two conditions.

---

## Test of one-sided affect steering on held paintings

![AFF against every comparator on held paintings](../../../archive/v2_conditioned/docs/reports/assets/2026-10-12_affect_steering_line/held_comparison.png)

- Paintings never used in tuning, 36,864 episodes: all seven checks set in advance passed
- R@1 19.44, against 15.80 for the best fine-tuned CLIP and 13.25 for CLIP alone
- Over our bar B′(A0): +0.60 [+0.46, +0.73]
- Condition gain +3.06; every true baseline is near 0
- Over B′(A1) and R1: no clear margin (the intervals include 0)

Notes: The seven checks: AFF beats cosine, RCA, B, B′(A0) and the matched control on R@1, and the condition-free scores and RCA on condition gain, with a Holm correction. Episodes came from seeds 52 to 54 on the 12,281 held paintings; the in-context MLLM ran on seed 52 only. The held test repeated the earlier fresh test (new seeds 49 to 51 on development paintings, weights re-picked per seed) almost exactly (+0.59 [+0.46, +0.73]). R1 also cleared all seven checks; AFF's lead over R1 was +0.20 on the fresh test and +0.02 on held paintings, so the value of the one-sided rule over R1 is not shown. Disclosures: AFF was found among about 50 label-free variants on development episodes, and about 10 more were tested there later; the held paintings were used in earlier tests of the old value task; 24 held rows have a training image at CLIP cosine of at least 0.99.

---

## Analysis of the held gain by aspect pair

![AFF minus B′(A0) per aspect pair](../../../archive/v2_conditioned/docs/reports/assets/2026-10-12_affect_steering_line/per_pair_fresh_held.png)

- Bars: R@1 margin over B′(A0); light for new development seeds, dark for held paintings
- The gain is on the two pairs with emotion: +0.59 and +1.64
- Style × genre loses: −0.43 against B′(A0), −1.24 against B′(A1)
- New seeds and held paintings show the same pattern

Notes: Held intervals: emotion × style +0.59 [+0.35, +0.83], emotion × genre +1.64 [+1.40, +1.90], style × genre −0.43 [−0.65, −0.21]. The test fixed in advance is on the margin pooled over the three aspect pairs, and that passed; the per-pair margins were not part of the test, and the loss on style × genre is reported beside it. The next part asks why.

---

## Analysis of where the gain comes from

| Example pairs | All episodes | Pairs show emotion | Pairs show style | Pairs show genre |
|---|---|---|---|---|
| 4 (as designed) | +0.70 | +2.88 | −0.09 | −0.69 |
| 3 | +0.63 | +2.92 | +0.04 | −1.06 |
| 2 | +0.19 | +2.58 | −0.17 | −1.84 |
| 1 | −1.73 | +4.41 | −1.15 | −8.44 |

- R@1 margin over B′(A0) on development episodes, split by the aspect the example pairs show
- All of the lead is in the emotion episodes, from an outside classifier (GoEmotions); style ties, genre loses
- A random gate that opens as often on each side scored the same: the gain is steering to emotion on the right side
- With 1 pair the method falls below the bar: the pairs mainly stop wrong steering on genre

Notes: Intervals for 4 pairs: all +0.70 [+0.46, +0.94]; emotion +2.88 [+2.40, +3.38]; style −0.09 [−0.46, +0.28]; genre −0.69 [−1.07, −0.31]. With 2 pairs the pooled interval includes 0 (+0.19 [−0.06, +0.44]); with 1 pair it is −1.73 [−2.05, −1.41]. The random gate was tested on fresh episodes (seeds 49 to 51): AFF minus it −0.04 [−0.14, +0.05]. It is given which side is which, so it is a control, not a method. With 1 pair the reader is confident but often wrong: at the lower gate threshold the gate opens in 62% of rankings under the swapped condition, against 30% with 4 pairs, so the method steers towards emotion on the wrong side and genre pays. The reader and all settings were fitted with 4 pairs and left unchanged, so a drop has two causes: less evidence and settings made for 4 pairs. These are development numbers (the seed every setting was picked on), so their levels are a little optimistic; on held paintings the pooled margin was +0.60.

---

## Analysis of style and genre

![Label-free style groups match genre as much as style](../../../archive/v2_conditioned/docs/reports/assets/2026-10-14_aff_scope_limits/csd_style_genre_ami.png)

- Figure: agreement (AMI, 0 to 1) of three groupings with the true style and genre labels; development episodes
- The groups of CSD, a style encoder, match style (0.34) and genre (0.33) about equally
- With the true style labels (an upper limit): +3.92 on style × genre
- Label-free: the best of 12 style scorers gained +0.25 (needed: +0.5); 20 learned groupings and 3 other encoders: none passed

Notes: The upper limit means the score is given the right grouping from the evaluation labels: an upper limit, not a method. The 12 style scorers came from six label-free groupings (CSD with content directions removed, CSD's neighbour graph with content edges cut, CLIP token statistics with the image embedding removed), each scored in two ways (R1, and AFF that may also steer on the style grouping). The learned groupings used a contrastive head whose negatives come from the query's own content group, and fair clustering that penalises agreement with content; the three encoders were ALADIN, DINOv2 and SigLIP 2. DINOv2 came closest: not clearly different from AFF (+0.29 [−0.01, +0.59], so its lower bound missed 0), and it also lost too much on emotion × style. The true style labels agree with the true genre labels at only 0.22, so style and genre are separable in the labels; no label-free source we tried separates them.

---

## Discussion of learning without aspect labels

- The method is not general in the usual sense: an outside model supplies the emotion grouping, and nothing we tried supplies style or genre
- All of this rests on one choice: we learn without aspect labels. With labels, three things change:
- **Baselines:** supervised methods, such as conditional similarity networks, join the comparison
- **Training:** a model per aspect, for example a style model trained to ignore genre
- **The condition:** the aspect could be named or learned, not only guessed from the example pairs
- **Open question:** should training use aspect labels?

Notes: The emotion grouping already rests on an outside labelled classifier (GoEmotions), and CSD was trained with style tags from LAION, yet it still mixes style and genre. The upper limits with the true labels show what labels could buy (+3.92 on style × genre). This question is open for the meeting.

---

## Experiment on two more backbones

| Margin of one-sided affect steering over … | CLIP | SigLIP 2 | Qwen3-VL-Embedding-2B |
|---|---|---|---|
| our bar B′(A0), R@1 | +0.60 | +0.51 | +0.63 |
| its matched control, R@1 | +0.63 | +0.83 | +0.78 |
| the encoder alone (cosine), R@1 | +6.19 | +6.59 | +4.82 |
| condition gain over the condition-free scores | +3.06 | +3.35 | +2.79 |
| seven checks | all pass | all pass | all pass |

- The same method on two more encoders: each part that uses the encoder is re-fitted the same automatic way, with no new tuning
- Both pass all seven checks on the same held episodes. Preliminary: the round report is still to come
- SigLIP 2 needed one fix (a group of 1 row; small groups now trigger a refit). Whether SigLIP 2 stays in the claim is open

Notes: SigLIP 2 is so400m/14-384. Re-fitted per encoder: the image and caption groupings, the heads, the factor model, the reader and the settings; the two GoEmotions groupings come from the captions alone and stay as they are. Intervals for SigLIP 2: B′(A0) +0.51 [+0.36, +0.66]; matched control +0.83 [+0.70, +0.96]; cosine +6.59 [+6.36, +6.82]; condition gain +3.35 [+3.15, +3.54]. Intervals for Qwen3-VL-Embedding-2B: B′(A0) +0.63 [+0.49, +0.76]; matched control +0.78 [+0.65, +0.90]; cosine +4.82 [+4.61, +5.03]; condition gain +2.79 [+2.61, +2.97]. For SigLIP 2, an independent re-computation agreed on all 166 numbers it compared. The prior, written before any number: a positive margin over B′(A0) on both new backbones, likely +0.2 to +0.6, since a stronger encoder lifts B as well. The refit rule changed nothing for CLIP and Qwen (their groupings refit byte-equal); whether SigLIP 2 stays in the claim with this rule is still to decide. The per-pair numbers and the stronger comparators (B′(A1), R1) come with the round report.

---

## Plan for more datasets

| Dataset | What it is | Aspects | Status |
|---|---|---|---|
| ArtELingo-28 | 746 paintings, captions in 27 other languages, translated to English | emotion, style, genre | CLIP and SigLIP 2: first level not passed (inconclusive); Qwen: running |
| CUB | bird photos, 10 captions each | primary colour, bill shape, wing colour | CLIP probes on development species done, not yet analysed |
| SemArt | paintings with catalogue descriptions | type, school, timeframe | CLIP probes on the validation split done, not yet analysed |
| Affection | 85,007 photos with emotion explanations | emotion, object category | waiting for the data |

- ArtELingo-28: other languages, same domain. Two levels: first beat B′(A0) on R@1 and the matched control on condition gain, then all seven checks
- CUB and SemArt: other domains; their held tests are decided after the probes
- Affection is dropped if its data does not arrive in time

Notes: ArtELingo-28 uses 746 paintings (248 of our held paintings plus 498 new); its English rows are left out because 88% copy our captions; the other languages are translated with NLLB and pooled, with per-language numbers reported as description. It runs on CLIP, Qwen3-VL-Embedding-2B and SigLIP 2 whatever the backbone results. CLIP's ArtELingo-28 test did not pass its first level: the condition gain passed (+3.23 [+2.97, +3.51]), but the R@1 margin over B′(A0) did not (+0.06 [−0.10, +0.23]), so the test counts it as inconclusive. SigLIP 2 shows the same pattern: condition gain passed (+3.51), the R@1 margin over B′(A0) did not (+0.15, with an interval that includes 0); inconclusive. Like the backbone results, this is preliminary, before the round report. CUB's held test would use the 50 unseen species of the standard zero-shot split; SemArt's the official test split (1,069 paintings). Affection's second aspect would come from a COCO-category join, after a feasibility check. Dates: the round 9 tests (the two backbones and ArtELingo-28) start by 18 October; the CUB, SemArt and Affection tests get their own decision, about 17 October; Affection is dropped if no data comes by 24 October. The CUB probe ran with a relaxed episode build (no third aspect; at least 10 images per value), since its value pools are small. Results enter the paper body by 6 November; CVPR abstract 10 November, paper 16 November.
