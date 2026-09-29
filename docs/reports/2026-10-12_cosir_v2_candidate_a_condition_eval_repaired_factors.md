# CoSiR v2 Candidate A: condition interface on the repaired factors (held-out evaluation)

## Verdict

**On held-out paintings, the repaired factors (R3) beat the collapsed factors (R0) at condition-aware matching for emotion and art-style conditions. R3 was selected under gates that were amended after the fact (Task 6). A condition-specific improvement over R0 is shown for text→image only. In image→text, R3 is better than R0 overall, but most of that gain (83%) is condition-independent, and the interaction CI spans 0 (post-hoc analysis from the final review, below). Two of the three pre-registered readiness criteria hold as written: the primary criterion and the floor. The secondary swap-reversal criterion does not hold.**

**Primary test.** 2,048 human-label episodes (1,024 emotion, 1,024 art style), built once from held rows and identical for both models.

| Measure (pooled R@1, image→text / text→image) | Value |
|---|---|
| R3, condition via the naive rule | **17.6% / 20.9%** |
| R0, naive rule | 13.1% / 16.7% |
| R3, uniform weights (condition ignored) | 15.6% / 15.7% |
| CLIP alone | 11.5% / 14.9% |
| Chance | 7.7% |
| R3 − R0 (paired, 95% CI) | **+4.5 [+2.7, +6.3]** and **+4.2 [+2.2, +6.2]** points |
| R3 naive − R3 uniform (paired) | **+2.0 [+0.5, +3.6]** and **+5.2 [+3.5, +6.8]** points |

**Readiness criteria**, applied literally:
1. **Primary: met.** Both paired differences have 95% CIs above 0 in both directions.
2. **Secondary: not met.**
   - R3's naive swap reversal is higher than R0's only in text→image: 65.2% against 44.4%.
   - In image→text it is lower: 62.5% (160/256) against 100% (9/9).
   - R0 has only 9 valid swap pairs on this split.
3. **Floor: met.** On factor-mined held episodes, R3 naive − uniform is +21.4 [+18.5, +24.2] and +22.7 [+19.8, +25.6] points.

**What the numbers say, in plain terms:**
- **The benefit is modest in absolute terms.** Given the condition, R3 puts the right item first about one time in five.
- **In image→text, most of R3's gain over R0 is not condition-specific.** With the condition ignored (uniform weights), R3 already beats R0 by +3.8 [+2.1, +5.4] points in image→text. That is 83% of the +4.5-point naive gain (86% at β=0). The interaction (R3 naive − R3 uniform) − (R0 naive − R0 uniform) is +0.8 [−1.0, +2.6] in image→text, a CI that spans 0, against +4.4 [+2.4, +6.4] in text→image. In text→image the uniform difference is −0.2 [−1.8, +1.4], so there R3's gain is condition-specific. Better use of the condition than R0 is therefore shown for text→image only; the image→text gain comes from a better space overall. The interaction numbers are a post-hoc analysis from the final review (see below).
- **The benefit is uneven.** For art style, the condition helps text→image strongly: +7.4 [+5.1, +9.9]. It does not clearly help image→text: +1.5 [−0.7, +3.6], a CI that includes 0. For emotion, the condition helps both directions by about 2.5-3 points.
- **Factor-mined episodes cannot compare the two models.** Each model mines its own episodes, so R0-vs-R3 comparisons there are unpaired and descriptive only. The repaired space also makes those episodes much harder for any scorer that ignores the condition: uniform weights reach about 11% R@1 on R3's mined episodes (chance is 7.7%), against 40-53% on R0's. Absolute mined R@1 therefore says nothing about which model is better.

**The user makes the stage (d) decision.** This report states only which criteria hold.

## Comparison: the split effect and the factor effect

The columns are:
- **Task 9:** collapsed factors on the row split.
- **R0:** collapsed factors on the painting split. R0 against Task 9 shows the split effect.
- **R3:** repaired factors on the painting split. R3 against R0 shows the factor effect.

All values are held R@1 in percent (image→text / text→image) unless stated otherwise. Differences are paired-bootstrap points with a 95% CI.

**Where the Task 9 numbers come from.** They are copied from `docs/reports/2026-10-07_cosir_v2_candidate_a_naive_rule_mechanism.md`, not recomputed. Their protocol differs from R0 and R3:
- The split is annotation-disjoint (the row split). 99.7% of held rows depict a painting that was seen in factor training.
- β was selected on 4,096 mined *train* episodes.
- Task 9's human-label test is its Q3b: emotion only, the catch-all label "something else" is used as a target, 6 of the 12 distractors are the CLIP-nearest other-emotion items (which forces CLIP-only to zero), negatives can depict a painting labelled with the target emotion, and the bootstrap used 2,000 resamples.

**The R0 and R3 protocol:**
- painting-grouped split;
- β selected on the val episodes;
- label episodes identical for both models, random clean negatives, "something else" never a target, bootstrap with 5,000 resamples.

| | Task 9 (collapsed, row split) | R0 (collapsed, painting split) | R3 (repaired, painting split) |
|---|---:|---:|---:|
| **Factor-mined held episodes** (each column mines its own, so these are unpaired) | | | |
| naive R@1, at its selected β | 58.1 / 49.6 (β .1) | 53.5 / 39.2 (β .1) | 32.3 / 34.0 (β 1) |
| uniform R@1 | 54.5 / 46.4 | 52.6 / 39.5 | 10.9 / 11.3 |
| oracle one-hot R@1 | 54.7 / 48.4 | 45.1 / 42.2 | 31.8 / 35.7 |
| CLIP-only R@1 | 13.7 / 21.7 | 12.2 / 21.2 | 8.3 / 8.6 |
| top-k R@1 | 62.1 / 56.5 (top-5 at β .1, k not selected) | 55.2 / 47.5 (k=5 chosen on val) | 31.3 / 35.0 (k=5 chosen on val) |
| naive − uniform, R@1, selected β | +3.6 [+1.7, +5.6] / +3.2 [+1.2, +5.3] | +0.9 [−1.1, +2.8] / −0.3 [−2.1, +1.6] | **+21.4 [+18.5, +24.2] / +22.7 [+19.8, +25.6]** |
| naive − uniform, R@1, β=0 | +3.5 [+2.0, +5.1] / +4.4 [+2.8, +6.0] | +1.6 [+0.3, +2.8] / +2.3 [+1.3, +3.5] | +17.9 [+14.9, +20.7] / +22.9 [+19.9, +25.9] |
| valid swap pairs | 61 | 9 | 256 |
| naive swap reversals (rate) | 13 / 15 (21% / 25%) | 9 / 4 (100% / 44%) | 160 / 167 (62.5% / 65.2%) |
| oracle swap reversals | 54 / 56 | 9 / 9 | 204 / 207 |
| per-role outrank at naive β, naive (hard-neg / condition-only / anchor-only, %), i2t; t2i | 1.7/40.5/2.8; 12.8/44.9/15.0 | 0.8/46.3/2.2; 8.8/56.8/13.3 | 15.5/49.1/45.2; 14.1/42.6/43.5 |
| the same for uniform | 4.3/41.2/7.6; 16.3/43.8/21.7 | 3.9/45.4/7.3; 11.3/55.8/18.9 | 55.0/28.5/88.7; 43.9/21.8/88.1 |
| **Human-label held episodes** | | | |
| emotion naive R@1 | 11.8 / 8.5 (Q3b, β 0) | 10.4 / 12.8 (β .3) | **15.2 / 16.2** (β .3) |
| emotion uniform R@1 | 9.7 / 5.3 | 9.4 / 12.4 | 12.7 / 13.3 |
| emotion CLIP-only R@1 | 0.0 / 0.0 (forced by CLIP-nearest distractors) | 9.7 / 12.3 | 9.7 / 12.3 |
| emotion naive − uniform, R@1 | +2.1 [+0.3, +4.1] / +3.2 [+1.5, +5.0] | +1.0 [−0.6, +2.6] / +0.4 [−1.3, +2.0] | +2.5 [+0.4, +4.7] / +2.9 [+0.7, +5.1] |
| art-style naive R@1 | not measured | 15.8 / 20.6 | **20.0 / 25.6** |
| pooled naive R@1 (emotion + art style) | not measured | 13.1 / 16.7 | **17.6 / 20.9** |
| pooled naive − uniform, R@1 | not measured | +1.2 [−0.05, +2.4] / +0.8 [−0.4, +2.0] | **+2.0 [+0.5, +3.6] / +5.2 [+3.5, +6.8]** |
| pooled R3 naive − R0 naive, R@1 (paired, same episodes) | not measured | | **+4.5 [+2.7, +6.3] / +4.2 [+2.2, +6.2]** |

**Reading the table.**
- **Split effect (Task 9 → R0), same recipe.**
  - On mined episodes, naive R@1 drops by 4.6 points (i2t) and 10.4 points (t2i).
  - The naive-over-uniform advantage that Task 9 found on mined episodes disappears at the selected β: +0.9 and −0.3, with both CIs spanning 0. It survives only at β=0 (+1.6, +2.3).
  - On R0's mined held episodes, only 9 pairs satisfy the swap-pair conditions, against 61 in Task 9.
  - These are different episodes on a different split, so this is descriptive.
- **Factor effect (R0 → R3), on the same painting split.**
  - The primary, paired comparison is the label-episode rows. R3 naive is higher than R0 naive in every pooled and per-type cell.
  - Only in text→image is that gain condition-specific. In image→text, R3 uniform − R0 uniform already accounts for most of it (83% pooled at the selected β). See the post-hoc interaction analysis below.
  - On mined episodes, the absolute R@1 falls, but the episodes themselves changed. With decorrelated factors, anchor-only distractors (like the anchor on everything except the target factor) become real competitors: uniform weights let at least one of them outrank the positive in 88-89% of R3's episodes, against 7-19% of R0's.
  - The large R3 naive-over-uniform gap on mined episodes is partly this construction effect. That is why the plan calls it a floor, not evidence of repair.

## Readiness criteria (pre-registered, brief Step 3), applied literally

**Primary evaluation point.** Each variant is evaluated at its own β, chosen on the val episodes. β=0 results are reported alongside for information. "CI above 0" means the lower bound of the 95% percentile interval is strictly greater than 0. All intervals below are for R@1 and in percentage points.

**1. Primary (held label episodes, pooled, paired): MET.**

| comparison | i2t | t2i | holds? |
|---|---:|---:|---|
| R3 naive − R0 naive | +4.5 [+2.7, +6.3] | +4.2 [+2.2, +6.2] | yes, in both directions |
| R3 naive − R3 uniform | +2.0 [+0.5, +3.6] | +5.2 [+3.5, +6.8] | yes, in both directions |

Per label type, reported alongside:

| comparison | label type | i2t | t2i |
|---|---|---:|---:|
| R3 naive − R0 naive | emotion | +4.9 [+2.5, +7.1] | +3.4 [+0.8, +6.1] |
| R3 naive − R0 naive | art style | +4.2 [+1.6, +6.9] | +5.0 [+2.1, +8.0] |
| R3 naive − R3 uniform | emotion | +2.5 [+0.4, +4.7] | +2.9 [+0.7, +5.1] |
| R3 naive − R3 uniform | art style | **+1.5 [−0.7, +3.6]** | +7.4 [+5.1, +9.9] |

- The one per-type cell whose CI includes 0 is art style image→text for naive − uniform.
- At β=0 (for information), the primary criterion also holds: R3 − R0 is +6.2 [+4.2, +8.2] / +7.5 [+5.5, +9.7], and naive − uniform is +2.6 [+0.9, +4.2] / +4.9 [+3.2, +6.6].
- The criterion is met as written, but it contains no interaction test (see Caveats). The post-hoc interaction analysis below shows better use of the condition than R0 in text→image only.

**2. Secondary (naive swap reversal on factor-mined held episodes higher than R0's, in both directions): NOT MET.**

| model | valid pairs | i2t reversals | t2i reversals |
|---|---:|---|---|
| R3 (β 1) | 256 | 160 = **62.5%** (exact 95% [56.3, 68.5]) | 167 = **65.2%** [59.1, 71.1] |
| R0 (β .1) | 9 | 9 = **100%** [66.4, 100] | 4 = **44.4%** [13.7, 78.8] |

- **How the criterion is applied.** It compares the rates, because the pair counts differ. R3 is higher in t2i and lower in i2t, so the criterion is not met.
- **The exact intervals** are Clopper-Pearson intervals computed from the counts. They are shown for information only; the criterion does not use them.
- **The evidence is weak in both directions.** R0's rates rest on 9 pairs drawn from different episodes. Its i2t interval overlaps R3's.
- **β=0 (for information) is also not met:** R3 79.7% / 78.5% against R0 88.9% (8/9) / 44.4% (4/9).

**3. Floor (naive − uniform on factor-mined held episodes, CI above 0 in both directions): MET.**
- R3: +21.4 [+18.5, +24.2] i2t and +22.7 [+19.8, +25.6] t2i. At β=0: +17.9 / +22.9.
- As the plan says, this cannot establish repair on its own. On the painting split, collapsed R0 does **not** meet this floor at its selected β (+0.9 [−1.1, +2.8] / −0.3 [−2.1, +1.6]). It does meet it at β=0.

## Post-hoc analysis from the final review (informed no decision)

This section was added after the final whole-branch review. **It is post hoc: it informed no decision, and it was not recomputed here.** Recomputing it would touch the held rows a third time. The numbers are the final reviewer's, cited as given.

**Method.** A difference-in-differences on the identical held label episodes: the interaction (R3 naive − R3 uniform) − (R0 naive − R0 uniform), in R@1 points. It uses a paired bootstrap over episodes, with 5,000 resamples and seed 42. The re-scored R@1 values match `results/eval_results.json` exactly. The per-type rows are at each variant's selected β.

| scope | selected β | β=0 |
|---|---:|---:|
| pooled i2t | +0.78 [−1.03, +2.64] | +0.88 [−1.12, +2.78] |
| pooled t2i | **+4.39 [+2.39, +6.40]** | **+4.54 [+2.54, +6.59]** |
| art-style i2t | 0.0 [−2.8, +2.8] | — |
| emotion i2t | +1.6 [−0.9, +4.0] | — |

**What it shows.**
- **Text→image:** the interaction is clearly positive at both β settings. R3 uses the condition better than R0 does.
- **Image→text:** the interaction CI spans 0 in the pooled set (at both β settings) and in each label type (at the selected β). The condition-independent share of R3's i2t gain over R0 is 83% at the selected β (the uniform difference is +3.76 of +4.54 points) and 86% at β=0. So the image→text gain comes from a better space overall, not from better use of the condition.
- **Criterion 1** is still literally met, and its verdict is unchanged. The analysis narrows what that verdict means.

## Evidence

### Setup and integrity checks

Run: `src/test/20261012_condition_eval_repaired_factors/run_eval.py`, once. Seed 42, RTX 3090, torch 2.11.0+cu130, **97.5 s**. A smoke run preceded it (26.5 s), using val rows in place of held rows; its numbers were discarded.

**Data**
- Split: `load_artelingo()` → `leakage_groups` → `grouped_split(seed=42)`, giving 216,107 / 30,872 / 61,744 rows. Leakage is zero by both the painting key and the image key (asserted).
- art_style: joined positionally via `annotations[int(i)]["art_style"]` (Task 6's loader). It gives 27 styles, with one style per painting and per leakage group (asserted).

**Checkpoints**
- Loaded with `load_factor_checkpoint`:
  - `R0_seed42.pt`: cosine agreement, no decorrelation;
  - `selected_seed42.pt` (R3): InfoNCE agreement, decorrelation 1.0.
  - Both configs are asserted.
- All 308,723 rows were encoded in 8,192-row batches under `no_grad`. **All codes are finite** (asserted).
- Active fraction: R0 71.4% (img) / 73.0% (txt); R3 45.3% / 48.5%.

**Val label episodes: identical to Task 6's**
- Rebuilt with Task 6's own `build_val_episodes`: 2,048 emotion episodes over 8 targets (with "something else" excluded) and 2,048 art-style episodes over 23 targets, from val, seed 42, following Ruling 13 (`paintings=` the leakage-group ids and clean negatives).
- Their metadata equals Task 6's stored episode metadata (asserted).
- `condition_lift` on them, with the reloaded checkpoints, reproduces Task 6's stored val lifts **exactly** for both models. That is 24/24 values per model: naive and uniform × R@1, R@3 and tie count × direction × label type.
- SHA-256: emotion `c67b32a9…a641`, art style `f01bec7f…cb60`.

**Held label episodes: built once, used for both models**
- 1,024 emotion episodes over 8 targets.
- 1,024 art-style episodes over 24 targets. New_Realism has at least 30 held paintings, so it is eligible on held; it was not eligible on val.
- The same three asserts as Task 6 apply: every row is held; no painting group repeats within an episode; building from held-only arrays gives the same episodes.
- SHA-256: emotion `e62ab41f7b54b500b25900a2e82c22d50f662c3843e767aeef3a8511c16a8c85`, art style `3a58cf9dc3cb670f39a5084f786968d2186bd511f036565a6f825116422e167f`.
- Each episode has an anchor, a positive, 4 supports, 4 contrasts and 12 distractors. That makes 13 candidates, so chance is 1/13 = 7.7% at R@1 and 3/13 = 23.1% at R@3.

**Scoring**
- Scores come from `conditional_score`. Ranks come from `tie_aware_rank`: ties count half against the positive, and any non-finite row gets the worst rank.
- On factor-mined episodes, `conditional_score` matches `score_pool` on the first 16 episodes of each part, for every variant at β=0 and at its selected β. The maximum absolute difference is 3.0e-8 (R0) and 2.4e-7 (R3), within the 1e-5 limit.

**Bootstrap**
- Paired bootstrap over episodes: 5,000 resamples, seed 42, 95% percentile intervals, using Task 9's `bootstrap_difference`.
- For "pooled", the 2,048 held label episodes (both label types) are resampled together.

### Step 1: human-label episodes (primary)

**β selection.** β was chosen on the 4,096 val episodes (both label types pooled), by mean bidirectional R@1. Ties go to the smaller β.

Selected β:

| model | naive | uniform | CLIP-only |
|---|---:|---:|---:|
| R3 | 0.3 | 0.1 | 0.001 |
| R0 | 0.3 | 0.3 | 0.001 |

Val curves, pooled R@1 in percent, i2t/t2i:

| model | variant | 0 | .001 | .003 | .01 | .03 | .1 | .3 | 1 | 3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 | naive | 18.5/20.8 | 18.5/20.8 | 18.5/20.9 | 18.5/21.0 | 18.6/21.0 | 18.6/21.1 | 18.8/20.9 | 17.8/20.5 | 16.6/18.9 |
| R3 | uniform | 15.2/15.7 | 15.2/15.7 | 15.2/15.7 | 15.3/15.7 | 15.3/15.8 | 15.2/15.9 | 15.2/15.7 | 14.6/15.9 | 13.4/15.2 |
| R0 | naive | 11.1/12.8 | 11.8/13.9 | 11.8/14.0 | 12.3/14.0 | 12.7/14.6 | 12.9/15.6 | 13.5/16.8 | 14.0/16.0 | 13.1/15.3 |
| R0 | uniform | 10.5/12.3 | 10.4/12.4 | 10.4/12.4 | 10.5/12.6 | 11.4/13.3 | 12.6/14.3 | 13.7/14.6 | 12.9/15.0 | 12.7/14.6 |
| both | CLIP-only | 0.0/0.0 (all tied) | 12.8/14.5 | 12.8/14.5 | 12.8/14.5 | 12.8/14.5 | 12.8/14.5 | 12.8/14.5 | 12.8/14.5 | 12.8/14.5 |

- R3 naive is nearly flat in β: the factor term carries the ranking.
- R0 depends much more on the CLIP term.
- No primary variant selected the upper grid edge.

**Held R@1 / R@3 at each variant's selected β.** Values are in percent. The number in brackets is the count of episodes with a tie against the positive. Chance is 7.7 / 23.1.

| model | variant | β | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art i2t | art t2i |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R3 | naive | .3 | **17.6/40.1** (0) | **20.9/44.5** (0) | 15.2/37.1 (0) | 16.2/36.6 (0) | 20.0/43.1 (0) | 25.6/52.4 (0) |
| R3 | uniform | .1 | 15.6/35.3 (0) | 15.7/38.7 (0) | 12.7/30.1 (0) | 13.3/33.6 (0) | 18.6/40.4 (0) | 18.2/43.8 (0) |
| R0 | naive | .3 | 13.1/33.3 (0) | 16.7/38.2 (0) | 10.4/27.5 (0) | 12.8/31.5 (0) | 15.8/39.2 (0) | 20.6/44.9 (0) |
| R0 | uniform | .3 | 11.9/31.2 (1) | 15.9/36.5 (0) | 9.4/27.9 (0) | 12.4/30.6 (0) | 14.4/34.5 (1) | 19.4/42.4 (0) |
| both | CLIP-only | .001 | 11.5/29.1 (0) | 14.9/37.2 (0) | 9.7/26.2 (0) | 12.3/32.5 (0) | 13.4/31.9 (0) | 17.5/41.8 (0) |

CLIP-only does not depend on the factor model. Its ranks are identical for R0 and R3 (asserted).

**Held R@1 / R@3 at β=0**, in the same format:

| model | variant | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art i2t | art t2i |
|---|---|---:|---:|---:|---:|---:|---:|
| R3 | naive | 18.2/40.7 (19) | 20.9/44.9 (33) | 16.6/37.1 (15) | 15.6/36.1 (23) | 19.7/44.2 (4) | 26.2/53.6 (10) |
| R3 | uniform | 15.6/35.1 (0) | 16.0/38.2 (0) | 12.9/29.5 (0) | 13.7/33.7 (0) | 18.3/40.7 (0) | 18.3/42.8 (0) |
| R0 | naive | 12.0/31.6 (331) | 13.4/32.1 (359) | 10.2/27.1 (201) | 10.1/25.6 (216) | 13.8/36.0 (130) | 16.7/38.7 (143) |
| R0 | uniform | 10.3/30.4 (85) | 13.0/31.9 (85) | 10.0/27.3 (52) | 11.2/27.7 (46) | 10.5/33.4 (33) | 14.7/36.0 (39) |
| both | CLIP-only | 0.0/0.0 (2,048) | 0.0/0.0 (2,048) | 0.0/0.0 (1,024) | 0.0/0.0 (1,024) | 0.0/0.0 (1,024) | 0.0/0.0 (1,024) |

- At β=0, CLIP-only has every score tied. Its zero is a null score, not a measurement.
- R0 naive has many tied episodes at β=0. On held, the naive weight vector is all zero in 5 episodes: 4 emotion and 1 art style. R3 has none.

**Paired bootstraps at the selected β, R@1 points [95% CI]:**

| comparison | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art i2t | art t2i |
|---|---:|---:|---:|---:|---:|---:|
| (a) R3 naive − R0 naive | +4.5 [+2.7, +6.3] | +4.2 [+2.2, +6.2] | +4.9 [+2.5, +7.1] | +3.4 [+0.8, +6.1] | +4.2 [+1.6, +6.9] | +5.0 [+2.1, +8.0] |
| (b) R3 naive − R3 uniform | +2.0 [+0.5, +3.6] | +5.2 [+3.5, +6.8] | +2.5 [+0.4, +4.7] | +2.9 [+0.7, +5.1] | +1.5 [−0.7, +3.6] | +7.4 [+5.1, +9.9] |
| (c) R3 naive − CLIP-only | +6.1 [+4.3, +7.9] | +6.0 [+4.2, +8.0] | +5.6 [+3.2, +7.9] | +3.9 [+1.5, +6.3] | +6.6 [+3.9, +9.4] | +8.1 [+5.2, +11.0] |
| context: R0 naive − R0 uniform | +1.2 [−0.05, +2.4] | +0.8 [−0.4, +2.0] | +1.0 [−0.6, +2.6] | +0.4 [−1.3, +2.0] | +1.5 [−0.4, +3.3] | +1.2 [−0.5, +2.9] |
| context: R0 naive − CLIP-only | +1.6 [+0.1, +3.0] | +1.8 [+0.4, +3.2] | +0.7 [−1.0, +2.4] | +0.5 [−1.3, +2.2] | +2.4 [0.0, +4.8] | +3.1 [+0.9, +5.4] |
| context: R3 uniform − R0 uniform | +3.8 [+2.1, +5.4] | −0.2 [−1.8, +1.4] | +3.3 [+1.3, +5.4] | +0.9 [−1.1, +2.9] | +4.2 [+1.8, +6.7] | −1.3 [−3.7, +1.2] |

**The same comparisons for R@3:**

| comparison | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art i2t | art t2i |
|---|---:|---:|---:|---:|---:|---:|
| (a) R3 naive − R0 naive | +6.7 [+4.5, +8.9] | +6.3 [+4.1, +8.5] | +9.6 [+6.4, +12.6] | +5.1 [+2.1, +8.3] | +3.9 [+0.9, +7.1] | +7.5 [+4.4, +10.6] |
| (b) R3 naive − R3 uniform | +4.8 [+3.0, +6.7] | +5.9 [+3.9, +7.8] | +7.0 [+4.2, +9.9] | +3.0 [+0.3, +5.7] | +2.6 [0.0, +5.3] | +8.7 [+6.1, +11.5] |
| (c) R3 naive − CLIP-only | +11.0 [+8.8, +13.2] | +7.4 [+5.1, +9.6] | +10.9 [+7.9, +14.0] | +4.1 [+1.1, +7.3] | +11.1 [+7.9, +14.5] | +10.6 [+7.3, +14.1] |

**Paired bootstraps at β=0, R@1 points:**

| comparison | pooled i2t | pooled t2i | emotion i2t | emotion t2i | art i2t | art t2i |
|---|---:|---:|---:|---:|---:|---:|
| (a) R3 naive − R0 naive | +6.2 [+4.2, +8.2] | +7.5 [+5.5, +9.7] | +6.4 [+3.6, +9.3] | +5.6 [+2.9, +8.3] | +6.0 [+3.0, +8.8] | +9.5 [+6.4, +12.5] |
| (b) R3 naive − R3 uniform | +2.6 [+0.9, +4.2] | +4.9 [+3.2, +6.6] | +3.7 [+1.4, +6.1] | +2.0 [−0.3, +4.2] | +1.5 [−0.8, +3.8] | +7.9 [+5.4, +10.4] |
| context: R0 naive − R0 uniform | +1.7 [+0.4, +2.9] | +0.4 [−0.8, +1.7] | +0.2 [−1.7, +2.0] | −1.2 [−2.8, +0.5] | +3.2 [+1.5, +5.1] | +2.0 [+0.2, +3.8] |
| context: R3 uniform − R0 uniform | +5.3 [+3.4, +7.2] | +3.0 [+1.1, +4.9] | +2.9 [+0.5, +5.4] | +2.4 [0.0, +5.0] | +7.7 [+4.9, +10.5] | +3.5 [+0.8, +6.2] |

Comparison (c) at β=0 is degenerate, because CLIP-only is all tied. Its point estimate equals R3 naive's R@1 (+18.2 / +20.9).

### Step 2: factor-mined episodes (secondary; unpaired between models)

**Protocol.** Unchanged from Task 7 of the condition-interface plan. Each model mines from its own codes, so the two models have different episodes.
- Episodes: `mine_episodes`, `EpisodeMiningConfig(seed=42)`. 4,096 on val rows and 1,024 on held rows, mined split-locally under single-threaded BLAS, then remapped to global rows.
- Every episode has 13 candidates: the positive, then 4 hard negatives, 4 condition-only distractors and 4 anchor-only distractors.
- All 32 factors appear as targets. The target sequence is identical for both models, because every factor is a valid target for both.
- Episode SHA-256 (held): R0 `6cf0286c…ecf9`, R3 `675bf3ae…6554`.

**β and k, chosen on the 4,096 val episodes:**

| model | naive | top-1 | top-3 | top-5 | uniform | oracle | CLIP-only | chosen k (val utility k1 / k3 / k5) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| R3 | 1 | **3** | **3** | 1 | .03 | **3** | .001 | 5 (32.6 / 33.1 / 33.3) |
| R0 | .1 | .3 | .1 | .1 | .1 | .3 | .001 | 5 (43.9 / 48.8 / 50.4) |

Bold marks a β at the upper edge of the grid. The grid was not extended (controller ruling). The variants the criteria use (naive, uniform) and the chosen top-5 are not at the edge.

**Held R@1 / R@3, in percent.** The bracket gives (episodes with any tie against the positive / episodes with all candidates tied).

| model | variant | selected β | selected: i2t | selected: t2i | β=0: i2t | β=0: t2i |
|---|---|---:|---:|---:|---:|---:|
| R3 | naive | 1 | 32.3/56.4 (0/0) | 34.0/55.4 (21/0) | 29.2/57.1 (0/0) | 34.1/59.4 (20/0) |
| R3 | top-5 | 1 | 31.3/57.7 (0/0) | 35.0/58.9 (21/0) | 27.5/56.7 (7/2) | 32.6/60.4 (24/2) |
| R3 | top-3 | 3 | 31.4/56.3 (0/0) | 33.6/56.0 (21/0) | 26.9/56.4 (16/3) | 30.9/60.3 (32/8) |
| R3 | top-1 | 3 | 29.9/61.3 (0/0) | 34.5/61.9 (21/0) | 22.3/53.1 (63/26) | 26.3/56.2 (76/45) |
| R3 | uniform | .03 | 10.9/21.8 (0/0) | 11.3/22.2 (20/0) | 11.3/21.6 (0/0) | 11.2/22.5 (20/0) |
| R3 | oracle | 3 | 31.8/63.9 (0/0) | 35.7/64.2 (21/0) | 23.7/56.1 (38/13) | 27.8/58.5 (58/33) |
| R3 | CLIP-only | .001 | 8.3/22.6 (0/0) | 8.6/20.3 (21/0) | 0.0/0.0 (all tied) | 0.0/0.0 (all tied) |
| R0 | naive | .1 | 53.5/82.6 (0/0) | 39.2/77.0 (0/0) | 51.8/79.1 (0/0) | 37.0/74.6 (0/0) |
| R0 | top-5 | .1 | 55.2/83.0 (0/0) | 47.5/80.2 (0/0) | 53.6/80.9 (0/0) | 45.4/79.9 (0/0) |
| R0 | top-3 | .1 | 52.5/82.2 (0/0) | 47.6/79.3 (0/0) | 51.4/79.7 (0/0) | 46.2/79.0 (0/0) |
| R0 | top-1 | .3 | 49.0/78.7 (0/0) | 45.3/74.9 (0/0) | 46.8/74.6 (3/3) | 44.4/74.7 (4/4) |
| R0 | uniform | .1 | 52.6/80.2 (0/0) | 39.5/74.7 (0/0) | 50.2/76.1 (0/0) | 34.7/71.2 (0/0) |
| R0 | oracle | .3 | 45.1/75.8 (0/0) | 42.2/75.0 (0/0) | 40.8/71.8 (0/0) | 31.7/70.4 (0/0) |
| R0 | CLIP-only | .001 | 12.2/36.0 (0/0) | 21.2/42.0 (0/0) | 0.0/0.0 (all tied) | 0.0/0.0 (all tied) |

About 20 of R3's held t2i episodes are tied in every variant, including CLIP-only. A candidate there shares the positive's painting, so it has the identical image (see Caveats).

**Paired bootstraps on each model's own held mined episodes, points [95% CI]:**

| model | comparison | β | R@1 i2t | R@3 i2t | R@1 t2i | R@3 t2i |
|---|---|---|---:|---:|---:|---:|
| R3 | naive − uniform | selected | **+21.4 [+18.5, +24.2]** | +34.7 [+31.6, +37.7] | **+22.7 [+19.8, +25.6]** | +33.2 [+30.2, +36.2] |
| R3 | naive − uniform | 0 | +17.9 [+14.9, +20.7] | +35.5 [+32.3, +38.9] | +22.9 [+19.9, +25.9] | +36.9 [+33.7, +40.1] |
| R3 | naive − CLIP-only | selected | +24.0 [+21.0, +27.1] | +33.9 [+30.5, +37.1] | +25.4 [+22.6, +28.3] | +35.1 [+31.9, +38.4] |
| R3 | top-5 − uniform | selected | +20.4 [+17.5, +23.4] | +35.9 [+32.7, +39.2] | +23.6 [+20.6, +26.8] | +36.7 [+33.6, +39.9] |
| R0 | naive − uniform | selected | +0.9 [−1.1, +2.8] | +2.4 [+0.8, +4.2] | −0.3 [−2.1, +1.6] | +2.2 [+0.3, +4.2] |
| R0 | naive − uniform | 0 | +1.6 [+0.3, +2.8] | +3.0 [+1.8, +4.3] | +2.3 [+1.3, +3.5] | +3.4 [+2.2, +4.6] |
| R0 | naive − CLIP-only | selected | +41.3 [+37.9, +44.8] | +46.6 [+43.2, +50.0] | +18.0 [+14.6, +21.5] | +35.0 [+31.4, +38.6] |
| R0 | top-5 − uniform | selected | +2.5 [+0.2, +4.9] | +2.8 [+0.9, +5.0] | +8.0 [+5.5, +10.5] | +5.5 [+3.0, +7.9] |

**Swap reversal on `choose_swap_pairs` pairs (held).** Counts are i2t / t2i, each variant at its selected β.

| model | pairs (distinct B episodes) | naive | top-1 | top-3 | top-5 | uniform | oracle | CLIP-only | naive at β=0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 | 256 (224) | 160/167 | 196/202 | 155/168 | 178/191 | 0/0 | 204/207 | 0/0 | 204/201 |
| R0 | 9 (7) | 9/4 | 9/9 | 9/9 | 9/9 | 0/0 | 9/9 | 0/0 | 8/4 |

The counts from `conditional_score` equal those from Task 9's `score_pool`-based `swap_reversal` for every variant and both models.

**Per-role breakdown (Task 9 Q2 format).** Each cell is the percentage of held episodes in which at least one member of a role strictly outranks the positive, written hard-negative / condition-only / anchor-only. "Naive β" is the naive variant's selected β: 1 for R3 and .1 for R0.

| model | variant | β=0 i2t | β=0 t2i | naive β i2t | naive β t2i |
|---|---|---:|---:|---:|---:|
| R3 | naive | 10.0/59.0/30.0 | 9.7/52.6/29.2 | 15.5/49.1/45.2 | 14.1/42.6/43.5 |
| R3 | top-5 | 9.5/63.8/22.8 | 8.2/57.2/22.1 | 12.6/55.8/35.6 | 11.0/49.2/33.8 |
| R3 | top-3 | 8.0/67.6/16.5 | 6.8/60.7/16.5 | 10.8/61.2/27.8 | 9.3/53.2/26.2 |
| R3 | top-1 | 5.4/72.8/5.0 | 3.9/66.3/3.6 | 9.7/69.4/12.7 | 8.1/63.2/12.2 |
| R3 | uniform | 38.0/21.1/87.0 | 35.4/17.3/86.5 | 55.0/28.5/88.7 | 43.9/21.8/88.1 |
| R3 | oracle | 2.4/75.0/2.1 | 1.3/68.9/1.1 | 5.8/70.4/8.9 | 5.3/64.8/8.5 |
| R3 | CLIP-only | all tied | all tied | 68.0/39.3/88.5 | 57.2/31.2/89.3 |
| R0 | naive | 1.1/48.0/1.2 | 8.4/60.3/10.1 | 0.8/46.3/2.2 | 8.8/56.8/13.3 |
| R0 | top-5 | 0.8/46.3/0.7 | 6.9/52.9/8.5 | 1.0/44.7/1.8 | 8.7/49.7/10.2 |
| R0 | top-3 | 1.0/48.5/0.9 | 7.2/51.8/8.3 | 0.8/47.5/1.6 | 9.0/49.2/10.6 |
| R0 | top-1 | 0.9/52.7/0.9 | 9.6/52.3/10.1 | 1.3/50.6/2.1 | 9.7/50.9/10.7 |
| R0 | uniform | 2.3/49.0/2.9 | 9.0/63.0/11.5 | 3.9/45.4/7.3 | 11.3/55.8/18.9 |
| R0 | oracle | 0.4/59.2/0.0 | 2.2/68.3/2.1 | 0.3/55.5/0.6 | 3.3/62.5/4.4 |
| R0 | CLIP-only | all tied | all tied | 70.5/53.8/74.5 | 56.9/52.5/62.6 |

- On R3's episodes, anchor-only distractors defeat any scorer that does not use the condition: uniform, CLIP-only, and naive with a large CLIP weight.
- Condition-only distractors defeat the oracle, as they did in Task 9.
- On R0's collapsed codes, anchor-only distractors are rarely competitive.

**Mined episode diagnostics.** Roles are distinct rows, not distinct paintings; this protocol is unchanged.

| model | part | episodes | with a repeated painting group among the 22 roles | with a candidate from the anchor's or positive's painting | naive all-zero weights |
|---|---|---:|---:|---:|---:|
| R3 | val | 4,096 | 3,350 | 157 | 0 |
| R3 | held | 1,024 | 791 | 34 | 0 |
| R0 | val | 4,096 | 2,020 | 13 | 0 |
| R0 | held | 1,024 | 331 | 2 | 0 |

## Caveats

1. **Selection used val episodes of the same two condition types.** Task 6 selected R3 by condition lift on val emotion and art-style episodes, and β was also chosen on val episodes of those types.
   - The held episodes use unseen paintings, so criterion 1 is a fair test for *emotion and style* conditions.
   - It is not evidence about arbitrary human-stated conditions. That is stage (e)'s human-judged set.
2. **The gates were amended after the fact.** Task 6's pre-registered rule stopped with no passing run. The user then relaxed two thresholds, on val results only (plan amendment `d490b96`):
   - readout from the absolute CLIP PCA-10 floor to "no worse than R0";
   - the sparsity cap from 37.5% to 50%.

   Under the original gates, R3 passes 7 of 9, on val and on held. R3 was chosen over R6 by the tie-break (mean readout, a 0.0012 margin). R6 scored 0.82 points higher on val. The held episodes here were not built or used in making those decisions. This report is the independent test of the condition benefit, but only for the recipe that the amended rule picked.
3. **One seed per model.** R0 and R3 are each a single seed-42 model. Task 6's replication seeds (43, 44) were not evaluated here, and Task 6 found their factor axes only partly stable.
   - The bootstrap CIs resample episodes under fixed models, split and β. They do not cover training randomness or the choice of split.
   - Episodes reuse rows and paintings *across* episodes. Within one label episode, groups are distinct.
4. **Emotion labels are per annotation, not per painting.** Each painting has about 5 annotations, and each annotator chose an emotion.
   - Clean negatives (Ruling 9) exclude contrasts and distractors from any painting that carries the target emotion in any row.
   - The positive is still one annotator's label, and emotion is subjective. This lowers the ceiling for every method and adds label noise to both the supports and the positives.
   - Art style is per painting and does not have this issue.
5. **Absolute levels are low.** The best held pooled R@1 is 20.9% (R3 naive, t2i). The episodes ask for a *different* painting with the same label, and CLIP alone reaches 11.5% / 14.9%.
6. **Factor-mined results are unpaired and partly a construction effect.** Each model mines its own episodes, so R0-vs-R3 comparisons there are descriptive only.
   - The anchor-only role behaves very differently on decorrelated and collapsed codes: uniform weights let it outrank the positive in 88-89% of R3's held episodes (at R3's naive β), against 7-19% of R0's.
   - The mined protocol also allows several annotation rows of one painting in an episode: 77% of R3's held episodes and 32% of R0's.
   - In 34 of R3's held episodes (2 of R0's), a candidate comes from the anchor's or the positive's painting.
   - In t2i, such a candidate has the identical image, so it ties exactly and counts against the positive. This produces about 20 tied R3 t2i episodes in every variant.
   - The protocol was kept unchanged for comparability, as the brief requires. The episode-role redesign is deferred to the stage (d) plan.
7. **Swap pairs.** Only 9 of R0's held episodes form valid swap pairs (Task 9 had 61, and R3 has 256). R0's swap rates therefore rest on 9 pairs.
8. **β at the grid edge.** R3's mined top-1, top-3 and oracle variants select β=3, the upper edge of the grid. Following the controller ruling, the grid was not extended. None of these variants is used by a criterion.
9. **The Task 9 column uses a different protocol.** It was copied from its report, not recomputed:
   - a row split (99.7% painting overlap with training);
   - β chosen on mined *train* episodes;
   - emotion-only human-label episodes with CLIP-nearest distractors and possibly ambiguous negatives.
10. **Uses of held rows.** This is the second and last use of held rows, after Task 6's held gate check.
    - A smoke run preceded the real run. It used val rows in place of held rows and computed no held metric.
    - Like the real run, it encoded all rows, held included, only to assert that every code is finite.
    - One code change followed the smoke run: recording "no valid swap pairs" instead of crashing. No number from the smoke run informed any choice.
    - Later, the final review re-scored the same held label episodes for its post-hoc interaction analysis (above). That use came after every decision and informed none. It is not repeated here.
11. **Criterion 1 is a conjunction with no interaction test.** It requires (a) R3 naive > R0 naive and (b) R3 naive > R3 uniform, each with a CI above 0. Both can hold while R3 uses the condition no better than R0: (a) can come from a better space overall, and (b) from a condition benefit that R0 shares. Only the interaction (R3 naive − R3 uniform) − (R0 naive − R0 uniform) isolates better use of the condition, and the pre-registered criteria did not include it. The final review's post-hoc interaction analysis (above) finds it in text→image only.

## Reproduction

```bash
# from the repository root (about 1.5 min on an RTX 3090); needs Task 6's gitignored checkpoints and results/*.json
/root/miniconda3/envs/CoSiR/bin/python src/test/20261012_condition_eval_repaired_factors/run_eval.py
# tables only, from results/eval_results.json
/root/miniconda3/envs/CoSiR/bin/python src/test/20261012_condition_eval_repaired_factors/run_eval.py --tables
```

The results JSON and `run_eval.log` are gitignored. The run log is `src/test/20261012_condition_eval_repaired_factors/20261012_condition_eval_repaired_factors_log.md`.
