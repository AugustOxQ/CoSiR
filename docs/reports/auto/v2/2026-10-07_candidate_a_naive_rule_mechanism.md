# CoSiR v2 Candidate A: why the naive rule works

**Verdict.** The naive rule has a real but modest condition-specific ranking benefit once every weight vector has the same L1 scale: at train-selected β, its held-out Recall@1 exceeds uniform by **3.6 points i2t and 3.2 points t2i**, with paired 95% intervals above zero. It works chiefly by weighting a **cluster of highly correlated factors**, not by finding one unique target dimension: the median target has 12 factors with train-item |Pearson correlation| ≥0.95, and keeping that cluster nearly preserves the naive result while removing it sharply hurts. This also explains why the single-factor oracle loses on high-target, condition-only distractors. A smaller naive-over-uniform gain survives episodes defined by human emotion labels instead of the factor miner, but this is **not a clean non-circular generalization test**: 61,567/61,745 held annotation rows depict paintings also present in factor training, and the CLIP-nearest negative construction makes CLIP-only Recall zero. The head architecture can reproduce naive almost exactly. Task 7's head failure is an optimization/interface failure of recovery training from scratch: recovery CE from a naive-mimicking start largely preserves retrieval and improves swaps, while direct ranking InfoNCE gives much stronger swap sensitivity and t2i ranking.

## Q1 — scale-fair comparison

All six weights are L1-normalized **per episode**; zero vectors remain zero. β was selected separately for each variant by mean train i2t/t2i R@1 on the original 4,096 train episodes, from `{0, .001, .003, .01, .03, .1, .3, 1, 3}`; ties choose the smaller β, and the upper grid would extend if selected. The table uses the same 1,024 hashed held episodes and both directions. Cells show R@1/R@3 percent. Swap counts are appropriate reversals among Task 7's same 61 pairs, at each variant's selected β.

| Weights | Selected β | β=0 i2t | β=0 t2i | Selected i2t | Selected t2i | Swap i2t/t2i |
|---|---:|---:|---:|---:|---:|---:|
| Naive | .1 | 52.8/77.1 | 46.8/68.8 | **58.1/82.8** | **49.6/73.3** | 13/15 |
| Uniform | .1 | 49.3/75.3 | 42.4/66.0 | 54.5/83.1 | 46.4/70.4 | 0/0 |
| Oracle one-hot | .3 | 42.9/73.6 | 38.2/64.6 | 54.7/81.9 | 48.4/73.3 | 54/56 |
| CLIP-only | .001 | 0.0/0.0* | 0.0/0.0* | 13.7/36.7 | 21.7/45.9 | 0/0 |
| Task 7 trained head | .1 | 49.5/75.5 | 42.6/66.1 | 55.3/82.7 | 46.2/70.4 | 0/0 |
| Shuffled trained condition | .1 | 49.4/75.3 | 42.3/65.9 | 54.4/83.0 | 46.0/70.4 | 0/0 |

At β=0, rank is `1 + #strictly higher + 0.5 × #tied distractors`. No Q1 variant except CLIP-only had a held positive-score tie. CLIP-only had **all 13 scores tied in all 1,024 episodes in each direction** (12,288 tied distractors per direction), giving midrank 7 and the starred zero R@1/R@3. This is an all-tied null score, **not** evidence that CLIP is worse than chance at β=0; expected random tie-breaking would give 1/13 and 3/13. The table retains the requested mechanical β=0 calculation, with the tie count exposed.

The next table gives paired episode-bootstrap differences in percentage points, `point [95% interval]`, with 5,000 seed-42 resamples. “Selected” compares each variant at its *own train-selected β*. The β=0 naive–CLIP-only row is mathematically defined by midranks but substantively uninformative because CLIP-only is all tied.

| Naive minus | i2t R@1 | i2t R@3 | t2i R@1 | t2i R@3 |
|---|---:|---:|---:|---:|
| Uniform, β=0 | +3.5 [+2.0,+5.1] | +1.9 [+0.5,+3.2] | +4.4 [+2.8,+6.0] | +2.7 [+1.4,+4.1] |
| Uniform, selected β | +3.6 [+1.7,+5.6] | −0.3 [−2.1,+1.6] | +3.2 [+1.2,+5.3] | +2.9 [+1.0,+4.9] |
| CLIP-only, β=0 (all tied) | +52.8 [+49.8,+55.9] | +77.1 [+74.5,+79.7] | +46.8 [+43.8,+49.9] | +68.8 [+65.8,+71.6] |
| CLIP-only, selected β | +44.4 [+40.9,+47.9] | +46.1 [+42.5,+49.5] | +27.9 [+24.4,+31.5] | +27.4 [+23.8,+31.0] |

Thus the original 17.4-point naive-over-uniform i2t R@1 gain at common unnormalized β=.3 was largely a scale comparison. The remaining R@1 lift is small but positive in both directions; selected-β i2t R@3 is indistinguishable from uniform under this bootstrap. The trained and shuffled heads remain close after normalization and never reverse the 61 pairs, so their earlier failure was not just excessive output magnitude. These intervals resample episodes, which reuse items and paintings, and should not be read as item- or painting-clustered uncertainty.

## Q2 — which part of the condition helps?

All ablations are normalized after editing weights. “Target-only” is the one-hot oracle. The additional ±cluster rows use the **train-item** pair-code Pearson matrix: for each episode, the target's cluster contains the target and every factor with absolute correlation ≥.95 to it. Cells are held R@1/R@3 percent at β=0 and naive's selected β=.1.

| Weights | β=0 i2t | β=0 t2i | β=.1 i2t | β=.1 t2i |
|---|---:|---:|---:|---:|
| Naive | 52.8/77.1 | 46.8/68.8 | 58.1/82.8 | 49.6/73.3 |
| Target-only | 42.9/73.6 | 38.2/64.6 | 51.4/80.5 | 43.5/71.8 |
| Naive minus named target | 50.3/75.0 | 44.2/66.7 | 55.0/81.2 | 46.1/71.2 |
| Top 1 naive weight | 52.6/75.7 | 50.5/73.3 | 55.6/80.4 | 52.7/77.2 |
| Top 3 | 57.1/80.3 | 54.1/76.6 | 59.7/84.4 | 55.3/79.1 |
| Top 5 | 56.8/80.2 | 54.5/75.5 | **62.1/84.8** | **56.5/78.6** |
| Support mean only | 52.1/76.7 | 46.1/69.0 | 57.9/82.6 | 48.4/73.4 |
| Anchor pair-code profile | 54.0/78.3 | 44.4/67.1 | 58.5/83.6 | 47.5/70.9 |
| Naive minus target-correlated cluster | 34.4/51.5 | 22.8/42.5 | 38.2/59.7 | 25.8/50.8 |
| Naive target-correlated cluster only | 50.2/76.6 | 47.9/69.3 | 56.7/83.3 | 50.8/74.7 |

The factor dictionary is highly redundant: **374/496** distinct factor pairs have train-item |r|≥.90, **189/496** have |r|≥.95, and **93/496** have |r|≥.98. An episode's direct target cluster has median **12** dimensions (range 1–19). Removing that cluster yields 73 zero weight vectors; at β=0, 83 episodes have all 13 scores tied. The near-naive performance of cluster-only weights and the collapse after removing the cluster make the “other independent factors explain everything” reading of the single-target ablation untenable. Top-3/5 improves on full naive, consistent with a sparse, redundant target-correlated signal plus noise from weak weights. Support-only is nearly naive, so contrast subtraction contributes little to this mined ranking; anchor-profile works well i2t and less well t2i, showing that a broad item profile also helps.

Per-role columns below are the percent of episodes where **at least one** of the four role members strictly outranks the positive, ordered hard-negative / condition-only / anchor-only. They are not per-candidate rates. Target-only is identical to oracle. The CLIP-only β=0 zeros mean all scores tied, not a successful ranking.

| Weights | β=0 i2t H/C/A | β=0 t2i H/C/A | β=.1 i2t H/C/A | β=.1 t2i H/C/A |
|---|---:|---:|---:|---:|
| Naive | 1.6/45.6/1.8 | 11.7/49.5/12.7 | 1.7/40.5/2.8 | 12.8/44.9/15.0 |
| Uniform | 2.3/47.7/4.0 | 13.0/52.9/15.0 | 4.3/41.2/7.6 | 16.3/43.8/21.7 |
| CLIP-only | 0/0/0* | 0/0/0* | 69.7/46.5/74.2 | 55.7/47.9/62.7 |
| Task 7 trained | 2.2/47.7/3.8 | 13.1/52.6/15.0 | 4.0/40.7/7.0 | 16.3/44.2/21.5 |
| Shuffled trained | 2.2/47.7/4.0 | 13.2/53.0/15.0 | 4.6/41.3/7.6 | 16.3/44.3/21.8 |
| Oracle/target-only | 0.3/57.1/0.3 | 3.3/61.7/2.0 | 0.5/48.6/0.7 | 5.7/56.2/4.5 |
| Naive minus named target | 4.2/45.2/5.3 | 15.8/48.2/17.2 | 4.6/39.8/6.4 | 16.5/43.9/19.5 |
| Top 1 | 1.2/46.5/1.6 | 12.4/44.2/12.8 | 1.3/43.4/2.1 | 13.3/41.1/13.8 |
| Top 3 | 0.8/42.5/0.8 | 10.7/41.3/10.6 | 0.9/39.2/1.6 | 11.2/39.3/13.3 |
| Top 5 | 1.1/42.2/1.4 | 9.6/41.7/10.1 | 1.0/36.9/1.9 | 10.6/37.8/11.8 |
| Support mean only | 2.0/46.1/2.1 | 12.0/50.1/12.8 | 2.3/40.1/3.4 | 13.6/45.3/15.9 |
| Anchor profile | 1.9/44.0/2.8 | 13.2/50.4/15.1 | 2.3/39.0/4.1 | 13.7/45.2/18.4 |
| Minus target cluster | 6.3/50.8/13.0 | 29.1/56.1/39.4 | 14.8/49.0/21.8 | 33.0/55.5/43.4 |
| Target cluster only | 0.9/49.6/0.6 | 8.7/51.5/8.7 | 1.0/43.1/1.4 | 10.0/46.3/11.4 |

Condition-only distractors dominate the oracle's failures: they are in the same high-target pool as the randomly chosen positive, so knowing the target dimension does not determine their order. The naive/top-k cluster uses small differences across correlated dimensions to separate some of them. Hard negatives and anchor-only distractors are comparatively easy for factor-based weights and much harder for CLIP-only.

## Q3 — circularity checks

**Q3a, seed 43 on fixed seed-42 episodes.** The second 2,000-epoch factor model used the same 246,978 train items and rebuilt the identical 2,463,679-edge train graph. Task 7's seed-42 episodes and 61 swap pairs were left unchanged; all condition weights and factor scores below came from seed-43 codes. β was again selected on the fixed 4,096 seed-42-mined *train* episodes, scored with seed-43 codes. Values are held R@1/R@3 percent and reversals.

| Weights | β | i2t | t2i | Swap i2t/t2i |
|---|---:|---:|---:|---:|
| Naive | .1 | 56.2/81.7 | 47.4/69.5 | 13/12 |
| Uniform | .1 | 54.0/82.4 | 45.4/67.9 | 0/0 |
| CLIP-only | .001 | 13.7/36.7 | 21.7/45.9 | 0/0 |

Seed-43 naive minus uniform R@1 is +2.2 points i2t (5,000-resample paired CI **+0.3 to +4.2**) and +2.0 t2i (**−0.1 to +4.1**). R@3 differences are −0.7 (**−2.4 to +1.1**) and +1.7 (**−0.3 to +3.7**). The factor spaces are too similar for this to be a strong independence test: per seed-42 factor, the maximum absolute held-item Pearson correlation to any seed-43 factor has mean **.982**, median **.995**, minimum **.708**; 20 distinct seed-43 factors are selected as maxima. One-to-one Hungarian matching still gives mean **.977**, median **.993**, minimum **.708**. Per-factor maxima (seed-42 indices in order) are:

| Factors | Maximum absolute Pearson correlations |
|---|---|
| 0–7 | .996, .998, .994, .982, .996, .998, .997, .985 |
| 8–15 | .998, .978, .990, .995, .995, .971, .998, .998 |
| 16–23 | .708, .998, .998, .971, .998, .945, .988, .996 |
| 24–31 | .995, .996, .998, .983, .993, .993, .988, .998 |

**Q3b, human emotion labels instead of factor pools.** `FeatureManager.get_all_sample_ids()` joins each feature row to `annotations[sample_id]`; the annotation field **`painting`** identifies the image (the record also has `image` and an annotation-specific `image_id`). Each of 1,024 held episodes and 4,096 analogous train episodes has 4 same-emotion supports, 4 other-emotion contrasts, a same-emotion anchor and positive, and 12 other-emotion distractors: six exact CLIP-nearest in that query direction and six random. No episode repeats a `painting`. Nearest candidates are drawn from all other-emotion rows in the item split; i2t and t2i therefore have direction-specific nearest candidates. We select β on train episodes by mean bidirectional R@1; CLIP-only excludes the all-tied β=0 and takes the best positive β. Cells are held R@1/R@3 percent; random candidate-order chance is 7.7/23.1.

| Weights | β | i2t | t2i |
|---|---:|---:|---:|
| Naive | 0 | **11.8/30.3** | **8.5/20.7** |
| Uniform | 0 | 9.7/23.7 | 5.3/14.8 |
| CLIP-only | .001 | 0.0/0.0 | 0.0/0.0 |

With 2,000 paired seed-42 episode resamples, naive minus uniform is **+2.1 [+0.3,+4.1]** i2t R@1, **+6.5 [+3.6,+9.5]** i2t R@3, **+3.2 [+1.5,+5.0]** t2i R@1, and **+5.9 [+3.3,+8.5]** t2i R@3, all in percentage points. Naive minus CLIP-only is +11.8 [+9.9,+13.8] / +30.3 [+27.4,+33.0] i2t and +8.5 [+6.7,+10.4] / +20.7 [+18.2,+23.2] t2i; this comparison is weak evidence because the six globally nearest negative captions/images make the CLIP-only row zero on every held episode at positive β. The label-defined naive-over-uniform advantage is the more meaningful result, though its R@1 magnitude is small.

Held episode counts by emotion: amusement **102**, anger **121**, awe **110**, contentment **117**, disgust **115**, excitement **119**, fear **96**, sadness **133**, something else **111** (total 1,024). Painting uniqueness holds *within* episodes, but the original row-level split contains **61,355 train and 41,048 held unique paintings with 41,001 in both**. Thus **61,567/61,745 (99.7%)** held annotation rows have a painting already seen during factor training. An image can also have annotations of different emotions; a negative annotation can therefore depict a painting labeled with the target emotion elsewhere. This check breaks the factor-miner episode recipe but does not establish painting-disjoint or unambiguous human relevance.

## Q4 — why did the learned head fail?

The existing `ConditionEncoder` was fitted by MSE to naive's **raw** train-episode weights. Its held R², with each factor centered separately, is **.999842** (MSE **1.29×10⁻⁶**); after L1 normalization it exactly matches naive's held ranking to the reported precision, **58.1/82.8 i2t and 49.6/73.3 t2i**, and its **13/15** swaps. The shared per-factor architecture therefore has ample capacity. All Q4 ranking and swaps use the fixed train-selected β=.1.

Starting at that MSE solution, the exact Task 7 recovery-CE sampling/optimizer recipe was run for 300 epochs: inverse-class-frequency replacement sampling, batch 64, Adam .01, seed 42. The table logs 13 checkpoints; CE is raw-logit recovery loss, R@1 is held i2t/t2i percent, and swaps are counts out of 61.

| Epoch | Train balanced CE | Held CE | Held R@1 i2t/t2i | Swaps i2t/t2i |
|---:|---:|---:|---:|---:|
| 0 | 3.359 | 3.357 | 58.1/49.6 | 13/15 |
| 25 | 2.384 | 2.314 | 58.1/50.0 | 21/24 |
| 50 | 2.350 | 2.289 | 58.1/49.6 | 21/23 |
| 75 | 2.346 | 2.287 | 58.1/49.5 | 21/23 |
| 100 | 2.345 | 2.285 | 58.1/49.5 | 21/23 |
| 125 | 2.344 | 2.287 | 58.1/49.4 | 21/23 |
| 150 | 2.351 | 2.292 | 58.0/49.8 | 21/23 |
| 175 | 2.344 | 2.293 | 58.0/49.1 | 21/22 |
| 200 | 2.345 | 2.286 | 58.0/49.4 | 21/23 |
| 225 | 2.341 | 2.288 | 57.6/49.1 | 18/22 |
| 250 | 2.350 | 2.303 | 57.6/49.0 | 17/21 |
| 275 | 2.352 | 2.309 | 57.4/49.1 | 18/22 |
| 300 | 2.341 | 2.292 | 57.5/49.1 | 17/21 |

The MSE init's raw held recovery CE is **3.357**, versus **2.439** for Task 7's scratch-trained head, but raw CE is confounded by logit scale. Fitting one positive scalar to each head's *train* logits and applying it to held logits gives **2.448** for the MSE init versus **2.438** for Task 7; their held target argmax rates are **23.93%** and **23.34%**. Warm-start CE lowers raw held loss to **2.292** and improves swaps without sacrificing much ranking. Recovery CE is therefore not inherently destructive to the naive solution; from-scratch recovery training found a worse, nearly condition-invariant solution.

For Q4c, a fresh seed-42 head was trained for 300 epochs on the same 4,096 train episodes with bidirectional 13-candidate InfoNCE, positive slot zero, mean of directions, fixed β=.1, temperature .1, batch 64, Adam .01. The comparison is:

| Weights | Held i2t R@1/R@3 | Held t2i R@1/R@3 | Swaps i2t/t2i |
|---|---:|---:|---:|
| Naive | **58.1/82.8** | 49.6/73.3 | 13/15 |
| Task 7 recovery head | 55.3/82.7 | 46.2/70.4 | 0/0 |
| Naive init → recovery CE, epoch 300 | 57.5/81.9 | 49.1/73.0 | 17/21 |
| Ranking InfoNCE from scratch | 56.1/81.7 | **53.5/76.7** | **41/44** |

Ranking training greatly increases condition-sensitive reversals and t2i ranking, but it does not beat naive i2t R@1 on these same mined candidates. This is evidence for a better training objective, not a held-out human-relevance validation of the trained scorer.

## Implications for stage (d)

Carry a scale-normalized naive rule and train-selected top-k variants as explicit stage-(d) baselines, calibrate β on train episodes, and evaluate both directions and swap reversals. The current 32-factor dictionary is strongly redundant; a stage-(d) model should be checked against correlated-factor pruning rather than assuming 32 independent semantic axes. Train the condition/scoring interface with a ranking objective, then test it on painting-disjoint, label-defined or human-judged episodes before claiming general retrieval quality. The present evidence supports a small condition-specific signal beyond this miner, but not painting-disjoint generalization.

## Reproduction and limits

From the repository root, run:

```bash
/root/miniconda3/envs/CoSiR/bin/python src/test/20261007_naive_rule_mechanism_analysis/run_mechanism.py --section all
```

The first seed-42 preparation, exact Task 8 mining, Task 7 head retraining, and Q1/Q2 run took **290.1 s**; the first seed-43 fit/evaluation took **35.3 s**; a subsequent complete integrated run with ignored local caches took **84.4 s** on an RTX 3090. The cold full run was not separately timed end to end. The fixed split is 246,978 train / 61,745 held annotation rows. Seed 42 is used throughout except the Q3a factor fit (seed 43). Mining runs under single-threaded BLAS; no cuML/cuGraph is used. The ordered held-episode SHA-256 is **`cb50ab7f026678d317dddde274d779674ec5e039921612c999cd19751b74410f`**, **matching Task 8 exactly**. The run also reproduces 2,463,679 train graph edges, 21 Stage 1 communities, and 61 swap pairs with 30 distinct B episodes.

All reported confidence intervals resample episodes under fixed items, factors, and selected β. Repeated paintings/items and a single split/model recipe limit their population meaning. Factor-defined candidate roles remain synthetic, and label-defined negatives can carry ambiguous emotions.

## Controller review addendum (independent verification)

Recomputed from the cached seed-42 codes on the same train split: the pairwise redundancy counts
(374/189/93 of 496 pairs at |r|≥.90/.95/.98) and the painting overlap (41,001 shared paintings;
61,567/61,745 held rows) reproduce exactly. The redundancy is stronger than the pairwise counts
suggest: on the **correlation** matrix of train pair codes, the first eigen-direction holds
**89.4%** of standardized variance (next: 5.2%, 2.4%, 1.6%), participation ratio **1.25**. Its
loadings are nearly equal in magnitude (|0.12–0.24|) with ~21 factors positive and ~10 negative
(factor 2 ≈ 0): the 32 factors are largely one bipolar axis replicated across the dictionary. It
is cross-modal (corr of image-code vs text-code projections .81) and only weakly emotion-related
(emotion means within ±0.17 on a unit-std axis). The factor-discovery validation (usage balance,
dead/modality-private counts, community spanning) measured *usage*, not *independence*, so it
did not detect this. The condition-specific signal studied here therefore lives in the small
residual beyond this axis. Separately, Task 7's "item-disjoint" split is annotation-disjoint,
not painting-disjoint. Also note: Task 9's brief (not the implementation) specified six CLIP-nearest
distractors in Q3b, which forces CLIP-only to zero there by construction; and the top-k rows in
Q2 were compared on held-out episodes without selecting k on train.
