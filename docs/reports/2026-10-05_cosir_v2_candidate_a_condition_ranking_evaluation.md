# CoSiR v2 Candidate A: item-disjoint condition ranking evaluation

**Verdict: the current factor space can support conditional ranking on mined episodes, but the trained condition head is a poor interface to it.** At the scale-matched evaluation point, `β=0.3`, the naive support-minus-contrast rule beats the uniform control by **+17.38/+11.52 percentage points** in i2t Recall@1/@3 and **+9.67/+8.59 points** in t2i. The one-hot true-factor control also beats uniform. The trained head underperforms the naive rule in all four metrics and is nearly indistinguishable from using another episode's condition. Thus the previous ~23% exact-factor recovery score was an incomplete proxy: useful retrieval signal exists even though the true factor is often not the largest support/contrast gap. These are automatically mined, factor-defined candidates; this does not establish human-judged retrieval quality or a trained stage-(d) scorer.

The true factor has **median within-episode gap rank 4/32**, appears in the top three in **456/1,024 (44.53%)** episodes, and the top ten in **870/1,024 (84.96%)**. A cross-factor comparison module could use this relative signal, but these numbers do **not** make cross-factor attention a well-targeted *next* fix by themselves: more than half the targets are outside the top three, while the naive full weight vector already ranks candidates substantially better than the trained head. The immediate measured failure is the head's almost condition-invariant, very large output scale; simply forcing a better exact-factor argmax would chase the proxy rather than the ranking result.

On 61 strictly constructed two-condition pairs at `β=0.3`, the desired positive-order reversal occurred for **naive weights in 16/61 i2t and 15/61 t2i**, and for the true-factor one-hot control in **54/61 and 56/61**. It occurred only **1/61 and 3/61** for the trained head, **0/61 and 1/61** for its shuffled-condition control, and never for uniform weights. The CLIP term therefore does not universally suppress the factor term; the naive/oracle factors can change the ordering appropriately, whereas the present learned head rarely does.

## Held-out ranking at the scale-matched point (`β=0.3`)

Every row uses the **same 1,024 held-out episodes** and all 13 candidates per episode: one positive, four hard negatives, four condition-only distractors, and four anchor-only distractors. The positive is at candidate position zero; rank is one plus the number of candidates scoring strictly above it. Recall@3 counts ranks 1–3. Percentages below include the exact successful-episode counts.

| Condition weights | i2t Recall@1 | i2t Recall@3 | t2i Recall@1 | t2i Recall@3 |
|---|---:|---:|---:|---:|
| Trained `ConditionEncoder` | 507/1,024 (49.51%) | 774/1,024 (75.59%) | 436/1,024 (42.58%) | 677/1,024 (66.11%) |
| Naive `ReLU(support_mean − contrast_mean)` | **590/1,024 (57.62%)** | **846/1,024 (82.62%)** | **499/1,024 (48.73%)** | **751/1,024 (73.34%)** |
| Uniform `1/32` | 412/1,024 (40.23%) | 728/1,024 (71.09%) | 400/1,024 (39.06%) | 663/1,024 (64.75%) |
| Shuffled condition, trained head | 507/1,024 (49.51%) | 771/1,024 (75.29%) | 434/1,024 (42.38%) | 675/1,024 (65.92%) |
| Oracle one-hot true factor | 560/1,024 (54.69%) | 839/1,024 (81.93%) | 496/1,024 (48.44%) | 751/1,024 (73.34%) |

The complete positive-rank distribution at this `β` is shown compactly below. Each group sums to 1,024; the script emits each episode's exact rank for every condition variant, direction, and `β`.

| Condition weights | i2t ranks 1 / 2 / 3 / 4–13 | t2i ranks 1 / 2 / 3 / 4–13 |
|---|---:|---:|
| Trained | 507 / 137 / 130 / 250 | 436 / 120 / 121 / 347 |
| Naive | 590 / 136 / 120 / 178 | 499 / 158 / 94 / 273 |
| Uniform | 412 / 179 / 137 / 296 | 400 / 154 / 109 / 361 |
| Shuffled trained | 507 / 138 / 126 / 253 | 434 / 119 / 122 / 349 |
| Oracle | 560 / 163 / 116 / 185 | 496 / 163 / 92 / 273 |

Using the same episodes for each comparison, a seed-42, 5,000-resample paired episode bootstrap gives naive-minus-uniform differences of **+17.38 points** (95% interval **+14.16 to +20.51**) for i2t Recall@1, **+11.52** (**+8.69 to +14.55**) for i2t Recall@3, **+9.67** (**+6.54 to +12.89**) for t2i Recall@1, and **+8.59** (**+5.76 to +11.52**) for t2i Recall@3. These intervals describe episode sampling under this fixed split and fixed factor model; mined episodes can reuse held-out items, so they are not item-clustered confidence intervals.

## `β` sensitivity and score scale

The score was evaluated **read-only** as `β · cos(CLIP_I(I), CLIP_T(T)) + Σ_l w_l(c) · a_I,l(I) · a_T,l(T)` in both directions. We evaluated `β ∈ {0, 0.03, 0.3}` without selecting an outcome after tuning. `β=0.3` is the scale-matched point for the untrained naive rule: mean candidate CLIP cosine is **0.1902**, so its mean CLIP contribution is about **0.0570**, close to the naive factor term's mean absolute magnitude **0.0728**. At the same point the mean absolute factor terms are **0.0128** for uniform, **0.0276** for one-hot oracle, and **73.28** for the trained head (**71.15** when its condition is shuffled). No single fixed `β` can balance all five unnormalized weight constructions. The trained head's factor term overwhelms CLIP at every tested `β`; the beta comparison must therefore be read with that scale limitation in view.

| `β` | Weights | i2t R@1 / R@3 | t2i R@1 / R@3 |
|---:|---|---:|---:|
| 0 | Trained | 49.51% / 75.49% | 42.58% / 66.11% |
| 0 | Naive | 52.83% / 77.15% | 46.78% / 68.75% |
| 0 | Uniform | 49.32% / 75.29% | 42.38% / 66.02% |
| 0 | Shuffled trained | 49.41% / 75.29% | 42.29% / 65.92% |
| 0 | Oracle | 42.87% / 73.63% | 38.18% / 64.65% |
| 0.03 | Trained | 49.51% / 75.49% | 42.58% / 66.11% |
| 0.03 | Naive | 53.03% / 77.93% | 47.07% / 69.04% |
| 0.03 | Uniform | 52.93% / 78.81% | 45.31% / 68.85% |
| 0.03 | Shuffled trained | 49.41% / 75.29% | 42.29% / 65.92% |
| 0.03 | Oracle | 46.78% / 76.46% | 39.45% / 68.46% |
| 0.3 | Trained | 49.51% / 75.59% | 42.58% / 66.11% |
| 0.3 | Naive | **57.62% / 82.62%** | **48.73% / 73.34%** |
| 0.3 | Uniform | 40.23% / 71.09% | 39.06% / 64.75% |
| 0.3 | Shuffled trained | 49.51% / 75.29% | 42.38% / 65.92% |
| 0.3 | Oracle | 54.69% / 81.93% | 48.44% / 73.34% |

At `β=0.03`, naive and uniform are close, with naive slightly worse on i2t Recall@3; the conclusion that naive is **meaningfully above uniform** depends on putting the CLIP and naive factor terms on comparable scales. At `β=0`, naive has a smaller Recall@1 advantage in both directions. The trained head's results barely change across the sweep, consistent with its outsized factor score. One-hot knowledge of the mined target is an **oracle for factor identity**, not a mathematical upper bound on retrieval Recall: another high-condition distractor may score above the randomly chosen high-condition positive, and other factors can help ranking. This explains why naive can match or exceed the one-hot row.

## Target factor within each episode

For every held-out episode, the script computes all 32 dimensions of `mean(pair_codes[support]) − mean(pair_codes[contrast])`, with `pair_codes = 0.5 · (image_codes + text_codes)`. The target's rank is `1 +` the number of strictly larger gaps. This measure uses no `w(c)` variant and no retrieval score.

| Target gap rank | Episodes | Fraction |
|---|---:|---:|
| 1 | 245 | 23.93% |
| 2 | 122 | 11.91% |
| 3 | 89 | 8.69% |
| 4–10 | 414 | 40.43% |
| 11–32 | 154 | 15.04% |
| **Median / top-3 / top-10** | **4 / 456 / 870** | **44.53% top-3 / 84.96% top-10** |

The naive rule's exact target argmax was **245/1,024 (23.93%)** and the trained head's was **239/1,024 (23.34%)**. The naive rule nevertheless reached 57.62%/48.73% Recall@1 at the scale-matched score. Exact target classification therefore throws away useful information in the rest of the weight vector. The rank distribution leaves room for a relative-factor mechanism to help, but it does not specifically validate attention: target rank is often 4–10, and the desired retrieval signal can already be used without making that factor the single largest weight.

## Same-anchor condition swap

Pair selection is deterministic from seed 42. The script permutes held-out episode indices and, for each episode A, searches a shuffled list for the first episode B with a different target factor. High and low mean the held-out pool's own 90th and 50th percentile cutoffs, exactly as in mining. A's anchor must be high on **both** factors; A's positive must be low on B's factor; B's positive must be low on A's factor. Both positives are high on their own factor by mining. B's support, contrast, and positive must exclude A's anchor. These conditions make both support/contrast conditions operationally valid for the same anchor while giving the two positives opposing factor labels. No alternative condition is inferred from the original episode's distractors.

The fixed comparison pool is the de-duplicated union of A's and B's positives and all six distractor-role lists, always scored against **A's same anchor**. It is rescored with A's and B's respective weights, with the reverse image/text roles for t2i. An *appropriate reversal* requires `score(A-positive | A-condition) > score(B-positive | A-condition) + 1e-8` **and** `score(B-positive | B-condition) > score(A-positive | B-condition) + 1e-8`. It measures a strict pairwise positive-order reversal, not necessarily a change in the top-1 item of the larger pool. Uniform uses the same weights under both conditions and therefore cannot reverse.

The search sought up to 256 A episodes and found **61** qualifying pairs among 1,024 held-out episodes; all 61 are reported. A episodes are unique, while B episodes can repeat (**30 distinct B episodes**), so swap rates are descriptive and not 61 independent-pair estimates.

| Weights | Appropriate i2t reversal | Appropriate t2i reversal | Any full-pool rank-order change, i2t / t2i |
|---|---:|---:|---:|
| Trained | 1/61 (1.64%) | 3/61 (4.92%) | 49.18% / 45.90% |
| Naive | **16/61 (26.23%)** | **15/61 (24.59%)** | 100% / 100% |
| Uniform | 0/61 | 0/61 | 0% / 0% |
| Shuffled trained | 0/61 | 1/61 (1.64%) | 36.07% / 36.07% |
| Oracle | **54/61 (88.52%)** | **56/61 (91.80%)** | 100% / 100% |

These are `β=0.3` results. The script also emits swap results for `β=0` and `0.03`: naive's appropriate reversal is **24.59% / 27.87%** and **22.95% / 26.23%** (i2t / t2i), respectively; oracle is **100% / 100%** at both. Thus factor-only scores already reverse under the right conditions, and the small naive rate is not an artifact of the CLIP term. The trained head changes the overall order in some pairs, but almost never makes the two condition-specific positives exchange order correctly.

## Reproduction and limits

Run `/root/miniconda3/envs/CoSiR/bin/python src/test/20261005_condition_ranking_evaluation/run_ranking_eval.py` from the repository root. The script uses Task 3's `FeatureManager`/`artelingo_train.json` positional join and validates 308,723 unique sample IDs. A seed-42 permutation assigns **246,978 items** to training and **61,745 unseen items** to evaluation. The content graph (**2,463,679 edges**), Stage 1 diagnostic (**21 communities**), and 32-factor `SharedFactorEncoder` are fit on training items only, with raw CLIP features, `lambda_usage_balance=0.1`, no whitening, and the existing 2,000-epoch factor-training settings. The frozen factor model then encodes held-out items. No held-out item enters graph construction, Stage 1 fitting, factor fitting, or condition-head training.

The unchanged miner draws **4,096** episodes from training items and **1,024** separately from held-out items, using its default 4-support/4-contrast/4-each-distractor sizes, 90th/50th percentile pools, and seed 42. The script checks that every role is unique within an episode, every role stays in its assigned item split, and all 13 candidate slots are present. It trains only `ConditionEncoder` using Task 5's validation-only `cross_entropy(w(c), targeted_factor)` objective: inverse-class-frequency replacement sampling, batch size 64, Adam learning rate 0.01, 300 epochs, seed 42. The shuffled-condition control applies this **same trained head** to another held-out episode's support/contrast; a randomized cyclic assignment guarantees no episode supplies its own shuffled condition. Uniform is exactly `ones(32)/32`; oracle is exactly one-hot. There is no swap-loss, scorer fitting, or architecture/miner change.

The factor target and candidate categories are **defined by the mined factor codes**, and the positive is a random item in the target's high pool. A high-pool condition-only distractor may be equally valid under the named factor; this is not a human relevance test. Factor fitting is truly item-disjoint here, but held-out episodes reuse items within the held-out pool, and all results come from one seed, one fixed code model, and one synthetic mining rule. `β` is uncalibrated; the raw weight scales differ sharply. Those limits preclude a claim that this score is ready for deployment, while the naive/oracle advantages and swap reversals are direct evidence that the factor space contains conditional ranking information for these mined candidates.
