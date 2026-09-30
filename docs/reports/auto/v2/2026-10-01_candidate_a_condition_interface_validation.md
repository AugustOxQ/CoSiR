# CoSiR v2 Candidate A: condition interface and episode mining on ArtELingo

**Verdict: the mining pipeline is usable, but the condition encoder needs a design decision before stage (d).** On 1,024 real mined episodes, all 32 factors were eligible and selected, and no requested role was shortened. Held-out condition recovery reached **45/205 (21.95%)**, above **1/32 (3.125%)** chance, so there is a learnable signal. Recovery is sharply concentrated: **19/32 factors had 0 correct held-out predictions**, while several others reached 67–100% on small denominators. The current interface is therefore not yet a uniformly reliable condition signal for the planned scoring and swap-loss stage.

## Method

Run `src/test/20261001_condition_interface_validation/run_validation.py` from the repository root with the CoSiR environment. The script uses `FeatureManager` at `/data/SSD2/pre_extract/artelingo/features` and validates unique, in-range sample IDs before the positional join with `/data/PDD/artelingo/artelingo_train.json`. It loaded all **308,723** training pairs. Following the Task 6 recipe, it rebuilt the raw-CLIP content graph (**3,130,544** edges), trained default Stage 1, detected **21** communities, and trained `SharedFactorEncoder` inline on the **raw, unwhitened 512-dimensional** image and text features. Factor training used 32 factors, seed 42, 2,000 epochs, batch size 1,024, and `lambda_usage_balance=0.1`; no parameter was retuned. Both resulting code arrays had shape `(308723, 32)` and finite values.

The script then called `mine_episodes` with its default 90th-percentile high pool, 50th-percentile low pool, four items per support/contrast/distractor role, `min_pool_size=50`, and seed 42. A separate seed-42 permutation selected the ten spot-check episodes. For recovery, another seed-42 random permutation of episode indices put the first `floor(0.8 × 1024)=819` episodes in training and the remaining **205** in held-out evaluation. `ConditionEncoder` used its default hidden width 16 and the Task 1 validation-only objective `F.cross_entropy(w(c), targeted_factor)` with Adam (`lr=0.05`), 100 epochs, and batches of 64. The code checks that vectorized calls agree with the public single-episode `forward`. `targeted_factor` appears only as this auxiliary validation label, never as an encoder input.

## Part A: mining quality

**Coverage:** 32/32 distinct factors were chosen as `targeted_factor`. **No factors were excluded** for insufficient high or low pool size (`[]`). Each factor had 30,873–30,874 high-pool items and 154,362–154,364 low-pool items, far above `min_pool_size=50`. The default threshold and minimum were unchanged.

**Role shortfall:** 0 of 5,120 requested role slots (1,024 episodes × five roles) were shortened, a **0% shortfall rate**. The shortfall-count distribution was `{0: 5120}`; mean missing items per role slot and mean missing items conditional on a shortened role were both **0**. Support, contrast, hard negative, condition-only distractor, and anchor-only distractor each had **0/1,024 shortened episodes** and **0 missing items**. The script also asserted that no item index appeared in two roles within an episode.

The table gives actual `pair_codes[:, targeted_factor] = 0.5 × (img_code + txt_code)` values for ten seed-42 sampled episodes. Each of the three low-pool roles is below the high-pool roles in its row. Hard negatives are selected by similarity to the positive on other factors; the two distractor roles use the anchor as their other-factor reference.

| Episode | Factor | Anchor | Support | Contrast | Positive | Hard negative | Condition-only | Anchor-only |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 87 | 13 | 0.2109 | 0.2178 | 0.0426 | 0.2047 | 0.1072 | 0.2086 | 0.1025 |
| 786 | 4 | 0.2205 | 0.2288 | 0.0000 | 0.2349 | 0.0965 | 0.2302 | 0.0925 |
| 90 | 2 | 0.2115 | 0.2228 | 0.0224 | 0.2092 | 0.1128 | 0.2115 | 0.1115 |
| 665 | 18 | 0.2177 | 0.2288 | 0.0003 | 0.2276 | 0.0944 | 0.2164 | 0.0985 |
| 446 | 21 | 0.1755 | 0.2285 | 0.0335 | 0.1737 | 0.0725 | 0.1747 | 0.0818 |
| 441 | 9 | 0.2543 | 0.2833 | 0.0000 | 0.2546 | 0.1243 | 0.2592 | 0.1269 |
| 712 | 1 | 0.2288 | 0.1955 | 0.0556 | 0.2399 | 0.0890 | 0.1937 | 0.0900 |
| 96 | 2 | 0.2107 | 0.2323 | 0.0090 | 0.2200 | 0.1087 | 0.2115 | 0.1126 |
| 206 | 9 | 0.3109 | 0.2701 | 0.0720 | 0.2756 | 0.1248 | 0.2548 | 0.1278 |
| 875 | 11 | 0.2033 | 0.2025 | 0.0059 | 0.2313 | 0.1028 | 0.1934 | 0.1025 |

## Part B: held-out condition recovery

The encoder predicted the mined factor for **45/205 episodes (21.95%)**, versus **3.125%** chance. All 32 factors appeared in held-out episodes, with 3–13 examples each. Accuracy is **concentrated**, not roughly uniform: 19 factors had 0 correct predictions, while 13 had at least one; the observed per-factor range is 0–100%.

| Factor | Correct / held-out | Accuracy | Factor | Correct / held-out | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/5 | 0.0% | 16 | 5/5 | 100.0% |
| 1 | 0/5 | 0.0% | 17 | 0/6 | 0.0% |
| 2 | 0/7 | 0.0% | 18 | 0/6 | 0.0% |
| 3 | 4/6 | 66.7% | 19 | 2/7 | 28.6% |
| 4 | 0/10 | 0.0% | 20 | 0/4 | 0.0% |
| 5 | 0/4 | 0.0% | 21 | 3/3 | 100.0% |
| 6 | 0/7 | 0.0% | 22 | 3/4 | 75.0% |
| 7 | 4/9 | 44.4% | 23 | 0/4 | 0.0% |
| 8 | 0/7 | 0.0% | 24 | 0/10 | 0.0% |
| 9 | 0/7 | 0.0% | 25 | 2/5 | 40.0% |
| 10 | 5/7 | 71.4% | 26 | 0/10 | 0.0% |
| 11 | 0/7 | 0.0% | 27 | 1/13 | 7.7% |
| 12 | 0/4 | 0.0% | 28 | 2/4 | 50.0% |
| 13 | 4/6 | 66.7% | 29 | 5/6 | 83.3% |
| 14 | 0/9 | 0.0% | 30 | 5/6 | 83.3% |
| 15 | 0/5 | 0.0% | 31 | 0/7 | 0.0% |

## Caveats

This is one seed-42 run on training-set factor codes. The split is by **episode**, so the same underlying ArtELingo item can appear in both train and held-out episodes; the 21.95% figure is episode-level recovery, not item-disjoint generalization. Each per-factor denominator is small, so its exact percentage is noisy, but the 19 zero-correct factors are a substantial warning against relying on the aggregate alone. Percentile pools show real high/low activation separation; they do not establish semantic factor identity or that candidate ranking will work. No scoring function or swap-loss training was run here.
