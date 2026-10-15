# CoSiR v2 Candidate A: class-balanced condition-recovery re-test

**Verdict: not fixed.** With 4,096 mined episodes, inverse-frequency class-balanced batch sampling, and the preselected lower-learning-rate/longer schedule, held-out recovery was **189/820 (23.05%)**. Only **16/32** factors were ever predicted, the top five still absorbed **569/820 (69.39%)** predictions, and **18/32** factors had exactly 0% accuracy. The minor accuracy increase does not change the collapse pattern. This result warrants reconsidering the condition encoder or its factor representation in a subsequent task; this re-test made no architectural or mining-logic change and did no further hyperparameter tuning.

## Direct comparison with Tasks 3 and 4

| Measure | Task 3/4 baseline | Balanced re-test |
|---|---:|---:|
| Held-out recovery | 45/205 (21.95%) | 189/820 (23.05%) |
| Distinct factors predicted at least once | 16/32 | 16/32 |
| Top-five prediction concentration | 141/205 (68.78%) | 569/820 (69.39%) |
| Factors with exactly 0% held-out accuracy | 19/32 | 18/32 |

The recovery difference is +1.10 percentage points on a larger mined episode set with a new 80/20 split, so it is a descriptive comparison rather than a paired estimate. The top-five share rose by 0.61 percentage points. This run's top five were 10:158, 29:133, 30:126, 25:82, 22:70; their counts sum to 569.

The **same factors mostly remain dead**: 18 of the baseline's 19 zero-correct factors are still zero-correct: 0, 1, 2, 4, 5, 6, 8, 9, 11, 14, 15, 17, 18, 20, 23, 24, 26, 31. Only factor **12** changed from zero-correct to **1/19 (5.26%)**. There are **no newly dead factors**. All 32 factors had held-out examples, so every 0% row is measured rather than missing. The baseline diagnostic's independent real-data-correlation check found 0/19 dead-factor top wrong destinations among their three closest factor columns; this run does not repeat that data-level analysis.

## Reproduction and training procedure

Run `python src/test/20261003_condition_recovery_balanced_retrain/run_retrain.py` from the repository root in the CoSiR environment. The script reuses Task 3's exact ArtELingo feature loading and prerequisite preparation: 308,723 positionally joined image/text rows, raw unwhitened CLIP features, a 3,130,544-edge content graph, 21 detected communities, and 32 factor columns trained with `lambda_usage_balance=0.1` and the default factor-training configuration. It calls the unchanged `mine_episodes` with `EpisodeMiningConfig(seed=42)` defaults and `num_episodes=4096`; all 32 factors were selected. A seed-42 random permutation of episode indices assigns the first `floor(0.8 × 4096) = 3276` to training and the remaining **820** to held-out evaluation. Training counts range from 88 to 135 episodes per factor.

The sole deliberate optimization-schedule choice was Adam **`lr=0.01` for 300 epochs**, instead of `lr=0.05` for 100 epochs; batch size stayed **64**. The objective remains `cross_entropy(w(c), targeted_factor)` on the unchanged `ConditionEncoder`, solely for validation. Each epoch, `WeightedRandomSampler` draws **3276** training-episode positions **with replacement**, using weight `1 / training count of that episode's targeted factor`. Thus every represented factor has equal expected sampling probability and roughly **102.4 expected draws per epoch**. The sampler uses an explicit seed-42 PyTorch generator. The script limits BLAS threads only around episode mining to reduce runtime overhead; this changes no mining settings. No second optimization adjustment was tried.

## Training dynamics

Accuracy uses the complete original training split and complete held-out split, not the resampled epoch batches. Diversity counts distinct `argmax(w(c))` factors on held-out episodes.

| Epoch | Training accuracy | Held-out accuracy | Distinct held-out predictions |
|---:|---:|---:|---:|
| 50 | 820/3276 (25.03%) | 189/820 (23.05%) | 16/32 |
| 100 | 821/3276 (25.06%) | 191/820 (23.29%) | 16/32 |
| 150 | 821/3276 (25.06%) | 190/820 (23.17%) | 16/32 |
| 200 | 819/3276 (25.00%) | 189/820 (23.05%) | 16/32 |
| 250 | 820/3276 (25.03%) | 189/820 (23.05%) | 16/32 |
| 300 | 822/3276 (25.09%) | 189/820 (23.05%) | 16/32 |

The trajectory is nearly flat from epoch 50 onward: roughly 25% training accuracy, 23% held-out accuracy, and exactly 16 predicted factors at every checkpoint. More episodes, balanced expected sampling, and the longer schedule therefore did not restore broad factor coverage.

## Held-out accuracy by targeted factor

| Factor | Held-out episodes | Correct | Accuracy |
|---:|---:|---:|---:|
| 0 | 24 | 0 | 0.00% |
| 1 | 32 | 0 | 0.00% |
| 2 | 23 | 0 | 0.00% |
| 3 | 21 | 14 | 66.67% |
| 4 | 27 | 0 | 0.00% |
| 5 | 29 | 0 | 0.00% |
| 6 | 26 | 0 | 0.00% |
| 7 | 28 | 12 | 42.86% |
| 8 | 25 | 0 | 0.00% |
| 9 | 25 | 0 | 0.00% |
| 10 | 21 | 15 | 71.43% |
| 11 | 27 | 0 | 0.00% |
| 12 | 19 | 1 | 5.26% |
| 13 | 24 | 14 | 58.33% |
| 14 | 24 | 0 | 0.00% |
| 15 | 23 | 0 | 0.00% |
| 16 | 23 | 23 | 100.00% |
| 17 | 24 | 0 | 0.00% |
| 18 | 27 | 0 | 0.00% |
| 19 | 19 | 1 | 5.26% |
| 20 | 26 | 0 | 0.00% |
| 21 | 27 | 27 | 100.00% |
| 22 | 26 | 20 | 76.92% |
| 23 | 24 | 0 | 0.00% |
| 24 | 29 | 0 | 0.00% |
| 25 | 29 | 19 | 65.52% |
| 26 | 26 | 0 | 0.00% |
| 27 | 27 | 7 | 25.93% |
| 28 | 24 | 1 | 4.17% |
| 29 | 28 | 14 | 50.00% |
| 30 | 25 | 21 | 84.00% |
| 31 | 38 | 0 | 0.00% |

The validation split is by mined episode, not by underlying ArtELingo item; support and contrast items can recur across episodes. The comparisons therefore describe this fixed-seed recoverability check rather than independent-item generalization or causal isolation of the three combined training changes.
