# CoSiR v2 Candidate A: held-out condition confusion on ArtELingo

**Verdict: story (b), prediction toward a small set of attractor factors, is the dominant observed pattern.** The five most-predicted factors received **141/205 (68.78%)** held-out predictions, and 98/124 (79.03%) predictions for the zero-correct factors. The most common wrong prediction was among a dead factor's three closest real-data factor columns in **0/19 (0%)** cases. This argues against story (a)'s specific nearest-neighbor confusion pattern; the predictions are also far from uniformly spread as in story (c). The measurement identifies prediction skew, but cannot by itself establish whether training, encoder capacity, factor representation, or their interaction causes it. No model or training change was made.

## Exact Task 3 reproduction and measurement

Run `src/test/20261002_condition_confusion_diagnostic/run_diagnostic.py` from the repository root in the CoSiR environment. It reuses Task 3's feature loading, factor-code preparation, and mining functions directly, then repeats Task 3's recovery training loop before retaining predictions. The 308,723 ArtELingo feature rows use the validated positional join and raw, unwhitened CLIP features. The run rebuilt the same 3,130,544-edge graph, detected 21 communities, and trained 32 factors with `lambda_usage_balance=0.1`, seed 42, and the default Task 6 factor-training configuration. It mined 1,024 episodes using `EpisodeMiningConfig(seed=42)` and `mine_episodes`; all 32 factors were selected, none excluded, and no role shortened. The recovery split is the same independent seed-42 episode permutation: 819 training and 205 held-out episodes. `ConditionEncoder(hidden_dim=16)` was trained with `F.cross_entropy(w(c), targeted_factor)`, Adam `lr=0.05`, 100 epochs, and batches of 64. This is the Task 1 validation-only objective, not stage (d) swap-loss training.

The diagnostic records `(episode index, true factor, argmax(w(c)))` for every held-out episode in its `RESULT_JSON` output. The rerun recovered **45/205 (21.95%)**, matching Task 3, versus 3.125% chance. All 32 factors had held-out examples. The zero-correct factors, derived from this run's confusion matrix rather than copied from Task 3, are **0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 23, 24, 26, 31**.

## Complete 32 × 32 confusion matrix

Rows are true factors; columns are predicted factors. The last column lists every nonzero cell as `predicted factor:count`; all omitted cells in that row are zero. Thus the table specifies all 1,024 matrix cells. Diagonal counts are shown separately for clarity.

| True factor | Held out | Correct | Nonzero predicted-factor cells |
|---:|---:|---:|---|
| 0 | 5 | 0 | 10:2, 29:3 |
| 1 | 5 | 0 | 7:1, 10:2, 29:2 |
| 2 | 7 | 0 | 12:2, 16:2, 30:3 |
| 3 | 6 | 4 | 3:4, 25:1, 30:1 |
| 4 | 10 | 0 | 3:2, 25:1, 30:7 |
| 5 | 4 | 0 | 0:1, 10:2, 29:1 |
| 6 | 7 | 0 | 7:1, 10:4, 22:1, 29:1 |
| 7 | 9 | 4 | 7:4, 10:2, 29:3 |
| 8 | 7 | 0 | 10:2, 13:1, 27:1, 29:3 |
| 9 | 7 | 0 | 3:1, 16:1, 30:5 |
| 10 | 7 | 5 | 10:5, 29:2 |
| 11 | 7 | 0 | 7:1, 10:2, 13:1, 22:1, 29:2 |
| 12 | 4 | 0 | 3:1, 25:2, 30:1 |
| 13 | 6 | 4 | 10:1, 13:4, 22:1 |
| 14 | 9 | 0 | 10:5, 13:1, 22:1, 29:2 |
| 15 | 5 | 0 | 3:2, 25:2, 30:1 |
| 16 | 5 | 5 | 16:5 |
| 17 | 6 | 0 | 7:2, 10:2, 22:1, 29:1 |
| 18 | 6 | 0 | 3:1, 25:2, 30:3 |
| 19 | 7 | 2 | 10:1, 13:3, 19:2, 29:1 |
| 20 | 4 | 0 | 3:1, 25:1, 30:2 |
| 21 | 3 | 3 | 21:3 |
| 22 | 4 | 3 | 7:1, 22:3 |
| 23 | 4 | 0 | 7:1, 10:2, 29:1 |
| 24 | 10 | 0 | 3:1, 9:1, 25:5, 30:3 |
| 25 | 5 | 2 | 16:2, 25:2, 30:1 |
| 26 | 10 | 0 | 7:4, 10:3, 29:3 |
| 27 | 13 | 1 | 0:1, 7:2, 10:1, 13:4, 22:3, 27:1, 29:1 |
| 28 | 4 | 2 | 7:1, 28:2, 29:1 |
| 29 | 6 | 5 | 10:1, 29:5 |
| 30 | 6 | 5 | 25:1, 30:5 |
| 31 | 7 | 0 | 10:4, 13:1, 27:1, 29:1 |

The row totals sum to 205; the diagonal sums to 45.

## Prediction skew across all held-out episodes

The complete predicted-factor frequency distribution is `0:2, 1:0, 2:0, 3:13, 4:0, 5:0, 6:0, 7:18, 8:0, 9:1, 10:41, 11:0, 12:2, 13:15, 14:0, 15:0, 16:10, 17:0, 18:0, 19:2, 20:0, 21:3, 22:11, 23:0, 24:0, 25:17, 26:0, 27:3, 28:2, 29:33, 30:32, 31:0`. Only 16/32 factors were ever predicted. The top five are **10:41, 29:33, 30:32, 7:18, 25:17**, totaling **141/205 (68.78%)**. Their uniform 32-way reference share would be 5/32 = 15.625%; this reference is descriptive, not a calibrated statistical null for a trained encoder. The skew includes both correct and incorrect predictions.

## Where the 19 zero-correct factors went

Each distribution lists **every** destination with a nonzero count, as `factor: count / row total (proportion)`. The most common wrong factor is the highest-count destination; ties use the lowest factor ID. Four rows have a tied mode (1, 11, 15, 17); none of their tied alternatives is a top-three correlated neighbor either.

For the independent data-level check, let `pair_codes = 0.5 × (img_codes + txt_codes)` for all **308,723** rows. The script computes the complete `(32, 32)` matrix `cosine(pair_codes[:, i], pair_codes[:, j])` in float64, without centering or using episodes or condition-training labels. Each row's top three excludes itself and ranks by descending cosine, breaking exact ties by factor ID. The neighbor column gives those three factors with their cosine values. Every top-mode neighbor check is **No**, yielding **0/19 (0%)**.

| Dead factor | Held out | Full wrong-prediction distribution | Top-3 real-data neighbors (cosine) | Top wrong in top 3? |
|---:|---:|---|---|:---:|
| 0 | 5 | 29: 3/5 (60.0%), 10: 2/5 (40.0%) | 10 (0.9984), 5 (0.9958), 17 (0.9950) | No |
| 1 | 5 | 10: 2/5 (40.0%), 29: 2/5 (40.0%), 7: 1/5 (20.0%) | 6 (0.9991), 31 (0.9989), 26 (0.9988) | No |
| 2 | 7 | 30: 3/7 (42.9%), 12: 2/7 (28.6%), 16: 2/7 (28.6%) | 9 (0.9965), 15 (0.9963), 20 (0.9962) | No |
| 4 | 10 | 30: 7/10 (70.0%), 3: 2/10 (20.0%), 25: 1/10 (10.0%) | 18 (0.9995), 20 (0.9994), 15 (0.9992) | No |
| 5 | 4 | 10: 2/4 (50.0%), 0: 1/4 (25.0%), 29: 1/4 (25.0%) | 31 (0.9999), 8 (0.9995), 17 (0.9992) | No |
| 6 | 7 | 10: 4/7 (57.1%), 7: 1/7 (14.3%), 22: 1/7 (14.3%), 29: 1/7 (14.3%) | 8 (0.9993), 31 (0.9993), 1 (0.9991) | No |
| 8 | 7 | 29: 3/7 (42.9%), 10: 2/7 (28.6%), 13: 1/7 (14.3%), 27: 1/7 (14.3%) | 31 (0.9998), 17 (0.9997), 5 (0.9995) | No |
| 9 | 7 | 30: 5/7 (71.4%), 3: 1/7 (14.3%), 16: 1/7 (14.3%) | 2 (0.9965), 15 (0.9941), 18 (0.9937) | No |
| 11 | 7 | 10: 2/7 (28.6%), 29: 2/7 (28.6%), 7: 1/7 (14.3%), 13: 1/7 (14.3%), 22: 1/7 (14.3%) | 14 (0.9991), 26 (0.9975), 17 (0.9974) | No |
| 12 | 4 | 25: 2/4 (50.0%), 3: 1/4 (25.0%), 30: 1/4 (25.0%) | 18 (0.9977), 15 (0.9977), 20 (0.9977) | No |
| 14 | 9 | 10: 5/9 (55.6%), 29: 2/9 (22.2%), 13: 1/9 (11.1%), 22: 1/9 (11.1%) | 11 (0.9991), 26 (0.9983), 17 (0.9979) | No |
| 15 | 5 | 3: 2/5 (40.0%), 25: 2/5 (40.0%), 30: 1/5 (20.0%) | 20 (0.9997), 18 (0.9996), 4 (0.9992) | No |
| 17 | 6 | 7: 2/6 (33.3%), 10: 2/6 (33.3%), 22: 1/6 (16.7%), 29: 1/6 (16.7%) | 8 (0.9997), 31 (0.9994), 26 (0.9994) | No |
| 18 | 6 | 30: 3/6 (50.0%), 25: 2/6 (33.3%), 3: 1/6 (16.7%) | 15 (0.9996), 20 (0.9995), 4 (0.9995) | No |
| 20 | 4 | 30: 2/4 (50.0%), 3: 1/4 (25.0%), 25: 1/4 (25.0%) | 15 (0.9997), 18 (0.9995), 4 (0.9994) | No |
| 23 | 4 | 10: 2/4 (50.0%), 7: 1/4 (25.0%), 29: 1/4 (25.0%) | 1 (0.9987), 7 (0.9975), 5 (0.9970) | No |
| 24 | 10 | 25: 5/10 (50.0%), 30: 3/10 (30.0%), 3: 1/10 (10.0%), 9: 1/10 (10.0%) | 18 (0.9991), 4 (0.9986), 20 (0.9983) | No |
| 26 | 10 | 7: 4/10 (40.0%), 10: 3/10 (30.0%), 29: 3/10 (30.0%) | 17 (0.9994), 31 (0.9993), 8 (0.9992) | No |
| 31 | 7 | 10: 4/7 (57.1%), 13: 1/7 (14.3%), 27: 1/7 (14.3%), 29: 1/7 (14.3%) | 5 (0.9999), 8 (0.9998), 17 (0.9994) | No |

These 19 rows contain 124 held-out episodes, all incorrect. Their destinations concentrate on the global attractors; for example, factor 4 went to factor 30 in 7/10 cases and factor 9 did so in 5/7. Many raw, noncentered column cosines are close to one, so this ranking does not prove the factor space is semantically independent. It does show that the *observed* wrong destinations do not follow its closest-neighbor ranking. Per-factor held-out counts of 4–10 and this single seed limit causal claims. The split is by episode, not by underlying ArtELingo item.
