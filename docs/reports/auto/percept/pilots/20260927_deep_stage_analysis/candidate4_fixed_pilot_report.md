# Candidate 4 fix — apples-to-apples re-scoring

Generated 2026-09-27 21:13:59. Fixes Critical finding C1 from an independent review of `run_candidate4_rich_multilabel_pilot.py`: its negative verdict compared each cutoff's AUC against that cutoff's OWN multi-label held-out targets, not against the same single-label targets the 0.8461 baseline was scored against -- an apples-to-oranges comparison, the same class of error this investigation already flagged in PercepT's own multi-hot-threshold numbers (master report §6b). Every model below is trained exactly as in the original candidate 4 script (same multi-label targets, same unweighted loss, same lr/epochs) but is additionally re-scored against the baseline's own single-label held-out targets (candidate 2's exact k=20 hard-vote transfer on the merged K=16 vocabulary) for a genuine like-for-like macro AUC. Baseline: 0.8461.

## Results

| cutoff | train mean/median/max labels | own-target macro AUC | vs-baseline-target macro AUC |
|---:|---|---:|---:|
| 0.50 | 1.240/1.000/7 | 0.8394 | 0.8504 |
| 0.30 | 1.485/1.000/9 | 0.8322 | 0.8521 |
| 0.15 | 1.878/2.000/11 | 0.8188 | 0.8533 |

## W2 control: single-label targets, unweighted loss

Macro AUC (vs. baseline single-label targets): **0.8476**. The candidate-2 baseline (same targets, same lr/epochs, but WITH class-balanced weighting) got 0.8461. The difference (+0.0015) isolates the effect of dropping the class-balanced weighting alone, independent of multi-labeling.

## Verdict

On the correct, like-for-like comparison (vs-baseline-target column), the best cutoff (0.15) scores **0.8533** against the 0.8461 baseline (+0.0072). **This clears the practical margin — the original negative verdict does not hold under a fair comparison.** Recommend 4-seed stressing this cutoff with the corrected evaluation before adopting.
