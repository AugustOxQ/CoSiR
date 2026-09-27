# Leiden resolution sweep on RedCaps' raw train teacher graph

Generated 2026-09-27 14:45:07; seed 42; reuses B1's saved split (`b1_redcaps_single_teacher_pilot_split.npz`) and rebuilds B1's raw train teacher union graph (K=30, device=cuda); Leiden and validation metrics run on CPU. Companion to [`docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md`](../../../docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md) and [`b1_redcaps_single_teacher_pilot_report.md`](b1_redcaps_single_teacher_pilot_report.md).

## Sanity check (resolution=1.0 vs B1's existing graph-only baseline)

Passed: resolution=1.0 reproduced B1's 1423 communities, 1404 below 1% occupancy, and 4.773× community lift at reported precision. The raw graph also matches B1's edge count exactly.

## Results

Community-level lift shows overall lift × (observed/expected same-subreddit fraction); expectation uses the edge-endpoint subreddit marginal, computed over validation pairs sharing a transferred community (cosine k=20 nearest-train-neighbor assignment), with any community above 2,000 members subsampled to 2,000 before pairs are enumerated (identical to B1's own protocol).

The cap changes the weighting of same-community pairs as communities grow; lift values use B1's capped metric at every resolution.

| Resolution | Communities | Min/median/max occupancy | Below-1% count | Transfer coverage | Community-level lift |
|---:|---:|---:|---:|---:|---:|
| 1 | 1423 | 1/1.0/21482 | 1404 | 27/1423 | 4.773× (0.1140/0.0239) |
| 0.5 | 1410 | 1/1.0/23853 | 1398 | 14/1410 | 4.707× (0.0862/0.0183) |
| 0.25 | 1403 | 1/1.0/28164 | 1396 | 7/1403 | 3.967× (0.0688/0.0173) |
| 0.1 | 1398 | 1/1.0/101040 | 1396 | 2/1398 | 1.926× (0.0535/0.0278) |
| 0.05 | 1397 | 1/1.0/118557 | 1396 | 1/1397 | 0.971× (0.0167/0.0172) |
| 0.01 | 1397 | 1/1.0/118557 | 1396 | 1/1397 | 0.971× (0.0167/0.0172) |
| 0.005 | 1397 | 1/1.0/118557 | 1396 | 1/1397 | 0.971× (0.0167/0.0172) |

## Verdict

**No tested resolution recovers a much smaller, healthier-occupancy partition while retaining community lift near 4.773×.** The smallest partition has 1397 communities, only 1.8% fewer than resolution=1.0, and its lift is 0.971×. The raw union has 1,397 disconnected components, including 1,354 isolated train nodes, which cannot merge across components under graph-only Leiden. At the smallest tested resolution, the largest community holds 118,557 of 120,000 train nodes; the partition is dominated by that community and tiny ones. Lowering resolution alone does not resolve the fragmentation; the disconnected-component floor explains the persistent count.
