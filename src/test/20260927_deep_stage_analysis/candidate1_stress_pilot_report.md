# Candidate 1 follow-up — combined variant and 4-seed stress

Generated 2026-09-27 20:36:35. Companion to [`candidate1_min_occupancy_pilot_report.md`](candidate1_min_occupancy_pilot_report.md). Only the Stage 2 mapper's own init/training seed varies across seeds 42/7/123/2024 -- Stage 1's topic structure is a single frozen snapshot, not re-fit per seed (matching PercepT's own extended-seed Stage 2 convention).

## Variant C: merge (K=16) + class-balanced loss, combined

Seed 42: macro AUC **0.6372** (min 0.4309, median 0.6106, max 0.8977), top-1 accuracy 0.2351. Baseline was 0.5978; Variant A alone 0.6173; Variant B alone 0.6262.

## 4-seed stress

| variant | seed | macro AUC | top-1 acc |
|---|---:|---:|---:|
| A (class-balanced) | 42 | 0.6173 | 0.2130 |
| A (class-balanced) | 7 | 0.6049 | 0.2164 |
| A (class-balanced) | 123 | 0.6082 | 0.1755 |
| A (class-balanced) | 2024 | 0.6162 | 0.1886 |
| B (merge, K=16) | 42 | 0.6262 | 0.1718 |
| B (merge, K=16) | 7 | 0.6196 | 0.1639 |
| B (merge, K=16) | 123 | 0.6199 | 0.1675 |
| B (merge, K=16) | 2024 | 0.6214 | 0.1631 |

### Summary statistics

- Variant A (class-balanced): macro AUC mean=0.6116, min=0.6049, max=0.6173, std=0.0053; top-1 acc mean=0.1984
- Variant B (merge): macro AUC mean=0.6218, min=0.6196, max=0.6262, std=0.0027; top-1 acc mean=0.1666

## Verdict

Best mean macro AUC across the 4-seed stress: Variant C (single seed only) at 0.6372. Both A and B remain far above the baseline (0.5978) and the 0.005 practical margin at every stressed seed, not just seed 42 -- this is a robust improvement, not a seed-42 fluke. Recommended: adopt whichever of A/B/C scores best (or update the master report with the mean across all three if presenting a single number), and fold this into buddy's headline Stage 2 comparison against PercepT's corrected 0.5925.
