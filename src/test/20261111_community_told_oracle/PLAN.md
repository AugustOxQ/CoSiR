# Told-partition oracle with community and coarser affect partitions (EXPLORATORY, seed 42 only)

Written 2026-10-05, before any number of this folder exists. Decides nothing about the GO; no fresh episode seed is read.

## Question

The told-partition oracle on the current partitions beats its matched condition-free counterpart by +1.14 [0.90, 1.41]
R@1 (`src/test/20261109_fix_diagnostics/results/diagnose_counterparts.txt`). The partition profile
(`src/test/20261110_partition_profile/`) found that the affect partition (k-means, 64 clusters on GoEmotions
probabilities) carries emotion as clusters (pair lift 2.17) but almost none of it survives the cross-modal heads
(1.11). Does a coarser affect partition, formed as Leiden communities the way the buddy pipeline forms them, raise the
told oracle? If the told oracle cannot clearly exceed +1.14, no label-free reader built on that partition can pass.

## Arms (only the affect partition changes; image and caption partitions, the told mapping and B stay as they are)

| Arm | Affect partition on scorer-train rows |
|---|---|
| R0 | current E2 affect partition (k-means, 64 clusters); must reproduce the stored numbers |
| L | Leiden communities: `src/model/communities.py::detect_communities` with its defaults (kNN union graph, k = 20, modularity, seed 42) on the 28 GoEmotions probabilities per row (`src/test/20261018_affect_factor_learning/cache/affect_prepare.npz`, `affect_probs`). Communities under 200 rows are merged into the community with the nearest centroid in the same 28-d space (label free) |
| K | k-means on the same probabilities with k = the number of communities of L after merging (E2's settings otherwise), so that L against K separates the algorithm from the granularity |

All partition settings are fixed above and are not tuned on any label or result.

## Measured for every arm (seed 42 development episodes)

1. Number of groups and sizes; the profile's pair statistics for the affect partition × emotion, on the groups and through the refitted heads.
2. Held-out head accuracy for the affect partition (image and caption heads, N6's `fit_heads`).
3. The told term (emotion to affect, style and genre to image) fused on B (stored C2, R@1 18.34), its matched counterpart, and the margin, with 95% painting-cluster intervals; per aspect pair.
4. The same for N6's label-free reader (T6), plus its pick accuracy.
5. Paired difference of each arm's told margin against R0's.

## How the numbers will be read

- R0 must reproduce +1.14 [0.90, 1.41] (told) and +0.14 [−0.04, 0.32] (reader). Otherwise stop and report.
- **Promising**: an arm's told margin over its counterpart is at least +2.0, or its paired difference against R0 has a
  95% lower bound above 0 and its reader margin moves toward +0.5.
- **Not better**: the paired difference against R0 has a lower bound at or below 0.
- L against K: if they are within each other's intervals, the change comes from granularity, not from Leiden.
- Exploratory on a reused development seed; a promising arm still needs a decision rule and the fresh-seed test.
