# Leiden affect communities: 3 x 3 sweep of graph k and resolution (EXPLORATORY, seed 42 only)

Written 2026-10-05, before any number of this folder exists. Decides nothing about the GO; no fresh episode seed is
read. Follows `src/test/20261111_community_told_oracle/` (PLAN.md sha 3b8f90bf…), where Leiden communities at
`detect_communities` defaults (k = 20, modularity; 41 groups after merging) raised the told margin over its matched
counterpart from +1.14 to +1.64 and the reader margin from +0.14 to +0.35, while k-means at 41 groups did not move.

## Grid (only the affect partition changes; image and caption partitions, the told mapping and B stay as they are)

- Graph: the kNN union graph of `detect_communities` (unweighted, simplified) on the 28 GoEmotions probabilities per
  scorer-train row, with **k in {10, 20, 40}**.
- Leiden: `leidenalg.RBConfigurationVertexPartition`, **resolution in {0.25, 1.0, 4.0}**, seed 42. (Resolution 1.0 is
  modularity; the k = 20, resolution 1.0 cell is compared with the stored arm L, which used
  `ModularityVertexPartition`, and any difference is reported.)
- Communities under 200 rows are merged into the nearest-centroid community of at least 200 rows (one pass, centroids
  before merging), as in arm L. Raw and merged counts are reported.
- Control: k-means (E2's settings) at every distinct merged group count of the grid.

Nine Leiden cells plus the k-means controls. No setting is tuned beyond this grid.

## Measured per cell

As in the previous folder: group count and sizes; affect × emotion pair statistics on the groups and through the refit
heads; held-out affect head accuracy; the told term and N6's label-free reader (T6) each fused on B with its matched
counterpart and margin (overall and per aspect pair), with 95% painting-cluster intervals; pick accuracy; paired
differences against R0 (k-means 64) and against arm L.

## Pick rule (fixed now)

The cell carried forward is the one with the **largest label-free reader margin over its counterpart** on seed 42 (the
method's own development metric). If two cells are within 0.05 of each other, the one with fewer groups is carried.
The told margin is reported as a diagnostic and does not pick. The untuned default (arm L) is reported beside the pick.

## How the numbers will be read

- The (k = 20, resolution 1.0) cell should be close to arm L; if it differs, both are reported and the grid is read
  as RBConfiguration throughout.
- Shape: how the told and reader margins move with the group count and with k; whether Leiden beats k-means at
  matched counts across the range (paired differences).
- The picked cell is a development pick on a reused seed. It goes into the reader-fix decision rule only after the
  user agrees, and still needs the fresh-seed test (seeds 49 to 51).
