# 2026-11-12 Leiden affect-community sweep (EXPLORATORY: decides nothing; seed 42 development episodes only)

**Problem.** In `src/test/20261111_community_told_oracle/`, Leiden communities on the 28 GoEmotions probabilities
(`detect_communities` defaults: kNN union graph k = 20, modularity, 41 groups after merging) replaced the affect k-means
partition (64 clusters) and raised the told margin over its matched counterpart from +1.14 to +1.64 and N6's label-free
reader margin from +0.14 to +0.35, while k-means at 41 groups did not move. This run asks how those margins depend on
the graph's k and on Leiden's resolution, and whether Leiden beats k-means at every group count.

**Plan.** `PLAN.md` (SHA-256 b477b287…, written 2026-10-05T09:50Z before any number): graph k in {10, 20, 40} x
resolution in {0.25, 1.0, 4.0} (`RBConfigurationVertexPartition`, seed 42), communities under 200 rows merged into the
nearest-centroid community, a k-means control (E2's settings) at every distinct merged count, the same measurements as
the previous folder, and a fixed pick rule: largest reader margin, ties within 0.05 to fewer groups.

**What was run.** `run_sweep.py` (built and smoke-tested by a subagent; the smoke cell k = 20, resolution 1.0 equalled
arm L exactly, ARI 1.0, and its k-means 41 control equalled arm K). Three CPU processes (one per k, 8 threads each),
423 to 504 s each, peak 5.5 GB each; every process first reproduced R0 (told +1.14 [0.90, 1.41], reader +0.14
[−0.04, 0.32]) and B (18.34) exactly. Then `run_sweep.py merge`. Outputs in `results/` (gitignored): `cells/*.json`,
`sweep.json`, `sweep.txt`, `per_anchor_sweep.npz`, `run_k{10,20,40}.log`. The controller re-derived the pick's margins,
one other cell and the pick's paired difference with k-means from `per_anchor_sweep.npz` with its own bootstrap; they
matched.

## Results (R@1 margin of fused term over fused matched counterpart, on B; 95% painting-cluster intervals)

Merged group count (raw): k = 10: 15 (15), 44 (45), 118 (122); k = 20: 14 (14), 41 (42), 95 (96); k = 40: 15 (15),
31 (32), 88 (89), for resolutions 0.25, 1.0 and 4.0.

| Told margin | res 0.25 | res 1.0 | res 4.0 |
|---|---|---|---|
| k = 10 | +1.95 [1.67, 2.24] | +1.83 [1.54, 2.12] | +1.85 [1.57, 2.13] |
| k = 20 | +1.70 [1.42, 1.98] | +1.64 [1.37, 1.92] | +1.68 [1.40, 1.95] |
| k = 40 | +1.88 [1.58, 2.18] | +1.68 [1.40, 1.95] | +1.65 [1.38, 1.92] |

| Reader margin | res 0.25 | res 1.0 | res 4.0 |
|---|---|---|---|
| k = 10 | +0.38 [0.17, 0.59] | +0.40 [0.18, 0.61] | +0.46 [0.23, 0.69] |
| k = 20 | +0.38 [0.18, 0.58] | +0.35 [0.15, 0.57] | +0.48 [0.25, 0.71] |
| k = 40 | +0.36 [0.16, 0.56] | +0.50 [0.30, 0.71] | +0.46 [0.23, 0.69] |

Baselines: k-means 64 (R0) told +1.14, reader +0.14; k-means controls at the nine counts: told +0.97 to +1.24, reader
+0.14 to +0.18. Reader pick accuracy: 54.1 to 55.9% in every Leiden cell (R0 52.4%; k-means controls 50.6 to 53.4%).

- **Leiden against k-means at the matched count:** told +0.48 to +0.88 in all nine cells, every lower bound above 0;
  reader +0.17 to +0.33, lower bound above 0 in six cells and between −0.02 and 0.00 in the other three.
- **Flat in k and resolution:** all nine told margins lie inside each other's intervals, and so do all nine reader
  margins. Cluster-level emotion lift rises with the group count (1.99 at 14 groups to 3.45 at 118), but the lift
  through the heads stays at 1.12 to 1.16, and head accuracy falls as groups multiply (image head 21.2% at 14 groups,
  4.8% at 118).
- **Per pair:** emotion x style and emotion x genre carry the gain (told +2.36 to +3.03 and +2.80 to +3.76); style x
  genre stays at −0.52 to −0.68 in every cell (it runs on the unchanged image partition).
- **Pick (PLAN.md rule):** k = 40, resolution 1.0, 31 groups: reader +0.50 [0.30, 0.71] (gain +1.44, either −0.43),
  told +1.68 [1.40, 1.95], pick accuracy 55.0%. Against the untuned default (k = 20, resolution 1.0) the pick's reader
  margin is +0.15 [−0.01, 0.31] higher; against k-means at 31 groups, +0.33 [0.13, 0.53].

## Reading (PLAN.md applied)

- The k = 20, resolution 1.0 cell equals arm L exactly, so the grid is RBConfiguration throughout with no discrepancy.
- Shape: no trend in k or resolution beyond noise; Leiden beats k-means at every count from 14 to 118 groups.
- The pick is a development pick on a reused seed and the maximum of nine cells. It goes into the reader-fix decision
  rule only after the user agrees, and still needs the fresh-seed test (seeds 49 to 51).

Caveats: exploratory, one episode seed reused many times; one Leiden seed and one head-fit draw per cell; the told
mapping was chosen with labels; B's cross-fit was tuned on the same parity halves the fusion reuses; B′ (B rebuilt with
the cell's averaged-heads term) was not computed, so the development bar's max(B′, counterpart) is not checked for the
pick (for arm L, B′ was 18.44, below its counterpart).
