# Partition profile (2026-11-10): log

## Problem
The N6 reader picks among three label-free k-means partitions (affect, image, caption; 64 clusters each, E2) from the
within-pair agreement Delta of image and caption heads. The affect partition's Delta is near 0. Two explanations were open:
(a) the partition fragments each emotion over many clusters, so two items with the same emotion rarely share a cluster;
(b) the partition is fine but the 64-class heads that predict it are too weak (held-out accuracy 13.5% image head, 34.6% caption head).
Only global AMI existed. This log gives per-cluster and pair-level descriptions. It is descriptive only, uses evaluation labels
on scorer-train rows (checks 1 to 3) and selection rows (check 3b), decides nothing, and no label-guided choice of partitions
may follow from it.

## What was run
`profile_partitions.py` (deterministic, CPU only, 14 s). Inputs: `artelingo_splits` scorer-train (183,694 rows) and selection
(32,413 rows), E2 `partitions.npz` (alignment with `local_groups` asserted), `n6_posteriors.npz` (selection set asserted equal).
Labelled rows: emotion 163,281 (catch-all "something else" excluded, row-level), style 183,694 (per painting), genre 148,956 (per painting).
Pair counts are exact from contingency counts, with same-painting pairs removed by subtracting counts keyed also on painting.
Every exact number was cross-checked against 2,000,000 uniformly sampled different-painting pairs (seed 0): 90 comparisons,
max |z| = 2.42, none beyond 4 SE. Full outputs: `results/profile.txt`, `results/profile.json` (gitignored), including the full
64 x 8 affect x emotion table with per-cluster size, dominant value, purity, lift, coverage.

## Check 1 and 2: cluster x value tables and fragmentation
Baseline for fragmentation: the cluster-size distribution of all labelled rows (what a value spread like everything else would give).
Purity and lift are size-weighted over clusters; base rate of the dominant value gives the lift denominator. k50/k80 = clusters to cover
50%/80% of a value's rows; effK = exp(entropy).

```
=== CHECK 1 and 2: cluster x value tables and fragmentation (scorer-train rows) ===
affect   x emotion  size-weighted purity 0.518, lift 3.33 | median over values k50 4.5 k80 15.0 effK 21.6 | baseline k50 9 k80 28 effK 37.5
affect   x style    size-weighted purity 0.167, lift 1.26 | median over values k50 9.0 k80 28.0 effK 37.6 | baseline k50 10 k80 31 effK 39.5
affect   x genre    size-weighted purity 0.269, lift 1.38 | median over values k50 9.5 k80 28.5 effK 36.6 | baseline k50 10 k80 31 effK 39.4
image    x emotion  size-weighted purity 0.342, lift 1.38 | median over values k50 19.0 k80 40.0 effK 55.2 | baseline k50 27 k80 48 effK 62.5
image    x style    size-weighted purity 0.419, lift 7.94 | median over values k50 4.0 k80 10.0 effK 14.9 | baseline k50 27 k80 47 effK 62.4
image    x genre    size-weighted purity 0.717, lift 6.52 | median over values k50 3.5 k80 10.0 effK 14.4 | baseline k50 24 k80 45 effK 60.6
caption  x emotion  size-weighted purity 0.355, lift 1.57 | median over values k50 13.5 k80 30.5 effK 44.9 | baseline k50 22 k80 42 effK 58.2
caption  x style    size-weighted purity 0.192, lift 1.74 | median over values k50 12.0 k80 29.0 effK 40.7 | baseline k50 21 k80 41 effK 57.4
caption  x genre    size-weighted purity 0.451, lift 3.14 | median over values k50 10.0 k80 24.5 effK 35.0 | baseline k50 21 k80 41 effK 57.5

```

Affect x emotion per emotion (base rate of the emotion; "best" = highest-purity cluster among those where the emotion is dominant,
which can be a small cluster; "largest" = the cluster holding most rows of the emotion):

```
--- affect x emotion, per emotion ---
best-purity cluster among clusters where the emotion is dominant (size, purity, lift, coverage) | largest-coverage cluster (id, purity of that cluster, coverage) | k50 k80 effK  (baseline of all labelled rows: k50 9 k80 28 effK 37.5)
amusement    base 0.112 | cl 47 size  2891 purity 0.959 lift 8.54 cov 0.151 | cl 57 pur 0.150 cov 0.155 | k50 5 k80 19 effK 25.3
anger        base 0.016 | cl 37 size   841 purity 0.610 lift 37.43 cov 0.193 | cl 37 pur 0.610 cov 0.193 | k50 4 k80 15 effK 20.7
awe          base 0.184 | cl 26 size   588 purity 0.709 lift 3.86 cov 0.014 | cl  7 pur 0.502 cov 0.267 | k50 4 k80 15 effK 19.7
contentment  base 0.319 | cl 61 size  3306 purity 0.861 lift 2.70 cov 0.055 | cl  7 pur 0.355 cov 0.108 | k50 7 k80 18 effK 26.9
disgust      base 0.054 | cl 15 size  1224 purity 0.898 lift 16.50 cov 0.124 | cl 57 pur 0.059 cov 0.127 | k50 5 k80 15 effK 22.5
excitement   base 0.094 | cl 10 size  2762 purity 0.875 lift 9.35 cov 0.158 | cl 10 pur 0.875 cov 0.158 | k50 6 k80 17 effK 25.0
fear         base 0.103 | cl 13 size  5638 purity 0.895 lift 8.68 cov 0.300 | cl 13 pur 0.895 cov 0.300 | k50 3 k80 11 effK 14.5
sadness      base 0.118 | cl 38 size  7236 purity 0.926 lift 7.87 cov 0.349 | cl 38 pur 0.926 cov 0.349 | k50 3 k80 10 effK 13.8

```

## Check 3: hard partition ceiling (pairs on different paintings, scorer-train)
Random-pair base = P(same cluster). P(same value) is the share of different-painting pairs sharing the aspect value.

```
=== CHECK 3: pair co-membership, hard partitions (scorer-train, different-painting pairs) ===
lift = P(same cluster | same value) / P(same cluster | different value), with the random-pair base rate P(same cluster)
part     aspect      p_same    p_diff      base    lift  P(same val)
affect   emotion    0.07672   0.03530   0.04295   2.174       0.1846
affect   style      0.04404   0.04011   0.04042   1.098       0.0791
affect   genre      0.04379   0.03974   0.04033   1.102       0.1459
image    emotion    0.01983   0.01558   0.01636   1.273       0.1846
image    style      0.05875   0.01280   0.01643   4.590       0.0791
image    genre      0.07435   0.00763   0.01736   9.748       0.1459
caption  emotion    0.02240   0.01782   0.01867   1.257       0.1846
caption  style      0.02279   0.01893   0.01924   1.204       0.0791
caption  genre      0.03841   0.01577   0.01907   2.436       0.1459

episode contrasts: s_AB = P(same cluster | same A, diff B), s_BA = P(same cluster | same B, diff A)
part     pair                 s_AB      s_BA       diff   ratio      base
affect   emotionxstyle     0.07698   0.03777   +0.03921   2.038   0.04295
affect   emotionxgenre     0.07467   0.03699   +0.03768   2.019   0.04272
affect   stylexgenre       0.04331   0.04343   -0.00012   0.997   0.04033
image    emotionxstyle     0.01529   0.05579   -0.04050   0.274   0.01636
image    emotionxgenre     0.00891   0.07070   -0.06179   0.126   0.01737
image    stylexgenre       0.02568   0.05850   -0.03281   0.439   0.01736
caption  emotionxstyle     0.02192   0.02122   +0.00070   1.033   0.01867
caption  emotionxgenre     0.01769   0.03676   -0.01907   0.481   0.01859
caption  stylexgenre       0.01608   0.03742   -0.02135   0.430   0.01907

```

## Check 3b: through the heads (selection rows, ordered cross-modal pairs)
```
=== CHECK 3b: through the heads, cross-modal (image head on row i, caption head on row j), selection rows ===
agreement = dot(image-head posterior of row i, caption-head posterior of row j); uniform-random 64-class baseline = 1/64 = 0.01562
part     aspect        same      diff   overall   ratio
affect   emotion    0.04460   0.04020   0.04101   1.109
affect   style      0.04283   0.03995   0.04018   1.072
affect   genre      0.04280   0.03963   0.04008   1.080
image    emotion    0.01879   0.01591   0.01644   1.181
image    style      0.02442   0.01581   0.01650   1.545
image    genre      0.03934   0.01311   0.01687   3.001
caption  emotion    0.01969   0.01868   0.01887   1.054
caption  style      0.02191   0.01903   0.01926   1.151
caption  genre      0.03412   0.01664   0.01915   2.050

episode contrasts through the heads: mean agreement (sameA,diffB) minus (sameB,diffA)
part     pair             sameA/dB  sameB/dA       diff   ratio
affect   emotionxstyle     0.04436   0.04257   +0.00179   1.042
affect   emotionxgenre     0.04387   0.04267   +0.00121   1.028
affect   stylexgenre       0.04236   0.04257   -0.00021   0.995
image    emotionxstyle     0.01780   0.02331   -0.00550   0.764
image    emotionxgenre     0.01415   0.03777   -0.02363   0.375
image    stylexgenre       0.01630   0.03677   -0.02047   0.443
caption  emotionxstyle     0.01934   0.02123   -0.00188   0.911
caption  emotionxgenre     0.01641   0.03355   -0.01714   0.489
caption  stylexgenre       0.01670   0.03330   -0.01660   0.502

```

## Exact vs sampled
```
=== EXACT vs SAMPLED (2000000 pairs, seed 0): 90 comparisons, max |z| = 2.42, mean |z| = 0.79, beyond 4 SE: 0 ===
runtime 13.8 s
```

## Reading what the numbers directly show
- Affect x emotion: the partition is not diffuse. Size-weighted purity 0.518 (lift 3.33); the median emotion needs 4.5 clusters for 50% and
  15 for 80% of its rows against 9 and 28 for the overall size spread, and exp(H) 21.6 against 37.5. Fear and sadness are the most
  concentrated (k80 11 and 10, one cluster holds 30% and 35% of the rows with purity 0.90 and 0.93). The pair-level lift of same emotion
  is 2.17 (p_same 0.0767 vs p_diff 0.0353; random-pair base 0.0430). Affect also carries almost no style or genre information (lift 1.10).
- Hard-partition contrasts for affect: emotion-same pairs share a cluster about twice as often as style-same or genre-same pairs
  (s_AB - s_BA = +0.039 and +0.038, ratios 2.04 and 2.02); style x genre is flat (-0.0001).
- Through the heads the same affect contrasts shrink to +0.0018 (ratio 1.04) and +0.0012 (ratio 1.03). Mean agreement for the affect
  partition is about 0.041 overall in all conditions, and affect x emotion same vs different is 0.0446 vs 0.0402 (ratio 1.11), against a hard-partition
  lift of 2.17. So the signal that exists in the partition is mostly lost between the partition and the cross-modal head dot product.
- Image partition: strong on genre (pair lift 9.75, 3b ratio 3.00) and style (4.59, 1.55); its contrasts are negative (emotion-same pairs
  are less alike than style- or genre-same pairs) in both hard (-0.041 to -0.062) and head (-0.0055 to -0.0236) form.
- Caption partition: genre lift 2.44 hard and 2.05 through the heads; emotion 1.26 and 1.05; style 1.20 and 1.15.

## Caveats
- Emotion labels are per row (per viewer); style and genre per painting. Same-emotion pairs are therefore pairs of independent viewer
  responses, and style or genre sharing is a property of the paintings.
- Checks 1 to 3 use scorer-train rows (where the partitions were fitted); check 3b uses selection rows, a different set and the
  head-posterior view. The numbers are not directly comparable across the two.
- 3b agreement is the dot of the image-head posterior of one row and the caption-head posterior of another; it is not the reader's episode Delta,
  which also depends on episode construction (not built here).
- The "best-purity cluster" in the per-emotion table can be a small cluster; the largest-coverage cluster is given beside it. This was a choice the brief left open.
- Descriptive only. No label-guided choice of partitions may follow from this.
