# Buddy versus PercepT: matched held-out silhouette and occupancy audit

## Population and protocol

Painting-set identity **passed**: both snapshots contain the same 61,402 unique train paintings and 9,365 unique held-out paintings. The PercepT held-out rows were aligned to buddy's painting order before a common seed-42 sample was drawn.

Buddy labels are k=20 cosine-neighbor majority votes onto its 19 frozen train Leiden communities, using the unchanged `assign_to_train_communities` helper. PercepT labels are its saved held-out assignments to 67 surviving train-fitted DEC centers. The independently reclustered buddy held-out communities were not used.

Both silhouettes use the same 6,000 held-out paintings out of 9,365: `np.random.default_rng(42).choice` followed by `silhouette_score(sample_size=min(4000, len(idx)), random_state=42)` with default Euclidean distance. Scores are still measured in each model's own space: 32-D buddy fused embeddings versus 128-D PercepT post-DEC latents. Thus the sampling and distance rule are matched, while the representation geometry is not fully controlled.

## Combined held-out comparison

The predeclared AMI bar requires emotion > 0.1236 and genre > 0.1954, both strict. Collapse means more than half of train labels receive fewer than 1% of held-out paintings. Zero occupancy is counted separately.

| system | emotion AMI / bar | genre AMI / bar | both AMI bars | silhouette | occupancy min / median / max | zero labels | below 1% | collapse rule |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Buddy Attention-h1 | 0.1364 / pass | 0.2530 / pass | pass | 0.0416 | 65 / 523.0 / 1,033 | 0/19 | 3/19 | not collapsed |
| PercepT Variant A | 0.1077 / fail | 0.3453 / pass | fail | 0.4973 | 0 / 13.0 / 1,264 | 21/67 | 50/67 | Collapsed |

Emotion AMI uses all held-out paintings. Genre AMI uses the 159 paintings with genre annotation for buddy and 159 for PercepT; saved genre values were checked against the pipeline genre map.

## Full held-out occupancy histograms

### Buddy: frozen train communities

Min 65; max 1,033; median 523.0; zero-count 0; below 1% 3/19.

| train label | held-out count |
|---:|---:|
| 0 | 1,033 |
| 1 | 811 |
| 2 | 917 |
| 3 | 795 |
| 4 | 945 |
| 5 | 620 |
| 6 | 526 |
| 7 | 594 |
| 8 | 424 |
| 9 | 558 |
| 10 | 523 |
| 11 | 367 |
| 12 | 278 |
| 13 | 240 |
| 14 | 230 |
| 15 | 283 |
| 16 | 87 |
| 17 | 69 |
| 18 | 65 |

### PercepT: surviving train centers

Min 0; max 1,264; median 13.0; zero-count 21; below 1% 50/67.

| train label | held-out count |
|---:|---:|
| 0 | 1,130 |
| 1 | 0 |
| 2 | 142 |
| 3 | 44 |
| 4 | 4 |
| 5 | 241 |
| 6 | 56 |
| 7 | 290 |
| 8 | 0 |
| 9 | 0 |
| 10 | 0 |
| 11 | 0 |
| 12 | 1,072 |
| 13 | 1 |
| 14 | 1 |
| 15 | 0 |
| 16 | 0 |
| 17 | 1 |
| 18 | 0 |
| 19 | 0 |
| 20 | 0 |
| 21 | 0 |
| 22 | 27 |
| 23 | 0 |
| 24 | 0 |
| 25 | 0 |
| 26 | 13 |
| 27 | 740 |
| 28 | 6 |
| 29 | 24 |
| 30 | 0 |
| 31 | 0 |
| 32 | 0 |
| 33 | 674 |
| 34 | 345 |
| 35 | 23 |
| 36 | 79 |
| 37 | 0 |
| 38 | 41 |
| 39 | 32 |
| 40 | 48 |
| 41 | 220 |
| 42 | 0 |
| 43 | 54 |
| 44 | 13 |
| 45 | 184 |
| 46 | 250 |
| 47 | 0 |
| 48 | 3 |
| 49 | 142 |
| 50 | 1 |
| 51 | 17 |
| 52 | 27 |
| 53 | 1 |
| 54 | 6 |
| 55 | 1 |
| 56 | 0 |
| 57 | 21 |
| 58 | 1,264 |
| 59 | 8 |
| 60 | 357 |
| 61 | 878 |
| 62 | 56 |
| 63 | 153 |
| 64 | 2 |
| 65 | 16 |
| 66 | 657 |

## Interpretation and verdict

The published faithful-recipe Variant-A seed-42 run had held-out emotion AMI **0.1092**, genre AMI **0.3288**, and full-split silhouette **0.5120**. Its verdict was **Collapsed**: **50/67** surviving held-out centers were below 1% occupancy, with minimum surviving-center count **0**. It missed the project's own joint AMI Pareto bar as well as its non-collapse criterion. The matched sampled silhouette above is a newly computed value and need not equal the published full-split 0.5120.

The faithful snapshot records full-split held-out silhouette **0.4840**. **WARNING:** this differs from the published 0.5120 beyond four-decimal rounding; inspect the snapshot run.

The earlier held-out label-transfer pilot reported **19/19** buddy train communities covered at k=20; the per-label counts above are recomputed from that same validated transfer rule.

PercepT retains a numerical silhouette advantage under matched sampling (0.4973 versus 0.0416). The occupancy picture is different: PercepT has 21 zero-occupancy surviving centers and 50/67 below 1%, versus buddy's 0 zero-occupancy communities and 3/19 below 1%. This means a larger silhouette alone is weak evidence of better topics: part of the gap is consistent with degenerate, over-confident clustering rather than better coverage or human-label agreement. Occupancy and silhouette do not prove the cause of that gap.

This audit does **not** test whether either system's labels are more useful downstream. Candidate 1's shared image-only mapper probe is deliberately out of scope for this pilot and remains a follow-up decision.
