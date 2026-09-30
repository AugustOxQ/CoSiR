# D_SHARED capacity sweep pilot

Generated automatically, 2026-09-26 23:41:01.

## Method

The shared architecture file was not modified. The pilot sets `arch.D_SHARED` on the imported sibling module immediately before constructing each `arch.LearnedStudent("attn1")`. Widths 64 and 128 use the unchanged snapshot pilot's train-only 50-D content PCA, 28-D affect probabilities, content and affect teacher graphs, symmetric two-teacher InfoNCE, temperature, fixed Adam learning rate, pair sampling, checkpoint cadence, recall plateau stopping, and independent train/held-out Leiden passes. No noise, cosine schedule, or clustering loss was added.

The D_SHARED=32 seed-42 AMIs and silhouette are cited from `attention_h1_embedding_snapshot_pilot_report.md` and `attention_h1_baseline_seed_stress_pilot_report.md`; it was not retrained. Its occupancy and embedding-only diagnostics are recovered from `attention_h1_embedding_snapshot.npz`, using the same `arch.evaluate_checkpoint` helper and seed-42 diagnostic draws as the new widths. The helper computes held-out content/affect teacher-graph recall, effective rank at 95% variance, and top covariance eigenvalue fraction. The cited AMIs use all held-out paintings for emotion and the genre-labelled subset for genre.

Held-out silhouette uses a fresh `np.random.default_rng(42)` draw of at most 6,000 points followed by `silhouette_score` with `sample_size=min(4000, len(idx))`, `random_state=42`, default Euclidean distance, and independent held-out Leiden labels. Occupancy counts all observed Leiden communities on each full split; below 1% means strictly fewer than 1% of that split's paintings. The held-out Pareto bar is emotion AMI > 0.1236 and genre AMI > 0.1954 (both strict).

Screen-to-stress gate: both AMI bars and held-out silhouette at least 0.0200 above the cited seed-42 D_SHARED=32 value 0.0377, i.e. at least 0.0577. If both widths qualify, only the higher-silhouette width is stressed (smaller width breaks an exact tie). This 0.0200 absolute bar exceeds the baseline's four-seed silhouette range (0.0063) and the schedule-only seed-42 increase (0.0111).

## Seed-42 capacity screen

| D_SHARED | seed | held-out effective rank 95% | top eigen fraction | content recall | affect recall | held-out emotion AMI | held-out genre AMI | train Leiden occupancy | held-out Leiden occupancy | held-out silhouette | Pareto bar |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 32 | 42 | 20 | 0.1027 | 0.0914 | 0.1457 | 0.1249 | 0.2404 | 19 communities; 430/3120.0/7132 min/median/max; 1/19 below 1% | 18 communities; 85/521.5/955 min/median/max; 2/18 below 1% | 0.0377 | clears |
| 64 | 42 | 33 | 0.0796 | 0.1421 | 0.1288 | 0.1153 | 0.2918 | 23 communities; 181/2470.0/6282 min/median/max; 2/23 below 1% | 18 communities; 100/488.0/959 min/median/max; 0/18 below 1% | 0.0367 | misses |
| 128 | 42 | 46 | 0.0697 | 0.1701 | 0.1271 | 0.1089 | 0.3319 | 24 communities; 16/1679.0/6993 min/median/max; 5/24 below 1% | 18 communities; 136/446.5/1078 min/median/max; 0/18 below 1% | 0.0250 | misses |

## Stress decision

No width cleared both AMI bars and the +0.0200 held-out silhouette gate, so no additional seed was trained.

## Final verdict

Capacity alone did not yield a qualifying improvement: best seed-42 widened silhouette was 0.0367 at D_SHARED=64 versus 0.0377 at 32, with Pareto bar missed. Under the unchanged two-InfoNCE objective, widening does not explain the reported silhouette gap to PercepT.
