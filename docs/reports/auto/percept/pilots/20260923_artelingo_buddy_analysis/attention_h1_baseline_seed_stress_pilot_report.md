# Unmodified Attention-h1 baseline: four-seed robustness check

Generated 2026-09-26 06:19:20.

Each seed runs `run_attention_h1_embedding_snapshot_pilot.py` as-is, including its Attention-h1 architecture, two InfoNCE losses, fixed Adam learning rate, stopping rule, and train/held-out Leiden passes. Only the module's seed and temporary output paths change. All four models were freshly fit. Held-out silhouette uses a seed-42 draw of at most 6,000 points followed by `silhouette_score` with `sample_size=min(4000, len(idx))`, `random_state=42`.

| seed | held-out emotion AMI | held-out genre AMI | held-out silhouette | Pareto bar | train Leiden communities | held-out Leiden communities |
|---:|---:|---:|---:|---|---:|---:|
| 42 | 0.1249 | 0.2404 | 0.0377 | clears | 19 | 18 |
| 7 | 0.1213 | 0.2544 | 0.0397 | does not clear | 19 | 17 |
| 123 | 0.1238 | 0.2386 | 0.0375 | clears | 20 | 15 |
| 2024 | 0.1264 | 0.2289 | 0.0438 | clears | 22 | 17 |

The held-out Pareto bar requires emotion AMI > 0.1236 **and** genre AMI > 0.1954 (strict inequalities).

## Held-out summary

| metric | mean | min | max |
|---|---:|---:|---:|
| Emotion AMI | 0.1241 | 0.1213 | 0.1264 |
| Genre AMI | 0.2406 | 0.2289 | 0.2544 |
| Silhouette | 0.0397 | 0.0375 | 0.0438 |

Pareto-bar clearance: **3/4 seeds**.

## Comparison with noise + cosine schedule

The noise + schedule pilot's reported four-seed values are listed alongside the untouched baseline below. Differences are schedule minus baseline, paired by seed.

| seed | schedule emotion / genre / silhouette | schedule − baseline emotion / genre / silhouette | schedule bar |
|---:|---:|---:|---|
| 42 | 0.1306 / 0.1973 / 0.0488 | +0.0057 / -0.0431 / +0.0111 | clears |
| 7 | 0.1244 / 0.2583 / 0.0438 | +0.0031 / +0.0039 / +0.0041 | clears |
| 123 | 0.1334 / 0.2452 / 0.0487 | +0.0096 / +0.0066 / +0.0112 | clears |
| 2024 | 0.1222 / 0.2576 / 0.0451 | -0.0042 / +0.0287 / +0.0013 | does not clear |

- emotion AMI: baseline mean 0.1241 (range 0.1213–0.1264); schedule mean 0.1276; mean difference +0.0036.
- genre AMI: baseline mean 0.2406 (range 0.2289–0.2544); schedule mean 0.2396; mean difference -0.0010.
- silhouette: baseline mean 0.0397 (range 0.0375–0.0438); schedule mean 0.0466; mean difference +0.0069.
- Pareto clearance: baseline 3/4; schedule 3/4. The schedule's 3/4 clearance also occurs without the schedule, so this count alone is within the baseline's observed seed variability.
- Per-seed increases occur in 3/4 emotion AMIs, 3/4 genre AMIs, and 4/4 silhouettes; these paired comparisons show whether gains are consistent across seeds.
