# Combined noise-schedule + pseudo-contrastive: four-seed stress

Generated 2026-09-26 06:29:19.

Seed 42's row is copied from `attention_h1_noise_schedule_pseudo_contrastive_pilot_report.md` (not re-run); seeds 7, 123, 2024 are freshly fit here using the same `run_seed` function, NOISE_STD, and context-building code as that pilot, unchanged.

| seed | held-out emotion AMI | held-out genre AMI | held-out silhouette | Pareto bar |
|---:|---:|---:|---:|---|
| 42 | 0.1210 | 0.2623 | 0.0789 | does not clear |
| 7 | 0.1141 | 0.3111 | 0.0721 | does not clear |
| 123 | 0.1180 | 0.3487 | 0.0692 | does not clear |
| 2024 | 0.1107 | 0.2822 | 0.0753 | does not clear |

## Summary

- Emotion AMI: mean=0.1160; min=0.1107; max=0.1210.
- Genre AMI: mean=0.3011; min=0.2623; max=0.3487.
- Silhouette: mean=0.0739; min=0.0692; max=0.0789.
- Both held-out Pareto bars clear simultaneously in **0/4 seeds**.
