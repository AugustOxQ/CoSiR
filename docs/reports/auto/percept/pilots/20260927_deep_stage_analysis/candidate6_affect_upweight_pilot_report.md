# Candidate 6 — affect InfoNCE upweighting pilot

Generated 2026-09-28 00:50:26. Local CUDA Stage-1 experiment; no DAS6.

## Method

Copied the established Attention-h1 noise/schedule pilot's data setup, teacher graphs, training loop, cosine LR schedule, plateau stopping, and clean-embedding post-hoc Leiden/AMI/silhouette evaluation. Set noise_std=0.0; the only training change is `total_loss = content_loss + LAMBDA_AFFECT * affect_loss`. Adam LR=0.001, CosineAnnealingLR(T_max=200, eta_min=1e-05). The held-out Pareto bar is emotion AMI > 0.1236 AND genre AMI > 0.1954. A winner also needs emotion AMI > 0.1356 (strictly more than +0.005 over 0.1306).

## Seed-42 screen

| LAMBDA_AFFECT | held-out emotion AMI | held-out genre AMI | held-out silhouette | Pareto bar | practical margin |
|---:|---:|---:|---:|---|---|
| 1 | 0.1306 | 0.1973 | 0.0488 | clears | does not clear |
| 2 | 0.1166 | 0.0352 | 0.0726 | does not clear | does not clear |
| 4 | 0.1180 | 0.0342 | 0.0492 | does not clear | does not clear |
| 8 | 0.1035 | 0.0212 | 0.0406 | does not clear | does not clear |

Baseline sanity check: expected 0.1306 / 0.1973 / 0.0488 (emotion AMI / genre AMI / silhouette), observed 0.1306 / 0.1973 / 0.0488; tolerance ±0.001 per metric. Passed.

## Winner selection

No upweighted point both beats the +0.005 emotion AMI margin and clears both Pareto bars. No non-qualifying point was stress-tested.

## Verdict

Upweighting the affect loss produced lower or unchanged held-out emotion AMI at best (0.1180 at LAMBDA_AFFECT=4), but no setting delivered a meaningful emotion gain while preserving the genre Pareto bar. This screen does not support the flat equal-weight loss as the bottleneck behind Finding D; the weak emotion separation likely depends on the teacher signal, representation, or clustering as well.
