# PercepT Stage 1 DEC reconstruction anneal pilot

Generated automatically, 2026-09-24 14:26:54.

## Controlled change

K=60/40, seed 42, the same forced-deterministic fit, and `LAMBDA_BALANCE=1000` are retained. The only training-objective change is the reconstruction coefficient: `1 - (1 - 0.1) * min(epoch, 200) / 200` for 1-indexed DEC epochs, then 0.1. The 200-epoch duration and 0.1 floor are first-pass, untuned choices. The hypothesis is that reconstruction weight 1 throughout the run opposes KL clustering and leaves the post-DEC latent geometry loose. The balance regularizer remains at 1000 because earlier unbalanced DEC collapsed.

DEC stopped at epoch 82 by stability criterion; whole-script wall time 409.4 seconds. Train N=61,402; held-out N=9,365.

## Comparison with established no-anneal baseline

The fixed-weight-1.0 AMI baseline comes from the prior `run_percept_stage1_embedding_snapshot_pilot.py` run. Its silhouette values come from an ad hoc check of that snapshot. The pre-DEC silhouette is a correctness check on the unchanged pretraining path, not a new result or a gate. Both silhouette calculations here use a seeded sample of at most 6,000 rows, followed by `silhouette_score(..., sample_size=min(4000, N), random_state=42)`.

| Metric | No anneal baseline | This annealed run | Absolute difference |
|---|---:|---:|---:|
| Held-out emotion AMI | 0.122500 | 0.123837 | 0.001337 |
| Held-out genre AMI | 0.227400 | 0.261700 | 0.034300 |
| Train silhouette pre-DEC | 0.048900 | 0.047397 | 0.001503 |
| Train silhouette post-DEC | 0.040200 | 0.039206 | 0.000994 |

Train post-DEC silhouette declined versus the 0.0402 no-anneal baseline by 0.000994.

## Surviving-topic size diagnostic (train hard assignments)

| Statistic | This run | Known K=60/40 baseline |
|---|---:|---:|
| Minimum topic size | 257 | not recorded |
| Maximum topic size | 5,290 | not recorded |
| Median topic size | 1069.0 | not recorded |
| Topics below 1% of train N | 8/40 | 0/40 |

## Verdict

**Collapsed.** Non-collapsed means no more than 0 of 40 surviving train topics below 1% of train N. The held-out Pareto bars are emotion AMI > 0.1236 and genre AMI > 0.1954; both must clear for Real success.
