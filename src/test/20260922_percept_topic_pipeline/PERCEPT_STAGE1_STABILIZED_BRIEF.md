# Brief: PercepT Stage 1 — diagnose and stabilize the DEC collapse

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_stabilized_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: what collapsed and the two concrete diagnoses

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`
(the exact base to copy and modify) and
`src/test/20260922_percept_topic_pipeline/percept_stage1_pilot_report.md`
(the collapsed result) in full first.

That pilot's DEC converged cleanly by its own stability criterion (epoch 275,
`fraction_changed < 0.001`) but collapsed: of 67 surviving topics, the median
cluster size was 0 and 65/67 clusters held under 1% of nodes, with the largest
holding ~50% of all 61,402 train paintings. Reconstruction loss and KL loss
were both logged throughout training — inspect them yourself, but note two
concrete, numeric diagnoses already visible in that report's training
trajectory:

1. **The reconstruction anchor is far too weak relative to the clustering
   term by the time collapse sets in.** By epoch 275, `KL=0.064992` vs.
   `reconstruction=0.000085` — roughly 760x smaller. `LAMBDA_RECONSTRUCTION =
   1.0` therefore contributes almost no gradient signal once training
   progresses; the loss is effectively pure DEC self-sharpening with no real
   reconstruction grounding, which is exactly the collapse mode described in
   this project's own PercepT-brainstorm risk note ("DEC can also
   over-sharpen arbitrary partitions").
2. **There is no diagnostic visibility into WHEN the collapse happened** —
   whether the K-means initialization already produced a skewed 100-cluster
   size distribution that DEC then only amplified, or whether DEC's
   self-sharpening created the skew from a reasonably balanced start. The
   collapsed pilot only logged `fraction_changed` (how many hard assignments
   flip per epoch), never the actual size distribution during training.

## What to change (two independent, additive interventions — implement both)

### 1. Add a balanced-assignment regularizer (the primary stabilization)

This is the standard, well-established fix for exactly this DEC/self-training
collapse mode (used broadly in self-supervised clustering, e.g. SwAV's
equipartition constraint): penalize the batch-average soft assignment for
drifting away from uniform across clusters. Because this pilot trains
full-batch (all 61,402 train nodes every epoch, not minibatches), the
"batch average" is the true global assignment distribution each epoch — the
ideal setting for this regularizer, no minibatch noise to worry about.

Add, inside the DEC training loop, after computing `q` (the soft assignment
over the current `N_INITIAL_CLUSTERS`-count centers, before any pruning):

```python
mean_q = q.mean(dim=0)  # shape [K], the global average assignment this epoch
uniform = torch.full_like(mean_q, 1.0 / mean_q.shape[0])
balance_loss = F.kl_div(mean_q.clamp_min(1e-8).log(), uniform, reduction="sum")
```

New loss: `total_loss = kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss
+ LAMBDA_BALANCE * balance_loss`, with a new module-level constant
`LAMBDA_BALANCE = 1.0`. Log `balance_loss` alongside `total`/`KL`/
`reconstruction` at the same cadence the base script already uses (every 25
epochs, plus at the stopping epoch). Do not change `LAMBDA_RECONSTRUCTION` —
isolate the balance term as the one new intervention so its effect is
attributable, rather than conflating it with a second untested change to the
reconstruction weight.

### 2. Log cluster-size distribution diagnostics throughout training

At the SAME checkpoints where losses are currently logged (every 25 epochs,
plus the stopping epoch), also compute and log the hard-assignment
(`argmax(q)`) cluster-size distribution over the full `N_INITIAL_CLUSTERS`
(100) centers — not just at the end, and not just on the pruned 67. Log:
min, max, median, and the count of clusters below 1% of `N` (the same
threshold the collapse rule uses). Additionally, log this same distribution
immediately after K-means initialization, BEFORE the first DEC epoch runs —
label it "epoch 0" — so the report can show whether the skew is already
present at init or purely DEC-induced. This diagnostic must appear in the
final report as its own trajectory (not just folded into the loss line),
because the reason for a stabilization result — collapse fixed vs. not fixed
vs. fixed but skewed toward a different degenerate shape — is only visible
through this distribution over time, not through the final size table alone.

## Everything else stays identical

Keep the fused embedding construction, autoencoder architecture, K-means
init (100 clusters, same seed), convergence criterion (`fraction_changed <
0.001`, ceiling 500 — unchanged; the balance term may change how many epochs
it takes to converge, that is expected and fine, just log the real number),
norm-threshold pruning to 67, and the full train/held-out evaluation
pipeline (same monkey-patch pattern, same `external_metrics`/`load_genre_map`
calls, same predeclared Pareto bar and collapse rule, same verdict language)
exactly as the base script implements them. This must be a controlled,
single-question test: does adding a balance regularizer fix the collapse,
holding everything else fixed.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_stabilized_pilot_report.md`
with the same structure as the base script's report, plus:
- A new "Cluster-size distribution over training" section showing the
  min/max/median/below-1%-count trajectory (including the epoch-0 K-means-
  init snapshot) at every logged checkpoint, so the reader can see whether
  collapse emerged gradually, immediately, or not at all.
- The balance-loss trajectory alongside the existing total/KL/reconstruction
  trajectory.
- An explicit before/after comparison against the collapsed base pilot's
  headline numbers (emotion AMI, genre AMI, collapse verdict, for both
  splits) in one table.
- The same predeclared Pareto bar and collapse-verdict language as the base
  script. Do not soften or reinterpret the collapse rule even if AMI numbers
  look attractive — the point of this pilot is exactly to test whether the
  balance regularizer earns a legitimately non-collapsed result under the
  SAME rule that caught the previous collapse.
