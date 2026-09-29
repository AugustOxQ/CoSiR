# Brief: decoupled clustering head, take two — detach the gradient path

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_attention_h1_decoupled_cluster_head_detached_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: the one variable this pilot isolates

Read, in full, before writing any code:
- `run_attention_h1_decoupled_cluster_head_pilot.py` (the sibling this pilot
  is a minimal variant of) and its report
  `attention_h1_decoupled_cluster_head_pilot_report.md`.

That sibling's own diagnosis, after running it: it fed `affect_embeddings`
(the clean fused embedding, still attached to the autograd graph) into
`cluster_head` without detaching it first. As a result, `dec_loss`'s
gradient flowed back through `cluster_head` **and through the shared
`LearnedStudent` trunk**, actively reshaping the same InfoNCE-optimized
embedding the DEC loss was supposed to be kept away from. That pilot's
result was the worst of three DEC-hybrid attempts tried in this
investigation: four-seed mean held-out fused silhouette -0.1568, genre AMI
collapsed to 0.0423, 0/4 seeds cleared the Pareto bar, and even the
cluster-head's own latent space only reached a near-zero mean silhouette
(-0.0062) — so decoupling the *representation* without decoupling the
*gradient path* achieved neither goal.

**This pilot changes exactly one thing**: detach `affect_embeddings` before
it enters `cluster_head`, so `dec_loss`'s gradient can only update
`cluster_head`'s own parameters and the DEC centers — never the shared
`LearnedStudent` trunk. `content_loss` and `affect_loss` continue to update
the trunk exactly as before, completely unaffected by this change. This
isolates whether the decoupled-clustering-head *hypothesis itself* (give
DEC a genuinely free Euclidean space to cluster in) has any merit, now that
the confound from the first attempt is removed.

## Implementation

Copy `run_attention_h1_decoupled_cluster_head_pilot.py` verbatim except for
this one change in `run_seed`:

```python
cluster_latent = cluster_head(affect_embeddings.detach())
```

(replacing the sibling's `cluster_latent = cluster_head(affect_embeddings)`
line). Everything else — `ClusterHead`'s definition, the K-means
initialization on the cluster-head's own initial latent, the shared
optimizer construction (`cluster_head.parameters()` still included — its
own weights must still train, only the *input* is detached, not its
parameters), `LAMBDA_DEC` screen values `(0.1, 0.5, 1.0)`, the 30-epoch
linear warm-up, `prune_centers`, dual-space silhouette reporting (cluster-
head latent and fused embedding), the seed-42 screen followed by a 3-seed
stress of the winner, and every diagnostic/report-writing helper — stays
identical to the sibling. Update the module docstring and the report's
"Method" section to state plainly that this variant detaches the cluster
head's input and why (one sentence, referencing the sibling's diagnosed
gradient-leakage failure).

Update `REPORT_PATH` to
`attention_h1_decoupled_cluster_head_detached_pilot_report.md`, and add one
more row to the `REFERENCES` tuple for "Decoupled cluster head, un-detached
(four-seed mean)" using that sibling's own actual four-seed numbers (read
them from its report file — 0.0845/0.0423 AMI, silhouette -0.1568 — rather
than guessing) so the new report can compare directly against the exact
failure it's trying to fix, alongside the same seven reference points the
sibling used.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/attention_h1_decoupled_cluster_head_detached_pilot_report.md`
with the same structure as the sibling's report (screen table, collapse
diagnostics, full trajectories, winner selection, four-seed stress table,
reference comparison, final numeric verdict, and the "do the two
silhouette spaces move together" section). State explicitly, in the final
verdict section, whether detaching the gradient path recovered the fused
embedding from the un-detached sibling's -0.1568 mean silhouette, and
whether the cluster-head's own latent space (now genuinely isolated from
InfoNCE's pressure) achieves a meaningfully separable clustering space this
time. This investigation's established convention is a blunt, numeric
verdict — if this still underperforms every simpler buddy variant, report
that plainly.

Do not touch git, do not modify any other file in the repository.
