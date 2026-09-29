# Brief: Attention-h1 + hypersphere-appropriate (vMF-style) DEC clustering loss

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_attention_h1_vmf_dec_hybrid_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: a diagnosed geometry mismatch, not a repeat of the failed attempt

Read `run_attention_h1_dec_hybrid_pilot.py` and its report in full first. That
pilot transplanted PercepT's exact DEC mechanism (Euclidean Student's-t soft
assignment, `soft_assignments`/`target_distribution` from
`../20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`) onto
Attention-h1's 32-D embedding and got a **robust, decisive failure**: held-out
silhouette went **negative** (-0.0202 to -0.0450 across all 4 stress seeds,
all at the selected LAMBDA_DEC=0.5), genre AMI collapsed to 0.10-0.18 (from
baseline 0.2404), and roughly half the 67 surviving centers were near-empty
in most runs. The diagnosed cause, stated in that report and confirmed by
the calling session's own reasoning: Attention-h1's embedding is
**LayerNorm'd and L2-normalized onto the unit hypersphere** (unlike
PercepT's free, unconstrained 128-D autoencoder latent). Euclidean distance
on a low-dimensional (32-D) unit sphere is compressed and a poor separation
signal, and 100 K-means centers plus a self-sharpening Euclidean kernel
fighting for space on that sphere, while InfoNCE simultaneously reshapes the
same coordinates, produced exactly the pathological interleaving a negative
silhouette measures.

This pilot tests the natural, geometry-appropriate fix: **replace DEC's
Euclidean/Student's-t kernel with a cosine-similarity-based (von
Mises-Fisher-style) soft assignment**, keeping every other part of the
mechanism (self-sharpening target, joint training, KL loss, learnable
centers, norm-based pruning) structurally the same. This is not a re-run of
the failed attempt with different hyperparameters — it changes the
mathematical form of the clustering kernel itself to match the space it
operates in.

## Implementation

Reuse `run_attention_h1_dec_hybrid_pilot.py`'s structure wholesale: sibling
imports (including its import of the PercepT base module for
`prune_centers`, which is **still Euclidean-norm-based and fine to reuse
unchanged** — pruning by center norm is a separate concern from the
assignment kernel), Attention-h1 architecture and teacher losses, cosine LR
schedule, plateau stopping, checkpoint cadence, and report-writing
conventions.

**Replace only the soft-assignment and target-distribution functions.**
Do not import `base.soft_assignments`/`base.target_distribution` for the
DEC loss in this pilot (still import `base.prune_centers` for the final
pruning step, unchanged). Implement, in this new file:

```python
def vmf_soft_assignments(embedding: torch.Tensor, centers: torch.Tensor, kappa: float) -> torch.Tensor:
    """Cosine-similarity soft assignment: q_ij ~ softmax_j(kappa * cos_sim(z_i, mu_j)).
    `embedding` is assumed already unit-normalized (Attention-h1's own output).
    `centers` must be renormalized to unit norm before the cosine similarity,
    every step, since gradient steps can move them off the unit sphere even
    though they were unit-norm at initialization."""
    unit_centers = torch.nn.functional.normalize(centers, dim=-1)
    logits = kappa * (embedding @ unit_centers.T)
    return torch.softmax(logits, dim=-1)


def vmf_target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Same self-sharpening construction as DEC/PercepT's target_distribution
    (square-and-frequency-normalize), applied to vMF-kernel assignments
    instead of Student's-t ones -- the sharpening idea is kernel-agnostic."""
    weight = q ** 2 / q.sum(dim=0)
    return (weight.t() / weight.sum(dim=1)).t()
```

Initialize centers by K-means on the clean epoch-0 embedding exactly as the
failed pilot did (100 clusters, `n_init=10`, `random_state=seed`), but
**immediately L2-normalize the K-means centroids to unit norm** before
wrapping them in `nn.Parameter` — K-means on already-unit-norm points
produces centroids that are the mean of unit vectors, which are NOT unit
norm themselves (their norm shrinks with within-cluster spread), so this
normalization step is required and is a real, necessary difference from the
failed pilot's initialization, not cosmetic. Pick a `kappa` (concentration
parameter — higher means sharper/more confident assignment, analogous to a
temperature) as a second thing to screen alongside `LAMBDA_DEC` (see below);
do not hardcode an unvalidated single value.

DEC loss: `dec_loss = F.kl_div(q.log(), p, reduction="batchmean")` using
`vmf_soft_assignments`/`vmf_target_distribution` in place of the Euclidean
versions — same KL formula, same warm-up ramp over 30 epochs, same joint
optimizer/schedule, same collapse-diagnostic logging (all-100-center
min/max/median/below-1%, at the same cadence) as the failed pilot. Final
pruning: L2-normalize centers, then call `base.prune_centers` on them
exactly as before (norm-based pruning still makes sense post-normalization
since gradient steps can still produce meaningfully different final norms
before this fixed final re-normalization — actually, re-examine this: if
centers are kept unit-norm at every step via `vmf_soft_assignments`'s own
internal renormalization, their raw `nn.Parameter` values may still drift in
norm between renormalized reads. Decide and document explicitly in a
comment whether pruning should act on the raw parameter's own norm or a
different signal such as each center's assignment mass or average
within-cluster cosine similarity — a norm-based prune on a direction-only
representation is not obviously meaningful the way it is in PercepT's free
Euclidean latent, and this is a real design decision the previous pilot did
not have to make. State your choice and reasoning plainly in the script's
docstring; if genuinely unsure, prefer pruning by **assignment mass** (drop
the lowest-total-soft-assignment-mass centers) as the more directly
meaningful criterion on a hypersphere, and say so.**

## Two-dimensional single-seed screen, then stress only the winner

**Seed 42, screen `LAMBDA_DEC` in `(0.1, 0.5, 1.0)` crossed with `kappa` in
`(4.0, 16.0)`** — six seed-42 runs total. `kappa` brackets a soft, low-
confidence assignment (4.0) against a much sharper one (16.0); do not test
more than these two initially, this is already a 6-run grid and a further
axis is out of scope for a single pilot. Report the same metrics as the
failed pilot's screen table for all six: held-out emotion AMI, genre AMI,
silhouette, held-out Pareto bar, and the collapse diagnostics (both the
all-100 pre-prune and post-prune surviving-center numbers).

**Winner selection:** among cells clearing the held-out Pareto bar, pick
the one with the highest held-out silhouette; if none clears, pick the
highest held-out silhouette among all six and label it explicitly as this
investigation's established "best available, does not clear" wording. Then
stress that single `(LAMBDA_DEC, kappa)` pair at `SEEDS = (7, 123, 2024)`,
same methodology as every other 4-seed stress in this project.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/attention_h1_vmf_dec_hybrid_pilot_report.md`
with: the 6-cell screen table, the pruning-criterion decision stated
plainly, the winner-selection rationale, the four-seed stress table and
summary statistics, and an explicit comparison against ALL SEVEN reference
points now established in this investigation: (a) Attention-h1 baseline
(0.1249/0.2404, silhouette 0.0392), (b) Leiden pseudo-contrastive alone
(0.1165/0.2351, silhouette 0.0877), (c) noise+schedule alone
(0.1306/0.1973 seed 42; four-seed mean 0.1277/0.2396, silhouette mean
0.0466), (d) noise-schedule+pseudo-contrastive (0.1210/0.2623, silhouette
0.0789, missed the bar), (e) the **Euclidean** DEC hybrid that failed
(four-seed held-out silhouette mean approximately -0.029, all 4 seeds
missed the Pareto bar — read its report for the exact mean), (f) PercepT-
replication standing balance-hack (0.1252/0.2486, silhouette never
measured), (g) PercepT-replication faithful recipe (0.1092/0.3288,
silhouette 0.5120). State plainly whether the geometry fix resolves the
collapse (silhouette back to positive and competitive with or better than
the best buddy variant so far), whether it clears the held-out AMI Pareto
bar, and whether that holds robustly across 4 seeds. If this also fails,
report that as bluntly as every other negative result in this project,
and note explicitly that the geometry-mismatch hypothesis itself would then
need to be questioned, not just this particular fix.

Do not touch git, do not modify any other file in the repository.
