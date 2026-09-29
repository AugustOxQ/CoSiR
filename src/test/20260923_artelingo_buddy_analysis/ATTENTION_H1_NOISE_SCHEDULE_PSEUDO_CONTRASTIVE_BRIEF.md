# Brief: Attention-h1 combining noise+schedule with the Leiden pseudo-contrastive loss

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_attention_h1_noise_schedule_pseudo_contrastive_pilot.py`.
Do NOT run it — execution happens separately, on a DAS6 GPU node, outside
this task.

## Context: combining two independently-positive mechanisms

Two separate, already-completed pilots each moved held-out silhouette up
from the Attention-h1 baseline (0.0392) without being combined with each
other yet:

1. `run_attention_h1_leiden_pseudo_contrastive_pilot.py`: keeps h1's two
   teacher InfoNCE losses, adds a **third** symmetric InfoNCE loss whose
   positives are same-Leiden-community pairs, reclustering every 20 epochs
   starting at epoch 1, with the third loss's weight ramping linearly from
   0 to `LAMBDA_CLUSTER_MAX=1.0` over `CLUSTER_WARMUP_EPOCHS=50`. Result:
   held-out emotion/genre AMI 0.1165/0.2351 (both slightly below the plain
   h1 baseline of 0.1249/0.2404), held-out silhouette 0.0877 (up from
   0.0392).
2. `run_attention_h1_noise_schedule_pilot.py` (sibling pilot, results not
   necessarily known yet when you write this): adds Gaussian noise before
   each training InfoNCE loss (renormalized to unit L2 norm afterward) plus
   an Adam 1e-3-to-1e-5 cosine-annealed learning rate in place of h1's fixed
   rate, screened over `noise_std in (0.0, 0.02, 0.05, 0.1)`.

This pilot tests whether combining both mechanisms compounds their
silhouette gains, or interacts destructively the way a separate PercepT-
replication combination attempt did (there, stacking an unrelated balance
regularizer onto a noise+cosine-LR recipe caused catastrophic collapse,
apparently because the balance term's gradient was calibrated for a flat,
lower learning rate and dominated early under the new schedule's higher
initial rate). Read both sibling reports named above in full, plus both
sibling scripts, before writing any code — you must reuse their exact
mechanisms, not reimplement close variants.

## Implementation

Start from `run_attention_h1_noise_schedule_pilot.py`'s structure (its
`run_seed`, `noisy_unit_embedding`, cosine-scheduled optimizer, sibling-
import pattern, and report-writing conventions) and add
`run_attention_h1_leiden_pseudo_contrastive_pilot.py`'s third loss on top,
unchanged in its own mechanics: reclustering cadence (every 20 epochs
starting at epoch 1), linear ramp of its weight from 0 to
`LAMBDA_CLUSTER_MAX=1.0` over `CLUSTER_WARMUP_EPOCHS=50`, its degenerate-
recluster-skip guard (`DOMINANT_FRACTION_GUARD=0.8`), and its own same-
community symmetric InfoNCE formulation exactly as that script implements
it. The third loss's embedding input should also go through
`noisy_unit_embedding` before entering its InfoNCE, for the same reason the
other two losses do (train-time-only perturbation, evaluation stays clean)
— state this explicitly as a design choice in the docstring since neither
sibling pilot combines these two changes on the same loss term.

**Do not re-screen noise_std or re-decide the cosine schedule's T_max/floor
from scratch.** Use whichever `noise_std` the noise-schedule sibling pilot's
report names as its four-seed stress winner (read that report; if it is not
yet written when you implement this, use a placeholder constant
`NOISE_STD = None` at the top of the file with a clear `TODO` comment
naming exactly where the calling session must fill in the winning value
before running, and raise a clear `RuntimeError` at the top of `main()` if
it is still `None` — do not guess a value). Use the same `LR_START=1e-3`,
`LR_FLOOR=1e-5`, `T_max=arch.MAX_EPOCHS` cosine schedule unchanged.

## Single-seed check, then stress only if it clears

**Seed 42 only, no further screening within this pilot** — the point is to
test one specific, well-motivated combination, not sweep a new grid. Report
held-out emotion AMI, held-out genre AMI, held-out silhouette, held-out
Leiden community count, and the held-out Pareto bar (emotion AMI > 0.1236
AND genre AMI > 0.1954), plus the full training/LR/loss trajectory
including the third loss's weight and value at each logged checkpoint and
the reclustering trajectory (community count, dominant fraction) the
pseudo-contrastive sibling already logs.

**If and only if seed 42 clears the held-out Pareto bar**, stress-test with
`SEEDS = (7, 123, 2024)`, same methodology as every other 4-seed stress in
this investigation. If it does not clear the bar, skip stress entirely and
report the seed-42 result as a plain miss — do not sweep the third loss's
weight, warmup length, or reclustering cadence; that is new scope outside
this brief.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/attention_h1_noise_schedule_pseudo_contrastive_pilot_report.md`
with the seed-42 result, full trajectories (including reclustering and the
third loss), the four-seed table if stress ran, and an explicit comparison
table against all of: (a) plain Attention-h1 baseline (0.1249/0.2404,
silhouette 0.0392), (b) Leiden pseudo-contrastive alone (0.1165/0.2351,
silhouette 0.0877), (c) noise+schedule alone (read its own report's winning
seed-42 and, if available, four-seed numbers), (d) PercepT-replication
standing balance-hack (0.1252/0.2486, silhouette never measured), (e)
PercepT-replication faithful recipe (0.1092/0.3288, silhouette 0.5120). A
final section stating plainly: does combining beat both individual buddy
mechanisms on silhouette, does it retain or lose AMI relative to each, and
does it beat either PercepT-replication reference on any axis. If the
combination collapses or clearly underperforms both individual mechanisms,
say so as bluntly as this investigation's other negative results have been
reported — do not soften it.

Do not touch git, do not modify any other file. This script is meant to
run on a DAS6 cluster node (via this project's `cluster-run` skill), not
necessarily this container — do not assume any container-specific paths
beyond what the sibling scripts already use, and do not hardcode a device
count or GPU index.
