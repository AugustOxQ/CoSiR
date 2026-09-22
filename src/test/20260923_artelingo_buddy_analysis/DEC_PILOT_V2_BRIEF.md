# Brief: DEC pilot v2 — proper convergence control

Write a new script `src/test/20260923_artelingo_buddy_analysis/run_dec_pilot_v2.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context

`run_dec_pilot.py` (already committed) ran DEC on the GoEmotions-affect-only
ArtELingo nodes for a FIXED 100 epochs at lr=1e-3, and got emotion AMI=0.1258
(vs. Leiden's 0.1180 reference on the same input) — a modest, inconclusive gain.
The loss trajectory (both KL and reconstruction terms) rose *monotonically*
across all 100 epochs, never plateauing — a sign the run had not reached an
equilibrium, not a sign DEC's real ceiling was found. The original DEC paper
does not train for a fixed epoch count; it stops based on an
assignment-stability criterion (halt once fewer than 0.1% of points change
their hard cluster assignment between consecutive epochs).

Read `run_dec_pilot.py` in full first. Reuse its `load_sibling_module()`,
`log()`, `build_autoencoder()`, `pretrain_autoencoder()`, `soft_assignments()`,
`target_distribution()`, `initialize_cluster_centers()`, `evaluate_clusters()`,
`final_silhouette()` by importing it as a library the same way it imports
`run_pipeline.py`/`run_affect_pilot.py` (`importlib.util.spec_from_file_location`).
Do not reimplement these — only the DEC joint-training loop and the report
change in this v2 script.

## What changes from v1

1. **Lower DEC-stage learning rate**: `1e-4` instead of `1e-3` for the joint
   training optimizer (Adam over encoder + decoder + centers). Pretraining
   stage is UNCHANGED (still lr=1e-3, 50 epochs, batch_size=1024) — reuse
   `pretrain_autoencoder()` from `run_dec_pilot.py` as-is.

2. **Assignment-stability stopping, with a safety ceiling**:
   - `MAX_DEC_EPOCHS = 500` (up from 100).
   - `STABILITY_THRESHOLD = 0.001` (0.1% of the 61,402 nodes = ~61 nodes).
   - Every epoch: after the optimizer step, recompute hard assignments
     (`argmax` of the soft assignment `Q` using the just-updated encoder/centers,
     no_grad, same as the final-evaluation pattern in `run_dec_pilot.py`).
     Compare to the previous epoch's hard assignments (keep the very first
     epoch's assignment as the initial "previous" state coming out of the
     KMeans initialization). Compute `fraction_changed = (num_differing) /
     61402`.
   - Stop the loop (break) as soon as `fraction_changed < STABILITY_THRESHOLD`
     for the first time, OR when `MAX_DEC_EPOCHS` is reached — whichever
     comes first. Record which one triggered the stop and at what epoch.
   - Log `fraction_changed` every 10 epochs via `log()`, and always log it
     (and the loss components) on the exact epoch where the stopping
     condition fires.
   - Also log total/KL/reconstruction loss every 25 epochs (this run may go
     much longer than v1's 100 epochs, so don't spam every-10-epoch logs for
     the loss the whole way — every 25 is enough; but DO log
     `fraction_changed` every 10 epochs as stated above, since that's the
     convergence signal we actually care about this time).

3. Everything else identical to v1: same input (L2-normalized, mean-pooled
   GoEmotions sigmoid probabilities via `extract_affect_nodes()` +
   `l2_normalize()`, same as `run_dec_pilot.py`), same architecture
   (28→64→32→16 encoder, mirrored decoder), same K=28 KMeans-initialized
   centers, same lambda_R=1.0, same seed=42.

## Evaluation and report

After the loop stops: hard-assign via final argmax(Q), same collapse-detection
check as v1 (cluster sizes, how many of 28 clusters hold <1% of nodes /
<614 nodes), community-vs-emotion and community-vs-genre AMI/V-measure via
`evaluate_clusters()`, and silhouette via `final_silhouette()` — all reused
from `run_dec_pilot.py`.

Write `src/test/20260923_artelingo_buddy_analysis/dec_pilot_v2_report.md`:
- State whether the run stopped via the stability criterion or hit the
  epoch ceiling, and at what epoch.
- Show the `fraction_changed` trajectory (a handful of checkpoint values,
  not every epoch) so convergence behavior is visible.
- Show the loss trajectory the same way.
- Cluster-size collapse-detection stats, same format as v1's report.
- A THREE-row comparison table (not two) — "Leiden on GoEmotions-only":
  emotion AMI 0.1180, genre AMI 0.0396; "DEC v1 (100 fixed epochs,
  lr=1e-3, non-converged)": emotion AMI 0.1258, genre AMI 0.0365; "DEC v2
  (converged or epoch-capped, lr=1e-4)": the actual numbers from this run.
  Hardcode the first two rows, don't recompute them.
- One paragraph: did convergence actually happen (stability criterion fired
  before the epoch ceiling)? Does the converged (or longer-trained) result
  clear the predeclared AMI > 0.177 bar this time? State plainly whether v2
  changes the v1 conclusion or confirms it holds even with proper
  convergence control.

Print clear timestamped progress logs matching the existing scripts'
`log()` format throughout (pretrain, DEC training with fraction_changed
progress, final evaluation).
