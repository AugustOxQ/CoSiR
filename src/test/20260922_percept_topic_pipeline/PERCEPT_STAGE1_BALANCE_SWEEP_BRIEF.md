# Brief: PercepT Stage 1 — LAMBDA_BALANCE sweep

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_balance_sweep_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: why this sweep, and the scale problem it must fix

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_pilot.py`,
`run_percept_stage1_stabilized_pilot.py`, and
`percept_stage1_stabilized_pilot_report.md` in full first.

The stabilized pilot added a balanced-assignment penalty (`LAMBDA_BALANCE =
1.0`) to fight DEC's collapse, but it still collapsed on both splits: 64/67
surviving topics ended up below 1% of nodes, with one cluster alone reaching
~38% of all 61,402 train paintings. Its own logged trajectory shows why: the
balance loss stayed in the 0.0001–0.0016 range for the entire 326-epoch run
while the KL clustering term grew from 0.007 to 0.093 — roughly 50-100x
larger throughout. `LAMBDA_BALANCE=1.0` was numerically negligible against
what it was supposed to counteract; this is a scale-mismatch failure, not
evidence the balancing approach is wrong (held-out emotion AMI still nearly
tripled, 0.0363 to 0.0925, versus the unregularized base pilot — a real,
directionally correct effect, just far too weak).

This sweep tests `LAMBDA_BALANCE ∈ {10, 50, 100, 500}` — chosen to bracket
the roughly 50-100x gap actually observed — to find whether some value
makes the balance term genuinely competitive with KL without erasing all
cluster structure (a very large lambda could instead force assignments back
toward uniform noise, destroying whatever real signal DEC was finding; watch
for and explicitly discuss this failure mode too, not just the collapse
failure mode).

## Efficiency: do the lambda-independent work exactly once

Extracting the GoEmotions-RoBERTa embeddings (train + held-out, ~3.5 minutes
total) and pretraining the autoencoder (100 epochs, lambda-independent — the
pretrain loss does not involve `LAMBDA_BALANCE` at all) do not depend on
`LAMBDA_BALANCE`. Do this work **once**, then reuse it for all 4 sweep
points:

1. Load train/held-out CLIP features, extract affect embeddings, build fused
   `h` for both splits — reuse `run_percept_stage1_pilot.py`'s
   `extract_affect_embedding_nodes()` and `fused_embeddings()` exactly
   (import as a sibling module, same `load_sibling_module()` pattern already
   used by `run_percept_stage1_stabilized_pilot.py`).
2. Build and pretrain ONE autoencoder (reuse `build_autoencoder()` and
   `pretrain_autoencoder()` exactly, same 100 epochs, same hyperparameters).
   Save its trained `state_dict()` for both encoder and decoder immediately
   after pretraining completes.

## For each `LAMBDA_BALANCE` value, run an independent, isolated DEC phase

**Critical correctness requirement:** each sweep point must start DEC
training from the SAME pretrained representation, not from whatever the
previous lambda's DEC phase left behind. For each value in `{10, 50, 100,
500}`:

1. Build a fresh encoder/decoder pair with `build_autoencoder()` and load
   the saved pretrained `state_dict()` into them (`copy.deepcopy` the
   state dict or reconstruct fresh modules and `load_state_dict()` — do not
   reuse the same module objects across sweep iterations, and do not let one
   lambda's DEC-trained weights leak into the next lambda's starting point).
2. Re-run `initialize_cluster_centers()` (same `SEED=42`) on this freshly-
   loaded pretrained encoder's latent — this will reproduce the exact same
   100 initial K-means centers every time, which is the correct controlled
   design (only `LAMBDA_BALANCE` should differ across sweep points).
3. Run DEC to convergence. Take `run_percept_stage1_stabilized_pilot.py`'s
   `train_dec_until_stable()` (Student's-t assignment, self-sharpening
   target, per-checkpoint cluster-size diagnostics including the epoch-0
   K-means-init snapshot) and modify it to accept `lambda_balance` as a
   function parameter instead of reading the module-level
   `LAMBDA_BALANCE` constant — this sweep calls it once per grid value, so
   a parameter is clearer than re-importing/monkey-patching the module 4
   times. Everything else about that training loop (convergence criterion,
   logging cadence, loss formula) stays identical.
4. Prune to 67 surviving topics (`prune_centers()`, unchanged) and evaluate
   train + held-out (`evaluate_assignments()`, `verdict()`, unchanged, same
   predeclared Pareto bar: emotion AMI > 0.1236 AND genre AMI > 0.1954).

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_balance_sweep_pilot_report.md`
with:

- A single results table across all 4 lambda values x 2 splits: lambda,
  split, emotion AMI, genre AMI, collapse verdict, final max cluster size
  (out of the 67 surviving, pre-collapse-check), fraction of clusters below
  1%. This is the primary artifact — it must make the lambda-vs-collapse
  relationship visible at a glance.
- For each lambda, the final-epoch cluster-size diagnostic (min/max/median/
  below-1%-count) so the reader can see whether collapse got better, stayed
  the same, or flipped into the opposite "flattened to uniform noise"
  failure mode at the high end of the sweep. Full per-epoch trajectories for
  all 4 runs would be too long for one report — final-state summary per
  lambda is sufficient, but explicitly note if any lambda's TRAJECTORY (not
  just final state) is qualitatively different from the others (e.g.
  collapses fast then recovers, vs. never collapses, vs. collapses and
  stays collapsed) — skim each run's own logged trajectory before writing
  this, don't infer it from final numbers alone.
- An explicit statement of which lambda values (if any) escape collapse
  under the predeclared rule, and separately, which (if any) additionally
  clear the held-out Pareto bar. Use this investigation's established
  verdict language exactly ("Collapsed" / "Merely a compromise" / "Real
  success").
- A short closing recommendation: if a lambda value exists that is both
  non-collapsed AND clears the held-out bar, say so plainly as the new
  standing PercepT-pipeline result. If none do, say so plainly too, and
  state whether the trend (does higher lambda monotonically reduce
  collapse, or is there a non-monotonic sweet spot, or does nothing work)
  suggests a further-refined lambda value would be worth trying, or whether
  this approach appears to have hit a wall.
