# Brief: PercepT Stage 2 — seed-stress the sweep's winning configuration

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage2_best_config_stress_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage2_sweep_pilot.py`
and `percept_stage2_sweep_pilot_report.md` in full first (both in this
directory).

That sweep's single best observed configuration is `q > 1.2/40` target
threshold, `lr=3e-3`, 100 epochs, mapper-init seed 42, held-out macro AUC
0.8256 (versus 0.5704 at the original `q > 2.0/40`, `lr=1e-3` config, and
0.5000 baseline). This is a large jump, and the raw training log shows it
is genuine convergence within the fixed epoch budget, not instability
(lr=3e-4 is still steeply underfit at epoch 100; lr=3e-3 converges smoothly
and monotonically to a materially lower loss with no oscillation) — but it
has only been tested at ONE mapper-init seed. Every other result in this
investigation that mattered got a 4-seed stress test before being trusted;
this one has not yet.

## What to do

Reuse `run_percept_stage2_sweep_pilot.py`'s exact structure: one shared
Stage-1 K=60/40 seed-42 re-fit (with the same reproduction check against
0.1238/0.2617, same tolerance, same stop-if-it-fails behavior), one set of
`q > 1.2/40` multi-label targets derived from that frozen fit, and the same
cached patch-feature loading. Then train and evaluate the
`AttentionPoolingMapper` at `lr=3e-3`, `q > 1.2/40` targets, 100 epochs,
across `SEEDS = (42, 7, 123, 2024)` — reuse
`train_and_evaluate_mapper()`/`AttentionPoolingMapper` from the sweep
script directly (import it as a sibling module, do not reimplement).

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage2_best_config_stress_pilot_report.md`
with:
- The shared Stage-1 reproduction check.
- A 4-seed results table (held-out macro AUC, min/median/max per-topic
  AUC) plus mean/min/max summary statistics, in the same format as every
  other seed-stress report in this directory.
- An explicit verdict: is the 0.8256 result (or something close to it)
  seed-robust, or was seed 42 unusually favorable? State the actual spread
  plainly — do not round a wide spread down to "still good."
- A one-paragraph closing comparison against the ORIGINAL threshold's own
  4-seed result from the sweep report (mean 0.5709, range 0.5644-0.5760):
  is the richer multi-label threshold + tuned learning rate a robust,
  reproducible improvement, or does its seed variance eat into the
  apparent gain?
