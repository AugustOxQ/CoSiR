# Brief: candidate 6 — upweight the affect/emotion signal in buddy's graph construction

## Context

Read `deep_stage_analysis_report.md` (this directory) §"Finding D" in full
before starting. Summary: buddy's Stage-1 topics are overwhelmingly
"contentment"-majority (14/19) with uniformly high emotion entropy, while
PercepT's topics show much more emotion-majority variety. The working
diagnosis: buddy's Attention-h1 student is trained with two InfoNCE losses
combined as a flat, equal-weight sum,
`total_loss = content_loss + affect_loss`
(see `run_attention_h1_noise_schedule_pilot.py`, function `run_seed`, and
its import `run_learned_student_arch_sweep_pilot.py` for
`symmetric_infonce`/`sample_positive_pairs`). Content structure appears to
dominate the resulting embedding; emotion "rides along" only weakly. This
candidate tests whether upweighting the affect loss term produces a real,
Pareto-bar-clearing gain in held-out emotion AMI without destroying genre
AMI — this is real Stage-1 InfoNCE retraining, not a frozen-embedding
post-hoc tweak (unlike candidates 1-5 in this directory).

This is explicitly the lowest-priority, most speculative candidate from the
deep analysis's updated list (§5, item 6) — "a follow-up if 1-5 don't close
the gap." Candidates 1-4 (occupancy handling, mapper tuning, richer
multi-label targets) already delivered large, validated Stage 2 gains
(0.5978 → 0.8534 macro AUC); candidate 5 (K sweep) was a clean negative.
This candidate is being run anyway per explicit instruction to continue
through the full list, but treat a negative or marginal result here as an
entirely acceptable, expected outcome — do not force a positive framing.

## Established baseline to build on

`run_attention_h1_noise_schedule_pilot.py`'s seed-42 winner
(`NOISE_STDS = (0.0, 0.02, 0.05, 0.1)`, winner `noise_std=0.0` — i.e. no
embedding noise, just the cosine LR schedule) is the standing best
Attention-h1 config referenced throughout this investigation as "Noise +
schedule alone (winning seed 42)":

| | held-out emotion AMI | held-out genre AMI | held-out silhouette |
|---|---:|---:|---:|
| Baseline (noise_std=0, LAMBDA_AFFECT=1.0, i.e. today's flat sum) | 0.1306 | 0.1973 | 0.0488 |

Held-out Pareto bar (must clear both, same convention used everywhere in
this investigation): emotion AMI > 0.1236 AND genre AMI > 0.1954.

## What to build

New script `src/test/20260927_deep_stage_analysis/run_candidate6_affect_upweight_pilot.py`.
Read `run_attention_h1_noise_schedule_pilot.py` in full first, and reuse its
`main()` setup flow (loading `run_learned_student_arch_sweep_pilot.py` as
`arch`, building `content_train`/`affect_train`/`content_edges`/
`affect_edges`/teacher graphs/`context` dict, exactly as that file's
`main()` does) by import or by copying that setup block verbatim — do not
reinvent data loading. Fix `noise_std=0.0` throughout (the established
winner); the only new variable is a `LAMBDA_AFFECT` multiplier applied to
the affect loss term:

```python
total_loss = content_loss + LAMBDA_AFFECT * affect_loss
```

(currently hardcoded as `content_loss + affect_loss`, i.e. `LAMBDA_AFFECT=1.0`,
inside `run_seed` in `run_attention_h1_noise_schedule_pilot.py` — copy
`run_seed` into the new script with this one line parameterized, everything
else identical: same optimizer, same `CosineAnnealingLR(T_max=arch.MAX_EPOCHS,
eta_min=1e-5)` starting at `LR_START=1e-3`, same plateau-based early stopping,
same post-hoc Leiden clustering + AMI + silhouette evaluation on the clean
final embedding).

### Screen (seed 42)

Sweep `LAMBDA_AFFECT` in `{1.0 (baseline, sanity check), 2.0, 4.0, 8.0}`.
For each, report held-out post-Leiden emotion AMI, genre AMI, silhouette,
and whether the Pareto bar clears. Sanity-check the `LAMBDA_AFFECT=1.0`
point reproduces 0.1306/0.1973/0.0488 within a small tolerance (this
validates the copied training loop is faithful) before trusting the other
sweep points — if it doesn't reproduce, stop and report the discrepancy
rather than continuing.

### Winner selection and stress

Practical margin: +0.005 absolute emotion AMI over the 0.1306 baseline,
same convention as every other candidate in this directory. A sweep point
only qualifies as a winner if it (a) beats this margin on emotion AMI AND
(b) still clears both Pareto bars (i.e. genre AMI does not collapse below
0.1954 as a side effect of the reweighting — this is the main risk and
should be reported plainly even if it happens). Among qualifying points,
pick the one with the highest emotion AMI. If no point qualifies, report
the full screen table, state the negative result plainly (matching this
directory's established honest-negative-result convention, e.g.
`candidate3_k_sweep_pilot_report.md`), and stop — do not stress-test a
non-qualifying point.

If a point qualifies, 4-seed stress it (seeds 42, 7, 123, 2024 — this
directory's standing convention) and report mean/min/max/std for emotion
AMI, genre AMI, and silhouette, plus Pareto-bar clear count out of 4.

### Optional Stage 2 follow-through

Only if a winner is validated across all 4 stress seeds: note in the
report that this would require re-running Stage 2 (Leiden re-clustering +
candidate-1/2/4 pipeline) on the new embedding to check whether it changes
the 0.8534 headline number, but do NOT implement that re-run yourself —
flag it as a follow-up decision point instead. This candidate's own scope
is Stage 1 (embedding/AMI/silhouette) only.

## Compute

Each seed run trains Attention-h1 to convergence (plateau-based early
stop, up to `MAX_EPOCHS=200`, full-batch). Comparable pilots in this
directory (e.g. the noise/schedule screen: 4 screen points + 3 stress
seeds = 7 runs) complete in well under 10 minutes total on a single GPU —
expect similar here (4 screen + up to 4 stress = 8 runs). Use the local
GPU (`torch.cuda.is_available()`); DAS6 is authorized if you judge it
useful but is very unlikely to be necessary at this scale — say explicitly
in your final summary whether you used it and why.

## Output

Write `candidate6_affect_upweight_pilot_report.md` in this directory,
following the Markdown table conventions of the other candidate reports in
this directory (see `candidate3_k_sweep_pilot_report.md` for a template of
a clean negative-result writeup, `candidate1_variant_c_stress_report.md`
for a template of a validated positive one). End with a one-paragraph
plain-English verdict: does upweighting affect materially help, hurt, or
have no effect on buddy's Stage-1 emotion signal, and what does that imply
about Finding D's diagnosis (is the flat equal-weight sum actually the
bottleneck, or is something else going on)?
