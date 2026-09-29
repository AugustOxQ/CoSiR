# Brief: PercepT Stage 1 — faithful recipe + modest balance term (combining two mechanisms)

Write a new script
`src/test/20260922_percept_topic_pipeline/run_percept_stage1_faithful_recipe_balance_pilot.py`.
Do NOT run it — execution happens separately, on GPU, outside this task.

## Context: two mechanisms with opposite strengths

Read `src/test/20260922_percept_topic_pipeline/run_percept_stage1_faithful_recipe_pilot.py`
(just-completed sibling pilot), its report
`percept_stage1_faithful_recipe_pilot_report.md`, and
`run_percept_stage1_balance_sweep_v2_pilot.py` in full first.

The faithful-recipe pilot (paper's latent noise injection `lambda_noise=0.1`
during pretraining + Adam 1e-3 cosine-annealed-to-1e-5 LR for both phases,
native K=100/67, no balance term) just ran both noise-persistence variants at
seed 42. Results (held-out, from the report above): Variant A (pretrain-only
noise) scored emotion AMI 0.1092 / genre AMI 0.3288; Variant B (noise
persists into DEC) scored 0.1086 / 0.3412 — the two variants are
statistically indistinguishable. Both **missed the held-out Pareto bar on
emotion only** (required > 0.1236; missed by ~0.014) while **clearing genre
by a wide margin** (required > 0.1954; cleared by ~0.13-0.15). This is the
mirror image of this project's own standing fix (`LAMBDA_BALANCE=1000`,
K=60/40), which wins emotion (0.1252) but is comparatively weaker on genre
(0.2486). The two mechanisms plausibly pull on complementary axes. This
pilot tests whether adding a small balance term on top of the faithful
recipe closes the emotion gap without giving back enough of the genre margin
to fall below its own bar.

**Fix noise handling to Variant A (pretrain-only noise) only** — the two
variants were empirically indistinguishable in the sibling pilot, and it is
also the more literal reading of the paper's equations (written under a
"Pretraining" heading). Do not re-test Variant B here; that would be a
second free dimension on top of the balance-weight screen this brief asks
for, which is out of scope.

## Implementation

Reuse `run_percept_stage1_faithful_recipe_pilot.py`'s pretraining loop
(latent noise injection, Adam 1e-3 cosine-annealed T_max=100 eta_min=1e-5),
K-means init (100 clusters), Student's-t soft assignment, self-sharpened
target, convergence criterion (`fraction_changed < 0.001`, ceiling 500), and
67-of-100 norm-threshold pruning — unchanged, via the same sibling-import
pattern. Reuse its DEC-phase Adam 1e-3 cosine-annealed (T_max=200,
eta_min=1e-5, clamped after epoch 200) schedule unchanged.

**The only change: add the balance term to the DEC joint loss.** Reuse
`run_percept_stage1_balance_sweep_v2_pilot.py`'s exact balance-loss
computation (the mean soft assignment pulled toward uniform) and combine it
with the faithful recipe's existing terms:

```python
total_loss = kl_loss + LAMBDA_RECONSTRUCTION * reconstruction_loss + LAMBDA_BALANCE * balance_loss
```

with `LAMBDA_RECONSTRUCTION = 1.0` unchanged from the faithful-recipe pilot.
Joint DEC reconstruction uses clean `z` (Variant A behavior — no noise
during the DEC phase). KL and balance loss both use clean `z`.

## Two-stage design: single-seed screen, then stress only the winner

**Stage 1 — single seed=42 screen across `LAMBDA_BALANCE` in {100, 300,
1000}.** This range deliberately brackets the prior investigation's
"too small" point (1) and its overshoot risk at very high values (5000
collapsed on train even though held-out looked fine) without repeating
either extreme; note in a comment that the DEC loss's KL magnitude at
convergence in the faithful-recipe pilot (~0.39-0.40) is a similar order of
magnitude to the original balance-sweep investigation's KL (~0.09), so the
same lambda range is a reasonable starting bracket, not an untested guess
transplanted from a differently-scaled loss. For each of the 3 values,
report held-out emotion AMI, held-out genre AMI, held-out Pareto bar
clearance, silhouette (same computation as the faithful-recipe pilot — this
is new territory this investigation has not measured before that pilot),
and the full collapse diagnostic (same format as prior pilots).

**Stage 2 — if and only if one or more of the 3 values clears the held-out
Pareto bar in Stage 1**, pick the single best-clearing value using this
investigation's established rule (prior report's language: "the point with
the largest emotion margin" when multiple clear — reuse that same
tie-break rule verbatim since emotion was this pilot's binding constraint,
not genre) and stress-test only that one value with the same 3 additional
seeds used throughout this investigation, `SEEDS = (7, 123, 2024)`, using
the exact multi-seed methodology already established (fresh
`torch.manual_seed`/CUDA seed before a fresh autoencoder build, fresh
100-epoch noisy pretrain per seed, `KMeans(random_state=seed)`, DEC with
the winning lambda_balance fixed). If none of the 3 values clears the bar in
Stage 1, skip Stage 2 entirely and report all 3 as plain misses — do not
widen the lambda range beyond {100, 300, 1000}; that is new scope outside
this brief.

## Report

Write
`src/test/20260922_percept_topic_pipeline/percept_stage1_faithful_recipe_balance_pilot_report.md`
with:
- The training trajectory (reconstruction/KL/balance/total loss, LR,
  convergence epoch and reason) and the cluster-size diagnostic trajectory,
  for all 3 screened lambda values.
- Stage 1 results table (seed 42, all 3 values, both splits): emotion AMI,
  genre AMI, silhouette, verdict, held-out Pareto bar clearance, and the
  collapse-check numbers.
- If Stage 2 ran: the 4-seed results table and summary statistics in the
  same format as `percept_stage1_seed_stress_pilot_report.md`, plus an
  explicit three-way comparison table against (a) the standing K=60/40
  `LAMBDA_BALANCE=1000` result (held-out emotion AMI mean 0.1252, genre AMI
  mean 0.2486, 4/4 seeds) and (b) the faithful-recipe-alone pilot's Variant A
  single-seed result (held-out emotion AMI 0.1092, genre AMI 0.3288,
  silhouette 0.5120) — does combining both mechanisms beat both individual
  mechanisms on the both-bar-clearing seed count, or does it just trade one
  mechanism's weakness for a smaller version of the same weakness?
- A final, honest verdict using this investigation's established language.
  If all 3 screened values miss the bar, say so plainly and do not silently
  widen the search — leave that judgment to the calling session, which has
  the full cross-pilot context this script does not.
