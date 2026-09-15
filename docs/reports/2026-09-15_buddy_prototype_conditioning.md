# Buddy-Graph Prototype Conditioning — Experiment 18

**Date:** 2026-09-15
**Spec:** `docs/superpowers/specs/2026-09-15-buddy-prototype-conditioning-design.md`
**Plan:** `docs/superpowers/plans/2026-09-15-buddy-prototype-conditioning.md`, Task 9
**Runs:** 2 arms (`baseline`=`free_vector`+`lowrank` combiner, `prototype_pooled`=16-prototype
attention-pooled bank + `lowrank` combiner) × 3 seeds × 100 epochs, RedCaps-150k,
DAS6 (`node411`/`node412`), `scripts/run_exp18_150k_sweep.sh`.
**Eval scripts (this report):** `src/test/20260915_exp18_eval/{recompute_oracle_retrieval,prototype_probe,prototype_coherence}.py`

## Verdict

**Prototype-pooled conditioning does not beat the architecture-fix-only baseline.**
Retrieval is a near-null (t2i, `mean/SEM`=1.73, below the significance bar) to
marginal (i2t, +0.6 R1 — at the very edge of this project's own ~0.1–0.7 R1 noise
floor) effect. More importantly, **the interpretability goal this design exists
for fails outright**: the trained condition space collapses to **1 effective
PCA dimension in all 3 seeds**, per-sample condition vectors are far more
mutually similar than the baseline's (mean pairwise cosine 0.16–0.61 vs.
≈0.000), the control-task-gated probe selectivity is near-null and seed-unstable
(warmth 0.010–0.263 vs. free-vector's stable 0.245–0.247; register ≈0 in all 3
seeds, vs. free-vector's stable 0.300–0.307), and the silhouette score on
argmax-prototype clusters is inconsistent across seeds (0.12, 0.31, −0.13) —
nowhere near PercepT's reported 0.97. **Neither on retrieval nor on
interpretability does this redesign clear the bar; on interpretability, its
own stated goal, it is a clear regression against the free-vector baseline.**

A second, load-bearing finding: **the one cluster-health metric this design added
(`prototype_usage_entropy`) does not detect this collapse.** Entropy ratio at
the real, final (100-epoch) checkpoints is 0.92–0.93 (vs. the smoke test's
2-epoch 0.9999) — a plausible-looking "mostly healthy, not collapsed" number —
while the argmax attention winner is the same single prototype for 88–99.5% of
all 150,000 samples in every seed, and the actual output vectors have collapsed
to ~1 effective dimension. Soft-attention entropy and functional/representational
diversity are not the same thing here; this gap is itself worth carrying forward.

## Retrieval (Task 9 Step 2 — recomputed against the oracle-eval bug fix)

**Bug and fix, briefly (full detail in `.superpowers/sdd/2026-09-15-buddy-prototype-conditioning/progress.md`):**
the live training run's own `test_oracle/*` numbers for `prototype_pooled` are
invalid — `_eval_snapshot` (`train_cosir.py:~986`) FPS-samples representatives
from `embedding_manager.embeddings`, which is buddy-graph-initialized once at
epoch 0 and never updated again in this mode (`batch_indices=None`,
`train_cosir.py:1650`). This report's retrieval numbers are **recomputed from
each run's saved checkpoint** instead: `prototype_pooled`'s representatives are
`model.prototype_bank.values` (the actual 16 trained prototypes) directly;
`baseline`'s representatives use the same FPS-over-`embedding_manager.embeddings`
method training's own in-loop eval uses (valid there — that table *is* updated
every step in `free_vector` mode), at `k=30` matching the final-epoch convention.
Both use each run's own cached `test_backbone_embeddings.pt` (frozen CLIP
features, byte-identical baseline-vs-prototype_pooled, confirmed: `raw/*` recall
is identical across all 6 runs), so no CLIP re-encoding or raw test images were
needed. Full per-run numbers: `src/test/20260915_exp18_eval/oracle_retrieval_results.json`.

| metric | baseline (free_vector) | prototype_pooled | Δ mean ± std | mean/SEM | vs. noise floor (~0.1–0.7 R1) |
|---|---|---|---|---|---|
| `oracle/t2i_R1` | 16.27 ± 0.06 | 16.37 ± 0.06 | +0.10 ± 0.10 | 1.73 | inside — null |
| `oracle/i2t_R1` | 16.20 ± 0.00 | 16.80 ± 0.10 | **+0.60 ± 0.10** | **10.39** | at the upper edge — flagged, not clearly beyond it |
| `oracle/t2i_R5` | 30.80 ± 0.00 | 30.93 ± 0.06 | +0.13 ± 0.06 | 4.00 | inside — marginal |
| `oracle/i2t_R5` | 29.50 ± 0.00 | 30.57 ± 0.32 | +1.07 ± 0.32 | 5.75 | beyond — real but small |
| `oracle/t2i_R10` | 37.70 ± 0.00 | 37.77 ± 0.06 | +0.07 ± 0.06 | 2.00 | inside — null |
| `oracle/i2t_R10` | 35.80 ± 0.00 | 37.20 ± 0.44 | +1.40 ± 0.44 | 5.56 | beyond — real but small |

Reading this the way this project's own convention (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md`
§5/§38) reads C7/C8's flagged-but-marginal cells: `mean/SEM ≥ 2` is necessary but
not sufficient — a tiny cross-seed variance can inflate `mean/SEM` for a magnitude
that is still inside the established noise floor. `i2t_R1`'s +0.6 sits exactly at
the noise floor's own upper bound, so it is reported as **directionally positive,
not a clean win**. `t2i` is flat. `i2t_R5`/`R10` clear the floor more clearly
(+1.07, +1.40) and are the closest thing to a real signal in this experiment —
still small in absolute terms, and this is the one axis prior C6/C9 results also
flagged as the more volatile direction (`combine_side="img"` here, matching
`prototype_pooled`'s own `combine_side` print — see raw JSON).

Both arms' oracle recall stays **below raw frozen-CLIP retrieval** on this test
set (`raw/t2i_R1=16.3`, `raw/i2t_R1=17.8` — identical across all 6 runs, a good
cross-run sanity check), consistent with C6's established RedCaps-150k/300k
finding that neither training arm beats the untouched CLIP backbone at this
scale — expected, not new.

## Interpretability probe (Task 9 Step 3 — reusing 17.1's harness)

`model.prototype_bank(query_features)` evaluated fresh over the full 150k
cached CLIP features per seed (not `final_embeddings/embeddings.npy`, which is
subject to the same staleness issue as the oracle-eval bug), fed through
17.1's unmodified `probe_selectivity` (Hewitt & Liang selectivity, 20-shuffle
control, `z ≥ 2 AND selectivity ≥ 0.10` verdict rule). Full output:
`src/test/20260915_exp18_eval/prototype_probe_results.json`.

| axis | raw-512D (17.1) | PCA-16 (17.1) | RandProj-16 (17.1) | free_vector checkpoint range (17.1, n=15) | **prototype_pooled (n=3 seeds)** |
|---|---|---|---|---|---|
| warmth | 0.2656 | 0.2625 | 0.1919 | 0.2445 – 0.2466 | **0.0101, 0.0138, 0.2629** — seed1 "positive" (barely), seed2 "null", seed3 "positive" |
| register | 0.3394 | 0.3125 | 0.1961 | 0.3004 – 0.3072 | **≈0.0000 in all 3 seeds — "null"** |

`prototype_pooled` is **not seed-replicated** on either axis: `register` is
exactly null (selectivity indistinguishable from the shuffled control, to
machine precision) in all 3 seeds — the proxy axis this design was meant to
make more accessible is not decodable from the learned condition space at all.
`warmth` ranges from null (seed2, 0.014) to roughly matching the free-vector
baseline (seed3, 0.263) — a 20× spread across seeds with identical
hyperparameters, itself evidence of an unstable, not-robustly-learned
representation, not just "sometimes it works." Compare to the free-vector
checkpoints, which land in a tight 0.2445–0.2466 band across 15 checkpoints
(§ of the 17.1 report) — this design is both weaker on average and far less
consistent.

## Prototype coherence / collapse diagnostic (Task 9 Step 4 + follow-up)

Reuses `condition_space_evaluator.py`'s `sklearn.metrics.silhouette_score`
machinery, with each sample's argmax-attention prototype id in place of the
HDBSCAN labels it was built around, per the plan. The near-null probe result
above prompted computing the same PCA-effective-dims / near-origin-ratio /
pairwise-diversity fields that diagnostic also reports, plus the real,
final-checkpoint `usage_entropy` (the smoke test's own entropy check only ever
covered 2 epochs / 1000 samples). Full output:
`src/test/20260915_exp18_eval/prototype_coherence_results.json`.

| | seed1 | seed2 | seed3 |
|---|---|---|---|
| `usage_entropy` (real, 100 ep, full 150k) | 2.554 | 2.577 | 2.590 |
| entropy ratio (vs. log 16 = 2.773) | 0.921 | 0.929 | 0.934 |
| argmax winner's share of all 150,000 samples | 90.7% (proto 14) | 99.5% (proto 15) | 87.9% (proto 5) |
| distinct prototypes ever chosen as argmax | 14 / 16 | 16 / 16 | 14 / 16 |
| silhouette (argmax labels, 20k subsample) | 0.122 | 0.313 | **−0.128** |
| PCA effective dims (95% var, 20k subsample) | **1** | **1** | **1** |
| near-origin ratio (‖v‖ < 0.5) | 100% | 100% | 100% |
| mean pairwise cosine sim (2k subsample) | 0.189 | 0.156 | 0.605 |
| — same metric, baseline free_vector | −0.0001 | 0.0003 | 0.0015 |

Every seed independently collapses to **1 effective PCA dimension** out of 16
— the condition space the model actually learned is, for practical purposes,
a single scalar axis, not a 16-D or even multi-D structure. The silhouette
score is not just weak but sign-inconsistent (seed3 is negative — argmax
clusters fit worse than random for that seed), far from PercepT's reported
0.97 (vs. 0.37 baseline) this design cited as the literature-aligned target.
Per-sample condition vectors are 100–1000× more mutually similar than the
free-vector baseline's (mean pairwise cosine 0.16–0.61 vs. ≈0.0000–0.0015) —
this is the same failure mode probe selectivity is catching: near-uniform,
high-entropy soft attention over a *shared, small* prototype set produces a
near-constant `attn @ values` output for nearly every sample, almost
independent of that sample's actual content.

**This is why the entropy metric looked fine while the space had collapsed.**
`usage_entropy` measures the softness of the per-sample attention distribution
(high here — 0.92–0.93× uniform, correctly reporting "not concentrated"), but
says nothing about whether the *argmax* decision or the *resulting output
vector* varies across samples. Here attention is soft-but-tilted: nearly every
sample's top logit lands on the same one or two prototypes (88–99.5% argmax
share), and because the softmax stays close to uniform (high entropy), the
actual weighted-sum output barely differs from the same near-uniform average
of all 16 prototype values, for every sample — softness and collapse turn out
not to be opposites here. A future revision to this design's own monitoring
should track the argmax-share/PCA-effective-dims pair this report computed,
not `usage_entropy` alone.

## Was prototype collapse observed in the real 150k/100-epoch runs?

Yes — see the table above. The 2-epoch smoke test's entropy check
(`entropy_ratio=0.99997`, near-perfectly uniform) was not representative of
the fully-trained state; entropy did drop over 100 epochs (0.92–0.93), a
directionally sensible trend, but the underlying representation collapsed to
~1 effective dimension regardless — a failure mode the smoke test's only
metric could not have caught even in principle, since it doesn't measure
argmax concentration or output-space dimensionality.

## Plain verdict

**Neither retrieval nor interpretability, the two success criteria this
design set out to clear (spec §7, plan Task 9 self-review), is met.**
Retrieval is a near-null-to-marginal, noise-floor-adjacent effect (real only
on i2t R5/R10, and small there). Interpretability — the actual point of
building a prototype bank instead of continuing with free per-sample vectors
— is a clear regression: near-null, seed-unstable probe selectivity and a
condition space that has collapsed to 1 effective dimension in every seed,
worse than PercepT's cited reference and worse than this project's own
free-vector baseline on the same axes. The architecture is real and trains
stably (no NaNs, no crashes, checkpoints load and forward-pass correctly, loss
curves converge normally per the sweep logs) but at `num_prototypes=16` with
the current temperature/query-projection setup, it does not deliver the
structured, interpretable condition space it was designed to produce.

## Caveats

- **Single configuration.** This is one point in the design space
  (`num_prototypes=16`, learned temperature initialized at 1.0, `lowrank`
  combiner, `combine_side` inherited per-arm from the sweep script). The
  collapse mechanism identified here (near-uniform-but-tilted attention →
  near-constant weighted output) is plausibly sensitive to temperature and to
  `num_prototypes`; this report does not test whether a sharper initial
  temperature or a smaller/larger prototype count avoids it. That would be
  the natural next step if this design is revisited, not attempted here.
- **`n=3` seeds** for `prototype_pooled`'s probe/coherence numbers (vs. 17.1's
  `n=15`, with its own caveat that several of those 15 are near-duplicate
  runs). The seed-to-seed instability reported here (warmth 0.010–0.263,
  silhouette 0.12 to −0.13) is itself the finding, not noise to average away,
  but a larger seed count would sharpen it.
- **`i2t_R1`'s "at the noise floor's upper edge" reading is a judgment call**,
  not a bright line — the ~0.1–0.7 R1 floor itself comes from one duplicate-
  config reference run (per the publication-plan spec's own §5 note); a
  dedicated duplicate-config check at this exact operating point was not run
  here.
- **Predictor-based eval (`test_pre_diff`) is deliberately not reported for
  `prototype_pooled`**, per the explicit user decision recorded in this
  branch's progress ledger: `condition_predictor` is trained unconditionally
  in both arms but is architecturally redundant for `prototype_pooled` (the
  bank itself is already a differentiable, amortized function of the query —
  it doesn't need a second network distilled to approximate it), and this
  repo's own git history (`bf12fa1`, `142506e`) already shows hard/discrete
  prototype-style assignment was tried and abandoned once before. Reporting a
  predictor-vs-CLIP gap for this arm would compare a fundamentally
  ill-motivated number, not a meaningful one.
- **This report's retrieval/probe/coherence scripts run outside the training
  loop**, reconstructing only the combiner, `other_proj`, and prototype bank
  from each checkpoint's saved state dicts (not the full `CoSiRModel`/CLIP
  backbone) — verified to match `CoSiRModel.combine`/`project_other` exactly
  (`label_encoder` is `nn.Identity()` in the real model, so skipping it here
  is not an approximation), and cross-checked via the `raw/*` recall numbers
  being byte-identical across all 6 independently-cached
  `test_backbone_embeddings.pt` files.
