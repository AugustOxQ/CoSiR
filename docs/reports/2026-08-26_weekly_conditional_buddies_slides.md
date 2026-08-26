# Conditional Buddies — What matters after initialization?

**Framing:** This week separates a robust initializer from the choices that do—and do not—change downstream retrieval.

**Date:** 2026-08-26  
**Branch:** `experiment/condition_drift_retrieval_correlation`

---

## From “buddy-init works” to “which pieces matter?”

**Last week:** buddy-graph initialization beat the generic initializer on RedCaps, including at 300k—but neither conditioned approach beat raw CLIP retrieval.

**This week:** ablated the graph source and edge weighting; explained RedCaps signal scaling; and separated the consequences of condition-table optimization for held-out retrieval versus assigned training rows.

| New result | Bottom line |
|---|---|
| **Experiment 8** | Source encoder pair matters for i2t, not t2i |
| **Experiment 9** | RedCaps signal is stable; apparent size effect is lift normalization |
| **Experiment 10** | Typed-edge correction changes init substantially, not retrieval |
| **Experiments 11.1–11.2** | Frozen wins held-out oracle i2t; trained table helps its own in-sample task |

All retrieval reads use paired-within-seed deltas and `mean/SEM`; the measured noise floor is ~0.1–0.7 R1.

---

## Experiment 8: the graph’s source pair matters—selectively

**Test:** all 16 cached vision × text encoder pairs as the buddy graph/init source; fixed frozen CLIP backbone and settings; 3 seeds each (48 runs). Deltas are versus `clip_img:clip_txt`.

`test_oracle/t2i_R1` is essentially invariant: among 15 non-baseline pairs, only `dinov2:minilm` clears `|mean/SEM| ≥ 2` (**−0.07**, mean/SEM **−2.0**)—still below the ~0.1–0.7 R1 noise floor.

`test_oracle/i2t_R1` differs materially:

| Encoder pair | Mean Δ vs. CLIP source | mean/SEM | Wins/3 |
|---|---:|---:|---:|
| `dinov2:minilm` | +2.30 | +6.5 | 3 |
| `dinov2:bge` | +1.97 | +4.5 | 3 |
| `vit_sup:clip_txt` | +1.77 | +4.0 | 3 |
| `clip_img:bge` | +1.43 | +6.6 | 3 |
| `siglip_v:e5` | −0.37 | −0.8 | 1 |
| `vit_sup:e5` | −0.80 | −6.9 | 0 |

**Verdict:** 12/15 non-baseline pairs are positive and significant on i2t (all 3/3 wins; 10 above noise floor); CLIP source is not the strongest i2t choice in this 150k grid. Survival rate is not a usefulness proxy: Pearson `r = +0.010` (t2i), `r = −0.410` (i2t), `n = 15`.

---

## Experiment 9: the subreddit signal is stable; the size story is mechanical

**Test:** decompose same-subreddit buddy-edge lift by subreddit; relate it to size, caption diversity, and visual homogeneity; independently sample all 350 subreddits at 150k, 300k, and 500k.

At 150k: aggregate lift **22.80×** across **159 of 350** qualifying subreddits. Individual lift: **4.41×** (`pics`) to **670.79×** (`f1porn`); median **83.98×**, mean **114.10×**.

| Property vs. lift, 150k | Pearson r | Spearman ρ | Read |
|---|---:|---:|---|
| Size | −0.328 | −0.523 | real curved/rank relationship |
| Caption diversity | −0.219 | −0.224 | null |
| Visual homogeneity | +0.204 | +0.230 | null |

| Metric | 150k | 300k | 500k |
|---|---:|---:|---:|
| Overall lift | 22.80× | 22.74× | 22.79× |
| Lift-qualifying subreddits | 159 (45%) | 197 (56%) | 214 (61%) |
| z-qualifying subreddits | 124 (35%) | 157 (45%) | 185 (53%) |
| Lift–size Spearman ρ | −0.523 | −0.532 | −0.590 |
| z–size Spearman ρ | +0.221 | +0.163 | +0.226 |

**Verdict:** C1’s aggregate signal is stable. Large subreddits are not shown to be topically diluted: purity rises slightly with size, while lift is pulled down by its degree-dependent denominator (`Spearman(deg_s, size) = +0.973`). Positive z–size is the expected confidence effect.

---

## Experiment 10: fixing edge typing moves the initializer, not retrieval

**Test:** `distance_mode="typed"` uses the supporting modality’s rank for image-only/text-only union-graph edges (~98% of edges); prior `blend` uses both modalities for every edge. 2 modes × 3 seeds.

| Paired metric, typed − blend | Mean Δ | ± std | mean/SEM | Typed wins |
|---|---:|---:|---:|---:|
| `test_oracle/t2i_R1` | +0.00 | 0.26 | +0.0 | 2/3 |
| `test_oracle/i2t_R1` | −0.37 | 0.59 | −1.1 | 1/3 |
| `test_pre_diff/t2i_R1` | +0.00 | 0.10 | +0.0 | 1/3 |
| `test_pre_diff/i2t_R1` | +0.17 | 0.55 | +0.5 | 2/3 |

`test_raw` was identical in all six runs: `t2i_R1 = 28.1`, `i2t_R1 = 29.7`.

This is a real initialization change: mean `|typed − blend| = 0.5117`, **1.02×** the mean absolute embedding value (0.5000); per-dimension correlation ranges **+0.985** to **−0.996**, mean **+0.240**.

**Verdict:** typed edges correct a genuine graph-level flaw, but retrieval and the gap to CLIP are unchanged within significance/noise criteria at 150k. Buddy-derived structure appears more load-bearing than this fine edge-weighting choice—a citable null.

---

## Experiment 11.1: freeze the table to win held-out oracle i2t

**Test:** train post-init condition table (`em_interval=-1`) versus fully freeze it (`em_interval=101`, beyond 100 epochs); buddy init, backbone, and loss stack fixed; 2 arms × 3 seeds.

| Metric, frozen − trained | Mean Δ | mean/SEM | Per-seed Δ | Read |
|---|---:|---:|---|---|
| `test_oracle/t2i_R1` | −0.27 | −2.0 | −0.40 / −0.40 / +0.00 | flagged, but inside noise floor |
| `test_oracle/i2t_R1` | +4.67 | +32.1 | +4.90 / +4.40 / +4.70 | decisive frozen win |

Sanity check: frozen drift from init = exactly **0** for all seeds; trained drift mean **0.0845**, range **[0.0834, 0.0855]**.

| Final-epoch geometry, trained − frozen | Mean Δ | mean/SEM |
|---|---:|---:|
| `shift_mean` | +0.0081 | +13.9 |
| `shift_std` | +0.0079 | +31.9 |
| Row diversity | +0.0141 | +15.9 |
| Column diversity | −0.0127 | −10.2 |
| Conditioned effective dimensions | 0 | exactly 301 in all six runs |

**Verdict:** training is not inert—it differentiates conditions and shifts embeddings further—but freezing decisively wins the held-out oracle i2t metric. The t2i difference is a noise-floor null.

---

## Experiment 11.2: drift has a large, graded rank effect when isolated

**Test:** reuse 11.1 checkpoints. For each seed: 3,000 in-sample queries, own assigned condition, full 150,000-item training gallery. The key counterfactual holds trained combiner, `other_proj`, and gallery fixed, replacing only trained conditions with buddy init.

| Spearman ρ; n = 3000 per seed | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| `rho(delta_rank, drift)` | +0.026 | +0.017 | +0.009 |
| `rho(|delta_rank|, drift)` | +0.160 | +0.141 | +0.156 |
| `rho(|delta_rank|, shift)` | −0.288 | −0.296 | −0.302 |
| `rho(delta_rank_swap, drift)` | +0.466 | +0.466 | +0.477 |

For the condition-only counterfactual, every `rho(delta_rank_swap, drift)` has **p < 1e−160**. Mean `delta_rank_swap`: **+343.5 / +352.0 / +357.1**; medians: **+6 / +7 / +7**.

**Verdict:** reverting high-drift conditions costs rank inside the fixed trained model. The naive signed cross-arm correlation is near zero because independently trained combiner/`other_proj` divergence dominates it (standard deviation **1055–1343** ranks); drift predicts magnitude, not direction.

---

## Trained wins its own in-sample, own-condition task

The same 11.2 diagnostic gives the trained arm the better assigned-training-row readout:

| Seed | Mean rank frozen / trained | R1 frozen / trained | Fraction improved |
|---:|---|---|---:|
| 1 | 1118.1 / 975.0 | 0.0923 / 0.1007 | 0.636 |
| 2 | 1122.8 / 980.4 | 0.0880 / 0.0997 | 0.640 |
| 3 | 1112.5 / 996.2 | 0.0893 / 0.1033 | 0.639 |

**Important scope boundary:** this is not a held-out result and not oracle-max-over-conditions retrieval. It shows trained conditions are useful for their assigned training rows—not that the learned table generalizes better as a held-out codebook.

---

## The apparent conflict disappears once we keep the constructs separate

| Experiment 11.1 | Experiment 11.2 |
|---|---|
| **Held-out** queries | **In-sample** training rows |
| **Oracle max over conditions** | **Own assigned condition** |
| Frozen beats trained on i2t: **+4.67 R1** | Trained conditions have drift-graded benefit: `rho = +0.466 / +0.466 / +0.477` |

**Simple reading:** training learns useful sample-specific co-adaptation for known rows, but the resulting table/codebook transfers worse to unseen queries when they may select their best table entry.

These are not competing estimates of one quantity. The held-out-vs-in-sample and oracle-vs-own-condition mismatches constrain the interpretation: 11.2 does not explain away 11.1’s frozen-wins result.

---

## Experiment 11.3: coupling test is implemented; results are pending

**Question:** should predictor-distillation remain one-way, or should it also pull the trainable table toward what `condition_predictor` can represent?

`loss.pred_stopgrad` defaults to `True`: table detached; gradients flow table → predictor. With `loss.pred_stopgrad=false`, the loss also updates the trainable table. The `pred_coupled` arm will run three RedCaps-150k seeds against 11.1’s trained/frozen arms, sharing the buddy-init template and wandb group.

| Implementation boundary | Evidence |
|---|---|
| Toggle | `038f956` |
| Sweep runner | `be0efb7` — one file, 85 insertions |
| Paired analysis | `c821513` |
| Results commit | None in requested `git log --oneline -10` |

**Status:** results are pending in a separate parallel run right now. No outcome should be inferred from 11.1/11.2. The future readout will report paired `pred_coupled − trained` and `pred_coupled − frozen` deltas for `test_oracle` and `test_pre_diff`, both directions, plus drift and predictor loss.

---

## Next: close the coupling test, then target the remaining gates

1. Complete and analyze Experiment 11.3’s three `pred_coupled` runs—report `test_oracle`, `test_pre_diff`, drift, and predictor loss without extrapolation from implementation commits.
2. Test Experiment 8’s strongest i2t source pairs on `test_pre_diff`; consider the 300k extension only if held-out feature re-extraction is justified.
3. Extend Experiment 9 coverage to 1M/3.1M; fit a joint model if explanation, rather than coverage, is the goal.
4. Treat Experiment 10 as a completed 150k null unless a scale or operating-point rationale emerges.
5. Keep Experiment 11 claims separated by construct: frozen tables are the held-out-oracle i2t winner here; a held-out-reachable predictor-based diagnostic is required to bridge the gap.

---

## Questions

**Current publication-safe claim:** buddy-graph structure is a robust, content-grounded initializer and a better in-model starting point than the generic alternative.

**This week’s sharper question:** how do we retain useful trained-row adaptation without sacrificing held-out codebook behavior?
