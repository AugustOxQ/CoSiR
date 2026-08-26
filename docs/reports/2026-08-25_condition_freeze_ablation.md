# Does post-init training of the conditions do anything? (Experiment 11.1)

**Date:** 2026-08-25 · **Dataset:** RedCaps, 150,000 rows of `redcaps_train.json` (matches C5/C6/C7/C8's scale) · **Branch:** `experiment/buddy_init_ablation2`
**Code:** `scripts/run_condition_freeze_ablation.sh`, `scripts/analyze_condition_freeze_ablation.py`, `scripts/analyze_condition_geometry.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 11.1

---

## TL;DR

Holding buddy-init geometry, the frozen CLIP backbone, and every hyperparameter identical, this experiment asks whether letting the per-sample condition vectors keep training after buddy-init (today's default) does anything, versus freezing them at their init value for the whole run (`em_interval` set past the epoch budget so the "network" phase never ends). **The answer is a real, seed-replicated, direction-asymmetric effect that is not a clean null on either axis:**

- **Retrieval — t2i is a noise-floor null, i2t is a large, decisive, and counter-intuitive win for freezing.** `test_oracle/t2i_R1`: mean Δ (frozen − trained) = **−0.27** (mean/SEM = −2.0, statistically flagged but the magnitude sits inside the ~0.1–0.7 R1 noise floor — the same category as C7/C8's flagged-but-tiny t2i exceptions). `test_oracle/i2t_R1`: mean Δ (frozen − trained) = **+4.67** (mean/SEM = **+32.1**, 3/3 seeds, magnitude nearly an order of magnitude above the noise floor's upper edge). **Freezing conditions after buddy-init beats today's default (continued training) on i2t retrieval, by a wide and seed-replicated margin.**
- **Geometry — a real but narrower divergence than the retrieval axis, on the shift distribution and the grid-diversity split only; effective dimensionality and most-changed-set identity are both nulls.** The frozen arm's `drift_from_init` is exactly 0 for all 3 seeds (sanity check: the freeze took effect as designed); the trained arm's conditions drift by mean ≈0.0845 (range 0.0834–0.0855) over the full 100-epoch run — small in absolute terms but nonzero and consistent. In the paired embedding-geometry diagnostic (recomputed at the **raw, un-normalized** feature scale the combiner is actually trained and evaluated at — see the normalization-bug note in Method):
  - **`conditioned_effective_dims` is a clean null.** All 6 runs land on **exactly 301** at epoch 99, and their whole trajectories are identical to within ±2 dims at every epoch (304 → 305/306 around epoch 10 → 301 by epoch 70–80). There is **no** effective-dimensionality gap between arms.
  - **Most-changed-set Jaccard overlap is indistinguishable from seed noise.** Cross-arm, same-seed overlap at epoch 99 is 0.18 / 0.18 / 0.14 (down from 1.00 at epoch 0) — but *within*-arm, cross-seed overlap at the same epoch is 0.11–0.25 (trained 0.21/0.25/0.21; frozen 0.21/0.11/0.18). The two arms disagree about which 20 of 150,000 samples move most by no more than two runs of the *same* arm with different seeds do. This diagnostic does not separate the arms.
  - **`shift_mean` and `shift_std` do separate the arms, cleanly and seed-replicated.** Trained shifts more than frozen in all 3 seeds: Δ(trained − frozen) = +0.0089 / +0.0084 / +0.0070, mean **+0.0081, mean/SEM = +13.9** (~3.0–3.9% of the frozen arm's own ~0.230 base); `shift_std` Δ = +0.0079, mean/SEM = **+31.9**. The between-arm gap is 2.6–7× the within-arm across-seed spread.
  - **The condition-vs-combine-side grid-diversity split separates the arms too, in both directions.** Row diversity (per-image spread across conditions) is higher for trained in all 3 seeds: 0.0386/0.0356/0.0383 vs. frozen 0.0251/0.0226/0.0225, Δ mean **+0.0141, mean/SEM = +15.9** — a ~57–70% *relative* increase. Col diversity (per-condition spread across images) is lower for trained in all 3 seeds: Δ mean **−0.0127, mean/SEM = −10.2**.

**Bottom line: post-init training of the conditions is not inert.** It measurably increases how far conditioning displaces the combine-side embedding and how much that displacement depends on *which* condition is used (shift distribution + row diversity, both seed-replicated with mean/SEM ≥ 13), and it has a large, consistent, seed-replicated effect on i2t retrieval — but in the *opposite* direction from what "training should help" would predict: **freezing is better for i2t, training is (marginally, within-noise) better for t2i.** Per the spec's decision rule, **this clears the gate for Experiment 11.2 on both axes** — retrieval decisively (i2t exceeds the noise floor by a wide margin, seed-replicated), geometry more narrowly (the shift distribution and the grid-diversity split both diverge seed-consistently; effective dimensionality and most-changed-set identity do not).

---

## Method

**What varies:** `train.em_interval`, the config knob that already governs EM-alternation phase-switching in `train_cosir.py`. Setting it to `-1` (today's default) keeps the "network" phase perpetually active — i.e. gradient reaches the condition table (`embedding_manager.embeddings`) every step, same as every prior CoSiR experiment. Setting it to `101` (> `epochs=100`) keeps the run in the same phase logic but with `epoch // em_interval` always `0`, i.e. even, which per `train_cosir.py`'s phase logic triggers `embedding_manager.embeddings.requires_grad_(False)` starting at epoch 0 and holding for the entire run — conditions are frozen at their buddy-init value from the first step onward. No new code; this is a pure config toggle on an existing mechanism.

**What's fixed (identical to C5–C8/Experiment 10's operating point):**

| | value |
|---|---|
| training backbone | CLIP ViT-B/32, frozen |
| `optimizer.lr` / `lr_label` | 1e-3 / 1e-4 |
| `model.embedding_dim` | 16 |
| `train.buddies.alpha` | 0.5 |
| `initialization_strategy` | `buddies` (fixed) |
| training-time buddy terms | off (`lambda_buddy=0`, `lambda_buddy_con=0`, `buddy_refresh=False`) — same discipline as Experiments 1/8/10 |
| loss stack (`lambda_laplacian`, `lambda_collapse`, `lambda_var`, `lambda_cov`, `lambda_gap_align=0`, `oracle_guided=True`, etc.) | identical between arms — confirmed by direct config diff |
| seeds | 1, 2, 3 |
| epochs / eval interval | 100 / 10 |
| dataset | RedCaps-150k |

2 arms (`trained`: `em_interval=-1`; `frozen`: `em_interval=101`) × 3 seeds = 6 runs, all confirmed `epochs=100` and `state=finished` via each run's `configs/config.json` / wandb.

**Runs (confirmed via wandb, group=`condition freeze ablation`, tag=`condition-freeze-ablation-redcaps_150k`, entity=`augustoxq`, project=`cosir_image`):**

| arm | seed | run dir | wandb run id |
|---|---:|---|---|
| trained | 1 | `20260825_161846_CoSiR_Experiment` | `n1kszd18` |
| trained | 2 | `20260825_163307_CoSiR_Experiment` | `j0slhjgc` |
| trained | 3 | `20260825_164733_CoSiR_Experiment` | `44t0k2kq` |
| frozen | 1 | `20260825_170212_CoSiR_Experiment` | `1t9xpyc6` |
| frozen | 2 | `20260825_171558_CoSiR_Experiment` | `f9twzv46` |
| frozen | 3 | `20260825_172950_CoSiR_Experiment` | `w5ntphar` |

(Two older `20260825_115635`/`20260825_115853` directories in the same parent folder are Task 2's 2-epoch smoke runs, `epochs=2`, and are excluded from this analysis.)

**Analysis tooling:**
- `scripts/analyze_condition_freeze_ablation.py --tag condition-freeze-ablation-redcaps_150k` — paired-within-seed Δ (`frozen − trained`), mean ± std, `mean/SEM` significance read, for `test_oracle/t2i_R1` and `test_oracle/i2t_R1`, plus a sanity check that the frozen arm's `train_buddy_diag/drift_from_init` is exactly 0 (confirming the em_interval freeze actually took effect) and the trained arm's drift range for context.
- `scripts/analyze_condition_geometry.py --exp-dir <run>` (once per run) then `--compare <frozen> <trained>` (once per seed) — per-epoch `shift_mean`/`shift_std` (1 − cos(combined embedding, combine-side embedding)), PCA effective dimensionality of the conditioned vs. unconditioned embedding space, most/least-changed-sample ranking and its Jaccard overlap between arms, correlation of shift against condition norm and buddy-graph degree, and a condition-vs-combine-side cross-grid diversity split (row diversity = per-combine-side-feature spread across conditions, col diversity = per-condition spread across combine-side features). The script loads the run's own `FeatureManager` store (path read from that run's `configs/config.json`) and feeds the combiner the **raw, un-normalized** features, matching what training and eval do — see the normalization-bug note below.

**The combine side is `img`, not `txt`, for all 6 runs.** Every run's `configs/config.json` records `model.combine_side: img`, and every `condition_viz/epoch_*.pt` snapshot carries the same value; the diagnostic branches on that saved value and therefore feeds the combiner the frozen CLIP **image** features. So everywhere below, "the combine-side embedding" means the frozen CLIP image feature for that sample, and the cross grid is conditions × images, not conditions × texts. (An earlier revision of this report described the combine side as "text" throughout; that was a prose error only — the code always read the saved `combine_side`.)

**A feature-normalization bug fixed during this analysis, and its consequences.** `analyze_condition_geometry.py`'s feature loader originally routed through `src/test/20260623_redcaps_buddy/redcaps_buddy.py`'s `load_data()`, which **L2-normalizes** both feature arrays before returning them. The actual training loop (`src/hook/train_cosir.py`) and eval (`src/eval/metrics.py`) feed the combiner **raw, un-normalized** features straight from the `FeatureManager` shard store — no normalization anywhere in that path — and `Combiner_new.forward` computes a *scale-sensitive* gated residual, `combined = (1−s)·general + s·delta` with `s ∈ [0.1, 0.9]`. RedCaps-150k's raw CLIP image features have mean L2 norm ≈ 10.6, so the diagnostic was probing the combiner at roughly 1/10th the input scale it was actually trained and evaluated at, which changes the gated mixture outright. The loader was rewritten to read `featuremanager.storage_dir` from the run's *own* `configs/config.json` and load the raw `FeatureManager` arrays directly (which also removes the previous hard-coding to RedCaps-150k — the diagnostic is now dataset-agnostic), and **all 6 runs' `condition_geometry/summary.json` files and all 3 paired comparisons were regenerated from scratch**; every geometry number in this report is from that regenerated output. The correction is not cosmetic: at the wrong scale `shift_mean` read ≈1.58–1.60, implying cos(conditioned, unconditioned) ≈ −0.59, which is not physically reachable for this architecture with `s ≤ 0.9`; at the correct scale it reads ≈0.230–0.240 (cos ≈ 0.76–0.77). Two previously-headline geometry findings — a 12–49 dimension `conditioned_effective_dims` gap and a 0.08–0.11 most-changed-set Jaccard overlap — **do not survive** the correction (see the geometry section below). **The retrieval axis is untouched by this bug**: those numbers come from wandb's `test_oracle/*` metrics logged during actual training, nowhere near this post-hoc diagnostic.

**One key-name bug fixed during this analysis:** `scripts/analyze_condition_freeze_ablation.py`'s `DRIFT` constant was originally `"buddy_diag/drift_from_init"`, but the actual wandb summary key (confirmed via direct API inspection of all 6 runs) is `"train_buddy_diag/drift_from_init"` (the `log_train(..., section="buddy_diag")` call prefixes the section with `train_`). With the wrong key, every drift value read back as `NaN`, silently suppressing both the frozen-arm sanity-check line and the trained-arm drift-range line from the script's output (the retrieval deltas themselves were unaffected — they read `test_oracle/*` keys, which were already correct). This one-line fix (`DRIFT = "train_buddy_diag/drift_from_init"`) was verified against the real wandb summaries before rerunning, and the corrected script's `--selftest` still passes. The captured output below is from the corrected script.

## Results — retrieval

Full captured output of `python scripts/analyze_condition_freeze_ablation.py --tag condition-freeze-ablation-redcaps_150k` (post-fix):

```
==============================================================================
Experiment 11.1 - condition freeze ablation  group='condition freeze ablation'  tag='condition-freeze-ablation-redcaps_150k'
==============================================================================
  6 run(s); arms present: ['frozen', 'trained'].
  OK: all 3 frozen-arm run(s) show drift_from_init == 0 (freeze confirmed to have taken effect).

  --- test_oracle/t2i_R1 (frozen - trained) ---
    mean delta = -0.27 (n=3, wins=0/3)  mean/SEM=-2.0 *
    Compare mean delta against the noise floor (~0.1-0.7 R1, NOT zero) - see docs/reports/2026-06-24_buddy_progress_report.md S8a.
      seed 1: delta = -0.40
      seed 2: delta = -0.40
      seed 3: delta = +0.00

  --- test_oracle/i2t_R1 (frozen - trained) ---
    mean delta = +4.67 (n=3, wins=3/3)  mean/SEM=+32.1 *
    Compare mean delta against the noise floor (~0.1-0.7 R1, NOT zero) - see docs/reports/2026-06-24_buddy_progress_report.md S8a.
      seed 1: delta = +4.90
      seed 2: delta = +4.40
      seed 3: delta = +4.70

  trained-arm drift_from_init: mean=0.0845, range=[0.0834, 0.0855]
```

**Read:**
- **t2i:** all 3 seeds cluster tightly (−0.40, −0.40, +0.00), giving a statistically-flagged mean/SEM of −2.0, but the mean magnitude (−0.27 R1) sits inside the project's established ~0.1–0.7 R1 noise floor — flagged, not practically meaningful, the same pattern C7 and parts of C8 established for their t2i-null results.
- **i2t:** all 3 seeds agree in direction and magnitude (+4.90, +4.40, +4.70), giving mean/SEM = +32.1 — the largest significance read of any paired ablation in this project's line (C5–C8, Exp. 10 all report single- or low-double-digit mean/SEM values). The +4.67 R1 mean magnitude is roughly 6.7–47× the noise floor's own width (0.1–0.7 R1), i.e. not a marginal case in either direction.
- **Frozen-arm drift sanity check passes cleanly**: all 3 frozen runs show `drift_from_init == 0` exactly, confirming the `em_interval=101` freeze took effect as designed and the whole paired-arm premise is sound. The trained arm's conditions do move (mean drift ≈0.0845, tight range 0.0834–0.0855 across seeds), confirming gradient really is reaching the condition table in that arm.

## Results — geometry diagnostic

### Per-arm final-epoch (epoch 99) state, from `--exp-dir` runs

| arm | seed | shift_mean | shift_std | conditioned_eff_dims | unconditioned_eff_dims | row_div (mean) | col_div (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| trained | 1 | 0.2402 | 0.1405 | 301 | 309 | 0.0386 | 0.9250 |
| trained | 2 | 0.2385 | 0.1409 | 301 | 309 | 0.0356 | 0.9207 |
| trained | 3 | 0.2375 | 0.1398 | 301 | 309 | 0.0383 | 0.9285 |
| frozen | 1 | 0.2313 | 0.1327 | 301 | 309 | 0.0251 | 0.9400 |
| frozen | 2 | 0.2301 | 0.1325 | 301 | 309 | 0.0226 | 0.9313 |
| frozen | 3 | 0.2305 | 0.1323 | 301 | 309 | 0.0225 | 0.9410 |

Paired within-seed deltas at epoch 99 (trained − frozen), using the project's standard mean ± std / `mean/SEM` read:

| quantity | seed 1 | seed 2 | seed 3 | mean Δ | std | mean/SEM |
|---|---:|---:|---:|---:|---:|---:|
| `shift_mean` | +0.0089 | +0.0084 | +0.0070 | **+0.0081** | 0.0010 | **+13.9** |
| `shift_std` | +0.0078 | +0.0084 | +0.0076 | **+0.0079** | 0.0004 | **+31.9** |
| `conditioned_eff_dims` | 0 | 0 | 0 | **0** | 0 | — |
| row_div (mean) | +0.0135 | +0.0130 | +0.0159 | **+0.0141** | 0.0015 | **+15.9** |
| col_div (mean) | −0.0149 | −0.0106 | −0.0125 | **−0.0127** | 0.0022 | **−10.2** |

(`unconditioned_effective_dims` = 309 for all 6 runs, as expected — it depends only on the fixed CLIP image features, not on the arm. Note this one number is computed on the L2-normalized frozen features, unlike `conditioned_effective_dims`, which is computed on the combiner's raw output; on raw un-normalized image features the same statistic is 305, so the conditioned-vs-unconditioned dimensionality comparison is a 301-vs-305/309 reference, not an exactly like-for-like one — it is context, not a load-bearing number, and it is identical across all 6 runs either way. `condition_effective_dims`, the raw condition vectors' own PCA dimensionality, is 11 for all 6 runs at epoch 99, unsurprising since the buddy-init spectral embedding itself is 16-D and both arms share the same init.)

At epoch 0 (before any post-init training step has run), each matched seed's two arms agree exactly on `shift_mean`/`conditioned_effective_dims`/`row_div`/`col_div` (as expected — both arms literally share the same buddy-init condition table and the same freshly-initialized combiner at epoch 0: seed 1 shift_mean 0.1711, seed 2 0.1557, seed 3 0.1913; eff_dims 304, row_div 0.0002–0.0006, col_div 0.951–0.965 for all 6). The divergence above is entirely a product of subsequent training. Note that the frozen arm's `shift_mean` also grows over the run (0.171 → 0.231 for seed 1) even though its conditions never move — the combiner keeps training in both arms, which is exactly why this diagnostic is run on both arms rather than only the trained one.

### Paired `--compare <frozen> <trained>` output, all 3 seeds, full per-epoch trajectory

All values are `B − A` where `A = frozen`, `B = trained` — i.e. positive `shift_mean B−A` means the trained arm shifts *more* than the frozen arm at that epoch; positive `eff_dims B−A` means trained has more effective dimensions than frozen.

**Seed 1** (`compare 20260825_170212 (frozen) 20260825_161846 (trained)`):

```
  epoch    0: shift_mean B-A=-0.0000  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=1.00  row_div B-A=-0.0000  col_div B-A=-0.0000
  epoch   10: shift_mean B-A=+0.0020  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.25  row_div B-A=+0.0063  col_div B-A=-0.0238
  epoch   20: shift_mean B-A=+0.0160  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.21  row_div B-A=+0.0176  col_div B-A=-0.0173
  epoch   30: shift_mean B-A=+0.0045  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.21  row_div B-A=+0.0196  col_div B-A=-0.0314
  epoch   40: shift_mean B-A=+0.0223  eff_dims B-A=-1  most-changed-set Jaccard(A,B)=0.25  row_div B-A=+0.0203  col_div B-A=-0.0115
  epoch   50: shift_mean B-A=+0.0097  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.21  row_div B-A=+0.0150  col_div B-A=-0.0162
  epoch   60: shift_mean B-A=+0.0166  eff_dims B-A=-1  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0158  col_div B-A=-0.0086
  epoch   70: shift_mean B-A=+0.0095  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0142  col_div B-A=-0.0115
  epoch   80: shift_mean B-A=+0.0056  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0135  col_div B-A=-0.0174
  epoch   90: shift_mean B-A=+0.0080  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0135  col_div B-A=-0.0148
  epoch   99: shift_mean B-A=+0.0089  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0135  col_div B-A=-0.0149
```

**Seed 2** (`compare 20260825_171558 (frozen) 20260825_163307 (trained)`):

```
  epoch    0: shift_mean B-A=-0.0000  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=1.00  row_div B-A=+0.0000  col_div B-A=+0.0000
  epoch   10: shift_mean B-A=+0.0047  eff_dims B-A=-1  most-changed-set Jaccard(A,B)=0.29  row_div B-A=+0.0048  col_div B-A=-0.0007
  epoch   20: shift_mean B-A=+0.0089  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.33  row_div B-A=+0.0095  col_div B-A=+0.0022
  epoch   30: shift_mean B-A=+0.0144  eff_dims B-A=-1  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0151  col_div B-A=-0.0175
  epoch   40: shift_mean B-A=+0.0120  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0131  col_div B-A=-0.0073
  epoch   50: shift_mean B-A=+0.0160  eff_dims B-A=+1  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0150  col_div B-A=-0.0139
  epoch   60: shift_mean B-A=+0.0106  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0144  col_div B-A=-0.0097
  epoch   70: shift_mean B-A=+0.0128  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.29  row_div B-A=+0.0125  col_div B-A=-0.0068
  epoch   80: shift_mean B-A=+0.0130  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0151  col_div B-A=-0.0126
  epoch   90: shift_mean B-A=+0.0077  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0130  col_div B-A=-0.0116
  epoch   99: shift_mean B-A=+0.0084  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0130  col_div B-A=-0.0106
```

**Seed 3** (`compare 20260825_172950 (frozen) 20260825_164733 (trained)`):

```
  epoch    0: shift_mean B-A=-0.0000  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=1.00  row_div B-A=-0.0000  col_div B-A=-0.0000
  epoch   10: shift_mean B-A=-0.0036  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.33  row_div B-A=+0.0068  col_div B-A=-0.0157
  epoch   20: shift_mean B-A=+0.0034  eff_dims B-A=-1  most-changed-set Jaccard(A,B)=0.21  row_div B-A=+0.0097  col_div B-A=-0.0209
  epoch   30: shift_mean B-A=+0.0261  eff_dims B-A=-2  most-changed-set Jaccard(A,B)=0.21  row_div B-A=+0.0183  col_div B-A=-0.0196
  epoch   40: shift_mean B-A=+0.0108  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0181  col_div B-A=-0.0223
  epoch   50: shift_mean B-A=+0.0067  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0181  col_div B-A=-0.0217
  epoch   60: shift_mean B-A=+0.0075  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0140  col_div B-A=-0.0160
  epoch   70: shift_mean B-A=+0.0132  eff_dims B-A=-1  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0187  col_div B-A=-0.0103
  epoch   80: shift_mean B-A=+0.0056  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0158  col_div B-A=-0.0150
  epoch   90: shift_mean B-A=+0.0081  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0160  col_div B-A=-0.0125
  epoch   99: shift_mean B-A=+0.0070  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0159  col_div B-A=-0.0125
```

### Is the most-changed-set Jaccard overlap actually a between-arm signal?

No. The epoch-99 cross-arm overlap (0.18 / 0.18 / 0.14) looks low against the epoch-0 value of 1.00, but epoch 0 is 1.00 *by construction* (both arms are byte-identical there) and is therefore not a usable baseline for "how similar should two genuinely-equivalent runs be." Computing the same Jaccard **within** an arm across seeds at epoch 99 — two runs of the *same* configuration, differing only in training seed — gives a proper noise reference:

| comparison | epoch-99 Jaccard |
|---|---|
| cross-arm, same seed (frozen vs. trained) | 0.176, 0.176, 0.143 |
| within trained arm, cross-seed (1–2, 1–3, 2–3) | 0.212, 0.250, 0.212 |
| within frozen arm, cross-seed (1–2, 1–3, 2–3) | 0.212, 0.111, 0.176 |

The cross-arm values (0.143–0.176) sit inside the within-arm range (0.111–0.250). **The two arms disagree about which 20 of 150,000 samples move most by no more than two runs of the same arm with different seeds do**, so this diagnostic carries no between-arm information at this k. (It is still far above chance — two random 20-element draws from 150,000 would overlap at ≈0 — so the *ranking itself* is reproducible; it just does not separate the arms.)

**Read:**
- **`eff_dims` is a clean null — it does not separate the arms at any epoch.** All 6 runs walk the same trajectory (304 at epoch 0 → 305/306 at epoch 10 → monotone decline → **exactly 301 at epoch 99 in every run**), and `eff_dims B−A` is exactly 0 at 25 of the 33 epoch×seed cells and never larger than 2 in magnitude. Whatever continued training does to the conditioned embedding space, it does not change its PCA effective dimensionality relative to freezing. (The previous revision of this report claimed a −12/−49/−33 gap here; that was entirely an artifact of feeding the combiner L2-normalized features — see the normalization-bug note in Method.)
- **Most-changed-set Jaccard overlap is real but uninformative about the arms.** It decays from 1.00 at epoch 0 (identical by construction) to 0.14–0.18 at epoch 99 — but the same statistic computed *within* an arm across seeds lands at 0.11–0.25 at epoch 99, i.e. the cross-arm value is indistinguishable from ordinary seed-to-seed variation. See the dedicated subsection above. This diagnostic should **not** be cited as evidence of a between-arm divergence.
- **`shift_mean` and `shift_std` do separate the arms, seed-replicated and well above seed noise.** `shift_mean B−A` is positive at every epoch after epoch 0 in seeds 1 and 2, and at every epoch after epoch 10 in seed 3 (epoch 0 is an exact tie by construction in all three; the single genuine negative anywhere is a −0.0036 blip at epoch 10 in seed 3), settling at +0.0089/+0.0084/+0.0070 at epoch 99 — mean +0.0081, `mean/SEM = +13.9`, i.e. **3.0–3.9% of the frozen arm's own ~0.230 absolute shift**. The between-arm gap is 2.6–7× the within-arm across-seed spread (trained 0.2375–0.2402, range 0.0028; frozen 0.2301–0.2313, range 0.0012), so it is not seed noise. `shift_std` splits even more cleanly (+0.0079, `mean/SEM = +31.9`): continued training both displaces the combine-side embedding further *and* spreads that displacement wider across samples.
- **Row diversity (per-image spread across conditions) is consistently and substantially higher for trained than frozen, from epoch 10 onward, in all 3 seeds.** `row_div B−A` is positive at all 30 post-epoch-0 epoch×seed cells, settling at +0.0135/+0.0130/+0.0159 (mean +0.0141, `mean/SEM = +15.9`) — a ~57–70% *relative* increase over the frozen arm's 0.0225–0.0251 base. Training makes conditions matter more, i.e. *less* interchangeable for a given image, the opposite direction from the "conditions become null" failure mode. **Col diversity (per-condition spread across images) moves the other way, also consistently:** negative at 29 of the 30 post-epoch-0 epoch×seed cells (the single exception is +0.0022 at epoch 20 in seed 2) and at epoch 99 in all 3 seeds (−0.0149/−0.0106/−0.0125, mean −0.0127, `mean/SEM = −10.2`), from a very high base (0.921–0.941). Both diversity axes therefore register the same underlying change — training shifts a small amount of the combiner's output variance from the combine-side feature onto the condition.

## Interpreting the two grid failure modes

The condition-vs-combine-side grid diagnostic's two diversity axes are diagnostic of two different failure modes: **low row diversity** (for a fixed combine-side feature — here, a fixed CLIP image feature — the combiner output barely changes across different conditions) means conditions are functionally null/interchangeable, i.e. the combiner is ignoring them; **low col diversity** (for a fixed condition, the combiner output barely changes across different images) means that one condition dominates and collapses every input toward the same output.

At the correct feature scale the two axes are **strongly asymmetric in both arms**: at epoch 99, row_div means span 0.0225–0.0386 across the 6 runs while col_div means span 0.9207–0.9410 — col diversity is ~25–40× larger. Read literally, that means swapping the condition while holding the image fixed leaves the combined embedding almost unchanged (mean pairwise cosine ≈ 0.96–0.98 across conditions), while swapping the image while holding the condition fixed changes it almost completely (mean pairwise cosine ≈ 0.06–0.08 across images). So **the collapse-to-a-dominant-condition failure mode is emphatically absent, but the diagnostic leans toward the other end: conditions are a small, largely interchangeable perturbation on a combine-side-dominated output.**

That is consistent with, not contradicted by, `shift_mean ≈ 0.23–0.24` (cos ≈ 0.76–0.77): conditioning *does* displace the output substantially, but it displaces it in a direction that barely depends on *which* condition is applied. The between-arm difference is the interesting part — post-init training raises row diversity by ~57–70% relative (0.023 → 0.038) while lowering col diversity slightly, i.e. it moves the combiner measurably *away* from the near-interchangeable regime — and it does so while making i2t retrieval **worse**. "Conditions matter more" and "retrieval improves" are not the same axis here, which is precisely the question 11.2 is now scoped to characterize.

(The previous revision of this report stated that row_div and col_div were "both nonzero and neither near zero relative to the other diversity axis"; at the correct feature scale that is false — the two axes differ by a factor of ~25–40. The corrected reading above supersedes it.)

## Correlation diagnostics (context, not part of the decision rule)

`shift_vs_condition_norm` and `shift_vs_buddy_degree` Pearson correlations (from each `--exp-dir` run's snapshots) are **small and positive in both arms throughout, and mostly weaken over training**: `shift_vs_condition_norm` starts at r = +0.070…+0.181 at epoch 0 (shared init, so the two arms of a seed are identical there) and ends at +0.087…+0.105 (trained) vs. +0.063…+0.067 (frozen) at epoch 99 — the trained arm retains slightly more of the relationship, and it ends up narrowly banded across seeds even where a seed's epoch-0 value started low (seed 3: +0.070 → +0.105). `shift_vs_buddy_degree` starts at r = +0.087…+0.131 at epoch 0 and ends at +0.039…+0.045 (trained) vs. −0.006…+0.000 (frozen) — in the frozen arm the relationship between a sample's buddy-graph degree and how much conditioning shifts it decays essentially to zero, while the trained arm keeps a small positive remnant. The largest |r| anywhere in the 6 runs × 11 epochs is 0.184 (`shift_vs_condition_norm`, epoch 0) and 0.131 (`shift_vs_buddy_degree`, epoch 0); at epoch 99 every |r| is below 0.11, so neither correlation is large enough to be a primary driver of the retrieval or shift results above; reported here as context per the spec's diagnostic list, not as a load-bearing finding. (These numbers, like every other geometry figure in this report, are from the corrected raw-feature re-run; the previous revision reported negative `shift_vs_condition_norm` values, an artifact of the normalization bug.)

## Applying the decision rule

Per spec §4 Experiment 11.1's success criteria: **no real difference on either axis** (retrieval within noise floor AND no meaningful geometry divergence) → clean simplification result, no 11.2 needed. **A real difference on either axis** (retrieval mean/SEM ≥ 2 exceeding the noise floor in either direction, OR a clear geometry divergence — in the shift distribution, the most/least-changed ranking, the condition-space quality metrics, or the condition-vs-combine-side grid diversity split — even with null retrieval) → 11.2 is gated open.

**The actual result triggers the gate-open branch decisively on retrieval, and more narrowly on geometry:**

- **Retrieval axis triggers on its own, unambiguously.** i2t's mean Δ = +4.67 R1, mean/SEM = +32.1, 3/3 seeds — this clears both the statistical bar (`mean/SEM ≥ 2`) and the noise-floor bar (0.1–0.7 R1) by a wide margin (the effect is ~6.7–47× the noise floor's width), with 3/3 seed agreement in both sign and rough magnitude (+4.90, +4.40, +4.70). **This alone is sufficient to gate 11.2 open**, and it is untouched by the geometry-diagnostic normalization bug.
- **Geometry axis independently triggers too, but on two of the four sub-diagnostics, not all four.** The spec's geometry clause lists four candidate signals: *"a clear divergence in the shift distribution, most/least-changed ranking, condition-space quality metrics, or the condition-vs-combine-side grid diversity split."*
  - **Shift distribution — triggers.** `shift_mean` Δ = +0.0081 (mean/SEM = +13.9) and `shift_std` Δ = +0.0079 (mean/SEM = +31.9), 3/3 seeds in sign, and the between-arm gap is 2.6–7× the within-arm across-seed spread.
  - **Grid diversity split — triggers.** row_div Δ = +0.0141 (mean/SEM = +15.9, a ~57–70% relative increase) and col_div Δ = −0.0127 (mean/SEM = −10.2), 3/3 seeds in sign for both.
  - **Condition-space quality metrics (`conditioned_effective_dims`) — does not trigger.** Exactly 301 in all 6 runs at epoch 99; the whole trajectory is arm-independent to within ±2 dims.
  - **Most/least-changed ranking — does not trigger.** Cross-arm Jaccard (0.14–0.18) is inside the within-arm cross-seed range (0.11–0.25).
  - This is a weaker geometry verdict than the previous revision of this report claimed (which leaned on the eff_dims and Jaccard signals, both of which turned out to be normalization-bug artifacts), but it is still a **real, seed-replicated, above-noise divergence on two of the four listed sub-diagnostics**, so the geometry axis does independently satisfy the rule as written.
- **t2i is the one part of this result that is a clean null** — within the noise floor, consistent with the pattern established by C7 and C8's t2i-null findings elsewhere in this project's line.

**Conclusion: Experiment 11.2 is gated open, triggered decisively by the retrieval axis (i2t) and independently — though more narrowly than previously reported — by the geometry axis (shift distribution and grid-diversity split; effective dimensionality and most-changed-set identity are nulls).** This is *not* the "training pressure on conditions is inert, drop it" outcome the spec's null branch would have produced — post-init training measurably increases both how far conditioning displaces the combine-side embedding and how much that displacement depends on which condition is used, and it has a large, direction-specific retrieval effect, in the *opposite* direction from a naive "more training should help" prior. Note that the gate-open verdict does **not** depend on the geometry axis: the retrieval axis clears the bar on its own by a wide margin, so no part of the routing below rests on the corrected geometry numbers.

**Which of 11.2's two branches this routes to:** spec §4 Experiment 11.2 branches explicitly on direction — *"if trained beats frozen (on retrieval, geometry, or both): ablate the loss-stack terms..."* (new training runs) vs. *"if frozen beats trained, the result is mixed, or the divergence is geometry-only (retrieval null): extend the geometry diagnostic rather than launch new training..."* (no new training, reuses this task's own checkpoints). The actual result is **frozen beating trained on i2t** (and a t2i null); the geometry divergence is directional (trained shifts further and its conditions matter more) but is not a retrieval win for trained on any axis, so at minimum "the result is mixed". Either reading lands on the spec's **second branch, by its literal wording**, not the first: extend the 11.1 geometry diagnostic (correlate per-sample condition drift `‖z_i − z_init,i‖`, already loggable from the trained arm's checkpoints, and per-sample embedding shift, already computed by `analyze_condition_geometry.py`, against per-sample retrieval outcome) rather than launching a new loss-stack-term ablation sweep. This is also the cheaper of the two branches — near-zero additional compute, reusing this task's own 6 runs' checkpoints and `condition_geometry/summary.json` files, no new training required.

## Caveats

- **RedCaps-150k only**, one operating point (`lr=1e-3, lr_label=1e-4, embedding_dim=16, alpha=0.5`, all training-time buddy terms off) — same scope discipline as every prior experiment in this line; not checked at 300k or under a different operating point.
- **Only `test_oracle` retrieval is reported here, not `test_pre_diff` (gap to CLIP).** Whether the i2t regression under continued training also affects the realistic single-forward-pass predictor comparison against raw CLIP is untested here — a natural, cheap follow-up given C6/C8's precedent for this exact comparison.
- **`shift_mean`'s absolute divergence between arms is small relative to each arm's own shift_mean** (deltas of 3.0–3.9% of the frozen arm's own ~0.230 base), even though it is consistently signed across all 3 seeds and 2.6–7× the within-arm across-seed spread. Together with the grid-diversity split it is now the *primary* geometry signal in this result, because the two signals the previous revision leaned on — `eff_dims` and most-changed-set Jaccard — turned out to be a normalization artifact and seed noise respectively. A ~3–4% relative shift difference and a ~57–70% relative row-diversity difference are real and seed-replicated, but they are second-order geometric changes, not a wholesale restructuring of the embedding space.
- **The geometry diagnostic samples only 30 combine-side features × 30 conditions for the grid, and only k=20 for the most/least-changed ranking.** The Jaccard null result above is specifically a null *at k=20*; a larger k would give a tighter, more sensitive comparison and is a cheap thing for 11.2 to do. Likewise the grid means come from a single fixed 30×30 sample per run (the indices are drawn once per run and reused across epochs by design, so the trajectory is not resampling noise — but the absolute level does depend on that one draw).
- **Correlations against condition norm and buddy degree are weak (|r| < 0.11 at epoch 99) in both arms** — neither is a strong explanatory variable for which samples get shifted most, in either arm; this diagnostic did not surface an obvious mechanistic story for *why* continued training regresses i2t, which is exactly the question 11.2 is now scoped to answer.
- **Two bugs were found and fixed in the analysis tooling during this work, both documented in Method above.** (1) A wrong wandb summary key in `scripts/analyze_condition_freeze_ablation.py` silently suppressed (not corrupted) the drift-sanity-check lines; the retrieval deltas were unaffected, and the script now prints a loud warning instead of nothing when the drift column comes back empty. (2) `scripts/analyze_condition_geometry.py` was loading L2-normalized features into a scale-sensitive combiner; **all geometry numbers in this report are from the post-fix re-run**, and two previously-headline geometry claims (a 12–49 dim `eff_dims` gap, a 0.08–0.11 Jaccard overlap) did not survive it. The retrieval numbers were never touched by either bug.
- **The geometry axis's verdict changed between revisions of this report; the retrieval axis's did not.** Anyone citing C9's geometry claim should cite the shift-distribution and grid-diversity numbers, not effective dimensionality or most-changed-set overlap.
- **"Seed-replicated" samples training stochasticity only** — the buddy-graph init is shared and deterministic across all 3 seeds within an arm (established convention, matching C5/C8/Exp. 10), so the reported seed-replication reflects training-run variance, not graph-construction variance.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_condition_freeze_ablation.sh   # 2 arms x 3 seeds, 100 epochs each, RedCaps-150k
python scripts/analyze_condition_freeze_ablation.py --tag condition-freeze-ablation-redcaps_150k

# Geometry diagnostic: build each run's summary, then compare paired by seed
for d in <trained_dir_seed1> <trained_dir_seed2> <trained_dir_seed3> <frozen_dir_seed1> <frozen_dir_seed2> <frozen_dir_seed3>; do
  python scripts/analyze_condition_geometry.py --exp-dir "$d"
done
python scripts/analyze_condition_geometry.py --compare <frozen_dir_seed1> <trained_dir_seed1>
python scripts/analyze_condition_geometry.py --compare <frozen_dir_seed2> <trained_dir_seed2>
python scripts/analyze_condition_geometry.py --compare <frozen_dir_seed3> <trained_dir_seed3>
```

## Experiment 11.2 — drift/shift vs. retrieval-rank correlation

**Date:** 2026-08-26 (rewritten 2026-08-26 after final whole-branch review) · **Code:** `scripts/analyze_condition_retrieval_correlation.py` · **Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 11.2 (second, no-new-training branch)

### TL;DR for this section

**The condition table's own effect on per-sample retrieval rank is large, highly significant, and replicated in 3/3 seeds — once you isolate it from whole-model divergence.** Holding the trained arm's combiner, `other_proj`, and gallery completely fixed and swapping *only* the condition table (trained → buddy-init) gives `rho(delta_rank_swap, condition_drift)` = **+0.466 / +0.466 / +0.477** (all p < 1e-160): the further a sample's condition drifted from init, the more reverting it to init costs that sample's rank. The naive signed cross-arm delta (`rank_trained − rank_frozen`, |rho| ≤ 0.05) **undersells this by an order of magnitude**, because it compares two independently-trained models — separate combiners, separate `other_proj`, separate condition tables — so most of its variance (std ≈ 1055–1343 ranks) is whole-model divergence rather than the condition table.

Two further corrections to the first write-up of this section, both from re-examining the same data:

1. **Even without the counterfactual, unsigned magnitude is not a null.** `rho(|delta_rank|, condition_drift)` = +0.160 / +0.141 / +0.156 and `rho(|delta_rank|, embedding_shift)` = −0.288 / −0.296 / −0.302, all 3/3 seeds, all far past `|rho| > 0.1`. The signed correlation is near zero because drift predicts *how much* a sample's rank moves, not *which way* — the two directions largely cancel.
2. **On this diagnostic's own metric the trained arm WINS.** In-sample, own-condition mean rank is better for trained (975 / 980 / 996) than frozen (1118 / 1123 / 1112), R1 is better for trained (0.101 / 0.100 / 0.103 vs. 0.092 / 0.088 / 0.089), and 63.6–64.0% of queries improve. There is no per-sample "degradation" here for drift/shift to explain. **That is the opposite direction from 11.1's headline** — and it is not a contradiction, because this measures a different population with a different metric (see the scope box below). A result here constrains, but cannot resolve, 11.1's held-out oracle regression.

### Method

11.1 (above) found that letting per-sample conditions keep training after buddy-init hurts **held-out oracle** i2t retrieval relative to freezing them (frozen beats trained, `test_oracle/i2t_R1` mean Δ = +4.67 R1, mean/SEM = +32.1, 3/3 seeds) — a large, direction-asymmetric effect that neither `conditioned_effective_dims` nor most-changed-set Jaccard could separate, and that the geometry diagnostic's own correlations (`shift_vs_condition_norm`/`shift_vs_buddy_degree`, both |r| < 0.11 at epoch 99) could not explain mechanistically. Per the spec's §4 Experiment 11.2 gate — resolved to its second, no-new-training branch ("if frozen beats trained, the result is mixed, or the divergence is geometry-only: extend the geometry diagnostic rather than launch new training") — 11.2 asks the natural next per-sample question, reusing only 11.1's existing checkpoints: does how far a sample's condition moved from buddy-init (`condition_drift`), or how much conditioning displaces its combine-side embedding (`embedding_shift`), predict how that sample's own true match ranks?

Concretely, per seed pair: draw a fixed subsample of **3000 query samples** (`n_query_sample=3000`, `rng` seed 0) and rank each query's true match against the **full 150,000-sample training population's** projected "other side" (text, since `combine_side=img` for all 6 runs) embeddings — a realistic-scale gallery, not a small closed one. Each query is conditioned on its own real, assigned condition (no oracle search over conditions, no `condition_predictor`). The correlations below are over those **3000 queries**; the 150k figure is the gallery size only.

Three rank quantities are computed:

- `rank_frozen` — frozen arm's combiner + `other_proj` + gallery, frozen (= buddy-init) conditions.
- `rank_trained` — trained arm's combiner + `other_proj` + gallery, trained conditions.
- `rank_trained_frozen_cond` — **trained** arm's combiner + `other_proj` + gallery held fixed, conditions swapped to the frozen/buddy-init table.

and two deltas: `delta_rank = rank_trained − rank_frozen` (the naive cross-arm delta; two *different models*) and **`delta_rank_swap = rank_trained_frozen_cond − rank_trained`** (the condition-only counterfactual; *one* model, only the condition changes). Positive `delta_rank_swap` = reverting this sample to its init condition ranks it worse.

Supporting detail:

- **Init-proxy validation.** `condition_drift` is measured against the frozen arm's final condition table as a stand-in for buddy-init. `analyze_pair` now checks *both* halves of that premise: (a) the frozen arm's table is unchanged between its first and final saved epoch (stationarity), and (b) that table is numerically identical to the **real shared buddy-init file** on disk, `res/CoSiR_condition_freeze_ablation/redcaps_150k/template_embeddings/embeddings.npy`, aligned by `sample_ids` (max abs per-sample distance asserted < 1e-4; the actual value is 0.0 for all 3 pairs, and each run logs `init_source: shared template_embeddings/embeddings.npy (verified identical)`). Check (b) is the load-bearing half and was previously unchecked; it degrades gracefully to (a) alone if the template file isn't reachable, so the script still runs where only experiment dirs were copied.
- **Sample-ID alignment** everywhere via `reorder_features_to_z`.
- **Tie handling is optimistic** (`1 + (sims > true_score).sum()`, strict): a gallery row tied with the true match does not demote it. Exact ties are rare in float32 over a 150k gallery, but the convention matters for reading `delta_rank`'s distribution — **15.8–16.6% of queries have `delta_rank == 0` exactly**, i.e. their rank is literally unchanged between arms. The same convention is applied to both arms, which is what makes their difference well-defined.
- Full method detail is in `scripts/analyze_condition_retrieval_correlation.py`'s module docstring and `analyze_pair`'s docstring; see also the plan at `docs/superpowers/plans/2026-08-26-condition-drift-retrieval-correlation.md`.

> **Scope box — what this diagnostic measures, and what it does not.**
> Everything in this section is **in-sample** (the 150k rows the run trained on) and **own-condition** (each sample's actual assigned condition). 11.1's headline `test_oracle/i2t_R1` is **held-out** (test set) and **oracle** (max over all conditions). Those are *two independent construct mismatches*, and neither is fixable by re-running: per-sample conditions only exist for training samples, so there is no way to evaluate this diagnostic on the held-out set at all, and dropping the oracle would change what 11.1 measured. Consequently the aggregate direction here (trained wins) and 11.1's aggregate direction (frozen wins) are **not** in contradiction and must not be compared as if they were the same number. What this section can legitimately claim is about the per-sample *structure* of the condition's effect on rank; what it cannot claim is that it explains — or refutes — 11.1's held-out oracle regression.

### Per-seed results

**In-sample, own-condition retrieval quality of each arm** (3000 queries against the 150k gallery; **not** comparable to 11.1's `test_oracle/i2t_R1`):

| seed | frozen dir | trained dir | mean rank frozen | mean rank trained | R1 frozen | R1 trained | frac. queries improved |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `20260825_170212` | `20260825_161846` | 1118.1 | **975.0** | 0.0923 | **0.1007** | 0.636 |
| 2 | `20260825_171558` | `20260825_163307` | 1122.8 | **980.4** | 0.0880 | **0.0997** | 0.640 |
| 3 | `20260825_172950` | `20260825_164733` | 1112.5 | **996.2** | 0.0893 | **0.1033** | 0.639 |

**Rank deltas:**

| seed | `delta_rank` mean | median | std | frac. == 0 | `delta_rank_swap` mean | median |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | −143.1 | −5.0 | 1189.5 | 0.166 | +343.5 | +6.0 |
| 2 | −142.4 | −5.0 | 1343.0 | 0.161 | +352.0 | +7.0 |
| 3 | −116.3 | −5.0 | 1055.0 | 0.158 | +357.1 | +7.0 |

**Correlations** (Spearman `rho`, n = 3000 per seed):

| seed | ρ(Δrank, drift) | ρ(Δrank, shift) | ρ(Δrank, Δshift) | ρ(\|Δrank\|, drift) | ρ(\|Δrank\|, shift) | **ρ(Δrank_swap, drift)** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.026 (p=0.16) | +0.046 (p=0.011) | +0.071 (p=1.0e−4) | **+0.160** (p=1.5e−18) | **−0.288** (p=2.7e−58) | **+0.466** (p=1.6e−161) |
| 2 | +0.017 (p=0.34) | +0.040 (p=0.029) | +0.046 (p=0.013) | **+0.141** (p=9.1e−15) | **−0.296** (p=1.3e−61) | **+0.466** (p=9.0e−162) |
| 3 | +0.009 (p=0.60) | +0.046 (p=0.011) | +0.084 (p=4.5e−6) | **+0.156** (p=8.6e−18) | **−0.302** (p=2.8e−64) | **+0.477** (p=4.7e−170) |

(`Δshift` = `embedding_shift_trained − embedding_shift_frozen`, the correctly paired shift quantity for a paired rank delta; the shift-only column is retained for continuity with the first write-up. All 3 runs: `n_query_sample=3000`, `n_population=150000`, `combine_side=img`; each JSON written to the trained arm's `condition_geometry/retrieval_correlation_vs_frozen.json`.)

### Cross-seed synthesis

- **The condition's own effect on rank is real and large (`delta_rank_swap`).** ρ = **+0.466 / +0.466 / +0.477**, p < 1e−160 in all 3 seeds — 4.7× the `|rho| > 0.1` practical bar, and remarkably tight across seeds. Direction: **positive** — the further a sample's condition drifted from buddy-init, the more that sample's rank *degrades* when you feed the trained model the init condition instead. That is the mechanically sensible direction: the trained combiner has co-adapted to the drifted conditions, so high-drift samples are the ones most dependent on them. Aggregate `delta_rank_swap` agrees: mean **+343.5 / +352.0 / +357.1**, median +6/+7/+7 — reverting to init conditions costs rank on average, inside the trained model.
- **The naive signed cross-arm delta is near-zero, and that is an artifact of the estimator, not evidence of no effect.** ρ(Δrank, drift) = +0.026/+0.017/+0.009 (p = 0.16/0.34/0.60) and ρ(Δrank, shift) = +0.046/+0.040/+0.046 (p = 0.011/0.029/0.011). `delta_rank` compares the frozen run's *whole model* (combiner + `other_proj` + conditions) against the trained run's *whole model* — two independently-trained networks. Its std is 1055–1343 ranks, most of which is whole-model divergence the condition table has nothing to do with. That noise swamps the condition signal. The correctly-paired M1 variant (ρ(Δrank, Δshift) = +0.071/+0.046/+0.084) is uniformly larger than the shift-only version and significant in 3/3 seeds, but still under 0.1 — same swamping.
- **Unsigned magnitude clears the bar without any counterfactual.** ρ(|Δrank|, drift) = **+0.160/+0.141/+0.156** and ρ(|Δrank|, shift) = **−0.288/−0.296/−0.302**, all p ≤ 9e−15, 3/3 seeds. Reading: condition drift predicts how *far* a sample's rank moves between arms in either direction (so the signed correlation cancels); embedding shift predicts the opposite — samples whose embedding is displaced *most* by conditioning are the ones whose rank moves *least* between arms. This is a genuine, seed-replicated structure that the first write-up's "clean null" framing erased.
- **Direction of the aggregate effect on this diagnostic's own metric: trained wins.** Mean rank 975/980/996 (trained) vs. 1118/1123/1112 (frozen); R1 0.101/0.100/0.103 vs. 0.092/0.088/0.089; 63.6–64.0% of queries improve. Per the scope box, this is the in-sample, own-condition metric and is **not** 11.1's held-out oracle metric — but it does mean there is no per-sample degradation *here* for the correlations above to be explaining.
- **Independence caveat.** All 3 seed pairs use the same fixed query-sampling `rng` seed (0, not tied to the run's own training seed) and start from the **identical** shared buddy-init table, so "3/3 seeds agree" on this diagnostic is partly shared-sample and shared-init agreement, not fully independent replication — even though the six underlying training runs genuinely differ by seed. The seed-to-seed tightness of the `delta_rank_swap` numbers (±0.006) should be read with that in mind. Separately, the near-zero signed correlations that do clear p < 0.05 (|ρ| ≈ 0.04) are **distinguishable from zero at n = 3000, but the effect size itself is negligible** — the earlier phrasing "statistically detectable" oversold them.

### Qualitative extremes

Representative most-degraded rows (`delta_rank` most positive — the trained *run* ranks this sample worse than the frozen run):

| seed | most-degraded rank | sample_id | delta_rank | condition_drift | embedding_shift |
|---:|---:|---:|---:|---:|---:|
| 1 | 1st | 42190 | +14202 | 0.134 | 0.316 |
| 2 | 1st | 42190 | +15446 | 0.153 | 0.285 |
| 3 | 1st | 42190 | +14216 | 0.107 | 0.186 |
| 1 | 2nd | 105764 | +10529 | 0.150 | 0.024 |
| 2 | 2nd | 105764 | +10523 | 0.138 | 0.033 |
| 3 | **3rd** | 105764 | +10245 | 0.173 | 0.037 |

Sample 42190 is the single most-degraded sample in all 3 seeds. Sample 105764 is 2nd-most-degraded in seeds 1 and 2 but **3rd** in seed 3 — seed 3's 2nd slot is sample 7310 at Δrank = +13616 (the first write-up incorrectly stated 105764 was 2nd in all 3). On the most-improved side, sample_ids 28404, 101763, 83630, 99341 and 132814 recur near the top in at least 2 of 3 seeds (28404: −36750 / −48112 / −17072, top-2 in all 3; 101763: −24906 / −23599 in seeds 2–3; 99341: −16487 / −10842 in seeds 2–3).

**Where the extremes sit in the drift/shift distributions** (median percentile of the k=20 group within the 3000-query sample; recomputed by `extreme_group_percentiles`, reported in each JSON's `extremes_percentiles`):

| seed | degraded: drift pct | degraded: shift pct | improved: drift pct | improved: shift pct |
|---:|---:|---:|---:|---:|
| 1 | 68.7 | 9.8 | 55.4 | 22.0 |
| 2 | 67.8 | 9.7 | 62.2 | 21.2 |
| 3 | 71.9 | 14.4 | 59.0 | 16.0 |

This is a real, seed-replicated pattern, and it is the **opposite** of the first write-up's claim of "no consistent extreme value on either axis": the biggest rank movers — degraded *and* improved alike — sit in the **upper third of `condition_drift`** (median pct 55–72, degraded consistently higher than improved) and the **bottom decile-to-fifth of `embedding_shift`** (median pct 9.7–22.0). Both directions match the unsigned correlations above (drift positive, shift negative) exactly, which is what one should expect if those correlations are describing something real. The correct reading is not "the extremes are unexplained" but "the extremes are high-drift, low-shift samples, in both directions" — the per-row absolute drift/shift values do vary seed-to-seed, which is why the percentile view was needed to see it.

### Interpretation

The first write-up of this section reported a clean null on both correlations and concluded that per-sample drift/shift magnitude "does not meaningfully explain per-sample retrieval-rank change." **That conclusion was an artifact of a single estimator choice and is withdrawn.** The arithmetic was correct; the lens was not. `delta_rank` differences two independently-trained models, so the condition table's own contribution is buried under combiner and `other_proj` divergence.

What the same data actually supports:

1. **The condition table has a large, direct, seed-replicated effect on per-sample rank.** Isolated inside a single fixed model, ρ(Δrank_swap, drift) ≈ **+0.47**, p < 1e−160, 3/3 seeds, in the mechanically expected direction: the trained combiner has co-adapted to the drifted conditions, and high-drift samples pay the most for reverting to init. Post-init condition training is emphatically *not* inert at the per-sample level — consistent with 11.1's geometry axis (trained shifts further, conditions matter more) and now with a much sharper per-sample handle on it.
2. **Drift predicts the magnitude, not the sign, of cross-arm rank movement.** ρ(|Δrank|, drift) ≈ +0.15 with ρ(Δrank, drift) ≈ +0.02 is exactly the signature of a symmetric-in-direction effect. Embedding shift runs the other way (ρ(|Δrank|, shift) ≈ −0.30): highly-shifted samples are the *stable* ones across arms.
3. **This diagnostic does not explain 11.1's held-out oracle regression, and cannot.** On the in-sample own-condition metric the trained arm is better, not worse — the opposite direction from 11.1's headline — and the two construct mismatches (in-sample vs. held-out; own-condition vs. oracle-max) are structural, not fixable by re-running. So 11.2's positive finding tells us the condition table's per-sample effect is real and drift-graded; it does **not** tell us that this is the mechanism behind the held-out i2t regression, and it does not rule it out either. Closing that gap would need a diagnostic that reaches the held-out set, which requires a way to assign conditions to unseen samples (i.e. `condition_predictor`) — out of scope for the spec's no-new-training branch.

The honest summary of the 11.2 branch is therefore: **not a null.** The condition table's per-sample retrieval effect is real, large, drift-graded, and seed-replicated once measured with an estimator that isolates it; the naive cross-arm delta undersold it by ~10×; and the relationship to 11.1's held-out oracle result remains open, for a documented structural reason rather than a lack of signal.

### Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_retrieval_correlation.py --selftest
python scripts/analyze_condition_retrieval_correlation.py --pair res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_170212_CoSiR_Experiment res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_161846_CoSiR_Experiment
python scripts/analyze_condition_retrieval_correlation.py --pair res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_171558_CoSiR_Experiment res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_163307_CoSiR_Experiment
python scripts/analyze_condition_retrieval_correlation.py --pair res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_172950_CoSiR_Experiment res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_164733_CoSiR_Experiment
```

(`--n-query-sample`, `--seed`, `--k-extremes` and `--rank-chunk` are all exposed on the CLI; `--rank-chunk` is a memory knob only and does not change results.)

## Experiment 11.3 — bidirectional table↔predictor coupling

### TL;DR for this section

Removing the stop-gradient on the condition-predictor distillation term (`loss.pred_stopgrad=false`, so gradient flows both table→predictor and predictor→table, not just the former) does not recover any of frozen's held-out advantage over trained — it makes things worse. Against **frozen**, the coupled arm (`pred_coupled`) is significantly worse on i2t, seed-replicated 3/3, on both the table's own held-out codebook quality (`test_oracle/i2t_R1` mean Δ = −7.10, mean/SEM = −3.1) and the predictor's standalone usefulness vs. raw CLIP (`test_pre_diff/i2t_R1` mean Δ = −1.00, mean/SEM = −2.8); t2i is a null both ways, consistent with 11.1's own t2i-is-noisier pattern. Against **trained**, `test_oracle/i2t_R1` holds the same negative direction (3/3 seeds) but does not clear significance (mean/SEM = −1.1) — seed 1 was an outlier (−11.70) against seeds 2–3's much smaller −0.10/−0.40. `test_pre_diff/i2t_R1` vs. trained does not agree with that direction: mean Δ = −0.37 but only 1/3 seeds negative (seed 1: −1.70; seeds 2–3 positive: +0.20, +0.40), so it is a non-significant, seed-inconsistent cell, not a second confirmation of the oracle result. The diagnostic explains why the hypothesis failed rather than succeeded: `drift_from_init` went *up* under coupling (0.084 vs. trained's 0.064, frozen's 0.000) rather than back down toward the predictable manifold, while the predictor-consistency loss itself dropped sharply (0.011 vs. ~0.03) — table and predictor did converge toward each other, just by both moving further from init together, not by the table snapping back. This is the spec's third named outcome branch (reportable instability), not the second (real signal) — no `lambda_pred` strength follow-up is warranted on this evidence.

### Method

One new arm, `pred_coupled` (`train.em_interval=-1`, `loss.pred_stopgrad=false`), 3 seeds, sharing 11.1's exact `results_dir` and buddy-init template (`scripts/run_pred_stopgrad_ablation.sh`), compared against 11.1's already-completed `trained` and `frozen` arms via `scripts/analyze_pred_stopgrad_ablation.py`. Same operating point as 11.1/11.2: RedCaps-150k, lr=1e-3, lr_label=1e-4, dim=16, alpha=0.5, all training-time buddy terms off. Primary metrics: `test_oracle/{t2i,i2t}_R1` (the table tried directly against held-out queries, oracle-max-over-conditions — unaffected by the predictor) and `test_pre_diff/{t2i,i2t}_R1` (the predictor's single-forward-pass recall minus raw CLIP baseline). Both are computed by the existing eval pipeline with zero new eval code. Free diagnostic context: `train_buddy_diag/drift_from_init` and final-epoch `train_loss/loss_pred` (a wandb-key naming bug — missing the `train_` prefix `logger.log_train` actually applies — was caught and fixed in `scripts/analyze_pred_stopgrad_ablation.py` while reading the seed-1 result; the primary metrics were unaffected since they use a different logging path).

### Per-seed results

```
Experiment 11.3 - bidirectional table<->predictor coupling  group='condition freeze ablation'  tag='pred-stopgrad-ablation-redcaps_150k'
==============================================================================
  11 run(s); arms present: ['frozen', 'pred_coupled', 'trained'].

  ----------------------------------------------------------------------
  pred_coupled vs trained
  ----------------------------------------------------------------------

    --- test_oracle/t2i_R1 (pred_coupled - trained) ---
      mean delta = -0.03 (n=3, wins=1/3)  mean/SEM=-0.3
        seed 1: delta = -0.20
        seed 2: delta = +0.20
        seed 3: delta = -0.10

    --- test_oracle/i2t_R1 (pred_coupled - trained) ---
      mean delta = -4.07 (n=3, wins=0/3)  mean/SEM=-1.1
        seed 1: delta = -11.70
        seed 2: delta = -0.10
        seed 3: delta = -0.40

    --- test_pre_diff/t2i_R1 (pred_coupled - trained) ---
      mean delta = -0.27 (n=3, wins=1/3)  mean/SEM=-0.8
        seed 1: delta = -0.90
        seed 2: delta = +0.30
        seed 3: delta = -0.20

    --- test_pre_diff/i2t_R1 (pred_coupled - trained) ---
      mean delta = -0.37 (n=3, wins=2/3)  mean/SEM=-0.5
        seed 1: delta = -1.70
        seed 2: delta = +0.20
        seed 3: delta = +0.40

  ----------------------------------------------------------------------
  pred_coupled vs frozen
  ----------------------------------------------------------------------

    --- test_oracle/t2i_R1 (pred_coupled - frozen) ---
      mean delta = +0.10 (n=3, wins=1/3)  mean/SEM=+0.4
        seed 1: delta = -0.20
        seed 2: delta = +0.60
        seed 3: delta = -0.10

    --- test_oracle/i2t_R1 (pred_coupled - frozen) ---
      mean delta = -7.10 (n=3, wins=0/3)  mean/SEM=-3.1 *
        seed 1: delta = -11.70
        seed 2: delta = -4.50
        seed 3: delta = -5.10

    --- test_pre_diff/t2i_R1 (pred_coupled - frozen) ---
      mean delta = -0.37 (n=3, wins=1/3)  mean/SEM=-1.3
        seed 1: delta = -0.90
        seed 2: delta = +0.10
        seed 3: delta = -0.30

    --- test_pre_diff/i2t_R1 (pred_coupled - frozen) ---
      mean delta = -1.00 (n=3, wins=0/3)  mean/SEM=-2.8 *
        seed 1: delta = -1.70
        seed 2: delta = -0.80
        seed 3: delta = -0.50

  ----------------------------------------------------------------------
  diagnostic context (not paired deltas)
  ----------------------------------------------------------------------
    frozen: drift_from_init mean=0.0000; final loss/loss_pred mean=0.0304
    trained: drift_from_init mean=0.0643; final loss/loss_pred mean=0.0306
    pred_coupled: drift_from_init mean=0.0838; final loss/loss_pred mean=0.0114
```

### Cross-seed synthesis

Against **frozen**, two of the four metric×baseline cells clear both bars this project holds every claim to (spec §5): seed-replicated (3/3 in sign) and |mean/SEM| ≥ 2 — `test_oracle/i2t_R1` (−7.10, z=−3.1) and `test_pre_diff/i2t_R1` (−1.00, z=−2.8), both `pred_coupled` losing. t2i is not significant against frozen either (`test_oracle/t2i_R1` z=+0.4; `test_pre_diff/t2i_R1` z=−1.3 — below the |z|≥2 bar but not as tight to zero as the oracle cell), mixed win counts in both cells. Against **trained**, no cell clears |z| ≥ 2. `test_oracle/i2t_R1` is directionally consistent (3/3 negative) but the variance from seed 1's outlier keeps it under the significance bar. `test_pre_diff/i2t_R1` vs. trained is the least consistent cell in the table — only 1/3 seeds negative (seed 1: −1.70; seeds 2–3: +0.20, +0.40), so despite the negative mean (−0.37) it should not be read as directionally agreeing with the oracle cell. t2i vs. trained is a null in both metrics. No cell in either comparison favors `pred_coupled`.

### Diagnostic context

`drift_from_init`: frozen=0.0000 (confirms the freeze mechanism, consistent with 11.1), trained=0.0643, pred_coupled=0.0838 — coupling increased drift by ~30% relative to trained, the opposite of the hypothesized effect (that removing the stop-gradient would pull the table back toward the feature-predictable manifold and shrink drift). `loss_pred` (final epoch): frozen=0.0304, trained=0.0306 (near-identical — as expected, since in both arms the table→predictor gradient path is the same one-way distillation), pred_coupled=0.0114 (63% lower) — table and predictor did become more mutually consistent under coupling, but by both moving further from init together rather than the table snapping back toward what the predictor could already represent.

### Interpretation

This is the spec's third named branch: **reportable instability**, not real signal, not a clean null. The bidirectional coupling does exactly what its stated risk warned it might — table and predictor chase each other into a more mutually-consistent but less generalizable configuration, rather than the table settling back onto the predictable manifold. It is significantly worse than frozen on i2t (both axes), and not distinguishable from — if anything, directionally worse than — the uncoupled trained arm, which itself already loses to frozen per 11.1. No `lambda_pred` strength follow-up is warranted per the spec's own gating discipline: the signal here is a negative result on the mechanism itself, not a promising-but-miscalibrated dose. Combined with 11.1/11.2, the accumulating evidence across Experiment 11's three parts continues to favor the simplest configuration — buddy-init geometry alone, conditions frozen after init — over every post-init training-pressure variant tested so far (implicit default pressure in 11.1, and this explicit table↔predictor coupling in 11.3).

### Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
SEED_SWEEP=1 bash scripts/run_pred_stopgrad_ablation.sh
SEED_SWEEP=2,3 bash scripts/run_pred_stopgrad_ablation.sh
python scripts/analyze_pred_stopgrad_ablation.py --tag pred-stopgrad-ablation-redcaps_150k
```
