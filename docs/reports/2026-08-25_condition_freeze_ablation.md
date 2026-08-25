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

**Date:** 2026-08-26 · **Code:** `scripts/analyze_condition_retrieval_correlation.py` · **Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 11.2 (second, no-new-training branch)

### Method

11.1 (above) found that letting per-sample conditions keep training after buddy-init hurts i2t retrieval relative to freezing them (frozen beats trained, mean Δ = +4.67 R1, mean/SEM = +32.1, 3/3 seeds) — a large, direction-asymmetric effect that neither `conditioned_effective_dims` nor most-changed-set Jaccard could separate, and that the geometry diagnostic's own correlations (`shift_vs_condition_norm`/`shift_vs_buddy_degree`, both |r| < 0.11 at epoch 99) could not explain mechanistically. Per the spec's §4 Experiment 11.2 gate — resolved to its second, no-new-training branch ("if frozen beats trained, the result is mixed, or the divergence is geometry-only: extend the geometry diagnostic rather than launch new training") because the actual 11.1 result was frozen beating trained on i2t with a t2i null — 11.2 asks the natural next per-sample question directly, reusing only 11.1's existing checkpoints: for each of a run's 150,000 training samples, does how far its condition moved from the frozen arm's (never-moving) buddy-init value (`condition_drift`), or how much conditioning displaces its combine-side embedding (`embedding_shift`), correlate with how much worse (or better) that exact sample's own true match ranks under its trained condition vs. its frozen condition? Ranking is against the FULL 150k-sample training population's projected "other side" (text, since `combine_side=img` for all 6 runs) embeddings — not a small closed gallery — using each sample's own real, assigned condition (no oracle search over conditions, no `condition_predictor`). Full method detail (sample-ID alignment via `reorder_features_to_z`, the frozen-arm first-vs-final-epoch sanity assertion that the "frozen arm never moved" premise actually holds before trusting it as the buddy-init proxy, Spearman `rho` for both correlations) is in `scripts/analyze_condition_retrieval_correlation.py`'s module docstring and `analyze_pair`'s docstring; see also the plan at `docs/superpowers/plans/2026-08-26-condition-drift-retrieval-correlation.md`. Each seed pair was run once with the default, statistically-motivated `n_query_sample=3000` (not reduced for speed).

**Scope note on `delta_rank`'s own scale:** this diagnostic ranks each sample against the full in-sample *training* population (the same 150k rows the run trained on), not 11.1's held-out `test_oracle/i2t_R1` metric — so `delta_rank`'s aggregate sign/magnitude below is a different quantity from 11.1's test-set retrieval delta and the two should not be compared directly. The load-bearing numbers for this task's question are the two Spearman `rho` correlations, not the aggregate `delta_rank` mean/median (reported below for completeness only).

### Per-seed results

| seed | frozen dir | trained dir | delta_rank mean | delta_rank median | rho(drift) | p(drift) | rho(shift) | p(shift) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `20260825_170212` | `20260825_161846` | −143.1 | −5.0 | +0.026 | 0.162 | +0.046 | 0.0115 |
| 2 | `20260825_171558` | `20260825_163307` | −142.4 | −5.0 | +0.017 | 0.338 | +0.040 | 0.0286 |
| 3 | `20260825_172950` | `20260825_164733` | −116.3 | −5.0 | +0.009 | 0.604 | +0.046 | 0.0115 |

(all 3 runs: `n_query_sample=3000`, `n_population=150000`, `combine_side=img`; each JSON written to the trained arm's `condition_geometry/retrieval_correlation_vs_frozen.json`.)

### Cross-seed synthesis

- **`condition_drift` vs. `delta_rank`:** sign agrees across all 3 seeds (rho = +0.026, +0.017, +0.009 — all positive, i.e. more drift weakly associates with a worse relative rank) but `|rho|` never exceeds 0.03, far short of the 0.1 practical-significance bar, and `p` does not clear 0.05 in any seed (0.162, 0.338, 0.604). **Clean null** — condition-drift magnitude does not explain per-sample retrieval-rank change.
- **`embedding_shift` vs. `delta_rank`:** sign also agrees across all 3 seeds (rho = +0.046, +0.040, +0.046) and `p` clears 0.05 in all 3 (0.0115, 0.0286, 0.0115) — statistically detectable at `n=3000`, sign-consistent. But `|rho|` tops out at 0.046, well under the 0.1 bar. **Statistically significant but practically negligible** — shift magnitude explains essentially none of the per-sample rank variance.
- **Neither correlation clears the modest `|rho| > 0.1` with `p < 0.05` bar, in the same direction, across all 3 seeds.** Per-sample drift/shift magnitude does not meaningfully explain i2t's retrieval-rank degradation under continued conditioning training. Whatever drives 11.1's aggregate i2t regression, it is not simply "samples whose condition moved further, or whose embedding shifted more, rank worse" — at least not at an effect size a linear rank correlation over 3000 queries can detect.

### Qualitative extremes

Representative most-degraded rows (`delta_rank` most positive — trained ranks this sample worse than frozen):

| seed | sample_id | delta_rank | condition_drift | embedding_shift |
|---:|---:|---:|---:|---:|
| 1 | 42190 | +14202 | 0.134 | 0.316 |
| 2 | 42190 | +15446 | 0.153 | 0.285 |
| 3 | 42190 | +14216 | 0.107 | 0.186 |

Sample 42190 is the single most-degraded sample in all 3 seeds, and sample 105764 is the 2nd-most-degraded in all 3 (Δrank = +10529 / +10523 / +10245) — despite `condition_drift`/`embedding_shift` for these two samples varying substantially seed-to-seed (e.g. 42190's `embedding_shift`: 0.316 / 0.285 / 0.186), with no consistent extreme value on either axis relative to the rest of the query sample. Symmetrically, on the most-improved side, sample_ids 28404, 101763, and 99341 recur at or near the top of `most_improved` in at least 2 of 3 seeds (28404: Δrank = −36750 / −48112 in seeds 1–2, present but lower in seed 3 at −17072; 101763: −24906 / −23599 in seeds 2–3; 99341: −16487 / −10842 in seeds 2–3), again with seed-inconsistent drift/shift values. This is consistent with the correlation numbers above: *which* samples swing most is somewhat sample-specific and seed-stable, but that swing does not track `condition_drift` or `embedding_shift` magnitude — something else about those particular samples (caption content, buddy-graph position, etc., unexamined here) drives the largest per-sample rank changes, not how far their own condition moved or how much their own embedding shifted.

### Interpretation

This is a genuine, reportable null result, per the plan's own Self-Review ("if the cross-seed synthesis... comes back null on both correlations, that is itself a valid, reportable finding... not a failure requiring more scope"). 11.1 established that continued conditioning training hurts i2t retrieval by a large, seed-replicated margin, and that it measurably changes the embedding-shift distribution and grid-diversity split (also seed-replicated). 11.2 asked whether those two natural per-sample geometric proxies — how far a sample's condition moved, and how much conditioning displaces its embedding — explain *which* samples get hurt. **They do not, at any practically meaningful effect size, in any of the 3 seeds.** The retrieval regression is real and robust; the two most obvious per-sample geometric explanations for it are not the mechanism, or at least not one this linear rank-correlation lens can see. Whatever drives i2t's degradation under continued training operates at a level these two per-sample scalars do not capture — a genuine "we looked, and it isn't this" result that narrows, rather than answers, the open mechanistic question. This closes out the Experiment 11 line of investigation as scoped: the spec's second 11.2 branch (extend the geometry diagnostic, no new training) has now been run to completion, with a null result reported honestly rather than dressed up as a positive finding.

### Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python scripts/analyze_condition_retrieval_correlation.py --pair res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_170212_CoSiR_Experiment res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_161846_CoSiR_Experiment
python scripts/analyze_condition_retrieval_correlation.py --pair res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_171558_CoSiR_Experiment res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_163307_CoSiR_Experiment
python scripts/analyze_condition_retrieval_correlation.py --pair res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_172950_CoSiR_Experiment res/CoSiR_condition_freeze_ablation/redcaps_150k/20260825_164733_CoSiR_Experiment
```
