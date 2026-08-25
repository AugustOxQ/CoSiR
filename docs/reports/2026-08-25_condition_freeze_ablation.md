# Does post-init training of the conditions do anything? (Experiment 11.1)

**Date:** 2026-08-25 · **Dataset:** RedCaps, 150,000 rows of `redcaps_train.json` (matches C5/C6/C7/C8's scale) · **Branch:** `experiment/buddy_init_ablation2`
**Code:** `scripts/run_condition_freeze_ablation.sh`, `scripts/analyze_condition_freeze_ablation.py`, `scripts/analyze_condition_geometry.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 11.1

---

## TL;DR

Holding buddy-init geometry, the frozen CLIP backbone, and every hyperparameter identical, this experiment asks whether letting the per-sample condition vectors keep training after buddy-init (today's default) does anything, versus freezing them at their init value for the whole run (`em_interval` set past the epoch budget so the "network" phase never ends). **The answer is a real, seed-replicated, direction-asymmetric effect that is not a clean null on either axis:**

- **Retrieval — t2i is a noise-floor null, i2t is a large, decisive, and counter-intuitive win for freezing.** `test_oracle/t2i_R1`: mean Δ (frozen − trained) = **−0.27** (mean/SEM = −2.0, statistically flagged but the magnitude sits inside the ~0.1–0.7 R1 noise floor — the same category as C7/C8's flagged-but-tiny t2i exceptions). `test_oracle/i2t_R1`: mean Δ (frozen − trained) = **+4.67** (mean/SEM = **+32.1**, 3/3 seeds, magnitude nearly an order of magnitude above the noise floor's upper edge). **Freezing conditions after buddy-init beats today's default (continued training) on i2t retrieval, by a wide and seed-replicated margin.**
- **Geometry — a real, growing divergence between arms, corroborating the retrieval finding rather than contradicting it.** The frozen arm's `drift_from_init` is exactly 0 for all 3 seeds (sanity check: the freeze took effect as designed); the trained arm's conditions drift by mean ≈0.0845 (range 0.0834–0.0855) over the full 100-epoch run — small in absolute terms but nonzero and consistent. The paired geometry diagnostic shows the two arms' conditioned-embedding spaces increasingly diverge over training: `conditioned_effective_dims` (PCA effective dimensionality of the combiner output) is essentially tied at epoch 0 (as expected, both arms start from the same init) but the trained arm ends up with **12–49 fewer effective dimensions than the frozen arm by epoch 99** (seed 1: −12, seed 2: −49, seed 3: −33), and the two arms' most-changed-sample sets — which 20 of 150,000 samples get shifted the most by conditioning — overlap by only **Jaccard 0.08–0.11 at epoch 99** (down from 1.00 at epoch 0), despite similar bulk shift magnitudes (`shift_mean` differs by only +0.007 to +0.018 between arms at epoch 99, small relative to the ~1.58–1.60 absolute shift_mean each arm reaches). Row/col diversity on the condition-vs-text grid diagnostic also diverges modestly and consistently in the trained arm's favor (row_div B−A ranges +0.0020 to +0.0150 across seeds at epoch 99 — trained arm shows *higher* row diversity, i.e. conditions are less interchangeable under training).

**Bottom line: post-init training of the conditions is not inert.** It reshapes which samples the combiner treats as most/least distinguishable (low most-changed-set overlap, diverging effective dimensionality) and it has a large, consistent, seed-replicated effect on i2t retrieval — but in the *opposite* direction from what "training should help" would predict: **freezing is better for i2t, training is (marginally, within-noise) better for t2i.** Per the spec's decision rule, **this clears the gate for Experiment 11.2 on both axes** — retrieval (i2t exceeds the noise floor by a wide margin, seed-replicated) and geometry (diverging effective dims, low most-changed-set overlap, corroborating rather than just co-occurring with the retrieval effect).

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
- `scripts/analyze_condition_geometry.py --exp-dir <run>` (once per run) then `--compare <frozen> <trained>` (once per seed) — per-epoch `shift_mean`/`shift_std` (1 − cos(combined embedding, text embedding)), PCA effective dimensionality of the conditioned vs. unconditioned embedding space, most/least-changed-sample ranking and its Jaccard overlap between arms, correlation of shift against condition norm and buddy-graph degree, and a condition-vs-text cross-grid diversity split (row diversity = per-text spread across conditions, col diversity = per-condition spread across texts).

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
| trained | 1 | 1.5961 | 0.1052 | 270 | 309 | 0.0171 | 0.0290 |
| trained | 2 | 1.5954 | 0.1045 | 239 | 309 | 0.0267 | 0.0311 |
| trained | 3 | 1.5961 | 0.1064 | 276 | 309 | 0.0126 | 0.0310 |
| frozen | 1 | 1.5851 | 0.1009 | 282 | 309 | 0.0151 | 0.0296 |
| frozen | 2 | 1.5883 | 0.0980 | 288 | 309 | 0.0118 | 0.0271 |
| frozen | 3 | 1.5777 | 0.1034 | 309 | 309 | 0.0041 | 0.0295 |

(`unconditioned_effective_dims` = 309 for all 6 runs, as expected — it depends only on the fixed CLIP text features, not on the arm. `condition_effective_dims`, the raw condition vectors' own PCA dimensionality, is 11 for all 6 runs at epoch 99, unsurprising since the buddy-init spectral embedding itself is 16-D and both arms share the same init.)

At epoch 0 (before any post-init training step has run), all 6 runs' `shift_mean`/`conditioned_effective_dims`/`row_div`/`col_div` agree within each matched seed (as expected — both arms literally share the same buddy-init condition table and the same freshly-initialized combiner at epoch 0). The divergence above is entirely a product of subsequent training, and grows monotonically-ish across the run (see per-epoch tables below).

### Paired `--compare <frozen> <trained>` output, all 3 seeds, full per-epoch trajectory

All values are `B − A` where `A = frozen`, `B = trained` — i.e. positive `shift_mean B−A` means the trained arm shifts *more* than the frozen arm at that epoch; positive `eff_dims B−A` means trained has more effective dimensions than frozen.

**Seed 1** (`compare 20260825_170212 (frozen) 20260825_161846 (trained)`):

```
  epoch    0: shift_mean B-A=-0.0000  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=1.00  row_div B-A=-0.0000  col_div B-A=+0.0000
  epoch   10: shift_mean B-A=-0.0024  eff_dims B-A=+1  most-changed-set Jaccard(A,B)=0.54  row_div B-A=+0.0056  col_div B-A=+0.0105
  epoch   20: shift_mean B-A=+0.0061  eff_dims B-A=+5  most-changed-set Jaccard(A,B)=0.33  row_div B-A=-0.0017  col_div B-A=-0.0016
  epoch   30: shift_mean B-A=+0.0059  eff_dims B-A=-12  most-changed-set Jaccard(A,B)=0.21  row_div B-A=+0.0010  col_div B-A=+0.0069
  epoch   40: shift_mean B-A=+0.0142  eff_dims B-A=-32  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0068  col_div B-A=+0.0045
  epoch   50: shift_mean B-A=+0.0119  eff_dims B-A=-8   most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0038  col_div B-A=+0.0006
  epoch   60: shift_mean B-A=+0.0132  eff_dims B-A=-26  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0053  col_div B-A=-0.0019
  epoch   70: shift_mean B-A=+0.0096  eff_dims B-A=-8   most-changed-set Jaccard(A,B)=0.08  row_div B-A=+0.0024  col_div B-A=-0.0008
  epoch   80: shift_mean B-A=+0.0062  eff_dims B-A=-9   most-changed-set Jaccard(A,B)=0.08  row_div B-A=+0.0006  col_div B-A=-0.0002
  epoch   90: shift_mean B-A=+0.0100  eff_dims B-A=-12  most-changed-set Jaccard(A,B)=0.08  row_div B-A=+0.0024  col_div B-A=-0.0001
  epoch   99: shift_mean B-A=+0.0111  eff_dims B-A=-12  most-changed-set Jaccard(A,B)=0.08  row_div B-A=+0.0020  col_div B-A=-0.0006
```

**Seed 2** (`compare 20260825_171558 (frozen) 20260825_163307 (trained)`):

```
  epoch    0: shift_mean B-A=-0.0000  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=1.00  row_div B-A=+0.0000  col_div B-A=-0.0000
  epoch   10: shift_mean B-A=+0.0029  eff_dims B-A=+2  most-changed-set Jaccard(A,B)=0.82  row_div B-A=+0.0092  col_div B-A=+0.0093
  epoch   20: shift_mean B-A=+0.0031  eff_dims B-A=+6  most-changed-set Jaccard(A,B)=0.29  row_div B-A=-0.0070  col_div B-A=-0.0069
  epoch   30: shift_mean B-A=+0.0059  eff_dims B-A=-24  most-changed-set Jaccard(A,B)=0.25  row_div B-A=+0.0038  col_div B-A=+0.0092
  epoch   40: shift_mean B-A=+0.0208  eff_dims B-A=-40  most-changed-set Jaccard(A,B)=0.21  row_div B-A=+0.0078  col_div B-A=+0.0009
  epoch   50: shift_mean B-A=+0.0114  eff_dims B-A=-53  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0181  col_div B-A=+0.0071
  epoch   60: shift_mean B-A=+0.0117  eff_dims B-A=-50  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0133  col_div B-A=+0.0050
  epoch   70: shift_mean B-A=+0.0153  eff_dims B-A=-55  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0140  col_div B-A=+0.0027
  epoch   80: shift_mean B-A=+0.0106  eff_dims B-A=-49  most-changed-set Jaccard(A,B)=0.05  row_div B-A=+0.0144  col_div B-A=+0.0030
  epoch   90: shift_mean B-A=+0.0062  eff_dims B-A=-49  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0151  col_div B-A=+0.0042
  epoch   99: shift_mean B-A=+0.0071  eff_dims B-A=-49  most-changed-set Jaccard(A,B)=0.08  row_div B-A=+0.0150  col_div B-A=+0.0039
```

**Seed 3** (`compare 20260825_172950 (frozen) 20260825_164733 (trained)`):

```
  epoch    0: shift_mean B-A=-0.0000  eff_dims B-A=+0  most-changed-set Jaccard(A,B)=1.00  row_div B-A=-0.0000  col_div B-A=+0.0000
  epoch   10: shift_mean B-A=-0.0050  eff_dims B-A=+3  most-changed-set Jaccard(A,B)=0.48  row_div B-A=+0.0106  col_div B-A=+0.0240
  epoch   20: shift_mean B-A=+0.0007  eff_dims B-A=+3  most-changed-set Jaccard(A,B)=0.25  row_div B-A=-0.0047  col_div B-A=-0.0022
  epoch   30: shift_mean B-A=+0.0181  eff_dims B-A=-18  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0036  col_div B-A=+0.0047
  epoch   40: shift_mean B-A=+0.0129  eff_dims B-A=-21  most-changed-set Jaccard(A,B)=0.33  row_div B-A=+0.0005  col_div B-A=-0.0013
  epoch   50: shift_mean B-A=+0.0180  eff_dims B-A=-23  most-changed-set Jaccard(A,B)=0.18  row_div B-A=+0.0058  col_div B-A=+0.0063
  epoch   60: shift_mean B-A=+0.0156  eff_dims B-A=-32  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0066  col_div B-A=+0.0060
  epoch   70: shift_mean B-A=+0.0229  eff_dims B-A=-37  most-changed-set Jaccard(A,B)=0.08  row_div B-A=+0.0098  col_div B-A=+0.0009
  epoch   80: shift_mean B-A=+0.0163  eff_dims B-A=-33  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0088  col_div B-A=+0.0039
  epoch   90: shift_mean B-A=+0.0182  eff_dims B-A=-34  most-changed-set Jaccard(A,B)=0.14  row_div B-A=+0.0091  col_div B-A=+0.0021
  epoch   99: shift_mean B-A=+0.0184  eff_dims B-A=-33  most-changed-set Jaccard(A,B)=0.11  row_div B-A=+0.0084  col_div B-A=+0.0015
```

**Read:**
- **`eff_dims` diverges consistently and, for 2 of 3 seeds, substantially.** After an initial small positive blip (epochs 10–20, where trained briefly has *more* effective dims than frozen — plausibly noise from the earliest post-init training steps), `eff_dims B−A` goes and stays negative from epoch 30 onward in all 3 seeds, settling at epoch 99 at −12 (seed 1), −49 (seed 2), and −33 (seed 3). Continued training collapses the conditioned embedding space into fewer effective dimensions than the frozen arm retains, by a magnitude that varies noticeably across seeds (4× difference between seed 1 and seed 2) but never reverses sign after epoch 20.
- **Most-changed-set Jaccard overlap decays from 1.00 (identical, at epoch 0 by construction) to 0.08–0.11 by epoch 99** across all 3 seeds (0.08, 0.08, 0.11 — recomputed directly from each run's `ranked.most_changed` sample-ID lists to confirm) — the specific 20 samples the combiner treats as most reshaped by conditioning become almost entirely different sets between the two arms. (Mid-training epochs show more scatter — the trajectory tables above show values up to 0.54 near epoch 10 and occasional local bumps to 0.14–0.33 at various mid-training epochs per seed — but the epoch-99 endpoint is consistently low and narrowly banded.) This is a genuine identity-level divergence, not just a shift in aggregate statistics.
- **`shift_mean` itself diverges only mildly in absolute terms** (final deltas +0.0111, +0.0071, +0.0184 — roughly 0.4–1.2% of each arm's own ~1.58–1.60 absolute shift_mean) but is **consistently positive from epoch 20 onward in all 3 seeds** (with one seed showing a small negative blip at epoch 10) — a small, growing, seed-consistent effect, not noise scattered around zero.
- **Row diversity (per-text spread across conditions) is consistently higher for trained than frozen from epoch 30 onward** (row_div B−A positive in all 3 seeds at epoch 99: +0.0020, +0.0150, +0.0084) — training makes conditions *less* interchangeable for a given text than freezing does, the opposite of the "conditions become null/interchangeable" failure mode. Col diversity (per-condition spread across texts) shows a smaller, less consistent split (seed 1 slightly negative at epoch 99, seeds 2–3 positive) — no clear collapse-to-a-dominant-condition failure mode in either arm.

## Interpreting the two grid failure modes

The condition-vs-text grid diagnostic's two diversity axes are diagnostic of two different failure modes: **low row diversity** (for a fixed text, the combiner output barely changes across different conditions) means conditions are functionally null/interchangeable — the combiner is ignoring them; **low col diversity** (for a fixed condition, the combiner output barely changes across different texts) means that one condition dominates and collapses every input toward the same output. Neither arm shows evidence of either failure mode in absolute terms: row_div means at epoch 99 range 0.0041–0.0267 and col_div means range 0.0271–0.0311 across all 6 runs — both nonzero and neither near zero relative to the other diversity axis, i.e. conditions are neither fully ignored nor fully dominant in either arm. The between-arm *difference* (trained showing modestly higher row diversity than frozen) is a real, seed-consistent effect but a second-order one on top of both arms being healthy on this diagnostic — it does not indicate either arm has collapsed.

## Correlation diagnostics (context, not part of the decision rule)

`shift_vs_condition_norm` and `shift_vs_buddy_degree` Pearson correlations (from each `--exp-dir` run's epoch-99 snapshot) are small in both arms and both directions: `shift_vs_condition_norm` r ranges from −0.161 (epoch 0, shared init) to −0.021…−0.109 (epoch 99, arm-dependent — trained arm's r moves closer to 0 than frozen's, e.g. seed 1: trained r=−0.041 vs. frozen r=−0.109); `shift_vs_buddy_degree` r ranges from +0.164 (epoch 0) to +0.215…+0.224 (trained, epoch 99) vs. +0.182…+0.191 (frozen, epoch 99) — a small positive relationship between a sample's buddy-graph degree and how much conditioning shifts it, marginally stronger under continued training. Neither correlation is large enough (|r| < 0.25 throughout) to be a primary driver of the retrieval or effective-dims results above; reported here as context per the spec's diagnostic list, not as a load-bearing finding.

## Applying the decision rule

Per spec §4 Experiment 11.1's success criteria: **no real difference on either axis** (retrieval within noise floor AND no meaningful geometry divergence) → clean simplification result, no 11.2 needed. **A real difference on either axis** (retrieval mean/SEM ≥ 2 exceeding the noise floor in either direction, OR a clear geometry divergence — large/growing `shift_mean` deltas, diverging `eff_dims`, or low most-changed-set Jaccard overlap — even with null retrieval) → 11.2 is gated open.

**The actual result triggers the gate-open branch decisively, and on both axes:**

- **Retrieval axis triggers on its own, unambiguously.** i2t's mean Δ = +4.67 R1, mean/SEM = +32.1, 3/3 seeds — this clears both the statistical bar (`mean/SEM ≥ 2`) and the noise-floor bar (0.1–0.7 R1) by a wide margin (the effect is ~6.7–47× the noise floor's width), with 3/3 seed agreement in both sign and rough magnitude (+4.90, +4.40, +4.70). This alone is sufficient to gate 11.2 open.
- **Geometry axis independently corroborates a real divergence, not just co-occurs with the retrieval finding.** `eff_dims` diverges consistently in sign from epoch 30 onward across all 3 seeds (never reversing after the initial epoch 10–20 settling period), reaching −12 to −49 effective dimensions by epoch 99 — a diverging trajectory, not a stable one. Most-changed-set Jaccard overlap decays from 1.00 to 0.08–0.11 by epoch 99, meaning the two arms disagree almost completely on which specific samples are most reshaped by conditioning, despite similar bulk shift magnitudes. This satisfies the decision rule's "diverging `eff_dims`, or low most-changed-set Jaccard overlap" clause independently of the retrieval result.
- **t2i is the one part of this result that is a clean null** — within the noise floor, consistent with the pattern established by C7 and C8's t2i-null findings elsewhere in this project's line.

**Conclusion: Experiment 11.2 is gated open, triggered by both the retrieval axis (i2t, decisively) and the geometry axis (diverging effective dims and most-changed-set identity).** This is *not* the "training pressure on conditions is inert, drop it" outcome the spec's null branch would have produced — post-init training measurably reshapes the condition-conditioned embedding space and has a large, direction-specific retrieval effect, in the *opposite* direction from a naive "more training should help" prior.

**Which of 11.2's two branches this routes to:** spec §4 Experiment 11.2 branches explicitly on direction — *"if trained beats frozen (on retrieval, geometry, or both): ablate the loss-stack terms..."* (new training runs) vs. *"if frozen beats trained, the result is mixed, or the divergence is geometry-only (retrieval null): extend the geometry diagnostic rather than launch new training..."* (no new training, reuses this task's own checkpoints). The actual result is **frozen beating trained on i2t** (and a t2i null) — on no axis does trained beat frozen — so this is the spec's **second branch, by its literal wording**, not the first: extend the 11.1 geometry diagnostic (correlate per-sample condition drift `‖z_i − z_init,i‖`, already loggable from the trained arm's checkpoints, and per-sample embedding shift, already computed by `analyze_condition_geometry.py`, against per-sample retrieval outcome) rather than launching a new loss-stack-term ablation sweep. This is also the cheaper of the two branches — near-zero additional compute, reusing this task's own 6 runs' checkpoints and `condition_geometry/summary.json` files, no new training required.

## Caveats

- **RedCaps-150k only**, one operating point (`lr=1e-3, lr_label=1e-4, embedding_dim=16, alpha=0.5`, all training-time buddy terms off) — same scope discipline as every prior experiment in this line; not checked at 300k or under a different operating point.
- **Only `test_oracle` retrieval is reported here, not `test_pre_diff` (gap to CLIP).** Whether the i2t regression under continued training also affects the realistic single-forward-pass predictor comparison against raw CLIP is untested here — a natural, cheap follow-up given C6/C8's precedent for this exact comparison.
- **`shift_mean`'s absolute divergence between arms is small relative to each arm's own shift_mean** (deltas of 0.4–1.2% of ~1.6 absolute shift), even though it is consistently signed and growing after epoch 20 across all 3 seeds. The `eff_dims` and Jaccard-overlap divergences are the stronger, more decisive geometry signals in this result; `shift_mean`'s own divergence should be read as corroborating, not as independently decisive on its own.
- **Correlations against condition norm and buddy degree are weak (|r| < 0.25) in both arms** — neither is a strong explanatory variable for which samples get shifted most, in either arm; this diagnostic did not surface an obvious mechanistic story for *why* continued training regresses i2t, which is exactly the question 11.2 is now scoped to answer.
- **A one-line bug fix was applied to `scripts/analyze_condition_freeze_ablation.py`** during this analysis (see Method above) — the wandb summary key for drift was wrong, silently suppressing (not corrupting) the drift-sanity-check lines. This did not affect the retrieval deltas, which use a different, already-correct set of keys.
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
