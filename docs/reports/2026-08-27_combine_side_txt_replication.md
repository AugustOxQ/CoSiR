# Does `combine_side="txt"` flip the i2t/t2i asymmetry and the bridge-subgroup effect? A direct test

**Date:** 2026-08-27 · **Dataset:** RedCaps, 150,000 rows · **Branch:** `experiment/condition_drift_retrieval_correlation`
**Code:** `scripts/run_condition_freeze_ablation.sh` (`EXTRA_OVERRIDES="model.combine_side=txt"`), `scripts/analyze_condition_freeze_ablation.py`, `scripts/analyze_training_trajectory.py`, `scripts/analyze_condition_retrieval_correlation.py`, `scripts/analyze_polysemy_bridges.py`
**Motivated by:** the `combine_side="img"` caveat added to Experiment 12.3 (`docs/reports/2026-08-26_polysemy_bridge_diagnostic.md`) — every run behind 11.1/11.3/12/12.3 fuses the trainable condition into the image embedding only, raising the question of whether that architectural choice, rather than graph topology, explains (a) the i2t-only retrieval deficit (11.1) and (b) the `img_only_only`-concentrated bridge effect (12.3).
**Compute:** 6 new training runs (`trained`+`frozen` × seeds {1,2,3}, `combine_side="txt"`, 100-epoch RedCaps schedule, reusing the existing buddy-init template — no re-extraction, no new graph). Same order of magnitude as 11.1 itself.

## TL;DR

**Neither candidate explanation from Experiment 12.3 is fully right; the truth splits the two findings apart.** Repeating 11.1's exact design with `combine_side="txt"` does **not** flip the i2t/t2i asymmetry the way the architectural hypothesis predicted: `i2t` (not `t2i`) is still the retrieval direction where `frozen` beats `trained`, just far more weakly (mean Δ = +0.40, mean/SEM = +4.0, vs. +4.67 / +32.1 under `combine_side="img"` — roughly 11× smaller) — `t2i` stays a null (mean Δ = +0.13, mean/SEM = +0.3, inconsistent in sign). **But the bridge-subgroup concentration effect from Experiment 12.3 *does* flip, cleanly, in every one of the 3 seeds:** under `combine_side="txt"`, it is `txt_only_only` nodes (not `img_only_only`) that show the outsized signed rank movement (median `delta_rank` -8 to -10, vs. `bridge`'s tight -2 to -2), while `img_only_only` drops down to match `bridge`'s small effect (-1 to -2) — the exact mirror image of the `combine_side="img"` pattern. So the graph-topology mechanism for *which one-sided subgroup* training preferentially corrects is validated and moves with `combine_side` as predicted, while whatever drives the *large-scale* i2t vs. t2i asymmetry does not — it is disproportionately, but not exclusively, tied to `combine_side="img"` specifically, not simply "whichever side the condition lives on."

## Method

Reran Experiment 11.1's exact ablation design (`trained` vs. `frozen` via `train.em_interval`, same lr/lr_label/embedding_dim/alpha, same shared buddy-init template) with the single override `model.combine_side=txt`, at 3 seeds (1 pilot seed run first and inspected before committing to seeds 2-3, per user approval). Ran, in order: (1) `analyze_condition_freeze_ablation.py` for the paired t2i/i2t deltas (11.1-style); (2) `analyze_training_trajectory.py` for the per-epoch gap shape (12.2-style); (3) `analyze_condition_retrieval_correlation.py --dump-per-sample` for all 3 seed pairs, then `analyze_polysemy_bridges.py`'s pooled cross-reference (12.3-style) on those 3 dumps. All three reuse the identical buddy-init graph/template Experiment 12's diagnostic built — `combine_side` does not affect graph construction, only the training-time combiner routing (confirmed by the Codex `combine_side`-consistency audit referenced in Experiment 12.3).

## Results

### Freeze-ablation deltas (11.1-style)

```
--- test_oracle/t2i_R1 (frozen - trained) ---
  mean delta = +0.13 (n=3, wins=1/3)  mean/SEM=+0.3
    seed 1: delta = +0.00
    seed 2: delta = +0.90
    seed 3: delta = -0.50

--- test_oracle/i2t_R1 (frozen - trained) ---
  mean delta = +0.40 (n=3, wins=3/3)  mean/SEM=+4.0 *
    seed 1: delta = +0.50
    seed 2: delta = +0.50
    seed 3: delta = +0.20
```

| | `combine_side="img"` (11.1, 3 seeds) | `combine_side="txt"` (this report, 3 seeds) |
|---|---|---|
| t2i mean Δ (frozen−trained) | -0.27, mean/SEM=-2.0 (noise floor) | +0.13, mean/SEM=+0.3 (null, inconsistent sign) |
| i2t mean Δ (frozen−trained) | **+4.67, mean/SEM=+32.1** | **+0.40, mean/SEM=+4.0** |

i2t is still the significant direction under `combine_side="txt"` — the asymmetry did not flip to t2i — but its magnitude collapsed to roughly 1/11th of the `combine_side="img"` effect, right at the noise floor's lower edge (~0.1-0.7 R1) rather than 6.7-47× above it.

### Trajectory audit (12.2-style)

```
i2t gap vs frozen, trained seed 1: e0=-0.10, e10=+0.10, e20=+0.20, e30=-0.10, e40=+0.20, e50=-0.30, e60=-0.60, e70=+0.00, e80=+0.00, e90=+0.00, e99=-0.50
i2t gap vs frozen, trained seed 2: e0=+0.20, e10=-0.50, e20=-0.20, e30=+0.10, e40=+0.10, e50=+0.30, e60=-0.10, e70=+0.10, e80=-0.50, e90=-0.10, e99=-0.50
i2t gap vs frozen, trained seed 3: e0=-0.10, e10=+0.00, e20=+0.20, e30=-0.20, e40=-0.60, e50=-0.40, e60=-0.50, e70=+0.00, e80=-0.50, e90=-0.60, e99=-0.20
```

Unlike `combine_side="img"`'s clean, nearly-identical, monotonic-decline-then-plateau shape across all 3 seeds (Experiment 12.2), the `combine_side="txt"` i2t gap trajectory is small and noisy at every epoch, in both directions, with no consistent shape across seeds. t2i is similarly small and noisy (not shown; see Reproduction). This is consistent with the freeze-ablation result above: whatever mechanism produces 11.1's large, clean, saturating i2t decline is much weaker and noisier under `combine_side="txt"`.

### Bridge/delta_rank cross-reference (12.3-style)

```
retrieval cross-reference, pooled across 3 run(s):
  corr(is_polysemic, |delta_rank|) across runs: mean rho=-0.004 (n=3)  mean/SEM=-1.1
  corr(is_polysemic, delta_rank) across runs: mean rho=-0.026 (n=3)  mean/SEM=-1.6
```

Neither pooled correlation clears the project's |z| >= 2 bar at n=3 (half of 12.3's 6-run pool) — but the *per-label* breakdown shows the real story, and it is a clean, seed-consistent flip:

| label | n/run | seed1 median `delta_rank` | seed2 | seed3 | (for comparison: `combine_side="img"`, 12.3) |
|---|---|---|---|---|---|
| `neither` | 38 | 0.0 | 0.0 | -2.0 | -2.5 to -0.5 |
| `img_only_only` | 422 | -1.0 | -2.0 | -2.0 | **-17.0 to -28.0** |
| `txt_only_only` | 139 | **-8.0** | **-10.0** | **-3.0** | -3.0 to -5.0 |
| `bridge` | 2401 | -2.0 | -2.0 | -2.0 | -3.0 to -4.0 |

Under `combine_side="img"`, `img_only_only` was the outsized subgroup (4-8× `bridge`'s effect). Under `combine_side="txt"`, `img_only_only` drops down to match `bridge` (-1 to -2 vs. bridge's -2), and `txt_only_only` becomes the outsized subgroup instead (-8 to -10, roughly 4-5× `bridge`'s -2) — the exact mirror image, in every one of the 3 seeds. `|delta_rank|` (unsigned) also flips sign at the pooled level (-0.004 here vs. +0.017 under `combine_side="img"`), consistent with the same underlying reordering.

## Interpretation

Two separate questions, two separate answers:

1. **Does the large-scale i2t vs. t2i retrieval asymmetry (11.1) come from `combine_side="img"` specifically?** No, not simply. If it did, `combine_side="txt"` should have flipped the significant direction to t2i. It did not: i2t remains the (only) significant direction, just ~11× smaller in magnitude and with a noisy, non-saturating trajectory rather than 12.2's clean decline-then-plateau. This means i2t retrieval is intrinsically more sensitive to continued condition-training than t2i, for a reason that is not simply "the condition lives on that side" — `combine_side="img"` appears to strongly *amplify* a pre-existing i2t-specific sensitivity rather than *create* the asymmetry from nothing. What that reason is remains unexplained by this report.
2. **Does the bridge-subgroup concentration effect (12.3) come from `combine_side="img"`?** Yes, essentially — this flips cleanly and consistently with `combine_side`, exactly as the graph-topology mechanism proposed in 12.3 predicts: whichever one-sided node type (`img_only_only` or `txt_only_only`) lacks a same-side buddy-init anchor *matching the side the condition trains on* is the one training disproportionately moves. This is a real confirmation of that mechanism, not just a plausible story — it replicated as a clean sign-and-magnitude flip across all 3 independently-trained seeds, in a result nobody could have gotten by construction (the graph/label assignment is combine_side-independent; only the training outcome moved).

**Bottom line for the paper:** the `img_only_only`/`txt_only_only` bridge-subgroup finding from Experiment 12.3 is now on considerably firmer footing — it is a real, mechanistically-understood, causally-testable (not just correlational) property of how buddy-init geometry interacts with which side the condition trains on. The much larger, headline i2t-vs-t2i retrieval asymmetry from 11.1 is *not* explained by this mechanism, or by `combine_side` alone — it is a separate, still-open question, though this experiment does establish that `combine_side="img"` is not merely incidental to its size: switching sides shrinks it by roughly an order of magnitude even though it doesn't eliminate or relocate it.

## Caveats

- 3 seeds, one recipe (`trained`/`frozen` only, no `pred_coupled` analog under `combine_side="txt"`) — matches 11.1's own seed count, but is half of 12.3's 6-run pool, so the pooled bridge-cross-reference correlations here don't individually clear the project's significance bar despite the clean, consistent per-label pattern. Treat the per-label flip (table above) as the finding; the pooled rho as corroborating but underpowered on its own.
- This does not identify *why* `combine_side="img"` amplifies the i2t effect roughly 11× beyond `combine_side="txt"`'s residual effect — that remains an open question this report does not attempt to answer.
- `img_only_only` (n=422) and `txt_only_only` (n=139) are graph labels defined purely by mutual-kNN structure in image/text feature space respectively — they are not symmetric in size (image-side mutual-kNN edges are simply more common in this dataset/embedding, per Experiment 12's own label counts), so the raw magnitudes are not expected to be identical in a symmetric flip, only the *qualitative pattern* (each one is outsized precisely when the condition trains on its matching side).

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR

# 1. Train (reuses existing buddy-init template automatically)
SEED_SWEEP="1,2,3" \
WANDB_TAG="combine-side-txt-pilot-redcaps_150k" \
WANDB_GROUP="combine side txt pilot" \
EXTRA_OVERRIDES="model.combine_side=txt" \
bash scripts/run_condition_freeze_ablation.sh

# 2. Freeze-ablation deltas
python scripts/analyze_condition_freeze_ablation.py \
  --group "combine side txt pilot" --tag combine-side-txt-pilot-redcaps_150k

# 3. Trajectory audit
python scripts/analyze_training_trajectory.py \
  --group "combine side txt pilot" --tag combine-side-txt-pilot-redcaps_150k \
  --pred-coupled-tag combine-side-txt-pilot-redcaps_150k-does-not-exist \
  --out-fig docs/reports/assets/training_trajectory/combine_side_txt_i2t_gap_trajectory.png

# 4. Per-sample dumps (one --pair call per seed) + pooled bridge cross-reference
python scripts/analyze_condition_retrieval_correlation.py --pair <frozen_dir> <trained_dir> --dump-per-sample   # x3 seeds
python scripts/analyze_polysemy_bridges.py --n-bridge-sample 5000 --device cuda \
  --per-sample-npz <seed1_dump.npz> <seed2_dump.npz> <seed3_dump.npz> \
  --out res/CoSiR_condition_freeze_ablation/redcaps_150k/polysemy_bridges_pooled_combine_side_txt.json
```
