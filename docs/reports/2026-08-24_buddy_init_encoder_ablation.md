# Does the buddy graph's source encoder pair matter for downstream retrieval? (and: does cross-VLM survival predict usefulness?)

**Date:** 2026-08-24 · **Dataset:** RedCaps, 150,000 rows of `redcaps_train.json` (matches C5's scale) · **Branch:** `experiment/buddy_init_ablation`
**Code:** `scripts/run_buddy_init_encoder_ablation.sh`, `scripts/analyze_buddy_init_encoder_ablation.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 8 (this report also folds in Experiment 6's deferred question — see "Survival-rate correlation" below)
**Precursor:** `docs/reports/2026-07-16_buddy_cross_vlm_survival.md` (C3, the 16-pair cross-VLM survival study whose encoder pairs and per-pair survival rates this experiment reuses and joins against)

---

## TL;DR

`compute_buddy_init` is a pure function over raw feature arrays — nothing ties it to CLIP specifically. Every prior run fed it CLIP image/text features. This experiment swaps in each of the 16 (vision × text) encoder pairs already validated in C3's cross-VLM survival study as the buddy-graph's *init source*, holding the frozen CLIP training backbone, gated combiner, and all training-time buddy terms (off) fixed — i.e. Experiment 1's `buddies` arm with encoder-pair id as the swept axis, compared against the current default (`clip_img:clip_txt`, using CLIP for both graph modalities).

**The result is genuinely direction-asymmetric, and interesting in both directions:**

- **t2i R@1: essentially null across the board.** All 15 non-baseline pairs show small deltas, mostly under ±0.2 R1 — well within this project's established noise floor (~0.1–0.7 R1). Only `dinov2:minilm` clears the significance bar (mean/SEM = −2.0), but its actual magnitude (−0.07 R1) sits *below* the noise floor — a statistically flagged but practically negligible result, not a real effect.
- **i2t R@1: a genuinely different, striking picture.** 12 of the 15 non-baseline pairs show a **positive**, significant delta (mean/SEM ≥ 2), several substantially so (`dinov2:minilm` +2.30, `dinov2:bge` +1.97, `vit_sup:clip_txt` +1.77, `clip_img:bge` +1.43). **The current default — sourcing the buddy graph from CLIP features, the same encoder used for the training backbone — is one of the *worse* choices among the 16 tested for i2t retrieval.** Most alternatives beat it, several strongly, all seed-replicated (3/3 seed wins for every significant positive pair). Two pairs go the other way, both pairing the "e5" text encoder with a non-CLIP vision encoder: `siglip_v:e5` (−0.37, not significant) and `vit_sup:e5` (−0.80, significant negative) — but `dinov2:e5` is positive and significant (+0.93), so this is pair-specific, not "e5 is uniformly bad."
- **Survival-rate correlation (closing Experiment 6's deferred question):** a pair's C3 cross-VLM survival rate shows essentially no relationship to its t2i usefulness (r = +0.01, n = 15) and a moderate, counterintuitive **negative** relationship to its i2t usefulness (r = −0.41, n = 15) — higher graph-consensus survival trends toward *less* downstream i2t benefit, not more. This substantially answers the question Experiment 6 was scoped to investigate; see "Consequence for Experiment 6" below.

This is a real, seed-replicated, direction-asymmetric positive result: for i2t retrieval, the graph-source encoder pair matters a lot, and the current default is not the best choice.

---

## Method

**What varies:** `train.buddies.encoder_pair`, the (vision, text) encoder pair fed into `compute_buddy_init` to build the buddy graph and its spectral-embedding-derived condition-vector init. 16 pairs total (4 vision encoders `{clip_img, dinov2, siglip_v, vit_sup}` × 4 text encoders `{clip_txt, bge, e5, minilm}`), matching C3's cross-VLM survival grid exactly, sourced from the same held-out feature cache (`heldout_feats/redcaps/*.npy`) C3 used.

**What's fixed (identical to C5/C6's operating point, for direct comparability):**

| | value |
|---|---|
| training backbone | CLIP ViT-B/32, frozen (unchanged by this experiment — only the graph's *source* features vary, per spec §9's out-of-scope boundary) |
| `optimizer.lr` / `lr_label` | 1e-3 / 1e-4 |
| `model.embedding_dim` | 16 |
| `train.buddies.alpha` | 0.5 |
| `initialization_strategy` | `buddies` (fixed) |
| training-time buddy terms | **off** — `lambda_buddy=0`, `lambda_buddy_con=0`, `buddy_refresh=False` (isolating the init-construction effect only, same discipline as Experiments 1 and 10) |
| seeds | 1, 2, 3 |
| epochs / eval interval | 100 / 10 |
| dataset | RedCaps-150k (matches C5's scale) |

16 pairs × 3 seeds = 48 runs. **"vs. `clip_img:clip_txt`"** means: for each of the other 15 pairs, the paired-within-seed delta is computed against the `clip_img:clip_txt` pair's matched-seed result — `clip_img:clip_txt` (CLIP image features + CLIP text features for the buddy graph) is the pre-existing default and the natural baseline, since it's what every prior CoSiR experiment (C5–C7) has silently used without ever varying it.

**Run bookkeeping:** the group contains one non-finished run (`clip_img:clip_txt` seed=1, state=`crashed`, run id `s33sb8ge`) — a leftover record from an earlier, separately-cancelled sweep attempt that was paused mid-run and later resumed; the crashed run was cleanly re-executed as part of the resume, and all 3 finished seeds of the baseline pair are present in the 48-run count below. This is correctly excluded by the analysis script's non-finished-run filter (Task 5) and is not a coverage gap.

**Analysis:** `scripts/analyze_buddy_init_encoder_ablation.py --tag buddy-encoder-ablation-redcaps_150k` — paired-within-seed Δ (`pair − clip_img:clip_txt`) per non-baseline pair, mean ± std (reported via `mean/SEM` as the significance read, `|mean/SEM| ≥ 2` convention), for both `test_oracle/t2i_R1` and `test_oracle/i2t_R1`; plus a Pearson correlation of each pair's mean delta against its already-measured C3 cross-VLM survival rate, over the 15 non-baseline pairs.

## Results

### `test_oracle/t2i_R1` (vs. `clip_img:clip_txt`)

| pair | mean Δ | mean/SEM | wins/3 |
|---|---:|---:|---:|
| clip_img:bge | +0.20 | +0.9 | 2 |
| clip_img:e5 | +0.20 | +1.2 | 2 |
| clip_img:minilm | +0.00 | +0.0 | 1 |
| dinov2:bge | +0.00 | +0.0 | 1 |
| dinov2:clip_txt | +0.13 | +1.1 | 2 |
| dinov2:e5 | −0.10 | −0.4 | 1 |
| dinov2:minilm | −0.07 | **−2.0*** | 0 |
| siglip_v:bge | −0.17 | −1.3 | 1 |
| siglip_v:clip_txt | +0.10 | +0.4 | 1 |
| siglip_v:e5 | +0.10 | +1.7 | 2 |
| siglip_v:minilm | −0.13 | −1.0 | 0 |
| vit_sup:bge | +0.17 | +1.1 | 2 |
| vit_sup:clip_txt | +0.10 | +0.6 | 2 |
| vit_sup:e5 | −0.03 | −0.4 | 1 |
| vit_sup:minilm | +0.00 | +0.0 | 1 |

(`*` = clears `|mean/SEM| ≥ 2`.)

**Read: essentially null.** Every mean delta lies within, or very close to, the ~0.1–0.7 R1 noise floor established in `2026-06-24_buddy_progress_report.md` §8a. `dinov2:minilm` is the only pair that clears the significance bar, but at −0.07 R1 its magnitude is *below* the noise floor itself — flagged by the statistical test, but not a practically meaningful negative finding. No pair shows a real t2i improvement or degradation over the CLIP-sourced default.

### `test_oracle/i2t_R1` (vs. `clip_img:clip_txt`)

| pair | mean Δ | mean/SEM | wins/3 |
|---|---:|---:|---:|
| clip_img:bge | +1.43 | **+6.6*** | 3 |
| clip_img:e5 | +0.60 | **+2.9*** | 3 |
| clip_img:minilm | +0.27 | +0.6 | 2 |
| dinov2:bge | +1.97 | **+4.5*** | 3 |
| dinov2:clip_txt | +1.40 | **+2.7*** | 3 |
| dinov2:e5 | +0.93 | **+3.3*** | 3 |
| dinov2:minilm | +2.30 | **+6.5*** | 3 |
| siglip_v:bge | +1.23 | **+4.2*** | 3 |
| siglip_v:clip_txt | +0.83 | **+3.1*** | 3 |
| siglip_v:e5 | −0.37 | −0.8 | 1 |
| siglip_v:minilm | +0.97 | **+2.6*** | 3 |
| vit_sup:bge | +0.73 | **+2.3*** | 3 |
| vit_sup:clip_txt | +1.77 | **+4.0*** | 3 |
| vit_sup:e5 | −0.80 | **−6.9*** | 0 |
| vit_sup:minilm | +1.20 | **+5.8*** | 3 |

**Read: a genuinely different, striking picture.** 12 of the 15 non-baseline pairs clear `|mean/SEM| ≥ 2`, all 12 **positive**, and every one of those 12 wins 3/3 seeds — this is not a noisy, cherry-picked subset, it's the large majority of the grid, seed-replicated across the board. **The CLIP-sourced default (`clip_img:clip_txt`) is one of the worse choices among the 16 pairs tested for i2t retrieval** — most alternatives beat it, several by more than 1.5 R1 pts (`dinov2:minilm` +2.30, `dinov2:bge` +1.97, `vit_sup:clip_txt` +1.77, `clip_img:bge` +1.43).

Two pairs go the other direction: `siglip_v:e5` (−0.37, not significant) and `vit_sup:e5` (−0.80, significant negative, 0/3 wins) — both pair the "e5" text encoder with a non-CLIP vision encoder. This is **not** simply "e5 is a bad text encoder for this purpose": `dinov2:e5` is positive and significant (+0.93, 3/3 wins). The negative effect is pair-specific (`{siglip_v, vit_sup} × e5`), not attributable to either encoder alone.

## Survival-rate correlation (closing Experiment 6's deferred question)

C3's cross-VLM survival study (`2026-07-16_buddy_cross_vlm_survival.md`) measured, for each of these same 16 encoder pairs, what fraction of that pair's buddy-graph edges survive into the 16-way consensus core — and explicitly deferred the question of whether that survival rate predicts downstream usefulness. This experiment answers it directly by joining each pair's Experiment 8 retrieval delta against its already-measured C3 survival rate (Pearson r, n = 15 non-baseline pairs):

| direction | r | reading |
|---|---:|---|
| t2i | **+0.010** | no relationship — unsurprising, since t2i itself shows no reliable per-pair effect to correlate with anything |
| i2t | **−0.410** | moderate, and counterintuitive |

The i2t result deserves to be reported honestly rather than forced into a tidy story. A naive hypothesis going in would be that a pair whose buddy-graph edges survive more consensus checks across independently-drawn encoders (more "robust"/agreed-upon graph structure) should be *more* useful downstream, not less — the data says the opposite, moderately (r = −0.41, n = 15). This is not an overwhelming correlation and should not be over-read as a mechanistic claim ("more survival causes less usefulness") — n = 15 pairs is a modest sample, and no causal story is tested here. But it is a real, measured, negative-direction result, and it means the answer to Experiment 6's question is genuinely **no** for t2i and **weakly-negative** for i2t — not the "strong positive would validate/extend the survival metric" outcome a reader might have expected.

### Consequence for Experiment 6

Per the spec's own scoping (`docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 8's "Why": *"Joining each pair's Exp. 8 retrieval Δ against its already-measured C3 cross-VLM survival rate also answers, for free, the question Experiment 6 explicitly deferred... Experiment 6 can likely be dropped from scope once this runs."*), this correlation check **is** that answer, run at the pair level across all 16 cells with 3-seed replication per cell. The result is clear enough in both directions (near-zero for t2i, moderately negative for i2t) that a separate per-sample-level Experiment 6 sub-study would not resolve genuine ambiguity — it would refine a already-clear "no"/"weakly negative" into a more granular version of the same answer. **Experiment 6 is subsumed by this result and is dropped from scope** (see the spec update, §6/§8).

## Overall picture

This is a genuine, direction-asymmetric positive result, consistent with this project's established pattern of direction-dependent findings (e.g. C5's Impressions t2i/i2t split). The graph-source encoder pair matters a lot for i2t retrieval and essentially not at all for t2i retrieval, and — for i2t — the current default is demonstrably not the best choice: this is not a marginal technicality where one non-CLIP pair edges out the baseline by a hair. It is the majority (12 of 15) of tested pairs, several with large, seed-replicated effects (up to +2.30 R1, mean/SEM up to +6.6).

## Applying the decision rule

Per the spec's §4 Experiment 8 success criteria: *"If one or more non-CLIP pairs beat the CLIP-sourced init, that's a stronger initializer candidate to lead the paper with. If all 16 cluster near the current CLIP-sourced result, that is itself evidence that the graph structure — not the specific encoder — is what drives usefulness."*

The actual result satisfies the **first branch, decisively, for i2t** and lands closer to the **second branch for t2i**:

- **i2t:** far more than "one or more" — 12 of 15 non-baseline pairs beat the CLIP-sourced default, seed-replicated, several by a wide margin. `dinov2:minilm`, `dinov2:bge`, `vit_sup:clip_txt`, and `clip_img:bge` are strong candidates for a stronger initializer to lead the paper with on this axis, pending the caveats below.
- **t2i:** all 16 (including the one flagged-but-tiny exception) cluster within noise of the CLIP-sourced result — the graph *structure*, not the specific source encoder, appears to drive t2i usefulness, consistent with the spec's second branch and with C2/C3's broader "not an encoder artifact" claim from a new (downstream-use, not just signal-survival) angle.

This is not a case that needs forcing into one bucket — the two directions genuinely answer the decision rule differently, and reporting both honestly is more informative than picking one.

## Caveats

- **RedCaps-150k only.** This experiment was built and validated at 150k, matching C5's scale and reusing the existing `heldout_feats/redcaps/*.npy` cache with no new extraction cost. The spec's planned 300k extension (which would also let this connect to C6's CLIP-baseline-gap framing) was not run in this report — 300k requires a fresh held-out feature extraction pass for all 6 non-CLIP encoders (`heldout_feats_300k/`, per the spec's §4 Exp. 8 tooling note), not yet done.
- **Only `test_oracle` retrieval is reported here, not `test_pre_diff` (gap to CLIP).** C6 and Experiment 10 both established that the *realistic*, deployable comparison is `test_pre_diff` (predictor-conditioned − raw CLIP), not just oracle retrieval, and that buddy-init narrows but does not close that gap against the current CLIP-sourced default. Whether any of the i2t-winning non-CLIP pairs found here also narrow the `test_pre_diff` gap further than `clip_img:clip_txt` does is untested — a natural, cheap follow-up given Experiment 10's precedent for exactly this comparison (the same `test_raw`/`test_pre_diff` metrics are already logged by every eval call in `src/eval/pipeline.py`, no new eval code needed).
- **Only one operating point tested** (`lr=1e-3, lr_label=1e-4, embedding_dim=16, alpha=0.5`, all training-time buddy terms off) — same scope discipline as every prior experiment in this line (C5–C7, Experiment 10); not checked for interaction with a different operating point or with training-time buddy terms enabled.
- **n = 15 for the survival-rate correlation.** A moderate correlation (r = −0.41) at this sample size is a real, reportable signal but not a tight, high-confidence estimate — treat the i2t finding as "counterintuitive and worth noting," not as a precisely-quantified relationship.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_buddy_init_encoder_ablation.sh   # 16 encoder pairs x 3 seeds, 100 epochs each, RedCaps-150k
python scripts/analyze_buddy_init_encoder_ablation.py --tag buddy-encoder-ablation-redcaps_150k
```
