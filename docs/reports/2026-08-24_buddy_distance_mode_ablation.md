# Fixing the buddy-graph's modality-dilution bug doesn't move retrieval or the gap to CLIP

**Date:** 2026-08-24 · **Dataset:** RedCaps, 150,000 rows of `redcaps_train.json` (matches C5's scale) · **Branch:** `experiment/buddy_init_ablation`
**Code:** `scripts/run_buddy_distance_mode_ablation.sh`, `scripts/analyze_buddy_distance_mode_ablation.py`
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 10
**Precursor:** `src/test/20260824_buddy_graph_disagreement/` (the no-training diagnostic that motivated this experiment and whose edge-composition/dilution findings this report does not re-litigate)

---

## TL;DR

**A real, diagnostic-confirmed graph-construction flaw — fixed — produces no measurable retrieval or gap-to-CLIP effect at this operating point/scale.** The diagnostic (`src/test/20260824_buddy_graph_disagreement/`) found that `compute_buddy_init`'s fixed-alpha distance blend collapses ~98% of a real RedCaps buddy graph's edges (the single-modality-only ones) from a good supporting-modality rank (median 0.22–0.30) down to statistical noise (median ~0.50). This experiment implemented an opt-in fix (`distance_mode="typed"`, Tasks 1–3) that uses each such edge's own supporting modality's rank instead of blending in the unsupported modality's noise, and ran the standard 3-seed paired ablation (`blend` vs `typed`) on RedCaps-150k to see whether the fix changes downstream training outcomes.

It doesn't, measurably:

- **Retrieval t2i R1:** mean Δ = **+0.00 ± 0.26**, mean/SEM = **+0.0** (typed wins 2/3 seeds)
- **Retrieval i2t R1:** mean Δ = **−0.37 ± 0.59**, mean/SEM = **−1.1** (typed wins 1/3 seeds)
- **Gap-to-CLIP t2i** (`test_pre_diff`, ours − CLIP): mean Δ = **+0.00 ± 0.10**, mean/SEM = **+0.0** (typed wins 1/3 seeds)
- **Gap-to-CLIP i2t** (`test_pre_diff`, ours − CLIP): mean Δ = **+0.17 ± 0.55**, mean/SEM = **+0.5** (typed wins 2/3 seeds)

Every delta is well inside the project's measured noise floor (~0.1–0.7 R1, `2026-06-24_buddy_progress_report.md` §8a) and none clear the `mean/SEM ≥ 2` significance bar — the largest magnitude observed is `−1.1`. Per the spec's decision rule (§4, Experiment 10), this is a **null** result: a legitimate, citable negative finding, not a failed experiment. See "Interpretation" below for why this does not undercut the diagnostic's own validity.

---

## Method

`compute_buddy_init` builds the union buddy graph `E = A_img ∪ A_txt` from independent per-modality mutual-kNN graphs, then needs a single distance value per edge of `E` to feed the spectral embedding step. The pre-existing behavior (`mix_distances`, still the default, `distance_mode="blend"`) computes both an image-cosine-distance and a text-cosine-distance for *every* edge and blends them with a fixed `alpha*D_img + (1-alpha)*D_txt`, regardless of which modality(ies) actually support that edge. For an edge that only one modality's mutual-kNN graph actually found (the diagnostic's ~98% case), the other modality's "distance" for that pair is not a signal — it's whatever cosine distance that unrelated modality happens to assign, and blending it in dilutes the real, single-modality signal toward noise.

This experiment's fix (`mix_distances_typed`, `src/conditional_buddy/buddy_graph.py`) classifies each edge of `E` by which of the original per-modality graphs (`A_img`, `A_txt`) support it, then:
- **img-only or txt-only edges** (the diluted ~98%): use that edge's own supporting modality's rank-normalized distance alone.
- **`both`-supported edges** (no disagreement to correct) and **`repair`/`neither` edges** (added by `ensure_min_degree`/`ensure_connected`, not owned by either modality): keep the existing fixed blend — there is no single supporting distance to prefer.

This is opt-in via `distance_mode: str = "blend"` on `compute_buddy_init` (default unchanged, byte-identical to pre-2026-08-24 behavior; only passing `"typed"` explicitly changes anything), threaded through to `train.buddies.distance_mode` in the training config (same integration pattern as Experiment 8's `encoder_pair`).

**Operating point** (identical to C5/C6, for direct comparability):

| | value |
|---|---|
| `optimizer.lr` / `lr_label` | 1e-3 / 1e-4 |
| `model.embedding_dim` | 16 |
| `train.buddies.alpha` | 0.5 |
| `initialization_strategy` | `buddies` (fixed) |
| training-time buddy terms | **off** — `lambda_buddy=0`, `lambda_buddy_con=0`, `buddy_refresh=False` (isolating the init-construction effect only, same discipline as Experiment 1) |
| swept axis | `train.buddies.distance_mode ∈ {blend, typed}` |
| seeds | 1, 2, 3 |
| epochs / eval interval | 100 / 10 |
| dataset | RedCaps-150k (matches C5's scale, cheapest iteration) |

6 runs total (2 modes × 3 seeds), all finished.

## Results

Paired-within-seed deltas (`typed − blend`), 3 seeds each, RedCaps-150k:

| Metric | mean Δ | ± std | mean/SEM | wins (typed/3) |
|---|---:|---:|---:|---:|
| retrieval `test_oracle/t2i_R1` | +0.00 | 0.26 | +0.0 | 2/3 |
| retrieval `test_oracle/i2t_R1` | −0.37 | 0.59 | −1.1 | 1/3 |
| gap-to-CLIP `test_pre_diff/t2i_R1` (ours − CLIP) | +0.00 | 0.10 | +0.0 | 1/3 |
| gap-to-CLIP `test_pre_diff/i2t_R1` (ours − CLIP) | +0.17 | 0.55 | +0.5 | 2/3 |

Sanity check — `test_raw` (frozen-CLIP embeddings, no conditioning) is identical across all 6 runs, confirming the only thing that differed between arms was the buddy-graph construction, not the frozen backbone or test set:

- `test_raw/t2i_R1` = **28.1** (single distinct value across all 6 runs)
- `test_raw/i2t_R1` = **29.7** (single distinct value across all 6 runs)

No metric reaches `|mean/SEM| ≥ 2`. The largest-magnitude effect (`−1.1` on retrieval i2t) is still short of the significance bar and well within the ~0.1–0.7 R1 duplicate-config noise floor established previously.

## Interpretation

**Two separate, both-true findings, not one:**

1. **The diagnostic's finding is real and independently verified.** `src/test/20260824_buddy_graph_disagreement/diagnose_disagreement.py` measured, on real RedCaps graph data at two scales (150k, 300k, agreeing), that ~98% of buddy-graph edges are single-modality-only, and that the pre-existing fixed blend collapses their supporting-modality rank from a good median (0.22–0.30) to statistical noise (median ~0.50), with ~80% of nodes acting as "bridge" nodes across non-overlapping modality neighborhoods — the structural precondition for spectral false-transitivity. Nothing in this experiment contradicts or weakens that measurement; this experiment did not re-run the diagnostic, only the downstream training consequence of fixing what it found.

2. **Fixing that dilution does not produce a measurable downstream training effect at this operating point/scale.** All four deltas above are statistically indistinguishable from zero. This means the RedCaps-150k spectral embedding step, at this operating point, is either robust to this particular distortion (the "noise" the blend introduces on 98% of edges doesn't change which broad spectral structure the embedding recovers, only fine-grained rank within it) or any real effect is too small to detect with 3 seeds (see Caveats).

Per the spec's own decision rule for this experiment (§4, Experiment 10): this is the **null** branch — "the dilution is real (diagnostic-confirmed) but doesn't matter for the downstream spectral embedding's usefulness — still a legitimate, citable negative result (a graph-construction flaw that doesn't propagate to a measurable training effect)." Do not read this as "the diagnostic was wrong" or "the fix doesn't work" — the fix does exactly what it was designed to do (restore single-modality edges' own rank instead of diluting it); it simply doesn't change retrieval outcomes measurably at this scale. This also means C6's still-open "does anything close the gap to CLIP" question remains open — `typed` does not narrow `test_pre_diff` any more than `blend` does.

## Caveats

- **Only 3 seeds.** The statistical standard used throughout this project (§5 of the spec) treats 3 seeds as sufficient to detect effects comparable to prior findings in this project (e.g. C5/C6's clean multi-point R1 wins), but a small true effect near or below the noise floor could be genuinely present and simply underpowered to detect at n=3. This report does not claim "no effect exists," only "no effect was detected at this scale/seed count."
- **Only RedCaps-150k was tested.** C6 showed that a different buddy-init question (buddy-init vs. generic init) can change with scale (150k → 300k, effect got larger/tighter, not smaller). Whether `typed` vs. `blend` behaves differently at 300k, 500k, or 1M is untested here — this experiment did not extend beyond 150k (matches the spec's stated scope for Experiment 10 exactly, no scale extension was in scope for this plan).
- **Only one operating point was tested** (`lr=1e-3, lr_label=1e-4, embedding_dim=16, alpha=0.5`, `buddies` init strategy, all training-time buddy terms off). A different alpha, embedding dimension, or with training-time buddy terms enabled might interact with the distance-mode change differently; not checked here.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
bash scripts/run_buddy_distance_mode_ablation.sh   # 2 distance modes x 3 seeds, 100 epochs each, RedCaps-150k
python scripts/analyze_buddy_distance_mode_ablation.py --tag buddy-distance-mode-ablation-redcaps_150k
```
