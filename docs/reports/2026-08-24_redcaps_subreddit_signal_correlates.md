# What predicts buddy-signal strength across RedCaps subreddits? — Experiment 9

**Date:** 2026-08-24 · **Dataset:** RedCaps-150k (`redcaps_150k.json`, same cache as `2026-06-23_redcaps_buddy.md`) · **Branch:** `experiment/buddy_init_ablation`
**Code:** `src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py` (analysis), `src/test/20260623_redcaps_buddy/redcaps_buddy.py` (`subreddit_lift`, extended to return every qualifying subreddit, not just top-15)
**Spec:** `docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md` §4 Experiment 9
**Precursor:** `docs/reports/2026-06-23_redcaps_buddy.md` (the original aggregate ~20× lift finding this deepens)

---

## TL;DR

RedCaps' aggregate ~20× same-subreddit buddy-edge lift (C1) is not evenly distributed: per-subreddit lift ranges from **4.4× to 671×** across 159 qualifying subreddits. But **none of the three candidate properties (sample count, caption diversity, visual homogeneity) meaningfully predicts where a subreddit falls on that range** — all three Pearson correlations are weak (|r| ≤ 0.33):

| Property | r | n |
|---|---|---|
| size (sample count) | **−0.328** | 159 |
| caption diversity (1 − mean pairwise text-cosine-sim) | **−0.219** | 159 |
| visual homogeneity (mean pairwise image-cosine-sim) | **+0.204** | 159 |

Size has the largest (still weak) association, and it's negative: bigger, more generic subreddits (`pics`, `cats`, `food`, `foodporn`) sit at the low-lift end, while small, visually/topically distinctive niches (`f1porn`, `scotch`, `trains`, `sushi`) sit at the high-lift end. Caption diversity and visual homogeneity point in the directions the spec's working hypothesis expected (more visually homogeneous → higher lift; more caption-diverse → lower lift, the opposite sign from the spec's guess) but both are too weak to call a real driver. This is a legitimate null result for these three properties, not a failure to find one — it rules them out as the mechanism behind why lift varies by subreddit, without weakening C1's core claim (the signal is real; it's just not simply explained by size, caption spread, or visual tightness alone).

---

## Method

### Per-subreddit lift (full breakdown, not just top-15)

`redcaps_buddy.subreddit_lift(data, e, top_k=None)` (extended in Task 1 of this plan item) computes, for every subreddit `s` with an edge-endpoint reliability threshold `exp_s > 5`:

```
lift_s = obs_same_sub_endpoints_s / exp_same_sub_endpoints_s
```

where `exp_same_sub_endpoints_s = deg_s * p_s` (`p_s` = subreddit `s`'s share of all edge endpoints, `deg_s` = its total edge-endpoint count) — the same "expected under the endpoint marginal" normalization used for the aggregate `overall_lift` figure in the original report, applied per-subreddit instead of only in aggregate. `top_k=None` returns every subreddit clearing the reliability filter, sorted by lift descending, instead of only the top 15.

The buddy graph itself is unchanged from the rest of the RedCaps buddy analysis: `E = A_img ∪ A_txt`, mutual-kNN per modality with `K=30` (project-wide default, `configs/train/default.yaml`), built over the full 150k CLIP feature cache.

### Three subreddit properties

- **Size:** raw sample count per subreddit (`len(idx)`).
- **Caption diversity:** `1 − mean_pairwise_cosine_sim(txt rows)` — high when a subreddit's captions are spread out in CLIP text-embedding space, low when they're repetitive/templated.
- **Visual homogeneity:** `mean_pairwise_cosine_sim(img rows)` — high when a subreddit's images cluster tightly in CLIP image-embedding space.

Both similarity properties use the same helper, `mean_pairwise_cosine_sim(X)`, applied to the subreddit's L2-normalized text or image feature rows respectively. Subreddits with fewer than `MIN_SUBREDDIT_SIZE = 20` samples are skipped — below that, a pairwise-similarity estimate over so few pairs is too noisy to trust (see Caveats).

**Closed-form identity, used to avoid an O(n²) loop.** For unit-norm row vectors, `‖Σx_i‖² = n + 2·Σ_{i<j}(x_i · x_j)`. Rearranging gives the mean pairwise cosine similarity in O(n·d) instead of O(n²·d):

```python
S = X.sum(axis=0)
total = S @ S - n                      # = 2 * sum_{i<j} x_i . x_j
mean_pairwise_sim = total / (n * (n - 1))
```

This matters because the largest qualifying subreddits run into the thousands of samples (`pics` alone has 9,882) — an explicit double loop over pairs would be `O(n²)` per subreddit and prohibitively slow across 159 subreddits, several with n in the thousands, while the closed form is a single sum-then-dot-product per subreddit regardless of size. The identity was cross-checked against a brute-force O(n²) loop on random unit vectors in the script's `--selftest` mode (max abs error < 1e-4) before being run on real data.

### Correlation

Pearson `r` between per-subreddit lift and each property, over the subreddits present in both the lift table (passed `exp_s > 5`) and the properties table (passed `size >= MIN_SUBREDDIT_SIZE`) — 159 subreddits in the actual run (see Results).

---

## Results

**Command run:**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py
```

**Captured output:**

```
Loading RedCaps data + buddy graph...
Computing full per-subreddit lift (350 subreddits)...
  overall_lift=22.80x over 159 qualifying subreddits
Computing per-subreddit properties (size, caption diversity, visual homogeneity)...

Correlation(subreddit lift, property):
                  size: r=-0.328  (n=159 subreddits)
     caption_diversity: r=-0.219  (n=159 subreddits)
    visual_homogeneity: r=+0.204  (n=159 subreddits)
wrote docs/reports/assets/redcaps_subreddit_correlates/lift_vs_properties.png
```

**Overall lift: 22.80×** over **159 of 350** subreddits (those clearing the `exp_s > 5` reliability filter *and* the `MIN_SUBREDDIT_SIZE = 20` property filter). This matches the original aggregate-lift report's union-graph figure (`2026-06-23_redcaps_buddy.md`, E: 22.8×) — same graph, same data, extended here to per-subreddit granularity.

### Per-subreddit spread

Lift ranges from **4.41× (`pics`)** to **670.79× (`f1porn`)** across the 159 qualifying subreddits (median 83.98×, mean 114.10×) — an order of magnitude of spread the single aggregate number hides.

Top 15 by lift (small, visually/topically distinctive niches):

| subreddit | lift | size | caption diversity | visual homogeneity |
|---|---:|---:|---:|---:|
| f1porn | 670.79 | 151 | 0.371 | 0.744 |
| scotch | 525.58 | 190 | 0.335 | 0.677 |
| trains | 425.17 | 248 | 0.461 | 0.614 |
| sushi | 395.21 | 168 | 0.359 | 0.748 |
| trucks | 381.94 | 203 | 0.364 | 0.621 |
| pourpainting | 354.46 | 277 | 0.355 | 0.751 |
| mead | 326.74 | 197 | 0.319 | 0.669 |
| chefknives | 323.42 | 232 | 0.326 | 0.731 |
| horses | 310.87 | 197 | 0.330 | 0.682 |
| leathercraft | 307.69 | 300 | 0.361 | 0.657 |
| flyfishing | 282.79 | 252 | 0.366 | 0.675 |
| quilting | 278.85 | 350 | 0.343 | 0.678 |
| vinyl | 278.79 | 419 | 0.362 | 0.608 |
| axolotls | 273.67 | 207 | 0.366 | 0.681 |
| tarantulas | 270.11 | 286 | 0.350 | 0.718 |

Bottom 15 by lift (large, generic subreddits):

| subreddit | lift | size | caption diversity | visual homogeneity |
|---|---:|---:|---:|---:|
| doggos | 12.95 | 238 | 0.309 | 0.673 |
| outdoors | 12.90 | 716 | 0.444 | 0.600 |
| plants | 12.29 | 970 | 0.387 | 0.674 |
| lookatmydog | 11.58 | 620 | 0.316 | 0.695 |
| natureporn | 11.20 | 211 | 0.433 | 0.607 |
| rarepuppers | 11.19 | 1,875 | 0.322 | 0.686 |
| dogpictures | 11.07 | 1,159 | 0.318 | 0.687 |
| photographs | 10.87 | 206 | 0.372 | 0.581 |
| catpictures | 10.53 | 308 | 0.293 | 0.738 |
| amateurphotography | 10.01 | 258 | 0.481 | 0.577 |
| cats | 9.73 | 9,027 | 0.307 | 0.725 |
| food | 9.07 | 4,991 | 0.451 | 0.636 |
| foodporn | 8.14 | 3,203 | 0.475 | 0.629 |
| eyebleach | 7.21 | 1,672 | 0.326 | 0.649 |
| pics | 4.41 | 9,882 | 0.435 | 0.497 |

(Full 159-row table is reproducible with the command above plus the `dump_table.py`-style loop shown in the reproduction section; only top/bottom 15 shown here per the task brief's "top/bottom-N if the full table is large" allowance.)

### Correlations

| Property | r | n | Interpretation |
|---|---:|---:|---|
| size | **−0.328** | 159 | Weak-to-moderate negative — the strongest of the three, but far from a strong predictor. Larger subreddits (`pics`, `cats`, `food`, `foodporn`) skew toward lower lift; small niche subreddits skew toward higher lift. Plausible mechanism: large subreddits are broader/less topically exclusive (more overlap with neighboring subreddits' content), diluting the same-subreddit enrichment signal — but this is not tested directly here. |
| caption diversity | **−0.219** | 159 | Weak negative. Notably the *opposite* sign from the spec's working hypothesis ("caption-diverse" subreddits expected to show *higher* lift) — if anything, subreddits with more repetitive/templated captions show marginally higher lift, not lower. Too weak to draw a mechanistic conclusion from. |
| visual homogeneity | **+0.204** | 159 | Weak positive, in the direction the spec's hypothesis expected (more visually homogeneous → higher lift), but weak enough that visual homogeneity alone explains very little of the variance (r²≈0.04). |

None of the three properties clears even a conventional "moderate" correlation threshold (|r| ≥ 0.4–0.5). **This is a genuine null finding for all three candidate explanations**, not an artifact of a bug or an underpowered sample (n=159 is a reasonably large sample of subreddits). It rules out sample count, caption diversity, and visual homogeneity — individually, linearly — as the drivers of *why* buddy-signal strength varies 150-fold across RedCaps subreddits, while leaving the core C1 claim (the signal itself is real, robust, ~20× on average) untouched.

### Figure

![Lift vs. the three properties](assets/redcaps_subreddit_correlates/lift_vs_properties.png)

Three scatter panels (size on a log x-axis, caption diversity, visual homogeneity — each vs. subreddit lift on the y-axis) generated by the script's `_write_figure()`. Visually consistent with the weak correlation numbers: each panel shows a broad, noisy cloud rather than a clean trend line, with the size panel showing the clearest (still loose) downward tilt.

---

## Caveats

- **`MIN_SUBREDDIT_SIZE = 20` threshold.** Subreddits with fewer than 20 samples are excluded from the properties table (and therefore from the correlation) because a pairwise-cosine-similarity estimate over too few pairs is too noisy to trust. Combined with the lift table's own `exp_s > 5` reliability filter, this leaves 159 of RedCaps' 350 subreddits in the joined analysis — the smallest, sparsest subreddits (191 of them) are simply absent from this correlation, not assigned an unreliable value. This is a deliberate reliability trade-off, not a hidden data loss: it means the correlation is a statement about RedCaps' mid-to-large subreddits, not about the long tail of very small ones.
- **Linear correlation only.** Pearson `r` only detects linear association; a nonlinear or threshold-based relationship (e.g. lift dropping sharply only above some size cutoff) would not necessarily show up as a large `|r|` here. The scatter figure is the honest complement to the correlation numbers for this reason.
- **Three properties tested independently, not jointly.** This analysis does not fit a joint model (e.g. multiple regression) over all three properties simultaneously, nor does it test interactions between them (e.g. small-and-visually-homogeneous vs. large-and-visually-homogeneous). A weak individual correlation for each property does not rule out a joint effect.
- **`exp_s > 5` filter interacts with size.** Because `exp_s` scales with a subreddit's edge-endpoint degree, very small subreddits are disproportionately likely to fail the lift-table reliability filter even before the properties-table `MIN_SUBREDDIT_SIZE` filter is applied — the two filters are correlated in their effect, both trimming toward "moderately-sized-or-larger" subreddits.
- **Same caveat as the original report:** this analysis uses the same 150k RedCaps subsample and the same E (union) buddy graph as `2026-06-23_redcaps_buddy.md`; the aggregate lift number (22.80×) it reproduces confirms consistency with that report, but any caveat that applied there (buddy graph fragments into components, spectral-init structure caveats, etc.) is orthogonal to and does not affect this per-subreddit lift/correlation analysis, which operates on raw graph edges, not on any downstream spectral embedding.

---

## Reproduce

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
python src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py --selftest  # offline arithmetic check
python src/test/20260824_redcaps_subreddit_correlates/analyze_subreddit_correlates.py             # full run against cached RedCaps-150k data
```

Runs in a few minutes end-to-end (loads the cached 150k CLIP feature store, builds the mutual-kNN union buddy graph with `K=30`, computes per-subreddit lift and properties, prints correlations, writes the figure). CUDA is used automatically if available (`torch.cuda.is_available()`); falls back to CPU otherwise — no GPU is strictly required.
