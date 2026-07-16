# Cross-VLM Buddy Survival — Log

Spec: docs/superpowers/specs/2026-07-16-buddy-cross-vlm-survival-design.md
Plan: docs/superpowers/plans/2026-07-16-buddy-cross-vlm-survival.md

## Prerequisite: held-out RedCaps caches

All six required caches already existed from the Jul 8 held-out grid run — no extraction needed.

**Verified caches:**
- dinov2.npy ✓
- siglip_v.npy ✓
- vit_sup.npy ✓
- minilm.npy ✓
- bge.npy ✓
- e5.npy ✓

**Verification command output (2026-07-16):**
```bash
$ ls -1 src/test/20260708_heldout_grid/heldout_feats/redcaps/{dinov2,siglip_v,vit_sup,minilm,bge,e5}.npy
src/test/20260708_heldout_grid/heldout_feats/redcaps/bge.npy
src/test/20260708_heldout_grid/heldout_feats/redcaps/dinov2.npy
src/test/20260708_heldout_grid/heldout_feats/redcaps/e5.npy
src/test/20260708_heldout_grid/heldout_feats/redcaps/minilm.npy
src/test/20260708_heldout_grid/heldout_feats/redcaps/siglip_v.npy
src/test/20260708_heldout_grid/heldout_feats/redcaps/vit_sup.npy
```

## Method note (deviation from original spec)

The chance-correction was specified as a Monte-Carlo permutation null. At full scale
(N=150,000, ~4.5M edges in the E graphs) that loop was intractable — a run was killed
after 154 min of CPU with no artifacts written. It was replaced (user-approved) with the
**closed-form analytic null** it was only ever estimating: under a uniform random
node-relabeling of graph `b`, `E[inter] = |a|·|b| / C(N,2)`, so null Jaccard =
`E[inter]/(|a|+|b|−E[inter])`. Exact in expectation, lower-variance, and the full run now
completes in **~31 s**. Spec updated; see `chance_null_jaccard`.

## Results (full RedCaps, N=150,000, K=30, 4×4 grid = 16 cells)

Run 2026-07-17, ~31 s. Artifacts in `docs/reports/assets/buddy_cross_vlm/`.
All 150,000 rows valid across every vision encoder (dropped 0).

**Headline — do buddies survive changing the VL model? Yes, far above chance.**

| graph | median off-diag Jaccard | median lift vs chance | Jaccard range | universal core (edges in all 16 cells) |
|-------|------------------------:|----------------------:|:-------------:|---------------------------------------:|
| B (strict, A_img ∩ A_txt) | 0.197 | ~176,000× | 0.130 – 0.538 | 2,915 |
| E (union,  A_img ∪ A_txt) | 0.196 | ~2,650×   | 0.128 – 0.605 | 174,161 |

- ~20% of the *exact* buddy pairs recur between any two arbitrary (vision×text) encoder
  choices — orders of magnitude above the ~1e-3–1e-5 chance rate (hence the huge lift).

**The vision encoder drives the variation (not the text encoder).**
- Highest agreement keeps CLIP's image tower and varies only text: vs the CLIP×CLIP cell,
  `clip_img×e5/minilm/bge` reach J≈0.35–0.38 (B) / 0.49–0.52 (E).
- Lowest agreement swaps vision to supervised ImageNet ViT: `vit_sup×{minilm,bge}` J≈0.13.
- Self-supervised DINOv2 and SigLIP vision towers sit in between.
Interpretation: the cross-modal buddy relation is anchored mostly by the image geometry;
text-encoder choice perturbs it much less.

**A stable "core" of buddies exists and is semantically real (subreddit-validated).**
- Survival `n_core(t)` (edges present in ≥ t of 16 cells):
  - B: 119,281 (t≥1) → 25,226 (t≥5) → 17,793 (t≥7) → 2,915 (t=16).
  - E: 4,558,062 (t≥1) → 1,208,538 (t≥8) → 174,161 (t=16).
  - (B shows plateaus at t={5,6},{7,8},… because edges tend to appear in cells sharing a
    vision encoder, so counts cluster near multiples of 4 — a structural artifact of the
    4-vision × 4-text design, not noise.)
- Same-subreddit lift of the ≥ t core (independent ground truth, used by nothing in the
  pipeline):
  - E: t=1 → 20.2×, t=8 → 22.5× (peak), t=16 → 19.0× — the surviving core stays ~20×
    more subreddit-coherent than chance across all consensus levels.
  - B: t=1 → 21.8×, t=16 → 12.8× — every level is strongly coherent; the mild decline
    likely reflects the universal strict-core concentrating on broader/hub subreddits.

**Reference-anchored view (free):** the CLIP×CLIP cell is one grid cell, so its row of the
agreement matrix answers the original "do CLIP buddies survive?" question directly — they
do, most strongly under text swaps and least under a supervised-vision swap (see above).

**Bottom line:** conditional buddies are not an artifact of the specific CLIP encoders. The
buddy relation is largely encoder-agnostic (driven by image geometry), ~20% of exact pairs
persist across arbitrary VL choices at 10³–10⁵× chance, and a consensus core survives every
combination while remaining strongly semantically coherent.

### Known minor caveats (for final review)
- `grid_agreement.json` contains non-standard `NaN` literals (lift-matrix diagonal;
  zero-edge cores) — fine for Python `json.load`, breaks strict/JS parsers.
- `survival_curves.png` uses a log y-axis; if any B core count hit 0 at high t it would be
  silently dropped (here the minimum is 2,915, so nothing is dropped).
