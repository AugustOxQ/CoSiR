# Do conditional buddies survive changing the vision–language model?

**Date:** 2026-07-16 (run 2026-07-17) · **Dataset:** RedCaps (N = 150,000) · **Branch:** `experiment/conditional_buddy_cross_vlms`
**Code:** `src/test/20260716_buddy_cross_vlm/` · **Spec:** `docs/superpowers/specs/2026-07-16-buddy-cross-vlm-survival-design.md`
**Artifacts:** `docs/reports/assets/buddy_cross_vlm/`

---

## TL;DR

Conditional buddies are **not an artifact of the specific CLIP encoders** used to build them. When we rebuild the cross-modal buddy graph with 16 different (vision × text) encoder combinations:

- **~20% of the exact buddy pairs recur** between any two arbitrary encoder choices — at **10³–10⁵× the chance rate**.
- The **vision encoder drives the variation**: keeping CLIP's image tower and swapping only the text encoder preserves buddies best; swapping the image tower (worst: supervised ImageNet ViT) degrades agreement most.
- A **stable "core"** of buddies survives *all 16* combinations (2,915 strict / 174,161 union edges), and that core is **13–22× more subreddit-coherent** than chance — i.e. the surviving buddies are semantically real, not mutually-agreed noise.

---

## The question (and how it differs from the held-out grid)

A "buddy" is an edge of the cross-modal mutual-nearest-neighbour graph:

```
A_img = mutual_knn(vision_features, K=30)   # mutual-NN graph in the vision encoder's space
A_txt = mutual_knn(text_features,   K=30)   # mutual-NN graph in the text encoder's space
B = A_img ∩ A_txt   (strict: mutual-NN in BOTH modalities)
E = A_img ∪ A_txt   (union: the edge set that seeds condition vectors)
```

The key property is that buddies combine **edges, not vectors** — only the per-modality graph *structure* is intersected/unioned, never the raw feature vectors. So heterogeneous encoders (different architectures, dims, no shared space) can be mixed freely; a "cell" is one (vision, text) pairing.

This is a different experiment from the **held-out grid** (`2026-07-08_heldout_grid.md`). There, the buddy graph was *fixed* at CLIP∪CLIP and six single-modality encoders only *scored* whether buddy pairs were closer than random in their own space — the graph was never rebuilt. **Here we rebuild the buddy graph for every (vision × text) encoder pair and ask whether the buddy relationships themselves persist.**

## Method

**The 4 × 4 grid (16 cells).** Every cell needs both a vision encoder (builds `A_img`) and a text encoder (builds `A_txt`):

| axis | encoders |
|------|----------|
| Vision | `clip_img` (CLIP ViT-B/32), `dinov2` (self-supervised), `siglip_v` (SigLIP vision tower), `vit_sup` (ImageNet-supervised ViT) |
| Text | `clip_txt` (CLIP text tower), `minilm`, `bge`, `e5` (sentence encoders) |

CLIP features come from the RedCaps `FeatureManager`; the other six from the held-out grid's cached features (`heldout_feats/redcaps/*.npy`), all in the same row order. The 8 distinct feature matrices are each turned into one mutual-kNN graph (built once, reused across cells); a cell is then cheap set algebra on the two relevant graphs.

**Common node set.** All rows valid across *every* vision encoder are kept (missing images would be zero-feature rows); all matrices are sliced once so every cell shares an identical node index space — the precondition for comparing edge sets. On this run, **0 of 150,000 rows were dropped**.

**Agreement (chance-corrected).** For each pair of cells we report Jaccard `|E₁∩E₂|/|E₁∪E₂|` and the overlap coefficient, for both `B` and `E`. Raw overlap is inflated by density and luck, so each value is divided by a **chance null**: the expected overlap under a uniform random node-relabeling of one graph. Under that relabeling each of graph `b`'s edges lands on a uniformly random node pair, so `E[inter] = |a|·|b| / C(N,2)` and the null Jaccard is `E[inter]/(|a|+|b|−E[inter])`. **`lift = observed / null`**; agreement counts only when lift clears ~1.

> **Method note.** The original spec used a Monte-Carlo *permutation* null. At N=150k (~4.5M edges in `E`) that loop was intractable (a run was killed after 154 min with no output). It was replaced — with approval — by the **closed-form analytic** null above, which is exactly the expectation the permutation loop estimated, is lower-variance, and makes the full run take **~31 s**. The spec was updated to match.

**Consensus core.** A co-occurrence graph weights each edge by how many of the 16 cells contain it (0–16). The **survival curve** `n_core(t)` counts edges present in ≥ t cells; the `t ≥ 8` majority set is the "core that survives VL-model choice."

**Core validation (independent ground truth).** RedCaps' **subreddit** label (parsed from the image path, used by nothing in the pipeline) validates the core: `subreddit_lift` measures how much more often core edges join same-subreddit images than chance. If the core is real, lift stays high across consensus levels.

## Results

### 1. Buddies survive far above chance

| graph | median off-diag Jaccard | median lift vs chance | Jaccard range | core surviving all 16 cells |
|-------|------------------------:|----------------------:|:-------------:|----------------------------:|
| **B** (strict) | 0.197 | **≈176,000×** | 0.130 – 0.538 | 2,915 edges |
| **E** (union)  | 0.196 | **≈2,650×**   | 0.128 – 0.605 | 174,161 edges |

Roughly one buddy pair in five is *exactly* reproduced between two arbitrary encoder choices — orders of magnitude above the ≈10⁻³–10⁻⁵ you would get by chance (see `agreement_B.png`, `agreement_E.png`).

### 2. The vision encoder drives the variation

Agreement of every cell with the reference **CLIP×CLIP** cell (union graph `E`):

| cell (vs CLIP×CLIP) | Jaccard | what changed |
|---------------------|--------:|--------------|
| `clip_img × bge` | 0.518 | text only |
| `clip_img × e5` | 0.506 | text only |
| `clip_img × minilm` | 0.489 | text only |
| `siglip_v × clip_txt` | 0.413 | vision only (SigLIP) |
| `vit_sup × clip_txt` | 0.356 | vision only (supervised) |
| `dinov2 × clip_txt` | 0.345 | vision only (self-sup) |
| `siglip_v × e5` | 0.173 | both |
| `dinov2 × {e5,minilm,bge}` | 0.134–0.138 | both |
| `vit_sup × {e5,minilm,bge}` | 0.128–0.132 | both |

The pattern (identical for `B`):

- **Keeping CLIP's image tower and swapping text preserves the most** (J ≈ 0.49–0.52). Swapping the image tower and keeping CLIP text preserves less (J ≈ 0.35–0.41). So **changing the vision encoder hurts buddy stability more than changing the text encoder** — the cross-modal buddy relation is anchored mostly by *image* geometry.
- Among vision swaps, **SigLIP** (a contrastive VLM tower, most CLIP-like) preserves buddies best (0.41); **self-supervised DINOv2** and **ImageNet-supervised ViT** preserve least (~0.35 / 0.36 with CLIP text, ~0.13 when text also changes).
- When **both** towers change, agreement drops to ~0.13–0.17 — still ~10³× above chance, but the two encoder choices compound.

### 3. A stable, semantically-real core

**Survival curves** (`survival_curves.png`) — edges present in ≥ t of 16 cells:

- `B`: 119,281 (t≥1) → 25,226 (t≥5) → 17,793 (t≥7) → **2,915 (t=16)**
- `E`: 4,558,062 (t≥1) → 1,208,538 (t≥8) → **174,161 (t=16)**

The curves step in plateaus (e.g. every `E` edge appears in **at least 4** cells) because an edge from one encoder's graph recurs across all cells sharing that encoder — a structural signature of the 4-vision × 4-text design, not noise.

**Subreddit lift of the ≥ t core** (`core_lift.png`) — the coherence check:

| t | `B` core lift | `E` core lift |
|---|--------------:|--------------:|
| 1 | 21.8× | 20.2× |
| 5 | 16.6× | 22.7× (peak) |
| 8 | 15.2× | 22.5× |
| 16 | 12.8× | 19.0× |

Every consensus level is **12–23× more subreddit-coherent than chance**. For the union graph `E`, coherence even *peaks* in the middle of the consensus range; for strict `B` it declines mildly with t (the universal strict core likely concentrates on broader/hub subreddits). Either way, the buddies that survive VL-model changes are strongly semantically meaningful.

## Interpretation

1. **Buddies are largely encoder-agnostic.** The buddy relation is a property of the underlying image–text data geometry, not of CLIP specifically: ~20% exact recurrence at 10³–10⁵× chance across arbitrary VL encoders, with a core that survives all of them.
2. **Image geometry anchors buddies.** Vision-encoder choice matters more than text-encoder choice; contrastive vision towers (CLIP, SigLIP) agree most, supervised/self-supervised towers least.
3. **The surviving core is real.** Independent subreddit ground truth confirms the consensus core is semantically coherent (12–23× lift), so "survival" reflects shared content structure, not a shared failure mode.

## Caveats / limitations

- **RedCaps only** (clean 1:1 image–caption). The same-photo / near-duplicate structure of Impressions is not tested here.
- **K = 30 fixed**; agreement magnitudes will shift with neighbourhood size (not swept).
- The **analytic null** is exact in expectation but is a plug-in of `E[inter]` into the Jaccard identity (not `E[Jaccard]`); it is a chance *reference*, not a p-value.
- The `B` survival curve's plateaus and the mild `B` core-lift decline are structural, discussed above — read the survival curve shape, not just endpoint counts.
- **Not tested:** whether higher cross-VLM survival translates into better downstream condition-vector initialization or training (that is the natural follow-up — "Approach C", embedding-level agreement, deliberately left as future work).

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate CoSiR
# fast pipeline sanity (first 512 nodes, no artifacts):
python src/test/20260716_buddy_cross_vlm/run_grid.py --smoke 512
# full run (~31 s, writes all artifacts):
python src/test/20260716_buddy_cross_vlm/run_grid.py
# unit tests:
python -m pytest src/test/20260716_buddy_cross_vlm/test_cross_vlm_buddy.py -v
```

Outputs (`docs/reports/assets/buddy_cross_vlm/`): `grid_agreement.json` (16×16 Jaccard / overlap / lift + survival + core-lift for `B` and `E`), `agreement_B.png`, `agreement_E.png`, `survival_curves.png`, `core_lift.png`, and `core_edges_{B,E}.npy` (the t ≥ 8 surviving edge lists). The `.npy`/`.json` outputs are git-ignored and regenerated by the run; the figures and this report are committed.
