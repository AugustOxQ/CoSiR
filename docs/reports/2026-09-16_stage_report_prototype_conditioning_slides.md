# Experiment 18 — Prototype-Pooled Conditioning

**Framing:** A stage report for a team member catching up after ~2 months away — how CoSiR conditions embeddings today, and what Experiment 18 tried, found, and is still deciding.

**Date:** 2026-09-16
**Branch:** `experiment/buddy_prototype_conditioning`

---

## Where CoSiR stood ~2 months ago

CoSiR keeps a **frozen** pretrained CLIP model — the image and text encoders are never fine-tuned — and attaches a small trainable **condition vector** to every training sample. That vector is fed into a small trainable **combiner** module, which nudges one side's CLIP embedding before the usual contrastive retrieval loss, aiming for retrieval better than raw CLIP while staying close to CLIP's own embedding space.

Each condition vector starts from the **buddy graph**: a graph connecting samples that are mutual nearest-neighbors in CLIP's image space, text space, or both, so that samples CLIP already considers close begin training with similar conditions rather than random ones.

*This buddy-graph idea, and its validation, is the established starting point — not new this report.*

---

## Two months of hardening the foundation (mid-July → early Sept)

The following months of work strengthened this basis without changing the buddy-graph construction Experiment 18 still relies on:

| Milestone | One-line takeaway |
|---|---|
| Cross-encoder buddy-graph check | Buddy structure holds across 16 different vision/text encoder pairs — not a CLIP-specific fluke |
| Buddy-init vs. generic init | Buddy-graph initialization wins on RedCaps and becomes the standing default; neither init beats raw CLIP outright at this scale |
| Robustness/diagnostic ablations | Graph-edge, encoder-pair, and bridge-node checks confirmed the construction was sound; none changed the architecture |
| Neighborhood-size (`K`) scaling | Fixed `K=30` is measurably suboptimal at 500k samples (predicted `K≈39` wins i2t R1 by +0.97) — real, still open, not yet in configs |
| Combiner redesign | A rank-16 low-rank residual adapter beat the older two-MLP-tower combiner, confirmed at full 500k/100-epoch scale — selected for Experiment 18, not yet a global default |

**Also standing, deliberately:** conditioning is asymmetric — the condition is fused only into the image embedding (`combine_side="img"`); text passes through a separate identity-initialized `other_proj` layer. A parallel, not-yet-merged line of work suggests this one-sidedness may cause a retrieval asymmetry between the two retrieval directions — a forward-pointer, not resolved here.

---

## Architecture right before Experiment 18

![CoSiR before Experiment 18](assets/2026-09-16_stage_report/architecture_before.png)

A large lookup table holds **one trainable condition vector per training sample** (16-d), initialized from the buddy graph and then trained directly at its own, much larger learning rate. The image embedding and that sample's condition vector enter the low-rank combiner; text passes through `other_proj` untouched; both final embeddings train against a contrastive retrieval loss.

**The problem this leaves open:** every row of the table is independent — there's no way to ask what a condition *means* in general, and no way to condition a brand-new sample without a separately-trained side network (`condition_predictor`) approximating the table.

---

## Experiment 18 — the idea

Replace the giant independent lookup table with a small, **shared bank of 16 learnable "prototype" vectors** — a small set of learned archetypes. Each sample's condition becomes a soft, weighted blend of those 16 shared prototypes, computed fresh from the sample's own CLIP feature rather than looked up from a table.

**The bet:**
- With 16 shared archetypes instead of one row per sample, the condition space should naturally organize into interpretable clusters.
- Because the condition is a function of the sample's own features, it works automatically for new, unseen samples — no separate predictor network needed.

---

## Implementation: attention over a small prototype bank

![Experiment 18 prototype-pooled conditioning](assets/2026-09-16_stage_report/architecture_after.png)

Each of the 16 prototypes has a learned **key** and a learned **value** vector. For a sample, its own CLIP feature (image and text, averaged) is projected into a **query**; the query is compared against all 16 keys to produce 16 similarity scores; a **temperature-controlled softmax** turns those scores into attention weights that sum to 1; the condition vector is the attention-weighted sum of the 16 prototypes' value vectors.

Everything downstream — the low-rank combiner, `other_proj`, the retrieval loss — is completely unchanged. Only the condition vector's source changes.

---

## A few terms this report leans on

- **Oracle retrieval (Recall@1/5/10):** retrieval accuracy given the best-available condition vector per query — always read against **raw retrieval** (plain, unconditioned CLIP) as the floor to beat.
- **Effective dimensions (95% variance):** how many independent directions are needed to explain 95% of the spread across a set of condition vectors. Close to 16 = using its full capacity; collapsing to 1 = nearly every vector differs only along one shared direction.
- **Silhouette score:** a standard clustering-quality number — positive and high means clean, separated groups; near-zero or negative means no real structure.
- **Probe selectivity:** train a simple classifier to guess a proxy label from only the condition vector; selectivity is its real accuracy minus its accuracy on shuffled labels. Meaningfully positive = the space actually encodes that distinction.
- **`warmth` / `register`:** two proxy groupings of RedCaps subreddits (companion-animal vs. curiosity-framed; polished-photography vs. casual-snapshot) used to probe the condition space. **Important hedge:** a dedicated audit found `warmth` is mostly explained by plain photo content, not confirmed emotional tone, and `register` carries little signal beyond content either — treat both as content-separation probes, not validated measures of emotion or formality.

---

## First pass: retrieval flat, interpretability collapsed

**Default settings, 3 random seeds, RedCaps-150k.**

Retrieval was roughly on par with the old lookup-table design — no clear win or loss, both designs slightly below plain unconditioned CLIP at this scale (an already-known pattern, not new).

Interpretability — the actual point of the redesign — failed outright:

| | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| Effective dimensions (of 16) | 1 | 1 | 1 |
| Silhouette score | 0.12 | 0.31 | −0.13 |
| `warmth` probe | weak positive | null | positive |
| `register` probe | null | null | null |

Nearly every sample ended up with almost the same condition vector, regardless of content — clearly worse than the old table, which reliably encoded both proxy properties.

---

## Why the built-in health check missed it

The one health-check metric this design shipped with, **usage entropy** (how spread out each sample's soft attention is over the 16 prototypes), looked reassuring — 92–93% of maximum spread, "not concentrated."

But on **88–99.5% of all 150,000 samples**, the single *strongest* prototype match was the same one or two prototypes. Because the softmax stayed close to uniform everywhere, the final blended output ended up nearly identical for nearly every sample anyway.

**Lesson:** a spread-out-attention metric and a diverse final output are not the same thing — this project's original monitoring only checked the first.

---

## Root-cause fix: a hidden 1,000× learning-rate gap

The prototype bank's own parameters were quietly being trained **1,000× slower** than the old per-sample table — an optimizer-configuration gap, not a flaw in the idea itself.

After giving the bank its own, correctly-scaled learning rate (plus a tunable starting "sharpness" for its attention), a quick single-seed sweep showed a clear, monotonic trend as the learning rate increased: entropy sharpened, attention concentration spread across more prototypes, and silhouette roughly quadrupled.

---

## Best setting, 3-seed confirmed: real interpretability gains

| | original design (n=3) | best fix: `temp=0.3`, `lr_prototype=1e-2` (n=3) |
|---|---:|---:|
| Silhouette score | 0.12, 0.31, −0.13 | **0.55, 0.56, 0.70** |
| `register` probe | null, null, null | **positive, positive**, null |
| `warmth` probe | 1 of 3 positive | 0 of 3 positive |

Silhouette moved to consistently positive and much higher across all 3 seeds — the condition space broke out of its 1-dimensional collapse. `register` became recoverable for the first time in this whole investigation. `warmth`'s one prior weak positive signal disappeared, a smaller real regression alongside the gain.

---

## ...bundled with a serious, seed-replicated retrieval cost

| oracle i2t Recall@1 | value |
|---|---:|
| Original prototype design | 16.8 |
| Best fix (`temp=0.3`, `lr_prototype=1e-2`) | **10.4** |
| No conditioning at all (raw CLIP) | 17.8 |

At this operating point, conditioning is **actively worse than doing nothing** on image-to-text retrieval — a real, tight drop, consistent across all 3 seeds, not seed noise. Text-to-image retrieval is unaffected either way.

**Bottom line:** this is a genuine, seed-replicated *trade-off*, not a finished fix. The learning-rate diagnosis is real and the interpretability problem is fixable in principle, but this exact setting is not yet usable as a replacement for the old design.
