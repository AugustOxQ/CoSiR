# Round 7, step 3: candidate ideas for design L (from the style source)

> Written 2026-10-11 by the `r7 decide` chat, step 3 of the research loop. Sources: the parked ideas
> (`docs/superpowers/direction_log.md`), our facts brief ([step3_facts.md](step3_facts.md)) and the literature check
> ([2026-11-27_style_sources.md](../../../../../docs/reports/literature/2026-11-27_style_sources.md)). Decision B (step 4) is
> the user's.

## What the three sources say

- **The source is the limit, not the scoring.** Told style × genre over B, seed 42 (R@1 points; condition gain in
  brackets): the label-free image grouping +0.35 [−0.10, +0.79] (gain 0.00, since the told map sends both aspects to
  it); CSD (A1c) +0.134 [−0.350, +0.595] (gain +3.67); label probes for every aspect (D0 told, 2026-10-04 diagnostics)
  +12.65 [+11.83, +13.46] (gain +22.72). D0 also has a genre probe, so it bounds the ceiling check from above.
- **Style and genre labels are only loosely tied:** AMI 0.217 on 148,956 scorer-train rows. A perfect style grouping
  would show a genre AMI near 0.22; CSD shows 0.33 at a style AMI of 0.34. There is room.
- **Placement is the binding side.** Heads are logistic regressions on frozen CLIP features. True styles (27) are placed
  at 60.9% from images and 25.4% from captions; the CSD grouping (17 groups) at 85.1% and 40.4%. A source purged of
  what CLIP carries is, by construction, harder for a CLIP head to place, captions above all.
- **The field has no ready answer.** No published style-against-genre measure exists for any style encoder. The two
  newest content-removing encoders (StyleDecoupler, MegaStyle; 2026 preprints) have no released weights. StyleShot's
  encoder saw WikiArt, so it can serve only as a reference. Removing a content direction label-free has one supporting
  preprint (StyleDecoupler); it measured artist retrieval, not leakage.
- **Everything is cheap:** Leiden 10 s, a head 20 to 36 s, scoring a configuration 20 s, round 1's reader 10 min on
  CPU; CSD and Gram extraction for 61,402 paintings took 754 s on the local GPU.

## Candidate ideas

1. **Take content out of the source (residual style).** Before grouping, remove from CSD (and Gram) what the CLIP image
   embedding predicts: a ridge regression fit on scorer-train paintings, keeping the residual, or StyleDecoupler's
   partial projection with the CLIP image embedding as the content vector. Then Leiden as before. Parked as "the
   30-minute label-free check of a de-genred style grouping". Cost: CPU minutes on cached features. Risk: the residual
   is what CLIP does not carry, so CLIP heads may not place it; CLIP's own style signal (AMI 0.28) goes too.
2. **Group conditionally on content (design L as first defined, grouping research synthesis §4).** Keep CSD; change the
   clustering so that groups cut across the CLIP image grouping: Leiden on CSD's neighbour graph with edges inside one
   image community removed or down-weighted, or a published conditional clustering (Gondek and Hofmann; kernel
   conditional clustering, KAIS 2019). Our only internal splitting success was conditional. Cost: CPU, a few hours of
   code. Risk: forcing independence from content also cuts the true style–genre tie, and the same placement risk.
3. **A new source (a wider screen).** Add encoders we can run label-free: ALADIN (Behance, weights released), DINOv2-L,
   SigLIP 2, and texture statistics on CLIP or DINO patch tokens (Gram without layout; no study found, so new). Cost:
   about an hour of GPU extraction plus loaders. Risk: no published evidence that any of them separates style from
   genre; more sources screened means more selection to disclose.

The ideas combine: the source screen can hold any mix. Every option starts with the ceiling check (told style labels,
genre still on the label-free image grouping) and is held to its matched B′ (C2).
