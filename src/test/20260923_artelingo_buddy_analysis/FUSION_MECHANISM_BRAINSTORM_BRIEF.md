# Brainstorm: a fusion mechanism that boosts emotion without trading away content

This is a **pure thinking task, not an implementation task**. Do not write or
run any code. Do not pause for confirmation — just write your findings to
`src/test/20260923_artelingo_buddy_analysis/fusion_brainstorm_codex_findings.md`
and stop.

## Context

We're building buddy graphs over ArtELingo (WikiArt paintings, real human
genre + emotion labels) to test fusion mechanisms between two signals:

- **Content** (CLIP image + CLIP text): excellent at genre (AMI=0.4384
  against a 1144-painting genre-labelled subset), weak at emotion
  (AMI=0.0593, against majority-vote emotion labels over ~5
  annotations/painting).
- **Affect** (GoEmotions, `SamLowe/roberta-base-go_emotions`, a 28-category
  sigmoid classifier mean-pooled per painting): decent at emotion
  (AMI=0.1180 single-modality ceiling), weak at genre (AMI=0.0396 — barely
  above chance). This near-chance genre score is important: it means content
  and affect structure are close to orthogonal organizing principles over
  this data, not just two noisy views of the same thing.

All graphs are mutual-kNN (K=20) + Leiden community detection (seed=42) +
the codebase's standard connectivity repairs
(`ensure_min_degree`/`ensure_connected`). Every clustering AMI/V-measure
number below is against real ground truth.

## Goal

We want emotion structure to become recoverable **without giving up genre
structure that the content graph already nails almost perfectly on its
own**. Ideally: something close to a Pareto improvement over content-only
(genre stays near 0.4384) while genuinely lifting emotion (well above
0.0593, ideally toward or past the best ceiling found so far, 0.1492).

## What's been tried, and why each one only trades off rather than adding

1. **Early (feature) fusion**: concatenate L2-normalized GoEmotions +
   CLIP-text vectors (with a scalar weight on the affect part), build ONE
   mutual-kNN graph in that combined space, sweep weight in {0, 0.5, 1, 2,
   4}. Best point (weight=4): emotion AMI=0.1160, genre AMI=0.0867 — genre
   collapsed to ~20% of its content-only value. Diagnosis: a single shared
   metric space makes every pairwise similarity a blend of two signals; there
   is no way for a pair to be "close for content reasons" without that also
   uniformly pulling in the affect term, and vice versa. The weight is a
   single global dial with no way to be selective about *where* affect gets
   to influence neighbor selection.

2. **Late (graph/edge) fusion, union**: build A_content and A_affect fully
   independently in their own native spaces (each already fully repaired /
   connected on its own), then `E = union_graph(A_content, A_affect)`
   (edge exists if either graph has it), then ONE Leiden pass on the merged
   edge set. Result: emotion AMI=0.1236 (actually beats the GoEmotions
   single-modality ceiling), genre AMI=0.1394 (much better than early
   fusion's 0.0867, but still only ~32% of content-only's 0.4384). Diagnosis:
   avoiding feature-space competition helped a lot (this is the best
   emotion/genre trade-off point found so far), but Leiden still computes
   ONE global partition of the merged graph, and any edges added between
   nodes that are content-similar-but-genre-different-yet-affect-similar can
   still merge two genre-pure communities into one, or split a genre-pure
   community across affect lines. Adding an entire second graph's worth of
   edges (620,608 affect edges on top of 711,878 content edges — nearly
   doubling total edges) is still a large, global structural perturbation.

3. **Late fusion, intersection**: `A_content ∩ A_affect`. Degenerate as
   expected — 98.96% of nodes had zero edges before repair (the two graphs'
   real neighbor structure barely overlaps at all, consistent with affect's
   near-chance genre score), so its metrics are dominated by the repair
   heuristic's synthetic bridge edges, not real shared structure. Confirms
   the two signals are close to orthogonal, not just under-fused.

## The structural pattern across all three results

Every attempt so far forces the SAME final object — one Leiden partition
over one graph — to simultaneously satisfy two objectives whose ground-truth
labels are nearly independent of each other. A single flat partition is, by
definition, one specific trade-off point along *some* frontier between
"resolve genre" and "resolve emotion" — it cannot represent "both, fully,
at once" if the two signals genuinely don't correlate at the painting level
(which the near-chance 0.0396 genre-on-GoEmotions number suggests they
mostly don't). Weight/threshold tuning (early fusion's scalar weight, late
fusion's binary union-vs-intersection choice) only moves you along that one
frontier — it cannot change the frontier's fundamental shape.

## The question for you

Think from scratch, not just about tuning the two approaches above. We are
NOT asking about a two-bank/dual-prototype-table architecture (separate
prototype vectors seeded from separate content and affect community labels)
— that is a deliberately deferred, bigger architectural change we are
holding off on for now. Constrain your brainstorm to mechanisms that still
produce a SINGLE community-label-per-node output compatible with the
existing buddy-graph → Leiden → community-ID pipeline (or a graph/clustering
preprocessing step that feeds into it), since that is what the current
prototype-seeding step in Experiment 18 actually consumes.

Ideas worth seriously considering (feel free to go beyond these, and feel
free to disagree with any of them if you find a real flaw):

- **Weighted (non-binary) graph fusion + weighted Leiden**: instead of a
  binary edge-exists-or-not union, build a WEIGHTED merged graph (e.g.
  `E_weighted = w_c * A_content + w_a * A_affect`, both as {0,1}-valued
  sparse matrices before the weighted sum, so an edge present in both graphs
  gets a higher weight than one present in only one), and run Leiden with
  edge weights (leidenalg supports this). This turns the binary "union vs.
  intersection" choice into a continuous dial, in graph space rather than
  feature space — sweep it and see whether there's a knee where most of the
  emotion gain survives while genre degrades much less than either fusion
  approach tried so far. Assess whether this is meaningfully different from
  early fusion's weight sweep, or just a graph-space reparameterization of
  the same underlying trade-off curve.

- **Hierarchical / nested clustering: content first, affect refines within**:
  run Leiden on the content graph ALONE first (giving communities that
  should closely match the 0.4384 genre ceiling, since nothing else has
  touched this partition), then, within each resulting community (or each
  community above some minimum size), run a SECOND clustering pass using
  ONLY the affect signal restricted to that community's nodes. Final label
  = (content_community_id, affect_sub_cluster_id). Reasoned argument for why
  this might structurally preserve genre much better than either fusion
  approach: it never merges nodes across different content communities and
  never moves a node out of its content community — it only ever splits
  communities into affect-coherent sub-groups. Interrogate this reasoning
  yourself: does finer partitioning change AMI's chance-correction in ways
  that could make this look artificially better or worse than it really is?
  Does this approach have a real conceptual downside (e.g. small/already-
  fragmented content communities having too few nodes for a meaningful
  affect sub-split, or emotion signal only being usable at all within
  genre-coherent neighborhoods, which might not be true)?

- **Leiden multi-resolution or constrained/restricted partitioning**: is
  there a way to get a similar hierarchical effect natively through Leiden's
  own resolution parameter or a "restrict to an existing partition" API
  (leidenalg supports partial/hierarchical refinement) rather than manually
  re-running Leiden per-community? Would this differ meaningfully from the
  manual hierarchical approach above?

- **Any other genuinely different mechanism** you can think of that doesn't
  fall into "tune a mixing weight" or "the two-bank architecture we're
  deferring." Consider, for example: distance-metric learning /
  re-ranking approaches, a two-pass Leiden where affect only breaks ties
  among near-equally-good content assignments, or anything else you find
  compelling. If you genuinely can't find anything better than the weighted-
  fusion and hierarchical ideas above, say so plainly and explain why, rather
  than inventing a weak alternative for the sake of having more options.

## What to write

In `fusion_brainstorm_codex_findings.md`:
1. Your own independent read of the structural pattern above — do you agree
   with the "orthogonal ground truths force a trade-off" diagnosis, or is
   there a different/better explanation for why both fusion attempts traded
   off rather than adding?
2. Each mechanism you seriously considered, with your honest assessment of
   its promise and its likely failure mode BEFORE anyone runs it (be
   concrete: what result would falsify it being useful, what result would
   make it worth implementing).
3. A clear ranked recommendation: which ONE mechanism (or combination) is
   most worth building and testing next, and why, given everything above.
4. Explicitly flag anything you think is a dead end not worth testing, and
   why, so we don't waste a pilot on it.

Be rigorous and skeptical of your own ideas — this investigation's whole
track record so far has been catching real trade-offs and dead ends (tie-
handling artifacts, non-independent encoder comparisons, degenerate
intersection graphs), not celebrating clean wins. Keep that standard here.
