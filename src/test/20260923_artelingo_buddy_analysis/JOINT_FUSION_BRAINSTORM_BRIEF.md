# Brainstorm: learned and classical joint-fusion mechanisms beyond graph union

This is a **pure thinking task, not an implementation task**. Do not write or
run any code. Do not pause for confirmation — just write your findings to
`src/test/20260923_artelingo_buddy_analysis/joint_fusion_brainstorm_codex_findings.md`
and stop.

## Context

Read `fusion_brainstorm_codex_findings.md` and
`hierarchical_refinement_pilot_report.md` in this directory first — this
brainstorm builds directly on both.

Recap of the evidence so far:

- Content (CLIP img+txt union) and affect (GoEmotions) mutual-kNN graphs are
  close to orthogonal at the node-neighbor level: their raw edge
  intersection left 98.96% of nodes with zero shared edge before repair.
  Genre AMI on GoEmotions-only is 0.0396 (near chance).
- Every mechanism tried that combines them into ONE shared adjacency
  structure or ONE shared feature space — early (feature) fusion, late
  (edge) union, hierarchical content-first refinement — trades emotion
  against genre rather than gaining on both. We independently verified the
  late-union mechanism directly causes specific community MERGES (28 content
  communities collapsed to 21 once affect edges were unioned in; the
  smallest community grew from 6 to 324 members), not a smooth dilution.
- Hierarchical refinement (real, control-verified affect signal, but genre
  cost dominated by partition granularity, not by affect specifically) is
  the current best-understood mechanism, but still not a Pareto win.

The user's newest hypothesis: maybe the problem isn't *how* we merge the two
graphs (union vs. weighted union vs. hierarchical splitting) — it's that
forcing content-structure and affect-structure into any ONE shared
adjacency/partition object at all is the wrong move, given they're close to
orthogonal. Instead: build both graphs independently (as already done), then
**learn or extract a genuinely joint/third representation FROM the two
finished graphs**, rather than operating on their edges directly.

## Two candidate families to assess (the user named three candidates; the
## first two collapse into one family — confirm or correct this framing)

### Family A — Learned joint fusion (cross-attention + contrastive/distillation objective)

A small trainable encoder (or pair of encoders) that takes each node's
content representation and affect representation and learns a fused
embedding, trained via a contrastive/dual-teacher objective: sample positive
pairs from content-graph edges (content teacher) and separately from
affect-graph edges (affect teacher), train the student embedding so that
each teacher's positive pairs stay close in the learned space (e.g.
InfoNCE-style, one loss term per teacher), with cross-attention as one
possible architecture for how the student combines the two input views per
node. The final buddy/community graph would then be built by mutual-kNN on
this learned embedding, same as every other pilot's final step.

Assess concretely:
- Is cross-attention actually load-bearing here, or would a much simpler
  student architecture (e.g. two small MLPs projecting each view into a
  shared space, summed or concatenated) achieve the same thing with far less
  risk? Cross-attention implies per-node, per-pair learned weighting; is
  there a real mechanism-level reason to expect that over a simpler shared
  projection, given only two input views (not many, unlike typical
  cross-attention settings with variable-length sequences)?
- What is the collapse risk with ~61,402 nodes, two views, and a purely
  self-supervised (graph-structure-only) contrastive objective? Is there a
  real risk the student just learns to satisfy whichever teacher's positive
  pairs are "easier" (recall content's own kNN graph is much denser/more
  internally consistent than affect's, per the graphs' own edge counts:
  E_content.nnz=711,878 vs. E_affect.nnz=620,608 despite affect's kNN being
  built with the same K) and effectively reproduces one of the fusion
  pilots' trade-off curves again, just via a trained loss weight instead of
  an explicit hyperparameter?
- Given the DEC pilots are this investigation's only precedent for a
  properly-converged small trained model (with a predeclared convergence
  criterion and collapse detection), what would the equivalent convergence
  and collapse-detection criteria look like here? Be concrete, not
  hand-wavy: what diagnostic would tell us this collapsed vs. genuinely
  learned two-source structure?
- Relation to the project's own history: this pushes back toward something
  resembling the PercepT paper's learned latent Z + DEC mechanism, which the
  whole ArtELingo investigation was originally launched to test buddy-init
  as a *replacement* for (see the very first brainstorm in this
  investigation's history, `src/test/20260922_percept_brainstorm/`). Is that
  an argument against this family (defeats the original motivation for
  buddy-graph-based, training-light initialization), or is it a reasonable
  scoped exception given everything else tried so far has failed to find a
  training-free win? Give your honest view.

### Family B — Classical manifold/spectral alignment (no gradient training)

Since both graphs share the exact same node set with KNOWN correspondence
(painting *i* is the same entity in both graphs — this is not the harder
unlabeled-correspondence manifold-alignment problem the literature usually
targets), assess methods that operate on this easier setting:

- **Similarity Network Fusion (SNF)**: iterative cross-view diffusion of
  each graph's similarity/kernel matrix, averaged into one fused similarity
  matrix after a few iterations, then clustered (e.g. spectral clustering).
  Given SNF's mechanism is amplifying weak-but-consistent cross-view
  agreement, and the measured cross-view agreement here is very low (98.96%
  empty intersection), assess honestly: is SNF likely to do meaningfully
  better than plain union, or likely to converge to something similar?
  What's the concrete falsifiable prediction — what would SNF's fused
  similarity matrix look like in the near-orthogonal case, mechanically?
- **Co-regularized multi-view spectral clustering**: an alternating
  optimization for a shared spectral embedding constrained to stay
  consistent with both graphs' Laplacians. Assess the same orthogonality
  concern: does the co-regularization term have any effect if the two
  Laplacians' natural eigenvectors are nearly unrelated to begin with? Under
  what condition would this method produce something meaningfully different
  from just running spectral clustering on each graph separately and
  concatenating the results?
- **(Kernel-)CCA**: find linear (or kernelized) projections of each view
  that maximize cross-view correlation. Given the known one-to-one
  correspondence here, this is the cheapest of the three to actually try.
  Assess: with near-orthogonal views, would CCA mostly find noise
  correlations (since it's explicitly optimizing to find *whatever*
  correlation exists, even if weak/spurious at this sample size), and is
  there a real risk of overfitting a spurious shared axis given ~61k samples
  and each view's own dimensionality? What predeclared validation would
  catch that (e.g. train/held-out CCA correlation checked on the already-
  available held-out ArtELingo split)?
- Is there a genuinely different classical method not listed here worth
  naming?

## What to write

In `joint_fusion_brainstorm_codex_findings.md`:

1. Confirm or correct the Family A/B framing above — do cross-attention and
   contrastive alignment really collapse into one family, or is there a
   meaningful independent version of cross-attention worth separating out?
2. For each concrete method assessed (at least SNF, co-regularized spectral
   clustering, CCA, and the learned cross-attention/contrastive-teacher
   approach), give an honest promise/risk assessment BEFORE anything is
   implemented: what result would make it worth building, what result would
   falsify it, and — critically, given the measured near-orthogonality — is
   this method's core mechanism even applicable when the two views mostly
   disagree, or does it implicitly assume more cross-view agreement than we
   actually have?
3. A clear ranked recommendation: which ONE method (or minimal combination)
   is most worth building and testing next, and why, given the full history
   of this investigation (three failed flat-fusion attempts, one partially-
   successful-but-not-Pareto hierarchical attempt).
4. Explicitly flag anything you think is a dead end given the orthogonality
   evidence, and why — don't let architectural sophistication substitute for
   a real mechanism-level reason to expect success here.
5. If, after honest analysis, you think NONE of these are likely to beat
   hierarchical refinement's result, say so plainly and explain why — and
   say what (if anything) would need to be true about the data for any of
   them to help, so we know what we'd be betting on.

Be as rigorous and skeptical as the prior brainstorm — this investigation's
track record is catching real trade-offs and dead ends, not celebrating
sophistication for its own sake.
