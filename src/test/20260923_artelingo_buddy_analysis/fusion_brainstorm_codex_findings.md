# Fusion-mechanism brainstorm: independent findings

## 1. What the results say structurally

I agree with the central diagnosis, with one qualification.  The evidence is
strong that content-neighborhood structure and affect-neighborhood structure
are very nearly orthogonal *at the scale and representation at which these
graphs are currently built*.  The almost empty intersection is especially
important: it is not merely that two encoders disagree about the strength of
the same local links; they select overwhelmingly different local relations.
In that situation, a union graph gives Leiden conflicting positive evidence
about which nodes belong together, while feature fusion makes that conflict
implicit in every similarity calculation.  Both naturally produce a
compromise partition.

The qualification is that the AMI numbers alone do not prove the human genre
and emotion labels themselves are independent.  They prove that GoEmotions
as mean-pooled per-painting features is almost uninformative for genre, and
that its mutual-kNN relation barely coincides with the content relation.
There could still be useful conditional dependence--for example, particular
emotions may be distinguishable within a genre even if they cut across genres
globally.  That conditional case is precisely the narrow opening for a
hierarchical scheme.  Conversely, if affect neighborhoods really cross genre
boundaries in the way the union result suggests, a single flat label cannot
be both a good genre identifier and a good cross-genre emotion identifier.

There is also a graph-construction issue rather than only a clustering issue.
Mutual kNN is deliberately selective and degree-constrained; two spaces can
have useful softer rank information even when their top-20 mutual edges have
almost no overlap.  But that does not make simple weighted union magically
additive: it only means a carefully designed *conditional or local* use of
that softer information remains plausible.

Finally, the desired claim should be framed carefully.  A final label of
`(content community, affect subcommunity)` can preserve or improve genre AMI
relative to the content partition if its splits align with genre detail, but
it cannot be assumed to preserve the exact 0.4384 score merely because it
never crosses a content boundary.  AMI is not monotone under refinement: an
arbitrary split of a cluster can lower (or occasionally raise) AMI after
chance correction.  Genre V-measure and a contingency-table inspection are
needed alongside AMI to tell controlled refinement from metric luck.

## 2. Mechanisms considered

### A. Weighted edge fusion followed by weighted Leiden

Constructing a weighted graph is a worthwhile diagnostic, but I do not view
it as the most promising route to a Pareto-like result.  It is meaningfully
different from early fusion operationally: early fusion changes every
candidate pair similarity and therefore the kNN competition itself, whereas
edge fusion preserves each modality's independently selected local support
and changes only the relative influence of retained edges.  This can make a
better practical frontier possible, particularly because the binary union
gave each affect-only edge the same force as a content-only edge despite the
large difference in genre reliability.

But it retains the essential limitation: one unconstrained global partition
must decide whether to honor cross-content affect edges.  Given that affect
adds roughly another full graph's worth of edges and the overlap is tiny,
shrinking its edge weight mostly approaches content-only; increasing it
mostly reproduces the genre-damaging merge pressure.  The fact that overlap
edges are rare also means assigning them extra weight is unlikely to supply
a substantial shared structural backbone.

This mechanism is falsified as useful if its Pareto curve is smooth and
monotonic from content-only to union-like behavior: any material emotion lift
then costs a comparably material genre decline.  It becomes worth retaining
if a low affect weight produces a reproducible emotion lift (for example,
well beyond run-to-run/seed variation and materially above 0.0593) while
genre remains near the content baseline, or if a clear knee dominates the
existing union point.  Sweep modularity resolution as well as affect weight;
otherwise a change in effective edge mass can be mistaken for a modality
effect.  Compare against degree/total-weight normalized variants, since raw
edge counts can otherwise decide the result.

### B. Content-first, affect-within-content hierarchical refinement

This is the best next mechanism to test.  It changes the constraint set,
rather than only changing a mixture coefficient: affect is prohibited from
creating a merge across a content community.  Its final composite community
label remains a single integer per node after relabeling, so it fits the
current pipeline while expressing a nested partition.

The structural bet is modest and testable: affect may have useful local
variation *conditional on content community*, even though it is unusable as
a global organizer.  If true, its splits can increase emotion homogeneity
without letting cross-genre affect relations destroy content separation.
This does not recover an emotion class spanning several genres under one
community ID--indeed it deliberately cannot--but that is the price of
protecting genre under the single-label constraint.

Important failure modes are substantial.  Content Leiden communities may not
be genre-pure enough; splitting them by affect could fragment a true genre
across many labels and lower genre AMI.  Some communities will be too small
or have too few affect edges to support stable clustering, and separate
Leiden calls create many implicit resolution choices and a multiple-testing
opportunity to overfit a pilot.  More fundamentally, global emotion labels
can cut across genres.  If so, within-community clusters will each contain
only a small slice of a global emotion and global emotion AMI may not rise
much even if the local clusters look semantically sensible.

Use conservative guards: retain the parent label unchanged for small,
sparse, or affect-unstable parent communities; only permit a split when it
has a predeclared minimum size and a nontrivial stability criterion across
seeds or perturbations.  Evaluate the composite labels against both targets,
but also report parent-versus-child refinement statistics: fraction of nodes
split, child-size distribution, per-parent emotion purity gain, and genre
contingency changes.  It is falsified if emotion AMI stays near content-only
while genre AMI falls, or if apparent emotion gains occur only in tiny,
unstable children.  It is compelling if conservative refinement yields a
repeatable emotion increase while genre AMI/V-measure remains close to the
parent partition and the gains occur in sizeable communities rather than
singletons.

### C. Leiden-native constrained refinement

If leidenalg can initialize/refine a partition while forbidding movement
between parent communities, it is an implementation variant of B, not a
different scientific hypothesis.  A native API may be useful for efficiency
and for ensuring the same objective semantics, but it does not avoid the
need for explicit per-parent eligibility and stability rules.  A resolution
parameter alone is not a constraint: raising resolution on a fused or
content graph merely creates more global splits and does not say that affect
is responsible for them.

Therefore, validate the mechanism first using the clearest manual
restriction conceptually (each induced parent subgraph receives affect-only
refinement).  Substitute a Leiden-native restricted implementation only if
it produces the same partitioning rule and avoids accidental parent-boundary
crossing.  It fails as a separate avenue if it cannot enforce that invariant
or changes the objective so much that it no longer matches the manual
baseline.

### D. Content-anchored local re-ranking / affect only as a tie-breaker

There is a second, genuinely different but lower-priority candidate:
construct the content candidate neighborhood first, then use affect only to
re-rank or select among content-near alternatives.  For example, affect
could break ties among nodes within a narrow content-similarity band, while
disallowing candidates whose content similarity is appreciably worse than
the current boundary.  The resulting graph still has one Leiden pass, but
affect cannot introduce arbitrary long-range cross-genre edges.

This targets a plausible weakness of hard top-k selection: content may have
many nearly interchangeable neighbors, and affect could choose the more
emotionally coherent ones at little content cost.  It is not ordinary global
feature fusion because the content band is an explicit eligibility
constraint.  Its weakness is that it may be too weak to add sufficient
emotion signal, especially when content rankings have a sharp margin or
when affect-relevant neighbors are not content-near.  It also has many ways
to tune the band and tie criterion, creating a high artifact risk.

It is worth a pilot only after B, with all thresholds predeclared from the
content-neighbor margin distribution rather than selected for label AMI.  It
is falsified if the re-ranked graph either has near-identical metrics to
content-only or follows the same genre-loss curve as union when the band is
relaxed.  A credible success would be an emotion gain with minimal changes
to the content edge set and no concentration of the gain in repair-induced
edges.

### E. Multi-layer/community-detection objectives

A true multiplex or multi-objective community method is tempting, but under
the requirement of one flat label it is mostly a more elaborate way to set a
global trade-off between layers.  It may provide a better optimizer than
weighted sum, but cannot represent simultaneously separate genre and
cross-genre emotion assignments.  I would not prioritize it until the
hierarchical conditional-dependence hypothesis has failed; otherwise it is
likely to consume effort to trace another trade-off frontier.

### F. Supervised metric learning or label-guided graph adaptation

This is not appropriate for the immediate comparison.  It could learn a
representation that optimizes a chosen composite of known genre and emotion
labels, but then the experiment changes from unsupervised signal fusion to
supervised task construction and introduces leakage/split-design questions.
It is only defensible if there is a clearly separated downstream evaluation
protocol and the scientific question is explicitly changed.

## 3. Ranked recommendation

1. **Test conservative content-first, affect-within-content refinement.**
   Keep the content partition as a hard parent constraint, apply affect only
   inside eligible parents, and emit one composite community ID per node.
   This is the only candidate that directly prevents the demonstrated source
   of genre loss--cross-content-boundary affect merges--while leaving room
   for conditional affect structure.  Pre-register the guards and report
   stability/size diagnostics so refinement cannot win merely by producing
   many tiny labels.

2. **Use weighted, normalized edge fusion as a small calibration baseline,
   not as the main bet.**  It is cheap and scientifically clarifies whether
   the harsh binary-union result is mostly an equal-edge-weight artifact.
   Treat it as a frontier-mapping control; do not keep extending its sweep
   if it exhibits the expected smooth trade-off.

3. **If hierarchical refinement fails but leaves large content-similarity
   ties, pilot content-anchored affect re-ranking.**  It is the cleanest
   alternative that lets affect influence topology only where content has
   already declared candidates plausibly interchangeable.

The first two form a sensible paired experiment because they answer distinct
questions: whether edge strength alone was the union problem, and whether a
hard nested constraint exposes conditional emotion structure.  The
hierarchical result should be compared to a control that splits the same
parents with random or content-only refinement at matched child-size
profiles; without that control, any change in AMI could be attributable to
partition granularity rather than affect.

## 4. Dead ends or low-value pilots

- **More early-fusion scalar-weight sweeps:** low value.  The observed
  collapse is the expected consequence of blending every pairwise metric;
  finer weights are unlikely to alter that geometry.

- **Binary intersection, or near-intersection thresholds:** dead end with
  the present graphs.  The measured overlap is so sparse that connectivity
  repair, not shared evidence, determines the result.

- **Resolution-only tuning of a fused graph:** low value.  It can change
  cluster granularity but cannot protect content boundaries from affect
  merges; any apparent win is especially vulnerable to AMI granularity
  effects.

- **Unconstrained multiplex/global multi-objective clustering as the next
  step:** defer.  It does not solve the representational bottleneck of one
  flat label and is likely another trade-off optimizer in more complex
  clothing.

- **Affect-first then content-within-affect hierarchy:** not worth a first
  pilot.  Affect's near-chance genre organization and the project goal make
  it the wrong hard constraint: it would explicitly forbid content links
  across affect parents and is therefore likely to destroy the reliable
  content structure.

- **Label-supervised fusion/metric learning:** out of scope for this
  unsupervised graph question and too easy to make look successful through
  leakage or objective engineering.
