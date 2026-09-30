# Joint-fusion brainstorm: independent findings

## Bottom line

The proposed shift is intellectually useful, but it does not by itself escape
the failure mode already observed.  Learning a third embedding from two graphs
still asks one geometry, and then one mutual-kNN graph and one partition, to
make incompatible local relations simultaneously close.  If an affect-positive
pair is deliberately cross-genre while a content-positive pair is not, a
single distance cannot satisfy both without either merging their neighborhoods
or allocating dimensions whose eventual kNN construction ignores one source.
Replacing a hand-set fusion weight with a learned loss balance does not remove
that conflict.

Consequently, I would not make SNF or co-regularized spectral clustering the
next substantial pilot.  Their premises are shared/reinforcing neighborhood
structure; the measured 98.96% empty mutual-edge intersection is unusually
direct evidence against that premise.  The one narrowly worthwhile next test
from this list is a **linear, regularized CCA audit on a train/held-out split**,
not because CCA is likely to yield a usable fused graph, but because it can
cheaply measure whether there is any stable shared continuous signal below the
hard mutual-kNN threshold.  It should be a go/no-go diagnostic, not a fishing
expedition over kernel and dimension choices.  If it finds little held-out
correlation, that is strong additional evidence that a shared representation is
the wrong target.  If it finds stable correlation, it only licenses a small
and tightly controlled learned-student pilot; it does not demonstrate that a
flat joint community graph can make a Pareto improvement.

My prior is that none of these will beat hierarchical refinement on the stated
two-AMIs objective.  The hierarchical pilot at least uncovered real,
non-granularity-driven affect signal (emotion AMI 0.1072 versus 0.0362 for the
matched random-split control).  Its failure was not absence of affect signal;
it was that splitting almost the entire data set (99.8% of nodes) into often
small children severely damaged the genre partition.  A global shared latent
has no hard protection against the even more damaging operation already seen:
cross-content merges.

## 1. Family framing: mostly correct, with an important separation

The first two user-named candidates do collapse into one scientific family
when cross-attention is merely an encoder inside a two-teacher contrastive
student.  In that formulation the important choice is the **objective**:
positive pairs from the content graph and affect graph both constrain one
student similarity space.  Cross-attention, concatenation followed by an MLP,
summed projected views, or a gated residual are alternative parameterizations
of that same hypothesis.

Cross-attention is not load-bearing for one fixed-length content vector and
one fixed-length affect vector per node.  Attention earns its complexity when
there are many tokens, variable-length evidence, or a need to align one
element of one sequence with a different element of another.  Here ordinary
two-vector attention reduces largely to a learned, data-dependent gate or
bilinear interaction.  A norm-balanced pair of small projection heads plus a
gate can express the useful part, exposes its modality weights for diagnosis,
and has fewer degrees of freedom with which to hide dominance by content.
Cross-attention might be independently meaningful only if the inputs become
structured sets--for example image patches attending to emotion-bearing caption
tokens, or several emotion-label/posterior tokens attending to visual regions.
That would be a new claim about *where* affect is expressed, requiring evidence
and an image-side inference story.  It is not justified by the present two
global-vector graph result.

There is a second, distinct cross-attention use that should not be conflated
with fusion: cross-view **edge/message passing**, where a node aggregates
content-neighbors and affect-neighbors into separate channels and preserves
both channels to the output.  That is a multiplex representation, not a
single fused embedding unless a later forced projection discards the separation.
It could be useful for a downstream multi-head task, but it does not solve the
current one-flat-community-label problem and should not be sold as a direct
fusion rescue.

## 2. Method assessments and falsifiable tests

### Learned two-teacher student (simple projections first; cross-attention optional)

**Mechanism and applicability.**  One can sample content-graph and
affect-graph positive pairs and optimize separate InfoNCE-style losses in a
student embedding.  This is genuinely different operationally from edge union:
it can use all positive-pair statistics, learns a continuous metric, and can
possibly use soft similarities rather than only retained mutual edges.  But it
still assumes that there is a metric in which both teachers' positives are
simultaneously local.  Near-orthogonality does not prove that no such metric
exists; hard mutual-kNN can discard shared lower-ranked structure.  It does
make that assumption costly and empirically doubtful.  In particular, the
teacher edge counts (711,878 content versus 620,608 affect) do not make their
losses equally easy: content's much more genre-coherent neighborhoods give a
more globally consistent set of constraints, while affect positives may form
many cross-cutting constraints.  Equal scalar loss weights are not equal
geometric pressure.

**Dominance/collapse risk.**  The likely pathology is not necessarily literal
constant-vector collapse--standard negative-pair contrastive loss normally
resists that--but *modal collapse*: the representation preserves content and
treats affect as noise, or learns a compromise that erases the sharp content
boundaries.  The reverse is also possible if loss scaling or temperature makes
affect gradients dominate.  Cross-attention adds a particularly convenient
failure route: a gate can saturate near one view while the nominal objective
continues to report an average loss.  A good aggregate loss is therefore not
evidence of joint information.

**Predeclared convergence and collapse criteria.**  Borrow DEC's discipline,
but evaluate each teacher separately on fixed, held-out edge and non-edge sets.
At each checkpoint report: (1) held-out content-edge and affect-edge retrieval
recall/mean rank in the learned space, each compared with its own input view;
(2) both edge types' similarity distributions versus matched random pairs;
(3) content and affect teacher losses and gradient-norm shares; (4) effective
embedding rank/covariance spectrum, per-dimension variance, and nearest-neighbor
overlap with each teacher; and (5) the learned gate distribution, if gated.
Use early stopping only after both held-out teacher retrieval measures plateau
for a fixed predeclared window, rather than after the summed training loss
plateaus.

Call it collapsed if either held-out teacher has no nontrivial separation from
random pairs, if its neighbor recall returns to approximately its random
baseline while the other improves, if one teacher consistently receives a
near-zero gradient contribution or the gate saturates near zero/one for most
nodes, or if the embedding's effective rank collapses.  Call it merely a
compromise, rather than success, if it retains both edge recalls but the final
mutual-kNN graph again merges content communities and produces only a genre /
emotion trade-off curve.  A meaningful success requires all of: reproducible
held-out lift over content-only for affect-edge retrieval, no material
content-edge retrieval loss, a nondegenerate contribution from both views,
and a final partition that clears predeclared two-metric thresholds relative
to the content and hierarchical baselines.  Do not select loss weights based
on label AMI; labels may evaluate the final shortlist but must not tune it.

**Worth building / falsification.**  It is worth a minimal pilot only if the
CCA audit below establishes stable out-of-sample shared signal, or if soft
cross-view rank correlation demonstrates a substantial shared candidate set
missed by mutual-kNN.  Start with linear projections and a scalar/vector gate;
cross-attention is an ablation after that baseline, not the first build.  It is
falsified as a route to a shared graph if every reasonable, predeclared balance
either reproduces content-only (affect edge retrieval absent) or yields the
same merge-driven genre loss as union.  Sophisticated architecture cannot turn
inconsistent positive constraints into a jointly local relation.

**Relation to PercepT and project scope.**  This is a scoped exception only if
the research question changes from “can a training-light buddy seed produce a
useful initialization?” to “does ArtELingo contain a learnable shared
content-affect latent useful for a separate topic artifact?”  It is not a
replacement for the original buddy-init motivation.  PercepT's latent is
reconstruction-grounded and DEC-shaped over many epochs; a two-teacher
contrastive student likewise introduces an ongoing representation-learning
stage.  The earlier analysis correctly identified that buddy init cannot
functionally substitute for that.  The exception is reasonable after the
training-free failures, but it should be evaluated offline and decoupled from
retrieval conditioning.  Otherwise this project risks rebuilding a weaker,
less well-anchored DEC analogue while still expecting it to solve retrieval.

### Similarity Network Fusion (SNF)

**Mechanism and applicability.**  SNF alternates within-view local diffusion
with cross-view replacement/averaging.  It is useful when two views contain
partly noisy observations of a common manifold: paths endorsed by one view can
be reinforced when the other supplies compatible local support.  Known node
correspondence makes the operation straightforward here, but does not create
the missing compatible support.  The almost-empty edge intersection says the
two sparse local transition operators mostly send probability mass to different
neighbors.

**Concrete prediction.**  In this near-orthogonal case, after the first update
each view will inject a smeared version of the other's neighborhoods.  Repeated
diffusion either (a) mixes those alternatives into a broader, denser
similarity whose strongest blocks are dominated by whichever view has more
coherent/high-conductance structure (likely content), or (b) creates bridge
paths through affect edges that reduce separation between content blocks.
The fused matrix should look like a smoothed union--more off-block mass and
less content-block contrast--not like a newly discovered sparse shared
backbone.  SNF's local normalization may change the exact frontier relative to
raw union, but it has no mechanism to distinguish a useful cross-view bridge
from the harmful cross-content bridge already demonstrated.

**Worth building / falsification.**  A tiny, fixed-parameter diagnostic could
be justified only if it first reports a materially increased *shared* local
neighborhood concentration without reducing content-block conductance.  A
success would be a fused affinity with substantially more high-confidence
pairs supported by both views (not merely nonzero diffusion mass), stable
content-community boundaries, and an emotion gain.  It is falsified if its
edge/block statistics interpolate between the two inputs, if content
communities merge as in union, or if its metric curve is no better than a
normalized weighted-union control.  My expectation is falsification.  SNF is
therefore a low-priority control, not a next direction.

### Co-regularized multi-view spectral clustering

**Mechanism and applicability.**  This learns spectral embeddings whose
columns are encouraged to agree across the two Laplacians.  It differs from
SNF in optimization form, but its shared-eigenspace penalty encodes the same
substantive prior: meaningful cluster indicators should be approximately common
to the views.  When natural low-frequency eigenvectors are unrelated, a strong
co-regularizer rotates both away from their good individual eigenspaces toward
a compromise; a weak one returns two near-separate spectral clusterings.

The method becomes meaningfully different from separately clustering and
concatenating labels only if the views have stable partially aligned eigenspaces
at the cluster scale--for example, principal angles between their leading
nontrivial eigenvector subspaces are small, or cross-view agreement is higher
within prospective blocks than between them.  Node correspondence alone is
not enough.  Concatenating independently obtained embeddings preserves two
factors; co-regularization deliberately sacrifices factor-specific directions
for common ones.

**Worth building / falsification.**  Before optimizing it, measure leading
eigenspace alignment and compare it to node-permuted affect controls.  If the
observed principal-angle/canonical-correlation signal is indistinguishable
from the permutation control, do not build it.  If there is a clearly
nonrandom shared subspace, a small sweep with a predeclared regularization grid
could test whether it gives a block structure that improves emotion without
content merging.  It is falsified if increasing regularization smoothly erodes
each view's own spectral quality, or if results equal a weighted-Laplacian
baseline.  Given the edge evidence, I regard this as a likely dead end.

### Linear CCA (kernel CCA explicitly deferred)

**Mechanism and applicability.**  CCA asks the right cheap diagnostic question:
is there a continuous linear subspace of CLIP content and GoEmotions features
that is reproducibly correlated across the same paintings?  This uses node
correspondence without assuming matching neighbor identities, so low kNN
overlap does not automatically rule it out.  That is its value.  It does not,
however, establish that the correlated component is emotion-relevant or that
it supports a good joint cluster graph.  It may be generic caption/image
semantics, language, image quality, or a small number of broad variables.

With about 61k examples, ordinary low-dimensional, regularized linear CCA is
not inherently a severe sample-size overfitting problem, provided dimensions
are modest and preprocessing is fit only on training data.  The danger is
selection and flexibility: trying many feature normalizations, components,
kernels, and graph constructions until an evaluation label moves converts a
large sample into a spurious research result.  Kernel CCA is especially risky:
its flexible feature map can produce impressive in-sample correlation from
nonlinear nuisance structure and has unfavorable memory/compute behavior at
this scale.  It should not be the first CCA experiment.

**Predeclared validation.**  Split by the already available ArtELingo
train/held-out division (and, if captions/languages have repeated structure,
respect the existing grouping rules).  Fit PCA/whitening and regularized CCA
only on train.  Predeclare a small component count and ridge grid selected by
training-only inner validation; transform held-out examples exactly once.
Report held-out canonical correlations for every retained component with
confidence intervals or a node-permutation null, plus stability of canonical
directions across train resamples.  Crucially, evaluate whether CCA-space
nearest neighbors preserve each source's held-out edge retrieval, rather than
only reporting correlation.  Keep genre/emotion AMI as a final, one-time
external assessment after selecting the dimensionality without labels.

**Worth building / falsification.**  This is the recommended next action as an
audit.  It is promising only if several components have material,
permutation-separated held-out correlation and stable directions, and a simple
CCA representation retains a nontrivial amount of both teacher neighborhoods.
It is falsified if held-out correlations collapse toward the permutation null,
are concentrated in one fragile component, or the CCA graph simply reproduces
content-only/union behavior.  Even a positive CCA result would justify testing
a simple gated student, not kernel CCA or cross-attention by default.

### A different classical option: partial/conditional alignment, not another fusion

The one classical alternative worth naming is **conditional (or partial) CCA /
residualization**: regress the affect representation on the content
representation (on train only), then ask whether the residual affect component
is stable, predicts held-out affect relationships, and has useful structure
*within* content neighborhoods.  This is not a route to one global joint graph;
it is an audit of the conditional-dependence opening that motivated hierarchy.
It can distinguish “there is no shared signal” from “there is affect signal
that is specifically orthogonal to content.”  The latter is exactly the case
where forced alignment is wrong but multi-head or conditional outputs may be
valuable.  It should be assessed with the same held-out and permutation
discipline, not optimized against labels.

Joint diagonalization of graph Laplacians is another possible name, but it is
mathematically close to co-regularized spectral clustering and inherits the
same common-eigenbasis assumption.  It is not a genuinely promising new bet.

## 3. Ranked recommendation

1. **Build only a regularized linear CCA + conditional-residual audit first.**
   This is a minimal combination because it answers two distinct questions
   before committing to a learned fusion system: is there stable shared signal,
   and is the useful affect signal instead conditional/orthogonal to content?
   It is cheap, admits clean held-out falsification, and its negative result
   saves a great deal of architecture work.  It must not be framed as a
   candidate final community method.

2. **Only on a strong audit pass, test the simplest two-teacher gated student.**
   Use predeclared balance and the per-teacher convergence/collapse metrics
   above.  Do not begin with cross-attention.  The final graph must beat the
   already observed baselines under a predeclared two-metric rule, not merely
   produce attractive embeddings or reduced contrastive loss.

3. **Defer SNF and co-regularized spectral clustering.**  If either is run at
   all, it should be a small diagnostic/control after demonstrating nonrandom
   common local/eigenspace structure.  Neither deserves a substantial search.

This ranking also preserves the most important lesson of the existing work:
first identify whether the relevant information is shared, conditional, or
separate; only then choose an object to optimize.  The requested methods are
all poor fits if the answer is “separate.”

## 4. Explicit dead ends and the data conditions required for success

At present, full SNF, strongly co-regularized spectral clustering, kernel CCA,
and cross-attention-first fusion are dead ends or near-dead ends.  Their
technical sophistication does not supply a mechanism for resolving an actual
contradiction between positive relations.  SNF needs locally reinforcing paths;
co-spectral methods need a common low-frequency structure; CCA needs stable
cross-view covariance; a contrastive student needs jointly satisfiable
positive-pair geometry.  The mutual-kNN evidence is not a perfect test of all
four requirements, but it is a strong warning against assuming them.

For any shared-fusion method to help, at least one of the following must prove
true in held-out diagnostics:

- The sparse mutual graphs hide substantial agreement in softer ranks or
  similarities, concentrated inside the same content blocks rather than on
  cross-content bridges.
- A stable continuous shared subspace exists and preserves meaningful local
  neighborhoods from both views; it is not merely caption language, generic
  semantic quality, or another nuisance axis.
- Affect distinctions are conditionally useful within content-near candidate
  sets, so a learned representation can improve local selection without
  relaxing the content boundary that protects genre.
- The evaluation's desired “emotion” structure is genuinely compatible with
  genre at the intended partition scale.  If emotions systematically span
  genres, a single flat community ID is a representational mismatch, regardless
  of optimizer.

Absent this evidence, the honest next research move is not a more elaborate
joint graph.  It is either to improve the conditional/refinement formulation
so it does not split nearly every content parent indiscriminately, or to change
the output to two linked representations--content communities plus affect
facets/topic scores--and stop demanding that one partition be both.  That is
not a defeat; it follows directly from the observed orthogonality.
