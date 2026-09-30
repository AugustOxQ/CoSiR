# Can buddy-graph initialization replace PercepT's latent `Z`?

## Bottom line

No. Buddy-graph initialization is a useful *prior* for CoSiR's prototype bank,
but it is not a functional substitute for PercepT's learned latent `Z` plus
DEC. The two methods both begin by turning a vision-language neighborhood
structure into a small set of groups, but that is where the meaningful analogy
ends. CoSiR seeds 16 parameters once and then lets retrieval gradients decide
what those parameters should be. PercepT spends its central training stage
learning a 128-dimensional representation whose explicit job is to reconstruct
the fused input while becoming clusterable. It then freezes the resulting
labels before asking a different model to solve a downstream task.

Experiment 18 is therefore asking one set of 16 shared vectors to serve two
objectives which need not agree: make conditions visibly partition samples, and
make an image-side residual improve image-to-text retrieval. The replicated
silhouette-versus-Recall@1 result is exactly the failure mode I would expect
when no objective protects the former goal from the latter. The higher bank LR
and lower temperature make the bank more willing to specialize; retrieval
gradients then have more power to make that specialization move the image
embedding in directions harmful to the frozen CLIP retrieval geometry. This is
strong evidence for objective conflict, not evidence that the graph seed was
bad.

That said, I would not claim that the missing clustering loss is the *sole*
root cause. CoSiR uses only 16 convexly blended condition values and applies
their learned effect asymmetrically to the image side. PercepT has roughly 67
surviving topics, an explicitly affect-enriched input, and evaluates topic
mapping rather than attempting to improve the same contrastive embedding space
from which topics were built. Capacity, target mismatch, and the one-sided
combiner can all contribute to the observed retrieval penalty. A DEC-like loss
could produce prettier clusters that are still irrelevant—or actively
destructive—to retrieval.

## Where the analogy holds, and where it breaks

The valid analogy is narrow:

- Both start from frozen vision-language features and use unsupervised
  neighborhood/group structure to avoid arbitrary random groups.
- Both seek reusable shared structure rather than a wholly independent vector
  per sample.
- Both can ultimately support a novel sample: PercepT through its mapper and
  CoSiR through `query_proj` plus softmax attention.

But their mechanisms and incentives differ materially.

| Question | CoSiR buddy seed + prototype bank | PercepT P-Topic formation |
| --- | --- | --- |
| What is learned after initialization? | Keys, values, query projection, temperature, and combiner under retrieval loss | Encoder, decoder, and centroids under reconstruction plus cluster self-training |
| Does the representation itself move? | The CLIP query features are frozen; only a small attention/readout mechanism moves | The encoder actively reshapes every point into latent `Z` |
| What preserves input information? | Nothing requires a condition to retain CLIP or affective information | Reconstruction loss anchors `Z` to fused input `h` |
| What creates compact/separated groups? | Initial community means only; thereafter indirect pressure from retrieval | DEC's repeated KL target sharpening over 200 epochs |
| Is clustering coupled to the downstream objective? | Completely coupled: retrieval is the only signal | Deliberately decoupled; downstream mapper learns frozen pseudo-labels |
| Group capacity | 16 soft attention prototypes | 100 initial centers, then about 67 retained topics |

There is a further implementation-level mismatch. Seeding maps community means
through the *current, randomly initialized* `query_proj`, then copies that
mapped vector into both keys and values. The same `query_proj` is subsequently
trainable. Thus even the coordinate system in which the one-time seed is
meaningful is not fixed; ordinary retrieval updates can rotate or distort it.
This is reasonable initialization for an attention module, but it is unlike a
cluster assignment procedure that keeps re-evaluating point-to-center geometry
while jointly updating its encoder.

The initial report's entropy finding makes the distinction vivid. Broad
attention was compatible with nearly identical output conditions because the
weighted prototype values did not differ in a useful way across samples. DEC
directly observes point/centroid assignments and optimizes their sharpening;
CoSiR's usage entropy observes neither output diversity nor cluster validity.
Adding an entropy target alone would therefore be a weak and potentially
misleading approximation to DEC.

## The affect problem may be more fundamental than the clustering problem

Yes: the `warmth` and `register` results are unsurprising if CoSiR is operating
only on ordinary frozen CLIP image/text embeddings. The report already says
`warmth` is largely animal-versus-non-animal content and `register` has little
signal beyond content. A probe that separates those subreddit sets is not an
affect detector, so the present evidence cannot say whether CoSiR encodes
emotion at all.

PercepT makes a design choice CoSiR has not made: it injects a
GoEmotions-tuned text representation into `h`, with an explicit one-third
weight after norm matching. That creates a representational route by which
affective distinctions can survive the initial clustering process, even when
CLIP's factual/semantic similarity would treat them as secondary. No clustering
algorithm can reliably recover a variable that is weakly represented in its
input. Better clustering can sharpen dogs versus buildings; it cannot turn that
division into warmth unless the input has a robust emotion signal and the
labels/test actually measure warmth.

I would prioritize this explanation over fine-tuning prototype temperature for
the affect claim. It does **not** explain the retrieval regression: affect
enrichment could make that regression worse if the retrieval corpus and loss
mostly reward literal caption matching. It does explain why an experiment
aiming for perceptual/affective structure needs new measurement and likely new
inputs before interpreting any silhouette improvement as success.

## New directions

### 1. Decouple perceptual-topic discovery from retrieval conditioning

Build an offline Stage-1 analogue before changing the retrieval model. Fuse
frozen CLIP image and text features with a frozen affect-aware text encoder;
norm-balance the branches; train an autoencoder latent; then use a DEC-style
reconstruction-plus-clustering objective to form a moderate number of topics.
Freeze either hard pseudo-labels or soft topic distributions. Train a separate
predictor from image features (or patches, if available) to predict those
frozen topics. CoSiR may then condition retrieval on the predictor's topic
distribution, but the topic system must be evaluated as its own artifact.

This is the closest principled test of the PercepT lesson. It prevents the
retrieval objective from erasing topic geometry and makes it possible to learn
whether the topics are meaningful before asking them to improve retrieval.

Smallest fast test: use a fixed 100k RedCaps subset and frozen global features,
not patch features. Compare (a) CLIP-only latent and (b) CLIP-plus-affect latent
on held-out cluster stability, output-space silhouette, and a *new manually
audited affect subset*. Do not run retrieval yet. If the affect branch does not
improve held-out affect-aligned discrimination or topic coherence, stop before
building the mapper.

Risks: this is substantial offline training and introduces an external encoder
and its domain/language biases. RedCaps lacks image-level emotion labels, so
weak pseudo-topics can look convincing without being affective. DEC can also
over-sharpen arbitrary partitions; reconstruction, stability across seeds, and
human/coherence checks are necessary counterweights. A topic predictor trained
on global embeddings is a weaker vision-only test than PercepT's patch mapper.

### 2. Make retrieval conditioning explicitly safe and topic-preserving

Keep the prototype architecture, but turn it into a multi-objective model with
a retrieval-preservation constraint. Maintain a frozen or slowly updated topic
teacher learned offline (from direction 1, or initially from buddy communities).
Train the online attention assignment to match the teacher's soft topic
distribution, while adding a distillation/identity constraint that keeps the
conditioned image embedding close to the original CLIP image embedding. The
retrieval loss remains, but it no longer has unilateral control of prototype
geometry. Use a gated residual whose initial and regularized magnitude is near
zero, and report the Pareto curve rather than one hand-selected operating point.

This directly targets Experiment 18's trade-off. The topic loss identifies
what must remain separated; the embedding-preservation loss limits how much
the combiner may damage raw CLIP ranking. It is more defensible than an
anti-collapse penalty alone because it supplies a semantic target for diversity.

Smallest fast test: retain the present 16-prototype model and use buddy-derived
soft labels as a temporary teacher. Sweep only the topic-loss and
embedding-preservation weights across one small validation split, plotting
Recall@1 against held-out topic-agreement and condition-space silhouette. A
promising result must contain a point near raw CLIP retrieval with better
topic-agreement than the unregularized bank; otherwise do not scale it up.

Risks: buddy communities may encode only CLIP content, so this can formalize the
wrong taxonomy. A strong identity constraint can make conditioning inert, while
a strong topic constraint may repeat the retrieval loss. The method adds tuning
burden and still cannot demonstrate affect without better targets.

### 3. Pivot: make CoSiR a perceptual annotation/mapping system, not a retrieval enhancer

The cleanest pivot is to stop requiring the same learned condition to improve
Recall@1. Treat CoSiR's frozen CLIP backbone and a new affect-aware branch as
inputs to an explicit perceptual-topic discovery system, then train an
image-only multi-label topic mapper. Its product is interpretable topic scores
for new images: factual, stylistic, and affective facets that can be used for
search filters, dataset analysis, or downstream ranking features. Retrieval can
remain raw CLIP as a protected baseline, optionally consuming topic scores only
in a separately evaluated re-ranking layer.

This is a real change of objective, but it aligns the architecture with the
evidence. PercepT's best mapping head was a simple pooled representation plus
linear multi-label classifier; its more prototype-like cross-attention topic
queries performed worse. CoSiR should not assume the 16-vector attention bank
is the right mapper merely because it is interpretable in principle.

Smallest fast test: construct a modest, human-labeled evaluation slice with
clear affect and style labels (or license an existing affect dataset for
evaluation), train simple linear/small-MLP multilabel heads on frozen features,
and compare CLIP-only against CLIP-plus-affect features. This deliberately
tests the cheapest mapper before attention pooling or retrieval integration.

Risks: it changes the project's success metric and may need a new dataset,
annotation protocol, and careful ethical review of emotion labeling. Models may
infer caption sentiment rather than visual affect. Scores will be domain-bound:
an affect encoder trained on textual emotion may transfer poorly to RedCaps or
to multilingual data. This direction may yield no retrieval improvement at all
because it no longer promises one.

### 4. Separate content, style, and affect rather than clustering their mixture

Use multiple explicit heads/latents: a content branch anchored to CLIP, a style
branch trained against image transformations or curated style data, and an
affect branch anchored to affective text/image supervision. Enforce
cross-branch independence with a decorrelation or adversarial leakage loss,
then cluster only the affect/style branches or their deliberately weighted
combination. Retrieval conditioning should read only a small, gated component
of these factors, leaving factual CLIP content mostly unchanged.

The motivation is diagnostic as much as architectural. Current proxy groups
confound content with the property they purport to test. Factorizing the inputs
creates falsifiable tests: can an affect probe predict affect after controlling
for content category, and does the content head remain useful for retrieval?

Smallest fast test: without end-to-end training, train probes for audited
affect/style labels from frozen CLIP, affect-encoder, and concatenated features;
evaluate within matched content strata (for example, animal images only). If
the affect encoder has no incremental within-stratum value, do not build the
factorized model.

Risks: disentanglement losses are fragile and can merely move information
between heads. The required supervision is precisely what RedCaps does not
offer. Compute, data curation, and evaluation complexity rise sharply, and an
overzealous factorization can discard interactions that genuinely matter to
perception.

## Recommendation and decision criteria

Do not treat a new seeding scheme or a temperature sweep as a PercepT analogue.
They are ablations of retrieval-conditioned attention, not perceptual-topic
learning. The immediate research priority should be a cheap representation
audit: acquire or construct a small evaluation set with valid affect/style
labels, then test whether adding an affect-aware encoder carries incremental
signal after controlling for content. In parallel, direction 1's offline
CLIP-plus-affect topic-discovery pilot can answer whether there is an
affect-relevant latent geometry worth protecting at all.

Only after that result should CoSiR attempt direction 2, using a frozen topic
teacher and an explicit raw-CLIP preservation term. Its success criterion should
be predeclared: a reproducible Pareto improvement over the current bank—topic
quality that exceeds the raw/baseline condition representation while image-to-
text Recall@1 remains statistically close to raw CLIP—not merely an attractive
silhouette. If no such point exists, the honest conclusion is that perceptual
topic mapping and retrieval optimization are different products, and direction
3 is the more coherent research program.
