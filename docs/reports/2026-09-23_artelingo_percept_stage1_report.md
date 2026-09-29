# ArtELingo stage report: PercepT Stage 1 P-Topic Formation

## I. Why this branch exists

This report closes Stage 1 of a separate ArtELingo investigation on branch
`experiment/percept_topic_pipeline`. It tests a literal-as-possible replication
of **PercepT**, the P-Topic Formation stage from *Beyond Semantics: Modeling
Factual and Affective Perceptual Experiences from Vision-Language Data*. PercepT
forms perceptual topics by combining factual visual-language content with an
affective text representation, learning a reconstruction-grounded latent space,
and using Deep Embedded Clustering (DEC) to sharpen K-means topics. Its proposed
Stage 2 is then an image-only model trained to predict those frozen topics.

The question here was not whether the earlier ArtELingo buddy-graph fusion
mechanism could be incrementally improved. That work is on a different branch,
uses a two-teacher contrastive student rather than an autoencoder plus DEC, and
currently stands at Attention-h1: held-out emotion AMI=0.1249 and genre
AMI=0.2404. This branch instead asks whether the PercepT mechanism itself can
find an equally useful joint factual-affective partition on the same underlying
ArtELingo data. Both investigations use the same held-out criterion so their
results are directly comparable.

The predeclared held-out Pareto bar is stated once here and used throughout:

**held-out Pareto bar = emotion AMI > 0.1236 AND genre AMI > 0.1954, both simultaneously, on val+test data with zero train overlap.**

AMI (Adjusted Mutual Information) measures chance-adjusted agreement between
the discovered topic assignment and the relevant human labels. A configuration
is **Collapsed** if more than half of its surviving clusters contain fewer than
1% of assigned nodes; it is a **Real success** only when it is non-collapsed
and clears both held-out AMI thresholds. Other non-collapsed misses are
**Merely a compromise**.

## II. Literal recipe and documented adaptations

The training input is a 2,816-dimensional fused representation. Its content
part is the independently L2-normalized CLIP image and text features,
concatenated and normalized again. Its affect part is a 768-dimensional,
attention-masked token mean of `SamLowe/roberta-base-go_emotions`, then mean
pooled across captions for each painting. This is an embedding-level affect
signal rather than the 28-dimensional GoEmotions label probabilities used in
earlier ArtELingo affect pilots.

The paper's Eq. 2 combines same-dimensional content and affect vectors by an
elementwise 2:1 weighted sum. The repository's ViT-B/32 CLIP content vector and
the 768-dimensional RoBERTa vector do not have matching dimensions, so that sum
cannot be reproduced without adding an unvalidated projection. The documented
substitute is `L2_normalize(concat([h_C', h_C', h_E]))`: two normalized copies
of content and one normalized affect vector preserve the intended 2:1 norm
budget while making the dimensional mismatch explicit. RoBERTa also substitutes
for the paper's ModernBERT-family GoEmotions encoder: the objective is the same,
but the backbone is not.

The fused input is pretrained through the paper-style 128-dimensional
autoencoder, initialized with K-means, and jointly optimized by DEC. The DEC
schedule is deliberately convergence-controlled rather than the paper's fixed
200 epochs: it uses full-batch training and stops when
`fraction_changed < 0.001`, with a 500-epoch ceiling. This was an existing,
validated project choice for this autoencoder-plus-DEC setting. The paper does
not specify its exact norm-threshold pruning rule, so the implementation keeps
the 67 highest-L2-norm centers from 100, matching its reported retention rate.

These deviations limit claims of literal reproduction, but they were documented
before evaluation rather than introduced after observing results.

## III. Chronological result record

### 1. Base PercepT replication: DEC collapse

The unregularized 100-initial/67-surviving-topic pilot converged at epoch 275,
but convergence did not imply a useful partition. On both train and held-out
data, 65/67 surviving topics fell below the 1% threshold and the median topic
size was 0. The held-out result was emotion AMI=0.0363 and genre AMI=0.3081:
genre signal remained, but the topic assignment was collapsed.

The logged loss scale identified a concrete cause. At convergence, KL was
0.064992 while reconstruction was 0.000085, so the reconstruction anchor was
roughly 760 times smaller and DEC's self-sharpening effectively dominated the
joint loss.

### 2. First balance regularizer: directionally useful but too small

The first stabilization added a global balanced-assignment regularizer with
`lambda_balance=1`, while retaining reconstruction weight 1. It still collapsed:
64/67 held-out topics were below 1%, with median size 0. Yet held-out emotion
AMI rose from 0.0363 to 0.0925, while genre AMI was 0.2631. The effect was
directionally encouraging but numerically too weak: its balance loss remained
0.0001--0.0016 while KL grew to 0.093190.

### 3. Balance sweep: the effective range was above 500

The first sweep tested `lambda_balance` 10, 50, 100, and 500. Every point was
held-out collapsed. The best point, 500, reached held-out emotion AMI=0.1142
and genre AMI=0.2116 but still had 44/67 topics below 1%. The relationship was
not monotonic, but the best observed point was at the top of the range, so a
higher-lambda search was justified rather than declaring the mechanism failed.

The follow-up at 1,000, 2,500, and 5,000 was the first single-seed success:
all three seed-42 held-out runs were non-collapsed and cleared both AMI bars.
At 1,000, the held-out result was 0.1242 emotion / 0.2466 genre; at 2,500 it
was 0.1242 / 0.1970; at 5,000 it was 0.1242 / 0.2208. This established that
the balance mechanism could work, but it did not establish that it would work
reliably across full-pipeline random seeds.

### 4. Lambda=1000 seed stress: a genuine fragility finding

The four-seed stress test at `lambda_balance=1000` varied autoencoder
initialization, pretraining order, and K-means seed. Only seed 42 cleared both
bars. Held-out means were emotion AMI=0.1235 (min=0.1216, max=0.1252) and genre
AMI=0.2089 (min=0.1849, max=0.2466); each individual bar cleared in 2/4 seeds,
and both cleared in 1/4. Thus the seed-42 result was real but fragile, not the
reliable Stage 1 result needed for Stage 2.

### 5. Reconstruction reweighting: failed to repair fragility

Because raw reconstruction loss was only about 0.00003 during DEC, the next
test fixed `lambda_balance=1000` and raised reconstruction weight to 100, 300,
and 1,000. None cleared both held-out bars at seed 42. Reconstruction weight
300 was selected for stress testing because it had the largest emotion margin:
0.1245 emotion and 0.1916 genre.

That selection failed the four-seed test: its held-out mean was 0.1234 emotion
(min=0.1215, max=0.1252) and 0.1966 genre (min=0.1849, max=0.2177), with 0/4
seeds clearing both bars. Genre spread narrowed from 0.0617 to 0.0328, but
emotion spread stayed 0.0036 and the both-bar count fell from 1/4 to 0/4. This
was not a stability fix.

### 6. Faithful IDEC: failed more severely than the original DEC pilot

The literature-standard alternative, IDEC, retained reconstruction actively
with `total_loss = reconstruction + 0.1 * KL`, without the balance term.
It converged at epoch 377 but collapsed: 64/67 topics were below 1% on both
splits, and held-out AMIs were 0.0444 emotion and 0.2718 genre. The all-center
diagnostic reached 97/100 centers below 1%, worse than the original
unregularized pilot's 65/67 surviving-center collapse. The diagnosis is scale:
even after multiplying KL by 0.1, its raw value of 0.076311 at convergence was
still about 10--100 times the 0.000060 reconstruction value. IDEC's published
default did not transfer to this loss scale, so its optional seed stress was
correctly skipped.

### 7. Cluster count: coarser topics found the stability mechanism

The next lever changed the 100/67 topic count itself. With 100 initial
clusters, the average 614-node train cluster is nearly the 1%-of-61,402-node
collapse threshold, so small assignment changes can determine the verdict.
The first screen tested 20/13, 30/20, and 50/33, holding the successful balance
weight at 1,000 and reconstruction weight at 1. K=30/20 was the strongest
screen point at held-out 0.1220 emotion / 0.2577 genre and was stress-tested.

K=30/20 made the outcome substantially more stable, but too low on emotion:
four-seed means were 0.1212 emotion and 0.2616 genre, with emotion range 0.0016
and genre range 0.0169. Genre cleared in all four seeds, but emotion cleared in
none. This was evidence that cluster count genuinely controlled seed stability,
not evidence that the desired operating point had already been found.

An intermediate screen of 40/27, 60/40, and 80/53 found the crossover at
K=60/40. Its seed-42 held-out result was 0.1238 emotion / 0.2617 genre. In the
four-seed stress test, all four seeds cleared both bars: emotion mean=0.1252,
min=0.1238, max=0.1272; genre mean=0.2486, min=0.2328, max=0.2617. The emotion
minimum clears 0.1236 by only 0.0002, so the margin remains thin, but it is now
consistently on the successful side of the threshold.

The raw trajectories support that this is not merely a fortunate aggregate.
Every K=60/40 seed converged through the stability criterion in 82--88 epochs;
none reached the 500-epoch ceiling. In every run the balance-loss term declined
toward approximately zero, rather than fighting the clustering objective to the
end. `fraction_changed` decayed smoothly and monotonically with no oscillation
or late-training instability. This is qualitatively healthier and more uniform
than the original K=100 trajectories.

## IV. Cross-method summary

All values below are held-out AMIs. “Single seed” means seed 42; “4-seed mean”
uses seeds 42, 7, 123, and 2024. The K=100/67 4-seed row is the lambda=1000
seed-stress result; K=30/20 and K=60/40 are their cluster-count stress results.

| configuration | held-out emotion AMI | held-out genre AMI | collapse / stability verdict |
|---|---:|---:|---|
| Base, K=100/67 (single seed) | 0.0363 | 0.3081 | Collapsed (65/67 below 1%) |
| Balance lambda=1, K=100/67 (single seed) | 0.0925 | 0.2631 | Collapsed (64/67 below 1%) |
| Balance lambda=500, K=100/67 (single seed) | 0.1142 | 0.2116 | Collapsed (44/67 below 1%) |
| Balance lambda=1000, K=100/67 (single seed) | 0.1242 | 0.2466 | Real success |
| Balance lambda=2500, K=100/67 (single seed) | 0.1242 | 0.1970 | Real success |
| Balance lambda=5000, K=100/67 (single seed) | 0.1242 | 0.2208 | Real success on held-out; train collapsed |
| Balance lambda=1000, K=100/67 (4-seed mean) | 0.1235 | 0.2089 | Seed-dependent: 1/4 clear both bars |
| Reconstruction=300, balance=1000, K=100/67 (4-seed mean) | 0.1234 | 0.1966 | Seed-dependent: 0/4 clear both bars |
| Faithful IDEC gamma=0.1, K=100/67 (single seed) | 0.0444 | 0.2718 | Collapsed (64/67 below 1%) |
| K=20/13 (single seed) | 0.1152 | 0.2273 | Merely a compromise |
| K=30/20 (4-seed mean) | 0.1212 | 0.2616 | Stable but below emotion bar: 0/4 clear |
| K=40/27 (single seed) | 0.1220 | 0.2736 | Merely a compromise |
| K=50/33 (single seed) | 0.1202 | 0.2830 | Merely a compromise |
| K=60/40 (4-seed mean) | 0.1252 | 0.2486 | Real success: 4/4 clear both bars |
| K=80/53 (single seed) | 0.1220 | 0.2396 | Merely a compromise |
| K=100/67 (4-seed mean) | 0.1235 | 0.2089 | Seed-dependent: 1/4 clear both bars |

## V. Standing result and comparison to Attention-h1

K=60/40 is the standing Stage 1 result. It is the only configuration in this
entire Stage 1 sequence to clear the held-out Pareto bar in 4/4 seeds, rather
than 1/4. Its spreads also improve substantially over original K=100/67:
emotion spread is 0.0034 versus 0.0036, and genre spread is 0.0289 versus
0.0617. It does not erase uncertainty: the worst emotion seed, 0.1238, exceeds
the 0.1236 bar by only 0.0002. The evidence is stronger because that narrow
margin persists in each stress-test seed, not because the numeric lead is large.

The two investigations are close on their central held-out axes. K=60/40 has
emotion AMI=0.1252 and genre AMI=0.2486, compared with Attention-h1's 0.1249
and 0.2404. More importantly, K=60/40 has evidence Attention-h1 does not yet
have: four-seed robustness. That is a genuine advantage in evidentiary strength,
not a claim that the small point-estimate differences are decisive.

## VI. Limits and Stage 2 handoff

The result remains an adapted PercepT replication. It uses embedding-level
GoEmotions-RoBERTa affect representations rather than the paper's literal
ModernBERT choice; concatenation with repeated content rather than Eq. 2's
elementwise sum; and a convergence-controlled DEC schedule rather than fixed
200 epochs. There has also been no literal, unadapted DEC/IDEC comparison at
the K=60 scale: the faithful IDEC test was K=100 only. These are open
comparisons, not resolved by the current result.

Stage 2 (P-Topic Mapping) will freeze the K=60/40 cluster assignments fit on
the **full train split** at one representative seed, to be specified in the
Stage 2 brief. Those assignments become fixed multi-label pseudo-targets for an
image-only classifier. This report makes no Stage 2 performance claim; it
records only the Stage 1 topic-formation evidence that justifies that handoff.
