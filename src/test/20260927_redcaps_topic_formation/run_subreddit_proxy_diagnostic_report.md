# Is 'subreddit' a noisy proxy for topic on RedCaps?

Generated 2026-09-27 14:45:40; seed 42; reuses B1's saved split (`b1_redcaps_single_teacher_pilot_split.npz`). Companion to [`docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md`](../../../docs/reports/2026-09-27_redcaps_b1_result_and_diagnosis.md).

Motivating hypothesis: the negative B1/repair result may partly reflect the subreddit label and the split, not the topic-formation method -- subreddits may be a noisier, more redundant, and more thinly-supported proxy for topic than ArtELingo's emotion/genre labels.

## A. Same- vs different-subreddit raw similarity separability

Sample: 3,000 validation points (77,839 same-subreddit pairs, 4,420,661 different-subreddit pairs). Raw cosine similarity (unit-normalized, concatenated image+text CLIP).

- Same-subreddit similarity: mean 0.6060, median 0.6045
- Different-subreddit similarity: mean 0.5135, median 0.5140
- **False-negative-rate estimate**: 15.9% of different-subreddit pairs are at least as similar as the *median* same-subreddit pair (at the same-subreddit 90th percentile instead: 0.6%).
- Nearest-neighbor check: 64.2% of validation points' single nearest neighbor (by raw similarity) is in a *different* subreddit; 63.1% of all points have a cross-subreddit nearest neighbor that is *at least as similar as a typical same-subreddit pair* -- these are the clearest false-negative candidates under the subreddit-lift metric.

## B. Subreddit-centroid redundancy

281 subreddits with >= 50 members (train+validation combined); centroid = mean unit-normalized raw feature, re-normalized.

- Inter-subreddit centroid similarity: mean 0.7986, 99th percentile 0.9723
- 45.79% of distinct-subreddit pairs have centroid similarity >= 0.8; 7.67% >= 0.9

Top 20 most similar *distinct* subreddit pairs (candidate redundant/overlapping communities):

| Subreddit A | Subreddit B | Centroid similarity | Size A | Size B |
|---|---|---:|---:|---:|
| dogpictures | lookatmydog | 0.9993 | 1,064 | 560 |
| dogpictures | rarepuppers | 0.9989 | 1,064 | 1,689 |
| catpictures | cats | 0.9986 | 279 | 8,062 |
| lookatmydog | rarepuppers | 0.9986 | 560 | 1,689 |
| doggos | rarepuppers | 0.9981 | 215 | 1,689 |
| doggos | dogpictures | 0.9977 | 215 | 1,064 |
| averagebattlestations | battlestations | 0.9973 | 148 | 815 |
| doggos | lookatmydog | 0.9970 | 215 | 560 |
| birding | birdpics | 0.9969 | 472 | 299 |
| bbq | smoking | 0.9968 | 322 | 762 |
| backyardchickens | chickens | 0.9965 | 250 | 191 |
| damnthatsinteresting | interestingasfuck | 0.9965 | 311 | 1,160 |
| beginnerwoodworking | woodworking | 0.9964 | 315 | 1,555 |
| houseplants | indoorgarden | 0.9963 | 4,454 | 476 |
| entomology | insects | 0.9954 | 151 | 349 |
| food | foodporn | 0.9949 | 4,506 | 2,899 |
| houseplants | plants | 0.9945 | 4,454 | 870 |
| equestrian | horses | 0.9945 | 102 | 176 |
| abandoned | urbanexploration | 0.9944 | 112 | 235 |
| pug | pugs | 0.9940 | 55 | 385 |

## C. Split/support coverage

350 total subreddits; 350 present in train. **4 of those (1.1%) have zero validation examples** under the random, non-subreddit-stratified split.

- Validation count per subreddit (subreddits with >= 1 val example): min 1, median 18.0, max 998
- The top 10 subreddits by validation count hold 32.7% of all validation examples; the top 20 hold 44.6%
- 16.2% of subreddits with any validation presence at all have fewer than 5 validation examples

## Reading these together

- (A) supports the false-negative hypothesis: 15.9% of different-subreddit pairs look as similar as a typical true match.
- (B) supports the redundancy hypothesis: 7.67% of distinct-subreddit pairs have near-duplicate centroids -- check the top-pairs table for whether these look like genuinely overlapping communities.
- (C) does not show a strongly thin or concentrated test set by these two measures.

This diagnostic bears on B1's metric ceiling, not on the trained students' training procedure itself -- it does not retract B1's finding that neither student beats raw CLIP features, but it does inform how much of that gap (and how much of the residual occupancy collapse from B1/repair) should be attributed to subreddit being a noisy, unevenly supported proxy label versus a genuine method shortfall.
