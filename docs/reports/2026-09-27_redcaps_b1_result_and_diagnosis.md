# RedCaps B1 pilot result and root-cause diagnosis

Generated 2026-09-27. Companion to
[the method-improvement/RedCaps brainstorm](2026-09-27_method_improvement_and_redcaps_brainstorm.md)
and
[`b1_redcaps_single_teacher_pilot_report.md`](../../src/test/20260927_redcaps_topic_formation/b1_redcaps_single_teacher_pilot_report.md).

## What was tested

The brainstorm memo's top-ranked candidate (B1): a single-teacher
Attention-h1-style student (image+text CLIP tokens, one-head attention,
32-D output) trained on RedCaps 150k's existing union buddy graph
(K=30, train-only, subreddit-blind 120k/15k/15k split), evaluated by the
already-validated `subreddit_lift` metric — no emotion/genre labels exist
on RedCaps, so this is the closest available proxy. A mean-pooling
architecture control and three no-training baselines (raw image features,
raw text features, and Leiden-on-the-raw-teacher-graph) were run
alongside it, all on local GPU, seed 42, ~2 minutes total wall-clock.

## Result: a genuine negative, with the cause visible in the data itself

| Method | Validation embedding-graph lift | Community-level lift | Below-1%-occupancy communities |
|---|---:|---:|---:|
| Attention student | 18.39x | 6.72x | 304/326 |
| Mean-pool control | 18.97x | 6.87x | 529/555 |
| Raw image (no training) | 24.31x | n/a | n/a |
| Raw text (no training) | 18.15x | n/a | n/a |
| **Graph-only baseline (Leiden on the raw teacher graph, no training at all)** | **27.11x** | 4.77x | **1404/1423** |

Two findings, both against this pilot's own predeclared verdict rule:

1. **Neither trained student beats simply using raw CLIP features
   directly** (18.4–19.0x vs. 24.3x for raw image alone) — training the
   single-teacher student did not improve on the untrained embedding it
   started from, on the primary lift metric.
2. **Attention does not beat mean-pooling here**, unlike on ArtELingo —
   the two are within noise of each other on both lift measures.

**The more important finding is in occupancy, and it rules out the
student architecture as the cause.** The completely untrained
graph-only baseline — Leiden run directly on the raw teacher graph, zero
training involved — is *also* ~99% collapsed (1,404 of 1,423 communities
below 1% occupancy). Since this baseline never touches the student
architecture, the training recipe, or the InfoNCE loss at all, the severe
fragmentation cannot be caused by anything this session built. **The
likely cause is Leiden's default modularity resolution behaving very
differently on RedCaps' graph structure than on ArtELingo's** — producing
hundreds of tiny, near-singleton communities instead of ArtELingo's clean
19-way partition, for reasons intrinsic to the graph (density, degree
distribution), not the topic-formation method.

## Why this stops here, not scales up

Per the pilot's own predeclared gate and the brainstorm memo's explicit
staged design: a null or negative result at this cheap first gate means
**diagnose before spending on B2 (patch-feature extraction) or any
300k/500k DAS6 training** — not push forward on an undiagnosed
fragmentation problem. Both DAS6 nodes (node403, node404) were confirmed
available and idle throughout tonight and were **not used** for this
reason; local GPU was sufficient and appropriate for a pilot this size,
exactly as the memo recommended.

## Follow-up (completed): the resolution hypothesis was wrong — the real cause is disconnected components, and there is a working fix

The resolution-aware sweep proposed below was run
([`run_leiden_resolution_sweep_pilot_report.md`](../../src/test/20260927_redcaps_topic_formation/run_leiden_resolution_sweep_pilot_report.md)),
and it **refutes** the resolution-calibration diagnosis: sweeping
`RBConfiguration` resolution from 1.0 down to 0.005 (three orders of
magnitude) barely moves community count (1423 → 1397) or the below-1%
count (1404 → 1396) at all — while the largest community balloons from
21,482 to 118,557 members and community-level lift *degrades* (4.773× →
0.971×, i.e. collapses toward no signal) as everything merges into one
giant blob.

A direct `scipy.sparse.csgraph.connected_components` check on the same
raw train teacher graph explains why: it has **1,397 connected
components** — 1,354 true singletons, ~40 tiny 2–3-node fragments, and
one giant component covering 98.8% of nodes. Leiden cannot merge across
disconnected components at *any* resolution, so 1,397 is a hard floor,
not a tuning artifact.

This codebase already has a fix for exactly this:
`ensure_min_degree` (`src/conditional_buddy/buddy_graph.py`), used
elsewhere in the B1 pilot but only for its sanity-check lift table, never
before Leiden. Applying it before every Leiden call
([`run_b1_repaired_graph_pilot_report.md`](../../src/test/20260927_redcaps_topic_formation/run_b1_repaired_graph_pilot_report.md)):

| Method | Isolated nodes repaired | Communities (before → after) | Below-1% (before → after) | Community-level lift (before → after) |
|---|---:|---:|---:|---:|
| Graph-only baseline | 1354 | 1423 → 71 | 1404 → 50 | 4.773× → 4.775× |
| Attention student | 279 | 326 → 46 | 304 → 23 | 6.720× → 6.577× |
| Mean-pool control | 497 | 555 → 56 | 529 → 31 | 6.865× → 6.195× |

Repair fixes the graph-only baseline's occupancy almost completely (98.7%
below 1% → 70.4%) at essentially zero cost to lift (4.773× → 4.775×), and
meaningfully improves both trained students' embedding-space graphs too
(e.g. attention student 93.3% → 50.0% below 1%), though it does not fully
resolve their occupancy the way it does for the graph-only baseline.

**This does not change B1's core verdict.** Even repaired, neither
trained student beats the graph-only baseline or the raw-feature
controls on validation embedding-graph lift (attention: 18.391× vs.
27.114× graph-only / 24.314× raw-image). The stop-before-B2 recommendation
stands. What changed is narrower but still useful: `ensure_min_degree`
should be applied before Leiden by default on RedCaps-scale graphs going
forward (it was already being computed for the sanity table and simply
wasn't wired into the partition step) — a one-line graph-hygiene fix, not
a fix for the negative training-value finding itself.

## Second follow-up: is 'subreddit' itself a noisy proxy for topic?

User hypothesis: RedCaps' negative result may partly reflect the dataset/metric,
not the method — subreddits aren't semantically clean topic labels the way
ArtELingo's emotion/genre are, and the split may leave too little validation
support for a reliable lift estimate. Tested directly
([`run_subreddit_proxy_diagnostic_report.md`](../../src/test/20260927_redcaps_topic_formation/run_subreddit_proxy_diagnostic_report.md)),
on the same B1 split, no new randomness:

- **False negatives are real and large.** On a 3,000-point validation sample,
  15.9% of *different*-subreddit pairs are at least as similar (raw CLIP) as
  the median *same*-subreddit pair. More strikingly: **63.1% of all validation
  points have a cross-subreddit nearest neighbor at least as similar as a
  typical same-subreddit pair** — the large majority of "best matches" in this
  dataset are being scored as negatives by the lift metric.
- **Subreddit redundancy is real and large.** Among 281 subreddits with ≥50
  members, 7.67% of distinct-subreddit *pairs* have centroid cosine similarity
  ≥0.9. The top offenders are unambiguous near-duplicates: `dogpictures` /
  `lookatmydog` / `rarepuppers` / `doggos` all pairwise ≥0.997, `catpictures` /
  `cats` at 0.9986, `food` / `foodporn` at 0.9949, `houseplants` / `plants` at
  0.9945, and 14 more equally clear pairs. These are the same topic, split
  across subreddit names.
- **The split itself is not the problem.** Only 4/350 subreddits (1.1%) are
  entirely absent from validation, and the top-20 subreddits hold 44.6% of
  validation mass — some concentration, but not the "thin, unreliable test
  set" the split-quality half of the hypothesis worried about.

**Net read:** the user's core intuition is confirmed on the label side, not
the split side. Subreddit is a substantially noisier, redundant proxy for
"topic" than ArtELingo's curated emotion/genre labels — a nontrivial share of
what the lift metric counts as errors are not errors. This does not retract
B1's finding that neither trained student beats raw CLIP feature lift (that
comparison is apples-to-apples under the same noisy metric for every method
tested), but it means **RedCaps' absolute lift numbers should not be read as
a clean ceiling on topic quality**, and any future RedCaps work should
either de-duplicate/merge near-identical subreddits before treating them as
distinct labels, or evaluate with a metric less sensitive to this specific
noise source (e.g. a fuzzy/embedding-aware match instead of exact subreddit
equality).

## Recommended next step (superseded by the follow-ups above)

~~A resolution-aware Leiden sweep on RedCaps' teacher graph~~ — completed;
see the follow-up section above. No further RedCaps work is recommended
at this time: the negative training-value finding from B1 (students do
not beat raw CLIP features on lift) is now confirmed robust to the
occupancy-collapse confound, which was the only open question left after
B1.
