# Brief: transfer held-out points onto the TRAIN Leiden vocabulary (Stage 2 blocker)

Write a new script
`src/test/20260923_artelingo_buddy_analysis/run_heldout_label_transfer_pilot.py`.
This script needs no GPU and no model training — implement it AND run it
yourself, then write its report (unlike other briefs in this directory, do
not stop at implementation only).

## Context: a real, identified structural gap, not a hypothetical

A comprehensive status memo on this project's buddy/Attention-h1 work
found: "The current Attention-h1 snapshot independently runs Leiden on the
held-out graph (18 communities) and train graph (19), with IDs unrelated
across splits... those two arrays cannot simply be used as shared
classifier target columns. A mechanism to assign held-out images to the
frozen train topic vocabulary is needed for meaningful Stage 2 evaluation."
This blocks ever plugging a buddy/Leiden-based Stage 1 into PercepT's
Stage 2 (P-Topic Mapping) pattern, which needs one shared `[N, K]` target
vocabulary across train and held-out. This pilot builds and validates the
missing mechanism using data that already exists on disk — no new training
run is needed to test it.

Read `run_attention_h1_embedding_snapshot_pilot.py` first (to see the exact
shapes/contents of `attention_h1_embedding_snapshot.npz`, which this pilot
loads directly rather than retraining anything), and `run_pipeline.py`'s
`external_metrics` function (reuse it for AMI scoring).

## What to implement

A single function, `assign_to_train_communities`:

```python
def assign_to_train_communities(
    train_embeddings: np.ndarray,   # [N_train, D], L2-normalized
    train_community: np.ndarray,    # [N_train] int labels, the frozen vocabulary
    query_embeddings: np.ndarray,   # [N_query, D], L2-normalized, same space
    k: int = 20,                    # matches this project's buddy-graph K=20 convention
) -> np.ndarray:                    # [N_query] int labels, values are a SUBSET of train_community's label set
    """Assign each query point to a train Leiden community by k-NN majority
    vote in cosine-similarity space (Leiden communities are graph-derived
    and not necessarily convex/globular, so a nearest-centroid approach
    would be a worse structural match than direct k-NN vote against actual
    train points)."""
```

Implement the k-NN search with `sklearn.neighbors.NearestNeighbors`
(`metric="cosine"`), fit on `train_embeddings`, batch-query
`query_embeddings` (batch to keep peak memory reasonable — 9,365 held-out
against 61,402 train in 32-D is not large, but do not build a single dense
9365x61402 matrix if `NearestNeighbors` doesn't already avoid this
internally; confirm rather than assume). For each query point, take the
majority `train_community` label among its k nearest train neighbors;
break ties by nearest single neighbor's label. If a query point's k
nearest train neighbors span more than half distinct labels with no
majority above 1 vote (a genuinely degenerate tie), fall back to the
single nearest neighbor's label and note in a running counter how often
this fallback fired.

## Validation: does transfer preserve or change agreement with human labels?

Load `attention_h1_embedding_snapshot.npz` (already on disk in this same
directory — do not retrain anything). Using the **post-training**
(`train_embedding_post`, `train_community_post`, `heldout_embedding_post`)
arrays:

1. Reproduce the **already-known, independently-computed** held-out
   emotion/genre AMI (0.1249 train... no — the CITED values in that script
   are `CITED_HELDOUT_EMOTION_AMI = 0.1249`, `CITED_HELDOUT_GENRE_AMI =
   0.2404`, using `heldout_community_post` as independently re-clustered by
   Leiden). Recompute these directly from the saved `.npz` arrays and the
   saved `heldout_emotion`/`heldout_genre` arrays plus `pipeline.load_genre_map()`-
   equivalent filtering (reuse `run_pipeline.py`'s `external_metrics` and
   the genre-overlap filtering pattern from the base scripts) as a
   sanity-check control — this MUST match the cited numbers closely (small
   floating point differences are fine; anything else means something is
   loaded wrong, and you must resolve this before proceeding, not report
   both a resolved and unresolved result).
2. Compute new held-out community labels via
   `assign_to_train_communities(train_embedding_post, train_community_post,
   heldout_embedding_post, k=20)` — this assigns every held-out point onto
   the TRAIN vocabulary instead of independently re-clustering held-out.
3. Compute held-out emotion/genre AMI using these NEW transferred labels
   (same `external_metrics` call, same genre-overlap filtering).
4. Report both numbers side by side, plus: how many of the frozen train
   communities receive at least one held-out point (coverage), the
   fallback-tie counter from above, and how many distinct labels appear in
   the transferred held-out assignment versus the original 18 independently-
   found held-out communities.
5. Repeat steps 2-4 for `k` in `(5, 10, 20, 50)` — report all four as a
   small table, do not just report k=20. State plainly whether AMI is
   sensitive to k in this range or stable.

## Report

Write
`src/test/20260923_artelingo_buddy_analysis/heldout_label_transfer_pilot_report.md`
with: the sanity-check reproduction numbers (must match cited values,
state the match explicitly), the k-sweep table (emotion AMI, genre AMI,
coverage, tie-fallback count, distinct-label count — for each k, plus the
"original independent Leiden" row as a reference), and a final section
answering directly: does k-NN transfer onto the frozen train vocabulary
preserve held-out AMI well enough to be a usable Stage-2 target-construction
mechanism, or does it degrade external-label agreement meaningfully versus
independent re-clustering? Since this is now a working, shared-vocabulary
mechanism regardless of the AMI answer, also state plainly that this
specific structural blocker (no shared train/held-out topic vocabulary) is
now resolved, and name the exact function (`assign_to_train_communities`)
and its file as the piece any future Stage-2-wiring work should import and
reuse, rather than reimplementing.

Do not touch git. Do not modify any other file. You have up to 20 minutes —
this is light CPU computation on already-saved arrays, not a training run.
