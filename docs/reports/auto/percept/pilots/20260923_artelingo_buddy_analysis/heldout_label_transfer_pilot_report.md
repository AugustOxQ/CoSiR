# Held-out label transfer onto the train Leiden vocabulary

## Sanity check

Using the saved post-training independent held-out Leiden labels, emotion AMI = **0.12488511** and genre AMI = **0.24042176**. These round to the cited **0.1249** and **0.2404**, respectively: both match. Genre AMI uses the 159 held-out paintings in `pipeline.load_genre_map()`; their saved genre values all match that map.

## k sweep

Post-training train vocabulary: 19 communities; held-out points: 9365. Cosine `NearestNeighbors` queries at most 128 held-out points per batch, so this script never passes the full held-out-by-train matrix to scikit-learn. A tied top vote uses the nearest train point among the tied winners (the interpretation of the brief's nearest-neighbor tie-break); fallback count records the all-distinct-label case. With only 19 train labels, that fallback is impossible when k exceeds 19; ordinary tied top votes are not counted.

| method | emotion AMI | genre AMI | train coverage | tie fallbacks | distinct held-out labels |
|---|---:|---:|---:|---:|---:|
| Original independent Leiden | 0.1249 | 0.2404 | n/a | n/a | 18 |
| k=5 | 0.1281 | 0.2243 | 19/19 | 45 | 19 |
| k=10 | 0.1327 | 0.2355 | 19/19 | 0 | 19 |
| k=20 | 0.1364 | 0.2530 | 19/19 | 0 | 19 |
| k=50 | 0.1390 | 0.2461 | 19/19 | 0 | 19 |

## Verdict

Across k=5–50, emotion AMI ranges 0.1281–0.1390 and genre AMI ranges 0.2243–0.2530. Emotion varies by 0.0109; genre varies by 0.0287 on 159 genre-labeled points. Genre is more sensitive to k in this sweep.

At the project-convention k=20, emotion AMI changes by +0.0115 and genre AMI by +0.0126 relative to independent held-out Leiden. Both AMIs are higher at k=20, so k-NN transfer preserves external-label agreement well enough to be a usable Stage-2 target-construction mechanism at that setting; the observed comparison shows no degradation. Genre AMI uses only 159 of 9365 held-out points, so its differences are less secure than the full-split emotion AMI. This is an observed comparison, not a formal equivalence test.

The specific structural blocker—the absence of a shared train/held-out topic vocabulary—is resolved for hard-label assignment: every transferred held-out label belongs to the frozen train vocabulary. Future Stage-2 wiring should import `assign_to_train_communities` from `run_heldout_label_transfer_pilot.py` and reuse it, rather than reimplementing the transfer. This pilot evaluates hard labels; it does not construct or evaluate soft [N, K] targets.
