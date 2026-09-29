# Attention-h1 embedding snapshot pilot

Generated automatically, 2026-09-24 02:59:15.

This is the buddy-graph "our own method" counterpart to the PercepT Stage-1 embedding snapshot pilot. It re-fits the standing Attention-h1 two-teacher contrastive student and dumps its 32-D joint embedding at initialization (epoch 0, before any InfoNCE step) and after training converges, with a genuine Leiden community assignment computed at both points (the original architecture-sweep script only ever runs Leiden once, on the final embedding).

## Reproduction check against the cited Attention-h1 result

| split | metric | cited | this re-fit | absolute difference |
|---|---|---:|---:|---:|
| train | emotion AMI | 0.1351 | 0.1351 | 0.0000 |
| train | genre AMI | 0.2397 | 0.2397 | 0.0000 |
| held-out | emotion AMI | 0.1249 | 0.1249 | 0.0000 |
| held-out | genre AMI | 0.2404 | 0.2404 | 0.0000 |

This is reported for transparency, not gated: unlike the PercepT branch, this specific script has no prior determinism investigation, so a small difference from the citation is expected and does not invalidate the snapshot below.

## Epoch-0 (pre-training) diagnostic AMI, newly computed

No prior run of this architecture ever computed these -- the original architecture-sweep script only evaluates recall/gate diagnostics at epoch 0, never Leiden AMI.

| split | metric | value |
|---|---|---:|
| train | emotion AMI | 0.0665 |
| train | genre AMI | 0.2734 |
| held-out | emotion AMI | 0.0500 |
| held-out | genre AMI | 0.2234 |

## Snapshot contents

Written to `attention_h1_embedding_snapshot.npz`.

| array | shape | meaning |
|---|---|---|
| train_paintings | (61,402,) | painting ids, train split |
| train_embedding_pre | (61,402, 32) | joint embedding at initialization (epoch 0) |
| train_embedding_post | (61,402, 32) | joint embedding after training converged |
| train_community_pre | (61,402,) | Leiden community id on the epoch-0 embedding (20 communities found) |
| train_community_post | (61,402,) | Leiden community id on the final embedding (19 communities found) |
| train_emotion | (61,402,) | majority caption emotion label (full coverage) |
| train_genre | (61,402,) | genre label, or "" where unavailable (1,144/61,402 covered) |
| heldout_paintings | (9,365,) | painting ids, held-out (val+test) split |
| heldout_embedding_pre | (9,365, 32) | same epoch-0 model applied out-of-sample |
| heldout_embedding_post | (9,365, 32) | same final model applied out-of-sample |
| heldout_community_pre | (9,365,) | Leiden community id on the epoch-0 held-out embedding (13 communities found) |
| heldout_community_post | (9,365,) | Leiden community id on the final held-out embedding (18 communities found) |
| heldout_emotion | (9,365,) | majority caption emotion label (full coverage) |
| heldout_genre | (9,365,) | genre label, or "" where unavailable (159/9,365 covered) |

Unlike PercepT's fixed K=60/40, Leiden chooses its own community count each time it runs -- the pre- and post-training community counts above are not necessarily equal, and that difference is itself a real, reportable fact about this method rather than a bug.
