# Candidate 4 — richer multi-label buddy Stage 2 targets

Generated 2026-09-27 21:14:14. Frozen buddy Stage 1 snapshot; candidate 1's three-smallest-community merge to K=16; candidate 2's AttentionPoolingMapper, lr=1e-2, 400 epochs. Seed 42 screen; practical margin +0.005 over the candidate 2 four-seed mean macro AUC 0.8461.

## Targets and training

The shared `assign_to_train_communities` returns only a hard winning label, not the neighbor vote fractions. A local batched helper therefore computes the same k=20 cosine nearest-neighbor vote fractions, counting neighbors in the merged K=16 vocabulary. Train paintings are queried against the train embeddings, including their own embedding; held-out paintings are queried against that same train vocabulary. A topic is positive when its fraction is strictly greater than the stated cutoff times that painting's largest fraction.

The loss is **unweighted BCEWithLogitsLoss** on multi-hot targets. Candidate 1's single-sample class-balanced weighting is dropped because its inverse hard-class frequency definition does not carry over to multi-hot paintings.

## Seed 42 screen

| relative cutoff | train mean / median / max labels | train multi-labeled | held-out mean / median / max labels | held-out multi-labeled | macro AUC | min / median / max topic AUC | skipped topics |
|---:|---|---:|---|---:|---:|---|---:|
| 0.50 | 1.240 / 1.000 / 7 | 20.55% | 1.288 / 1.000 / 5 | 23.88% | 0.8394 | 0.6651 / 0.8447 / 0.9701 | 0 |
| 0.30 | 1.485 / 1.000 / 9 | 35.90% | 1.572 / 1.000 / 7 | 40.48% | 0.8322 | 0.6639 / 0.8379 / 0.9530 | 0 |
| 0.15 | 1.878 / 2.000 / 11 | 51.94% | 2.022 / 2.000 / 10 | 56.72% | 0.8188 | 0.6531 / 0.8251 / 0.9199 | 0 |

The baseline uses strict single-label held-out targets, so its AUC and these AUCs evaluate different target definitions. Their difference is a practical screen, not a like-for-like metric improvement on identical labels.

## Verdict

**Candidate 4 is closed as tested with a negative result.** Four-seed stress was skipped because the best seed 42 macro AUC was 0.8394 at cutoff 0.50, below the 0.8511 stress gate. No additional cutoffs were tested.
