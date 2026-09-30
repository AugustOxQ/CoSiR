# ArtELingo late-fusion buddy-graph pilot

Generated automatically, 2026-09-22 17:12:22.

**Late fusion in plain language:** the content view and the affect view each build a buddy graph independently in their own native feature space. Their graphs are then combined by their edges, rather than by combining feature vectors before neighbour search. This differs from the earlier **early fusion** pilot, which concatenated GoEmotions and CLIP-text features into one vector and built one mutual-kNN graph in that shared space.

**Setup:** K=20, alpha=0.5, Leiden seed=42; genre metrics use the 1144-painting genre-labelled overlap. The content graph has `E_content.nnz=711,878`, the GoEmotions affect graph has `E_affect.nnz=620,608`, and their late union has `E_late_union.nnz=1,331,820`. The raw late intersection left 98.96% of nodes isolated before any repair.

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Content-only (CLIP img+txt union, reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
| GoEmotions-affect-only (reference, single-modality ceiling) | 0.1180 | 0.1189 | 0.0396 | 0.0937 |
| Early fusion, best point (weight=4.0, reference) | 0.1160 | 0.1166 | 0.0867 | 0.1237 |
| Late fusion — union (this run) | 0.1236 | 0.1241 | 0.1394 | 0.1677 |
| Late fusion — intersection (this run) | 0.0510 | 0.0554 | 0.2399 | 0.3955 |

## Conclusion

**(a) Original affect-pilot bar:** this requires emotion AMI > 0.09 (at least a 50% relative improvement over the 0.0593 content-only baseline) while retaining at least 80% of the 0.4384 genre baseline (genre AMI >= 0.3507). Late union does not clear this bar: emotion AMI=0.1236, genre AMI=0.1394.

**(b) Stricter DEC-pilot bar:** this requires emotion AMI > 0.177, a 50% relative improvement over GoEmotions' 0.1180 single-modality ceiling. Late union does not clear this bar: emotion AMI=0.1236.

Late fusion preserved genre structure better than early fusion: late-union genre AMI=0.1394 versus the early-fusion value of 0.0867.

## Intersection diagnostic

The raw edge intersection isolated 98.96% of nodes before repair. After minimum-degree and connectivity repair, its emotion/genre AMI values were 0.0510/0.2399, versus 0.1236/0.1394 for union. This suggests the intersection is degenerate or noise-dominated after heavy repair, with the raw isolated-node fraction providing the necessary context for that interpretation.
