# ArtELingo content-first hierarchical refinement pilot

Generated automatically, 2026-09-22 19:13:27.

This pilot first fixes a CLIP content-parent Leiden partition, then permits a GoEmotions-only Leiden split *inside* each parent. It never merges nodes across content parents, so affect cannot create the cross-genre merges seen in flat fusion.

**AMI caveat:** AMI is not monotone under refinement. Splitting a community can raise or lower chance-corrected AMI even when the split has no real affect information. Therefore the raw hierarchical values alone cannot answer the question; only its comparison with the exactly size-matched random-split Control A can.

**Setup:** 61,402 deduplicated paintings; content graph K=20, alpha=0.5; affect subgraphs use K_sub=min(20, N_parent - 1); Leiden seeds 42 and 43; genre overlap n=1144.

## Content-only reproduction check

The parent labels are the untouched content-only graph partition. Compare this run's computed values to the reference emotion AMI=0.0593 / V-measure=0.0600 and genre AMI=0.4384 / V-measure=0.4572 before interpreting refinements.

## Split diagnostics

There were 28 content parents. 23 were eligible (N >= 50); 23 eligible parents met the stability threshold (ARI >= 0.5); and 23 were stable and genuinely split (>=2 children). Those genuinely-split parents contain 61,298/61,402 nodes (99.8%).

Across all children in genuinely-split parents, child size was min/median/max = 5/119.0/425.

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Content-only (reference) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
| Content-parent partition (this run's reproduction check) | 0.0593 | 0.0600 | 0.4384 | 0.4572 |
| Hierarchical (content-parent + affect-child) | 0.1072 | 0.1141 | 0.1954 | 0.3901 |
| Control A — random split (matched sizes) | 0.0362 | 0.0437 | 0.2014 | 0.3904 |
| Control B — content re-split (not size-matched) | 0.0671 | 0.0717 | 0.3076 | 0.4293 |

## Decision-rule conclusion

Decision rule: real, non-granularity-driven emotion signal requires hierarchical emotion AMI - Control A emotion AMI >= 0.02. Here the difference is 0.0709; Real emotion signal above Control A, but genre retention was lost. Separately, hierarchical genre AMI=0.1954 is below the 0.3507 80%-retention floor.

## Control B child-count caveat

Control B is not size-matched. Across the genuinely-split parents, the mean absolute difference between its natural content-child count and the real affect-child count was 7.96 (min/median/max 0/8.0/15). Per-parent counts were logged during the run.
