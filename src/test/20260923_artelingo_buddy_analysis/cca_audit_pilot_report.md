# ArtELingo linear CCA + conditional-residual audit

Generated automatically, 2026-09-22 20:04:37.

## What this audit tests

CCA asks whether matched CLIP content and GoEmotions affect features contain a linear subspace that stays correlated on paintings the fit never saw. The residual test removes the part of affect linearly predicted by content, then asks whether the leftover affect still carries emotion structure. Together these are a go/no-go gate for the learned two-teacher student, SNF, and co-regularized spectral clustering ideas: they distinguish no useful signal, a shared signal, and affect that is useful but orthogonal to content.

## Held-out CCA

CCA fit only on train paintings after a train-only 50-component content PCA. Each null fit scrambles train content/affect correspondence, but always evaluates on the same correctly paired held-out paintings.

| component | held-out correlation | null mean | null 95th percentile | > 0.15 | > null 95th |
|---:|---:|---:|---:|---|---|
| 1 | 0.7285 | 0.0081 | 0.0699 | pass | pass |
| 2 | 0.5531 | -0.0037 | 0.0594 | pass | pass |
| 3 | 0.4985 | -0.0089 | 0.0875 | pass | pass |
| 4 | 0.4197 | 0.0013 | 0.0740 | pass | pass |
| 5 | 0.4250 | -0.0066 | 0.0514 | pass | pass |
| 6 | 0.2791 | 0.0082 | 0.0846 | pass | pass |
| 7 | 0.2896 | -0.0043 | 0.0274 | pass | pass |
| 8 | 0.2371 | -0.0062 | 0.0497 | pass | pass |
| 9 | 0.2315 | 0.0010 | 0.0659 | pass | pass |
| 10 | 0.1933 | -0.0014 | 0.0551 | pass | pass |

**Stable shared signal found: yes.** The predeclared rule is that the first component must exceed both 0.15 and its own permutation-null 95th percentile.

## Held-out edge retrieval

For the same 2,000 held-out paintings, this measures how much each true content or affect graph's neighborhood is retained in the joint CCA-space mutual-kNN graph. The chance floor replaces each true neighbor set with a random set of equal size.

| reference graph | joint-CCA neighbor recall | random-neighbor chance floor |
|---|---:|---:|
| content | 0.0718 | 0.0011 |
| affect | 0.0783 | 0.0012 |

## Conditional residual

Held-out R² for linear content → affect prediction: **0.1157**. This is the fraction of held-out affect variance explained by content.

| signal | emotion AMI | emotion V-measure | genre AMI | genre V-measure |
|---|---:|---:|---:|---:|
| Raw GoEmotions-affect-only reference | 0.1180 | — | 0.0396 | — |
| Residual-affect-only (genre n=1,144) | 0.0914 | 0.0921 | 0.0302 | 0.0649 |

**Informative residual: no.** The predeclared rule is residual emotion AMI ≥ 0.0944 (80% of the raw GoEmotions reference, 0.1180).

## Final synthesis

Shared signal found: a small learned-student pilot is licensed, consistent with the second brainstorm's ranking.
