# Research Brief

**Title**: Judging a grouping before use without labels, and sibling-aware agreement (T5)
**Date**: 2026-10-05
**Mode**: Quick Research Brief (ARS deep-research)
**AI Disclosure**: This brief was produced with AI-assisted research tools (Claude). Citations were checked against retrieved pages; the reading depth of each is stated in "Brief citation check" and "Not verified". Formulas marked "reader-inferred" are our own constructions, not literature results.

---

## Executive Summary

We asked which label-free criteria can judge a grouping before the reader uses it, and how to make the agreement score p_img · p_txt treat sibling groups as related. Stability criteria (Ben-Hur et al., 2002; Lange et al., 2004) are well established for choosing K, but theory shows they depend on the clustering objective and, in the form used on its own, miss over-coarse solutions (Ben-David et al., 2006; Mourer et al., 2023). We found no study, within this search, showing that label-free criteria predict downstream usefulness of a grouping in a head-plus-reader pipeline; comparative studies find no universally best index (Arbelaitz et al., 2013). Placeability (cross-modal mutual information of paired rows) has a close precedent in IIC's objective (Ji et al., 2019) but not as a selection criterion. For agreement, a soft-cosine bilinear form, a co-membership matrix and a hierarchy kernel are cheap (under 2 GFLOP for 100k pairs); Sinkhorn OT is feasible but about 50 times costlier.

---

## Background & Research Question

### Context
The reader scores an image and a caption by the dot product of their posteriors over a grouping's groups. Two gaps follow. First, groupings are chosen by told or reader margins, which read labelled episodes (brief §4 R5, §9). Second, siblings of one property (sadness over clusters 38, 3, 59, 32, 35) score near 0 when image and caption land in different siblings (R3).

### Research Question
> Which label-free criteria select groupings that the reader can use, and what group-similarity-aware agreement score reduces to the plain dot product when groups are unrelated?

### Scope
- **In scope**: stability, cross-view predictability, non-redundancy measures and the evidence that they predict usefulness; soft similarity, hierarchy kernels and OT for agreement.
- **Out of scope**: running the checks of brief §10, fusion designs (T1), losses (T3), stage-2 interface (T6).

---

## Key Findings

### Finding 1: Stability is a mature, objective-dependent criterion for K
Stability methods choose the number of clusters whose solutions are most reproducible under resampling or perturbation (Ben-Hur et al., 2002; Lange et al., 2004), and von Luxburg (2010) surveys the theory. Ben-David et al. (2006) show that for large samples stability is fully determined by the behaviour of the objective the algorithm minimises, so it says whether the objective has a unique minimiser, not whether the groups are useful. Mourer et al. (2023) argue stability alone cannot detect too few clusters and add a second criterion: no stable partition inside a cluster.

**Evidence strength**: Strong for the theory; Moderate for coarse-solution failure (Mourer et al. read at abstract level).
**Source(s)**: Ben-Hur et al. (2002); Lange et al. (2004); von Luxburg (2010); Ben-David et al. (2006); Mourer et al. (2023).

### Finding 2: No evidence found that label-free validity criteria predict downstream usefulness
Arbelaitz et al. (2013) compared 30 validity indices over 6,480 configurations; per the search record we read, no index was universally best and overlap and noise degraded all. Gösgens et al. (2021) address the converse problem, that external indices disagree and favour different algorithms. To our knowledge, within this search, no paper tests label-free criteria against a downstream head-plus-reader margin like ours. Our brief §10 check 1 would therefore be new evidence, not a replication.

**Evidence strength**: Moderate for "no universal index"; Emerging for the absence claim (bounded).
**Source(s)**: Arbelaitz et al. (2013); Gösgens et al. (2021).

### Finding 3: Placeability has a precedent as an objective, not as a diagnostic
IIC trains classifiers by maximising mutual information between the class assignments of paired samples (Ji et al., 2019). Our placeability, the MI of image-head and caption-head assignments of one row against random pairs, is the same quantity evaluated after the fact. Meilă's (2007) variation of information gives a metric view of the same MI/entropy terms, useful for comparing groupings and for non-redundancy (low MI between groupings), but we found no validation of MI as a predictor of usefulness. Caveat (reader-inferred): MI is large for many fine groups even when head accuracy is low, so a chance-corrected form against shuffled pairs, as proposed, is needed; the sweep's rising cluster lift with falling head accuracy (brief §2) is the pattern to expect.

**Evidence strength**: Moderate (objective), Emerging (as criterion).
**Source(s)**: Ji et al. (2019); Meilă (2007).

### Finding 4: Class-similarity kernels and hierarchy-aware scoring exist and reduce to the plain case
The soft cosine measure inserts a feature-similarity matrix into the cosine and falls back to ordinary cosine when no feature similarity exists (Sidorov et al., 2014). Bertinetto et al. (2020) use a class hierarchy to make errors between close classes cheaper in a cross-entropy loss. Both rely on externally supplied similarity (a dictionary, WordNet), not on a label-free S, so building S from centroids or a Leiden hierarchy is our extension.

**Evidence strength**: Strong for the form; Emerging for label-free S.
**Source(s)**: Sidorov et al. (2014); Bertinetto et al. (2020).

### Finding 5: OT with a ground cost between groups is established and tractable
Word Mover's Distance compares two distributions over embedded items with an earth-mover cost (Kusner et al., 2015); Sinkhorn's entropic regularisation makes it orders of magnitude faster than exact LP (Cuturi, 2013).

**Evidence strength**: Strong.
**Source(s)**: Kusner et al. (2015); Cuturi (2013).

---

## Analysis & Implications

### What This Means
Stability will tell us whether a Leiden or k-means setting is reproducible, not whether it is usable; coarse, content-dominated solutions can be perfectly stable. Because the told margin was nearly flat in group count while head accuracy fell (brief §2), the criteria worth checking are those tied to the reader's mechanism: placeability (R2), stability of the heads' posteriors, and non-redundancy (R4). Each is a hypothesis for check 1, with the told margin as the outcome, over the 9 + 8 sweep cells; with 17 cells the rank correlation will have wide intervals.

### Recommendations
1. Use stability only as a gate (reject unstable settings), and report it beside placeability; do not select on it alone (Ben-David et al., 2006; Mourer et al., 2023).
2. Run brief §10 check 1 with placeability against a shuffled-pair null, plus head-posterior stability across two 60k-row draws, and report rank correlation with told margins.
3. Try the three formulas below on existing Leiden groups (check 2), unnormalised first, with the S = I run as the matched control.

---

## Formulas for sibling-aware agreement

Notation: p, q in the K-simplex (image and caption posteriors); K = 30 to 120.

**F1. Soft-cosine bilinear form, S from centroids (reader-inferred, form from Sidorov et al., 2014).**
S_jk = max(0, cos(μ_j, μ_k))^γ, μ = group centroid in the grouping's source space (GoEmotions probabilities, CLIP features), γ ≥ 1 sharpens; unit diagonal. Score: a₁ = pᵀSq. Normalised: a₁ / √(pᵀSp · qᵀSq). With S = I the unnormalised score is exactly the plain dot product (the normalised one is cosine of posteriors). Cost: P_img S once (N·K²), then a row-wise dot per pair; 100k pairs at K = 120 is about 1.4 GFLOP, seconds. Property: the max(0, ·)^γ cut does not guarantee PSD; add a small ridge or use the eigen-clipped matrix if normalising.

**F2. Co-membership S, with a cross-modal variant (reader-inferred).**
(a) Within-source: S_jk = ⟨c_j, c_k⟩ / (‖c_j‖‖c_k‖), c_j = column j of the soft-assignment matrix of scorer-train rows (rows that fall in both groups). (b) Cross-modal: C = P_imgᵀ P_txt over paired rows of one grouping, S = row-normalised C plus I; this learns which image group tends to accompany which caption group, and uses no labels or support pairs (a leakage check on which rows enter C is needed). Equals the dot product when columns are disjoint (hard, non-overlapping) or C is diagonal. Cost: one K×N by N×K product, then as F1. Risk: where placement is noisy, C is blurred and S inflates agreement for random pairs; calibrate against shuffled pairs.

**F3. Hierarchy kernel (reader-inferred; hierarchy-derived distances after Bertinetto et al., 2020).**
Build nested levels ℓ = 1..L by multi-resolution Leiden or agglomerative merging of centroids; A_ℓ is the K_ℓ × K ancestor-aggregation matrix. Score: a₃ = Σ_ℓ w_ℓ (A_ℓ p)·(A_ℓ q) = pᵀ(Σ_ℓ w_ℓ A_ℓᵀA_ℓ)q, w_ℓ ≥ 0, Σ w_ℓ = 1. S is PSD by construction and S = I when only the finest level has weight; weights are the single hyperparameter. Nested resolutions are not guaranteed by Leiden; use agglomerative cuts if they are not. Cost: as F1 with a precomputed S; interpretable (siblings share all coarser levels).

**OT alternative (Kusner et al., 2015; Cuturi, 2013).** Cost C_jk = 1 − S_jk, agreement a_OT = 1 − W_ε(p, q; C) / max C. With C = 1 − I it gives Σ_j min(p_j, q_j), the overlap, not the dot product, so no exact reduction. Cost: Sinkhorn iterations of K² each (about 50), so 100k pairs at K = 120 is about 7·10¹⁰ flops, minutes on the CPU or seconds on a GPU batch; exact LP is far costlier. We expect little gain over F1 to F3 for 30 to 120 groups and would use OT only as a check.

---

## Bearing on the designs P0, L, G, E

- **P0**: stability and placeability can screen each source's Leiden setting before the reader sees it (reader-inferred; precedents Ben-Hur et al., 2002; Ji et al., 2019). Sibling-aware agreement is the only change to the reader, so P0 plus F1 to F3 is a clean control (reader-inferred).
- **L**: the non-redundancy term has an MI/VI formulation (Meilă, 2007); conditional MI is not covered by anything we verified (reader-inferred). A refinement model that sharpens placement may reduce sibling mixing and so the gain from S; compare S gains before and after (reader-inferred).
- **G**: multi-resolution community levels give the F3 hierarchy directly (reader-inferred); layer-specific stability is untested in what we read.
- **E**: IIC's paired-sample MI is a trainable form of placeability (Ji et al., 2019), so the criterion would be circular if E trains on it; evaluate E on a criterion it does not optimise (reader-inferred).

---

## Limitations
- Search limited to roughly a dozen queries; four publisher pages (Ben-Hur, Lange, Meilă, Arbelaitz) were not readable here, so claims about their content are kept minimal.
- No evidence was found on cross-view predictability as a grouping criterion beyond IIC; the absence is bounded to this search.
- All formulas are untested; computation estimates are arithmetic, not benchmarks.

---

## References

Arbelaitz, O., Gurrutxaga, I., Muguerza, J., Pérez, J. M., & Perona, I. (2013). An extensive comparative study of cluster validity indices. *Pattern Recognition, 46*(1), 243-256. https://doi.org/10.1016/j.patcog.2012.07.021

Ben-David, S., von Luxburg, U., & Pál, D. (2006). A sober look at clustering stability. In *Learning Theory (COLT 2006)* (pp. 5-19). Springer. https://doi.org/10.1007/11776420_4

Ben-Hur, A., Elisseeff, A., & Guyon, I. (2002). A stability based method for discovering structure in clustered data. *Pacific Symposium on Biocomputing*, 6-17. World Scientific.

Bertinetto, L., Mueller, R., Tertikas, K., Samangooei, S., & Lord, N. A. (2020). Making better mistakes: Leveraging class hierarchies with deep networks. *CVPR 2020*. https://arxiv.org/abs/1912.09393

Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transportation distances. *Advances in Neural Information Processing Systems 26*. https://arxiv.org/abs/1306.0895

Gösgens, M. M., Tikhonov, A., & Prokhorenkova, L. (2021). Systematic analysis of cluster similarity indices: How to validate validation measures. *Proceedings of ICML 2021* (PMLR 139, pp. 3799-3808). https://proceedings.mlr.press/v139/gosgens21a.html

Ji, X., Henriques, J. F., & Vedaldi, A. (2019). Invariant information clustering for unsupervised image classification and segmentation. *ICCV 2019*. https://arxiv.org/abs/1807.06653

Kusner, M. J., Sun, Y., Kolkin, N. I., & Weinberger, K. Q. (2015). From word embeddings to document distances. *Proceedings of ICML 2015* (PMLR 37, pp. 957-966). https://proceedings.mlr.press/v37/kusnerb15.html

Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-based validation of clustering solutions. *Neural Computation, 16*(6), 1299-1323.

Meilă, M. (2007). Comparing clusterings: An information based distance. *Journal of Multivariate Analysis, 98*(5), 873-895. https://doi.org/10.1016/j.jmva.2006.11.013

Mourer, A., Forest, F., Lebbah, M., Azzag, H., & Lacaille, J. (2023). Selecting the number of clusters K with a stability trade-off: An internal validation criterion. *PAKDD 2023*. https://arxiv.org/abs/2006.08530

Sidorov, G., Gelbukh, A., Gómez-Adorno, H., & Pinto, D. (2014). Soft similarity and soft cosine measure: Similarity of features in vector space model. *Computación y Sistemas, 18*(3), 491-504.

von Luxburg, U. (2010). Clustering stability: An overview. *Foundations and Trends in Machine Learning, 2*(3), 235-274. https://arxiv.org/abs/1007.1075

---

## Brief citation check

| Brief mention | Result |
|---|---|
| Ben-Hur et al. 2002 | Correct: authors, title, Pacific Symposium on Biocomputing, 2002, pp. 6-17 (bibliographic record via search; publisher page returned 403). Volume number not confirmed. |
| Lange et al. 2004 | Correct: Neural Computation 16(6), 1299-1323 (bibliographic record via search; MIT Press page 403). |
| von Luxburg 2010 | Correct, arXiv abstract page read: Foundations and Trends in ML 2(3), 235-274. |
| Other T5 items (hierarchy-based kernels, OT) | Not named as papers in the brief; supplied here (Sidorov, Bertinetto, Kusner, Cuturi). |

No corrections needed.

## Not verified

- Content of Ben-Hur et al. (2002), Lange et al. (2004), Meilă (2007) and Arbelaitz et al. (2013) beyond bibliographic data and search-record summaries; publisher pages were blocked. Claims about them are limited to what von Luxburg's survey context and those summaries say.
- Whether Ben-Hur et al. report the coarse-solution failure: not claimed.
- Kusner et al. (2015) and Cuturi (2013) cost figures: abstract-level only; our flop counts are our arithmetic.
- Candidate papers on non-redundancy (Cui, Fern and Dy 2007; Niu, Dy and Jordan 2010) belong to T1 and were not checked here.
