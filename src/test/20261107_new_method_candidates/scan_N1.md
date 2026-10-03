# Three-way scan for N1 (centered covariance agreement rule)

Date: 2026-10-04. Mode: ARS deep-research `three-way-scan` (bibliography + source verification). Candidate: N1 in `candidates_draft.md` §2 (weights w_l = ReLU(cov_S(a_I,l, a_T,l) − cov_C(a_I,l, a_T,l)) over a learned sparse shared image-text basis, scored with centered codes).

Search bound. WebSearch queries (standard mode): "Conditional Similarity Networks Veit Belongie Karaletsos CVPR 2017"; "Learning distance functions using equivalence relations ... RCA ICML 2003"; "Large scale metric learning from equivalence constraints KISSME CVPR 2012"; "Contextual Visual Similarity Wang Kitani Hebert"; "Rasiwasia A new approach to cross-modal multimedia retrieval CCA"; "few-shot image-text retrieval aspect-conditioned similarity from example pairs covariance feature weighting test-time"; "metric learning from very few equivalence constraints covariance estimation shrinkage small sample Mahalanobis". Pages opened with WebFetch: KISSME PDF (TU Graz), RCA JMLR page, CSN arXiv abstract page (CVPR page returned 403), Contextual Visual Similarity (ar5iv full text), CRML arXiv abstract page, Rasiwasia PDF (binary, unparsed). The project's own review files were read to avoid re-reporting (the 2026-10-24 novelty check already lists KISSME, RCA, CSN, CVS, CRML, CCA and a "diagonal KISSME" baseline). Not searched: Semantic Scholar, Google Scholar, kernel-alignment papers (Cortes; Cristianini), PLS literature. Those gaps limit the absence claims below.

## Shortlist

## Large Scale Metric Learning from Equivalence Constraints (KISSME)
Source: CVPR proceedings (PDF at TU Graz, opened) | Year: 2012 | Link: https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/lrs/pubs/koestinger_cvpr_2012.pdf
- WHY: Mahalanobis learners needed slow optimisation and full supervision; the authors wanted a closed form from similar/dissimilar pairs.
- HOW: Gaussian likelihood-ratio test on pair differences gives M = Σ_S⁻¹ − Σ_D⁻¹ (covariances of pair differences), clipped to the PSD cone; PCA first.
- WHAT: Face verification (LFW), re-identification (VIPeR) and matching of unseen instances; trained on many pairs offline, one feature space.
  - Method weaknesses: read scope: sections (fetched PDF summarised by the tool; equations 12 to 16 were verified earlier in `2026-10-28_citation_check.md`). (a) Inverting a covariance of pair differences with few pairs is rank-deficient without PCA or shrinkage; reader-inferred. (b) Only "limited to learning a linear transformation of the input space" (author-acknowledged, conclusion section as returned by the tool; exact section not located). Checklist (empirical): small-n regime checked: none found in the text I could access.

## Learning a Mahalanobis Metric from Equivalence Constraints (RCA)
Source: JMLR 6 | Year: 2005 (conference version ICML 2003) | Link: https://www.jmlr.org/papers/v6/bar-hillel05a.html
- WHY: Side information as chunklets of points known to share a class, without class labels.
- HOW: Whitens by the within-chunklet covariance (maximum likelihood under Gaussian assumptions); positive constraints only.
- WHAT: Improves clustering and classification over using the raw metric (abstract).
  - Method weaknesses: read scope: abstract_only for this pass (the full text was checked on 2026-10-28). Not assessed (read scope: abstract_only).

## Conditional Similarity Networks
Source: CVPR 2017 (arXiv 1603.07810, opened; CVF page returned 403 but the search result listed it) | Year: 2017 | Link: https://arxiv.org/abs/1603.07810
- WHY: Images are similar in several, sometimes contradictory, ways, which one embedding cannot hold.
- HOW: A disentangled embedding with learned per-condition masks that select and reweight dimensions; the condition is given as an id.
- WHAT: Beats separately trained specialists on triplet questions across notions (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Contextual Visual Similarity
Source: arXiv 1612.02534 (ar5iv full text opened) | Year: 2016 | Link: https://arxiv.org/pdf/1612.02534
- WHY: Visual similarity is ambiguous until a context is given.
- HOW: Learns per-dimension feature weights by gradient descent on a triplet ranking loss, at query time, from k positive and negative images, with a norm regulariser.
- WHAT: k = 1, 3, 5 positives; MAP with fc7 features rises 0.440, 0.519, 0.557 (Table 1 as returned by the tool). Positives "contain the target attribute but in a different object category with the query" (shares the query's value), image-only.
  - Method weaknesses: read scope: sections (ar5iv, tool-summarised). (a) Single-attribute labels: author-acknowledged, §4.3, "A more comprehensive evaluation requires multi-label data, which we leave for future work". (b) Positives share the query's attribute value, so it does not test value-disjoint transfer; reader-inferred.

## Few-shot Metric Learning: Online Adaptation of Embedding for Retrieval (CRML)
Source: ACCV 2022 (arXiv 2211.07116, opened) | Year: 2022 | Link: https://arxiv.org/abs/2211.07116
- WHY: Learned metrics fail on unseen classes under domain shift.
- HOW: Online adaptation of intermediate channels from a few labelled target examples, meta-trained.
- WHAT: Gains on miniImageNet, CUB-200-2011, MPII, miniDeepFashion, larger under large domain gaps (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## A New Approach to Cross-Modal Multimedia Retrieval (CCA on image and text)
Source: ACM Multimedia 2010 (MIT-hosted PDF located; binary text not parsed) | Year: 2010 | Link: https://www.mit.edu/~rplevy/papers/rasiwasia-etal-2010-acm.pdf
- WHY: Joint modelling of the image and text parts of documents for retrieval in both directions.
- HOW: Canonical correlation analysis between a topic-model text space and a bag-of-SIFT image space.
- WHAT: Cross-modal correlation and semantic abstraction both improve retrieval (abstract as returned by search).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Cross-paper synthesis
- Common WHY: similarity is relative to a context, and the context should be extracted from side information (pairs, masks, examples) instead of a single fixed metric.
- Divergent HOW: KISSME and RCA estimate global second moments of within-pair differences (closed form, offline, many pairs); CSN picks learned masks by id; CVS fits per-dimension weights at test time by gradient from 1 to 5 shared-value examples; CRML meta-learns online adaptation; CCA estimates cross-modal covariance offline.
- Strongest WHAT: CVS shows a per-dimension weight rule can improve with k = 1 to 5 examples; KISSME and RCA show closed-form pair statistics work with thousands of pairs; CCA shows cross-modal covariance carries retrieval signal.
- Unresolved gap: no source tests a covariance or difference-variance estimator from about 4 pairs, with cross-item cross-modal pairs, whose values never include the query's.

## Verdict for N1

**Exists already: partially.** Searches within the bound above found no paper that weights shared image-text factors by support-minus-contrast cross-modal covariance across pairs, but each ingredient is published: contrast-of-statistics (KISSME), pair-based equivalence (RCA), per-dimension test-time weights from few examples (CVS), masks over dimensions (CSN), cross-modal covariance (CCA). Absence is claimed only for the exact combination and within this bound.

**Reviewer citation.** KISSME first: M = Σ_S⁻¹ − Σ_D⁻¹ is "similar-pair statistic minus dissimilar-pair statistic". A reviewer would also cite CVS for few-example per-dimension weights and CCA (Rasiwasia et al. 2010) for cross-modal covariance. The difference to state: N1 uses covariance across pairs of the two modalities' codes (a diagonal cross-covariance, PLS/CCA numerator, rewarding shared pair-to-pair variation), whereas KISSME uses the covariance of within-pair differences. By the identity var(a_I − a_T) = var(a_I) + var(a_T) − 2cov(a_I, a_T), a diagonal KISSME on differences contains N1's cross-covariance but also the marginal variances (reader-inferred, my derivation). The project's planned "diagonal KISSME from 4+4 pairs" baseline is therefore the necessary control, ideally together with a diagonal cross-covariance on raw CLIP.

**Evidence for the regime.**
- Few examples: CVS reports gains from k = 1 to 5, but with gradient training, image-only data and positives sharing the query's value.
- Value-disjoint: none found. KISSME's matching of unseen instances is offline and in one space (inferred from its protocol; tables not re-checked).
- Cross-modal: CCA works offline on many pairs; no few-pair evidence found.
- A rough statistical point (reader-inferred, untested): a sample correlation from n = 4 pairs has 3 degrees of freedom, so any single factor's weight is very noisy; the rule depends on summing over 32 factors and two conditions, as the draft says.

**Stronger published variant to adopt.** None found. The nearest stronger relatives (KISSME, RCA, CVS) should be run as baselines on the same 4+4 episodes rather than adopted, and KISSME's shrinkage or PCA would be the first repair if N1's noise is the issue. Do not write the rule as new; write it as "a centered cross-modal diagonal variant of KISSME/CCA applied to a learned basis", as the 2026-10-24 check already advises.

## AI-assistance note
This scan was produced by a Claude agent using ARS three-way-scan rules, with web search and page fetches. Every listed link was opened or returned by a search listing, but several pages were read only as tool-generated summaries or abstracts, and the Rasiwasia, CSN CVF and CRML pages were not read in full; the method-weakness fields say so. Statements marked reader-inferred are the agent's own reasoning. A scholar should read the primary papers before citing them.
