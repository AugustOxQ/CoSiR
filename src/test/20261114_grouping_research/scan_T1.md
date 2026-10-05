# Three-way scan for T1 (fusing several groupings or graphs, then splitting)

Thread: T1 of `research_brief.md` §7. Scan date 2026-10-05. Mode: ARS deep-research `three-way-scan` (bibliography + source verification), standard search depth. Retrieved text was treated as data.

**Read scope.** Per paper, stated in its entry. In summary: `abstract_only` for most; `partial full text` (ar5iv HTML rendering of the paper, read through WebFetch's summariser with targeted questions, not by me line by line) for MFCVAE, SCE-Net and DiscoverNet; `metadata_only` (Crossref or Semantic Scholar record, no abstract page opened) for Gondek and Hofmann and for Cui, Fern and Dy. Method weaknesses follow SKILL.md rule 5: `not assessed (read scope: abstract_only)` where only the abstract was read; where partial full text was read, the weakness is named with its condition and labelled author-acknowledged (with locator) or reader-inferred. Several PDFs (JMLR Strehl and Ghosh, SCE-Net, MFCVAE, Niu) were fetched as binary and could not be converted here (no PDF tool); they were not read as PDFs.

**AI disclosure.** Produced by a Claude agent (Sonnet 5.5) using WebSearch and WebFetch. Every shortlisted item was opened at a publisher, arXiv, proceedings, ORA, Crossref, S2 or docs page; claims go no further than what that page showed. A human should read Niu et al., Cui et al., Gondek and Hofmann, and Mucha et al. in full before citing method details.

**Queries run (WebSearch).** (1) "Learning Similarity Conditions Without Explicit Supervision Tan ... SCE-Net"; (2) "Multiple Non-Redundant Spectral Clustering Views Niu Dy Jordan ICML 2010"; (3) "Community Structure in Time-Dependent, Multiscale, and Multiplex Networks Mucha ..."; (4) "Strehl Ghosh Cluster Ensembles ... JMLR 2002"; (5) "DiscoverNet Ye Shi Zhan CVPR 2022 ..."; (6) "Gondek Hofmann Non-redundant data clustering ICDM 2004 KAIS 2007" (two variants); (7) "Cui Fern Dy Non-redundant multi-view clustering via orthogonalization ICDM 2007"; (8) "deep multiple clustering non-redundant Miklautz ENRC"; (9) "Multi-MaP multiple clustering ... IC|TC"; (10) "multi-view clustering one view dominates fused clustering weak views degrade view weighting"; (11) "multilayer community detection layer-specific community structure ...". Plus direct fetches of the arXiv, NeurIPS, ORA, AAAI, mlanthology, Crossref, S2 and leidenalg pages listed below.

**Sources searched.** arXiv, NeurIPS/NIPS proceedings, AAAI OJS, ORA (Oxford), mlanthology, Crossref, Semantic Scholar (one query; two others hit 429), leidenalg ReadTheDocs, PMC, general web search. **Not searched.** OpenReview, ACL Anthology, CVF full-text search, Google Scholar, DBLP (Anubis block), Springer, IEEE Xplore and ACM DL (403 or redirect), PDF full texts. No query on SCAN-style or contrastive clustering heads (T3), nor on identifiability (T2).

## Non-redundant clustering given an existing clustering

## Non-redundant data clustering
Source: Knowledge and Information Systems 12, pp. 1-24 (Crossref: issued online 2006, volume year 2007); conference version IEEE ICDM 2004, pp. 75-82 (from search result only) | Year: 2004 / 2006-07 | Link: https://link.springer.com/article/10.1007/s10115-006-0009-7 (Springer page redirected; metadata confirmed at https://api.crossref.org/works/10.1007/s10115-006-0009-7)
- Authors confirmed: David Gondek, Thomas Hofmann.
- WHY: A user who already knows one grouping wants a different, complementary one; ordinary clustering returns the dominant, known structure again. (Search-result summary, not the abstract page.)
- HOW: Extends the information bottleneck to a coordinated conditional information bottleneck (CCIB) that maximises conditional mutual information of the new clustering with the data, given the known clustering. (Search-result summary only.)
- WHAT: Not assessed (read scope: metadata_only; Crossref confirmed title, authors, journal, volume, pages; the abstract page itself did not load).
  - Method weaknesses: not assessed (read scope: metadata_only).
- Verification status: citation verified (Crossref); content claims unverified at source.

## Non-redundant Multi-view Clustering via Orthogonalization
Source: IEEE ICDM 2007 (S2 lists the venue as "Industrial Conference on Data Mining", which is a garbled string; DOI 10.1109/ICDM.2007.94 and DBLP key conf/icdm/CuiFD07 are the ICDM ones), pp. 133-142 from a search result | Year: 2007 | Link: https://doi.org/10.1109/ICDM.2007.94 (S2 record opened via the Semantic Scholar API)
- Authors confirmed: Ying Cui, Xiaoli Z. Fern, Jennifer G. Dy.
- WHY, HOW, WHAT: not assessed (read scope: metadata_only; no abstract was opened). Search snippets only say it "established a direction" of multi-view clustering by orthogonalization, which I do not rely on.
  - Method weaknesses: not assessed (read scope: metadata_only).
- Verification status: citation verified (title, authors, year, DOI); content unverified.

## Multiple Non-Redundant Spectral Clustering Views
Source: ICML 2010, pp. 831-838 | Year: 2010 | Link: https://mlanthology.org/icml/2010/niu2010icml-multiple/
- Authors confirmed: Donglin Niu, Jennifer G. Dy, Michael I. Jordan.
- WHY: High-dimensional data admit several clusterings in different feature subspaces; one partition hides the others.
- HOW: Learns non-redundant subspaces and a spectral clustering in each, with a dimensionality-reduction step and a redundancy penalty between views (mlanthology summary of the abstract, which quotes "simultaneously learns non-redundant subspaces that provide multiple views and finds a clustering solution in each view").
- WHAT: The page gives no numbers.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Deep Embedded Non-Redundant Clustering (ENRC)
Source: AAAI 2020, vol. 34(4), pp. 5174-5181, DOI 10.1609/aaai.v34i04.5961 (DOI and pages from a search result; the OJS page confirmed title, authors, venue, year) | Year: 2020 | Link: https://ojs.aaai.org/index.php/AAAI/article/view/5961
- Authors confirmed: Lukas Miklautz, Dominik Mautz, Muzaffer Can Altinigneli, Christian Böhm, Claudia Plant.
- WHY: Complex data such as images can be clustered in several valid ways; deep clustering finds one.
- HOW: An autoencoder embeds the data; each embedded dimension is (softly) assigned to one of several clusterings, which gives separate clusterings with different dimensionalities.
- WHAT: The abstract says the method can group image objects by colour, material or shape without feature engineering, and that merging representation learning with non-redundant clustering beats existing approaches. No numbers on the page.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Multi-facet and multi-view clustering

## Multi-Facet Clustering Variational Autoencoders (MFCVAE)
Source: NeurIPS 2021 / arXiv 2106.05241 | Year: 2021 | Link: https://arxiv.org/abs/2106.05241
- Authors confirmed: Fabian Falck, Haoting Zhang, Matthew Willetts, George Nicholson, Christopher Yau, Chris Holmes.
- Read scope: abstract page plus partial full text (ar5iv render, targeted questions on results, limitations and mechanism).
- WHY: Deep clustering returns one partition although data carry several clusterable properties.
- HOW: A ladder VAE with several latent layers ("facets"), each with its own mixture-of-Gaussians prior; prior independence across facets plus the ladder's different depths push facets towards distinct subspaces; progressive training adds facets one at a time so each keeps representing the same aspect.
- WHAT: Facet recovery on MNIST, 3DShapes and SVHN. Supervised probes: the facet matching the label gives 94 to 100% accuracy, the other facet 17 to 73% (Table 1). Unsupervised clustering accuracy: MNIST 92.02 ± 3.18, SVHN 56.25 ± 0.93, 3DShapes floor colour 99.46 ± 1.10 and object shape 88.47 ± 1.82 (Table 2). Facet-to-property assignment is emergent, not guaranteed.
  - Method weaknesses: (i) no label-free way to pick the number of facets J or clusters per facet K_j: author-acknowledged (§5, Conclusion; the summariser's quote: "lack of a procedure to find good hyperparameters through a metric known at training time"). (ii) Compositionality "works less so on SVHN", attributed by the authors to a more diverse dataset and lower fit (§4.2, author-acknowledged). (iii) Reader-inferred: the evidence is on synthetic or near-synthetic benchmarks whose facets are independent by construction (digit and style; floor colour and shape); where facets are correlated, as emotion, genre and style are in artworks, the independence prior may split a correlated property across facets or merge it. Not tested in the paper.

## Co-regularized Multi-view Spectral Clustering
Source: NIPS 2011 (Advances in Neural Information Processing Systems 24) | Year: 2011 | Link: https://papers.nips.cc/paper/2011/hash/31839b036f63806cba3f47b93af8ccb5-Abstract.html
- Authors confirmed: Abhishek Kumar, Piyush Rai, Hal Daumé III (listed as "Hal Daume").
- WHY: With several data representations, using all views jointly should beat clustering each independently.
- HOW: Spectral clustering per view with co-regularisation terms that force the views' cluster memberships to agree ("corresponding data points in each view should have same cluster membership", abstract).
- WHAT: Reported better results than existing alternatives on synthetic and real data (abstract; no numbers on the page).
  - Method weaknesses: not assessed (read scope: abstract_only). Reader-inferred from the stated objective (agreement is enforced): the method returns one shared partition, so a view whose structure differs from the others is pulled toward them; it is the multi-view analogue of consensus, not of keeping layer-specific groupings.

## Balanced Multi-view Clustering (BMvC)
Source: arXiv 2501.02564 (preprint; venue not confirmed) | Year: 2025 | Link: https://arxiv.org/abs/2501.02564
- Authors confirmed: Zhenglai Li, Jun Wang, Chang Tang, Xinzhong Zhu, Wei Zhang, Xinwang Liu.
- WHY: In joint training, views with more discriminative information dominate the learning and other views are under-optimised (abstract, direct quote in the page summary).
- HOW: View-specific contrastive regularisation that adaptively modulates the gradient magnitudes per view.
- WHAT: The abstract reports improved clustering; no numbers read. This is the only direct statement of strongest-view dominance in a fused clustering that I opened, and it concerns deep joint training, not graph fusion.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Multiplex and multilayer community detection

## Community Structure in Time-Dependent, Multiscale, and Multiplex Networks
Source: Science 328(5980), 876-878 | Year: 2010 | Link: https://ora.ox.ac.uk/objects/uuid:aa9b41b7-d8f7-4e45-8fcf-efe95081b412 (DOI 10.1126/science.1184819; the Science page returned 403)
- Authors confirmed: Peter J. Mucha, Thomas Richardson, Kevin Macon, Mason A. Porter, Jukka-Pekka Onnela.
- WHY: Community detection on one static graph cannot handle networks that evolve, have several link types (multiplexity) or span scales.
- HOW: Generalises modularity to multislice networks: a quality function over slices linked through corresponding nodes; the application is U.S. Senate roll-call similarity networks.
- WHAT: A general framework; the ORA summary names no numbers.
  - Method weaknesses: not assessed (read scope: abstract_only). The inter-slice coupling that ties a node's community across slices is, per the leidenalg docs (below), what makes the partition shared across layers.

## Multiplex optimisation in leidenalg (software documentation, gray literature)
Source: leidenalg documentation, "Multiplex" page | Year: n/a (current stable docs) | Link: https://leidenalg.readthedocs.io/en/stable/multiplex.html
- WHY: Several graphs on one vertex set (layers) should yield one community structure that reflects all of them.
- HOW: Quality is the weighted sum q = Σ_k w_k q_k over layers; "each node is identified with a single community, and cannot have different communities for different graphs." Layer weights set relative importance; with a normalised partition type such as modularity, layers are weighted equally irrespective of their link counts unless weights are given. Negative layer weights (for example [1, -1]) give "many negative links between communities". A separate "slices" mode allows differing vertex sets and community membership per layer.
- WHAT: Documentation only; no experimental evidence of any kind. Gray literature.
  - Method weaknesses: not assessed (gray literature, no study). Reader-inferred: with shared membership and summed quality, the layer that has the most separable structure contributes the largest quality gain per move, so the fused partition may follow it; the docs do not say this.

## Community extraction in multilayer networks with heterogeneous community structure
Source: Journal of Machine Learning Research (accepted 2017; volume and pages not confirmed) / arXiv 1610.06511 | Year: 2016 (submitted), 2017 | Link: https://arxiv.org/abs/1610.06511
- Authors confirmed: James D. Wilson, John Palowitch, Shankar Bhamidi, Andrew B. Nobel.
- WHY: Communities in multilayer networks can differ across layers; shared-partition methods assume otherwise.
- HOW: "Multilayer Extraction" finds densely connected vertex-layer sets by a significance score against a random-graph null; communities can sit on a subset of layers and overlap.
- WHAT: Theoretical guarantees under a multilayer stochastic block model plus simulations and real applications (abstract; no numbers read).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Conditions from mixed similarities

## Conditional Similarity Networks (CSN)
Source: CVPR 2017 / arXiv 1603.07810 | Year: 2017 (arXiv 2016) | Link: https://arxiv.org/abs/1603.07810
- Authors confirmed: Andreas Veit, Serge Belongie, Theofanis Karaletsos.
- WHY: One feature space cannot serve several valid similarity notions.
- HOW: One embedding whose dimensions split into semantically distinct subspaces, with learned masks that select and reweight dimensions per similarity notion.
- WHAT: Beats separate per-notion networks on triplet questions (abstract); interpretable subspaces. The abstract does not state that the condition is given at test time.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Learning Similarity Conditions Without Explicit Supervision (SCE-Net)
Source: ICCV 2019 / arXiv 1908.08589 | Year: 2019 | Link: https://arxiv.org/abs/1908.08589
- Authors confirmed: Reuben Tan, Mariya I. Vasileva, Kate Saenko, Bryan A. Plummer.
- Read scope: abstract page plus partial full text (ar5iv render, targeted questions).
- WHY: Similarity depends on a condition (colour, category, shape) that is usually not annotated at test time.
- HOW: K parallel condition masks on the shared embedding; a weight branch takes the compared images' features, outputs a softmax over the K masks, and assigns each pair its mask mixture as a latent variable (head competition by softmax).
- WHAT: Zappos error 7.53% against CSN's 10.73% is quoted in the project review (T5 of the paper; I did not re-derive it). From this read: Fig. 4 t-SNE shows the four learned masks align with shoe class, closing mechanism, heel height and gender; Table 2 shows Polyvore Outfits AUC peaking at 5 masks (0.92; 0.86 with 1, 0.89 with 20), and Zappos best at 4 masks, which equals its number of ground-truth attributes.
  - Method weaknesses: (i) the number of masks K is a manual hyperparameter and picking it without labels is left as future work (author-acknowledged). (ii) Training triplets are sampled from ground-truth attributes (the model sees only similar and dissimilar, never the condition id), so the alignment of masks with single properties is obtained under attribute-structured triplet sampling; reader-inferred: with unstructured pooled edges, as in design E, the alignment is not shown by this paper.

## Identifying Ambiguous Similarity Conditions via Semantic Matching (DiscoverNet)
Source: CVPR 2022 / arXiv 2204.04053 | Year: 2022 | Link: https://arxiv.org/abs/2204.04053
- Authors confirmed: Han-Jia Ye, Yi Shi, De-Chuan Zhan. DiscoverNet is the method's name inside this paper (the brief's name is correct; the title is not "DiscoverNet").
- Read scope: abstract page plus partial full text (ar5iv render, targeted questions).
- WHY: Triplets under weak supervision carry no condition label; the same triplet can be right under one condition and wrong under another (aircraft is closer to bird by flight, to train by transport).
- HOW: Decompose-and-fuse: several condition-specific projections on a shared base embedding, a latent multinomial condition per triplet, a set module (order-agnostic, built from pairwise concatenations) and a semantic regulariser that pushes the condition distributions of a triplet and its reverse apart (histogram intersection kernel).
- WHAT: Introduces a coverage metric that aligns each embedding to an optimal ground-truth condition before scoring. Greedy accuracy on UT-Zappos-50k 77.84% against 72.21% for the best competing method; the best embedding count is near 8 for 8 conditions, with the method not requiring an exact match (Table 3, 2 to 10 embeddings).
  - Method weaknesses: (i) the authors acknowledge that the coverage criterion "does not fit the case when the goal is not to learn a model similar to its supervised counterpart" (author-acknowledged; locator not recorded by the summariser). (ii) Reader-inferred: evaluation is against ground-truth conditions whose attributes also generate the triplets, so condition recovery is shown only where the true conditions are a small known set of independent attributes.

## Consensus clustering (the contrast case)

## Cluster Ensembles: A Knowledge Reuse Framework for Combining Multiple Partitions
Source: Journal of Machine Learning Research 3, 583-617 | Year: 2002 | Link: https://www.jmlr.org/papers/v3/strehl02a.html
- Authors confirmed: Alexander Strehl, Joydeep Ghosh.
- WHY: Combine several partitionings of the same objects into one consolidated clustering without access to the original features or algorithms.
- HOW: A combinatorial problem of shared mutual information, solved by three heuristics (similarity-based reclustering, hypergraph partitioning, meta-clustering); a meta-algorithm picks the best of the three.
- WHAT: Tested on three scenarios, including clustering on different feature sets (the case closest to ours), different object subsets, and robustness from the same data (abstract). The page gives no numbers.
  - Method weaknesses: not assessed (read scope: abstract_only). Reader-inferred from the stated objective: the output is one partition, so which source put two rows together is not recorded, which is the reason the brief dropped this design.

## Cross-paper synthesis

**Common WHY.** Data carry several valid groupings and a single clustering or embedding returns the dominant one (Gondek and Hofmann, Niu et al., ENRC, MFCVAE, CSN, SCE-Net, DiscoverNet, Wilson et al.). The papers split on what to do about it: combine (Strehl and Ghosh, Kumar et al., Mucha et al., leidenalg), or keep several (the others).

**Divergent HOW.** (1) Fuse into one shared structure: consensus (Strehl and Ghosh), co-regularised agreement (Kumar et al.), multislice or multiplex modularity with a shared membership (Mucha et al., leidenalg). (2) Condition on what is known: Gondek and Hofmann (conditional mutual information given a known clustering) and, per its title, Cui et al. (not assessed). (3) Learn several clusterings jointly from one representation and separate them by independence or subspace assignment: Niu et al. (subspaces plus redundancy penalty), ENRC (dimension-to-clustering assignment), MFCVAE (independent priors across ladder levels). (4) Learn condition-specific subspaces with competing masks: CSN (given condition), SCE-Net and DiscoverNet (latent, softmax or multinomial). (5) Allow layer-specific membership: leidenalg slices, Wilson et al.

**Strongest WHAT.** Aligned, interpretable single-property structure is shown only in (3) and (4), on benchmarks where properties are independent or attribute-structured by construction: MFCVAE's facet probes (94 to 100% on the matching facet against 17 to 73% on the other), SCE-Net's four masks matching the four Zappos attributes, DiscoverNet's 77.84% against 72.21% on coverage-aligned accuracy. None of these fuses several sources first; they separate within one representation.

**Evidence on the two specific questions.**
- (a) The strongest view dominating a fused result. Directly stated for deep multi-view clustering (BMvC abstract: "views with more discriminative information could dominate the learning process"). Not found, within this search, for graph-level multiplex fusion; the leidenalg documentation says membership is shared and layer weights matter but does not report dominance. Our own percept result (content graph dominates, genre AMI 0.438 against 0.059 emotion; brief §2) is the only graph-level evidence, and it is internal.
- (b) Splitting after fusion recovering single-property structure. To our knowledge, within this search, no paper fuses several sources into one structure and then splits it back into single-property groupings. The nearest cases separate without fusing first (MFCVAE, ENRC, Niu et al.) or compute a new clustering conditional on an existing one (Gondek and Hofmann). The only positive fuse-then-split evidence is our own: hierarchical refinement of content communities by affect kept emotion AMI 0.107 against 0.036 for a random split of the same sizes (brief §2).

**Unresolved gap.** Whether a split mechanism driven by label-free signals (source provenance, modality, painting membership) separates correlated properties (emotion, style, genre) after fusion. All separation evidence above comes from independent or attribute-sampled properties, and none of the papers has a weak property that a strong one absorbs, which is our central risk.

## Bearing on P0, L, G, E (brief §5)

Statements are tagged **[lit]** (literature-supported, with citation) or **[inferred]** (reader-inferred).

**P0 (no merge, one grouping per source, sibling-aware agreement).**
- Supported: keeping separate source-specific groupings is the setting in which non-redundant and multi-facet papers say a single partition fails to show every property [lit: Niu et al. 2010; Miklautz et al. 2020; Falck et al. 2021]. Not-merging avoids the shared-membership constraint [lit: leidenalg docs "cannot have different communities for different graphs"].
- Open: whether per-source groupings stay non-redundant on correlated sources; the papers measure redundancy through penalties or independence priors, none run on this regime [inferred]. Sibling-aware agreement (R3) is T5's subject, not covered here.

**L (keep the M groupings, refine jointly given the others).**
- Supported: the idea of a clustering conditioned on an existing clustering, via conditional mutual information, exists [lit: Gondek and Hofmann 2006/2007, title and summary only]. Joint non-redundancy across several clusterings from one model also exists [lit: ENRC, MFCVAE, Niu et al.].
- Contradicts or warns: MFCVAE's independence prior is built for independent facets; L's staying-close-to-source anchor is its own addition, with no paper tested in this search [inferred]. Cui et al.'s orthogonalisation is the likely closest iterative variant, not assessed here.
- Open: refinement of existing groupings (not their discovery) with an anchor; to our knowledge, within this search, not found.

**G (multiplex graph, one shared Leiden partition across layers, then layer-specific sub-groupings).**
- Supported: the multiplex machinery exists and is implemented [lit: Mucha et al. 2010; leidenalg multiplex docs]. A shared partition is by construction one partition for all layers [lit: leidenalg]; layer-specific structure needs a different formulation [lit: leidenalg slices; Wilson et al. 2017].
- Warns: shared-membership fusion has the same shape as co-regularised multi-view clustering, where agreement across views is the objective [lit: Kumar et al. 2011 abstract], and dominance by a stronger view is documented for deep joint fusion [lit: BMvC]. Whether the dominant layer wins in a Leiden multiplex objective is not shown in anything I opened; it follows if its per-layer quality gain is larger [inferred].
- Open: the fuse-then-split step (the sub-grouping stage) has only our own hierarchical-refinement result behind it [brief §2]; no paper found.

**E (pool tagged edges, competing property heads, two-tower trunk).**
- Supported: competing heads or masks over a shared embedding that infer the condition per pair exist and give condition-aligned subspaces [lit: SCE-Net, DiscoverNet; CSN with given condition]. Heads defined by dimension assignment also exist for clusterings [lit: ENRC].
- Warns: SCE-Net's and DiscoverNet's alignment with single properties was shown where triplets are drawn per attribute and K matches the attribute count; K without labels is left as future work [lit: SCE-Net, author-acknowledged; DiscoverNet Table 3]. Pooled tagged edges whose properties are correlated and unequally strong have no such evidence [inferred].
- Open: edges tagged by source rather than by a sampled attribute; to our knowledge, within this search, not found.

**Dropped design (consensus).** The literature agrees with the brief's reading: consensus returns one partition and the combination does not record its sources [lit: Strehl and Ghosh 2002 abstract for the objective; the consequence is reader-inferred].

## Answers to the thread deliverable

For each fusion point: evidence that a later split recovers single-property structure, the split mechanism, and failure modes.

| Fusion point | Evidence that splitting recovers single-property structure | Split mechanism in the literature | Failure modes |
|---|---|---|---|
| **Groupings (L; also consensus)** | Conditional clustering given a known clustering exists [Gondek and Hofmann; summary only]. Consensus gives one partition and no split [Strehl and Ghosh]. Our own: affect plus image sources jointly beat either alone (SE, brief §2). No paper tests splitting a fused grouping. | Conditional mutual information with the known clustering (Gondek and Hofmann); orthogonalisation (Cui et al., not assessed). | [inferred] The refinement may re-learn the dominant property; a hyperparameter-free stopping rule is absent. |
| **Graphs (G)** | Multislice modularity (Mucha et al.) and leidenalg multiplex give a shared partition, so the "split" has to come from layer-specific formulations (slices, Wilson et al.) or from sub-partitioning each community. Only our own refinement result (emotion AMI 0.107 against 0.036 for a random split; brief §2) shows a split that kept the property. | Per-layer support for a community's edges (Wilson et al. significance); slice-wise membership (leidenalg slices); layer weights (leidenalg). | Shared membership forces one partition on all layers [lit: leidenalg]; the layer with the strongest structure can steer the partition [inferred; same shape as the dominance reported for deep fusion in BMvC [lit]]; in the percept line a union graph gave emotion 0.124 and genre 0.139, both mixed [brief §2]. |
| **Edges (E)** | SCE-Net: masks aligned with four Zappos attributes (Fig. 4); DiscoverNet: 77.84% coverage-aligned accuracy; MFCVAE: facet probes 94 to 100% against 17 to 73%. All with independent or per-attribute-sampled properties. | Softmax or multinomial competition among K masks (SCE-Net, DiscoverNet); independent facet priors and ladder depth (MFCVAE); dimension assignment (ENRC). | K must be set (SCE-Net, DiscoverNet, MFCVAE: author-acknowledged); hyperparameters cannot be chosen by a training-time metric (MFCVAE §5); correlated properties are not tested (all) [inferred]. Our own DEC variants lowered separation and agreement together (brief §2). |

**Overall.** To our knowledge, within this search, no published method fuses several sources and then splits them into single-property groupings with measured recovery. The published successes are joint separation of facets inside one representation (MFCVAE, ENRC, Niu et al., SCE-Net, DiscoverNet) on independent-by-construction properties. This favours P0 as the control and treats L, G and E as untested adaptations rather than replications.

## Brief citation check

| Brief wording | Result |
|---|---|
| Gondek and Hofmann | Verified. "Non-redundant data clustering", ICDM 2004 (search result) and Knowledge and Information Systems 12, 1-24 (Crossref; issued 2006 online, volume 2007). |
| Cui, Fern and Dy 2007 | Verified: "Non-redundant Multi-view Clustering via Orthogonalization", ICDM 2007, DOI 10.1109/ICDM.2007.94 (S2 record). |
| Niu, Dy and Jordan 2010 | Verified: "Multiple Non-Redundant Spectral Clustering Views", ICML 2010, pp. 831-838. |
| MFCVAE, Falck et al. 2021 | Verified. Full author list: Falck, Zhang, Willetts, Nicholson, Yau, Holmes; NeurIPS 2021; arXiv 2106.05241. |
| Kumar, Rai and Daumé 2011 | Verified: "Co-regularized Multi-view Spectral Clustering", NIPS 2011. |
| Mucha et al. 2010 | Verified: Science 328(5980), 876-878. |
| leidenalg multiplex optimisation | Verified as documentation (gray literature). |
| Conditional Similarity Networks (Veit et al. 2017) | Verified: Veit, Belongie, Karaletsos, CVPR 2017, arXiv 1603.07810. |
| SCE-Net (Tan et al. 2019) | Verified: Tan, Vasileva, Saenko, Plummer, ICCV 2019, arXiv 1908.08589. |
| DiscoverNet (named in the literature review) | Corrected in form: the paper is "Identifying Ambiguous Similarity Conditions via Semantic Matching", Ye, Shi, Zhan, CVPR 2022, arXiv 2204.04053; DiscoverNet is the method name. The review's attribution (Ye, Shi, Zhan, CVPR 2022) is right. |
| Strehl and Ghosh 2002 | Verified: JMLR 3, 583-617. |

No wrong citation found. Caveats: the brief's "Kumar, Rai and Daumé" lists Daumé III as "Daume" on the NIPS page; the Gondek and Hofmann year is 2004 (conference) or 2006/2007 (journal) depending on what is cited.

## Opened but not shortlisted (context only, no claims)
- Dual-disentangled Deep Multiple Clustering (DDMC), Yao and Hu, SDM 2024, arXiv 2402.05310: variational EM learning coarse and fine disentangled representations for multiple clusterings; abstract opened.
- Multi-View Multiple Clustering (MVMC), Yao, Yu, Wang, Domeniconi, Zhang, arXiv 1905.05053: HSIC to reduce redundancy between view-specific matrices plus a shared commonality matrix; venue not confirmed.
- Multi-Modal Proxy Learning Towards Personalized Visual Multiple Clustering (Multi-MaP), Yao, Qian, Hu, CVPR 2024, arXiv 2404.15655: CLIP and GPT-4 text proxies pick which clustering a user wants; relevant to T3 or T6 more than T1.
- Finding multifaceted communities in multiplex networks, Gadár and Abonyi, Scientific Reports 2024: a modularity variant with an inclusion or exclusion choice for how edge overlap between layers is treated (abstract-level; PMC page opened).

## Not verified
- Springer, IEEE and ACM pages for Gondek and Hofmann, Cui et al. and the Niu et al. ICML paper itself: not opened (redirect or 403); the first two are verified through Crossref or S2 metadata only.
- Pages and volume of the Wilson et al. JMLR article; venue of MVMC.
- The SCE-Net Zappos 7.53 against 10.73 numbers: taken from the project literature review, not re-derived.
- The ICDM 2004 pagination (75-82) of Gondek and Hofmann and ICDM 2007 pagination (133-142) of Cui et al.: from search-result text only.
- Any claim about Cui et al.'s mechanism: none was made beyond the title.
