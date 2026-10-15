# Three-way scan for N6 (cross-modal pseudo-partition heads)

Scan date 2026-10-04. Mode: ARS deep-research three-way-scan (bibliography + source verification). Candidate text: `candidates_draft.md` §2 N6.
Read scope for every paper below: **abstract_only** (arXiv or proceedings abstract page opened with WebFetch; no full text). Per the method-weakness rule 5, method-level weaknesses are therefore "not assessed". Retrieved text was treated as data.

Queries run (WebSearch, standard mode): "Self-Supervised Learning by Cross-Modal Audio-Video Clustering XDC"; "Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos"; "Conditional Similarity Networks Veit Belongie"; "Deep clustering multiple clusterings multi-facet disentangled representation"; "Self-labelling via simultaneous clustering and representation learning Asano"; "few-shot attribute-conditioned image retrieval infer which attribute from example pairs". Sources: arXiv abs pages, CVF and NeurIPS pages. Not searched: Google Scholar, Semantic Scholar, ACL Anthology, OpenReview, mixture-of-experts gating literature (no query was run on it), the three project literature reviews.

## Self-Supervised Learning by Cross-Modal Audio-Video Clustering (XDC)
Source: arXiv / NeurIPS 2020 | Year: 2019 (v1), 2020 | Link: https://arxiv.org/abs/1911.12667
- WHY: Single-modality pretext tasks miss the correlation between audio and video; cross-modal prediction is argued to be a richer signal.
- HOW: Unsupervised clustering in one modality gives pseudo-labels that supervise the other modality's network (abstract, verified).
- WHAT: Beats single-modality clustering and other multimodal variants; video model pretrained this way outperforms supervised ImageNet and Kinetics pretraining on HMDB51 and UCF101 (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos (MCN)
Source: ICCV 2021 / arXiv 2104.12671 | Year: 2021 | Link: https://arxiv.org/abs/2104.12671
- WHY: Instance-level contrast ignores semantic similarity between different instances across modalities.
- HOW: Adds a multimodal clustering step to contrastive training, giving a joint embedding space in which semantically similar instances (video, audio, text) are grouped.
- WHAT: State-of-the-art zero-shot text-to-video retrieval and action localization on four datasets after HowTo100M training (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)
Source: NeurIPS 2020 / arXiv 2006.09882 | Year: 2020 | Link: https://arxiv.org/abs/2006.09882
- WHY: Pairwise feature comparison is memory heavy; cluster assignments can be the contrast target.
- HOW: Online clustering with swapped prediction: the assignment of one view is predicted from another view; multi-crop augmentation.
- WHAT: 75.3% ImageNet top-1 with ResNet-50 and better transfer than supervised pretraining (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Deep Clustering for Unsupervised Learning of Visual Features (DeepCluster)
Source: ECCV 2018 / arXiv 1807.05520 | Year: 2018 | Link: https://arxiv.org/abs/1807.05520
- WHY: Unsupervised features without hand-designed pretext tasks.
- HOW: Alternates k-means on features with training the network to predict the cluster assignments (abstract). SeLa (Asano et al., arXiv 1911.05371, ICLR 2020) replaces k-means by an optimal-transport self-labelling step; seen only via search-result summary, not opened, so cited here as context only.
- WHAT: State of the art on unsupervised visual feature benchmarks at the time (ImageNet, YFCC100M).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Multi-Facet Clustering Variational Autoencoders (MFCVAE)
Source: NeurIPS 2021 | Year: 2021 | Link: https://neurips.cc/virtual/2021/poster/26795
- WHY: High-dimensional data contain several characteristics one could cluster over; single-partition deep clustering picks one.
- HOW: VAE with a hierarchy of latents, each with a mixture-of-Gaussians prior, trained end to end to learn several clusterings at once.
- WHAT: Different facets are clustered separately (shape versus background colour) on image benchmarks (abstract). Dual-disentangled Deep Multiple Clustering (arXiv 2402.05310, seen in search results only) is a later variant.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Conditional Similarity Networks (CSN)
Source: CVPR 2017 / arXiv 1603.07810 | Year: 2017 | Link: https://arxiv.org/abs/1603.07810
- WHY: Images have several, sometimes contradictory, similarity notions; a single embedding cannot serve all.
- HOW: One embedding with disentangled subspaces plus learned masks that select the dimensions for a given similarity notion.
- WHAT: Outperforms training a separate network per notion on triplet questions (abstract). The abstract does not say how the condition is supplied at test time; the usual reading is that it is given, but I did not verify that (reader-inferred).
  - Method weaknesses: not assessed (read scope: abstract_only).

## Cross-paper synthesis
- Shared WHY: cluster assignments are a free supervisory signal for semantic structure (DeepCluster, SwAV, XDC, MCN), and one data set carries several partitions (MFCVAE, CSN).
- Divergent HOW: XDC is the closest in mechanism, since a classifier on modality B is trained on modality A's k-means clusters. Those works use the result as pretraining for a representation; none of the abstracts reads posteriors over several cluster heads as a retrieval representation. MFCVAE and CSN supply the multi-partition and condition-selection ideas, but CSN's condition is supplied by design, and MFCVAE is unimodal generative.
- Gaps visible from the abstracts: nothing found about picking the relevant partition from few example pairs, nor about whether cluster-posterior agreement tracks human semantic attributes (affect, style) in a value-disjoint setting.

## Verdict for N6
- **Partially exists.** Each ingredient is published: cross-modal cluster-prediction heads (XDC), cluster assignments as the shared code (MCN, SwAV, DeepCluster), several clusterings per dataset (MFCVAE), and condition-to-subspace selection (CSN). I did not find, in the six queries above on arXiv, CVF, NeurIPS and search-engine results, a method that combines per-pseudo-partition cross-modal heads with example-based head selection by within-pair agreement. This is a bounded absence claim: no full-text search, no ACL or OpenReview search, no mixture-of-experts gating query.
- **Strongest citation a reviewer would raise:** XDC (Alwassel et al., NeurIPS 2020), as "cross-modal pseudo-label heads", then CSN for "condition-selected subspaces" and MFCVAE for "multiple clusterings".
- **Evidence relevant to our regime:** the abstracts give evidence that cluster-derived supervision transfers to action recognition and zero-shot video retrieval (XDC, MCN), a different regime from affect or style aspects with disjoint values. I found no published evidence that cluster-posterior agreement transfers to semantic attributes, or supports few-shot choice of the clustering. Our own numbers (AMI 0.20 to 0.40 between partitions and aspects, K8 failing for a held-out aspect) remain the only evidence, and they are against transfer.
- **Stronger published variant to adopt:** none identified. XDC's cross-modal cluster supervision is the design to cite and replicate (cluster one modality, train the other modality's head). MFCVAE-style jointly learned multiple clusterings could replace separate k-means partitions but is generative and unimodal.
- Novelty therefore rests on the selection step and the value-disjoint cross-modal evaluation, not on the heads.

AI-assistance note: this scan was produced by a Claude agent using WebSearch and WebFetch; every listed reference was opened at its arXiv, CVF or NeurIPS page, but only abstracts were read. Papers seen only in search results are marked and carry no claims beyond their titles. A human should read XDC and CSN in full before citing them.
