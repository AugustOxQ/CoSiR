# PercepT, its neighbourhood, and where the buddy model sits

> Literature check for the new line (buddy vs PercepT, GeneCIS as the core idea), step 3 of the r10 decide chat,
> 2026-10-17 (Amsterdam). Opus, deep depth: no note on PercepT existed, and the paper's comparator claim rests on it.
> The agent used 29 of 40 searches and 15 of 20 page reads, and read the full PercepT text. The file name uses CoSiR's
> sequence date (2026-12-05). Builds on [2026-10-21](2026-10-21_cvpr_literature_review.md) and
> [2026-11-28](2026-11-28_design_l_novelty.md); the conditional-similarity check beside it is
> [2026-12-04](2026-12-04_conditional_similarity_update.md); our code against the paper is in the
> [code audit](../auto/percept/2026-12-04_percept_code_audit.md).
>
> **Checked by the controller** against the paper text: the no-DEC rows. ArtELingo (Table 1): h without DEC gives
> AMI 0.16, H 0.32, V 0.19; PercepT gives 0.18, 0.30, 0.19. Affection (Table 3): h without DEC gives AMI 0.21 and
> h_concat without DEC 0.25; PercepT gives 0.18. Both hold.

**Summary.** PercepT is a preprint that no paper cites yet (0 citations), and its code is not released. Its own tables
show that the DEC step adds little to AMI against the labels. Its image-only stage is scored only against its own
clusters. The buddy graph is not new as a construction; what may be new is how the graphs serve the aspect task.

## Goal

1. PercepT: task, data, method, metrics, numbers, weaknesses, code, authors.
2. Follow-ups and competing methods.
3. The neighbourhood (2023 to 2026): multimodal topic models, deep clustering on VL embeddings, affect in VL data, aspects "beyond semantics".
4. What a CVPR reviewer would require from an "improvement over PercepT".
5. Prior work on mutual-kNN or cross-modal neighbour graphs with community detection.

## Q1. PercepT (high confidence: from the full text, arXiv v1, 2 June 2026, "8 pages", preprint)

- **Task.** A P-topic is a hard cluster of image–caption pairs in a fused embedding. The task has two parts: (1) discover the clusters; (2) learn per-topic "mapping functions" that predict, from the image alone, which topics apply. This is multi-label: O[i]=1 if any caption of the image is in cluster i. The paper sets four desiderata: distinctiveness, image relevance, coverage, semantic coherence.
- **Fused embedding h.**
  - Inputs: CLIP ViT-L/14 image and text (h_C = mean of the two), plus ModernBERT-base fine-tuned on GoEmotions (h_E).
  - Fusion: each part is scaled by the other's norm, then h = (2h_C′ + h_E)/3. Concatenation did worse.
  - The affect signal therefore comes from a supervised text-emotion classifier. That is not label-free.
- **Stage 1 (this is IDEC; the paper does not cite IDEC).**
  - Autoencoder: MLP 500-500-500-2000, latent d=128, denoising pretraining for 100 epochs.
  - Clustering: DEC KL loss plus the reconstruction loss (λ_R=1), 200 epochs. k0=100 centres, initialised by k-means.
  - Pruning: centres whose norm lies past the largest gap in the sorted norms are dropped (100 → 67 on ArtELingo).
  - DEC plus a reconstruction term is IDEC (Guo et al., IJCAI 2017).
- **Stage 2.** Attention pooling over CLIP patch tokens, then sigmoid outputs, trained with BCE.
  - Inconsistency: §4.4 describes per-topic vectors and heads F_i, but Appendix A.2 is a single linear projection after the pooling.
  - The appendix also calls the clusters "text-derived", although h includes the image.
- **Data.**
  - ArtELingo plus ArtELingo-28: about 1.2M pairs, captions machine-translated to English with GemmaX2-28-9B.
  - Affection: about 500K pairs, no genre labels.
  - The intro's "1.2M art images" is wrong. ArtELingo covers about 80K paintings, and ArtELingo-28 has about 200K annotations at 140 per image. This settles the open conflict in our 2026-12-01 note.
- **Metrics.**
  - Stage 1 internal: SI, CHI, DBI, computed in each method's own space.
  - Stage 1 external: ARI, AMI, H, C, V, FMI, reported as the harmonic mean of the emotion and genre scores.
  - Stage 2: multi-label AUC (the primary metric), F1, P, R, exact-match accuracy, against Stage 1 assignments.
- **Baselines.** BERTopic multimodal (ViT-B/32, MiniLM, UMAP+HDBSCAN, merged to 100 topics), CEMTM, VLM2Vec v1 and v2, k-means on CLIP, k-means on CLIP+Emo.
- **Headline numbers.**
  - ArtELingo Stage 1: SI 0.97 vs 0.37 (BERTopic); AMI 0.18 vs 0.16 (VLM2Vec v1); V 0.19.
  - ArtELingo Stage 2: AUC 0.94 vs 0.77 (BERTopic) and 0.86 (PCA).
  - Affection Stage 1: AMI 0.18.
  - Affection Stage 2: AUC 0.89 vs 0.87 (BERTopic) and 0.88 (PCA); F1 0.28 vs 0.41 (BERTopic).
- **Human evaluation.** PercepT 58.4%, BERTopic 19.5%, ties or neither 22.1%. Raters: 4 CS PhD students, 113 samples. Clusters were matched by keyword Jaccard. No agreement statistic is reported.
- **Limitations the authors state:** image relevance is not measured; translation may smooth cultural signal; dependence on caption quality; only two domains.
- **Weaknesses the authors do not state (the agent's reading, each backed by their own tables):**
  - a. **DEC barely helps the external metrics.**
    - ArtELingo: h without DEC gives AMI 0.16, V 0.19, H 0.32; PercepT gives 0.18, 0.19, 0.30.
    - Affection (Table 3): h without DEC gives AMI 0.21, and h_concat without DEC gives 0.25, both above PercepT's 0.18.
    - The SI of 0.97 is the DEC objective scored in its own latent space, so it is not comparable across methods (Wang and Ye 2024).
  - b. **Stage 2 is circular.** Its labels are Stage 1's clusters, so AUC measures how learnable a method's own clusters are. Methods with different k are not comparable on it.
  - c. **No held-out protocol, seeds or variance** is described for Stage 1.
  - d. **k is not matched.** k0=100 plus a heuristic prune gives 67 topics, against 100 for BERTopic, and AMI depends on k.
  - e. **Label leakage.** The emotion teacher is supervised, and ArtEmis-style captions name the emotion, so "never seen" emotion labels partly leak.
  - f. **No retrieval evaluation.**
  - g. **Affection's reference labels are not named**, yet a harmonic mean of emotion and genre is claimed.
  - h. **Method novelty is thin.** The method is IDEC, the IDEC paper is not cited, and the size of the BERTopic encoder differs.
- **Code.** "Will be made public"; no link on arXiv or in search results. Not released as of 2026-10-17.
- **Group.** Youssef Mohamed (KAUST Vision-CAIR, PhD student of Mohamed Elhoseiny), with Kenneth Church (VecML). Related earlier work: ArtEmis (CVPR 2021), ArtEmis 2.0 (CVPR 2022), ArtELingo (EMNLP 2022), Affection (CVPR 2023), ArtELingo-28 (EMNLP 2024), Affective Visual Dialog (ECCV 2024).
- **Venue guess (the agent's inference):** the "Limitations" section and the 8 pages suggest an ACL ARR submission.

## Q2. Follow-ups (medium-high confidence)

Semantic Scholar shows 0 citations. Two broad searches found no extension and no competing method on the P-topic task.
Adjacent new work, not about P-topics:

- ArtECulture (arXiv 2608.03358, MLLM benchmark of culture-conditioned emotion; snippet only, unverified)
- ProFocus (arXiv 2608.13974; snippet only, unverified)
- AffectiveArt Challenge 2026 (OpenReview; snippet only, unverified)

## Q3. The neighbourhood (medium confidence; 2025 to 2026 coverage is thin where noted)

- **a. Multimodal topic models.**
  - BERTopic-MM; M3L (COLING 2022).
  - Multimodal-ZeroShotTM and Multimodal-Contrast (González-Pizarro and Carenini, LREC-COLING 2024): a large evaluation with new metrics.
  - CEMTM (EMNLP 2025): LVLM embeddings, scored with an LLM rating.
  - DMNTM (Information Processing & Management, 2026; snippet only).
  - All of these score coherence and diversity on text topics. None does image-only prediction of the topics, which is PercepT's stated gap.
- **b. Deep clustering on VL embeddings.**
  - DEC, IDEC; SCAN; TEMI (BMVC 2023): neighbour self-distillation on frozen backbones; SIC (AAAI 2023).
  - TAC (ICML 2024 oral): WordNet-noun text counterpart, cross-modal neighbour distillation. ImageNet-1K NMI/ACC/ARI 79.9/58.2/43.5 against CLIP k-means 72.3/38.9/27.1.
  - CAE (AAAI 2026): training-free, reports +4.2 ACC over the best training-free method.
  - NTK spectral clustering on VL representations (ICLR 2026).
  - Text-guided clustering (arXiv 2402.02996), IC|TC; Multi-MaP and X-Cluster for multiple clustering (in our 2026-11-28 note).
  - Caveat from CLUBench (arXiv 2605.29933, snippet only): CLIP-era clustering benchmarks overlap the pretraining data.
- **c. Affect in VL data.**
  - Datasets: ArtEmis, ArtEmis 2.0, ArtELingo, ArtELingo-28, Affection, EmoSet (all in our notes).
  - EmotionCLIP (CVPR 2023): emotion representation from uncurated data. EEmo-Bench (2025): MLLM evaluation of evoked emotion.
  - No affective image–text retrieval benchmark found. This repeats our 2026-10-21 finding.
- **d. Beyond semantics.**
  - CSD style descriptor (ECCV 2024): WikiArt style retrieval, mAP@k.
  - TPIPS (Wang, …, Efros, Zhang, arXiv 2607.18237, preprint): text-prompted perceptual similarity with human triplets over free-form aspects. The closest concurrent "similarity depends on an aspect" work.
  - VLM probing of aesthetic attributes (ACL Findings 2026): attributes are present but correlated.
  - GeneCIS and its descendants: in our notes.

## Q4. What a reviewer would require (medium confidence; the agent's synthesis from the weaknesses above)

- **Metrics.**
  - Per-label AMI and NMI on a held-out split, not only the harmonic mean, with k matched or AMI shown against k.
  - Hungarian-matched accuracy (ACC) and ARI; seeds and variance.
  - Silhouette only within one space, or dropped.
  - Stage 2 scored against external labels (held-out emotion and genre predicted from image-only topic vectors), not against the method's own clusters.
  - Retrieval R@K (image↔text) plus a conditional benchmark (GeneCIS).
- **Baselines.** A PercepT reimplementation and its no-DEC row; IDEC; BERTopic-MM, CEMTM; k-means on CLIP+Emo; TAC or TEMI; Leiden on a kNN graph; Multi-MaP or X-Cluster; a supervised ceiling.
- **Datasets.** ArtELingo plus Affection, and one more domain, which PercepT itself names as a gap.
- **Most credible targets:** weakness a (DEC adds little), b (circular Stage 2), d and e (fixed k, a supervised affect teacher), f (no retrieval).

## Q5. Where buddy sits (medium confidence)

"Mutual neighbours in image space and in text space" is a known construction in at least four forms:

- **UMAP model intersection** (umap-learn docs): keeps the edges both views support, and requires one-to-one samples.
- **Seurat WNN** (Hao et al., Cell 2021): two-modality neighbour graphs fused into a shared-neighbour graph, then community clustering. Whether it uses Leiden or Louvain is not checked.
- **DJSRH** (ICCV 2019): a joint image–text neighbourhood affinity for unsupervised cross-modal hashing.
- **The mutual-kNN alignment metric** (Platonic Representation Hypothesis, ICML 2024). It measures image–text neighbour overlap and is not used for clustering.

Related:

- Koepke et al. (arXiv 2604.18572, preprint) show that image–text mutual-kNN overlap drops at scale and with many-to-many data. This fits buddy's use of separate per-modality graphs.
- Leiden on kNN graphs is the scanpy standard (Traag 2019). Community detection as pseudo-labels for deep clustering: DCvCD (arXiv 2501.02036, preprint).
- Cross-modal neighbour distillation: TAC (closest in vision). Student from a teacher graph: Parametric UMAP (Sainburg et al. 2021).
- No work found uses such graphs to discover aspects and evaluates against emotion and genre. That is "not found", not "proven new".

## Sources

| Title, authors, year | Venue | URL | Shows, for |
|---|---|---|---|
| Beyond Semantics (PercepT), Mohamed, Church, Elhoseiny 2026 | preprint | https://arxiv.org/abs/2606.03345 | everything in Q1 |
| Semantic Scholar record of PercepT | database | https://api.semanticscholar.org/graph/v1/paper/arXiv:2606.03345 | Q2: 0 citations |
| IDEC, Guo et al. 2017 | IJCAI | https://ijcai.org/proceedings/2017/243 | Q1: DEC+reconstruction is prior |
| Deep Clustering Evaluation, Wang and Ye 2024 | preprint | https://arxiv.org/abs/2403.14830 | Q1, Q4: internal indices are not comparable across spaces |
| CEMTM, Abaskohi et al. 2025 | EMNLP | https://aclanthology.org/2025.emnlp-main.590 | Q3a |
| Neural Multimodal Topic Modeling, González-Pizarro, Carenini 2024 | LREC-COLING | https://aclanthology.org/2024.lrec-main.1064 | Q3a |
| Image Clustering with External Guidance (TAC), Li et al. 2024 | ICML | https://arxiv.org/html/2310.11989v3 | Q3b, Q5 |
| TEMI, Adaloglou et al. 2023 | BMVC | https://arxiv.org/abs/2303.17896 | Q3b |
| Hierarchical Semantic Alignment (CAE), Zhu et al. 2025 | AAAI 2026 (per arXiv comment) | https://arxiv.org/abs/2512.00904 | Q3b |
| Spectral Clustering with VL Representations, Peng et al. 2026 | ICLR 2026 (per arXiv comment) | https://arxiv.org/abs/2602.09586 | Q3b, Q5 |
| Text-Guided Image Clustering, 2024 | preprint (search snippet) | https://arxiv.org/abs/2402.02996 | Q3b |
| EmotionCLIP, Zhang et al. 2023 | CVPR | https://arxiv.org/abs/2305.13500 | Q3c |
| ArtELingo, Mohamed et al. 2022 | EMNLP | https://aclweb.org/anthology/2022.emnlp-main.600.pdf | Q1 group, data size |
| Affective Visual Dialog, Haydarov et al. 2024 | ECCV | https://arxiv.org/abs/2308.16349 | Q1 group |
| CSD style similarity, Somepalli et al. 2024 | ECCV | https://arxiv.org/abs/2404.01292 | Q3d |
| TPIPS, Wang et al. 2026 | preprint | https://arxiv.org/abs/2607.18237 | Q3d |
| VLMs for personalised aesthetics, 2026 | ACL Findings | https://aclanthology.org/2026.findings-acl.1706/ | Q3d (snippet) |
| Back into Plato's Cave, Koepke et al. 2026 | preprint | https://arxiv.org/abs/2604.18572 | Q5 |
| Platonic Representation Hypothesis, Huh et al. 2024 | ICML | https://arxiv.org/abs/2405.07987 | Q5 (snippet) |
| UMAP composing models | documentation | https://umap-learn.readthedocs.io/en/latest/composing_models.html | Q5 |
| WNN, Hao et al. 2021 | Cell | https://www.satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis | Q5 |
| DJSRH, Su et al. 2019 | ICCV | https://openaccess.thecvf.com/content_ICCV_2019/papers/Su_Deep_Joint-Semantics_Reconstructing_Hashing_for_Large-Scale_Unsupervised_Cross-Modal_Retrieval_ICCV_2019_paper.pdf | Q5 |
| DCvCD, Cheng, Chen 2025 | preprint | https://arxiv.org/abs/2501.02036 | Q5 |
| Parametric UMAP, Sainburg et al. 2021 | Neural Computation (venue not checked) | https://arxiv.org/abs/2009.12981 | Q5 |

## Open

- Whether PercepT is under review, and where. A search of OpenReview and ARR listings would settle it; OpenReview blocked the fetch.
- Whether WNN clusters with Leiden or Louvain: read the Seurat FindClusters defaults.
- 2026 multimodal topic models beyond CEMTM: search ACL Anthology 2026 for "multimodal topic".
- The AffectiveArt Challenge 2026 details (organisers, data): blocked by OpenReview.
- PhenoGraph (kNN + Jaccard + Louvain) is from memory and was not searched.

## What this means for the new line

1. PercepT has no code and no citations yet, so we compare against our own replication. We state that it is a reimplementation.
2. The strongest attack uses their own Table 1 and Table 3 rows: DEC adds little to AMI, and Stage 2 is scored against its own clusters. We should beat the no-DEC row, not only PercepT.
3. Buddy's graph is a known construction (UMAP intersection, WNN, TAC). Claim the use, not the graph: aspect discovery, label-free, with all topics used.
4. Add what PercepT lacks: held-out AMI per label at matched k, external-label Stage 2, retrieval and GeneCIS-style conditional retrieval.
5. A second domain is needed. Affection has no genre labels, so a second aspect must come from elsewhere.

## Works a paper in this line must cite or compare against

1. **PercepT (preprint):** the task and the main comparator.
2. **IDEC / DEC:** PercepT's real method; the obvious clustering baseline.
3. **BERTopic-MM and CEMTM (EMNLP 2025):** the multimodal topic baselines PercepT used.
4. **TAC (ICML 2024):** the strongest VL clustering method with cross-modal neighbours, the nearest relative of buddy.
5. **TEMI (BMVC 2023) or SCAN:** neighbour-based clustering on frozen features.
6. **GeneCIS (CVPR 2023):** conditional similarity, the core idea and a benchmark.
7. **Multi-MaP (CVPR 2024) or X-Cluster:** discovery of several clusterings, one per aspect.
8. **TPIPS (preprint, 2026):** concurrent work on aspect-dependent similarity; cite it as concurrent.

Unverified (search snippet only): DMNTM, CLUBench, ArtECulture, ProFocus, the AffectiveArt Challenge, the ACL Findings
aesthetics paper, the PRH metric details, Text-Guided Image Clustering.
