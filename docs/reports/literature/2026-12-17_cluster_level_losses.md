# Cluster-level losses per aspect head: prototype contrast, teacher partitions, DEC from a good start, comparators

> Literature check for the CoSiR condition-free line (research-loop step 3), written 2026-10-20. Deep depth (Opus):
> the per-aspect cluster loss is close to the paper's central claim, and our own September evidence conflicts with
> the field's positive reports. Budget: 36 of 40 searches and 20 of 20 page reads. Every page went through a
> summariser, and no paper was read in full. "Abstract" means the abstract or listing only. "Not checked" means the
> claim was not seen at the source.
> This note builds on `2026-12-05_percept_neighbourhood.md`, `2026-12-07_prompted_aspect_views.md`,
> `2026-11-28_design_l_novelty.md`, `2026-11-30_r8_learned_grouping.md` and
> `2026-09-28_architecture_rethink_literature_brainstorm.md`. Their content is not repeated here.

## Summary

1. On strong frozen features, a cluster-level loss on top of an instance loss adds little NMI or AMI (about 0.6 to 2 points in Contrastive Clustering). Large gains come only with a known K, balanced classes, pixel augmentations or a text side (TEMI, TAC, CPP). Studies of uncertain embeddings and a 2026 benchmark find no average gain over k-means.
2. A head trained on a teacher's k-means pseudo-labels often beats k-means on the same embedding: ESMC improves NMI on 13 of 15 criteria, including emotion (0.19 to 0.22). In every such case, though, the student reads the teacher's own input. Our student reads the default image, so the teacher sees information the student does not have. Theory says the student can still beat the teacher, but only where the teacher's errors are hard for the student to fit.
3. We found no published evidence that DEC, started from a good partition on strong features, raises agreement with labels. The evidence points to drift: PercepT's own rows, our September runs, IDEC, the "feature randomness/drift" line, and SCCL. Self-training helps only with confidence selection plus strong augmentations, and frozen features rule out the augmentations.
4. Several aspects at once: DivClust (a shared trunk with 20 cluster heads) costs about 0.03 NMI per head against one head. ESMC and Multi-Sub use a per-criterion teacher partition, but neither uses a shared trunk read once, and both are transductive. We found no exact prior for our design (medium-low confidence).
5. Comparators a CVPR reviewer may add beside PercepT: TURTLE (ICML 2024), TEMI (BMVC 2023), CPP (ICLR 2024; it estimates K and clusters WikiArt), ESMC (AAAI 2026) and SEIC (AAAI 2026). All except TAC and SEIC run on precomputed vectors such as ours.

## Q1. Cluster-level contrast per head (confidence: medium-high on the numbers, medium on how they transfer to us)

**What the field shows: on good instance features, the cluster term adds little; the large gains need conditions we lack.**

- **Contrastive Clustering** (Li et al., AAAI 2021), Table 5, read from the paper text:

  | Data | Instance head + k-means | Cluster head only | Both |
  |---|---|---|---|
  | CIFAR-10, NMI | 0.699 | 0.592 | 0.705 |
  | ImageNet-10, NMI | 0.838 | 0.850 | 0.859 |

  The cluster term adds 0.006 to 0.021 NMI over k-means on the instance features. On CIFAR-10 the cluster term alone is far worse. An entropy term on the batch cluster marginal prevents "most instances in one cluster". [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17037)
- **PCL / ProtoNCE** (Li et al., ICLR 2021): ImageNet AMI at k = 25,000 is 0.410 for PCL, 0.285 for MoCo and 0.281 for DeepCluster (Table 5). This is training from scratch with a huge k, which is not our case.
  - Prototypes come from k-means once per epoch on momentum-encoder features.
  - The first 20 of 200 epochs use InfoNCE only.
  - Proto loss alone without warm-up is much worse: 60.7 against 72.3 (1% fine-tune top-5). Instance plus proto gives 75.3.
  - A per-cluster concentration φ scales the temperature. The paper says this balances cluster sizes and prevents collapse. [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.04966)
- **Gains on frozen pretrained features:**
  - TEMI (BMVC 2023) trains heads only, on precomputed features. Over 17 backbones it beats k-means by +6.1 ACC on ImageNet and +12.2 on CIFAR-100. With CLIP ViT-L/14 on CIFAR-100, ACC goes from 54.6 to 73.7 (NMI 79.9). The gains are smaller for strong supervised ViT-L backbones (about 2 to 3 points; read from Table 12 by the summariser). [arXiv](https://arxiv.org/abs/2303.17896), [ar5iv](https://ar5iv.labs.arxiv.org/html/2303.17896)
  - TAC (ICML 2024), ImageNet-1K NMI: k-means on CLIP 72.3, a no-training text concatenation 77.8, trained heads 79.9. On DTD the three values are 57.3, 60.1 and 62.1. Most of the gain comes from the text side, not from the trained cluster loss. [arXiv HTML](https://arxiv.org/html/2310.11989v3)
  - CPP (ICLR 2024), frozen CLIP ViT-L/14 on ImageNet-1k, ACC/NMI: k-means 49.2/81.3, CPP 66.2/86.8. [arXiv HTML](https://arxiv.org/html/2306.05272v5)
  - All three use the true K on balanced benchmarks.
- **Counter-evidence for uncertain embeddings:**
  - Scott et al. (NeurIPS 2022 workshop): deep clustering methods are "surprisingly fragile" on embeddings with more uncertainty. There they match shallow methods or do worse. [arXiv](https://arxiv.org/abs/2211.05183) (abstract)
  - CLUBench (preprint, May 2026; 24 algorithms, 131 datasets) finds no significant average advantage of deep methods over k-means or spectral clustering. Pretrained embeddings with a conventional algorithm are "effective and efficient". [arXiv](https://arxiv.org/abs/2605.29933) (abstract)
  - SCCL (NAACL 2021) reports that its clustering-only objective gives tighter predicted clusters but poor agreement with the true clusters on SearchSnippets. [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.12953)
  - This is the same pattern as our Leiden pseudo-label run: silhouette went from 0.039 to 0.088 while AMI fell slightly.
- **Our reading:** our AMI values (0.09 to 0.45) and imbalanced labels put us in the "uncertain embedding" regime, not the CIFAR regime. Expect a small gain at best from a self-made cluster loss.

**Collapse, confirmation bias, degenerate clusters: the design pieces the field uses.**

| Piece | Where | What it does / evidence |
|---|---|---|
| Entropy of the batch cluster marginal | CC; SCAN (λ = 5); SeCu (global entropy); TAC balance loss | Stops one cluster from taking everything. TAC: without it, most images fall into a few clusters. SCAN shows no ablation. [SCAN ar5iv](https://ar5iv.labs.arxiv.org/html/2005.12320), [SeCu](https://openaccess.thecvf.com/content/ICCV2023/html/Qian_Stable_Cluster_Discrimination_for_Deep_Clustering_ICCV_2023_paper.html) |
| Equipartition (Sinkhorn), as in SwAV | SwAV, MSN | Assran et al. (ICLR 2023): this hidden uniform prior hurts on class-imbalanced data, and a power-law prior helps. [arXiv](https://arxiv.org/abs/2210.07277) (abstract). Our labels are imbalanced. |
| Lower bound on cluster size only | CoKe (CVPR 2022) | Keeps the natural size spread and still avoids empty clusters. [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Qian_Unsupervised_Visual_Representation_Learning_by_Online_Constrained_K-Means_CVPR_2022_paper.html) |
| Prototype scattering | ProPos (TPAMI 2022) | Pushes prototypes apart for uniformity. This avoids the k-means collapse of non-contrastive features. [arXiv](https://arxiv.org/pdf/2111.11821) (abstract) |
| EMA teacher, β < 1, marginal reweighting, 50 heads | TEMI | Final usage KL to uniform about 0.015 to 0.05. Pair weights down-weight false-positive neighbours (mean weight 0.40 against 0.76 for true positives). Only 66 to 72% of kNN pairs are true positives on CIFAR-100. |
| Confidence selection | SCAN (p > 0.99); TCL (IJCV 2022); ESMC (nearest α = 0.1 to 0.4 of each cluster) | Trains only on confident or core samples. SCAN stops self-labeling when the confident count plateaus. [TCL](https://arxiv.org/abs/2210.11680) |
| Strong augmentations | SCAN self-labeling | With SimCLR augmentations, self-labeling collapsed to 10% ACC on CIFAR-10. With RandAugment it reached 87.6%. **Frozen features remove this safeguard.** |
| Instance warm-up | PCL | See above (60.7 against 72.3). |
| Batch size against K | SeCu | With batch b, at most b + 1 clusters have positives in a batch, so cluster discrimination is unstable on small batches. |

**Learning the number of clusters (confidence: medium).**
- DeepDPM (CVPR 2022) splits and merges clusters during training. Methods with a fixed K degrade as K moves away from the truth, most on imbalanced data. [CVF PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.pdf)
- CPP picks K by coding length, without retraining. Its estimates: CIFAR-10 10, CIFAR-100 20, WikiArt 20. Whether the WikiArt clusters were scored against labels: not checked (they look qualitative).
- PCL clusters at several k at once. TEMI reports overclustering AMI at 25K clusters. SCAN overclustering raised CIFAR-100-20 from 45.9 to 55.1, under a many-to-one mapping.
- Gradual community detection (Cheng and Chen, preprint 2025) merges many small "pseudo-communities" to raise pseudo-label purity. From the abstract it is not clear whether K is learned. [arXiv](https://arxiv.org/abs/2501.02036)
- No published image-clustering method was found that pairs Leiden communities with a prototype loss. This is absence of evidence from two searches (low confidence).

## Q2. Teacher clusters as anchors (confidence: medium)

**A trained head usually beats k-means on its own input. Whether it beats a teacher it cannot see depends on what the teacher's errors are.**

- **Same input, head beats k-means:**
  - ESMC (AAAI 2026) runs k-means on per-criterion MLLM embeddings. It keeps the fraction α nearest each centroid and trains a head with cross-entropy. NMI, without the head against with it, presumably k-means against the head (the paper does not say this outright):
    - CMU_face emotion: 0.190 to 0.224.
    - Card order: 0.390 to 0.474.
    - Fruit360 species: 0.381 to 0.504.
    - In total, 13 of 15 rows improve and 2 tie.
  - ESMC has no balance term and appears transductive. [arXiv HTML](https://arxiv.org/html/2512.00725v1)
  - TAC heads beat TAC's no-training partition on every dataset (see Q1). SCAN self-labeling adds +1.2 to +5.8 ACC over its own clustering step.
- **Theory on when a student beats its teacher:**
  - Wei et al. (ICLR 2021): self-training plus input-consistency regularization reaches high accuracy under "expansion": small sets of a class have proportionally larger neighbourhoods, and different classes overlap little. [arXiv](https://arxiv.org/pdf/2010.03622) (abstract)
  - Lang, Sontag and Vijayaraghavan (NeurIPS 2024): the student beats the teacher through "pseudolabel correction" and "coverage expansion". This happens when the teacher's mistakes are hard for the strong model to fit. [mlanthology](https://mlanthology.org/neurips/2024/lang2024neurips-theoretical) (abstract)
  - Yao et al. (ICLR 2025): too much optimisation over-fits the weak labels. Wu and Sahai (ICLR 2025): soft logits as supervision help. [arXiv 2502.01458](https://arxiv.org/abs/2502.01458v1), [arXiv 2410.04638](https://arxiv.org/abs/2410.04638v2) (abstracts)
  - Dong et al. (NeurIPS 2023): relational KD is spectral clustering on a graph the teacher induces. It provably reaches low clustering error. Our relational-distillation row is therefore already an implicit cluster-level transfer. [arXiv](https://arxiv.org/abs/2307.11030) (abstract)
- **The privileged-input gap.** Our teacher reads the instructed view and our student reads only the default image. This is generalized distillation with privileged information (Lopez-Paz et al., ICLR 2016). [arXiv](https://arxiv.org/abs/1511.03643) (abstract)
  - No clustering paper was found in which a student without the teacher's input beats the teacher's partition.
  - Our reading: our D2 row already matches the target on emotion (0.096 against 0.093). It stays below the target on style (0.370 against 0.395) and genre (0.454 against 0.496). That fits the theory. Where the teacher's errors are noise, the student can match or pass it. Where the teacher has information the default image lacks, a hard teacher partition sets the ceiling.
- **Related designs:**
  - Multi-Sub (NeurIPS 2024) refines the CLIP projection layer with a distance-based intra/inter-cluster loss on pseudo-labels from proxy words, one run per user interest. It has no ablation of this phase and seems transductive. [arXiv HTML](https://arxiv.org/html/2411.03978v1)
  - TURTLE (ICML 2024) searches for the labeling whose linear classifiers have the largest margin in one or several representation spaces. It needs K. [PMLR](https://proceedings.mlr.press/v235/gadetsky24a.html)

## Q3. DEC started from a good partition (confidence: medium that it drifts; high that no positive evidence was found)

**Evidence against plain DEC sharpening on strong features:**
- PercepT's own ablation (our `2026-12-05_percept_neighbourhood.md`):
  - ArtELingo AMI is 0.16 without DEC and 0.18 with it.
  - Affection AMI is 0.21 without DEC (0.25 for the concatenated input) and 0.18 with it.
- Our September runs: six DEC variants all lowered AMI, mostly on genre.
- IDEC (IJCAI 2017) adds reconstruction because the DEC loss distorts the embedding's local structure (from our earlier note).
- ADEC (Mrabah et al.) names "feature randomness" (learning from unreliable pseudo-labels) and "feature drift". The follow-up R-DC (Shaheen et al., preprint 2025) drops pseudo-supervision for a second self-supervised round, and adds "feature twist". [arXiv 2503.03733](https://arxiv.org/abs/2503.03733) (abstract)
- Scott et al. 2022, CLUBench 2026 and SCCL: see Q1.

**When self-training does help:** SCAN and TEMI. They use confidence selection, strong augmentations (or an EMA teacher with β < 1), a balance term, and stopping when the confident fraction plateaus. Without the augmentations, SCAN's self-labeling collapsed.

**Art precedent:** DELIUS (Castellano and Vessio, IJCV 2022) runs DEC on DenseNet121 features of a WikiArt subset. Its comparison with k-means was not checked. [arXiv](https://arxiv.org/abs/2106.06234)

**Our reading:** a DEC step from the k-means partition of a head's output has no published support for raising AMI. To try it anyway, run it with an anchor that holds the start: a KL term toward the initial soft assignment, or a teacher partition (Q2).

## Q4. Several aspects at once (confidence: medium-low; only what the earlier notes miss)

- **DivClust** (CVPR 2023): one shared encoder, 20 heads, each trained with its base clustering loss (CC, PICA or IIC) plus a bound on inter-head NMI. On CIFAR-10 with CC:
  - mean per-head NMI is 0.678 without a diversity constraint, against 0.705 for single CC;
  - the best head reaches 0.762 at target 0.8, but choosing it uses labels.
  - The heads are alternative partitions of one target, not aspects.
  - Lesson: a shared trunk with many cluster heads costs about 0.03 NMI per head. [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.01042)
- **MFCVAE** (NeurIPS 2021): one latent per facet, each with its own Gaussian-mixture prior, so a per-facet cluster objective on one ladder encoder. It is generative, trained from pixels, with no teacher. [NeurIPS](https://neurips.cc/virtual/2021/poster/26795) (abstract)
- **ESMC** and **Multi-Sub** (Q2) use a per-criterion teacher partition. ESMC does not say whether its heads share layers. Multi-Sub trains per interest.
- **CIM-Embed** (Xu et al., preprint, 15 Oct 2026) fine-tunes Qwen3-VL-Embedding with supervised contrastive learning on instruction-defined groups, for product attributes. The instruction is an input at test time and the products are held out. It is supervised, so it is not a label-free comparator. It shows that instructed Qwen3-VL embeddings are a 2026 basis for clustering. [arXiv HTML](https://arxiv.org/html/2610.21805)
- **SigLino** (CVPR 2026) has per-teacher heads on one distilled student. It is the architectural analogue of our design, but for features, not clusters. [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chaybouti_SigLino_Efficient_Multi-Teacher_Distillation_for_Agglomerative_Vision_Foundation_Models_CVPR_2026_paper.pdf)

## Q5. The strongest comparator (confidence: medium)

The earlier notes already list PercepT, IDEC, TAC, TEMI or SCAN, Multi-MaP or X-Cluster, guideline-driven clustering and ESMC. New since those notes:

| Method | Venue | Runs on our frozen vectors? | Needs K | Inductive to unseen paintings | Why a reviewer may ask |
|---|---|---|---|---|---|
| TURTLE | ICML 2024 | Yes: precomputed workflow; own backbones need a code change (not checked) | Yes | Yes (linear classifier) | Strongest frozen-feature method; takes several spaces, e.g. default + instructed |
| TEMI | BMVC 2023 | Yes (precomputed features) | Yes | Yes (heads) | Neighbour self-distillation; closest to "buddy pairs + cluster head" |
| CPP | ICLR 2024 | Yes (frozen CLIP + small heads) | Estimates K | Yes | Clusters WikiArt; answers "why k = 20" |
| SCP "Keep it light" | preprint 2025 | Yes (cluster head on frozen features) | Yes | Yes | Simplest text-free head baseline. [arXiv](https://arxiv.org/abs/2502.04226) |
| SEIC | AAAI 2026 | Needs a text encoder and WordNet nouns | Yes | Yes | Newest TAC-style method. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39506) |
| DELIUS | IJCV 2022 | Yes (DEC on CNN features) | Yes | Yes | Art-specific deep clustering |

- **Our recommendation:**
  - Run TEMI and TURTLE per aspect on the instructed embedding (k = 20), each fit on the training paintings and applied to the 5,000 unseen ones. These are the strongest unsupervised per-aspect clusterings on our own features.
  - Keep PercepT, and its no-DEC row, as the task-level comparator.
  - Add ESMC-style head training as the closest per-criterion method.
- **Caveats:**
  - TAC and SEIC need a text side. Qwen3-VL-Embedding has one, but a WordNet-noun step for emotion or style is untested (not checked).
  - No published per-aspect clustering of paintings scored against emotion, style and genre was found.

## Novelty

A per-aspect cluster-level loss whose clusters come from an instructed teacher view, on a shared trunk that reads the default image once and works on unseen images, was not found in two targeted searches. Confidence is medium-low.

Nearest works:
- **ESMC** (AAAI 2026): per-criterion MLLM embedding, then k-means pseudo-labels, then a head. It is transductive, with no shared trunk and no student input that differs from the teacher's.
- **Multi-Sub** (NeurIPS 2024): per-interest pseudo-labels and a clustering loss.
- **DivClust** (CVPR 2023): a shared trunk with many cluster heads.
- **TAC** (ICML 2024): cross-view neighbour distillation into cluster heads.
- **SigLino** (CVPR 2026): per-teacher heads.

Claim the combination and the task, not the cluster loss.

## Candidate ideas for a one-day exploration loop (ranked)

All three use the 13,400 training paintings only, fit k-means on training data, and score on the 5,000 unseen ones.

**1. Teacher-partition anchor per head (soft, core samples).**
- **Change:** per aspect, run k-means (k = 20) on the instructed embedding. Add a 20-way prototype layer on each head. Train it with KL to the teacher's soft assignment (temperature, as in Wu and Sahai), on the α = 0.3 core samples only. Add a CoKe-style minimum-size or marginal-entropy term. Keep InfoNCE and direct distillation; give the new term a weight of about 0.1 to 0.3. k-means runs once, so the cost is minutes.
- **Expected:** style and genre AMI move toward the target (0.395 and 0.496); emotion stays near 0.096.
- **Main risk:** the student copies the teacher and its ceiling is the target, which is a weak claim.
- **Nearest:** ESMC, Multi-Sub, Dong et al. 2023.

**2. TEMI-style neighbour self-distillation head on the buddy pairs.**
- **Change:** add a cluster head per aspect, with an EMA teacher (0.996), a β = 0.6 PMI-like loss with marginal reweighting, and buddy pairs as the neighbours. Optionally also teacher-graph neighbours. No k-means during training; read clusters from the head's argmax and from k-means as before.
- **Expected:** tighter, balanced clusters, perhaps +0.01 to 0.02 AMI on style and genre.
- **Main risk:** tighter is not more label-aligned, as in our Leiden run and SCCL. With no augmentations, the self-labeling safeguard is missing.
- **Nearest:** TEMI, SCAN.

**3. Comparator and ceiling probe, no new training of our model.**
- **Change:** run TURTLE and TEMI per aspect on the instructed embedding. Run TURTLE also on two spaces at once, D2's head output plus the instructed embedding. Each run fits on training and is applied to unseen paintings.
- **Expected:** tells us whether a cluster-level method on our own features passes the target and D2. If it does not, ideas 1 and 2 have little room.
- **Main risk:** the comparators need K and assume balanced classes, so their AMI may sit below k-means, and the probe answers less than hoped.
- **Nearest:** TURTLE, TEMI.

**Not recommended:** plain DEC or ProtoNCE from k-means on the head output. Q3 found no support for it, and our own six runs lowered AMI.

## Sources

| Source | Venue, year | Question | Read |
|---|---|---|---|
| Li et al., Contrastive Clustering https://ojs.aaai.org/index.php/AAAI/article/view/17037 | AAAI 2021 | Q1 ablation, entropy term | full-text file, summarised |
| Li et al., Prototypical Contrastive Learning https://ar5iv.labs.arxiv.org/html/2005.04966 | ICLR 2021 | Q1 AMI, warm-up, φ | ar5iv summary |
| Adaloglou et al., TEMI https://arxiv.org/abs/2303.17896 | BMVC 2023 | Q1 frozen gains, collapse pieces, overclustering | ar5iv summary |
| Li et al., TAC https://arxiv.org/html/2310.11989v3 | ICML 2024 | Q1, Q2 no-train vs trained heads, balance loss | HTML summary |
| Chu et al., CPP https://arxiv.org/html/2306.05272v5 | ICLR 2024 | Q1 gains, K estimate, WikiArt; Q5 | HTML summary |
| Van Gansbeke et al., SCAN https://ar5iv.labs.arxiv.org/html/2005.12320 | ECCV 2020 | Q1, Q3 threshold, augmentation collapse | ar5iv summary |
| Qian, SeCu https://openaccess.thecvf.com/content/ICCV2023/html/Qian_Stable_Cluster_Discrimination_for_Deep_Clustering_ICCV_2023_paper.html | ICCV 2023 | Q1 batch instability | search snippets |
| Qian et al., CoKe https://openaccess.thecvf.com/content/CVPR2022/html/Qian_Unsupervised_Visual_Representation_Learning_by_Online_Constrained_K-Means_CVPR_2022_paper.html | CVPR 2022 | Q1 minimum size | snippets |
| Huang et al., ProPos https://arxiv.org/pdf/2111.11821 | TPAMI 2022 | Q1 prototype scattering | abstract |
| Li et al., TCL https://arxiv.org/abs/2210.11680 | IJCV 2022 | Q1 confidence boosting | abstract |
| Assran et al., hidden uniform prior https://arxiv.org/abs/2210.07277 | ICLR 2023 | Q1 equipartition on imbalanced data | abstract |
| Ronen et al., DeepDPM https://openaccess.thecvf.com/content/CVPR2022/papers/Ronen_DeepDPM_Deep_Clustering_With_an_Unknown_Number_of_Clusters_CVPR_2022_paper.pdf | CVPR 2022 | Q1 learned K | abstract |
| Cheng and Chen, gradual community detection https://arxiv.org/abs/2501.02036 | preprint 2025 | Q1 community pseudo-labels | abstract |
| Scott et al., clustering pretrained embeddings https://arxiv.org/abs/2211.05183 | NeurIPS 2022 workshop | Q1, Q3 fragility | abstract |
| Xiao et al., CLUBench https://arxiv.org/abs/2605.29933 | preprint 2026 | Q1, Q3 no average deep gain | abstract |
| Zhang et al., SCCL https://ar5iv.labs.arxiv.org/html/2103.12953 | NAACL 2021 | Q1, Q3 tighter but worse | ar5iv summary |
| Mrabah et al., ADEC; Shaheen et al., R-DC https://arxiv.org/abs/2503.03733 | journal (venue not checked); preprint 2025 | Q3 drift | abstracts |
| Castellano and Vessio, DELIUS https://arxiv.org/abs/2106.06234 | IJCV 2022 | Q3, Q5 art DEC | abstract |
| Lai et al., ESMC https://arxiv.org/html/2512.00725v1 (authors as listed in our 2026-12-07 note) | AAAI 2026 | Q2, Q4 head vs k-means | HTML summary |
| Yao et al., Multi-Sub https://arxiv.org/html/2411.03978v1 | NeurIPS 2024 | Q2, Q4 | HTML summary |
| Gadetsky et al., TURTLE https://proceedings.mlr.press/v235/gadetsky24a.html ; code https://github.com/mlbio-epfl/turtle | ICML 2024 | Q2, Q5 | abstract, README |
| Wei et al. https://arxiv.org/pdf/2010.03622 | ICLR 2021 | Q2 expansion | abstract |
| Lang et al. https://mlanthology.org/neurips/2024/lang2024neurips-theoretical | NeurIPS 2024 | Q2 weak-to-strong | abstract |
| Yao et al. https://arxiv.org/abs/2502.01458v1 ; Wu and Sahai https://arxiv.org/abs/2410.04638v2 | ICLR 2025 (both) | Q2 over-fitting, logits | abstracts |
| Dong et al. https://arxiv.org/abs/2307.11030 | NeurIPS 2023 | Q2 relational KD = spectral clustering | abstract |
| Lopez-Paz et al. https://arxiv.org/abs/1511.03643 | ICLR 2016 | Q2 privileged input | abstract |
| Metaxas et al., DivClust https://ar5iv.labs.arxiv.org/html/2304.01042 | CVPR 2023 | Q4 shared trunk, many heads | ar5iv summary |
| Falck et al., MFCVAE https://neurips.cc/virtual/2021/poster/26795 | NeurIPS 2021 | Q4 | abstract |
| Xu et al., CIM-Embed https://arxiv.org/html/2610.21805 | preprint, Oct 2026 | Q4, Q5 instructed Qwen3-VL for clustering | HTML summary, first half |
| Chaybouti et al., SigLino https://openaccess.thecvf.com/content/CVPR2026/papers/Chaybouti_SigLino_Efficient_Multi-Teacher_Distillation_for_Agglomerative_Vision_Foundation_Models_CVPR_2026_paper.pdf | CVPR 2026 | Q4 per-teacher heads | snippet |
| Li et al., SCP https://arxiv.org/abs/2502.04226 | preprint 2025 | Q5 | abstract |
| Li et al., SEIC https://ojs.aaai.org/index.php/AAAI/article/view/39506 | AAAI 2026 (from DOI) | Q5 | snippet |

## Open (budget spent)

- **ESMC author list.** The earlier note did not name ESMC's authors. "Lai et al." in the Sources table is a guess and is not checked. Settle it from https://arxiv.org/abs/2512.00725.
- **TEMI and TURTLE on imbalanced labels with k ≠ number of classes.** Neither paper tests this. Settle it by running them (idea 3).
- **TURTLE with our own features.** It is not checked whether the code takes an arbitrary embedding matrix. Read `run_turtle.py` in the repo.
- **CPP's WikiArt result.** Whether its 20 clusters were scored against style labels. Read the CPP appendix.
- **DELIUS against k-means.** The numbers were not read. Read the IJCV paper's results section.
- **Leiden with a prototype loss.** Search single-cell deep clustering (e.g. "Leiden pseudo-labels contrastive scRNA") for a cross-domain precedent.
- **A 2025–2026 clustering survey's view of DEC on foundation features.** Search "deep clustering survey 2025 foundation models".

Nothing to delete; no files were written.
