# Three-way scan for T3: losses that update the representation and the groups together (not DEC)

Scan date 2026-10-05. Mode: ARS deep-research three-way-scan (bibliography + source verification). Thread: T3 of `research_brief.md` (§7).

**Read scope.** Per paper, stated in each entry. Full text was read (PDF text extracted locally, the cited sections opened) for SCAN, SwAV, SeLa, IIC, XDC and TEMI. Abstract only for DeepCluster, MCN, TAC (abstract plus a WebFetch summary of the HTML full text, which I did not read line by line), IC|TC, DDMC, Multi-Sub, Zhou and Zhang 2022, R-DC. Where a WebFetch summary of a PDF disagreed with the PDF text, the PDF text was used (SCAN neighbours are mined once, not online; SCAN's ten heads select the lowest-loss head, they do not hold different targets).

**AI-disclosure.** This scan was produced by a Claude agent (WebSearch, WebFetch, local PDF text extraction). Retrieved content was treated as data. A human should read SCAN §2, SeLa §3.5, XDC §3 to §5 and TEMI §3 before citing mechanisms from this file.

**Queries run (WebSearch).** (1) Image Clustering with External Guidance TAC text-aided clustering Li ICML 2024; (2) the same with arXiv id lookup; (3) multi-view clustering representation alignment collapse (Trosten CVPR 2021); (4) deep multiple clustering with different targets, disentangled, image text guided; (5) swapped prediction between image and text modalities with shared prototypes; (6) deep clustering regularisation toward pretrained feature structure, frozen teacher, drift. Plus direct WebFetch of arXiv abs/PDF pages for every listed id.

**Sources searched.** arXiv abs, PDF and HTML, PMLR, CVF (blocked, HTTP 403 for CVPR 2021 and ICCV 2021 pages), Semantic Scholar (blocked, 403), general web search.

**Not searched.** OpenReview, ACL Anthology, Google Scholar, Zhou and Zhang or R-DC beyond abstracts, CLIP-specific clustering papers beyond TAC and IC|TC, multimodal SwAV-style image-text prototype papers (query 5 returned no paper that does it; see the gap), supplementary material of any paper. Network egress by curl was refused by the sandbox, so I did not download anything outside WebFetch.

---

## SCAN: Learning to Classify Images without Labels
Source: ECCV 2020 (Van Gansbeke, Vandenhende, Georgoulis, Proesmans, Van Gool) | Year: 2020 | Link: https://arxiv.org/abs/2005.12320
Read scope: full text (§2.1 to §2.3, supplementary ImageNet setup).
- WHY: End-to-end deep clustering latches onto low-level features and degenerates; decouple representation learning from clustering and use the pretext features as a prior (§1, §2.1).
- HOW: (i) Learn an embedding by a pretext task (instance discrimination), (ii) mine each sample's K nearest neighbours once in that embedding (§2.1), (iii) train a softmax clustering head with Eq. 2: `-mean log<Phi(X), Phi(k)>` over neighbours k, plus lambda times the negative entropy of the batch-mean assignment (lambda = 5), (iv) self-label: keep samples with max-probability above 0.99 and train with cross-entropy under strong augmentation (§2.3). Collapse is prevented by the entropy term; uniform assignment is the target, replaceable by a KL to a known prior (§2.2). ImageNet: backbone frozen, only a linear head trained, ten heads in parallel, the lowest-loss head is kept (supplement B.1).
- WHAT: +26.6 (CIFAR10), +25.0 (CIFAR100-20), +21.3 (STL10) points over prior work (abstract). The authors state that neighbours are noisy (some of the K neighbours are not in the same cluster) and that the self-labelling step exists to correct this (§2.3); failure cases are background-focused or look-alike images (supplement B.3).
  - Method weaknesses: (a) the neighbour set is fixed by the pretext embedding, so whatever property that embedding does not make neighbours on is invisible to the loss; under our data a CLIP-image neighbour graph links paintings by genre, so affect cannot be recovered from it. Reader-inferred. (b) The entropy term pushes toward uniform group sizes, which distorts the result when the true property classes are skewed. Author-acknowledged for the imbalance case only as a note that the class distribution can be replaced by a KL to a known one (§2.2); the distortion under skew is reader-inferred.
- Fit: needs a kNN graph (our buddy graph) and a head; per-row placement of a lone input works because the head is a function of one embedding. Anchoring to the teacher graph is built in (the neighbours are the teacher).

## Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV)
Source: NeurIPS 2020 (Caron, Misra, Mairal, Goyal, Bojanowski, Joulin) | Year: 2020 | Link: https://arxiv.org/abs/2006.09882
Read scope: full text (§3 method, appendix C).
- WHY: Pairwise instance comparison is memory heavy; compare cluster codes instead.
- HOW: Features z of two views are scored against K trainable prototypes C. Codes Q are computed online per batch by Sinkhorn-Knopp under an equipartition constraint (Eq. 3 to 4), and each view's code is predicted from the other view's features (swapped prediction, cross-entropy). The paper states the equipartition "ensures that the codes for different images in a batch are distinct, thus preventing the trivial solution where every image has the same code" (§3), and that a high entropy weight epsilon collapses codes to uniform, so epsilon is kept low (0.05). Appendix C notes DeepCluster-style tricks (re-assignment, balanced sampling) were unnecessary.
- WHAT: 75.3% ImageNet top-1 linear (ResNet-50), beating supervised pretraining on transfer (abstract).
  - Method weaknesses: (a) the swapped-prediction fixed point rewards whatever the two views share. With augmentation views that is object content; with an image and its own caption it would be the content CLIP already aligns, which in our data is genre and subject, not caption affect. Reader-inferred. (b) Equipartition over a batch forces roughly equal-sized groups; skewed affect classes (our median emotion spreads over 15 of 64 clusters) would be cut. Reader-inferred; the paper itself does not test skewed data in the sections read.
- Fit: needs two views per row (here image and caption, our adaptation, not in the paper) and a batch for Sinkhorn. Lone-modality placement at test time is direct (softmax over prototypes from one tower). No cross-modal use of SwAV was found by query 5.

## Self-labelling via Simultaneous Clustering and Representation Learning (SeLa)
Source: ICLR 2020 (Asano, Rupprecht, Vedaldi) | Year: 2019 (arXiv), 2020 | Link: https://arxiv.org/abs/1911.05371
Read scope: full text (§3.1 to §3.5, ablations).
- WHY: Cross-entropy on self-assigned labels has a degenerate solution (all points one label); fix it with a constraint, not a heuristic.
- HOW: Alternate (1) train the network on current pseudo-labels, (2) re-solve the labels by optimal transport under the constraint that the N points split uniformly among K classes, using fast Sinkhorn-Knopp (§3.1 to §3.2). The objective is read as maximising mutual information between labels and data indices, and the equipartition is what avoids degeneracy (§3.2). §3.5 "multiple simultaneous self-labellings": the same representation feeds T heads, one per clustering task, possibly with different numbers of labels, "which can potentially capture different and complementary clustering axis".
- WHAT: state of the art on SVHN, CIFAR, ImageNet at the time for AlexNet and ResNet-50 (abstract); ablations on K and T (Tables 2 and 3).
  - Method weaknesses: the multiple heads are all driven by the same loss on the same input and differ only through initialisation and K; nothing in the loss makes them capture different axes. Reader-inferred from §3.5 (no diversity term is described there). Equipartition shares the skew caveat of SwAV (reader-inferred; the paper's class-distribution test in its appendix was not read in detail).
- Fit: heads with different K on one shared representation is the nearest published form of "several groupings at once". It gives no source separation.

## Invariant Information Clustering for Unsupervised Image Classification and Segmentation (IIC)
Source: ICCV 2019 (Ji, Henriques, Vedaldi) | Year: 2018 (arXiv), 2019 | Link: https://arxiv.org/abs/1807.06653
Read scope: full text (§3 objective, overclustering, ablation).
- WHY: Maximise mutual information between the cluster assignments of two views of the same sample; no reconstruction, no k-means.
- HOW: For paired soft assignments z, z', form the joint P = mean of outer products over the batch, symmetrise, and maximise I(z; z') (six lines, Fig. 4). Degeneracy is avoided by the entropy of the marginal (maximising H(z)) while minimising H(z|z') (§3.1). Auxiliary overclustering head (more clusters than classes, trained on all data, ignored at test) and several sub-heads (Fig. 2, ablation: no auxiliary head 43.8, single sub-head 57.6 on the reported ablation). It is stated to work "on any paired dataset, not just images" (abstract summary).
- WHAT: beats closest competitors by 6.6 (STL10) and 9.5 (CIFAR10) points (abstract).
  - Method weaknesses: the objective is symmetric in what the two views share, so any property carried by only one view contributes nothing to I(z; z'). Reader-inferred. Paper-acknowledged related point: a distractor subset needs the auxiliary head to be usable (§3, "Auxiliary overclustering").
- Fit: the cleanest published loss for "agreement between an image's and a caption's assignment" (z from image tower, z' from caption tower). Placement of a lone input is direct. Nearest to our reader's agreement p_img . p_txt.

## Deep Clustering for Unsupervised Learning of Visual Features (DeepCluster)
Source: ECCV 2018 (Caron, Bojanowski, Joulin, Douze) | Year: 2018 | Link: https://arxiv.org/abs/1807.05520
Read scope: abstract_only (the degeneracy discussion was read in its quotations by SeLa and SwAV, second hand).
- WHY: Unsupervised visual features without a hand-designed pretext task.
- HOW: Alternate k-means on the network's features with training the network to predict the assignments (abstract). Second-hand: it avoids the all-one-label solution via empty-cluster reassignment and uniform sampling over clusters (SeLa §3.3; SwAV appendix C).
- WHAT: beat prior unsupervised methods on ImageNet and YFCC100M (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).
- Fit: the template for XDC. It is a hard-assignment self-sharpening loop in the same family as DEC in spirit (the model learns the labels it produced), so the percept-line failure of DEC is a warning for it; this link is reader-inferred.

## Self-Supervised Learning by Cross-Modal Audio-Video Clustering (XDC)
Source: NeurIPS 2020 (Alwassel, Mahajan, Korbar, Torresani, Ghanem, Tran) | Year: 2019 (arXiv), 2020 | Link: https://arxiv.org/abs/1911.12667
Read scope: full text (§3 to §5, supplementary on trivial solutions).
- WHY: Audio and video are correlated but carry distinct information; one modality's clusters can supervise the other (§1).
- HOW: Four variants: SDC (DeepCluster), MDC (a second head per encoder supervised by the other modality), CDC (concatenated features clustered), XDC (each encoder trained exclusively on the other modality's k-means pseudo-labels). Collapse: the paper says SDC may reach trivial solutions because it learns the classifier on the same input it takes labels from; XDC is "less prone to trivial solutions because they learn the discriminative classifier on one modality and obtain the labels from a different modality", and no empty clusters were observed (supplement, "Trivial solutions").
- WHAT: XDC beats SDC, MDC, CDC (Study 1); same-modality XDC is 8 to 12 points worse than cross-modal XDC, which the authors read as the benefit of the other modality, not of the optimisation; the best k (64 to 1024 tried) is not tied to the number of downstream labels (Study 2). Explanation offered for XDC over CDC: XDC groups samples similar in either modality, CDC only those similar in both (§4, Study 1 (III)).
  - Method weaknesses: the supervision of each tower is the other tower's k-means partition of its own raw features, so a property that neither modality clusters on is never supervised; hard labels inherit the k-means shape. Reader-inferred. Author-acknowledged: MDC underperforms, attributed to different learning speeds of the modalities when trained jointly (§4, Study 1 (III)).
- Fit: the same as the N6 heads, plus an update of the encoders. One-row placement from a lone modality is native (each encoder predicts from its own input). Whether affect (weak in images) survives is open; XDC's own result that either-modality similarity helps is mild support for keeping separate groupings per source.

## Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos (MCN)
Source: ICCV 2021 (Chen, Rouditchenko, Duarte, Kuehne, Thomas, Boggust, Panda, Kingsbury, Feris, Harwath, Glass, Picheny, Chang) | Year: 2021 | Link: https://arxiv.org/abs/2104.12671
Read scope: abstract_only (the full PDF exceeded WebFetch's size limit; the CVF page returned 403).
- WHY: Instance-level contrast treats semantically similar instances of other videos as negatives.
- HOW: Adds a multimodal clustering step to contrastive training of a joint embedding over video, audio and text (abstract).
- WHAT: state-of-the-art zero-shot text-to-video retrieval and temporal action localisation after HowTo100M training (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).
- Fit: cluster centres shared across modalities inside a joint space; collapse prevention and the loss weights are not assessed here.

## Exploring the Limits of Deep Image Clustering using Pretrained Models (TEMI)
Source: BMVC 2023 (Adaloglou, Michels, Kalisch, Kollmann) | Year: 2023 | Link: https://arxiv.org/abs/2303.17896
Read scope: full text (§1, §3, §4, §5, appendix A).
- WHY: Frozen pretrained features already have usable neighbourhood structure; k-means on them is imbalanced and suboptimal; train only a head.
- HOW: Student and EMA teacher heads on frozen features g(x). For a pair (x, x') mined as 25-nearest neighbours in g, a pointwise-mutual-information loss with a hyperparameter beta in (0.5, 1] (0.6 for clustering; 1 for overclustering) that "avoids collapsing all sample pairs in a single cluster" (§3.3). H = 50 independent heads, with an instance weight from the mean of the teacher ensemble's agreement, to damp false-positive neighbours (§3.4). Choose beta by keeping the teacher's marginal entropy sufficiently low (appendix A.1).
- WHAT: +6.1 (ImageNet) and +12.2 (CIFAR100) points of accuracy over k-means across 17 pretrained models; 61.6% ImageNet with self-supervised ViTs (abstract). With only true-positive neighbours, CIFAR100 accuracy rises 67.1 to 82.6 (§5, noise analysis), so neighbour noise is the limit; DINO ViT-B has 72% true positives in 20-NN (same section).
  - Method weaknesses: author-acknowledged: (a) cluster use is far from uniform in training, attributed to over-confident predictions early on (§3.3), which is why beta exists; (b) jointly fine-tuning the backbone helped only when the pretraining dataset differed from the downstream one (§5, "Joint learning of encoder and cluster head"). Reader-inferred: the loss trusts the neighbour graph, so a graph that links by the dominant property (CLIP image kNN linking by genre) trains a head that reproduces that property.
- Fit: this is "buddy kNN graph plus a head" in a recent form, on frozen features, cheap, and with the teacher graph as the anchor. Per-row placement from one embedding is native.

## Image Clustering with External Guidance (TAC)
Source: ICML 2024 (Li, Hu, Peng, Lv, Fan, Peng), PMLR 235 | Year: 2024 | Link: https://arxiv.org/abs/2310.11989 (PMLR: https://proceedings.mlr.press/v235/li24aa.html)
Read scope: abstract_only plus a WebFetch summary of the HTML full text (§3.1, §3.2, §4.3.2, §5 as reported by that summary; I did not read those sections line by line).
- WHY: Image-only clustering is limited; external semantics can supervise it.
- HOW: Text counterparts are not captions: WordNet nouns are retrieved per image by classifying nouns into k-means centres and soft-weighting the top nouns (per the summary, §3.1). Then two heads (image, text) are trained by mutual distillation of cluster assignments over each other's neighbourhoods, with a confidence loss and a balance (entropy) loss against collapse, `L_Dis + L_Con - 5 L_Bal` (per the summary, §3.2).
- WHAT: state of the art on five standard and three harder image clustering benchmarks including ImageNet-1K (abstract). Ablation (per the summary, §4.3.2): text-to-image 69.4, image-to-text 67.1, both 72.2 ARI on ImageNet-Dogs.
  - Method weaknesses: not assessed (read scope: abstract_only). The conclusion, per the summary, says performance depends on data bias and the cluster number must be set by hand; I did not confirm this wording.
- Fit: correction to the brief: TAC uses retrieved noun counterparts, not paired captions. Its cross-modal neighbourhood distillation and balance loss are directly reusable on our true image-caption pairs.

## Image Clustering Conditioned on Text Criteria (IC|TC)
Source: arXiv (Kwon, Park, Kim, Cho, Ryu, Lee); venue not confirmed on the page | Year: 2023 | Link: https://arxiv.org/abs/2310.18297
Read scope: abstract_only.
- WHY: Users want clusterings by a stated criterion (action, location, emotion) rather than the dominant one.
- HOW: Vision-language and large language models steer the clustering by a text description of the criterion (abstract).
- WHAT: better than baselines across criteria types (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).
- Fit: evidence that the same images cluster differently under different criteria and that the dominant criterion has to be overridden by an explicit signal; it needs a text criterion at run time, which our unnamed aspects do not supply.

## Dual-disentangled Deep Multiple Clustering (DDMC)
Source: SDM 2024 (Yao, Hu) | Year: 2024 | Link: https://arxiv.org/abs/2402.05310
Read scope: abstract_only.
- WHY: Several hidden clusterings exist; earlier work controls dissimilarity but not clustering focus.
- HOW: Variational EM: E-step disentangles coarse and fine latent factors, M-step assigns clusters (abstract).
- WHAT: better than existing multiple-clustering methods on seven benchmark tasks (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).
- Fit: a multi-head method with a diversity mechanism, unimodal. Same family as MFCVAE (T1).

## Customized Multiple Clustering via Multi-Modal Subspace Proxy Learning (Multi-Sub)
Source: NeurIPS 2024 (Yao, Qian, Hu) | Year: 2024 | Link: https://arxiv.org/abs/2411.03978
Read scope: abstract_only.
- WHY: Existing multiple clustering methods cannot follow a user's interest.
- HOW: CLIP and GPT-4: an LLM generates proxy words that act as subspace bases, aligned with image features, to give an interest-specific clustering (abstract).
- WHAT: better than baselines on visual clustering benchmarks (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).
- Fit: separates properties inside CLIP space by a text-defined subspace; needs the property named, so it applies to hand-matched sources only.

## Deep Clustering with Features from Self-Supervised Pretraining
Source: arXiv 2022 (Zhou, Zhang) | Year: 2022 | Link: https://arxiv.org/abs/2207.13364
Read scope: abstract_only.
- WHY: Training feature extractor and head jointly is costly and fragile; use pretrained self-supervised features.
- HOW: Two-stage: frozen pretrained ViT features, then a clustering head (abstract).
- WHAT: CIFAR-10 94.0%, CIFAR-100 55.6%, STL-10 97.9% accuracy, above prior bests 84.3, 47.7, 80.8 (abstract). The abstract notes possible vulnerability to domain shift.
  - Method weaknesses: not assessed (read scope: abstract_only).
- Fit: supports a frozen-trunk design for L and P0 (heads only).

## Rethinking Deep Clustering Paradigms: Self-Supervision Is All You Need (R-DC)
Source: arXiv 2025 (Shaheena, Mrabahb, Ksantinia, Alqaddoumia) | Year: 2025 | Link: https://arxiv.org/abs/2503.03733
Read scope: abstract_only.
- WHY: Pseudo-supervision and self-supervision conflict, producing feature randomness, feature drift and feature twist.
- HOW: Replace pseudo-supervision by a second round of self-supervision moving from instance level to neighbourhood level (abstract).
- WHAT: substantial improvements on six datasets (abstract).
  - Method weaknesses: not assessed (read scope: abstract_only).
- Fit: independent statement of the drift problem that matches our DEC evidence; the paper has not been read beyond its abstract, so I rely on it only as a pointer.

---

## Cross-paper synthesis
**Common WHY.** A clustering loss needs a self-supplied target that cannot be met by collapse. Everything above supplies it from consistency between two things: two augmented views (SwAV, IIC), a sample and its neighbours (SCAN, TEMI), or two modalities (XDC, MCN, TAC).

**Divergent HOW.**
- Collapse prevention: marginal entropy (SCAN, IIC, TAC balance loss), equipartition by optimal transport (SeLa, SwAV), a tempered PMI with beta (TEMI), or label source different from input (XDC).
- Where the positive pair comes from: augmentation (SwAV, IIC), mined neighbours (SCAN, TEMI), the other modality's cluster (XDC), the other modality's neighbourhood (TAC).
- Multiple heads: SCAN (ten heads, keep the best), TEMI (50 heads, ensembled), SeLa §3.5 (heads for different tasks, different K), IIC (sub-heads, an overclustering head), XDC MDC (a head per modality). In none of the papers read is each head given a different signal; they differ by initialisation or K.

**Strongest WHAT.** (1) Heads on frozen pretrained features with a kNN-consistency loss reach large gains over k-means (TEMI: +6.1 ImageNet, +12.2 CIFAR100; Zhou and Zhang; SCAN with a frozen ImageNet backbone), at the cost of a head trained in minutes to hours. (2) Cross-modal supervision beats same-modality supervision by 8 to 12 points in XDC. (3) Neighbour noise is the limit: TEMI gains 15 points when false-positive neighbours are removed.

**Unresolved gap (to our knowledge, within this search).** I found no paper that (a) trains per-source clusterings so that a weak source (affect) is protected from a dominant one (content) inside a single shared loss, nor (b) applies SwAV or IIC across image and caption towers with the pair given by the dataset rather than by augmentation or retrieval. The closest are TAC (retrieved nouns, one clustering) and XDC (one modality supervises the other, one k-means each). No paper read analyses which property a swapped-prediction fixed point favours; the dominance statements above are reader-inferred.

## Comparison of candidate losses
Applied to our setting (one row = image + caption; lone image or caption at test time; frozen CLIP features as input).

| Loss | Inputs | Collapse prevention | Dominance risk (reader-inferred unless noted) | Lone-modality placement | Cost |
|---|---|---|---|---|---|
| SCAN consistency + self-label | kNN graph from a teacher embedding, head | entropy of mean assignment (lambda 5) | the graph's dominant property wins; a content graph yields genre groups | yes (head on one embedding) | low (head only) |
| TEMI (WPMI, EMA teacher, 50 heads) | kNN graph on frozen features | beta in (0.5, 1], teacher marginal entropy | same as SCAN; its instance weights help against noisy neighbours (author-shown on CIFAR100) | yes | low (minutes to hours, frozen) |
| SwAV-style swapped prediction (image <-> caption of a row) | paired rows, prototypes, batch | Sinkhorn equipartition | shared information between image and caption is mostly content; affect, which the image head hardly sees, is weakly rewarded; equipartition conflicts with skewed groups | yes (two towers, shared prototypes) | medium (batch Sinkhorn, tower training) |
| SeLa | one tower, full-data OT labelling | global equipartition | same skew issue; no cross-modal term | yes | medium |
| IIC (image tower vs caption tower) | paired rows | marginal entropy minus conditional entropy | symmetric in the shared part: a property in one view only adds nothing | yes | low to medium |
| XDC (cluster one modality, train the other's head) | per-modality k-means, paired rows | label from the other modality (author-reported none observed) | each head inherits the other modality's raw-feature clusters; image k-means is genre; caption GoEmotions k-means would give affect to the image head, which is the weak side | yes | low (our N6 heads already do this once) |
| TAC-style cross-modal neighbourhood distillation | two kNN graphs (image, caption), two heads | balance (entropy) loss, confidence loss | graph-limited, as SCAN; both directions better than one (author-reported, Dogs ARI 72.2 vs 69.4 and 67.1) | yes | low to medium |
| Multi-head SeLa / SCAN heads with separate sources | one trunk, one head per source | per-head entropy or OT | per-head targets keep the source; shared trunk may let content leak into the affect head (unknown) | yes | low |

## Bearing on the designs P0, L, G, E

**P0 (one grouping per source, heads).**
- Literature-supported: TEMI and Zhou and Zhang report that a head on frozen features is competitive; XDC supports training each modality's head on the other's clusters. A loss swap inside P0 is cheap: replace the logistic-regression heads by a TEMI- or SCAN-style head on a source graph with entropy balancing.
- Reader-inferred: placeability of a lone image into the affect grouping is bounded by what the image carries about caption affect; no loss in this list changes the information in the CLIP image feature (frozen trunk), so a head loss can raise placeability only toward that bound.

**L (groupings kept, a small model refines them jointly).**
- Reader-inferred best match: one trunk on frozen CLIP features with one head per grouping, each head trained with a SCAN/TEMI consistency term on its own source's graph (the source anchor) plus a cross-grouping term. SeLa §3.5 is the published precedent for several heads on a shared trunk, with the caveat that its heads are not made different by the loss. Source anchor: keep each head's positives defined by its own source graph, and add a penalty to the original source assignment (KL from the head's posterior to the source cluster membership, as SCAN's KL variant in §2.2 does with a known prior). Not tested anywhere I found.
- Non-redundancy between heads: no loss here supplies it; IIC between two heads' outputs would be the wrong sign (it rewards agreement), so a penalty on I(z_a; z_b) would have to be added, and is not published in these papers (T1 and T2 cover it).

**G (multiplex graph, shared Leiden partition across layers).**
- These losses are not needed to build the partition. If a head is added after the partition, SCAN or TEMI on the fused graph is the published form; the dominance risk is inherited from the graph (TEMI §5: neighbour noise is the limit, and a pooled graph adds disagreeing neighbours). Percept-line evidence in the brief (union graph merged content communities) agrees; reader-inferred.

**E (pooled edges, two-tower trunk, competing property heads).**
- The two-tower trunk is where SwAV, IIC and TAC apply: IIC between the image-tower and caption-tower assignments of the same row gives cross-modal agreement directly; SwAV adds equipartition; TAC's two-way neighbourhood distillation uses both modality graphs. Fixed-point dominance (reader-inferred): all three reward information shared by the two towers. Content is shared; caption affect is not. Without a separate affect head anchored to its source, the weak property is expected to lose, which is consistent with the percept-line Attention-h1 student that traded emotion for genre.
- Anchoring options for E, none verified for our setting: (a) keep the backbone frozen (TEMI §5 reports joint fine-tuning helped only under a domain gap; R-DC and the drift statement in its abstract point the same way); (b) positives from the source graph only (SCAN prior); (c) a KL or distillation term to the source posterior (SCAN §2.2 KL variant). Zhou and Zhang's frozen-feature result supports (a).

**Per-row, lone-modality use.** All of the losses above place a lone input with a head on one embedding, so test-time use is unproblematic; the loss differs only in training-time inputs (a paired row, a neighbour list).

## Brief citation check
| Brief says | Result |
|---|---|
| SCAN (Van Gansbeke et al. 2020) | Verified. Title "SCAN: Learning to Classify Images without Labels", ECCV 2020, arXiv 2005.12320. The brief's gloss "neighbour consistency, close to buddy graph plus a clustering head" is accurate; neighbours are mined once from the pretext embedding. |
| SwAV (Caron et al. 2020) | Verified. NeurIPS 2020, arXiv 2006.09882. "Swapped prediction between a row's image and caption" is our adaptation; the paper uses image augmentations. |
| SeLa (Asano et al. 2020) | Verified. ICLR 2020, arXiv 1911.05371 (full title "Self-labelling via Simultaneous Clustering and Representation Learning"); the first author is Yuki Markus Asano with Rupprecht and Vedaldi. |
| IIC (Ji et al. 2019) | Verified. ICCV 2019, arXiv 1807.06653 (submitted 2018). |
| DeepCluster (Caron et al. 2018) | Verified. ECCV 2018, arXiv 1807.05520. |
| XDC (Alwassel et al. 2020) | Verified. NeurIPS 2020, arXiv 1911.12667. |
| MCN (named in scan_N6, not in this brief's T3 list) | Verified at abstract level: ICCV 2021, Chen et al., arXiv 2104.12671. |
| TAC, Li et al. ICML 2024 "if it exists" | Exists, with a correction: "Image Clustering with External Guidance" (TAC = Text-Aided Clustering), Li, Hu, Peng, Lv, Fan, Peng, ICML 2024, arXiv 2310.11989. It uses retrieved WordNet nouns as text counterparts, not paired captions. |
| TEMI, Adaloglou et al. 2023 | Verified. "Exploring the Limits of Deep Image Clustering using Pretrained Models", BMVC 2023, arXiv 2303.17896. TEMI is the name of its final loss (teacher ensemble-weighted pointwise mutual information). |
| scan_N6 id for SCAN | I tried arXiv 2005.04200 at first, which is an unrelated paper; the correct id is 2005.12320. scan_N6 does not list a SCAN id, so no correction is needed there. |

## Not verified
- Trosten, Løkse, Jenssen, Kampffmeyer, "Reconsidering Representation Alignment for Multi-view Clustering", CVPR 2021 (CoMVC). Seen in search results (title, venue and a summary), but both the CVF page and Semantic Scholar returned 403, so I could not confirm authors or text. Not cited above. Its reported point (naive alignment of view representations prevents prioritising views and blurs clusters) would bear on dominance in cross-modal alignment if confirmed.
- A claim from a search-result summary that fine-tuning the feature extractor during a clustering phase dropped accuracy from 92.1% to 65.7% and merged clusters. I did not open the paper it came from, so it is not used.
- Any paper using SwAV, IIC or SeLa across an image and a text tower (the gap above): none found in query 5; absence is bounded by that search.
- DR-Tune (ICCV 2023, arXiv 2308.12058) appeared in search results for distribution regularisation of a task head toward pretrained features; I did not open it.
