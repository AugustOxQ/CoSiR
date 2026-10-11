# Design L: clustering across a known partition, style statistics from a generic backbone, novelty, comparators and a second dataset

> Deep literature check for round 7 (design L), r7 decide chat, 2026-10-11; Opus, deep depth; builds on 2026-11-27_style_sources.md.

**Budget:** all 40 searches and about 19 page reads are used. "Abstract only" means only the abstract or a listing was seen. "Unverified" means the claim rests on a search snippet or on memory. "Reader-inferred" means it is our reasoning, not a source's claim.

## 1. Results brief

1. **The literature supports the mechanism but takes away its novelty.** Building a clustering that is non-redundant with a known partition is about 20 years old (high confidence):
   - Gondek and Hofmann (ICDM 2004).
   - Cui, Fern and Dy, "orthogonalization" (ICDM 2007): project out the first clustering, then cluster again.
   - Qi and Davidson (KDD 2009).
   - Fair clustering is the deep version, with code: DFC (CVPR 2020) and FCMI (CVPR 2023). FCMI minimises I(cluster; protected group) and maximises I(X; cluster | group).

   The paper cannot claim the non-redundancy step. It can claim that such groupings are used as the conditions an example-conditioned scorer steers by, and that they carry style apart from genre on frozen features.
2. **It changes the design in three places.**
   - (a) **The ridge residual removes only what is linear.** It is the same object as LEACE's least-squares residual, which only guarantees that no linear predictor can recover the removed part. Nonlinear content survives linear erasure (Ravfogel et al., EMNLP 2022), and Leiden on a kNN graph is nonlinear. So measure leftover content with the grouping itself (AMI against the CLIP-image grouping and against genre), not with a linear probe.
   - (b) **The residual may also delete style.** CLIP linearly carries style: a linear probe on CLIP reaches 51.2% WikiArt style accuracy (GOYA, from the earlier note). The ridge residual of CSD on CLIP may therefore come out close to noise. That is the classic failure of alternative clustering: dissimilarity is bought with quality (Qi and Davidson make the trade-off an explicit knob). Pair every arm with a stability check against a size-matched random grouping (already in synthesis §VI).
   - (c) **The generic arm should rely on a different source, not on a second cut of the same embedding.** On Clevr-4 (NeurIPS 2023), k-means on CLIP ViT-B/16 features follows shape (78.7% accuracy) and sits at chance on texture (12.4) and colour (12.3). MAE follows colour (64.7). A second CLIP-image grouping made to cut across the first has little non-content structure to find. Patch-token statistics are the better generic bet (reader-inferred from that table).
3. **Closest prior for the grouping step.** Fair clustering with the content grouping as the "protected attribute" (FCMI), and Cui et al.'s orthogonal re-clustering. Neither has been run on frozen CLIP features against a content grouping, or on art. Closest prior for the whole method: none found (§4).
4. **Strongest comparator** (our reading). A bank of criteria discovered by an MLLM (X-Cluster, CVPR 2026 Findings: no human input, LLaVA-NeXT captions plus a Llama-3.1 proposer), placed under the same reader. A reviewer will ask why we don't let an MLLM name "style". Next come the cheap label-free baselines on the same CLIP features: ClustPy's Orth1/Orth2 (Cui et al.), NrKmeans and ENRC, and FCMI with the CLIP-image grouping as the protected attribute. As a privileged ceiling: Multi-MaP (CVPR 2024, frozen CLIP, code public) given the keyword "style".
5. **Parked ideas** (one line each):
   - A learned cross-cutting source: an InfoNCE head on frozen CLIP whose negatives come from the query's own content community (conditional contrastive learning, Tsai et al. 2021 and ICLR 2022), so content cannot separate the pairs.
   - FCMI on frozen features with the content grouping as the protected attribute, as a grouping builder rather than only as a comparator.
   - Mean and standard deviation of intermediate-layer CLIP tokens, as StyLIP uses them, with high-norm "register" tokens dropped, instead of final-layer statistics.
6. **Best second dataset:** COCO-4c, from the X-Cluster paper.
   - 5,000 COCO val2017 images, human-labelled for activity (64 clusters), location (19), mood (20) and time of day (6).
   - The standard COCO captions exist for these images (reader-inferred join on the COCO image ID).
   - Mood mirrors our affect aspect.
   - A controlled, image-only test: Clevr-4 (four independent 10-class facets, CC BY 4.0).

## 2. Q1: clustering against or across a known partition

**Answer** (high confidence that the area exists, medium on details).
- The shallow line (2004 to 2010) takes a given partition explicitly.
- The deep multiple-clustering line (2020 to 2026) moved to text criteria once CLIP arrived.
- Fair clustering is the only deep line that takes a given partition and outputs a clustering independent of it.

| Work | Venue | Needs labels or text? | Works on frozen features? | Evidence the facets separate | Known failure |
|---|---|---|---|---|---|
| Conditional information bottleneck: Gondek and Hofmann | ICDM 2004; KAIS 2007 | Given partition only | Any feature vectors | Content from a summary only (T1) | n/a |
| Orth1/Orth2: Cui, Fern, Dy | ICDM 2007 | None | Yes (implemented in ClustPy) | Not read | Not read |
| Davidson and Qi; Qi and Davidson | ICDM 2008; KDD 2009 | Given clustering (as constraints) | Yes | Abstract only | Alternativeness traded against quality, a user knob (KDD 2009 abstract) |
| Niu, Dy, Jordan; NrKmeans; AutoNR | ICML 2010; KDD 2018; SDM 2022 | None | Yes (ClustPy) | Fruit demo, NMI (PyPI page) | n/a |
| ENRC (Miklautz et al.) | AAAI 2020 | None | No: trains an autoencoder (ClustPy has it) | In Multi-MaP's table: NMI 0.43 (Fruit360 colour), 0.25 (Cars colour) | Weak on real images, judged by those numbers |
| AugDMC (Yao et al.) | arXiv 2306.13023; venue unverified | Hand-picked augmentations act as aspect selectors | No | NMI 0.46 on Fruit360 colour (Multi-MaP table) | The augmentation choice encodes the aspect |
| MFCVAE | NeurIPS 2021 | None | Its own VAE | Facet probes 94 to 100% (T1) | Assumes independent facets |
| IC\|TC (Kwon et al.) | ICLR 2024 | Text criterion; VLM + LLM | No CLIP clustering; captions | Stanford40 action/location/mood, PPMI (abstract) | Needs the criterion named |
| Multi-MaP (Yao, Qian, Hu) | CVPR 2024 | Keyword + GPT-4 candidate words | Frozen CLIP; learns proxy words | NMI 0.62 Fruit360 colour, 0.74 Cars colour, against AugDMC's 0.46 and 0.27 | "limited by data with semantic meaningful labels"; WordNet words for identity |
| Multi-Sub | NeurIPS 2024 | Keyword + GPT-4 | Text encoder frozen; image projection trained | Fruit360 colour 0.67 NMI | GPT-4 bias; small datasets (its own statement) |
| TGAICC (Stephan et al.) | RepL4NLP workshop 2024 | Text prompts + VLM | Embeddings; code Apache-2.0 | 4 benchmarks (abstract) | n/a |
| X-Cluster / OpenSMC (Liu et al.) | CVPR 2026 Findings | None: criteria proposed from captions | MLLM captions, not CLIP | Average 59.9 clustering accuracy; mood hardest (38.1) on COCO-4c | Style criteria only in the "hard" list, never scored; per-image VQA noise |
| DFC (Li, Zhao, Liu) | CVPR 2020 | Protected attribute | Trained | MNIST-USPS with domain as protected (via later papers) | n/a |
| FCMI (Zeng et al.) | CVPR 2023 | Protected attribute | Trained; code public | 6 benchmarks, 11 baselines (abstract) | Trade-off not stated on the abstract page |
| Conditional contrastive learning (Tsai et al.) | arXiv 2106.02866; CCL-K at ICLR 2022 | Conditioning variable | Learns a representation | Fairness and weak supervision (abstract) | Too few pairs for some condition values (CCL-K's own motivation) |

**Closest prior to "a label-free grouping built to cut across the content grouping, on frozen CLIP features":** FCMI with the content grouping as the protected attribute, and Cui et al.'s Orth1. Both are generic and label-free once the first partition is given. In fair-clustering benchmarks the protected attribute is the domain (MNIST vs USPS) and the target is content. Ours is the inverse.

**Comparators for the grouping step:** ClustPy (Orth1, Orth2, NrKmeans, AutoNR, ENRC) runs on any feature matrix, at CPU or GPU minutes. FCMI's official code needs adapting to feature input (unverified).

## 3. Q2: appearance against content in a generic backbone, label-free

**Answer** (medium confidence). Evidence exists that ViT statistics carry appearance. None measures leftover content or genre, and none removes the image direction. Our arm would be a new test, not a replication.

- **Splice** (Tumanyan et al., CVPR 2022), read on ar5iv:
  - The deepest DINO-ViT [CLS] carries appearance, and the self-similarity of keys carries structure, shown by inversion and PCA (Figs. 3 to 5).
  - In shallow layers the [CLS] holds "local texture patterns", in deep layers "object parts". The deep [CLS] is therefore not content-free.
  - No Gram matrix or patch statistics are compared. A TOG extension exists (arXiv 2311.12193; not read).
- **StyLIP** (arXiv 2302.09251; WACV 2024 per memory, unverified):
  - Takes the per-layer channel mean and standard deviation of frozen CLIP token features as style and domain features.
  - The premise, that these statistics carry domain information, is cited, not tested. Ablations only show both statistics help accuracy.
- **Clevr-4** (Vaze, Vedaldi, Zisserman, NeurIPS 2023, Table 2): final features of CLIP B/16, DINO, DINOv2 and iBOT cluster by shape, at chance on texture and colour. MAE clusters by colour (64.7). Different pretraining exposes different facets; no single one exposes all.
- **Naseer et al.** (NeurIPS 2021): ViTs are "highly permutation invariant to patch positions". Late-layer patch tokens are close to a bag of global semantics, so their pooled statistics will carry content (reader-inferred; the paper gives its numbers only in figures).
- **Darcet et al.** ("Vision Transformers Need Registers", ICLR 2024; arXiv 2309.16588, snippet only): high-norm outlier patch tokens hold global information. Drop them before pooling statistics.
- **Archetypal Style Analysis** (Wynen, Schmid, Mairal, NeurIPS 2018, abstract only): unsupervised painting styles from deep-feature statistics (VGG).
- **Toronto poster** (Pournajafian, not peer reviewed): has a "Gram vs CLS" plot without numbers. Its wrong style neighbours share the query's object class more often as probe depth grows.

**"Texture statistics of CLIP patch tokens with the image direction removed": not found.** Confidence medium-low, from three searches. Use intermediate layers and filter outlier tokens.

## 4. Q3: novelty

| Statement | Our verdict | Confidence | Closest prior and what separates us |
|---|---|---|---|
| Making a grouping non-redundant with a known (content) grouping | **Not new** | High | Gondek and Hofmann 2004, Cui et al. 2007, FCMI 2023 |
| Choosing among several precomputed views from user examples | **Not new as an idea** | Medium | Relevance feedback re-weights feature types from marked examples (Rui et al., IEEE TCSVT 1998, abstract only). COBS picks among candidate clusterings by pairwise constraints (Van Craenendonck and Blockeel, arXiv 1609.07272; a Machine Learning journal version per a listing). Neither scores cross-modal pairs, uses contrast pairs, or works per query. |
| Label-free groupings on frozen CLIP, with no text criterion | Partly new | Medium-low | Multi-MaP, Multi-Sub and IC\|TC need a named criterion. X-Cluster discovers criteria, but through captions and an LLM. ENRC needs an autoencoder. |
| Label-free groupings made non-redundant with a content grouping, used as the conditions an example-conditioned image–caption scorer steers by | **New, to our knowledge** | Medium-low | No work found that combines these; the earlier notes (GeneCIS, CLAY, CRL, SCE-Net, DiscoverNet) take text or train masks. |

The defensible claim is the combination plus the measured result: a cross-cut grouping that carries style apart from genre and steers retrieval. The mechanism alone is not defensible.

## 5. Q4: strongest comparators

| Comparator | Criterion given as | Public code | Runs on frozen CLIP features? | Role |
|---|---|---|---|---|
| X-Cluster criteria, under our reader | Discovered (MLLM) | Abstract says "will be open-sourced"; project page oatmealliu.github.io/xcluster.html; release unverified | No: captions; we would cluster per criterion, then the reader picks | **Must-run**: answers "why not let an MLLM find style?". Risk: an MLLM knows WikiArt style names, a knowledge leak to discuss. |
| Verbalise-then-name + CRL (from the 2026-11-25 note) | Examples → text | CRL public | Yes | Must-run (already planned) |
| ClustPy Orth1/Orth2, NrKmeans, ENRC on CLIP features | Discovered | Yes | Yes | Cheap label-free baseline for the grouping step |
| FCMI with the CLIP-image grouping as protected attribute | Discovered (given partition) | github PengxinZeng/2023-CVPR-FCMI | Needs adaptation (unverified) | Strongest prior for the non-redundancy step |
| Multi-MaP or Multi-Sub with the keyword "style" or "emotion" | Text keyword | Yes (Alexander-Yao) | Multi-MaP: yes (CLIP variant unstated in the README); Multi-Sub trains a projection | Privileged ceiling |
| IC\|TC, TGAICC | Text criterion | Yes | No | Report-only |

## 6. Q5: datasets for a generality test

| Dataset | Size | Aspects (clusters) | Captions | Licence | Fit for aspect episodes |
|---|---|---|---|---|---|
| **COCO-4c** (X-Cluster) | 5,000 | activity 64, location 19, mood 20, time of day 6 | COCO val2017 has 5 per image (join reader-inferred) | COCO terms; new labels follow source licences (paper) | **Best**: image–text, 4 human-labelled aspects, mood ≈ affect; small |
| Food-4c | 25,250 | food type 101, cuisine 15, course 5, diet 4 | No | Food-101 terms | Image–image only |
| Clevr-4 | 10k or 100k | texture, shape, colour, count (10 each) | No (synthetic) | CC BY 4.0 | Controlled test of "cut across"; templated captions would name the aspect |
| SemArt | 21,384 (19,244 / 1,069 / 1,069) | author, type, school, timeframe, technique | Yes, art comments | Non-commercial research (Aston) | Second art set: type ≈ genre, school/timeframe ≈ style. Same domain, so not a cross-domain test. |
| Stanford40 (IC\|TC labels) | about 9.5k (unverified) | action, location, mood | No | Not checked | Image–image |
| CUB + Reed captions | 11,788 (5,994 / 5,794) | species 200 + 312 binary attributes | 10 per image | Not checked | Attributes are binary; aspects need grouping |
| DeepFashion-MultiModal | 11,484 or 44,096 (sources conflict) | shape, fabric, colour | Yes | Unclear | Captions describe the same attributes, so leakage |
| Multi-clustering sets: Fruit, Fruit360, Cards, CMUface, Cars, Flowers | 105 / 4,856 / 8,029 / 640 / 1,200 / 1,600 | 2 to 4 facets | No | Mixed | Too small; reviewers know them from Multi-MaP |

## 7. Corrections to earlier notes

1. **"No deep 2024 to 2025 version" of conditional clustering** (2026-11-27 note, §3, last row): partly wrong. Deep clustering independent of a given partition exists as fair clustering (DFC, CVPR 2020; FCMI, CVPR 2023). Deep multiple clustering with CLIP is active: Multi-MaP (CVPR 2024), Multi-Sub (NeurIPS 2024), X-Cluster (CVPR 2026 Findings). What is still missing is a deep method that takes a given partition and needs no text.
2. **"No study of texture statistics on CLIP or DINO tokens"** (same note): still true for Gram matrices. StyLIP uses mean and standard deviation of frozen CLIP intermediate tokens as style statistics, without testing them. That is a partial precedent.
3. **Cui et al. "not assessed"** (T1 and synthesis): it is the direct prior of the residual arm (project out, re-cluster), and ClustPy implements it as Orth1 and Orth2.
4. **AugDMC:** listings disagree on two versus three datasets; venue unconfirmed.

**Unverified:**
- StyLIP's venue.
- X-Cluster's code release.
- Multi-MaP's CLIP variant.
- That COCO-4c IDs join to COCO captions.
- DFC's own setup (seen only through later papers).
- Rui et al. and Qi and Davidson beyond their abstracts.
- Stanford40's size.
- Naseer et al.'s shuffle numbers.

## 8. Open, with the search that would settle each

- **Whether X-Cluster finds a "style" criterion on art.** Run it on ArtELingo, or search "X-Cluster painting style criterion".
- **The FCMI input format.** Read the code at github.com/PengxinZeng/2023-CVPR-FCMI.
- **A multiple-clustering survey from 2024 to 2026.** Search "Multiple clusterings: recent advances and perspectives".
- **Clevr-4 on CLIP ViT-B/32 intermediate layers.** A cheap local run that would test the patch-statistics arm directly.

## 9. References

- Multi-MaP. Yao, Qian, Hu, CVPR 2024. https://arxiv.org/html/2404.15655v1 ; code https://github.com/Alexander-Yao/Multi-MaP
- Multi-Sub. Yao, Qian, Hu, NeurIPS 2024. https://arxiv.org/html/2411.03978v1 ; https://proceedings.neurips.cc/paper_files/paper/2024/hash/96b8167534ef3cc30c230bbeb55a524d-Abstract.html
- IC|TC. Kwon et al., ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/05b69cc4c8ff6e24c5de1ecd27223d37-Abstract-Conference.html ; https://github.com/sehyunkwon/ICTC
- X-Cluster / OpenSMC. Liu et al., CVPR 2026 Findings. https://arxiv.org/html/2410.05217v5 ; https://openaccess.thecvf.com/content/CVPR2026F/html/Liu_Organizing_Unstructured_Image_Collections_using_Natural_Language_CVPRF_2026_paper.html
- AugDMC. Yao et al., preprint. https://arxiv.org/abs/2306.13023
- TGAICC. Stephan et al., RepL4NLP 2024. https://arxiv.org/html/2406.18589v1 ; https://github.com/AndSt/alternative_image_clustering
- ENRC. Miklautz et al., AAAI 2020. https://ojs.aaai.org/index.php/AAAI/article/view/5961
- ACe/DeC. Miklautz et al., IJCAI 2021 (context only). https://ijcai.org/proceedings/2021/389
- ClustPy, alternative package (Cui et al. ICDM 2007 Orth1/Orth2; NrKmeans KDD 2018; AutoNR SDM 2022). https://clustpy.readthedocs.io/en/latest/clustpy.alternative.html
- Qi and Davidson, KDD 2009. https://web.cs.ucdavis.edu/~davidson/Publications/KDD2009.pdf (abstract via search)
- COBS. Van Craenendonck and Blockeel, preprint (journal version per listing https://link.springer.com/article/10.1007/s10994-017-5643-7). https://ar5iv.arxiv.org/html/1609.07272
- DFC. Li, Zhao, Liu, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Deep_Fair_Clustering_for_Visual_Learning_CVPR_2020_paper.html
- FCMI. Zeng et al., CVPR 2023. https://arxiv.org/abs/2209.12396 ; https://openaccess.thecvf.com/content/CVPR2023/papers/Zeng_Deep_Fair_Clustering_via_Maximizing_and_Minimizing_Mutual_Information_Theory_CVPR_2023_paper.pdf
- Conditional contrastive learning. Tsai et al., preprint https://arxiv.org/pdf/2106.02866 ; CCL-K, ICLR 2022, https://arxiv.org/abs/2202.05458
- LEACE. Belrose et al., NeurIPS 2023. https://arxiv.org/pdf/2306.03819
- Kernelized concept erasure. Ravfogel et al., EMNLP 2022. https://aclanthology.org/2022.emnlp-main.405
- Splice. Tumanyan et al., CVPR 2022. https://ar5iv.labs.arxiv.org/html/2201.00424 ; TOG extension https://arxiv.org/abs/2311.12193
- StyLIP. Bose et al., arXiv 2302.09251 (venue unverified). https://ar5iv.labs.arxiv.org/html/2302.09251
- No Representation Rules Them All (Clevr-4). Vaze, Vedaldi, Zisserman, NeurIPS 2023. https://arxiv.org/html/2311.17055 ; data https://www.robots.ox.ac.uk/~vgg/data/clevr4/
- Intriguing Properties of ViTs. Naseer et al., NeurIPS 2021. https://ar5iv.labs.arxiv.org/html/2105.10497
- Vision Transformers Need Registers. Darcet et al., ICLR 2024 (snippet only). https://arxiv.org/pdf/2309.16588
- Archetypal Style Analysis. Wynen, Schmid, Mairal, NeurIPS 2018 (abstract only). https://arxiv.org/abs/1805.11155
- Relevance feedback. Rui, Huang, Ortega, Mehrotra, IEEE TCSVT 1998 (abstract only). https://ieeexplore.ieee.org/document/718510
- SemArt. Garcia and Vogiatzis, ECCV Workshops 2018. https://arxiv.org/abs/1810.09617 ; https://research.aston.ac.uk/en/datasets/semart-dataset/
- DeepFashion-MultiModal / Text2Human. https://arxiv.org/pdf/2205.15996
- CUB captions. Reed et al. 2016, via https://www.tensorflow.org/datasets/catalog/caltech_birds2011 (snippet)
- Toronto zero-shot style page. Pournajafian, not peer reviewed. https://www.cs.toronto.edu/~ary/zero-shot-style/
