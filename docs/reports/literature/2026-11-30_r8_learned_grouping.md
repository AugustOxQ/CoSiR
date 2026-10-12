# Round 8: the learned grouping across content, novelty, positive pairs, comparators and FCMI in practice

> Literature check for round 8 (step 3), r8 decide 2 chat, 2026-10-12; Sonnet, standard depth (the 2026-11-28 note
> already covers the area, and the grouping is one part of the AFF paper, not its central claim). Builds on
> `2026-11-28_design_l_novelty.md` and `2026-11-27_style_sources.md`. Checked by the controller (2026-10-12 18:26):
> MCL's arXiv abstract confirms "the negative samples of an anchor are exclusively selected from the cluster it was
> assigned to in preceding stages", and its comments field says ECCV 2024 camera-ready.

**Budget:** 19 of 20 searches and 9 of 10 page reads were used. The agent read the two earlier notes first and searched
only for what they do not cover. Every page went through a summariser. Two PDF fetches failed, so the HTML versions
were used. No source was read in full. "Abstract" means abstract or listing only. "Snippet" means search result only.

## Goal

1. Is (ii) new?
2. How do others form positive pairs for an unlabelled style?
3. Is there a stronger comparator for the grouping step in 2025–2026?
4. Does the FCMI code take precomputed features, and does it cope with 64 protected groups?

## Answers

**Q1. Novelty of (ii): partly new, and closer to "not new as a mechanism" (medium confidence).**
- **Closest prior:** Multistage Contrastive Learning (MCL), Zhang et al., ECCV 2024. It trains in stages. After each
  stage it runs k-means on the learned representation. The next stage draws each anchor's negatives only from the
  anchor's own cluster in earlier stages. Features learned earlier then cannot separate the pairs, so the model must
  learn new ones. This is our idea with a self-made partition in place of a given one.
  - Read: arXiv v3 HTML page.
  - Positives are augmentations (unimodal) or image-text pairs (CLIP).
  - The CLIP experiment fine-tunes the last 6 image blocks on CC12M. It does not use frozen features.
  - Data: Trifeature, CIFAR-MNIST, CelebA, STL-10, MMVP. No art.
  - The appendix reports AMI between stage groupings of 0.24 to 0.33, the same kind of "does it cut across" check we
    use.
- **Other priors:** attribute-matched negatives exist in fair contrastive learning. Shen et al. (2021) is the closest:
  it pushes apart instances that share a protected attribute (preprint; abstract only). A search snippet described
  FAREContrast as pairing each sample only with negatives of similar protected-attribute status (not verified). Tsai et
  al. (2021) and CCL-K (ICLR 2022) are in the earlier note.
- **What separates us:**
  - (a) The partition is given from outside (the content grouping of CLIP image features), not self-made.
  - (b) The head is small and sits on frozen CLIP features.
  - (c) The target is style in art.
  - (d) The output is a grouping used as a condition for AFF.
  - Points (b) to (d) have no prior that was found.
- **Advice (reader-inferred):** cite MCL as the mechanism. Do not claim the negative-sampling idea. Consider MCL-style
  iteration (k-means, same-cluster negatives, repeat) as a named comparator.

**Q2. Positive pairs when style has no labels (medium confidence on the list, low on which works best).** No source
reports an option shown to keep style and drop content on art.

| Option | Source | Notes |
|---|---|---|
| Two augmented views of one image, with colour jitter and blur removed. Flips, rotation and resize are kept as style-safe. | CSD, ECCV 2024 (arXiv HTML) | The page does not confirm random crops. The paper does not measure leakage. Its supervised term uses shared style tags from captions, which is not label-free. |
| Standard SimCLR augmentations as positives | MCL; von Kügelgen et al., NeurIPS 2021 | Von Kügelgen et al. show that such augmentations isolate content (what the views share) and discard style (what they change). Default positives would therefore teach content, the opposite of our aim (reader-inferred application; abstract only). |
| Random patches of one image (1/8 to 1/4 of the side) as same-texture positives | Swapping Autoencoder, Park et al., NeurIPS 2020 (search results) | Label-free. It is a GAN discriminator, not InfoNCE. |
| Same style, different content, from synthetic generation | ALADIN-NST, GOYA, MegaStyle (from the 2026-11-27 note) | Needs a generator or about 1M images. Not for stored features. |
| Style image, same content in style A, same content in style B (triplet) | Kotovenko et al., ICCV 2019 (abstract and supplement snippet) | Needs style transfer. |
| Neighbours outside the query's content group, or same artist | Nothing found | Neighbour positives are our reasoning only. Same-artist positives were not found as a label-free choice. Artist leakage is a known risk: "Style or Signature?", arXiv 2608.14435, 2026 (snippet). |

One practical point (reader-inferred): on stored CLIP features, pixel augmentations are not available. Positives
would need a new extraction of cropped or flipped views, or must come from feature-space neighbours.

**Q3. Strongest comparator, 2025–2026: nothing newer found (medium-low confidence, because searches return snippets).**
- No 2025–2026 deep method was found that takes a given partition and returns a non-redundant grouping on pretrained
  features without text.
- Newer or missed items:
  - DDMC, dual-disentangled deep multiple clustering, arXiv 2402.05310, 2024 (snippet).
  - DivClust, Metaxas et al., CVPR 2023 (snippet).
  - Deep fair clustering with continuous attributes, Zhang et al., arXiv 2403.16201, 2024 (snippet).
  - Fair visual clustering with a transformer, Nguyen et al., arXiv 2304.07408, 2023 (snippet).
- **Advice:** keep the earlier list (X-Cluster, Orth1/Orth2, NrKmeans, ENRC, FCMI, DFC, Multi-MaP). Add MCL-style
  iteration as the closest contrastive comparator.

**Q4. FCMI in practice (medium-high on the code facts; low on behaviour at 64 groups).**
- **Input:** the README does not say. Its datasets are Color Reverse MNIST, Office-31, MTFL, MNIST-USPS and HAR. The
  paper says MTFL and Office-31 use ResNet50 features as input (ar5iv summary). The code has a class for flat vectors,
  `NetFCN`, with an MLP encoder (input_dim to 1024 to 1024 to 512). The image class has hard-coded 28×28 conv sizes.
  So precomputed vectors are supported in practice but not documented. We would add a loader under `DataSetMaster/`.
- **Pretrained models:** no run of FCMI or another deep fair-clustering method on frozen foundation-model features was
  found. The closest is a CLIP-based clustering used to infer sensitive groups for a downstream classifier (arXiv
  2403.10624, snippet). It is not fair clustering.
- **Many groups:** the paper says it extends to any number of groups. Its decoder is one branch per group. The code
  makes the encoder and decoder per group only when `GroupWiseEncoder` or `GroupWiseDecoder` is set; their defaults
  were not seen. One discriminator MLP (rep_dim to 512 to 256 to 1) is always built per group: with 64 groups, 64
  small MLPs, cheap on 512-d features. Group IDs must be 0..G−1. The balance loss uses a clusters × groups matrix. The
  soft-assignment scale 700 is hard-coded.
- **Evidence for 64 groups:** none. The paper's largest tested case is HAR ("three or more" groups, exact count
  unverified). With 64 groups and small batches, many groups will have few samples per batch. Treat the first run as
  a test.

## Sources

| Source | Venue, year | What it shows (question) | Read |
|---|---|---|---|
| Zhang, Lan, Qu, Cheng, Feng, Hooi, "Learning the Unlearned: Mitigating Feature Suppression in Contrastive Learning" https://arxiv.org/html/2402.11816v3 ; venue page https://eccv2024.ecva.net/virtual/2024/poster/822 | ECCV 2024 (code github.com/MajorDavidZhang/MCL) | Same-cluster negatives, k-means per stage, CLIP fine-tune (Q1, Q2, Q3) | HTML summary; abstract checked by the controller |
| Shen, Han, Cohn, Baldwin, Frermann, "Contrastive Learning for Fair Representations" https://arxiv.org/abs/2109.10645 | preprint, 2021 | Negatives tied to a protected attribute (Q1) | abstract |
| Tsai et al. https://arxiv.org/pdf/2106.02866 ; CCL-K https://arxiv.org/abs/2202.05458 | preprint 2021; ICLR 2022 | Conditional contrastive learning (Q1) | earlier note |
| Sundaram et al. (CSD), "Investigating Style Similarity in Diffusion Models" https://arxiv.org/html/2404.01292 | ECCV 2024 | Positive pairs and style-safe augmentations (Q2) | HTML summary |
| von Kügelgen et al., "Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style" https://arxiv.org/abs/2106.04619 | NeurIPS 2021 | Augmentation positives keep content (Q2) | abstract |
| Park et al., "Swapping Autoencoder for Deep Image Manipulation" https://arxiv.org/abs/2007.00653 | NeurIPS 2020 | Patch positives for texture (Q2) | search snippets |
| Kotovenko et al., "Content and Style Disentanglement for Artistic Style Transfer" https://openaccess.thecvf.com/content_ICCV_2019/papers/Kotovenko_Content_and_Style_Disentanglement_for_Artistic_Style_Transfer_ICCV_2019_paper.pdf | ICCV 2019 | Same-content, different-style triplets (Q2) | snippet |
| "Style or Signature? Artist-Disjoint Evaluation of Style Classification in Frozen Vision Embeddings" https://arxiv.org/pdf/2608.14435 | preprint, 2026 | Artist leakage in frozen embeddings (Q2) | snippet |
| Zeng et al., FCMI https://arxiv.org/abs/2209.12396 (read via https://ar5iv.labs.arxiv.org/html/2209.12396) | CVPR 2023 | Per-group decoders; ResNet50 features for MTFL and Office-31 (Q4) | HTML summary |
| FCMI code https://github.com/PengxinZeng/2023-CVPR-FCMI (README; `networkExplicitDecoupleShuffle.py`) | official code | `NetFCN` for vectors; per-group lists (Q4) | summaries of the README and file |
| DDMC https://arxiv.org/abs/2402.05310 ; DivClust https://openaccess.thecvf.com/content/CVPR2023/html/Metaxas_DivClust_Controlling_Diversity_in_Deep_Clustering_CVPR_2023_paper.html ; Zhang et al. https://arxiv.org/pdf/2403.16201 ; Nguyen et al. https://arxiv.org/abs/2304.07408 | 2023–2024; venues unverified except DivClust (CVPR 2023) | Related alternative or fair clustering (Q3) | snippets |
| "Leveraging CLIP for Inferring Sensitive Information and Improving Model Fairness" https://arxiv.org/abs/2403.10624 | preprint, 2024 | CLIP clusters used for fairness, not fair clustering (Q4) | snippet |

## Open

- **MCL details:** how its first-stage clusters are formed for CLIP, and whether its code runs on stored features.
  Read the MCL repo.
- **FCMI defaults:** `main.py` and `QuickConfig/` were not read, so the defaults for `GroupWiseEncoder` and
  `GroupWiseDecoder` are unknown. Read those files.
- **FCMI group counts:** the group counts for HAR and Office-31 are unverified. Read the supplement.
- **Neighbour and same-artist positives:** no source found. Search "nearest-neighbour positives style" and
  "artist-level positives".
- **CSD crops:** whether CSD uses random crops. Read the ECCV PDF.
- **Q3 coverage:** one search per angle. Check the citing papers of FCMI and Multi-MaP on Semantic Scholar.
