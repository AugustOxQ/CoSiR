# Style sources that separate style from content and genre: literature check for design L (round 7)

> Literature check for round 7 (design L, from the style source), step 3 of the r7 decide chat, 2026-10-11; Sonnet, standard depth; builds on scan T4 (src/test/20261114_grouping_research/).

Budget: 18 searches and about 10 page reads, so the standard budget is spent. Mostly abstracts, HTML summaries and search snippets. Confidence per item is marked. "Preprint" means arXiv only. "Unverified" means the item rests on a search snippet or listing.

## 1. Results brief

1. **Best lead: remove a content direction from a style embedding, label-free.** StyleDecoupler (preprint, Jan 2026) projects a DINOv2-based content vector out of a CLIP/SigLIP style vector. The projection is partial: `s_pure = Norm(s − α·proj_c(s))` with `α = max(0, 1 − sim(s, c))`.
   - Evidence: WikiArt mAP@1 (artist as the style proxy) is 63.6 against CLIP ViT-L 58.4 and CSD 63.1. It also leads on its own WeART benchmark.
   - Risk: it reports no direct measure of leftover content or genre. Its content vector needs GPT-4o text descriptions and a DINOv2 model aligned to CLIP on CC3M. No release link for weights was found.
   - A simpler, untested variant (reader-inferred) needs none of that. Take the CSD embedding, regress out or project out the CLIP-image and/or DINOv2 embedding, keep the residual, then run Leiden and AMI on it. It reuses features we already have.
2. **Subtract a content text embedding from the image embedding (InstantStyle).** Evidence is qualitative only: figures in a generation paper (preprint), no metrics, and manual tuning of the strength. In our setting the "content text" would be a generated or viewer caption. That makes the style source depend on the weak caption side, so rate it a cheap probe and not a source.
3. **Content-fixed synthetic training (ALADIN-NST, GOYA, MegaStyle).** These give the best content-removal evidence, but all of it comes from synthetic benchmarks.
   - WikiArt exposure: GOYA's prompts use the 27 WikiArt style names. MegaStyle's style pool contains about 80K WikiArt images (from the page summary).
   - Weights: GOYA has code only. MegaStyle states no release. Neither is a clean label-free source.
   - GOYA works on frozen CLIP ViT-B/32 features, our backbone, but retraining it needs about 1M Stable Diffusion images.
4. **Question 3, how separable are style and genre:** no published contingency, NMI or AMI between WikiArt style and genre was found. The nearest evidence is GOYA's Table 3. Tuning CLIP on genre lowers style accuracy from 51.2 to 35.0, so the two labels compete in the embedding. Confidence: low-medium. The label-side AMI(style, genre) on ArtELingo's own metadata costs nothing and should be computed first (§4).
5. **Question 5, novelty:** no prior work found that uses a label-free, content-de-confounded style grouping to steer an aspect from example supports. Confidence: medium-low (one search per angle). Closest work: GeneCIS (CVPR 2023) and CLAY (arXiv 2026, CVPR 2026 per the arXiv comments field), both with an explicit text condition. The comparator a reviewer will most likely name is a style embedding (CSD, or StyleDecoupler) with the supports' prototype, used as a plain retriever.

## 2. Candidate style sources

Cost assumes about 61k images on one RTX 3090. These are reader-inferred estimates; none was run.

| Name | Venue/year | Training data and WikiArt exposure | Weights | How it separates style from content | Evidence | Cost |
|---|---|---|---|---|---|---|
| CSD ViT-L (control, from T4) | ECCV 2024 as "Investigating Style Similarity in Diffusion Models" (see §6); arXiv 2404.01292 | LAION-Styles, CLIP init. WikiArt is evaluation only. | github.com/learn2phoenix/CSD (ViT-L) | Multi-label contrastive on style tags plus a self-supervised loss | WikiArt mAP@1 (artist proxy) 64.56 vs CLIP-L 59.4 (CSD paper). 63.1 vs 58.4 in StyleDecoupler's table. MegaStyle's synthetic benchmark: CSD 45.6, CLIP 9.3. Our own AMI: 0.34 style / 0.33 genre. | A few minutes of GPU, I/O bound |
| StyleDecoupler | preprint, arXiv 2601.17697, Jan 2026 (Jia, Zhang, Zhou) | Frozen CLIP-L or SigLIP; DINOv2-L aligned to CLIP on CC3M (the paper says 0.5M in one place, 2M in another). GPT-4o text descriptions. Evaluated on WikiArt and WeART. | Not found. Only the WeART dataset link is given (huggingface.co/datasets/wechat-prcap/weart). | Partial projection of a DINOv2-based content vector out of the style vector | WikiArt mAP@1: CLIP-L 58.4, CSD 63.1, SigLIP 58.5, DINOv2-L 40.0 (a snippet of its own table), StyleDecoupler-CLIP 63.6, StyleDecoupler-SigLIP 62.7. Clustering against artist labels: ACC 41.76 vs CLIP 36.09. Leakage is not measured. | Cheap for the projection. GPT-4o on 61k images is an API cost, or a local captioner takes hours. |
| MegaStyle-Encoder | preprint, arXiv 2604.08364, Apr 2026 | Fine-tuned from SigLIP so400m-384 (a ViT-L variant also reported) on 1.4M text-to-image images (170K style prompts × 400K content prompts). WikiArt supplies about 80K images to the style pool (page summary, unclear role). | None stated (project page only) | Same style, many contents, by construction | StyleRetrieval mAP@1: 87.3 (ViT-L), CSD 45.6, CLIP 9.3, SigLIP 10.4. Own synthetic benchmark. It says SigLIP matches are driven by semantic content. | If weights appear: SoViT-384 is several times slower than ViT-L, tens of minutes. |
| GOYA | ICMR 2023 ("Not only generative art"); J. Imaging 10(7), 2024 | Two projections on frozen CLIP, trained on 981k Stable Diffusion images. Prompts use the 27 WikiArt style names and 43,610 painting titles (origin not stated). WikiArt is evaluation only. The paper says SD's LAION data contains some WikiArt test paintings. | Code only, github.com/yankungou/GOYA | Contrastive content and style subspaces | Distance correlation (lower is better), Table 2: labels 0.269, GOYA 0.367, CLIP fine-tuned 0.460, DINO 0.518, ResNet50 0.635. Table 3 gives one accuracy pair per model: CLIP 71.56 genre / 51.23 style; GOYA 69.70 / 50.90. | Needs about 1M SD generations to retrain. Not cheap. |
| ALADIN / ALADIN-NST (from T4) | ICCV 2021 / preprint 2023 | Behance. No WikiArt. | ALADIN: weights in README | Separate content and style branches / NST pairs | Unchanged from T4 | Low |
| StyleShot style encoder | preprint, arXiv 2407.01414 (Gao et al., 2024) | StyleGallery, 5.7M pairs from sources including JourneyDB and WikiArt. **Exposure: yes.** Reference or ceiling only. | github.com/open-mmlab/StyleShot | "Decoupling training strategy" (details not read) | Style-transfer benchmark StyleBench; no retrieval numbers seen | n/a |
| StyleTokenizer | ECCV 2024 (Li et al.) | Style30k: 30k images, over 300 style categories, contrastive encoder. WikiArt use unknown. | github.com/alipay/style-tokenizer | Encoder trained to represent style and exclude content | Generation metrics only; retrieval unseen | Low (unverified) |
| DEADiff | CVPR 2024 per the arXiv listing (2403.06951); venue not confirmed here | Not read | Not checked | Q-Former conditioned on a text instruction, trained on same-style/different-semantics pairs | Generation only | n/a |
| CSGO | NeurIPS 2025; arXiv 2408.16766 | IMAGStyle, 210K synthetic content-style-stylised triplets | github.com/instantX-research/CSGO | Separate injection of content and style features | Generation only; no standalone descriptor | n/a |
| InstantStyle | preprint, arXiv 2404.02733 | None (training-free, on top of IP-Adapter) | github.com/instantX-research/InstantStyle | Text subtraction plus style-only attention layers | See §3 | n/a |
| DINOv2 / DINOv3 / SigLIP 2 as plain backbones | n/a | Web-scale, not disclosed | Public | None. Not style-specific. | DINOv2-L 40.0 vs CLIP-L 58.4 mAP@1 on WikiArt (StyleDecoupler's table). An unvetted demo page reports linear probes with DINOv2-B 57.5 style / 71.0 genre and CLIP-L 67.1 / 75.0 (unverified). No DINOv3 or SigLIP 2 WikiArt numbers found. A Toronto student poster (not peer reviewed) finds DINOv3 raw CLS style retrieval below 1% R@1, and that a trained style probe "is picking style while still leaking content". | Low |

No art-specific CLIP variant that is also label-free was found. Any CLIP fine-tuned on WikiArt styles is a ceiling only. GOYA's Table 3 gives one: CLIP fine-tuned on style reaches 63.02 style and 56.28 genre.

## 3. De-confounding methods (question 2)

| Method | Evidence found | Quality | What it does not show |
|---|---|---|---|
| Project out a content vector (StyleDecoupler) | Higher artist-proxy retrieval and clustering; qualitative maps | Preprint, no venue; baselines mostly its own table | No measure of leftover genre or subject |
| Subtract a content text embedding (InstantStyle) | Figs 13 and 14; "no longer any content leakage" at higher scale | Preprint; qualitative only; the paper says it needs manual strength tuning and assumes CLIP is additive | No numbers; judged by eye on generated images |
| SVD rectification of the style embedding with an LLM-built content subspace (CSD-VAR, arXiv 2507.13984) | Seen in search results only | Preprint, unverified | Generation, not retrieval |
| Orthogonal-complement subspaces (Parts-of-Speech-Grounded Subspaces, arXiv 2305.14053) | Seen in search results only | Unverified | Uses POS supervision |
| Gram or texture statistics | Gram statistics are in T4 and in our own result (VGG-19 Gram: AMI 0.14 / 0.19, ratio 0.77). No study found of Gram on CLIP or DINO patch tokens, nor of patch-shuffled inputs for art style. | Gap | Nothing found. A test of these would be new, not borrowed. |
| Content-fixed synthetic training | ALADIN-NST (T4); MegaStyle says semantic content drives SigLIP matches while its encoder resolves the pair | Synthetic evidence only | Real WikiArt style versus genre |
| Clustering conditioned on a partition, or non-redundant clustering | Kernel conditional clustering (KAIS 2019, Springer s10115-019-01334-5) and the iterative spectral method for alternative clustering (arXiv 1909.03441). Both are shallow methods. | Venue-listed, abstracts only | No deep 2024 to 2025 version found. T1 stays the reference. |
| HSIC independence penalty | HSIC-InfoGAN (arXiv 2208.03563), unsupervised disentanglement in a GAN | Preprint | Nothing on clustering or art |

One caution is a measured one. The Toronto poster (not peer reviewed) reports that a stronger style probe returns more wrong neighbours that share the query's object class. That is leakage that grows with probe capacity.

## 4. WikiArt style and genre separability (question 3) and screen precedent (question 4)

**Separability.**
- No paper reports a contingency, NMI or AMI between WikiArt style and genre. This is a gap, to our knowledge, within this search. Confidence: medium-low.
- Probe results (all label-supervised):
  - GOYA Table 3: pre-trained CLIP reaches 71.56 genre / 51.23 style. Fine-tuned on genre it gives 80.43 / 34.98. Fine-tuned on style it gives 56.28 / 63.02.
  - Bianco et al., "Multitask Painting Categorization" (arXiv 1812.08052; Expert Systems with Applications 2019 per the search listing): 56.5% artist, 57.2% style, 63.6% genre on about 100K WikiArt paintings.
  - Ashton (preprint, in T4): artist-disjoint kNN drops from 0.869 to 0.766.
- MegaStyle argues WikiArt "categorizes styles by artist names", so labels are inconsistent within a style (preprint, Section 5).
- **Reading (reader-inferred):** the two labels overlap in meaning. Examples would be Abstract Expressionism with the abstract genre, and Ukiyo-e with its typical subjects. Tuning for one costs the other, so a perfect style grouping will itself show a nonzero genre AMI. Compute AMI(style, genre) and the pair ratio of the true labels from ArtELingo's metadata. Judge the told-ceiling check and the screen's pair ratio against that number, not against a ratio of 1. This costs nothing: it uses the labels only and trains no encoder.

**Precedent for the screen's measure.**
- GOYA (Table 2) measures separation as distance correlation between the content and style spaces. Its Figs 5 and 6 show confusion matrices of genre in the content space and of style in the style space. It is the closest published two-label evaluation of an embedding on WikiArt.
- Huckle, Garcia and Nakashima (ECCV Workshops 2020) cluster style embeddings and score them by AMI against WikiArt style labels, best 0.191, which they read as near random. This is from a search snippet; the PDF was too large to fetch (unverified). It is a cited form for AMI against style labels. Our CSD value of 0.34 can be set beside it with caution.
- "Substance or Style: What Does Your Image Embedding Know?" (arXiv 2307.05610) probes embeddings for both. Title only seen, so check it before citing (unverified).
- StyleDecoupler reports clustering ACC/ARI against artist labels. Our own stage report used AMI against style (0.227 against 0.318).
- No work found using AMI with a style label set against a genre label set for the same embedding. The screen's pair ratio would be new as a form.

## 5. Novelty and strongest comparator (question 5)

- **Closest prior work.**
  - GeneCIS (Vaze, Carion, Misra, CVPR 2023): conditional image similarity where the condition is given as text, with attribute and object focus/change tasks. CLIP-based baselines struggle.
  - CLAY (arXiv 2604.11539, comments field says CVPR 2026): text-conditional similarity in a frozen VLM space, without training. It does not take example images.
  - Conditional Similarity Networks, SCE-Net and DiscoverNet (in the earlier notes) have no art or example supports.
  - Collomosse et al., "Sketching with Style" (ICCV 2017): style embeddings that ignore structure, with the style set by reference images.
  - MosAIc (Hamilton et al.): conditional retrieval by a metadata filter.
  - ALADIN and StyleBabel: style search by image or text.
- **Not found:** a method that reads the aspect from support examples, uses label-free groupings, and de-confounds the style one. Confidence: medium-low (one search per angle).
- **Strongest comparators a reviewer would name** (reader-inferred):
  1. A plain style embedding (CSD, StyleDecoupler or MegaStyle-Encoder) with a prototype from the supports, used as a retriever without any grouping.
  2. A GeneCIS-style CLIP baseline that combines the supports' prototype with the query.
  3. As a ceiling, a CLIP fine-tuned on WikiArt style labels (63.02 style in GOYA's table), flagged as leaking.
- DIOR (in the earlier review, row 109) loses to plain CLIP on art style, which supports needing a style-specific source.

## 6. Corrections and unverified items

**Corrections to the earlier scans.**
1. **CSD venue.** T4, the synthesis and the CVPR review call it an "arXiv preprint, venue unconfirmed". The ECVA listing (www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08294.pdf) and search metadata give ECCV 2024, titled "Investigating Style Similarity in Diffusion Models". That PDF was not opened, so confirm the title and the table numbers before citing. The numbers may differ from the arXiv v1 ones.
2. **CSD against CLIP on WikiArt: three sources disagree.**
   - CSD paper (via T4): +5.2 mAP@1 (64.56 vs 59.4).
   - StyleDecoupler's table: +4.7 (63.1 vs 58.4).
   - DIOR (in the CVPR review row): CSD 58.2 below CLIP-L 59.3.
   The sets and splits differ, so do not say CSD beats CLIP by a fixed margin.
3. **GOYA, previously "unclear".**
   - WikiArt is used for evaluation only.
   - The training prompts use WikiArt's 27 style names. The label vocabulary is therefore in the training signal; no images or per-image labels are.
   - The painting titles' origin is not stated.
   - The paper says SD's LAION data contains some WikiArt test paintings.
   - The paper does not say which embedding gives the 69.70 / 50.90 pair.
   - Its text says GOYA is "less than 1%" below CLIP in genre and style, but Table 3 shows 1.86 points on genre.
   - No weights are mentioned; code is at github.com/yankungou/GOYA.
4. **Research gap in T4.** T4 found no work on the style-versus-genre balance of a style descriptor. That still holds. StyleDecoupler and MegaStyle score artist or synthetic-style retrieval, not genre.

**Unverified or not read.**
- StyleDecoupler: weights and code link; the embedding dimension; the table layout (a summariser read it).
- MegaStyle: weights, and whether its WikiArt images are used in the encoder's training or only to build prompts.
- StyleShot, StyleTokenizer, DEADiff and CSGO: abstract level only; WikiArt exposure of StyleTokenizer and DEADiff unknown.
- IP-Adapter style variants: not searched.
- The Huckle AMI value, "Substance or Style" and CSD-VAR: snippets or titles only.
- The ArtSleuth demo numbers and the Toronto poster: not peer reviewed.
- Costs: estimates, no run made.

**Open (searches that would settle them, over this budget).**
- Open the ECVA CSD paper and read arXiv 2307.05610 and the Huckle paper.
- Look for a StyleDecoupler or MegaStyle release.
- Search "Gram matrix on vision transformer tokens, style retrieval".
- Compute the style-genre AMI from the metadata (being done in the round 7 facts brief, `src/test/20261126_design_l_style_source/step3_facts.md`).

## 7. References

Venue = peer-reviewed venue as listed; preprint = arXiv only.

- StyleDecoupler. Jia, Zhang, Zhou, 2026, preprint. https://arxiv.org/abs/2601.17697 (HTML https://arxiv.org/html/2601.17697v1). Q1, Q2.
- MegaStyle. Gao et al., 2026, preprint. https://arxiv.org/abs/2604.08364 (HTML https://arxiv.org/html/2604.08364v2). Q1, Q3.
- GOYA, J. Imaging 10(7), 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11278509/. ICMR 2023 version: https://arxiv.org/abs/2304.10278. Q1, Q3, Q4.
- InstantStyle. Wang et al., 2024, preprint. https://arxiv.org/abs/2404.02733. Q2.
- StyleShot. Gao et al., 2024, preprint. https://arxiv.org/abs/2407.01414. Q1.
- StyleTokenizer. Li et al., ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04025.pdf. Q1.
- DEADiff. 2024. https://arxiv.org/html/2403.06951v2. Q1.
- CSGO. NeurIPS 2025. https://arxiv.org/abs/2408.16766. Q1.
- CSD, "Investigating Style Similarity in Diffusion Models", ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08294.pdf (arXiv https://arxiv.org/abs/2404.01292). Q1, Corrections.
- CSD-VAR. 2025, preprint. https://arxiv.org/abs/2507.13984. Q2.
- Parts of Speech-Grounded Subspaces in VLMs. https://arxiv.org/abs/2305.14053. Q2 (unverified).
- Kernel conditional clustering. KAIS 2019. https://link.springer.com/article/10.1007/s10115-019-01334-5. Q2.
- Iterative spectral method for alternative clustering. https://arxiv.org/abs/1909.03441. Q2.
- HSIC-InfoGAN. 2022, preprint. https://arxiv.org/abs/2208.03563. Q2.
- Multitask Painting Categorization by Deep Multibranch Neural Network. Bianco et al., ESWA 2019 (listing). https://arxiv.org/abs/1812.08052. Q3.
- Demographic Influences on Contemporary Art with Unsupervised Style Embeddings. Huckle, Garcia, Nakashima, ECCV Workshops 2020. https://arxiv.org/abs/2009.14545. Q4 (snippet only).
- Substance or Style: What Does Your Image Embedding Know? 2023. https://arxiv.org/abs/2307.05610. Q4 (title only).
- Probing vision transformers across abstraction and style. Poster, not peer reviewed. https://www.cs.toronto.edu/~ary/probing-poster/. Q1, Q2.
- GeneCIS. Vaze, Carion, Misra, CVPR 2023. https://arxiv.org/html/2306.07969v1. Q5.
- CLAY. Lim et al., preprint 2026 (CVPR 2026 per comments). https://arxiv.org/abs/2604.11539. Q5.
- Sketching With Style. Collomosse et al., ICCV 2017. https://openaccess.thecvf.com/content_iccv_2017/html/Collomosse_Sketching_With_Style_ICCV_2017_paper.html. Q5.
- Style or Signature? (in T4), preprint 2026. https://arxiv.org/abs/2608.14435. Q3.
