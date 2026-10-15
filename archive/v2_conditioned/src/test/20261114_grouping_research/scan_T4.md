# Three-way scan for T4 (style sources not trained on WikiArt style labels; backbone candidates)

Thread T4 of `research_brief.md` §7. Scan date 2026-10-05. Mode: ARS deep-research three-way-scan (bibliography + source verification).

**Read scope.** Every item was opened (arXiv abs or HTML, ACL Anthology, BMVA archive, ICLR page, GitHub README, Semantic Scholar API record) and title, authors, venue, year and id were confirmed from the opened page unless a line says otherwise. Body text was reached only through the WebFetch summariser (arXiv HTML or ar5iv pages), never as raw full text; two PDFs came back binary and were not readable. So the read scope is `abstract_only` for most items and `html_section_summary` (summariser answers to targeted questions, with section locators) for CSD, ALADIN, ALADIN-NST, Johnson, ArtEmis, Long-CLIP and ShareGPT4V. Method weaknesses are "not assessed (read scope: abstract_only)" where only abstracts were read; elsewhere they are labelled author-acknowledged (with locator) or reader-inferred. Retrieved text was treated as data. Absence claims are bounded: "to our knowledge, within this search".

**AI disclosure.** This scan was produced by a Claude agent (Sonnet 5.5) with WebSearch and WebFetch. A human should read CSD (§4 to §6), ALADIN-NST (§3) and ArtEmis (dataset statistics) in full before any number here is cited in a paper.

**Queries run (WebSearch, 11):** ALADIN BAM-FG training data; ArtELingo EMNLP 2022; Gram matrix features WikiArt style classification; self-supervised style representation without style labels (content-style disentangled); Karayev Recognizing Image Style; SemArt; ArtEmis captions and art style words; CLIP zero-shot WikiArt style and content bias; GOYA; Gatys CVPR 2016; Geirhos texture bias; text-only art style prediction. Plus about 25 WebFetch opens of the pages named below.

**Sources searched:** arXiv (abs, HTML, ar5iv), ACL Anthology, BMVA archive, ICLR virtual site, GitHub READMEs (gray literature: CSD, ALADIN), Semantic Scholar API, general web search. **Not searched:** CVF pages (HTTP 403 on fetch; used only as search-result listings), IEEE Xplore, MDPI and PMC pages (403 or captcha), Google Scholar, OpenReview, Hugging Face model cards (so weight availability for GOYA and ALADIN-NST is unchecked), art-history literature on style in verbal description, non-English caption tokenisation studies.

---

## Image Style Transfer Using Convolutional Neural Networks (Gram-matrix style statistics)
Source: CVPR 2016 | Year: 2016 | Link: https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html (the CVF page returned 403 on fetch; title, authors, venue, year and DOI 10.1109/CVPR.2016.265 confirmed from the Semantic Scholar API record, DBLP key conf/cvpr/GatysEB16, and search-result listings)
- WHY: Content and style of a natural image can be separated in a CNN's representation, which allows recombination (search-result abstract text).
- HOW: Style is the set of feature correlations (Gram matrices) across layers of an ImageNet-trained VGG; content is the activations at a deeper layer. Authors: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge.
- WHAT: Image synthesis, not a recognition benchmark. The paper itself does not measure how well Gram statistics recover art-style labels. That evidence comes from Johnson (below).
  - Method weaknesses: not assessed (read scope: abstract_only; the body was not read).
- Training data: ImageNet-trained VGG, no style labels | WikiArt style labels: no | WikiArt images: unclear for the ImageNet network; ImageNet is not WikiArt (reader-inferred; ImageNet provenance not checked)

## Neural Style Representations and the Large-Scale Classification of Artistic Style
Source: Proceedings of the Future Technologies Conference (per arXiv page) | Year: 2017 (arXiv v1 2016-11) | Link: https://arxiv.org/abs/1611.05368
- WHY: Test whether the neural-style (Gram) representation works as a feature for classifying painting style.
- HOW: Gram matrices of ImageNet VGG-19 layers on WikiArt paintings, fed to a random forest per layer, compared with fine-tuned ResNet-50 and a baseline CNN (HTML, §4, via summariser).
- WHAT: Random forest on one Gram layer (ReLU3_1) reached 33.46% top-1, a fine-tuned ResNet-50 36.99%, a CNN from scratch 27.47%; a linear classifier on the full style representation reached only 13.21%; deeper layers (ReLU4_1, ReLU5_1) were worse (HTML, §4). The summariser reported "70 style categories", which does not match WikiArt's 27 styles used elsewhere in this scan; treat the class count and absolute numbers as unconfirmed until the table is read. Author conclusion (§4): the representation is competitive, but fine-tuning remains better.
  - Method weaknesses: reader-inferred: the classifier on top is trained with WikiArt style labels, so the 33.46% measures how linearly or non-linearly separable the labels are in Gram space, not label-free recovery. It is evidence that Gram statistics carry style information, with no measurement of genre or subject content in the same space.
- Training data: features from ImageNet VGG-19 (no labels from WikiArt); the classifier uses WikiArt labels | WikiArt style labels: no for the representation, yes for the probe | WikiArt images: not in the representation (HTML summary)

## ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness
Source: ICLR 2019 | Year: 2019 | Link: https://iclr.cc/virtual/2019/poster/697
- WHY: Whether ImageNet CNNs rely on shape or texture.
- HOW: Psychophysics on cue-conflict images (about 50,000 trials, 97 observers) and Stylized-ImageNet training. Authors: Geirhos, Rubisch, Michaelis, Bethge, Wichmann, Brendel.
- WHAT: ImageNet-trained CNNs are strongly biased towards texture. For us this supports the idea that an ImageNet network's lower-layer statistics are texture-heavy, hence a plausible style source; it says nothing about art style as such (reader-inferred bridge).
  - Method weaknesses: not assessed (read scope: abstract_only).
- Training data: ImageNet | WikiArt style labels: no | WikiArt images: not applicable (no art data; reader-inferred)

## Recognizing Image Style (Karayev et al.)
Source: BMVC 2014 | Year: 2014 | Link: https://bmva-archive.org.uk/bmvc/2014/papers/paper121/index.html
- WHY: Image style (composition, colour, technique) is a recognisable property worth predicting.
- HOW: Compare colour histograms, low-level and deep features (ImageNet-trained multi-layer networks) as inputs to style classifiers on Flickr Style (80K photos) and Wikipaintings (85K paintings with style and genre labels). Authors: Karayev, Trentacoste, Han, Agarwala, Darrell, Hertzmann, Winnemoeller.
- WHAT: Abstract: features learned in a multi-layer network generally perform best, even when trained with object-class labels rather than style labels. The HTML summary also reports the authors' hypothesis that "style is content-dependent" (§6.1), which supports our worry that a content-trained network entangles the two; the summariser's AP numbers were internally inconsistent and are not used.
  - Method weaknesses: reader-inferred: classifiers are trained on Wikipaintings style labels, so this is a label-supervised probe and the Wikipaintings labels are the same family as our evaluation taxonomy.
- Training data: ImageNet features, probes trained on Wikipaintings style and genre labels | WikiArt style labels: yes in the probes (not a candidate source) | WikiArt images: yes (Wikipaintings is the same site family; reader-inferred)

## Measuring Style Similarity in Diffusion Models (CSD)
Source: arXiv preprint (cs.CV); no venue shown on the abstract page | Year: 2024 | Link: https://arxiv.org/abs/2404.01292
- WHY: Text-to-image models copy artists' styles; there is no good style descriptor to detect it.
- HOW: ViT backbones initialised from CLIP, trained on LAION-Styles (511,921 images, 3,840 style tags, curated from LAION-Aesthetics) with L = L_MCL + λ L_SSL: a multi-label contrastive loss on style tags plus a self-supervised loss (HTML §4, §5). Authors: Somepalli, Gupta, Gupta, Palta, Goldblum, Geiping, Shrivastava, Goldstein. Code: https://github.com/learn2phoenix/CSD (MIT licence, ViT-L; README, gray literature).
- WHAT: Zero-shot on WikiArt (80,096 images; artist used as the style proxy, §6) CSD ViT-L reaches mAP@1 of 64.56 against CLIP ViT-L at 59.4 (HTML, Table 1, via summariser); untrained humans did worse than many feature extractors (§6.2). WikiArt is evaluation only.
  - Method weaknesses: reader-inferred. (1) The evaluation treats the artist as style, so a high score can come from artist signatures, not the movement labels we use; a style-versus-genre balance on WikiArt movements is not reported. (2) The backbone starts from CLIP, so any genre dominance of CLIP can persist. (3) LAION-Styles tags come from web text and are not a movement taxonomy. Author-acknowledged weaknesses were not located (read scope: html_section_summary).
- Training data: LAION-Styles, from LAION-Aesthetics | WikiArt style labels: no (WikiArt is zero-shot evaluation only, HTML §6) | WikiArt images: unclear (not in the training set by construction as described; LAION is a web crawl and overlap with WikiArt paintings was not checked in this search; CLIP initialisation data is not public)

## ALADIN: All Layer Adaptive Instance Normalization for Fine-grained Style Similarity
Source: ICCV 2021 | Year: 2021 | Link: https://arxiv.org/abs/2103.09776 (CVF PDF: https://openaccess.thecvf.com/content/ICCV2021/papers/Ruta_ALADIN_All_Layer_Adaptive_Instance_Normalization_for_Fine-Grained_Style_Similarity_ICCV_2021_paper.pdf)
- WHY: Style similarity search needs a descriptor that is fine-grained and not tied to coarse labels.
- HOW: Encoder-decoder with pooled AdaIN statistics from all encoder layers, trained with a reconstruction plus contrastive loss under weak supervision from project-group co-membership on BAM-FG (2.62M Behance images, 310,000 groupings). It "explicitly disentangles content and style" with separate branches (HTML §3). Authors: Ruta, Motiian, Faieta, Lin, Jin, Filipkowski, Gilbert, Collomosse. Released weights `aladinVIT_bamfg_64.47.pt` (README, gray literature; no licence stated).
- WHAT: 56.89% fine-grained IR-1 on BAM-FG against a prior 3.57%; human study precision 0.974 at rank 1 (HTML, via summariser). WikiArt is not mentioned in the paper.
  - Method weaknesses: reader-inferred: the groups are Behance project co-membership, so images from one project share subject and artist as well as style; the descriptor may encode project-specific content. No test on a movement taxonomy such as WikiArt's.
- Training data: BAM-FG (Behance); BAM coarse labels used for a separate benchmark | WikiArt style labels: no | WikiArt images: no (the paper does not use WikiArt; Behance is a different source; reader-inferred from the dataset descriptions)

## ALADIN-NST: Self-supervised disentangled representation learning of artistic style through Neural Style Transfer
Source: arXiv preprint, April 2023 (a Springer chapter with a matching title exists, DOI 10.1007/978-3-031-91572-7_1; its page was blocked, so the venue is unconfirmed) | Year: 2023 | Link: https://arxiv.org/abs/2304.05755
- WHY: Earlier style descriptors still carry content; train style to ignore content.
- HOW: Trains on synthetic stylised images from neural style transfer (NeAT, PAMA, SANet) built from the style subset of BBST-4M (Behance, 2M style images) so that style is constant while content varies (HTML §3, §4.1). Authors: Ruta, Canet Tarres, Black, Gilbert, Collomosse.
- WHAT: State of the art on explicitly disentangled metrics with less semantic content than prior models (abstract); tested on 40k synthetic stylised images (400 Behance styles x 100 Flickr contents), with content retrieval used as the entanglement measure (HTML §4.2, §4.4).
  - Method weaknesses: reader-inferred: the evaluation is on synthetic style-transfer images, so "style" means the transferred artwork's appearance; this may not match what separates Baroque from Impressionism, and it was not tested on real paintings in the pages read. Weight availability not checked.
- Training data: BBST-4M style subset plus NST-generated images | WikiArt style labels: no | WikiArt images: no (not mentioned in the datasets described; reader-inferred)

## StyleBabel: Artistic Style Tagging and Captioning
Source: arXiv March 2022; ECCV 2022 per the ALADIN README citation (gray, not confirmed on the paper's own page) | Year: 2022 | Link: https://arxiv.org/abs/2203.05321
- WHY: Natural-language descriptions of style are rare; style is usually only a label.
- HOW: 135K digital artworks annotated by art and design experts with style tags and captions (Grounded-Theory protocol), plus ALADIN-with-ViT models for tag generation, style captioning and text-to-style search. Authors: Ruta, Gilbert, Aggarwal, Marri, Kale, Briggs, Speed, Jin, Faieta, Filipkowski, Lin, Collomosse (abstract page).
- WHAT: State of the art in fine-grained style retrieval (abstract). The abstract does not state the artwork source.
  - Method weaknesses: not assessed (read scope: abstract_only).
- Training data: StyleBabel annotations on 135K artworks (source not stated on the page read) | WikiArt style labels: unclear | WikiArt images: unclear (abstract page does not say; the ALADIN family uses Behance, reader-inferred only)

## GOYA: Leveraging Generative Art for Content-Style Disentanglement
Source: Journal of Imaging 10(7):156 | Year: 2024 | Link: https://www.mdpi.com/2313-433X/10/7/156 (MDPI page 403; title, authors, venue and year confirmed from the Semantic Scholar API record for DOI 10.3390/jimaging10070156; method text from a search-result summary)
- WHY: Disentangling content and style of paintings; models trained on real paintings entangle them. Authors: Yankun Wu, Yuta Nakashima, Noa Garcia.
- HOW (search-result summary, not opened): extract CLIP image features, then train two transformation networks with contrastive learning on Stable Diffusion images generated from (content, style) prompts, giving separate content and style embeddings.
- WHAT: Reported to disentangle better than models trained on real paintings, and to help similarity retrieval and art classification (search-result summary). A companion paper, "Not Only Generative Art: Stable Diffusion for Content-Style Disentanglement in Art Analysis" (ACM, ICMR 2023 per the URL; page returned 403), was seen in search results only and carries no claims here.
  - Method weaknesses: not assessed (read scope: abstract_only, plus search summary).
- Training data: Stable Diffusion synthetic images from style and content prompts, on top of CLIP features | WikiArt style labels: unclear (the prompts use style names; whether they were taken from WikiArt's list was not seen) | WikiArt images: unclear (synthetic training set; WikiArt use for evaluation not confirmed)

## Style or Signature? Artist-Disjoint Evaluation of Style Classification in Frozen Vision Embeddings
Source: VISART VIII workshop at ECCV 2026 (per arXiv page) | Year: 2026 | Link: https://arxiv.org/abs/2608.14435
- WHY: Frozen embeddings classify movements well, but this may be artist recognition.
- HOW: 5-NN on frozen CLIP ViT-B/32, CLIP ViT-L/14, DINOv2 ViT-B/14 and ResNet-50 embeddings, 320 paintings from four movements, random versus artist-disjoint evaluation (HTML §4, via summariser). Author: Rory Ashton.
- WHAT: Accuracy 0.869 with random splits and 0.766 artist-disjoint; Surrealism fell about 20 points (0.513) while Impressionism (0.900) and Cubism (0.800) held (HTML §5 to §6). Author limitation (§7): movements are not matched for subject, so style stays entangled with content.
  - Method weaknesses: author-acknowledged (§7): style entangled with content. Reader-inferred: 320 paintings and four movements is a small, easy test, and kNN uses movement labels at test time.
- Training data: frozen encoders (CLIP, DINOv2, ImageNet ResNet); no training | WikiArt style labels: not used for training; used as kNN reference labels (reader-inferred) | WikiArt images: unclear (data source of the 320 paintings not checked)

## Have Large Vision-Language Models Mastered Art History?
Source: arXiv cs.CV | Year: 2024 | Link: https://arxiv.org/abs/2409.03521
- WHY: Can VLMs identify style, author and date of paintings? Authors: Strafforello, Soydaner, Willems, Maerten, De Winter.
- HOW: Zero-shot classification of WikiArt (27 styles) and JenAesthetics (11 periods) with CLIP (text prompts "the art style of the painting is [style]"), LLaVA and GPT-4o (HTML, via summariser).
- WHAT: WikiArt style top-1: CLIP 24.57%, LLaVA 28.32%, GPT-4o 53.42%; the best earlier CNN method 71.24% (HTML). The paper reports prompt-sensitivity and failure analyses. This is evidence about reading style from images through text prompts, not from captions of paintings.
  - Method weaknesses: not assessed (read scope: abstract_only plus HTML summary of one results table).
- Training data: pretrained VLMs, no training | WikiArt style labels: not used for training (zero-shot) | WikiArt images: unclear (CLIP, LLaVA and GPT-4o training sets are not fully public)

## Long-CLIP: Unlocking the Long-Text Capability of CLIP
Source: ECCV 2024 | Year: 2024 | Link: https://arxiv.org/abs/2403.15378
- WHY: CLIP's text encoder accepts 77 tokens, but its effective length is "merely 20 tokens" (HTML §3.1).
- HOW: Positional-embedding stretching with knowledge-preserving interpolation plus feature matching to CLIP's latent space; maximum input 248 tokens; fine-tuned on about 1M (long caption, image) pairs from ShareGPT4V (HTML §4.1, §4.3). ViT-B/16 and ViT-L/14. Authors: Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Jiaqi Wang.
- WHAT: About +20% on long-caption retrieval and +6% on COCO and Flickr30k (abstract). Author-acknowledged (§5): still has an input-length upper bound; only 1M long-text pairs were used.
  - Method weaknesses: author-acknowledged (§5): limited long-text training data. Reader-inferred: the gains are on long captions (ShareGPT4V captions average 826 characters, §2) and a distribution shift to short captions is possible; short-caption performance is only reported through COCO and Flickr30k, not art captions.
- Training data: ~1M ShareGPT4V pairs on top of OpenAI CLIP weights | WikiArt style labels: no | WikiArt images: unclear and small: ShareGPT4V's 100K GPT4-Vision caption set includes 500 WikiArt images (ShareGPT4V appendix A, below); the 1.2M PT set does not list WikiArt, and which subset Long-CLIP's 1M draws on was not checked

## ShareGPT4V: Improving Large Multi-Modal Models with Better Captions (provenance check for Long-CLIP)
Source: arXiv cs.CV | Year: 2023 | Link: https://arxiv.org/abs/2311.12793
- WHY, HOW, WHAT (abstract and appendix only): 1.2M descriptive captions seeded from 100K GPT4-Vision captions. Authors: Chen, Li, Dong, Zhang, He, Wang, Zhao, Lin. Appendix A (HTML, via summariser): the 100K seed has 50K COCO, 30K LCS, 20K SAM, 500 TextCaps, 500 WikiArt and 1K web images; the 1.2M PT set has 118K COCO, 570K SAM and 558K LLaVA-1.5 pretraining images.
  - Method weaknesses: not assessed (read scope: abstract_only for the method).
- Training data: as above | WikiArt style labels: no | WikiArt images: yes, 500 in the seed set; unknown overlap with our 36,518 evaluation paintings

## ArtEmis: Affective Language for Visual Art
Source: CVPR 2021 (CVF listing seen in search results; the CVF page itself returned 403 and the arXiv page does not state the venue) | Year: 2021 | Link: https://arxiv.org/abs/2101.07396
- WHY: Affective language about artworks is not captured by objective captioning datasets. Authors: Achlioptas, Ovsjanikov, Haydarov, Elhoseiny, Guibas.
- HOW: Annotators give a dominant emotion and a free-text explanation for WikiArt artworks; baseline neural speakers.
- WHAT: Caption statistics (ar5iv HTML via summariser; confirm in the paper): 439,121 explanations, average length 15.8 words against 10.5 for COCO and 9.6 for Conceptual Captions; vocabulary 36,347; over 20% use similes or metaphors; 83.5% contain sentimental language against 22.6% for COCO. The abstract page says 439K attributions on 81K artworks; the project site says 455K on 80K; ArtELingo cites 0.45M for 80K artworks. These counts differ by version, so quote the paper's number only after a check. The paper does not quantify how often explanations name an art style or movement (summariser answer, §unknown).
  - Method weaknesses: reader-inferred: the instruction asks for an emotional reaction and its reason, so captions are shaped to describe content and feeling; they say little about movement or technique, which limits text as a style source.
- Training data: not applicable (dataset) | WikiArt style labels: not used by the dataset (WikiArt's style label is metadata of the paintings) | WikiArt images: yes (the dataset is WikiArt images; this is our evaluation set)

## ArtELingo: A Million Emotion Annotations of WikiArt with Emphasis on Diversity over Language and Culture
Source: EMNLP 2022 (ACL Anthology 2022.emnlp-main.600, pp. 8770 to 8785) | Year: 2022 | Link: https://aclanthology.org/2022.emnlp-main.600/
- WHY: Emotional responses differ across languages and cultures. Authors: Mohamed, Abdelfattah, Alhuwaider, Li, Zhang, Church, Elhoseiny.
- HOW: Adds about 0.79M Arabic and Chinese and 4.8K Spanish annotations to ArtEmis's 0.45M English ones.
- WHAT: 51K artworks have five or more annotations in three languages; the paper lists 27 styles and analyses agreement by 10 genres (HTML, via summariser). It gives no caption length statistics (summariser answer).
  - Method weaknesses: not assessed (read scope: abstract_only plus HTML summary).
- Training data: not applicable (dataset) | WikiArt style labels: not used for training | WikiArt images: yes (our evaluation set)

## SemArt (How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval)
Source: ECCV 2018 workshops (arXiv v1 2018-10) | Year: 2018 | Link: https://arxiv.org/abs/1810.09617
- WHY: Art understanding beyond style classification. Authors: Noa Garcia, George Vogiatzis.
- HOW: 21,384 paintings from the Web Gallery of Art with expert comments and metadata (author, title, date, technique, type, school, timeframe); text-to-art retrieval benchmark.
- WHAT: Comments are longer than ArtEmis captions (about 70% have 100 words or less) and refer to content, technique, context and artistic movements (HTML, via summariser); "Type" has 10 genres and "School" 26 traditions. This is the one source here where expert text does name style-like information, but it is curator prose, not viewer captions.
  - Method weaknesses: not assessed (read scope: abstract_only plus HTML summary).
- Training data: not applicable | WikiArt style labels: no | WikiArt images: no (Web Gallery of Art; some paintings can overlap with WikiArt, not checked)

## Lower-confidence items seen and used for context only
- WikiArtVectors (Desikan, Shimao, Miton; Entropy 2022; https://doi.org/10.3390/e24091175; confirmed via Semantic Scholar record, abstract only): style and colour vectors for 250,000+ WikiArt images. It uses WikiArt images by construction; whether style labels enter is unclear. Not shortlisted.
- Harnessing Self-Supervised Features for Art Classification (Melis et al.; IRCDL 2026; arXiv 2605.18974; abstract only): DINO family and CLIP as feature extractors for artwork classification. Probe labels not stated. Not shortlisted.

---

## Cross-paper synthesis
- **Common WHY.** Content (what is depicted) dominates generic embeddings, and every style method tries to remove or ignore it: Gram statistics ignore spatial layout (Gatys, Johnson), ALADIN and ALADIN-NST separate branches, GOYA and CSD use contrastive signals that hold style fixed while content varies. Karayev's "style is content-dependent" and Ashton's "style stays entangled with content" are the cautions.
- **Divergent HOW.** (1) Hand-designed statistics on an ImageNet net (Gram). (2) Weak supervision by co-membership groups (ALADIN, BAM-FG). (3) Tag supervision plus self-supervised augmentation invariance (CSD, LAION-Styles). (4) Synthetic content-fixed or style-fixed pairs (ALADIN-NST from style transfer, GOYA from Stable Diffusion). Only (1) needs no art data at all.
- **Strongest WHAT.** CSD beats CLIP on artist-proxy retrieval on WikiArt (64.56 vs 59.4 mAP@1 for ViT-L, §6, Table 1) with WikiArt used for evaluation only. Johnson shows that Gram features carry usable style information (33.46% top-1 with a random forest on one layer, label-supervised probe, class count to be rechecked). ALADIN-NST has the best explicit content-removal evidence but on synthetic data.
- **Unresolved gap (to our knowledge, within this search).** I found no paper that measures, for any label-free style descriptor, how its nearest-neighbour or cluster structure splits between WikiArt movement and WikiArt genre. CSD scores artist retrieval, ALADIN scores Behance groups, ALADIN-NST scores synthetic stylisations, and Ashton's CLIP, DINOv2 and ResNet-50 results use four movements and 320 paintings. So the style-versus-genre balance on our data is open and must be measured, once and disclosed, as a check of the grouping (brief §9). No source reports ArtEmis-style captions naming art style; the text route has no direct evidence.

---

## Shortlist

Expected style-versus-genre balance is **reader-inferred** unless a citation is given. Cost counts one forward pass per unique painting: the 183,694 scorer-train rows cover 36,518 paintings (brief §1), so images need about 36.5K passes, not 300K; ViT-L on the shared RTX 3090 is minutes to well under an hour (reader-inferred estimate; no run made).

| Candidate style source | Training data | WikiArt label / image status | Expected style-vs-genre balance | Readable from images / from captions | Cost to extract |
|---|---|---|---|---|---|
| CSD ViT-L (Somepalli 2024) | LAION-Styles, CLIP init | labels: no (eval only); images: unclear | Better than CLIP on artist retrieval (Table 1: 64.56 vs 59.4 mAP@1, cited); style-vs-genre untested; reader-inferred to be less genre-heavy than CLIP | images: yes; captions: no (image-only encoder) | Low: 36.5K ViT-L passes, 3090 shared |
| ALADIN ViT (Ruta 2021) | BAM-FG, Behance | labels: no; images: no (reader-inferred) | Built to separate content from style (cited, §3); domain gap to classical paintings likely (Behance digital art); reader-inferred | images: yes; captions: no (StyleBabel adds a text side, not tested) | Low to medium: weights from README, no licence stated |
| ALADIN-NST (Ruta 2023) | BBST-4M + NST synthetic | labels: no; images: no (reader-inferred) | Strongest content removal evidence (abstract), synthetic tests only | images: yes; captions: no | Unknown: weight availability not checked |
| GOYA (Wu 2024) | SD-generated images on CLIP features | labels: unclear; images: unclear | Claimed better disentanglement than real-painting models (search summary only) | images: yes; captions: no | Unknown: weights not checked; needs CLIP features first |
| Gram matrices, ImageNet VGG-19 (Gatys 2016; Johnson 2017) | ImageNet only | labels: no; images: no | Style information present (Johnson: 33.46% top-1 single layer with a label-trained probe); no label-free genre measurement; texture bias supported (Geirhos 2019); reader-inferred | images: yes; captions: no | Low compute, but descriptor is large (4,096+ dims per layer set); needs PCA to 256 to 512 (search-result detail) |
| Colour and texture statistics (Karayev 2014) | none | labels: no for the descriptor; images: no | Weak on style relative to deep features per the abstract; likely captures palette and period only (reader-inferred) | images: yes; captions: no | Very low |
| Current CLIP ViT-B/32 image features (control) | web, undisclosed | labels: no; images: unclear | Genre dominant in our data (brief §2: lift genre 9.75, style 4.59); CLIP recovers movements partly (24.57% zero-shot, 0.87 vs 0.77 artist-disjoint kNN) | images: yes; captions: partly | Already extracted |
| Caption text (any encoder) | n/a | n/a | No evidence that ArtEmis captions name style (not quantified in the paper); our data: caption lift style 1.20 (brief §2) | images: n/a; captions: weak | Already extracted |

## Long-CLIP and caption-length verdict
- **Numbers.** ArtEmis explanations average **15.8 words** (ar5iv HTML via summariser; COCO 10.5, Conceptual Captions 9.6). With about 1.3 BPE tokens per English word (reader-inferred conversion, not measured) that is about **20 tokens**, equal to the "effective length" of CLIP that Long-CLIP reports (§3.1: "merely 20 tokens") and far below the 77-token limit. The paper reports a mean only; the maximum and the share over 77 tokens were not found, so I cannot exclude a thin tail of long captions.
- **Verdict.** The 77-token limit **does not bind** for the typical English ArtEmis or ArtELingo caption. Long-CLIP is trained on ShareGPT4V captions averaging 826 characters (§2) and reports its gains on long-caption retrieval; for 20-token captions the expected gain is small (reader-inferred), so it is a low-priority backbone. One check is worth doing before dropping it: the token-length histogram of our rows with the CLIP tokenizer (CPU, minutes).
- **Risk not covered by the paper numbers.** ArtELingo adds Arabic and Chinese captions (about 369K and 426K annotations, Table 3). CLIP's English byte-level BPE gives many tokens per character for such scripts (ArtELingo-28 reports byte-to-character ratios of about 3:1 for Thai and 4:1 for Korean; Chinese and Arabic were not reported). If non-English rows are in the scorer-train set, truncation at 77 tokens may bind for them. I could not confirm which languages the 183,694 rows contain; this is for the project to check.
- **Overlap flag for Long-CLIP.** ShareGPT4V's 100K seed set has 500 WikiArt images (appendix A), so Long-CLIP's 1M pairs could contain a few of our paintings (unknown, at most a few hundred); negligible against 36,518 evaluation paintings but should be disclosed if used.

## Bearing on P0, L, G, E
- **P0 (one grouping per source, style source swapped in).** Literature-supported: CSD is the only candidate with a published WikiArt zero-shot evaluation, no use of WikiArt labels in training (§6), and a released ViT-L checkpoint; ALADIN, ALADIN-NST and GOYA are alternatives with no published WikiArt result. Gram statistics are the cheapest label-free source and have style evidence only through a label-supervised probe (Johnson). The measured gap (style versus genre balance) is open and must be disclosed as one labelled description of each candidate grouping (brief §9). Reader-inferred.
- **L (refine groupings jointly, anchored to sources).** A style source adds a grouping whose non-redundancy with the image grouping is checkable label-free (mutual information against the CLIP image grouping). Reader-inferred: CSD and the CLIP image grouping share a CLIP initialisation, so they may be more redundant than a Gram or ALADIN source; this is testable.
- **G (multiplex graph, one layer per source).** Each style candidate is one more kNN layer; layers built from different training paradigms (Gram vs CSD vs CLIP) give the most independent edges. Reader-inferred; no published multiplex test on art style.
- **E (earliest fusion, augmentation invariance).** CSD's L_SSL and its augmentation recipe are the published precedent for the brief's "augmentation invariance" signal (§6) as a style signal; ALADIN-NST and GOYA are precedents for generating content-held-fixed pairs, usable only if synthetic data is acceptable. Literature-supported for the existence of the signal; applicability to our rows is reader-inferred.
- **Text route.** No source here shows style being read from ArtEmis-type captions; SemArt's expert comments are the only text found that refers to movement and technique. So style has to come from images in all four designs (reader-inferred, consistent with the brief's weak-modality finding: style from images 60.8%, from captions 25.4%).

## Brief citation check
| Brief mention | Status |
|---|---|
| Gatys et al. 2016, CVPR, "Image Style Transfer Using Convolutional Neural Networks" | Verified (Semantic Scholar API: Gatys, Ecker, Bethge; CVPR 2016; DOI 10.1109/CVPR.2016.265). CVF page itself not opened (403). |
| Somepalli et al. 2024, "Measuring Style Similarity in Diffusion Models" (CSD) | Title, authors, arXiv 2404.01292 verified. **Venue not confirmed:** the abstract page shows only cs.CV, April 2024; cite it as an arXiv preprint. Training data: LAION-Styles (511,921 images), WikiArt evaluation only. |
| Ruta et al. 2021 ALADIN | Verified: ICCV 2021, arXiv 2103.09776, eight authors; trained on BAM-FG. Venue from the CVF listing in search results and the README citation, not from an opened CVF page. |
| Zhang et al. 2024 Long-CLIP, ECCV | Verified: arXiv 2403.15378, ECCV 2024, authors Zhang, Zhang, Dong, Zang, Wang. |
| Achlioptas et al. 2021 ArtEmis, CVPR | Title, authors, arXiv 2101.07396 verified; CVPR 2021 venue taken from the CVF listing in search results (page 403), not from the arXiv page. Counts differ by source (439K, 455K). |
| Mohamed et al. 2022 ArtELingo, EMNLP | Verified on the ACL Anthology: EMNLP 2022, pp. 8770 to 8785. |
| "Texture and style statistics (Gram matrices of an ImageNet network)" | Verified through Gatys and Johnson. The brief's 183,694 rows and 77-token claim were not checked against the repo. |
| Brief claim that Long-CLIP was not tested | Not examined here (project fact). |

## Not verified
- Maximum ArtEmis caption length and the share of captions over 77 CLIP tokens (not in the pages read).
- Whether LAION-Styles, GOYA's training or Long-CLIP's 1M pairs contain any of our WikiArt paintings (overlap unmeasured; only the 500 WikiArt images in ShareGPT4V's seed set are documented).
- GOYA's and ALADIN-NST's released weights, licences and venue for ALADIN-NST (Springer page blocked).
- StyleBabel's ECCV 2022 venue (README citation only) and image source.
- Johnson 2017: the number of style classes behind the 33.46% (summariser said 70, WikiArt has 27) and the exact table.
- Karayev 2014 per-feature AP values (summariser output was inconsistent; not used).
- CLIP's own pretraining data (undisclosed), hence image overlap of every CLIP-initialised candidate (CSD, GOYA) with WikiArt.
