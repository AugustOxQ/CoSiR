# Cross-modal conditional similarity: close work and missing citations for related work v1.1

Date: 2026-11-09 (sequence date 2027-01-13)
Run by: paper chat, gap-fill for related work v1.1 (Sonnet, `high`). Searches on the open web plus page reads; "verified
(page opened)" means the official page (CVF open access, PMLR, ACL Anthology, arXiv abstract or HTML) was opened in this run
and the stated fields were read there. Builds on the 2026-12-04, 2027-01-03 and 2027-01-08 notes and on `refs.bib` (197 entries).
Limits: the search tool is a general web search, not a search of OpenReview or the ACL Anthology by field; a negative below
means "not found in about 12 searches", not "does not exist".

## 1. Results brief

- **No work found that does what we do**: learning one embedding space per aspect for images and for text from an
  instructed embedding model's neighbour graphs, without aspect labels.
- Closest three: (1) **Lenses** (CVPR 2026): several lens embeddings per image and caption, scored per lens across
  modalities; trained with annotated lens categories. (2) **CANVAS** (ECCV 2026, not a preprint as the 2027-01-08 note
  said): one embedding per relation type for art, scored across modalities; trained with relation labels. (3) **Show Me the
  Infographic I Imagine** (arXiv 2604.07989, preprint): one MLP head per facet on the image embedding, per-facet text-image
  cosine; trained with facet captions that an MLLM wrote.
- The difference we can state is how ours is learned: the supervision comes from an instructed embedding model's mutual
  neighbour graphs, with no aspect, lens, relation or facet labels, and the test-time model reads a plain embedding with no
  instruction or facet token. This is unchanged from the brief; no new work weakens it.
- New near-neighbours that need a sentence: QARE (CVPR 2026 Findings; per-attribute embeddings from a frozen VLM, a query
  at test time, training-free) and AspectCSE (RANLP 2023; text-only aspect embeddings with knowledge-graph labels).
- AspectSim (arXiv 2601.03435) is a text-document benchmark and framework, background only.
- Missing from `refs.bib`: TURTLE, UT-Zappos50k (Yu and Grauman), the three fashion sets that CSN's successors use
  (DARN, DeepFashion, FashionAI), and all the close works above. DEC, TEMI and GeneCIS are present.

## 2. Close works

Group letters follow the drafting brief: (a) datasets, (b) conditional similarity in models, (c) general VLMs and MLLMs
with multi-space scoring, (d) learning from a teacher's neighbours.

**Lenses: Toward Polysemous Vision-Language Understanding** (Alomari, Asgarov, Thomas; CVPR 2026, pp. 37810-37820).
URL: https://openaccess.thecvf.com/content/CVPR2026/html/Alomari_Lenses_Toward_Polysemous_Vision-Language_Understanding_CVPR_2026_paper.html .
Status: verified (page opened; abstract only, paper body not read). Its page states: a multi-prompt embedding model on a
multimodal LLM; learned lens tokens give lens-specific embeddings for every image and caption; a lens-masking similarity
with a global fallback; five annotated lens categories (Literal, Figurative, Abstract, Background, Emotional); a
category-aware multi-positive contrastive loss. Overlap: partial (several spaces scored across modalities, includes an
emotional lens). Difference: lens categories are annotated, lenses are named interpretation types, not style or genre. Group (c)/(b).

**Art Beyond Semantics: Sheaf-Informed Contrastive Learning for Multi-Relational Representations (CANVAS)** (Schaerf,
Purificato, Riccio, Silvestri, Garcia; ECCV 2026, arXiv 2607.16321).
URL: https://arxiv.org/abs/2607.16321 . Status: verified (page opened; abstract and comment "Accepted to ECCV 2026"; body
not read here). Its page states: each artwork projected into several embeddings, one per relation type, a new contrastive
loss, no external data at inference, three new art benchmarks (WikiArt+, HertzianaDP, SemArt+). Overlap: partial (art,
several spaces, image-text retrieval). Difference (from the 2027-01-08 note, which read the body): relation-labelled edges
supervise it. Group (c)/(b). Correction: the 2027-01-08 note called it a preprint; the arXiv page now says ECCV 2026.

**Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for Authoring Support** (Xu, Hu, Shuai, Chen, Yang;
arXiv 2604.07989, cs.IR, 9 April 2026; no venue listed).
URL: https://arxiv.org/abs/2604.07989 and https://arxiv.org/html/2604.07989 . Status: verified (page opened, HTML full
text read to its Sec. 3.3 and 5.1; last 14k characters, references, not read). The paper states, in Sec. 3.2.2, "a lightweight
MLP head per embedding facet" on the image feature, a shared text encoder conditioned by a facet-specific token, and
per-facet cosine similarity between text and image; the final relevance is a weighted sum over facets. Five facets
(content, chart type, layout, illustration, style); chart type uses a discrete kernel, the other four use embeddings. The
whole model (both encoders and heads) is fine-tuned (Fig. 3 caption) with an in-batch contrastive loss (Sec. 3.3). The
supervision is synthetic: Qwen3-VL-8B writes a description of each of 52,000 training infographics, which is turned into
facet captions (two paraphrases per facet). The base encoders are not named. Overlap: partial. It shares the
head-per-facet design and per-facet image-text similarity, and it uses no human facet labels. It differs in that an MLLM
generates text per facet (no embedding teacher, no neighbour graphs), the text side gets a facet token (ours reads the plain embedding), the domain is
infographics, and the facets come from a formative-study taxonomy. This is the closest in architecture; cite it in (b). Its
page does not support "found tonight" claims of a teacher embedding model. Group (b)/(c).

**Towards Text-Guided Attribute-Disentangled Multimodal Representation Learning (QARE)** (Wei, Katakol, Brack, Lin, Bai, Li,
Zhang, Shechtman, Ravi, Kale; CVPR 2026 Findings, pp. 1883-1892).
URL: https://openaccess.thecvf.com/content/CVPR2026F/html/Wei_Towards_Text-Guided_Attribute-Disentangled_Multimodal_Representation_Learning_CVPRF_2026_paper.html .
Status: verified (page opened; paper PDF text read in part: abstract, introduction, method opening). Its page states: a task
that gives an embedding E(image, attribute) sensitive only to the queried attribute (object, style, background), a benchmark
(QARE-Bench), and TF-QARE, a training-free method that pools the hidden states of reply tokens that a frozen VLM generates
for a structured prompt; it outperforms fine-tuned VLM2Vec. Overlap: partial (attribute-specific embeddings from an
instruction-following VLM). Difference: the attribute is named at test time, nothing is trained, the paper is about images
(from what we read). Group (b)/(c). It is a baseline-like route (prompted frozen model), as Qwen3-VL-Embedding is for us.

**AspectCSE** (Schopf, Gerber, Ostendorff, Matthes; RANLP 2023, pp. 1054-1065).
URL: https://aclanthology.org/2023.ranlp-1.113 . Status: verified (page opened). Its page states: contrastive learning for
aspect-based sentence embeddings, with Wikidata knowledge-graph properties as supervision; multi-aspect embeddings beat
single-aspect ones on aspect-specific retrieval. Overlap: background (text only, aspect labels from a knowledge graph).
Group (b), text side.

**The Critical Role of Aspects in Measuring Document Similarity (AspectSim)** (Hossain, Hazra, Bhuiyan, Karmaker; arXiv
2601.03435, cs.CL, 6 January 2026; 24 pages; no venue listed).
URL: https://arxiv.org/abs/2601.03435 . Status: verified (page opened; abstract only). Its page states: document
similarity conditioned on an explicitly given aspect; a benchmark of 26K aspect-document pairs with human judgments; GPT-4o
prompting agrees with humans better than holistic similarity; a meta-evaluation of 16 open LLMs and 9 embedding models
(direct prompting is weak, a two-stage refinement helps). No training method on the page. Overlap: background (text
documents, no images, no training of aspect spaces). Group (a)/(b). Cite as one sentence, if at all, on aspect-conditioned
text similarity next to C-STS.

**Not close, checked and set aside:** DistillCIR (ICCV 2025; distills an instruction-aware MLLM embedding into a
projection model for composed image retrieval; seen in a search snippet only, not opened), Factorized MLLM Embeddings for CIR
(arXiv 2611.05193; abstract opened; composed image retrieval, no aspects), the "Many Senses of Visual Similarity" and
"Controlling Embedding Spaces" papers (already in `refs.bib` as `wang2026tpips` and `fioresi2026controlling`).

**Searched for and not found:** (i) aspect-specific students distilled from an instruction-following multimodal embedding
model; (ii) per-aspect image-text spaces learned from mutual-kNN graphs of a teacher's embeddings; (iii) unsupervised aspect
discovery with image-text heads on paintings. Searches ran on arXiv-indexed pages and the CVF site through a web search;
OpenReview and the ACL Anthology were reached only through the same search, so a gap there is possible.

## 3. Missing citations

Present in `refs.bib` (key only; not re-verified here): DEC `xie2016dec`, TEMI `adaloglou2023temi`, GeneCIS
`vaze2023genecis`, CSN `veit2017csn` (pages 830-838, matches the CVF page), ASEN `ma2020asen`, ASEN++ `dong2021asenpp`.
PMLR also lists DEC at ICML 2016, PMLR 48, pp. 478-487, as in the entry.

| Work | In refs.bib? | Status | New key |
|---|---|---|---|
| TURTLE: Gadetsky, Jiang, Brbic, "Let Go of Your Labels with Unsupervised Transfer", ICML 2024, PMLR 235, pp. 14382-14407 | No | verified (PMLR page opened) | `gadetsky2024let` |
| UT-Zappos50k: Yu and Grauman, "Fine-Grained Visual Comparisons with Local Learning", CVPR 2014, pp. 192-199 | No | verified (CVF page opened). The UT-Zap50K project page asks to cite this paper (and Yu and Grauman, "Semantic Jitter", ICCV 2017, not added) | `yu2014fine` |
| DARN: Huang, Feris, Chen, Yan, "Cross-Domain Image Retrieval With a Dual Attribute-Aware Ranking Network", ICCV 2015, pp. 1062-1070 | No | verified (CVF page opened) | `huang2015cross` |
| DeepFashion: Liu, Luo, Qiu, Wang, Tang, CVPR 2016, pp. 1096-1104 | No | verified (CVF page opened) | `liu2016deepfashion` |
| FashionAI: Zou, Kong, Wong, Wang, Liu, Cao, CVPR 2019 Workshops | No | verified (CVF page opened); the page shows no page range, so pages are left out | `zou2019fashionai` |
| Lenses | No | verified (above) | `alomari2026lenses` |
| CANVAS | No | verified (above) | `schaerf2026art` |
| Infographic retrieval | No | verified (above) | `xu2026show` |
| QARE | No | verified (above) | `wei2026towards` |
| AspectCSE | No | verified (above) | `schopf2023aspectcse` |
| AspectSim | No | verified (above) | `hossain2026critical` |
| Fonts set of CSN (Bernhardsson's font data) | No | not verified: CSN's text names it only as a font dataset collected by Bernhardsson (a footnote URL); no paper to cite. State it in prose, no entry. | none |

**Attribute-similarity datasets.** From CSN's text (the arXiv PDF, read as text): Fonts (62 characters in 50,000 fonts,
first 1,000 used; triplets from same character or same font) and UT-Zappos50k (50,000 shoe images, four characteristics:
type, gender, heel height, closing mechanism; plus brand for a classification test). From ASEN's text (arXiv 2002.02814,
read as text): FashionAI (reconstructed from the 2018 challenge release, 180,335 images, 8 attributes), DARN,
DeepFashion for attribute-specific retrieval, and Zappos50k with CSN's split for triplet prediction. ASEN++ (arXiv
2104.02429) uses FashionAI, DARN and DeepFashion. Not checked here: SCE-Net and DiscoverNet datasets (`tan2019scenet`,
`ye2022discovernet`), the CUB and CelebA uses in the 2026-12-04 note.

**Multiple-clustering benchmark sets** (from each paper's own page; these are the sets for related-work group (a)):

| Paper (key in refs.bib) | Sets | Source opened |
|---|---|---|
| Multi-MaP, CVPR 2024 (`yao2024multimap`) | ALOI (color, shape), Fruit (color, species), Fruit360 (color, species), Flowers (color, species), Card (order, suits), CMUface (emotion, glass, identity, pose), Stanford Cars (color, type) | arXiv 2404.15655 HTML |
| Multi-Sub, NeurIPS 2024 (`yao2024multisub`) | not checked (arXiv 2411.03978 abstract page only: title, authors Yao, Qian, Hu, "accepted to NeurIPS 2024") | partly |
| ICTC, ICLR 2024 (`kwon2024ictc`) | Stanford 40 Action (action, location, mood), PPMI, CIFAR-10, STL-10, CIFAR-100 (20 superclasses), CIFAR-10-Gen | arXiv 2310.18297 HTML |
| X-Cluster, CVPR 2026 Findings (`liu2026xcluster`) | Fruit-2c, Card-2c, Action-3c, Clevr-4c, plus new COCO-4c and Food-4c | arXiv 2410.05217 HTML |
| ESMC, AAAI 2026 (`wang2026esmc`; its arXiv page lists no venue) | Stanford Cars, Flower, Fruit, Card, CMU face, Fruit360, CIFAR10 | arXiv 2512.00725 HTML |
| Guideline-driven clustering, CVPR 2026 (`zhong2026guideline`) | multiple clustering: Fruit, Cards, CIFAR10-MC; also CIFAR-10, STL-10, ImageNet-10, CUB, Stanford Dogs, Stanford Cars, Oxford Flowers, ABO-LC | arXiv 2606.24094 HTML (first 100k characters) |
| CIM-Embed, arXiv 2610.21805 (`xu2026cimembed`) | new ProductMIC: 5,373 products, 449 grouping tasks from Amazon Reviews 2023 | arXiv abstract page |

Reading: Fruit, Card, Flowers, Cars, CMUface and Fruit360 are the shared multiple-clustering sets; the "aspects" there are
attributes such as color and species, with no emotion or art set except CMUface emotion and ICTC's Mood. No multiple-clustering paper above
uses paintings.

## 4. BibTeX

Keys follow `refs.bib`. Fields not seen on the page are left out (for example the FashionAI pages, the CSN-style
volume numbers). Authors are as the pages list them.

```bibtex
% verified: https://proceedings.mlr.press/v235/gadetsky24a.html (title, authors, PMLR 235, pp 14382-14407, ICML 2024)
@inproceedings{gadetsky2024let,
  author    = {Gadetsky, Artyom and Jiang, Yulun and Brbic, Maria},
  title     = {Let Go of Your Labels with Unsupervised Transfer},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {14382--14407},
  year      = {2024}
}
```

```bibtex
% verified: https://openaccess.thecvf.com/content_cvpr_2014/html/Yu_Fine-Grained_Visual_Comparisons_2014_CVPR_paper.html (pp 192-199); dataset page https://vision.cs.utexas.edu/projects/finegrained/utzap50k asks to cite it
@inproceedings{yu2014fine,
  author    = {Yu, Aron and Grauman, Kristen},
  title     = {Fine-Grained Visual Comparisons with Local Learning},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {192--199},
  year      = {2014}
}
```

```bibtex
% verified: https://openaccess.thecvf.com/content_iccv_2015/html/Huang_Cross-Domain_Image_Retrieval_ICCV_2015_paper.html (pp 1062-1070)
@inproceedings{huang2015cross,
  author    = {Huang, Junshi and Feris, Rogerio S. and Chen, Qiang and Yan, Shuicheng},
  title     = {Cross-Domain Image Retrieval With a Dual Attribute-Aware Ranking Network},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {1062--1070},
  year      = {2015}
}
```

```bibtex
% verified: https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html (pp 1096-1104)
@inproceedings{liu2016deepfashion,
  author    = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
  title     = {{DeepFashion}: Powering Robust Clothes Recognition and Retrieval With Rich Annotations},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1096--1104},
  year      = {2016}
}
```

```bibtex
% verified: https://openaccess.thecvf.com/content_CVPRW_2019/html/FFSS-USAD/Zou_FashionAI_A_Hierarchical_Dataset_for_Fashion_Understanding_CVPRW_2019_paper.html (the page shows no page range, so none given)
@inproceedings{zou2019fashionai,
  author    = {Zou, Xingxing and Kong, Xiangheng and Wong, Waikeung and Wang, Congde and Liu, Yuguang and Cao, Yang},
  title     = {{FashionAI}: A Hierarchical Dataset for Fashion Understanding},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2019}
}
```

```bibtex
% verified: https://openaccess.thecvf.com/content/CVPR2026/html/Alomari_Lenses_Toward_Polysemous_Vision-Language_Understanding_CVPR_2026_paper.html (pp 37810-37820)
@inproceedings{alomari2026lenses,
  author    = {Alomari, Hani and Asgarov, Ali and Thomas, Chris},
  title     = {Lenses: Toward Polysemous Vision-Language Understanding},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {37810--37820},
  year      = {2026}
}
```

```bibtex
% verified: https://arxiv.org/abs/2607.16321 (arXiv comment: Accepted to ECCV 2026; the proceedings page was not opened, so no pages)
@misc{schaerf2026art,
  author        = {Schaerf, Ludovica and Purificato, Antonio and Riccio, Piera and Silvestri, Fabrizio and Garcia, Noa},
  title         = {Art Beyond Semantics: Sheaf-Informed Contrastive Learning for Multi-Relational Representations},
  year          = {2026},
  eprint        = {2607.16321},
  archivePrefix = {arXiv},
  note          = {Accepted to ECCV 2026, per the arXiv comment}
}
```

```bibtex
% verified: https://arxiv.org/abs/2604.07989 (arXiv only, cs.IR, v1 9 April 2026; no venue listed)
@misc{xu2026show,
  author        = {Xu, Jing and Hu, Jiarui and Shuai, Zhihao and Chen, Yiyun and Yang, Weikai},
  title         = {Show Me the Infographic I Imagine: Intent-Aware Infographic Retrieval for Authoring Support},
  year          = {2026},
  eprint        = {2604.07989},
  archivePrefix = {arXiv}
}
```

```bibtex
% verified: https://openaccess.thecvf.com/content/CVPR2026F/html/Wei_Towards_Text-Guided_Attribute-Disentangled_Multimodal_Representation_Learning_CVPRF_2026_paper.html (CVPR Findings, pp 1883-1892)
@inproceedings{wei2026towards,
  author    = {Wei, Yibing and Katakol, Sudeep and Brack, Manuel and Lin, Jinhong and Bai, Haoyue and Li, Yu-Teng and Zhang, Richard and Shechtman, Eli and Ravi, Hareesh and Kale, Ajinkya},
  title     = {Towards Text-Guided Attribute-Disentangled Multimodal Representation Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  pages     = {1883--1892},
  year      = {2026}
}
```

```bibtex
% verified: https://aclanthology.org/2023.ranlp-1.113 (RANLP 2023, pp 1054-1065)
@inproceedings{schopf2023aspectcse,
  author    = {Schopf, Tim and Gerber, Emanuel and Ostendorff, Malte and Matthes, Florian},
  title     = {{AspectCSE}: Sentence Embeddings for Aspect-based Semantic Textual Similarity Using Contrastive Learning and Structured Knowledge},
  booktitle = {Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing (RANLP)},
  pages     = {1054--1065},
  year      = {2023}
}
```

```bibtex
% verified: https://arxiv.org/abs/2601.03435 (arXiv only, cs.CL, v1 6 January 2026; no venue listed)
@misc{hossain2026critical,
  author        = {Hossain, Eftekhar and Hazra, Tarnika and Bhuiyan, Ahatesham and Karmaker, Santu},
  title         = {The Critical Role of Aspects in Measuring Document Similarity},
  year          = {2026},
  eprint        = {2601.03435},
  archivePrefix = {arXiv}
}
```

## Not verified, for the record

- Page ranges of FashionAI (CVF page shows none; a search snippet said pp. 296-304, not confirmed).
- The Multi-Sub (`yao2024multisub`) dataset list; TEMI's and DEC's entries were not re-opened.
- Bodies of Lenses, CANVAS (read in the 2027-01-08 note), Infographic after Sec. 5.1, and QARE after its method opening.
- DistillCIR, RCML (arXiv 2508.17497), PVSE, SLiCS: from earlier notes or snippets only.
- Searches were not exhaustive; a work posted after early November 2026 may be missing.
- TURTLE wording rule from the brief applies when citing: any sentence that says we ran it adds "at its released code's default
  settings, without its hyperparameter selection".

Temporary files (PDFs and HTML for text extraction) are in the session scratchpad
`/tmp/claude-0/-project-CoSiR/116c719b-0cee-44b4-8450-03a59687901b/scratchpad/pdf/`; they can be deleted.
