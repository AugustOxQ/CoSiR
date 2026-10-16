# Prompted aspect views: aspect instructions to a multimodal embedder, then per-aspect topics and an image-only student

> Literature check, 2026-10-17 (Amsterdam), r10 decide chat, on the user's idea of 2026-10-17: one instruction
> embedder (Qwen3-VL-Embedding) prompted once per aspect, buddy graphs on top. The file name uses CoSiR's sequence date.
> Opus deep-research agent, standard depth: 20 of 20 searches and 8 of 10 page reads used. Read first:
> [2026-12-04](2026-12-04_conditional_similarity_update.md), [2026-12-05](2026-12-05_percept_neighbourhood.md),
> [2026-11-28](2026-11-28_design_l_novelty.md), and the embedder and conditional-similarity tables of
> [2026-10-21](2026-10-21_cvpr_literature_review.md). This note adds only what those notes do not cover. "P" means preprint.
> "Snippet" means the claim was seen only in a search summary and is not checked.
>
> **Checked by the controller** against the arXiv pages: DIOR's WikiArt table (Table 3: CLIP ViT-L/14 59.3, CSD 58.2,
> DIOR 45.6 MAP@1; the authors blame the LVLM's thin art knowledge); COCO-Facet's image-side prompt rows (Table 13,
> R@1 averages, as below); guideline-driven clustering's title, authors and "CVPR 2026" comment (clustering only). The
> arXiv pages of DIOR and COCO-Facet name no venue; WACV 2026 and NeurIPS 2025 come from our earlier notes.
> Guideline-driven clustering's backbones (Qwen2.5-VL, GME) are as read by the agent; the abstract does not name them.

## Brief

- **Novelty verdict: the parts are known, but the whole line was not found.** Earlier work already turns an aspect prompt into an aspect-specific embedding, for clustering and for retrieval: DIOR (WACV 2026), COCO-Facet (NeurIPS 2025), ESMC (AAAI 2026) and guideline-driven clustering (CVPR 2026). No work found does all of these together: prompt both images and captions per aspect, build one cross-modal neighbour graph per aspect, score the topics against affect, style and genre labels, and distil the topics into a multi-head image-only student that serves as the retrieval condition.
- **Most likely attack: "this is just prompting Qwen".** The answer must be a measured gap. The student (or the graph) must beat Qwen3-VL-Embedding used alone, with the aspect instruction on the image, on image-only emotion topics and on aspect retrieval.
- **Instructions do move image embeddings, but unevenly.** On COCO-Facet, an image-side prompt raised VLM2Vec R@1 from 44.5 to 53.4, while MM-Embed did not respond. On CtrlBench, GME sits close to the ceiling that a model ignoring the instruction would reach. No instruction ablation for Qwen3-VL-Embedding was found.
- **Negation ("not its subject") is a known weak point** for CLIP-type models (NegBench, CVPR 2025). No result on negated instructions in MLLM embedders was found. Test it, do not assume it.
- **Art warning:** DIOR's prompted LVLM embedding loses to plain CLIP on WikiArt style (MAP@1 45.6 vs 59.3). No result was found for prompted embedders on art emotion (ArtEmis, ArtELingo, EmoSet).
- **The bar:** training-free, prompt-based pipelines did get into CVPR 2026 and NeurIPS 2025. They usually came with a new benchmark or many datasets. No 2026 preprint was found that scoops this line.

## Q1. Prompted aspect embeddings (confidence: medium-high; several peer-reviewed works found)

The mechanism is established. All of the works below are image-only or text-to-image, and none uses affect or art captions.

- **DIOR** (WACV 2026) prompts an LVLM with "Describe the image in one word regarding {condition}:" and takes the hidden state of the last token. It is used for conditional retrieval and for AMI clustering: car colour AMI 0.297 against 0.011 for CLIP. The wording matters: 14.7 with the condition in the prompt, 6.6 without it.
- **COCO-Facet** (NeurIPS 2025) encodes the gallery image together with an attribute question, "Represent the given image with the following question:". This is the closest match to our image-side use.
- **ESMC** (AAAI 2026) prompts LLaVA-1.5 with a criterion ("The color of the car is"). It takes the hidden states of a chosen token and layer, then trains a 2-layer clustering head. It tests multiple clustering on Cars, Flowers, Fruit, Cards, CMU-face and CIFAR-10. Cars type NMI is 0.88 against 0.66 for Multi-Sub.
- **Guideline-driven image clustering** (Zhong et al., CVPR 2026) is training-free. Qwen2.5-VL writes captions focused on the guideline, and GME-Qwen2-VL-7B encodes caption plus image under a clustering instruction. It covers several criteria on the same images (Fruit, Cards, CIFAR10-MC), with no retrieval and no art.
- **X-Cluster** (CVPR 2026 Findings) and **TPIPS** (P) are already in our notes.
- **Text side, briefly:**
  - InBedder (ACL 2024) clusters one corpus under different instructions and gets different, interpretable clusters.
  - GoalEx (EMNLP 2023) clusters text by a stated goal, through LLM-written explanations.
  - INSTRUCTOR was not re-checked.

Our reading: "aspect name in, aspect-specific embedding out, cluster per aspect" is prior art. Do not claim it.

## Q2. Do instructions change the embedding? (confidence: medium; mixed evidence, none for Qwen3-VL-Embedding)

Instructions change image-side embeddings for most MLLM embedders, but by uneven amounts, and some models ignore them.

| Evidence | Venue | Numbers |
|---|---|---|
| COCO-Facet, image-side prompt, R@1 without → with (Table 13) | NeurIPS 2025 | VLM2Vec 44.5 → 53.4; E5-V 46.0 → 54.9; LLaVE-2B 45.6 → 56.1; MM-Embed 42.8 → 41.5 |
| CtrlBench (ABC v2): a model that ignores the instruction gets about 20 R@1 at most | TMLR 2025 per mlanthology (one aggregator) | VLM2Vec 2.95; GME-2B 20.92; GME-7B 24.75; MagicLens 17.0; ABC 50.94 |
| Qwen3-VL-8B-Embedding zero-shot on GeneCIS, from our 2026-12-04 note | P (TPIPS) | focus attribute 30.85, the best zero-shot row we know |
| MMEB-V3: shifts caused by instructions are "too weak or point the wrong way" | P (abstract on an author page; snippet) | none seen |

- ABC's v1 snippet reports different numbers (VLM2Vec 24.0, ABC 39.7). We use v2, the version read.
- **Negation:** NegBench (CVPR 2025) puts CLIP-type models near chance on negated queries. Training-free fixes exist (CVPRW 2026, ICLR 2026, both snippets). No test of negated instructions on MLLM embedders was found.
- **Qwen3-VL-Embedding:** no instruction ablation was found. The vLLM tutorial says documents need no instruction, so instructions were probably trained mostly on the query side. Our reading, not checked.
- Our own fact: the default instruction moved image-side emotion by at most 1.5 points. That says nothing yet about an aspect instruction. A one-day pilot (aspect instruction against the default, AMI per label) settles it.

## Q3. Emotion and style with prompted embedders (confidence: low-medium; absence of evidence)

No published result was found for art emotion retrieval or clustering with a prompted MLLM embedder. On style, the one result is negative.

- **DIOR on WikiArt style, MAP@1 (Table 3):** CLIP ViT-L/14 59.3, CSD 58.2, DIOR 45.6. The authors blame the LVLM's thin art knowledge.
- **Emotion-near results:**
  - ESMC's "emotion" is facial expression on CMU-face.
  - X-Cluster's mood criterion is its hardest on COCO-4c (38.1, from our notes).
  - EmoGist (EMNLP 2025 Findings, snippet) classifies emotion training-free with LVLM explanations, without embeddings.
  - ArtECulture and ArtSociety (P, 2026, snippets) are MLLM emotion benchmarks or agents, not embedders.

Our reading: the art-emotion cell is open. That is good for novelty but a risk for results. Given our caption-over-image facts, the emotion view may work on captions and fail on images. That gap is the student's job.

## Q4. Novelty verdict (confidence: medium; one reviewer-style search for the whole line, not exhaustive)

| Part | Verdict | Closest prior |
|---|---|---|
| Aspect instruction → aspect-specific embedding | Known | DIOR, COCO-Facet, ESMC, guideline-driven clustering |
| LLM proposes the aspect names | Known | X-Cluster, TPIPS |
| Mutual-kNN graph in image and in caption space, then Leiden | Known construction | UMAP intersection, WNN, TAC (2026-12-05 note) |
| One graph per prompted aspect view on image-text data, scored against emotion, style and genre | Not found | PercepT does one fused partition with a supervised affect teacher |
| Multi-head image-only student; picking a head is the retrieval condition, with no condition input during training | Not found | SCE-Net and DiscoverNet learn masks from triplets; TPIPS trains on human votes |

**The three closest papers:**
1. **DIOR:** per-condition LVLM embeddings for retrieval and clustering. It is image-only and needs an LVLM pass per condition at query time. It has no captions, no graph, no student and no affect, and it loses on art style.
2. **Guideline-driven clustering:** multi-criterion clustering with GME and an instruction, at CVPR 2026. It has no retrieval, no image-only student and no art or affect.
3. **COCO-Facet:** image-side prompts for attribute retrieval. The attribute is given, nothing is discovered, there is no clustering and the attributes are factual (counts, weather, scenes).

ESMC and PercepT are next in line.

**Comparators a CVPR reviewer would require:**
- a. Qwen3-VL-Embedding with the aspect instruction, used alone, for retrieval and for k-means or Leiden per aspect. This is the "just prompting" baseline.
- b. Qwen3-VL-Embedding with the default instruction, as one partition.
- c. DIOR with the same aspect words.
- d. ESMC and/or guideline-driven clustering for the per-aspect clustering.
- e. The PercepT reimplementation and its no-DEC row.
- f. CRL or CLAY, given the aspect name, for conditional retrieval; TPIPS zero-shot if released.

**The attack and the answer.** The attack: "the method is Qwen with K prompts plus a known graph". Three results would answer it (our reading):
- i. On image-only queries, the student's emotion head beats the prompted Qwen image embedding at emotion AMI and aspect retrieval. Our facts make this likely, since emotion lives in the captions.
- ii. Per aspect view, the AMI matrix (view × label) is diagonal-dominant, and the graph view beats the plain prompted embedding.
- iii. The student needs one small forward pass instead of K MLLM passes; give the measured cost.

If prompted Qwen on the image alone already matches the student, the contribution shrinks to an analysis.

## Q5. The publication bar (confidence: medium; venues checked on CVF, NeurIPS or AAAI pages unless noted)

Training-free or prompt-based pipelines do get in. They come with breadth: many datasets, a new benchmark, or both.

| Paper | Venue | Contribution type | Evaluation that carried it |
|---|---|---|---|
| Guideline-driven clustering | CVPR 2026 | training-free agent | about 11 datasets (general, fine-grained, multi-criterion), new ABO-LC set; ACC/NMI/ARI |
| CLAY | CVPR 2026 | training-free, on frozen VL embeddings | new CLAY-EVAL benchmark, GeneCIS; mAP |
| X-Cluster | CVPR 2026 Findings (not the main track) | training-free criterion discovery | new COCO-4c and Food-4c benchmarks |
| COCO-Facet | NeurIPS 2025 | benchmark plus a prompting method | 9,112 queries, about 12 embedders; R@1/R@5 |
| CRL | NeurIPS 2025 | training-free conditioning | GeneCIS and others |
| ESMC | AAAI 2026 | frozen MLLM plus a small head | 7 multi-clustering sets; NMI/RI |
| DIOR | WACV 2026 | training-free LVLM prompt | LanZ-DML, WikiArt, GeneCIS focus |

- **Scoops:** none found. The nearest 2026 items are guideline-driven clustering (June), TPIPS (July, P) and MMEB-V3 (P).
- **Bar for us (our reading):** ArtELingo plus a second dataset (COCO-4c mood, or Affection). Report per-label AMI at matched k, image-only prediction scored on external labels, retrieval R@K, and GeneCIS focus tasks. Include all comparators a to f above.

## Sources

| Title, authors, year | Venue | URL | Q |
|---|---|---|---|
| Training-free Conditional Image Embedding (DIOR), Kawarada et al. 2025 | WACV 2026 | https://arxiv.org/html/2512.21860v1 | 1, 3, 4, 5 |
| Highlighting What Matters (COCO-Facet), Li, Gao, Du 2025 | NeurIPS 2025 | https://arxiv.org/html/2505.15877v1 | 1, 2, 4, 5 |
| ESMC, Wang, Jia, Liu, Hou 2025 | AAAI 2026 | https://arxiv.org/html/2512.00725v1 | 1, 3, 4 |
| Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent, Zhong et al. 2026 | CVPR 2026 | https://arxiv.org/html/2606.24094 ; https://cvpr.thecvf.com/virtual/2026/poster/37785 | 1, 4, 5 |
| ABC / CtrlBench, Schneider, Kerschbaum, Chen 2025 | TMLR 2025 (aggregator only) | https://arxiv.org/html/2503.00329v2 | 2 |
| Vision-Language Models Do Not Understand Negation (NegBench), Alhamoud et al. 2025 | CVPR 2025 | https://arxiv.org/abs/2501.09425 | 2 |
| MMEB-V3 (abstract only) | P, snippet | https://www.catalyzex.com/author/Mingyi%20Su | 2 |
| Qwen3-VL-Embedding, Li et al. 2026; vLLM tutorial | P; docs | https://arxiv.org/abs/2601.04720 ; https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/Qwen3-VL-Embedding.html | 2 |
| InBedder, Peng et al. 2024 | ACL 2024 | https://arxiv.org/abs/2402.09642 | 1 |
| GoalEx, Wang, Shang, Zhong 2023 | EMNLP 2023 | https://arxiv.org/abs/2305.13749 | 1 |
| X-Cluster, Liu et al. | CVPR 2026 Findings | https://arxiv.org/html/2410.05217v5 | 1, 5 |
| TPIPS, Wang et al. 2026 | P | https://arxiv.org/abs/2607.18237 | 2, 4 |
| EmoGist 2025 | EMNLP 2025 Findings (snippet) | https://arxiv.org/abs/2505.14660 | 3 |
| ArtECulture 2026; ArtSociety 2026 | P (snippets) | https://arxiv.org/abs/2608.03358 ; https://arxiv.org/abs/2609.13240 | 3 |

## Open

- **Instruction ablation for Qwen3-VL-Embedding**, especially on the image side. Read the report's appendix (arXiv 2601.04720); the cheaper answer is our own pilot.
- **Negated instructions in MLLM embedders.** Nothing found. A pilot with "style, not subject" should check that genre AMI drops in the style view.
- **Guideline-driven clustering's per-criterion numbers and GME details** (Appendix C.3). Fetch arXiv 2606.24094 HTML from offset 100000.
- **MMEB-V3's arXiv ID and numbers.** Search arXiv for "OmniSET MMEB-V3".
- **ABC's venue.** Check the TMLR OpenReview page.
- **ICLR 2026 and ECCV 2026 papers in this space** were not searched separately. Search "instruction-conditioned image embedding ICLR 2026".
