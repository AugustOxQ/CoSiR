# Literature check for the ArtELingo held test: is the claim still new, and against what

> Searched 2026-10-09 (Amsterdam), read-only: web search and the arXiv API (2024 to 2026-10-09), each work then
> checked on a primary page. Builds on the
> [2026-10-21 review](../../../../docs/reports/auto/v2/2026-10-21_cvpr_literature_review.md) and the
> [2026-10-24 novelty check](../../../../docs/reports/auto/v2/2026-10-24_aspect_task_novelty_check.md); their rows are not
> repeated. Project numbers come from the round 3 report, the backbone check and the 8B probe log.

## 1. Verdict

**New, to our knowledge.** Nothing found up to 2026-10-09 scores image↔caption of different items under an aspect
given only by support and contrast pairs. Newer work names the aspect in text (TPIPS, CLAY, CRL, SteerViT) or takes
examples only for text (RICE, BGE-EN-ICL). The protocol needs careful wording (§3.2); the method stays modest (§3.3).

**Strongest applicable comparator (our reading): verbalise, then name, built on CRL.** Qwen3-VL-8B-Instruct, the
model of the 8B probe, reads the episode's 4 support and 4 contrast pairs and states what the supports share and the
contrasts lack (a VisDiff-style set-difference prompt). CRL (NeurIPS 2025) turns that phrase into a text basis on the
same frozen CLIP ViT-B/32; anchor and candidates are projected and compared, fused with β·cos on the same cross-fitted
grid as every baseline. Its matched control drops the examples. A privileged variant (true aspect name) is its ceiling.

- **Why this one.** Episode examples only, no labels, same frozen backbone, built from published parts; it is what a
  reviewer reaches for after CIReVL and OSrCIR, and it answers "why examples instead of names?".
- **In our baseline set?** Partly. CRL and privileged names are plan §8 tier 2, verbalise-then-name is only a
  stretch, and none has been run. The in-context MLLM reranker (tier 2, required) is the only example-reading MLLM
  comparator scored so far, on development seeds.
- **Cost (our estimate).** Half a day to build; one 8B call per episode (the probe needed 3.59 s per episode for 4
  prompts, so under an hour per 1,800 episodes); CRL is CPU minutes on cached features.
- **Can it reach affect steering's 18.9 R@1 (+0.6 over the best condition-free scorer)?** Possible, not likely.
  For: the gain sits on emotion, captions name emotions, and CLAY's text-named mood condition gained +4.9 mAP over
  CLIP B/32 on Stanford40. Against: CRL's gains on DeepFashion are small (7.93 vs 6.08 MAP), images carry emotion
  weakly (image-side probe 35.1), and the 8B model reading the same examples had condition gain +0.21 [−0.51, 0.94].
- **Must-add**, frozen on development seeds before any held row is scored; run the privileged variant first.

**Report-only.** The in-context MLLM reranker on held episodes (about 2 GPU hours per 1,800); a RICE-style
embedder (Qwen3-VL-Embedding-2B with the 8 pairs in context; not in §8, cheap, likely weak); Qwen3-VL-Embedding
with the aspect in its instruction (privileged, another backbone).

## 2. Relevant papers (new since the earlier review, or newly verified)

| Title | Venue, year | Relevance | Link |
|---|---|---|---|
| The Many Senses of Visual Similarity: A Text-Prompted Image Perceptual Metric (TPIPS; S.-Y. Wang et al.) | arXiv, 2026 | Concurrent: aspect named in free text, image triplets; frontier VLMs well short of human consensus | [2607.18237](https://arxiv.org/abs/2607.18237) |
| CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space | CVPR 2026 | Training-free text conditions on frozen CLIP B/32; image only; includes mood | [2604.11539](https://arxiv.org/abs/2604.11539) |
| Conditional Representation Learning for Customized Tasks (CRL; H. Liu et al.) | NeurIPS 2025 | LLM-written basis for a named criterion, frozen CLIP; code public; core of the comparator | [2510.04564](https://arxiv.org/abs/2510.04564) |
| Steerable Visual Representations (SteerViT) | ECCV 2026 | Text-steered frozen ViT; venue new since the earlier review | [2604.02327](https://arxiv.org/abs/2604.02327) |
| Relational Visual Similarity (Nguyen et al.) | CVPR 2026 | One new notion of similarity, fine-tuned VLM; no conditioning | [2512.07833](https://arxiv.org/abs/2512.07833) |
| Effective Dense Retrieval using Only In-Context Examples (RICE; Jedidi et al.) | arXiv, 2026 | Training-free LLM embeddings conditioned on example pairs; text only | [2609.38099](https://arxiv.org/abs/2609.38099) |
| Making Text Embedders Few-Shot Learners (BGE-EN-ICL) | ICLR 2025 | Examples on the query side; text only | [2409.15700](https://arxiv.org/abs/2409.15700) |
| Lens: Bringing the Right Semantic Perspective into Focus for Training-Free Multimodal Representation Learning | arXiv, 2026 | Training-free MLLM embeddings steered by a text task phrase | [2609.20252](https://arxiv.org/abs/2609.20252) |
| FreeRet: MLLMs as Training-Free Retrievers | ICML 2026 | Off-the-shelf MLLM embeds, then reranks | [2509.24621](https://arxiv.org/abs/2509.24621) |
| Qwen3-VL-Embedding and Qwen3-VL-Reranker | arXiv, 2026 | Instruction embedder and reranker for the report-only baselines | [2601.04720](https://arxiv.org/abs/2601.04720) |
| Describing Differences in Image Sets with Natural Language (VisDiff) | CVPR 2024 | Set-difference description: the verbaliser step | [2312.02974](https://arxiv.org/abs/2312.02974) |
| Do Composed Image Retrieval Benchmarks Require Multimodal Composition? (Attimonelli et al.) | arXiv, 2026 | 32.2% to 83.6% of CIR queries solvable from one modality | [2605.14787](https://arxiv.org/abs/2605.14787) |
| C-STS: Conditional Semantic Textual Similarity (Deshpande et al.) | EMNLP 2023 | One sentence pair under a high and a low condition: a text swap test | [2305.15093](https://arxiv.org/abs/2305.15093) |
| Unsupervised Learning via Meta-Learning (CACTUs; Hsu et al.) | ICLR 2019 | Training tasks from k-means partitions: practice episodes from label-free groupings | [1810.02334](https://arxiv.org/abs/1810.02334) |
| Multimodal Model-Agnostic Meta-Learning via Task-Aware Modulation (Vuorio et al.) | NeurIPS 2019 | Infers the task mode from the support set: the reader's role | [1910.13616](https://arxiv.org/abs/1910.13616) |
| Identifying Ambiguous Similarity Conditions via Semantic Matching (DiscoverNet; Ye, Shi, Zhan) | CVPR 2022 | Conditions from triplets without condition labels | [2204.04053](https://arxiv.org/abs/2204.04053) |
| Multi-Modal Proxy Learning Towards Personalized Visual Multiple Clustering (Multi-MaP) | CVPR 2024 | Picks among CLIP clusterings by a user's keyword | [2404.15655](https://arxiv.org/abs/2404.15655) |
| Multiview Triplet Embedding: Learning Attributes in Multiple Maps (Amid, Ukkonen) | ICML 2015 | One map per hidden attribute (plan E18 lead) | [PMLR v37](https://proceedings.mlr.press/v37/amid15.html) |
| ArtECulture: Benchmarking Culture-Conditioned Visual Emotion Understanding in MLLMs | arXiv, 2026 | ArtELingo rebuilt for MLLM emotion reading; no retrieval | [2608.03358](https://arxiv.org/abs/2608.03358) |

**Corrections to earlier citations.** CSD's venue version is "Investigating Style Similarity in Diffusion Models",
ECCV 2024 ([ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8294_ECCV_2024_paper.php)). TEVI is in the
Findings of EMNLP 2026. Fioresi et al. (ECCV 2026) is titled "Controlling Embedding Spaces with Text-Conditioned
Transformations". No ArtEmis or ArtELingo retrieval paper appeared (2026 art-affect work is MLLM understanding).

## 3. Novelty risks

### 3.1 The task (claim 1): low to moderate
All 2026 neighbours name the aspect in text or are text-only when examples define the task (§1). TPIPS is concurrent
and a short step from example conditioning: search arXiv again the week before the abstract (2026-11-10), and keep
"to our knowledge" with the novelty check's wording.

### 3.2 The protocol (claim 2): moderate, on wording
Condition-blind controls exist: GeneCIS reports image-only and text-only rows and uses conditional distractors; the
2026 CIR audit shows single-modality shortcuts; C-STS scores one pair under two conditions, a text swap test. Claim the
matched control (drop only the examples), condition gain (zero for any scorer ignoring them) and R@1 = (either rate +
gain)/2. Cite the others as motivation; do not claim to be first to test for condition-blind shortcuts.

### 3.3 The method (claim 3): high for each part, low for the combination
Episodes from label-free partitions are CACTUs; inferring the task mode from the support set is MMAML; condition
weights without labels are SCE-Net and DiscoverNet; choosing among clusterings by user intent is Multi-MaP;
distant emotion supervision is EmotionCLIP. None combines a reader over label-free groupings with a gated term for
example-conditioned cross-modal similarity. Present it as a transparent, label-free method and cite them.

## 4. Limits
Semantic Scholar refused queries (HTTP 429), so ACM MM and journal coverage is thin. Nothing above is UNVERIFIED:
every title, author list and venue was opened on a primary page.
