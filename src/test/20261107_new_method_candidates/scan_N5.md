# Three-way scan: N5 (infer the aspect name with an MLLM, then embed with the name)

Date: 2026-10-04. Mode: ARS deep-research `three-way-scan`. Candidate: show an MLLM the 4 support and 4 contrast image-caption pairs, ask it to name what the support pairs share that the contrast pairs do not, then use the phrase as the instruction of an instruction-conditioned multimodal embedder and rank candidates with it.

Search bound. Tools: WebSearch (about 14 queries) and WebFetch of arXiv abstract pages (plus the VisDiff v2 HTML page). Queries covered: VisDiff / set difference captioning; Qwen3-VL-Embedding; GeneCIS / conditional similarity; LLM concept inference from image sets; CIReVL / caption-LLM-retrieve; think-then-embed; visual instruction inversion; instruction-following multimodal embedders; MLLM few-shot common-attribute / Bongard reasoning; MLLM-generated instruction then instruction-conditioned embedding; user-specified similarity criteria (CLAY, MCMR). Not searched: Google Scholar, ACL Anthology, OpenReview directly, Semantic Scholar; GME, E5-V, VLM2Vec and MM-Embed papers were not individually opened. Every paper below was opened on its arXiv page this session. **Read scope for all entries: abstract_only (the fetch tool returns a model-written summary of the page, not the full text), except VisDiff, where the results table and limitations were returned from the v2 HTML page (scope: sections: results, limitations; table values are the fetch summary's, not independently re-read).**

## Shortlist

## Describing Differences in Image Sets with Natural Language (VisDiff)
Source: arXiv 2312.02974, CVPR 2024 (oral) | Year: 2024 | Link: https://arxiv.org/abs/2312.02974

- WHY: Humans cannot sift thousands of images to see how two sets differ; the task is to state the difference in words (Set Difference Captioning).
- HOW: A proposer captions images (BLIP-2) and a language model (GPT-4) proposes candidate difference phrases; a CLIP ranker re-ranks them by how well each separates the two sets. Caption-based proposing beat image-based (LLaVA) and feature-based proposing.
- WHAT: On VisDiffBench (187 paired sets) the caption proposer plus CLIP ranker gets Acc@1 of 88 / 75 / 61 % on Easy / Medium / Hard and 78 % on ImageNet-R subsets; on Hard, image-based LLaVA proposing gets 28 % and feature-based 12 %.
  - Method weaknesses: (a) author-acknowledged (limitations, per fetch summary): abstract concepts are hard and errors of CLIP, GPT and BLIP propagate; (b) reader-inferred: the sets differ in one dominant attribute by construction, whereas in our episodes the contrast pairs share a different aspect B, so a phrase must separate A from B among distractors, which the benchmark does not test; (c) reader-inferred: the caption bottleneck drops non-describable cues such as mood. Section-level locators not verified beyond the fetch summary.

## Vision-by-Language for Training-Free Compositional Image Retrieval (CIReVL)
Source: arXiv 2310.09291, ICLR 2024 | Year: 2024 | Link: https://arxiv.org/abs/2310.09291

- WHY: Compositional image retrieval needs triplet training data; a modular training-free alternative is wanted.
- HOW: Caption the reference image with a generative VLM, have an LLM rewrite the caption according to the edit text, retrieve with CLIP. All reasoning stays in language.
- WHAT: Competitive or state-of-the-art on four zero-shot CIR benchmarks, in part beating supervised methods; intervenable because the intermediate text is readable.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Qwen3-VL-Embedding and Qwen3-VL-Reranker
Source: arXiv 2601.04720 | Year: 2026 | Link: https://arxiv.org/abs/2601.04720

- WHY: One unified embedding and reranking stack for text, image, document and video retrieval.
- HOW: Qwen3-VL backbone; multi-stage training from contrastive pre-training to reranker distillation; instruction-aware (embedding quality depends on the instruction given); 2B and 8B; Matryoshka dimensions; a cross-encoder reranker companion.
- WHAT: The 8B embedder scores 77.8 on MMEB-V2, reported first among all models as of the paper's date. This is the embedder N5 would call.
  - Method weaknesses: not assessed (read scope: abstract_only). The abstract reports no test of instructions naming abstract aspects such as emotion or style.

## Think Then Embed: Generative Context Improves Multimodal Embedding (TTE)
Source: arXiv 2510.05014, ICLR 2026 | Year: 2025/2026 | Link: https://arxiv.org/abs/2510.05014

- WHY: Encoder-only MLLM embedders weaken when instructions become complex or need compositional reasoning.
- HOW: A reasoner MLLM first writes a reasoning trace about the query, then an embedder encodes the query plus the trace; a smaller reasoner is fine-tuned on embedding-centric traces; unified single-model variants studied.
- WHAT: State of the art on MMEB-V2, above proprietary models; the fine-tuned small reasoner gives a 7 % absolute gain over recent open models. This is the nearest published form of "generate words, then embed conditioned on them", though the generated text is a rationale for one query, not a name inferred from example sets.
  - Method weaknesses: not assessed (read scope: abstract_only).

## CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space
Source: arXiv 2604.11539, CVPR 2026 | Year: 2026 | Link: https://arxiv.org/abs/2604.11539

- WHY: Human similarity depends on the viewer's focus, but retrieval systems use one fixed metric.
- HOW: Training-free; recasts a pretrained VLM embedding space as a text-conditional similarity space so the user's text condition modulates fixed image embeddings; builds the CLAY-EVAL synthetic set.
- WHAT: Multi-condition retrieval with precomputed visual embeddings. Conditions are supplied as text; the fetch summary shows no inference of the condition from examples.
  - Method weaknesses: not assessed (read scope: abstract_only). Evaluation set is synthetic (author-stated in the abstract summary).

## Supporting papers (opened, not in the shortlist table)

- GeneCIS (arXiv 2306.07969, CVPR 2023): conditional similarity benchmark; CLIP baselines are weak and the condition is given as text.
- ABC (arXiv 2503.00329, TMLR 2025) and CtrlBench: VLM-backbone embedder steered by instructions. A search snippet (not the abstract page) reports R@1 on CtrlBench of 0.0 (UniIR), 9.7 (MagicLens) and 24.0 (VLM2Vec); I did not verify these numbers on the paper.
- Promptable Embeddings for Attribute-Focused Retrieval (arXiv 2505.15877, NeurIPS 2025): prompting MLLM retrievers with the needed attribute helps; 15 % R@5 gain with predefined prompts on COCO-Facet. The prompts are author-written, concrete visual attributes.
- Reasoning Limitations of MLLMs, Bongard Problems (arXiv 2411.01173, ICML 2025): 8 MLLMs find shared concepts across image sets better on real-world than on synthetic problems; performance on classical Bongard problems is poor.
- Show Me Examples: Inferring Visual Concepts from Image Sets (VICIS, arXiv 2607.02402, 2026): infers a concept from an image set into embeddings (not words) for generation; states state-of-the-art VLMs perform poorly on the task.
- Visual Instruction Inversion (arXiv 2307.14331, NeurIPS 2023): recovers a text edit instruction from one before-and-after pair by optimisation.

## Cross-paper synthesis

Shared WHY: similarity is not fixed, and a language handle on "which aspect" is attractive because it is readable and works with frozen models. Divergent HOW: (i) language bottleneck from examples (VisDiff, Visual Instruction Inversion, CIReVL), (ii) text-conditioned embedders with a user-given condition (CLAY, GeneCIS, ABC, Qwen3-VL-Embedding, Promptable Embeddings), (iii) generate-then-embed (TTE), (iv) non-verbal set-to-concept inference (VICIS). N5 is the composition (i) then (ii), with episodes where the contrast set carries a second aspect. Remaining gaps in what we found: no paper in our search tests a naming step on aspects that are abstract (emotion, style, mood), cross-modal (image and caption pairs), and discriminated against a contrast set sharing another aspect. VisDiff is the only one that quantifies naming accuracy, and it flags abstract concepts as hard.

## Verdict for N5

**Partially exists, not found as a single published pipeline.** Within the stated search bound (queries and sources above) I found no paper that infers an aspect phrase from few support and contrast example pairs with an MLLM and feeds it as the instruction of an instruction-aware embedder for example-conditioned retrieval. Each half exists: set-to-phrase naming (VisDiff, which uses a contrast set and a CLIP re-ranker) and phrase-conditioned embedding (CLAY, Qwen3-VL-Embedding, ABC). Absence is bounded by the search above; GME, E5-V, VLM2Vec and MM-Embed were not individually checked, and the search used no OpenReview or Semantic Scholar queries.

Strongest prior work a reviewer would cite: VisDiff (naming from contrast sets, with a CLIP verification step), plus CIReVL (language as the interface to a frozen retriever) and TTE (MLLM-generated text then conditioned embedding). The reviewer's line: "N5 is VisDiff's proposer glued to a conditional embedder; novelty lies only in the regime."

Published evidence for our regime:
- Naming works when the difference is describable (VisDiff Acc@1 88 / 75 / 61 % by difficulty) and degrades on abstract concepts (author-acknowledged).
- MLLMs are weak at shared-concept inference across image sets in the Bongard study (read at abstract level only), consistent with our 8B in-context probe (R@1 +1.07, gain +0.21).
- Conditional embedders are tested mostly on concrete attributes with human-written conditions (Promptable Embeddings, CLAY, CtrlBench); I found no evidence that they follow abstract-aspect instructions such as emotion or style. Our own result (true names to CLIP gave +1.5 R@1) is the only data point for abstract aspects, and it bounds N5: a perfect naming step through a CLIP-class embedder is worth about +1.5, so N5's ceiling depends on the instruction-conditioned embedder being much stronger than CLIP at the named aspect. That is untested in the literature we found.

Stronger published variants to consider instead: (1) VisDiff-style generate-and-verify: sample several candidate phrases, then pick the one whose embedder scores best separate support from contrast (a built-in check that costs no labels); (2) TTE-style reasoner-plus-embedder in one model if N5 underperforms. Both are reader suggestions from the papers, not published tests of this task.

AI-assistance note: this scan was run by an AI agent (Claude Sonnet 5.5) using web search and page fetches. Page contents came from a fetch tool that summarises pages, so figures are second-hand and read scope is abstract-level except as marked; the CtrlBench numbers come from a search snippet and are unverified. A human should open the papers before citing any number.
