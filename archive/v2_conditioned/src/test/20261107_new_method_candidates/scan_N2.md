# Three-way scan for N2 (find-then-select cascade)

Date 2026-10-04. Mode: ARS deep-research `three-way-scan`. Candidate: N2 in `candidates_draft.md`. Read scope for every paper below: **abstract_only** (arXiv or proceedings abstract page opened via WebFetch; full texts not read). Method weaknesses are therefore "not assessed" throughout, per rule 5.

Searches run (WebSearch, 2026-10-04): "GeneCIS benchmark general conditional image similarity"; "composed image retrieval two-stage coarse-to-fine reranking VLM CIRR FashionIQ"; "retrieve then rerank cascade fusion vs reranking trade-off retrieval analysis short list"; "relevance feedback reranking few-shot examples image retrieval query expansion top-k rerank"; "hybrid retrieval score interpolation versus cascade reranking top-k noisy signal"; "few-shot composed image retrieval support examples aspect attribute-conditioned rerank CLIP". Fetched pages: arxiv.org abs pages and the NeurIPS/mlanthology pages listed below. Not searched: Google Scholar, Semantic Scholar, ACM/IEEE databases, OpenReview, 2026 CVPR/ECCV proceedings in depth.

## GeneCIS: A Benchmark for General Conditional Image Similarity
Source: arXiv 2306.07969 (CVPR 2023, highlight) | Year: 2023 | Link: https://arxiv.org/abs/2306.07969
- WHY: a fixed embedding encodes one notion of similarity; users want similarity under a condition (focus or change an attribute or object).
- HOW: zero-shot benchmark with open-set text conditions; the proposed model is trained on mined caption data. Baselines are CLIP combiners.
- WHAT: CLIP baselines are weak, and ImageNet accuracy is only weakly correlated with GeneCIS accuracy. No retrieve-then-rerank design is part of the benchmark's baselines per the abstract.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Candidate Set Re-ranking for Composed Image Retrieval with Dual Multi-modal Encoder
Source: arXiv 2305.16304 (TMLR) | Year: 2023 (final 2024) | Link: https://arxiv.org/abs/2305.16304
- WHY: composed retrieval needs fine interaction between reference, modification text and candidate, which is too costly to run on the whole gallery.
- HOW: stage 1 prunes candidates with vector distance; stage 2 re-ranks the survivors with a dual-encoder attending over (reference, text, candidate) triplets. Trained.
- WHAT: reports consistent gains over prior state of the art on CIRR and FashionIQ (abstract). Stage 2 is a strong learned condition-dependent scorer; the cascade is justified by cost, not by a weak-signal argument.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Training-free Zero-shot Composed Image Retrieval with Local Concept Reranking
Source: arXiv 2312.08924 | Year: 2023 (rev. 2024) | Link: https://arxiv.org/abs/2312.08924
- WHY: avoid triplet training data for composed retrieval.
- HOW: stage 1 global retrieval (query converted to text); stage 2 local concept re-ranking that emphasises discriminative details of the modification instruction. Training-free.
- WHAT: comparable to supervised methods and clearly above other training-free ones on CIRR, CIRCO, COCO and FashionIQ. This is the closest in form to N2: training-free, condition-free-ish first stage, condition-specific rerank.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Incorporating Relevance Feedback for Information-Seeking Retrieval using Few-Shot Document Re-Ranking
Source: arXiv 2210.10695 (EMNLP 2022) | Year: 2022 | Link: https://arxiv.org/abs/2210.10695
- WHY: users give a few relevant documents; neural rerankers ignore them.
- HOW: lexical retrieval, neural rerank, then kNN similarity to the feedback documents and meta-learned cross-encoders tuned on them, fused with the rerank score.
- WHAT: feedback-conditioned reranking beats competing methods by a substantial margin on four converted IR datasets (abstract). Closest to "few examples define the condition, applied to a short list", in text, with relevance labels rather than aspect-agreement examples.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Re-Ranking for Image Retrieval and Transductive Few-Shot Classification
Source: NeurIPS 2021 (Shen, Xiao, Hu, Sbai, Aubry) | Year: 2021 | Link: https://mlanthology.org/neurips/2021/shen2021neurips-reranking (index page that was opened; the NeurIPS proceedings page for this paper was not opened)
- WHY: initial similarity from pre-trained features is imperfect; the gallery structure can refine it.
- HOW: meta-learned re-ranking updates on the similarity graph.
- WHAT: gains on CUB, Cars, SOP and three few-shot benchmarks. A precedent for reranking a retrieved set using few-shot structure; the condition here is class identity, not an aspect.
  - Method weaknesses: not assessed (read scope: abstract_only).

## An Analysis of Fusion Functions for Hybrid Retrieval
Source: arXiv 2210.11934 | Year: 2022 | Link: https://arxiv.org/abs/2210.11934
- WHY: how to combine lexical and semantic scores.
- HOW: compares convex combination and Reciprocal Rank Fusion.
- WHAT: convex combination beats RRF in and out of domain and needs few tuning examples. This is the fusion side of the fusion-versus-cascade comparison; it does not test cascades.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Cross-paper synthesis
- **Common WHY.** A cheap global score is good at finding the neighbourhood, and a condition-specific signal is too costly or too noisy to apply everywhere.
- **Divergent HOW.** Learned triplet rerankers (2305.16304), training-free concept or text reranking (2312.08924), few-shot feedback rerankers (2210.10695), graph re-ranking (Shen 2021), versus score-level fusion (2210.11934). Only the feedback paper conditions on a handful of examples; none conditions on within-pair agreement across support pairs with a value-disjoint anchor.
- **Strongest WHAT.** Two-stage pruning then condition-aware rerank is a standard and effective design in composed retrieval. Convex fusion is a robust one-stage alternative in hybrid text retrieval.
- **Unresolved gap.** None of the abstracts compares cascade against fusion when the conditioning signal is weak, or reports how often the target survives the first stage as a ceiling. I found no paper that argues a weak condition term helps more inside a short list than in full-list fusion.

## Verdict for N2
- **Exists already?** Partially. The architecture (condition-free first stage, top-k, condition-dependent rerank) exists in the composed-retrieval papers above, so the mechanism is not novel. The specific instantiation (aspect inferred from 4 support and 4 contrast pairs, factor-agreement term, displacement-of-negatives motivation) was **not found**. Search bound: the six queries and sources listed above, abstract-level only; this is not a claim of exhaustive absence.
- **Strongest prior a reviewer would cite.** Liu et al., Candidate Set Re-ranking for CIR (arXiv 2305.16304), together with Sun et al., Local Concept Reranking (arXiv 2312.08924) as the training-free form. Add Baumgartner et al. (arXiv 2210.10695) for example-conditioned few-shot reranking.
- **Evidence for our regime.** No published evidence found that a weak condition signal gains more in a short list than under fusion. The generic cascade-benefit claims seen in search results came from blog-level or RAG benchmark snippets that I did not verify and do not cite. The only verified fusion result is that tuned convex combination is strong (2210.11934), which is the comparator N2 must beat (the existing nested score is itself a convex-style fusion). So N2's value is an empirical question for our data, bounded by the diagnostic already in the draft (how often p_A and p_B both reach the top k).
- **Stronger published variant to adopt?** None found for the training-free, weak-signal setting. The learned triplet reranker (2305.16304) is stronger but needs training data in our setting that the project does not have. If N2 is run, frame it as a cheap control on cascade versus fusion, not as a method contribution.

## AI-assistance note
This scan was produced by an AI agent (Claude) using web search and abstract-level page fetches only. Every listed paper's existence was confirmed by opening its arXiv or index page; no full texts were read, so method weaknesses are not assessed. The Shen et al. entry was verified through an index page, not the official proceedings page. A human should read the full texts of 2305.16304, 2312.08924 and 2210.10695 before citing them.
