# Three-way scan for N4 (amortized, meta-learned conditioner)

Date 2026-10-04. Mode: ARS `three-way-scan`. Candidate: a set encoder over 4 support and 4 contrast cross-item image-caption pairs emits an episode-specific projection or gating of frozen CLIP features; trained episodically on k-means pseudo-aspect episodes plus distant supervision; tested on value-disjoint episodes of held-out labelled aspects.

**Read scope (honest).** Every paper below was opened at its arXiv abstract page. For the six shortlisted papers I also fetched the ar5iv HTML full text, but through a fetch tool that returns a model-written summary of the page, not the raw text, so section numbers and quotes are as reported by that summary and not checked line by line. Our local PDF extraction failed (no PDF tool), so no PDF was read directly. Treat the scope as `sections` (method, ablation and limitation sections as reported), not full text. Papers outside the shortlist were read at abstract level only (`abstract_only`).

## Shortlist

## Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions (FEAT)
Source: CVPR 2020 (Ye, Hu, Zhan, Sha) | Year: 2020 | Link: https://arxiv.org/abs/1812.03664

- WHY: a fixed embedding learned on seen classes is not discriminative for the classes of a new task.
- HOW: a set-to-set function (best: Transformer) maps the task's support embeddings to task-specific ones (§4.1), trained jointly with the embedding on N-way M-shot episodes from seen classes (§3, Alg. 1).
- WHAT: state of the art on two benchmarks; Transformer beat BiLSTM, DeepSets and GCN (§5.2.1, Table 1); Clipart to Real World domain-shift test 30.89 vs ProtoNet 29.47 (§5.3.1).
  - Method weaknesses: no search for a limitations section was successful in the fetched text, so no author-acknowledged item is listed. Reader-inferred: the episodes are class-discrimination tasks with a fixed label space style, so a conditioner trained this way could encode "separate these classes", not "find what pairs share"; the condition is read from single items with labels, not from cross-item pairs.

## Finding Task-Relevant Features for Few-Shot Learning by Category Traversal (CTM)
Source: CVPR 2019 (Li, Eigen, Dodge, Zeiler, Wang) | Year: 2019 | Link: https://arxiv.org/abs/1905.11116

- WHY: metric few-shot learners use the same features for every task.
- HOW: a concentrator extracts intra-class commonality and a projector uses all support classes at once to build an inter-class uniqueness mask over features, trained end to end episodically (§3.1, §3.2.1, §3.2.2).
- WHAT: miniImageNet 62.05 (1-shot) and 78.63 (5-shot); 3 to 4 point gains when plugged into metric learners (Table 3).
  - Method weaknesses: author-acknowledged: slower inference, 0.0688 vs 0.0632 s per episode (as reported in the fetched summary; locator not recovered). Reader-inferred: the mask is read from commonality within a class, close to our "agree within pairs, vary across pairs" rule, so it is the nearest mechanism, but it is evaluated only with test classes drawn like training classes.

## TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning
Source: NeurIPS 2018 (Oreshkin, Rodriguez, Lacoste) | Year: 2018 | Link: https://arxiv.org/abs/1805.10123

- WHY: a metric space shared by all tasks is suboptimal.
- HOW: a task encoder takes the mean of class prototypes and predicts FiLM scale and shift for the embedding layers (§2.2), with an auxiliary co-training task.
- WHAT: 1-shot 58.5, 5-shot 76.7 on mini-ImageNet; task conditioning alone gave 75.6 vs 74.2 and only paid off with auxiliary training (§3.4, Table 3).
  - Method weaknesses: author-acknowledged (§2.4): "overly challenging" to optimize the conditioner and the filters together, which needed co-training. Reader-inferred: mean pooling of prototypes drops the pair structure that carries the aspect in our task.

## Unsupervised Learning via Meta-Learning (CACTUs)
Source: ICLR 2019 (Hsu, Levine, Finn) | Year: 2019 | Link: https://arxiv.org/abs/1810.02334

- WHY: learn a fast learner without labels.
- HOW: k-means on unsupervised embeddings, P partitions with random dimension scaling (§2.3), tasks sampled as N clusters, then MAML or ProtoNets.
- WHAT: beats embedding-only baselines; Omniglot 20-way 5-shot 73.36 vs Oracle-MAML 96.29 (Tables 1-2, "substantial to severe" label penalty, §4.2).
  - Method weaknesses: author-acknowledged (§5): the meta-training task distribution is not human-designed to mimic evaluation, and clusters can be "uninterpretable" or follow image artifacts (Fig. 2). Reader-inferred: evaluation classes are still class-membership tasks; a gap in aspect structure (our case) is untested.

## Revisiting Unsupervised Meta-Learning via the Characteristics of Few-Shot Tasks
Source: IEEE TPAMI (Ye, Han, Zhan; per arXiv listing) | Year: 2020 (arXiv 2011.14663) | Link: https://arxiv.org/abs/2011.14663

- WHY: pseudo-class tasks differ from real semantic tasks.
- HOW: augmentation-based pseudo-classes plus a Transformer set-to-set adapter, keeping a pre-adapted embedding for test (§III, §V-B).
- WHAT: the adapted embedding does better on pseudo tasks, the pre-adapted one on real tasks (Table IX); the gap "makes the embedding... hard to fit" (§V-B).
  - Method weaknesses: author-acknowledged (§V-B): distribution gap between pseudo and real tasks. Reader-inferred: it is the closest direct evidence against N4, because N4's set-encoder is exactly the component that overfits pseudo tasks here.

## Making Text Embedders Few-Shot Learners (BGE-EN-ICL)
Source: arXiv (Li, Qin, Xiao et al.) | Year: 2024 | Link: https://arxiv.org/abs/2409.15700

- WHY: embedders ignore task examples.
- HOW: 0 to 5 in-batch demonstrations placed on the query side with an instruction; trained contrastively (§3.1).
- WHAT: state of the art on MTEB and AIR-Bench, with gains on tasks absent from training (+1.43 QA, +1.08 long-doc, Tables 2-3); text only.
  - Method weaknesses: author-acknowledged: passage-side prompts hurt most tasks (§4.3 to 4.5). Reader-inferred: examples are query-passage pairs defining a task, so the demonstrated pairs share the query's relation value; the value-disjoint, cross-modal setting is not tested.

## Cross-paper synthesis

Shared WHY: a single metric space under-serves tasks, so read the task from a support set (FEAT, CTM, TADAM, BGE-EN-ICL) and adapt the embedding. Divergent HOW: Transformer set function (FEAT), commonality and uniqueness masks (CTM), FiLM from pooled prototypes (TADAM), instruction plus demonstrations in a decoder (BGE-EN-ICL). Unsupervised task construction is CACTUs and Ye et al. Related lines read at abstract level only: SCE-Net (https://arxiv.org/abs/1908.08589, ICCV 2019) and DiscoverNet (https://arxiv.org/abs/2204.04053, CVPR 2022) infer a condition mixture from the compared items without condition labels; Contextual Visual Similarity (https://arxiv.org/abs/1612.02534) fits per-dimension weights from a positive and a negative image; MARS (https://arxiv.org/abs/2210.00312) is example-pair analogy over knowledge graphs; CLAY (https://arxiv.org/abs/2604.11539, CVPR 2026) modulates VLM similarity from text; UMTRA (https://arxiv.org/abs/1811.11819) and PsCo (https://arxiv.org/abs/2303.00996) are pseudo-task meta-learning; Adaptive Cross-Modal Few-Shot Learning (https://arxiv.org/abs/1902.07104) mixes text and image prototypes.

Gaps that remain across all of them: (a) the condition is read from support items that share the query's value (class or relation), not from cross-item pairs whose values differ from the query's; (b) cross-modal image-caption pairs as the support set; (c) transfer from pseudo-partition episodes to held-out labelled aspects. Only Ye et al. measure pseudo-to-real transfer for a set-function adapter, and they find it hurts.

## Verdict for N4

**Partially exists.** The mechanism (a meta-trained set encoder that conditions a frozen embedding per episode, with pseudo-task episodes built by clustering) exists in parts: FEAT and CTM for the set-conditioned embedding, TADAM for FiLM conditioning, CACTUs for k-means pseudo-episodes, and Ye et al. for the combination of the two. We found no paper that applies it to the cross-modal, cross-item, value-disjoint aspect task, but that is the same novelty basis as the task itself, so N4 adds little beyond it.
Search bound: WebSearch (standard and extended) on about 12 queries covering FEAT, TADAM, CrossTransformers, CACTUs and UMTRA, unsupervised meta-learning gap, BGE-EN-ICL and multimodal in-context embedders, MARS, Contextual Visual Similarity, conditional similarity with meta-learning, and few-shot cross-modal retrieval; arXiv abstract pages for 16 papers; the three project reports. Not searched: Google Scholar, Semantic Scholar, OpenReview, ACL and CVF listings directly, and anything after the search engine's index of early October 2026. The absence claim is therefore bounded to those queries and sources.

**Strongest citations a reviewer would use against novelty.** FEAT (set-to-set embedding adaptation) first, then CTM (commonality and uniqueness masks, the same signal as our agreement rule), then CACTUs for the pseudo-task recipe, and BGE-EN-ICL for the in-context framing. Ye et al. would be cited against the method's soundness, not its novelty.

**Published evidence for our regime.** Pseudo-task meta-training transfers partly in class-discrimination settings (CACTUs gets well above embedding baselines, but 73.36 vs 96.29 oracle on Omniglot 20-way 5-shot); the authors flag the task-distribution mismatch (§5). Ye et al. show a Transformer adapter improved pseudo-task accuracy and not real-task accuracy (Table IX). Generalization to unseen condition types: SCE-Net reports generalization to unseen fashion categories (abstract only), and BGE-EN-ICL to unseen text tasks (AIR-Bench), but no paper tests unseen condition types for an image-caption conditioner. Cross-modal versions: only the text-and-image prototype mixing paper, which is not conditioning on pairs. Our own K8 failure and the E3 weakness of pseudo-partitions fit this picture, so the published evidence predicts N4 learns a selector among training partitions.

**Stronger published variant to adopt instead.** None replaces N4 for our task. If N4 is run, the design that follows from the literature is Ye et al.'s: keep a condition-free pre-adapted pathway and add the set-encoder as a residual or gate, initialised to identity (TADAM's delta regime), so a failed conditioner degrades to the control. Our own N1 and N2 remain cheaper first tests.

## AI-assistance note

This scan was run by an AI agent (Claude) using web search and page fetches. Page content came through a summarising fetch tool, so quotes and section numbers should be checked against the PDFs before any appears in a paper. All listed links were opened or returned by search in this session. No reference was included that could not be opened.
