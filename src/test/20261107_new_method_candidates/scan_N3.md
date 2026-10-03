# Three-way scan for N3 (grouped concept basis)

Date: 2026-10-04. Mode: ARS deep-research `three-way-scan`. Candidate: N3 in `candidates_draft.md` (SpLiCE-style concept codes, LLM-grouped into aspect groups, group chosen by within-pair agreement across 4 support vs 4 contrast pairs, query and candidate compared inside that group).

**Read scope, stated once.** Every paper below was verified to exist by opening its arXiv abstract page (or the CVPR/NeurIPS page where noted) with WebFetch; I read abstracts and the fetch summaries only, not full texts. Per the method-weaknesses rule 5, every WHAT entry therefore says "not assessed (read scope: abstract_only)". Statements about SpLiCE Appendix B.3 come from the project's own `2026-10-28_citation_check.md` (which read the PDF locally), not from this scan.

## SpLiCE: Interpreting CLIP with Sparse Linear Concept Embeddings
Source: arXiv / NeurIPS 2024 | Year: 2024 | Link: https://arxiv.org/abs/2402.10376

- WHY: CLIP's dense vectors are not interpretable; the paper wants a post-hoc decomposition without concept labels.
- HOW: sparse non-negative recovery of a CLIP embedding as a linear combination of text embeddings of a fixed concept vocabulary; training-free, task-agnostic.
- WHAT: sparse codes keep downstream performance and support spurious-correlation detection and model editing. The vocabulary is flat: no aspect groups, no condition.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Conditional Representation Learning for Customized Tasks (CRL)
Source: arXiv / NeurIPS 2025 | Year: 2025 | Link: https://arxiv.org/abs/2510.04564

- WHY: universal embeddings capture dominant semantics (object) and miss the criterion a user cares about (habitat, colour).
- HOW: for a user-given criterion text, an LLM writes descriptive words that serve as the basis of a criterion-specific space; the VLM image embedding is projected onto that basis. Frozen backbone.
- WHAT: gains on customized classification, clustering and retrieval. This is the closest published form of "concept basis per aspect, compare inside it". The criterion is given as text; it is not inferred from examples.
  - Method weaknesses: not assessed (read scope: abstract_only).
- Follow-up: SP-CRL, arXiv 2602.05464 (https://arxiv.org/abs/2602.05464), reports that VLMs are not trained to disentangle criteria, so projection leaks semantics, and adds basis decomposition and null-space projection. Abstract only; the leakage claim is the authors'.

## CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space
Source: arXiv / CVPR 2026 | Year: 2026 | Link: https://arxiv.org/abs/2604.11539

- WHY: retrieval metrics are fixed while human similarity depends on the aspect of interest.
- HOW: training-free modulation of a frozen VLM similarity space by a text condition (species, location, action, colour), with fixed visual embeddings.
- WHAT: competitive conditional retrieval and a synthetic benchmark (CLAY-EVAL). Condition is text; no concept vocabulary groups, no example-based inference.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Learning Similarity Conditions Without Explicit Supervision (SCE-Net); Conditional Similarity Networks (CSN)
Source: arXiv / ICCV 2019 | Year: 2019 | Link: https://arxiv.org/abs/1908.08589 (CSN, CVPR 2017: https://arxiv.org/abs/1603.07810)

- WHY: images are similar under several notions at once; the notion should not need to be supplied.
- HOW: CSN learns masks over embedding dimensions selected by a known notion id. SCE-Net treats the condition as a latent variable and infers the mixture of learned masks from the compared items.
- WHAT: SCE-Net beats supervised conditional methods on three fashion datasets. Conditions are learned subspaces inferred per triplet, not a human-readable grouped vocabulary, and not inferred from labelled example pairs.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Language in a Bottle (LaBo)
Source: arXiv / CVPR 2023 | Year: 2023 | Link: https://arxiv.org/abs/2211.11158

- WHY: concept bottleneck models needed manual concept lists.
- HOW: GPT-3 writes candidate concept sentences per class, CLIP scores images against them, a submodular utility picks discriminative, diverse concepts, and a linear layer classifies.
- WHAT: 11.7% above linear probes at 1 shot, comparable with more data. It is an LLM-generated CLIP concept bank, but organised by class, not by aspect.
  - Method weaknesses: not assessed (read scope: abstract_only).
- Related, verified but not shortlisted: Label-free CBM (ICLR 2023, listed by search, ICLR page not opened; the project citation check holds it), Discover-then-Name (ECCV 2024, https://arxiv.org/abs/2407.14499).

## Describing Differences in Image Sets with Natural Language (VisDiff)
Source: arXiv / CVPR 2024 | Year: 2024 | Link: https://arxiv.org/abs/2312.02974

- WHY: finding what separates two image sets is manual.
- HOW: captions propose candidate difference descriptions via an LLM; CLIP re-ranks how well each separates the two sets.
- WHAT: the nearest published "infer the distinguishing attribute from example sets", but it outputs a phrase for a human and does not use it to condition retrieval.
  - Method weaknesses: not assessed (read scope: abstract_only).

## Cross-paper synthesis

- Shared WHY: similarity should depend on an aspect, and CLIP's single score does not say which. CRL, CLAY, CSN and SCE-Net all attack this.
- Divergent HOW on the three N3 ingredients:
  1. Concept codes: SpLiCE and LaBo give flat or class-organised CLIP concept scores.
  2. Aspect-specific basis: CRL builds one per criterion with an LLM, and SP-CRL tries to remove leakage between criteria.
  3. Condition source: CSN uses an id, CRL and CLAY a text, SCE-Net a latent inferred from the compared items, VisDiff a text inferred from two image sets. None infers it from within-pair agreement of cross-modal example pairs.
- Remaining gaps: a pre-grouped vocabulary whose group is selected by example agreement, and value-disjoint transfer across modalities. I found no paper that does both.

## Verdict for N3

**Partially exists; the full combination was not found.** Search bound: WebSearch queries (standard mode, Oct 2026) on SpLiCE, LaBo, label-free CBM, conditional similarity networks, "condition-aware retrieval concept vocabulary", "conditional image similarity CLAY / CRL", "infer attribute from examples analogy retrieval", "concept bottleneck retrieval concept-grouped", "VisDiff", "CLIP art style emotion"; page fetches on arxiv.org and the CVPR 2017 page (403). Not searched: Google Scholar, Semantic Scholar, OpenReview, ACL anthology, proceedings of ICCV/ECCV 2026 not yet indexed. Absence is therefore bounded to those queries and sources.

- **What a reviewer cites against novelty.** (1) CRL (arXiv 2510.04564): an LLM-written descriptive basis per criterion, projection of CLIP embeddings, retrieval; N3's "grouped basis, compare inside the group" is that with the group chosen by a different rule. (2) SpLiCE for the sparse codes; the project's citation check says its App. B.3 already applies codes to captions. (3) SCE-Net for inferring the condition from the compared items without labels. (4) VisDiff for naming the distinguishing attribute from example sets. N3's own distinct pieces: inferring the group from within-pair agreement across pairs, and testing on values absent from the examples; neither is claimed by the abstracts I read.
- **Evidence relevant to our regime (limited).** I found no paper showing that concept-grouped CLIP codes support attribute-conditioned retrieval from examples, few-shot attribute-type inference, or per-group image-versus-caption matching. CRL and CLAY report text-conditioned gains on object, colour and action style attributes, which supports "concept basis per aspect works when the aspect is named and concrete".
- **Known failure modes for abstract attributes.** Kazmierczak et al., arXiv 2510.07115 (CHILI), state that CLIP hallucinates concepts, predicting presence or absence from context, which undermines CLIP-concept bottlenecks (abstract only). Widhoelzl and Takmaz, arXiv 2405.06319 (CogSci 2024), report CLIP's zero-shot emotion recognition on abstract art is above baseline but modest and misaligned with humans, with colour-emotion pairings stronger than in human annotators. A source I opened for CLIP's zero-shot art-style performance (arXiv 2605.18974) did not state a figure in the fetched summary, so I make no claim about style. SP-CRL's leakage claim suggests concept groups will not be clean in CLIP space; this is the authors' claim from an abstract.
- **Stronger variant to adopt.** None found that does the inference. For the basis side, CRL with SP-CRL's purification is the stronger published way to build per-aspect subspaces; N3 could reuse it as a comparator or component rather than hand-rolled groups. CRL should be a named baseline (it needs a criterion text, so run it with the true aspect name as a privileged reference).

## AI-assistance note

This scan was run by an AI agent (Claude) with WebSearch and WebFetch. Fetched pages were summarised by a small model before I saw them, so quoted claims are second-hand from abstracts; verify against the PDFs before any citation in the paper. Retrieved text was treated as data.
