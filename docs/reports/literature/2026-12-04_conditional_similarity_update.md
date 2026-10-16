# Discovered aspects as retrieval conditions: GeneCIS state of the art, discovered conditions, the bridge, benchmarks

> Literature check for the new line (buddy vs PercepT, GeneCIS as the core idea), step 3 of the r10 decide chat,
> 2026-10-17 00:53 (Amsterdam). Sonnet, standard depth: the 2026-10-21 and 2026-11-28 notes already cover GeneCIS,
> CSN, SCE-Net and multiple clustering, so this note adds only what is new or contradicts them. The file name uses
> CoSiR's sequence date (2026-12-04). Builds on [2026-10-21](2026-10-21_cvpr_literature_review.md) and
> [2026-11-28](2026-11-28_design_l_novelty.md); the PercepT check beside it is
> [2026-12-05](2026-12-05_percept_neighbourhood.md).

Budget: all 20 searches used; about 7 page fetches (3 returned little: two arXiv PDFs were too large, one patent PDF
was an unreadable scan). "P" = preprint. "Snippet" = seen only in a search snippet, unverified.

## Goal

1. State of the art on GeneCIS and the approaches behind it.
2. What is new on conditions that are discovered, not given.
3. The bridge: discovered aspects used for conditional retrieval.
4. Benchmarks with several aspect labels.
5. How crowded the area is, and the strongest comparator.

## Q1. GeneCIS today (confidence: medium; no clean leaderboard exists)

- The GeneCIS repo is archived and has no v1. Papers use five table formats, so rows cannot be compared directly (our 2026-10-21 note, §2.2).
- No four-task average above 19.7 was found for a frozen CLIP-class model. That figure is SQUARE with an MLLM reranker (16.4 without it).
- TPIPS (P, 2607.18237, Table 4) calls CRL fine-tuned on GeneCIS "the state-of-the-art". CRL's object half is the only part reported.
- An instruction-conditioned MLLM embedder is a strong zero-shot row on the focus tasks. TPIPS ran Qwen3-VL-8B-Embedding: focus attribute 30.85, focus object 18.11.

| Method | Year, venue | Approach | GeneCIS R@1 |
|---|---|---|---|
| GeneCIS Combiner | 2023, CVPR | RN50x4 fine-tuned on mined triplets | avg 16.8 (17.6 with ViT-B/16) |
| CIReVL | 2024, ICLR | training-free, caption + LLM, CLIP B/32 | avg 15.9 |
| OSrCIR | 2025, CVPR | training-free, GPT-4o, CLIP B/32 | avg 17.4 (reproduction disputed) |
| STiTch | 2026, P 2605.21261 | training-free, transport, CLIP B/32 | avg 18.4 |
| SQUARE | 2025, P 2509.26330 | MLLM captions + reranker, B/32 | avg 19.7 |
| CRL pretrained / fine-tuned | 2025, NeurIPS | criterion text to LLM basis, B/16 | focus object 15.4 / 19.7 (TPIPS T4) |
| DIOR | 2026, WACV (venue new) | training-free LVLM prompt, I to I | focus attribute 24.0, focus object 21.1 |
| Qwen3-VL-8B-Embedding (zero-shot) | 2025, P | instruction embedder | focus attribute 30.85, focus object 18.11 (TPIPS T4) |
| TPIPS | 2026, P 2607.18237 | Qwen3-VL-8B-Emb fine-tuned on human odd-one-out votes | focus attribute 31.35, focus object 19.95 |

Rows 1 to 5 come from our earlier note; the last four rows are the new ones. Not checked: the full four-task numbers
of CLAY (CVPR 2026) and DeCIR (P 2605.08389, 16.5 per the earlier note); the CLAY fetch was truncated.

## Q2. Discovered conditions: what is new since the notes (confidence: medium)

- **Multiple clustering with MLLMs.** All of these cluster and do not retrieve.
  - X-Cluster / OpenSMC (CVPR 2026 Findings) finds criteria itself. It was already in our notes.
  - ESMC (Wang, Jia, Liu, Hou; AAAI 2026, 40(31):26588) selects MLLM token embeddings for a user criterion. Code is public.
  - MLLMMC (AAAI 2026) is the same family (snippet only).
  - Agent-Centric Personalized Multiple Clustering (P 2503.22241, Mar 2025) uses MLLM agents on a graph. It is tested only on Card Order and Card Suits.
- **TPIPS (P 2607.18237, in our notes) matters more than the notes say.**
  - An LLM proposes the aspects per triplet, a VLM refines them, and humans prune them. So aspects are LLM-discovered.
  - They are given as text at test time, and the model is trained on about 1M human votes.
  - It is image-to-image, not image-text.
- **Label-free condition learning is still the old line.** SCE-Net (ICCV 2019), DiscoverNet (CVPR 2022) and Generalized CSL (TPAMI 2025) are image-only and triplet-supervised. No 2026 paper on this was found.
- **CRL, CLAY, DIOR, FocalLens, SteerViT** all take a text condition (see our notes and Q1).

## Q3. The bridge (confidence: medium that no end-to-end work exists; the search was incomplete)

No work found does "discover aspects from image-text data, then retrieve conditioned on a discovered aspect". Closest works:

- **(a) Discover, then condition.**
  - X-Cluster, ESMC and TGAICC (RepL4NLP 2024) discover or take criteria but stop at clusters. None scores retrieval.
  - TPIPS (above) is the nearest in pipeline shape. It differs in being supervised, image-to-image and trained on human votes. It does not learn aspects from the data.
  - A US patent, 12670695, "Conditional similarity-based image identification and matching with reduced labels" (snippet only; the PDF is an unreadable scan). A VLM assigns pseudo-labels for attributes that are defined in advance, and a Conditional Similarity Network (CSN) is trained on them. Attributes are defined, not discovered. Unverified; its owner and date are unknown.
  - A 2017 paper (arXiv 1703.09928, snippet) learns one embedding space per latent aspect from triplets without aspect labels. Image-only and old. Unverified.
- **(b) Evaluate discovered clusters by retrieval.** None found. X-Cluster scores clustering accuracy. CLAY scores mAP on Stanford40 mood, whose labels came from IC|TC clustering. That is retrieval scored against clustering-derived labels, and it is the nearest case.
- **(c) Affect or perceptual topics with conditional similarity.** None found.
  - Affective image retrieval (Yao ICCV 2019; Yang AAAI 2018) uses one query and no condition.
  - MosAIc (Hamilton et al., PMLR v133, 2021) does conditional retrieval for art, with user-supplied filters such as culture or medium. It is image-only, pre-CLIP and has no affect.
  - PercepT has no retrieval.

## Q4. Benchmarks (the earlier note's §2.7 and design-L §6 have the base list; additions here)

| Dataset | Aspect labels | Size, licence |
|---|---|---|
| COCO-4c (X-Cluster) | activity 64, location 19, mood 20, time 6; COCO captions exist (join not checked) | 5,000 val2017 images; licence "follows source licences" (snippet) |
| Food-4c | food type, cuisine, course, diet; no captions | 25,250 (earlier note) |
| Clevr-4 | texture, shape, colour, count; synthetic | CC BY 4.0 (earlier note) |
| EmoArt (P 2506.03652, 2025) | 56 styles, 5 visual attributes, valence/arousal, 12 emotions | about 132.7k paintings (a later paper says 132,895); annotations made by GPT-4o (snippet); licence not found |
| WikiArt, SemArt | genre, style, artist; SemArt adds type, school, timeframe, comments | SemArt 21,384, non-commercial research |
| ArtEmis / ArtELingo | emotion, captions | our current data |
| Multiple-clustering sets (Fruit, Card, CMUface, Cars, Flowers) | 2 to 4 facets each | too small, no captions |
| CUB + Reed captions, CelebA, UT-Zappos50k | attributes; Zappos is the CSN benchmark | see the earlier note |

EmoArt is new since the notes and is the only art set with style, content and emotion on the same images. Its labels
come from a model, not from people.

## Q5. Crowdedness (confidence: medium)

- **Text-given conditional similarity is crowded.** CRL (NeurIPS 2025), DIOR (WACV 2026), CLAY (CVPR 2026), FocalLens, SteerViT, TPIPS, COCO-Facet (NeurIPS 2025) and Fioresi et al. (ECCV 2026) are all in 2025 to 2026.
- **Aspect discovery by multiple clustering is also active:** X-Cluster, ESMC, MLLMMC and the 2024 Multi-MaP / Multi-Sub.
- **The intersection looks empty.** No paper found takes discovered aspects as the condition for image-text retrieval. A reviewer can build the obvious baseline: X-Cluster criteria named by an LLM, then fed to CRL or CLAY. Expect that baseline.
- **Strongest comparator: CRL** (NeurIPS 2025). It is training-free on frozen CLIP, has public code, reports GeneCIS, and is the stated state of the art on its object half. Run it with the aspect names produced by X-Cluster. Close seconds are CLAY (includes a mood task) and TPIPS (human-judged aspects).

## What this means for the new line

1. Claim the combination "aspects learned from image-text data without text names, used as retrieval conditions". Each part has prior art (see Q2).
2. Compare against LLM-named aspects fed to CRL, CLAY and an instruction embedder (Qwen3-VL-Embedding); the last scored 30.85 / 18.11 zero-shot on GeneCIS focus tasks.
3. Report GeneCIS as four R@1 columns, and state the backbone (B/32 frozen: 15.9 to 19.7).
4. Add one benchmark with image-text pairs and several labelled aspects: COCO-4c (mood) or an art set (EmoArt, SemArt).
5. Check before the 2026-11-10 abstract whether the patent and ESMC-style work have an image-text retrieval follow-up.

## Open (budget gone)

- Full CLAY and DeCIR GeneCIS tables: fetch arXiv HTML of 2604.11539 from offset 100000 and the supplement.
- Patent 12670695: owner, date, claims; look up on Google Patents.
- Whether COCO-4c IDs join to COCO captions and its licence; read the X-Cluster project page.
- Search "image-text retrieval conditioned on discovered clusters" at ICCV 2025 and NeurIPS 2025 proceedings.

## Sources

- [TPIPS, arXiv 2607.18237 (P)](https://arxiv.org/html/2607.18237): GeneCIS Table 4, aspects discovered by LLM, Qwen3-VL row (Q1, Q3, Q5).
- [CLAY, CVPR 2026, arXiv 2604.11539](https://arxiv.org/pdf/2604.11539): mood task, text condition (Q2, Q3). Truncated fetch.
- [Conditional Representation Learning (CRL), NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119073) and [arXiv 2510.04564](https://arxiv.org/html/2510.04564v2): comparator (Q1, Q5).
- [DIOR, WACV 2026, arXiv 2512.21860](https://wacv.thecvf.com/virtual/2026/poster/654): venue confirmed (Q1).
- [STiTch, arXiv 2605.21261 (P)](https://arxiv.org/pdf/2605.21261), [DeCIR, arXiv 2605.08389 (P)](https://arxiv.org/pdf/2605.08389), [ZeroSight, arXiv 2606.07032 (P)](https://www.alphaxiv.org/abs/2606.07032): GeneCIS rows, snippets only (Q1).
- [X-Cluster / OpenSMC, CVPR 2026 Findings, arXiv 2410.05217](https://arxiv.org/html/2410.05217v5): discovery, COCO-4c, Food-4c (Q2, Q4).
- [ESMC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39867) and [MLLMMC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40066): multiple clustering (Q2). MLLMMC snippet only.
- [Agent-Centric Personalized Multiple Clustering, arXiv 2503.22241 (P)](https://arxiv.org/abs/2503.22241): abstract page read (Q2).
- [TGAICC, arXiv 2406.18589](https://arxiv.org/html/2406.18589v1): prompt-guided alternative clustering (Q3).
- [SCE-Net, ICCV 2019, arXiv 1908.08589](https://arxiv.org/pdf/1908.08589): label-free conditions (Q2).
- [US patent 12670695](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12670695): snippet only, unverified (Q3).
- [Multi-aspect embeddings, arXiv 1703.09928](https://ar5iv.arxiv.org/html/1703.09928): snippet only (Q3).
- [MosAIc, PMLR v133, 2021](https://proceedings.mlr.press/v133/hamilton21a.html): conditional retrieval for art (Q3).
- [Yao et al., ICCV 2019, affective image retrieval](https://openaccess.thecvf.com/content_ICCV_2019/html/Yao_Attention-Aware_Polarity_Sensitive_Embedding_for_Affective_Image_Retrieval_ICCV_2019_paper.html): (Q3).
- [PercepT, arXiv 2606.03345 (P)](https://www.emergentmind.com/papers/2606.03345): abstract-level facts only (Q3).
- [EmoArt, arXiv 2506.03652 (P)](https://arxiv.org/html/2506.03652v1): dataset (Q4); size and GPT-4o annotation from snippets.
