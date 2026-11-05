# Cross-modal and text-to-text scoring for aspect retrieval: protocols, comparators, published numbers

> Literature check for r22 (r22 decide, step 3, 2026-11-05 16:42; Opus `high`, deep depth: the area is new to our notes). The file name uses
> CoSiR's sequence date. Builds on [2026-10-21](2026-10-21_cvpr_literature_review.md),
> [2026-11-28](2026-11-28_design_l_novelty.md), [2026-12-04](2026-12-04_conditional_similarity_update.md),
> [2026-11-25](2026-11-25_held_claim_check.md) and [2026-12-05](2026-12-05_percept_neighbourhood.md).

Budget used: about 38 of 40 searches, 20 of 20 page reads (4 PDF reads failed: MMArt, CANVAS and MULTI3IR were recovered through HTML; the
Lenses CVPR PDF returned 403 and was not recovered). "Snippet" means seen only in a search result. "Ours" means our own reasoning, not a
source's claim. All "runs in hours" entries are our estimates.

**Main finding:** two works in no earlier note weaken the novelty claim. Lenses (CVPR 2026) is one model with several lens embeddings,
including an emotional lens, scored per lens across modalities. CANVAS (preprint, July 2026) is one model with relation-conditioned embeddings
on art, scored per relation across modalities. Both are supervised by category or relation annotations. "No labels" survives as the difference;
"several aspect spaces across modalities" does not.

## Goal

1. How the field scores aspect-conditional retrieval across modalities and text to text.
2. Which comparators a CVPR reviewer expects, and whether each can run without training.
3. What cross-modal numbers exist on paintings and affect, and whether ours can be comparable.
4. How papers treat captions that name the label, and class-name queries.
5. The closest prior work, 2024 to 2026, and what weakens novelty.

## Q1. Scoring (confidence: high on the families, medium on "no standard protocol")

No standard protocol for aspect-specific cross-modal retrieval was found. The field has separate families, and our two hit rules each match
one. Arora et al. (arXiv 2407.20114, snippet) name the split: instance-level matching against category-level retrieval.

| Family | Hit rule | Metrics | Examples |
|---|---|---|---|
| Known pair | The item's own pair | R@K, MedR, MRR, nDCG | SemArt Text2Art (R@1/5/10, MedR, test set of 1,069); Recipe1M (10 random subsets of 1k and 10k pairs, medR; one paper calls 1k too noisy); MMArt (R@1/5/10, P@k, nDCG, MRR on the full gallery) |
| Known pair among distractors | Own image among random distractors | Accuracy | Affection listener |
| Label sharing | At least one label in common with the query | mAP, P@N, PR curves; some unsupervised work uses mAP@50 | Cross-modal hashing on MIRFlickr and NUS-WIDE, I→T and T→I reported separately. Query set is small (1% or 5% in one setup) |
| Graded human score | Human similarity under a condition | Spearman only, no retrieval metric | C-STS and CASE |
| Same aspect label, image only | Same label on one aspect | mAP, MAP@R | CLAY, CRL, InDiReCT (earlier notes) |
| Aspect coverage | Any supporting document for a perspective in the top k | Hard and soft coverage @k | MULTI3IR; Lenses "lens-coverage" |

**C-STS.** The original set has 18,908 examples and all baselines scored Spearman below 50. Tu et al. (2024) found annotator disagreement on 55%
of the validation set; their re-annotation agreed with the original at 50.4 Spearman. On a re-annotated set (851 test items), CASE (EACL 2026)
reports:

- CASE best: 74.94
- Fine-tuned SimCSE-large: 79.51
- LoRA-tuned Qwen3-8B: 79.91
- Frozen Qwen3-8B plus a small feed-forward network: 71.64

Our reading: C-STS needs graded human ratings, which we lack on art. Cite it, but do not compute it.

**CANVAS and Lenses.** CANVAS's "correct match" is not defined in the text (Recall@K, P@K, nDCG@K). Lenses reports per-lens R@1; its hit rule is
not verified.

**Text to text under an aspect.** AspectCSE and aspect-based document similarity exist (snippets), but neither gives a same-label mAP protocol. No
same-emotion mAP benchmark for text was found.

**Label-sharing mAP on our data (ours, not from a source).**

- A painting with 5 captions holds several emotions. So the "at least one label" rule inflates hits in emotion I→T and T→I.
- Prevalence is skewed: 62% of ArtEmis captions carry a positive emotion. mAP therefore needs a chance row per aspect.

## Q2. Comparators (confidence: medium)

Weights are checked on model cards unless marked otherwise. Our features for the first two rows already exist.

**a. Text to image and image to text**

| Model | Weights | Hours, no training? | Why a reviewer asks |
|---|---|---|---|
| CLIP L/14, B/32; SigLIP; SigLIP 2 | Public | Yes, cached | Floor in every neighbouring paper |
| Qwen3-VL-Embedding-2B, plain and with an aspect instruction | Apache-2.0 | Yes | Our own backbone, and the zero-training answer to "why train?" |
| GME-Qwen2-VL-2B-Instruct | Apache-2.0 | Yes | Sibling instruction embedder; used in MULTI3IR, MMEB-V3 |
| VLM2Vec (TIGER-Lab, Full and Qwen2VL) | Referenced in the MMEB leaderboard; licence not checked | Yes | PercepT baseline; 58.9 to 75.5 R@5 with prompts on COCO-Facet |
| Jina-CLIP-v2 | CC BY-NC 4.0 | Yes | MMArt's second model |
| BGE-VL (MegaPairs) | Public per the 2026-10-21 note | Yes | Baseline in Lenses |
| CRL basis on CLIP | Code public | Yes, but cross-modal use is our adaptation; the paper is image only | Strongest text-named method in our notes |
| CANVAS | Code public; weights not mentioned | No: needs relation edges and partial fine-tuning | Quote its published rows |
| Lenses | Code "Coming Soon"; data by request at most | No | Quote its published rows |
| PaintingCLIP (preprint 2602.11349); Long-CLIP or BLIP on paintings (SIGIR 2025) | Not confirmed | No | Art-specific; report only |

CSD is image only, so it does not apply here. PercepT has no retrieval and no released code.

**Evidence that instructions can fail.** MMEB-V3 (preprint) finds instruction shifts "insufficient or misaligned" in Qwen3-VL-Embedding, VLM2Vec and
GME. CtrlBench reports VLM2Vec 2.95 and GME-2B 20.92 (earlier notes). So an instruction row needs testing; it is not an automatic ceiling.

**b. Text to text**

| Model | Weights | Hours, no training? | Note |
|---|---|---|---|
| Qwen3 and CLIP text towers | Public | Yes, cached | Floor |
| all-MiniLM-L6-v2 | Public | Yes | Generic sentence embedder; PercepT's BERTopic and MMArt's retrieval index use it |
| RoBERTa GoEmotions (SamLowe), 28-d probabilities | MIT | Yes | Emotion only. Model-card F1 differs across revisions, so do not quote one number |
| INSTRUCTOR-large | Apache-2.0 | Yes | Instruction-tuned, covers all three aspects |
| InBedder | Checkpoints on Hugging Face; licence per checkpoint not checked | Yes | Instruction-following (ACL 2024) |
| ArtEmis BERT text-to-emotion classifier | Notebook, trains on labels | Hours, supervised | Supervised ceiling (65.7%, earlier note) |
| CASE | Not checked | Needs C-STS training | Report only |

## Q3. Published numbers (confidence: high on SemArt and MMArt, medium on the Lenses row, medium on the negative for ArtEmis)

| Work | Data, gallery | Direction | Best numbers | Comparable? |
|---|---|---|---|---|
| Garcia, Vogiatzis 2018 (paper) | SemArt, test 1,069 | T→I; I→T | R@1 0.144; 0.138; R@10 about 0.455 | Yes for a known-pair hit, if the text input matches |
| Same (repo README) | Same | T→I | CML 0.164/0.386/0.505, MedR 10; CCA 0.117/0.283/0.377, MedR 25 | Conflicts with the paper; the cause is not stated |
| Garcia et al. 2019 (1904.10615; Table 1, Att&ContextNet, Author) | Same; gallery "test samples", size not stated | T→I; I→T | 0.247/0.477/0.581, MR 6; 0.212/0.446/0.563, MR 7 | Only if attributes are used too; it encodes title, comment and one-hot metadata |
| CANVAS 2026 (preprint), Table 1 | SemArt+, 1,069 test images, 5,169 test texts, original split | T→I R@5/R@10 | CLIP 0.305/0.385; SigLIP 0.334/0.423; SigLIP-ft 0.337/0.446; CANVAS 0.376/0.489 | No: hit undefined, several texts per image |
| Same | Same | I→T R@5/R@10 | CLIP 0.078/0.111; SigLIP 0.095/0.133; SigLIP-ft 0.102/0.150; CANVAS 0.645/0.714 | No |
| Same | WikiArt+ R@10 | T→I; I→T | SigLIP-ft 0.307; CANVAS 0.217 (not best); CANVAS 0.781 | No |
| MMArt (preprint) | 1,000 query paintings, gallery 74,234 | T→I R@1 / MRR | Narrative 44.0% / 0.537; formal 7.8%; historical 2.2% / 0.037; unified 45.1% / 0.548 | Yes for known pair, on Qwen3-VL-Embedding-2B with the same 1,000 queries |
| Affection (CVPR 2023) | Random distractors | T→I accuracy | CLIP B/32, not fine-tuned: 97.9% (1 distractor), 89.7% (10); custom LSTM + ResNet 91.1%, 63.4% | Only with distractor sampling, not a full gallery |
| Lenses (CVPR 2026, project page only) | Lenses test set, size unknown | I→T R@1, emotional lens / overall | BGE-VL zero-shot 14.8 / 70.7; fine-tuned 33.8 / 80.9; Lenses 40.4 / 89.1 | No: closed data |

Further points:

- **MMArt caveats.** The Emotional single-perspective R@1 appears only in Figure 5, not in the text. The text does not say which of its two models
  gives each number.
- **ArtEmis and ArtELingo.** No published retrieval numbers (three notes and two searches here). Our rows there would be first, with nothing to
  compare against.
- **CLIP on Text2Art.** No frozen-CLIP run on the original Text2Art task was found. CANVAS's frozen CLIP R@10 of 0.385 is on a different task, so it
  cannot be set against 0.581.
- **Human bound on SemArt.** 10-way choice among random paintings 0.889, same-type paintings 0.714; best model 0.830 and 0.680.
- **Generated captions (ours).** MMArt captions were generated from the image, so known-pair hits may run higher than on human captions. Treat the
  two as different difficulty levels.

## Q4. Captions that name the label; word queries (confidence: medium, both are negative results)

**Masking is not standard in retrieval.** No retrieval paper found masks label words. Affection does not describe it; SemArt, MMArt and CANVAS do not
mention it in what we read. Lenses is not known.

Precedents and motivation come from elsewhere:

- WASSA-2018 IEST masks the emotion word in every tweet. Macro F1 ran 21% to 71%; the bag-of-words baseline reached 60%.
- VL-Taboo: zero-shot models succeed mainly when class labels appear in the pretraining text.
- The CIR audit (Attimonelli et al.): 32.2% to 83.6% of queries are solvable from one modality.
- ArtEmis 2.0: a naive nearest-neighbour model did abnormally well, attributed to unbalanced emotions and generic captions.

So masking is a defensible robustness row that we would have to justify ourselves. No figure was found for the share of ArtEmis captions that name
their emotion. MMArt's emotional captions draw on ArtEmis utterances and the dominant emotion for 99.0% of paintings, so the leak carries over.
Whether they contain the word is not stated.

**Word queries: no controlled comparison found.**

- CRL, CLAY and InDiReCT use short text to steer image-to-image similarity. RCML and CANVAS put relation text into the query.
- CANVAS gave CLIP, SigLIP and MSC the form "{relationship}: {text}" and found it did not help.
- Rodriguez et al. (ConQA) find CLIP weaker on conceptual queries than descriptive ones.
- Hashing uses real text as queries, not class names.

Ours: a word query is zero-shot classification read as retrieval. It gives one query per class, so it should be a separate row with mAP averaged over
classes, not pooled with caption queries.

## Q5. Closest prior work (confidence: medium; the Lenses paper body was not read)

| Work | Venue | One model, several spaces? | Across modalities? | Supervision | Remaining gap to us |
|---|---|---|---|---|---|
| **Lenses** (Alomari, Asgarov, Thomas) | CVPR 2026 | Yes: lens tokens on an MLLM, same-lens masking, global fallback | Yes, per-lens R@1 reported | Category-aware loss on annotated lenses | Lens labels; interpretive lenses, not style or genre |
| **CANVAS** (Schaerf, ..., Garcia) | Preprint, July 2026 | Yes: one relation-conditioned embedding per pair | T2I and I2T; per-relation T2I in a figure | Relation-labelled edges; last 3 blocks fine-tuned | Labels; no emotion |
| RCML (Qiao, Hu, Zhao) | arXiv 2508.17497; venue not checked | Relation-conditioned | Yes, product data | Named relations | No art, no discovery |
| PVSE (Song, Soleymani) | CVPR 2019 | Multiple diverse embeddings | Yes | Instance pairs | Slots are not aspects |
| TPIPS, CLAY, CRL | 2025 to 2026 | One conditional model | Image only | Text-named | Earlier notes |
| SLiCS | Preprint 2508.20322 | Concept subspaces | Image only at inference | Text pseudo-labels in its unsupervised variant | No cross-modal scoring |

**Strongest comparator.** In print: Lenses, because it is peer-reviewed at the target venue and has an emotional lens. In our domain: CANVAS, whose last
author also made SemArt. Strongest one we can run: Qwen3-VL-Embedding with an aspect instruction, since it needs no training.

**What still appears new (our reading).** Aspects learned with no aspect or relation labels, scored across modalities on cross-item same-aspect hits.
This depends on two unchecked facts: whether Lenses and CANVAS already score cross-item hits. CANVAS may do so if attribute text nodes are shared by
images. Also, the earlier note's "the intersection looks empty" (2026-12-04, Q5) now holds only for aspects that are discovered, not for multi-space
cross-modal models.

## Sources

| Title | Authors, year | Venue | URL | Shows (question) |
|---|---|---|---|---|
| Lenses: Toward Polysemous Vision-Language Understanding | Alomari, Asgarov, Thomas 2026 | CVPR 2026 | https://openaccess.thecvf.com/content/CVPR2026/html/Alomari_Lenses_Toward_Polysemous_Vision-Language_Understanding_CVPR_2026_paper.html ; https://hanialomari.github.io/lenses | Lens model, per-lens numbers from the project page (Q1, Q3, Q5) |
| Art Beyond Semantics (CANVAS) | Schaerf et al. 2026 | Preprint | https://arxiv.org/html/2607.16321 | Relation-conditioned art embeddings, Table 1 (Q1 to Q5) |
| MMArt | Wang, Ding et al. 2026 | Preprint; project page says ACM MM 2026 Dataset Track, the dataset README says under review | https://arxiv.org/html/2608.10706 ; https://shuaiwang97.github.io/MMArt | Retrieval baseline, emotional grounding (Q3, Q4) |
| How to Read Paintings | Garcia, Vogiatzis 2018 | ECCV Workshops 2018 | https://arxiv.org/abs/1810.09617 ; https://github.com/noagarcia/SemArt | Text2Art protocol; paper and repo conflict (Q3) |
| Understanding Art through Multi-Modal Retrieval in Paintings | Garcia et al. 2019 | Venue not stated on the page read | https://ar5iv.arxiv.org/html/1904.10615 | Best Text2Art rows, human bound (Q3) |
| Affection | Achlioptas et al. 2023 | CVPR 2023 (per notes) | https://ar5iv.labs.arxiv.org/html/2210.01946 | Listener numbers (Q3, Q4) |
| C-STS | Deshpande et al. 2023 | EMNLP 2023 | https://arxiv.org/abs/2305.15093 | Task, Spearman (Q1) |
| Linguistically Conditioned STS | Tu et al. 2024 | Peer-reviewed proceedings per the Brandeis listing; venue not read | https://arxiv.org/abs/2406.03673 | Annotation problems (Q1) |
| CASE | Zhang, Zhou, Bollegala | EACL 2026 | https://arxiv.org/html/2503.17279v5 | Spearman on the re-annotated set (Q1) |
| Deep Cross-modal Hashing via Margin-dynamic-softmax Loss | 2020 | Preprint listing, snippet | https://arxiv.org/pdf/2011.03451 | "At least one label" rule (Q1) |
| Cross-media retrieval paper | 2017 | Snippet | https://arxiv.org/pdf/1703.03567v1.pdf | Same rule on Wikipedia, NUS-WIDE (Q1) |
| Image-text retrieval subfields | 2024 | Snippet | https://arxiv.org/pdf/2407.20114v3 | Instance against category (Q1) |
| Recipe retrieval papers | Various | Snippets | https://arxiv.org/pdf/1804.11146 ; https://arxiv.org/pdf/1911.12763 | 1k/10k sampling, medR (Q1) |
| MULTI3IR | Song, Kim, Kim 2026 | Preprint | https://arxiv.org/html/2608.30949 | Coverage metrics (Q1) |
| COCO-Facet | Li, Gao, Du 2025 | NeurIPS 2025 | https://arxiv.org/abs/2505.15877 | Attribute-focused T→I (Q1, Q2) |
| MMEB-V3 | Huang et al. 2026 | Preprint | https://www.emergentmind.com/papers/2604.23321 | Instruction failures (Q2) |
| Qwen3-VL-Embedding-2B card | Qwen 2026 | Model card | https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B | Apache-2.0, 2048-d (Q2) |
| GME, RoBERTa GoEmotions, Jina-CLIP-v2, INSTRUCTOR, InBedder cards | Various | Model cards | https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct ; https://huggingface.co/SamLowe/roberta-base-go_emotions ; https://huggingface.co/jinaai/jina-clip-v2 ; https://huggingface.co/hku-nlp/instructor-large ; https://arxiv.org/pdf/2402.09642.pdf | Weights, licences (Q2) |
| IEST: WASSA-2018 | Klinger et al. 2018 | WASSA 2018 | https://arxiv.org/abs/1809.01083 | Masked emotion word (Q4) |
| VL-Taboo | 2022 | Snippet | https://arxiv.org/pdf/2209.06103.pdf | Class names in pretraining text (Q4) |
| ArtEmis 2.0 | Mohamed et al. 2022 | CVPR 2022 | https://arxiv.org/pdf/2204.07660v1.pdf | Nearest-neighbour shortcut, imbalance (Q4) |
| ConQA | Rodriguez et al. | Snippet | https://vbn.aau.dk/en/publications/does-theperformance-oftext-to-image-retrieval-models-generalize-b | Conceptual queries (Q4) |
| RCML | Qiao, Hu, Zhao 2025 | arXiv 2508.17497; venue not checked | https://arxiv.org/html/2508.17497v1 | Relation-conditioned CLIP (Q5) |
| PVSE | Song, Soleymani 2019 | CVPR 2019 | https://arxiv.org/abs/1906.04402 | Multiple embeddings (Q5) |
| SLiCS | Li et al. 2025 | Preprint | https://arxiv.org/html/2508.20322v1 | Concept subspaces (Q5) |
| Augmenting Vision-Language Retrieval | Bell, Gore, Mansouri 2025 | SIGIR 2025 | https://dl.acm.org/doi/10.1145/3726302.3730167 | Painting retrieval, numbers not seen (Q2) |
| PaintingCLIP | 2026 | Preprint, snippet | https://arxiv.org/abs/2602.11349 | Art-tuned CLIP (Q2) |

## Conflicts and corrections to the notes

- SemArt T→I: paper R@1 0.144 against README 0.164; R@10 about 0.455 against 0.505.
- SemArt split: 19,244 + 1,069 + 1,069 = 21,382, not the stated 21,384.
- MMArt status: the project page says ACM MM 2026; arXiv names no venue; the dataset README says under review.
- MMArt embeddings: the abstract snippet says released; the project page is silent.
- Lenses data: "Data" link to a form in one reading, "Coming Soon" in another.
- The 2026-10-21 note says Affection listener results sit in a figure. The numbers are in the text (Q3 table).

## Open

1. **Lenses paper body:** hit rule, label source, T→I and text-to-text rows. The CVF PDF returned 403, so open the PDF and supplement by hand. Settle
   before the abstract, because the novelty wording depends on it.
2. **CANVAS hit rule:** read the eval code in the `artistic_sheaf` repo, and check whether attribute nodes create cross-item hits.
3. **MMArt Emotional R@1, and which model gives which number:** read them from the figure data, or ask the MMArt authors.
4. **Share of ArtEmis captions naming their emotion:** count it directly, or see the repo's linguistic notebook.
5. **SIGIR 2025 numbers:** ACM page.
6. **RCML venue and PaintingCLIP weights:** check their pages.
7. **Frozen CLIP on the original Text2Art:** no paper found; a Google Scholar cited-by search or our own run settles it.
8. **New arXiv papers:** search again in the week before the abstract, per the 2026-11-25 note.

## Ranked list

**Comparators to run**

1. Frozen backbones, no instruction (Qwen3-VL-Embedding-2B, CLIP L/14, SigLIP 2, SigLIP): the floor; cached; appears in every neighbouring paper.
2. Qwen3-VL-Embedding-2B with an aspect instruction on both sides: the strongest zero-training answer to "why train?".
3. RoBERTa GoEmotions probabilities for text to text: the natural rival on emotion, and free to run.
4. INSTRUCTOR-large with an aspect instruction: the instruction-tuned text row for all three aspects.
5. GME and VLM2Vec instructed: reviewers know them, and they show whether instruction response varies by model.
6. Jina-CLIP-v2 on MMArt: matches that dataset's second model.
7. Class-name prompts on frozen CLIP and Qwen3: the word-query row.
8. CRL basis on both sides of CLIP: the text-named method in our notes, adapted (ours).
9. Quote only: CANVAS, Lenses, Affection listener, Text2Art rows. These cannot be run in hours, or their protocols differ.

**Metrics to report**

1. mAP with the same-aspect hit, per aspect and per direction (I→T, T→I, T→T, with I→I kept): matches the hashing protocol and our existing tables.
2. A prevalence (chance) row beside every label mAP: mAP depends on class share, and emotion is skewed.
3. Known-pair R@1/5/10 and MedR, per direction: the Text2Art standard.
4. MRR and nDCG on MMArt at gallery 74,234 with 1,000 queries: the only setup directly comparable with a published baseline.
5. The strict case for emotion hits (ours): when an image holds several caption labels, state the rule, since "at least one label" inflates hits.
6. Masked and unmasked caption rows: not standard, but WASSA, VL-Taboo and the CIR audit give the motivation.
7. Word-query mAP averaged over classes, as a separate row: one query per class, so it should not be pooled with caption queries.
