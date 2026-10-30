# Passing the teacher: second sources, retrieval-specific training, noisy targets, framing

> Literature check for the CoSiR condition-free line (research-loop step 3), a side job of r12 decide that prepares r13. Written 2026-10-30 by the `deep-research` agent; the file name uses CoSiR's sequence date (2026-12-30).
> The last section, "Controller's note", was added by the r12 decide chat, not by the agent.
> Depth: standard (Sonnet). Budget: 18 of 20 searches, 10 of 10 page reads. Four reads returned raw PDF bytes or a 403 (Born-Again PDF, Zhao et al. BMVC 2018, Laskar and Kannala ACCV 2020, arXiv 2510.18680); the Born-Again numbers were redone through ar5iv. Every page went through a summariser; no paper was read in full.
> Notes read first: `2026-12-20_ways_to_improve.md` (all), `2026-12-26_relational_genre_term.md` (all), `2026-12-17_cluster_level_losses.md` (Summary), and a grep of `2026-12-07_prompted_aspect_views.md` and `2026-11-25_held_claim_check.md` for CLAY. Their content is not repeated.
> Conventions: "abstract" or "snippet" = seen only in an abstract or a search summary. "Not checked" = not seen at its source. Peer-reviewed work and preprints are told apart.

## Summary

1. **Q1, second sources (confidence: medium on the numbers, low on transfer to us).** No source shows a frozen-input, label-free student passing a one-teacher target on retrieval by a clear margin. Where students pass a teacher, the gains are small or they come from a second source or from labels:
   - two-teacher ensemble over one teacher: +2.4 ImageNet and +1.5 Flickr30k (MobileCLIP);
   - student over its weaker teacher only: +0.35 MTEB average (Jasper);
   - same-size student with true labels in the loss: +0.8 to +1.3 error points on CIFAR-100 (Born-Again).
   Nothing found uses captions as privileged information for an image-retrieval student.
2. **Q2, retrieval-specific training (confidence: low-medium).** One label-free, retrieval-style case passes its teacher: RKD on Cars-196, R@1 82.50 against 77.17 (+5.33). It fails on CUB-200 (-0.57 against the teacher, 60.67 against 61.24). The teacher there is weak, and the student re-trains a whole backbone. Training a network to copy a diffusion re-ranking is reported to beat the diffusion itself, but only the abstract was seen (BMVC 2018). No ranking or listwise loss has a published ablation against the teacher's cosine retrieval.
3. **Q3, noisy and near-chance targets (confidence: low).** The evidence is for supervised students only. The distilled VLM of EmoVLM-KD passes its ViT teacher by +1.56 on EmoSet and +10.65 on FI, but it is a larger model with the prompt and labels. Theory says a same-size student can beat its teacher under label noise, but with labels in the loss. We found no label-free result on emotion.
4. **Q4, framing (confidence: medium).** "Level at a fraction of the cost" is an accepted contribution when the cost is measured and the comparison set is complete. MobileCLIP-S0 equals OpenAI ViT-B/16 on the 38-dataset average (58.1 against 58.1) at about 4.8x lower latency. A CVPR reviewer for multi-aspect retrieval will likely ask for a CLAY-style training-free projection of the default embedding (one pass too) and a K-pass cost table. No source states a formal non-inferiority margin (as in the 12-26 note).
5. **Most useful finding for r13:** the cheapest lever with published support is to train toward the teacher's read-out (centering plus query expansion), not its raw embedding. This is our reading from Zhao et al. (abstract only). It adds no new information, so it can lift the student to teacher-plus-read-out at best.

## Q1. Second sources that let a student pass its teacher (confidence: medium on numbers)

**Several teachers**

- **MobileCLIP** (Vasu et al., arXiv 2311.17049, preprint; CVPR 2024, venue not checked here; read via the HTML page). ViT-B/16 student, 12.8M-sample scale.
  - Table 2: a two-teacher ensemble raises IN-val 59.3 to 61.7 (+2.4) and Flickr30k 70.5 to 72.0 (+1.5) over the single teacher.
  - Table 15 (8M subset): ensemble 56.3 IN-val and 49.6 average over 38 datasets, against 54.4 and 48.3 for the best single teacher. Flickr30k I2T against the OpenAI teacher: 59.2 against 54.5.
  - The student does not pass its teacher: MobileCLIP-B 76.8 IN-val and 65.2 average (Table 7), against the ensemble teacher 80.1 and 67.3.
  - Synthetic captions: at lambda = 0, IN-val 44.5 to 51.9 and Flickr30k 41.8 to 69.3. Both caption kinds used together give +5.3 IN-val and +15.0 Flickr30k over real captions only (Table 3a). These captions enter the student's own contrastive loss. They are not privileged information for the student.
  - The best single teacher is not the most accurate one: ViT-bigG-14 53.4 against ViT-L-14 DataComp 54.0 (Table 15). This fits Cho and Hariharan (ICCV 2019, in the 12-20 note).
  - MobileCLIP2 (ICLR 2026, [poster page](https://iclr.cc/virtual/2026/poster/10014020)): a better teacher ensemble and captioners add +2.2 IN-1k for the B model (abstract).
- **Jasper and Stella** (Zhang et al., [arXiv 2412.19048](https://arxiv.org/html/2412.19048v2), preprint Dec 2024), Table 1, MTEB 56 tasks, text inputs.

  | Model | Avg | Classification | Clustering | Retrieval |
  |---|---|---|---|---|
  | Jasper (student, 2B) | 71.54 | 88.49 | 58.04 | 61.33 |
  | Stella 1.5B (teacher; also the student's initialisation) | 71.19 | 87.63 | 57.69 | 61.01 |
  | NV-Embed-v2 7B (teacher) | 72.31 | 90.37 | 58.46 | 62.65 |

  The student passes the weaker teacher by 0.35 and stays 0.77 under the stronger one. Losses: cosine, an MSE on pairwise similarity, and a margin ranking hinge (weights 10, 200, 20). The paper gives no per-loss ablation.
- **AM-RADIO, UNIC, PHI-S** are in the 12-20 note (b). Students pass teachers where the teachers are complementary, not on a teacher's home task.
- **Our reading:** a second teacher helps when it adds information. For emotion that could be the caption text. For style, a CLIP or SigLIP 2 image, but the 12-20 note records that CLIP style is weaker (0.300 against 0.395).

**Captions or text as privileged information**

- Nothing found for an image-retrieval student with caption text as the privileged input. The search returned:
  - DCLIP ([arXiv 2505.21549](https://arxiv.org/pdf/2505.21549), preprint, snippet): a teacher with box features the student never sees; no claim that the student passes the teacher.
  - VL2V-ADiP (Addepalli et al., CVPR 2024, snippet): distils a VLM into a vision-only student for out-of-distribution accuracy; no claim that the student passes the teacher.
  - LUGPI ([arXiv 2309.15238](https://arxiv.org/pdf/2309.15238), snippet): text classification, not retrieval.
- Provodin et al. (TMLR 2025, in the 12-20 note) remains the main caution: gains vanish with more than about 1,000 samples or longer training.

**Same-size and self-training students**

- **Born-Again Networks** (Furlanello et al., ICML 2018, [ar5iv](https://ar5iv.labs.arxiv.org/html/1805.04770)). Supervised, with true labels in the loss.
  - CIFAR-100, DenseNet (Table 2): teacher 18.25, 17.69, 17.16, 16.87; student 16.95, 16.69, 16.36, 16.00 (gain 0.80 to 1.30 error points). Two or three students in an ensemble: 15.1 to 15.7.
  - CIFAR-10 (Table 1): gain +0.23 and +0.31 for the two smaller nets; none for the two largest (-0.01, -0.17).
  - No noisy-label test.
- **Noisy Student** (Xie et al., CVPR 2020, [arXiv 1911.04252](https://arxiv.org/pdf/1911.04252), snippet): 88.4% ImageNet top-1, 2.0 above the prior best; pseudo-labels on 300M unlabeled images; the student is as large as or larger than the teacher and gets noise (dropout, stochastic depth, RandAugment). Gain over its own teacher: not checked.
- **Das and Sanghavi** (ICML 2023, [PMLR](https://proceedings.mlr.press/v202/das23d.html), abstract): in regularised linear and logistic regression with label noise, a same-size student can beat the teacher, and the best teacher weight can exceed 1. Labels are in the loss. We did not read the theorems.
- **More unlabeled images and prompt ensembles:** nothing newer than the 12-20 note (PromptKD Fig. 4 levels off; Beyer et al. Fig. 6; CLIP's 80 prompts +3.5%).
- **Our reading:** self-training from the student's own outputs has no label-free precedent on frozen embeddings. The 12-17 note's drift warning applies.

## Q2. Retrieval-specific training on neighbour graphs (confidence: low-medium)

- **RKD** (Park et al., CVPR 2019, [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.05068)). Metric learning, teacher ResNet50-512 trained with a triplet loss, student ResNet18-128 trained with distance plus angle relational losses. The 12-26 note says this setting uses no task loss.

  | Set | Teacher R@1 | RKD student R@1 | Difference |
  |---|---|---|---|
  | CUB-200-2011 (Table 1a) | 61.24 | 59.69 (with l2 norm), 60.67 (without) | -0.57 (without norm) |
  | Cars-196 (Table 1b) | 77.17 | 80.00 (with norm), 82.50 (without) | +5.33 (without norm) |

  On SOP (Tables 2 and 3) a self-distilled ResNet50-512 student gains +1.03, and an RKD ResNet50-128 student +0.92. Conditions: the teacher is weak (small fine-grained sets), the student re-trains a full backbone with pixel augmentation, and the gain depends on the l2-norm choice. This is the only label-free, retrieval-style pass over a teacher we found, and it is not on frozen inputs.
- **Zhao et al.**, "Modelling diffusion process by deep neural networks for image retrieval" (BMVC 2018, [BMVA](https://bmva-archive.org.uk/bmvc/2018/contents/papers/0861.pdf), abstract and search summary only). Unsupervised. Diffusion is treated as a kernel mapping and learned by metric learning. Retrieval then uses plain Euclidean distance. The authors say the result can beat the original diffusion. We did not see the mAP numbers.
- **InsCLR** (AAAI 2022, snippet): self-supervised instance retrieval with pseudo positives from batch and memory bank; 73.1 mAP on ROxford medium against 76.2 for its supervised counterpart. The base descriptor's score was not seen.
- **Laskar and Kannala** (ACCV 2020, [CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Laskar_Data-Efficient_Ranking_Distillation_for_Image_Retrieval_ACCV_2020_paper.html), snippet): the student mimics the teacher's ranking with a black-box teacher; it beats a fully supervised baseline at low sample counts. No student-against-teacher claim seen. The page returned 403, so this is from the search summary.
- **Jasper's ranking hinge** (above): no ablation, so it is no evidence for or against.
- **Not found:** an adapter on frozen embeddings trained on unsupervised kNN-graph positives that beats the base embedding in mAP (one search). NUDGE and PEFA (snippets) need training queries.
- **Our reading:**
  - The teacher's own read-out (centering plus average query expansion k = 10) lifted our teacher to level with c*. The student is therefore at "teacher plus read-out". To pass it, the student needs a signal the read-out does not use.
  - The read-out is a graph operation. Training toward the read-out's output (Zhao's route) would move that gain into the model. After that a second read-out probably adds less (not tested).

## Q3. Noisy and near-chance targets (confidence: low)

- **EmoVLM-KD** ([arXiv 2505.07164](https://arxiv.org/html/2505.07164v1), preprint; CVPR 2025 workshop listing seen as a snippet). The student reads image plus prompt and uses labels. Accuracy, ViT teacher against the distilled VLM (Table 1):

  | Set | ViT teacher | EmoVLM-KD | Difference |
  |---|---|---|---|
  | EmoSet | 78.27 | 79.83 | +1.56 |
  | FI | 68.86 | 79.51 | +10.65 |
  | Emotion6 | 61.95 | 73.91 | +11.96 |

  The gain is smallest on the biggest set. The student is a 7B VLM, so size and input differ from ours. The paper's text and its Table 2 disagree on the distillation module alone (79.51 against 77.37).
- **Multi-teacher distillation for visual emotion** (Cladière et al., Pattern Analysis and Applications 2025): a simple student beats more complex methods (PhD defence abstract only). No numbers seen.
- **Label noise and ambiguity:** soft labels and label-distribution learning are standard for image emotion (Yang et al., AAAI 2017, snippet), and a 2026 preprint ([arXiv 2606.21069](https://arxiv.org/pdf/2606.21069), snippet) argues annotator disagreement is structured, not noise. We found no head-to-head of soft against majority-vote training on one image benchmark.
- **Self-distillation under noise:** Das and Sanghavi above; Born-Again does not test it.
- **Our reading:**
  - No source shows a label-free lever helping on emotion retrieval. All the emotion gains seen are supervised.
  - Emotion mAP of 0.27 against chance 0.20 leaves little room. Differences of 0.001 to 0.002 (our +0.0007 and -0.0023) sit below the seed spread. Report them with several seeds and intervals.

## Q4. Framing "level with the teacher at 1/K the encoder cost" (confidence: medium)

- **MobileCLIP** (above) is the model case. Its claims are latency-accuracy trade-offs at matched accuracy:
  - S0 against OpenAI ViT-B/16: 58.1 against 58.1 average, about 4.8x lower latency (3.1 ms against 14.8 ms), 2.8x fewer parameters.
  - S2 against ViT-B/16: about 2.1x lower latency, more accurate.
  The paper never claims the student passes the teacher.
- **PromptKD** (CVPR 2024, in the 12-20 note): student 83.73 against about 84.4 for the teacher; published as a gain over prior methods, not over the teacher.
- **jina-v5** (preprint 2026; clustering 53.4 against 57.1) frames the student as one base model with a small adapter per task.
- **CLAY** (Apr 2026, [arXiv 2604.11539](https://www.alphaxiv.org/abs/2604.11539.md); CVPR 2026 per the 12-07 note) is the nearest multi-aspect comparator and states a cost axis.
  - Training-free; it keeps one fixed visual embedding per image and adds a text-derived projection per condition.
  - Baselines: CLIP-B, SigLIP-B, GeneCIS, SEARLE, MagicLens, InstructBLIP; Qwen3-VL-Embedding and VLM2Vec in the symmetric comparison.
  - Data and metrics: Stanford40, fine-grained sets, CLEVR4, its own CLAY-EVAL, GeneCIS; mAP, and R@1 to 3 on GeneCIS.
  - Per-query latency for a second condition 0.09 to 0.13 s, against 1.66 s for symmetric GeneCIS and 12.74 s for symmetric Qwen3-VL.
- **Comparators a reviewer may expect (our reading, from the above and the 12-07 and 12-04 notes):**
  1. the default Qwen image (we have it);
  2. the instructed Qwen image, K passes (the teacher);
  3. a CLAY-style projection of the default embedding, also one pass;
  4. a regression probe from the default to the instructed embedding (the 12-20 "information ceiling" idea 1);
  5. PercepT, plus CLIP or SigLIP 2 stores;
  6. a cost table with encoder passes (K against 1), measured latency, and storage. Our student still stores K vectors per image, so the saving is encoder time, not index size.
- **No margin convention:** the 12-26 note found no non-inferiority margin applied to distillation; we found none either.

## Levers ranked

| # | Lever | Extra signal | Best evidence | Cost for us | Fit to limits |
|---|---|---|---|---|---|
| 1 | Train toward the teacher's read-out (centered, query-expanded teacher target); keep the cosine read-out as a check | none new; moves the read-out gain into the model | Zhao et al., BMVC 2018 (abstract: beats the diffusion it copies; numbers not seen) | minutes per run | fits both limits; caps at teacher plus read-out |
| 2 | Second teacher per head: caption text embedding for emotion, CLIP or SigLIP 2 image for style, with teacher dropping | new information (text; another encoder) | MobileCLIP +2.4 IN, +1.5 Flickr over one teacher; Jasper +0.35 over the weaker teacher; RADIO and UNIC (12-20) | about 15 min, stores exist | label-free yes; one read yes; captions were dropped earlier, so the user decides; CLIP style weaker |
| 3 | More unlabeled paintings for the distillation term (the r11 FI images, other affective sets) | more images, same teacher | Noisy Student (snippet, 300M images); Beyer (12-20); PromptKD levels off | extraction of default plus instructed passes (about 0.094 s per instructed image, 12-20 note; not rechecked) | label-free yes; held overlap must be checked |
| 4 | RKD-style relational or ranking loss on the student's own neighbours (queue of teacher anchors, hinge on teacher order) | none new; ranks instead of coordinates | RKD Cars-196 +5.33 R@1, CUB -0.57; Jasper hinge (no ablation) | about 20 min | fits; evidence is from full-backbone students with a weak teacher |
| 5 | Prompt-ensembled instructed teacher (4 to 8 paraphrases per aspect, fixed before runs) | averaged instruction views | CLIP +3.5% (12-20); jina-v5: the instruction matters | about 1 h per aspect and paraphrase pass; raises the bar | fits; the teacher gets stronger too |
| 6 | Born-again second generation (student 1 is the teacher of student 2), or the student's own pseudo-clusters | none new | Born-Again +0.8 to +1.3 CIFAR-100 with true labels; Noisy Student with noise | minutes | no label-free precedent; drift risk (12-17) |
| 7 | Soft-label or distribution targets for emotion | annotator spread | LDL for image emotion (snippet only) | needs per-caption votes, which exist | uses labels or votes, so it breaks label-free; not recommended |

## Sources

| Source | Venue | Used for |
|---|---|---|
| MobileCLIP, [2311.17049](https://arxiv.org/html/2311.17049v2) | preprint; CVPR 2024 (not checked here) | Q1, Q4 |
| MobileCLIP2, [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10014020) | ICLR 2026 (abstract) | Q1 |
| Jasper and Stella, [2412.19048](https://arxiv.org/html/2412.19048v2) | preprint 2024 | Q1, Q2 |
| Born-Again Networks, [ar5iv](https://ar5iv.labs.arxiv.org/html/1805.04770) | ICML 2018 | Q1, Q3 |
| Noisy Student, [1911.04252](https://arxiv.org/pdf/1911.04252) | CVPR 2020 (snippet) | Q1 |
| Das and Sanghavi, [PMLR](https://proceedings.mlr.press/v202/das23d.html) | ICML 2023 (abstract) | Q1, Q3 |
| RKD, [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.05068) | CVPR 2019 | Q2 |
| Zhao et al., [BMVA](https://bmva-archive.org.uk/bmvc/2018/contents/papers/0861.pdf) | BMVC 2018 (abstract) | Q2 |
| Laskar and Kannala, [CVF](https://openaccess.thecvf.com/content/ACCV2020/html/Laskar_Data-Efficient_Ranking_Distillation_for_Image_Retrieval_ACCV_2020_paper.html) | ACCV 2020 (snippet) | Q2 |
| InsCLR (AAAI 2022), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/19930/19689) | AAAI 2022 (snippet) | Q2 |
| EmoVLM-KD, [2505.07164](https://arxiv.org/html/2505.07164v1) | preprint 2025; CVPR workshop (snippet) | Q3 |
| Cladière et al., [lab page](https://mudialbot.univ-avignon.fr/2025/12/01/new-paper-visual-emotion-analysis-using-skill-based-multi-teacher-knowledge-distillation/) | Pattern Anal. Appl. 2025 (defence abstract) | Q3 |
| Quality and agreement in multilabel emotion annotation, [2606.21069](https://arxiv.org/pdf/2606.21069) | preprint 2026 (snippet) | Q3 |
| CLAY, [2604.11539](https://www.alphaxiv.org/abs/2604.11539.md) | preprint Apr 2026; CVPR 2026 per the 12-07 note | Q4 |
| DCLIP [2505.21549](https://arxiv.org/pdf/2505.21549); VL2V-ADiP (CVPR 2024); LUGPI [2309.15238](https://arxiv.org/pdf/2309.15238) | preprint; CVPR 2024; preprint (snippets) | Q1 |

## Not checked

- Teacher-versus-student gain of Noisy Student; Born-Again's venue pages beyond ar5iv; Das and Sanghavi's theorems.
- Zhao et al.'s mAP numbers, Laskar and Kannala's tables, InsCLR's base-descriptor score. A fetch of the arXiv versions (1805.08587 for Zhao et al.) as ar5iv pages would give them.
- Cladière et al.'s numbers (journal DOI 10.1007/s10044-025-01426-9).
- The RKD, MobileCLIP and Jasper numbers came through a summariser, not from the tables directly. The MobileCLIP page was cut at 100,000 of 112,595 characters.
- arXiv 2510.18680 (multi-teacher task-agnostic distillation) returned unreadable PDF bytes; it may bear on Q1 and was not read.
- That the RKD student uses no task loss comes from the 12-26 note, not rechecked.

## Open

- **Does training toward a read-out beat reading out afterwards?** Fetch Zhao et al. as an ar5iv page for its numbers; our own run would settle it in minutes.
- **Caption text as privileged information for an image student:** a search on "privileged information" with "image-text retrieval" in LUPI venues, or the 2510.18680 text.
- **Label-free emotion on ArtEmis or Affection:** no number found; search "ArtEmis" with "image-only emotion" and "retrieval".

## Controller's note: candidate ideas for r13

Added by the r12 decide chat (2026-10-30), from the direction log's parked ideas, our own results and this note. Our
reading; nothing here is tested. r13's decide chat chooses (step 4); the candidates not chosen go to "Parked ideas".

1. **A second teacher that adds information (levers 2 and 5 above).** Caption text for the emotion head, a SigLIP 2
   or CLIP L/14 image for style; both stores exist. The field's passes over a teacher come with a second source
   (MobileCLIP, the 12-20 note). Our own caution: the own-caption terms were dropped in report 12-24, because the
   recipe without them was better on development data (emotion +0.0061). A caption teacher must therefore differ
   from those terms, e.g. graphs built on the caption view (the parked "within-painting caption variation" and
   "aspect words as a caption signal" ideas are forms of it). Cost: minutes to an hour per run on stores that exist.
2. **More unlabeled images for the distillation terms (lever 3).** r12 builds default and instructed Qwen views for
   Affection (85,007 images) and Impressions; r13 could train on them. Same teacher, more data: the gain may be small
   (PromptKD levels off). Conflict to settle first: images r13 trains on are no longer a clean transfer test, so
   r12's confirmation on those datasets would need a held part that r13 never trains on.
3. **Train toward the teacher's read-out (lever 1).** The target becomes the centered, query-expanded teacher
   embedding. It adds no information, so it caps at "teacher plus read-out", but it may lift c*'s untuned (plain
   cosine) retrieval, where emotion is BELOW the teacher (report 12-29, stage 1). Cost: minutes per run. Zhao et al.'s
   numbers (BMVC 2018) should be fetched first.

Not candidates: born-again or self-training students (no label-free precedent; drift, 12-17 note) and soft emotion
targets (they use votes, so they break the label-free aim).
