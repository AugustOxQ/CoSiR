# Ways to raise D2 to the instructed target: instruction distillation, multi-teacher losses, relational and privileged distillation, graphs, heads, data

> Literature check for the CoSiR condition-free line (research-loop step 3). Written 2026-10-21 in the r10 run 12 chat
> by the `deep-research` agent (Opus); the file name uses CoSiR's sequence date (2026-12-20).
> Depth: deep (Opus), because this is the paper's central gap. Budget: 38 of 40 searches and 14 of 20 page reads; 3 of
> the reads returned PDFs that could not be read. Every page went through a summariser, and no paper was read in full.
> "Abstract" or "snippet" marks a claim that was not seen in the paper's tables.
> Question: what does the field know that could raise D2's style and genre AMI to or above the instructed Qwen image
> (and emotion further)? Two limits hold: one read of the default image at test time, and label-free training.
> Notes read first: `2026-12-17_cluster_level_losses.md` (format model), `2026-12-07_prompted_aspect_views.md`,
> `2026-12-05_percept_neighbourhood.md`, `2026-12-04_conditional_similarity_update.md`,
> `2026-10-21_cvpr_literature_review.md`, `2026-10-24_aspect_task_novelty_check.md`, and reports_sum rows 12-16 to
> 12-19. Their content is not repeated here. StyleDecoupler and Fioresi et al. are already covered in
> `2026-11-27_style_sources.md` and the 10-21 review.
> The D2 ablation of the same night (report 12-22) was not available to the agent; the "Row" column below names the
> ablation rows that bear on each idea.

## Summary

1. On the teacher's own task, a one-teacher student matches its teacher at best; no source shows it passing. Examples:
   jina-v5 clustering 53.4 against 57.1; RADIO zero-shot 82.93 against 83.90; UNIC on ImageNet-1K 83.8 against 84.0;
   Beyer et al. 82.31 against 83.0. Students pass their teachers only where a second source adds information. Our
   captions do this for emotion.
2. The cheapest levers with the best support are in the distillation loss:
   - standardised teacher targets (UNIC: +2.7 top-1);
   - a discarded MLP adaptor (SimReg: +3.7 kNN-1 over a linear one);
   - regression instead of a softmax-over-batch term (jina-v5);
   - matching soft similarity distributions over teacher anchors (PromptKD: 76.2 against 69.4 HM for feature L1).
3. Instructed inputs still fall below the target on style and genre (one seed). So D2's other losses probably pull the
   heads away from the teacher, and missing information may not be the only cause. A regression probe from the default
   to the instructed embedding separates the two causes in minutes.
4. Precomputed fixed targets are the case where distillation over-fits. Beyer et al.: with a fixed teacher, accuracy
   falls from 77.8 to 77.0 as training grows longer; with consistent views it rises to 81.8. Multi-crop targets are
   the strongest data lever, but they cost an extraction night.
5. Work on learning with privileged information (LUPI) warns that information missing from the student's input does
   not transfer (Provodin et al., TMLR 2025).
6. No comparator newer than our notes was found. The instructed Qwen image is still the bar.

## a. Instruction distillation into a student without the instruction (confidence: medium)

- **jina-embeddings-v5-text** (preprint, Feb 2026, [arXiv 2602.15547](https://arxiv.org/html/2602.15547v2)) is the
  closest precedent.
  - Teacher and student: the teacher is Qwen3-Embedding-4B with instructions. The student sees only "Query:" or
    "Document:".
  - Architecture: one shared base model, plus one LoRA adapter per task (retrieval, STS, clustering, classification).
  - Loss: cosine distance to a linear projection on the student side. A projection on the teacher side "simply does
    not work".
  - Clustering needed its own teacher instruction ("Identify the topic or theme…"). The generic instruction was
    "distinctly suboptimal".
  - Plain embedding distillation converges more slowly than the score-based (in-batch softmax) and InfoNCE forms, but
    ends highest (Fig. 3, no numbers given).
  - Retrieval, MTEB (Table 5): full objective 64.50, distillation only 63.16, InfoNCE only 63.38.
  - Clustering stays below the teacher: MMTEB clustering is 57.1 for the teacher against 53.4 and 52.7 for the two
    students (Table 2). The adapter stage raises clustering from 46.5 to 54.7 for the small student (Table A7).
  - The classification adapter adds a relational distillation term toward the base model (weight 20) to prevent drift.
  - Unlike our student, these students read the raw text, not a frozen vector.
- **PromptKD** (CVPR 2024,
  [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptKD_Unsupervised_Prompt_Distillation_for_Vision-Language_Models_CVPR_2024_paper.html),
  [arXiv HTML](https://arxiv.org/html/2403.02781v1)): label-free distillation of a prompted CLIP-L teacher into a
  CLIP-B student on unlabeled images.
  - Mean HM over 11 datasets: student 83.73 (Table 1a); teacher about 84.4 (the summariser's mean of Table 10).
  - ImageNet loss ablation (Table 4): a KL over the teacher's class-text similarities gives HM 76.22, feature L1 69.35,
    feature MSE 68.91.
  - Projector depth (Table 12): 1 layer 75.81, 2 layers 76.22, 3 layers 75.78.
  - More unlabeled images help, then level off (Fig. 4, no numbers).
- **Context distillation** (Snell et al., preprint 2022, [arXiv 2209.15189](https://ar5iv.labs.arxiv.org/html/2209.15189))
  trains the model to give its instructed answer from the bare input. It internalises instructions, reasoning and
  examples, and gains 9% on SPIDER over gradient descent on the same examples.
- **Eddy-VL** (preprint, Jul 2026, [arXiv 2607.16316](https://arxiv.org/pdf/2607.16316), snippet) distils
  Qwen3-VL-Embedding-2B into a pruned 1.9B model. It keeps the instruction format.
- **Not found:** distillation of an embedder's instructed image embeddings into a per-aspect student without
  instructions.

## b. Multi-teacher and agglomerative distillation (confidence: medium-high on the numbers)

| Source | Loss and heads | Effect sizes |
|---|---|---|
| AM-RADIO, CVPR 2024 ([arXiv](https://arxiv.org/html/2312.06709v4)) | Summary token: cosine, which "produced better models compared to L1, MSE, Smooth-L1". Spatial features: 0.9 cosine + 0.1 smooth-L1. One 2-layer MLP adaptor per teacher, depth not ablated | Beats its teachers on 6 of 9 tasks, e.g. kNN 86.06 against 83.41 (DINOv2-g). Below DFN CLIP on zero-shot, 82.93 against 83.90 (Table 1). Loss-balancing schemes within 0.35 mIoU of each other (Table 4) |
| PHI-S, preprint 2024 ([arXiv 2410.01680](https://arxiv.org/html/2410.01680v1)) | Normalisations of the teacher's distribution | Mean rank, 1 = best (Table 1): MSE 7.83, cosine 5.81, cosine + smooth-L1 5.31, global standardisation 3.81, per-channel 4.21, PHI-S 2.92; whitening worse (PCA-W 6.29). Cosine fits the teacher badly (MSE 105.9 against 5.1 ×10⁻⁴) but ranks better than raw MSE downstream |
| UNIC, ECCV 2024 ([arXiv](https://arxiv.org/html/2408.05088)) | Mean of cosine and smooth-L1; teacher features standardised with an EMA; per-teacher 2-layer MLP (hidden 4d), discarded; "teacher dropping" | ImageNet top-1 (Table 1): basic 78.7, + standardisation 81.4, + per-teacher projectors 82.2, + ladder 82.7, + dropping 83.2. Beats the best teacher on fine-grained (57.9 against 53.0) and ADE (39.6 against 36.6), not on ImageNet (83.8 against 84.0) |
| Theia, CoRL 2024 ([PMLR](https://proceedings.mlr.press/v270/shang25a.html)) | Cosine + smooth-L1 beat MSE (secondary summary only) | not checked |

- **When a student passes its teachers:** several complementary teachers, scored on tasks where their strengths
  combine. On one teacher's home task, students only match it.
- **Our reading:** style and genre are the instructed teacher's home task, so a one-teacher student should land at or
  below it. A second source can change this, as the captions do for emotion.

## c. Relational and structural distillation (confidence: low-medium; no head-to-head on clustering AMI found)

- RKD (CVPR 2019, [arXiv](https://arxiv.org/pdf/1904.05068)): in metric learning, a student with a smaller backbone and
  fewer dimensions beat its teacher on Cars-196 (abstract; numbers not checked).
- SEED (ICLR 2021, [arXiv](https://arxiv.org/pdf/2101.04731)) and CompRess (NeurIPS 2020,
  [arXiv](https://arxiv.org/pdf/2010.14713)) match the student's similarity distribution over a queue of teacher
  features to the teacher's. Numbers not checked.
- PromptKD (section a): the KL over anchors beats feature regression by about 7 HM. jina-v5 points the other way:
  in-batch score distillation plateaus below plain embedding distillation.
- RRD (preprint 2024, [arXiv 2407.12073](https://arxiv.org/html/2407.12073v5)): better NMI than CRD on supervised
  CIFAR-100 (Table 7, numbers not checked).
- Dong et al. (NeurIPS 2023): relational distillation is spectral clustering on the teacher's graph. This is already in
  the 12-17 note.
- **Our reading:** D2's distillation term is a softmax over in-batch cosines, which sits closer to the score-based form.
  Similarities to fixed anchors only ask for the teacher's neighbour order. That may suit a student that cannot
  reproduce the teacher's coordinates.

## d. Privileged information (confidence: medium)

- Provodin et al., "Rethinking Knowledge Transfer in LUPI" (TMLR 2025; NeurIPS 2024 workshop;
  [arXiv 2408.14319](https://arxiv.org/html/2408.14319v1)): no solid evidence of transfer in realistic settings.
  - Gains vanish once there are more than about 1,000 training samples, or with longer training.
  - On real data (Table 2) the teacher with privileged information is far ahead (Repeat Buyers 73.23), but the
    students equal the model without it (62.88 against 63.13).
- Modality focusing hypothesis (ICLR 2023, oral; [arXiv 2206.06487](https://arxiv.org/abs/2206.06487)): cross-modal
  distillation works in proportion to the teacher's decisive features that both modalities share (abstract).
- Generalized distillation (Lopez-Paz, ICLR 2016) is in the 12-17 note. Classic vision work on privileged information
  used attributes, boxes and text, with small data ([arXiv 1410.0389](https://pdf.arxiv.org/pdf/1410.0389)).
- **Our reading:**
  - The instruction is the same for every painting of an aspect. So the extra information lives in the pixels, and the
    question is what the default 2048-d vector kept.
  - The captions are true privileged information, and they helped emotion. That fits the hypothesis above: emotion
    cues in the captions are often visible in the image.

## e. Neighbour-graph quality (confidence: low-medium; beyond the 12-17 note)

- Hubness reduction (Amblard et al., Bioinformatics 2022,
  [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.03.18.435808.full.pdf)) gives better kNN graphs and better
  clustering, most at high intrinsic dimension. It beats other ways of improving the graph (abstract; numbers not
  read).
- QB-Norm (CVPR 2022,
  [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.html))
  fixes hubness in cross-modal retrieval without retraining (abstract).
- PhenoGraph ([docs](https://scanpy.readthedocs.io/en/stable/external/generated/scanpy.external.tl.phenograph.html))
  weights kNN edges by Jaccard overlap and runs Leiden on the result. Its `prune` option multiplies the graph by its
  transpose, which is close to a mutual graph.
- NNM (CVPR 2021,
  [CVF](https://openaccess.thecvf.com/content/CVPR2021/papers/Dang_Nearest_Neighbor_Matching_for_Deep_Clustering_CVPR_2021_paper.pdf))
  takes neighbours from batch and global features as they change: +3.7% over the best prior method on CIFAR-100
  (abstract).
- Mutual kNN graphs suit the search for the most significant clusters (Maier et al.,
  [arXiv 0912.3408](https://arxiv.org/pdf/0912.3408), snippet).
- **Our reading:** these change the buddy pairs and the Leiden readout, not what the teacher passes on. Expect small
  effects on style and genre; they may matter for the Leiden emotion gap.

## f. Disentangled aspect heads (confidence: low)

- Already covered: CSN, SCE-Net, DivClust (a shared trunk costs about 0.03 NMI per head), MFCVAE, Fioresi et al.,
  StyleDecoupler.
- Nr-Kmeans and ENRC make the subspaces orthogonal so that the clusterings do not repeat each other. AugDMC (preprint
  2023, [arXiv 2306.13023](https://arxiv.org/html/2306.13023v1)) gets the diversity from augmentations instead.
  Per-clustering numbers not checked.
- **Not found:** evidence that an orthogonality term between aspect heads raises AMI per aspect.
- **Our reading:** WikiArt style and genre are probably correlated (not checked), so a forced orthogonality may remove
  signal the two share.

## g. Augmentation, capacity, data (confidence: medium-high for Beyer and SimReg)

- **Beyer et al.**, "patient and consistent" (CVPR 2022, [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.05237)),
  ImageNet, Table 3:
  - Fixed precomputed teacher: 77.76 at 300 epochs, over-fits after 600, 76.95 at 9,600.
  - Consistent crops: 80.08 at 300 and 81.76 at 9,600. With mixup added: 82.31 at 9,600.
  - On Flowers102 (Table 4), training from 1k to 1M epochs raises the student from 77.51 to 96.93, against a teacher
    at 97.82.
  - Related images work nearly as well as images of the target data, given a longer schedule (Fig. 6).
  - **Our targets are the fixed-teacher case.**
- **SimReg** (venue not checked, [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.05131)), Table 1:
  - A 4-layer MLP predictor, discarded after training, against a linear one: kNN-1 54.5 against 50.8; linear probe
    68.5 against 58.3.
  - Features under the predictor are as good as its output: 65.7 against 65.2 (Table 2).
  - Teacher and student work best on the same weak view (Table 7).
- Cho and Hariharan (ICCV 2019,
  [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html)):
  a larger or more accurate teacher is often not a better one (abstract).
- **Noise and mixup on frozen embeddings** stop heads from over-fitting ([arXiv 2303.05180](https://arxiv.org/pdf/2303.05180);
  E-Stitchup, [arXiv 1912.00772](https://arxiv.org/pdf/1912.00772)). Not tested for distillation.
- **Intermediate layers** beat the final layer by up to 16% on 32 MTEB tasks (Skean et al., ICML 2025,
  [arXiv 2502.02013](https://arxiv.org/html/2502.02013v2)). ESMC picks its layer and token (12-07 note).
- **Prompt ensembling:** CLIP's 80 averaged prompts add 3.5% on ImageNet over one prompt
  ([arXiv 2103.00020](https://arxiv.org/pdf/2103.00020)).
- **WiSE-FT** (CVPR 2022,
  [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)):
  interpolating toward the original model gives +4 to 6 pp under distribution shift and +1.6 pp on ImageNet.
- Distillation scaling laws (ICML 2025, [PMLR](https://proceedings.mlr.press/v267/busbridge25a.html)): when a teacher
  already exists, distillation beats supervised training, up to a compute level that grows with student size
  (abstract).

## Candidate ideas, one ranking

Cost assumes about 35 s per D2 run on the 3090, with 3 seeds. "Row" names the ablation row (report 12-22) that decides
whether the idea is worth trying.

| # | Idea, mechanism, fit | Evidence | Cost | Risk | Row |
|---|---|---|---|---|---|
| 1 | **Information ceiling probe** (diagnostic, first). Per aspect, regress the instructed image from the default one: a ridge model and a 4-layer MLP, cosine + smooth-L1 on standardised targets, long training, early stopping on the dev regression loss (no labels). k-means AMI of the predictions on dev | Provodin; SimReg; Beyer | about 10 GPU min | low | teacher; instructed inputs |
| 2 | **Teacher-faithful style and genre heads.** Per-aspect loss weights (more distillation on style and genre), or a relational distillation term toward the teacher (jina-v5), or output interpolation with probe 1's regressor (as in WiSE-FT) | jina-v5; WiSE-FT; drift (12-17 note) | 10 to 20 min | weights picked on dev labels must be reported; the emotion lead may come from the same terms; the ceiling is the target | graph terms; buddy pairs; own caption; caption side |
| 3 | **Standardised regression distillation.** Per-dimension standardisation or PHI-S on the teacher; cosine + smooth-L1 through a per-aspect 2-layer adaptor that is discarded; cluster the head under it | UNIC +2.7; PHI-S rank 2.92 against 7.83; SimReg +3.7; jina-v5 | about 15 min | the adaptor may absorb the gain | teacher; default-view teacher |
| 4 | **Soft teacher anchors (12-17 #1, now merged).** KL between teacher and student similarity distributions over the teacher's k-means centroids (k = 20 and about 200) or a queue of teacher embeddings | PromptKD +6.9 HM; ESMC 13/15; SEED, CompRess | about 20 min | copies the teacher's partition | teacher; graph terms |
| 5 | **A second teacher per head.** The instructed caption embedding (per painting) for emotion; a CLIP or SigLIP 2 image for style, from the existing stores; UNIC-style teacher dropping | UNIC, RADIO: students pass teachers only by combining them; modality focusing hypothesis | about 15 min, no extraction | the information may be missing from the Qwen default (probe 1); CLIP style is weaker (0.300 against 0.395, 12-15) | caption side; own caption |
| 6 | **Consistent multi-crop targets.** Default and instructed embeddings of K = 4 crops per training painting, the same crop for teacher and student, longer training; test time unchanged | Beyer: fixed teacher over-fits, consistent views +4.8 at 9,600 epochs | extraction hours (Qwen throughput not checked); about 2.8 GB fp16 on /data/SSD2 | crops may cut genre cues | teacher |
| 7 | **Prompt-ensembled teacher.** Average 4 to 8 paraphrases per aspect, written before any run | CLIP +3.5%; jina-v5 (the instruction matters) | extraction hours | a stronger instructed baseline to beat; never select paraphrases by AMI | no instruction anywhere; default-view teacher |
| 8 | **Richer input from the same default pass.** Intermediate-layer or token states | Skean et al.; ESMC | one re-extraction, about 1.4 GB for 6 layers | changes the test-time contract (the user decides); try only if probe 1 shows the information is missing | — |
| 9 | **More unlabeled paintings** for the distillation term only | PromptKD Fig. 4; Beyer Fig. 6 | extraction | overlap with the held paintings; licences | — |
| 10 | **TURTLE and TEMI on the instructed embedding (12-17 #3).** A comparator for the paper, not a lever | 12-17 note | as in 12-17 | as in 12-17 | — |
| 11 | **Graph refinement.** Mutual proximity or Jaccard-weighted graphs for the buddy pairs and Leiden; neighbours refreshed from the student | Amblard; NNM | minutes | adds no teacher information | random graph of equal degree; default-view graph |
| 12 | **TEMI-style cluster head (12-17 #2)** | 12-17 note | minutes | tighter clusters, not better aligned with labels | graph terms |

**Not recommended:** an orthogonality term between heads (no support found), and plain DEC (12-17 note).

## Comparators

- Three searches (one extended) found nothing newer than our notes for label-free aspect clustering from one default
  image embedding.
- Mechanism precedents a reviewer may name:
  - jina-v5 (task students without instructions, text only);
  - SAGA (preprint, Jun 2026, [arXiv 2606.15134](https://arxiv.org/html/2606.15134v1)): a frozen MLLM supervises an
    image encoder during training; one embedding per image, supervised by class pairs, fine-grained retrieval;
  - Eddy-VL (keeps instructions).
- The bar is still the instructed Qwen image (an ensemble of prompts, if idea 7 runs). Beside it: TURTLE and TEMI on
  that image, and PercepT core.

## Sources

| Source | Venue | Area |
|---|---|---|
| jina-embeddings-v5-text, [2602.15547](https://arxiv.org/html/2602.15547v2) | preprint 2026 | a, b, c |
| PromptKD, Li et al. | CVPR 2024 | a, c, g |
| Snell et al., context distillation | preprint 2022 | a |
| Eddy-VL [2607.16316](https://arxiv.org/pdf/2607.16316) | preprint 2026 (snippet) | a |
| AM-RADIO, Ranzinger et al. | CVPR 2024 | b |
| PHI-S, Ranzinger et al. | preprint 2024 (venue not checked) | b |
| UNIC, Sariyildiz et al. | ECCV 2024 | b |
| Theia, Shang et al. | CoRL 2024 (secondary only) | b |
| RKD; SEED; CompRess; RRD | CVPR 2019; ICLR 2021; NeurIPS 2020; preprint 2024 | c |
| Provodin et al. | TMLR 2025 | d |
| Xue et al., modality focusing | ICLR 2023 | d |
| Sharmanska et al. [1410.0389](https://pdf.arxiv.org/pdf/1410.0389) | preprint 2014 | d |
| Amblard et al.; QB-Norm; NNM; PhenoGraph | Bioinformatics 2022; CVPR 2022; CVPR 2021; docs | e |
| AugDMC [2306.13023](https://arxiv.org/html/2306.13023v1) | preprint 2023 | f |
| Beyer et al. | CVPR 2022 | g |
| SimReg [2201.05131](https://ar5iv.labs.arxiv.org/html/2201.05131) | venue not checked | c, g |
| Cho and Hariharan; WiSE-FT | ICCV 2019; CVPR 2022 | g |
| Skean et al.; Busbridge et al. | ICML 2025 (both) | g |
| CLIP, Radford et al. | ICML 2021 | g |
| SAGA [2606.15134](https://arxiv.org/html/2606.15134v1) | preprint 2026 | comparators |

## Not checked

- The numbers of Theia, RKD, SEED, CompRess, RRD, Amblard, AugDMC and the modality focusing hypothesis.
- The venues of PHI-S, SimReg and Snell et al.
- jina-v5 Table A15: the clustering-instruction ablation.
- Whether born-again networks pass their teacher (the search returned no results text).
- Qwen3-VL-Embedding extraction throughput on the 3090. It sets the cost of ideas 6 to 9 (the framing note 12-21
  measured about 0.094 s per instructed image from the claim test's extraction logs).

## Open (budget spent)

- **A head-to-head of relational against direct distillation on clustering AMI.** Search "distillation k-means NMI
  student teacher frozen embeddings".
- **jina-v5 clustering instruction effect.** Fetch arXiv 2602.15547 from offset 100000.
- **Amblard effect sizes.** Read the Bioinformatics HTML (DOI 10.1093/bioinformatics/btab795).
