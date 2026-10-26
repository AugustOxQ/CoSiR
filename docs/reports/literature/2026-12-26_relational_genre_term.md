# Relational genre term: name and form, weight regime, relational beside direct, genre comparators

> Literature check for the CoSiR condition-free line (research-loop step 3). Written 2026-10-26 by the `deep-research` agent for r10 decide 23; the file name uses CoSiR's sequence date (2026-12-26).
> Depth: standard (Sonnet). Budget: 14 of 20 searches, 10 of 10 page reads. Four of the reads returned raw PDFs that could not be read (RKD, SP-KD, SEED, CompRess as arxiv.org/pdf); they were redone as ar5iv pages, which count in the 10. Every page went through a summariser; no paper was read in full.
> Notes read first: `2026-12-20_ways_to_improve.md` (section c, candidate ideas, comparators), `2026-12-17_cluster_level_losses.md` (Summary), `2026-10-24_aspect_task_novelty_check.md` (section 1). Their content is not repeated.
> Conventions: "Abstract" or "snippet" marks a claim seen only in an abstract or a search-result summary. "Not checked" marks a claim not seen at its source. Peer-reviewed work and preprints are told apart.
> The last section, "Controller's note", was added by the decide 23 chat from the project's code, not by the agent.

## Summary

1. **Q1, name and form (confidence: medium-high).** Our term is not RKD. RKD matches pairwise distances and angles. Ours matches softmax rows of similarities, which is the form of probabilistic knowledge transfer (PKT), CompRess, SEED, NanoVDR's "ranking loss" and CLIP-KD's "contrastive relational distillation". Call it "relational (similarity-distribution) distillation" and cite RKD, PKT, SP, CompRess, SEED and CLIP-KD. The usual form is KL(teacher || student), teacher first. Anchors are a queue of teacher features in SEED and CompRess; in-batch rows are used in CLIP-KD, SP-KD and NanoVDR.
2. **Q2, weight regime (confidence: medium-low).** Published weights are not comparable across papers, because each normalises its loss differently. The sweeps seen are flat or peaked, not monotone: CLIP-KD's relational weight is flat from 0.5 to 2 and lower at 10. No source shows gains rising steadily with the relational weight. No source tests one head's relational weight against other heads on a shared trunk. Our dip on seed 0 that vanishes on fresh seeds has no precedent either way.
3. **Q3, relational alone against beside direct (confidence: medium on direction, low on transfer).** No head-to-head on clustering or AMI was found. On retrieval and classification the evidence splits. In CLIP-KD, direct feature regression beats the relational term alone (34.2 against 31.9 ImageNet zero-shot), and adding the relational term beside it gains a little (+0.3). In NanoVDR, adding a KL ranking term beside the direct term lowers NDCG@5. In SEED and CompRess, the relational term beats plain regression on ImageNet linear probes.
4. **Q4, comparators (confidence: low).** No label-free genre AMI or NMI on WikiArt or ArtELingo was found. The instructed Qwen image stays the bar. No paper states a "student reaches teacher level" claim with a formal non-inferiority margin. Plan to define and state our own margin.

## Q1. Name and form

- **RKD** (Park, Kim, Lee, Cho; CVPR 2019, [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html), [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.05068)) defines a distance-wise loss and an angle-wise loss, each a Huber loss on teacher and student potentials. It does not use a softmax row, so it is not the same form as ours. The summariser did not confirm the Huber threshold or normalisation from the paper; the Huber description is from a secondary 2025 paper ([arXiv 2503.21269](https://arxiv.org/pdf/2503.21269), snippet).
- **SP-KD** (Tung and Mori; ICCV 2019, [arXiv 1907.09682](https://arxiv.org/pdf/1907.09682), read via [ar5iv](https://ar5iv.labs.arxiv.org/html/1907.09682)): a b x b in-batch similarity matrix of layer activations, rows L2-normalised, loss = squared Frobenius distance divided by b^2. It is in-batch and layerwise, not softmax, not KL.
- **PKT** (Passalis and Tefas; ECCV 2018, [arXiv 1803.10837](https://arxiv.org/pdf/1803.10837), abstract only): matches the probability distribution of the data in feature space instead of the representation. That the kernel is cosine and the divergence KL was not confirmed; from memory only. Not checked.
- **CompRess** (Abbasi Koohpayegani, Tejankar, Pirsiavash; NeurIPS 2020, [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.14713)): per query, the loss is KL(p_teacher || p_student). Each p is a softmax over cosine similarities to anchors, divided by a temperature. The anchors are a FIFO queue of teacher features (128,000); the teacher is frozen. Temperature 0.04 for two teachers, 0.007 for SwAV; one temperature for both sides. This is the closest published form to ours, with a queue in place of the batch.
- **SEED** (Fang et al.; ICLR 2021, [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.04731)): cross-entropy between teacher and student similarity distributions over a queue of 65,536 teacher features, with the teacher feature appended. Cross-entropy with the teacher as target equals KL up to a constant. Temperatures differ: teacher 0.01, student 0.2. The teacher temperature matters (Table 5, ResNet-18 student, ResNet-50 teacher, ImageNet top-1): 0.3 gives 54.8, 0.1 gives 54.9, 0.05 gives 56.5, 0.01 gives 57.9, 0.001 gives 57.6.
- **CLIP-KD** (Yang et al.; CVPR 2024, [arXiv 2307.12732](https://arxiv.org/html/2307.12732)): "contrastive relational distillation" (CRD in that paper) aligns teacher and student in-batch image-to-text and text-to-image softmax distributions with KL, teacher first. Separate teacher and student temperatures (values not stated on the page). This is the closest in-batch precedent. Naming clash: this CRD is not Tian et al.'s CRD (ICLR 2020, contrastive representation distillation), which was not read here. Not checked.
- **DarkRank** (Chen et al., AAAI 2018) was not found or read. Not checked.
- **Which name a reviewer would expect (our reading).** "Relational distillation" is the umbrella (RKD). Because our term is a softmax row compared with KL, "similarity-distribution distillation toward the teacher" is the accurate form, and PKT and CompRess are the closest cites. The one choice that decides the family is anchors: in-batch rows (CLIP-KD, SP-KD, NanoVDR) or a queue of fixed teacher features (CompRess, SEED). A queue is the stronger published form (CompRess says larger banks help, Figure 3b, no numbers in text; SEED gains about +1.5 top-1 from 256 to 65,536 anchors, Appendix A.8).
- **Direction.** Teacher distribution first (KL(teacher || student)) in CompRess and CLIP-KD; equivalent cross-entropy in SEED. This is the usual direction; no paper in this check argues for the reverse.
- **Temperature.** Published values are low and often tuned: 0.04 and 0.007 (CompRess), 0.01 for the teacher (SEED). A teacher temperature well below the student's sharpens the target (SEED). Our direct term uses 0.05; the brief does not give the relational term's temperature. Whether the relational term uses the same 0.05 matters for comparison with these papers.

## Q2. Weight regime and saturation

- **Weights vary by orders of magnitude, since the losses are normalised differently.**
  - RKD (peer-reviewed): metric learning, lambda_RKD-D = 1 and lambda_RKD-A = 2, without the task loss; classification, 25 and 50 beside cross-entropy; few-shot, 50 and 100. The paper gives no ablation of these weights in the page read.
  - SP-KD (peer-reviewed): gamma = 3000 on CIFAR-10 and 2000 on CINIC-10, beside cross-entropy. The paper reports robust performance for gamma from 10 to 64,000 (summariser's reading; no table seen).
  - CLIP-KD (peer-reviewed): lambda_CRD = 1 against lambda_FD = 2000 for the direct feature term (Section 4.1). So the relational weight is far smaller than the direct one there, though the scales differ.
  - RRD (preprint 2024, [arXiv 2407.12073](https://arxiv.org/pdf/2407.12073), snippet): coefficients between 0.5 and 1.5 work best; the relational weight is usually fixed at 1.
- **Saturation seen (CLIP-KD, Appendix C, ImageNet zero-shot):**

  | lambda_CRD | 0.5 | 1 | 2 | 10 |
  |---|---|---|---|---|
  | IN acc | 31.6 | 31.9 | 31.7 | 31.2 |

  Flat, then a small fall at 10. The direct weight rises then saturates (Table 6: 10 gives 31.1, 100 gives 32.3, 1000 gives 33.7, 2000 gives 34.2, 3000 gives 34.1).
- **Monotone gains with weight:** none seen. Our curve (genre AMI 0.4935, 0.4948, 0.4938, 0.5018 at weights 0.5 to 8) is flat to weight 4 and then rises; it has no precedent in these sources. Where a sweep ends is where the evidence stops: only CLIP-KD swept up to 20 times its default (10 against 0.5) and fell.
- **Collapse at high weight:** a secondary source (Multi-level KD via alignment and correlation, [arXiv 2012.00573](https://arxiv.org/pdf/2012.00573), snippet) says RKD's distance-wise loss differs greatly in scale from other losses and makes training unstable. Not checked at the source.
- **Interference with other heads on a shared trunk:** no source tests this for distillation. The general problem (negative transfer in shared-trunk multi-task models) is described by Jacob et al., "Online Knowledge Distillation for Multi-Task Learning" (WACV 2023, [CVF](https://openaccess.thecvf.com/content/WACV2023/html/Jacob_Online_Knowledge_Distillation_for_Multi-Task_Learning_WACV_2023_paper.html), snippet); they add per-task weighting. Nothing found on raising one head's relational weight. Our seed-0 dip (emotion -0.003, style -0.005) that does not appear on fresh seeds is within what the sources would call noise, but no source gives a size for it. Not checked.

## Q3. Relational alone against relational beside direct

No head-to-head on NMI, AMI or R@1 for a clustering student was found. Closest evidence, with the contradiction shown:

| Source (venue) | Setting | Direct alone | Relational alone | Both |
|---|---|---|---|---|
| CLIP-KD (CVPR 2024), Table 1 | ViT-T/16 student, IN zero-shot | FD 34.2 | CRD 31.9 | FD+ICL 34.6 -> +CRD 34.9 |
| NanoVDR (preprint, Mar 2026, [arXiv 2603.12824](https://arxiv.org/pdf/2603.12824)), Table 3 (snippet of table via search summary) | text student of a VLM retriever, NDCG@5 v1 / v2 / v3 | alignment 82.2 / 61.4 / 44.1 | KL ranking 81.1 / 57.4 / 41.6 | 1 + 0.5 rank: 81.6 / 59.8 / 42.8; 1 + 1: 81.5 / 59.1 / 42.5 |
| RKD (CVPR 2019), Table 4 | CIFAR-100 accuracy | Hinton KD 74.26 | RKD-DA 72.97 | KD + RKD-DA 74.66 |
| SP-KD (ICCV 2019), CINIC-10 | top-1 error, lower is better | KD 18.62, AT 18.50 | SP 18.56 | KD+SP 18.35, AT+SP 18.20 |
| SEED (ICLR 2021), Table 4 | ImageNet linear top-1 | L2 regression 55.3 | SEED 57.9 | not tested; Appendix A.5 says an L2 + cross-entropy form gives equal results when the teacher temperature is tuned |
| CompRess (NeurIPS 2020), Table 1 | MoCo R50 -> R18, linear | regression 52.2 | 62.6 (one queue) | not tested |

- **Reading.** Where the relational term replaces the direct one, the result depends on the setting: it wins over plain regression in self-supervised ImageNet distillation (SEED, CompRess), and it loses to direct regression in CLIP-style and retrieval students (CLIP-KD, NanoVDR). Added beside a direct term, the gain is small and positive in three sources (RKD +0.4, CLIP-KD +0.3, SP-KD -0.15 to -0.5 error) and negative in NanoVDR (-0.6 to -1.8 at weight 0.5 to 1). This matches the earlier note's jina-v5 finding that in-batch score distillation plateaus below embedding distillation (see the 12-20 note).
- **Relevance to us.** Our result (relational beside direct, genre AMI up 0.03 at weight 8) sits with the "small gain beside direct" group, but the gain is larger than in CLIP-KD or RKD. The NanoVDR trend (more relational weight, less score) is the opposite of ours, and a reviewer may cite it. Different task: NanoVDR students read different inputs from the teacher; ours reads a default image against an instructed one.
- **Dense-sweep note.** A relational-alone arm (relational on genre without the direct term) would answer Q3 for our setting directly; no source does.

## Q4. Comparators for label-free genre clustering of paintings

- **Strongest published label-free genre number: not found** on WikiArt or ArtELingo (three searches, no AMI or NMI against genre from frozen embeddings, k-means or Leiden).
- **Nearest items, none comparable:**
  - Description Guided Clustering (DGC, preprint 2025, [arXiv 2507.04769](https://arxiv.org/html/2507.04769), Table 1): WikiART, CLIP + k-means, ACC / NMI / ARI = 36.09 / 58.98 / 14.85; DGC 41.45 / 61.20 / 15.72. The caption says "styles of real artists in the AICD dataset"; the labels (style, genre or artist) are not stated. NMI of 59 is far above our genre AMI, so these are very probably different labels and many classes. Do not put it beside our numbers.
  - CPP (ICLR 2024, [arXiv 2306.05272](https://arxiv.org/pdf/2306.05272)): clusters WikiArt into 20 clusters with frozen CLIP; reported as a qualitative case study only (search summary). Already in the 12-17 note as a comparator for other reasons.
  - DELIUS, "A deep learning approach to clustering visual arts" (Castellano and Vessio; IJCV 2022, [arXiv 2106.06234](https://arxiv.org/pdf/2106.06234), snippet): clusters with k from 2 to 10 against 10 genres; reports silhouette and says genre accuracy is below 50%; NMI lower than ACC. Numbers not seen. Not checked.
  - Supervised genre accuracy (CLIP ViT-L/14 linear probe 75.0%, 11 genres) is from an informal Hugging Face space ([ArtSleuth page](https://ladyfaye1998-artsleuth.hf.space/sitemap.xml) was the only link found; not checked). "Harnessing Self-Supervised Features for Art Classification" (preprint 2026, [arXiv 2605.18974](https://arxiv.org/pdf/2605.18974), snippet) says zero-shot style and genre are poor and CLIP-linear is best. Both are supervised and are no label-free comparator.
- **Framing a student as "reaching its teacher":** no paper with a pre-set non-inferiority margin was found (three searches). What was found: students are said to "retain" a share of the teacher (DistilBERT 97% of BERT, from a glossary page, secondary, not checked at source); "Does Knowledge Distillation Really Work?" (Stanton et al., NeurIPS 2021, [arXiv 2106.05945](https://arxiv.org/pdf/2106.05945), abstract) shows students often stay far from the teacher's predictions even when matching is possible, and that closer matching does not always generalise better. Menon et al. (ICML 2021) is a bias-variance account and does not set a margin. Earlier note: jina-v5 clustering 53.4 against 57.1 and similar ones in the 12-20 note are students at or below the teacher.
- **Our reading.** The margin has to be our own. State it in AMI points, set before the held read, as "genre AMI of the student minus the instructed Qwen image, one-sided lower bound above minus X", with X justified by the seed-to-seed spread (about 0.003 to 0.005 AMI in our own notes; not rechecked here). The usual statistics name is a non-inferiority margin from clinical trials; no source in this check applies it to distillation. Not checked.

## Comparators

- Beside PercepT and the instructed Qwen image (0.4989 genre AMI on development data): nothing new for label-free genre clustering of paintings from frozen embeddings.
- Mechanism precedents a reviewer may name for the term: CompRess and SEED (anchors), PKT, CLIP-KD (in-batch, CLIP student), NanoVDR (counter-evidence), RKD and SP-KD (relational family). TURTLE and TEMI stay as in the 12-17 note.

## Sources

| Source | Venue, year | Used for |
|---|---|---|
| Park et al., Relational Knowledge Distillation, [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.05068) | CVPR 2019 | Q1, Q2, Q3 (weights, Table 4) |
| Tung and Mori, Similarity-Preserving KD, [ar5iv](https://ar5iv.labs.arxiv.org/html/1907.09682) | ICCV 2019 | Q1, Q2, Q3 (gamma, tables) |
| Passalis and Tefas, PKT, [arXiv 1803.10837](https://arxiv.org/pdf/1803.10837) | ECCV 2018 (abstract only) | Q1 |
| Abbasi Koohpayegani et al., CompRess, [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.14713) | NeurIPS 2020 | Q1, Q3 |
| Fang et al., SEED, [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.04731) | ICLR 2021 | Q1, Q3 |
| Yang et al., CLIP-KD, [arXiv 2307.12732](https://arxiv.org/html/2307.12732) | CVPR 2024 | Q1, Q2, Q3 (weights, Tables 1, 6, 7) |
| Liu et al., NanoVDR, [arXiv 2603.12824](https://arxiv.org/pdf/2603.12824) | preprint, Mar 2026 (table via search summary) | Q2, Q3 |
| RRD, [arXiv 2407.12073](https://arxiv.org/pdf/2407.12073) | preprint 2024 (snippet) | Q2 |
| Multi-level KD via alignment and correlation, [arXiv 2012.00573](https://arxiv.org/pdf/2012.00573) | venue not checked (snippet) | Q2 |
| Jacob et al., Online KD for Multi-Task Learning, [CVF](https://openaccess.thecvf.com/content/WACV2023/html/Jacob_Online_Knowledge_Distillation_for_Multi-Task_Learning_WACV_2023_paper.html) | WACV 2023 (snippet) | Q2 |
| DGC, [arXiv 2507.04769](https://arxiv.org/html/2507.04769) | preprint 2025 | Q4 |
| CPP, [arXiv 2306.05272](https://arxiv.org/pdf/2306.05272) | ICLR 2024 (snippet) | Q4 |
| DELIUS, [arXiv 2106.06234](https://arxiv.org/pdf/2106.06234) | IJCV 2022 (snippet) | Q4 |
| Harnessing Self-Supervised Features for Art Classification, [arXiv 2605.18974](https://arxiv.org/pdf/2605.18974) | preprint 2026 (snippet) | Q4 |
| Stanton et al., Does KD Really Work?, [arXiv 2106.05945](https://arxiv.org/pdf/2106.05945) | NeurIPS 2021 (abstract) | Q4 |

## Not checked

- PKT's kernel and divergence; DarkRank; Tian et al.'s CRD (ICLR 2020), and the temperatures in CLIP-KD and in the NanoVDR loss.
- The RKD, SP-KD, CLIP-KD, SEED and CompRess numbers came through a summariser, not from reading the tables myself; the SP-KD "robust from 10 to 64,000" claim and the CompRess queue-size figures were not seen as numbers.
- NanoVDR Table 3 is the search summary's copy; the per-backbone breakdown (its Appendix D) was not seen.
- PercepT's own genre result (arXiv 2606.03345): a search found the paper's title and datasets, but not an AMI or genre number.
- Whether DGC's WikiART labels are genre. Whether the ArtSleuth 75.0% figure is correct.

## Open

- **A relational-alone against relational-beside-direct comparison on clustering AMI.** Not found in 14 searches. Our own arm (relational alone on the genre head) would settle it for us in minutes; a search for "distillation k-means NMI student teacher frozen embeddings" was already listed open in the 12-20 note and was not repeated.
- **Weight sweeps past 8, and the relational term's temperature.** No source covers the region. The next experiment is the evidence. Suggested pre-check: report the relational term's temperature and loss scale (the mean of the two losses at weight 1) so a reviewer can compare with CompRess and CLIP-KD.
- **A queue of fixed teacher anchors in place of in-batch rows (CompRess, SEED form).** Not tested by us; the earlier note lists it as idea 4. The search that would settle the best anchor count is SEED Appendix A.8, Figure 7 (a figure, not read).
- **Label-free genre clustering on WikiArt.** Open DGC's code or supplement for the label set, and search "genre" with "unsupervised" in art-analysis venues (ICCVW AI4VA, CVPRW VISART).
- **Non-inferiority framing.** A targeted search on "non-inferiority" with "model compression" in clinical-ML evaluation venues, or a statistics reference such as the CONSORT non-inferiority extension (not read), would give a citable margin convention.

## Controller's note (r10 decide 23, from the code)

The term as implemented (`src/test/20261019_model_sheet/ms_train.py`, `distill_loss` with kind `relational`, called by `src/test/20261025_encoders_genre/eg_train3.py`, `switched_head_loss`): for each painting in the batch, the KL divergence from the teacher's row to the student's row, where each row is a softmax over that painting's similarities to the other paintings in the batch (entries of the same painting excluded), temperature 0.05 on both sides, the same similarity and temperature as the direct term; the mean over the batch. The teacher rows come from the instructed Qwen genre image, the student rows from the genre head's image output. So it is the in-batch form (CLIP-KD, SP-KD), teacher first, with one shared temperature and no queue of anchors. The loss scale of the two terms at weight 1 is not recorded in the reports.
