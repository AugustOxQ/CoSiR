# T6 scan: the interface between stage 1 (groupings) and stage 2 (factor learning)

Date 2026-10-05. Mode: ARS deep-research `quick` (research brief). AI disclosure: this scan was run by an AI agent (Claude) with WebSearch and WebFetch; page content came through a summarising fetch tool, so quotes and section numbers must be checked against the PDFs before any enters a paper. Read scope: arXiv or conference abstract pages for all 11 cited papers; full text (via ar5iv, summarised) only for CACTUs and PCL.

## Executive Summary

Published pseudo-task work offers little direct evidence on what a bank should take from a grouping, and none for our setting (cross-modal, value-disjoint, aspect-conditioned). Three points follow. (1) Pseudo-task meta-learning transfers only partly: CACTUs reached 73.36 against 96.29 for an oracle on Omniglot 20-way 5-shot (as reported in `scan_N4.md`), and the cluster-label mismatch is acknowledged by its authors. (2) Hard-group episodes assume that every non-group row is a negative; the contrastive literature treats this as the main failure and offers elimination, attraction or soft weights, which map onto a group-similarity matrix S. (3) Several partitions help: CACTUs ablates one partition against 100 and finds multiple better, and PCL clusters at three granularities. This agrees with our SE result. We recommend a staged interface: keep hard groups per grouping (the P0 control), add sibling exclusion from episode negatives using S, and weight groupings only if a label-free criterion exists.

## Background and Research Question

A bank episode copies an evaluation episode with groups in place of values, so support pairs share a group by construction (brief §1). Real episodes share a value that the groupings only partly track (handoff Step 1). Question: which parts of a grouping should a bank consume (hard group id, posterior, group similarity, per-grouping weight) so that factors transfer to labelled aspects?

## Key Findings

| # | Finding | Strength | Source |
|---|---|---|---|
| 1 | Tasks from k-means clusters of unsupervised embeddings (k = 500; P = 100 partitions on Omniglot, 50 on miniImageNet and CelebA; random dimension scaling per partition) beat embedding-only baselines. P = 1 against P = 100 ablations favour many partitions. | Moderate (full text summarised) | Hsu et al. (2019) |
| 2 | The authors flag that clusters sometimes follow labels, sometimes are uninterpretable or follow image artifacts, and name the pseudo-to-real task gap as open. | Moderate | Hsu et al. (2019), §5, Fig. 2 |
| 3 | Pseudo-tasks from augmentation (one image per class, augmented copies) match CACTUs, so cluster semantics are not needed for transfer in class tasks. | Moderate (abstract) | Khodadadeh et al. (2019) |
| 4 | Fixed pseudo-labels are replaced in PsCo by momentum-network pseudo-labels and a queue, motivated by building diverse tasks with improving labels. | Weak (abstract) | Jang et al. (2023) |
| 5 | A set-function adapter fitted on pseudo-tasks does better on pseudo tasks and worse on real ones (Table IX, from `scan_N4.md`). | Moderate (as reported) | Ye et al. (2020/2022) |
| 6 | Mixture-of-Gaussians episode inference treats each mode as a class-concept of the episode, a soft assignment inside each episode rather than fixed pseudo-labels. | Weak (abstract) | Lee et al. (2021) |
| 7 | Instance-level or cluster-level negatives include same-class items ("false negatives"); harm grows with the number of semantic concepts. Responses: cancel or attract them (FNC), detect and remove them incrementally (IFND), or replace one-hot targets with soft inter-sample relations (ASCL). | Moderate (abstracts) | Huynh et al. (2022); Chen et al. (2022); Feng & Patras (2022) |
| 8 | Clustering at several granularities (K = 25k, 50k, 100k) gives a more robust estimate of prototypes and encodes hierarchy; the paper gives limited explicit ablation of K. | Moderate | Li et al. (2021) |
| 9 | Task-distribution properties (diversity, difficulty, domain, curriculum) change few-shot accuracy materially; a class-partition diversity metric and disentangled-latent clusters (one partition per latent dimension) are proposed. | Weak to moderate (abstracts) | Bansal et al. (2021); Cui et al. (2025) |

Granularity: the only direct evidence we found is indirect (finding 8, finding 1's k = 500 fixed). We found no controlled study of cluster count against transfer for pseudo-task banks, to our knowledge, within this search. Our own sweep (brief §2: group count is not the lever; cluster lift rises while lift through heads stays flat) is consistent.

## Analysis and Implications

Reader-inferred. (a) Hard groups with random negatives are exactly what FNC, IFND and ASCL fix: our affect grouping splits sadness over about seven groups (brief §2), so an episode whose negative group is a sibling of the support group trains the factors to separate sad from sad. Over-split groups raise false-negative rate at fixed K regardless of purity. (b) CACTUs and UMTRA suggest the bank does not need cluster semantics, only consistent within-group structure and enough task variety; this favours keeping several groupings (SE's gain, +1.31 emotion and +1.12 style, is the stage-2 analogue of CACTUs' P = 1 against P = 100). (c) Ye et al. and the CACTUs caveat say the conditioner can fit the bank, not the aspect; the safeguard is a condition-free pathway, as in `scan_N4.md`. (d) Quality against diversity: no source ranks them for our case; SE's result (each single grouping lost the other aspect) says diversity across groupings beat quality of one.

## Interface options

| Option | What stage 2 receives | Evidence for | Evidence against | Cost |
|---|---|---|---|---|
| H. Hard groups | one id per row per grouping; support share an id, negatives from other ids | CACTUs, UMTRA, SE all use hard task construction | false negatives from siblings (FNC, IFND); signal lost in placement only matters at test | none (today) |
| H+X. Hard groups, sibling-excluded negatives | ids plus S; episodes skip negatives whose group has S above a threshold | IFND removes detected false negatives; FNC cancels them | threshold has no label-free choice yet; detection can mis-remove (reader-inferred) | low |
| Soft. Soft assignments | posterior vector per row; episode agreement p_imgᵀ S p_txt | ASCL soft inter-sample relations; Meta-GMVAE soft episode modes; FNC attraction | posteriors from heads are weak for affect (13.5% image head, brief §2); Li et al. use hard assignments with prototypes | medium (needs heads at bank time) |
| S. Groups plus similarity matrix | ids and S, used as soft targets or weights in the loss | PCL multi-granularity prototypes carry hierarchy; ASCL | no pseudo-task paper tests it; S from a Leiden hierarchy or centroids is untested | low to medium |
| W. Per-grouping weights | one scalar per grouping, sampling probability of its episodes | Bansal et al. and Cui et al.: task distribution matters | no label-free quality criterion known (T5); CACTUs samples partitions uniformly | low if uniform, unknown otherwise |

## Bearing on the designs P0, L, G, E

- **P0** (one grouping per source, sibling-aware agreement): the bank is H or H+X per grouping. Literature-supported: mixing partitions helps (Hsu et al., 2019; SE internal). Sibling-aware agreement at read time is reader-inferred from the false-negative literature (Huynh et al., 2022; Feng & Patras, 2022).
- **L** (kept groupings refined jointly): refined groupings should still be handed over as separate hard banks plus S; mixing refined and original banks follows the multi-partition logic (Hsu et al., 2019). Reader-inferred.
- **G** (one shared Leiden partition across layers): a single partition removes the multi-partition benefit (Hsu et al., 2019) unless layer-specific sub-groupings are exported as extra banks; the hierarchy can supply S (Li et al., 2021 for multi-granularity). Reader-inferred.
- **E** (competing property heads): heads give soft assignments, so the Soft option applies, with the Meta-GMVAE precedent for soft episode modes (Lee et al., 2021), but overlap with stage 2 is high and the pseudo-to-real gap risk (Ye et al., 2020/2022) rises when the conditioner is trained on its own heads. Reader-inferred.

## Recommendations

1. Keep H as the control. Add H+X with S from the Leiden hierarchy or centroid similarity; compare to H on the same banks and seeds, with the matched condition-free control.
2. Keep all groupings in the bank and sample them uniformly (SE, CACTUs); test weighting only with a label-free criterion from T5.
3. Preserve a condition-free pathway so a bank-specific conditioner cannot hurt real tasks.
4. Do not tune K for the bank; test granularity only through the existing Leiden sweep.

## Limitations

Abstract-level reads for most papers; no pseudo-task paper matches our cross-modal, per-row, value-disjoint setting; the false-negative literature is for self-supervised instance contrast, not episode banks. Search bound: about 10 WebSearch queries and 12 page fetches; Semantic Scholar, CVF and Google Scholar not searched directly. Absence claims hold only within this search.

## References

Bansal, T., Gunasekaran, K., Wang, T., Munkhdalai, T., & McCallum, A. (2021). Diverse distributions of self-supervised tasks for meta-learning in NLP. *Proceedings of EMNLP 2021*. arXiv:2111.01322. https://arxiv.org/abs/2111.01322

Chen, T.-S., Hung, W.-C., Tseng, H.-Y., Chien, S.-Y., & Yang, M.-H. (2022). Incremental false negative detection for contrastive learning. *ICLR 2022*. arXiv:2106.03719. https://arxiv.org/abs/2106.03719

Cui, W., Wu, T., Cresswell, J. C., Sui, Y., & Golestan, K. (2025). DRESS: Disentangled representation-based self-supervised meta-learning for diverse tasks. arXiv:2503.09679. https://arxiv.org/abs/2503.09679

Feng, C., & Patras, I. (2022). Adaptive soft contrastive learning. *ICPR 2022*. arXiv:2207.11163. https://arxiv.org/abs/2207.11163

Hsu, K., Levine, S., & Finn, C. (2019). Unsupervised learning via meta-learning. *ICLR 2019*. arXiv:1810.02334. https://arxiv.org/abs/1810.02334

Huynh, T., Kornblith, S., Walter, M. R., Maire, M., & Khademi, M. (2022). Boosting contrastive self-supervised learning with false negative cancellation. *WACV 2022* (venue not confirmed on the page read; arXiv v-revised January 2022). arXiv:2011.11765. https://arxiv.org/abs/2011.11765

Jang, H., Lee, H., & Shin, J. (2023). Unsupervised meta-learning via few-shot pseudo-supervised contrastive learning. *ICLR 2023*. arXiv:2303.00996. https://arxiv.org/abs/2303.00996

Khodadadeh, S., Bölöni, L., & Shah, M. (2019). Unsupervised meta-learning for few-shot image classification. *NeurIPS 2019* (venue not confirmed on the page read). arXiv:1811.11819. https://arxiv.org/abs/1811.11819

Lee, D. B., Min, D., Lee, S., & Hwang, S. J. (2021). Meta-GMVAE: Mixture of Gaussian VAE for unsupervised meta-learning. *ICLR 2021*. OpenReview wS0UFjsNYjn. https://iclr.cc/virtual/2021/poster/3317

Li, J., Zhou, P., Xiong, C., & Hoi, S. C. H. (2021). Prototypical contrastive learning of unsupervised representations. *ICLR 2021*. arXiv:2005.04966. https://arxiv.org/abs/2005.04966

Ye, H.-J., Han, L., & Zhan, D.-C. (2020). Revisiting unsupervised meta-learning via the characteristics of few-shot tasks. *IEEE TPAMI*. arXiv:2011.14663. https://arxiv.org/abs/2011.14663

## Brief citation check

- CACTUs, Hsu, Levine and Finn, ICLR 2019, "Unsupervised Learning via Meta-Learning": confirmed (arXiv 1810.02334).
- UMTRA, Khodadadeh et al. 2019: confirmed; the title is "Unsupervised Meta-Learning For Few-Shot Image Classification" (arXiv 1811.11819), authors Khodadadeh, Bölöni, Shah. Venue NeurIPS 2019 not confirmed on the page.
- PsCo: confirmed, Jang, Lee, Shin, ICLR 2023 (spotlight), arXiv 2303.00996.
- Meta-GMVAE: confirmed, Lee, Min, Lee, Hwang, ICLR 2021; no arXiv id found, cite by OpenReview.
- "Hsu et al. 2019" in the brief is the CACTUs paper; the year is the ICLR 2019 venue year (arXiv 2018).
- Ye, Han and Zhan: the TPAMI year is not confirmed; arXiv 2020.

## Not verified

- FNC venue (WACV 2022 recalled, not confirmed on page); UMTRA NeurIPS 2019 venue.
- Whether Ye et al.'s Table IX numbers hold: relayed from `scan_N4.md`, which read them through a summary tool.
- "Coarse-to-fine pseudo supervision guided meta-task optimization" (Pattern Recognition 2021) appeared in search only; not opened, not cited.
- Any quantitative effect of cluster count on pseudo-task transfer: none found.
