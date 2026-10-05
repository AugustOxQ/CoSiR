# Research brief: the grouping component of CoSiR v2 (stage 1) and its interface to stage 2

Written 2026-10-05 from a design discussion with the user. Input for an ARS deep-research run; the scans and the
synthesis go next to this file (`scan_T1.md` to `scan_T6.md`, `synthesis.md`), as in
`src/test/20261107_new_method_candidates/`. **The designs below are hypotheses, not literature findings. Every paper
named here was named from memory and must be verified (title, authors, venue, year, arXiv id) before it is cited.**

## 1. The task and the pipeline, for a reader without the project's context

**Task** (spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §3, §5.1). A query (an
artwork image or a caption) comes with 4 *support pairs* and 4 *contrast pairs*. A support pair is an image of one
painting and a caption of another that share a value of an unnamed aspect A; the four pairs show four different values,
never the query's. Contrast pairs do the same for a second aspect B. The system ranks 13 candidates in the other
modality; the target shares the query's value of A, a distractor its value of B. On ArtELingo the aspects are
**emotion** (8 values, labelled per viewer row), **style** (WikiArt art style such as Impressionism or Baroque, 23
values used, per painting) and **genre** (10 values, per painting). Metrics: R@1, condition gain (R@1 minus the rate at
which B's candidate ranks first; 0 for any scorer that ignores the condition) and either rate, with
R@1 = (either + gain) / 2. Rows are split by painting; all groupings and heads use the 183,694 scorer-train rows
(36,518 paintings); development episodes use selection rows.

**Stage 1, grouping (the subject of this brief).** Label-free groupings of the scorer-train rows that define the
"pseudo-aspects" the system can read. Today there are three (E2, `docs/reports/auto/v2/2026-10-31_pseudo_partitions.md`):

| Grouping | Built from | Method |
|---|---|---|
| affect | the 28 GoEmotions probabilities of the caption (`SamLowe/roberta-base-go_emotions`, frozen, never trained on ArtELingo) | MiniBatch k-means, 64 clusters, raw probabilities |
| image | CLIP ViT-B/32 image features | MiniBatch k-means, 64 clusters, L2-normalised |
| caption | CLIP ViT-B/32 caption features | MiniBatch k-means, 64 clusters, L2-normalised |

**Placement.** For each grouping and modality, a logistic-regression *head* on frozen CLIP features (60,000
scorer-train rows) gives a lone image or a lone caption a posterior over the groups (N6,
`src/test/20261108_new_method_quick_checks/run_n6.py`).

**Reader.** *Agreement* of an image and a caption on a grouping is the dot product of their posteriors. For each
grouping, Δ = mean agreement over the support pairs minus over the contrast pairs; the reader picks the grouping with
the largest Δ and scores candidates by agreement with the query on it (`src/eval/aspect_quick_checks.py`,
`aspect_deltas`, `inferred_scores`).

**Stage 2, factor learning (method A).** 32 shared non-negative image and caption factors trained on *pseudo-aspect
episode banks* built from the stage-1 groupings: a bank episode is a task episode with groups in place of labelled
values, so its support pairs share a group by construction. A3, whose factors enter today's best condition-free score
B (R@1 18.34 on development seed 42), was trained on E2's banks. The design of factor learning is handled in a
separate discussion ("leiden-plan"); this brief covers stage 1 and **what stage 2 consumes from it**.

**Oracle layers** (stage report `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md` §14; R@1 margin
of a fused term over its matched condition-free counterpart, seed 42, 95% painting-cluster intervals):

| Reader | Margin |
|---|---|
| label-free reader on today's groupings (N6) | +0.14 [−0.04, 0.32] |
| same heads, right grouping told (emotion to affect, style and genre to image; mapping chosen from AMI) | +1.14 [0.90, 1.41] |
| heads trained on the evaluation labels themselves (same 60,000-row draw), told | +6.66 [6.31, 7.00] |

The gap between the second and third rows is mostly the grouping (pseudo-groups against the true label classes, and
style and genre in separate blocks), since the head family and the training rows are the same.

## 2. What we know about the groupings

| Fact | Number | Source |
|---|---|---|
| Groupings were hand-matched to evaluation aspects | "each is hand-matched to one evaluation aspect; disclosed" | spec §6 table |
| 64 was inherited, never tested | default `n_clusters` of the stage (d) CLIP-cluster source, copied by the affect run and E2 | `src/train/condition_sources.py`; `src/test/20261018_affect_factor_learning/run_affect.py` |
| AMI with labels (affect / image / caption) | emotion 0.204 / 0.037 / 0.057; style 0.016 / 0.318 / 0.058; genre 0.023 / 0.397 / 0.161 | E2 §3 |
| Affect is close to one property | pair lift for emotion 2.17; style and genre 1.10 | `src/test/20261110_partition_profile/` log, check 3 |
| Emotion is spread over many affect clusters | median emotion needs 15 of 64 clusters for 80% of its rows (28 for a value spread like all rows); sadness: 35% in cluster 38 (93% sad), about 21% in five smaller mostly sad clusters, about 19% in two large mixed clusters | draft report `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md` §3 |
| Image mixes genre and style, genre dominant | pair lift genre 9.75, style 4.59; style-versus-genre contrast ratio 0.44; told gain on style × genre exactly 0 | profile check 3; stage report §14 |
| Caption repeats content | lift genre 2.44, emotion 1.26, style 1.20; never used by the told mapping | profile check 3 |
| Affect and image groupings are nearly independent | AMI 0.026 | affect report Result 1 |
| Signal is lost in placement | affect × emotion lift 2.17 on the groups, 1.11 through the heads | draft §3 |
| Head accuracy (image / caption head, held-out) | affect 13.5 / 34.6; image 92.7 / 22.9; caption 21.6 / 89.7 | draft Table 2 |
| Leiden beats k-means on affect | at 41 groups told +1.64 against +1.10; reader +0.35 against +0.16 | draft §4 |
| Group count is not the lever | Leiden sweep, 14 to 118 groups: told +1.64 to +1.95, k-means +0.97 to +1.24; cluster lift rises 1.99 to 3.45 while lift through heads stays 1.12 to 1.16 and image-head accuracy falls 21.2% to 4.8% | draft §5 |
| Leiden's count is set indirectly | graph k ∈ {10, 20, 40} and resolution ∈ {0.25, 1, 4} gave 14 to 118 groups | `src/test/20261112_community_sweep/results/sweep.txt` §1 |
| Every aspect has a weak modality, across backbones | emotion from captions 56.9% against images 35.2%; style from images 60.8% against captions 25.4%; CLIP, SigLIP 2, PE-Core, Qwen3-VL-Embedding-2B move the weak side by at most 1.5 points. Long-CLIP not tested | stage report §3 |

**Evidence from the percept line on fusing sources** (weekly report `docs/reports/weekly/2026-09-30_percept_buddy_to_v2.md`
§2 to §4; one node per painting, GoEmotions averaged over a painting's captions, genre from a different 1,303-painting
label set, training-split AMI unless noted, so not directly comparable with v2's numbers):

| Construction | Emotion AMI | Genre AMI | Reading |
|---|---|---|---|
| content buddy graph + Leiden | 0.059 | 0.438 | content dominates |
| affect-only graph + Leiden | 0.118 | 0.040 | |
| late fusion, union of both graphs, one partition | 0.124 | 0.139 | merged content communities (28 to 21) |
| hierarchical refinement: content communities split by affect | 0.107 (random split of the same sizes 0.036) | 0.195 (control 0.201) | a split inside a parent grouping kept real emotion signal |
| Attention-h1 student (InfoNCE on both teacher graphs) + Leiden | 0.135 (held 0.125) | 0.240 (held 0.240) | the fused embedding trades one property for the other |

The two graphs barely agree on neighbours (their intersection left 98.96% of paintings without an edge) although
the views share broad directions (top CCA correlation 0.73 against 0.07 shuffled). Six ways of attaching a DEC loss to
the student's embedding, plus one rerun after bug fixes, all lowered cluster separation and label agreement together;
PercepT's clearer clusters gave no emotion advantage downstream (0.0221 against 0.0231). A content-only student restates
CLIP (Block 1: emotion AMI 0.0369 against 0.0358).

**Evidence from stage 2 on keeping groupings separate.** Factor training whose conditions came half from affect clusters
and half from image clusters (SE) gained on emotion (+1.31 [0.54, 2.06]) and style (+1.12 [0.29, 1.93]) over the matched
control, where affect alone lost style (−1.23) and image alone lost emotion (−0.85) (affect report
`docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md`, verdict table; value episodes). As
condition sources for the stage (d) scorer, Leiden communities on a content graph scored +3.00 against +2.54 for CLIP
k-means (tie band, not tested on held; `docs/reports/auto/v2/2026-10-13_candidate_a_stage_d_selection.md`).

## 3. The problem as the user framed it, and decisions so far

The user's concerns: (1) one fixed K for every grouping; (2) the sources (CLIP image, CLIP caption, GoEmotions) were not
well chosen; (3) three groupings, none holding a single property; (4) groupings do not interact and are never updated,
unlike PercepT, whose latent space is reshaped during training. First idea: fuse several sources into one structure and
then split it ("fission") into clearer single-property sub-distributions that feed the heads.

Decided in the discussion (user, 2026-10-05):

- **Group count** (K for k-means, graph k and resolution for Leiden) is not the main lever; tune later, by label-free
  criteria.
- **Sources:** start with hand-matched sources (GoEmotions for emotion, CLIP image for genre, a style source), then test
  whether a **fixed generic source menu**, written down before the hand-matched results, does as well. Hand-matching is
  easier for a reviewer to question; the generic menu is the stronger claim.
- **Style source:** ideally not trained on WikiArt style labels (ArtELingo's images are WikiArt, so such an encoder would
  carry the evaluation taxonomy).
- **Updating the representation and the groups** during training is wanted; DEC is excluded by the evidence above.
- **Purity is a direction, not a requirement.** The reader's actual need is weaker: each aspect must dominate at least one
  grouping (affect is not pure yet works for emotion; image fails for style because genre dominates it).
- **Sibling-aware agreement** (Section 4, R3) is adopted as an idea.
- **Late fusion is kept** (design L). Merging the groupings into one consensus partition and then splitting it is
  dropped: nothing would record which source put rows together.
- **Scope:** stage 1 and its interface to stage 2. Factor learning itself, the GO bars, the reader design, the margins
  and the robustness runs are out of scope.

## 4. Working requirements for a grouping

- **R1, dominance.** Every aspect dominates at least one grouping, so that under its condition the right grouping's Δ is
  the largest. Simulation: mixed codes give 0.00 condition gain against 0.70 to 0.95 for aspect-block codes (spec §6).
- **R2, placeability.** A lone image and a lone caption can each be placed in the grouping, and agree. Today the affect
  image head is right 13.5% of the time, about 1.24 times the majority rate (told-oracle log §3). Likely ceiling: the rows
  of one painting share its image but can carry different viewers' emotions.
- **R3, sibling groups count as related.** Agreement is p_img · p_txt, so a sad image placed in cluster 3 and a sad
  caption placed in cluster 38 score near 0. Proposed: p_imgᵀ S p_txt with a label-free group-similarity matrix S
  (centroid similarity, co-membership or a Leiden hierarchy). Merging clusters instead costs purity.
- **R4, non-redundancy** across groupings.
- **R5, generic sources chosen by label-free criteria.** CUB, SemArt and GeneCIS follow; spec §6 already requires generic
  partitions on CUB. Settings are chosen by label-free scores, not by told or reader margins (which read labelled
  episodes).

## 5. Candidate designs (hypotheses to evaluate)

| | Sources merged at | Separation by | What is trained | Cost | Role |
|---|---|---|---|---|---|
| **P0** | not merged: one grouping per source (GoEmotions, CLIP image, a style source), Leiden each, sibling-aware agreement | not needed | heads | low | the control every fusion design must beat; tests the source swap |
| **L** | groupings (latest): the M groupings are kept, a small model refines them jointly | non-redundancy given the other groupings, placeability, staying close to its own source; source kept by construction | refinement model and heads | low to medium | cleans each grouping (genre out of a style grouping), adds cross-grouping links |
| **G** | graphs: one layer per source in a multiplex graph, one shared Leiden partition across layers | which layers support each community's internal edges; layer-specific sub-groupings | heads (or a SCAN-style head) | low to medium | first fuse-then-split; precedent: hierarchical refinement |
| **E** | edges (earliest): all sources' kNN edges pooled, each tagged with its source | competing property heads (each edge explained mainly by one head), source tags, image versus caption, non-redundancy between heads | a two-tower trunk (lone image or caption in) and property heads | high | the ambitious version; the representation and groups are updated. Its heads are aspect-block codes, so it overlaps stage 2 |
| dropped | one consensus partition | nothing left to split by | | | source lost |

Working order (to be confirmed after the research): P0, then L and G side by side, then E.

Practical notes: the groupings must be per row (emotion is per viewer), unlike the percept line's per-painting nodes; at
test time items arrive alone, so any trunk needs separate image and caption encoders; Leiden cannot place new items, so
placement stays with heads or a trained model.

## 6. Separation signals that need no evaluation labels

| Signal | What it separates | Where it is in our data | Used by |
|---|---|---|---|
| source provenance | structure each source carries alone | which source graph an edge came from | G, E |
| modality | what a lone image can place versus what only a caption can | image and caption heads | E; check for all |
| painting membership | painting-level properties (style, genre) from viewer-level ones (emotion) | about 5 rows per painting; painting id comes from the split, not from evaluation labels | E; ceiling check for all |
| non-redundancy | each grouping explains what the others do not | mutual information between groupings | G, E |
| conditional non-redundancy | what a grouping carries beyond the others | conditional mutual information | L |
| agreement versus disagreement between groupings | shared structure (redundant) from structure unique to one source | rows co-grouped in two groupings or in one only | L, G |
| head competition | each relation explained mainly by one head | per-edge head weights (as in SCE-Net) | E |
| augmentation invariance | appearance (changes under colour or texture jitter) from content (changes under crops) | image augmentations | style source, E |
| anchoring to the source | keeps a weak property (affect) from being absorbed by content | distance to the original grouping or graph | L, E |

## 7. Research threads

For each thread: what exists, the strongest published version of each idea, what it assumes, known failure modes, and
how it bears on P0, L, G and E. Mark each idea as exists, partially exists or not found (to our knowledge, within the
search).

**T1. Fusing several groupings or graphs, then splitting.**
- Multiple, alternative and non-redundant clustering, including clustering given an existing clustering (for example
  Gondek and Hofmann; Cui, Fern and Dy 2007; Niu, Dy and Jordan 2010).
- Multi-facet clustering (MFCVAE, Falck et al. 2021).
- Multi-view clustering with co-regularisation or co-training (for example Kumar, Rai and Daumé 2011).
- Multiplex and multilayer community detection (Mucha et al. 2010; leidenalg's multiplex optimisation), and
  layer-specific community structure.
- Condition discovery from mixed similarities: Conditional Similarity Networks (Veit et al. 2017), SCE-Net (Tan et al.
  2019), DiscoverNet (named in `docs/reports/auto/v2/2026-10-21_cvpr_literature_review.md`).
- Consensus or ensemble clustering (Strehl and Ghosh 2002) as the contrast case we dropped.
- Deliverable: for each fusion point (groupings, graphs, edges), the evidence that splitting afterwards recovers
  single-property structure, the split mechanism, and failure modes (in particular dominance of the strongest source).

**T2. Supervision for splitting without labels.**
- Identifiability of disentanglement: the negative result for unsupervised disentanglement (Locatello et al. 2019) and
  the routes that restore it: pairs of observations sharing some factors (Locatello et al. 2020), auxiliary variables
  (iVAE, Khemakhem et al. 2020), multiple views and augmentations (von Kügelgen et al. 2021, content and style).
- Deliverable: which signals of Section 6 have theory or evidence behind them and under which assumptions. The rows of a
  painting are the candidate analogue of pairs that share some factors (painting-level ones) and differ in others
  (viewer-level ones).

**T3. Losses that update the representation and the groups together (not DEC).**
- SCAN (Van Gansbeke et al. 2020; neighbour consistency, close to "buddy graph plus a clustering head"), SwAV (Caron et
  al. 2020; swapped prediction, here between a row's image and caption), SeLa (Asano et al. 2020), IIC (Ji et al. 2019),
  DeepCluster (Caron et al. 2018), cross-modal deep clustering (XDC, Alwassel et al. 2020).
- Deliverable: candidate losses for placeability across modalities, how each prevents collapse, whether each favours the
  dominant signal (the risk that content absorbs affect), and how a source anchor can be added.

**T4. Style sources not trained on WikiArt style labels, and backbone candidates.**
- Classical texture and style statistics (Gram matrices of an ImageNet network, Gatys et al. 2016; colour and texture
  descriptors); self-supervised or contrastive style encoders (for example CSD, Somepalli et al. 2024) and published
  art-style embeddings, **each with its training data checked for WikiArt style labels**.
- Style from text: whether captions name style, and any evidence on reading style from text.
- Long-CLIP (Zhang et al. 2024) and similar long-caption models as untested backbone candidates (ArtEmis caption lengths
  to be checked).
- Deliverable: a shortlist with training-data provenance, the expected style-versus-genre balance, and readability from
  each modality.

**T5. Judging a grouping before use, without labels; sibling-aware agreement.**
- Clustering stability across seeds and subsamples (for example Ben-Hur et al. 2002; Lange et al. 2004), cross-view
  predictability, non-redundancy measures, and any evidence that such criteria predict downstream usefulness.
- Group-to-group similarity inside an agreement score: hierarchy-based kernels, optimal transport between assignments
  with a ground cost between groups, soft clustering.
- Deliverable: a short list of criteria with their evidence, and two or three concrete formulas for p_imgᵀ S p_txt.

**T6. The interface between stage 1 and stage 2.**
- What a pseudo-aspect episode bank needs from the groupings: hard groups, soft assignments, or similarity-aware episodes
  (siblings as near-positives); evidence from meta-learning on pseudo-tasks built by clustering (for example CACTUs, Hsu
  et al. 2019) on how task construction from clusters transfers to real tasks.
- Deliverable: options for the interface and their evidence, to hand to the factor-learning discussion.

## 8. Output

- One file per thread, `scan_T1.md` to `scan_T6.md`: answers to the thread's questions, a short annotated list of the
  key papers (verified), and the bearing on P0, L, G and E.
- `synthesis.md`: a ranking of P0, L, G and E with reasons, the open risks, the label-free criteria to design against,
  and the first experiments the evidence supports, each with its matched control.
- Suggested modes (the user decides): a three-way scan per thread for T1 to T4, a quick brief for T5 and T6, then a
  methodology-focus review of the plan that follows.

## 9. Constraints on any recommended experiment

- No grouping choice reads the evaluation labels; labels may describe a chosen grouping once, disclosed.
- Matched controls: every configuration against the same score with only the condition removed.
- Develop on episode seed 42, test the final configuration on fresh seeds 49, 50 and 51.
- Compute: local CPU (32 cores; the RTX 3090 is shared with other projects) and DAS6 node404 if needed.
- CVPR abstract on 10 November 2026.

## 10. Checks queued for after the research (not run; label-free, CPU, minutes each)

1. Placeability (mutual information between image-head and caption-head assignments of the same row, against random
   pairs) for the 9 Leiden cells and 8 k-means controls of the sweep: does it order them as their told margins do?
2. Sibling-aware agreement on the existing Leiden groups: change in told and reader margins.
3. The same-painting ceiling of the affect grouping: how often two rows of one painting share an affect group, against
   two random rows.

## Sources read for this brief

Draft report `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md` (commit 33ff34b); E2 report and
`src/test/20261031_pseudo_partitions/build_partitions.py`; `src/train/pseudo_partitions.py`, `src/train/condition_sources.py`;
stage (d) and affect factor-learning reports; stage report of 2026-10-04 (§2, §3, §5, §9 to §11, §14); CVPR plan spec §3
to §6; handoff `docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`; weekly report of 2026-09-30 (§2 to §4);
`src/model/communities.py`, `src/model/graph.py`, `src/conditional_buddy/buddy_graph.py`; run folders
`src/test/20261110_partition_profile/`, `src/test/20261111_community_told_oracle/`, `src/test/20261112_community_sweep/`
(commit a6287af).
