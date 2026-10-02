# CoSiR v2: CVPR publication plan (design)

**Date:** 2026-10-02
**Status:** sections 1 to 6 approved by the user in brainstorming; the written spec awaits the user's review.
**Replaces:** the archived conditional-buddies plan (`docs/archive/buddy_publication_plan/`) as the project's
publication target.
**Builds on:**
- the handoff `docs/superpowers/handoffs/2026-10-02-cosir-v2-cvpr-publication-handoff.md`;
- the brainstorm progress note `docs/superpowers/handoffs/2026-10-02-cvpr-plan-brainstorm-progress.md`;
- the reports listed in §2.

## 1. Goal, venue and dates

Publish CoSiR v2 at **CVPR**. Hard dates (from the user):

| Milestone | Date |
|---|---|
| Abstract registration | Tue 2026-11-10 |
| Paper | Mon 2026-11-16 |
| Supplementary material | Mon 2026-11-23 |

The user's four requirements: strong results, a good storyline, clear baselines on comparable benchmarks, and a
precise problem definition with a method contribution and an explanation of why it works.

**Compute.**
- DAS6 node404 is available (normally up to 3 GPUs; 3 to 6 nodes when the cluster is idle; the user books them).
- The local RTX 3090 is shared with other sessions, so every GPU job runs under `flock -n -o -E 75 /tmp/gpu0.lock`.
- Long jobs are launched by the main session in the background.

## 2. Evidence that shaped this plan (2026-10-02; selection rows only, held rows untouched)

| Report | What it established |
|---|---|
| [affect factor learning, held test](../../reports/auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md) | SE (factors trained with value-type condition episodes from GoEmotions affect clusters and CLIP image clusters) beats its matched control C0 on value episodes: emotion +2.08, style +0.65 R@1. Distant supervision, not label-free discovery. |
| [CVPR literature review](../../reports/auto/v2/2026-10-21_cvpr_literature_review.md) | No prior example-conditioned cross-item image–text similarity (to our knowledge). The naive rule is Rocchio relevance feedback; the weighted factor term is a CSN mask. "Unsupervised condition discovery" is already taken (SCE-Net, DiscoverNet; EmotionCLIP for distant affect supervision). PercepT itself uses ModernBERT-GoEmotions and CLIP ViT-L/14, so our RoBERTa teacher must be named exactly, not called "PercepT's teacher". |
| [support-baseline spike](../../reports/auto/v2/2026-10-22_support_baseline_spike.md) | On value episodes, raw-CLIP support baselines beat SE (logistic probe 24.10 vs 21.22 pooled R@1). A prototype that ignores the query reaches 22.83, so value episodes are few-shot classification. SE leads only cross-modally. |
| [aspect-episode spike](../../reports/auto/v2/2026-10-23_aspect_episode_spike.md) | On aspect episodes no factor model beats CLIP (SE 11.32 vs CLIP 11.13). A label-probe ceiling reaches about 23, so the task is learnable. Emotion lives in captions and style in images, which caps cross-modal matching on ArtELingo. |
| [aspect novelty check](../../reports/auto/v2/2026-10-24_aspect_task_novelty_check.md) | No paper defines the aspect task, but every property exists separately. Closest: metric learning from pairs (Xing, RCA, ITML, KISSME), Contextual Visual Similarity (arXiv 1612.02534), MARS (ICLR 2023), in-context text embedders. The expected reviewer line is "a few-shot cross-modal diagonal KISSME". |
| [backbone check](../../reports/auto/v2/2026-10-25_backbone_check.md) | Weak-side probes are flat across four encoders (emotion from images 35.1 to 36.6, style from captions 25.4 to 26.3), so the asymmetry is in the data. The ceiling sums tie among the strong backbones. CUB primary colour is symmetric across modalities. The user chose Qwen3-VL-Embedding-2B. |

## 3. Problem definition (approved)

**Example-conditioned aspect similarity across modalities.**

- **Input:** a query x (an image or a caption) and a condition c = (S, C).
  - S holds a few **support pairs**: each an image of one item and a caption of another item, that agree on the
    wanted aspect with any values.
  - C holds a few **contrast pairs** that agree on a different aspect.
- **Output:** a ranking of candidates y in the other modality by s(x, y | c), such that candidates sharing the
  query's value on the aspect demonstrated by S rank first.
- **Rules:**
  - The condition never shows the query's value and never names the aspect.
  - Both directions (image→text, text→image) are scored.
  - **Swap test:** the same query and gallery under the swapped condition (S and C exchanged) must re-rank.

**Relation to GeneCIS.** GeneCIS focus conditions name the aspect in text and compare image with image. Ours shows
the aspect by examples and compares image with caption. GeneCIS focus attribute is the closest standard benchmark
(condition = attribute type, i.e. an aspect). Focus object conditions on a value (a named object), so it is
supplementary only.

## 4. Contributions and claims (approved)

- **C1, task and protocol.** Novelty statement: *"to our knowledge, the first cross-modal similarity in which the
  aspect is fixed at test time only by value-disjoint, cross-item image–caption examples with a contrast aspect,
  evaluated in both directions with a paired swap test."*
  - Neighbours we cite: metric learning from pairs, Contextual Visual Similarity, MARS, GeneCIS, CLAY, CRL, in-context
    embedders.
  - Motivation: value episodes are solved by supports alone (K1).
  - We do **not** claim that inferring a notion of similarity from examples or pairs, test-time reweighting, or
    relation-by-example retrieval is new.
- **C2, method.** A shared sparse image–text factor basis, trained on pseudo-aspect episodes with no labels from the
  evaluation taxonomy, is the prior that makes test-time metric estimation from four pairs work.
  - The training-free agreement rule is presented explicitly as a Rocchio/KISSME-style estimator, not as the novelty.
  - Distant affect supervision (GoEmotions RoBERTa `SamLowe/roberta-base-go_emotions`, named exactly) is disclosed
    where a dataset uses it.
- **C3, analysis.** Aspects live asymmetrically across modalities: emotion in captions, style in images,
  colour in both. The weaker modality caps cross-modal matching, and this holds across four backbones, so it is a
  property of the data.

| # | Claim | Evidence | Status |
|---|---|---|---|
| K1 | Value episodes are solved by the supports alone | prototype vs model, query ablation | **done** (support spike) |
| K2 | Our method beats backbone-only, raw-feature metric-from-pairs baselines and the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset | pre-registered final tests (§10) | open; factors are at CLIP level today |
| K3 | Examples beat names on subjective aspects and match them on objective ones | CRL, Qwen with the aspect in its instruction, privileged names | open |
| K4 | Gains hold on a strong backbone (Qwen3-VL-Embedding-2B) | second backbone | open |
| K5 | GeneCIS focus attribute: competitive with the published frozen-B/32 rows (example and text protocols reported apart) | GeneCIS runs | open |
| K6 | The rule selects aspect factors, and the factors are shared across modalities | ablations, swap analysis, per-factor analysis | open |
| K7 | The gain comes from the learned basis | same rule on raw, PCA/NMF/SpLiCE and learned factors | open |

**Fallback.** If K2 fails, K1, K3 and C3 can carry a task, benchmark and analysis paper but not the method paper. The
switch is discussed with the user (decision point Oct 9, §11).

## 5. Benchmarks and protocols (approved; one item open)

### 5.1 Common protocol

- **Aspect episodes** (as in the aspect spike):
  - an anchor, and 13 candidates [p_A, p_B, 11 that share neither aspect];
  - P_A and P_B are each 4 cross-item pairs, sharing A (resp. B) with values distinct from each other and from the
    anchor's;
  - condition A uses S = P_A and C = P_B; condition B swaps them;
  - every item in an episode comes from a distinct painting or photo.
- **Aspect pairs:** with three aspects, all three pairs are used.
- **Primary metric:** R@1, the mean over both directions and all conditions.
- **Secondary metrics:** per condition and direction; swap success (pairwise: p_A above p_B under A, and p_B above p_A
  under B), always read next to R@1; the other-aspect rate; a 101-candidate gallery variant on ArtELingo and CUB.
- **Splits:** factor training uses image–caption pairs and pseudo-partitions only, no labels. A labelled development
  split serves model selection. The test split is read within the budget of §10.

### 5.2 Datasets (priority order)

| Dataset | Aspects | Captions | Test split | Notes |
|---|---|---|---|---|
| **ArtELingo** (primary) | emotion (8) × style (23) × **genre** (open item, §5.3) | human, affective | held rows, fresh-seed aspect episodes | asymmetric aspects; label-probe ceiling about 23 R@1 |
| **CUB-200-2011 + Reed captions** | primary colour (15), bill shape (9), and a third group chosen in E0 (§12) | human, 10 per image, no species names | the zero-shot split's **50 unseen species** (approved); development on 30 of the 150 training species, kept out of factor training | colour is symmetric across modalities; captions name colours |
| **GeneCIS focus attribute** | condition = attribute type | none (Visual Genome crops) | the benchmark (2,000 templates) | image→image; example protocol (supports from other templates with the same condition) is ours; text protocol is a stretch (§5.4) |
| **SemArt** | type (10), school (26), timeframe (22) | catalogue descriptions, artist names and dates scrubbed | official test (1,069); development = official val | main paper if on time, otherwise supplementary |
| GeneCIS focus object | condition = an object (value-type) | COCO | benchmark | supplementary only |
| Affection | | | | potential; the user will check |

Data on disk (`/data/SSD/`):
- `cub/` (11,788 images, Reed captions in `captions/extracted/text_c10`);
- `semart/SemArt` (21,384 images);
- `visual_genome/VG_100K_all` (108,249 images, all 23,640 GeneCIS attribute images present);
- GeneCIS COCO half preprocessed earlier (`/data/PDD/genecis/`).

Point `/project/genecis/config.py` at `VG_100K_all`. Do not edit the GeneCIS clone in place; pass the path through
our own loader.

### 5.3 Open item: ArtELingo genre

The user wants genre kept: it is the one ArtELingo aspect visible in both modalities (needed for C3), and three
aspects avoid the "binary toggle" objection.

- The genre labels on disk cover only 1,160 of 61,402 paintings (from ArtELingo-28).
- WikiArt's 10-class genre labels come from ArtGAN's `genre_train.csv` / `genre_val.csv`. The user runs the download
  (an egress hook needs their approval) into `/data/SSD/wikiart_genre/`.
- **Rule:** keep genre if, after joining by file name, at least half of the selection and held paintings are covered
  and every genre has at least 30 paintings in each of those splits.
  - Otherwise, use a coarse subject label from the WikiArt tags (portrait, landscape, religious, still life),
    disclosed as derived.
  - If that also fails coverage, keep two aspects.

### 5.4 GeneCIS protocols

1. **Example protocol (ours):** image→image on image codes; supports and contrasts from other templates; not
   comparable with published numbers.
2. **Text protocol (stretch):** a phrase-to-weights adapter w = a_T(phrase), needed for a row beside SEARLE 14.4,
   CIReVL 15.9 and OSrCIR 17.4 (OSrCIR's reproduction at 14.0 noted).

Always reported: image-only, text-only and image+text on our backbones. GeneCIS's CC3M training triplets are never
used.

## 6. Method A: aspect-trained factors (approved)

**Model.**
- Two small encoders map frozen backbone features of each modality into a shared sparse non-negative space R₊^L
  (L = 32; 64 is a grid point).
- Score: s = β·cos + Σ_l w_l a_I,l(x) a_T,l(y).
- **Agreement rule** (training-free):
  w = ReLU(mean over S of a_I(x_i) ⊙ a_T(y_i) − mean over C of a_I(x_j) ⊙ a_T(y_j)), L1-normalized.

**Training: pseudo-aspect episodes.**
- Each training row carries cluster ids from K pseudo-partitions. Episodes copy §5.1 with pseudo-labels: for a
  partition pair (k, k′), supports are 4 cross-item pairs sharing a k-cluster different from the anchor's; contrasts
  are 4 pairs sharing a k′-cluster; candidates are one sharing only the anchor's k-cluster, one sharing only its
  k′-cluster, and negatives.
- **Losses:**
  - a ranking loss through the differentiable agreement rule, in both directions;
  - a swap term on the same episode with roles exchanged;
  - the R3 recipe's pair agreement (InfoNCE) and decorrelation terms, so that codes stay cross-modal and do not
    collapse.

| Dataset | Pseudo-partitions (no evaluation labels) |
|---|---|
| ArtELingo | GoEmotions affect k-means of captions (emotion-like; distant supervision), backbone image k-means (style-like; AMI 0.32 with style), caption-content k-means (genre-like) |
| CUB (150 training species) | backbone image k-means; **per-sentence** caption k-means (Reed's captions focus on different parts) |
| SemArt | image k-means; description k-means |
| GeneCIS | a factor model trained on COCO train2014 pairs (no GeneCIS images), with image and caption k-means |

**Go/no-go (pre-registered before the first run; ArtELingo, B/32, selection rows, decided Fri Oct 9).**
- At most about 10 runs (partition set × L × loss weights), about 10 minutes each locally. The best run is
  **picked on seed-42** selection aspect episodes.
- The GO test then uses **fresh seed-43 episodes** from the same selection rows, so picking the best of about ten
  runs does not inflate the result.
- **GO** if, on those fresh episodes, the picked run's pooled aspect R@1 (mean of directions and conditions) has a
  95% CI lower bound above 0 against **both** backbone-only (11.1) **and** the best raw-feature metric-from-pairs
  baseline, with that baseline's fusion weight cross-fitted. Swap success is reported at matched R@1.
- **Strong GO:** pooled R@1 of about 15 or more (a third of the gap to the ceiling of about 23).
- **NO-GO:** stop the method line and discuss the fallback framing with the user.
- A winning run is replicated at seeds 43 and 44 before moving to other datasets and Qwen.

**Open risk R-pseudo** (flagged by the user, accepted for now): the pseudo-partitions are proxies, and the model may
learn the episode format rather than aspects. The tests are the cross-dataset and unseen-species results.

## 7. Backbones (approved)

- **Development** on CLIP ViT-B/32 (cached).
- **Final tables** also on **Qwen3-VL-Embedding-2B** (user's choice). Its instruction interface gives the "aspect
  named in the instruction" baseline on the same backbone.
- Before any paper use, the reimplementation in `src/test/20261025_backbone_check/extract.py` (transformers 5.6.2,
  last-token pooling, default instruction, max_pixels capped) must be **checked against the official
  implementation**. If it fails, fall back to PE-Core L/14.
- Packages go in `pip --target` directories, never in the CoSiR env.

## 8. Baselines (approved)

All run on the same features and episodes.
- **On development splits,** fusion weights (and β) are cross-fitted by anchor parity.
- **For the final reads,** they are chosen on the development split and frozen. GeneCIS, which has no development
  split, uses the weights frozen on the other datasets.

| Tier | Baseline | Reviewer question |
|---|---|---|
| 1 | backbone only (cosine) | is conditioning needed? |
| 1 | metric from pairs on raw features: diagonal KISSME (our rule on raw coordinates, signed and ReLU), low-rank KISSME with shrinkage, RCA, Xing-style per-episode fit, Wang et al. per-query weights | "your rule is few-shot KISSME" |
| 1 | the agreement rule on PCA-32/64, NMF-32, SpLiCE (B/32) | K7 |
| 1 | value prototype or Rocchio | value-type baselines on aspect episodes |
| 1 | C0, SE, R3 (our ablations) | what aspect episodes add |
| 2 | names with privileged vocabulary (projection onto true value names) | upper reference for naming |
| 2 | CRL (LLM-listed values from one aspect word, projection) | K3 |
| 2 | Qwen3-VL-Embedding with the aspect in its instruction | would an instruction embedder suffice? |
| 3 | label-supervised probe ceiling (converged, fixed threads) | distance to supervised |
| 3 | GeneCIS published frozen-B/32 rows and image / text / image+text | K5 |
| stretch | CLAY reimplementation; in-context MLLM reranker given the example pairs | strongest names and in-context comparisons |

**Out of the tables.**
- **Teacher-only** (GoEmotions cannot read images) becomes a text-side analysis inside C3.
- **PercepT** is a side branch that showed the buddy graph works on an existing benchmark and inspired v2. A
  buddy-vs-PercepT comparison is **optional and storyline-dependent** (supplement or analysis), never required.

## 9. What this plan does not do

- No backbone fine-tuning.
- No evaluation labels in factor training.
- No GeneCIS CC3M triplets.
- No merging of percept-branch code.
- Codex only if the user asks.
- No held read outside §10.

## 10. Held budget and statistics (approved)

**Ledger.** Every final-split read is recorded in `docs/superpowers/held_ledger.md` (date, script SHA-256, episode
SHA-256, purpose). Each final script refuses a second run.

| Dataset | Development | Final test | Budget |
|---|---|---|---|
| ArtELingo | selection rows (free to reuse; read many times) | held rows, fresh-seed aspect episodes (a new task on rows read 3 times before for value episodes) | 1 main + 1 reserve |
| CUB | 30 of the 150 training species | 50 unseen species | 1 + 1 |
| SemArt | official val | official test | 1 + 1 |
| GeneCIS | none (zero-shot; selection on the other datasets) | benchmark | 1 + 1 |

- **Main read:** one pre-registered run per dataset, covering all models and baselines on both backbones, with
  checkpoints, fusion weights, β and episode counts frozen.
- **Reserve read:** only for a pre-registered fix after a final-review finding.
- **Disclosure:** ArtELingo's held rows shaped earlier design decisions; the paper discloses this.

**Statistics.**
- Primary metric: aspect R@1, the mean over directions and aspect pairs.
- Uncertainty: paired bootstrap over anchors (5,000). Our models train with 3 seeds; the headline is the 3-seed
  mean, with its CI from bootstrapping anchors on per-anchor seed means; every seed is also reported.
- **Pre-declared primary comparisons** ("beats" means CI lower bound > 0):
  - ours vs backbone only;
  - ours vs the best raw metric-from-pairs baseline;
  - ours vs Qwen with the aspect in its instruction ("beats" or "matches" per aspect type, declared in the
    pre-registration).
  - All other comparisons are descriptive.
- Episode counts come from a power calculation on selection variance (default 4,096 anchors per aspect pair).
- Swap success is always reported next to R@1.

## 11. Experiment plan and schedule (approved)

| # | Experiment | Dates | Output or decision |
|---|---|---|---|
| E0 | Setup: a generic aspect-episode module (`src/eval/`) for all datasets; held ledger; genre coverage (§5.3); CUB third aspect group (highest min(image, caption) probe gain over majority among `has_shape`, `has_wing_pattern`, `has_breast_pattern`, `has_wing_color`); Qwen fidelity check | Oct 2 to 4 | module with tests; decisions recorded |
| E1 | Tier-1 baselines on ArtELingo selection aspect episodes (B/32) | Oct 3 to 6 | baseline table; the KISSME bar for the go/no-go |
| E2 | ArtELingo pseudo-partitions (affect and image k-means exist; caption-content k-means new) | Oct 4 to 5 | partitions plus their AMI with the labels (diagnostic) |
| E3 | Method A training and go/no-go (at most about 10 runs) | Oct 5 to 9 | **GO / NO-GO, Fri Oct 9** |
| E4 | node404 extraction: Qwen on full ArtELingo; B/32 and Qwen on SemArt, GeneCIS VG crops and COCO, COCO train2014 | Oct 5 to 12 | feature caches under `/data/SSD2/pre_extract/` |
| E5 | Replication seeds 43 and 44 on ArtELingo | Oct 10 to 11 | seed table |
| E6 | CUB: partitions, training, selection evaluation | Oct 10 to 16 | **CUB check, Oct 16** (if A does not beat the baselines on CUB development, claims narrow to ArtELingo plus analysis) |
| E7 | GeneCIS focus attribute: example protocol, COCO-trained factors, baselines; text protocol as stretch | Oct 13 to 20 | GeneCIS table |
| E8 | SemArt (scrubbing, partitions, training, selection) | Oct 15 to 22 | main or supplementary |
| E9 | Qwen backbone runs on every dataset | Oct 12 to 22 | K4 |
| E10 | Tier-2 baselines: CRL, Qwen instruction names, privileged names | Oct 12 to 20 | K3 |
| E11 | Ablations: rule on bases (K7), no episodes and value episodes, factor-group analysis, swap analysis (K6) | Oct 14 to 23 | ablation tables |
| E12 | C3 analysis: converged ceilings across datasets and backbones, teacher text-side analysis | Oct 16 to 30 | analysis figures |
| E13 | Pre-registration and power for the final reads | Oct 21 to 23 | **methods frozen Oct 23** |
| E14 | Final reads (one main read per dataset) | Oct 24 to Nov 1 | **main-paper experiments frozen Nov 1** |
| E15 | Writing: draft from Oct 26, full draft Nov 6; title and abstract fixed Nov 7; registration Nov 10 | Oct 26 to Nov 16 | paper |
| E16 | Whole-branch final review (most capable model; re-derives every load-bearing number) and fix wave | Nov 11 to 14 | review report |
| E17 | Supplementary: slipped datasets, GeneCIS focus object, 101-candidate galleries, stretch baselines, optional buddy vs PercepT | Nov 16 to 23 | supplementary |

Every experiment ends with a report in `docs/reports/auto/v2/`, one row in `reports_sum.md`, and a commit. Results
update §4's claims table.

**Planning scope.**
- The first implementation plan covers **E0 to E5** (setup, baselines, partitions, go/no-go, replication, and the
  node404 extraction that runs alongside) in task-level detail.
- E6 onward gets its own plan after a GO on Oct 9, informed by what E3 finds.
- After a NO-GO, the next plan is the fallback framing, agreed with the user.

## 12. Risks

| ID | Risk | Mitigation or signal |
|---|---|---|
| **R-pseudo** | pseudo-aspect partitions are proxies; the model may learn the episode format | cross-dataset and unseen-species results; disclose distant supervision |
| R-nogo | A fails on Oct 9 | fallback framing (K1, K3, C3), discussed with the user before switching |
| R-kissme | a raw-feature metric from pairs matches A | K7 fails and the basis is not the contribution; known by Oct 9 |
| R-names | Qwen with the aspect in its instruction beats examples | narrow K3 to where names fail (subjective aspects) |
| R-ceiling | ArtELingo's ceiling is low (about 23) | relative gains; CUB colour as the symmetric case; C3 turns the cap into a finding |
| R-time | 4 datasets × 2 backbones in about 5 weeks | staged priority; SemArt moves to the supplementary first |
| R-qwen | reimplementation differs from the official one | E0 fidelity check; PE-Core fallback |
| R-infra | local GPU loss (2026-10-02); a shared machine | node404 as backup; GPU lock; commit after every task |
| R-held | ArtELingo held rows shaped earlier design | fresh-seed episodes on a new task; disclosure |
| R-licence | CUB terms conflict (non-commercial vs cc-by); SemArt and GeneCIS are CC BY-NC | academic use; state the terms in the paper |
| R-concurrent | TPIPS, SteerViT, CLAY, Fioresi et al., COCO-Facet | position on the example interface and the cross-modal score |

## 13. Conventions

- Implementation goes to Claude Code subagents sized to the task; the controller reviews every result.
- Reports follow `docs/reports/` layout and the user's report rules: a real baseline beside every number, figures,
  paper-draft style, no dashes.
- Other sessions share main: stage files by explicit path.
- Debug and experiment folders: `src/test/<sequence-date>_<name>/`.
- Change logs for edits to existing source go in `.claude/yyyymmdd_log.md`.
