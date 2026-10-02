<!--block:B0001-->
# CoSiR v2: CVPR publication plan (design)

<!--block:B0002-->
**Date:** 2026-10-02
**Status:** sections approved by the user in brainstorming; genre resolved (kept); this written spec awaits the user's review.
**Replaces:** the archived conditional-buddies plan (`docs/archive/buddy_publication_plan/`) as the project's
publication target.

<!--block:B0003-->
**In one paragraph.**
<!--block:B0004-->
- **The problem.** We will submit CoSiR v2 to CVPR (paper deadline Nov 16) on a newly defined problem:
  *example-conditioned aspect similarity across modalities*. A user shows, by a few example image–caption pairs,
  *in which respect* two things should count as similar, for example "similar in the emotion they convey". The
  system then matches images with captions of other items in that respect.
- **Why the problem changed.** On 2026-10-02 we found that our previous evaluation (examples of a *value*, such as
  "sad, like these") is solved by a 4-shot classifier that ignores the query. We also found that our current model
  cannot yet do the *aspect* version, although supervised probes show the task is learnable.
- **The plan.** Train the model's shared image–text factors for aspect selection. A go/no-go is set for Oct 9.
  The method is then tested on four benchmarks and two frozen backbones, against baselines that a reviewer will
  ask for. Each held-out test set is read once.

<!--block:B0005-->
![Timeline of the plan](assets/2026-10-02_cvpr_plan_timeline.png)

<!--block:B0006-->
*Figure 1. Experiments E0 to E17 (§11) by phase, with decision points (dashed lines) and hard deadlines (solid lines).
Built by `assets/build_2026-10-02_cvpr_plan_timeline.py`; its rows mirror the §11 table.*

<!--block:B0007-->
Terms in *italics* at first use are defined in the glossary (Appendix A).

<!--block:B0008-->
---

<!--block:B0009-->
## 1. Goal, venue and resources

<!--block:B0010-->
**Goal.** Publish CoSiR v2 at CVPR. The user's four requirements for the paper are:
<!--block:B0011-->
1. strong results;
2. a good storyline;
3. clear baselines on comparable benchmarks;
4. a precise problem definition with a clear method contribution and an explanation of why it works.

<!--block:B0012-->
**Dates** (from the user).

<!--block:B0013-->
| Milestone | Date |
|---|---|
| Abstract registration | Tue 2026-11-10 |
| Paper | Mon 2026-11-16 |
| Supplementary material | Mon 2026-11-23 |

<!--block:B0014-->
**Compute.**
<!--block:B0015-->
- **DAS6 GPU cluster.** The node and GPU count depend on what the user reserves: normally one node with up to 3
  GPUs, and 3 to 6 nodes when the cluster is idle.
  - Jobs go through the `cluster-run` CLI, and all node-side files live under `/local/wding/`.
- **Local RTX 3090.** It is shared with other Claude sessions and used for smoke tests and small runs. Every GPU job
  takes the shared lock `flock -n -o -E 75 /tmp/gpu0.lock`.
- **Long jobs** are launched by the main session in the background.

<!--block:B0016-->
## 2. Background: what CoSiR is and how we got here

<!--block:B0017-->
### 2.1 The research question

<!--block:B0018-->
CoSiR studies **conditional image–text similarity**. Whether an image and a caption "match" should depend on a
*condition*, i.e. on what the user cares about.
<!--block:B0019-->
- The direct ancestor is GeneCIS (Vaze, Carion and Misra, CVPR 2023). It ranks images for a reference image under a
  short text condition such as "colour". CoSiR's original `combiner.py` credits it.
- CoSiR set out to differ in two ways:
  1. **Conditions not drawn from a hand-designed list.** This is weaker now (§2.4).
  2. **Genuinely cross-modal similarity:** image against caption, rather than image against image.

<!--block:B0020-->
### 2.2 Three lines of work

<!--block:B0021-->
| Line | Period | What it was | Outcome |
|---|---|---|---|
| buddy | Jun to Sep 15 | per-sample condition vectors and a combiner on a "buddy graph" (mutual nearest neighbours across modalities) | archived plan; the condition mechanism stayed weak and asymmetric |
| percept | Sep 15 to 28 | ArtELingo affect work; a buddy-graph replacement for PercepT's (arXiv 2606.03345) topic stage | buddy topics matched PercepT on held-out label agreement; a matched head-to-head showed no PercepT lead. A side branch that showed the buddy graph works on an existing benchmark |
| **v2** | from Sep 28 | a ground-up rewrite, block by block (spec `2026-09-28-cosir-v2-ground-up-redesign.md`) | the subject of this plan |

<!--block:B0022-->
### 2.3 CoSiR v2 as it stood on the morning of 2026-10-02

<!--block:B0023-->
**Data: ArtELingo** (English part, `/data/PDD/artelingo/artelingo_train.json`).
<!--block:B0024-->
- WikiArt paintings with ArtEmis-style annotations. Each annotation is one viewer's *emotion* label plus a caption
  explaining it ("the dark colours make me feel sad").
- 308,723 rows over 61,402 paintings. A *row* is one image–caption pair (the painting and one caption). Every
  painting also has one *art style* (Impressionism, Baroque and so on).
- We evaluate 8 emotions (the catch-all "something else" is excluded) and the art styles with at least 30
  paintings.

<!--block:B0025-->
**Splits.** All splits are grouped by painting, so no painting crosses a split.

<!--block:B0026-->
| Split | Rows | Paintings | Role |
|---|---:|---:|---|
| *scorer-train* | 183,694 | 36,518 | training factor models |
| *selection* | 32,413 | 6,451 | development; read many times |
| val | 30,872 | | unused in recent work |
| *held* | 61,744 | 12,281 | final tests only; read 3 times so far |

<!--block:B0027-->
(Scorer-train and selection together form the 216,107-row train part.)

<!--block:B0028-->
**Features.** Frozen CLIP ViT-B/32 image and caption embeddings, cached once. CLIP is never fine-tuned.

<!--block:B0029-->
**Model (Candidate A).**
<!--block:B0030-->
- Two small encoders map a row's image feature and caption feature into one shared space of 32 non-negative
  *factors*. An item's vector there is its *code*.
- A *condition* is given by 4 *support* pairs and 4 *contrast* pairs.
- The parameter-free *naive rule* turns them into factor weights:
  `w = ReLU(mean support pair code − mean contrast pair code)`, L1-normalized. A pair code is the mean of a row's
  image code and caption code.
- A query q and a candidate c (opposite modalities) score `0.3·cos(CLIP_q, CLIP_c) + Σ_l w_l q_l c_l`.

<!--block:B0031-->
**Factor recipes.**
<!--block:B0032-->
- **R0:** the first recipe. It collapsed to about one effective dimension through a cosine agreement loss.
- **R3:** the repair, with an InfoNCE pair loss plus decorrelation.
- **C0:** R3's recipe refit on scorer-train rows; the matched control.
- **SE:** C0 plus *condition episodes*. Each episode's condition comes, with probability ½ each, from 64 k-means
  clusters of a text emotion classifier's outputs on the captions (GoEmotions RoBERTa,
  `SamLowe/roberta-base-go_emotions`, 28 emotion probabilities) or from 64 k-means clusters of CLIP image features.
  Because the classifier is supervised, SE is *distantly supervised*, not label-free.

<!--block:B0033-->
**Evaluation (label episodes).**
<!--block:B0034-->
- An *anchor* (the query) has a target label: one emotion or one style.
- The 4 supports carry that label; the 4 contrasts do not.
- There are 13 *candidates*: one positive with the label, and 12 from paintings never given it.
- Metric: *R@1*, the share of episodes where the positive ranks first (chance 7.69%). Both directions are scored:
  *i2t* (image query, captions ranked) and *t2i* (caption query, images ranked).
- Held results (pooled R@1): CLIP only 13.16, R3 19.45, C0 19.88, SE 21.24. SE's emotion gain over C0 (+2.08) held
  at three seeds. A trained condition scorer did not beat the naive rule (stage (d)).

<!--block:B0035-->
### 2.4 What changed on 2026-10-02

<!--block:B0036-->
Six investigations, all on selection rows (held rows untouched), changed the plan.

<!--block:B0037-->
| Report | Finding | Consequence |
|---|---|---|
| [CVPR literature review](../../reports/auto/v2/2026-10-21_cvpr_literature_review.md) | No prior example-conditioned cross-item image–text similarity (to our knowledge). The naive rule is Rocchio relevance feedback (1971), and the weighted factor score is a Conditional Similarity Network (CSN, CVPR 2017) mask. Learning conditions without labels is taken (SCE-Net ICCV 2019, DiscoverNet CVPR 2022), and so is distant affect supervision (EmotionCLIP CVPR 2023) | drop "unsupervised condition discovery"; claim "no labels from the evaluation taxonomy". Also, PercepT uses a different emotion model (ModernBERT) and CLIP ViT-L/14, so our teacher must be named exactly |
| [support-baseline spike](../../reports/auto/v2/2026-10-22_support_baseline_spike.md) | A logistic probe fit on the 4+4 examples in raw CLIP space beats SE (24.10 vs 21.22). A prototype of the supports that **ignores the query** reaches 22.83; adding the query adds only +0.57 | label episodes test few-shot recognition of a *value*, not conditional similarity. The problem has to change |
| [aspect-episode spike](../../reports/auto/v2/2026-10-23_aspect_episode_spike.md) | On *aspect episodes* (§3), no factor model beats CLIP (SE 11.32 vs 11.13). Probes trained with labels reach about 23 (the *ceiling*). Emotion lives in captions and style in images | the task is learnable, but the factors must be trained for it. ArtELingo caps cross-modal matching through its weaker modality |
| [aspect novelty check](../../reports/auto/v2/2026-10-24_aspect_task_novelty_check.md) | No paper defines the aspect task, but every ingredient exists. Closest: metric learning from pairs (Xing 2002, RCA 2005, KISSME 2012), Contextual Visual Similarity (arXiv 1612.02534), MARS (ICLR 2023), in-context text embedders | the novelty is the combination. Expected reviewer line: "a few-shot cross-modal diagonal KISSME" |
| [backbone check](../../reports/auto/v2/2026-10-25_backbone_check.md) | Across four frozen encoders, the weak-side probes barely move (emotion from images 35 to 37, style from captions 25 to 26). CUB's primary colour is read equally from images and captions | the asymmetry is in the data (claim C3). The user chose Qwen3-VL-Embedding-2B as the second backbone |
| [GeneCIS feasibility](../../reports/auto/v2/2026-10-20_genecis_feasibility.md) (Oct 1) | GeneCIS is usable for evaluation; its focus-attribute task names an aspect | GeneCIS focus attribute joins the benchmarks |

<!--block:B0038-->
**Value versus aspect, by example.**
<!--block:B0039-->
- **Value condition (old):** "sad, like these". The supports are 4 sad paintings, and the positive is the one sad
  candidate among 13. The supports alone find it; the query is nearly irrelevant.
- **Aspect condition (new):** "similar in emotion, the way these pairs are".
  - Supports: 4 pairs, each a fearful image matched with a fearful caption of *another* painting in another style,
    an awe image with an awe caption, and so on.
  - Contrasts: 4 pairs that share a style instead.
  - Query: a sad Baroque image.
  - The right answer is the caption that is sad but not Baroque; a caption that is Baroque but joyful is the
    distractor.
  - No support shows "sad". The model must work out *which respect* the examples share and then match the query on
    it. Swapping supports and contrasts makes the Baroque caption the right answer.

<!--block:B0040-->
## 3. Problem definition (approved)

<!--block:B0041-->
**Example-conditioned aspect similarity across modalities.**
<!--block:B0042-->
- **Input:** a query x (an image or a caption) and a condition c = (S, C).
  - S holds a few *support pairs*: an image of one item and a caption of another item (*cross-item*) that agree on
    the wanted *aspect*, with any *values*.
  - C holds a few *contrast pairs* that agree on a different aspect.
- **Output:** a ranking of candidates y in the other modality by s(x, y | c). Candidates that share the query's
  value on the aspect demonstrated by S rank first.
- **Rules:**
  - The condition never shows the query's value and never names the aspect.
  - Both directions are scored.
  - **Swap test:** the same query and candidates under the swapped condition (S and C exchanged) must re-rank.

<!--block:B0043-->
**Against GeneCIS.** GeneCIS focus conditions name the aspect in text and compare image with image. Ours shows the
aspect by examples and compares image with caption.
<!--block:B0044-->
- GeneCIS *focus attribute* (condition = an attribute type such as "colour") is the closest standard benchmark.
- *Focus object* conditions on a named object, which is a value, so it is supplementary only.

<!--block:B0045-->
## 4. Contributions and claims (approved)

<!--block:B0046-->
- **C1, task and protocol.**
  - Novelty statement: *"to our knowledge, the first cross-modal similarity in which the aspect is fixed at test
    time only by value-disjoint, cross-item image–caption examples with a contrast aspect, evaluated in both
    directions with a paired swap test."*
  - We cite the neighbours (metric learning from pairs, Contextual Visual Similarity, MARS, GeneCIS, CLAY, CRL,
    in-context embedders) and motivate value-disjoint examples by K1.
  - We do **not** claim that inferring a notion of similarity from examples, test-time reweighting, or
    relation-by-example retrieval is new.
- **C2, method.** A shared sparse image–text factor basis, trained on *pseudo-aspect episodes* with no labels from
  the evaluation taxonomy, is the prior that makes it possible to estimate a similarity from just four example pairs
  at test time.
  - The training-free *agreement rule* is presented openly as a Rocchio/KISSME-style estimator, not as the novelty.
  - Distant affect supervision is disclosed where a dataset uses it.
- **C3, analysis.** Aspects live asymmetrically across modalities: emotion in captions, style in images, colour in
  both. The weaker modality caps cross-modal matching, and the effect persists across four backbones, so it belongs
  to the data.

<!--block:B0047-->
| # | Claim | Evidence | Status |
|---|---|---|---|
| K1 | Value episodes are solved by the supports alone | prototype vs model, query ablation | **done** (support spike) |
| K2 | Our method beats backbone-only, raw-feature metric-from-pairs baselines and the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset | pre-registered final tests (§10) | open; factors are at CLIP level today |
| K3 | Examples beat naming the aspect on subjective aspects and match it on objective ones | CRL, Qwen with the aspect in its instruction, privileged names | open |
| K4 | Gains hold on a strong backbone (Qwen3-VL-Embedding-2B) | second backbone | open |
| K5 | GeneCIS focus attribute: competitive with the published frozen-B/32 rows (example and text protocols reported apart) | GeneCIS runs | open |
| K6 | The rule selects aspect factors, and the factors are shared across modalities | ablations, swap analysis, per-factor analysis | open |
| K7 | The gain comes from the *learned* basis | same rule on raw, PCA/NMF/SpLiCE and learned factors | open |

<!--block:B0048-->
**Fallback.** If K2 fails, K1, K3 and C3 could carry a task, benchmark and analysis paper but not the method paper.
The switch is discussed with the user at the Oct 9 decision (§11).

<!--block:B0049-->
## 5. Benchmarks and protocols (approved)

<!--block:B0050-->
### 5.1 Common protocol

<!--block:B0051-->
- **Aspect episodes:**
  - an anchor, and 13 candidates: one sharing aspect A with the anchor (p_A), one sharing aspect B (p_B), and 11
    sharing neither;
  - two sets of 4 cross-item example pairs, P_A and P_B. Each pair shares one aspect, with values distinct from
    each other and from the anchor's;
  - condition A uses S = P_A, C = P_B; condition B swaps them;
  - every item in an episode comes from a distinct painting or photo. With three aspects, all three aspect pairs
    are used.
- **Primary metric:** R@1, the mean over both directions and all conditions.
- **Secondary metrics:**
  - per condition and direction;
  - *swap success*: p_A above p_B under condition A *and* p_B above p_A under condition B, always reported next to
    R@1, because a scorer can flip with the condition without being right;
  - the other-aspect rate;
  - a 101-candidate gallery variant on ArtELingo and CUB.
- **Splits.** Factor training uses image–caption pairs and pseudo-partitions only, no labels. A labelled
  development split serves model selection. The test split is read within §10's budget.

<!--block:B0052-->
### 5.2 Datasets, in priority order

<!--block:B0053-->
| Dataset | Aspects | Captions | Test split | Notes |
|---|---|---|---|---|
| **ArtELingo** (primary) | emotion (8) × style (23) × genre (10; 81% of paintings labelled, §5.3) | human, affective | held rows, fresh-seed aspect episodes | asymmetric aspects; label-probe ceiling about 23 R@1 |
| **CUB-200-2011 + Reed et al. captions** (bird photos) | primary colour (15), bill shape (9), and a third attribute group chosen in E0 | human, 10 per image, written without species names | the standard zero-shot split's **50 unseen species**; development on 30 of the 150 training species, kept out of factor training | colour is symmetric across modalities; captions name colours |
| **GeneCIS focus attribute** | condition = attribute type | none (Visual Genome object crops) | the benchmark (2,000 templates) | image to image; see §5.4 |
| **SemArt** (paintings with catalogue descriptions) | type (10), school (26), timeframe (22) | catalogue text, artist names and dates scrubbed | official test (1,069 paintings); development = official val | main paper if on time, otherwise supplementary |
| GeneCIS focus object | condition = an object (value-type) | COCO | benchmark | supplementary only |
| Affection | | | | potential; the user will check |

<!--block:B0054-->
**Data on disk** (`/data/SSD/`; downloaded 2026-10-02 by `scripts/download_benchmarks.sh`):
<!--block:B0055-->
- `cub/` (11,788 images; captions in `captions/extracted/text_c10`);
- `semart/SemArt` (21,384 images);
- `visual_genome/VG_100K_all` (108,249 images, including all 23,640 GeneCIS attribute images);
- the GeneCIS COCO half, preprocessed earlier into `/data/PDD/genecis/`.

<!--block:B0056-->
Our loaders pass the Visual Genome path themselves; the GeneCIS clone is not edited.

<!--block:B0057-->
### 5.3 ArtELingo genre (resolved 2026-10-02: kept)

<!--block:B0058-->
**Result of the coverage check** (`src/test/20261026_genre_coverage/`). ArtGAN's labels cover 81% of paintings in
every split. The smallest genre still has 149 paintings in selection and 296 in held.

<!--block:B0059-->
| Split | Paintings | With genre |
|---|---:|---:|
| scorer-train | 36,518 | 29,593 (81.0%) |
| selection | 6,451 | 5,256 (81.5%) |
| held | 12,281 | 9,949 (81.0%) |

<!--block:B0060-->
- The rule below passes, so **genre is the third ArtELingo aspect**. Genre episodes draw only from the labelled
  paintings.
- The class-name file is gone upstream. The id-to-name mapping was recovered from the ArtELingo-28 genre names (9 of
  10 ids, purity 1.0). Id 5 (`nude_painting`) is inferred by elimination and alphabetical order.

<!--block:B0061-->
The original decision record follows.

<!--block:B0062-->
- **Why keep it.** The user wants genre (subject matter: portrait, landscape, still life and so on). It is the one
  ArtELingo aspect visible in both modalities, which C3 needs as its within-dataset control. Three aspects also
  answer the objection that two aspects make the condition a binary switch.
- **The gap.** The genre labels on disk come from ArtELingo-28 and cover only 1,160 of 61,402 paintings. WikiArt's
  10-class genre labels are published with ArtGAN (`genre_train.csv`, `genre_val.csv`). The user runs that download
  (an egress hook needs their approval) into `/data/SSD/wikiart_genre/`.
- **Rule:** keep genre if, after joining by file name, at least half of the selection and held paintings are covered
  and every genre has at least 30 paintings in each of those splits.
  - Otherwise, use a coarse subject label derived from WikiArt tags (portrait, landscape, religious, still life),
    disclosed as derived.
  - If that also fails, keep two aspects.

<!--block:B0063-->
### 5.4 GeneCIS protocols

<!--block:B0064-->
1. **Example protocol (ours).** Image to image on image codes, with supports and contrasts drawn from other
   templates that share the condition. It is not comparable with published numbers.
2. **Text protocol (stretch).** A phrase-to-weights adapter `w = a_T(phrase)`. It is needed for a row beside the
   published frozen ViT-B/32 results: SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1 (OSrCIR's independent
   reproduction gave 14.0).

<!--block:B0065-->
Always reported: image only, text only and image + text on our backbones. GeneCIS's CC3M training triplets are never
used.

<!--block:B0066-->
## 6. Method A: aspect-trained factors (approved)

<!--block:B0067-->
**Model.** The v2 structure is kept.
<!--block:B0068-->
- Two small encoders map frozen backbone features of each modality into a shared sparse non-negative space R₊^L
  (L = 32; 64 is a grid point).
- Score: s = β·cos + Σ_l w_l a_I,l(x) a_T,l(y).
- The *agreement rule* replaces the naive rule, because the examples are now pairs that agree on an aspect:
  w = ReLU(mean over S of a_I(x_i) ⊙ a_T(y_i) − mean over C of a_I(x_j) ⊙ a_T(y_j)), L1-normalized.
  It keeps the factors on which support pairs co-activate more than contrast pairs. It is training-free at test
  time.

<!--block:B0069-->
**Training: pseudo-aspect episodes.**
<!--block:B0070-->
- Each training row carries cluster ids from K *pseudo-partitions*, i.e. clusterings that stand in for unknown
  aspects.
- Episodes copy §5.1 with clusters in place of labels. For a partition pair (k, k′):
  - supports are 4 cross-item pairs sharing a k-cluster different from the anchor's;
  - contrasts are 4 pairs sharing a k′-cluster;
  - candidates are one sharing only the anchor's k-cluster, one sharing only its k′-cluster, and negatives.
- **Losses:**
  - a ranking loss through the differentiable agreement rule, in both directions;
  - a swap term on the same episode with roles exchanged;
  - R3's pair agreement (InfoNCE) and decorrelation terms, so that codes stay cross-modal and do not collapse.
- **Why this could work where SE did not.** SE was only trained on value episodes, so it learned to recognise a
  value from supports. These episodes reward exactly what the aspect spike found missing:
  - putting the anchor's value on an aspect into *both* modalities' codes;
  - keeping aspects in separable factors that the rule can select.

<!--block:B0071-->
| Dataset | Pseudo-partitions (no evaluation labels) |
|---|---|
| ArtELingo | GoEmotions affect k-means of captions (emotion-like; distant supervision); backbone image k-means (style-like; adjusted mutual information 0.32 with style); caption-content k-means (genre-like) |
| CUB (150 training species) | backbone image k-means; **per-sentence** caption k-means (Reed's captions focus on different parts, such as "red crown" or "short beak") |
| SemArt | image k-means; description k-means |
| GeneCIS | a factor model trained on COCO train2014 pairs (none are GeneCIS images), with image and caption k-means |

<!--block:B0072-->
**Go/no-go.** Pre-registered before the first run; ArtELingo, CLIP B/32, selection rows; decided Fri Oct 9.
<!--block:B0073-->
- **Grid:** at most about 10 runs (partition set × L × loss weights), about 10 minutes each locally.
- **Picking:** the best run is picked on seed-42 selection aspect episodes. The GO test uses **fresh seed-43
  episodes** from the same rows, so picking the best of about ten runs does not inflate the result.
- **GO** if, on those fresh episodes, pooled aspect R@1 has a 95% CI lower bound above 0 against **both**
  backbone-only (11.1) **and** the best raw-feature metric-from-pairs baseline (§8), with that baseline's fusion
  weight cross-fitted. Swap success is reported at matched R@1.
- **Strong GO:** pooled R@1 of about 15 or more, i.e. a third of the way to the ceiling of about 23.
- **NO-GO:** stop the method line and discuss the fallback framing with the user.
- **Replication:** a winner is re-run at seeds 43 and 44 before it moves to other datasets and Qwen.

<!--block:B0074-->
**Open risk R-pseudo** (flagged by the user, accepted for now): the pseudo-partitions are proxies, and the model may
learn the episode format rather than aspects. The cross-dataset and unseen-species results are the test.

<!--block:B0075-->
## 7. Backbones (approved)

<!--block:B0076-->
- **Development** on CLIP ViT-B/32 (cached; runs take minutes).
- **Final tables** also on **Qwen3-VL-Embedding-2B**, the user's choice. It is a multimodal embedder built on a
  vision–language model that takes a task instruction at encoding time. That lets the "aspect named in the
  instruction" baseline run on the very same backbone.
- **Fidelity check first.** Our reimplementation (`src/test/20261025_backbone_check/extract.py`: transformers
  5.6.2, last-token pooling, default instruction, capped image resolution) must be checked against the official
  implementation before paper use. If it fails, fall back to PE-Core L/14.
- **Packages** go in `pip --target` directories, never into the CoSiR env.

<!--block:B0077-->
## 8. Baselines (approved)

<!--block:B0078-->
All baselines run on the same features and episodes.
<!--block:B0079-->
- **On development splits,** fusion weights (and β) are cross-fitted: chosen on one half of the anchors (split by
  parity) and applied to the other, then swapped.
- **For final reads,** they are chosen on the development split and frozen. GeneCIS, which has no development
  split, takes them from the other datasets.

<!--block:B0080-->
| Tier | Baseline | What it does | Reviewer question |
|---|---|---|---|
| 1 | backbone only | cosine of query and candidate | is conditioning needed at all? |
| 1 | metric from pairs on raw features | estimate a similarity from the 4+4 example pairs directly on raw features: the diagonal agreement rule (signed and ReLU), low-rank KISSME with shrinkage (inverse covariance of similar pairs minus that of dissimilar pairs), RCA (whitening by within-pair variation), a Xing-style metric fit per episode, and Wang et al.'s per-query weights | "your rule is few-shot KISSME" |
| 1 | the agreement rule on unsupervised bases | PCA-32/64, NMF-32 and SpLiCE sparse concept codes instead of our factors | K7: is it the learned basis? |
| 1 | value prototype or Rocchio | mean of support items minus mean of contrasts | do value baselines fail on aspects? |
| 1 | C0, SE, R3 | our earlier factor recipes | what do aspect episodes add? |
| 2 | names, privileged | project items onto the true value names ("a painting that evokes fear", …) and compare | upper reference for naming |
| 2 | CRL (NeurIPS 2025) | an LLM lists an aspect's values from one word ("emotion"); project and compare | K3, training-free names |
| 2 | Qwen3-VL-Embedding with the aspect in its instruction | "represent this painting by the emotion it evokes" | would an instruction embedder make the method unnecessary? |
| 3 | label-supervised probe ceiling | per-modality classifiers trained with the labels (converged, fixed thread count) | distance to supervision |
| 3 | GeneCIS rows | published frozen-B/32 numbers; image / text / image + text | K5 |
| stretch | CLAY (CVPR 2026) reimplementation; an in-context MLLM reranker given the example pairs | | strongest naming and in-context comparisons |

<!--block:B0081-->
**Out of the tables.**
<!--block:B0082-->
- **Teacher-only:** GoEmotions cannot read images, so it cannot score cross-modal pairs. It becomes a text-side
  analysis inside C3.
- **PercepT** was a side branch that showed the buddy graph works on an existing benchmark and inspired v2's design.
  A buddy-vs-PercepT comparison is **optional and storyline-dependent** (supplement or analysis), never required.

<!--block:B0083-->
## 9. Out of scope

<!--block:B0084-->
- Fine-tuning the backbone.
- Evaluation labels in factor training.
- GeneCIS CC3M triplets.
- Merging percept-branch code.
- Codex (unless the user asks).
- Any held read outside §10.

<!--block:B0085-->
## 10. Held budget and statistics (approved)

<!--block:B0086-->
**Ledger.** Every read of a final split is logged in `docs/superpowers/held_ledger.md`: date, script SHA-256,
episode SHA-256 and purpose. Each final script refuses a second run.

<!--block:B0087-->
| Dataset | Development | Final test | Budget |
|---|---|---|---|
| ArtELingo | selection rows (free to reuse) | held rows, fresh-seed aspect episodes (a new task on rows read 3 times before for value episodes) | 1 main + 1 reserve |
| CUB | 30 of the 150 training species | 50 unseen species | 1 + 1 |
| SemArt | official val | official test | 1 + 1 |
| GeneCIS | none (zero-shot; selection on the other datasets) | benchmark | 1 + 1 |

<!--block:B0088-->
- **Main read:** one pre-registered run per dataset, covering all models and baselines on both backbones, with
  checkpoints, fusion weights, β and episode counts frozen.
- **Reserve read:** only for a pre-registered fix after a final-review finding, never for a second attempt at a
  better number.
- **Disclosure:** ArtELingo's held rows shaped earlier design decisions; the paper says so.

<!--block:B0089-->
**Statistics.**
<!--block:B0090-->
- **Primary metric:** aspect R@1, the mean over directions and aspect pairs.
- **Uncertainty:** paired bootstrap over anchors (5,000 resamples).
- **Seeds:** our models train with 3 seeds. The headline is the 3-seed mean, with its CI from bootstrapping anchors
  on per-anchor seed means; every seed is also reported.
- **Pre-declared primary comparisons** ("beats" means CI lower bound > 0):
  - ours vs backbone only;
  - ours vs the best raw metric-from-pairs baseline;
  - ours vs Qwen with the aspect in its instruction ("beats" or "matches", per aspect type, declared in the
    pre-registration).
  - All other comparisons are descriptive.
- **Episode counts** come from a power calculation on selection variance (default 4,096 anchors per aspect pair).
- **Swap success** is always shown next to R@1.

<!--block:B0091-->
## 11. Experiment plan and schedule (approved)

<!--block:B0092-->
| # | Experiment | Dates | Output or decision |
|---|---|---|---|
| E0 | **Setup:** a generic aspect-episode module (`src/eval/`) for all datasets, with tests; held ledger; genre coverage (§5.3); CUB third aspect group (the highest min(image, caption) probe gain over the majority rate, among `has_shape`, `has_wing_pattern`, `has_breast_pattern`, `has_wing_color`); Qwen fidelity check | Oct 2 to 4 | module; decisions recorded |
| E1 | Tier-1 baselines on ArtELingo selection aspect episodes (B/32) | Oct 3 to 6 | baseline table; the metric-from-pairs bar for the go/no-go |
| E2 | ArtELingo pseudo-partitions (affect and image k-means exist; caption-content k-means is new) | Oct 4 to 5 | partitions and their agreement with the labels (diagnostic only) |
| E3 | Method A training and go/no-go (at most about 10 runs) | Oct 5 to 9 | **GO / NO-GO, Fri Oct 9** |
| E4 | DAS6 extraction: Qwen on all ArtELingo rows; B/32 and Qwen on SemArt, GeneCIS crops and COCO, COCO train2014 | Oct 5 to 12 | caches under `/data/SSD2/pre_extract/` |
| E5 | Replication seeds 43 and 44 on ArtELingo | Oct 10 to 11 | seed table |
| E6 | CUB: partitions, training, development evaluation | Oct 10 to 16 | **CUB check, Oct 16:** if A does not beat the baselines on CUB development, claims narrow to ArtELingo plus analysis |
| E7 | GeneCIS focus attribute: example protocol, COCO-trained factors, baselines; text protocol as stretch | Oct 13 to 20 | GeneCIS table |
| E8 | SemArt: scrubbing, partitions, training, development | Oct 15 to 22 | main or supplementary |
| E9 | Qwen backbone runs on every dataset | Oct 12 to 22 | K4 |
| E10 | Tier-2 baselines: CRL, Qwen instruction names, privileged names | Oct 12 to 20 | K3 |
| E11 | Ablations: rule on bases (K7); no episodes vs value episodes; factor-group and swap analysis (K6) | Oct 14 to 23 | ablation tables |
| E12 | C3 analysis: converged ceilings across datasets and backbones; GoEmotions text-side analysis | Oct 16 to 30 | analysis figures |
| E13 | Pre-registration and power for the final reads | Oct 21 to 23 | **methods frozen Oct 23** |
| E14 | Final reads, one main read per dataset | Oct 24 to Nov 1 | **main-paper experiments frozen Nov 1** |
| E15 | Writing: draft from Oct 26, full draft Nov 6; title and abstract fixed Nov 7; registration Nov 10 | Oct 26 to Nov 16 | the paper |
| E16 | Whole-branch final review (most capable model, re-deriving every load-bearing number) and fix wave | Nov 11 to 14 | review report |
| E17 | Supplementary: slipped datasets, GeneCIS focus object, 101-candidate galleries, stretch baselines, optional buddy vs PercepT | Nov 16 to 23 | supplementary |

<!--block:B0093-->
Every experiment ends with a report in `docs/reports/auto/v2/`, one row in `reports_sum.md`, a commit, and an update
of §4's claims table.

<!--block:B0094-->
**Planning scope.**
<!--block:B0095-->
- The first implementation plan covers **E0 to E5** in task-level detail.
- E6 onward gets its own plan after a GO on Oct 9, informed by what E3 finds.
- After a NO-GO, the next plan is the fallback framing, agreed with the user.

<!--block:B0096-->
## 12. Risks

<!--block:B0097-->
| ID | Risk | Mitigation or signal |
|---|---|---|
| **R-pseudo** | pseudo-aspect partitions are proxies; the model may learn the episode format | cross-dataset and unseen-species results; disclose distant supervision |
| R-nogo | A fails on Oct 9 | fallback framing (K1, K3, C3), discussed with the user before switching |
| R-kissme | a raw-feature metric from pairs matches A | K7 fails and the basis is not the contribution; known by Oct 9 |
| R-names | Qwen with the aspect in its instruction beats examples | narrow K3 to where names fail (subjective aspects) |
| R-ceiling | ArtELingo's ceiling is low (about 23) | relative gains; CUB colour as the symmetric case; C3 turns the cap into a finding |
| R-time | 4 datasets × 2 backbones in about 5 weeks | staged priority; SemArt moves to the supplementary first |
| R-qwen | our reimplementation differs from the official one | E0 fidelity check; PE-Core fallback |
| R-infra | local GPU loss (happened 2026-10-02); a shared machine | DAS6 as backup; GPU lock; commit after every task |
| R-held | ArtELingo held rows shaped earlier design | fresh-seed episodes on a new task; disclosure |
| R-licence | CUB terms conflict (non-commercial vs cc-by); SemArt and GeneCIS are CC BY-NC | academic use; state the terms in the paper |
| R-concurrent | TPIPS, SteerViT, CLAY, Fioresi et al., COCO-Facet condition similarity on text | position on the example interface and the cross-modal score |

<!--block:B0098-->
## 13. Conventions

<!--block:B0099-->
- Implementation goes to Claude Code subagents sized to the task; the controller reviews each result.
- Reports follow the `docs/reports/` layout and the user's report rules: a real baseline beside every number,
  figures, paper-draft style, no dashes.
- Other sessions share main: stage files by explicit path.
- Experiment folders are `src/test/<sequence-date>_<name>/`. Edits to existing source get a change log in
  `.claude/yyyymmdd_log.md`.

<!--block:B0100-->
---

<!--block:B0101-->
## Appendix A. Glossary

<!--block:B0102-->
| Term | Meaning here |
|---|---|
| agreement rule | the training-free rule of §6 that weights factors by how much support pairs co-activate them, minus contrast pairs |
| anchor (query) | the item a ranking is made for: an image (i2t) or a caption (t2i) |
| aspect / value | an aspect is a respect in which items can be similar (emotion, style, colour); a value is one setting of it (sad, Baroque, red) |
| backbone | the frozen encoder whose features everything builds on (CLIP ViT-B/32; Qwen3-VL-Embedding-2B) |
| β | weight of the backbone cosine in the score |
| candidate | one of the 13 items ranked in an episode, in the modality opposite the anchor |
| ceiling (label-probe) | R@1 when factor codes are replaced by classifiers trained with the true labels: a diagnostic upper reference, never a model |
| code / factor | an item's 32 non-negative numbers in the shared space / one of those dimensions |
| condition | what the user cares about, given here by support and contrast pairs |
| condition episode / pseudo-aspect episode | a training episode built from clusters instead of labels; value-type for SE, aspect-type for method A |
| contrast pair | an example pair that agrees on a different aspect than the one wanted |
| cross-fitting | choosing a tuning weight on one half of the anchors and applying it to the other half, so tuning does not flatter the result |
| cross-item pair | an image of one item with a caption of another item |
| distant supervision | training signal from an external supervised model (here a GoEmotions classifier), not from our labels |
| episode | one ranking problem: anchor, condition and candidates |
| held / selection / scorer-train rows | the final-test, development and training parts of ArtELingo (§2.3) |
| i2t / t2i | image query ranking captions / caption query ranking images |
| KISSME, RCA, Xing | classic ways to learn a similarity metric from similar and dissimilar pairs (2012, 2005, 2002) |
| naive rule | the earlier rule for value conditions: mean support pair code minus mean contrast pair code |
| pseudo-partition | a clustering of training rows that stands in for an unknown aspect |
| R@1 | share of episodes where the right candidate ranks first; chance is 1/13 = 7.69% |
| R0, R3, C0, SE | earlier factor recipes (§2.3) |
| Rocchio, CSN | relevance feedback (1971) and Conditional Similarity Networks (2017), the classic ancestors of our rule and score |
| support pair | an example pair that agrees on the wanted aspect |
| swap test / swap success | exchanging supports and contrasts must flip which aspect candidate wins |

<!--block:B0103-->
## Appendix B. Sources

<!--block:B0104-->
- **Handoffs:** `docs/superpowers/handoffs/2026-10-02-cosir-v2-cvpr-publication-handoff.md`,
  `docs/superpowers/handoffs/2026-10-02-cvpr-plan-brainstorm-progress.md`.
- **v2 foundation spec:** `docs/superpowers/specs/2026-09-28-cosir-v2-ground-up-redesign.md`.
- **Reports:** everything in `docs/reports/auto/v2/` (the chain from 2026-10-04 to 2026-10-25), indexed in
  `docs/reports/reports_sum.md`.
- **Code used so far:**
  - `src/model/factors.py`, `src/model/conditioning.py`;
  - `src/eval/label_episodes.py`, `src/eval/condition_eval.py`;
  - `src/test/20261018_affect_factor_learning/run_affect.py` (SE and C0);
  - `src/test/20261023_aspect_episode_spike/` (aspect episodes, ceilings);
  - `src/test/20261025_backbone_check/` (feature extraction for four backbones).


<!--block:B0105-->
---

<!--block:B0106-->
# Appendix (evidence report): 2026-10-21_cvpr_literature_review

<!--block:B0107-->
## CVPR literature review for CoSiR v2: prior art, baselines and benchmarks

<!--block:B0108-->
**Report date:** 2026-10-21 (sequence date in this folder; the research was done on 2026-10-02).
**Scope:** read-only literature review for the CVPR submission (abstract 2026-11-10, paper 2026-11-16, supplementary 2026-11-23). We trained nothing and ran no evaluation. We fetched every cited paper's arXiv, venue or repository page and checked title and authors. Numbers are copied from each paper's own tables, with the table number (written T2 for Table 2). Anything we could not open is marked **UNVERIFIED**; where we draw a conclusion a paper does not state, we write "we infer". Three subagents swept threads in parallel (GeneCIS numbers, benchmarks, affect and instruction-conditioned embeddings); we spot-checked their key rows (CIReVL T3, OSrCIR T2, CLAY T2(a), CRL T3, GeneCIS T6, FocalLens T1, COCO-Facet T1, TEVI T2, the PercepT encoder) against the arXiv HTML and found them correct. The session's web search budget (200 searches) ran out late in the sweep; after that we searched arXiv titles and abstracts through the arXiv API and fetched known pages directly.
**Builds on:** the [Stage 1 to GeneCIS synthesis](2026-09-28_stage1_genecis_synthesis_brainstorm.md) (an earlier, shorter survey) and the [GeneCIS feasibility check](2026-10-20_genecis_feasibility.md) (GeneCIS counts, licence and on-disk data). We extend those and do not repeat them.

<!--block:B0109-->
**CoSiR v2 in the terms used below.** CoSiR v2 scores an image I and a caption T under a condition c, s(I,T|c), in both retrieval directions. The condition is given **by examples**: 4 support image–caption pairs that have the wanted aspect and 4 contrast pairs that do not. Two small encoders map frozen CLIP ViT-B/32 image and text features into a shared, sparse, non-negative 32-d factor space; the parameter-free naive rule sets w = ReLU(mean support code − mean contrast code), L1-normalised; the score is β·cos(CLIP_I, CLIP_T) + Σ_l w_l a_I,l(I) a_T,l(T). The factors are trained without ArtELingo labels, from condition episodes built on k-means of GoEmotions RoBERTa probabilities of the captions and k-means of CLIP image features (distant supervision). Held ArtELingo label episodes (anchor, 4 supports, 4 contrasts, 13 candidates with 1 positive, R@1): CLIP only 13.2, SE 21.2 pooled (emotion 17.0, style 25.5). After this review started, the user widened the backbone: any frozen encoder is eligible, including encoder pairs that do not share an embedding space (§2.8).

<!--block:B0110-->
### 1. Verdict

<!--block:B0111-->
The closest prior art falls into three lines that no paper we found combines: text-conditioned similarity (GeneCIS, and on frozen vision–language encoders CLAY, CVPR 2026, and CRL, NeurIPS 2025), image-only similarity whose condition is inferred from examples or learned without condition labels (SCE-Net, DiscoverNet, few-shot attribute learning, classic relevance feedback), and cross-modal retrieval of a personal instance from a few example images (PALAVRA, POLAR). (a) We found no paper that scores an image against a caption of another item under a condition given by support and contrast image–caption pairs, so the novelty risk for example-conditioned cross-modal similarity is moderate; but the naive rule is a Rocchio relevance-feedback update applied to factor codes and the weighted factor score is a Conditional Similarity Network mask, so the novelty must rest on the combination and on the shared image–text factors, not on either mechanism. (b) The risk for condition learning without condition labels is high: SCE-Net (ICCV 2019) and DiscoverNet (CVPR 2022) already learn similarity conditions without condition labels, GeneCIS and MagicLens mine conditions from captions, and EmotionCLIP (CVPR 2023) already used a frozen text sentiment classifier as distant supervision for vision–text contrastive learning, so the claim has to shrink to "no labels from the evaluation taxonomy". A second risk sits in our own protocol: in the ArtELingo label episodes the positive is the only candidate that carries the support label, so an anchor-free 4-shot prototype on raw CLIP may solve them, and a reviewer will ask for that baseline first. The must-have baselines are a support prototype and a Rocchio query on raw features (with and without the anchor), a Tip-Adapter style cache and a linear probe on the eight support and contrast pairs, a text-named condition on the same frozen encoder (label prompt, CRL projection), the naive rule on unsupervised codes and on the GoEmotions teacher, PercepT, and on GeneCIS the published frozen ViT-B/32 rows (SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1). Because any frozen encoder is now eligible, an instruction-following embedder with the condition written into its instruction (Qwen3-VL-Embedding, GME or VLM2Vec-V2, all with open weights) is both a must-have text-named baseline and a candidate backbone; a pair of unimodal encoders would remove the β·cos term and need a LiT, SAIL or ASIF style alignment. The most promising extra benchmark is CUB-200-2011 with Reed et al.'s human captions (species crossed with attribute groups, captions written without species names), with GeneCIS kept as the reviewer-facing standard and SemArt as a second art benchmark.

<!--block:B0112-->
### 2. Literature by thread

<!--block:B0113-->
#### 2.1 Conditional similarity learning

<!--block:B0114-->
Conditional similarity began as image-only metric learning with a known condition id (CSN), moved to conditions inferred without labels (SCE-Net, DiscoverNet), and since 2023 has moved to open text conditions on large pretrained encoders (GeneCIS, then the training-free methods of §2.6). GeneCIS numbers of later methods are in Table 2.

<!--block:B0115-->
**Table 1. Conditional similarity methods.**

<!--block:B0116-->
| Paper | Condition form | Modalities | Backbone, frozen? | Benchmarks, metric | Headline numbers (source) | Code | Closeness to CoSiR v2 |
|---|---|---|---|---|---|---|---|
| CSN, Veit, Belongie, Karaletsos, CVPR 2017 | attribute id selects a learned mask over embedding dimensions | I→I triplets | CNN trained end to end | UT-Zappos50k (4 notions), fonts; triplet error | 10.73% error, learned masks (quoted in SCE-Net T5) | yes | our factor term is a CSN-style masked similarity whose mask comes from examples instead of an id |
| SCE-Net, Tan, Vasileva, Saenko, Plummer, ICCV 2019 | none at test time; a weight branch mixes K learned masks computed from the inputs (on Zappos, the concatenated triplet images) | I→I; outfit items | ResNet-18 trained | Polyvore Outfits, Maryland Polyvore (compatibility AUC, FITB), UT-Zappos50k (triplet error) | Zappos error 7.53% with 4 masks vs CSN 10.73% (T5); Polyvore Outfits AUC 0.91, FITB 61.6 vs 0.86, 55.3 for the type-aware network (T1) | yes | the closest image-only analogue of inferring the condition from examples; training triplets are still sampled per attribute |
| DiscoverNet, Ye, Shi, Zhan, CVPR 2022 | weakly supervised: triplets without condition labels; a set module matches triplets to embeddings | I→I | trained CNN | UT-Zappos-50K, Celeb-A | not verified (CVF PDF returned 403) | not found | label-free condition discovery from triplets |
| Generalized CSL, Shi, Li, Gan, Zhan, Ye, TPAMI 2025 | supervised, weakly supervised or semi-supervised conditions | I→I | trained | not verified | not verified | not checked | the latest paper of the label-free line |
| Few-shot attribute learning, Ren et al., arXiv 2012.05895 (technical report; v1 titled "Flexible Few-Shot Learning with Contextual Similarity") | a support set of positives and negatives that share one or two attributes | image classification episodes | trained, self-supervised vs supervised pretraining | Celeb-A, Zappos-50K, ImageNet-with-attributes | claims supervised pretraining generalises poorly to unseen attributes; no numbers copied | not checked | the same episode design as our label episodes, image only |
| ASEN, Ma et al., AAAI 2020; ASEN++, arXiv 2104.02429 | attribute id drives spatial and channel attention | I→I | trained CNN | FashionAI, DARN, DeepFashion, Zappos50k; MAP | DeepFashion mean MAP 8.74 and 9.64 (quoted in CRL T4) | yes | supervised attribute-specific similarity |
| Conditional Image-Text Embedding Networks, Plummer et al., ECCV 2018 | the phrase is softly assigned to K conditional embeddings | phrase↔region grounding | trained | Flickr30K Entities, ReferIt, Visual Genome | +3 to 4% grounding (abstract only) | not checked | an early cross-modal model with several conditional subspaces; the condition is inferred from the phrase itself |
| GeneCIS, Vaze, Carion, Misra, CVPR 2023 | free text phrase | I+T→I | CLIP RN50x4 fine-tuned on 1.6M triplets mined from CC3M | GeneCIS, R@1/2/3 | average R@1 16.8 (T2), 15.1 with the backbone frozen (T5), 17.6 with ViT-B/16 (T6) | yes; RN50x4 and ViT-B/16 weights, CC3M only | ancestor; conditions mined from captions |
| InDiReCT, Kobs, Steininger, Hotho, WACV 2023 | a few text prompts define the similarity notion | I→I | frozen CLIP; a projection trained on text embeddings | LanZ-DML (5 datasets, 13 notions), MAP@R | not copied | yes | the earliest frozen-CLIP, text-defined notion of similarity |

<!--block:B0117-->
**Learning conditions without condition labels.** Three groups of papers already claim it, and each narrows our claim. SCE-Net and DiscoverNet learn condition subspaces from triplets without condition labels and infer the condition from the compared items at test time; their supervision is the triplet itself, which in UT-Zappos50k is generated per attribute. GeneCIS and MagicLens mine conditions automatically from caption structure (scene graphs; LLM-written instructions): no labelled taxonomy, yet still supervision. Multi-clustering methods that find several partitions of the same images under a user criterion (IC|TC, ICLR 2024; Multi-MaP, CVPR 2024) resemble our use of two k-means pseudo-partitions to make condition episodes, although they cluster and do not retrieve. Distant supervision from an external affect classifier also has a precedent: EmotionCLIP (Zhang, Pan, Wang, CVPR 2023) passes captions through a frozen DistilBERT sentiment model (7-d pseudo sentiment scores) and reweights the negatives of its contrastive loss by the KL divergence between those scores (§3.4, Eq. 5 and 6). We infer that a reviewer will accept "trained without labels from the evaluation taxonomy, with distant affect supervision from GoEmotions" and will reject "unsupervised condition discovery".

<!--block:B0118-->
#### 2.2 Composed image retrieval and zero-shot CIR

<!--block:B0119-->
Composed image retrieval (CIR) ranks target images for a reference image plus a modification text; zero-shot CIR (ZS-CIR) trains without CIR triplets. The standard benchmarks are FashionIQ, CIRR, CIRCO and GeneCIS. The GeneCIS subagent found 26 papers from 2023 to 2026 that report GeneCIS numbers in some form. Table 2 lists the rows that matter for us, as **focus attribute / change attribute / focus object / change object / average R@1**. All conditions are text phrases; none is example-conditioned.

<!--block:B0120-->
**Table 2. Methods that report GeneCIS numbers (selected).**

<!--block:B0121-->
| Method, venue | Condition and modality | Backbone, frozen? | GeneCIS R@1 (source table) | Code | Closeness |
|---|---|---|---|---|---|
| GeneCIS Combiner, CVPR 2023 | text; I+T→I | RN50x4 fine-tuned | 19.0 / 16.6 / 14.7 / 16.8 / 16.8 (T2); 15.1 frozen (T5) | yes | ancestor |
| CIReVL, Karthik et al., ICLR 2024 | text via BLIP-2 caption and LLM rewrite, then CLIP text→image | frozen CLIP, training-free | **ViT-B/32 17.9 / 14.8 / 14.6 / 16.1 / 15.9**; SEARLE re-run at B/32 18.9 / 13.0 / 12.2 / 13.6 / 14.4 (T3) | yes | training-free on frozen CLIP; needs an LLM |
| LinCIR, Gu et al., CVPR 2024 | text; projection trained on text only | frozen CLIP L, H, G | ViT-L 16.90 / 16.19 / 8.27 / 7.40 / 12.19; re-runs Pic2Word (L avg 11.16) and SEARLE (L avg 12.26) (T B.3); no B/32 row | yes | light projection on frozen CLIP |
| CompoDiff, Gu et al., TMLR 2024 | text (optional negative text); diffusion over CLIP embeddings | frozen CLIP RN50 to ViT-G | average only: RN50 14.65 to ViT-G 15.48 (T3) | yes | far |
| MagicLens, Zhang et al., ICML 2024 | open text instruction | CLIP or CoCa, fine-tuned | CLIP-B (B/16) 15.5 / 12.3 / 14.4 / 17.7 / 15.0 (T16) | yes | far |
| OSrCIR, Tang et al., CVPR 2025 | text; GPT-4o reasons over image and text, then CLIP retrieval | frozen CLIP, training-free | **ViT-B/32 19.4 / 16.4 / 15.7 / 18.2 / 17.4** (T2); Paracosm v1 (ECCV 2026) reproduced 14.0 with GPT-4o | yes | training-free; reproduction disputed |
| PrediCIR, Tang et al., CVPR 2025 | text; trained mapper | frozen CLIP L/14 | 18.2 / 18.7 / 12.7 / 16.9 / 16.6 (T9) | yes | medium |
| RTD, Byun et al., ICCV 2025 | text; post-hoc text-encoder tuning | image encoder frozen | B/32 average: Pic2Word 11.13, SEARLE 12.19, LinCIR 12.23 before RTD (T10) | announced | B/32 runs of three ZS-CIR methods |
| MegaPairs (MMRet), Zhou et al., ACL 2025 | text plus image | CLIP-B, CLIP-L unfrozen; MLLM | MMRet-Base 18.3 / 15.2 / 16.6 / 21.7 / 18.0 (T9) | yes | far |
| LamRA, Liu et al., CVPR 2025 | instruction, image and text | Qwen2-VL-7B with LoRA | one R@1: 18.9, 24.8 with reranking; its own runs of CLIP-L 13.3, E5-V 18.5, UniIR-CLIP 16.8, MagicLens-L 16.3, EVA-CLIP-8B 13.1 (T4) | yes | far |
| STiTch, Li et al., arXiv 2605.21261 | text; transition vector and optimal transport | frozen CLIP | B/32 21.1 / 17.9 / 16.4 / 18.3 / 18.4 (T8) | not found | medium |
| SQUARE, Wu et al., arXiv 2509.26330 | text; GPT-4o captions and MLLM reranking | frozen CLIP | B/32 25.6 / 19.0 / 17.3 / 16.8 / 19.7; 16.4 without the reranker (T3) | not found | far |
| DIOR, Kawarada et al., arXiv 2512.21860 | condition word in an LVLM prompt; I→I | frozen LVLM (Llama-3.2-Vision 11B is the main model) | focus tasks only: focus attribute 24.0, focus object 21.1 (T C-1) | yes | training-free conditional similarity |
| CRL, Liu et al., NeurIPS 2025 | criterion text → LLM-written basis | frozen CLIP ViT-B/16 | object half, training-free: focus object 15.4, change object 17.0 vs CLIP image + text 11.5 / 9.8 and the CC3M Combiner 16.6 / 18.0 (T3); we infer the object half from the CLIP rows, which equal GeneCIS T6 | yes | training-free, frozen CLIP |
| SteerViT, Ruthardt et al., arXiv 2604.02327 | text cross-attention inside a frozen ViT | DINOv2 ViT-B/14 frozen plus about 21M trained parameters | focus object only: 25.4 vs DINOv2 9.6 and an unnamed "specialized" baseline 18.7 (T6) | project page | conditional representation, text condition |
| FocalLens, Hsieh et al., arXiv 2504.08368 | text instruction | CLIP ViT-L/14-336 with the vision tower trained | focus attribute and focus object only, at **R@3**: 43.30 / 43.72 (T1); not comparable with R@1 | not found | close in problem |

<!--block:B0122-->
**Frozen CLIP ViT-B/32 rows on GeneCIS** (average R@1): Pic2Word 11.13, SEARLE 12.19 and LinCIR 12.23 (RTD T10 runs); SEARLE 14.4 (CIReVL T3 run); CIReVL 15.9; Paracosm 16.1; DeCIR 16.5 (LoRA on CLIP); PACT 17.23 (LoRA on the text encoder, validation split); OSrCIR 17.4 (disputed); STiTch 18.4; SQUARE 19.7 with an MLLM reranker. Paracosm v1 gives trivial B/32 baselines of 11.5 (image only), 8.4 (text only) and 12.7 (image + text). Pic2Word, SEARLE, Context-I2W, KEDs, ISA, LDRE, SEIZE, Slerp, Denoise-I2W, TransAgg, Visual Delta Generator, CoVR, E5-V, VLM2Vec, UniIR, GENIUS, MM-Embed and CoLLM do not report GeneCIS in their own papers; their GeneCIS rows, where they exist, are re-runs by others.

<!--block:B0123-->
Three pitfalls matter for a GeneCIS table. Papers use five formats (the four tasks; average only; an undefined single R@1; overlapping pairwise averages, as in FoCo, DiffComp and PACT; focus-only rows, one at R@3). SEARLE at ViT-L/14 circulates as 14.4 (CIReVL's run) and 12.26 (LinCIR's run). Two printed averages are arithmetically wrong (Context-I2W L/14 in OSrCIR T2; PrediCIR G/14 in T9). The GeneCIS README calls the release "v0" and reports a 0.2 point seed standard deviation in average R@1; the repository was archived in August 2024 and no "GeneCIS v1" exists.

<!--block:B0124-->
**Successor benchmarks.** CLAY-EVAL (synthetic FLUX images with object and human attributes, mAP), CORE (SteerViT; SUN397 scenes with inpainted objects), COCO-Facet (Li, Gao, Du, NeurIPS 2025; 9,112 attribute-focused text→image queries, 1 positive among 100), ZeroSight (video-sourced ZS-CIR that criticises overlap with CLIP pretraining) and MCMR (CVPR 2026; multi-condition product retrieval). None uses example conditions or scores cross-item image↔text pairs.

<!--block:B0125-->
#### 2.3 Example-conditioned and few-shot retrieval

<!--block:B0126-->
This thread answers the key novelty question.

<!--block:B0127-->
**Table 3. Retrieval or similarity conditioned on examples.**

<!--block:B0128-->
| Paper | What the examples define | Modalities | Backbone, frozen? | Benchmark | Code | Closeness to CoSiR v2 |
|---|---|---|---|---|---|---|
| Rocchio, "Relevance feedback in information retrieval", in The SMART Retrieval System, 1971 | the query moves towards the mean relevant and away from the mean non-relevant document | text documents | vector space | n/a | n/a | our naive rule w = ReLU(μ⁺ − μ⁻) is a Rocchio update applied to factor codes |
| Rui, Huang, Ortega, Mehrotra, IEEE TCSVT 1998 | feature weights updated from images the user marks relevant | I→I (content-based retrieval) | hand-crafted features | not checked | n/a | per-feature reweighting from examples, as in our rule |
| MindReader, Ishikawa, Subramanya, Faloutsos, VLDB 1998 | several examples, optionally scored, define the hidden distance function | database records | n/a | synthetic and real data | n/a | infers which attributes matter from examples |
| VISALOGY, Sadeghi, Zitnick, Farhadi, NIPS 2015 | an image pair A:B defines a transformation used to retrieve D for C | I→I | quadruple Siamese CNN, trained | VAQA | not checked | a pair as the condition, image only |
| Few-shot attribute learning, Ren et al., arXiv 2012.05895 | positive and negative supports define an attribute or a conjunction of two | image classification | trained | Celeb-A, Zappos-50K, ImageNet-with-attributes | not checked | the closest episode design; unimodal classification |
| PALAVRA (PerVL), Cohen et al., ECCV 2022 | a few images of a personal concept learn a new word embedding | T→I retrieval with the new word; segmentation | frozen CLIP | two new PerVL benchmarks | yes | cross-modal and example-conditioned, but the examples define an instance, not an aspect shared by different items |
| POLAR, Ryan et al., CVPR 2025 | a few images of a personal concept; low-rank update of the text encoder's last layer | T→I | dual encoder, partly tuned | DeepFashion2, ConCon-Chi | yes (CC BY-NC-SA 4.0) | as PALAVRA |
| Relevance feedback for CLIP, Nara et al., ECCV Workshops 2024 | binary feedback on retrieved images | I→I, T→I | frozen CLIP, training-free | category-based retrieval | not checked | Rocchio-style feedback on CLIP |
| CLIP-Branches, Lülf et al., SIGIR 2024 (arXiv 2406.13322) | positive and negative images marked after a text query train a classifier | T→I | frozen CLIP features | interactive search | not checked | example-refined text-to-image search |
| FSIR, Idan et al., arXiv 2603.25891 | a text query plus exemplar positives and hard negatives | T→I | encoder-agnostic | FSIR-BD: 38,353 images, 303 queries | not stated | the nearest recent cross-modal setting; examples refine a text query, they do not define an aspect for I↔T matching |
| Tip-Adapter, Zhang et al., ECCV 2022 | few-shot labelled images form a key–value cache | image classification | frozen CLIP | 11 datasets | yes | support-set baseline we can run |
| LP++, Huang et al., CVPR 2024; CLAP, Silva-Rodríguez et al., CVPR 2024 | few-shot labelled images fit a linear probe | image classification | frozen CLIP | 11 datasets | yes | linear-probe baselines we can run |
| ProtoNets (NeurIPS 2017), TADAM (NeurIPS 2018), FEAT (CVPR 2020) | the support set defines prototypes or adapts the metric | image classification | trained | miniImageNet and others | yes | support-conditioned metric, unimodal |
| Rankability of visual embeddings, Sonthalia, Uselis, Oh, arXiv 2507.03683 | two extreme examples recover an ordinal axis (age, aesthetics and others) | image embeddings | 7 frozen encoders | 9 datasets | yes | evidence that raw-feature prototypes from very few examples are already strong |
| Affection CLIP listener, Achlioptas et al., CVPR 2023 | none (affective explanation → target image among distractors) | T→I | CLIP | Affection; accuracy reported in a figure (Fig. 7), not copied | yes | affective caption→image discrimination, unconditioned |

<!--block:B0129-->
**Is there prior work on cross-modal image↔text similarity conditioned by example pairs?** We found none. Every cross-modal method that takes examples uses them either to name an instance (PALAVRA, POLAR) or to refine a text query towards a set of images (CLIP-Branches, FSIR, relevance feedback). None scores a caption against an image of a different item under an aspect defined by support and contrast image–caption pairs, and none evaluates both retrieval directions under the same condition. The only cross-modal conditional retrieval with a learned non-text condition we found is user-conditioned hashtag retrieval (Veit, Nickel, Belongie, van der Maaten, arXiv 1711.09825), where the condition is a per-user embedding. This is a negative search result from an incomplete search (the search budget ran out; arXiv API queries such as "support set" with "image-text retrieval" returned nothing), so the paper should say "to our knowledge". "Few-shot cross-modal retrieval" already names a different setting (novel categories with few training pairs, for example GCRDP, arXiv 2505.13306); the paper should define its own term to avoid the confusion.

<!--block:B0130-->
#### 2.4 Affective and subjective cross-modal retrieval

<!--block:B0131-->
The affect literature on our data is about classification, captioning and topic discovery; we found no conditional or even plain image↔text retrieval benchmark on ArtEmis or ArtELingo.

<!--block:B0132-->
**Table 4. Affective, subjective and art-style work.**

<!--block:B0133-->
| Paper | Task, condition | Backbone, frozen? | Benchmarks, metric | Headline numbers (source) | Code | Closeness |
|---|---|---|---|---|---|---|
| PercepT, Mohamed, Church, Elhoseiny, arXiv 2606.03345 | no condition; discovers perception topics from image–caption pairs, then maps images to topics | frozen CLIP **ViT-L/14**; affect encoder **ModernBERT-base fine-tuned on GoEmotions** (fused with CLIP) | ArtELingo, ArtELingo-28, Affection; clustering (SI, AMI) and multi-label AUC, F1 | SI 0.97 vs BERTopic 0.37, AMI 0.18 vs 0.07 (T1); AUC 0.94 vs 0.77 (T2); no retrieval | "will be made public" | same data and the same kind of affect signal; topic discovery, not conditional scoring |
| ArtEmis, Achlioptas et al., CVPR 2021 | emotion prediction and affective captioning | ResNet, LSTM, BERT | 439,121 explanations, 81K WikiArt paintings, 9 classes | text→emotion BERT 65.7% (§6); best speaker emotional alignment 0.522 (T4) | yes | our data; no retrieval |
| ArtEmis 2.0, Mohamed et al., CVPR 2022 | captioning with contrastive data | SAT speaker | 260,533 new explanations collected on visually similar paintings with opposite valence | combined METEOR 0.144, CIDEr 0.111 vs 0.135, 0.091 (T3) | yes | a ready source of similar-image, opposite-emotion contrast pairs |
| ArtELingo, EMNLP 2022; ArtELingo-28, EMNLP 2024 | multilingual affective captions | captioning models | about 0.8M added annotations; 28 languages | captioning metrics | yes | our data |
| Affection, Achlioptas et al., CVPR 2023 | affective explanations of real photos; CLIP listener | CLIP | 85,007 images, 526,749 explanations | listener results in a figure (§2.3); pragmatic speaker CLIPScore 69.2 (T3) | yes | the photo counterpart of ArtEmis |
| EmoSet, Yang et al., ICCV 2023 | emotion classification with 6 attributes | CNNs | 118,102 human-labelled images | best top-1 78.40% (T3) | yes | alternative affect benchmark, no captions |
| EmotionCLIP, Zhang, Pan, Wang, CVPR 2023 | sentiment-guided contrastive pretraining | trained | BoLD, Emotic and others; linear probe mAP | BoLD 22.51 vs X-CLIP 13.26 (T4) | yes | distant supervision from a text sentiment model (§2.1) |
| GOYA, Wu, Nakashima, Garcia, ICMR 2023 | content and style subspaces, I→I | **frozen CLIP ViT-B/32** plus two projections trained on Stable Diffusion images | WikiArt (10 genres, 27 styles): distance correlation, classification | style 50.90 vs pre-trained CLIP 51.23 (T2) | yes | the closest style-side work: same backbone, one subspace per aspect |
| CSD, Somepalli et al., arXiv 2404.01292 | style descriptor, I→I | CLIP ViT-L fine-tuned | WikiArt with artist as style, mAP@1 | 64.56 vs CLIP ViT-L 59.4, ViT-B/16 52.2 (T1) | yes | strong style baseline |
| DIOR, arXiv 2512.21860 | condition word in an LVLM prompt | frozen LVLM | WikiArt style mAP@1 | CLIP ViT-L 59.3, CSD 58.2, DIOR 45.6 (T3) | yes | a conditional embedding that loses to plain CLIP on art style |
| SemArt, Garcia, Vogiatzis, arXiv 1810.09617 | text↔painting retrieval, no condition | ResNet-50 and bag of words | 1,069 test paintings; R@K, median rank | T→I R@1 0.144, I→T R@1 0.138 (T4) | yes | art I↔T retrieval, unconditioned |
| HPIR, Zhang et al., arXiv 2406.09397 | aesthetic preference in T→I retrieval | CLIP fine-tuned with preference RL | 150 queries, human group choice | CLIP 62.1% → 71.7% (T1) | not found | the only subjective-retrieval benchmark we found; no condition |

<!--block:B0134-->
**Correction for the project.** PercepT's paper uses ModernBERT-base fine-tuned on GoEmotions and CLIP ViT-L/14, while our handoff calls SamLowe/roberta-base-go_emotions "PercepT's affect teacher" and our factors use ViT-B/32. The teacher shares PercepT's label set but not its model; the paper should describe the teacher by name and not as "PercepT's teacher", and any PercepT comparison should state which encoder our port uses.

<!--block:B0135-->
#### 2.5 Sparse or interpretable shared concept spaces on CLIP

<!--block:B0136-->
**Table 5. Sparse and concept codes on CLIP-like encoders.**

<!--block:B0137-->
| Paper | Code type | Shared image and text? | Backbone, frozen? | Used for retrieval? | Condition-weighted retrieval? | Closeness |
|---|---|---|---|---|---|---|
| SpLiCE, Bhalla et al., NeurIPS 2024 | sparse non-negative combination of concept-word embeddings (LAION vocabulary, top 10k recommended) | yes, both modalities decompose over one vocabulary | frozen CLIP ViT-B/32, ViT-B/16, RN50, OpenCLIP ViT-B-32 (repository, Apache-2.0) | spurious-correlation and editing uses | not shown | a training-free shared sparse code for a dimension-matched baseline |
| Discover-then-Name, Rao, Mahajan, Böhle, Schiele, ECCV 2024 | SAE on CLIP image features, concepts named with CLIP text | named through text | frozen CLIP | classification | no | concept bottleneck, image side |
| Matryoshka SAE, Zaigrajew, Baniecki, Biecek, ICML 2025 | hierarchical SAE | image side | frozen CLIP ViT-L/14 | §5.2: boosting one concept in SAE space and mapping back changes ImageNet nearest neighbours | qualitative only | the only image retrieval steered by sparse concepts we found |
| Papadimitriou, Su, Fel, Gil, Kakade, COLM 2025 | SAEs on CLIP, SigLIP, SigLIP2, AIMv2 | finds mostly single-modality concepts that "bridge" across modalities | frozen | analysis | no | warns that shared dictionaries split by modality |
| MGSAE, Kaushik, Barch, Fanelli, arXiv 2601.20028 | group-sparse SAE with cross-modal masking | yes, built against split dictionaries | frozen CLIP and CLAP | cross-modal control | no | a shared-dictionary method to cite and possibly compare |
| LUCID-SAE, Gu et al., arXiv 2602.07311 | shared patch and token dictionary, optimal-transport matching | yes, plus private capacity | frozen | grounding, interpretation | no | token-level analogue |
| SCoCCA, Gordon, Levi, Gilboa, arXiv 2603.13884 | sparse concept decomposition via CCA | yes | frozen CLIP | concept ablation | no | sparse CCA alternative |
| SPARC, Nasiri-Sarvi, Rivaz, Hosseini, TMLR 2026 | concept-aligned SAE, global TopK across models and modalities | yes | frozen | cross-modal concept retrieval | no | shared sparse space |
| TEVI, Mahajan, Rao, Xie, Koller, Schiele, EMNLP 2026 | caption → MLP → sigmoid mask over TopK-SAE latents of the CLIP image embedding | mask conditioned on text | frozen CLIP vision tower; text tower fine-tuned | image↔text retrieval on COCO, Flickr, DOCCI, IIW | yes, but the condition is the caption being matched: DOCCI B/16 I→T R@1 20.38 → 24.52, T→I 7.16 → 8.30 (T2) | yes | the closest mechanism to our weighted sparse factors |
| STAIR (EMNLP 2023), LexLIP (ICCV 2023), VDR (ICLR 2024) | sparse lexical image and text codes | yes, trained jointly | trained | standard I↔T retrieval | not shown | shared sparse I↔T spaces without conditions |
| Kang, Wang, Xiong, arXiv 2411.00786 | SAE with a retrieval contrastive loss on a dense text retriever | text only | frozen retriever | yes | yes: editing latents prioritises documents "from specific perspectives" | the closest precedent for user-steered retrieval through sparse latents, in text IR |

<!--block:B0138-->
**Has anyone used sparse codes for condition-weighted retrieval?** Partly. TEVI masks sparse latents of frozen CLIP for image↔text retrieval, but its mask comes from the caption being matched, not from a user condition; Matryoshka SAE shows one qualitative concept-boosted search; Kang et al. steer text retrieval. Per-dimension weights that select a notion of similarity are CSN's masks. We infer that a shared sparse image–text factor space whose weights come from example pairs, evaluated on condition-defined cross-modal episodes, is not in the literature, while each of its parts is.

<!--block:B0139-->
#### 2.6 Text-conditioned and instruction-conditioned embeddings

<!--block:B0140-->
**Table 6. Conditional embeddings with a text or instruction condition.** Universal multimodal embedders are in §2.8.

<!--block:B0141-->
| Paper | Condition | Modalities | Backbone, frozen? | Benchmarks, metric | Headline (source) | Code | Closeness |
|---|---|---|---|---|---|---|---|
| CLAY, Lim, Lee, Park, Oh, CVPR 2026 | text keyword → LLM descriptions → SVD subspace and log map; training-free | I→I | **frozen CLIP ViT-B/32**, SigLIP-B (L variants in the supplement) | Stanford40 action, location and **mood** (mood labels from IC\|TC), 7 fine-grained sets, CLEVR4, CLAY-EVAL; mAP | Stanford40 action / location / mood: CLIP-B 43.0 / 47.0 / 53.0, GeneCIS Combiner 50.0 / 50.9 / 51.8, CLAY 66.0 / 55.4 / 57.9 (T2(a)) | project page only | the closest frozen ViT-B/32 conditional similarity; text instead of examples, image only; includes a subjective condition |
| CRL, Liu, Sun, Hu, Li, Peng, NeurIPS 2025 | criterion → LLM-written descriptive texts → basis T; R = I·Tᵀ; training-free | I→I | frozen CLIP ViT-B/32 or B/16 | Clevr4-10k, Cards, GeneCIS object half, DeepFashion (MAP) | DeepFashion mean MAP 7.93 training-free vs CLIP 6.08 (T4); GeneCIS in Table 2 | yes | trivially runnable on our features |
| SP-CRL, Wang, Lyu, Li, Jia, arXiv 2602.05464 | CRL bases purified by truncation and null-space projection | I→I | frozen VLM | clustering, few-shot, retrieval | not copied | not found | CRL successor |
| Fioresi, Caba Heilbron, Nathani, Shah, Kafle, ECCV 2026 (arXiv 2607.22919) | attribute text → hypernetwork → affine map of frozen embeddings | I→I | frozen CLIP ViT-L/14; hypernetwork trained with attribute labels (19 attributes) | Clevr-4, Stanford40 (with mood), ShotBench; mAP and clustering | mAP base 33.8 / 43.7 / 25.2 vs 77.0 / 86.3 / 41.0 (T1, as read by the subagent) | project page | supervised light transform on frozen features |
| FocalLens, Hsieh et al., arXiv 2504.08368 (ICLR 2025 workshop per mlanthology) | free text instruction | I→I | CLIP ViT-L/14-336 trained with condition tokens; MLLM variant | CelebA-Attribute (29 attributes, scaled mAP), GeneCIS (R@3), SugarCrepe, MMVP-VLM | CelebA average: CLIP 13.59, MagicLens 13.42, InstructBLIP 16.19, FocalLens-CLIP 21.32, FocalLens-MLLM 22.67 (T1) | not found | reports both GeneCIS and a multi-attribute face benchmark |
| DIOR, arXiv 2512.21860 | condition word in an LVLM prompt | I→I | frozen Llama-3.2-Vision 11B | LanZ-DML, WikiArt and DomainNet style, GeneCIS focus | LanZ-DML average CLIP 31.0 vs 44.5 (T1); WikiArt in Table 4 | yes | training-free LVLM conditioning |
| SteerViT, arXiv 2604.02327 | text injected by gated cross-attention | I→I | frozen ViT plus about 21M parameters trained on referring data | CORE, GeneCIS focus object, MOSAIC, PODS | Table 2 | project page | text-steered encoder |
| TPIPS, Wang, Nitzan, Hertzmann, Zhu, Shechtman, Efros, Zhang, arXiv 2607.18237 | free-form text aspect | I→I perceptual similarity | Qwen3-VL-8B-Embedding fine-tuned | new odd-one-out data: 24,342 triplets, 257,391 triplet–aspect pairs, about 1M votes | main results are in a figure; not copied | yes | concurrent human-judged aspect-conditioned similarity |
| COCO-Facet promptable embeddings, Li, Gao, Du, NeurIPS 2025 | a GPT-4o question about the attribute used as the gallery-side prompt | T→I | MLLM embedders, not fine-tuned | COCO-Facet: 9,112 queries, 8 attribute types; R@1, R@5 | R@5: CLIP ViT-L 47.0, SigLIP2 52.6, VLM2Vec 58.9, 75.5 with prompts (T1) | yes | attribute-conditioned cross-modal retrieval with a text condition |
| FLAIR, Xiao et al., arXiv 2412.03561 | the caption embedding is the pooling query over patch tokens | I↔T | ViT-B/16 trained from scratch | COCO, Flickr, fine-grained retrieval | COCO T→I R@1 53.3 vs SigLIP 47.2 (T1) | yes | text-dependent image embedding, with the caption as the condition |
| Alpha-CLIP, Sun et al., CVPR 2024 | region mask (alpha channel) | I↔T | image encoder fine-tuned | ImageNet-S, referring tasks | L/14 zero-shot 73.48 → 77.41 (T2) | yes | spatial focus only |

<!--block:B0142-->
#### 2.7 Candidate benchmarks

<!--block:B0143-->
The benchmark subagent verified 37 datasets; the ranked shortlist with all requested fields is in §4. Two facts shape the choice. Many multi-attribute face and fashion sets generate their captions from the labels (MM-CelebA-HQ, CelebA-Dialog, and, we infer from released examples, DeepFashion-MultiModal), so the text side leaks the label and cannot test cross-modal conditional matching. The datasets with human captions and independent labels are few: CUB-200-2011 with Reed et al. captions, SemArt, Affection, CelebAText-HQ and our ArtEmis/ArtELingo. Prior conditional-similarity papers used mostly caption-free sets (UT-Zappos50k, Polyvore, DeepFashion, FashionAI, DARN, CelebA attributes, Stanford40, CLEVR4).

<!--block:B0144-->
#### 2.8 Backbones and universal or instruction-following multimodal embedders

<!--block:B0145-->
Universal multimodal embedders turn an MLLM into a retriever; most take a task instruction at encode time, so a condition can be written into the instruction ("Represent this painting with respect to the emotion it evokes"). That makes them both candidate frozen backbones and text-named-condition baselines. Their standard benchmark is MMEB (36 datasets: classification, VQA, retrieval, grounding; precision@1) or MMEB-V2 (78 datasets, adding video and visual documents); neither is a conditional-similarity benchmark, so the rows closest to our problem are GeneCIS (Table 2), COCO-Facet (Table 6) and CelebA-Attribute (FocalLens). Memory figures below are our estimate of bf16 weight memory (2 bytes per parameter); inference needs more.

<!--block:B0146-->
**Table 7. Frozen backbone candidates.**

<!--block:B0147-->
| Model (paper) | Type, size | Embedding dim | Weights memory (bf16, our estimate) | Licence, weights | Instruction or text condition at encode time | MMEB or related (source) | GeneCIS (source) |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (current) | dual encoder, about 151M | 512 | under 1 GB | MIT, open | no | CLIP 37.8, variant not stated (VLM2Vec T2) | B/32 image + text 12.7 average (Paracosm v1 T3) |
| OpenCLIP ViT-L/14, ViT-H/14 (Cherti et al., CVPR 2023) | dual encoder, LAION-2B | 768, 1024 | about 1 to 2 GB | open (licence not re-checked) | no | OpenCLIP 39.7 (VLM2Vec T2) | LinCIR uses H as a backbone (Table 2) |
| EVA-CLIP, EVA-CLIP-18B (Sun et al. 2023, 2024) | dual encoder up to 18B | varies | up to about 36 GB | open (licence not re-checked) | no | EVA-CLIP 8B 43.7 (UniME T1) | EVA-CLIP-8B 13.1, 18B 13.6 single R@1 (LamRA T4) |
| SigLIP 2 (Tschannen et al., arXiv 2502.14786) | sigmoid dual encoder: B, L, So400m (about 1B total per the card), g | not stated on the card | about 2 GB for So400m | Apache-2.0, open | no | SigLIP (v1) 34.8 (VLM2Vec T2); SigLIP2 R@5 52.6 on COCO-Facet (Li et al. T1) | none found |
| Perception Encoder, PE-Core (Bolya et al., arXiv 2504.13181) | CLIP-style, B/16, L/14, G/14 | not checked | up to a few GB | licence not verified | no | ImageNet zero-shot 83.5 for L14-336 (repository) | none found |
| jina-clip-v2 (Koukounas et al., arXiv 2412.08802) | dual encoder: 561M text (XLM-RoBERTa) plus 304M EVA02-L14 vision | 1024, Matryoshka down to 64 | about 2 GB | CC BY-NC 4.0, open | task prefix only (for example retrieval.query), no free instruction | not found | none found |
| BLIP-2 (Li et al., ICML 2023) | Q-Former on a frozen ViT-g; LLM for generation | 256 (projected ITC features) | about 2 to 8 GB for the feature extractor (our estimate) | BSD-3-Clause (LAVIS, archived September 2026), open | no | 25.2 (VLM2Vec T2) | none found |
| UniIR CLIP_SF, BLIP_FF (Wei et al., ECCV 2024) | fine-tuned CLIP-L or BLIP fusion | 768 for CLIP_SF (we infer from CLIP-L) | 5.1 GB and 7.5 GB checkpoints | MIT, open | instruction prefix | 44.7 (VLM2Vec T2); M-BEIR 48.9 CLIP_SF (UniIR T2) | 16.8 single R@1 (LamRA T4) |
| E5-V (Jiang et al., arXiv 2407.12580) | LLaVA-NeXT-8B trained on text pairs only | 4096 | about 16 GB | not stated on the card | fixed "in one word" prompt; a condition can be written into the prompt string (we infer; not evaluated by its authors) | 13.3 (VLM2Vec T2) but 37.5 (UniME T1); CIRR R@1 33.90 (E5-V T2) | 18.5 single R@1 (LamRA T4) |
| VLM2Vec (Jiang et al., ICLR 2025) | Phi-3.5-V 4.2B or LLaVA-1.6 with LoRA | backbone hidden size | about 8 to 16 GB | open (licence not re-checked) | yes: "Instruct: {task} Query: {input}" | 62.9 best variant (T2) | none in own paper |
| VLM2Vec-V2 (Meng et al., arXiv 2507.04590) | Qwen2-VL-2B with LoRA | backbone hidden size | about 4 GB | model card returned 401 | yes | MMEB-V2 58.0 overall, 64.9 image (T2); GME-7B 57.8, LamRA 40.4 in the same table | none |
| GME (Zhang et al., CVPR 2025) | Qwen2-VL 2B (2.21B) or 7B | 1536 or 3584 | about 4.5 or 16 GB | Apache-2.0, open | yes, through a `prompt` or `instruction` argument | UMRB 64.45 (2B), 67.44 (7B) (T3); CIRR 51.79 for 7B (T7) | none |
| MM-Embed (Lin et al., ICLR 2025) | LLaVA-1.6-Mistral-7B plus NV-Embed-v1, 8B | not stated on the card | about 16 GB | CC BY-NC 4.0, open | yes, required for queries | M-BEIR 52.7 vs CLIP-SF 48.3 (T1) | none in own paper |
| LamRA (Liu et al., CVPR 2025) | Qwen2-VL-7B (Qwen2.5-VL branch) | backbone hidden size | about 16 GB | MIT, open | instruction-tuned; condition through the query text | MMEB-V2 40.4 (VLM2Vec-V2 T2) | 18.9, 24.8 with reranking (T4) |
| mmE5 (Chen et al., arXiv 2502.08468) | Llama-3.2-11B-Vision | backbone hidden size | about 22 GB | not stated | yes | 69.8 supervised, 58.6 zero-shot (T2) | none |
| UniME (Gu et al., ACM MM 2025) | Phi-3.5-V 4.2B, LLaVA-1.6 7B, LLaVA-OneVision 7B | backbone hidden size | about 8 to 16 GB | open (licence not re-checked) | yes, VLM2Vec task prompts | 66.6 (LLaVA-1.6), 70.7 (OneVision) (T1) | none |
| LLaVE (Lan et al., EMNLP 2025 Findings) | 0.5B, 2B, 7B | backbone hidden size | about 1 to 16 GB | not checked | yes | 59.1, 65.2, 70.3 (T2) | none |
| B3 (Thirukovalluru et al., arXiv 2505.11293) | Qwen2-VL 2B or 7B, InternVL3 | backbone hidden size | about 4 to 16 GB | not checked | yes | 68.1 (2B), 72.0 (7B) (T1) | none |
| MetaEmbed (Xiao et al., ICLR 2026) | multi-vector late interaction on Qwen2.5-VL and others | multi-vector | up to 32B models | not checked | not central | 76.6 (7B), 78.7 (32B) (T1) | none |
| Qwen3-VL-Embedding (Li et al., arXiv 2601.04720) | 2B or 8B | 64 to 2048 (2B, Matryoshka), 4096 (8B) | about 4 or 16 GB | Apache-2.0, open | yes, a custom instruction replaces the default "Represent the user's input." | MMEB-V2 73.2 (2B), 77.8 (8B) (model card table) | none |
| jina-embeddings-v4 (Günther et al., arXiv 2506.18902) | Qwen2.5-VL-3B, about 4B | 2048 dense (Matryoshka to 128) or 128 multi-vector | about 8 GB | Qwen Research License | three task adapters (retrieval, text matching, code); no free instruction | not copied | none |

<!--block:B0148-->
Three observations follow. First, the best open, permissively licensed instruction-following embedders that fit one GPU are Qwen3-VL-Embedding-2B and GME-Qwen2-VL-2B (Apache-2.0, about 2B parameters); MM-Embed and jina-clip-v2 are non-commercial, and jina-embeddings-v4 uses the Qwen Research License. Second, MMEB numbers for the same model disagree across papers (E5-V 13.3 in VLM2Vec T2 vs 37.5 in UniME T1), so we should cite each number with its source and re-run what we compare. Third, no universal embedder reports GeneCIS in its own paper; LamRA's single-number GeneCIS evaluation (T4) is the only source, and it puts E5-V (18.5) and LamRA (18.9) above the frozen ViT-B/32 ZS-CIR methods but below training-free MLLM pipelines such as SQUARE (19.7 with reranking).

<!--block:B0149-->
**When the image and text encoders do not share a space.** With a pair of unimodal encoders (for example an e5 text encoder and a DINOv2 image encoder), cos(f_I(I), f_T(T)) is meaningless, so our β·cos term is unavailable and the factor term must carry all cross-modal matching. Prior work offers three ways to restore a shared term. Trained alignment on locked towers: LiT (Zhai et al., CVPR 2022) locks a pretrained image tower and trains the text tower; Three Towers (Kossen et al., NeurIPS 2023) adds a frozen pretrained tower as a teacher; Maniparambil et al. (CVPR 2025, arXiv 2409.19425) train only MLP projectors on frozen DINOv2 and All-RoBERTa-Large and reach 76% ImageNet accuracy with 20 times less data and 65 times less compute than training from scratch; SAIL (Zhang, Yang, Agrawal, CVPR 2025) trains an alignment layer on frozen unimodal encoders with 6% of CLIP's paired data and reports 73.4% ImageNet zero-shot against CLIP's 72.7%. Training-free alignment: ASIF (Norelli et al., arXiv 2210.01738) represents each input by its similarities to a set of anchor image–text pairs, so every dimension is "the similarity of the input to a unique image-text pair", building on relative representations (Moschella et al., ICLR 2023). Evidence that it can work: Maniparambil et al. (CVPR 2024, arXiv 2401.05224) find that unimodal vision and language encoders have similar similarity structure, Merullo et al. (ICLR 2023) map image features into a language model's input space with one linear layer, the Platonic representation hypothesis (Huh et al., arXiv 2405.07987) argues for convergence, and Schnaus, Araslanov, Cremers (CVPR 2025) match vision and language without parallel data. For CoSiR this means the factor encoders already map each modality into one shared 32-d space and survive the switch, while the β term needs a LiT or SAIL style projector or an ASIF anchor representation. ASIF is worth noting: its anchors are image–text pairs, so our support and contrast pairs could serve as anchors, which makes it a natural example-conditioned baseline for the non-shared case (we infer; untested). Any non-shared-backbone result must be read against a shared CLIP control, since alignment quality and conditioning would otherwise be confounded.

<!--block:B0150-->
### 3. Baselines we can run

<!--block:B0151-->
All baselines run on the same frozen features as CoSiR (CLIP ViT-B/32 caches exist for ArtELingo; if we change backbone, every baseline moves with it) and on the same episodes, within the held-row budget of the handoff. We rank by how directly each answers a question a CVPR reviewer will ask, then by cost. "Needs" lists what goes beyond our own episode inputs (anchor, 4 supports, 4 contrasts, 13 candidates).

<!--block:B0152-->
**Table 8. Ranked baselines.**

<!--block:B0153-->
| Rank | Baseline | What it computes | Needs | Reviewer question it answers |
|---|---|---|---|---|
| 1 | **Support prototype and Rocchio query on raw features** | anchor-free: score each candidate by cos(f(c), μ⁺ − μ⁻), μ the mean feature of the support or contrast items in the candidate's modality (or the pair mean); anchor-plus: q = f(anchor) + α(μ⁺ − μ⁻), scored cross-modally | nothing new; α tuned on selection rows | "Is this just 4-shot classification of the candidates?" In our label episodes the positive is the only candidate with the support label, so the anchor-free prototype can in principle solve the task; it also tests learned factors against textbook relevance feedback |
| 2 | **Tip-Adapter style cache (training-free)** | candidate affinity Σ exp(−γ(1 − cos)) to the supports minus the same to the contrasts, plus β·cos(anchor, candidate) | nothing new; γ, β on selection rows | "Did you compare with the standard training-free few-shot CLIP adapter?" |
| 3 | **Linear probe on the eight pairs** | logistic regression on 4 support vs 4 contrast pairs (image and text features), candidate score = probe logit + β·cos(anchor, candidate) | scikit-learn | "Does a fitted classifier on raw features beat a parameter-free rule on learned factors?" |
| 4 | **Text-named condition on the same frozen encoder** | (a) label prompt: cos(anchor, c) + λ·cos(c, text("a painting that evokes sadness")); (b) CRL projection: an LLM lists the values of the criterion ("emotion", "art style"), anchor and candidates are projected onto that text basis and compared; (c) GeneCIS-style image + text sum | label names at test time; one LLM call per criterion for (b); CRL code is public | "Why examples instead of naming the condition?" Examples must win where the aspect is hard to name, or the paper needs another argument |
| 5 | **Instruction-following embedder with the condition in the instruction** | Qwen3-VL-Embedding-2B or GME-Qwen2-VL-2B (Apache-2.0), or VLM2Vec-V2, encoding anchor and candidates under an instruction such as "Represent this painting by the emotion it evokes"; optionally E5-V with an edited prompt or a DIOR-style LVLM prompt | about 4 to 5 GB of weights, a few GPU hours to encode ArtELingo (our estimate) | "Would a 2026 instruction-following embedder make the method unnecessary?" |
| 6 | **Naive rule on unsupervised codes** | the same w = ReLU(μ⁺ − μ⁻) on PCA-32, ICA-32 or NMF-32 of the training features, on SpLiCE codes (supports ViT-B/32, Apache-2.0), and on the positive part of centred raw features | SpLiCE vocabulary | "Is the gain from the learned factors or from the rule plus any sparse basis?" |
| 7 | **Teacher-only** | naive rule on the 28-d GoEmotions probabilities of the captions | the GoEmotions model already used | "Is SE more than its distant teacher?" (planned in the handoff) |
| 8 | **PercepT topic codes** | naive rule on PercepT Stage 1 topic memberships | the project's PercepT port as a clean v2 module (state its encoder, §2.4) | "How does the closest image–caption topic method do on the same episodes?" |
| 9 | **Published text-conditioned models** | SEARLE ViT-B/32 (torch.hub weights, CC BY-NC 4.0) with the label name as the modification text; CLAY reimplemented from its formula; GeneCIS Combiner (RN50x4 or ViT-B/16 CC3M weights, fine-tuned backbone) | released weights; label names | "How do established CIR and conditional-similarity models do when told the condition?" |
| 10 | **Supervised upper reference** | a CSN-style mask or linear head trained with ArtELingo labels on training rows | labels on training rows | "How far is training without evaluation labels from training with them?" (the headroom probe's label-aligned code at 49.8% R@1 is a first answer) |
| 11 | **Style subspace** (optional) | GOYA's style projection on frozen ViT-B/32 for style episodes | GOYA code | "Does a fixed style subspace match the learned factors on style?" |

<!--block:B0154-->
On GeneCIS itself, a frozen ViT-B/32 model should be compared with image only, text only and image + text (re-run on our cache), with SEARLE 14.4, CIReVL 15.9 and OSrCIR 17.4 average R@1 (published, with OSrCIR's reproduction at 14.0 noted), and with the GeneCIS Combiner at 15.1 with a frozen RN50x4 (T5) as the closest frozen trained reference. Our model enters through an image→image mode and either a text adapter or few-shot supports drawn from other templates, reported separately, as the feasibility check recommends.

<!--block:B0155-->
### 4. Benchmarks

<!--block:B0156-->
Ranked for example-conditioned image↔text episodes with at least two independent aspects, small compute and the deadline. Facts come from the benchmark subagent's fetched pages; "(we infer)" marks inference.

<!--block:B0157-->
**Table 9. Ranked candidate benchmarks.**

<!--block:B0158-->
| Rank | Benchmark | Size | Aspects | Caption source | Licence | Download | Prior conditional-similarity use | Main risk |
|---|---|---|---|---|---|---|---|---|
| 1 | **CUB-200-2011 + Reed et al. captions** (CVPR 2016) | 11,788 images, 200 species, 312 binary attributes; 10 captions per image (about 118k, we infer) | species; attribute groups (colour, bill shape, size and others) | human (AMT); annotators were told not to name the species | Caltech page: non-commercial research and education; the CaltechDATA record says "cc-by" (conflict unresolved); caption licence not stated | live (1.2 GB); caption archive on Google Drive | attributes used in few-shot attribute work (PAN, ICCV 2021); CUB captions used for cross-modal retrieval by PCME (CVPR 2021) | noisy per-image attributes; colour correlates with species; captions name colours, so colour conditions are easy on the text side |
| 2 | **GeneCIS** (object half now; attribute half after a Visual Genome 1.2 download) | 4 tasks, 1,960 to 2,112 templates each, galleries of 10 or 15 | text keyword (object or attribute) | none native; COCO captions exist for the object half | CC BY-NC 4.0 | live; object half preprocessed in our repository | the standard benchmark (Table 2) | image→image with text conditions; measures transfer, not the primary claim |
| 3 | **SemArt** (arXiv 1810.09617) | 21,384 paintings | type (10), school (26), timeframe (22), author, technique | human catalogue comments (Web Gallery of Art) | CC BY-NC 4.0 | live (3 GB, DOI) | Text2Art retrieval (R@K) | comments often name artist and date; school and timeframe correlate |
| 4 | **ArtEmis/ArtELingo plus WikiArt genre** | current data | emotion, style, and genre as a third aspect | human | ArtEmis terms of use | on disk; genre through the ArtGAN WikiArt class lists (we infer) | our current benchmark | same domain; enables same-anchor swaps across three aspects, and ArtEmis 2.0 adds similar-image, opposite-emotion pairs for hard contrasts |
| 5 | **Affection** (CVPR 2023) | 85,007 images, 526,749 explanations | emotion (ArtEmis taxonomy); objects from its COCO, VG and Flickr30k sources (we infer) | human explanations | Affection terms of use | request form | none found | access latency; 71.3% positive vs 21.1% negative |
| 6 | CelebAText-HQ + CelebA attributes | 15,010 images, 10 human captions each | 40 binary attributes, identity | human | not stated | Drive folder (not tested) | CelebA attributes used by FocalLens | faces and ethics; binary aspects |
| 7 | DeepFashion-MultiModal (SIGGRAPH 2022) | 44,096 images | 12 shape attributes, fabric (8), colour or pattern (8) | one description per image, apparently templated from the labels (we infer) | non-commercial research | live | DeepFashion used by ASEN and CRL | text leaks the label |
| runner-up | EmoSet-118K (ICCV 2023) | 118,102 human-labelled | emotion (8), scene, object, facial expression, action, brightness, colourfulness | none | non-commercial | live | none | captions would have to be generated; attributes machine-predicted |
| not recommended | MM-CelebA-HQ, CelebA-Dialog | 30k, 202k | face attributes | generated from the labels | non-commercial | MM-CelebA-HQ links removed | none | total label leakage |
| not recommended | UT-Zappos50k | 50,025 | category, gender, heel height, closure | none | academic | live | CSN, SCE-Net, ASEN | no captions |

<!--block:B0159-->
### 5. Novelty risks and how a reviewer would phrase them

<!--block:B0160-->
1. **"The condition rule is relevance feedback."** "ReLU(mean support − mean contrast) is the Rocchio update (1971), and per-feature reweighting from examples goes back to MindReader and Rui et al. (1998)." Answer: present the rule as a deliberately simple interface, claim the learned shared factors, and report Rocchio on raw features (baseline 1).
2. **"This is few-shot classification, not conditional similarity."** "The positive is the only candidate with the support label, so a 4-shot prototype that ignores the anchor solves the task; Ren et al. already studied episodes where positive and negative supports define an attribute." Answer: report the anchor-free prototype, and add episodes where the anchor matters, such as GeneCIS-style condition-only distractors (same label as the supports, different second aspect) and same-anchor swaps across emotion, style and genre.
3. **"Weighted per-dimension similarity is a Conditional Similarity Network."** "Σ w_l a_I,l a_T,l is CSN's masked distance, and SCE-Net infers the mask from the compared items without condition labels." Answer: show what is new with ablations: factors shared by image and text (against a split dictionary, as MGSAE warns), masks set at test time from example pairs, both retrieval directions.
4. **"Frozen-encoder conditional similarity already exists without training."** "CLAY (CVPR 2026) and CRL (NeurIPS 2025) reshape frozen CLIP similarity from a text condition; InDiReCT did so in 2023; Qwen3-VL-Embedding takes the condition as an instruction." Answer: baselines 4 and 5, and evidence that examples beat names where the aspect is subjective (CLAY's own mood condition gains only 4.9 mAP over CLIP-B, T2(a)).
5. **"The no-label claim is overstated."** "The factors are trained on outputs of a supervised GoEmotions classifier that names 6 of the 8 emotions; EmotionCLIP used this kind of distant supervision in 2023." Answer: claim "no labels from the evaluation taxonomy", name the teacher model exactly (it is not the model PercepT used, §2.4), and report the teacher-only baseline and the caption-word shortcut analysis from the held report.
6. **"Small numbers on an in-house benchmark."** "17 to 25% R@1 among 13 candidates on home-made episodes." Answer: add GeneCIS with its published frozen B/32 rows and CUB with Reed captions.
7. **"Missing state-of-the-art comparisons."** "No CIR, conditional-similarity or universal-embedder baseline." Answer: SEARLE, CIReVL and OSrCIR on GeneCIS; CRL or CLAY and an instruction-following embedder on our episodes.
8. **"ViT-B/32 is outdated."** Now that any frozen encoder is allowed, the answer is to report the main result on one strong open backbone (for example SigLIP 2 or Qwen3-VL-Embedding) beside B/32, keeping the backbone frozen.
9. **"Concurrent work crowds the problem."** TPIPS (July 2026), SteerViT, Fioresi et al. (ECCV 2026), COCO-Facet (NeurIPS 2025), TEVI (EMNLP 2026) and CLAY all make similarity or retrieval conditional on text. Answer: cite them and position the paper on the example interface and the cross-modal score, where none of them works.

<!--block:B0161-->
### 6. References

<!--block:B0162-->
Links were fetched during this review unless marked UNVERIFIED. Venues come from the arXiv comment, the venue page or the repository; where only a secondary source gave the venue we say so.

<!--block:B0163-->
**Conditional similarity (§2.1)**
<!--block:B0164-->
- Veit, Belongie, Karaletsos. Conditional Similarity Networks. CVPR 2017. https://arxiv.org/abs/1603.07810
- Tan, Vasileva, Saenko, Plummer. Learning Similarity Conditions Without Explicit Supervision. ICCV 2019. https://arxiv.org/abs/1908.08589
- Ye, Shi, Zhan. Identifying Ambiguous Similarity Conditions via Semantic Matching. CVPR 2022. https://arxiv.org/abs/2204.04053
- Shi, Li, Gan, Zhan, Ye. Generalized Conditional Similarity Learning via Semantic Matching. IEEE TPAMI 2025. https://doi.org/10.1109/TPAMI.2025.3535730 (abstract page not fetched; metadata from search)
- Ren, Triantafillou, Wang, Lucas, Snell, Pitkow, Tolias, Zemel. Probing Few-Shot Generalization with Attributes. arXiv 2012.05895. https://arxiv.org/abs/2012.05895
- Ma, Dong, Long, Zhang, He, Xue, Ji. Fine-Grained Fashion Similarity Learning by Attribute-Specific Embedding Network. AAAI 2020. https://arxiv.org/abs/2002.02814
- Dong et al. Fine-Grained Fashion Similarity Prediction by Attribute-Specific Embedding Learning (ASEN++). https://arxiv.org/abs/2104.02429 (venue UNVERIFIED)
- Plummer, Kordas, Kiapour, Zheng, Piramuthu, Lazebnik. Conditional Image-Text Embedding Networks. ECCV 2018. https://arxiv.org/abs/1711.08389
- Vaze, Carion, Misra. GeneCIS: A Benchmark for General Conditional Image Similarity. CVPR 2023. https://arxiv.org/abs/2306.07969 ; weights https://github.com/facebookresearch/genecis/blob/main/DOWNLOAD.md
- Kobs, Steininger, Hotho. InDiReCT: Language-Guided Zero-Shot Deep Metric Learning for Images. WACV 2023. https://arxiv.org/abs/2211.12760
- Kwon et al. Image Clustering Conditioned on Text Criteria. ICLR 2024. https://arxiv.org/abs/2310.18297
- Yao, Qian, Hu. Multi-Modal Proxy Learning Towards Personalized Visual Multiple Clustering. CVPR 2024. https://arxiv.org/abs/2404.15655
- Zhang, Pan, Wang. Learning Emotion Representations from Verbal and Nonverbal Communication (EmotionCLIP). CVPR 2023. https://arxiv.org/abs/2305.13500

<!--block:B0165-->
**Composed image retrieval (§2.2)**
<!--block:B0166-->
- Saito et al. Pic2Word. CVPR 2023. https://arxiv.org/abs/2302.03084
- Baldrati et al. SEARLE (and CIRCO). ICCV 2023. https://arxiv.org/abs/2303.15247 ; code https://github.com/miccunifi/SEARLE
- Agnolucci et al. iSEARLE. https://arxiv.org/abs/2405.02951
- Gu et al. Language-only Efficient Training of Zero-shot Composed Image Retrieval (LinCIR). CVPR 2024. https://arxiv.org/abs/2312.01998
- Gu et al. CompoDiff. TMLR 2024. https://arxiv.org/abs/2303.11916
- Zhang et al. MagicLens. ICML 2024. https://arxiv.org/abs/2403.19651
- Karthik et al. Vision-by-Language for Training-Free Compositional Image Retrieval (CIReVL). ICLR 2024. https://arxiv.org/abs/2310.09291
- Tang et al. Reason-before-Retrieve (OSrCIR). CVPR 2025. https://arxiv.org/abs/2412.11077
- Tang et al. PrediCIR. CVPR 2025. https://arxiv.org/abs/2503.17109
- Byun et al. RTD. ICCV 2025. https://arxiv.org/abs/2406.09188
- Zhou et al. MegaPairs. ACL 2025. https://arxiv.org/abs/2412.14475
- Liu et al. LamRA. CVPR 2025. https://arxiv.org/abs/2412.01720
- Li et al. STiTch. https://arxiv.org/abs/2605.21261
- Wu, Lin, Yang. SQUARE. https://arxiv.org/abs/2509.26330
- Wang, Zhao, Kong. Generating a Paracosm for Training-Free ZS-CIR. ECCV 2026. https://arxiv.org/abs/2602.00813
- Liu et al. DeCIR. https://arxiv.org/abs/2605.08389
- Kwon. PACT. https://arxiv.org/abs/2609.31202
- Zhang et al. FoCo. ECCV 2026. https://arxiv.org/abs/2607.00374
- Yang, Du, Qian, Xu. ZeroSight. https://arxiv.org/abs/2606.07032
- Li et al. COMBINER. IEEE TIP 2026. https://arxiv.org/abs/2606.04604
- Lu et al. MCMR. CVPR 2026. https://arxiv.org/abs/2603.01082
- Wu et al. Fashion IQ. CVPR 2021. https://arxiv.org/abs/1905.12794
- Liu et al. CIRR. ICCV 2021. https://arxiv.org/abs/2108.04024

<!--block:B0167-->
**Example-conditioned retrieval (§2.3)**
<!--block:B0168-->
- Rocchio. Relevance feedback in information retrieval. In Salton (ed.), The SMART Retrieval System, Prentice-Hall, 1971, pp. 313 to 323 (book chapter; metadata from search)
- Rui, Huang, Ortega, Mehrotra. Relevance feedback: a power tool for interactive content-based image retrieval. IEEE TCSVT 8(5), 1998. https://doi.org/10.1109/76.718510
- Ishikawa, Subramanya, Faloutsos. MindReader: Querying Databases Through Multiple Examples. VLDB 1998. https://www.semanticscholar.org/paper/04938be9fd727ea6363cc950efd263ff82d02b77
- Sadeghi, Zitnick, Farhadi. VISALOGY: Answering Visual Analogy Questions. NIPS 2015. https://arxiv.org/abs/1510.08973
- Cohen, Gal, Meirom, Chechik, Atzmon. "This is my unicorn, Fluffy": Personalizing frozen vision-language representations (PALAVRA). ECCV 2022. https://arxiv.org/abs/2204.01694
- Ryan, Sivic, Caba Heilbron, Hoffman, Rehg, Russell. Improving Personalized Search with Regularized Low-Rank Parameter Updates. CVPR 2025. https://arxiv.org/abs/2506.10182
- Nara et al. Revisiting Relevance Feedback for CLIP-based Interactive Image Retrieval. ECCV Workshops 2024. https://arxiv.org/abs/2404.16398
- Lülf, Martins, Salles, Zhou, Gieseke. CLIP-Branches: Interactive Fine-Tuning for Text-Image Retrieval. SIGIR 2024. https://arxiv.org/abs/2406.13322
- Idan et al. Few Shots Text to Image Retrieval. https://arxiv.org/abs/2603.25891
- Zhang et al. Tip-Adapter. ECCV 2022. https://arxiv.org/abs/2207.09519
- Huang et al. LP++. CVPR 2024. https://arxiv.org/abs/2404.02285
- Silva-Rodríguez, Hajimiri, Ben Ayed, Dolz. A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models (CLAP). CVPR 2024. https://arxiv.org/abs/2312.12730
- Snell, Swersky, Zemel. Prototypical Networks for Few-shot Learning. NeurIPS 2017. https://arxiv.org/abs/1703.05175
- Oreshkin, Rodriguez, Lacoste. TADAM. NeurIPS 2018. https://arxiv.org/abs/1805.10123
- Ye, Hu, Zhan, Sha. FEAT. CVPR 2020. https://arxiv.org/abs/1812.03664
- Sonthalia, Uselis, Oh. On the rankability of visual embeddings. https://arxiv.org/abs/2507.03683
- Veit, Nickel, Belongie, van der Maaten. Separating Self-Expression and Visual Content in Hashtag Supervision. https://arxiv.org/abs/1711.09825
- Sun et al. GCRDP, few-shot cross-modal retrieval. https://arxiv.org/abs/2505.13306
- Nguyen et al. Visual Instruction Inversion. NeurIPS 2023. https://arxiv.org/abs/2307.14331

<!--block:B0169-->
**Affect and art (§2.4)**
<!--block:B0170-->
- Mohamed, Church, Elhoseiny. Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data (PercepT). https://arxiv.org/abs/2606.03345
- Achlioptas, Ovsjanikov, Haydarov, Elhoseiny, Guibas. ArtEmis. CVPR 2021. https://arxiv.org/abs/2101.07396
- Mohamed, Khan, Haydarov, Elhoseiny. It is Okay to Not Be Okay (ArtEmis 2.0). CVPR 2022. https://arxiv.org/abs/2204.07660
- Mohamed et al. ArtELingo. EMNLP 2022. https://arxiv.org/abs/2211.10780 ; ArtELingo-28. EMNLP 2024. https://arxiv.org/abs/2411.03769
- Achlioptas, Ovsjanikov, Guibas, Tulyakov. Affection. CVPR 2023. https://arxiv.org/abs/2210.01946
- Yang et al. EmoSet. ICCV 2023. https://arxiv.org/abs/2307.07961
- Wu, Nakashima, Garcia. Not Only Generative Art (GOYA). ICMR 2023 (DOI 10.1145/3591106.3592262). https://arxiv.org/abs/2304.10278
- Somepalli et al. Measuring Style Similarity in Diffusion Models (CSD). https://arxiv.org/abs/2404.01292
- Garcia, Vogiatzis. How to Read Paintings (SemArt). https://arxiv.org/abs/1810.09617 (ECCV Workshops 2018 per secondary sources)
- Zhang et al. Aligning Vision Models with Human Aesthetics in Retrieval (HPIR). https://arxiv.org/abs/2406.09397

<!--block:B0171-->
**Sparse and concept codes (§2.5)**
<!--block:B0172-->
- Bhalla et al. Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE). NeurIPS 2024. https://arxiv.org/abs/2402.10376 ; code https://github.com/AI4LIFE-GROUP/SpLiCE
- Rao, Mahajan, Böhle, Schiele. Discover-then-Name. ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09973.pdf
- Zaigrajew, Baniecki, Biecek. Interpreting CLIP with Hierarchical Sparse Autoencoders. ICML 2025. https://arxiv.org/abs/2502.20578
- Papadimitriou, Su, Fel, Gil, Kakade. Interpreting the linear structure of vision-language model embedding spaces. COLM 2025. https://arxiv.org/abs/2504.11695
- Kaushik, Barch, Fanelli. Decomposing multimodal embedding spaces with group-sparse autoencoders. https://arxiv.org/abs/2601.20028
- Gu et al. LUCID-SAE. https://arxiv.org/abs/2602.07311
- Gordon, Levi, Gilboa. SCoCCA. https://arxiv.org/abs/2603.13884
- Nasiri-Sarvi, Rivaz, Hosseini. SPARC. TMLR 2026. https://arxiv.org/abs/2507.06265
- Kubaty et al. Conceptualizing Embeddings (CEDAR). https://arxiv.org/abs/2605.22679
- Mahajan, Rao, Xie, Koller, Schiele. TEVI. EMNLP 2026. https://arxiv.org/abs/2606.07451
- Chen et al. STAIR. EMNLP 2023. https://arxiv.org/abs/2301.13081
- Luo et al. LexLIP. ICCV 2023. https://arxiv.org/abs/2302.02908
- Zhou et al. Retrieval-based Disentangled Representation Learning with Natural Language Supervision (VDR). ICLR 2024. https://arxiv.org/abs/2212.07699
- Kang, Wang, Xiong. Interpret and Control Dense Retrieval with Sparse Latent Features. https://arxiv.org/abs/2411.00786

<!--block:B0173-->
**Text-conditioned embeddings (§2.6)**
<!--block:B0174-->
- Lim, Lee, Park, Oh. CLAY. CVPR 2026. https://arxiv.org/abs/2604.11539
- Liu, Sun, Hu, Li, Peng. Conditional Representation Learning for Customized Tasks (CRL). NeurIPS 2025. https://arxiv.org/abs/2510.04564 ; code https://github.com/XLearning-SCU/2025-NeurIPS-CRL
- Wang, Lyu, Li, Jia. Semantic Purification for Conditional Representation Learning (SP-CRL). https://arxiv.org/abs/2602.05464
- Fioresi, Caba Heilbron, Nathani, Shah, Kafle. Controlling Embedding Spaces with Text-Conditioned Transformations. ECCV 2026. https://arxiv.org/abs/2607.22919
- Hsieh et al. FocalLens. https://arxiv.org/abs/2504.08368
- Kawarada, Yamada, Tejero-de-Pablos, Inoue. Training-free Conditional Image Embedding Framework Leveraging Large Vision Language Models (DIOR). https://arxiv.org/abs/2512.21860
- Ruthardt, Gaur, Ramanan, Tapaswi, Asano. Steerable Visual Representations (SteerViT). https://arxiv.org/abs/2604.02327 (ECCV 2026 per secondary sources)
- Wang, Nitzan, Hertzmann, Zhu, Shechtman, Efros, Zhang. The Many Senses of Visual Similarity (TPIPS). https://arxiv.org/abs/2607.18237
- Li, Gao, Du. Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval (COCO-Facet). NeurIPS 2025. https://arxiv.org/abs/2505.15877
- Xiao et al. FLAIR. https://arxiv.org/abs/2412.03561
- Sun et al. Alpha-CLIP. CVPR 2024. https://arxiv.org/abs/2312.03818

<!--block:B0175-->
**Benchmarks (§2.7, §4)**
<!--block:B0176-->
- Reed, Akata, Schiele, Lee. Learning Deep Representations of Fine-grained Visual Descriptions. CVPR 2016. https://arxiv.org/abs/1605.05395 ; CUB https://www.vision.caltech.edu/datasets/cub_200_2011/
- Chun et al. Probabilistic Embeddings for Cross-Modal Retrieval (PCME). CVPR 2021. https://arxiv.org/abs/2101.05068
- Jiang et al. Text2Human (DeepFashion-MultiModal). SIGGRAPH 2022. https://arxiv.org/abs/2205.15996
- Xia et al. TediGAN (MM-CelebA-HQ). CVPR 2021. https://arxiv.org/abs/2012.03308
- Yu, Grauman. UT-Zappos50K. https://vision.cs.utexas.edu/projects/finegrained/utzap50k/
- SemArt data: https://researchdata.aston.ac.uk/id/eprint/380

<!--block:B0177-->
**Backbones and universal embedders (§2.8)**
<!--block:B0178-->
- Radford et al. CLIP. https://arxiv.org/abs/2103.00020 (not re-fetched)
- Cherti et al. Reproducible scaling laws for contrastive language-image learning (OpenCLIP). CVPR 2023. https://arxiv.org/abs/2212.07143
- Sun et al. EVA-CLIP. https://arxiv.org/abs/2303.15389 ; EVA-CLIP-18B https://arxiv.org/abs/2402.04252
- Tschannen et al. SigLIP 2. https://arxiv.org/abs/2502.14786 ; card https://huggingface.co/google/siglip2-so400m-patch14-384
- Bolya et al. Perception Encoder. https://arxiv.org/abs/2504.13181
- Koukounas et al. jina-clip-v2. https://arxiv.org/abs/2412.08802 ; card https://huggingface.co/jinaai/jina-clip-v2
- Li, Li, Savarese, Hoi. BLIP-2. ICML 2023. https://arxiv.org/abs/2301.12597 ; LAVIS https://github.com/salesforce/LAVIS
- Wei et al. UniIR. ECCV 2024. https://arxiv.org/abs/2311.17136 ; https://github.com/TIGER-AI-Lab/UniIR
- Jiang et al. E5-V. https://arxiv.org/abs/2407.12580 ; card https://huggingface.co/royokong/e5-v
- Jiang et al. VLM2Vec. ICLR 2025. https://arxiv.org/abs/2410.05160
- Meng et al. VLM2Vec-V2. https://arxiv.org/abs/2507.04590
- Zhang et al. GME. CVPR 2025. https://arxiv.org/abs/2412.16855 ; card https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
- Lin et al. MM-Embed. ICLR 2025. https://arxiv.org/abs/2411.02571 ; card https://huggingface.co/nvidia/MM-Embed
- Chen et al. mmE5. https://arxiv.org/abs/2502.08468
- Gu et al. UniME. ACM MM 2025. https://arxiv.org/abs/2504.17432
- Lan et al. LLaVE. EMNLP 2025 Findings. https://arxiv.org/abs/2503.04812
- Thirukovalluru et al. B3. https://arxiv.org/abs/2505.11293
- Xiao et al. MetaEmbed. ICLR 2026. https://arxiv.org/abs/2509.18095
- Li et al. Qwen3-VL-Embedding and Qwen3-VL-Reranker. https://arxiv.org/abs/2601.04720 ; card https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
- Günther et al. jina-embeddings-v4. https://arxiv.org/abs/2506.18902 ; card https://huggingface.co/jinaai/jina-embeddings-v4
- Zhai et al. LiT. CVPR 2022. https://arxiv.org/abs/2111.07991
- Kossen et al. Three Towers. NeurIPS 2023. https://arxiv.org/abs/2305.16999
- Maniparambil et al. Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment. CVPR 2025. https://arxiv.org/abs/2409.19425
- Maniparambil et al. Do Vision and Language Encoders Represent the World Similarly? CVPR 2024. https://arxiv.org/abs/2401.05224
- Zhang, Yang, Agrawal. Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL). CVPR 2025. https://arxiv.org/abs/2412.04616
- Norelli et al. ASIF: Coupled Data Turns Unimodal Models to Multimodal Without Training. https://arxiv.org/abs/2210.01738
- Moschella et al. Relative representations enable zero-shot latent space communication. ICLR 2023. https://arxiv.org/abs/2209.15430
- Merullo, Castricato, Eickhoff, Pavlick. Linearly Mapping from Image to Text Space. ICLR 2023. https://arxiv.org/abs/2209.15162
- Huh, Cheung, Wang, Isola. The Platonic Representation Hypothesis. https://arxiv.org/abs/2405.07987
- Schnaus, Araslanov, Cremers. It's a (Blind) Match! CVPR 2025. https://arxiv.org/abs/2503.24129

<!--block:B0179-->
**UNVERIFIED or second-hand items.** DistillCIR, DiffComp and CIG GeneCIS tables (CVF returned 403; numbers only as quoted by others); SEIZE's GeneCIS numbers; whether MCL (ICML 2024) reports GeneCIS; DiscoverNet and Generalized CSL numbers; the identity of SteerViT's "specialized" baseline; MMRet's CLIP-B patch size; licences of VLM2Vec-V2, E5-V, mmE5, PE-Core, the Reed caption files and CelebAText-HQ; the CUB licence conflict; how DeepFashion-MultiModal captions were made; venues of FocalLens, SemArt, SteerViT and DIOR from secondary sources only.


<!--block:B0180-->
---

<!--block:B0181-->
# Appendix (evidence report): 2026-10-22_support_baseline_spike

<!--block:B0182-->
## CoSiR v2: do simple support-set baselines match SE on the label episodes? (spike)

<!--block:B0183-->
### Verdict

<!--block:B0184-->
**No, SE does not hold the lead. Simple support-set baselines on raw CLIP features beat SE pooled, and the gain
is all in art style.** The best is a logistic probe fitted on the 4 + 4 examples of both modalities: pooled R@1
24.10% against SE's 21.22% (difference +2.87 [+2.14, +3.64]) and CLIP only's 13.45%. On art style it gains +5.62
[+4.48, +6.71] over SE; on emotion it is matched, not beaten (+0.13 [−0.89, +1.15]). **The query adds little.** A
prototype scorer that never looks at the query reaches 22.83, already above SE, and adding the query moves it by
+0.57 [+0.20, +0.96]. In these label episodes every support carries the anchor's label and every distractor lacks
it, so the supports alone identify the positive. The episodes therefore mostly measure few-shot recognition of a
label value from examples, not conditional similarity between a query and a candidate. Where SE does win, it wins
cross-modally (style i2t, emotion t2i), which a within-modality baseline cannot do. The claim "learned factors beat
the obvious few-shot baseline" does not hold today. This is a diagnostic spike on selection rows, input to the CVPR
plan, not a decision.

<!--block:B0185-->
### What we tested

<!--block:B0186-->
All numbers are R@1 (%) on the 4,096 emotion and 4,096 art-style label episodes of the affect selection run
([report](2026-10-18_candidate_a_affect_factor_learning_selection.md)), 13 candidates each, chance 7.69%, mean of
the two retrieval directions unless a direction is named. In an episode the *anchor* is the query, the *positive* is
the candidate with the anchor's label, the *supports* are 4 other items with that label and the *contrasts* are 4
items without it. i2t ranks captions for an image query, t2i ranks images for a caption query. Differences are
paired bootstrap over episodes, 5,000 resamples, seed 42.

<!--block:B0187-->
Scorers (all on raw CLIP ViT-B/32 features, no training except the probe fit inside each episode):

<!--block:B0188-->
- **CLIP only**: cosine of query and candidate. **C0** and **SE** are the factor models of the affect run, scored
  with the *naive rule* (the parameter-free rule that turns the supports and contrasts into condition weights,
  β 0.3, stored ranks reproduced exactly). SE is the affect-signal model that passed the held test; C0 is its
  matched control.
- **Prototype**: score the candidate by its cosine to the mean support minus its cosine to the mean contrast, in
  the candidate's own modality. **Condition only** means the query is ignored (`proto` at λ = ∞). **Prototype +
  query** (`proto_cv`) adds the query: z(cos(query, candidate)) + λ · z(prototype term), with λ picked by
  cross-fitting.
- **Probe**: an L2 logistic regression (C = 1) fitted per episode on the supports (label 1) and contrasts (label
  0), applied to the candidates. `probe` uses the candidate modality only; **`probe_x` uses both modalities**
  (4 + 4 images and 4 + 4 captions).
- Other terms are in the table below: `proto_pos` (supports only), `dir` (a support minus contrast direction applied
  to the query and the candidate), `SE_term` and `C0_term` (the factor term alone, tuned), `raw_naive` (the naive
  rule on raw coordinates).

<!--block:B0189-->
λ was chosen on the pooled R@1 of one episode parity half and applied to the other (cross-fitting). The spike
reproduced the stored episodes (SHA-256), CLIP only, C0 and SE ranks exactly before any new scorer was run.

<!--block:B0190-->
### Results

<!--block:B0191-->
![Support-set baselines against SE](../../assets/2026-10-22_support_baseline_spike/support_baselines.png)

<!--block:B0192-->
*Figure 1. R@1 (%) on the selection episodes. (a) Pooled, emotion and art style for CLIP only, C0, SE, the
condition-only prototype, the prototype with the query and the both-modality probe. (b) The four direction by label
cells for CLIP only, SE and the prototype with the query. Dotted line: chance, 7.69%. Cross-fitted λ for the
prototype and probe, naive β 0.3 for SE and C0.*

<!--block:B0193-->
#### Result 1: support-set baselines beat SE pooled

<!--block:B0194-->
| Scorer | Pooled | Emotion | Style | Pooled − SE, paired 95% CI |
|---|---:|---:|---:|---|
| CLIP only (baseline of everything) | 13.45 | 10.83 | 16.06 | |
| C0 (SE's matched control) | 20.01 | 15.12 | 24.89 | |
| **SE** (naive rule, β 0.3) | 21.22 | 16.43 | 26.01 | |
| Prototype, condition only | 22.83 | 15.44 | 30.21 | +1.61 [+0.82, +2.40] |
| Prototype + query (`proto_cv`) | 23.40 | 15.77 | 31.03 | +2.18 [+1.43, +2.96] |
| Probe, one modality (`probe_cv`) | 23.57 | 15.89 | 31.24 | +2.34 [+1.61, +3.13] |
| **Probe, both modalities (`probe_x_cv`)** | **24.10** | **16.56** | **31.63** | **+2.87 [+2.14, +3.64]** |
| Supports only prototype (`proto_pos_cv`) | 20.81 | 12.76 | 28.87 | −0.41 [−1.20, +0.42] |
| Direction (`dir_cv`) | 18.73 | 11.76 | 25.71 | −2.49 [−3.23, −1.76] |
| SE factor term alone, tuned (`SE_term_cv`) | 21.91 | 17.46 | 26.35 | +0.68 [+0.33, +1.04] |
| C0 factor term alone, tuned (`C0_term_cv`) | 20.45 | 15.83 | 25.07 | −0.77 [−1.39, −0.18] |
| `raw_naive_cv` (scale artifact, see below) | 13.46 | 10.88 | 16.05 | −7.76 [−8.47, −7.06] |

<!--block:B0195-->
Paired differences by label against SE:

<!--block:B0196-->
| Scorer minus SE | Emotion | Style |
|---|---|---|
| Condition-only prototype | −0.99 [−2.08, +0.10] | +4.20 [+3.02, +5.40] |
| Prototype + query | −0.66 [−1.68, +0.35] | +5.02 [+3.87, +6.16] |
| Probe, both modalities | +0.13 [−0.89, +1.15] | +5.62 [+4.48, +6.71] |

<!--block:B0197-->
Reading. The probe beats SE pooled by +2.87 points, and nearly all of it is art style (+5.62). On emotion no
support baseline is distinguishable from SE (every interval contains 0), so we say matched, not beaten. SE's
emotion edge over C0 (+1.31 on selection, +2.08 on held) is real, but it does not translate into an edge over a
prototype of the supports. The fair SE comparator is `SE_term_cv` (the factor term alone, tuned), which is +0.68
above the untuned stored SE; against it the probe's pooled lead is smaller but still positive, and on emotion
`SE_term_cv` (17.46) is the highest scorer in the table, 0.9 above the probe. We did not pair those two, so we make
no claim about that gap.

<!--block:B0198-->
#### Result 2: the query adds little, so the episodes mostly test few-shot recognition

<!--block:B0199-->
| Scorer | Pooled | Emotion | Style |
|---|---:|---:|---:|
| Prototype, condition only | 22.83 | 15.44 | 30.21 |
| Prototype + query | 23.40 | 15.77 | 31.03 |
| Query adds (paired) | +0.57 [+0.20, +0.96] | +0.33 [−0.16, +0.83] | +0.82 [+0.22, +1.39] |

<!--block:B0200-->
A scorer that ignores the query reaches 22.83, above SE (21.22) and above C0 (20.01), and the query lifts it by 0.57
points. The mechanism is in the episode design (`src/eval/label_episodes.py`): the positive shares the anchor's label
with every support, and no distractor does. Evidence about the label therefore ranks the positive first without any
reference to the query. This is not conditional similarity, where the right answer depends on both the query and the
condition. GeneCIS avoids the shortcut: its gallery mixes distractors that match only the reference and distractors
that match only the condition ([feasibility check](2026-10-20_genecis_feasibility.md) §1 and §7, where slots 0 to
8 are similar scenes without the condition and slots 9 to 13 other scenes with it). A condition-only scorer cannot
win there. The label-episode benchmark can, and does.

<!--block:B0201-->
#### Result 3: where SE wins, it wins cross-modally

<!--block:B0202-->
| Cell (candidate ranked) | CLIP only | SE | Prototype + query | Leader |
|---|---:|---:|---:|---|
| Style i2t (captions) | 14.70 | **22.51** | 14.75 | SE, +7.76 |
| Emotion t2i (images) | 11.94 | **15.84** | 13.55 | SE, +2.29 |
| Emotion i2t (captions) | 9.72 | 17.02 | **17.99** | prototype, +0.97 |
| Style t2i (images) | 17.43 | 29.52 | **47.31** | prototype, +17.79 |

<!--block:B0203-->
SE leads where the candidate's modality carries the aspect weakly: art style in captions and emotion in images. The
prototype leads where the candidate modality carries the aspect directly: art style in images (47.31 against 29.52)
and emotion in captions (17.99 against 17.02). The prototype on style i2t (14.75) is no better than CLIP only
(14.70), while SE reaches 22.51. We did not test the cell-level differences with intervals.

<!--block:B0204-->
Reading, not a tested result: the shared factor space carries a condition shown in one modality over to the other,
which a within-modality prototype cannot do. This matches the headroom probe, where style was found to be visual and
emotion textual ([headroom probe](2026-10-15_candidate_a_factor_headroom_probe.md), Result 3). The probe on both
modalities recovers part of the cross-modal gap (style i2t 17.90, emotion t2i 14.62) but does not close it.

<!--block:B0205-->
#### Result 4: `raw_naive` is not evidence

<!--block:B0206-->
`raw_naive_cv` (13.46 pooled) equals CLIP only (13.45). The naive rule's factor term is computed on raw CLIP
coordinates, which are unit-norm with entries near 1/√512, so the term is about 1e-3 against a cosine in [−1, 1]
and β has no effect (the grid edge β = 1 was picked). It is a scale artifact, not a test of the naive rule on raw
features, and we draw nothing from it.

<!--block:B0207-->
### What this means for the publication plan

<!--block:B0208-->
- The claim "learned factors beat the obvious few-shot baseline" does not hold on these episodes today.
- The label-episode benchmark needs query-dependent episodes before it can test conditional similarity: for example
  condition-only distractors (match the label, wrong query relation), reference-only distractors, or conditions
  that name an aspect ("style") rather than a value.
- A method that combines within-modality support evidence with cross-modal factor transfer is the natural next
  candidate, since each side wins in the cells the other loses (Result 3).
- All of this is input to the CVPR plan being brainstormed, not a decision.

<!--block:B0209-->
### Caveats

<!--block:B0210-->
- Selection rows, read many times. Diagnostic, not pre-registered. One seed of SE, episode level CIs only.
- The λ grid ended at 4 and every pick for the prototype and probe terms sat at that edge (λ 4 in both folds, except
  `proto_pos` and the factor terms at ∞), so the prototype and probe numbers are, if anything, conservative.
- Cross-fitting splits by episode parity. Paintings can recur across the halves, so the split does not separate
  paintings, but only one λ is chosen per fold.
- SE is scored untuned at β 0.3; its tuned factor term is 0.68 points higher pooled (Result 1).
- The probe uses a batched Newton solver in the dual form of sklearn's objective. Within episode ordering equals
  sklearn at tol 1e−10 on 100% of 50 random episodes per setting; against sklearn's default tolerance it was 96% on
  emotion i2t (stopping noise, not the objective).

<!--block:B0211-->
### Files

<!--block:B0212-->
- Script: `src/test/20261022_support_baseline_spike/run_spike.py` (CPU, 22 s, throwaway; selection rows only).
- Log: `src/test/20261022_support_baseline_spike/20261022_support_baseline_spike_log.md`.
- Gitignored, local only: `src/test/20261022_support_baseline_spike/results/spike_results.json`, `spike_ranks.npz`,
  `run_spike.log`.
- Figure: `docs/reports/assets/2026-10-22_support_baseline_spike/support_baselines.png`, built by
  `docs/reports/assets/build_2026-10-22_support_baseline_spike_figures.py`.
- Previous step: [affect factor-learning held test](2026-10-19_candidate_a_affect_factor_learning_held.md).


<!--block:B0213-->
---

<!--block:B0214-->
# Appendix (evidence report): 2026-10-23_aspect_episode_spike

<!--block:B0215-->
## CoSiR v2: can a factor model select the aspect on cross-modal aspect episodes? (spike)

<!--block:B0216-->
### Verdict

<!--block:B0217-->
**No. On cross-modal aspect episodes no factor model selects the aspect, although the task is learnable.** SE with
the agreement rule reaches 11.32% pooled R@1 against CLIP only's 11.13% (difference +0.20 [−0.12, +0.50], the
interval includes 0). C0 (11.09), R3 (11.05), the raw CLIP agreement rule (11.16) and the value prototype (11.08)
are also at CLIP level, and chance is 7.69%. Only the privileged `names` reference, which is told the label names,
gains: 12.63, +1.51 [+0.94, +2.09] over CLIP only. The task itself is not the obstacle. Probes fitted with the human
labels reach about 24 (emotion) and 22 (art style) cross-modally against CLIP's 10 and 12, a pooled ceiling of 23.09
against CLIP's 11.13. Our factors capture none of that headroom. Emotion is carried by captions and style by images,
so a cross-modal match depends on the weaker modality, which caps this task on ArtELingo. This is a diagnostic spike
on selection rows, input to the CVPR plan, not a decision.

<!--block:B0218-->
### What we tested

<!--block:B0219-->
All numbers are R@1 (%) on 4,096 *aspect episodes* built on the selection rows of the affect run
([report](2026-10-18_candidate_a_affect_factor_learning_selection.md)), 13 candidates each, chance 7.69%, mean of
the two retrieval directions unless a direction is named. i2t ranks captions for an image anchor, t2i ranks images
for a caption anchor. The *anchor* is the query. Differences are paired bootstrap over anchors, 5,000 resamples,
seed 42.

<!--block:B0220-->
**Episode design.** The previous spike showed that value episodes (the supports carry the anchor's own label) are
solved by the supports alone ([support-baseline spike](2026-10-22_support_baseline_spike.md), Result 2). So the
examples here never carry the anchor's value; they name an *aspect* and the target is the candidate that shares the
anchor's value of that aspect.

<!--block:B0221-->
- The 13 candidates are shared by both conditions: [p_emo, p_style, 11 negatives]. p_emo shares the anchor's
  emotion, p_style shares its art style, and the 11 negatives share neither.
- Two sets of 4 cross-item example pairs (an image from one painting, a caption from another). In P_emo the 4 pairs
  share an emotion that differs from the anchor's; in P_style they share a style that differs from the anchor's.
  No example row has the anchor's emotion or style, and all 30 rows of an episode come from 30 distinct paintings.
- Under the *emotion* condition P_emo are the supports and P_style the contrasts, and the target is p_emo. Under the
  *style* condition the roles swap and the target is p_style. The other-aspect candidate is therefore the
  hardest distractor.
- Eligible values: 8 emotions (not "something else") and 23 styles with at least 30 selection paintings. 0 draw
  failures; every constraint was asserted on the final arrays.

<!--block:B0222-->
**Scorers.** *CLIP only* is the cosine of anchor and candidate. **SE**, **C0** and **R3** are the factor models of
the affect and candidate A runs (SE: affect signal; C0: its matched control; R3: the repaired factor recipe).
Each gets the *agreement rule*, which turns the examples into condition weights with no training:

<!--block:B0223-->
w = ReLU( mean over supports of a_I(x) ⊙ a_T(y) − mean over contrasts of a_I(x') ⊙ a_T(y') ), L1-normalized,

<!--block:B0224-->
where a_I and a_T are the image and caption factor codes of an example pair (x, y) and ⊙ is the elementwise
product. The factor term is Σ w · a(anchor) ⊙ a(candidate). The final score is z(cos) + λ · z(term), with z a
per-episode standardization and λ chosen by cross-fitting on anchor parity (grid 0, 0.25, 0.5, 1, 2, 4, 8, 16, ∞).
*Raw agree* applies the same rule to raw CLIP coordinates (with and without the ReLU). The *value prototype* scores a
candidate by its cosine to the mean support minus its cosine to the mean contrast, in the candidate's own modality.
*Names* (privileged) uses the label names as text prompts and is a reference, not a scorer we could ship. We also
report the *swap success*: the share of anchors where p_emo scores above p_style under the emotion condition and
p_style scores above p_emo under the style condition (a pairwise order between the two aspect candidates; the
*strict* variant asks for R@1 = 1 under both conditions). The *label ceiling* is described in Result 3.

<!--block:B0225-->
### Results

<!--block:B0226-->
![Aspect episodes](../../assets/2026-10-23_aspect_episode_spike/aspect_episodes.png)

<!--block:B0227-->
*Figure 1. R@1 (%) on the aspect episodes. (a) Pooled over both aspects and directions: CLIP only, the agreement
rule on SE, C0, R3 and raw CLIP, the value prototype, privileged names and the cross-modal label ceiling. (b) Per
aspect: CLIP only, SE agree, the cross-modal label ceiling and the same-modality ceiling (caption to caption for
emotion, image to image for style). Dotted line: chance, 7.69%. cv rows cross-fitted by anchor parity.*

<!--block:B0228-->
#### Result 1: no factor model beats CLIP only

<!--block:B0229-->
| Scorer | Emotion | Style | Pooled | Pooled minus CLIP, paired 95% CI |
|---|---:|---:|---:|---|
| CLIP only (baseline of everything) | 10.14 | 12.11 | 11.13 | |
| **SE agree (cv)** | 10.34 | 12.30 | 11.32 | +0.20 [−0.12, +0.50] |
| C0 agree (cv) | 10.03 | 12.15 | 11.09 | −0.04 [−0.34, +0.26] |
| R3 agree (cv) | 10.27 | 11.84 | 11.05 | −0.07 [−0.47, +0.31] |
| Raw CLIP agree (cv) | 10.29 | 12.02 | 11.16 | +0.03 [−0.25, +0.32] |
| Value prototype (cv) | 10.16 | 12.01 | 11.08 | −0.04 [−0.26, +0.17] |
| Names (privileged, cv) | 13.31 | 11.96 | 12.63 | +1.51 [+0.94, +2.09] |
| SE agree at β 0.3, uncross-fitted | 10.39 | 11.51 | 10.95 | |
| SE value rule at β 0.3 | 11.91 | 11.98 | 11.94 | |

<!--block:B0230-->
Reading. Every non-privileged scorer is within 0.3 points of CLIP only and every interval contains 0. SE against
the control C0 is +0.23 [−0.12, +0.59], so even SE's affect advantage on label episodes does not carry over. The
other-aspect rate (how often the other-aspect candidate ranks first, 11.13 for CLIP only) stays at 10.7 to 11.0
for the cv scorers, so they do not confuse the aspects more or less than CLIP does. The cross-fitted λ picks are
small (0.25 to 0.5 for SE, C0, R3 and raw agree, 0 for one fold of C0 and of the prototype), because R@1 falls as
λ grows. The raw agreement rule with the ReLU picked λ = 0 in both folds and equals CLIP only exactly. Names helps
on emotion only (+3.2 over CLIP only, 13.31 against 10.14) and is limited by weak zero-shot accuracy: emotion
prompts on captions reach 31.3% (majority class 31.8%) and style prompts on images 27.5% (majority 16.1%).

<!--block:B0231-->
#### Result 2: the swap test must be read next to R@1

<!--block:B0232-->
The swap success of the cv scorers is small (SE 4.96, C0 4.43, R3 6.65, raw agree 6.96, prototype 3.49, names
20.12; CLIP only 0.00). That is not a finding of aspect selection. Under λ = ∞ the prototype and raw agree scores
are exactly antisymmetric between the two conditions (swapping the roles negates the term), so swap success is
about 50% for any ordering: it reached 51.6 and 51.8 while R@1 was 8.0 and 8.6, at or below chance. An independent
random scorer reaches 25%. The uncross-fitted factor rows at β 0.3 show 16 to 18% swap because the factor term
dominates the cosine at raw code scale, again with R@1 at chance level. Swap success rises with λ while R@1 falls,
so a high swap with a CLIP-level R@1 means a scorer that moves with the condition without being correct. We report
swap only next to R@1.

<!--block:B0233-->
#### Result 3: the task is learnable, and each aspect lives in one modality

<!--block:B0234-->
The *label ceiling* is diagnostic only: for each aspect we fit a multinomial logistic regression with the human
labels on 60,000 scorer-train rows (one per modality, on the normalized CLIP features), take an item's class
posterior, and score a candidate by the dot product of its posterior with the anchor's, using the image probe on
image sides and the caption probe on caption sides. It scores this term alone, without CLIP. We evaluated it on the
same episodes. "Same-modality" scores compare the anchor and the candidates inside one modality, which is not the
cross-modal task.

<!--block:B0235-->
| Aspect | CLIP i2t | CLIP t2i | Ceiling i2t | Ceiling t2i | Same-modality ceiling |
|---|---:|---:|---:|---:|---:|
| Emotion | 9.42 | 10.86 | 23.58 | 24.29 | 36.62 (caption to caption) |
| Art style | 10.77 | 13.45 | 21.41 | 23.07 | 49.80 (image to image) |

<!--block:B0236-->
| Probe selection accuracy (%) | From image | From caption |
|---|---:|---:|
| Emotion | 35.2 | **56.9** |
| Art style | **60.8** | 25.4 |

<!--block:B0237-->
Reading. Cross-modal ceilings are 21 to 24 against CLIP's 9 to 13, so about 11 to 14 points of headroom exist and
SE takes +0.2 of them. Emotion is carried by captions (caption probe 56.9 against 35.2 from images, caption to
caption ceiling 36.6 against 17.9 image to image) and art style by images (60.8 against 25.4; image to image 49.8
against 13.1 caption to caption). A cross-modal match must read one aspect through its weaker modality, which is
why the cross-modal ceilings (about 24 and 22) sit well below the best same-modality ones (36.6 and 49.8). This is
the same pattern as the earlier headroom probe ([report](2026-10-15_candidate_a_factor_headroom_probe.md)). On
datasets whose captions describe what the image shows, such as CUB and COCO, we expect both modalities to carry the
aspect and the cross-modal ceiling to sit closer to the same-modality one. That is an expectation to test, not a
result.

<!--block:B0238-->
### What this means for the publication plan

<!--block:B0239-->
- The user chose method A: train the shared factors for aspect selection, with pseudo-aspect episodes built from
  two or more pseudo-partitions (so that no human label is needed at training time).
- The go/no-go target is selection-row aspect R@1 above CLIP (about 11 pooled), measured against the label ceiling
  (about 23). A trained factor model that does not leave the CLIP band on these episodes does not pass.
- CUB (now downloading) gets the same two checks first: CLIP only and the label-supervised ceiling. If CLIP is
  already near the ceiling there, the benchmark has no headroom and we should not build on it.
- The agreement rule on existing factors is closed as a route here. SE's earlier edge on value episodes did not
  survive an episode design where the supports cannot identify the positive by themselves.
- All of this is input to the CVPR plan being brainstormed, not a decision.

<!--block:B0240-->
### Caveats

<!--block:B0241-->
- Selection rows, read many times. Diagnostic, not pre-registered. One seed per model, anchor-level CIs only.
- The λ grid was shared by all cv rows and the picks are small (Result 1); the cv rows are tuned on R@1, so they
  cannot show a larger swap success at a price in R@1.
- Emotion labels are per annotation, and an image takes the emotion of its row, which is a noisy image label. The
  image emotion probe (35.2) and the emotion ceilings are therefore conservative.
- The label ceiling uses 60,000 of the scorer-train rows and one probe family. It is a diagnostic, not a bound in
  the strict sense; a better probe could score higher.
- Zero-shot names is no better than the majority class on emotion (31.3 against 31.8), so its gain is not a
  measure of what names can give.

<!--block:B0242-->
### Files

<!--block:B0243-->
- Script: `src/test/20261023_aspect_episode_spike/run_aspect.py` (35 s, one GPU process for CLIP text encoding,
  throwaway; selection rows asserted). Ceiling check: `src/test/20261023_aspect_episode_spike/aspect_ceiling.py`
  (about 2 minutes, CPU).
- Log: `src/test/20261023_aspect_episode_spike/20261023_aspect_episode_spike_log.md`.
- Gitignored, local only: `src/test/20261023_aspect_episode_spike/results/aspect_results.json`, `aspect_ranks.npz`,
  `aspect_episodes.npz`, `run_aspect.log`.
- Figure: `docs/reports/assets/2026-10-23_aspect_episode_spike/aspect_episodes.png`, built by
  `docs/reports/assets/build_2026-10-23_aspect_episode_spike_figures.py`.
- Previous step: [support-baseline spike](2026-10-22_support_baseline_spike.md).


<!--block:B0244-->
---

<!--block:B0245-->
# Appendix (evidence report): 2026-10-24_aspect_task_novelty_check

<!--block:B0246-->
## Novelty check: example-conditioned aspect similarity across modalities

<!--block:B0247-->
**Report date:** 2026-10-24 (sequence date in this folder; the research was done on 2026-10-02).
**Scope:** a narrow, read-only novelty check of the task defined in the [aspect episode spike](2026-10-23_aspect_episode_spike.md). We trained nothing and ran no evaluation. The [CVPR literature review](2026-10-21_cvpr_literature_review.md) already covered GeneCIS, CIR and ZS-CIR, CLAY, CRL, InDiReCT, CSN, SCE-Net, DiscoverNet, ASEN, PALAVRA, POLAR, CLIP-Branches, FSIR, Rocchio and relevance feedback, Tip-Adapter, few-shot attribute learning, sparse CLIP codes and universal embedders; we cite those rows from it and do not repeat them. That review searched for value-conditioned support sets; here we searched for the aspect formulation.
**Method.** The session's WebSearch budget was exhausted, shell network access was blocked, Semantic Scholar returned HTTP 429 after one query and dblp returned an access-denied page. We therefore searched the arXiv API (about 25 queries over titles and abstracts, 2002 to October 2026), fetched arXiv abstract pages and ar5iv or arXiv HTML full texts, and verified non-arXiv papers through Crossref, the JMLR page or one Semantic Scholar record. Every paper in the tables was opened in this session or in the 2026-10-21 review. Facts we did not re-check (mostly dataset lists of pre-2013 metric-learning papers) are marked "not re-checked"; where we draw a conclusion a paper does not state, we write "we infer".

<!--block:B0248-->
**The task under test.** A query x (an image or a caption) and a condition c = (S, C). S holds a few (4) support pairs, each an image of one item and a caption of another item, that agree on an unnamed aspect; different pairs show different values of that aspect and none shows the query's value. C holds a few contrast pairs that agree on a second aspect. The model ranks candidates in the other modality so that the candidate sharing the query's value on the S aspect ranks first, in both directions, and a swap test exchanges S and C and must flip the ranking between the aspect-1 and aspect-2 candidates. The method under review uses frozen encoders, a shared sparse image-text factor space and a training-free rule that weights factors by support-pair co-activation minus contrast-pair co-activation.

<!--block:B0249-->
### 1. Verdict

<!--block:B0250-->
We found no paper that defines this task: none scores image↔text across different items under an aspect given only by a few cross-item demonstration pairs whose values differ from the query's, with contrast pairs on a second aspect and a paired swap test. Every property exists separately. Metric learning from equivalence constraints (Xing et al. 2002, RCA 2005, ITML 2007, KISSME 2012) already defines a notion of similarity by pairs that agree under it, and KISSME's closed form subtracts a dissimilar-pair statistic from a similar-pair statistic, but these methods fit offline on many pairs in one feature space. The closest single paper in mechanism and protocol is Contextual Visual Similarity (Wang, Kitani, Hebert 2016), which fits per-dimension weights over fixed features at test time from 1 to 5 positive and negative images, but its positives share the query's attribute value and it is image only. The closest cross-modal formulation is multimodal analogical reasoning (MARS, ICLR 2023), where one image or text example pair defines a hidden relation for a query entity, but its relations are knowledge-graph relations learned on a training split, with no contrast set and no swap. In-context embedders (BGE-EN-ICL 2024, RICE 2026) let example pairs define a retrieval task without retraining, but only for text, and our sweep of 2024 to October 2026 found no multimodal embedder that takes example pairs in context. We can safely write that, to our knowledge, this is the first formulation of cross-modal similarity in which the aspect is fixed at test time only by value-disjoint, cross-item image-caption demonstrations with a contrast aspect, evaluated in both directions with a paired swap test. We must not claim that inferring a notion of similarity from examples or pairs, test-time feature reweighting, or relation-by-example retrieval is new, and a reviewer will call the rule "a few-shot, cross-modal, diagonal KISSME" unless we report that baseline (§5).

<!--block:B0251-->
### 2. Property table

<!--block:B0252-->
Properties: **(a)** the condition is an aspect, not a value or an instance; **(b)** it is given by examples, not a name or text; **(c)** image↔text across different items; **(d)** few-shot at test time without retraining; **(e)** support values are disjoint from the query's; **(f)** both retrieval directions and a swap-style test. "n/a" means the paper has no support examples. Rows marked † were verified in the 2026-10-21 review.

<!--block:B0253-->
**Table 1. Prior work against the six properties.**

<!--block:B0254-->
| Paper | (a) aspect | (b) examples | (c) I↔T, cross-item | (d) test-time few-shot | (e) disjoint values | (f) both directions, swap | Benchmark |
|---|---|---|---|---|---|---|---|
| **Thread 1: metric learning from pairs** | | | | | | | |
| Xing, Ng, Jordan, Russell, NIPS 2002 | yes: side information selects the notion of similarity | yes: similar and dissimilar pairs | no: one vector space | no: convex fit on the full constraint set | no: constraints come from the data later clustered | no | clustering with side information (datasets not re-checked) |
| RCA, Bar-Hillel, Hertz, Shental, Weinshall, JMLR 2005 | yes: equivalence constraints mark irrelevant variation | yes: chunklets of equivalent points (positive constraints only) | no | partly: closed form (inverse within-chunklet covariance), but many chunklets | no | no | clustering and classification (abstract) |
| ITML, Davis, Kulis, Jain, Sra, Dhillon, ICML 2007 | yes | yes: similarity and dissimilarity constraints | no | no: optimisation over all constraints | no | no | not re-checked |
| KISSME, Köstinger, Hirzer, Wohlhart, Roth, Bischof, CVPR 2012 | yes | yes: similar and dissimilar pairs; closed form Σ_S⁻¹ − Σ_D⁻¹ | no | no: thousands of pairs, offline | partly: applied to identities unseen in training (we infer from its verification and re-identification protocols) | no | face verification, person re-identification (datasets not re-checked) |
| He, Zhang, Wang et al., arXiv 1411.7798 | no: content matching | partly: pairwise constraints at training time | partly: image-text, matched pairs | no | no | not checked | cross-modal retrieval (not re-checked) |
| Few-shot metric learning (CRML), Jung, Kang, Kwak, Cho, arXiv 2211.07116 | no: class | yes: labelled support | no | partly: gradient steps on channel scale and shift at test time | no | no | miniImageNet, CUB, miniDeepFashion; CUB 5-way 5-shot mAP 82.7 (T2) |
| Category Traversal Module, Li, Eigen, Dodge, Zeiler, Wang, CVPR 2019 | no: class | yes: support set | no | yes: meta-trained module masks features per episode from intra-class commonality and inter-class uniqueness | no | no | few-shot classification (not re-checked) |
| Personalized clustering, Geng et al., AAAI 2025 | yes: the user's criterion (object vs background) | yes: must-link and cannot-link pairs | no | no: 10k constraints, gradient training | no | partly: the same images clustered under a default and a personalised criterion (T3); no per-query swap | CIFAR10-2, CIFAR100-4, ImageNet10-2 |
| Bernard et al., arXiv 1703.03385; Loeffler et al., TMLR 2023 | yes: the user's subjective notion | yes: labelled pairs (Bernard) or triplets (Loeffler) | no: soccer players, football trajectories | no: active learning with training | no | no | user studies |
| **Thread 2: example-conditioned embedders** | | | | | | | |
| BGE-EN-ICL, Li et al., arXiv 2409.15700 | partly: examples define the retrieval task | partly: query-passage pairs plus a task instruction | no: text only | yes: in context, no gradient at test time (trained with 0 to n examples) | yes: examples come from other queries | no | MTEB 64.67 → 66.08 with examples; AIR-Bench QA 52.93 → 54.36 |
| RICE, Jedidi, Ali, Li, Lin, arXiv 2609.38099 | partly: task | yes: 10 query-document pairs | no: text only | yes: training-free | yes | no | BEIR R@100 0.558 vs PromptReps 0.534 (T1) |
| Promptagator, Dai et al., arXiv 2209.11755 | partly: task | yes: 8 examples | no | no: trains a task retriever on generated queries | yes | no | not re-checked |
| Contextual Document Embeddings, Morris, Rush, arXiv 2410.02525 | no: corpus context | partly: neighbour documents, not chosen examples | no | yes | n/a | no | MTEB |
| Universal multimodal embedders† (VLM2Vec, GME, Qwen3-VL-Embedding; MuCo, CVPR 2026; Lens, arXiv 2609.20252) | partly: a text instruction names the task | no: instructions, not demonstrations | yes | yes | n/a | no | MMEB |
| **Thread 3: analogy and relation by example** | | | | | | | |
| VISALOGY, Sadeghi, Zitnick, Farhadi, NIPS 2015† | partly: an image pair defines a transformation | yes: one pair | no: images | partly: trained network applied to new analogies | yes: C differs from A | no | VAQA |
| VASR, Bitton et al., AAAI 2023 | partly: a change of situation role | yes: one pair | no: images | yes for zero-shot CLIP baselines | yes | no | 3,820 gold analogies; models about 53% vs humans 90% with chosen distractors |
| MARS, Zhang et al., ICLR 2023 | partly: one example pair defines a hidden knowledge-graph relation | yes: one pair | partly: entities are images or text, single or blended | no: models trained on the MARS training split (relation overlap not checked) | yes | partly: both modality orders appear; no swap | MARS: best Hits@1 0.301, MRR 0.341 |
| Relational Visual Similarity, Nguyen et al., CVPR 2026 | yes: one fixed relational notion | no: fixed by fine-tuning | no: I→I | no | n/a | no | own 114k anonymised captions |
| Visual Instruction Inversion (arXiv 2307.14331); Visual Prompting via Inpainting (arXiv 2209.00647); Painter (CVPR 2023) | partly: a before/after pair defines an edit or task | yes | no | yes: per-pair inversion or in-context prompting | yes | no | generation, not retrieval |
| **Thread 4: criterion defined by examples** | | | | | | | |
| Contextual Visual Similarity, Wang, Kitani, Hebert, arXiv 1612.02534 | partly: the context selects an attribute, but positives share the query's value | yes: k ∈ {1, 3, 5} positive and negative images | no: images | yes: per-query feature weights by gradient descent on fixed fc7 features | no | no: the swap appears only as the motivating example | own 2,197 images, 8 categories × 8 attributes; MAP 0.334 → 0.557 at k = 5 (T1) |
| Few-shot attribute learning, Ren et al., arXiv 2012.05895† | no: attribute value | yes: positive and negative supports | no | yes | no | no | Celeb-A, Zappos-50K, ImageNet-with-attributes |
| Bongard-HOI (CVPR 2022), Bongard-OpenWorld (ICLR 2024), support-set context (TMLR 2024) | no: one concept, a value | yes: positive and negative sets; HOI negatives differ only in the action | no: image examples | yes | no | no | HOI 62% vs human 91%; OpenWorld 64% vs 91%; 76.4% on HOI with support-set context |
| Agile Modeling, Stretcu et al., arXiv 2302.12948 | no: a subjective concept value | partly: named concept plus user labels | no | no: trains a classifier | no | no | user study, N = 14 |
| VisDiff, Dunlap et al., CVPR 2024 | partly: names what separates set A from set B | yes: two image sets | no: outputs text | yes | n/a | no | not retrieval |
| **Covered in the 2026-10-21 review** | | | | | | | |
| GeneCIS, CVPR 2023† | yes: "focus on" an attribute or object | no: text | no: I+T→I | yes: zero-shot evaluation | n/a | partly: focus and change tasks; no paired swap | GeneCIS |
| CSN 2017, SCE-Net 2019, DiscoverNet 2022† | yes | no (CSN: id) or partly (SCE-Net infers the mask from compared items) | no | no for CSN; yes at inference for SCE-Net | n/a | no | Zappos, Polyvore, Celeb-A |
| CLAY 2026, CRL 2025, InDiReCT 2023† | yes | no: text | no: I→I | yes: training-free or light | n/a | partly: CLAY scores one Stanford40 gallery under action, location and mood | Stanford40, GeneCIS, LanZ-DML |
| PALAVRA 2022, POLAR 2025† | no: instance | yes: a few images | partly: T→I for the same instance | partly: per-concept optimisation | no | no | PerVL, DeepFashion2, ConCon-Chi |
| Rocchio, relevance feedback for CLIP, CLIP-Branches, FSIR† | no: value or instance | yes | partly: T→I refinement | yes | no | no | FSIR-BD and others |
| Tip-Adapter, LP++† | no: class | yes: labelled shots | no | yes | no | no | 11 classification sets |
| COCO-Facet, NeurIPS 2025† | yes: attribute type in a text prompt | no | partly: T→I | yes | n/a | no | COCO-Facet |
| **Thread 5: fixed aspect across modalities** | | | | | | | |
| Zhao et al. (IMEMNet), ACM MM 2020 | fixed: emotion in valence-arousal, set at training | no | yes: image↔music | no | n/a | not checked | IMEMNet, 140K image-music pairs |
| Stewart, Avramidis, Feng, Narayanan, ICASSP 2024 | fixed: emotion | no | yes: image↔music | no | n/a | no | cross-modal retrieval, music tagging |
| MMVA, Choi, Kim, Kang, AAAI 2025 AI for Music workshop | fixed: valence-arousal | no | yes: images, music, music captions | no | n/a | no | IMEMNet-C |
| Won et al., ISMIR 2021 | fixed: emotion | no | yes: story text↔music | no | n/a | no | story-to-music retrieval |
| Song, Soleymani, arXiv 1804.04318 | fixed: implicit reaction concepts | no | yes: GIF↔sentence | no | n/a | no | 47K GIF-sentence pairs |
| Liu, Fu, Kato, Yoshikawa, ACM MM 2018 | fixed: poetic clues including sentiment | no | yes: image→poem (generation) | no | n/a | no | poem generation |
| **This task (CoSiR v2 aspect episodes)** | yes | yes: 4 support and 4 contrast cross-item pairs | yes | yes: training-free rule | yes | yes | ArtELingo aspect episodes |

<!--block:B0255-->
No row reaches more than three "yes" marks (RICE, VASR and the visual-prompting row reach three). Three gaps recur across all threads. Test-time demonstrations that differ from the query (d with e) appear only where they define a text retrieval task (RICE, BGE-EN-ICL) or a transformation between two images (VASR, Visual Instruction Inversion), never an aspect on which two items agree; KISSME reaches unseen values of such an aspect only offline. No prior work uses cross-item image-caption demonstrations (b with c); MARS comes nearest, with knowledge-graph relations. And we found no paired swap test (f) anywhere.

<!--block:B0256-->
### 3. Closest papers in detail

<!--block:B0257-->
**3.1 Contextual Visual Similarity** (Wang, Kitani, Hebert, arXiv 1612.02534, 2016; the arXiv comment mentions a CVPR 2017 submission and we found no published venue). *What it does.* A query image comes with k positive and k negative images (k = 1, 3, 5). The positives share the query's value of some attribute (a black dog with black horses), and the negatives belong to the query's category with another value (white dogs). The method learns a non-negative weight per feature dimension of fixed AlexNet fc7 features by gradient descent on a triplet ranking loss with regularisation, then searches with the weighted feature. *What it reports.* Attribute-specific search on its own 2,197 images (8 categories, 8 attributes): MAP rises from 0.334 (unweighted fc7) to 0.557 at k = 5 (T1). It also answers 4,280 generated visual analogy questions better than its baseline and finds 72 meaningful clusters among 116 in unsupervised attribute discovery. *Difference.* The positives carry the query's own value, so the examples define the target rather than the aspect (our support-baseline spike showed that such value episodes are solved by the supports alone); it is image only; it has no second-aspect contrast set and no swap evaluation. *Reviewer sentence.* "Test-time per-dimension weights over frozen features, fitted from a few positive and negative examples, were proposed by Wang et al. (2016); this paper changes the feature space and the modality."

<!--block:B0258-->
**3.2 Metric learning from equivalence constraints** (Xing et al., NIPS 2002; RCA, Bar-Hillel et al., JMLR 2005; ITML, Davis et al., ICML 2007; KISSME, Köstinger et al., CVPR 2012). *What they do.* They learn a Mahalanobis metric from pairs labelled similar (and, except RCA, dissimilar). RCA whitens by the covariance within chunklets of equivalent points, in closed form. KISSME sets M = Σ_S⁻¹ − Σ_D⁻¹ from the covariances of pair differences, also in closed form, which makes a metric from a similar-pair statistic minus a dissimilar-pair statistic. *What they report.* RCA improves clustering and classification and can be read as maximum-likelihood estimation of the within-class covariance (abstract); KISSME targets large-scale face verification and person re-identification (we did not re-check its tables). *Difference.* They fit on many pairs offline, all points live in one feature space, and the negative constraints are dissimilar pairs. Our contrast pairs instead agree on a second aspect, so they say which agreement to ignore. Our demonstrations are 4 plus 4 cross-item image-caption pairs given at test time, and the values are disjoint from the query's within the episode. *Reviewer sentence.* "Defining which similarity holds by pairs that are equivalent under it is RCA, and subtracting a contrast-pair statistic from a support-pair statistic in closed form is KISSME; the proposed rule is a diagonal KISSME on factor co-activations."

<!--block:B0259-->
**3.3 Personalized Clustering via Targeted Representation Learning** (Geng et al., AAAI 2025, arXiv 2412.13690). *What it does.* It queries a user with informative must-link and cannot-link pairs and trains a representation with an attention module and a constrained contrastive loss, so that clustering follows the user's criterion instead of the dataset's default one. *What it reports.* On CIFAR10-2 and ImageNet10-2 the personalised target groups images by background where the default groups by object, and CIFAR100-4 uses four super-groups (large animals, small animals, plants, artifacts); with 10k pairwise constraints it beats the constrained and active baselines on the personalised orientation (T3). *Difference.* It uses ten thousand pairs and gradient training, it clusters rather than retrieves, it is image only, and its two orientations are compared at the dataset level, not as a per-query swap. *Reviewer sentence.* "Steering which aspect a representation groups by through pairwise constraints, and testing the same images under two criteria, was already done by Geng et al. (2025)."

<!--block:B0260-->
**3.4 BGE-EN-ICL** (Li et al., arXiv 2409.15700, 2024), with **RICE** (Jedidi et al., arXiv 2609.38099, September 2026). *What they do.* BGE-EN-ICL prepends n query-passage example pairs and a task instruction to the query of a decoder-only embedder and trains with a variable number of examples (0 to n), so that examples define the task at test time. RICE extracts dense representations from an off-the-shelf LLM conditioned on 10 query-document pairs, without training. *What they report.* BGE-EN-ICL: MTEB average 64.67 zero-shot and 66.08 few-shot (public-data setting), AIR-Bench QA nDCG@10 52.93 and 54.36. RICE: BEIR average R@100 0.558 against 0.534 for PromptReps and 0.488 for HyDE (T1). *Difference.* Both are text only; the examples define a relevance task, usually alongside a text instruction, and not an aspect that must transfer to a new value; neither has contrast pairs or a swap. Our sweep of multimodal embedders (MMEB-line papers, 2024 to October 2026) found none that takes demonstrations at encode time: VLM2Vec, GME, Qwen3-VL-Embedding, MuCo and Lens all condition on text instructions. *Reviewer sentence.* "In-context embedders already let a few example pairs define what 'similar' means at test time without retraining; the paper moves this to images."

<!--block:B0261-->
**3.5 MARS: Multimodal Analogical Reasoning over Knowledge Graphs** (Zhang et al., ICLR 2023, arXiv 2210.00312). *What it does.* Given an example pair (e_h, e_t) and a question entity e_q, the model predicts e_a such that (e_q, e_a) stands in the same hidden relation as (e_h, e_t). Entities appear as images or text, in single-modality forms such as (I_h, I_t):(T_q, ?) and blended forms such as (I_h, T_t):(I_q, ?). The candidates are all knowledge-graph entities, and the dataset has 27 relations. *What it reports.* 10,685 training, 1,228 validation and 1,415 test instances; the best model, MarT on MKGformer, reaches Hits@1 0.301 and MRR 0.341. *Difference.* The relation links head and tail (for example part of), not agreement of two items on an aspect. It uses one example pair and no contrast set, models are trained on the same relation inventory, and there is no swap. *Reviewer sentence.* "Cross-modal relation-by-example retrieval, where an example pair defines a hidden relation and the answer is a different entity, is multimodal analogical reasoning (MARS); the task is an analogy whose relation is 'shares the aspect value'."

<!--block:B0262-->
### 4. Recommended novelty statement and must-cite list

<!--block:B0263-->
**Statement we can defend.** "We introduce example-conditioned aspect similarity across modalities. Given a query image or caption and a few support image-caption pairs, each pairing two different items that agree on an unnamed aspect at values other than the query's, together with contrast pairs that agree on a second aspect, a model must rank candidates in the other modality by agreement with the query on the demonstrated aspect, in both directions, and must reverse its ranking when support and contrast are swapped. To our knowledge, prior work learns a notion of similarity from equivalence constraints offline in one feature space [Xing; RCA; ITML; KISSME], adapts feature weights at test time to examples that share the query's value [Wang et al. 2016; Ren et al.; Bongard], lets example pairs define a text retrieval task [BGE-EN-ICL; RICE], or retrieves by a relation shown in one example pair [VISALOGY; VASR; MARS]; none combines test-time, value-disjoint, cross-item image-caption demonstrations with a contrast aspect and a paired swap test."

<!--block:B0264-->
**Claims to avoid.** We must not call new: inferring a notion of similarity from examples (Tversky's diagnosticity principle, contextual visual similarity, Bongard problems); learning it from pairs (RCA, KISSME); test-time per-dimension reweighting (Wang et al. 2016, Category Traversal, Rocchio); example pairs that define a task (BGE-EN-ICL, Promptagator, Painter); emotion-matched cross-modal retrieval (IMEMNet and successors). "Cross-modal analogy" is also taken (MARS).

<!--block:B0265-->
**Must cite.**
<!--block:B0266-->
- Metric learning from pairs: Xing et al. 2002; RCA (Bar-Hillel et al. 2005); ITML (Davis et al. 2007); KISSME (Köstinger et al. 2012). Cite KISSME next to the rule.
- Test-time criterion from examples: Contextual Visual Similarity (Wang et al. 2016); few-shot attribute learning (Ren et al.); Bongard-HOI and Bongard-OpenWorld; Category Traversal Module (Li et al. 2019); Personalized Clustering (Geng et al. 2025).
- Example-conditioned embedders: BGE-EN-ICL; RICE; Promptagator (as "examples define the task"); one universal multimodal embedder with text instructions (Qwen3-VL-Embedding or VLM2Vec) to show that the multimodal line conditions on text.
- Analogy: VISALOGY; VASR; MARS.
- Fixed aspect across modalities: Zhao et al. 2020 (IMEMNet); Stewart et al. 2024; Won et al. 2021; plus Affection and EmotionCLIP from the earlier review.
- Background: Tversky 1977 (similarity depends on the comparison context); CCA cross-modal retrieval (Rasiwasia et al. 2010) if we describe the rule as a diagonal cross-covariance on support pairs.
- From the earlier review: GeneCIS, CSN, SCE-Net, CLAY, CRL, PALAVRA, Rocchio, Tip-Adapter.

<!--block:B0267-->
### 5. Baselines this implies

<!--block:B0268-->
These complement the baselines in §3 of the 2026-10-21 review (prototype, Rocchio, Tip-Adapter, linear probe, text-named conditions, instruction embedders, unsupervised codes). All run on the same frozen features and the same aspect episodes; "needs" lists what goes beyond the episode inputs. Two facts frame them. The aspect spike already ran the raw-feature analogue of our rule (raw CLIP agreement rule, 11.16 pooled R@1 against CLIP only 11.13 on selection rows), and our factor models currently sit at CLIP level on these episodes (SE 11.32). So the pair-metric baselines below are a risk as well as a comparison: if a 4 plus 4 KISSME on raw CLIP beats our factors, the factor space is not the contribution.

<!--block:B0269-->
**Table 2. Baselines from the pair-metric, in-context and analogy lines.**

<!--block:B0270-->
| Baseline | Lineage | What it computes | Needs | Reviewer question |
|---|---|---|---|---|
| **Diagonal or low-rank KISSME from 4 + 4 pairs** | KISSME | d_s = f_I(I_s) − f_T(T_s) for support pairs, d_c for contrast pairs; per dimension m_j = 1/(var_S,j + λ) − 1/(var_C,j + λ), clipped at 0, or a rank-k version from the shrunk covariances; score = −(f(x) − f(y))ᵀ M (f(x) − f(y)) or cos in the M-weighted space, plus β·cos(x, y) | a shared image-text space (CLIP), because pair differences cross modalities; λ, k, β tuned on selection rows | "Is your rule more than a closed-form metric from similar and dissimilar pairs?" |
| **RCA with shrinkage** | RCA | treat each support pair as a two-point chunklet; W = (C_S + λI)⁻¹ from the 8 centred support points; score in the W-whitened space | shared space; λ; optionally contrast chunklets with the sign reversed, which is our own variant and must be described as such | "Does whitening by within-pair variation alone select the aspect?" |
| **Xing-style diagonal metric, fitted per episode** | Xing et al. 2002 | minimise Σ_S ‖d_s‖²_A subject to Σ_C ‖d_c‖_A ≥ 1 over diagonal A ≥ 0, a few projected-gradient steps | step count and regulariser on selection rows | "Did you try fitting a metric from the pairs instead of a fixed rule?" |
| **Per-episode weight fitting on raw features** | Contextual Visual Similarity | non-negative weights w on raw CLIP dimensions, fitted by a few gradient steps so that each support caption scores higher with its paired support image than with contrast images (and the reverse for contrast pairs), with a pull towards uniform weights | a loss and a step budget; about 1 ms per episode at 512 dimensions (we infer) | "Wang et al. 2016 on frozen features, made cross-modal: does a learned factor space beat it?" |
| **Analogy offset** | VISALOGY, MARS, word-vector analogies | score = cos(f(y) − f(x), mean_S(f(T_s) − f(I_s)) − mean_C(f(T_c) − f(I_c))) + β·cos(x, y) | β; nothing else | "This is an analogy task; did you compare with the standard offset?" We expect it to be weak, because the offset is dominated by the modality gap. |
| **Off-label in-context multimodal embedder** | BGE-EN-ICL, applied to a multimodal embedder | encode the query with the support pairs (interleaved image and caption) and the contrast pairs in its instruction context, for example "Pairs in A share one hidden property; pairs in B share another. Represent the query by the property of A", in Qwen3-VL-Embedding-2B or GME-2B; encode candidates without context | interleaved input support (to check per model), about 4 to 5 GB of weights, one context-heavy encode per query and episode | "Would a 2026 embedder given the same examples solve this?" No published multimodal embedder is trained on demonstrations, so we must present this as our construction. |
| **In-context MLLM reranker** | Bongard and MARS VLM baselines | an open MLLM receives support pairs, contrast pairs, the query and the 13 candidates and returns a ranking | 8 to 16 images per prompt, 2 directions per episode; cost limits it to a subsample | upper reference: "Can a general VLM do example-conditioned aspect matching?" |
| **Verbalise, then name** | VisDiff, then the text-named baselines of the earlier review | an MLLM states what the support pairs share and the contrast pairs do not; the phrase feeds the label prompt, CRL projection or instruction embedder | one MLLM call per episode | "Do the examples carry more than an inferred aspect name?" It separates the value of examples from the value of naming. |

<!--block:B0271-->
The first four rows need a shared image-text space, which CLIP provides; with unimodal encoder pairs, pair differences are undefined and only rows 5 to 8 apply. The swap test applies to every row: exchanging S and C must flip the aspect-1 and aspect-2 candidates. Diagonal KISSME passes it by construction, because exchanging the sets negates M before clipping, much as our rule's ReLU(support − contrast) does; RCA ignores C and cannot flip. A swap pass therefore does not separate our rule from KISSME, and that comparison has to rest on R@1 and on the size of the swap margin.

<!--block:B0272-->
### 6. References

<!--block:B0273-->
Links were opened in this session unless marked †, meaning verified in the [2026-10-21 review](2026-10-21_cvpr_literature_review.md). Non-arXiv papers were verified through Crossref, the JMLR page or Semantic Scholar, as noted.

<!--block:B0274-->
**Metric learning from pairs (thread 1)**
<!--block:B0275-->
- Xing, Ng, Jordan, Russell. Distance Metric Learning with Application to Clustering with Side-Information. NIPS 2002. Semantic Scholar record: https://www.semanticscholar.org/paper/d1a2d203733208deda7427c8e20318334193d9d7
- Bar-Hillel, Hertz, Shental, Weinshall. Learning a Mahalanobis Metric from Equivalence Constraints. JMLR 6:937 to 965, 2005. https://www.jmlr.org/papers/v6/bar-hillel05a.html
- Davis, Kulis, Jain, Sra, Dhillon. Information-Theoretic Metric Learning. ICML 2007. https://doi.org/10.1145/1273496.1273523 (Crossref)
- Köstinger, Hirzer, Wohlhart, Roth, Bischof. Large Scale Metric Learning from Equivalence Constraints. CVPR 2012. https://doi.org/10.1109/CVPR.2012.6247939 (Crossref)
- He, Zhang, Wang et al. Cross-Modal Learning via Pairwise Constraints. arXiv 1411.7798. https://arxiv.org/abs/1411.7798
- Jung, Kang, Kwak, Cho. Few-Shot Metric Learning: Online Adaptation of Embedding for Retrieval. arXiv 2211.07116 (venue not verified). https://arxiv.org/abs/2211.07116
- Li, Eigen, Dodge, Zeiler, Wang. Finding Task-Relevant Features for Few-Shot Learning by Category Traversal. CVPR 2019. https://arxiv.org/abs/1905.11116
- Geng, Zhao, Yu, Peng, Du, Chen, Li, Wang. Personalized Clustering via Targeted Representation Learning. AAAI 2025. https://arxiv.org/abs/2412.13690
- Bernard, Ritter, Sessler, Zeppelzauer, Kohlhammer, Fellner. Visual-Interactive Similarity Search for Complex Objects by Example of Soccer Player Analysis. arXiv 1703.03385. https://arxiv.org/abs/1703.03385
- Loeffler, Fallah, Fenu, Zanca, Eskofier, Rozell, Mutschler. Active Learning of Ordinal Embeddings: A User Study on Football Data. TMLR 2023. https://arxiv.org/abs/2207.12710

<!--block:B0276-->
**Example-conditioned embedders (thread 2)**
<!--block:B0277-->
- Li, Qin, Xiao, Chen, Luo, Shao, Lian, Liu. Making Text Embedders Few-Shot Learners (BGE-EN-ICL). arXiv 2409.15700. https://arxiv.org/abs/2409.15700
- Jedidi, Ali, Li, Lin. Effective Dense Retrieval using Only In-Context Examples (RICE). arXiv 2609.38099. https://arxiv.org/abs/2609.38099
- Dai et al. Promptagator: Few-shot Dense Retrieval From 8 Examples. arXiv 2209.11755. https://arxiv.org/abs/2209.11755
- Asai et al. Task-aware Retrieval with Instructions (TART). arXiv 2211.09260. https://arxiv.org/abs/2211.09260
- Morris, Rush. Contextual Document Embeddings. arXiv 2410.02525. https://arxiv.org/abs/2410.02525
- Gu et al. MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model. CVPR 2026. https://arxiv.org/abs/2602.06393
- Liu et al. Lens: Bringing the Right Semantic Perspective into Focus for Training-Free Multimodal Representation Learning. arXiv 2609.20252. https://arxiv.org/abs/2609.20252
- Hu et al. Vela: Scalable Embeddings with Voice Large Language Models for Multimodal Retrieval. Interspeech 2025. https://arxiv.org/abs/2506.14445
- Qwen3-VL-Embedding, VLM2Vec, GME†.

<!--block:B0278-->
**Analogy and relation by example (thread 3)**
<!--block:B0279-->
- Sadeghi, Zitnick, Farhadi. VISALOGY: Answering Visual Analogy Questions. NIPS 2015. https://arxiv.org/abs/1510.08973
- Bitton, Yosef, Strugo, Shahaf, Schwartz, Stanovsky. VASR: Visual Analogies of Situation Recognition. AAAI 2023. https://arxiv.org/abs/2212.04542
- Zhang, Li, Chen, Liang, Deng, Chen. Multimodal Analogical Reasoning over Knowledge Graphs (MARS). ICLR 2023. https://arxiv.org/abs/2210.00312
- Nguyen, Mo, Singh, Wang, Shi, Kolkin, Shechtman, Lee, Li. Relational Visual Similarity. CVPR 2026. https://arxiv.org/abs/2512.07833
- Nguyen, Li, Ojha, Lee. Visual Instruction Inversion: Image Editing via Visual Prompting. arXiv 2307.14331. https://arxiv.org/abs/2307.14331
- Bar, Gandelsman, Darrell, Globerson, Efros. Visual Prompting via Image Inpainting. arXiv 2209.00647. https://arxiv.org/abs/2209.00647
- Wang, Wang, Cao, Shen, Huang. Images Speak in Images: A Generalist Painter for In-Context Visual Learning. CVPR 2023. https://arxiv.org/abs/2212.02499

<!--block:B0280-->
**Criterion defined by examples (thread 4)**
<!--block:B0281-->
- Wang, Kitani, Hebert. Contextual Visual Similarity. arXiv 1612.02534, 2016. https://arxiv.org/abs/1612.02534 (full text read through https://ar5iv.labs.arxiv.org/html/1612.02534)
- Tversky. Features of Similarity. Psychological Review 84(4), 1977. https://doi.org/10.1037/0033-295X.84.4.327 (Crossref)
- Jiang et al. Bongard-HOI: Benchmarking Few-Shot Visual Reasoning for Human-Object Interactions. CVPR 2022. https://arxiv.org/abs/2205.13803
- Wu et al. Bongard-OpenWorld: Few-Shot Reasoning for Free-form Visual Concepts in the Real World. ICLR 2024. https://arxiv.org/abs/2310.10207
- Raghuraman, Harley, Guibas. Support-Set Context Matters for Bongard Problems. TMLR 2024. https://arxiv.org/abs/2309.03468
- Stretcu et al. Agile Modeling: From Concept to Classifier in Minutes. arXiv 2302.12948. https://arxiv.org/abs/2302.12948
- Dunlap et al. Describing Differences in Image Sets with Natural Language (VisDiff). CVPR 2024. https://arxiv.org/abs/2312.02974
- Ye, Shi, Zhan. Identifying Ambiguous Similarity Conditions via Semantic Matching (DiscoverNet). CVPR 2022. https://arxiv.org/abs/2204.04053
- Ren et al., few-shot attribute learning, arXiv 2012.05895†; GeneCIS, CSN, SCE-Net, CLAY, CRL, InDiReCT, PALAVRA, POLAR, FSIR, CLIP-Branches, Rocchio, Tip-Adapter, LP++, COCO-Facet†.

<!--block:B0282-->
**Fixed aspect across modalities (thread 5)**
<!--block:B0283-->
- Zhao, Li, Yao, Nie, Xu, Yang, Keutzer. Emotion-Based End-to-End Matching Between Image and Music in Valence-Arousal Space. ACM MM 2020. https://arxiv.org/abs/2009.05103
- Stewart, Avramidis, Feng, Narayanan. Emotion-Aligned Contrastive Learning Between Images and Music. ICASSP 2024. https://arxiv.org/abs/2308.12610
- Choi, Kim, Kang. MMVA: Multimodal Matching Based on Valence and Arousal across Images, Music, and Musical Captions. AAAI 2025 AI for Music workshop. https://arxiv.org/abs/2501.01094
- Won, Salamon, Bryan, Mysore, Serra. Emotion Embedding Spaces for Matching Music to Stories. ISMIR 2021. https://arxiv.org/abs/2111.13468
- Song, Soleymani. Cross-Modal Retrieval with Implicit Concept Association. arXiv 1804.04318. https://arxiv.org/abs/1804.04318
- Liu, Fu, Kato, Yoshikawa. Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training. ACM MM 2018. https://arxiv.org/abs/1804.08473
- Rasiwasia et al. A New Approach to Cross-Modal Multimedia Retrieval. ACM MM 2010. https://doi.org/10.1145/1873951.1873987 (Crossref)
- Affection, EmotionCLIP, GOYA, CSD†.

<!--block:B0284-->
**Searched, nothing found.** arXiv abstract queries for in-context or demonstration-conditioned multimodal embedders (four query forms, 2024 to October 2026), for "conditional similarity" with few-shot, exemplar, meta-learning or test-time terms, for "user-defined similarity" and "similarity by example", for cross-modal metric learning from pairwise constraints, and for composed retrieval conditioned by an example pair returned no paper that conditions image↔text similarity on example pairs. This is a negative result from an incomplete search (no web search engine, Semantic Scholar rate-limited, no Google Scholar), so the paper should keep "to our knowledge".


<!--block:B0285-->
---

<!--block:B0286-->
# Appendix (evidence report): 2026-10-25_backbone_check

<!--block:B0287-->
## CoSiR v2: which frozen backbone joins CLIP ViT-B/32? (backbone check)

<!--block:B0288-->
### Verdict

<!--block:B0289-->
**The weaker-modality probes barely move across four backbones, so the modality asymmetry is a property of the data,
not of the encoder.** On ArtELingo, emotion read from images scores 35.1 (CLIP ViT-B/32), 36.3 (SigLIP 2), 36.6
(PE-Core) and 36.4 (Qwen3-VL-Embedding-2B), and style read from captions scores 25.4, 25.9, 26.3 and 26.1. The spread is
1.5 and 0.9 points. The strong-side probes move much more: emotion from captions rises from 56.9 to 62.5 (Qwen), style
from images from 60.9 to 71.9 (PE). This supports our analysis claim C3 (a cross-modal aspect match is capped by the
weaker modality). On the criterion fixed in advance, the cross-modal label ceiling, the three strong backbones are
within 0.5 points of each other (sum of the emotion and style ceilings: Qwen 50.99, PE 50.49, SigLIP 2 50.49) and all
sit about 4 points above CLIP (46.42). The ceiling does not separate them, so the choice rests on the trade-offs
below. We did not choose here; on 2026-10-02 the user picked Qwen3-VL-Embedding-2B, mainly because the aspect named in its instruction gives an examples versus names baseline on the same backbone, subject to a fidelity check of our reimplementation against the official code. This is a throwaway diagnostic on selection rows, a follow-up to the
[aspect-episode spike](2026-10-23_aspect_episode_spike.md).

<!--block:B0290-->
### What we tested

<!--block:B0291-->
**Question.** Which frozen backbone sits beside CLIP ViT-B/32 in the final tables? The criterion fixed in advance was
the *label-probe ceiling* on ArtELingo aspect episodes (R@1 % when the factor coordinates are replaced by logistic
probes fitted with the human labels; probes use 60,000 scorer-train rows, episodes use selection rows, val and held
were never read), plus attribute probes on CUB. Baseline of every number: CLIP ViT-B/32 through the same pipeline.

<!--block:B0292-->
**Setup.** Four frozen backbones: CLIP B/32, SigLIP 2 So400m/14-384, PE-Core L/14-336 and Qwen3-VL-Embedding-2B.
Features were extracted for 37,738 selection paintings with 92,413 captions (ArtELingo) and all 11,788 images with
117,880 captions (CUB). A *probe* is a logistic regression (C=1.0, max_iter 300) on L2-normalised features, scored as
accuracy. The *cross-modal ceiling* is aspect R@1 with the probe-derived scorer, averaged over the i2t and t2i
directions. *Backbone-only aspect R@1* is the cosine of anchor and candidate, mean of the four aspect and direction
cells (chance 7.69%).

<!--block:B0293-->
### Results

<!--block:B0294-->
![Backbone check](../../assets/2026-10-25_backbone_check/backbone_check.png)

<!--block:B0295-->
*Figure 1. (a) ArtELingo probe accuracy (%) per backbone. Light bars are the weaker modality for each aspect
(emotion from images, style from captions), dark bars the stronger one. Dotted line: emotion majority class, 31.8
(from the spike). (b) Cross-modal label ceilings for emotion and style (R@1 %, mean of i2t and t2i), with the
backbone-only aspect R@1 as a diamond. Dotted line: chance, 7.69.*

<!--block:B0296-->
#### Result 1: weak-side probes are flat, strong-side probes move (claim C3)

<!--block:B0297-->
| Probe accuracy (%) | CLIP | SigLIP 2 | PE | Qwen | Spread |
|---|---:|---:|---:|---:|---:|
| Emotion from images (weak) | 35.1 | 36.3 | 36.6 | 36.4 | 1.5 |
| Style from captions (weak) | 25.4 | 25.9 | 26.3 | 26.1 | 0.9 |
| Emotion from captions (strong) | 56.9 | 58.7 | 58.9 | 62.5 | 5.6 |
| Style from images (strong) | 60.9 | 70.7 | 71.9 | 64.2 | 11.0 |

<!--block:B0298-->
Reading. Four different encoders and training recipes add about one point to the weak side. The weak-side emotion probe sits 3 to 5 points above the 31.8 majority class and does not
leave that band under any backbone. The strong side gains up to 5.6 (emotion) and 11.0 (style). So the weak modality
carries little of the aspect in the data itself, and a better encoder cannot read what the pixels or words do not
hold. Caveat: the weak-side labels are noisy (an image takes the emotion of its row), which the earlier spike already
noted.

<!--block:B0299-->
#### Result 2: ceilings separate CLIP from the rest, not the rest from each other

<!--block:B0300-->
| (%) | CLIP | SigLIP 2 | PE | Qwen |
|---|---:|---:|---:|---:|
| Emotion cross-modal ceiling | 24.16 | 26.17 | 25.72 | 27.37 |
| Style cross-modal ceiling | 22.27 | 24.32 | 24.77 | 23.62 |
| Sum | 46.42 | 50.49 | 50.49 | 50.99 |
| Backbone-only aspect R@1, pooled | 11.13 | 10.45 | 10.94 | 11.93 |

<!--block:B0301-->
The sum for CLIP is 46.42 on recomputation from the JSON (the controller's note had 46.43). Qwen has the best emotion
ceiling (+3.2 over CLIP), PE the best style ceiling (+2.5). Backbone-only aspect R@1 stays in a 10.45 to 11.93 band,
7.69 chance, so the headroom of about 12 to 16 points above it is the same for every backbone. Qwen's pooled 11.93 is
+0.8 over CLIP; SigLIP 2 is 0.7 below CLIP.

<!--block:B0302-->
#### Result 3: CUB attribute probes and retrieval

<!--block:B0303-->
Probe accuracy (%), image / caption, test split, label = the single attribute value with certainty at least 3.

<!--block:B0304-->
| Group (classes, majority) | CLIP | SigLIP 2 | PE | Qwen |
|---|---|---|---|---|
| Primary colour (15, 21.0) | 61.3 / 62.6 | 61.9 / 64.5 | 62.5 / 64.2 | 64.3 / 64.3 |
| Bill shape (9, 41.4) | 60.8 / 50.2 | 67.3 / 55.3 | 67.6 / 52.9 | 66.3 / 55.5 |
| Size (5, 50.7) | 62.2 / 57.8 | 62.5 / 60.2 | 63.2 / 59.3 | 62.4 / 60.0 |

<!--block:B0305-->
- **Primary colour is symmetric.** Image and caption probes are within 2.3 points under every backbone, three times the
  majority rate. This is the first symmetric aspect we have measured, against ArtELingo's gaps of 21 to 35 points.
- **Bill shape leans to images** (images 66.3 to 67.6 against captions 52.9 to 55.5 for the three new backbones; CLIP
  60.8 against 50.2). The majority is 41.4.
- **Size is a poor aspect.** Every cell is within 13 points of the 50.7 majority and the best caption cell is 60.2, so
  neither modality carries it well.

<!--block:B0306-->
Retrieval sanity check, 5,794 test images, first caption each, R@1 %:

<!--block:B0307-->
| | CLIP | SigLIP 2 | PE | Qwen |
|---|---:|---:|---:|---:|
| i2t | 0.88 | 2.26 | 1.90 | 1.12 |
| t2i | 0.64 | 1.38 | 1.48 | 1.02 |

<!--block:B0308-->
Instance retrieval is near zero everywhere probably because the 5,794 test images hold about 29 photos per species (5,794 over 200 classes), so an exact
image is hard to pick out (untested). Because of that, the controller also checked species-level alignment (t2i, a hit is
the query's own image or an image of the same species; species chance 0.52%): CLIP 6.71, SigLIP 2 15.27, PE 12.72,
Qwen 10.23 (instance: 0.64, 1.38, 1.48, 1.02). All four are well above chance, so the features are aligned and the
extraction is sane, with SigLIP 2 best.

<!--block:B0309-->
#### Result 4: CLIP reproduction

<!--block:B0310-->
The CLIP aspect R@1 values (9.42, 10.86, 10.77, 13.45) match the spike exactly. The probes and ceilings drift by up to
0.25 points (style image probe 60.92 against 60.8; emotion cross ceilings 23.78 and 24.54 against 23.58 and 24.29).
Rerunning the unmodified `aspect_ceiling.py` gives the new numbers, so the drift is not in this pipeline. The likely
cause, untested, is unconverged lbfgs (max_iter 300) that is not bit-reproducible under a different thread count
(OMP 8 now). Differences between backbones of 0.5 points on the ceiling sum are therefore close to this noise, which is
one more reason not to rank the three strong backbones on it. Paper-grade ceilings should use converged probes and a
fixed thread count.

<!--block:B0311-->
### Implementation notes and cost

<!--block:B0312-->
- PE-Core was loaded through open_clip from `hf-hub:timm/PE-Core-L-14-336` (text context 32). open_clip_torch, timm,
  ftfy and regex went in with `pip --target` (no deps); the CoSiR env is untouched.
- SigLIP 2 used transformers `get_*_features`, lower-cased text and padding to 64, bf16.
- **Qwen3-VL-Embedding-2B was reimplemented, not run with the official script,** on transformers 5.6.2: system prompt
  "Represent the user's input.", last-token pooling, L2 norm, bf16, image `max_pixels` capped at 512·32·32 (default
  about 1.31M). It must be validated against the official implementation before paper use.
- GPU extraction times (s, RTX 3090, 8 loader workers):

<!--block:B0313-->
| Model | ArtELingo images | ArtELingo captions | CUB images | CUB captions | Total |
|---|---:|---:|---:|---:|---:|
| CLIP | cached | cached | 7 | 15 | 22 (CUB only) |
| SigLIP 2 | 558 | 99 | 184 | 131 | 972 |
| PE-Core | 361 | 44 | 113 | 56 | 574 |
| Qwen | 2,691 | 268 | 390 | 340 | 3,689 |

<!--block:B0314-->
Qwen takes 3.8 times SigLIP 2 and 6.4 times PE in total (4.5 and 7.3 times on ArtELingo alone). The whole run took
about 62 minutes of GPU time.

<!--block:B0315-->
### What this means

<!--block:B0316-->
We do not pick a backbone here. The trade-off:

<!--block:B0317-->
- **Qwen3-VL-Embedding-2B:** best emotion ceiling (27.37) and best backbone-only aspect R@1 (11.93). It is an
  instruction-following embedder, so the baseline "aspect named in the instruction" can run on the same backbone. It
  is 4 to 7 times slower to extract and needs a fidelity check against the official implementation first.
- **PE-Core L/14-336:** best style ceiling (24.77) and best image-side style probe (71.9), fast, and a plain dual
  encoder.
- **SigLIP 2:** best CUB species retrieval (15.27 against CLIP 6.71), mid-cost, and the same ceiling sum as PE.
- For claim C3 the choice does not matter: the weak-side probes are the same under all four.

<!--block:B0318-->
### Caveats

<!--block:B0319-->
- Diagnostic on selection rows, one seed per backbone, no confidence intervals. The 0.5 point ceiling differences are
  within the probe drift of Result 4.
- The emotion majority (31.8) comes from the earlier spike; we did not recompute majority rates for the style probes.
- CUB probes use a single-value label subset (2,431 to 5,266 test images), and attribute labels are crowd-sourced.
- The species alignment numbers come from a controller check run outside the logged script, not from the JSON.

<!--block:B0320-->
### Files

<!--block:B0321-->
- Scripts: `src/test/20261025_backbone_check/run_backbone_check.py`, `extract.py`. Log:
  `src/test/20261025_backbone_check/20261025_backbone_check_log.md`.
- Gitignored, local only: `src/test/20261025_backbone_check/results/backbone_check.json`, `timings.json`; features in
  `/data/SSD2/pre_extract/backbone_check/`.
- Figure: `docs/reports/assets/2026-10-25_backbone_check/backbone_check.png`, built by
  `docs/reports/assets/build_2026-10-25_backbone_check_figures.py`.
- Previous step: [aspect-episode spike](2026-10-23_aspect_episode_spike.md).
