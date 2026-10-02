# CoSiR v2: CVPR publication plan (design)

**Date:** 2026-10-02
**Status:** sections approved by the user in brainstorming; this written spec awaits the user's review.
**Replaces:** the archived conditional-buddies plan (`docs/archive/buddy_publication_plan/`) as the project's
publication target.

**In one paragraph.**
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

![Timeline of the plan](assets/2026-10-02_cvpr_plan_timeline.png)

*Figure 1. Experiments E0 to E17 (§11) by phase, with decision points (dashed lines) and hard deadlines (solid lines).
Built by `assets/build_2026-10-02_cvpr_plan_timeline.py`; its rows mirror the §11 table.*

Terms in *italics* at first use are defined in the glossary (Appendix A).

---

## 1. Goal, venue and resources

**Goal.** Publish CoSiR v2 at CVPR. The user's four requirements for the paper are:
1. strong results;
2. a good storyline;
3. clear baselines on comparable benchmarks;
4. a precise problem definition with a clear method contribution and an explanation of why it works.

**Dates** (from the user).

| Milestone | Date |
|---|---|
| Abstract registration | Tue 2026-11-10 |
| Paper | Mon 2026-11-16 |
| Supplementary material | Mon 2026-11-23 |

**Compute.**
- **DAS6 GPU cluster.** The node and GPU count depend on what the user reserves: normally one node with up to 3
  GPUs, and 3 to 6 nodes when the cluster is idle.
  - Jobs go through the `cluster-run` CLI, and all node-side files live under `/local/wding/`.
- **Local RTX 3090.** It is shared with other Claude sessions and used for smoke tests and small runs. Every GPU job
  takes the shared lock `flock -n -o -E 75 /tmp/gpu0.lock`.
- **Long jobs** are launched by the main session in the background.

## 2. Background: what CoSiR is and how we got here

### 2.1 The research question

CoSiR studies **conditional image–text similarity**. Whether an image and a caption "match" should depend on a
*condition*, i.e. on what the user cares about.
- The direct ancestor is GeneCIS (Vaze, Carion and Misra, CVPR 2023). It ranks images for a reference image under a
  short text condition such as "colour". CoSiR's original `combiner.py` credits it.
- CoSiR set out to differ in two ways:
  1. **Conditions not drawn from a hand-designed list.** This is weaker now (§2.4).
  2. **Genuinely cross-modal similarity:** image against caption, rather than image against image.

### 2.2 Three lines of work

| Line | Period | What it was | Outcome |
|---|---|---|---|
| buddy | Jun to Sep 15 | per-sample condition vectors and a combiner on a "buddy graph" (mutual nearest neighbours across modalities) | archived plan; the condition mechanism stayed weak and asymmetric |
| percept | Sep 15 to 28 | ArtELingo affect work; a buddy-graph replacement for PercepT's (arXiv 2606.03345) topic stage | buddy topics matched PercepT on held-out label agreement; a matched head-to-head showed no PercepT lead. A side branch that showed the buddy graph works on an existing benchmark |
| **v2** | from Sep 28 | a ground-up rewrite, block by block (spec `2026-09-28-cosir-v2-ground-up-redesign.md`) | the subject of this plan |

### 2.3 CoSiR v2 as it stood on the morning of 2026-10-02

**Data: ArtELingo** (English part, `/data/PDD/artelingo/artelingo_train.json`).
- WikiArt paintings with ArtEmis-style annotations. Each annotation is one viewer's *emotion* label plus a caption
  explaining it ("the dark colours make me feel sad").
- 308,723 rows over 61,402 paintings. A *row* is one image–caption pair (the painting and one caption). Every
  painting also has one *art style* (Impressionism, Baroque and so on).
- We evaluate 8 emotions (the catch-all "something else" is excluded) and the art styles with at least 30
  paintings.

**Splits.** All splits are grouped by painting, so no painting crosses a split.

| Split | Rows | Paintings | Role |
|---|---:|---:|---|
| *scorer-train* | 183,694 | 36,518 | training factor models |
| *selection* | 32,413 | 6,451 | development; read many times |
| val | 30,872 | | unused in recent work |
| *held* | 61,744 | 12,281 | final tests only; read 3 times so far |

(Scorer-train and selection together form the 216,107-row train part.)

**Features.** Frozen CLIP ViT-B/32 image and caption embeddings, cached once. CLIP is never fine-tuned.

**Model (Candidate A).**
- Two small encoders map a row's image feature and caption feature into one shared space of 32 non-negative
  *factors*. An item's vector there is its *code*.
- A *condition* is given by 4 *support* pairs and 4 *contrast* pairs.
- The parameter-free *naive rule* turns them into factor weights:
  `w = ReLU(mean support pair code − mean contrast pair code)`, L1-normalized. A pair code is the mean of a row's
  image code and caption code.
- A query q and a candidate c (opposite modalities) score `0.3·cos(CLIP_q, CLIP_c) + Σ_l w_l q_l c_l`.

**Factor recipes.**
- **R0:** the first recipe. It collapsed to about one effective dimension through a cosine agreement loss.
- **R3:** the repair, with an InfoNCE pair loss plus decorrelation.
- **C0:** R3's recipe refit on scorer-train rows; the matched control.
- **SE:** C0 plus *condition episodes*. Each episode's condition comes, with probability ½ each, from 64 k-means
  clusters of a text emotion classifier's outputs on the captions (GoEmotions RoBERTa,
  `SamLowe/roberta-base-go_emotions`, 28 emotion probabilities) or from 64 k-means clusters of CLIP image features.
  Because the classifier is supervised, SE is *distantly supervised*, not label-free.

**Evaluation (label episodes).**
- An *anchor* (the query) has a target label: one emotion or one style.
- The 4 supports carry that label; the 4 contrasts do not.
- There are 13 *candidates*: one positive with the label, and 12 from paintings never given it.
- Metric: *R@1*, the share of episodes where the positive ranks first (chance 7.69%). Both directions are scored:
  *i2t* (image query, captions ranked) and *t2i* (caption query, images ranked).
- Held results (pooled R@1): CLIP only 13.16, R3 19.45, C0 19.88, SE 21.24. SE's emotion gain over C0 (+2.08) held
  at three seeds. A trained condition scorer did not beat the naive rule (stage (d)).

### 2.4 What changed on 2026-10-02

Six investigations, all on selection rows (held rows untouched), changed the plan.

| Report | Finding | Consequence |
|---|---|---|
| [CVPR literature review](../../reports/auto/v2/2026-10-21_cvpr_literature_review.md) | No prior example-conditioned cross-item image–text similarity (to our knowledge). The naive rule is Rocchio relevance feedback (1971), and the weighted factor score is a Conditional Similarity Network (CSN, CVPR 2017) mask. Learning conditions without labels is taken (SCE-Net ICCV 2019, DiscoverNet CVPR 2022), and so is distant affect supervision (EmotionCLIP CVPR 2023) | drop "unsupervised condition discovery"; claim "no labels from the evaluation taxonomy". Also, PercepT uses a different emotion model (ModernBERT) and CLIP ViT-L/14, so our teacher must be named exactly |
| [support-baseline spike](../../reports/auto/v2/2026-10-22_support_baseline_spike.md) | A logistic probe fit on the 4+4 examples in raw CLIP space beats SE (24.10 vs 21.22). A prototype of the supports that **ignores the query** reaches 22.83; adding the query adds only +0.57 | label episodes test few-shot recognition of a *value*, not conditional similarity. The problem has to change |
| [aspect-episode spike](../../reports/auto/v2/2026-10-23_aspect_episode_spike.md) | On *aspect episodes* (§3), no factor model beats CLIP (SE 11.32 vs 11.13). Probes trained with labels reach about 23 (the *ceiling*). Emotion lives in captions and style in images | the task is learnable, but the factors must be trained for it. ArtELingo caps cross-modal matching through its weaker modality |
| [aspect novelty check](../../reports/auto/v2/2026-10-24_aspect_task_novelty_check.md) | No paper defines the aspect task, but every ingredient exists. Closest: metric learning from pairs (Xing 2002, RCA 2005, KISSME 2012), Contextual Visual Similarity (arXiv 1612.02534), MARS (ICLR 2023), in-context text embedders | the novelty is the combination. Expected reviewer line: "a few-shot cross-modal diagonal KISSME" |
| [backbone check](../../reports/auto/v2/2026-10-25_backbone_check.md) | Across four frozen encoders, the weak-side probes barely move (emotion from images 35 to 37, style from captions 25 to 26). CUB's primary colour is read equally from images and captions | the asymmetry is in the data (claim C3). The user chose Qwen3-VL-Embedding-2B as the second backbone |
| [GeneCIS feasibility](../../reports/auto/v2/2026-10-20_genecis_feasibility.md) (Oct 1) | GeneCIS is usable for evaluation; its focus-attribute task names an aspect | GeneCIS focus attribute joins the benchmarks |

**Value versus aspect, by example.**
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

## 3. Problem definition (approved)

**Example-conditioned aspect similarity across modalities.**
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

**Against GeneCIS.** GeneCIS focus conditions name the aspect in text and compare image with image. Ours shows the
aspect by examples and compares image with caption.
- GeneCIS *focus attribute* (condition = an attribute type such as "colour") is the closest standard benchmark.
- *Focus object* conditions on a named object, which is a value, so it is supplementary only.

## 4. Contributions and claims (approved)

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

| # | Claim | Evidence | Status |
|---|---|---|---|
| K1 | Value episodes are solved by the supports alone | prototype vs model, query ablation | **done** (support spike) |
| K2 | Our method beats backbone-only, raw-feature metric-from-pairs baselines and the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset | pre-registered final tests (§10) | open; factors are at CLIP level today |
| K3 | Examples beat naming the aspect on subjective aspects and match it on objective ones | CRL, Qwen with the aspect in its instruction, privileged names | open |
| K4 | Gains hold on a strong backbone (Qwen3-VL-Embedding-2B) | second backbone | open |
| K5 | GeneCIS focus attribute: competitive with the published frozen-B/32 rows (example and text protocols reported apart) | GeneCIS runs | open |
| K6 | The rule selects aspect factors, and the factors are shared across modalities | ablations, swap analysis, per-factor analysis | open |
| K7 | The gain comes from the *learned* basis | same rule on raw, PCA/NMF/SpLiCE and learned factors | open |

**Fallback.** If K2 fails, K1, K3 and C3 could carry a task, benchmark and analysis paper but not the method paper.
The switch is discussed with the user at the Oct 9 decision (§11).

## 5. Benchmarks and protocols (approved; genre open)

### 5.1 Common protocol

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

### 5.2 Datasets, in priority order

| Dataset | Aspects | Captions | Test split | Notes |
|---|---|---|---|---|
| **ArtELingo** (primary) | emotion (8) × style (23) × **genre** (open, §5.3) | human, affective | held rows, fresh-seed aspect episodes | asymmetric aspects; label-probe ceiling about 23 R@1 |
| **CUB-200-2011 + Reed et al. captions** (bird photos) | primary colour (15), bill shape (9), and a third attribute group chosen in E0 | human, 10 per image, written without species names | the standard zero-shot split's **50 unseen species**; development on 30 of the 150 training species, kept out of factor training | colour is symmetric across modalities; captions name colours |
| **GeneCIS focus attribute** | condition = attribute type | none (Visual Genome object crops) | the benchmark (2,000 templates) | image to image; see §5.4 |
| **SemArt** (paintings with catalogue descriptions) | type (10), school (26), timeframe (22) | catalogue text, artist names and dates scrubbed | official test (1,069 paintings); development = official val | main paper if on time, otherwise supplementary |
| GeneCIS focus object | condition = an object (value-type) | COCO | benchmark | supplementary only |
| Affection | | | | potential; the user will check |

**Data on disk** (`/data/SSD/`; downloaded 2026-10-02 by `scripts/download_benchmarks.sh`):
- `cub/` (11,788 images; captions in `captions/extracted/text_c10`);
- `semart/SemArt` (21,384 images);
- `visual_genome/VG_100K_all` (108,249 images, including all 23,640 GeneCIS attribute images);
- the GeneCIS COCO half, preprocessed earlier into `/data/PDD/genecis/`.

Our loaders pass the Visual Genome path themselves; the GeneCIS clone is not edited.

### 5.3 Open item: ArtELingo genre

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

### 5.4 GeneCIS protocols

1. **Example protocol (ours).** Image to image on image codes, with supports and contrasts drawn from other
   templates that share the condition. It is not comparable with published numbers.
2. **Text protocol (stretch).** A phrase-to-weights adapter `w = a_T(phrase)`. It is needed for a row beside the
   published frozen ViT-B/32 results: SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1 (OSrCIR's independent
   reproduction gave 14.0).

Always reported: image only, text only and image + text on our backbones. GeneCIS's CC3M training triplets are never
used.

## 6. Method A: aspect-trained factors (approved)

**Model.** The v2 structure is kept.
- Two small encoders map frozen backbone features of each modality into a shared sparse non-negative space R₊^L
  (L = 32; 64 is a grid point).
- Score: s = β·cos + Σ_l w_l a_I,l(x) a_T,l(y).
- The *agreement rule* replaces the naive rule, because the examples are now pairs that agree on an aspect:
  w = ReLU(mean over S of a_I(x_i) ⊙ a_T(y_i) − mean over C of a_I(x_j) ⊙ a_T(y_j)), L1-normalized.
  It keeps the factors on which support pairs co-activate more than contrast pairs. It is training-free at test
  time.

**Training: pseudo-aspect episodes.**
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

| Dataset | Pseudo-partitions (no evaluation labels) |
|---|---|
| ArtELingo | GoEmotions affect k-means of captions (emotion-like; distant supervision); backbone image k-means (style-like; adjusted mutual information 0.32 with style); caption-content k-means (genre-like) |
| CUB (150 training species) | backbone image k-means; **per-sentence** caption k-means (Reed's captions focus on different parts, such as "red crown" or "short beak") |
| SemArt | image k-means; description k-means |
| GeneCIS | a factor model trained on COCO train2014 pairs (none are GeneCIS images), with image and caption k-means |

**Go/no-go.** Pre-registered before the first run; ArtELingo, CLIP B/32, selection rows; decided Fri Oct 9.
- **Grid:** at most about 10 runs (partition set × L × loss weights), about 10 minutes each locally.
- **Picking:** the best run is picked on seed-42 selection aspect episodes. The GO test uses **fresh seed-43
  episodes** from the same rows, so picking the best of about ten runs does not inflate the result.
- **GO** if, on those fresh episodes, pooled aspect R@1 has a 95% CI lower bound above 0 against **both**
  backbone-only (11.1) **and** the best raw-feature metric-from-pairs baseline (§8), with that baseline's fusion
  weight cross-fitted. Swap success is reported at matched R@1.
- **Strong GO:** pooled R@1 of about 15 or more, i.e. a third of the way to the ceiling of about 23.
- **NO-GO:** stop the method line and discuss the fallback framing with the user.
- **Replication:** a winner is re-run at seeds 43 and 44 before it moves to other datasets and Qwen.

**Open risk R-pseudo** (flagged by the user, accepted for now): the pseudo-partitions are proxies, and the model may
learn the episode format rather than aspects. The cross-dataset and unseen-species results are the test.

## 7. Backbones (approved)

- **Development** on CLIP ViT-B/32 (cached; runs take minutes).
- **Final tables** also on **Qwen3-VL-Embedding-2B**, the user's choice. It is a multimodal embedder built on a
  vision–language model that takes a task instruction at encoding time. That lets the "aspect named in the
  instruction" baseline run on the very same backbone.
- **Fidelity check first.** Our reimplementation (`src/test/20261025_backbone_check/extract.py`: transformers
  5.6.2, last-token pooling, default instruction, capped image resolution) must be checked against the official
  implementation before paper use. If it fails, fall back to PE-Core L/14.
- **Packages** go in `pip --target` directories, never into the CoSiR env.

## 8. Baselines (approved)

All baselines run on the same features and episodes.
- **On development splits,** fusion weights (and β) are cross-fitted: chosen on one half of the anchors (split by
  parity) and applied to the other, then swapped.
- **For final reads,** they are chosen on the development split and frozen. GeneCIS, which has no development
  split, takes them from the other datasets.

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

**Out of the tables.**
- **Teacher-only:** GoEmotions cannot read images, so it cannot score cross-modal pairs. It becomes a text-side
  analysis inside C3.
- **PercepT** was a side branch that showed the buddy graph works on an existing benchmark and inspired v2's design.
  A buddy-vs-PercepT comparison is **optional and storyline-dependent** (supplement or analysis), never required.

## 9. Out of scope

- Fine-tuning the backbone.
- Evaluation labels in factor training.
- GeneCIS CC3M triplets.
- Merging percept-branch code.
- Codex (unless the user asks).
- Any held read outside §10.

## 10. Held budget and statistics (approved)

**Ledger.** Every read of a final split is logged in `docs/superpowers/held_ledger.md`: date, script SHA-256,
episode SHA-256 and purpose. Each final script refuses a second run.

| Dataset | Development | Final test | Budget |
|---|---|---|---|
| ArtELingo | selection rows (free to reuse) | held rows, fresh-seed aspect episodes (a new task on rows read 3 times before for value episodes) | 1 main + 1 reserve |
| CUB | 30 of the 150 training species | 50 unseen species | 1 + 1 |
| SemArt | official val | official test | 1 + 1 |
| GeneCIS | none (zero-shot; selection on the other datasets) | benchmark | 1 + 1 |

- **Main read:** one pre-registered run per dataset, covering all models and baselines on both backbones, with
  checkpoints, fusion weights, β and episode counts frozen.
- **Reserve read:** only for a pre-registered fix after a final-review finding, never for a second attempt at a
  better number.
- **Disclosure:** ArtELingo's held rows shaped earlier design decisions; the paper says so.

**Statistics.**
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

## 11. Experiment plan and schedule (approved)

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

Every experiment ends with a report in `docs/reports/auto/v2/`, one row in `reports_sum.md`, a commit, and an update
of §4's claims table.

**Planning scope.**
- The first implementation plan covers **E0 to E5** in task-level detail.
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
| R-qwen | our reimplementation differs from the official one | E0 fidelity check; PE-Core fallback |
| R-infra | local GPU loss (happened 2026-10-02); a shared machine | DAS6 as backup; GPU lock; commit after every task |
| R-held | ArtELingo held rows shaped earlier design | fresh-seed episodes on a new task; disclosure |
| R-licence | CUB terms conflict (non-commercial vs cc-by); SemArt and GeneCIS are CC BY-NC | academic use; state the terms in the paper |
| R-concurrent | TPIPS, SteerViT, CLAY, Fioresi et al., COCO-Facet condition similarity on text | position on the example interface and the cross-modal score |

## 13. Conventions

- Implementation goes to Claude Code subagents sized to the task; the controller reviews each result.
- Reports follow the `docs/reports/` layout and the user's report rules: a real baseline beside every number,
  figures, paper-draft style, no dashes.
- Other sessions share main: stage files by explicit path.
- Experiment folders are `src/test/<sequence-date>_<name>/`. Edits to existing source get a change log in
  `.claude/yyyymmdd_log.md`.

---

## Appendix A. Glossary

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

## Appendix B. Sources

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
