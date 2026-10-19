# Aspect graph and multi-head sweep: design

> Date: 2026-10-18 16:40 · Status: draft · Kind: experiment
> Builds on: [purification pilot](../../reports/auto/percept/2026-12-10_purified_buddy_pilot.md), the grill of
> [decide 6 handoff](../handoffs/2026-10-18-r10-decide-6-handoff.md) §2 · Constitution: `docs/superpowers/constitution.md`
> Decision rule: none (descriptive; every stage is an exploration)
> Stage 2 (model sheet) moved on 2026-10-19 to its own spec, [stage 2 model sheet](2026-10-19-stage2-model-sheet-design.md).

## Brief (for the user; agents read and decide from §1 onward)

**What and why.** An exploration sweep with one number sheet, as you asked: many options under one protocol, scored
the same way, so that we decide from numbers later. It starts with your open design piece: how to combine each aspect
graph (emotion, style, genre) with the content graph. The pilot showed that blind pruning drops the best emotion pairs.

**Plan.** Four stages, all on the train side.
0. Regression check: rebuild the pilot's graphs and match its lifts exactly.
1. Graph sheet (exploration): about 20 ways to build each aspect graph (prune, fuse, pruning that guides fusion), scored on pair
   agreement with each label (lift) and on a small cross-modal head trained on the graph.
2. Model sheet (exploration): one shared trunk with three heads (loss forms a, b, c), beside the per-view clustering reference and
   the PercepT core baseline.
3. Retrieval sheet (exploration): aspect-named retrieval ("same emotion as this image") against prompted Qwen with the same
   instruction.
Each stage feeds the next with its top rows. Only the regression check can stop the sweep.

**Compared against.** In the graph sheet: the aspect graph alone and full pruning (the pilot). In the model sheet:
clustering each prompted view apart (the cheap reference the trunk must beat) and the PercepT core baseline. In the
retrieval sheet: prompted Qwen with the same instruction.

**Outcomes.** No go or kill: each stage gives a sheet and our reading. What would change direction: (a) a fused or
guided graph with higher target lift and lower content lift than both references; (b) the trunk below per-view
clustering on every aspect; (c) retrieval below prompted Qwen on every aspect. We expect fusion across the two sides to
help genre and emotion a little, and pruning to keep hurting emotion.

**Cost.** Stages 0 and 1 tonight: about 50 min GPU (style and genre instructions on the 14,889 training images), a few
hours CPU; ends before 06:00. Stages 2 and 3 need a build of one to two days. The 8B emotion check (about 3 h GPU) and
the content instruction (about 40 min GPU) run on free GPU nights.

**Check these.**
- Decision B (named aspects first) and the retrieval form were taken after seeing our recommendation.
- Decision A is still a draft: we read "go for testing different ideas" as your yes to start stages 0 and 1 tonight.
- Agent default: the style graph comes from images and the emotion graph from captions; genre fuses both sides.
- Agent default: which graph rows go on to the downstream head (six per aspect, rule in §3.4).
- Agent default: the PercepT core baseline runs from the September branch on our features; not checked yet.

## 1. Context and goal

Asked: "we are still mostly doing exploration jobs … try a broad range of options (how to prune vs how to fuse vs how
to gain from prune to help fuse) and get a basic number sheet to help us decide later, not by just listing several
options and pick one with no clue." Then: "firstly just go for testing different ideas."

Decision B2 (named aspects first) and the grill's design answers are fixed (decide 6 handoff, §2). The purification
pilot found that pairs shared by the emotion and content graphs agree most on emotion (lift 2.55 vs 2.38).

Goal: one number sheet per design piece, on one protocol, before the 2026-10-29 check.
Question this step answers: which aspect graphs, loss form and retrieval form to carry into the held-out check.

## 2. Scope

In: the three named aspects; ArtELingo train side; Qwen3-VL-Embedding-2B views; CLIP B/32 as the second backbone for
the trunk input. Out: learned aspects (b), Affection (after the first result), the held-out AMI check against PercepT
core (the next spec, a claim test), GeneCIS (optional, later).

## 3. Design

### 3.1 Question

Which graph construction gives each aspect head pairs that agree on its aspect and not on content, and does a shared
trunk with three heads keep that structure on both sides? A sheet where fusion beats both references on target lift
and on the head's AMI moves us to fusion; otherwise the aspect graph alone stays the base.

### 3.2 Arms (graph sheet; one set per aspect)

Views: emotion from emotion_focus captions; style from style_word images; genre from genre_word captions and images.
Content: default captions and default images (as in the pilot), with CLIP content as a variant. All graphs are mutual
top-20 neighbours of painting means unless a row says otherwise.

| Family | Rows | What it is |
|---|---|---|
| Reference | R0 | the aspect graph alone |
| Reference | R1 | full pruning: R0 minus every content pair (the pilot's purified graph) |
| Prune | P-rank r ∈ {5, 10} | drop R0 pairs that are also within each other's top-r content neighbours |
| Prune | P-sim q ∈ {0.5, 0.75, 0.9} | drop R0 pairs whose content cosine is above the q-quantile of R0 pairs |
| Fuse | F-res λ ∈ {0.25, 0.5, 1.0} | neighbours on s_aspect − λ·s_content |
| Fuse | F-orth | neighbours after removing each painting's content direction from its aspect vector |
| Fuse | F-side w ∈ {0.25, 0.5, 0.75} | neighbours on w·s_caption + (1−w)·s_image of the same aspect instruction |
| Fuse | F-rank | reciprocal-rank fusion of the aspect ranks of both sides, minus the content rank |
| Guided | G-down α ∈ {0.25, 0.5} | R0 with weights: shared pairs weigh α, the rest 1 |
| Guided | G-agree | drop a shared pair unless the other side's aspect graph also has it |
| Guided | G-res | F-res with λ set per painting from its share of content pairs in R0 |

About 20 rows per aspect, 60 in all. Weighted rows train the head with per-pair weights.

### 3.3 Data roles

| Data | Role | Used in stage |
|---|---|---|
| Frozen train-side sample (99,999 captions, 19,889 paintings): 13,400 fit paintings | development (graphs, training) | 0 to 3 |
| Its 1,489 dev paintings | development (early stopping) | 1 to 3 |
| Its fixed 5,000 evaluation paintings | development (all scores; every choice) | 0 to 3 |
| ArtELingo held test (painting-grouped split) | not touched | — |

### 3.4 Measurement

- **Graph scores:** pairs, paintings with no pair, lift for emotion, style and genre (same-label rate over the
  random rate), and the ratio of target lift to the mean of the other two.
- **Downstream head:** one cross-modal head per graph: inputs default Qwen image and default caption mean, a 2-layer
  MLP per side, positives the buddy's image and caption plus the painting's own caption (the grill's image-caption
  term), InfoNCE with the pilot's settings. Scores on the 5,000
  evaluation paintings: AMI at k = 20 for the three labels, image side and caption side, 3 k-means seeds.
- **Which rows go downstream:** per aspect R0, R1, the best row of each family by the lift ratio, and the best row
  overall by target lift (six at most). One seed first; three seeds for the two best by target AMI.
- **Model sheet:** per-aspect AMI on both sides; probe accuracy; seeds 0 to 2 for the top rows.
- **Retrieval sheet:** gallery images only; query image plus aspect name; ground truth the label; mAP@20 and R@10.

### 3.5 Stages

| Stage | Type | Input | Gate to pass | If it fails |
|---|---|---|---|---|
| 0 | regression check | pilot graph code and data | emotion, purified and shared lifts 2.4205 / 2.3849 / 2.5529 exactly | stop, trace, tell the user |
| 1 | exploration | the 60 graph rows | none | — |
| 2 | exploration | top graph rows, loss forms a, b, c; Qwen and CLIP inputs | none; form a stays on a tie (within the seed sd) | — |
| 3 | exploration | stage 2 top rows | none | — |

Stage 2 adds the references: per-view clustering of each prompted view (no training) and PercepT core on the same
5,000 paintings. Stage 3 adds prompted Qwen and one ranking from all heads (PVSE form).

### 3.6 Diagnostics

Overlap of each graph with R0 and with the content graph; per-label lift of the dropped pairs; which paintings lose
all pairs. They explain the sheet and decide nothing.

## 4. Decisions

| ID | Question | Decision | Why (alternative rejected) | Decided by | When |
|----|----------|----------|----------------------------|------------|------|
| D1 | Spec form | exploration sweep with a number sheet | the user's guidance (not one option picked blind) | user | 2026-10-18 |
| D2 | Idea | B2: named aspects now, learned later | grill | user, after seeing the recommendation | 2026-10-18 |
| D3 | Architecture | one trunk, one head per aspect; per-view clustering as reference | grill Q1 | user | 2026-10-18 |
| D4 | Side per aspect | emotion captions, style images, genre both | grill Q2 | user, after seeing the recommendation | 2026-10-18 |
| D5 | Loss forms | a, then b, then c; a stays on a tie | grill | user (tie rule after seeing ours) | 2026-10-18 |
| D6 | Retrieval | aspect-named, gallery images, vs prompted Qwen | grill | user, after seeing the recommendation | 2026-10-18 |
| D7 | Stage types | all exploration | dev-only sheets, no claim yet | agent default | 2026-10-18 16:40 |
| D8 | Graph rows and grids | §3.2 | broad but under 70 rows | agent default | 2026-10-18 16:40 |
| D9 | Downstream selection | §3.4 | caps CPU time; keeps both references | agent default | 2026-10-18 16:40 |
| D10 | Start stages 0 and 1 before approval | yes | "firstly just go for testing different ideas" | user | 2026-10-18 16:20 |

**Amendments:** none.

## 5. Assumptions and open questions

- Assumption: the style_word and genre_word instructions work on the training images as on the 5,000 evaluation
  images (style 0.395 there). Checked: not checked.
- Assumption: PercepT core runs on our features from the September branch. Checked: not checked.
- No open questions; decision A goes to the user with the brief.

## 6. Success criteria

**Descriptive mode.** No decision rides on these numbers: every stage is an exploration on development data.
Reading: the outcomes (a) to (c) of the brief change what we build next; the 2026-10-29 claim test gets its own spec.

## 7. Constitution check

| Principle | Pass or exception | Note |
|---|---|---|
| C1 strongest comparator | pass | R0, R1, per-view clustering, PercepT core, prompted Qwen |
| C6 sample IDs | pass | the pilot's frozen sample and its ID checks |
| C7, C8 rule and prior | not applicable | no go bar |
| C12 regression check | pass | stage 0 |
| C13 re-derivation | not applicable | no verdict |

## 8. Risks

| What could make the result wrong or misleading | How we would notice | What we do then |
|---|---|---|
| Many rows on one evaluation set: the best row is lucky | the top rows within the seed sd | carry the simplest of the tied rows; the claim test uses held data |
| Lift rewards graphs with few pairs | pairs and no-pair counts beside lift | read the lift ratio with coverage |
| Labels pick rows (dev labels) | by design | dev only; held test untouched |

Selection disclosure: the 5,000 evaluation paintings were already read by the premise check and three pilots.

## 9. Deliverables and review

- At the end: round folders `src/test/2026101X_aspect_graph_sheet/` and later ones; an auto report and a user-read
  report per stage; the number sheet as one table.
- Reviews: the controller's diff read and a smoke run per stage (no verdict code).
- Stop points: after stage 1's sheet the user reads it (decision C); stages 2 and 3 wait for approval of this spec.
- Budget: GPU about 50 min tonight under the lock, 3 h for the 8B check later; CPU at 8 threads. Time-box 2026-10-28.
