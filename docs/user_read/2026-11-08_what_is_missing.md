# What we have and what is missing after r24: the planned comparisons on paintings are done; the paper, one figure and a few checks are left

> 2026-11-08 · Updates the overview of 2026-11-03 ([overview](2026-11-03_experiment_overview.md)) and the coverage
> tables of the [stage report of 2026-11-04](../reports/stage/2026-11-04_datasets_and_retrieval.md). Every cell with
> its source: [facts inventory](../../src/test/20261108_what_is_missing/facts_inventory.md).

## Summary

- The 11-03 plan had six loops. Four ran (r17, r18, r19, r22), and text retrieval grew into r23 and r24.
- Clustering and image retrieval on ArtELingo and MMArt now have each planned comparator, at its planned scope:
  plain image, teacher, PercepT, CRL, CLAY, TPIPS, and TURTLE in clustering at its default settings.
- Text retrieval has a claim on Qwen only. It holds on ArtELingo and fails on MMArt; there the counts follow MMArt's
  narrative text, and the reason stays open.
- Of the 23 to-do items, 15 are done or done in part. Two loops wait: the cluster figure (r20) and the head variants (r21).
- No run is recorded for one check you decided on 11-04: the photo model trained from random weights.
- The paper is the largest open item: only the related work and the method have a first draft (11-02).

## How we got here

c* is our model: one image in, one embedding each for emotion, style and genre. It learns these from the instructed
Qwen3-VL-Embedding views, which we call the teacher. On 11-03 we listed what a reviewer would expect and planned six
loops to fill the gaps. This page checks that list against the results of r17 to r24.

## Clustering: our model against each comparator

Each cell says what ran and how our model did; the outcome words are in the glossary. "4 encoders" are Qwen, SigLIP 2,
SigLIP 1 and CLIP L/14; "3 more" are DINOv2, CLIP B/32 and CLIP B/16. On the 4 encoders, r17's claims use four
metrics (AMI, NMI, ARI, ACC) at the class count. The 3 more encoders and SemArt use AMI at 20 clusters.

| Against | ArtELingo held | MMArt test | Affection test (photos) | EmoSet test (photos) | SemArt new artists |
|---|---|---|---|---|---|
| Plain image | 4 encoders: WIN in all four metrics; 3 more: AMI WIN | 4 encoders: WIN in all four metrics; 3 more: AMI WIN | photo model, 7 encoders: WIN | photo model, Qwen: WIN | 7 encoders, genre: WIN |
| Teacher | 4 encoders: 33 WIN, 13 LEVEL, 2 BELOW of 48; 3 more: no "better than" | 4 encoders: 35 WIN, 10 LEVEL, 3 BELOW of 48; 3 more: mostly BELOW | photo model, 4 encoders: 14 WIN, 2 LEVEL; 3 more mixed. Frozen c*, Qwen: failed | photo model, Qwen: WIN in all four metrics. Frozen c*: descriptive | 4 encoders LEVEL; 3 more mostly BELOW |
| PercepT | 7 encoders: WIN (one ACC LEVEL) | 7 encoders: WIN (two ACC LEVEL) | Qwen, descriptive only | not run | 7 encoders: WIN |
| CRL | CLIP L/14: WIN, emotion ARI and ACC LEVEL | CLIP L/14: WIN, emotion ACC LEVEL, style ACC BELOW | descriptive only | descriptive only | not run |
| CLAY | c* on Qwen and CLIP L/14: WIN, one style LEVEL | WIN, two style LEVEL, one style BELOW | not run | not run | not run |
| TPIPS | c* on Qwen: WIN, one emotion ARI LEVEL | c* on Qwen: all WIN | not run | not run | not run |
| Plain image with TURTLE, default settings | 4 encoders: WIN | 4 encoders: WIN | not run | not run | not run |
| TEMI | not run (deferred) | not run (deferred) | not run | not run | not run |

- The photo model is c* fine-tuned on Affection's unlabeled train photos. Frozen c* is the painting model as it is.
- CLAY runs on CLIP L/14 features. TPIPS is the released 8B Qwen model, four times c*'s base size.
- With TURTLE on c*'s heads too, c* is WIN in 114 of 116 cells; the two LEVEL cells are emotion ARI and ACC.
- In r17's teacher cells on the 4 encoders, all five BELOW are emotion. On ArtELingo, one class, contentment, carries
  that gap. The 3 more encoders add other BELOW cells, genre included.
- TURTLE ran without its own setting selection. It scored below plain k-means in 203 of 232 comparisons, so it does not
  yet answer whether a stronger clustering of the plain image closes the gap.

## Image retrieval: our model against each comparator

| Against | ArtELingo held | MMArt test | Affection test (photos) | EmoSet test (photos) | SemArt new artists |
|---|---|---|---|---|---|
| Plain image | 7 encoders: WIN | 7 encoders: WIN | photo model, 7 encoders: WIN | photo model, Qwen: WIN | 7 encoders, genre: WIN |
| Teacher | 4 encoders, as on 11-04: emotion LEVEL, CLIP L/14 emotion BELOW; 3 more: no "better than" | 4 encoders, as on 11-04: emotion BELOW on Qwen and CLIP L/14, LEVEL on the others; 3 more: mostly BELOW | photo model, 4 encoders: holds; 3 more mixed, DINOv2 BELOW | photo model, Qwen: WIN | 3 encoders LEVEL, CLIP L/14 BELOW; 3 more mostly BELOW |
| PercepT | 4 encoders: WIN | 4 encoders: WIN | Qwen, descriptive only | not run | not run |
| CRL | CLIP L/14: WIN | CLIP L/14: WIN | CLIP L/14: photo model WIN; frozen c* LEVEL | descriptive only | not run |
| CLAY | c* on Qwen and CLIP L/14: WIN | c* on Qwen and CLIP L/14: WIN | not run | not run | not run |
| TPIPS | c* on Qwen: WIN | c* on Qwen: WIN | not run | not run | not run |
| Plain image, no aspect named (untold) | 4 encoders: WIN, one style BELOW, one genre LEVEL | 4 encoders: WIN | not run | not run | not run |

## Text retrieval: c*'s caption side

Text retrieval is new since 11-03. A cell is one aspect in one direction: text to image, image to text, or text to text.
The claim counts WIN cells of 9 in label mAP, on Qwen. "Holds" needs 9 of 9 WIN over plain Qwen and 5 or more over the
teacher's text. Late fusion also needs pair R@10 no more than 0.02 below plain Qwen at the lower bound; it passes on both
sets.

| Row | ArtELingo validation (one seed, no bar) | ArtELingo held | MMArt test | MMArt paintings shared with ArtELingo (no bar, either label set) |
|---|---|---|---|---|
| c* against plain Qwen | 9 of 9 above | claim holds: 9 WIN | claim fails: 8 WIN | captions 9; narrative text 8 |
| c* against the teacher's text | 6 of 9 above (emotion below) | 6 WIN (emotion BELOW) | 3 WIN (style only) | captions 6; narrative text 3 |
| Late fusion against plain Qwen | 9 of 9 above | claim holds: 9 WIN | claim fails: 9 WIN | 9 with either text |
| Late fusion against the teacher's text | 5 of 9 above | 5 WIN | 0 WIN | captions 4; narrative text 0 |
| Qwen c* against other plain encoders (CLIP L/14, SigLIP 2, SigLIP 1) | descriptive: c* above | descriptive, text to image only: c* above | not run | not run |

- Late fusion is c*'s score joined with plain Qwen's at weight 0.7. On the shared paintings with captions, its count
  over the teacher's text is 4, against 5 on all held paintings. r23's verdict stands; the row set differs.
- Why weight 0.7 helps: r23 ran the per-query hit split you asked for. It describes the overlap and tests no cause.
- The own-caption term raises pair hits and lowers label mAP in all 9 cells, on older saved models only. c* was not retrained.
- Pair hits of c* are in the r22 and r23 reports. Their wording waits for your talk with the other authors.

## The to-do list of 11-03: status

| Status | Items |
|---|---|
| Done | 1 standard metrics, 2 cluster counts, 6 SemArt new artists, 7 and 8 CRL, 9 untold retrieval, 10 text queries, 11 emotion against the teacher, 13 and 14 TPIPS, 15 CLAY, 16 three more encoders, 19 cost table; CRL clustering and PercepT retrieval, added by you |
| Done in part | 3 a stronger clustering method: TURTLE at default settings only. 12 MMArt texts as queries: five texts explored on validation; narrative text claim and shared-row check done; the other four have descriptive values on the new paintings only |
| Waiting | 4 DEC loss and 18 deeper heads (r21), 20 the cluster figure (r20) |
| Not run, your call | 17 a second instructed embedder (important; priority not decided) |
| Paper text only | 5 criterion-driven clustering, 21 a second photo aspect, 22 the note on the encoder's own retrieval, 23 the domain column and context rows |

## What is missing

**Runs that you decided or planned, not yet run:**

- The cluster figure, r20: example paintings per cluster, c* against the plain image. A small loop, with design work with you.
- The head variants, r21: DEC loss on each head, plain and with a pull toward the start clusters, retrieval reported
  too; and a deeper head (3 residual layers). Planned for the supplementary.
- The photo model from random weights: five seeds, 80 neighbours, on the Affection test. You decided it on 11-04;
  no run is recorded. About 30 minutes of GPU (our estimate).

**Deferred or parked by you:**

- TURTLE with its own setting selection on validation, then on the plain image (about 4 GPU hours, not measured). A
  small spec must first settle a conflict between two of r18's rules. TEMI: up to 10 GPU hours.
- A second instructed embedder, such as GME or VLM2Vec, as a told comparator (3 to 4 GPU hours).
- MMArt's text: the other four texts on the shared paintings (about 50 minutes of GPU), and tests of which text
  property matters. You see directions here.
- MMArt's perspective names (narrative, formal, historical) as new instructions: new aspects with new heads.
- Later: TPIPS as c*'s teacher, a photo style aspect, emotion and style on new artists, cost against a one-pass teacher.

**Gaps nobody planned to fill** (our view: say them in the paper, do not run them):

- Text retrieval against outside methods (CRL, CLAY, TPIPS), with c* on another encoder, and on photos.
- Frozen c* on photos with encoders other than Qwen; PercepT on EmoSet.

**Paper items, no experiment:**

- The draft: abstract, introduction, experiments, limits. The related work and the method have a first draft (11-02;
  paper folder, checked 2026-11-08).
- Put in r18's outside-method rows; decide whether r19's rows go to an ablation or the appendix.
- Word MMArt's narrative text result, with its reason open. The pair-hit wording waits for your talk with the other authors.
- Explain, without runs: criterion-driven clustering (IC|TC and others), and why GeneCIS and CLAY's Stanford40 set use
  conditions c* has no head for.
- Items 22 and 23, and the note for the collaborator and the supervisor.

## The plan, updated (our proposal, for you to decide)

The deadlines are close: abstract 11-10, paper 11-16, supplementary 11-23 (each 23:59 AoE, 12:59 the next day here).

| Step | What | Size | Machine | When (our estimate) |
|---|---|---|---|---|
| 1. Abstract | Title, abstract and claim sentence, in a paper chat | writing | none | by 11-10 |
| 2. Paper | Introduction, experiments with r15 to r24 rows, limits | writing | none | 11-10 to 11-16 |
| 3. Cluster figure (r20) | As planned | small | CPU minutes | by 11-12 |
| 4. Photo model from random weights | Your 11-04 decision | small | GPU about 30 min | a night before 11-14 |
| 5. Head variants (r21) | DEC loss, deeper heads | medium | GPU about 2 h | 11-16 to 11-20 |
| 6. Supplementary checks | MMArt's other texts; TURTLE selection and TEMI | small, medium | GPU about 1 h; up to 14 h | 11-16 to 11-22 |
| Open | A second instructed embedder (item 17) | medium | GPU 3 to 4 h | your call |

<details>
<summary>Why we propose this order (our view)</summary>

- The abstract and the paper are now the critical path. Each new run takes your time to read, and that time is short.
- The figure is the one planned item the main paper needs. The random-weights check is short and you already decided it.
- The planned comparisons are enough for the main tables. CLAY, TPIPS and CRL now stand beside PercepT, so the
  outside-method risk of 11-03 is smaller.
- TURTLE's selection still matters for one reviewer question. It fits the supplementary with TEMI, as one loop.
- MMArt's text follow-ups add detail. The main paper can word the narrative-text result as a limit without them.

</details>

## Glossary

- **c\*:** our frozen model, with emotion, style and genre spaces and a caption side for text.
- **Teacher:** the instructed Qwen3-VL-Embedding view of each aspect.
- **Plain image, plain Qwen:** the encoder's own embedding, with no instruction.
- **Late fusion:** c*'s score joined with plain Qwen's, at weight 0.7 on plain Qwen.
- **WIN, LEVEL, BELOW:** WIN, the lower bound of the 95% interval is above 0. LEVEL, it is not, but it is above minus
  the seed range on validation. BELOW, otherwise.
- **Told, untold:** the query names the aspect, or it does not.
