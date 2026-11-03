# What we have and what is missing: clustering and retrieval are covered on four encoders and four test sets; the gaps are outside methods, clustering metrics and a few checks

> 2026-11-03 · An overview on request, for choosing the next experiments. Every number comes from the auto reports of
> r10 to r15 ([report index](../reports/reports_sum.md)). The facts sheet behind it, with a source for each row:
> [facts inventory](../../src/test/20261103_experiment_overview/facts_inventory.md).

## Summary

- Clustering, our main task, has claim results for our painting model on four encoders and two painting test sets.
- On photos, a fine-tuned photo model carries the claims: on Affection with four encoders, on EmoSet with Qwen.
- Every comparator we planned is filled: the plain image, PercepT on the same encoder, and the Qwen teacher.
- Retrieval, the second task, has the same grid. Its weak spot is emotion against the teacher.
- The gaps sit outside our grid: almost no outside methods, and AMI as the only clustering metric.
- Most gaps cost a few CPU hours and a small script. TPIPS and a criterion-driven clustering method cost the most.

## How we got here

The project builds c*, a model that gives emotion, style and genre spaces from one read of a painting's plain image.
c* learns from a teacher: the Qwen image embedding, given an instruction for one aspect.
The aim is to improve on PercepT, a published method that finds perceptual topics, such as emotion, in image embeddings.
From 2026-10-27 to 2026-11-03, six rounds (r10 to r15) built c* and then tested it.
On 2026-11-03 you paused experiments, to report to the collaborator and the supervisor.
You asked for this overview to choose what to run before the deadlines: abstract 2026-11-10, paper 2026-11-16,
supplementary 2026-11-23.

The four test sets:

- **ArtELingo held:** 12,281 ArtELingo paintings kept for tests only, with emotion and style labels (genre labels on
  9,949).
- **MMArt test:** 16,384 WikiArt paintings outside ArtELingo, with MMArt's own labels.
- **Affection test and EmoSet test:** photos with emotion labels.

The four encoders are Qwen3-VL-Embedding-2B (Qwen), SigLIP 2, SigLIP 1 and CLIP L/14.

In this overview, "missing" means a hole in our own grid, or a check that a reviewer would expect for robust results,
as our literature notes name it.

## Clustering (main task): what we have

Clustering is scored with AMI: the agreement between 20 k-means clusters and the labels, corrected for chance.
Each trained model has five seeds, and each comparison has a 95% interval.
Roles in the tables:

- **claim:** a comparison whose bar was set before the run; it decides a sentence in the paper;
- **comparator:** a reference row in a claim; the bar against it is a WIN, or "LEVEL or WIN", as set before the run;
- **descriptive:** reported, but it decides nothing.

E, S and G are emotion, style and genre.

| Method | ArtELingo held (E, S, G) | MMArt test (E, S, G) | Affection test (E) | EmoSet test (E) |
|---|---|---|---|---|
| c*, the painting model | 4 encoders, claim | 4 encoders, claim | Qwen, failed confirmation (below the teacher) | Qwen, descriptive |
| Photo model | none | none | 4 encoders, claim | Qwen, claim; 3 encoders, descriptive |
| Plain image | 4 encoders, comparator | 4 encoders, comparator | 4 encoders, comparator | Qwen, comparator; 3 descriptive |
| Teacher (instructed Qwen image) | comparator | comparator | comparator | comparator |
| PercepT on the same encoder | 4 encoders, comparator (3 more in an earlier round) | 4 encoders, comparator | Qwen, descriptive | none |
| CRL (an outside method, our re-implementation) | none | none | Qwen and CLIP L/14, descriptive | CLIP L/14, descriptive |
| CLAY, TPIPS, other outside methods | none | none | none | none |

The headline numbers on Qwen (AMI):

| Test set | c* (photo model on photos) | Teacher | Plain image | PercepT |
|---|---|---|---|---|
| ArtELingo held, E / S / G | 0.1066 / 0.4066 / 0.5001 | 0.0921 / 0.4026 / 0.5062 | 0.0708 / 0.2233 / 0.4521 | 0.0874 (E) |
| MMArt test, E / S / G | 0.0609 / 0.4227 / 0.5131 | c* is a WIN on all three (values not in the report) | 0.0407 / 0.2314 / 0.4498 | 0.0514 / 0.1959 / 0.4257 |
| Affection test, E | 0.1838 | 0.1666 | 0.1041 | 0.0870 (frozen, descriptive) |
| EmoSet test, E | 0.4571 | 0.4390 | 0.3918 | none |

A WIN means the 95% interval of c*'s difference lies above zero.
On ArtELingo held, genre is level with the teacher, not a win: the whole interval of the difference lies below zero.

Two caveats on photos:

- The frozen c* was below the teacher on Affection (0.1384). We built the photo model after we saw that failure. It
  trained on separate, unlabelled Affection train photos, but the same test photos scored it, and r14 used them again.
  EmoSet is the photo test that shaped nothing.
- EmoSet scores use 17,348 test photos, after we removed 368 near-copies of Affection training photos. With them kept,
  the photo model's gain over the plain image falls from +0.0653 to +0.0390, still a WIN.

> Facts inventory, sections A1 and B1; reports 12-28, 12-31, 2027-01-01, 2027-01-02 and 2027-01-03.

## Clustering: what is missing

The holes in our own grid are small:

- Frozen c* on Affection and EmoSet uses Qwen only. The photo model carries the claim there, so this matters little.
- PercepT has no EmoSet row, and only a descriptive Qwen row on Affection.
- CRL has no clustering row on paintings.

The checks a reviewer would expect matter more:

- **Standard metrics.** Every claim uses AMI only. Clustering papers also report NMI, ARI and clustering accuracy
  (ACC: the best one-to-one match of clusters to classes). We have NMI and ARI only on development data, in an early
  ablation. (to-do 1)
- **The number of clusters.** Every claim uses 20 clusters. Readings at the class count exist only as diagnostics (8
  clusters on EmoSet, 9 on MMArt). (to-do 2)
- **A strong clustering method on the plain image.** TEMI and TURTLE learn clusters on frozen features. One of them
  would answer "does better clustering of the plain image close the gap?" Nothing ran. (to-do 3)
- **A criterion-driven clustering method.** IC|TC, X-Cluster and Multi-MaP group images by a named criterion, the
  closest published task. Most need a large language or vision-language model per image and per criterion. Nothing
  ran. (to-do 5)
- **New artists.** MMArt shares most artists with ArtELingo: only 15 MMArt test paintings come from new artists.
  SemArt, a painting dataset from the Web Gallery of Art, has 11,162 usable paintings by 2,717 new artists. Its genre
  test is parked. (to-do 6)
- **The encoder pick.** We picked the four encoders after an earlier emotion test on the same paintings, with an
  earlier model. In that test DINOv2 (an image-only encoder) failed, and CLIP B/32 and B/16 were unclear. A reviewer
  may read the pick as selection. (to-do 16)
- **A qualitative figure** of example paintings per cluster, c* against the plain image. None exists. (to-do 20)

> Facts inventory, sections B1 and D; reports 12-24, 12-25 and 2027-01-03.

## Retrieval (second task): what we have

The query is a painting plus an aspect name. The aspect name picks c*'s head, its output for that aspect.
A hit is a gallery painting with the same label. The score is mAP: how high the same-label paintings rank.
A read-out is how the embeddings are compared, for example after centering. Each model's read-out was picked on
validation data.

| Method | ArtELingo held (E, S, G) | MMArt test (E, S, G) | Affection test (E) | EmoSet test (E) |
|---|---|---|---|---|
| c*, the painting model | 4 encoders, claim | 4 encoders, claim | Qwen, failed confirmation (below the teacher) | Qwen, descriptive |
| Photo model | none | none | 4 encoders, claim | Qwen, claim; 3 encoders, descriptive |
| Plain image | 4 encoders, comparator | 4 encoders, comparator | 4 encoders, comparator | Qwen, comparator; 3 descriptive |
| Teacher (instructed Qwen image) | comparator | comparator | comparator | comparator |
| PercepT | Qwen, descriptive | none | Qwen, descriptive | none |
| CRL (our re-implementation) | CLIP L/14, comparator; Qwen, descriptive | none | CLIP L/14, comparator; Qwen, descriptive | CLIP L/14, descriptive |
| CLAY, TPIPS | none | none | none | none |

The headline numbers on Qwen (mAP):

| Test set | c* (photo model on photos) | Teacher | Plain image | CRL (CLIP L/14) |
|---|---|---|---|---|
| ArtELingo held, E / S / G | 0.2691 / 0.3235 / 0.6151 | 0.2684 / 0.3217 / 0.6124 | 0.2483 / 0.1924 / 0.5361 | 0.2554 / 0.2816 / 0.5033 |
| MMArt test, E / S / G | 0.2011 / 0.3472 / 0.6199 | c* is BELOW on E (by 0.0010), LEVEL on S, a WIN on G | 0.1935 / 0.2046 / 0.5151 | none |
| Affection test, E | 0.3172 | 0.2945 | 0.2452 | 0.2739 |
| EmoSet test, E | 0.4838 | 0.4519 | 0.3864 | descriptive (value not in the report) |

LEVEL means not a WIN, with the interval's lower end above minus one seed range. BELOW means the lower end is at or
below minus one seed range.
CRL's bars on ArtELingo were set after we had seen its held scores, so that comparison is weaker than the others.

Against the teacher, emotion is the weak aspect. Three of the eight painting cells are below it, by at most 0.0013
mAP. They are CLIP L/14 on ArtELingo, and Qwen and CLIP L/14 on MMArt. The other five are level.

> Facts inventory, sections A2 and B2; reports 12-29, 12-31, 2027-01-01, 2027-01-02 and 2027-01-03.

## Retrieval: what is missing

The holes in our own grid:

- CRL has no MMArt row. (to-do 8)
- PercepT has retrieval rows only for Qwen. PercepT is a clustering method, so this matters little.

The checks a reviewer would expect:

- **Outside methods that make similarity depend on an aspect.** CLAY (needs no training, public code) and TPIPS (a
  released 8B model) are on your list. (to-dos 13 to 15)
- **Other query forms.** All our results are "told": the aspect name picks the head. Untold retrieval (the image only)
  and text queries through c*'s caption side are on your list. (to-dos 9 and 10)
- **Emotion against the teacher.** Either we state the three cells as a limit, or we look at them per class. On
  MMArt, the gap sits mostly in one class, contentment. (to-do 11)
- **A second instructed embedder.** GME and VLM2Vec are other embedding models that follow instructions. Given the
  same aspect instructions, one of them is a second told comparator. A reviewer may ask whether Qwen is the strongest.
  (to-do 17)
- **Standard benchmarks for conditional retrieval** (GeneCIS, CLAY's Stanford40 set) use conditions that c* has no
  head for. Our view: explain this in the paper, do not run them.

> Facts inventory, sections B2 and D.

## Ablations and analysis: what we have and what is missing

| Set | Data | Encoders | Tasks | Seeds |
|---|---|---|---|---|
| Eight removals of c*'s parts | ArtELingo held and development | Qwen | both | 5 |
| 13 settings ("knobs"), three values each | ArtELingo validation | Qwen | both | 3 |
| 14 photo settings | Affection validation | Qwen | both | 3 |
| Read-out choices | ArtELingo and Affection validation | Qwen, CLIP L/14, SigLIP 2 | retrieval | 5 for trained models |
| The wrong-head test: each aspect's query read through another aspect's head | ArtELingo held | Qwen | retrieval | 5 |
| Earlier design sheets (graphs, sharing, teacher forms, genre terms) | ArtELingo development | Qwen | clustering | 1 to 3 |
| An earlier model on seven encoders | ArtELingo held | 7 | clustering | 5 |

The eight removed parts are: the graphs of related paintings built from the instructed views (replaced by graphs from
the plain view); the "buddy" losses on those graphs; the relational genre term, which copies the teacher's similarity pattern; the content correction, which
removes image content before the emotion and style graphs; the direct teacher term; the sharper teacher; the caption
side of the buddy losses; and the shared trunk.

What is missing: the DEC loss on every head (to-do 4) and deeper heads (to-do 18), both on your list. All ablations use
Qwen only; our view is that the paper can say so.

> Facts inventory, section C.

## The to-do list

Build size is a rough guess of our work, from smallest to largest:

- **None:** text only.
- **Small:** one script on saved outputs, or a thin adapter around released code. One chat, a few hours.
- **Medium:** a new variant in the training code, or a new data source, with tests. A medium loop (short spec, build
  chat, run chat), about a day.
- **Large:** a new model, teacher, aspect or pipeline. A full loop with a literature check, two days or more.

Machine costs come from our run logs where they exist. On the 3090, one c* training run takes 3.5 to 5.5 minutes, and
Qwen reads about 11 images per second per view. DAS6 is the university GPU cluster. "Fits before" is our estimate of
the earliest deadline. Origin: "yours" is an item you decided or parked in the last chats; "reviewer" is a gap this
overview found.

| # | Item | Task | Origin | Machine cost | Build | Fits before |
|---|---|---|---|---|---|---|
| 1 | NMI, ARI and ACC beside AMI on every claim cell | clustering | reviewer | CPU 2 to 4 h | small | 11-10 |
| 2 | Cluster count at the class count, and 10 and 40, on the claim cells | clustering | reviewer | CPU 2 to 3 h | small | 11-10 |
| 3 | A strong clustering method on the plain image (TEMI or TURTLE), four encoders | clustering | reviewer | GPU under 1 h per encoder (not measured) | small to medium | 11-16 |
| 4 | DEC loss on every head, plain and with a pull toward the start clusters; retrieval reported too | clustering | yours | GPU about 1 h; CPU 1 to 2 h | medium | 11-23 |
| 5 | A criterion-driven clustering method (IC\|TC, X-Cluster or Multi-MaP) | clustering | reviewer | tens of GPU hours (rough guess) | large | after the paper |
| 6 | SemArt genre test on new artists | both, genre only | yours (parked) | GPU under 1 h; CPU 2 to 3 h | medium | 11-23 |
| 7 | CRL against c* on the same encoder (CLIP L/14), with the WIN bar set before we look | retrieval | yours | CPU under 1 h | small | 11-10 |
| 8 | CRL on MMArt | retrieval | yours | CPU about 1 h | small | 11-10 |
| 9 | Untold retrieval: the image only, c*'s three heads joined into one vector | retrieval | yours | CPU about 1 h | small | 11-16 |
| 10 | Text queries through c*'s caption side, emotion first | retrieval | yours | under 1 h | small to explore; medium for a claim | 11-16 to explore |
| 11 | Emotion below the teacher: a limit sentence, or a per-class look | retrieval | yours (open) | none, or CPU under 1 h | none or small | 11-16 |
| 12 | MMArt perspective captions as text queries | retrieval | yours (parked) | GPU under 1 h | medium | 11-23 |
| 13 | TPIPS timing probe (8B model) | both | yours | GPU about 10 min, after a 16.5 GB download into its own env | small | 11-10 |
| 14 | TPIPS as a comparator: ArtELingo held and MMArt, three aspects | both | yours | GPU 7.5 to 11 h (a guess); DAS6 or a night | small | 11-16 |
| 15 | CLAY on CLIP L/14: ArtELingo held and MMArt | both, retrieval first | yours | GPU minutes; CPU about 1 h | small, plus aspect prompts we write | 11-16 |
| 16 | c* on the encoders that failed or were unclear before (DINOv2, CLIP B/32, B/16) | both | reviewer | GPU 1 to 2 h; CPU 1 to 2 h | small | 11-16 |
| 17 | A second instructed embedder (such as GME or VLM2Vec) as a told comparator | both | reviewer | GPU about 3 to 4 h | medium | 11-23 |
| 18 | Deeper heads (a 3-layer residual head) | both | yours (supplementary) | GPU about 1 h; CPU 1 to 2 h | medium | 11-23 |
| 19 | Cost table: c*'s one pass against the teacher's one pass per aspect | both | reviewer | GPU about 30 min | small | 11-16 |
| 20 | Qualitative figure: example paintings per cluster, c* against the plain image | both, clustering first | reviewer | CPU minutes | small, plus design work | 11-16 |
| 21 | A second photo aspect (Flickr Style or AVA) | both | yours (parked) | large | large (access risk) | after the paper |
| 22 | The note for the collaborator and the supervisor: one line that c* leaves the frozen encoder's own retrieval unchanged | paper | yours (open) | none | none | 11-10 |
| 23 | Paper: a "training images from the domain" column; CRL, CLAY and TPIPS as context rows | paper | yours | none | none | 11-16 |

Costs marked "not measured" or "a guess" are ours, not from a log. The TPIPS time assumes the 8B model is 3 to 4 times
slower than Qwen.

## Advice (our view)

- **Before the abstract (11-10):** the abstract needs nothing new. The cheapest items firm up the main tables: 1, 2,
  7, 8 and 13. Most are CPU work. Item 13 needs a download, its own env and minutes of GPU. One small loop can do them.
- **Before the paper (11-16):** the outside methods in both tasks (14, 15) and a clustering baseline (3). Also the
  encoder-pick question (16), the cost table and the figure (19, 20), and items 9 and 11. TPIPS needs DAS6 or a night.
- **Supplementary (11-23):** 4, 6, 10, 12, 17 and 18.
- **Cite and explain only:** 5 and 21.
- **The largest review risks:** AMI as the only clustering metric (1), and no outside clustering method except PercepT
  (3, 14, 15). Both are cheap to address.

## Next

- Yours to decide: which items run, and in what order. Experiments stay paused until you say so.
- Yours to decide: the form of the note for the collaborator and the supervisor, after your other discussions, and
  item 22.
- After your choice, we write one plan for the chosen items. The small items can share one loop.

## Glossary

- **c\*:** our painting model; a shared trunk with emotion, style and genre heads on a frozen encoder.
- **Photo model:** c* fine-tuned on unlabelled Affection photos.
- **Teacher:** the Qwen image embedding given an instruction for one aspect; c* learns from it.
- **Plain image:** an encoder's image embedding without an instruction.
- **PercepT:** the published method we improve on; it finds perceptual topics in image embeddings. "On the same
  encoder" (matched): trained on that encoder's features.
- **CRL, CLAY, TPIPS:** outside methods that make similarity depend on a named aspect. CRL and CLAY need no training;
  TPIPS is a tuned 8B model.
- **AMI:** agreement between clusters and labels, corrected for chance. **mAP:** how high same-label paintings rank.
- **WIN:** the 95% interval of c*'s difference lies above zero. **LEVEL:** not a WIN, and the interval's lower end lies
  above minus one seed range. **BELOW:** the lower end is at or below minus one seed range.
- **Seed range:** the spread of a score across training seeds, set on validation data; the margin for "level".
- **Told, untold:** the aspect name picks the head (told), or the query is the image only (untold).
- **Held, validation:** held data is for tests only; validation data sets choices such as the read-out.
