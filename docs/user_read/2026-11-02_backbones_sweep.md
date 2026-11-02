# The recipe works on three more encoders; the frozen photo model beats its teacher on EmoSet

> 2026-11-02 · [Full report](../reports/auto/percept/2027-01-02_backbones_sweep.md)

## Summary

- The recipe carries over to three more image encoders, on paintings and on photos, within the limit of how we picked them.
- On EmoSet, a second photo dataset, the frozen photo model beats both the plain image and its teacher on emotion.
- One encoder comparison falls below the teacher: CLIP L/14 emotion retrieval on paintings. It supports no claim.
- The analysis sweep shows how parts and settings move the scores. It names no better setting.
- The results support claims on the tested encoders and datasets only.

## How we got here

CoSiR aims to cluster and retrieve images by aspects such as emotion, style and genre.
It builds on PercepT, a published method that finds perceptual topics in image features.
This line started with a model trained on ArtELingo, a set of paintings with emotion captions and style and genre labels.
Its training uses captions, graphs of related paintings, and image features from Qwen3-VL-Embedding (Qwen).
Qwen is an image-text embedding model that can follow a text instruction.
Qwen image features instructed to focus on one aspect are the teacher, and also the main comparator.
At test time, the model reads only the plain image embedding, with no aspect instruction.

Affection is a photo dataset with emotion labels.
On Affection, the painting model used as it is scored below its teacher.
In the next round, we fine-tuned it on Affection's train photos, without their emotion labels.
That brought it level with or better than the teacher; this fine-tuned model is the photo model.

This round asks whether the recipe works with other image encoders, and whether the frozen photo model works on another photo dataset.
It also sweeps the recipe's parts and settings to explain the results, without picking a replacement.

## The recipe works with each added encoder

**Problem.** We had tested the recipe only on the Qwen image encoder.

**Idea.** We trained the painting model on SigLIP 2, SigLIP 1 and CLIP L/14 image embeddings, with the same recipe.
Qwen still supplied the instructed graphs, captions and teacher during training.
Training still depended on Qwen; only the image input changed.

We also fine-tuned each painting model on Affection's train photos, without emotion labels.
The photo training settings came from the earlier Qwen work.
Validation data picked the retrieval read-out (how stored embeddings are compared) before any test scoring.
Each claim used five seeds, that is, five training runs from different random starts.

We compared each model with its encoder's plain image embedding.
On paintings, we also compared clustering with matched PercepT: PercepT trained on the same encoder's image features.
Separate comparisons asked whether each model reached the Qwen teacher.
Painting tests use the held paintings: ArtELingo's test split, kept apart from training and validation.

Clustering uses AMI: agreement between clusters and labels, adjusted for chance.
Retrieval uses mAP: how high images with the same aspect label rank.
Higher is better for both.

A decision rule, fixed before the run, reads each difference through its 95% interval:

- **Win:** the whole interval lies above zero.
- **Level:** the interval's lower end lies above minus the model's seed range (the spread of its validation scores across seeds).
- **Below:** neither.

A level result supports only "level with or better than".

**Did it work.** Yes, under the rule, for each tested encoder.
The painting models beat their plain images in clustering and retrieval for emotion, style and genre.
They also beat matched PercepT in clustering for each aspect.
Each photo model beats its plain image in both emotion tasks on Affection.

The teacher comparisons are more mixed.
Of 24, 18 are wins and 5 are level.
In four of the level cells, the model scores above the teacher.
In the fifth, CLIP L/14 genre clustering on paintings, it scores slightly below (-0.0052) but stays inside the level margin.
One cell is below: CLIP L/14 emotion retrieval on paintings.
Its whole interval lies below zero, so it supports no claim.

These are claim tests on fixed data and seeds.
We picked the three encoders because they passed an earlier emotion test on these same held paintings.
Affection's test photos were also scored in earlier rounds.

**Key evidence.** All differences below compare the model with its encoder's plain image.
The painting columns show the smallest difference across the three aspects, which is emotion for each encoder.

| Encoder | Painting AMI difference | Painting mAP difference | Affection AMI difference | Affection mAP difference |
|---|---|---|---|---|
| SigLIP 2 | +0.0477 | +0.0249 | +0.0782 | +0.0711 |
| SigLIP 1 | +0.0399 | +0.0236 | +0.0728 | +0.0688 |
| CLIP L/14 | +0.0377 | +0.0192 | +0.0815 | +0.0640 |

All of these are wins under the rule.
The separate teacher comparisons include the one exception:

| Encoder comparisons with the teacher | Result |
|---|---|
| Wins | 18 |
| Level | 5 |
| Below | 1 |
| CLIP L/14 emotion retrieval, model minus teacher, 95% interval | -0.0013 [-0.0023, -0.0003] |

> [Full report: Results](../reports/auto/percept/2027-01-02_backbones_sweep.md#results), with the level cells and the licensed claims.

## The frozen photo model beats its teacher on EmoSet

**Problem.** The photo result needed a test beyond Affection, whose test photos earlier rounds had already scored.

**Idea.** We tested the existing Qwen photo model on EmoSet, a second photo dataset with emotion labels.
The model stayed frozen and never trained on EmoSet.
This tests whether the photo model carries over beyond its training dataset.

EmoSet validation picked each model's read-out and the teacher's instruction wording, by a rule fixed in an earlier round.
These picks were frozen before test scoring.
The comparators were the plain Qwen image and the teacher, the Qwen image instructed for emotion.
The rule required wins over both, in both tasks.

We removed near-copies from EmoSet: photos very close to an Affection train photo in Qwen's image embedding.
This cuts overlap with the photo model's training data.
We also scored the test set with the near-copies kept, to see how much the filter matters.

**Did it work.** Yes. The photo model beats both comparators in clustering and retrieval.
EmoSet is the only test set in this round that shaped no earlier choice.
The result covers only emotion, and only this one more photo set.

The gain over the teacher is not the same for every emotion.
On amusement, the photo model's retrieval score is below the teacher's (0.3203 against 0.3471).

With the near-copies kept, the clustering gain over plain Qwen shrinks.
Every comparison still counts as a win.

**Key evidence.** The filtered test set holds 17,348 photos.
Both intervals of the difference from the teacher lie above zero.

| Emotion task | Photo model | Plain Qwen image | Teacher | Photo minus teacher, 95% interval |
|---|---|---|---|---|
| Clustering, AMI | 0.4571 | 0.3918 | 0.4390 | +0.0181 [+0.0121, +0.0242] |
| Retrieval, mAP | 0.4838 | 0.3864 | 0.4519 | +0.0320 [+0.0295, +0.0347] |

The filter removes 368 test photos and changes the size of the clustering gain:

| EmoSet test version | Photo minus plain Qwen, AMI |
|---|---|
| Near-copies removed | +0.0653 |
| Near-copies kept | +0.0390 |

> [Full report: Results](../reports/auto/percept/2027-01-02_backbones_sweep.md#results) for the claim; [Analysis](../reports/auto/percept/2027-01-02_backbones_sweep.md#analysis) for the near-copy and per-emotion checks.

## The analysis shows how parts and settings move the scores

**Problem.** The claim tests do not show which parts of the recipe matter, or how scores change with training settings.

**Idea.** Three sets of runs, all for analysis only:

- **Painting settings:** loss weights, graph settings and model settings, swept on validation paintings.
- **Photo settings:** the photo model's training settings and fixed training lengths, swept on Affection validation photos.
- **Ablations:** stored runs that each remove one training part, compared with the unchanged painting model on held paintings.

The ablations remove, among other parts, these four:

- the instructed graphs: graphs of related paintings built from the instructed Qwen features;
- the graph losses: the losses on the instructed graphs;
- the relational term: a genre loss that makes each painting's pattern of similarities to other paintings copy the teacher's;
- content correction: the step that removes image content from the emotion and style features before their graphs are built.

These runs measure how scores move around the frozen models.
They cannot name a setting better than either model's own.

**Did it work.** The analysis is complete; it has no verdict.
Our reading of the ablations:

- Removing the instructed graphs gives the largest style clustering drop.
- Removing the relational term gives the largest genre clustering drop. It also lowers genre retrieval.
- Removing the graph losses or content correction also cuts style clustering, so no single part carries style.
- The effects of training settings vary by aspect and task.

**Key evidence.** Each change is against the unchanged model, painting or photo, in the named metric's units.

| Change tested | Data and metric | Score change |
|---|---|---|
| Remove instructed graphs | Held paintings, style AMI | -0.1903 |
| Remove graph losses | Held paintings, style AMI | -0.1512 |
| Remove content correction | Held paintings, style AMI | -0.0752 |
| Remove relational term | Held paintings, genre AMI | -0.0440 |
| Remove relational term | Held paintings, genre mAP | -0.1175 |
| Turn content correction off | Photo validation, emotion AMI | -0.0320 |

> [Full report: Analysis](../reports/auto/percept/2027-01-02_backbones_sweep.md#analysis), with all setting curves, ablations and photo training lengths.

## Our advice

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** Freeze the painting model, the photo model and the claims, and write for the submission deadlines.
The recipe carries over across two encoder families, within the limit of how we picked the encoders.
EmoSet gives the strongest photo evidence so far, because its test set shaped no earlier choice.
The one below cell is small: its point value lies inside one seed range, but its interval does not.
Keep it visible in the paper.
Use the analysis sheets for the paper's ablations and analysis.

</details>

## Next

The encoder and EmoSet claim tests are complete and hold under the rule.
The analysis picks no replacement for either frozen model.
Your next step: give your reading of these results, then choose the direction.
The full report offers three options:

1. Freeze and write. This needs no compute and fixes the claims for the abstract and the paper.
2. Test on PolyArt (MMArt), WikiArt paintings not in ArtELingo. Its data is ready, and a side chat extracts its features tonight. Then a test design and one scoring round.
3. Test a fourth encoder that no earlier held test picked: about 10 runs on DAS6 and about 1 hour of CPU scoring. This shows whether the encoder pick shaped the result.

## Glossary

| Plain name | Meaning | Name in the full report |
|---|---|---|
| Painting model | Model trained on ArtELingo paintings; only the plain image embedding enters at test | c* |
| Photo model | Painting model fine-tuned on Affection train photos without emotion labels | photo model |
| Recipe | The training loss terms and their weights | c*'s recipe |
| Qwen | Qwen3-VL-Embedding, an image-text embedding model that follows a text instruction | Qwen; Qwen3-VL-Embedding-2B |
| Plain image | An encoder's image embedding without an aspect instruction | plain image; frozen image; plain Qwen |
| Teacher | Qwen image embedding instructed to focus on one aspect | teacher; instructed Qwen image |
| Matched PercepT | PercepT trained on the same encoder's image features | matched PercepT |
| Held paintings | ArtELingo's test paintings, kept apart from training and validation | ArtELingo held |
| Near-copy | An EmoSet photo very close to an Affection train photo | near-copy |
| Read-out | How stored embeddings are compared for retrieval | read-out |
| Instructed graphs | Graphs of related paintings built from instructed Qwen features | instructed graphs |
| Graph losses | The losses on the instructed graphs | buddy terms |
| Relational term | Genre loss that copies the teacher's pattern of similarities between paintings | relational term |
| Content correction | Step that removes image content from emotion and style features before graph building | content correction |
| Clustering score | Agreement between clusters and labels, adjusted for chance | AMI |
| Retrieval score | How high images with the same aspect label rank | mAP |
| Seed range | Highest minus lowest validation score across seeds | own range; seed range |
| Win, level, below | The rule's three outcomes for a difference interval | WIN, LEVEL, BELOW |
| Claim test | A test whose verdict licenses a claim sentence | claim test |
