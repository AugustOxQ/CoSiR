# The recipe works on more encoders; the frozen photo model beats its teacher on EmoSet

> 2026-11-02 · [Full report](../reports/auto/percept/2027-01-02_backbones_sweep.md)

## Summary

- The recipe carries over to three more image encoders on paintings and photos, within the limits of their earlier selection.
- On EmoSet, a second photo dataset, the frozen photo model beats both the plain image and its teacher for emotion.
- One encoder comparison falls below the teacher; it supports no claim for that task.
- The analysis sweep explains which parts matter, but selects no new setting.
- These results support claims on the tested data; they do not establish the same result on untested encoders or datasets.

## How we got here

CoSiR aims to cluster and retrieve images by aspects such as emotion, style and genre.
This line started with a model trained on ArtELingo paintings.
Its training uses captions, graphs of related examples, and Qwen image features instructed to focus on each aspect.
These instructed features are the teacher, which also serves as a comparator.
At test time, the model takes only the plain image embedding, without an aspect instruction.

The photo model starts from the painting model and fine-tunes on Affection's train photos without emotion labels.
This round keeps that photo model frozen for its EmoSet test.
This round asks whether the recipe works with other image encoders and whether the frozen photo model works on another photo dataset.
It also tests changes to the training recipe to explain the results, without choosing a replacement.

## The recipe works with each added encoder

**Problem.** We needed to test the recipe beyond its original Qwen image encoder.

**Idea.** We trained the painting model on SigLIP 2, SigLIP 1 and CLIP L/14 image embeddings.
Each encoder used the same recipe.
Qwen still supplied the instructed graphs, captions and teacher during training.
This tests a change in image input; it does not test training without Qwen.

We also fine-tuned each painting model on Affection photos without emotion labels.
The photo training settings came from the earlier Qwen work.
Validation data set the retrieval read-out, the method used to compare embeddings, before test scoring.
Each claim used five seeds.

We compared each model with its encoder's plain image embedding.
On paintings, we also compared clustering with PercepT, an earlier method, trained on the same encoder's image features.
Separate comparisons asked whether each model reached the Qwen teacher.

Clustering uses AMI, agreement between clusters and labels adjusted for chance.
Retrieval uses mAP, which measures how high images with the same aspect label rank.
Higher scores are better for both metrics.

**Did it work.** Yes, under the fixed rule, for each tested encoder.
The painting models beat their plain images in clustering and retrieval for emotion, style and genre.
They also beat the matched PercepT comparator in clustering for each aspect.
Each photo model beats its plain image in both emotion tasks on Affection.

The teacher comparisons are more mixed.
Most are wins; some support only “level with or better than”.
CLIP L/14 emotion retrieval on paintings falls below the teacher and supports no claim.
Its difference is small, but its interval fails the rule.

These are claim tests on fixed data and seeds.
The encoders passed an earlier emotion test on these same held paintings, which shaped their selection.
Affection's test photos were also scored in earlier rounds.
The result therefore covers these selected encoders and these datasets.

**Key evidence.** All differences below compare the model with its encoder's plain image.
The painting columns show the smallest difference across the tested aspects, which is emotion in each case.

| Encoder | Painting AMI difference | Painting mAP difference | Affection AMI difference | Affection mAP difference |
|---|---|---|---|---|
| SigLIP 2 | +0.0477 | +0.0249 | +0.0782 | +0.0711 |
| SigLIP 1 | +0.0399 | +0.0236 | +0.0728 | +0.0688 |
| CLIP L/14 | +0.0377 | +0.0192 | +0.0815 | +0.0640 |

All these comparisons count as wins under the rule.
The separate teacher comparisons include an exception:

| Encoder comparisons with the teacher | Result |
|---|---|
| Wins | 18 |
| Level | 5 |
| Below | 1 |
| CLIP L/14 emotion retrieval, model minus teacher, 95% interval | -0.0013 [-0.0023, -0.0003] |

> [Full report: Results](../reports/auto/percept/2027-01-02_backbones_sweep.md#results), including the teacher exceptions and licensed claims.

## The frozen photo model beats its teacher on EmoSet

**Problem.** The photo result needed a test beyond Affection, whose test set had already informed earlier work.

**Idea.** We tested the existing Qwen photo model on EmoSet, a second photo dataset with emotion labels.
The model stayed frozen and never trained on EmoSet.
This tests whether the learned photo representation carries over beyond its training dataset.

EmoSet validation selected each model's retrieval read-out and the teacher's instruction wording under the existing rule.
Those choices froze before test scoring.
The comparisons used the plain Qwen image and the Qwen image instructed for emotion.
The rule required wins over both comparators in both tasks.

We removed EmoSet photos that were near-copies of Affection train photos.
This reduces overlap with the photo model's training data.
We also scored the set with those photos kept to check how much the filter mattered.

**Did it work.** Yes. The photo model beats both comparators in clustering and retrieval.
EmoSet is the only test set in this round that shaped no earlier choice.
The result covers emotion on this additional photo set, rather than other attributes or photo datasets.

The near-copy filter matters for the size of the clustering difference over plain Qwen.
Keeping near-copies reduces that difference, although all comparisons still count as wins.

**Key evidence.** The filtered test contains 17,348 photos.
The intervals below decide whether the photo model beats the teacher.

| Emotion task | Photo model | Plain Qwen image | Teacher | Photo minus teacher, 95% interval |
|---|---|---|---|---|
| Clustering, AMI | 0.4571 | 0.3918 | 0.4390 | +0.0181 [+0.0121, +0.0242] |
| Retrieval, mAP | 0.4838 | 0.3864 | 0.4519 | +0.0320 [+0.0295, +0.0347] |

Both intervals stay above zero.
The filter changes the size of the clustering difference:

| EmoSet test version | Photo minus plain Qwen, AMI |
|---|---|
| Near-copies removed | +0.0653 |
| Near-copies kept | +0.0390 |

The filter removes 368 test photos.

> [Full report: Results](../reports/auto/percept/2027-01-02_backbones_sweep.md#results) for the claim; [Analysis](../reports/auto/percept/2027-01-02_backbones_sweep.md#analysis) for the near-copy check.

## The analysis shows which parts matter, without selecting a new setting

**Problem.** The claim tests do not explain which recipe parts matter or how scores change with training settings.

**Idea.** We swept the painting model's training settings on validation data.
These include loss weights, graph settings and model settings.
We also tested the photo model's training settings and fixed training lengths on Affection validation.
These include content correction, the recipe's correction for image content.
These sweeps measure sensitivity around the frozen models.

Separately, we added retrieval scores to existing ablations, which remove individual training parts.
These use held paintings and compare each removal with the unchanged painting model.
We compared each result with a reference on the same machine to limit device effects.

The ablations test graphs built from instructed features and the relational term, which copies the teacher's pattern of similarities between paintings.
They also test other parts of the recipe.
The analysis explains the frozen models; it cannot name a setting better than either model's settings.

**Did it work.** The analysis is complete and has no verdict.
Removing instructed graphs produces the largest style clustering drop among the tested removals.
Removing the relational term produces the largest genre clustering drop and also reduces genre retrieval.
Other removals reduce style clustering too, so no single part carries style.
The effects of training settings vary by aspect and task.

**Key evidence.** These are changes from the unchanged model on the same machine, in the named metric's units.

| Change tested | Data and metric | Comparator | Score change |
|---|---|---|---|
| Remove instructed graphs | Held paintings, style AMI | Painting model | -0.1903 |
| Remove relational term | Held paintings, genre AMI | Painting model | -0.0440 |
| Remove relational term | Held paintings, genre mAP | Painting model | -0.1175 |
| Turn content correction off | Photo validation, emotion AMI | Photo model | -0.0320 |

These comparisons explain score changes and select nothing.

> [Full report: Analysis](../reports/auto/percept/2027-01-02_backbones_sweep.md#analysis), including all setting curves, ablations and photo training lengths.

## Our advice

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** Freeze the painting model, the photo model and the claims, and write for the submission deadlines.
The recipe carries over across the tested encoder families, within the limit of their earlier selection.
EmoSet provides the strongest photo evidence so far because its test set shaped no earlier choice.
Keep the failed teacher comparison visible.
Use the analysis sheets for the paper's ablations and analysis.

</details>

## Next

The encoder and EmoSet claim tests are complete and hold under the rule.
The analysis selects no replacement for either frozen model.
Your next decision is which direction to take after giving your reading of these results.
The full report offers these options:

1. Freeze and write. This needs no new compute and fixes the claims for the abstract and paper.
2. Test PolyArt when you supply its link. Data preparation and a scoring round would test the painting model on another art source.
3. Test an encoder that no earlier held test selected. This would examine whether the encoder selection shaped the result.

## Glossary

| Plain name | Meaning | Name in the full report |
|---|---|---|
| Painting model | Network trained on ArtELingo paintings; only the plain image embedding enters at test | c* |
| Photo model | Painting model fine-tuned on Affection train photos without emotion labels | photo model |
| Recipe | Training loss terms and their weights | c*'s recipe |
| Plain image | Encoder's image embedding without an aspect instruction | plain image; frozen image; plain Qwen |
| Teacher | Qwen image embedding instructed to focus on an aspect | teacher; instructed Qwen image |
| Matched PercepT | PercepT comparator trained on the same encoder's image features | matched PercepT |
| Read-out | Method used to compare saved embeddings for retrieval | read-out |
| Clustering score | Agreement between clusters and labels, adjusted for chance | AMI |
| Retrieval score | Mean average precision; how high images with the same aspect label rank | mAP |
| Relational term | Loss that copies the teacher's pattern of similarities between paintings | relational term |
| Content correction | Recipe correction for image content | content correction |
| Seed range | Highest minus lowest validation score across model seeds for the relevant metric | own range; seed range |
| Win | Difference interval's lower bound is above zero | WIN |
| Level | Lower bound is above minus the model's validation seed range, without a win | LEVEL |
| Below | Comparison meets neither the win nor the level rule | BELOW |
| Claim test | Test whose result supports the paper's stated claim | claim test |

