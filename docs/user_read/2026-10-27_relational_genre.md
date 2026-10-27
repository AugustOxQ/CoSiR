# A sharper teacher in the genre loss brings held genre level with the instructed Qwen genre image

> 2026-10-27 · [Full report 12-27 (CoSiR numbers its reports in sequence, not by date)](../reports/auto/percept/2026-12-27_relational_genre.md) · Previous briefing: [encoders and genre, 2026-10-26](2026-10-26_encoders_genre.md)

## Summary

- We tried six changes to the relational genre loss on development paintings, at one seed.
- A sharper teacher target won that seed. On three fresh seeds it kept the highest genre AMI of four confirmed variants.
- Its lead over the earlier weight-8 variant is small, inside one seed range: the normal spread of genre AMI between seeds (0.0128).
- We trained it on the full training set and tested it on held paintings. Its genre clusters are level with the instructed Qwen genre image within our margin: 0.5001 against 0.5062.
- It is still slightly below that image, so we may say "level within one seed range", not "matches or exceeds".
- The gain over the current recipe on held paintings, +0.0467, is larger than on development paintings.

## How we got here

CoSiR seeks emotion, style and genre clusters in paintings, each from its own head of one model.
The model reads a default image, an embedding made without an aspect instruction.
An instructed image is an embedding made with an instruction for one aspect, here from Qwen3-VL-Embedding.
The model learns from a teacher: it is pulled toward the instructed image of each aspect.

PercepT is the published method that finds perceptual topics in paintings; we compare against it.
On held paintings, the current recipe beat PercepT on emotion. On genre it stayed 0.0527 AMI below the instructed Qwen genre image.
AMI measures how well clusters agree with labels.

The previous round, on 2026-10-26, added a relational term to the genre loss.
The relational term makes each painting's similarities to the other paintings of the batch copy the teacher's similarities.
At weight 8 it reached the instructed image's genre level on development paintings, and fresh seeds repeated the gain.

You then chose to close the genre gap first, as one loop of three stages.
First a wider search of the relational term on one seed. Then a pick on fresh seeds. Then one held genre test of the pick.
This briefing reports that loop. It ran on the night of 2026-10-27 in under two hours.

> Context: [What ran](../reports/auto/percept/2026-12-27_relational_genre.md#what-ran).

## A wider search of the relational term

**Problem.** Weight 8 was the highest weight we had tried. We did not know which form of the term works best.

**Idea.** We kept the current recipe and changed only the genre loss, one change per variant.
All variants ran at seed 0 on 29,232 development paintings, 23,682 of them with a genre label.

- Higher weights, 16 and 32, with more steps (64, 128, 256) only while genre kept rising.
- The relational term alone, without the direct pull toward the teacher.
- A sharper teacher target: the teacher's similarities at temperature 0.02 or 0.01 in place of 0.05.
- Teacher anchors: 4,096 fixed teacher vectors in place of the other paintings of the batch.

A sharper target makes the teacher's nearest neighbours count more in what the model copies.
Teacher anchors give a larger and fixed set of reference paintings.
A guard blocks a variant that costs emotion or style more than one seed range.
The variant with the highest genre AMI among those that pass the guard wins.

**Did it work?** Yes, at one seed. The sharper target at 0.02 won.
Weight 32 scored below weight 16, so the higher steps did not run.
The sharper target at 0.01 failed the style guard.
The relational term alone and the teacher anchors both lost genre against weight 8.
This is development data at one seed, and the winner was picked among seven variants on the same paintings.

**Key evidence.** Seed-0 genre AMI per variant.

| Change to the recipe | Genre AMI | Passed the guard? |
|---|---:|---|
| Current recipe (reference) | 0.4674 | — |
| Relational term, weight 8 (previous round) | 0.5018 | yes |
| Weight 16 | 0.5022 | yes |
| Weight 32 | 0.4964 | yes |
| Relational term alone, weight 8 | 0.4972 | yes |
| Sharper target 0.02, weight 8 | **0.5074** | yes |
| Sharper target 0.01, weight 8 | 0.4882 | no: style |
| Teacher anchors, weight 8 | 0.4900 | yes |
| Instructed Qwen genre image (reference) | 0.4989 | — |

> Full report: [Stage A](../reports/auto/percept/2026-12-27_relational_genre.md#stage-a-seed-0-development-paintings).

## A pick on fresh seeds

**Problem.** A pick on one seed can be luck. The lower weights 2 and 4 had never run on fresh seeds.

**Idea.** The seed-0 winner and weights 2 and 4 trained on seeds 1, 2 and 3. Weight 8 already had those seeds.
A variant is confirmed when its genre gain over the recipe is at least one seed range, with its interval above zero.
Both guards must also hold.
The confirmed variant with the highest three-seed mean goes to the held test.

**Did it work?** Yes. All four variants are confirmed, and the sharper target has the highest mean.
But the four means lie within 0.0085 of each other, less than one genre seed range (0.0128).
So the pick follows the rule we fixed in advance; the data do not show a clear winner.

**Key evidence.** Genre AMI, means over seeds 1 to 3.

| Variant | Genre AMI | Gain over the recipe (0.4694) [95% interval] |
|---|---:|---|
| Weight 2 | 0.4965 | +0.0272 [+0.0180, +0.0387] |
| Weight 4 | 0.5005 | +0.0311 [+0.0216, +0.0387] |
| Weight 8 | 0.5022 | +0.0328 [+0.0235, +0.0442] |
| Sharper target 0.02, weight 8 | **0.5050** | +0.0356 [+0.0263, +0.0464] |

Emotion and style stay within one seed range of the recipe for all four.

> Full report: [Stage B](../reports/auto/percept/2026-12-27_relational_genre.md#stage-b-seeds-1-to-3-development-paintings).

## The held genre test

**Problem.** Development results choose variants but support no claim. Held paintings test the pick once.

**Idea.** We trained the sharper-target variant at full scale, on all 42,969 training paintings, with five seeds.
We clustered its genre output on 12,281 held paintings and scored the 9,949 with a genre label.
The bar is the instructed Qwen genre image, the strongest genre reading we have.
"Level" means the lower end of the interval is less than one seed range (0.0128) below the bar.
We fixed this margin, and the other outcomes, before any held number existed.

**Did it work?** Yes. The model is level with the instructed genre image within one seed range.
It is still below the image by 0.0060, and the whole interval is below zero.
The test covers Qwen, k-means clustering and ArtELingo only.
The held gain over the recipe, +0.0467, is larger than the development gain, +0.0356.
Our reading, not checked: at full scale the recipe's genre head stops after one epoch, while this model's trains for four to seven.

**Key evidence.** Held genre AMI, five seeds, 9,949 labelled paintings.

| Model | Genre AMI | The new model minus it [95% interval] |
|---|---:|---|
| New model (sharper target 0.02) | 0.5001 | — |
| Instructed Qwen genre image (the bar) | 0.5062 | −0.0060 [−0.0105, −0.0014] |
| Current recipe | 0.4534 | +0.0467 [+0.0402, +0.0531] |
| Strongest PercepT entry | 0.4269 | +0.0732 [+0.0622, +0.0795] |

On held paintings the same model also gives emotion AMI 0.1039 (recipe 0.1029) and style 0.4057 (recipe 0.3996).
We did not compute its emotion margin over PercepT, the bar of the emotion claim.

> Full report: [Stage C](../reports/auto/percept/2026-12-27_relational_genre.md#stage-c-held-paintings-full-scale-5-seeds) and [the held readings](../reports/auto/percept/2026-12-27_relational_genre.md#analysis).

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: next, make this model the recipe and run the held emotion test with it.
A paper needs one model for its emotion, style and genre claims. This test is cheap, about one hour.
We chose this model after we read its held numbers, so the report must say so.

After that, write up the encoder claims and the genre claim for the abstract on 2026-11-10.
We are not sure that the sharper target matters. Its lead over weight 8 is inside one seed range, and weight 8 had no held test.
In the paper, describe the result as the relational term's, with the sharper target as its setting.

</details>

## Next

The decide chat comes next: the chat where you make this loop's decisions. There you first give your reading of these results.
Then you choose among the full report's three options:

a. Make this model the recipe and run the held emotion test with it (about 1 hour).

b. Test the same change on SigLIP 2, SigLIP 1 and CLIP L/14: development first, then held (about half a day).

c. Write up the encoder and genre claims for the 2026-11-10 abstract.

> Full report: options and our recommendation in the [Results brief](../reports/auto/percept/2026-12-27_relational_genre.md#results-brief).

## Glossary

- AMI: adjusted mutual information, the agreement between clusters and labels.
- Default and instructed image: embeddings without or with an aspect instruction.
- Current recipe: R6-5; the loss without own-caption terms, teacher share one fifth. Its held run is K4.
- Relational term: GR-λ in the full report, at weight λ. Relational term alone: GN-8.
- Sharper target: GS-0.02 and GS-0.01, the teacher's temperature. The held model: G.
- Teacher anchors: GQ-8.
- Seed range: the spread of a previous model's AMI over three seeds; 0.0128 for genre. The unit for a small effect and for the guards.
- Outcomes: A-NEW (a new seed-0 winner), B-PICK (a confirmed pick), G-GO (level on held paintings).
- Development paintings (P) serve choices; held paintings serve one final test.
