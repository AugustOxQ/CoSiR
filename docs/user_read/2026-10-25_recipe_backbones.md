# SigLIP 2 and Qwen pass the held emotion test; CLIP B/32 stays inconclusive

> 2026-10-25 · [Full report 12-24 (CoSiR numbers its reports in sequence, not by date)](../reports/auto/percept/2026-12-24_recipe_backbones.md)

## Summary

- Removing terms that align each image with its caption won on fresh development paintings, with the teacher at one fifth of the loss.
- The pick used one training seed; the recipe with only image pairs and the teacher came close.
- K-means groups similar embeddings into clusters. On held paintings, reserved for evaluation, SigLIP 2 and Qwen passed at both training sizes (k-means clusters); CLIP B/32 stayed inconclusive.
- Every model we train still needs Qwen3-VL-Embedding, a vision-language embedding model, during training.
- No model established an emotion lead on ArtELingo-28; its small test has wide intervals.

## How we got here

CoSiR seeks emotion, style and genre clusters in paintings. Each is an aspect.
The model reads a default image, an embedding made without an aspect instruction.
An instructed image is an embedding made with an aspect instruction.
Qwen3-VL-Embedding supplies these instructed images for training graphs and the teacher, the target the model learns to match.
Training graphs link paintings that are close in an instructed view.

PercepT is the published topic method we compare against; PercepT core is our reimplementation.
Our previous model passed both held emotion claim tests against PercepT core.
It stayed below the instructed Qwen image on style and genre.

The earlier ablation found gains without own-caption terms, which align each image with its own caption.
That change also raised the teacher share, the teacher's fraction of total loss weight.
Those development paintings had also helped choose our previous model.
On 2026-10-25, this round checked recipes on fresh development paintings, then tested other image encoders and ArtELingo-28.

> Context: [What ran](../reports/auto/percept/2026-12-24_recipe_backbones.md#what-ran).

## Picking the recipe on fresh development paintings

**Problem.** The earlier caption removal changed both the loss terms and their weights.
We needed a recipe before the held tests, plus a check of genre information in the default image.

**Idea.** We compared our previous model with five candidates on development paintings that had never chosen it.
A recipe means loss terms and weights; candidates changed caption terms, teacher share, or graph endpoints.
The recipes with only image endpoints keep image-to-image graph terms and the teacher.
They have no caption network, the part that reads captions.

We scored emotion, style and genre with AMI, adjusted mutual information, which measures agreement between clusters and labels.
A seed range is our previous model's AMI spread over three seeds, used as the unit for the development rule.

On at least one aspect, a candidate needed a gain of at least one seed range with an interval above zero.
No aspect could lose more than one seed range.
Among candidates that passed, the largest mean gain in seed ranges chose the recipe.

The ceiling probe checked how much instructed structure a mapping could predict from the default image.
It predicted each instructed Qwen image from the default one with ridge, regularized linear regression, and a multilayer network.

The probe stopped on regression loss without labels.
It then measured cluster agreement on fresh development paintings and decided nothing.

**Did it work?** Yes, under the development rule: removing own-caption terms at teacher share one fifth won.
This used one training seed on development data only.
The winner was picked among five candidates, which can inflate its apparent gain, and the runner-up was close.
Both recipes with only image endpoints also passed.
Raising the teacher share in the previous recipe failed the style and genre guards, the rule's limits on losses.
Caption removal at the larger share had no qualifying gain.

**Key evidence.** The pick used 29,232 fresh development paintings.
The table gives AMI changes against our previous model, with 95% intervals.
Genre rose, but its gain stayed below its seed range of 0.0128.

| Aspect | Pick minus previous model | 95% interval |
|---|---:|---|
| Emotion | +0.0061 | [+0.0046, +0.0076] |
| Style | +0.0240 | [+0.0216, +0.0261] |
| Genre | +0.0070 | [+0.0039, +0.0102] |

The pick's mean gain was 1.60 seed ranges.
The runner-up, with only image endpoints at the same teacher share, reached 1.55.

The ridge probe, the linear version of that check, reached the instructed image's genre AMI level on these development paintings.

| Genre representation | AMI |
|---|---:|
| Ridge prediction from the default image | 0.5040 |
| Instructed Qwen image | 0.4989 |
| Default Qwen image | 0.4623 |
| Previous model | 0.4604 |

> Full report: [Part 1 pick](../reports/auto/percept/2026-12-24_recipe_backbones.md#part-1-pick) and the probe in [Analysis](../reports/auto/percept/2026-12-24_recipe_backbones.md#analysis).

## Testing other image encoders against PercepT core

**Problem.** The held emotion lead with Qwen left other image encoders untested.
The new recipe also needed a held Qwen test, separate from the development choice.

**Idea.** We trained students with SigLIP 2 and CLIP B/32 image inputs, plus a Qwen student.
All three use the picked recipe.
A student learns from Qwen's graphs, teacher and captions, while its image side reads its encoder's default embedding.
We tested each at sample and full training sizes, on the same held paintings.

The bar was the strongest PercepT core entry, including a matched arm.
A matched arm retrains PercepT's second stage on the student's image encoder, keeping the Qwen topics and settings.
For every student's held test, PercepT core on the Qwen image was strongest.

You made these four choices in the design chat, after you saw our recommendation:

- The image encoders to test.
- Qwen captions for every student.
- The strongest PercepT entry as the bar, including the matched arm.
- Full training unless the sample result showed no gain at all.

The tests averaged five training seeds.

A pass required an emotion AMI gain of at least 0.0100 and an interval wholly above zero.
The held Qwen tests evaluated the development pick; they did not choose between Qwen recipes.

**Did it work?** SigLIP 2 and Qwen passed at both training sizes.
CLIP B/32 was inconclusive at both sizes; that is not a fail.
Every student still needed Qwen3-VL-Embedding's instructed graphs, teacher and default captions during training.

The emotion claims cover k-means only.

Leiden, a graph clustering method checked beside k-means, gave full-scale emotion intervals that include zero for SigLIP 2 and Qwen.

These passes cover neither style and genre claims nor retrieval, other datasets, untested encoders, or PercepT with its own encoders.

**Key evidence.** Both sizes used the same 12,281 held paintings.
Gain means student AMI minus the strongest PercepT core AMI.

| Training size | Training paintings | PercepT core on the Qwen image, AMI |
|---|---:|---:|
| Sample | 13,400 | 0.0855 |
| Full | 42,969 | 0.0874 |

| Student and training size | AMI | Gain [95% interval] | Outcome |
|---|---:|---|---|
| SigLIP 2, sample | 0.1020 | +0.0165 [+0.0110, +0.0177] | Passed |
| SigLIP 2, full | 0.1024 | +0.0150 [+0.0096, +0.0163] | Passed |
| CLIP B/32, sample | 0.0901 | +0.0046 | Inconclusive |
| CLIP B/32, full | 0.0885 | +0.0011 | Inconclusive |
| Qwen, sample | 0.1045 | +0.0190 [+0.0137, +0.0201] | Passed |
| Qwen, full | 0.1029 | +0.0154 [+0.0100, +0.0169] | Passed |

Both CLIP B/32 gains fell below the required size, so their intervals do not decide the outcome.

> Full report: [Part 2 held emotion](../reports/auto/percept/2026-12-24_recipe_backbones.md#part-2-held-emotion-stages-3-and-4) and [full-scale diagnostics](../reports/auto/percept/2026-12-24_recipe_backbones.md#stage-4-reported-readings).

## Checking the lead on ArtELingo-28

**Problem.** The held passes concern ArtELingo and leave the lead on ArtELingo-28 untested.

**Idea.** We tested every student and the previous Qwen recipe on ArtELingo-28 with full-scale weights.
The check used its own images and English-translated captions, against each model's strongest PercepT comparator.
SigLIP 2's matched arm set its bar; PercepT core on the Qwen image set the other bars.

To say the lead holds required a positive gain with the interval above zero.
A positive gain with an interval reaching below zero received the outcome "qualified".
We also reported separate results for paintings shared with ArtELingo and paintings outside our store, the project's saved painting data.
All intervals in both subsets included zero.

**Did it work?** No model established that its lead holds.
Both Qwen recipes and SigLIP 2 were qualified: positive gains, with intervals reaching below zero.
CLIP B/32 was below PercepT core here.
That result is specific to ArtELingo-28; its earlier held result remains inconclusive.
This test is small, with wide intervals.

**Key evidence.** The check used 746 paintings.
Gain uses each model's strongest comparator.

| Model | Emotion AMI | Gain [95% interval] | Outcome |
|---|---:|---|---|
| Previous Qwen recipe | 0.1344 | +0.0067 [-0.0200, +0.0116] | Qualified |
| SigLIP 2 | 0.1536 | +0.0244 [-0.0044, +0.0271] | Qualified |
| CLIP B/32 | 0.1198 | -0.0079 | Below PercepT core |
| Qwen, picked recipe | 0.1443 | +0.0167 [-0.0103, +0.0212] | Qualified |

> Full report: [ArtELingo-28 result](../reports/auto/percept/2026-12-24_recipe_backbones.md#stage-5-artelingo-28) and [reported context](../reports/auto/percept/2026-12-24_recipe_backbones.md#artelingo-28-reported-context).

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: write the narrow emotion claim, then test genre losses on development data.
SigLIP 2 meets the framing condition, the requirement for a claim beyond Qwen: a SigLIP 2 or CLIP student must beat PercepT core.
The claim covers Qwen and SigLIP 2 image encoders on the held ArtELingo paintings, with k-means clusters.
It still requires Qwen3-VL-Embedding at training.
CLIP B/32 remains inconclusive, and ArtELingo-28 gives only qualified support.

Our reading: removal at the lower teacher share gains on emotion and style; genre rises by less than one seed range.
Raising the share alone failed the style and genre guards.
Removal at the larger share gave no qualifying gain.
This separates the earlier changes only in that setting.

Our reading of the probe is that the default image holds genre information, while the losses limit its use.
That supports a development test of heads or losses that preserve the teacher's structure more closely.

</details>

## Next

All results are in for the next chat, where you give your reading first.
Then you choose the next direction from the full report's options:

a. Test heads or losses that follow the teacher more closely on development data, to check whether they preserve genre information.

b. Write the narrow emotion claim for the abstract, with its Qwen training requirement and inconclusive CLIP B/32 result.

> Full report: options in [Results brief](../reports/auto/percept/2026-12-24_recipe_backbones.md#results-brief).

## Glossary

- Previous model: D2, the recipe used as the development comparator; K1 is the previous Qwen model in the later tests.
- Picked recipe: R6-5, no own-caption terms with teacher share one fifth; K4 is its Qwen student.
- SigLIP 2 student: K2; the student whose image input comes from SigLIP 2.
- PercepT core on the Qwen image: B1; the strongest held comparator.
- SigLIP 2 matched arm: M2; PercepT retrained on SigLIP 2 images, keeping Qwen topics and settings.
- Passed at full scale: S2-GO; gain meets the emotion threshold and its interval stays above zero.
- Qualified on ArtELingo-28: A-QUAL; positive gain whose interval reaches below zero.
- Encoder choice: D8; SigLIP 2 then CLIP B/32.
- AMI: adjusted mutual information, agreement between clusters and labels.
- Seed range: the previous model's AMI spread over three seeds.
- Own-caption terms: align each image with its own caption.
- Teacher share: the teacher's fraction of loss weight.
- Held paintings: reserved for evaluation.
- Default and instructed image: embeddings without or with an aspect instruction.
- K-means and Leiden: the main clustering method and the graph clustering check.
