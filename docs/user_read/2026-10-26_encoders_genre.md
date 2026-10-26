# CLIP L/14 and SigLIP 1 each pass the held emotion test; a relational genre loss reaches the instructed image's level on development data

> 2026-10-26 · [Full report 12-25 (CoSiR numbers its reports in sequence, not by date)](../reports/auto/percept/2026-12-25_encoders_genre.md) · Previous briefing: [recipe and backbones, 2026-10-25](2026-10-25_recipe_backbones.md)

## Summary

- CLIP L/14 passes its held emotion test, and SigLIP 1 passes its own. Both passes are k-means only and need Qwen3-VL-Embedding at training.
- CLIP B/16 stays inconclusive; DINOv2-large fails at sample scale. On ArtELingo-28, the small second test, the passes are qualified only.
- Every one of seven students exceeds its own PercepT arm on development emotion.
- A relational genre loss on Qwen reaches the instructed image's level on development data: genre AMI 0.5022 against the recipe's 0.4694 on fresh seeds.
- That genre result is promising only: picked among seven variants on one seed, only this variant on fresh seeds, no held test yet.

## How we got here

CoSiR seeks emotion, style and genre clusters in paintings. Each of the three is an aspect.
The model reads a default image, an embedding made without an aspect instruction.
An instructed image is an embedding made with an instruction for one aspect.
Qwen3-VL-Embedding supplies the instructed images for the training graphs and for the teacher, the target the model learns to match.
A student is our model trained on one image encoder's default image.

PercepT is the published method that finds perceptual topics in paintings; we compare against it.
PercepT core is our reimplementation of it.
The data is ArtELingo, paintings with emotion captions and style and genre labels.
Held paintings are reserved for evaluation; development paintings serve choices and exploration.
ArtELingo-28 is a related set of 746 paintings with captions in other languages, translated to English.

The previous briefing, dated 2026-10-25, picked the current recipe: the loss terms and their weights.
It removes the terms that align each image with its own caption and gives the teacher one fifth of the loss weight.
The Qwen student passed its held emotion test. The SigLIP 2 student passed its own. CLIP B/32 stayed inconclusive.
Every student stayed below the instructed Qwen image on genre.

We decided this round on 2026-10-25 and ran it overnight.
It asks three questions. Do more encoders pass the held emotion test? Does the gain over PercepT hold on every encoder? Can another genre loss lift genre toward the instructed image?

> Context: [What ran](../reports/auto/percept/2026-12-25_encoders_genre.md#what-ran).

## Four more encoders take the held emotion test

**Problem.** The earlier held tests covered three image encoders. Four common ones were untested.

**Idea.** We gave CLIP B/16, CLIP L/14, SigLIP 1 and DINOv2-large the same recipe and the same held emotion rule.
The test uses 12,281 held paintings and five training seeds.
Sample scale trains on 13,400 paintings; full scale trains on 42,969.
A sample-scale failure stops the full-scale test.

We cluster the model's emotion output with k-means (20 clusters) and measure AMI, adjusted mutual information: agreement between clusters and labels.
Each student must beat the strongest PercepT entry, its matched arm included.
A matched arm is PercepT core retrained on the student's own encoder image, with the same Qwen topics and settings.
Here, PercepT core on the Qwen image was the strongest entry in every comparison.
A pass needs a gain of at least 0.0100 with its 95% interval above zero.

Every student still needs Qwen3-VL-Embedding at training, for the instructed graphs, the teacher and the default captions.

**Did it work?** CLIP L/14 passes at both scales. SigLIP 1 passes at both scales.
CLIP B/16 is inconclusive at both scales: positive, but below the required size.
DINOv2-large fails at sample scale and got no full-scale test.

Two caveats hold for both passes.
Leiden, a graph clustering method, is our check. Its emotion intervals include zero for CLIP L/14 and SigLIP 1, and its margin is negative for CLIP B/16.
So the claims cover k-means only.
On ArtELingo-28, CLIP L/14, SigLIP 1 and DINOv2-large are qualified: positive gains with intervals that include zero. CLIP B/16 fails there.
That test has 746 paintings and wide intervals.

**Key evidence.** Gains are against the strongest PercepT entry: 0.0855 at sample scale, 0.0874 at full scale.

| Encoder | Sample-scale gain | Full-scale AMI | Full-scale gain [95% interval] | Held outcome |
|---|---:|---:|---|---|
| CLIP B/16 | +0.0089 | 0.0945 | +0.0070 [+0.0016, +0.0085] | Inconclusive at both scales |
| CLIP L/14 | +0.0138 | 0.0996 | +0.0122 [+0.0068, +0.0136] | Pass at both scales |
| SigLIP 1 | +0.0191 | 0.1036 | +0.0162 [+0.0106, +0.0179] | Pass at both scales |
| DINOv2-large | -0.0041 [-0.0094, -0.0030] | not run | not run | Fail at sample scale |

| Encoder on ArtELingo-28 | Gain [95% interval] | Outcome |
|---|---|---|
| CLIP B/16 | -0.0001 [-0.0259, +0.0077] | Fails |
| CLIP L/14 | +0.0125 [-0.0140, +0.0191] | Qualified |
| SigLIP 1 | +0.0200 [-0.0061, +0.0250] | Qualified |
| DINOv2-large, sample scale | +0.0024 [-0.0227, +0.0069] | Qualified |

> Full report: [Held emotion, per candidate](../reports/auto/percept/2026-12-25_encoders_genre.md#stage-i-held-emotion-per-candidate) and [its diagnostics](../reports/auto/percept/2026-12-25_encoders_genre.md#stage-i-diagnostics-decide-nothing).

## Every student gains over its own PercepT arm on development paintings

**Problem.** The held test uses one common bar, PercepT core on the Qwen image.
It does not say what the student adds over PercepT on the same encoder.

**Idea.** We compared each of the seven students with its own PercepT arm on 29,232 development paintings, at sample scale with five seeds.
The lift is the student's emotion AMI minus its own arm's best reading.
This is a development reading; it makes no held claim.

We also computed statistics of each encoder's default image. These are the frozen input's own AMI, label purity among 20 nearest neighbours, and the cosine silhouette, a measure of label separation.
We then asked which statistic tracks the lift across the seven encoders.

**Did it work?** All seven lifts have intervals above zero.
One caveat: the recipe was picked on these same paintings with the Qwen student, so Qwen's lift is inflated, and so are the correlations.
The held lifts of the same encoders follow the same order (Spearman +0.893), although six of the seven encoders change rank.
SigLIP 1 and SigLIP 2 lead on both, and DINOv2-large is last on both.

**Key evidence.** Each row compares the student with its own PercepT arm on the same development paintings.

| Encoder | Student AMI | Own PercepT arm AMI | Lift [95% interval] |
|---|---:|---:|---|
| Qwen | 0.1050 | 0.0844 | +0.0206 [+0.0171, +0.0223] |
| SigLIP 2 | 0.1042 | 0.0756 | +0.0286 [+0.0249, +0.0302] |
| CLIP B/32 | 0.0905 | 0.0708 | +0.0196 [+0.0163, +0.0209] |
| CLIP B/16 | 0.0952 | 0.0728 | +0.0225 [+0.0195, +0.0235] |
| CLIP L/14 | 0.1031 | 0.0772 | +0.0259 [+0.0229, +0.0271] |
| SigLIP 1 | 0.1073 | 0.0784 | +0.0288 [+0.0250, +0.0307] |
| DINOv2-large | 0.0839 | 0.0722 | +0.0117 [+0.0090, +0.0127] |

The lift does not follow the frozen input's AMI. The style statistics (style is the art-style label) track it best.
Seven encoders are too few to claim a cause, and the style pattern may only separate larger encoders from smaller ones.

| Encoder statistic | Correlation with the emotion lift (Spearman) |
|---|---:|
| Frozen input emotion AMI | +0.036 |
| Own PercepT arm emotion AMI | +0.571 |
| Style purity among 20 nearest neighbours | +0.929 |
| Style cosine silhouette | +0.964 |
| Held sample-scale lift | +0.893 |

> Full report: [Development lift, per encoder](../reports/auto/percept/2026-12-25_encoders_genre.md#stage-ii-development-lift-per-encoder) and [its statistics](../reports/auto/percept/2026-12-25_encoders_genre.md#stage-ii-statistics-descriptive-seven-points).

## A relational genre loss reaches the target on development data and repeats on fresh seeds

**Problem.** The current recipe leaves genre below the instructed Qwen image.
We wanted more genre structure without a loss on emotion or style.

**Idea.** We changed only the genre loss, on Qwen at seed 0, on the development paintings (genre labels cover 23,682 of 29,232).
Five variants first: the teacher weight doubled and quadrupled; a relational term at weights 0.5 and 2; a regression term at weight 1.
Then two extra weights for the strongest method that passed the guards.

- The teacher weight multiplies the loss toward the instructed genre image.
- The relational term makes the student copy the similarities between paintings in the instructed genre image.
- The regression term predicts the instructed genre image from the genre head's output through an adaptor, a small network after the head.

A guard limits the loss on another aspect: emotion and style may fall by at most one seed range.
A seed range is the spread of our previous model's AMI over three seeds, the unit for a small effect. It is 0.0046 for emotion, 0.0082 for style and 0.0128 for genre.
The genre target is the instructed Qwen image's AMI, 0.4989.
The winner, the variant with the highest genre AMI among those that passed the guards, then ran on three fresh seeds beside the recipe.

**Did it work?** Yes, on development data. The relational term at weight 8 reached the target at seed 0 and passed both guards.
On fresh seeds its genre gain repeats, with emotion and style within one seed range of the recipe.
Raising the teacher weight lowered genre AMI.
The regression term left the head unchanged and failed the style guard. The adaptor's own output reached 0.4985, while the head below it stayed at 0.4675. Our reading: the adaptor absorbs the gain.

Four limits hold. The result covers Qwen only.
The winner was picked among seven variants on one seed, on the same development paintings that the fresh seeds reuse.
Only weight 8 ran on fresh seeds, and weight 8 is the highest weight tried, so the best weight is open.
No genre variant has a held test; the result is "promising on development data".

**Key evidence.** Seed-0 genre AMI per variant, against the recipe and the target.

| Change to the recipe | Genre AMI | Passed both guards? |
|---|---:|---|
| Current recipe (reference) | 0.4674 | — |
| Teacher weight doubled | 0.4594 | yes |
| Teacher weight quadrupled | 0.4478 | yes |
| Relational term, weight 0.5 | 0.4935 | no: emotion |
| Relational term, weight 2 | 0.4948 | yes |
| Regression term, weight 1 | 0.4675 | no: style |
| Relational term, weight 4 | 0.4938 | yes |
| Relational term, weight 8 | 0.5018 | yes |
| Instructed Qwen genre image (target) | 0.4989 | — |

Fresh seeds 1 to 3, means over the three seeds:

| Aspect | Relational term, weight 8 | Current recipe | Difference [95% interval] |
|---|---:|---:|---|
| Emotion | 0.1049 | 0.1045 | +0.0004 [-0.0012, +0.0022] |
| Style | 0.3931 | 0.3941 | -0.0010 [-0.0056, +0.0028] |
| Genre | 0.5022 | 0.4694 | +0.0328 [+0.0235, +0.0442] |

> Full report: [Genre variants on Qwen](../reports/auto/percept/2026-12-25_encoders_genre.md#stage-iii-genre-variants-on-qwen-seed-0) and [its diagnostics](../reports/auto/percept/2026-12-25_encoders_genre.md#stage-iii-diagnostics-decide-nothing).

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: first compare the relational term at weights 2, 4, 8 and above on fresh seeds. Then give one fixed variant a held genre test.
The fresh seeds support the gain, but weight 8 is the highest weight tried, so the best weight is open.
A held test of a variant that a later weight replaces would be wasted.

Keep the encoder results as separate claims, one per encoder, each with the Qwen training requirement, the k-means limit and the qualified ArtELingo-28 result.

</details>

## Next

The decide chat comes next. There you give your reading of these results first.
Then you choose among the full report's three options:

a. Give the weight-8 variant a held genre test, with a new rule.

b. Do more development work on the relational term: weights above 8, and the other weights on fresh seeds.

c. Pause genre and write up the CLIP L/14 and SigLIP 1 results.

> Full report: options and our recommendation in the [Results brief](../reports/auto/percept/2026-12-25_encoders_genre.md#results-brief).

## Glossary

- Current recipe: R6-5; the loss without own-caption terms, teacher share one fifth.
- Students on CLIP B/16, CLIP L/14, SigLIP 1, DINOv2-large: K6, K7, K8, K5; their matched arms: M6, M7, M8, M5.
- Strongest PercepT entry: B1-own; PercepT core's own reading on the Qwen image.
- Pass at sample or full scale: S1-GO, S2-GO. Inconclusive: S1-IN, S2-IN. Fail at sample scale: S1-NO.
- Qualified or failed on ArtELingo-28: A-QUAL, A-FAILS.
- Development lift holds: L-HOLDS; lift with its interval above zero.
- Genre target reached at seed 0: III-GO. Repeats on fresh seeds: C-HOLDS.
- Variants: GT (teacher weight), GR (relational term), GA (regression term); GR-8 is the winner. Target: T_g.
- AMI: adjusted mutual information, agreement between clusters and labels.
- Seed range: the spread of the previous model's AMI over three seeds; the unit for a small effect and for the guards.
- Held paintings: reserved for evaluation. Development paintings (P): for choices and exploration.
- Default and instructed image: embeddings without or with an aspect instruction.
- Matched arm: PercepT core retrained on an encoder's image, with Qwen topics and settings.
- Frozen input: an encoder's default image clustered directly, without training (F with the encoder's number).
- Teacher weight: the multiplier on the loss toward the instructed image. Teacher share: that weight divided by the total loss weight.
