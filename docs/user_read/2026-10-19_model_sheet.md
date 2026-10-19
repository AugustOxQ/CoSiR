# One shared model reaches the instructed Qwen image on emotion, but not on style or genre

> 2026-10-19 (Amsterdam). Full report:
> [stage 2 model sheet](../reports/auto/percept/2026-12-16_model_sheet.md). It follows the
> [PercepT core K = 40 report](2026-10-19_percept_core_k40.md), which set the two comparators.

## Summary

- The emotion line for 2026-10-29 holds on development data. Several settings keep emotion above PercepT core + 0.01.
- On emotion, four settings are above the instructed Qwen image (our target), one of them only by 0.0001. On
  style and genre, no setting reaches it.
- Distillation (a pull toward the instructed image) helps. Its two forms help different aspects, and no setting
  gets the best of both yet.
- One shared layer for the three aspects costs nothing. It is level with or above three separate networks.
- With the instructed images as input (three passes), our model still stays below the target on style and genre.
- You give your reading of this sheet. Then we choose between a second short loop for style and genre and the
  claim test (the test on held data for 2026-10-29).

## The sheet at a glance

Image-side AMI ↑ (k-means with 20 clusters on 5,000 development paintings; genre on its 4,053 labelled ones). Mean
over 3 model seeds where the Seeds column says 3. Bold: best default-input value per column.

| Group | Method | Seeds | Emotion ↑ | Style ↑ | Genre ↑ |
|---|---|---:|---:|---:|---:|
| Comparators | PercepT core, best setting per aspect | 3 | 0.077 | 0.229 | 0.436 |
| | Image encoder clustered directly, best per aspect | — | 0.070 | 0.300 | 0.449 |
| | **Target:** instructed Qwen image clustered directly | — | 0.093 | 0.395 | 0.496 |
| | Linear probe (uses the labels; a ceiling) | — | 0.121 | 0.527 | 0.657 |
| Earlier | Stage 1 network | 3 | 0.081 | 0.348 | 0.458 |
| This sheet | Three separate networks, new graph picks | 3 | 0.089 | 0.354 | 0.457 |
| | One shared layer (the base setting) | 3 | 0.093 | 0.359 | 0.460 |
| | Shared layer + direct distillation, weight 0.5 | 3 | **0.096** | 0.370 | 0.454 |
| | Shared layer + relational distillation | 3 | 0.088 | **0.370** | **0.489** |
| | Distillation only, no graphs | 1 | 0.082 | 0.275 | 0.437 |
| Second result | Instructed inputs (three passes) | 1 | 0.095 | 0.384 | 0.485 |

Against the target, the best default-input values are +0.003 on emotion, −0.025 on style and −0.007 on genre.
Seed spreads of the shared-layer setting: 0.001 (emotion), 0.007 (style), 0.005 (genre). The full sheet has 13
settings; the table shows the ones that matter.

## How we got here

This project builds a model that finds aspects of paintings (emotion, style, genre) without conditions. It aims to
improve over PercepT, a published method for perceptual topics. The test for 2026-10-29: emotion at least 0.01 above
PercepT core on held data.

The stage 1 network (2026-10-18) trains one small network per aspect on "buddy graphs": pairs of paintings that are
near neighbours in an aspect view. It reached 0.081 / 0.348 / 0.458. On 2026-10-19 you set a harder target for
stage 2. One pass on the default Qwen image should reach the instructed Qwen image clustered directly. "Instructed"
means Qwen reads the image with an instruction that names the aspect, so three aspects need three passes.

This run tried the parts that could close that gap, on one number sheet. All numbers come from development data:
5,000 paintings of the training part of the dataset, set aside for scoring. The held test paintings were not read.

## Does one shared layer work as well as three separate networks?

**Problem.** Three separate networks mean three models. One model with a shared first layer is simpler, if it costs
nothing.

**Idea.** Share the first layer of the network between the three aspects. Each aspect keeps its own second layer
(its head). The comparison is three separate networks on the same graphs, with only the sharing removed. We also
picked each aspect's graph by the score of the image outputs this time, because the claim is about images.

**Did it work?** Yes. The shared layer is above the separate networks on all three aspects. On style and genre the
difference is within one seed spread.

**Key evidence.** Shared layer against separate networks: +0.004 emotion, +0.005 style, +0.003 genre (3 seeds each).
The new emotion graph gives most of the emotion gain over stage 1. On stage 1's emotion graph, the shared layer
falls back to 0.078.

> Full report: "Results", the per-aspect tables, and "Analysis".

## Do the two other loss forms help?

**Problem.** The design left two other ways to train three heads together. Form b adds other-aspect neighbours as
extra negatives. Form c uses one output with a learned mask per aspect.

**Idea.** If a head learns to push away neighbours that are near only in another aspect, it could separate the
aspects better.

**Did it work?** Partly, and not clearly. Form b is above the base setting on emotion and genre, on one seed only.
Form c is above on emotion and below on style. The masks of form c overlap a lot: each pair of aspects shares
about 1,500 of the 2,048 output dimensions. So the masks moved little toward a split by aspect.

**Key evidence.** Change against the base setting (emotion / style / genre): form b +0.003 / −0.003 / +0.007 (1
seed); form c +0.002 / −0.009 / +0.003 (3 seeds). Form b on style and form c on genre are within the base setting's
seed spread. Form b's result started a planned rerun of the best distillation
setting on form b. That rerun (one seed) did not change the picture.

> Full report: "Analysis", loss forms and the rerun.

## Does distillation toward the instructed image close the gap?

**Problem.** The instructed image clusters better than our network on all three aspects. Our model must read the
default image only.

**Idea.** During training, pull each head's output toward the instructed image of its aspect. At test time the model
still reads the default image once. We tried two forms of distillation:

- **Direct:** each painting's output should point at its own instructed embedding. Three weights: 0.5, 1, 2.
- **Relational:** the similarities between paintings in a batch should match the similarities of the instructed
  embeddings. It copies the structure, not the coordinates.

A control removes the graphs and keeps only the direct pull.

**Did it work?** Partly. Each form helps two aspects and hurts the third. Neither reaches the target on style or
genre. Distillation alone is far weaker. That control drops the graphs and the caption terms together, so together
they carry most of the result.

**Key evidence.** Change against the base setting (emotion / style / genre, 3 seeds):

| Setting | Emotion | Style | Genre |
|---|---:|---:|---:|
| Direct, weight 0.5 | +0.003 | +0.011 | −0.006 |
| Direct, weight 1 | +0.003 | −0.005 | −0.018 |
| Relational, weight 1 | −0.005 | +0.011 | +0.029 |
| Direct only, no graphs (1 seed) | −0.011 | −0.084 | −0.024 |

Direct weight 1 on style is within the base setting's seed spread (0.007). A higher direct weight is worse, most
on genre. The relational form gives the largest single gain of the sheet:
genre +0.029, close to the target. Its emotion result varies across seeds (spread 0.005).

> Full report: "Analysis", distillation and the diagnostics.

## Does the model reach the target when it gets the instructed images?

**Problem.** If our model falls short on style and genre, is the default input the limit?

**Idea.** Give the instructed images and captions as input (three passes) to one setting: relational distillation
on form b, the best mean over the three aspects on the first seed. This result does not carry the claim; it shows
what the training can reach with the best input.

**Did it work?** No, on style and genre. With instructed inputs the model is still 0.012 below the target on style
and 0.010 below on genre. On emotion it is above (one seed).

**Key evidence.** Instructed inputs against the same setting on default inputs: +0.003 emotion, +0.014 style, +0.001
genre. So the input explains only part of the style gap and almost none of the genre gap.

> Full report: "Analysis", the second result.

## Advice

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our reading.** The plan named four possible outcomes; the sheet gives "the target holds on some aspects only", here
emotion. The parts that help differ by aspect. Direct distillation at weight 0.5 helps emotion and style. Relational
distillation helps style and genre. The
instructed-input result says the training, not only the input, limits style and genre. We are fairly sure about the
direction of each part. The sizes come from one development set that also chose the settings, so they are
optimistic.

**Our recommendation.** One short second loop for style and genre, then the claim test. The loop would try
relational distillation alone, at weights 0.5 and 2, and together with direct distillation at 0.5. Cost: a small
build and about 30 minutes on the GPU, inside the 2026-10-23 time-box. Genre (0.007 short) may close; style (0.025
short) probably will not. For the 2026-10-29 emotion test, the shared layer with direct distillation at 0.5 is our
first choice (+0.019 over PercepT core on development data).

</details>

## Next

- **Where things stand:** the emotion line for 2026-10-29 holds on development data. The instructed-image target holds
  on emotion only.
- **Your decision now:** your reading of this sheet. Then the next step:
  1. a second short stage 2 loop for style and genre;
  2. the claim test spec now, with the style and genre target left for 2026-11-10;
  3. both: the claim test spec, and the short loop in the same night as its build.
- The next chat ("r10 decide 12") starts with your reading.

## Glossary

- **AMI:** adjusted mutual information between the clusters and the labels; 0 is chance, 1 is a perfect match.
- **Instructed image:** the Qwen3-VL-Embedding of the image with an instruction that names the aspect; "default" has
  no instruction.
- **Buddy graph:** pairs of paintings that are near neighbours in an aspect view; the network trains on these pairs.
- **Shared layer (trunk):** the first network layer, shared by the three aspects; each aspect keeps its own head.
- **Direct distillation:** a training pull from each output toward the same painting's instructed embedding.
- **Relational distillation:** a training pull that matches the similarities between paintings in a batch.
- **Form b, form c:** the other two loss forms: other-aspect negatives (b), one output with a mask per aspect (c).
- **Seed spread:** the standard deviation over three model seeds; the full report calls the base setting's spread the
  "tie band".
