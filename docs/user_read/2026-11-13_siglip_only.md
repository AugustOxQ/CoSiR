# A SigLIP-only CoSiR loses emotion text scores, so it becomes an extra row in the paper

> 2026-11-13 · Full report: [SigLIP-only adapter](../reports/auto/percept/2027-01-18_siglip_only.md)

## Summary

- We trained CoSiR so that texts go through SigLIP 1's own text encoder. Then a text needs no Qwen3 pass at test time.
- On validation data, the image scores held. The text scores fell below the bar that we set before the run.
- So the new model becomes an extra row in the paper. The current model stays the main model.
- The text loss sits almost all in emotion. Style and genre text move little.
- On the test set, training adds over plain SigLIP 1 in 13 of 15 cells. Against the instructed Qwen3, the new model is
  above in most cells but BELOW in all three emotion text cells.
- The SigLIP 1 text pass is about three times faster than the Qwen3 pass.

## How we got here

CoSiR trains c*, a small model on top of frozen encoders. It gives one embedding per aspect: emotion, style and genre.
It has two parts. The image tower maps an image to the three aspect embeddings. The caption tower does the same for a
text. Training pulls these embeddings together along neighbour graphs that we build from Qwen3-VL-Embedding, a large
image and text model.

Since 2026-11-12, the paper's main model reads images through SigLIP 1. But its caption tower still reads each text
through Qwen3. So the paper's claim of a light model is weak for texts.

This round (2026-11-12 to 2026-11-13) asks one question. Can the caption tower read SigLIP 1's text embedding instead
and keep its scores? Before the run, you decided the outcome rule. If it holds on validation, it becomes the main
model. If not, it becomes an extra row.

## Read texts with SigLIP 1 instead of Qwen3

**Problem.** At test time, each text passes through Qwen3: 3.6 ms per caption on one RTX 3090. The image side already
uses only SigLIP 1.

**Idea.** Only the caption tower's input changes. It reads the SigLIP 1 text embedding of the captions instead of the
Qwen3 one. The graphs and the training targets stay on Qwen3. The tower learns its aspect spaces from the graphs, so a
weaker input may cost little. Against this, SigLIP 1's text carries less emotion than Qwen3's.

We first tried two caption forms on one seed: the plain caption, and the caption with the prefix "Represent the
caption: ". The prefix won by a very small margin and went on to five seeds.

**The bar.** On validation data, with five seeds, the new model must keep two group means. Each must stay within one
seed range below the current model's mean:

- the image group: clustering and image retrieval, for the three aspects;
- the text group: retrieval from text to image, image to text and text to text, for the three aspects.

**Did it work?** No. The image group held. The text group fell below the bar in all five seeds.

- The loss is mostly emotion text to text: 0.025 below the current model on validation.
- The new model trained on our local GPU, the current one on the cluster. At one seed, this machine change moves the
  text group by only 0.0001, far less than the gap. We did not measure it on the other seeds.
- The plain caption form ran on one seed only. Its text group is below the bar too.

**Key evidence.** Five-seed means on validation data.

| Group | New model | Current model | Bar |
|---|---|---|---|
| Image group | 0.3882 | 0.3863 | 0.3833 |
| Text group | 0.2549 | 0.2594 | 0.2573 |

> Full report: [results](../reports/auto/percept/2027-01-18_siglip_only.md#results) and
> [analysis](../reports/auto/percept/2027-01-18_siglip_only.md#analysis).

## Score the paper cells on the test set

**Problem.** The paper needs the new model's test scores beside two comparators. The first is plain SigLIP 1, with no
training. The second is the instructed Qwen3, our strongest comparator. This step was planned to run after a fail too.

**Idea.** We scored 15 cells on the test set of ArtELingo, our main dataset of paintings with captions and emotion,
style and genre labels. Each aspect has a clustering cell and four retrieval cells. Each cell compares two models over
five seeds, with a 95% interval for the difference:

- WIN: the whole interval lies above zero.
- LEVEL: no WIN, but the lower end of the interval lies above minus one seed range.
- BELOW: all other cases.

Each interval is for one cell. There is no correction for the many cells.

**Did it work?** Partly. Training adds over plain SigLIP 1 in 13 of 15 cells.

- Style image retrieval is BELOW plain SigLIP 1. This cell reads one joined vector of all three aspects, because plain
  SigLIP 1 has one vector too. The current model is BELOW in this cell too. With the style embedding alone, the new
  model is clearly above.
- Genre text to image stays open. Our scores and an independent re-check both find a WIN. But one number differs by
  slightly more than the allowed tolerance, so the rule leaves the cell open.
- Against the instructed Qwen3, the new model is above in 12 cells. It is BELOW in the three emotion text cells: each
  interval lies wholly below zero. The current model also trails Qwen3 in these three cells, but by less.

**Key evidence.** Outcomes of the 15 test cells.

| Test cells | Against plain SigLIP 1 | Against the instructed Qwen3 |
|---|---|---|
| Clustering, 3 aspects | 3 WIN | 3 WIN |
| Image retrieval, 3 aspects | 2 WIN, style BELOW | 3 WIN |
| Emotion text retrieval, 3 directions | 3 WIN | 3 BELOW |
| Style text retrieval, 3 directions | 3 WIN | 3 WIN |
| Genre text retrieval, 3 directions | 2 WIN, 1 open | 3 WIN |

Emotion text retrieval, test mAP:

| Direction | Plain SigLIP 1 | New model | Current model | Instructed Qwen3 |
|---|---|---|---|---|
| Text to image | 0.211 | 0.262 | 0.264 | 0.268 |
| Image to text | 0.220 | 0.281 | 0.286 | 0.304 |
| Text to text | 0.206 | 0.327 | 0.346 | 0.377 |

> Full report: [test cells](../reports/auto/percept/2027-01-18_siglip_only.md#results) and
> [agent choices](../reports/auto/percept/2027-01-18_siglip_only.md#agent-choices-checked).

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: take option a below. Keep the current model as the main model and add the SigLIP-only model as an extra row.
The rule you set says so, and the paper is due 11-16.

In the paper chat, choose how the paper shows image retrieval against plain SigLIP 1. The joined vector is the fair
form for one plain vector. The aspect embedding alone is a clear WIN. Park option c until after 11-16. Run option b
only if the paper needs the extra row on MMArt or on photos.

</details>

## Next

The round is complete. The bigger second stage (MMArt, photos, other encoders, the ablation) did not run, as the rule
said. The next step is your reading of these results. Then you choose one of three options:

a. Keep the current model as the main model. Put the SigLIP-only model in the paper as an extra row, with its speed
   and its emotion text cost. It needs no runs.
b. Run part of the second stage for the extra row: MMArt, the photo model and the ablation. It costs about 6 GPU-hours
   and tells if the extra row holds outside ArtELingo.
c. Start a new loop to close the emotion text gap with no Qwen3 pass at test time. For example, the caption tower
   could also learn to match the Qwen3 caption embedding. It costs a medium loop, about 1 to 2 GPU-hours on
   validation data.

One more choice is yours, in the paper chat: how the paper shows image retrieval against plain SigLIP 1 (joined vector
or aspect embedding).

## Glossary

- **c\*:** the current CoSiR model; one shared trunk with an emotion, a style and a genre head.
- **Image tower, caption tower:** the parts of c* that map an image or a text to the three aspect embeddings.
- **Current model:** c* on SigLIP 1 images with Qwen3 text input. The full report calls it K0.
- **New model (SigLIP-only model):** c* with SigLIP 1 text input and the prefix. The full report calls it K*.
- **Plain SigLIP 1:** SigLIP 1's own image and text embeddings with no training. The full report calls it P1.
- **Instructed Qwen3:** Qwen3-VL-Embedding under an instruction per aspect, such as "focus only on the emotion". The
  full report calls it Q1.
- **Clustering (AMI):** adjusted mutual information between k-means clusters and the labels.
- **Retrieval (mAP):** mean average precision, where a hit shares the query's label.
- **Joined vector:** the three aspect embeddings of the new model placed side by side as one vector.
- **The bar:** the pass rule on validation data. The full report calls it SC2.
- **Seed range:** the spread between the best and the worst of five seeds on validation data. The bar uses the
  current model's range; each test cell uses the new model's range.
- **WIN, LEVEL, BELOW:** the three outcomes of a test cell (section above).
- **Independent re-check:** Codex recomputes every number from the rule and the data. The full report calls it C13.
