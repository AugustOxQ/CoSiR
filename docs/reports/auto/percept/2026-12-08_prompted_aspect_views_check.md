# Prompted aspect views: does an aspect instruction move Qwen3-VL-Embedding toward that aspect?

> Premise check of the r10 decide chat, 2026-10-17 05:30 (Amsterdam). The file name uses CoSiR's sequence date. It
> tests the user's idea (one instruction per aspect to an instruction embedder, buddy graphs on top) before the grill
> goes on. Train side only; labels used only to score. Round folder: `src/test/20261017_prompted_aspect_views/` (log,
> `tables.md`, `results.json`). Literature context: [2026-12-07_prompted_aspect_views](../../literature/2026-12-07_prompted_aspect_views.md).

## Results brief

**What we tried.** We gave Qwen3-VL-Embedding-2B ten instructions (the default, and three wordings each for emotion,
genre and style). Then we scored its embeddings of 99,999 ArtELingo captions and 5,000 paintings against the emotion,
genre and style labels. The scores are AMI of k-means clusters, probes and buddy-pair lift.

**Outcome against the bar.** Two of the three bars are met clearly. The third is met on captions, but only weakly on images.
- **Emotion, captions:** caption emotion AMI (k = 20) is 0.278 with the emotion instruction. The default gives 0.122,
  CLIP text 0.047, and the supervised GoEmotions classifier 0.209.
- **Style, images:** image style AMI is 0.395 with the one-word style instruction. The default gives 0.222, CLIP 0.300,
  and the CSD style encoder 0.317.
- **Different partitions:** the partitions of two aspect prompts agree at AMI 0.27 to 0.47. The same view under two
  seeds agrees at 0.61 to 0.69.
- **Emotion, images:** image emotion AMI moves only from 0.069 to 0.093, and the probe only from 39.2% to 40.9%
  (majority class 35.2%).

**What it does not show.** One sample, one model size (2B), one dataset, the train side. Any instruction lifts caption
emotion somewhat (genre and style prompts give 0.17 to 0.22). Captions often name the emotion, so the lift may come
from reading the word (not checked). There is no retrieval measure and no student yet. We compared the prompt wordings
with labels.

**Options for next.**
- a. **Image-only transfer pilot** (the student premise): a small head on image embeddings learns the caption-side
  emotion targets. We compare it with prompted Qwen on the image (0.093) and with the teacher (0.217), and we use a
  control trained toward default caption targets. About 2 h, GPU minutes.
- b. **Emotion words removed from captions:** we embed the captions again with the emotion words masked (2 keys, about
  15 min GPU). It tells us whether the caption lift is more than word reading.
- c. **The 8B model** on the focus and one-word forms: about 4 times slower (about 3 h). It tells us whether a bigger
  embedder moves image emotion.

**Your reading.** <left for the user>

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The user's idea works where the signal is: the instruction makes the embedding geometry follow the
named aspect. Emotion now comes from captions without an emotion-trained encoder, and above one. Style comes from
images, above a dedicated style encoder. This answers the problem the user named ("the emotion credits come from the
emotion encoder"). The probes barely move, so the instruction changes what cosine similarity sees, not what the
embedding contains. That is the part that clustering, buddy graphs and retrieval use. Image emotion stays weak, and
the image probe says there is little linear emotion signal in images to find. So a student that "beats prompted Qwen
on the image" is a real test, not a formality, and its expected gain is small. Fairly sure of the caption and style
results (large gaps, seed sd ≤ 0.015); not sure how much of the caption gain is word reading.

**Our recommendation.** Run (a) tonight, then (b) if time allows; keep (c) for later. If (a) fails, the paper story
leans on captions as the aspect source, with style from images. Our view only; decision B is the user's.

</details>

## What ran

- **Sample:** the earlier check-1 sample (99,999 captions, 19,889 paintings, seed 0, train side) and a fixed
  5,000-painting image subset (seed 0, chosen without labels).
- **Instructions** (they replace the default system prompt "Represent the user's input."; the same text for images
  and captions):
  - `_focus`: "Represent the <aspect> of this artwork, or of this description of an artwork. Focus only on the <aspect>."
  - `_word`: "Describe the <aspect> of this artwork in one word." (the DIOR form; it still returns an embedding).
  - `_neg`: "Represent the <aspect> of this artwork, not its <other> and not its <other>."
  - Exact strings: `pav_prompts.py`.
- **Measures:**
  - shift (cosine to the default);
  - caption emotion AMI at k = 9 / 20 / 40;
  - painting AMI at k = 20 for the image and the mean caption (emotion is the majority label; genre only where known);
  - probes (standardised logistic regression, C = 0.1, GroupKFold by painting);
  - pairwise AMI between partitions;
  - buddy-pair lift on mutual top-20 cosine graphs.
- **Reference rows:** CLIP B/32, CSD, VGG Gram and GoEmotions. They reproduce the earlier check (CLIP text 0.037 /
  0.047 / 0.052; GoEmotions 0.197 / 0.209 / 0.199).
- **Who:** Codex (gpt-6.1-sol, medium) wrote the scripts, and the controller reviewed them. The controller ran them
  (02:35 to 05:20) and read the key numbers from `results.json`.

## Results

| View | Caption emotion AMI k = 9 / 20 / 40 | Image AMI k = 20: emotion / genre / style | Mean-caption AMI k = 20: emotion / genre / style |
|---|---|---|---|
| CLIP B/32 | 0.037 / 0.047 / 0.052 | 0.065 / 0.412 / 0.300 | 0.067 / 0.330 / 0.110 |
| CSD (style encoder) | | 0.044 / 0.254 / 0.317 | |
| GoEmotions (supervised) | 0.197 / 0.209 / 0.199 | | |
| Qwen default | 0.098 / 0.122 / 0.129 | 0.069 / 0.449 / 0.222 | 0.107 / 0.338 / 0.120 |
| emotion_focus | 0.305 / **0.278** / 0.264 | **0.093** / 0.410 / 0.168 | **0.217** / 0.244 / 0.101 |
| emotion_word | 0.292 / 0.274 / 0.261 | 0.094 / 0.414 / 0.180 | 0.211 / 0.247 / 0.106 |
| emotion_neg | 0.249 / 0.254 / 0.243 | 0.077 / 0.445 / 0.188 | 0.185 / 0.290 / 0.117 |
| genre_focus | 0.184 / 0.188 / 0.185 | 0.069 / 0.482 / 0.257 | 0.139 / 0.336 / 0.130 |
| genre_word | 0.149 / 0.169 / 0.159 | 0.070 / **0.496** / 0.264 | 0.138 / 0.353 / 0.140 |
| genre_neg | 0.189 / 0.198 / 0.195 | 0.069 / 0.474 / 0.251 | 0.141 / 0.344 / 0.133 |
| style_focus | 0.180 / 0.176 / 0.174 | 0.062 / 0.410 / 0.356 | 0.143 / 0.339 / 0.134 |
| style_word | 0.236 / 0.215 / 0.209 | 0.063 / 0.349 / **0.395** | 0.178 / 0.310 / 0.135 |
| style_neg | 0.189 / 0.189 / 0.183 | 0.062 / 0.405 / 0.356 | 0.138 / 0.332 / 0.130 |

Seed sd is at most 0.028 for caption AMI and at most 0.020 for painting AMI. Full sd values are in `tables.md`.

| Measure | Default | Best aspect prompt | Reference |
|---|---|---|---|
| Probe, image emotion (majority 35.2%) | 39.2% | 40.9% (emotion_focus) | CLIP 39.2% |
| Probe, image style (majority 15.1%) | 57.8% | 62.9% (style_word) | CSD 62.0% |
| Probe, caption emotion (majority 28.3%) | 56.4% | 58.6% (emotion_focus) | GoEmotions 51.6% |
| Strict buddy graph, style lift | 4.88 | 7.56 (style_word) | CLIP image 6.49 |
| Mean-caption buddy graph, emotion lift | 2.06 | 2.37 (emotion_focus) | CLIP text graph 1.81 |
| Image partition AMI vs default | 1 | 0.33 (style_word), 0.47 (emotion_focus) | seed rerun 0.61 |

## Analysis

- **Is the shift toward the named aspect?** It is. The named aspect rises and the other aspects fall. For example,
  emotion_focus lowers mean-caption genre AMI from 0.338 to 0.244, and style_word lowers image genre AMI from 0.449 to
  0.349.
- **The default Qwen view is content-led:** genre_focus partitions agree with the default at AMI 0.64, near the seed
  level.
- **Wording:**
  - The one-word form is best for style.
  - The focus and one-word forms tie for emotion.
  - Negation is the weakest emotion form (caption AMI 0.254 vs 0.278; image 0.077 vs 0.093), as NegBench predicts.
  - Forms within an aspect agree with each other (partition AMI 0.52 to 0.60). So the choice of form changes little,
    and the label use in that choice matters little.
- **Shift size does not predict effect:** the one-word forms move images most (cosine to the default 0.41 to 0.46),
  while style_focus moves them least (0.82), yet style_focus still lifts image style to 0.356.
- **Geometry, not content:** the probes change by 1 to 5 points, but cluster AMI changes by up to 0.17. The instruction
  reorders what dominates cosine similarity, and the linear information stays about the same.
- **Image emotion is the hard part.** The image probe sits 4 to 6 points above the majority class for every view,
  CLIP included.

## Deviations

None from the brief that Codex received. The waiter and the queue ran as planned. The scoring needed 30 min, inside
the 45 min estimate.

## Files and storage

- Store: `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/<key>/{img.npy,txt.npy,ids.npz}`
  (4.6 GB; a disk-ledger row was added, and the decision is deferred to the user's next storage summary).
- Round folder: scripts, `sample.npz` (67 MB, not in git), `results.json` (5.7 MB, gitignored), `tables.md`, the log.
- Small leftovers (Codex's fake fixtures, planning notes, the GPU smoke output) are listed in
  `.scratch/pending_deletions.md`.
