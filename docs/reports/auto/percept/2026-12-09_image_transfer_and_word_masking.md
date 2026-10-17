# Image-only transfer and emotion-word masking: most of the caption lift survives masking 132 affect words; an image-only student does not learn the caption emotion geometry

> Two night pilots of the r10 decide chat, 2026-10-17 05:30 to 06:45 (Amsterdam). The file name uses CoSiR's
> sequence date. They follow the premise check
> [2026-12-08_prompted_aspect_views_check](2026-12-08_prompted_aspect_views_check.md) and test two of its open points
> before the grill goes on. Decision rule: the bar of the handoff
> ([2026-10-17-r10-decide-4-handoff](../../../superpowers/handoffs/2026-10-17-r10-decide-4-handoff.md), section 4),
> stated before the run. Train side only; labels used only to score. Round folders and logs:
> `src/test/20261017_image_transfer_pilot/` and `src/test/20261017_emotion_word_masking/`.

## Results brief

**What we tried.** Two checks on the prompted Qwen3-VL-Embedding-2B views. First, we masked the emotion words in the
99,999 captions, embedded them again and scored caption emotion AMI. Second, we trained a small image-only network (the
student) toward each painting's emotion-instructed caption embedding, and scored it on 5,000 held-aside training-side
paintings.

**Outcome against the bar.**
- **Word masking:** most of the emotion instruction's caption lift stays.
  - With the label words masked, caption emotion AMI (k = 20) is 0.231; the default instruction on the same text gives 0.095.
  - With a broad list of affect words masked as well, it is 0.187; the default gives 0.069. Unmasked it was 0.278 vs 0.122.
  - On captions that have no affect word at all, it is 0.197 vs 0.070.
- **Image-only student: fails the bar** (image emotion AMI at least 0.12, and above the control student).
  - The student gives 0.074 ± 0.000 over 3 seeds; the control student, trained toward default captions, gives 0.074 ± 0.002.
  - The default image gives 0.070, the emotion-instructed image 0.093, and the teacher 0.217.
  - With the emotion-instructed image as input, the student gives 0.074 and the control 0.075.
- **Neighbour transfer** (no training: the mean teacher of the 20 nearest training images): it keeps a little more. The best image-only view is 0.102 (emotion_focus image neighbours, 20 of them), against 0.083 for the same neighbours with control targets and 0.093 for the emotion_focus image itself. That is still below 0.12.

**What it does not show.** One sample, the 2B model, ArtELingo, the train side. The broad list does not hold every
affect-bearing word ("dark", "death", "smile" stay), so the masked lift can still come from reading affect in the text.
The student had one fixed design (a 2-layer MLP, InfoNCE toward the whole caption vector, 13,400 training paintings);
other targets (an emotion-only residual, buddy pairs) were not tried.

**Options for next.**
- a. **Keep emotion on the caption side** and use the image side for style and content, where the instruction works
  on images (style 0.395). No new run; it changes what the student must show (grill Q4).
- b. **Student for style** instead of emotion: distil the style-instructed views into an image head, and compare with
  prompted Qwen (0.395) and CSD (0.317). About 1 h.
- c. **Student with an emotion-only target** (the teacher minus the control target, or the caption-side buddy pairs).
  About 1 to 2 h; the neighbour transfer above says how much room there is.
- d. **The 8B model** for image emotion: about 3 h.

**Your reading.** <left for the user>

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** The caption result is solid. The emotion instruction makes the embedding read the affect of the
description, and most of it is not the emotion name: with every listed affect word masked, it stays near the
supervised GoEmotions classifier on unmasked text (0.187 vs 0.209). The student result is a clear negative for
emotion. Both students land at the default image's level, and the emotion target adds no cluster gain over the control (1 probe point). The
student learns what captions share with the image, which is mostly content. The emotion part of a caption is not
predictable from the image in this data: the image emotion probe is only 4 to 11 points above the majority class for
every view, the students included. The neighbour transfer shows a small transferable part (about +0.02 AMI over the control targets) that the trained student lost; even so, the best image-only view (0.102) is less than half of the teacher (0.217). Fairly sure of both results (gaps far above the seed sd); the student
result holds only for this design.

**Our recommendation.** Option a for the grill, with b as the next pilot: the paper story becomes "captions give the
emotion aspect, images give the style aspect, one instruction per aspect", and the student must show its worth on
style or on retrieval. Do not spend more night runs on image-only emotion unless the 8B model moves the image probe.
Our view only; decision B is the user's.

</details>

## What ran

- **Data:** the frozen sample of the premise check (99,999 captions, 19,889 paintings; train side, seed 0). The
  5,000-painting image subset is the evaluation set. The other 14,889 paintings train the student, with a fixed 10%
  dev slice (1,489 paintings) for early stopping on the training loss.
- **Masking check:** whole-word, case-insensitive masks that replace each matched word with `[MASK]`.
  - The label mask holds the eight emotion names and their common forms (sad, scary, afraid, angry, awe, content, ...).
  - The broad mask adds 88 common affect words (132 words in all), including the forms of "feel".
  - Both lists were fixed before the run; the exact lists are in `ewm_masks.py`.
  - We re-embedded the masked captions with the default and emotion_focus instructions. A re-encode of 256 unmasked
    captions reproduced the store (cosine 0.9999).
- **Student:** input is the Qwen image embedding (default, or emotion_focus); target is the l2-normalised mean of a
  painting's caption embeddings (emotion_focus = teacher, default = control).
  - Model: Linear 2048 → 1024, GELU, dropout 0.1, Linear → 2048, l2 output.
  - Training: symmetric InfoNCE, temperature 0.05, AdamW lr 1e-3, batch 512, early stop after 15 epochs without a
    better dev loss. Seeds 0, 1, 2. Nothing was tuned.
  - The emotion_focus input needed a new extraction of the 14,889 training images (24 min GPU).
- **Scores:** the conventions of the premise check (`pav_score.py` functions). They include painting AMI at k = 20
  (3 k-means seeds), the grouped emotion probe, top-20 neighbour overlap with the teacher, and single-view buddy-graph
  lift. All four baseline views reproduced the premise check (largest difference 0.001).
- **Who:** Codex (gpt-6.1-sol, high) wrote both checks. The controller reviewed the code, ran both queues (05:40 to
  06:31) and read the numbers below from `itp_results.json` and `ewm_results.json`. The neighbour transfer is
  controller exploration after the bar failed (`itp_knn.py`).

## Results

**Masking.** Caption emotion AMI over all 99,999 captions (mean of 3 k-means seeds; sd at most 0.010). The last
column is the instruction effect (emotion_focus minus default).

| Text | Default k = 9 / 20 / 40 | emotion_focus k = 9 / 20 / 40 | Effect at k = 20 |
|---|---|---|---:|
| Unmasked | 0.098 / 0.122 / 0.129 | 0.305 / 0.278 / 0.264 | +0.156 |
| Label words masked | 0.079 / 0.095 / 0.109 | 0.239 / 0.231 / 0.222 | +0.136 |
| Broad words masked | 0.059 / 0.069 / 0.082 | 0.188 / 0.187 / 0.184 | +0.118 |
| Unmasked, captions with no broad word (59,413) | 0.056 / 0.070 / 0.085 | 0.203 / 0.197 / 0.191 | +0.127 |
| Unmasked, captions with no label word (86,804) | 0.087 / 0.102 / 0.104 | 0.238 / 0.226 / 0.220 | +0.124 |

- **Word shares:** 13.2% of captions hold a label word, and 40.6% a broad word. In 88.0% of the captions with a label
  word, a matched word names the caption's own label. The most frequent words: sad (3,105), content (1,396), scary
  (1,095); for the broad mask: feel (8,956), love (3,177), sad, peaceful (2,554).
- **Mean-caption painting emotion AMI** (5,000 paintings): default 0.107 / 0.103 / 0.094, emotion_focus 0.217 / 0.189
  / 0.166 (unmasked / label / broad). Its genre AMI rises with masking (0.244 / 0.252 / 0.282).
- **Caption emotion probe:** default 56.4% / 52.7% / 49.1%, emotion_focus 58.6% / 54.7% / 51.3% (majority 28.3%).

**Student.** On the 5,000 evaluation paintings; students are mean ± sd over 3 training seeds.

| View | Emotion AMI | Genre AMI | Style AMI | Emotion probe | Overlap with teacher | Emotion lift |
|---|---:|---:|---:|---:|---:|---:|
| Default image | 0.070 | 0.449 | 0.222 | 39.3% | 0.081 | 1.56 |
| emotion_focus image | 0.093 | 0.410 | 0.168 | 40.9% | 0.090 | 1.64 |
| Student, default input | 0.074 ± 0.000 | 0.424 | 0.165 | 45.3% | 0.096 | 1.64 |
| Control student, default input | 0.074 ± 0.002 | 0.405 | 0.165 | 44.4% | 0.091 | 1.58 |
| Student, emotion_focus input | 0.074 ± 0.003 | 0.420 | 0.166 | 46.0% | 0.099 | 1.64 |
| Control student, emotion_focus input | 0.075 ± 0.002 | 0.405 | 0.158 | 44.9% | 0.095 | 1.60 |
| Neighbour transfer, default input, 20 neighbours | 0.091 | 0.427 | 0.188 | 40.2% | — | — |
| The same, control targets | 0.070 | 0.443 | 0.174 | 39.3% | — | — |
| Neighbour transfer, emotion_focus input, 20 neighbours | 0.102 | 0.416 | 0.166 | 41.0% | — | — |
| The same, control targets | 0.083 | 0.436 | 0.163 | 39.8% | — | — |
| Teacher (mean emotion_focus caption) | 0.217 | 0.244 | 0.101 | 56.1% | 1 | 2.37 |
| Control target (mean default caption) | 0.107 | 0.338 | 0.120 | 53.5% | 0.282 | 2.06 |

Majority class 35.2%. Overlap is the share of a painting's top-20 cosine neighbours that are also its top-20 in the
teacher space (chance 0.004). Emotion lift is on the single-view mutual top-20 graph.

| Bar | Student | Control | Outcome |
|---|---:|---:|---|
| Default input: emotion AMI at least 0.12, and above the control by more than the seed sd | 0.074 | 0.074 (gap +0.000, sd 0.002) | fail |
| emotion_focus input: the same | 0.074 | 0.075 (gap −0.001, sd 0.003) | fail |

**Verdict under the rule:** the student premise fails for emotion with this design.

## Analysis

- **The instruction reads affect, not only the emotion name.** Masking lowers both instructions. The instruction
  effect shrinks only from +0.156 to +0.118. On the 59,413 captions with no listed affect word, the effect is +0.127.
  So about three quarters of the lift does not need an affect word from our list.
- **Masking also removes signal from the default.** The default caption AMI falls from 0.122 to 0.069. So part of the
  default's emotion structure was word reading too. The emotion_focus view keeps more of its structure than the default.
- **Masking moves emotion_focus toward content:** its mean-caption genre AMI rises from 0.244 to 0.282.
- **The student learns the shared part of image and caption.** Both students lift the emotion probe by 5 to 7 points
  over the default image. The emotion target adds only 1 point over the control (45.3% vs 44.4%; 46.0% vs 44.9%).
  The cluster AMI does not move. The InfoNCE target is the whole caption vector, and its content part is the part the
  image can predict.
- **The emotion_focus input loses its own gain:** 0.093 as input, 0.074 after the student. The training pulls the
  geometry toward content (genre 0.410 → 0.420).
- **Training fits only partly:** the best dev loss is 4.25 for 1,489 candidates (chance 7.31). The early stop came at
  epochs 10 to 16 (default input) and 32 to 50 (emotion_focus input).
- **Neighbour transfer finds a small emotion part that the student lost.** Averaging the teacher targets of the 20
  nearest training images gives +0.019 to +0.021 AMI over the same neighbours with control targets (sd at most
  0.002). With 100 neighbours the numbers are about the same (0.089 and 0.102). So the image neighbourhoods hold a
  little of the caption emotion structure, and the InfoNCE student does not keep it. The ceiling looks low: the best
  image-only view, 0.102, is near the emotion_focus image (0.093) and far from the teacher (0.217).

## Deviations

- The neighbour transfer (`itp_knn.py`) was not in the brief. It is controller exploration, added after the bar
  failed, to tell a design problem from a missing signal.
- No other deviation from the two Codex briefs. The training images for the emotion_focus input were extracted at 10.2
  images per second (24 min), so the queue ended at 06:31, well before 11:00.

## Files and storage

- **Round folders:** scripts, logs, `itp_tables.md`, `ewm_tables.md` (in git); `itp_results.json`,
  `ewm_results.json`, `ewm_word_statistics.json`, `masked_captions_*.json` (16 MB, gitignored); student outputs in
  `src/test/20261017_image_transfer_pilot/outputs/` (118 MB, gitignored).
- **Store:** `/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/` gained `masked_label/` and
  `masked_broad/` (879 MB each) and `emotion_focus_trainimg/` (about 61 MB). Disk-ledger rows: 2026-10-17.
- Codex's planning notes, check logs and the smoke folders in `/tmp/claude-0/-project-CoSiR/` are listed in
  `.scratch/pending_deletions.md`.
