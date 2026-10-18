# Purified buddy pilot: caption emotion buddy pairs lift an image-only student to 0.094 emotion AMI (control 0.070), below the 0.12 bar; removing content edges adds little

> 2026-10-18 (Amsterdam), run on CPU, queue ended 15:49. The file name uses CoSiR's sequence date. Spec: none (a
> pilot of the r10 decide chat, agent default design; it follows the failed student of
> [2026-12-09_image_transfer_and_word_masking](2026-12-09_image_transfer_and_word_masking.md)). Decision rule: the bar
> stated before the run in the run log. Run log and tables: `src/test/20261018_purified_buddy_pilot/`
> (`20261018_purified_buddy_pilot_log.md`, `pbp_tables.md`, `clip_content_stats.md`). Train side only; labels used
> only to score.

## Results brief

**What we tried.** The night student learned toward each painting's own emotion-instructed caption and kept mostly
content. Here the same image-only network learned from pairs of paintings instead. A pair is two paintings whose
emotion-instructed captions are mutual nearest neighbours (a caption emotion buddy pair). The purified variant drops
every pair that is also a content pair (mutual neighbours in the default caption or default image embedding). A
second check rebuilt the content graph with CLIP, to see if the shared Qwen model explains the overlap.

**Outcome against the bar.** Fail (image emotion AMI of the purified student at least 0.12, and above the control by
more than the seed sd). On 5,000 train-side paintings, k = 20, 3 seeds:
- Purified student 0.094 ± 0.002; control student (default caption pairs) 0.070 ± 0.002. The gap is clear, the level
  is not: the emotion-instructed image alone gives 0.093, the neighbour average of the night pilot 0.102.
- Emotion pairs vs default caption pairs: +0.020. Purification: +0.004 (about 2 sd). Content-only hard negatives:
  +0.001 (inside the sd).
- The graphs say why purification adds little. Dropping the shared pairs lowers their genre and style lift but not
  their emotion lift (2.42 to 2.38). The shared pairs are the most emotion-consistent ones (2.55).
- With a CLIP content graph, the shared pairs again have the highest emotion lift (2.61 vs 2.42 for the whole emotion graph).

**What it does not show.** One sample, the 2B model, ArtELingo, the train side. One student design (the night pilot's
2-layer MLP, InfoNCE on pairs). Every run had its best dev loss at epoch 1 to 3 and worse after, so the students fit little. The CLIP check has
no uncertainty estimate.

**Options for next.**
- a. Keep image-only emotion out of the plan and use the graphs as built: the open design piece (prune, fuse, or
  pruning that guides fusion) goes into the spec's sweep, scored on graph lift and a downstream head.
- b. One 8B check of image emotion (about 3 h GPU), as in the grill's minor batch.
- c. A second student design (a deeper network, or pairs plus the own-caption term). Cheap on CPU, but the neighbour
  average (0.102) bounds what an image-only view reaches so far.

**Your reading.** (user, before ours, 2026-10-18) "Something but not conclusive; the teacher-student net failed on
cross-modal emotion transfer; the Stage 1 finding matters: purification lowers content-likeness but does not raise
emotion purity; maybe the views of one model share a lot."

<details>
<summary>Our reading and recommendation (open after writing yours)</summary>

**Our reading.** Buddy pairs carry more emotion to the image than the own-caption target: +0.020 over the control,
where the night student gained nothing. The level stays at the prompted image's (0.093) and below the neighbour
average (0.102). So the image-only route to emotion has a low ceiling with this data. The purification result is a
data fact more than a model fact: in art, content and emotion co-occur (storms are fearful, still lifes content).
The CLIP check supports this, since a content graph from another model gives the same pattern. Fairly sure of the
direction of each effect; the purification effect itself is small (3 seeds, about 2 sd).

**Our recommendation.** Option a. Do not prune the shared pairs blind; score prune, fuse and pruning that guides
fusion side by side in the spec's sweep. Keep the 8B check as the one remaining image-emotion run. Our view only.

</details>

## What ran

- **Stages and types:** one exploration pilot (four student variants, 3 seeds each) and one graph check (exploration,
  added after the user's reading).
- **Data and seeds:** the frozen sample of the premise check (99,999 captions, 19,889 paintings, train side, seed 0).
  Graphs and students use 13,400 training paintings, with 1,489 dev paintings for early stopping on the dev loss.
  Scores are on the fixed 5,000 evaluation paintings of the night pilots. Student seeds 0, 1, 2.
- **Graphs:** mutual top-20 cosine neighbours of the painting means, built apart on the training and the dev
  paintings. Emotion graph: emotion_focus captions. Content graph: the union of the default caption graph and the
  default image graph. Purified graph: the emotion graph minus the content graph. Content-only graph: the content
  graph minus the emotion graph.
- **Students:** the night pilot's network (Linear 2048 to 1024, GELU, dropout 0.1, Linear to 2048, l2 output) on the
  default Qwen image. Symmetric InfoNCE between the two images of a pair, in-batch negatives, temperature 0.05, AdamW
  lr 1e-3, weight decay 0.05, 512 pairs per batch, early stop after 15 epochs without a better dev loss. Nothing was
  tuned.
- **Variants:** S_D (default caption graph, the control), S_E (emotion graph), S_P (purified graph), S_PN (purified
  graph plus up to four content-only neighbours per anchor as extra negatives).
- **Scores:** the premise check's functions: painting AMI at k = 20 (3 k-means seeds), the grouped 5-fold emotion
  probe, top-20 overlap with the teacher, graph lift (same-label rate of a graph's pairs over the random rate).
- **CLIP check:** CLIP B/32 image and caption-mean graphs on the same training paintings, the same mutual top-20 rule.
- **Code and who:** Codex wrote the pilot and its CPU checks; the controller reviewed it and ran the queue on CPU (the
  GPU was down before the container restart). A Sonnet worker wrote and ran the CLIP check
  (`pbp_clip_content.py`); the controller checked its numbers against `clip_content_stats.json`.

## Results

**Students** (5,000 evaluation paintings; mean ± sd over 3 student seeds).

| View | Emotion AMI | Genre AMI | Style AMI | Emotion probe |
|---|---:|---:|---:|---:|
| Default image | 0.070 | 0.449 | 0.222 | 39.3% |
| emotion_focus image | 0.093 | 0.410 | 0.168 | 40.9% |
| Night student (own-caption target) | 0.074 ± 0.000 | 0.424 | 0.165 | 45.3% |
| Neighbour average (night pilot, best) | 0.102 | 0.416 | 0.166 | 41.0% |
| S_D, default caption pairs (control) | 0.070 ± 0.002 | 0.419 | 0.155 | 46.5% |
| S_E, emotion pairs | 0.090 ± 0.003 | 0.425 | 0.167 | 46.8% |
| S_P, purified pairs | 0.094 ± 0.002 | 0.414 | 0.171 | 46.6% |
| S_PN, purified pairs plus content negatives | 0.096 ± 0.001 | 0.419 | 0.169 | 46.6% |
| Teacher (mean emotion_focus caption) | 0.217 | 0.244 | 0.101 | 56.1% |

| ID | Measured | Number | Compared against | Bar | Outcome |
|---|---|---|---|---|---|
| Pilot bar | S_P image emotion AMI | 0.094 ± 0.002 | S_D 0.070 ± 0.002 (gap +0.024, max sd 0.002) | at least 0.12 and gap above the max sd | fail (level) |

**Verdict under the rule:** fail. The gap part passes; the level part fails.

**Effects** (emotion AMI, differences of student means): S_E − S_D +0.020; S_P − S_E +0.004 (sds 0.002 and 0.003);
S_PN − S_P +0.001 (sds 0.001 and 0.002).

**Graphs** (training paintings; lift = same-label rate over the random rate).

| Graph | Pairs | Emotion lift | Genre lift | Style lift |
|---|---:|---:|---:|---:|
| Emotion | 57,727 | 2.42 | 3.95 | 2.28 |
| Default caption | 47,516 | 2.19 | 4.45 | 2.74 |
| Default image | 69,853 | 1.65 | 5.21 | 5.37 |
| Purified (emotion minus content) | 45,505 | 2.38 | 3.75 | 2.04 |
| Shared (emotion and content) | 12,222 | 2.55 | 4.66 | 3.18 |
| CLIP content | 86,075 | 1.73 | 4.92 | 4.89 |
| Shared with CLIP content | 5,131 | 2.61 | 5.10 | 4.07 |
| Emotion minus CLIP content | 52,596 | 2.40 | 3.84 | 2.11 |

- 21% of emotion pairs are Qwen content pairs (19% default caption, 5% default image); 9% are CLIP content pairs.
- 705 training paintings have no purified pairs: 440 had no emotion pairs, and 265 lose their remaining pairs.
- The CLIP and Qwen content graphs share 33% of the CLIP pairs.

## Analysis

- **Pairs transfer more than the own-caption target.** The emotion graph student gains +0.020 over the control; the
  night student gained 0.000. Our reading: the network learns from image pairs chosen with caption embeddings, and does
  not predict the caption vector itself. Content can still shape which pairs are chosen.
- **The level stays at the prompted image's.** S_P (0.094) matches the emotion_focus image (0.093) and stays below
  the neighbour average (0.102), less than half of the teacher (0.217).
- **Purification removes the best emotion pairs too.** The shared pairs have emotion lift 2.55, above the rest of the
  emotion graph (2.38). So dropping them lowers content-likeness (style lift 2.28 to 2.04) without a gain in emotion
  purity. Our reading: this is a likely reason why the student effect (+0.004) is small; it is not checked.
- **The pattern is not only an artifact of the shared Qwen model.** With a CLIP content graph, the overlap falls from
  21% to 9%, and the shared pairs still have the highest emotion lift (2.61). Our reading: content and emotion co-occur
  in this data. Two content graphs do not rule out a model effect fully.
- **The students fit little.** Every run had its best dev loss at epoch 1 to 3 and rose after. The best dev loss is
  about 4.8 for the control and 5.2 to 5.6 for the emotion variants. The emotion probe is about 46.5% for all four,
  the control included, so the gain is in the cluster geometry only.
- **Genre and style fall.** All students lose genre (−0.02 to −0.03) and style (−0.05 to −0.07) against the default
  image.

## Deviations

- The CLIP check was not in the pilot design. It is exploration, added after the user's reading ("maybe the views of
  one model share a lot").
- The run log is Codex's hand-back, written before the queue ran ("the full queue was NOT run"); the controller's run
  section at its end records the run.
- Codex fixed one choice the brief left open: dev negatives are redrawn from the run seed each epoch, so early stopping
  compares the same negatives.
- The pilot ran on CPU (the GPU was down before the container restart at 15:53). No other deviation.

## Files and storage

- **Outputs:** scripts, the log, `pbp_tables.md`, `clip_content_stats.md` (in git); `pbp_results.json`,
  `clip_content_stats.json` (gitignored); graphs `graphs/fit.npz`, `dev.npz` (7.8 MB, untracked; rebuild in 35 s with
  `pbp_graphs.py`).
- **Storage:** the 12 student arrays (`outputs/`, 235 MB) and Codex's smoke folder (about 100 MB) are on
  `.scratch/pending_deletions.md` for the next storage summary. No item of 1 GB or more.
