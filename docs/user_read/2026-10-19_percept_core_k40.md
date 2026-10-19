# PercepT core keeps 40 of its 100 topics and now passes every check; our network is above it on style and genre, level on emotion

> 2026-10-19 (Amsterdam). Full report:
> [PercepT core with K = 40](../reports/auto/percept/2026-12-15_percept_core_k40.md). It follows the
> [PercepT core arms report](2026-10-19_percept_core_arms.md), where the topic count fell to 1 to 3 on some seeds.

## Summary

- PercepT core now keeps 40 of its 100 trained topics. All five settings passed every check on all three seeds.
- The best PercepT core setting per aspect is now the comparator: emotion 0.077, style 0.229, genre 0.436.
- Our network is higher on style and genre. On emotion it is level (a difference below 0.01).
- PercepT core is level with or below its own image encoder clustered directly.
- The training before the topic cut matched the last run. Development data only.
- You decide the comparator for 2026-10-29, then the design of our model's next stage.

## How we got here

This project builds a model that finds aspects of paintings (emotion, style, genre) without conditions. It aims to
improve over PercepT, a published method that learns perceptual topics from images and captions. Our target: higher
per-aspect clustering scores than PercepT core by 2026-10-29.

PercepT core is our version of PercepT, on our own features, with four fixes from our review of PercepT's code. The
fixes: train on image-caption pairs, use the paper's fusion formula, change the topic count, and train the image
reader on all topics of a painting's captions.

The last run (this morning) used PercepT's own rule for the topic count. That rule kept 99 topics on some seeds and 1
to 3 on others. Only one setting passed the checks, and its score was weak. You then chose a fixed count of 40 topics.
This run repeats the five settings with that count.

Two encoders appear below. The September encoders are the ones PercepT used in our September work: CLIP for images
and captions, and an emotion text model (RoBERTa trained on GoEmotions). Qwen is Qwen3-VL-Embedding, our own encoder;
"default" means without an instruction that names the aspect.

All scores are AMI (adjusted mutual information between clusters and labels; 0 is chance, 1 is a perfect match).
PercepT core reads only images at test time, so we compare image-side scores: k-means with 20 clusters on the 5,000
evaluation paintings. We read a difference below about 0.01 as level; three seeds give a spread, not a test.

## Does a fixed topic count make PercepT core stable?

**Problem.** PercepT's rule cuts the topic list at the largest gap between topic lengths. On our data that gap
always sat at one end of the list, so the count was 99 or 1 to 3, by seed.

**Idea.** Keep a fixed number of topics. PercepT core still trains 100 topics as before. After training, each
training pair goes to its nearest topic, and the 40 topics with the most pairs stay. We count pairs, not lengths:
in the last run, short topics were the little-used ones. PercepT's rule is still computed, but only reported.

**Did it work?** Yes, on the three seeds we ran. Every setting passed all three checks on every seed: the
regression check, the unit tests of the fixes, and how well the image reader predicts topics (dev AUC at least
0.75). The seed spread (standard deviation) is now at most 0.007 per aspect, against up to 0.143 last time.

**Key evidence.**

- The topic lengths after training match the last run bit for bit. So the training was very likely the same, and
  the topic count made the difference.
- The kept 40 topics hold 57 to 63% of the training pairs, and each holds at least 1% of them.
- On the six seeds that kept 99 topics last time, 40 topics change the scores by at most 0.02.
- On the four seeds that kept 2 or 3 topics, 40 topics raise the genre score by 0.16 to 0.32.
- The 40th and 41st topics differ by only 1 to 14 pairs. We did not test whether 39 or 41 topics change the result.

> Full report: "Results" (kept topics, the run 2 against run 3 table) and "Analysis", bullets one and two.

## Where does our network stand against PercepT core?

**Problem.** Our model needs a fair and strong PercepT comparator, per aspect, on our data and features.

**Idea.** Take the best of the five PercepT core settings per aspect. Set it beside our network and beside the image
encoders clustered directly. "Our network" is the head that stage 1 picked per aspect, read on default image
embeddings.

| Setting | Topics learned from | Image reader |
|---|---|---|
| P1 | September encoders (CLIP, emotion text model) | CLIP patches |
| P2 | September encoders | CLIP image vector |
| P3 | Qwen, with Qwen's emotion-instructed captions | Qwen image vector |
| P4 | as P3 | CLIP patches |
| P5 | Qwen, with the emotion text model | Qwen image vector |

**Did it work?** Yes. The comparator is P3 for emotion and genre and P2 for style. Our network is above it on style
(+0.120) and genre (+0.022), and level on emotion (+0.003). Two plain image encoders also beat PercepT core: CLIP
images on style, Qwen default images on genre.

**Key evidence.** Image-side AMI at 20 clusters, 5,000 paintings, mean over 3 seeds. Row numbers follow the full
ablation table.

| Row | Method | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| 1 | CLIP image, clustered directly | 0.065 | 0.300 | 0.412 |
| 2 | Qwen default image, clustered directly | 0.070 | 0.222 | 0.449 |
| 4a | PercepT core P1 | 0.053 | 0.167 | 0.356 |
| 4b | PercepT core P2 | 0.066 | **0.229** | 0.400 |
| 5a | PercepT core P3 | **0.077** | 0.189 | **0.436** |
| 5b | PercepT core P4 | 0.054 | 0.162 | 0.361 |
| 5c | PercepT core P5 | 0.074 | 0.181 | 0.435 |
| 9 (genre 10) | Our network, no instruction at test time | 0.081 | 0.348 | 0.458 |

Bold: the comparator per aspect. Seed spreads are at most 0.007.

Caveats: development data only. The same 5,000 paintings chose our network's settings and the comparator setting.
On emotion, our network, PercepT core and the plain Qwen image all lie within 0.011. Two other heads of our network
reach 0.085 and 0.089 on emotion, but stage 1 did not pick them.

> Full report: "Results brief", "Results" (arm table, ablation table) and "Analysis", last bullet.

## What carries PercepT core's score?

**Problem.** The five settings were built to show which encoder and which part of PercepT carry the score.

**Idea.** Compare settings that differ in one part.

**Did it work?** Partly. The comparisons below are descriptive, on development data. Our reading of them:

- The encoder sets the profile. The settings with a Qwen image vector lead on emotion and genre, the CLIP vector
  setting on style, as with the encoders clustered directly.
- The emotion source made little difference: P3 and P5 differ by at most 0.008.
- The CLIP patch reader is the weak part. On the same topics, CLIP patches score below an image vector. On Qwen
  topics they also score below CLIP images clustered directly.

**Key evidence.** Differences in image-side AMI:

| Change | Settings | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| CLIP vector instead of CLIP patches | P2 against P1 | +0.013 | +0.061 | +0.044 |
| CLIP patches instead of Qwen vector, same Qwen topics | P4 against P3 | −0.024 | −0.026 | −0.074 |
| Emotion text model instead of Qwen emotion | P5 against P3 | −0.003 | −0.008 | −0.001 |

> Full report: "Results" (arm differences) and "Analysis", bullets three to five.

<details>
<summary>Our advice (open after giving your reading)</summary>

Take PercepT core with 40 topics as the comparator. Add the image encoders clustered directly as a second bar per
aspect: CLIP images beat PercepT core on style (0.300 against 0.229), Qwen default images on genre (0.449 against
0.436). This bar is stricter than the 2026-10-29 target. Then go on to the design of our model's next stage in the
same chat.

Emotion is the aspect that the next stage must move. A gain of 0.003 is level, so our network does not beat PercepT
core on emotion yet.

A check of other topic counts can wait. On the same training, 40 against 99 topics changed scores by at most 0.02,
and emotion by at most 0.005, on six seeds. But 40 is our choice, and a count below 40 could move the comparator. Our
view.

</details>

## Next

The next chat is the decide chat, where you give your reading first and then decide. The options for the
comparator:

- a. PercepT core with 40 topics only (P3, P2, P3 per aspect). No new run.
- b. As a, plus the image encoders clustered directly as a second bar per aspect. No new run.
- c. A check of two more topic counts (for example 20 and 60) first: a short build and about 35 GPU minutes
  (estimate).

After that comes the design of our own model's next stage. The current plan (direction log): one shared model with a
head per aspect.

## Glossary

- **PercepT core:** our version of PercepT with four fixes from the code review, on our features.
- **Setting (arm):** one PercepT core combination of encoders and image reader, P1 to P5.
- **Topic count (K):** how many topics PercepT core keeps; now 40 of the 100 trained topics, those with the most
  training pairs.
- **Image-side AMI:** k-means with 20 clusters on a method's image output, scored against the aspect's labels.
- **Comparator:** the best PercepT core setting per aspect, by mean image-side AMI over 3 seeds.
- **Dev AUC:** how well the image reader predicts each painting's topics on held-apart development paintings; a
  setting needs at least 0.75 on every seed.
- **Level:** a difference below about 0.01, which we read as no difference.
