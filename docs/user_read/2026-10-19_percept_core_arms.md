# PercepT core now runs, but its topic count often falls to 1 to 3 topics, so its comparator score is weak

> 2026-10-19 (Amsterdam). Full report:
> [PercepT core arms](../reports/auto/percept/2026-12-14_percept_core_arms.md). It follows the
> [first PercepT core run](2026-10-19_percept_core.md).

## Summary

- The changed regression check passed. One PercepT core run, repeated, gave identical files.
- PercepT core ran in five settings with 3 seeds each. Its rule for the number of topics kept 99 topics on some seeds
  and only 1 to 3 on others.
- Only one setting passed all checks: the September encoders (CLIP and an emotion text model) with a CLIP image
  vector reader. It is now the PercepT core comparator, and it is weak (image-side genre score 0.217 against 0.458
  for our network).
- With 99 topics, the two best settings come close to their image encoder clustered directly on emotion and genre,
  and stay lower on style.
- Against the best 99-topic seed per aspect, our network is slightly higher on emotion and genre (about 0.01) and
  higher on style (about 0.10). Development data only.
- You decide whether to fix the topic count and rerun (about 20 GPU minutes), or go on to the stage 2 decision (the
  design of our model's next stage).

## How we got here

This project builds a model that finds aspects of paintings (emotion, style, genre) without conditions. It aims to
improve over PercepT, a published method that learns perceptual topics from images and captions. Our target: higher
per-aspect clustering scores than PercepT core by 2026-10-29.

PercepT core is our version of PercepT, with four fixes from our review of PercepT's code, on our own features. Its
first run stopped at the regression check, because the September PercepT code does not repeat its own result. You
then changed the check to the parts that repeat exactly and asked for PercepT core to run (this morning, 08:20).

Two encoders appear below. The September encoders are the ones PercepT used in our September work: CLIP for images
and captions, and an emotion text model (RoBERTa trained on the GoEmotions dataset) for captions. Qwen is
Qwen3-VL-Embedding, our own encoder; "default" means without an instruction that names the aspect.

All scores are AMI (adjusted mutual information between clusters and labels; 0 is chance, 1 is a perfect match).
PercepT core reads only images at test time, so we compare image-side scores: k-means with 20 clusters on the
5,000 evaluation paintings.

## Does the changed check pass?

**Problem.** The first run's check required a rerun of the September code to give its recorded scores. That rerun
gives slightly different results each time, so the check could not pass.

**Idea.** Check only what repeats exactly. The saved September model must give its recorded scores. The reused code
must be unchanged. K-means on the saved data must give the saved clusters. Also, PercepT core's own k-means runs on
one thread, so a run repeated twice must give the same files.

**Did it work?** Yes. Every check passed in under a minute. One run of PercepT core's first stage, repeated, gave
byte-identical files. The other runs were not repeated.

**Key evidence.** The saved September model gives emotion 0.1225 and genre 0.2274, its recorded scores. The reused
code differs in 0 files. K-means repeats all of its cluster labels.

> Full report: "Results", the stage 0 table.

## Where does PercepT core stand?

**Problem.** Our model needs a fair PercepT comparator, on our data and our features. PercepT core gives it. It
learns topics from image-caption pairs in a first stage. A second stage then trains an image reader to predict a
painting's topics from the image alone.

**Idea.** Run PercepT core in five settings, to see which encoder and which reader carry the score. Take the best
setting per aspect as the comparator. A setting counts only if all three seeds train well (dev AUC at least 0.75).

| Setting | Topics learned from | Image reader |
|---|---|---|
| P1 | September encoders (CLIP, emotion text model) | CLIP patches |
| P2 | September encoders | CLIP image vector |
| P3 | Qwen, with Qwen's emotion-instructed captions | Qwen image vector |
| P4 | as P3 | CLIP patches |
| P5 | Qwen, with the emotion text model | Qwen image vector |

**Did it work?** Partly. PercepT core starts with 100 topics. PercepT assumes that unused topics get long center
vectors, so it sorts the center lengths (norms) and drops the topics above the largest gap. On our data the lengths
spread evenly, so the largest gap always sits at one end. Then the rule keeps 99 topics, or only 1 to 3, depending
on the seed. Only the CLIP vector setting (P2) passed all checks. Its two seeds with 2 or 3
topics pull its score far down.

**Key evidence.** Image-side AMI at 20 clusters, 5,000 paintings, mean over 3 seeds:

| Row | Method | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| 1 | CLIP image, clustered directly | 0.065 | 0.300 | 0.412 |
| 2 | Qwen default image, clustered directly | 0.070 | 0.222 | 0.449 |
| 4a | PercepT core, P1: September encoders, CLIP patches (fails one seed) | 0.029 | 0.137 | 0.234 |
| 4b | PercepT core, P2: September encoders, CLIP image vector (the comparator) | 0.036 | 0.162 | 0.217 |
| 5a to 5c | PercepT core, P3 to P5 | none | none | none |
| 9 (genre 10) | Our best network, no instruction at test time | 0.081 | 0.348 | 0.458 |

Topics kept per seed: P1 and P2 99, 2, 3; P3 and P4 1, 99, 99; P5 1, 1, 1. With 1 topic the image reader has
nothing to learn, so it does not train.

With 99 topics, the two best settings come close to their image encoder clustered directly on emotion and genre, and
stay lower on style. The other two 99-topic settings (P1 seed 0, P4) are lower than CLIP images on every aspect.

| Seeds with 99 topics | Emotion | Style | Genre | Same encoder, clustered directly |
|---|---:|---:|---:|---|
| P3, 2 seeds | 0.072 | 0.194 | 0.446 | Qwen: 0.070, 0.222, 0.449 |
| P2, 1 seed | 0.063 | 0.250 | 0.412 | CLIP: 0.065, 0.300, 0.412 |

Against the best 99-topic seed per aspect, our network is 0.009 higher on emotion, 0.098 on style and 0.012 on
genre. That comparison picks the best seed for PercepT core after the fact.

Caveat: development data only. The same 5,000 paintings chose our network's settings (graph and input) and the
comparator setting.

> Full report: "Results brief", "Results" (arm and per-seed tables) and "Analysis", bullets one to three.

## Do the topics carry over to another image reader?

**Problem.** It was open whether topics learned with Qwen also help a CLIP image reader.

**Idea.** Train the CLIP patch reader on the Qwen topics (P4) and compare it with the Qwen reader (P3).

**Did it work?** Only in part. On the two seeds that trained, the CLIP reader scores lower on every aspect. It is
also below CLIP images clustered directly.

**Key evidence.** Same topics, two readers (2 seeds): Qwen reader 0.072, 0.194, 0.446; CLIP reader 0.052, 0.156,
0.348 (emotion, style, genre).

> Full report: "Analysis", fourth bullet.

<details>
<summary>Our advice (open after giving your reading)</summary>

Fix the topic count, then rerun the five settings (a short build and about 20 GPU minutes). A rule that keeps 2 or 3
topics on two of three seeds does not give the strongest real comparator, and a reviewer would see that. Two choices
for the new rule: a fixed number of topics, or a cut only inside the range of center lengths, never at an end. Then go on to
the stage 2 decision in the same chat. Our view.

</details>

## Next

The decide chat opens next. It asks for your reading of these results first. Your decision has three options:

- a. Keep the CLIP vector setting as the comparator. Add the image encoders clustered directly as a second bar. Go
  to the stage 2 decision.
- b. Change the topic count rule and rerun the five settings (about 20 GPU minutes after a short build).
- c. Keep the rule and run 7 more seeds, to see how often the count falls (about 15 GPU minutes).

After that comes the design of our own model's next stage: one shared model with a head per aspect.

## Glossary

- **PercepT core:** our version of PercepT with four fixes from the code review, on our features.
- **Setting (arm):** one PercepT core combination of encoders and image reader, P1 to P5.
- **Topic count rule (Algorithm 1):** start with 100 topics, sort the lengths (norms) of their center vectors, keep
  the topics below the largest gap.
- **Image-side AMI:** k-means with 20 clusters on a method's image output, scored against the aspect's labels.
- **Dev AUC:** how well the image reader predicts each painting's topics on held-apart development paintings.
- **Regression check (stage 0):** the check that the reused September code is unchanged before PercepT core runs.
