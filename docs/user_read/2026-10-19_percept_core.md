# The September PercepT code does not repeat its own result, so PercepT core waits; new rows point to the aspect graph

> 2026-10-19 (Amsterdam). Full report:
> [PercepT core run](../reports/auto/percept/2026-12-13_percept_core.md).

## Summary

- Before PercepT core trains, a check reruns the September PercepT code. It must give the recorded scores. It
  failed, so PercepT core did not run tonight.
- The cause is noise in the September code itself. On this machine it gives slightly different results from run to
  run, so the check cannot pass reliably.
- Four missing rows of the ablation table ran. On a graph without the aspect instruction, our network is slightly
  worse than no training. Our reading: the aspect graph gives the gain (development data only).
- The side of the graph matters: captions for emotion, images for style and genre.
- The September encoders are strong on one side each: the emotion text model on captions, CLIP on images for style.
- You decide how to change the check. Then PercepT core runs in about 25 GPU minutes.

## How we got here

This project builds a model that finds aspects of paintings (emotion, style, genre) without conditions. It aims to
improve over PercepT, a published method for perceptual topics. Our target: per-aspect clustering scores above
PercepT core by 2026-10-29.

PercepT core is our version of PercepT, with four fixes from our review of PercepT's code (the code audit). It runs
on our own features and gives the comparator that our model must beat. We approved its design yesterday. The build
(the coding step, done by agents overnight) finished tonight, and this run was the first use.

The run also filled empty rows of the ablation table. That table changes one part of our method per row. Its scores
are AMI (adjusted mutual information between clusters and labels; 0 is chance, 1 is a perfect match).

## Does the September PercepT still give its numbers?

**Problem.** PercepT core reuses the September PercepT code. The stage 0 check first reruns that code unchanged. It
must give the recorded scores to 4 decimals. So a silent change in the code cannot slip through.

**Idea.** If the rerun matches, the reused code is safe. If not, PercepT core waits, and we find the cause. The
build already found one cause: the recorded numbers came from a cluster machine (node404), not this one. The build
then believed that this machine at least repeats its own numbers.

**Did it work?** No. Tonight the two September scripts gave two new results. They differ from each other and from
the build. A short check found the source of the noise.

The first step of the September code is k-means on several threads. The threads add up the cluster centers in a
different order in every run, so the centers differ in the last digit. With one thread, the clustering step gives the
same centers every time, and two full runs gave the same result. We think the later training step grows the tiny
difference into a different end result. That last link is our inference.

The same k-means step is in PercepT core. So PercepT core's own runs carry this noise too, unless its k-means uses one
thread.

**Key evidence.** Eight runs of the same code on this machine. The scores are on September's held-out paintings; the
genre score rests on 159 paintings only.

| Result (emotion / genre) | Runs |
|---|---:|
| 0.1238 / 0.2617 | 5 |
| 0.1244 / 0.2617 | 1 |
| 0.1287 / 0.2804 | 2 |
| Recorded on node404 | 0.1225 / 0.2274 |

The parts of the check that need no rerun still pass. The saved September model gives its recorded scores exactly
with today's code. The reused code is unchanged. Whether node404 repeats itself, we did not check (its five September
runs were equal).

> Full report: "Results brief", "Analysis", first bullet.

## Which part of our network does the work?

**Problem.** Our network learns from a graph of painting pairs. The best graphs use an instruction that names the
aspect. It was open whether the network itself helps, or only the aspect pairs. It was also open how much the side
of the graph (images or captions) matters.

**Idea.** Train the same network on a graph without any instruction (the default graph). Also train it on the
graph from the other side alone, with default and with prompted inputs.

**Did it work?** Yes, the rows answer both questions. On the default graph, the network is 0.005 to 0.008 below the
default embedding clustered directly. Our reading: the network alone adds nothing, and the aspect graph gives the
gain. The other side's graph loses most of the gain for emotion and style, and wins for genre. Caveat: development
data only (the same paintings chose our earlier rows).

**Key evidence.** Mean of image-side and caption-side AMI at 20 clusters, 5,000 paintings, 3 seeds:

| Row | What changes | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| 2 | No training, default embedding | 0.089 | 0.171 | 0.393 |
| 6 | Network on the default graph | 0.082 | 0.163 | 0.388 |
| 7 | Network on the plain aspect graph | 0.118 | 0.213 | 0.402 |
| 10 | Network on the other side's graph | 0.086 | 0.154 | 0.428 |
| 13 | Same, prompted input | 0.089 | 0.156 | 0.434 |

Row 13's genre score (0.434) is the highest genre value in the ablation table. It needs the instruction at test time.

> Full report: "Results" (the ablation table) and "Analysis", bullets two and three.

## How strong are the September encoders?

**Problem.** PercepT uses CLIP for images and an emotion text model (RoBERTa trained on the GoEmotions dataset) for
captions. We use Qwen3-VL-Embedding (Qwen) for both. The table had no row for the September encoders without
training.

**Idea.** Cluster the September encoders directly, as row 2 does for Qwen: CLIP images, and captions once with CLIP
text and once with the emotion text model.

**Did it work?** Yes. Each September encoder is strong on one side. On captions, the emotion text model scores much
higher on emotion than Qwen's default captions. On images, CLIP scores higher on style than Qwen.

**Key evidence.** One side at a time, k-means at 20 clusters, 5,000 paintings; the spread covers k-means seeds only:

| Side | Encoder | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| Captions | Qwen default | 0.107 | 0.120 | 0.338 |
| Captions | Emotion text model | 0.189 | 0.049 | 0.068 |
| Captions | Qwen with emotion instruction | 0.217 | — | — |
| Images | Qwen default | 0.070 | 0.222 | 0.449 |
| Images | CLIP | 0.065 | 0.300 | 0.412 |

This matters for PercepT core, which learns its topics with the emotion text model. So it may be a strong comparator
on emotion.

> Full report: "Results", side table, and "Analysis", fourth bullet.

<details>
<summary>Our advice (open after giving your reading)</summary>

Change the check to the parts that repeat exactly. Those are the saved September model's scores, the unchanged code,
and k-means on the saved data. Also give PercepT core's own k-means one thread, so its runs repeat. Then run PercepT
core in one evening. Our view: this meets the check's purpose without a rerun that cannot repeat.

</details>

## Next

PercepT core waits for your decision on the check. The options:

- a. Check only the parts that repeat exactly (our advice), then run PercepT core (about 25 GPU minutes). This
  changes the approved design, so the spec gets an amendment.
- b. Keep a rerun of the September code, with a tolerance from tonight's spread. A weak check: the spread comes from
  eight runs.
- c. Rerun on node404 as first planned. It needs a cluster booking for that machine, which may not repeat itself
  either.

After PercepT core comes the design of our own model's next stage: one shared model with a head per aspect.

## Glossary

- **PercepT core:** our version of PercepT with four fixes from the code audit, on our features.
- **Stage 0 check:** the rerun of the unchanged September PercepT that must give its recorded scores.
- **AMI:** adjusted mutual information between clusters and labels; 0 is chance, 1 is a perfect match.
- **Default graph:** pairs of paintings that are among each other's 20 nearest neighbours in the default embedding.
- **Plain aspect graph:** the same pairs rule on the prompted embedding (Qwen with an instruction that names the
  aspect).
- **Other side's graph:** the plain aspect graph built from the other side (images instead of captions for emotion).
- **Prompted input:** the network reads the prompted embeddings, so it needs the instruction at test time.
