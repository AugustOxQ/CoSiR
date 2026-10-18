# The residual graphs stay best on three seeds, and a network on prompted inputs comes close to the prompted embedding but does not beat it

> 2026-10-18 (Amsterdam). Full report:
> [aspect graph follow-up](../reports/auto/percept/2026-12-12_aspect_graph_followup.md).

## Summary

- We repeated the stage 1 networks with more seeds, tried one more genre graph, and fed some networks the prompted
  embeddings instead of the default ones.
- With three seeds, the residual graphs still train the best emotion and style networks.
- For genre, the image graph alone is best. Mixing in the caption graph does not help.
- On prompted inputs, the residual networks come within 0.003 of the prompted embedding on its stronger side. They
  do not beat it there.
- So the gap of stage 1 comes mostly from the input. This matters for the next model, which reads default inputs.
- An ablation table (in the middle of this report) changes one part at a time. The instruction gives the largest
  gain. The best network keeps most of it without an instruction at test time. The PercepT core rows are still open.

## How we got here

This project builds a model that finds aspects (emotion, style, genre) without conditions. It aims to improve over
PercepT, a published method for perceptual topics.

Stage 1 (earlier today) built many pair graphs per aspect. An aspect graph links two paintings when each is among
the other's 20 nearest neighbours in the prompted embedding (Qwen with an instruction that names the aspect). On
each graph we trained a small network and clustered its outputs. The residual graphs gave the best networks. They
rank pairs by aspect similarity minus content similarity.

Two points stayed open. Five reference networks had one seed only. And no network reached the prompted embedding on
its stronger side: the side (image or caption) where the prompted embedding clusters best. You asked to make this
pilot more robust before the next steps: PercepT core (our version of PercepT with four fixes, on our features), then
stage 2 (one shared model with a head per aspect).

## Do the stage 1 results hold?

**Problem.** Stage 1 compared the residual graphs with reference graphs that had one seed. For genre, the image
graph alone got no network.

**Idea.** Give every reference network three seeds, and train a genre network on the image graph alone.

**Did it work?** Yes. The order of stage 1 holds. The residual graphs stay ahead of the plain aspect graph by about
ten times the seed spread. For genre, the image graph alone beats the mix of the image graph (weight 0.75) and the
caption graph (0.25).

Caveat: development data only. Stage 1's scores on the same paintings picked these graphs.

**Key evidence.** AMI at 20 clusters on 5,000 paintings, mean of the image and caption side, 3 seeds:

| Aspect | Plain aspect graph | Best graph |
|---|---:|---:|
| Emotion | 0.118 | 0.137 (residual) |
| Style | 0.213 | 0.260 (residual) |
| Genre | 0.402 (captions) | 0.428 (image graph alone) |

> Full report: "Results" and "Analysis", first two bullets.

## Can a network beat the prompted embedding?

**Problem.** In stage 1 every emotion and style network stayed below the prompted embedding on its stronger side.
The networks read default embeddings, so the input may be the cause.

**Idea.** Train the same networks on the prompted embeddings of the aspect. If they then reach the prompted
embedding, the input explains the gap. Caveat by design: such a network needs the aspect instruction at test time.

**Did it work?** Partly. With prompted inputs, the residual networks come within 0.003 of the prompted embedding on
its stronger side (mean of three seeds). Among these networks, only style beats it on the mean of both sides, through
the caption side. Emotion stays just below on both sides, and genre is level. The genre network on the image graph
alone, with default inputs, is at 0.428.

**Key evidence.** Same measure, residual graph (genre: the image-weighted mix), 3 seeds:

| Aspect | Network, default inputs | Network, prompted inputs | Prompted embedding, no training |
|---|---:|---:|---:|
| Emotion | 0.137 | 0.151 | 0.155 |
| Style | 0.260 | 0.284 | 0.265 |
| Genre | 0.420 | 0.424 | 0.424 |

> Full report: "Results" and "Analysis", bullets three and four.

## Which part of the design does the work?

**The instruction gives the largest gain, and a network trained on the right graph keeps most of that gain without
an instruction at test time.** The table changes one part at a time: the input embedding, the method, and the graph
that the network learns from. You asked for it as an ablation, with placeholders where a number is still missing.

How to read it:

- Measure: AMI at 20 clusters on the 5,000 evaluation paintings, mean of the image side and the caption side, for
  the aspect's own label.
- Rows with a network: mean of 3 seeds, sd at most 0.005. Rows with k-means on an embedding: sd over k-means seeds
  at most 0.007. Differences below about 0.01 are close to this spread.
- Every graph except the default graph comes from the prompted embedding. So rows 7 to 10 use the instruction in
  training only.
- The plain aspect graph uses captions for emotion and genre, and images for style. The default graph is built the
  same way on the default embedding, so it uses no instruction. The stage 1 pick is the residual graph for emotion
  and style, and the image-weighted mix (image 0.75, caption 0.25) for genre. "Other side alone" is the graph from
  the other side only (for genre, the image graph).
- "ongoing": part of PercepT core, the next step (row 1 needs the RoBERTa caption features that the PercepT core
  build extracts). "not run": no run planned yet; each is a short CPU run.

| # | Input embedding | Method | Graph the network learns from | Emotion | Style | Genre |
|---|---|---|---|---:|---:|---:|
| | **No training** | | | | | |
| 1 | CLIP B/32 + RoBERTa-GoEmotions | k-means on the embedding | none | ongoing | ongoing | ongoing |
| 2 | Qwen default | k-means on the embedding | none | 0.089 | 0.171 | 0.393 |
| 3 | Qwen prompted | k-means on the embedding | none | 0.155 | 0.265 | 0.424 |
| | **PercepT core** | | | | | |
| 4 | CLIP B/32 + RoBERTa-GoEmotions | PercepT core | none | ongoing | ongoing | ongoing |
| 5 | Qwen default | PercepT core | none | ongoing | ongoing | ongoing |
| | **Our network, default input (no instruction at test time)** | | | | | |
| 6 | Qwen default | network | default graph | not run | not run | not run |
| 7 | Qwen default | network | plain aspect graph | 0.118 | 0.213 | 0.402 |
| 8 | Qwen default | network | aspect graph minus content pairs | 0.123 | 0.221 | 0.404 |
| 9 | Qwen default | network | stage 1 pick | 0.137 | 0.260 | 0.420 |
| 10 | Qwen default | network | other side alone | not run | not run | 0.428 |
| | **Our network, prompted input (instruction at test time)** | | | | | |
| 11 | Qwen prompted | network | plain aspect graph | 0.122 | 0.222 | 0.404 |
| 12 | Qwen prompted | network | stage 1 pick | 0.151 | 0.284 | 0.424 |
| 13 | Qwen prompted | network | other side alone | not run | not run | not run |

**What each change adds** (differences of the unrounded means, so a difference can be 0.001 off the table's
rounded numbers):

| Change, one part at a time | Rows | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| Instruction at test time, no training | 3 vs 2 | +0.066 | +0.094 | +0.031 |
| Network on the plain aspect graph, default input | 7 vs 2 | +0.029 | +0.042 | +0.009 |
| Instruction in the graph (default graph to plain aspect graph) | 7 vs 6 | not run | not run | not run |
| Remove content pairs from the graph | 8 vs 7 | +0.005 | +0.008 | +0.002 |
| Stage 1 pick instead of the plain aspect graph | 9 vs 7 | +0.019 | +0.047 | +0.018 |
| Other side alone instead of the plain aspect graph | 10 vs 7 | not run | not run | +0.026 |
| Prompted input instead of default, plain aspect graph | 11 vs 7 | +0.004 | +0.009 | +0.002 |
| Prompted input instead of default, stage 1 pick | 12 vs 9 | +0.014 | +0.023 | +0.004 |
| Best network with default input vs default embedding | 9 (genre 10) vs 2 | +0.049 | +0.089 | +0.035 |
| Best network with default input vs prompted embedding | 9 (genre 10) vs 3 | −0.018 | −0.005 | +0.003 |
| Best network with prompted input vs prompted embedding | 12 vs 3 | −0.004 | +0.018 | 0.000 |
| Encoder: Qwen default vs the September encoders | 2 vs 1 | ongoing | ongoing | ongoing |
| Method: our network vs PercepT core, same input | 9 (genre 10) vs 5 | ongoing | ongoing | ongoing |

**What the table says:**

- The instruction at test time is the largest single gain (row 3 vs row 2).
- Without an instruction at test time, the best network keeps most of that gain: 0.049 of 0.066 for emotion and
  0.089 of 0.094 for style. For genre it gains more than the instruction does (0.035 vs 0.031).
- The choice of graph adds more than the removal of content pairs: 0.018 to 0.047, against at most 0.008.
- A prompted input adds little on the plain aspect graph (at most 0.009). It adds more on the stage 1 pick (0.014
  for emotion, 0.023 for style).
- Still open: whether the instruction in the graph is what helps (row 6), how the September encoders compare
  (row 1), and where PercepT core stands (rows 4 and 5).

The September PercepT numbers (emotion 0.123, genre 0.227) used another split, 40 topics and another scoring. They
are not on this protocol, so the table leaves them out.

<details>
<summary>Image side and caption side for each row</summary>

Same measure and rows; "img" is the image side, "cap" the caption side. Rows 1, 4 and 5 are ongoing; rows 6 and 13
are not run.

| # | Emotion img | Emotion cap | Style img | Style cap | Genre img | Genre cap |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.070 | 0.107 | 0.222 | 0.120 | 0.449 | 0.338 |
| 3 | 0.093 | 0.217 | 0.395 | 0.135 | 0.496 | 0.353 |
| 7 | 0.085 | 0.151 | 0.267 | 0.160 | 0.435 | 0.369 |
| 8 | 0.089 | 0.157 | 0.283 | 0.160 | 0.437 | 0.370 |
| 9 | 0.081 | 0.194 | 0.348 | 0.172 | 0.454 | 0.386 |
| 10 | not run | not run | not run | not run | 0.458 | 0.397 |
| 11 | 0.085 | 0.159 | 0.289 | 0.156 | 0.437 | 0.371 |
| 12 | 0.087 | 0.216 | 0.393 | 0.175 | 0.458 | 0.390 |

</details>

> Full report: "Analysis", "Ablation view".

<details>
<summary>Our advice (open after giving your reading)</summary>

Go on with PercepT core, as decided. In the next stage, keep the residual graph for emotion and style and the image
graph alone for genre. One open point: the emotion residual graph leaves 57% of paintings without a pair, and we did
not test what that costs. Test the model's input as an option: the default embeddings (no instruction at test time),
or the prompted ones (one fixed instruction per aspect). Our view.

</details>

## Next

PercepT core comes next; its design questions are open in the chat. After that, you decide on stage 2. The network
outputs of stage 1 and this follow-up (2.3 GB) wait for the storage decision at the end of the loop.

## Glossary

- **Aspect graph:** pairs of paintings that are among each other's 20 nearest neighbours in the prompted view (R0).
- **Residual graph:** the same, ranked by aspect similarity minus content similarity (the report's "residual λ").
- **Default graph:** the same pairs rule on the default embedding, so no instruction anywhere.
- **Stage 1 pick:** the graph per aspect that stage 1 took to three seeds: residual for emotion and style,
  image-weighted mix for genre.
- **Prompted embedding:** Qwen3-VL-Embedding with an instruction that names the aspect; "default" has no instruction.
- **Stronger side:** the side where the prompted embedding has the higher AMI: captions for emotion, images for
  style and genre.
- **AMI:** adjusted mutual information between the clusters and the labels; 0 is chance, 1 is a perfect match.
