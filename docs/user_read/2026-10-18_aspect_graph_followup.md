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
- **Prompted embedding:** Qwen3-VL-Embedding with an instruction that names the aspect; "default" has no instruction.
- **Stronger side:** the side where the prompted embedding has the higher AMI: captions for emotion, images for
  style and genre.
- **AMI:** adjusted mutual information between the clusters and the labels; 0 is chance, 1 is a perfect match.
