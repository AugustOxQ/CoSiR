# The aspect graph alone has the cleanest pairs, but graphs that subtract content train better networks

> 2026-10-18 (Amsterdam). Full report:
> [aspect graph sheet, stages 0 and 1](../reports/auto/percept/2026-12-11_aspect_graph_sheet.md).

## Summary

- We built 21 kinds of pair graph for each aspect (emotion, style, genre) and scored all 63 the same way.
- For emotion and style, the plain aspect graph has the pairs that agree most on the aspect. Every way to remove
  content that we tried lowers that agreement.
- When we train a small network on each graph, graphs that subtract content similarity give the best emotion and
  style clusters.
- For genre, the image side carries the gain: a graph that weighs images more beats the caption graph.
- For emotion and style, no trained network reaches the prompted embedding on the side the aspect uses. This is
  the main risk for the next stage.

## How we got here

The new line builds a model that finds aspects (emotion, style, genre) without conditions. It must improve over
PercepT, a published method for perceptual topics. For each aspect we give the embedding model (Qwen3-VL-Embedding)
one instruction that names the aspect. This gives one prompted embedding per aspect. On it we build a buddy
graph: pairs of paintings that rank among each other's 20 nearest neighbours.

The pilot before this one showed that pairs which share content (the same subject) are also the pairs that agree most
on emotion. So removing them blindly hurts. You asked for a broad sweep with one table of numbers: prune, fuse, or let
pruning guide fusion. This report covers its first two stages.

## Which graph gives the cleanest aspect pairs?

**Problem.** An aspect graph mixes two kinds of pairs: pairs with the same aspect, and pairs with the same subject.
We want the first kind without losing good pairs.

**Idea.** Build each aspect graph in 21 ways and compare them. Some ways prune: they drop pairs whose content is too
similar. Some fuse: they rank pairs by aspect similarity minus content similarity (a residual graph), or mix the
caption and image sides. Some guide: they use the content overlap to weigh or pick pairs. We score each graph by its
lift: how often its pairs share a label, over the rate for random pairs.

**Did it work?** For emotion and style, no new graph beats the plain aspect graph on lift. The pairs with the most
similar content agree most, on every label. For genre, a graph with images at 0.75 and captions at 0.25 beats the
caption graph. The image graph alone has the same lift (to two decimals), so the image side is the likely reason.
First check passed: we rebuilt the pilot's graphs exactly.

**Key evidence.** Lift for the aspect's own label, on 13,400 training paintings:

| Aspect | Plain aspect graph | Best other graph |
|---|---:|---:|
| Emotion (captions) | 2.42 | 2.42 is the best; all others lower |
| Style (images) | 7.07 | 7.07 is the best; all others lower |
| Genre (captions) | 4.41 | 5.35 (images 0.75, captions 0.25) |

> Full report: "Results" (graphs and heads) and "Analysis", first bullet.

## Which graph trains the best network?

**Problem.** Lift counts pairs. It does not tell which pairs teach a network the aspect.

**Idea.** On five or six graphs per aspect, train the same small network. It reads the default embeddings (Qwen
with no aspect instruction) of each image and caption, and learns to pull each pair together. Then we cluster its
outputs and compare the clusters with the labels (AMI). We average the image side and the caption side.

**Did it work?** Partly. The residual graphs give the best networks for emotion and style, although their lift is
lower. Their pairs share the aspect, but share the other labels less often. Our reading: the network then learns to
ignore the other labels. For genre, the image-weighted graph wins; a network on the image graph alone was not tried.
For emotion and style, every network stays below the prompted embedding on the aspect's own side.

Caveats. Development data only. We picked the best graphs with the labels, on the same paintings, and the plain
graphs for emotion and style have one seed. The best emotion graph leaves more than half the paintings without a
pair. Each network reaches its best development loss within four epochs, so it learns little; it is a cheap test of
each graph.

**Key evidence.** AMI at 20 clusters, 5,000 paintings, mean of the image and caption side; 3 seeds unless marked:

| Aspect | Network on the plain graph | Best network |
|---|---:|---:|
| Emotion | 0.120 (1 seed) | 0.137 (residual) |
| Style | 0.218 (1 seed) | 0.260 (residual) |
| Genre | 0.402 | 0.420 (image-weighted) |

The same networks on one side at a time, against the prompted embedding with no training:

| Aspect, side | Prompted embedding | Best network |
|---|---:|---:|
| Emotion, captions (its side) | 0.217 | 0.194 |
| Style, images (its side) | 0.395 | 0.348 |
| Genre, images | 0.496 | 0.454 |
| Style, captions (other side) | 0.135 | 0.172 |
| Genre, captions | 0.353 | 0.386 |
| Emotion, images (other side) | 0.093 | 0.091 |

One gain is in the last three rows: for style and genre, the network carries the aspect to the other side. For
emotion, the image side stays weak.

> Full report: "Results", the second table, and "Analysis".

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** Drop the pruning graphs. Carry the plain aspect graph and the best residual graph for emotion and
style, and the image-side graphs for genre. Go on to the shared model (stage 2), but start it with two cheap steps:
fill the missing seeds and test a network that reads the prompted embeddings; and run PercepT core (our
reimplementation of PercepT on our features), to learn early whether the prompted embeddings already beat it.

</details>

## Next

Your reading of these results comes first. Then you choose the direction (decision A), and then you approve the spec.
The spec's next stage is one network with a shared body and one output head per aspect. It is compared with the
prompted embeddings and with PercepT core (our reimplementation of PercepT).

## Glossary

- **Aspect graph**: the buddy graph of one aspect's prompted embedding (full report: R0).
- **AMI** (adjusted mutual information): agreement between a clustering and the labels; 0 is chance.
- **Buddy pair**: two paintings that rank among each other's 20 nearest neighbours in one embedding.
- **Lift**: how often a graph's pairs share a label, divided by the rate for random pairs.
- **Prompted embedding**: Qwen3-VL-Embedding with one aspect instruction (full report: prompted view).
- **Residual graph**: neighbours ranked by aspect similarity minus content similarity (full report: Fres, Gres).
