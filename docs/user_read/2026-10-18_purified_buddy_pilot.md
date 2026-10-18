# Emotion pairs from captions help an image-only network a little; removing content pairs does not help more

> 2026-10-18 (Amsterdam). Full report:
> [the purified buddy pilot](../reports/auto/percept/2026-12-10_purified_buddy_pilot.md).

## Summary

- We trained a small image-only network on pairs of paintings whose emotion captions are close.
- It finds more emotion in images than a control trained on ordinary caption pairs, but not more than the
  emotion-prompted image alone.
- Removing the pairs that are also content pairs (same subject) adds almost nothing.
- The pairs that share content are also the pairs that agree most on emotion. This is a likely reason, not checked.
- A content graph from CLIP shows the same pattern, so it is not only an effect of the shared Qwen model.
- In this pilot, image-only emotion stays below the bar. The next step compares ways to combine the graphs.

## How we got here

The new line builds a model that finds aspects (emotion, style, genre) without conditions. It must improve over
PercepT, a published method that finds perceptual topics in images. Its central idea: prompt Qwen3-VL-Embedding with
one instruction per aspect. Each instruction gives one embedding set (a view). On each view we build a buddy graph:
pairs of paintings that rank among each other's 20 nearest neighbours.

A night pilot on 2026-10-17 trained an image-only network toward each painting's own emotion caption. It did not
learn caption emotion. This pilot asked if pairs work better.

## Can caption emotion pairs teach an image-only network?

**Problem.** Toward its own caption, the network learned mostly content, the part of a caption that the image can
predict.

**Idea.** Train on pairs instead. Two paintings form a pair when their emotion-instructed captions are buddies. The
network sees only the two images and pulls them together. So it learns from pairs chosen with the captions, and does
not predict the caption vector itself. A second form, the purified one, drops every emotion pair that is also a
content pair.

**Did it work?** Partly. The pairs carry more emotion than the control pairs, but the result stays below the bar
(0.12) and level with the emotion-prompted image. Purification adds 0.004, about two seed spreads (the spread
between runs with different random seeds). One sample, the small (2B) Qwen model and one network design.

**Key evidence.** Image emotion AMI at 20 clusters, 5,000 paintings, 3 seeds:

| View | Image emotion AMI |
|---|---:|
| Default image | 0.070 |
| Control network (ordinary caption pairs) | 0.070 |
| Network on emotion pairs | 0.090 |
| Network on purified emotion pairs | 0.094 |
| Emotion-prompted image, no training | 0.093 |
| Teacher (from captions) | 0.217 |

Pair check: we count how often a graph's pairs share an emotion label, over the rate for random pairs.
- Emotion pairs that are also content pairs: 2.55 times the random rate.
- The other emotion pairs (the purified graph): 2.38.
- With a CLIP content graph: 2.61 for the shared pairs, 2.42 for all emotion pairs.

> Full report: [the purified buddy pilot](../reports/auto/percept/2026-12-10_purified_buddy_pilot.md), "Results" and "Analysis".

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** Content and emotion go together in art, so pruning the shared pairs throws away the best emotion
pairs. Do not prune blind: compare pruning, fusion and pruning that guides fusion in one sweep, with the same
scores. Keep one run with the larger (8B) Qwen model as the last image-emotion run.

</details>

## Next

You gave your reading before ours. The next step is an exploration sweep. It tries many options under one protocol,
and gives one table of numbers (the number sheet). It starts with how to combine the aspect graphs with the content
graph. Then come the model with one head per aspect, retrieval and the baselines. Decision A (the direction) still
needs your confirmation.

## Glossary

- **AMI** (adjusted mutual information): agreement between a clustering and the labels; 0 is chance.
- **Buddy pair**: two paintings that rank among each other's 20 nearest neighbours in one embedding.
- **Content pair**: a buddy pair in the default caption or default image embedding.
- **Purified pairs**: emotion pairs minus content pairs (full report: S_P).
- **Control network**: the same network trained on default caption pairs (full report: S_D).
- **Teacher**: the average of a painting's emotion-instructed caption embeddings.
