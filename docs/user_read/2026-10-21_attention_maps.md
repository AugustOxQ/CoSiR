# Each Qwen instruction changes where the model looks; the sharper image focus needs a check

> 2026-10-21 · [Full report](../reports/auto/percept/2026-12-23_attention_maps.md) (its name follows CoSiR's report
> numbering, not the date)

## Summary

- We mapped where Qwen's embedding looks on 5,000 development paintings, under its four instructions.
- Each instruction gives a clearly different image map. Style and genre maps are the most alike.
- With an instruction, the model reads less of the image: about half the share it gives without one.
- On captions, the emotion instruction clearly picks out emotion words. Style and genre do this only in part.
- Instructed image maps are more focused, but often on a border patch. Such patches may be encoder artifacts; not checked.
- A local viewer shows the maps for any painting, instruction and input.

## How we got here

We want one model that finds emotion, style and genre clusters in paintings. Its best signal comes from
Qwen3-VL-Embedding, which gives a different image embedding for each instruction (default, emotion, style, genre).
[Our last ablation](2026-10-21_d2_ablation.md) showed that the graphs built from these instructed embeddings carry most
of our model's gain. In this side step you asked where the instructed model looks, on the image and on the captions.

## Attention maps under the four instructions

**Problem.** We knew that the instructions help, but not what they change inside the model.

**Idea.** Qwen's embedding is the state of its last token, the pooled token. It plays the role of a ViT's class
token. We recorded the pooled token's attention on each image patch and each caption word. We averaged it over the
attention heads and summed it over layers 16 to 27, where the four instructions start to differ.

We first tried rollout, the usual ViT method. On this left-to-right model almost all of its mass goes to the first
token, so you chose the raw attention instead. We ran three inputs: the image alone, each caption alone, and the two
together. The joint input also gives each caption word its own map over the image.

**Did it work.** Yes, as a description. The maps differ far above rounding noise. Attention shows where the model
looks, which is not always what drives the embedding. Development paintings and one model size only.

**Key evidence.** In layers 16 to 27, the image gets 0.22 of the pooled token's attention without an instruction and
0.12 with one. The figure shows how far two instructions' maps are apart (blue bars). The gap is far above rounding
noise (red line) and below a map with the same values at random positions (grey, "shuffled baseline"). Each panel
uses a different set of layers; the bars run from the 10th to the 90th percentile of paintings.

![Image maps: difference between instruction views](assets/2026-10-21_attention_maps/fig_change_l1.png)

On captions alone, the words with the most weight are:

| Instruction | Top caption words |
|---|---|
| default | horses, statue, beach |
| emotion | bored, anger, lonely |
| style | cartoon, abstract, rainbow |
| genre | portrait, abstract, statue |

The border caveat: in layers 16 to 27 the hottest patch is on the border in 63% of emotion maps, against 29% of
default maps. In the last layer, and over all layers, every map peaks on the border most of the time.

> Full report: "Results" and "Analysis".

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: two results are clear. The instructions move the image map and lower the image's share, and the emotion
instruction moves caption attention to emotion words. The sharper instructed image maps are not yet clear, because of
the border patches. A short check of these patches (about an hour) should come before any paper figure uses the maps.
After that, go back to the main loop.

</details>

## Next

The main research loop is at its decision step: choosing the next direction. Your decisions here:

1. Your reading of these results.
2. Whether to check the border patches, test word maps against hand-marked regions, or stop here.

The viewer runs at http://localhost:8765/.

## Glossary

- **Pooled token:** Qwen's last token; its state is the embedding.
- **Attention sink:** the first token, which takes about 60 to 70% of the pooled token's attention in every case.
- **Instruction (view):** the task text given to Qwen (`default`, `emotion_focus`, `style_word`, `genre_word`).
- **Layers 16 to 27:** the layers we sum over by default; the viewer also offers the last layer and all 28 layers.
- **L1 gap:** the summed absolute difference between two maps, from 0 (same) to 2 (no overlap).
- **Rounding noise:** how much 16-bit arithmetic alone moves a map, against full 32-bit precision.
- **Rollout:** a ViT method that multiplies attention through the layers; it failed here.
