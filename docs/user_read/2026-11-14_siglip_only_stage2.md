# The SigLIP-only model passes the added image tests and stays a supplementary result

> 2026-11-14 · Full report: [SigLIP-only stage 2](../reports/auto/percept/2027-01-19_siglip_only_stage2.md)

## Summary

- The added image tests reached their top claim level on other encoders, separate paintings and photos.
- Training adds over each plain embedding in every tested image cell.
- Against the instructed Qwen3, emotion image retrieval is level under the rule on paintings. The other cells are above.
- The removals link graphs and graph-positive loss to style scores, and the relational term to genre scores.
- SigLIP 1 leads the validation encoder pick. The margin remains inside its seed range.
- The earlier text fail stands. The SigLIP-only model stays an extra row, as you and the co-authors decided.

## How we got here

CoSiR learns separate embeddings for emotion, style and genre. An image tower maps images to these embeddings. A caption tower does the same for text.
The current model uses SigLIP 1 for images and Qwen3 for captions.

The SigLIP-only model gives the caption tower SigLIP 1's own text embedding. It needs no Qwen3 pass at test time.
Qwen3 still supplies graphs and targets during training.

The earlier stage failed its validation rule on text. On 2026-11-13, after you saw those results, you approved the added tests anyway.
You and the co-authors kept this model as an ablation or supplementary row. This stage checks how far its image results extend.
The paper deadline is close.

## Other encoders also reach the top image claim

**Problem.** The earlier result covered only SigLIP 1. We did not know whether the caption input change would work with other encoders.

**Idea.** We trained the same adapter on SigLIP 2 and CLIP L/14. Each caption tower reads its own encoder's text embedding.
Both use the caption prefix picked earlier on SigLIP 1. We did not pick a separate caption form for each encoder.
We picked retrieval read-outs on validation, then tested clustering and image retrieval for each aspect.

**Did it work?** Yes, for these image tests. Each encoder reaches the top claim level on its own.
Each beats its plain embedding in every tested cell. Each is level with or above the instructed Qwen3 in every cell.
“Level” uses a validation seed range. It does not prove equality.
These tests do not measure text retrieval for the new encoders.

**Key evidence.** Each encoder has six image cells. The current model on that encoder appears only as a descriptive comparison.

| Encoder | Against its plain embedding | Against instructed Qwen3 |
|---|---|---|
| SigLIP 2 | 6 WIN | 5 WIN, emotion retrieval LEVEL |
| CLIP L/14 | 6 WIN | 5 WIN, emotion retrieval LEVEL |

The differences from each encoder's current model all have intervals across zero.
For CLIP L/14, emotion retrieval is lower than instructed Qwen3 throughout its interval. The rule still calls it level.

> Full report: [other encoders](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#other-encoders-sc4) and [analysis](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#analysis).

## The result extends to separate WikiArt paintings

**Problem.** Results on ArtELingo alone cannot show whether the image model works on other paintings.

**Idea.** We applied the SigLIP-only model to MMArt, a set of WikiArt paintings outside ArtELingo.
No new model was trained on those test paintings. Validation data set the retrieval read-outs and seed ranges.
We compared clustering and image retrieval with plain SigLIP 1 and the instructed Qwen3.
Against plain SigLIP 1, retrieval uses the joined vector of the aspect embeddings. Against instructed Qwen3, it uses the relevant aspect embedding.

**Did it work?** Yes, within these image tests. It reaches the top claim level.
Training adds over plain SigLIP 1 in every cell. The model is level with or above the instructed Qwen3 in every cell.
This does not test MMArt text retrieval.

**Key evidence.** The test covers 16,384 paintings, with fewer labeled rows for emotion and genre.

| Test | Against plain SigLIP 1 | Against instructed Qwen3 |
|---|---|---|
| Clustering | All WIN | All WIN |
| Image retrieval | All WIN | Style and genre WIN, emotion LEVEL |

> Full report: [MMArt](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#mmart-sc5).

## The photo model beats both comparators

**Problem.** A painting model may not carry emotion information across to photos.

**Idea.** We fine-tuned the SigLIP-only model's emotion pathway on Affection photos. Its caption tower reads SigLIP 1 embeddings of photo explanations.
Each training seed starts from the matching painting model checkpoint.
We used validation data to pick the read-outs. We then tested image clustering and retrieval on Affection and EmoSet.
EmoSet test data shaped no training or selection.

**Did it work?** Yes, for image emotion. Every test cell is above plain SigLIP 1 and the instructed Qwen3.
The photo store cuts explanations that exceed the text limit and keeps those rows. These results do not test photo captions at test time.

**Key evidence.** Image retrieval mAP on test data:

| Dataset | Plain SigLIP 1 | Photo model | Instructed Qwen3 |
|---|---|---|---|
| Affection | 0.255 | 0.325 | 0.295 |
| EmoSet | 0.364 | 0.487 | 0.452 |

The clustering comparisons are also above both comparators. All difference intervals are wholly positive.

> Full report: [photos](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#photos-sc6).

## The removals show which training parts support each aspect

**Problem.** A successful full model does not show which parts contribute to its scores.

**Idea.** We removed each training part in turn and retrained the SigLIP-only model. We compared each removal with the full model.
We measured clustering and plain-cosine retrieval for each aspect. We paired the same test rows and training seeds in each difference interval.
The graph-positive loss pulls neighbours in the graphs together. Content correction reduces shared content in those graphs.
The relational term teaches the genre head to match Qwen3's pattern of image similarities.

**Did it work?** The removals support a limited attribution. This test has no claim level.
Graphs and graph-positive loss support emotion and style scores under these settings. The relational term supports genre scores.
A removal measures its effect within the full design. It does not measure interactions between parts.

**Key evidence.** Selected changes from the full model. Negative values mean lower scores after removal.

| Removed part | Aspect | AMI change | mAP change |
|---|---|---|---|
| Instructed graphs | Style | -0.2147 | -0.1416 |
| Content correction | Style | -0.0876 | -0.0674 |
| Graph-positive loss | Style | -0.1695 | -0.1167 |
| Relational term | Genre | -0.0597 | -0.1144 |

Each listed interval is wholly below zero. The full report gives every aspect and interval.
The small genre retrieval gain without content correction depends on shared seed draws. With independent seed draws, its interval crosses zero.

The scorer first refused a stale gate record. A separate check repeated the trainer and frozen-score checks before the ablation was scored.

> Full report: [ablation](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#ablation-sc7) and [agent choices](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#agent-choices-checked).

## SigLIP 1 leads the validation encoder pick narrowly

**Problem.** The earlier encoder pick applied to the current model, whose caption tower reads Qwen3.

**Idea.** We compared the SigLIP-only models on the same validation image group. It averages clustering and plain-cosine retrieval across aspects and seeds.
The highest mean wins. We also checked the margin and whether it was smaller than the leader's seed range.
This pick uses validation only. It sets no test claim and does not change the decided supplementary placement.

**Did it work?** “Among the SigLIP-only models, SigLIP 1 did best on the validation split (0.3882 against 0.3838 and 0.3762).”
The alternatives are SigLIP 2 and CLIP L/14. SigLIP 1 leads in all five matched seeds, but this is a close call.

**Key evidence.** Five-seed image group means on validation:

| Encoder | Mean | Seed range |
|---|---|---|
| SigLIP 1 | 0.3882 | 0.0055 |
| SigLIP 2 | 0.3838 | 0.0061 |
| CLIP L/14 | 0.3762 | 0.0034 |

The margin over SigLIP 2 is 0.0045, below the leader's seed range.
SigLIP 2's first start-up failure was set aside as a wiring fault. If counted as a training crash, this pick would be inconclusive.

> Full report: [encoder pick](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#encoder-pick-sc9) and [deviations](../reports/auto/percept/2027-01-19_siglip_only_stage2.md#deviations).

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: write the tested image claims into the decided extra-row place. The current model stays the main model.
State the earlier text failure beside the image results. The added tests do not cancel it.

Then complete the planned saved-output check of emotion groups. Keep new caption training and encoder-specific prefix searches for after the paper deadline.
The image results support the supplementary story. They do not show equal text performance or general success outside the tested datasets.

</details>

## Next

The added tests are complete. Your reading of these results remains open.
The planned next check asks what the saved emotion groups separate, with a shared map of the embeddings.

The options are to write the supplementary results now, complete that saved-output check, or start new caption experiments after the deadline.
Writing needs paper work and no runs. The saved-output check needs bounded CPU analysis. New caption experiments need validation training and scoring.

The paper chat still needs to choose how image retrieval appears beside a plain embedding: joined aspect vector or individual aspect embedding.
The full report keeps the pre-set joined-vector comparison against plain embeddings. Individual aspect embeddings answer a different retrieval question.

All results use fixed settings and five seeds. Intervals apply to individual cells, with no correction for many comparisons.

## Glossary

- **SigLIP-only model:** the adapter with its encoder's own caption input. The full report calls the SigLIP 1 version K*.
- **Current model:** the adapter with Qwen3 caption input. The full report calls its SigLIP 1 version K0.
- **Plain SigLIP 1:** the frozen encoder's own embeddings, without the adapter. The full report calls this comparator P1.
- **Instructed Qwen3:** Qwen3 under an instruction for each aspect. The full report calls this comparator Q1.
- **Other encoder tests:** image claim tests for SigLIP 2 and CLIP L/14, called E2 and E3 under SC4.
- **MMArt test:** the separate painting image test, called SC5.
- **Photo model:** the SigLIP-only emotion pathway fine-tuned on Affection, called F1 under SC6.
- **Ablation:** the removal tests, called SC7. They have no claim level.
- **Encoder pick:** validation selection among the SigLIP-only models, called SC9 under amendment A2.
- **Run-on-fail amendment:** your decision after the earlier results, called A1. The earlier validation failure is SC2.
- **AMI and mAP:** clustering agreement with labels, and mean average precision for retrieval.
- **Seed range:** the highest score minus the lowest across training seeds on validation.
- **WIN and LEVEL:** the interval rules for “above” and “level with”. LEVEL allows a loss within the stated seed-range rule.
- **Independent re-derivation:** a separate recomputation from saved outputs, called C13.
