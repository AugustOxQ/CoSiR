# SigLIP 1 now has the paper's results, and you kept the reference settings

> 2026-11-12 · Full report: [SigLIP 1 main result](../reports/auto/percept/2027-01-16_siglip_main.md)

## Summary

- We completed the scores needed to make SigLIP 1 the paper's main image encoder.
- Validation supports that encoder choice, although the lead over SigLIP 2 is small.
- The component removals repeat the earlier pattern: instructed graphs matter most for style, and the relational term for genre.
- Two settings beat the reference on one validation seed, within its seed spread. You kept the reference settings, so we did not retrain.
- The text replay used a looser tolerance than authorized. An independent check confirmed the inference path.

## How we got here

CoSiR learns separate image embeddings for emotion, style and genre. Qwen3 supplies the instructed examples that guide training.
The earlier main model also read Qwen3 images. You moved the main result to SigLIP 1, a lighter image encoder.
This round completed the missing comparisons while the caption side kept Qwen3 text embeddings.

## Complete the tables and check the settings

**Problem.** The encoder change left gaps in the paper's tables and left the earlier training settings unchecked.

**Idea.** Score saved models, match the outside methods to SigLIP 1, and choose every setting on validation data.
Remove each of four training components in turn to see which results depend on it. Prepare figure candidates for your choice.

**Did it work?** Yes for the requested results. The two settings leads are small and use only one seed, so keeping the reference fits.
The encoder lead and the style comparison with CLAY also vary across seeds. The caption replay needs your tolerance decision.

**Key evidence.** Test clustering agreement, averaged over model seeds:

| Model | Emotion AMI | Style AMI | Genre AMI |
|---|---:|---:|---:|
| CoSiR on SigLIP 1 | 0.099 | 0.441 | 0.528 |
| Instructed Qwen3 | 0.089 | 0.406 | 0.501 |
| Plain SigLIP 1 | 0.061 | 0.382 | 0.402 |

The image retrieval means also exceed instructed Qwen3. Text retrieval remains below it for emotion.
These are descriptive comparisons, with no new statistical test. The speed comparison covers test time only.

> Full report: [results and limits](../reports/auto/percept/2027-01-16_siglip_main.md#results).

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: keep the reference settings, as you decided. A check of both leading settings on the other seeds costs about eight short GPU runs.
It is not needed unless the learning rate matters in the next round. Resolve the caption replay tolerance explicitly.

</details>

## Next

The reference settings stay. The learning rate can come up again in the next round's spec.
The optional seed check can show whether the small leads repeat.
The figure candidates await your selection. No candidate was chosen for you.
The caption replay tolerance also needs your decision.

## Glossary

- **AMI:** adjusted mutual information, a measure of agreement between clusters and labels.
- **Instructed graphs:** neighbours defined by Qwen3 under an aspect's instruction.
- **Relational term:** the training loss that copies the instructed model's pattern of genre similarities.
