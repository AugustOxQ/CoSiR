# On ArtELingo, the teacher's lead in ARI and ACC comes from one large class

> 2026-11-05 · [Full report](../reports/auto/percept/2027-01-07_teacher_gap.md)

## Summary

* c*, our small model on a frozen image encoder, often misses its teacher in ARI and ACC, while it wins in AMI and NMI.
* Most of those misses show no clear difference; 11 of the 26 are clearly lower.
* On ArtELingo emotion, one class explains the gap: contentment, the largest class.
* The teacher forms a larger cluster for that class, mostly landscapes, at about the same precision.
* On MMArt the reason is partial: fear also matters, and we did not test fear.
* Genre labels alone match both models in ACC on ArtELingo, which a reviewer may ask about.

## How we got here

We build c*, a small network on a frozen image encoder, with one output head each for emotion, style and genre.
Its teacher is the embedding of Qwen, a large image model, instructed for one aspect.
The last round scored c* with four standard cluster metrics on two painting sets, ArtELingo and MMArt.
Against the teacher, 46 of 48 AMI and NMI results won, but only 22 of 48 ARI and ACC results did.
You asked for the reason.

## Why ARI and ACC miss the teacher

**Problem.** We did not know why two metrics favour c* and two do not.

**Idea.** ARI and ACC weigh large classes more than AMI does. So we split each gap into one part per class.
Then we left one class out of the score after clustering and scored again.
We also weighted all classes equally, and compared the clusters with the genre labels.
This used saved outputs only, with no training.

**Did it work.** Yes on ArtELingo; partly on MMArt.

* **ArtELingo.** Contentment carries the whole gap, and the other eight classes favour c*.
* **With contentment left out.** c* is clearly above the teacher in ARI on all four encoders.
  In ACC it is clearly above on two encoders. On the other two the difference could be zero.
* **Controls.** Leaving out sadness, amusement or awe keeps the ARI gap. These classes are smaller than contentment.
* **MMArt.** ARI has no gap. In ACC, contentment and fear carry the gap together.
* **Validation rows.** The ArtELingo gap is smaller and mostly absent there.

This shows where the gap sits, not what in training causes it.
We chose these tests after seeing the results, so they give no claim. The results of the last round stay as they are.

**Key evidence.** The cluster that matches contentment, on ArtELingo:

| | Teacher | c*, four encoders |
|---|---:|---:|
| Share of all rows | 20% | 15% to 16% |
| Precision for contentment | 0.65 | 0.67 to 0.69 |
| Contentment landscapes caught | 80% | 58% to 65% |

Genre labels used directly as the clusters, on ArtELingo emotion, with c* on Qwen:

| Metric | Genre labels | Teacher | c* |
|---|---:|---:|---:|
| AMI | 0.062 | 0.086 | 0.099 |
| ARI | 0.055 | 0.077 | 0.062 |
| ACC | 0.258 | 0.263 | 0.256 |

> [Full report: Results and Analysis](../reports/auto/percept/2027-01-07_teacher_gap.md#results)

## Our advice

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** Use this as analysis text beside the teacher results. Keep AMI beside ARI and ACC.
Say that emotion clusters on paintings track genre in both c* and the teacher.
Add genre labels as a stated baseline row, with its test fixed before we look, and with intervals.
Do not expect a new head to close this gap; we did not test that.

</details>

## Next

The paper needs this text. You decide the next step, including a genre baseline row (under 1 hour of CPU).
Finding where c*'s other landscape rows go takes minutes. It matters only if the paper needs the mechanism.

## Glossary

|Term in the full report|Meaning here|
|---|---|
|Aspect|Emotion, style or genre.|
|c*|Our small model on a frozen image encoder, with one head per aspect.|
|Head|A small output layer of c*, one per aspect.|
|Teacher|The Qwen image embedding instructed for an aspect.|
|ArtELingo, MMArt|Our two painting test sets.|
|AMI, NMI, ARI, ACC|The standard cluster metrics.|
|Precision|The share of a cluster's rows that belong to the class.|
