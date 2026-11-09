# The parts of c* that matter for clustering also matter for retrieval

> 2026-11-09 · Full report: [ablation in image retrieval](../reports/auto/percept/2027-01-13_ablation_retrieval.md)

## Summary

- We took c* apart one piece at a time and measured image retrieval for each version. Before, we had clustering only.
- The same pieces matter in both tasks. The instructed graphs carry emotion and style; the relational term carries genre.
- Without the instructed graphs, emotion and style retrieval fall to about the plain image's level.
- Without the relational term, genre retrieval falls below the plain image.
- Small effects mostly match too. Of 24 comparisons (8 pieces, 3 aspects), the two tasks differ on 6, all small.
- Both paper figures are built, and you kept them as they are.

## How we got here

c* is our model. It takes one image and gives three embeddings: emotion, style and genre. The paper's ablation removes one piece of its training at a time. An earlier report (the recipe sweep) measured these eight versions in clustering only. The paper's second task is retrieval, so this step asks whether the ablation holds there too. The same small loop also built two figures for the paper.

## Retrieval for the ablation

**Problem.** A reviewer can ask whether the pieces of c* matter for retrieval, not only for clustering. We had no numbers for that.

**Idea.** Reuse the eight saved versions of c* and score them in image-to-image retrieval. The queries are ArtELingo's held paintings, which no training step or choice used. A hit is a painting with the query's label in that aspect. Each version has five training seeds.

**Did it work.** Yes. Every large change in clustering also shows in retrieval, with the same sign. This is an exploration with no pass threshold set in advance, so it decides nothing by itself.

**Key evidence.** Change in retrieval score (mean average precision) when one piece is removed; a negative value means retrieval gets worse. c* scores 0.26, 0.31 and 0.59 (emotion, style, genre); the plain image 0.24, 0.18 and 0.48. Each change of 0.005 or more is outside its 95% interval.

| Piece removed | Emotion | Style | Genre |
|---|---|---|---|
| Instructed graphs | -0.021 | -0.122 | 0.001 |
| Buddy terms | -0.018 | -0.098 | 0.000 |
| Content correction | -0.007 | -0.058 | -0.001 |
| Relational term | 0.000 | -0.001 | -0.129 |

The other four pieces change each aspect by less than 0.01.

> Full report, Results.

## The two paper figures

The cluster figure shows one class per aspect. For each model, c* and the plain image, it shows the cluster where that class has its largest share. In c*'s cluster the share is larger: Ukiyo-e is 93% against 14%. Each class is the one where c* gains most, so the figure shows the largest difference among the classes. The supplementary will show all classes.

The teaser shows one painting and its five nearest neighbours in each aspect. The painting came from candidates that already had at least 3 of 5 neighbours with matching labels in each aspect. You saw every candidate first. You picked the teaser painting, and we picked the cluster design at your request. So the captions state the rule, but not that it was fixed in advance.

> Full report, Results ("Figures") and Deviations.

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our reading.** Retrieval confirms the clustering ablation for every large piece. We are confident in this: the intervals of the large changes are far from zero, over five seeds, in two tasks.

**Our recommendation.** Add the retrieval column to the paper's ablation table; it costs only writing. Later, with the supplementary work, make a version of the figures that compares with the teacher.

</details>

## Next

You give your reading of the retrieval table, then we wrap up this loop. One small choice is yours: whether the figures also get a teacher version for the supplementary.

## Glossary

- **c\*:** our model (one trunk, three aspect heads), trained on ArtELingo paintings.
- **Plain image:** the frozen Qwen image embedding with no instruction.
- **Teacher:** the Qwen image embedding given an instruction for one aspect; c* learns from it.
- **Instructed graphs:** for each aspect, links between paintings that the teacher sees as close.
- **Buddy terms:** the training losses that pull linked paintings together.
- **Content correction:** a step that removes links caused by shared subject matter.
- **Relational term:** a loss that makes the genre head copy the teacher's pattern of similarities between paintings.
