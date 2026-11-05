# c* wins most cells against TPIPS, CLAY and the plain image under default-settings TURTLE

> 2026-11-05 · [Full report](../reports/auto/percept/2027-01-06_outside_methods.md)

## Summary

* We put c*, our student network, beside outside methods on two painting sets, in clustering and retrieval.
* Each result gets its own label and there is no overall verdict. Of 337 cells, 330 are WIN for c*, the label that allows "better than" for that cell.
* Against TPIPS, a text-conditioned embedder on a much larger base model, one cell is not WIN. Against CLAY, a text projection of CLIP, every cell that is not WIN concerns style.
* We also clustered with TURTLE at its released code's default settings, without its hyperparameter selection. Against the plain image under that TURTLE, c* has no BELOW cell.
* In a diagnostic with no label, default TURTLE scored below k-means in 203 of 232 comparisons. So these cells do not show that c* holds against a clustering stronger than k-means.
* TEMI, a second clustering method, did not run. The independent check recomputes the scoring from saved outputs, and it redoes no feature extraction or training.

## How we got here

We want a model that separates paintings by emotion, style and genre.
We call each of these an aspect.
c* is our student network, with one head for each aspect.
The plain image is the encoder's own embedding, with nothing on top.
Qwen, SigLIP 2, SigLIP 1 and CLIP L/14 are the four frozen image encoders we use.

Readers will ask two things that earlier rounds leave open.
First, how does c* compare with outside methods that name the aspect in text?
Second, does c*'s lead over the plain image hold when k-means gives way to another clustering method?

This step answers both on ArtELingo's held split and the MMArt test set.
c*'s test values were known from earlier rounds, and the outside methods' values were new.
The results finished on 2026-11-05. The abstract is due 2026-11-10, and later results enter the paper before 2026-11-16.

A cell is one metric on one set, encoder, aspect and cluster count.
AMI, NMI, ARI and accuracy score clustering, and mAP scores retrieval.
Each cell takes one label from the difference, c* minus its comparator, and that difference's 95% interval:

* **WIN:** the lower bound is above zero. We may write "better than" for that cell.
* **LEVEL:** not WIN, and the lower bound stays above minus c*'s validation seed range. We may write "level with", which does not establish equality.
* **BELOW:** any other case. The label gives no claim for c*. The comparator is better only if the whole interval is below zero.

The seed range is the highest minus the lowest validation score across c*'s seeds.
Claims hold at the class count, the number of label classes, and at twenty clusters only.
No correction covers the number of cells.

In the table, default TURTLE means TURTLE at its released code's default settings, without its hyperparameter selection.

|Comparison|WIN|LEVEL|BELOW|
|---|---:|---:|---:|
|Qwen c* against TPIPS, clustering|28|1|0|
|Qwen c* against TPIPS, retrieval|6|0|0|
|Qwen and CLIP L/14 c* against CLAY, clustering|54|3|1|
|Qwen and CLIP L/14 c* against CLAY, retrieval|12|0|0|
|c* against plain images, default TURTLE on both sides|114|2|0|
|c* with k-means against plain images with default TURTLE|116|0|0|

The independent check matched the scoring from saved outputs and recomputed CLAY's vectors.
It redoes no feature extraction or training.

## TPIPS: most cells favour c* on Qwen

**Problem.** A reviewer can ask how c* compares with an outside method that embeds an image under a named aspect.

**Idea.** TPIPS takes a factor named in text and produces an image embedding for it.
We compare its aspect embeddings with c* on Qwen.
Its base model is four times c*'s size.
The comparison covers clustering and retrieval on both painting sets.

The clustering comparison uses k-means on each method's vectors.
Retrieval settings were picked on validation data before the test results.

The claim covers Qwen c* only.
TPIPS ran with one library version, and its vectors change with that version.
We did not check whether it saw WikiArt paintings in training.

**Did it work.** Most cells permit "better than" for c*: clustering has 28 WINs out of 29, and retrieval has six WINs.
The remaining clustering cell is LEVEL for ArtELingo emotion ARI.
No claim reaches other encoders, or cluster counts other than the class count and twenty.

**Key evidence.** These ArtELingo emotion examples compare c* on Qwen with TPIPS.

|Task and metric|c*|TPIPS|
|---|---:|---:|
|Clustering, AMI at nine clusters|0.1024|0.0718|
|Retrieval, mAP|0.2691|0.2565|

The LEVEL emotion ARI cell has difference +0.0056 [-0.0001, +0.0119].
Its interval reaches below zero, while the lower bound stays within c*'s seed range.

> Full report: [Results](../reports/auto/percept/2027-01-06_outside_methods.md#results) and [TPIPS extraction limits](../reports/auto/percept/2027-01-06_outside_methods.md#deviations).

## CLAY: style contains the exceptions

**Problem.** A reviewer can ask whether a text projection of frozen CLIP, with no training, already gives useful aspect embeddings.

**Idea.** CLAY projects a CLIP image embedding through text values for an aspect.
The texts provide an aspect-specific space for the image vectors.
We use CLIP L/14 and compare CLAY with c* on both CLIP L/14 and Qwen.
This gives a comparison on the same encoder and a comparison with Qwen c*.

We had two text lists: values from earlier work, and values we generated.
Validation data chose between them for each task and painting set, and the earlier list won every time.
The chosen list holds four of nine emotion class names, three of ten genre names and one MMArt style name. We removed none.

CLAY uses the scored set's mean image without labels.
Its clustering uses k-means, while its retrieval uses the validation pick.
The test results selected no setting.
No claim reaches other encoders for CLAY, or cluster counts other than the class count and twenty.

**Did it work.** Clustering has 54 WIN cells, three LEVEL cells and one BELOW cell.
Retrieval has twelve WIN cells.
Every non-WIN cell concerns style.
The BELOW cell gives neither method a "better than" claim, although c*'s value is slightly higher.

**Key evidence.** The retrieval example is a WIN; the style accuracy example is BELOW under the rule.

|Cell|c*|CLAY|
|---|---:|---:|
|ArtELingo Qwen emotion retrieval mAP|0.2691|0.2486|
|MMArt CLIP L/14 style accuracy at twenty clusters|0.3518|0.3510|

The style accuracy difference is +0.0008 [-0.0104, +0.0113].
The lower bound is below minus the seed range (0.0081), so the rule gives no claim for c*.
The upper bound is above zero, so "CLAY better than c*" cannot be claimed either.

> Full report: [Results and every non-WIN cell](../reports/auto/percept/2027-01-06_outside_methods.md#results) and [CLAY analysis](../reports/auto/percept/2027-01-06_outside_methods.md#analysis).

## TURTLE: these cells do not show a clustering stronger than k-means

**Problem.** A reviewer can ask whether c*'s advantage depends on using k-means for clustering.
TURTLE, at its released code's default settings and without its hyperparameter selection, supplies a different way to cluster the same vectors.

**Idea.** Default TURTLE fits clusters on the scored images without reading their labels.
We compare two procedures across Qwen, SigLIP 2, SigLIP 1 and CLIP L/14.
First, we apply default TURTLE to both c*'s vectors and the plain image vectors.
This tests the two spaces with the same clustering method.

Second, we keep c* with k-means and compare it with plain images clustered by default TURTLE.
This asks whether the plain image reaches c* when its clustering procedure changes.
The LEVEL margin remains c*'s k-means validation seed range.

Default TURTLE is our re-implementation.
We checked it against the official code only in a short run.
The comparison covers these released defaults without hyperparameter selection.
It does not establish how selected settings would perform.

**Did it work.** With default TURTLE on both sides, c* has 114 WIN cells and two LEVEL cells.
With c* using k-means, all 116 cells against plain images with default TURTLE are WIN.
"Better than" applies separately to each WIN cell.
The closest cells are in emotion: in eleven of them the difference is smaller than c*'s seed range. WIN reads only the interval, so ten of the eleven are still WIN, and one is LEVEL.

On the same vectors, default TURTLE scored below k-means in 203 of 232 diagnostic comparisons.
This diagnostic has no label.
On plain images, ARI and accuracy are mixed, so the direction depends on the metric.
These results do not establish performance against clustering stronger than k-means.

TEMI, which trains clustering heads through neighbours and a teacher copy, did not run.

**Key evidence.** These ArtELingo Qwen emotion examples use AMI at nine clusters.
Each is a WIN against the plain image with default TURTLE.

|c* procedure|c* AMI|Plain image with default TURTLE AMI|
|---|---:|---:|
|k-means|0.1024|0.0669|
|Default TURTLE|0.0998|0.0669|

> [Full report: Analysis on what default TURTLE adds, and its limits](../reports/auto/percept/2027-01-06_outside_methods.md#analysis)

## Our advice

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our reading.** c* stands above the outside embeddings in most cells, with CLAY closest in style.
The default TURTLE cells provide limited evidence about stronger clustering because the comparator usually scores below k-means.
The independent check supports the scoring from saved outputs; it redoes no feature extraction or training.

**Our view.** Put these results into the paper now.
The abstract needs no new run.
Then build TEMI and the validation selection for TURTLE together under one spec.
A reader will ask whether TURTLE was tuned.
Keep the released-defaults clause beside its claims until that comparison exists.

</details>

## Next

The reported comparisons are complete. TEMI did not run, and its own build comes later.
You decide what these results mean and which option to take.
Head variants and text queries also remain on the plan.

|Option|Cost|What it could tell us|
|---|---|---|
|Put these results into the paper's tables, related work and method text.|Writing only, with no GPU run.|It shows readers the existing comparisons and adds no evidence.|
|Build and run TEMI.|One unattended build and up to ten GPU hours.|It could show whether a second method, at its paper's settings, closes c*'s gap. The ten-hour limit may leave c*'s heads without a result.|
|Select TURTLE settings on validation, then repeat plain-image TURTLE.|A small spec, a build and about 134 TURTLE fits. About four GPU hours by our arithmetic, not measured.|It could show whether selected TURTLE reaches c* with k-means. A selected setting may change fit time.|

> [Full report: Results brief, including the options and their costs](../reports/auto/percept/2027-01-06_outside_methods.md#results-brief)

## Glossary

|Term in the full report|Meaning here|
|---|---|
|Aspect|Emotion, style or genre.|
|c*|Our student network, with a head for each aspect.|
|Plain image|The encoder's own embedding before the student transforms it.|
|TPIPS|The outside method that embeds an image under a factor named in text.|
|CLAY|The method that projects CLIP image embeddings through text values without training.|
|Default TURTLE|The label-free clustering method at its released code's default settings, without its hyperparameter selection.|
|TEMI|The clustering method that trains heads through neighbours and a teacher copy; it did not run here.|
|ArtELingo held; MMArt test|The painting splits used for these comparisons.|
|AMI; NMI; ARI; ACC|Clustering metrics; ACC is accuracy.|
|mAP|The retrieval score.|
|Cell|One metric result for a comparison, set, encoder, aspect and cluster count where applicable.|
|Seed range|Highest minus lowest validation score across model seeds.|
|WIN; LEVEL; BELOW|The rule's separate label for each cell; there is no overall verdict.|
|Independent check (full report: independent re-derivation, C13)|Separate code that recomputes the scoring from saved outputs, and CLAY's vectors; it redoes no feature extraction or training.|
