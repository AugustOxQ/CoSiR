# c* scores above VLM2Vec-V2 and text-only BERTopic in all 12 ArtELingo test cells

> 2026-11-11 · [Full report](../reports/auto/percept/2027-01-14_paper_baselines.md) (CoSiR names reports by a sequence date, here 2027-01-14)

## Summary

- The paper needed two outside clustering baselines beside c*, our model: VLM2Vec-V2 and BERTopic.
- We scored both on the ArtELingo test paintings, with the same labels and cluster counts as c*.
- c* is above both in all 12 cells: 3 aspects, 2 cluster counts, 2 methods. Each cell passes its own interval test.
- BERTopic reads captions. c* and VLM2Vec-V2 read the images. The user chose this design.
- The paper chat already has the numbers. A paper-row cutoff applies (see Next).

## How we got here

c* learns one image space per aspect (emotion, style, genre) for paintings. The paper compares it with PercepT, an
earlier paper on the same task, which used BERTopic and VLM2Vec as clustering baselines. On 2026-11-11 the user added
both to the paper's main clustering table. ArtELingo is a set of WikiArt paintings with emotion captions and style and
genre labels.

## Two outside baselines for the paper's clustering table

**Problem.** The paper's main clustering table had empty rows for two outside methods.

**Idea.** Each method puts the test paintings into clusters. AMI (adjusted mutual information) measures how well the
clusters follow an aspect's labels: 0 is chance, higher is better. VLM2Vec-V2 embeds each painting under the aspect's
instruction, and k-means makes the clusters. BERTopic is a topic model on text: it joins each painting's captions into
one document and gives each painting one topic.

We scored at each aspect's class count (emotion 9, style 27, genre 10) for the paper's table, and at 20 clusters for
the supplementary. A cell is a WIN when the 95% interval of c* minus the method stays above zero. A WIN allows "better
than" for that one cell only.

**Did it work.** Yes for these 12 cells, on ArtELingo test with the Qwen version of c*. The closest cell is
VLM2Vec-V2 on emotion at 20 clusters: +0.0356 AMI [+0.0320, +0.0391]. Caveats:

- Captions often name the emotion, so BERTopic's emotion cells may read label words. We did not measure the size.
- BERTopic's topics are very unbalanced: at 20 clusters the four largest hold about 8,000 to 9,000 of 12,281
  paintings. This probably lowers its AMI.
- Emotion AMI is low for every method (c* 0.1024), so the emotion gaps are small in absolute terms.
- Not checked: topics per caption, the image-and-text BERTopic of the PercepT paper, other VLM2Vec-V2 prompts and image
  sizes. A side check, k-means on BERTopic's caption embeddings, also gives six WINs, with a smaller emotion gap
  (+0.0151).

**Key evidence.** Mean AMI at each aspect's class count, higher is better (12,281 test paintings; genre: the 9,949
with labels):

|Aspect, class count|c*|VLM2Vec-V2|BERTopic|
|---|---:|---:|---:|
|emotion, 9|0.1024|0.0638|0.0375|
|style, 27|0.4109|0.3730|0.1050|
|genre, 10|0.5101|0.4593|0.2605|

> [Full report: all 12 cells with intervals, and the setup limits](../reports/auto/percept/2027-01-14_paper_baselines.md#results)

<details>
<summary>Our advice (open after giving your reading)</summary>

Our view: put these rows into Table 2 now, before the cutoff. Beside BERTopic, state that it reads
captions, that captions often name the emotion, and that its topics are unbalanced. We are sure of the scoring: c*'s
numbers equal the earlier round exactly, and an independent recompute matches all 12 cells. A further run helps only
if it answers a named concern, such as the text-only caveat.

</details>

## Next

Your reading of these results comes first. Then the choice, with the paper-row cutoff on 2026-11-14:

1. Put these rows into Table 2 as they are. Cost: writing only.
2. Run BERTopic on images and captions, as the PercepT paper did. Cost: new features and CPU fits, not measured. It
   could answer the text-only caveat.
3. Run VLM2Vec-V2 on MMArt, the paper's second painting set. Cost: a new GPU extraction and scoring, not measured.

> [Full report: options and costs](../reports/auto/percept/2027-01-14_paper_baselines.md#results-brief)

## Glossary

|Term|Meaning|
|---|---|
|c*|Our model: one image space per aspect, here on the Qwen encoder.|
|AMI|Adjusted mutual information: agreement between clusters and labels; 0 is chance.|
|WIN|The 95% interval of c* minus the method is above zero for that cell.|
|VLM2Vec-V2|An outside model that embeds images under a text instruction.|
|BERTopic|An outside topic model that groups text documents into topics.|
|Class count|The number of label classes of an aspect.|
