# The frozen painting model beats the plain image and PercepT on new MMArt paintings, on all four encoders

> 2026-11-03 · [Full report](../reports/auto/percept/2027-01-03_mmart_test.md) (its file name uses the project's
> sequence date, 2027-01-03)

## Summary

- Our painting model, c*, passes the decision rule on all four image encoders, with no new training.
- On 16,384 paintings outside its training set, it beats the plain image in clustering and retrieval.
- It also beats PercepT, the method we improve on, in clustering.
- Against the teacher, every comparison chosen in advance for a "better than" claim is a win.
- Emotion retrieval is below the teacher for two encoders, so those two give no teacher claim.
- The paintings and labels are new, but most artists also appear in the training paintings.

## How we got here

The project builds an image-text model that finds aspects of paintings, label-free if possible.
An aspect is a way to group paintings, such as emotion, style or genre.
The model gets no aspect instruction at test, and it should improve on PercepT.
PercepT is a published method that finds perceptual topics, such as emotion and genre, in image embeddings.

Our network, c*, learns from a teacher: the Qwen image embedding given an instruction for one aspect.
Qwen is the Qwen3-VL-Embedding-2B encoder.
At test, c* reads only the plain image: the encoder's image embedding without an instruction.
c* was trained on ArtELingo, a set of WikiArt paintings with emotion labels and captions.
WikiArt is a large online collection of paintings.

Earlier rounds on ArtELingo showed that c* beats the plain image and PercepT.
The previous round repeated the recipe on three more encoders: SigLIP 2, SigLIP 1 and CLIP L/14.
There, most comparisons with the teacher were wins, and most of the others were level.
Those three encoders had passed an earlier held emotion test on ArtELingo.

This round asks whether frozen c* still wins on paintings it never saw, with labels from another dataset.
MMArt supplies WikiArt paintings outside ArtELingo, with its own emotion, style and genre labels.
The paper's abstract is due on 2026-11-10 and the paper on 2026-11-16.

## The gains carry to MMArt; against the teacher, emotion retrieval falls short

**Problem.** Results on ArtELingo alone could not show whether the gains carry to new paintings and to another
dataset's labels.

**Idea.** We tested the saved networks on MMArt without more training.
First we removed every MMArt painting that is an exact or near copy of an ArtELingo training or validation painting.
No test painting shaped a model or a setting.

A read-out is how stored embeddings are compared for retrieval, for example with centering or query expansion.
We picked the read-outs on ArtELingo validation paintings with MMArt labels, and froze them before the test.
Each encoder has five seeds: five training runs with different random starts.

We compared c* with three comparators for each aspect:

- the plain image, in clustering and in retrieval;
- PercepT on the same encoder, in clustering, with its saved topic model applied unchanged;
- the teacher, in clustering and in retrieval.

Clustering is scored with AMI over 20 clusters: the agreement between clusters and labels, corrected for chance.
Retrieval is scored with mAP: how high the paintings with the same label rank.
Higher is better for both.
A painting without a label for an aspect is left out of that aspect's score (16,218 have an emotion label, 13,420 a
genre label).

The decision rule was fixed before the test.
It reads each difference, c* minus the comparator, through its 95% interval from a bootstrap over paintings and seeds:

- **WIN:** the whole interval is above zero.
- **LEVEL:** not a win, and the lower end is less than one seed range below zero.
- **BELOW:** the lower end is further below zero.

The seed range is the highest minus the lowest validation score over the five seeds.
A WIN against the plain image or PercepT licenses "better than".
Against the teacher, only 15 comparisons chosen in advance can license "better than".
Any other teacher WIN, and every LEVEL, licenses "level with or better than"; a BELOW licenses nothing.

The verdict "holds" means every required comparison with the plain image and PercepT is a WIN.
"GO" means the claim may be used in the paper.
A separate program, written from the rule and not from our code, recomputed every number from the saved inputs.
It matched all 6,697 values exactly.

**Did it work.** Yes: the verdict is "holds, GO" for each of the four encoders.
c* beats the plain image in clustering and in retrieval for every aspect.
It beats PercepT in clustering for every aspect.
All 15 teacher comparisons chosen for "better than" are wins, but some are narrow: the closest lower end is +0.0004.

The teacher comparisons have these exceptions:

- **BELOW** (no claim): emotion retrieval for Qwen and for CLIP L/14.
- **LEVEL** ("level with or better than"): Qwen style retrieval, SigLIP 2 and SigLIP 1 emotion retrieval, and CLIP
  L/14 genre clustering.

The test covers new paintings and MMArt labels, all from WikiArt.
It says nothing about other art sources or about photos.
Only 15 test paintings have an artist with no ArtELingo training painting.
That is fewer than the 20 clusters, so this subset was not scored, and the round makes no claim about new artists.

**Extra analyses.** These decide nothing, and the readings are ours.

- **Shared paintings.** 11,528 paintings are in both MMArt and ArtELingo's held set, so they have both label sets.
  For Qwen, in clustering against the plain image, the emotion gain with MMArt labels is about half the gain with
  ArtELingo labels. The style gain is the same. Our reading: the smaller emotion gain on MMArt comes from the new
  labels, not from the new paintings. The two label sets cover slightly different paintings.
- **Paintings far from ArtELingo** (a stricter copy cut, 10,403 paintings). The gains over the plain image remain.
  Qwen's emotion and style retrieval are below the teacher there; on the main test the style comparison is LEVEL.
  This subset supports no claim.
- **Seed spread.** Our reading: the validation seed range is a good guide to the test spread in retrieval and a
  loose one in clustering.

**Key evidence.** Each entry is c* minus the comparator for emotion, the aspect with the lowest scores.
Emotion AMI is low for every model (about 0.06 for c*), so these differences are small in absolute terms.
For example, Qwen's c* has an emotion AMI of 0.0609 against 0.0407 for the plain image.

| Comparison (emotion) | Qwen | SigLIP 2 | SigLIP 1 | CLIP L/14 |
|---|---:|---:|---:|---:|
| AMI against the plain image | +0.0201 | +0.0276 | +0.0249 | +0.0204 |
| mAP against the plain image | +0.0075 | +0.0120 | +0.0108 | +0.0098 |
| AMI against PercepT | +0.0094 | +0.0158 | +0.0152 | +0.0136 |
| AMI against the teacher | +0.0043 | +0.0043 | +0.0054 | +0.0024 |
| mAP against the teacher | −0.0010 (BELOW) | −0.0005 (LEVEL) | −0.0000 (LEVEL) | −0.0011 (BELOW) |

Every entry without a label is a WIN.

> [Full report: Results](../reports/auto/percept/2027-01-03_mmart_test.md#results): every aspect, the intervals and
> the claim each result licenses.

## Our advice

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** Freeze these results and write the paper.
The paper gains a second painting set, tested on four encoders.
The separate recomputation matched every number, so we trust the verdict.
In the paper, state the two emotion-retrieval exceptions against the teacher and the limit on new artists.

</details>

## Next

The test is complete, and nothing still runs.
Next is the planning session of the next loop.
There you give your reading of these results first, and then you pick the direction.
The full report gives these options:

1. Freeze and write. This needs no compute and frees the week before the abstract.
2. Test another aspect on photos, such as style or mood, for example with Flickr Style, a photo set with style labels.
   This needs about one loop, a new data path and a check of data access. It can wait until after the abstract.
3. Test paintings by artists outside ArtELingo. A data search must come first; it has not started.

What to delete from the disk is decided with you in that session.

## Glossary

| Term | Meaning (name in the full report) |
|---|---|
| Aspect | A way to group paintings: emotion, style or genre. |
| c* | Our network trained on ArtELingo paintings; at test it reads only the plain image (c*). |
| Plain image | The encoder's image embedding without an aspect instruction (plain image, frozen image). |
| Teacher | The Qwen image embedding given an instruction for one aspect (teacher). |
| PercepT | A published method that finds perceptual topics in image embeddings; here applied to the same encoder (PercepT's matched arm). |
| AMI | Agreement between clusters and labels, corrected for chance; higher is better (AMI). |
| mAP | How high paintings with the same label rank in retrieval; higher is better (mAP). |
| Read-out | How stored embeddings are compared for retrieval (read-out). |
| Seed range | Highest minus lowest validation score over the five seeds (seed range r). |
| WIN, LEVEL, BELOW | The three outcomes of a comparison under the decision rule (WIN, LEVEL, BELOW). |
| Holds, GO | Every required comparison is a WIN, and the claim may be used in the paper (holds, GO). |
| Near-copy cut | Removes test paintings too close to an ArtELingo training or validation painting (the cut). |
| MMArt, ArtELingo | Painting sets from WikiArt: MMArt gives the test paintings and labels; ArtELingo trained c* (MMArt, ArtELingo). |
