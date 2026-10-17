# One instruction per aspect works for caption emotion and image style; an image-only emotion student does not

> 2026-10-17 (Amsterdam). Full reports:
> [the prompted-aspect check](../reports/auto/percept/2026-12-08_prompted_aspect_views_check.md) and
> [the word-masking check and the student pilot](../reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md).

## Summary

- We gave an instruction-following embedder one aspect at a time, for example "focus only on the emotion".
- On captions, the emotion instruction makes the clusters follow emotion, above a supervised emotion classifier.
- On images, the style instruction makes the clusters follow style, above a dedicated style encoder.
- Most of the caption gain stays when we mask a list of 132 affect words, so the emotion name is not the main source.
- Image emotion stays weak. A small image-only network that learned from the emotion captions did not improve it.
- So emotion comes from the captions and style from the images. The grill decides what this means for the plan.

## How we got here

The conditioned line (CoSiR v2) stopped on 2026-10-15. The new line builds a model without conditions. It starts
from the buddy model (our model that learns from pairs of mutual nearest neighbours). It must improve over PercepT,
a published method that finds perceptual topics such as emotion and genre in images. It must find aspects such as
emotion, genre and style, without labels if possible. Your central idea: prompt Qwen3-VL-Embedding with one instruction per aspect, get one embedding set per
aspect, and build buddy graphs on each.

On the night of 2026-10-17 we tested three premises of this idea before the grill (the question round in which you
make the key design decisions). All three ran on one sample of the ArtELingo training side (99,999 captions, 19,889
paintings), with Qwen3-VL-Embedding-2B. Labels were used to score, and to compare the instruction wordings. They did
not train anything.

## 1. Does an aspect instruction move the embedding toward that aspect?

**Problem.** Our earlier models found emotion only through an encoder trained on emotion labels. You named this
problem: "the emotion credits come from the emotion encoder". That is, the emotion result came from the supervised
encoder, not from our method.

**Idea.** Qwen3-VL-Embedding takes a system instruction before each input. We replaced the default instruction
with ten texts. Besides the default, each aspect (emotion, genre, style) got three wordings:
- "focus only on the emotion";
- "describe the emotion in one word";
- "the emotion, not the subject and not the style".

Images and captions got the same text. We then clustered each embedding set and measured how well the clusters
agree with the labels (AMI, below).

**Did it work?** Partly. Yes for caption emotion and for image style; only weakly for image emotion. The partitions of
different aspects differ from each other. For emotion, the "focus" and one-word wordings work best and negation
works least; for image style, the one-word wording is best. The gain is in the cluster geometry: linear probes
(classifiers on the embedding) change much less, by 1 to 5 points.

**Key evidence.** AMI at 20 clusters. A painting's emotion label is the majority emotion of its captions. The seed
spread is at most 0.03 for captions and 0.02 for paintings.

| What we measure | Default instruction | Aspect instruction | Strongest reference |
|---|---:|---:|---|
| Caption emotion | 0.122 | 0.278 | supervised emotion classifier 0.209 |
| Image style | 0.222 | 0.395 | dedicated style encoder 0.317 |
| Image emotion | 0.069 | 0.093 | CLIP 0.065 |

> Full report: [the prompted-aspect check](../reports/auto/percept/2026-12-08_prompted_aspect_views_check.md), "Results".

## 2. Is the caption gain only reading the emotion word?

**Problem.** Many captions name the emotion ("it makes me feel sad"). If the instruction only finds that word, the
gain is cheap, and a keyword list could do the same.

**Idea.** We replaced emotion words with a mask token and embedded the captions again. We used two word lists:
- the emotion names and their forms (44 words, in 13% of the captions);
- a broad list of 132 affect words, with "feel" and "love" (in 41% of the captions).

We also scored, without masks, only the captions that hold no word from the broad list.

**Did it work?** Mostly. About three quarters of the gain stays. One caveat: some affect words are not on the list
("dark", "death", "smile"). So the instruction can still read affect from other words. That is near what we want,
emotion read from the description, but it is not checked.

**Key evidence.** Caption emotion AMI at 20 clusters:

| Captions | Default instruction | Emotion instruction |
|---|---:|---:|
| All captions, nothing masked | 0.122 | 0.278 |
| Emotion names masked | 0.095 | 0.231 |
| All 132 affect words masked | 0.069 | 0.187 |
| Only captions without a word from the broad list (59,413) | 0.070 | 0.197 |

> Full report: [the word-masking check](../reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md), "Results", masking.

## 3. Can an image-only network learn the caption emotion geometry?

**Problem.** A reviewer can say that the method "just prompts Qwen". A student could answer this. It is a small
network that sees only the image and learns from the prompted captions. If it beat prompted Qwen on the image alone,
the method would add something.

The bar, set before the run: image emotion AMI of at least 0.12, and a gap over a control student larger than the
seed spread.

**Idea.** The student maps the Qwen image embedding to the teacher. The teacher is the average of the painting's
emotion-instructed caption embeddings. The control student learns the average of the default captions instead. So
the two differ only by the emotion instruction.

Both trained on 13,400 paintings, with three seeds, and we scored them on 5,000 other paintings. After the bar
failed, we added a check with no training: average the teacher of the 20 most similar training images.

**Did it work?** No. The student and the control both reach 0.074, so the emotion target gives no cluster gain (the
probe gain is 1 point). Our reading: the student learns what captions share with the image, which is mostly content. The neighbour check reaches 0.102 at best. That
is a small gain over the prompted image (0.093), far from the teacher (0.217) and below the bar. The student also
loses genre (0.449 to 0.424) and style (0.222 to 0.165). Caveat: we tried one student design only.

**Key evidence.** Image emotion AMI at 20 clusters, on the 5,000 paintings:

| View | Image emotion AMI |
|---|---:|
| Default image | 0.070 |
| Emotion-instructed image | 0.093 |
| Student (three seeds) | 0.074 |
| Control student (three seeds) | 0.074 |
| Neighbour average, emotion-instructed image | 0.102 |
| Teacher (from captions, not images) | 0.217 |

> Full report: [the student pilot](../reports/auto/percept/2026-12-09_image_transfer_and_word_masking.md), "Results", student, and "Analysis".

<details>
<summary>Our advice (open after giving your reading)</summary>

**Our view.** Keep emotion on the caption side, and let the images give style and content. The paper story could
read: "one instruction per aspect; each aspect comes from the side where it is visible". The student then must show its
worth on style or on retrieval, not on image emotion. The next pilot we suggest is a style student, about 1 hour. We
would not spend more runs on image-only emotion unless the 8B model raises the image emotion probe.

</details>

## Next

Decision A (the direction) is still a draft, and decision B (which idea) is open. The grill goes on with you:

1. Your reading of these results (decision C), before you open our advice.
2. The discussion of candidate (b), learned aspects (the model finds its own aspects, with no aspect names given),
   and then decision B.
3. What the student must show: image-only transfer (now weak for emotion), better topics than the teacher, a gain
   from a shared trunk, or a gain on style or retrieval.
4. Which aspects: content, affect and style; content and style; or content and affect.
5. What counts as "something works" by 2026-10-29.

## Glossary

- **AMI** (adjusted mutual information): agreement between a clustering and the labels; 0 is chance, 1 is the same partition.
- **Instruction**: the system text Qwen3-VL-Embedding reads before each input; an aspect instruction names one aspect.
  In the full reports: `default`, `emotion_focus`, `style_word`, and so on.
- **Buddy graph**: the graph of mutual nearest neighbours in one embedding set, the base of the buddy model.
- **Teacher**: the average of a painting's emotion-instructed caption embeddings.
- **Student**: a small image-only network trained toward the teacher. **Control student**: the same network trained
  toward default-instruction captions.
- **Neighbour average**: no training; the average teacher of the 20 most similar training images (in the full
  report, "neighbour transfer").
- **Probe**: a linear classifier on the embedding, to measure how much label information it holds.
