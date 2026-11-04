# CoSiR weekly report, 4 November 2026

CoSiR: one read of an image gives separate emotion, style and genre spaces

Retrieval, more datasets, what is missing and what we fill

Notes: This deck summarises the weekly report docs/reports/weekly/2026-11-04_datasets_and_retrieval.md. Every number is taken from that report and its sources. It follows last week's deck and stands alone: the opener and the setting slides give what you need. Abbreviations are avoided in the text; some figures still use the short names.

---

## Problem and goal of CoSiR

- PercepT finds perceptual topics, but its emotion gain leans on an emotion-tuned caption encoder chosen in advance.
- Such an aspect-specific encoder is hard to carry to new datasets, and maybe to new backbones.
- We want a more general framework: one general embedder, Qwen3-VL-Embedding-2B ("Qwen"), told which aspect to look at.
- The bar is Qwen's image with an aspect instruction: our teacher, and the strongest label-free comparator we have.
- Goal: from one plain image pass, beat PercepT and match or beat the instructed Qwen image on emotion, style and genre.

Notes: An aspect is emotion, style or genre. PercepT is "Beyond Semantics" (arXiv 2606.03345). The plain image is an encoder's image embedding with no instruction. Qwen's default prompt is "Represent the user's input." The instructed views replace it. Emotion: "Represent the emotion that this artwork, or this description of an artwork, conveys. Focus only on the emotion." Style: "Describe the artistic style of this artwork in one word." Genre: "Describe the genre of this artwork in one word." The teacher needs one encoder pass per aspect, and the aspect must be named at test time. If one plain pass matches it, we get three aspect spaces for the price of one pass. On 12,281 held ArtELingo paintings, the teacher gives emotion AMI 0.0921 and the plain image 0.0708. AMI is adjusted mutual information between k-means clusters and labels; 0 is chance.

---

## Approach of CoSiR in one picture

![Train on instructed views, read the plain image once](../assets/2026-10-28_weekly/slides/approach.png)

In the figure, "default image" is the plain image.

- Training uses Qwen's instructed views next to the plain image embedding.
- Buddy graphs link paintings that rank each other among their closest neighbours; the network pulls buddies together.
- At test time, one image pass feeds three heads; k-means makes 20 clusters per head, or retrieval uses one named head.

Notes: Top: training paintings only; Qwen stays frozen. Each instructed Qwen image builds a buddy graph and is a teacher target for its head; the caption side adds training signal. A relational term on the genre head copies the teacher's similarity pattern. The model now uses 40 graph neighbours per item. Bottom: only the image side runs at test time. We call this network the painting model: a shared trunk with emotion, style and genre heads on a frozen image encoder. Training uses no labels. Labels score the results and, on validation data, also pick settings and read-outs, so the footer line of this figure is narrower than the current practice.

---

## Progress of CoSiR so far

![Timeline](../assets/2026-11-04_weekly/slides/progress_timeline.png)

- A week ago: one painting model, one held dataset (ArtELingo) and no retrieval result.
- This week: retrieval on paintings, a photo model, three more encoders, a second painting set and a coverage check.

Notes: Where last week ended, on 12,281 held ArtELingo paintings: emotion AMI 0.1066 against 0.0921 for the teacher (+0.0145 [+0.0113, +0.0177]) and 0.0874 for the strongest PercepT entry; style narrowly above the teacher; genre level. Retrieval: can the model search by a named aspect? Photo model: the painting model fine-tuned on unlabelled photos. Three more encoders: SigLIP 2, SigLIP 1 and CLIP L/14. Second painting set: MMArt. Coverage check: every method against every test set, then three rounds of work to fill the gaps. A round is one planned set of runs. The rounds have no results here.

---

## Setting of this week's tests: test sets and comparators

| Test set | Items | Labels | Role |
|---|---:|---|---|
| ArtELingo held | 12,281 paintings | emotion, style, genre | scored in earlier work |
| MMArt test | 16,384 paintings | MMArt's own | new paintings and labels |
| Affection test | 8,501 photos | emotion | scored in earlier work |
| EmoSet test | 17,348 photos | emotion | shaped no earlier choice |

- **Plain image:** an encoder's image embedding with no instruction.
- **Matched PercepT:** our re-implementation of PercepT on the same image encoder as the model.
- **Teacher:** the instructed Qwen image, with one encoder pass per aspect.
- **CRL:** an outside method that projects frozen features onto texts written for a criterion; our re-implementation.

Notes: ArtELingo has genre labels on 9,949 of the held paintings. Training used 42,969 ArtELingo paintings (validation: 6,152); the photo model also used 69,927 Affection photos. ArtELingo and MMArt are WikiArt paintings. A fifth test set, SemArt, holds 11,159 paintings from the Web Gallery of Art with genre labels; it is running and has no results here. A held set is a test split kept out of training. Validation data pick settings and read-outs. Comparisons used five model seeds and 95% bootstrap intervals. CRL is from NeurIPS 2025.

---

## Setting of this week's tests: outcomes

| Outcome | Meaning |
|---|---|
| Win | the whole 95% interval of the difference is above 0 |
| Level | the lower end of the interval is above minus one seed range |
| Below | any other case |

- Seed range: a model's validation spread over its seeds, fixed before the test.
- A claim test fixes its bar before the read; a descriptive result decides nothing.
- Clustering score: AMI, the agreement between clusters and labels, corrected for chance; 0 is chance.

Notes: A difference is the model minus its comparator, and outcomes use unrounded values. A seed range is the maximum minus the minimum. Qwen ArtELingo clustering uses last week's fixed ranges, from three development seeds: 0.0046, 0.0082 and 0.0128 for emotion, style and genre. Example: emotion retrieval against the teacher gave +0.0007 [−0.0002, +0.0016], which is level. Level supports no claim of superiority. A test that failed stays failed in the record.

---

## Setting of retrieval

- **Task:** find items with the same label as the query, for one named aspect.
- **Query:** one image plus an aspect name. The name only picks the head; the model gets no instruction text.
- **Gallery:** the other labelled paintings of the test set.
- **Score:** mAP (mean average precision): how high the items with the same label rank, averaged over queries.

Notes: We call this told retrieval. All retrieval here used image queries. A hit has the same label. AP (average precision) is the same score for one query or one class. Retrieval intervals are a joint bootstrap over queries and seeds with a fixed gallery, so they do not cover a new gallery. Emotion's chance mAP on ArtELingo, from the class shares, is 0.2007, and the majority labels from captions are noisy.

---

## Experiments on retrieval: plain cosine

| Aspect | Painting model | Plain image | Teacher | Outcome against teacher |
|---|---:|---:|---:|---|
| Emotion | 0.2642 | 0.2424 | 0.2665 | below |
| Style | 0.3110 | 0.1801 | 0.3091 | win |
| Genre | 0.5896 | 0.4760 | 0.5860 | win |

- **Purpose:** test whether the painting model also retrieves by a named aspect.
- **Design:** 12,281 held ArtELingo paintings, Qwen image queries, plain cosine (no change to the saved embeddings); the bar was level or win against the teacher.
- **Result:** the model beat the plain image on every aspect, but emotion fell below the teacher, so this first form failed.

Notes: Model minus teacher: emotion −0.0023 [−0.0032, −0.0014], style +0.0019 [+0.0008, +0.0030], genre +0.0037 [+0.0024, +0.0051]. Genre labels cover 9,949 paintings. One class carried the emotion miss: contentment, with an AP difference of −0.0324 across 4,244 of the 12,281 queries. The other eight classes were above the teacher, by +0.0027 to +0.0340 AP. These are diagnostics, with no per-class claims. mAP weights each query equally, so the largest class weighs most. Graded emotion (nDCG, a graded ranking score on the overlap of caption-emotion distributions) gave 0.9343 for the model against 0.9341 for the teacher, +0.0002 [+0.0001, +0.0004], and 0.9292 for the plain image. One unchecked explanation: training did not directly optimise retrieval of each painting's majority label.

---

## Experiments on retrieval: read-out picked on validation

![The painting model minus the teacher, with plain cosine and with the picked read-out](../assets/2026-11-04_weekly/slides/teacher_gap_slide.png)
<!-- NEW SLIDE FIGURE 1 of 2: teacher_gap_slide.png. A relabelled copy of ../assets/2026-11-04_datasets_and_retrieval/teacher_gap.png, needed because that figure prints the internal code c* in its title, axis and footer. To go back to the report figure, swap the path.
Data: same as the report figure. Script: docs/user_read/assets/2026-10-29_retrieval/make_teacher_gap.py, which reads the verdict.json files of each stage. Points and intervals agree with Table 3 (plain cosine) and Table 4 (picked read-out) of the report. The grey level zones are the seed ranges in those verdict files.
Layout: unchanged (three rows, open marker plain cosine, filled marker picked read-out, outcome word at each interval). Larger fonts for a slide. Size 16:9 or close to it.
Text: title "With the picked read-out, emotion moves from below to level; style moves from win to level". Axis "Painting model minus the teacher (mAP × 0.001), 95% interval". Footer "12,281 held ArtELingo paintings (genre: 9,949), five model seeds." Aspect names capitalised. -->

- **Purpose:** check whether the way saved embeddings are compared, with no training, can close the emotion miss.
- **Design:** each model picked one read-out for all aspects on 6,152 validation paintings: centering, whitening or neighbour query expansion.
- **Result:** model against teacher: emotion 0.2691 against 0.2684 (level), style 0.3235 against 0.3217 (level), genre 0.6151 against 0.6124 (win).
- **Result:** the plain image stayed lower: emotion 0.2483, style 0.1924, genre 0.5361.

Notes: A read-out is how saved embeddings are compared at retrieval time; plain cosine is the comparison with no change. Up to 25 read-out options. Model minus teacher: emotion +0.0007 [−0.0002, +0.0016], style +0.0017 [−0.0001, +0.0036], genre +0.0027 [+0.0013, +0.0042]. Style moved the other way, from win to level. The painting model, the plain image and the teacher all picked centering plus ten-neighbour expansion. No whitening option passed its validation check. Every pick used the largest neighbour count offered, so the options may not exhaust read-out gains. Selection reused data: validation chose the painting model's epochs within development data that had already picked the model among about 60 configurations. Verdict (reference): The retrieval claim on paintings rests on picked read-outs; for emotion it is level with the teacher. The large margins are over the plain image; against the teacher, the model matches.

---

## Experiments on retrieval: each head and CRL

| Label queried | Correct head | First wrong head | Second wrong head |
|---|---:|---|---|
| Emotion | 0.2642 | style head, 0.2487 | genre head, 0.2539 |
| Style | 0.3110 | emotion head, 0.1529 | genre head, 0.2167 |
| Genre | 0.5896 | emotion head, 0.4520 | style head, 0.3593 |

- **Purpose:** check that each head serves its own aspect, and compare with an outside method.
- **Design:** query each label set through each head with plain cosine; compare the picked read-out with the CRL version picked on validation.
- **Result:** each head beat both wrong heads on its own labels, with all intervals above 0.
- **Result:** the model won against CRL: emotion 0.2691 against 0.2554, style 0.3235 against 0.2816, genre 0.6151 against 0.5033.

Notes: The teacher and CRL showed the same head pattern as point values, so the pattern is not unique to our model. Our model gives it from one read of the plain image. The picked CRL version is CLIP L/14 with criterion texts worded for paintings. Differences: emotion +0.0137 [+0.0124, +0.0149], style +0.0418 [+0.0392, +0.0444], genre +0.1118 [+0.1087, +0.1149]. The CRL bars came after we had seen CRL's descriptive held scores, so this evidence is weaker than a comparison fixed earlier. The two arms use different encoders. On CLIP L/14 for both, emotion mAP was 0.2671 for our model against 0.2554, a point comparison only. An unpicked Qwen version of CRL exceeded the picked one on held emotion; picks use validation only, so the pick stayed.

---

## Experiments on photos: frozen painting model

| Task | Painting model | Plain image | Teacher | Outcome against teacher |
|---|---:|---:|---:|---|
| AMI | 0.1384 | 0.1041 | 0.1666 | below |
| mAP | 0.2755 | 0.2452 | 0.2945 | below |

- **Purpose:** test whether heads trained on paintings transfer to photos with no photo training.
- **Design:** plain Qwen image of 8,501 Affection test photos, frozen heads; bar: win over the plain image, level or win against the teacher.
- **Result:** the model beat the plain image and fell below the teacher in both tasks, so the test failed.
- **Result:** against CRL it was level (0.2755 against 0.2739), which failed its win bar.

Notes: Model minus plain: AMI +0.0343 [+0.0298, +0.0387], mAP +0.0304 [+0.0293, +0.0315]. Model minus teacher: AMI −0.0282 [−0.0333, −0.0233], mAP −0.0190 [−0.0205, −0.0176]. CRL difference: +0.0016 [−0.0001, +0.0032]. Labels are plurality votes; 1,917 of the test photos (22.6%) had tied votes, broken alphabetically. Read-outs and wording were picked on 4,251 validation photos. The order of model, plain image and teacher stayed the same with ties dropped, with plain cosine, with graded nDCG and in each image collection. Contentment, excitement and sadness carried most of the deficit. Painting-only training is a possible cause; this test did not check it. We chose to try photo training next. The failed test stays failed.

---

## Method of the photo model: design

![The photo model: the emotion pathway of the painting model, fine-tuned on unlabelled photos](../assets/2026-11-04_datasets_and_retrieval/photo_model_structure.png)

- Start from the painting model and keep the image encoder frozen; train only the emotion pathway.
- Train on 69,927 unlabelled Affection photos and their free-text explanations.
- Graphs come from the explanations, with content similarity subtracted; validation chose 80 graph neighbours against 40.
- At test time, one plain image gives one emotion output, with no instruction.

Notes: Orange marks what changed and teal what is new. The emotion pathway is the image-side and caption-side trunks and the emotion head. The explanations take the place of captions. The buddy terms and the teacher are of the same kinds as in the painting recipe, with the same teacher weight and learning rate. Train labels were never read. We removed 2,328 of 72,255 train photos as near-copies of validation or test photos (plain-image cosine at least 0.85). The photo model and the painting model are separate models.

---

## Experiments on photos: fine-tune on Affection

![The gain of each model over the plain image, as a share of the teacher's gain](../assets/2026-11-04_weekly/slides/teacher_share_slide.png)
<!-- NEW SLIDE FIGURE 2 of 2: teacher_share_slide.png. A reduced copy of ../assets/2026-11-04_datasets_and_retrieval/teacher_share.png, needed because that figure prints the internal code c* in its legend, carries a painting row with no value in the report, and says "held photos" for a test set that shaped earlier work. To go back to the report figure, swap the path.
Data: Figure 4 caption of the report. Frozen painting model: AMI 54.9%, mAP 61.5%. Photo model: AMI 127.5%, mAP 145.9%. Teacher 100%, plain Qwen image 0%. Tables 7 and 8 give the scores behind them. Script: docs/user_read/assets/2026-11-01_photo_finetune/make_teacher_share.py, which reads the verdict.json files of the Affection and photo fine-tune rounds.
Layout: two rows only, Affection AMI and Affection mAP. Drop the row "Paintings (ArtELingo), retrieval mAP". Same marker colours (plain grey, frozen open blue, photo model dark blue, teacher green). Direct labels beside the points. Larger fonts for a slide.
Text: title "On Affection, photo training moved the model from under to over the teacher's gain". Axis "Share of the teacher's gain over the plain image (%)". Legend "plain Qwen image", "frozen painting model", "photo model", "teacher". Footer "Point values. 8,501 Affection test photos, emotion, five model seeds." -->

A point at 100% has the same gain over the plain Qwen image as the teacher.

- **Purpose:** test whether photo training closes the gap to the teacher on Affection.
- **Design:** frozen picks from nine validation rows, then the 8,501 test photos; the bar was level or win against the teacher.
- **Result:** AMI 0.1838 against 0.1666 for the teacher; mAP 0.3172 against 0.2945.
- **Limit:** we trained after the failure on these same test photos, so this is no fresh confirmation.

Notes: Photo minus teacher: AMI +0.0172 [+0.0121, +0.0225], mAP +0.0227 [+0.0210, +0.0244]. Photo minus plain: AMI +0.0797 [+0.0741, +0.0857], mAP +0.0720 [+0.0699, +0.0742]. Against CRL (0.2739), the mAP difference was +0.0433 [+0.0411, +0.0455], a win. The order held with ties dropped, with plain cosine and in each image collection. The bar supports only the sentence "level with or better than the teacher"; a "better than" claim needed a new test with a win bar. The result gives no claim of transfer without photo training, and no claim of one model for both domains. The exact pick is close to noise: its lead over the second-best row was 0.0032 AMI, under half of the seed range (0.00707). All five final runs stopped at epoch 1. Verdict (reference): The Affection win is weaker evidence, because it follows a failure on the same photos.

---

## Experiments on photos: EmoSet

| Task | Photo model | Plain image | Teacher | Outcome against teacher |
|---|---:|---:|---:|---|
| AMI | 0.4571 | 0.3918 | 0.4390 | win |
| mAP | 0.4838 | 0.3864 | 0.4519 | win |

- **Purpose:** test the photo model, with a win bar set in advance, on a test set that shaped no earlier choice.
- **Design:** frozen photo model on 17,348 EmoSet test photos; only wording, seed ranges and read-outs came from EmoSet validation photos.
- **Result:** the model beat the plain image and the teacher in both tasks: +0.0181 AMI and +0.0320 mAP above the teacher.
- **Scope:** Qwen and emotion only.

Notes: No model trained on EmoSet. Photo minus teacher: AMI +0.0181 [+0.0121, +0.0242], mAP +0.0320 [+0.0295, +0.0347]. Photo minus plain: AMI +0.0653 [+0.0589, +0.0719], mAP +0.0975 [+0.0948, +0.1004]. We removed 368 of 17,716 test photos as near-copies of Affection train photos. With them kept, the AMI gain over the plain image fell to +0.0390 [+0.0318, +0.0464], still with a positive lower bound. The gain was uneven: amusement AP was 0.3203 against the teacher's 0.3471, and awe was also below the teacher with plain cosine. On the other three encoders, EmoSet gave descriptive results only. Verdict (reference): EmoSet is the strongest photo evidence so far, since it shaped no earlier choice.

---

## Experiments on encoders: three more encoders

| Encoder | AMI gain over own plain image | mAP gain over own plain image | AMI gain over matched PercepT |
|---|---:|---:|---:|
| SigLIP 2 | +0.0477 | +0.0249 | +0.0253 |
| SigLIP 1 | +0.0399 | +0.0236 | +0.0250 |
| CLIP L/14 | +0.0377 | +0.0192 | +0.0243 |

- **Purpose:** test whether the recipe depends on the image encoder.
- **Design:** the same recipe on three more encoders; each entry is the smallest of three aspects on 12,281 held paintings.
- **Result:** all bars against the plain image and matched PercepT passed; against the teacher, 18 wins, 5 level and 1 below.
- **Limit:** we chose these encoders after an earlier test of ours on the same paintings; Qwen still supplies training.

Notes: The recipe is the fixed training procedure. These encoders replaced only the image input; Qwen's instructed graphs, captions and teacher still supplied training. We did not test training without Qwen. Emotion was the smallest aspect every time. Intervals: SigLIP 2 +0.0477 [+0.0439, +0.0514], +0.0249 [+0.0238, +0.0260], +0.0253 [+0.0198, +0.0267]. SigLIP 1 +0.0399 [+0.0363, +0.0434], +0.0236 [+0.0225, +0.0247], +0.0250 [+0.0195, +0.0264]. CLIP L/14 +0.0377 [+0.0343, +0.0411], +0.0192 [+0.0181, +0.0203], +0.0243 [+0.0192, +0.0252]. The 24 comparisons are six on ArtELingo and two on Affection per encoder. The one below: CLIP L/14 emotion retrieval, 0.2671 against 0.2684, −0.0013 [−0.0023, −0.0003]. Photo model on Affection, AMI against own plain image: Qwen 0.1838 against 0.1041, SigLIP 2 0.1924 against 0.1142, SigLIP 1 0.1928 against 0.1200, CLIP L/14 0.1941 against 0.1127. Verdict (reference): The recipe carried to three encoders of two families (SigLIP and CLIP), within the limit of the pick. Qwen remains a training dependency on every encoder.

---

## Experiments on new paintings: MMArt

| Encoder | Emotion AMI, model | Plain image | Matched PercepT |
|---|---:|---:|---:|
| Qwen | 0.0609 | 0.0407 | 0.0514 |
| SigLIP 2 | 0.0608 | 0.0332 | 0.0451 |
| SigLIP 1 | 0.0620 | 0.0370 | 0.0468 |
| CLIP L/14 | 0.0590 | 0.0386 | 0.0454 |

- **Purpose:** test the frozen painting model on new paintings. **Design:** 16,384 WikiArt paintings outside ArtELingo, with MMArt's own labels.
- **Result:** the model won all nine comparisons per encoder against the plain image and matched PercepT.
- **Result:** against the teacher, 18 wins, 4 level and 2 below; both below are emotion retrieval (Qwen, CLIP L/14).
- **Limit:** only 15 test paintings come from new artists, so we claim new paintings and new labels only.

Notes: No MMArt test result selected a model. Emotion gave the smallest gain over the plain image in every row. Nine comparisons: AMI and mAP against the plain image and AMI against matched PercepT, on each of the three aspects. Qwen emotion AMI gain over the plain image: +0.0201 [+0.0178, +0.0227]. Read-outs and seed ranges came from 5,769 ArtELingo validation paintings with MMArt labels. Below cells against the teacher, emotion mAP: Qwen −0.0010 [−0.0015, −0.0004], CLIP L/14 −0.0011 [−0.0015, −0.0006]. All 15 comparisons set in advance for a "better than" claim won. The test set kept paintings with plain-image cosine below 0.95 to every ArtELingo train and validation painting. On the farther subset at 0.85 (10,403 paintings) the gains over the plain image survived; Qwen style retrieval also fell below the teacher there, −0.0039 [−0.0058, −0.0020]. On 11,528 shared paintings, the Qwen emotion AMI gain was +0.0192 [+0.0166, +0.0220] under MMArt labels and +0.0364 [+0.0332, +0.0398] under ArtELingo labels; style did not change. Our reading: the changed labels explain much of the smaller emotion effect. The intervals are not corrected for the number of comparisons. Verdict (reference): The gains over the plain image and matched PercepT carried to new paintings and labels. Against the teacher the ArtELingo pattern repeated: wins in clustering, level or slightly below in emotion retrieval.

---

## Analysis of the results: against the teacher

![Every comparison with the teacher, as win, level or below](../assets/2026-11-04_weekly/teacher_cells.png)

Each cell is one comparison with the teacher. Left: ArtELingo paintings. Middle: MMArt paintings. Right: photos, emotion only.

- Emotion retrieval is the only column with no win among painting comparisons: of eight, five were level and three below.
- Those three fell short by at most 0.0013 mAP.
- The frozen painting model on Affection stays in the record as the failed test.
- All photo model comparisons won, on Affection and EmoSet.

Notes: The Qwen ArtELingo clustering comparisons used last week's seed ranges. In the share of the teacher's gain over the plain image that the model keeps, MMArt gives over 100% in every clustering comparison and 88.2% to 99.8% in emotion retrieval. Per class, contentment is the recurring weak spot in retrieval: it is the largest emotion class on ArtELingo and on Affection, so it weighs most in mAP. On EmoSet the exception was amusement. On Qwen, MMArt contentment AP was 0.2413 against the teacher's 0.2547. Why emotion is weaker in retrieval than in clustering is not checked. Our related-work note found no case of a small student on frozen features that exceeds the instruction-following embedder on the named aspect. Our result holds in clustering on most comparisons and in retrieval on photos and painting genre, and not on painting emotion. Verdict (reference): One plain-image read gave a gain over the plain image on every test set. The result is strongest on style, genre and photo emotion, and weakest on painting emotion retrieval against the teacher.

---

## Analysis of the results: parts and settings

![Held change when each part is removed](../assets/2026-10-28_weekly/slides/held_ablation.png)

Each panel shows the held AMI change for one aspect when one part is removed. In the figure, "the pick" is the painting model.

- **Purpose:** find which parts and settings carry the result.
- **Design:** remove one part at a time on held paintings; change one setting at a time on validation.
- **Result:** the graphs, buddy terms and content correction carry emotion and style; the relational term carries genre (−0.0440).
- **Result:** no setting row was better on all three aspects, and the settings stayed frozen.

Notes: The removals are last week's saved runs on Qwen, five seeds, nothing retuned. They are analysis only. In retrieval the relational term also mattered: removing it changed held genre mAP by −0.1175 [−0.1214, −0.1134]. Three parts did not help: removing the direct teacher raised held emotion (+0.0035 [+0.0008, +0.0056]) and style (+0.0041 [+0.0016, +0.0067]), removing the caption side raised style (+0.0072 [+0.0015, +0.0121]), and removing the shared trunk changed nothing we could tell from zero. "Caption endpoints" in the figure is the caption side of the buddy terms. The painting sheet had 39 validation rows over 13 settings, three seeds; it chose nothing. The photo sheet had 14 rows with three seeds. Turning off the content correction changed validation emotion AMI by −0.0320 [−0.0391, −0.0247]. A random start gave similar scores to the painting start on one seed (AMI 0.1801 against 0.1769), so on photos the painting start may add little; one seed cannot show it. Against the photo model, twenty fixed epochs changed validation AMI by +0.0067 [+0.0014, +0.0127]; the analysis named no better setting.

---

## Analysis of coverage: what we have and what is missing

![Methods against test sets: scored, running or none](../assets/2026-11-04_weekly/slides/coverage.png)

One row per method and task. Scored means a claim test or its comparator; it does not mean a win.

- **What we have:** our own comparisons are broad: plain image, matched PercepT and teacher across the test sets.
- **What is missing:** outside methods, standard clustering metrics, cluster counts, new artists and encoders past our own picks.
- **Not covered:** training without Qwen, one model for both domains, text queries and a second photo aspect.

Notes: "Score only" means a descriptive result that decides nothing. On photos, "our model" is the photo model; on paintings and SemArt it is the painting model. "Both tasks" means clustering and retrieval. "4 encoders" are Qwen, SigLIP 2, SigLIP 1 and CLIP L/14; "3 running" are DINOv2, CLIP B/32 and CLIP B/16; "7" is all seven. The failed frozen painting model test on Affection is not a row. Gaps with no scheduled work: a second instructed embedder (important; we deferred its priority) and criterion-driven clustering and a second photo aspect (cite and explain only). Selection limit: only EmoSet shaped no earlier choice. Every claim uses AMI at 20 clusters and mAP. All claims need Qwen at training.

---

## Plan: rounds in progress

| Round | Question | Status |
|---|---|---|
| New scores from saved outputs | Do standard metrics, cluster counts and the missing retrieval comparisons support the picture so far? | Claim scoring done; descriptive sweeps running; report pending |
| Outside methods | How does the model compare with text-conditioned outside methods and stronger clustering of plain images? | Build and checks passed; TPIPS features extracting |
| Past our own picks | Do results survive our encoder selection, transfer to new artists and support a cost claim? | Build finished; features extracting; training running on a compute-cluster node |

- These three rounds have no results here.

Notes: A round is one planned set of runs. Status is the one in the report. New scores from saved outputs: no training or GPU work; it compares the painting model with plain images, matched PercepT, the teacher and same-encoder CRL, and it tests untold retrieval (three heads joined into one vector, no aspect name). Standard metrics are normalised mutual information, adjusted Rand index and accuracy. Outside methods: CLAY is a training-free projection of a frozen image onto text values of an aspect. TPIPS is an embedder fine-tuned on human similarity votes, queried along a factor named in text. TURTLE clusters plain images and model heads without labels. TEMI follows as a separate build. Past our own picks: adds DINOv2, CLIP B/32 and CLIP B/16; SemArt genre covers seven encoders on artists whose names match no ArtELingo training artist; the cost question is one pass for the plain image against one pass per aspect for the teacher.

---

## Plan: later work and open decisions

- **Later work:** a qualitative cluster figure, head variants (deep embedded clustering loss, deeper heads) and text queries.
- **Open decision:** the priority of a second instructed embedder.
- **Open decision:** the paper's wording for emotion retrieval below the teacher, after the per-class look.
- **Open decision:** the wording of a paper note that the frozen encoder kept its own retrieval.

Notes: The qualitative figure follows the cluster counts. Go/no-go status of the required items: the final recipe on SigLIP 2, SigLIP 1 and CLIP L/14 is met, with one teacher cell below; aspect-named retrieval is met with picked read-outs and not met with plain cosine; photos at least level with the teacher failed for the frozen model and is met by the photo model; "better than the teacher" on data that shaped no earlier choice is met on EmoSet, emotion, Qwen. The paper options: one read for all aspects, instruction supervision across encoders, or a fair comparison table with a column that states whether a method trained on images from the test domain. PercepT with its own encoders is not built and has no scheduled work.
