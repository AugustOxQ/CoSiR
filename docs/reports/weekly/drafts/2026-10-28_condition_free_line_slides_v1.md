# CoSiR weekly report, 28 October 2026

CoSiR: one read of a painting gives separate emotion, style and genre spaces

From the conditioned line to a condition-free model trained on instructed views

Notes: This deck summarises the weekly report docs/reports/weekly/drafts/2026-10-28_condition_free_line_v2.md (draft v2). Every number is taken from that report and its sources. Abbreviations are avoided in the text; some figures still use short names (for example "c*" for the sweep's pick and "G" for the previous recipe). AMI is adjusted mutual information between clusters and labels; 0 is chance.

---

## Problem and goal of CoSiR

- Image-text embedders such as CLIP and Qwen3-VL-Embedding place each painting at one point, with no choice of aspect.
- Their default clusters carry little emotion: the default Qwen image gives emotion AMI 0.069 (0 is chance).
- A curator may want to group one collection by mood, by style or by genre.
- Goal: one image pass gives three aspect spaces, each above PercepT and level with or above the instructed Qwen image.

Notes: An aspect is emotion, style or genre. AMI (adjusted mutual information) measures how well k-means clusters agree with the labels of one aspect; 0 is chance. Example: on development paintings the default Qwen image clustered into 20 groups agrees with the emotion labels at AMI 0.069 (held paintings: 0.0708), against style 0.222 for the same image. The closest prior method is PercepT ("Beyond Semantics", arXiv 2606.03345), which clusters image-caption pairs into perceptual topics. The instructed Qwen image is Qwen's image embedding made with one aspect's instruction; it needs one encoder pass per aspect. The second task of the plan, retrieval of paintings by the name of an aspect, is not measured yet.

---

## Approach of CoSiR in one picture

![Train on instructed views, read the default image once](../../assets/2026-10-28_weekly/slides/approach.png)

- At training, Qwen3-VL-Embedding embeds each painting once by default and once per aspect instruction.
- The instructed views link each painting to its nearest paintings in that aspect; a small network learns these links from the default image.
- At test time, one default image pass gives three outputs; k-means on each is scored against that aspect's labels.

Notes: Top row (training, training paintings only): one frozen embedder gives a default view and three instructed views (emotion, style, genre). Each instructed view builds one buddy graph: a painting's buddies are its mutual nearest neighbours in that aspect. The network reads only the default view and pulls buddies together in the output of each aspect; the instructed image of each aspect is also a teacher target. Bottom row (test): one default image pass, one network pass, three outputs, k-means with 20 clusters on each. Labels never enter training, graph building or early stopping; they only score the clusters. The held paintings are separate from every painting used to pick a recipe.

---

## Progress of CoSiR so far

![Project timeline with this report marked](../../assets/2026-10-28_weekly/slides/progress_timeline.png)

- Until mid-October the project followed a conditioned line: similarity under an aspect that example pairs showed.
- The supervisors stopped that line; a condition-free model, built as an improvement over PercepT, replaced it.
- This report covers the first two weeks of the new line, from the first checks to held tests and a recipe sweep.

Notes: Milestones: June to early September, the conditional buddy model; late September, the PercepT topic pipeline and the buddy model against PercepT on ArtELingo; end of September to mid-October, the conditioned line (CoSiR v2) up to one-sided affect steering; then the stop and the archive. In the new line: first checks of one Qwen instruction per aspect; the first shared model; its held emotion claim test (pass); a recipe without own-caption terms and four image encoders that pass held emotion; a relational genre term that brings held genre level with the instructed Qwen image; a recipe sweep and its held test and ablation.

---

## Experiment on one-sided affect steering

| Image encoder | Held margin over a condition-free scorer on its own groupings (R@1 points) |
|---|---|
| CLIP | +0.60 |
| Qwen3-VL-Embedding-2B | +0.63 |
| SigLIP 2 | +0.51 |

- The conditioned line scored image-caption pairs under an aspect that example pairs showed.
- Its best method, one-sided affect steering, steered toward emotion only when it was confident that the examples shared emotion.
- On held episodes it beat a condition-free scorer built on its own three groupings, on three image encoders.

Notes: R@1 is recall at rank 1 in an episode (one ranking test). Intervals: Qwen +0.63 [+0.49, +0.76]; SigLIP 2 +0.51 [+0.36, +0.66]; the CLIP result (round 6) has no interval. The groupings were label-free clusterings of the training rows; the emotion-like one came from an external emotion classifier run over the captions. The line is archived with its code, reports and round folders.

---

## Pitfall in one-sided affect steering

- The comparator of the table was a weaker bar: a scorer built on the method's own groupings.
- Against the strongest condition-free scorer, the margins ranged from −0.52 to +0.13 R@1 points over five reads.
- So the method never established a lead over a scorer that ignores the condition; it also lost on the style and genre pair.
- The supervisors judged the line too specific and too complicated for a CVPR paper, and stopped it.

Notes: The supervisors' decision, relayed by the user: "the line became too specific and too complicated to make a good CVPR paper; stop it and archive the CVPR plan". The user drafted the new direction the same day: a condition-free model built from the earlier buddy model (redesigned), as an improvement over PercepT, on PercepT's task plus retrieval, label-free if possible. Lesson carried over: every claim names its strongest real comparator.

---

## Plan for the CVPR submission

| Item | Kind | Status |
|---|---|---|
| Held emotion above PercepT by at least 0.0100, interval above 0 | required | met |
| One model for emotion, style and genre; level with the instructed Qwen image accepted | required | rule reports two wins and one level; the user's reading is open |
| The final recipe on SigLIP 2, SigLIP 1 and CLIP L/14 | required | not run |
| Retrieval of paintings by the name of an aspect | required | not started |

- Soft target: submit the CVPR abstract if a result holds; the abstract deadline is the hard limit.
- The aim of this loop: one recipe that is strongest on all three aspects.

Notes: The user set a soft CVPR target when the new line started: submit if a result holds by about the end of this week, else keep trying until the abstract deadline, about two weeks from now. The aim of the current loop was set by the user: one recipe strongest on all three aspects, with "level" against the instructed Qwen image accepted. Level means not more than one seed range below it. Reported but not required: other encoders with the earlier recipe, Leiden clustering, ArtELingo-28, a second dataset, PercepT with its own encoders. Paper options: one read for all aspects; instruction supervision across backbones; a fair topic evaluation of PercepT.

---

## Background on aspect-specific embeddings

- PercepT clusters image-caption pairs into perceptual topics, then trains a classifier to predict the topic from the image.
- It has one topic space, scored against emotion and genre labels with AMI.
- Prompted methods (DIOR, COCO-Facet, ESMC, guideline-driven clustering) make aspect-specific embeddings, but need the aspect at test time.
- Gap: no method gives separate aspect spaces from one read of the image.

Notes: PercepT's two stages: Stage 1 clusters fused image-caption pairs into topics; Stage 2 trains an image classifier that predicts the topic. Prompted methods need one pass per aspect at use. Distillation work (RKD, PKT, CompRess, SEED, CLIP-KD) trains a student toward a teacher's embedding or its similarity pattern; we reuse that idea for the teacher terms.

---

## Idea of one instruction per aspect

![An instruction raises own-aspect AMI](../../assets/2026-10-28_weekly/slides/instruction_effect.png)

- We replace Qwen's default prompt with one instruction per aspect, for example "Describe the style of this artwork in one word."
- The instruction raises own-aspect AMI where the aspect is visible: image style 0.395 against 0.222 for the default image.
- We train on the instructed views and test on the default view, so use needs no instruction and no caption.

Notes: Numbers from 99,999 captions and 5,000 development paintings. Caption emotion: 0.278 instructed against 0.122 default (a supervised GoEmotions classifier gives 0.209). Image style: 0.395 against 0.222 (CSD style encoder 0.317, frozen CLIP image 0.300). Image emotion: 0.093 against 0.069. A linear probe on image emotion moved only from 39.2% to 40.9% accuracy, so the instruction changed the cosine geometry more than the information. With 132 affect words masked, the instructed caption emotion AMI fell to 0.187, still above 0.069 for the default caption on the same masked captions. Emotion is visible mostly in captions, style in images, genre in both.

---

## Overview of the task

- Input: the default image embedding of one painting, read once; no caption and no instruction.
- Output: one vector per aspect (emotion, style, genre).
- Score: k-means with 20 clusters on each output; AMI of the clusters against that aspect's labels.

Notes: AMI is averaged over three clustering seeds and then over model seeds. Genre excludes unlabelled paintings. Leiden community detection, a graph method that picks its own cluster count, is a second clustering method we report beside k-means.

---

## Measures for the task

| Measure | Meaning |
|---|---|
| AMI | agreement between clusters and labels; 0 is chance |
| Seed range | spread of AMI over three seeds: emotion 0.0046, style 0.0082, genre 0.0128 |
| Win | the interval of the difference to the instructed Qwen image lies above 0 |
| Level | the lower bound lies above minus one seed range |
| Mean gain | the gain per aspect divided by its seed range, averaged over the three aspects |

- A claim test fixes its rule before the held read and uses five model seeds and paired 95% intervals.
- Separate code re-derives every verdict number.

Notes: The seed range is the maximum minus the minimum of the first shared model's development AMI over three seeds. Worked example: held emotion of the sweep's pick minus the instructed Qwen image is +0.0145; divided by 0.0046 this is +3.155 seed ranges. Intervals are paired 95% bootstrap intervals over paintings and seeds, written +0.0145 [+0.0113, +0.0177].

---

## Controls and comparators

![Development AMI before the first shared model](../../assets/2026-10-19_qwen_buddy_line/image_side_ami.png)

Development paintings, before the first shared model: our separate networks (blue) trailed the instructed Qwen image (green) on every aspect.

- **PercepT core**: our reimplementation of PercepT with 40 topics and five arms on our feature stores.
- **Instructed Qwen image**: the teacher and the hardest comparator; one encoder pass per aspect.
- **Frozen image**: the same encoder without training, clustered directly.

Notes: PercepT core fixes the topic count at 40 because the paper's own topic-count rule kept 1, 2, 3 or 99 topics depending on the seed. All five arms passed every reproducibility check (Stage 2 AUC 0.85 to 0.96, seed standard deviation at most 0.007). Its strongest held entries in the Qwen claim-test pool: emotion 0.0874 (Qwen arm, each painting's top topic), style 0.2297, genre 0.4269. A linear probe trained on labels gives a ceiling reference, not a rival: emotion 0.1215, style 0.5275, genre 0.6566.

---

## Dataset: ArtELingo

| Set | Paintings | Role |
|---|---|---|
| Train | 42,969 | training |
| Validation | 6,152 | label-free early stopping |
| Development | 29,232 (genre on 23,682) | recipe choice |
| Held | 12,281 (genre on 9,949) | claim tests only |

- WikiArt paintings with viewers' captions; each caption carries one of nine emotions.
- A painting's emotion is the mode of its caption labels; style and genre come from WikiArt.
- Splits group the captions by painting; a second dataset is planned but not run.

Notes: Early work used a smaller developed sample: 13,400 fit paintings for training and graphs, 5,000 for scores (genre on 4,053). ArtELingo-28 (746 paintings, translated captions) is a second test store. Artist overlap between splits was not measured.

---

## Novelty against prior work

| Neighbour | What it shares with us | What it lacks |
|---|---|---|
| DIOR, COCO-Facet, ESMC, guideline-driven clustering | aspect-specific embeddings | need the aspect at test time |
| ESMC, Multi-Sub | one teacher partition per criterion | no shared-trunk student read once from a default input |
| PercepT | topics predicted from the image | one topic space; emotion and genre only |
| RKD, PKT, CompRess, SEED, CLIP-KD | a student trained toward a teacher | no one-teacher student passing its teacher |

- To our knowledge, no published work combines instructed views, one buddy graph per aspect and a multi-head student that reads the image once.
- Not claimed as new: aspect prompting, distillation toward a teacher, the form of the relational genre term.

Notes: The literature note calls ESMC and Multi-Sub apparently transductive (trained on the items they are scored on). It found no held-out protocol for PercepT's Stage 1 scores against emotion and genre labels. It found no one-teacher student that beat its teacher on the teacher's task; students passed "only where a second source adds information". Our reading: captions may be such a source for emotion; not tested.

---

## Contributions, now and at full success

| Contribution | What we have now | What a full success would add |
|---|---|---|
| Method: one default-image read gives three aspect spaces | on Qwen, held emotion +0.0145 and style +0.0041 above the instructed Qwen image, genre level; emotion +0.0192 over PercepT | the final recipe on the other encoders that passed; retrieval |
| Protocol: a fair PercepT comparator and a fixed claim test | PercepT core with five arms; claim tests with a fixed rule, five seeds, paired intervals, separate re-derivation | a second dataset; PercepT with its own encoders |
| Analysis: which parts carry each aspect | two ablations; seven image encoders on development data | an analysis of what the heads learned |

Notes: The method row is the sweep's pick on 12,281 held paintings. "Level" means within one seed range below the instructed Qwen image (genre −0.0060, seed range 0.0128). The PercepT comparator is the strongest entry within the Qwen claim-test pool.

---

## Experiment on image-only students toward caption emotion

| Student | Image emotion AMI | Control | Required level |
|---|---|---|---|
| Toward instructed caption targets | 0.074 | 0.074 | 0.12 |
| On purified buddy pairs | 0.094 | 0.070 | 0.12 |

- Idea: instructed captions carry emotion (AMI 0.278); an image-only student might learn it.
- Setting: a student on the default Qwen image of 13,400 paintings, pulled toward the instructed caption or trained on caption-emotion buddy pairs.
- Both students missed the required level; purified pairs transferred some signal (+0.024 over control).

Notes: Purified pairs are caption-emotion buddy pairs with the pairs that the content graph also linked removed. Each control used default-caption targets or pairs in place of instructed ones. Scores on 5,000 development paintings, three seeds. The user read it as "there are aspects that we should not expect an image-only student model can learn". Follow-up: compare graph constructions instead of removing content (the aspect graph sheet).

---

## Experiment on PercepT with the paper's own topic-count settings

- Setting: an autoencoder with deep embedded clustering on fused image-caption pairs, then an image classifier for the topic.
- The paper's rule keeps the topics up to the largest gap between the sorted norms of the cluster centres.
- All 20 strict runs collapsed: they kept 1 to 8 or 98 to 99 topics, so the count followed the seed.
- With 40 topics fixed, held emotion was 0.0644 to 0.0711, below PercepT core's 0.0874.

Notes: A run counts as collapsed if it keeps under 10 or over 90 topics, or one topic holds over half the pairs; 10 strict runs at each training scale. The largest gap always sat at an end of the sorted list. Held emotion on 12,281 paintings, five seeds. The user read it as "paper settings PercepT seems worse than our PercepT core setting which fixed many things". A faithful version with the paper's encoders (CLIP L/14 and a GoEmotions-tuned ModernBERT) is not built.

---

## Method of the aspect-graph network: design

![Our model: one read of the default image gives emotion, style and genre spaces](../../assets/2026-10-28_recipe_sweep/model_structure.png)

- Example: a painting enters Qwen four times, by default and with the emotion, style and genre instructions.
- Each instructed view links it to at most 40 mutual nearest paintings in that aspect: its buddies.
- The network reads only the default image and pulls its three outputs toward its buddies' outputs.
- At test time one default image pass gives three outputs; k-means clusters each one.

Notes: Box 1: four views from one frozen embedder, Qwen3-VL-Embedding-2B. Box 2: one buddy graph per aspect; emotion and style subtract a content similarity. Box 3: one trunk and three heads, trained with buddy terms and the instructed image as teacher. Box 4: one read of the default image. Orange marks the part the sweep changed (40 neighbours in place of 20); teal marks the relational genre term. Nothing uses labels. New is the combination; each part is known. The network has 16,791,552 trainable parameters, against 7,220,152 for PercepT core's Qwen arm.

---

## Method of the aspect-graph network: components

- **Buddy graphs**: mutual top-40 neighbours per aspect; emotion and style subtract a weighted default-view (content) similarity.
- **Network**: image and caption sides; a shared trunk (2048 to 1024) and one head per aspect (1024 to 2048).
- **Buddy terms**: an InfoNCE loss pulls a painting's outputs toward its buddies' outputs, against the rest of the batch.
- **Teacher terms**: each image head is pulled toward its instructed image; the genre head also matches the teacher's similarity rows.

Notes: The content weight is set per painting for emotion and is 0.5 for style; genre uses the plain instructed image graph. Emotion graphs use instructed captions; style and genre graphs use instructed images. Batches hold 512 graph pairs (1,024 painting rows); buddy weight 1. Direct teacher: InfoNCE, temperature 0.05, weight 0.25. Relational genre term: KL divergence between the head's in-batch similarities and the teacher's row, sharpened by temperature 0.02, weight 8. Captions enter training only; the test-time read uses the image trunk and the three heads. Each aspect stops on its label-free validation loss.

---

## Setting of the recipe sweep

- The sweep changed the previous recipe's settings and picked one change by a rule fixed beforehand.
- It tested 13 settings one at a time on one seed, a joint grid of the best four, then three candidates on three more seeds.
- Pick rule: the largest mean gain, with every aspect at least level with the previous recipe.
- The pick: 40 graph neighbours in place of 20, mean gain +0.683 [+0.452, +0.959] seed ranges on development paintings.

Notes: The previous recipe is the first shared model without own-caption terms, plus the relational genre term with the sharper teacher row. At seed 0, 40 neighbours gave +0.924 seed ranges alone; a teacher-row temperature of 0.035 gave +0.371 alone, and both together +0.751, so combinations did not add. The confirmation used 29,232 development paintings, seeds 1 to 3: emotion +0.0026, style +0.0141, genre −0.0031 (interval across zero). The second-best candidate reached +0.369; the lead was within the intervals. The seed-0 search evaluated 57 configurations, which inflates the best seed-0 gains. Held data chose nothing in the sweep.

---

## Test of the aspect-graph network on held paintings

![Held results of the sweep's pick](../../assets/2026-10-28_recipe_sweep/held_results.png)

Left: held AMI per aspect for PercepT, the instructed Qwen image, the previous recipe and the sweep's pick. Right: the pick minus the instructed image.

- 12,281 held paintings (genre on 9,949), five seeds, paired 95% intervals; the rule was fixed before the read.
- Against the instructed Qwen image: emotion +0.0145, style +0.0041, genre −0.0060 (within one seed range).
- Against the strongest PercepT entry of the Qwen claim-test pool: emotion +0.0192, style +0.1769, genre +0.0732.
- Verdict (reference): under the fixed rule emotion and style are wins and genre is level; the user's reading is open.

Notes: Against the instructed Qwen image: emotion 0.1066 against 0.0921, +0.0145 [+0.0113, +0.0177] (+3.155 seed ranges); style 0.4066 against 0.4026, +0.0041 [+0.0004, +0.0081] (narrow: lower bound about 0.05 seed ranges above 0); genre 0.5001 against 0.5062, −0.0060 [−0.0103, −0.0015] (level bar −0.0128). Against PercepT: +0.0192 [+0.0137, +0.0207], +0.1769 [+0.1663, +0.1809], +0.0732 [+0.0622, +0.0796]. These PercepT entries are strongest only within the Qwen claim-test pool; in the encoder tests the CLIP L/14 matched arm reached style 0.2306 and the SigLIP 2 matched arm genre 0.4558. Separate code re-derived every gate and verdict number with 0 differences. Under Leiden (genre only) the pick reached 0.5082. Caveat: the previous recipe became the starting point after its held genre result.

---

## Test of the earlier recipe on seven image encoders

![Held emotion per image encoder](../../assets/2026-10-26_condition_free_line/held_emotion_by_encoder.png)

Left: held emotion AMI per image encoder against PercepT core's Qwen arm. Right: development lift over PercepT core on each encoder.

- Only the image input layer changes; graphs, teacher and captions stay Qwen's.
- Four encoders pass held emotion: SigLIP 1 0.1036, Qwen 0.1029, SigLIP 2 0.1024, CLIP L/14 0.0996.
- CLIP B/16 and CLIP B/32 are inconclusive; DINOv2-large fails at sample scale.
- This is the recipe before the genre term; the sweep's pick ran on Qwen only.

Notes: A pass needs a margin of at least 0.0100 over the strongest eligible PercepT entry (0.0874), interval above 0. Margins: SigLIP 1 +0.0162 [+0.0106, +0.0179]; Qwen +0.0154 [+0.0100, +0.0169]; SigLIP 2 +0.0150 [+0.0096, +0.0163]; CLIP L/14 +0.0122 [+0.0068, +0.0136]; CLIP B/16 +0.0070; CLIP B/32 +0.0011; DINOv2-large −0.0041 (sample scale, against 0.0855). On development paintings every student lifted over its own PercepT arm, +0.0117 to +0.0288; the lift correlated with style silhouette (Spearman +0.964) and not with frozen emotion AMI (+0.036). Under Leiden, the emotion intervals of every passing student included zero. The user: the failures "possibly means our design is not really a plug&play module that fits any model backbones".

---

## Ablation of the aspect-graph network

![Held change when each part is removed](../../assets/2026-10-28_weekly/slides/held_ablation.png)

- Each part was removed in turn and nothing was retuned; the figure shows the held paintings.
- Instructed graphs and buddy terms carry style and emotion: style −0.1903 and −0.1512 without them.
- The relational genre term carries genre: −0.0440 without it; no other removal changed genre by more than 0.0072.
- Removing the direct teacher or the caption endpoints raised held style (+0.0041, +0.0072).

Notes: Held, 12,281 paintings, seeds 0 to 4; intervals: instructed graphs style −0.1903 [−0.1959, −0.1822], emotion −0.0327 [−0.0361, −0.0292]; buddy terms style −0.1512 [−0.1547, −0.1451], emotion −0.0277; content correction style −0.0752, emotion −0.0090, genre +0.0072 [+0.0030, +0.0113]; relational term genre −0.0440 [−0.0512, −0.0372]; direct teacher emotion +0.0035 [+0.0008, +0.0056], style +0.0041 [+0.0016, +0.0067]; caption endpoints style +0.0072 [+0.0015, +0.0121]; sharper teacher row genre −0.0038 [−0.0074, −0.0002]; shared trunk no held change above 0.0023, every interval across zero (it saves parameters: 16,791,552 against 25,184,256). The development block (29,232 paintings, seeds 1 to 3) agrees in direction. Removal measures each part inside this recipe; it does not show the path that built the recipe.

---

## Analysis of where the gain sits

| Aspect | Over the instructed Qwen image | Over the strongest PercepT entry (Qwen pool) |
|---|---|---|
| Emotion | +0.0145 | +0.0192 |
| Style | +0.0041 | +0.1769 |
| Genre | −0.0060 | +0.0732 |

- Against the teacher the gain sits on emotion: about three seed ranges.
- Against PercepT it sits on style.
- Over the previous recipe the held gain was mostly emotion (+0.0027); the development style gain did not hold.
- Our reading: captions in the emotion graph may be a second source that lets the student pass its teacher; not tested.

Notes: Held, 12,281 paintings, five seeds. Over the previous recipe: emotion +0.0027 [+0.0001, +0.0054], style +0.0009 [−0.0044, +0.0065], genre 0.0000. Each label set favoured the head of its own name: against genre labels the emotion head reached 0.3889, against 0.5001 for the genre head. The clustering method changes the emotion lead: under Leiden the first shared model's emotion lead over PercepT core was −0.0005 [−0.0048, +0.0023], while k-means at Leiden's cluster count kept +0.0190. Style has no caption graph, and why style passes the teacher is not checked.

---

## Analysis of the scope of the result

- Covered: one model on held ArtELingo paintings, read once on the default Qwen image, scored with k-means.
- Not covered: the pick on other encoders, training without Qwen's instructed views, and Leiden clustering for emotion and style.
- Not covered: other data, retrieval, and PercepT with its own encoders.
- Cost at test time: one default pass (0.0871 s per painting) against three instructed passes (0.0936 s each).

Notes: Every student needs Qwen's graphs, teacher and captions at training. Under Leiden no emotion lead held for earlier models; for the pick Leiden ran on genre only. ArtELingo-28 was not read for the previous recipe or the pick. The two timing rates come from different runs and include image loading. Selection: 57 configurations at seed 0.

---

## Ongoing work and plan

- **Running**: nothing; the sweep and its held ablation are complete.
- **Open decision (the user's)**: the reading of the sweep, then the next work.
- **Options**: remove the caption endpoints; test the recipe on SigLIP 2, SigLIP 1 and CLIP L/14; draft the CVPR abstract.
- **Not started**: retrieval by aspect name; a second dataset; PercepT with its own encoders.

Notes: Option costs from the sweep report: removing the caption endpoints takes under one hour, and the held ablation already measured that arm, so a held test would be close to known; the encoder test takes about half a day of GPU on the cluster node, which is idle for about two more days. Go or no-go status: the held emotion requirement over PercepT is met; the one-model claim has two wins and one level under the rule, waiting for the user's reading; the final recipe on the other encoders and retrieval are not done. Paper options: one read for all aspects; instruction supervision across backbones; a fair topic evaluation of PercepT.
