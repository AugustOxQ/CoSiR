# Stage report, 2026-10-15 to 2026-10-19: our condition-free network clearly led PercepT core and plain image clustering only on style; instructed Qwen led all aspects

> Covers 2026-10-15 to 2026-10-19. Previous stage report: [the archived conditioned line](../../../archive/v2_conditioned/docs/reports/stage/2026-10-12_affect_steering_line.md). Every number comes from the auto report or experiment log linked in its section. Decisions come from the direction log. Development data only. No held-out number exists for this line in this period.

## Summary

- We stopped the conditioned line and started a condition-free model for emotion, style and genre.
- Aspect instructions gave Qwen useful caption emotion and image style views. They supplied the premise for graphs that could train our network.
- Both image-only emotion transfer pilots missed their preset bar. Content removal also removed pairs that shared emotion.
- Residual graphs trained better emotion and style networks than plain aspect graphs. For genre, the image graph alone and the image-weighted combination were level.
- We established PercepT core with a fixed topic count, then kept plain image clustering as another comparator.
- Our network clearly led both comparators only on style. Instructed Qwen remained ahead of our network on every aspect, which set the next target.

## Where we started

We began with a simpler question: could one condition-free model represent several aspects of a painting? The supervisors judged the earlier conditioned line too specific and too complicated for a good paper, and it had no margin over a condition-free scorer. We stopped and archived it on 2026-10-15. The new direction kept the buddy idea and used PercepT as the comparison task. We aimed to learn aspect spaces and later use them for retrieval.

PercepT learned topics from image-caption pairs, then trained an image reader to predict those topics. Our literature note found that, in the paper's own tables, its clustering step (DEC) added little label agreement. Its image reader learned its own discovered topics, so topic prediction alone could not establish agreement with independent aspect labels. The note also identified its supervised emotion source and lack of retrieval evidence. These concerns made a direct image-encoder comparator necessary.

Our earlier PercepT code also differed from the paper. We therefore began without an established PercepT comparator for the new protocol. The audit showed that the earlier claim, that our buddy model matched PercepT, compared it with a PercepT-like baseline, not PercepT as published. The first task was to separate the effect of the input features from the effect of our own network.

The literature check also limited what we could call new. Aspect prompts already produced aspect-specific features in earlier work. Our reading: the possible contribution lay in graph supervision and a shared image model that preserved several aspects. The immediate attack was “just prompting Qwen.” We needed a measured comparison against an instructed image encoder.

All steps used ArtELingo, paintings with captions written by viewers; each painting has an emotion, a style and a genre label. Qwen meant Qwen3-VL-Embedding-2B, and CLIP meant CLIP ViT-B/32. We used AMI, adjusted mutual information between k-means clusters and labels; 0 meant chance. “Image side” meant clusters of image features only. Our network meant a small network trained on aspect graphs and read on the default image. Default input used no aspect instruction at test time. The aspects were named in advance, so this stage did not discover their names. Level means a difference below 0.01.

Sources: [direction log](../../superpowers/direction_log.md), [PercepT literature](../literature/2026-12-05_percept_neighbourhood.md), [prompted-view literature](../literature/2026-12-07_prompted_aspect_views.md), [code audit](../auto/percept/2026-12-04_percept_code_audit.md).

## Timeline

1. 2026-10-17 · Audited our PercepT code and checked the new line's premises (check).
2. 2026-10-17 · Tested one Qwen instruction per aspect (check).
3. 2026-10-17 · Trained an image-only student toward instructed caption targets (did not work).
4. 2026-10-17 · Masked emotion words in captions (check).
5. 2026-10-18 · Trained on purified buddy pairs (did not work).
6. 2026-10-18 · Compared aspect graphs and small networks (check).
7. 2026-10-18 · Repeated graph comparisons and tested instructed inputs (check).
8. 2026-10-19 · Traced the failed September reproduction check to k-means thread order (check).
9. 2026-10-19 · Ran PercepT core with its own topic-count rule (check).
10. 2026-10-19 · Established PercepT core with 40 topics and selected comparators per aspect (check).
11. 2026-10-19 · Measured the image-side ceiling with labelled linear probes (check).

## The steps, in date order

### 2026-10-17 · Code audit and premise checks (check)

The audit found that our earlier PercepT code averaged captions per painting and differed in fusion, encoders, topic count and image-reader targets. The premise checks also showed weak caption emotion in CLIP text, AMI 0.047, beside 0.209 from a supervised GoEmotions classifier. We retained PercepT as a baseline to repair and explored one space per aspect before learned aspects or a protocol-only paper.

Source: [code audit](../auto/percept/2026-12-04_percept_code_audit.md), [new-line candidates and premise checks](../auto/percept/2026-12-06_new_line_candidates.md).

### 2026-10-17 · One instruction per aspect in Qwen (check)

The instructed views supplied the premise of the line. AMI at k = 20, on 99,999 captions or 5,000 paintings:

- Caption emotion: 0.278 with the emotion instruction, beside 0.122 for default Qwen and 0.209 for the supervised GoEmotions classifier.
- Image style: 0.395, beside 0.222 for default Qwen and 0.317 for the dedicated style encoder CSD.
- Image emotion: 0.093, beside 0.069 for default Qwen (0.070 when later reports rescored it) and 0.065 for CLIP.

We kept the visible side of each aspect as its source, while the named-aspect design remained an exploration.

Source: [prompted aspect views](../auto/percept/2026-12-08_prompted_aspect_views_check.md).

### 2026-10-17 · Image-only student toward caption targets (did not work)

The student missed the preset image emotion AMI bar of 0.12. Image emotion AMI on 5,000 paintings, three seeds:

- Student: 0.074 ± 0.000. Control student, trained toward default captions: 0.074 ± 0.002.
- Instructed image alone: 0.093. Instructed caption teacher: 0.217.

Our reading: this student design transferred little caption emotion, so we moved to pair supervision.

Source: [image transfer and word masking](../auto/percept/2026-12-09_image_transfer_and_word_masking.md).

### 2026-10-17 · Emotion-word masking (check)

Most of the caption emotion lift survived affect-word masking. Caption emotion AMI on 99,999 captions, instructed beside default:

- Unmasked: 0.278 beside 0.122.
- Broad list of 132 affect words masked: 0.187 beside 0.069.
- Only captions without a listed affect word: 0.197 beside 0.070.

The list could miss affect-bearing words. We kept the caption view as a useful source without claiming that masking removed every emotion cue.

Source: [word-masking pilot](../auto/percept/2026-12-09_image_transfer_and_word_masking.md).

### 2026-10-18 · Purification pilot (did not work)

The purified buddy student passed the gap part of its preset bar and failed the level part (0.12). Image emotion AMI on 5,000 paintings, three seeds:

- Purified student: 0.094 ± 0.002. Control student, trained on default caption pairs: 0.070 ± 0.002.
- Purification added +0.004 over the unpurified emotion pairs (0.090).
- The removed shared pairs had emotion lift 2.55 beside 2.42 for the whole emotion graph.

Our reading: content and emotion overlapped in these paintings, so we compared graph constructions instead of assuming that content removal would help.

Source: [purified buddy pilot](../auto/percept/2026-12-10_purified_buddy_pilot.md).

### 2026-10-18 · Aspect graph sheet and network checks (check)

**Idea:** We treated graph construction as an open empirical question. The purification pilot showed that content overlap could carry useful emotion. We therefore compared several ways to keep, remove or combine pairs before building the shared model. This was exploration with no success bar.

**Design:** A buddy graph linked paintings that were among each other's 20 nearest neighbours. Each aspect graph used its instructed view. A residual graph ranked pairs by aspect similarity minus content similarity. It changed the neighbour ranking instead of removing every pair with similar content. We also compared graphs that combined image and caption similarities.

We rebuilt the pilot graphs and recovered their results exactly. Then we compared 21 graphs per aspect. Pair agreement measured how often graph neighbours shared a label, relative to the background rate. We trained small networks on selected graphs to test which pairs gave useful supervision.

Target AMI meant the mean of image-side and caption-side AMI. We used it to select networks in this sheet. Scores used k = 20 on 5,000 evaluation paintings. The plain aspect graph was the reference for emotion and style; the caption graph was the genre reference.

**Result:** Residual graphs trained the best emotion and style networks, although the plain aspect graphs had the highest pair agreement.

| Aspect | Graph for our network | Target AMI | Reference graph target AMI |
|---|---|---:|---:|
| Emotion | Residual | 0.137 ± 0.002 | Plain aspect, 0.120, one seed |
| Style | Residual | 0.260 ± 0.003 | Plain aspect, 0.218, one seed |
| Genre | Image-weighted combination | 0.420 ± 0.002 | Caption, 0.402 ± 0.005 |

The side-specific result set a limit. Style captions reached 0.172 beside instructed captions 0.135. Style images reached 0.348 beside instructed images 0.395. Thus the mean could hide a shortfall on the side required for image-only use.

**What it changed:** We separated graph pair agreement from downstream usefulness. Our reading: the cleanest pairs did not necessarily teach the best space. The project chose a follow-up to complete the reference seeds and test instructed inputs. PercepT core would follow, before the shared model. These were checks on selected development rows, so the sheet did not establish a general advantage.

Source: [aspect graph sheet](../auto/percept/2026-12-11_aspect_graph_sheet.md).

### 2026-10-18 · Graph follow-up with instructed inputs (check)

**Idea:** We checked whether the residual graph order survived more seeds. We also asked whether default input explained the gap to instructed Qwen. The earlier plain-graph references had incomplete seed coverage. The follow-up again had no success bar.

**Design:** We completed the missing reference seeds, trained a genre network on the image graph alone, and supplied instructed features to selected networks. The graph comparison retained target AMI on the same evaluation paintings. Instructed-input networks needed an aspect instruction at test time, so they answered a different input question from the default-input model.

**Result:** Residual graphs stayed ahead of plain aspect graphs across three seeds. For genre, the image graph alone and the image-weighted combination were level (a difference of 0.008).

| Aspect | Graph for default-input network | Target AMI | Comparator target AMI |
|---|---|---:|---:|
| Emotion | Residual | 0.137 ± 0.002 | Plain aspect, 0.118 ± 0.002 |
| Style | Residual | 0.260 ± 0.003 | Plain aspect, 0.213 ± 0.005 |
| Genre | Image alone | 0.428 ± 0.003 | Image-weighted combination, 0.420 ± 0.002 |

Instructed inputs raised the residual networks' target AMI. The side-specific comparison remained stricter, and no network passed the instructed view on its stronger side (three-seed means):

- Target AMI, default input then instructed input: emotion 0.137 then 0.151; style 0.260 then 0.284.
- Emotion captions: 0.216 beside the instructed view's 0.217.
- Style images: 0.393 beside the instructed view's 0.395.

**What it changed:** We carried residual emotion and style graphs forward, with the image graph alone for genre. The project requested an ablation table in the user-read report, with open rows for PercepT core to fill, to expose which component carried each result. Our reading: the instruction supplied much of the useful geometry. Graph supervision retained some of that geometry on default input, while the strongest instructed side remained the target.

The follow-up measured repeatability across seeds after row selection. It did not remove the effect of selecting graphs on the same paintings. It also left the instructed-input genre network on the image graph untested. Those limits mattered when we chose the scope of the next model.

Source: [aspect graph follow-up](../auto/percept/2026-12-12_aspect_graph_followup.md).

### 2026-10-19 · First PercepT core run (check)

The September reproduction check failed because k-means accumulated its initial centres in a thread order that changed across runs, so PercepT core waited. Meanwhile, our network on the default graph reached target style AMI 0.163 beside direct default clustering 0.171 (level). We changed the regression check to exact reusable parts and chose one-thread k-means for PercepT core.

Source: [first PercepT core run](../auto/percept/2026-12-13_percept_core.md).

### 2026-10-19 · PercepT core with its own topic-count rule (check)

The amended checks passed, yet the topic-count rule kept 1, 2, 3 or 99 topics according to seed. Only the CLIP image-vector setting passed all checks, with image-side style AMI 0.162 beside our network's 0.348. The largest norm gap lay at an end of the topic list, so we fixed the count before accepting a comparator.

Source: [PercepT core arms](../auto/percept/2026-12-14_percept_core_arms.md).

### 2026-10-19 · PercepT core with 40 topics (check)

**Idea:** We needed a comparator whose topic count did not collapse on some seeds. We fixed K = 40 and retained the trained centres with the most fit pairs. The original topic-count rule remained a reported diagnostic. This established a named variant of PercepT core for our protocol.

**Design:** PercepT core included the audit's four fixes: image-caption pairs, the paper's fusion formula, its topic-count rule (here replaced by the fixed count), and multi-hot caption-union targets. The image reader learned every topic assigned to a painting's captions. We used our features, so the comparison did not reproduce PercepT as published. The fixed count was also our choice.

The settings separated feature sources and image readers. We scored each reader's image outputs under the same image-side protocol. The table gives mean AMI across three seeds.

| Setting | Topic features | Image reader | Emotion | Style | Genre |
|---|---|---|---:|---:|---:|
| P1 | CLIP and RoBERTa emotion captions | CLIP patches | 0.053 | 0.167 | 0.356 |
| P2 | CLIP and RoBERTa emotion captions | CLIP image vector | 0.066 | 0.229 | 0.400 |
| P3 | Qwen and instructed emotion captions | Qwen image vector | 0.077 | 0.189 | 0.436 |
| P4 | Qwen and instructed emotion captions | CLIP patches | 0.054 | 0.162 | 0.361 |
| P5 | Qwen and RoBERTa emotion captions | Qwen image vector | 0.074 | 0.181 | 0.435 |

**Result:** Every setting passed every check. AUC on the development split ranged from 0.85 to 0.96 beside the required 0.75. Seed standard deviation was at most 0.007, beside the earlier maximum 0.143. The centre norms and topic counts matched the previous run bit for bit. This separated the topic-count change from the variation of a new training run.

The best setting per aspect was level with or below the image encoder it reads. It was lower on style (0.229 beside 0.300 for plain CLIP images) and on genre (0.436 beside 0.449 for plain Qwen images). The table under "Where we stand" gives all values.

**What it changed:** We accepted PercepT core with the fixed count as a comparator per aspect. We also retained the strongest plain image encoder per aspect. Our reading: PercepT core alone set a weak bar for style and genre. Neither the fixed count nor these development comparisons established the published method's best possible performance. Other topic counts remained unchecked.

Source: [PercepT core with 40 topics](../auto/percept/2026-12-15_percept_core_k40.md).

### 2026-10-19 · Image-side ceiling check (check)

A linear probe trained with labels gave a ceiling reference on default Qwen image features, not a method result. Image-side AMI, k = 20, 5,000 paintings:

| Method | Emotion | Style | Genre |
|---|---:|---:|---:|
| Linear probe trained with labels | 0.121 | 0.527 | 0.657 |
| Direct clustering | 0.070 | 0.222 | 0.449 |

Our reading: the features contained more aspect information than our unsupervised geometry exposed, although this probe was no strict upper bound.

Source: [image-side ceiling log](../../../src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md).

## All attempts at a glance

| Date | Step | Result against its bar or comparator | Outcome |
|---|---|---|---|
| 2026-10-17 | Audit and premise checks | Earlier code differed from PercepT; comparator required repair | check |
| 2026-10-17 | Aspect instructions | Caption emotion 0.278 beside default 0.122 | check |
| 2026-10-17 | Own-caption student | Image emotion 0.074 beside control 0.074; bar 0.12 | did not work |
| 2026-10-17 | Word masking | Masked caption emotion 0.187 beside default 0.069 | check |
| 2026-10-18 | Purified buddy student | Image emotion 0.094 beside control 0.070; bar 0.12 | did not work |
| 2026-10-18 | Graph sheet | Style target AMI 0.260 beside plain graph 0.218; no bar | check |
| 2026-10-18 | Graph follow-up | Style target AMI 0.260 beside plain graph 0.213; no bar | check |
| 2026-10-19 | First core run | Reproduction failed on thread order; comparison waited | check |
| 2026-10-19 | Original topic-count rule | Only one setting passed all checks | check |
| 2026-10-19 | Fixed topic count | Every setting passed; style 0.229 beside CLIP 0.300 | check |
| 2026-10-19 | Labelled ceiling | Default Qwen style probe 0.527 beside raw 0.222 | check |

Sources: the experiment links in the corresponding steps above. Neither graph exploration had a preset success bar.

## Where we stand

Our network clearly led both chosen comparators only on style. The headline comparison below used image-side AMI, k = 20, on 5,000 development paintings. Genre scores used only paintings with a genre label. No held-out number exists for this line in this period.

| Method | Emotion | Style | Genre |
|---|---:|---:|---:|
| Qwen default image, clustered directly | 0.070 | 0.222 | 0.449 |
| CLIP image, clustered directly | 0.065 | 0.300 | 0.412 |
| PercepT core with 40 topics, best setting per aspect | 0.077 | 0.229 | 0.436 |
| Our network, default input, one pass | 0.081 | 0.348 | 0.458 |
| Instructed Qwen image, clustered directly, three passes, one per aspect | 0.093 | 0.395 | 0.496 |
| Label-trained linear probe on default Qwen images, ceiling reference | 0.121 | 0.527 | 0.657 |

Our network's advantage over PercepT core was +0.003 on emotion, +0.120 on style and +0.022 on genre. These differences came from the source's unrounded means. We did not subtract the displayed rounded values. Emotion was level with PercepT core and +0.011 above plain Qwen images. Genre was level with plain Qwen images (+0.0097). The strongest comparator per aspect therefore narrowed the claim to style.

The instructed Qwen row remained ahead of our network on every aspect (by 0.012, 0.047 and 0.038, from the rounded table values). Our reading: the graph-trained network recovered useful aspect structure from default input, yet the instruction itself still supplied the stronger image geometry. The ceiling probe showed further available signal under label supervision. It did not establish that an unsupervised method could reach that signal.

The network row combined separate aspect networks: the graph sheet picks for emotion and style, and the image-graph network for genre (the sheet pick scores 0.454). It was no result for a shared trunk. Target AMI selected those networks from both sides, which could favour a higher caption score over a higher image score. Other default-input emotion networks reached 0.085 and 0.089, while the selected network reached 0.081. We retained the selected result and changed the next selection rule explicitly.

The same evaluation paintings influenced graph choice, instruction choice and comparator choice. Seed spread described variation after those choices. It did not cover selection effects or establish a statistical test. The fixed topic count also limited the comparator claim. We established PercepT core with our features and our count; faithful reproduction remained separate work.

Sources: [fixed-count auto report](../auto/percept/2026-12-15_percept_core_k40.md), [graph follow-up](../auto/percept/2026-12-12_aspect_graph_followup.md), [two-bar user-read table](../../user_read/2026-10-19_percept_core_k40.md), [ceiling log](../../../src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md).

## Next

We decided to build the shared model next: one shared trunk with emotion, style and genre heads. It had no results yet. Its target was the instructed Qwen image, reached through one pass on the default image. The best graphs so far were residual for emotion and style, and the image graph alone for genre. The default input carried the claim; instructed input would be a second result.

We chose image-side AMI to select shared-model settings. Retrieval selection would wait for the retrieval stage. The next comparison tables would include PercepT core, direct image clustering, instructed Qwen and the labelled probe ceiling. Additional metrics would accompany AMI, while AMI would continue to govern decisions.

The project set the dates and required evidence:

| Date | What had to hold |
|---|---|
| 2026-10-29 | Held-out AMI at least +0.01 above PercepT core, and clearly above the other comparators; emotion had to meet this requirement. |
| 2026-11-10 | Reach the instructed Qwen image bar and show aspect-named retrieval above prompted Qwen. |

The instructed Qwen row in the headline table sets the target, and our current network is below it on all three aspects. We allowed the next work to take more than one loop. The later instructed-Qwen target did not replace the earlier comparator gate. We chose to specify and build the shared model after the container restart.

Source: [direction log, current state and 2026-10-19 decision](../../superpowers/direction_log.md), [aspect graph design](../../superpowers/specs/2026-10-18-aspect-graph-sweep-design.md).

## Glossary

- **Aspect:** the named property under study, here emotion, style or genre.
- **View:** features from a particular encoder instruction and data side.
- **Default input:** Qwen features without an aspect instruction at test time.
- **Buddy graph:** mutual nearest-neighbour pairs, as defined in the graph-sheet step.
- **Residual graph:** neighbours ranked after content similarity was subtracted from aspect similarity.
- **Our network:** the small graph-trained network read on default image features.
- **AMI:** adjusted mutual information between clusters and labels.
- **Image-side AMI:** AMI from clusters of image outputs alone.
- **Target AMI:** the selection score for the graph sheet networks that averaged image-side and caption-side AMI.
- **PercepT core:** our repaired implementation on our own features, as scoped in the project glossary.
- **Linear probe ceiling:** a labelled diagnostic reference for information available in frozen image features.

## Sources

The auto reports supplied experimental numbers. The direction log supplied the decisions and their order. File dates in December followed the project's sequence dates; the timeline used the recorded October experiment dates.

Auto reports:

- [Code audit](../auto/percept/2026-12-04_percept_code_audit.md)
- [New-line candidates and premise checks](../auto/percept/2026-12-06_new_line_candidates.md)
- [Prompted aspect views](../auto/percept/2026-12-08_prompted_aspect_views_check.md)
- [Image transfer and word masking](../auto/percept/2026-12-09_image_transfer_and_word_masking.md)
- [Purified buddy pilot](../auto/percept/2026-12-10_purified_buddy_pilot.md)
- [Aspect graph sheet](../auto/percept/2026-12-11_aspect_graph_sheet.md)
- [Aspect graph follow-up](../auto/percept/2026-12-12_aspect_graph_followup.md)
- [First PercepT core run](../auto/percept/2026-12-13_percept_core.md)
- [PercepT core arms](../auto/percept/2026-12-14_percept_core_arms.md)
- [PercepT core with 40 topics](../auto/percept/2026-12-15_percept_core_k40.md)

User-read reports:

- [Prompted aspects and student pilot](../../user_read/2026-10-17_prompted_aspects_and_student_pilot.md)
- [Purified buddy pilot](../../user_read/2026-10-18_purified_buddy_pilot.md)
- [Aspect graph sheet](../../user_read/2026-10-18_aspect_graph_sheet.md)
- [Aspect graph follow-up](../../user_read/2026-10-18_aspect_graph_followup.md)
- [First PercepT core run](../../user_read/2026-10-19_percept_core.md)
- [PercepT core arms](../../user_read/2026-10-19_percept_core_arms.md)
- [PercepT core and the two-bar table](../../user_read/2026-10-19_percept_core_k40.md)

Designs, context and decisions:

- [Aspect graph sweep design](../../superpowers/specs/2026-10-18-aspect-graph-sweep-design.md)
- [PercepT core design](../../superpowers/specs/2026-10-18-percept-core-design.md)
- [Image-side ceiling log](../../../src/test/20261019_emotion_ceiling/20261019_emotion_ceiling_log.md)
- [PercepT neighbourhood literature](../literature/2026-12-05_percept_neighbourhood.md)
- [Prompted aspect views literature](../literature/2026-12-07_prompted_aspect_views.md)
- [Direction log](../../superpowers/direction_log.md)
- [Project glossary](../../../GLOSSARY.md)
