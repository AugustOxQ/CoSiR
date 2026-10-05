# Reading an unnamed aspect from four example pairs: method attempts, controls and outcomes (2 to 4 October 2026)

**Stage report, CoSiR v2.** Self-contained summary of three days of work towards a CVPR submission (abstract 10
November 2026). Every number below comes from a reviewed report or result file named in the section's sources;
95% intervals resample anchor paintings (5,000 resamples, seed 42) unless stated otherwise.

*Revised 5 October 2026 after a question-and-answer review of the design: the metric and episode definitions and the
label levels (Section 2), where the method came from (Section 3), the evidence for learnability (Section 9.1), the
condition-free bar (Section 11.3) and the oracle layers (Section 14). No earlier number changed.*

## Abstract

We study *example-conditioned aspect similarity across modalities*: a system receives a query (an artwork image or a
caption), four image and caption pairs that agree on an unnamed aspect (for example emotion or style) and four pairs
that agree on a different aspect, and must rank captions (or images) by agreement with the query on the demonstrated
aspect. On ArtELingo the task is learnable in principle when items are represented in the evaluation taxonomy:
logistic probes trained on the evaluation labels reach R@1 25.02 on our development episodes (episode seed 42) when
they read the aspect from the pairs, 2.48 [2.11, 2.84] above the same probes with the condition ignored (22.54) and
far above CLIP cosine (12.96); told which aspect is meant, they reach 30.66. Over three days we tested a learned factor basis
read by a training-free agreement rule (method A), a repaired fusion of it (A′), in-context multimodal LLMs (2B and 8B),
and five further readers and fusions. None passed its pre-registered test. The common failure is arithmetic:
R@1 = (either rate + condition gain) / 2, and every scorer trained without ArtELingo labels that reads the condition lost nearly as much *either rate* (how often one of the two candidates that share an aspect with the query ranks first, whichever it is) as it gained in *condition gain* (how often the demonstrated aspect's
candidate ranks first minus how often the other aspect's candidate does; exactly 0 for any scorer that ignores the condition), or more, so none beat its *matched control* (the same score with only the condition removed) on R@1 with a lower bound above 0 (best +0.23 [−0.01, 0.47]). Two results change the outlook. The first is that the condition can be read: inferring
the aspect from the pairs keeps 63% of the told probes' gain, and a reader on cross-modal heads trained without ArtELingo labels (the affect partition is distantly supervised by a GoEmotions classifier whose categories name 6 of the 8 evaluation emotions, and the image clusters carry style and genre, Table 4) on k-means partitions reaches a condition gain of 4.41 [3.96, 4.87], 3.5 times the best rule on the learned factor codes (1.25) and 4.5 times method A's own rule (0.99). The second is that condition-free
scores that use the examples without reading the condition rose from 16.55 to 18.34 R@1 (B, a control built from our own ingredients rather than a published baseline), so the bar for a method is now higher and better defined, with a caveat about the episode construction (Section 11.3). We describe the task, the protocol, each attempt with its baseline, the controls that caught
two false passes, and what a passing method has to do.

## 1. Introduction

Similarity between an image and a text is usually a single number. People, however, compare things *in a respect*:
two paintings can be alike in mood and unlike in style. Prior work fixes the respect by name (conditional similarity
networks, instruction embedders) or by training on one respect at a time. We ask whether a model can be told the
respect only by examples, the way a curator would show a few pairs and say "similar like these", and whether it can
apply that respect across modalities, ranking captions for an image or images for a caption.

The CVPR plan (spec of 2 October, revised after an automated multi-persona LLM review, ARS) set three contributions: the task and its
benchmark (C1), a method (C2: a shared sparse image and text factor basis read by a training-free agreement rule) and
an analysis of how aspects are carried by the two modalities (C3). It also fixed in advance what would count as a GO for
the method and what paper each outcome leads to (Table 1).

*Table 1. Pre-registered branches of the plan (spec §4).*

| Branch | Condition | Paper |
|---|---|---|
| 1, GO | the method beats backbone-only cosine, the best raw "metric from pairs" baseline (RCA) and its own condition-free control on both R@1 and condition gain (95% lower bounds above 0) on a fresh episode draw | method paper |
| 2 | no GO, but an in-context MLLM has both lower bounds above 0 against cosine | benchmark paper with the MLLM as the scorer |
| 3 | neither | analysis and negative-results paper |

The three days covered: redefining the task after two spikes (Section 3, which also traces how the method grew out
of the buddy graph and the first v2 design); nine baselines (Section 4); method A and its
GO test (Section 5); a repair (Section 6); MLLM probes (Section 7); a literature-checked list of new candidates
(Section 8); three quick checks and two follow-ups run overnight on 4 October (Sections 9 and 10). Figure 2 shows the
chain. Section 11 puts all scorers on one plot and explains the common failure; Section 12 lists what the process
caught; Section 13 the limits.

![Chain of the three days](../assets/2026-10-04_aspect_conditioned_similarity_methods/chain.png)

*Figure 2. The three days as a chain of experiments, each with its headline number. Grey: context and baselines;
orange: an attempt that failed its pre-registered GO or gate; teal: a positive finding.*

**Main findings.**

1. **The task is learnable but not by the methods tried.** Told label probes reach R@1 30.66 and condition gain 21.06
   on the development episodes (cosine 12.96 and 0). Reading the aspect from the pairs with the same probes reaches
   25.02, 2.48 [2.11, 2.84] above the probes with the condition ignored, so the reading step pays off when items carry
   one block of features per aspect (Section 9.1). No method trained without ArtELingo labels beat its matched condition-free control on R@1 with a lower bound above 0 (best: N6 +0.23 [−0.01, 0.47], N6c +0.15 [−0.06, 0.38]), although fused condition gains reached 1.69 [1.35, 2.02].
2. **Method A failed its GO** on fresh episodes: R@1 13.76 against 13.53 for cosine and 16.72 for its own
   condition-free control, gain 0.26 [−0.04, 0.56]. Its agreement term alone selects the aspect (gain 0.97) but finds
   aspect-sharing candidates less often than cosine (either 21.4 against 27.1).
3. **Every repair hit the same wall.** A nested fusion (A′), factors trained on the labels as a diagnostic, a centered rule (N1), a
   find-then-select cascade (N2) and two head-based readers (N6, N6c) all landed within 0.3 R@1 of, or below, their
   matched condition-free controls.
4. **The condition can be read.** With label probes, the aspect inferred from the pairs keeps 63% of the told gain.
   With cross-modal heads on k-means partitions trained without ArtELingo labels (the affect partition is distantly supervised by a GoEmotions classifier whose categories name 6 of the 8 evaluation emotions, and the image clusters carry style and genre, Table 4), the reader's condition gain is 4.41 (A3's rule: 0.99).
5. **The condition-free bar rose.** Centering factor codes on the episode's example items, and averaging partition
   heads, lifted the best score that ignores the condition to 18.34 R@1 (RCA 13.38), part of which may come from how
   the episodes are built (Section 11.3). This score, B, is our own control and sets the R@1 bar a method must clear.
6. **Controls matter.** Two configurations passed against a control that removed more than the condition; matched
   controls (the same score with only the condition removed) removed both passes: N1 fell 1.15 [0.92, 1.38] below its matched control, and N6c's margin shrank from +0.55 [0.31, 0.79] to +0.15 [−0.06, 0.38].

## 2. Task, data and protocol

### 2.1 The aspect episode

![Aspect episode](../assets/2026-10-04_aspect_conditioned_similarity_methods/task_schematic.png)

*Figure 1. One aspect episode. The query has values on aspects A and B. Each of the four support pairs (teal) is an
image of one painting with a caption of another that agree on a value of A; the four pairs show four different values.
The four contrast pairs (orange) do the same for B. Candidate p_A shares the query's value of A, p_B its value of B, eleven negatives (grey) share neither; all 13 candidates differ from the query on the third aspect. Swapping
supports and contrasts must move p_B to the top.*

An *aspect episode* (spec §5.1) consists of:

- a **query** taken from one *anchor* row of a selection painting (a *row* is one image of a painting with one
  viewer's caption): the anchor's image when captions are ranked (image to caption, i2t) and its caption when images
  are ranked (caption to image, t2i), so every episode is scored in both directions;
- **4 support pairs**, each an image of one painting with a caption of another painting (*cross-item*), the two sharing
  a value of the conditioned aspect A and differing on B; the four pairs show four different values, never the query's own (*value-disjoint*);
- **4 contrast pairs** built the same way for a second aspect B;
- **13 candidates** in the other modality: p_A shares the query's value of A, p_B its value of B, 11 negatives share
  neither; all 13 candidates differ from the query on the third aspect. All 30 rows of an episode come from 30 different paintings.

Under condition A the target is p_A; swapping supports and contrasts (condition B) makes p_B the target. The aspect is
never named. On ArtELingo the aspects are **emotion** (8 values, the catch-all "something else" removed), **style**
(23 values with at least 30 selection paintings) and **genre** (10 values, from WikiArt via ArtGAN), giving three aspect
pairs: emotion × style, emotion × genre and style × genre. We draw 4,096 episodes per pair, 12,288 per *episode seed*.

Every *episode* in this report is an evaluation episode of this form. One episode holds both conditions (the same
query, candidates and eight pairs, with the roles of supports and contrasts exchanged) and both directions, so it
yields four rankings, and its per-episode metrics average over them (`per_anchor` in `src/eval/aspect_metrics.py`).
Anchors are drawn with replacement, so the 12,288 seed-42 episodes fall on 4,602 anchor paintings, which is why the
bootstrap resamples paintings. The pseudo-aspect episodes that trained method A (Section 5.1) are a separate set built
from scorer-train rows, with k-means clusters in place of labels; no result in this report is scored on them.

The construction controls the example pairs only partly (`_pairs` and `build_aspect_episodes` in
`src/eval/aspect_episodes.py`). Every one of the 16 example rows comes from a painting that has no row with the query's
value of A or of B, so the example items are drawn from the same "shares neither value" population as the 11
negatives (Section 11.3 returns to this). The third-aspect control applies to the 13 candidates only. The pairs are
not constrained on the third aspect, so an emotion pair in an emotion × style episode may also share a genre, and
anything correlated with the shared value (artist, period, palette) is free; how often pairs share the third aspect
has not been counted. The four support pairs show four different values, and what they have in common is that each
pair agrees within itself on A. This is why the readers of Sections 5 to 14 compare agreement within pairs; a prototype
of the supports would average four different values.

### 2.2 Data

ArtELingo (English): 308,723 image and caption rows over 61,402 paintings. Every item is represented by frozen CLIP
ViT-B/32 image or caption features; no backbone is fine-tuned.

The aspects are labelled at different levels (`artelingo_aspect_labels` in `src/data/artelingo_splits.py`). A row is
one viewer's caption with that viewer's emotion, and a painting has about five rows (308,723 / 61,402 ≈ 5.0), which can
carry different emotions. Style and genre belong to the painting and are shared by all its rows (genre is looked up by
painting name). For emotion the episode builder therefore mixes two strengths of rule. Membership is per row: two rows
share emotion v when both are labelled v. The image half of an emotion pair is then "a painting that at least one viewer
felt v about", labelled through a caption the episode does not show, while the caption half states the emotion of the
viewer who wrote it; and two rows that differ on emotion can still share it through other viewers of the same paintings.
Exclusion is per painting: p_B, the negatives and the example rows come from paintings that no viewer labelled with
the query's emotion (in style × genre episodes, where emotion is the third aspect, so do all 13 candidates), which
keeps false negatives out. How mixed the paintings' emotions are has not been measured
(spec E12 planned an annotator-agreement count; it was not run).

Rows are split by painting:

*Table 2. Splits (spec §2.3).*

| Split | Rows | Paintings | Use |
|---|---:|---:|---|
| scorer-train | 183,694 | 36,518 | training any model or probe |
| selection | 32,413 | 6,451 | all development and test episodes of this stage |
| val | 30,872 | | unused |
| held | 61,744 | 12,281 | reserved for the final paper test; not read for aspect episodes |

Episodes are drawn from selection rows only, so every development or test number below is on paintings no model was
trained on. Different episode seeds reuse the same 6,451 selection paintings: a fresh seed is a fresh draw of episodes,
not of paintings.

### 2.3 Metrics

For each ranking (two directions, image to caption and caption to image, times two conditions):

- **R@1**: the target ranks strictly first; ties count as misses. Chance is 1/13 = 7.69%.
- **Other-aspect rate**: the other aspect's candidate ranks first.
- **Condition gain** = R@1 − other-aspect rate. It is exactly 0 for any scorer that ignores the condition, whatever its
  R@1.
- **Either rate** = R@1 + other-aspect rate: how often *some* aspect-sharing candidate ranks first.

Each ranking ends one of three ways, and R@1 and the other-aspect rate are the shares of rankings that end the first
and the second way:

| What ranks strictly first | Adds to R@1 | Adds to the other-aspect rate |
|---|---|---|
| the target (p_A under condition A, p_B under B) | 1 | 0 |
| the other aspect's candidate (p_B under A, p_A under B) | 0 | 1 |
| a negative, or a tie at the top | 0 | 0 |

The code stores R@1 and the other-aspect rate; either rate and gain are derived from them. Gain counts first places
only, as a net difference. The pairwise *swap* statistic of spec §5.1 (p_A above p_B under A and p_B above p_A under B)
is a different metric: a ranking in which the target beats the other aspect's candidate while a negative ranks first
adds nothing to gain.

From the last two, **R@1 = (either + gain) / 2**. This identity organises the whole stage: a scorer can raise R@1 by
finding aspect-sharing candidates more often or by choosing the conditioned one more often, and a scorer that reads the
condition must not lose more of the first than it gains in the second.

The identity itself is algebra, ((R + O) + (R − O)) / 2 = R; its use is that it splits R@1 into two parts that scorers
move separately. Equivalently, R@1 = either × q, where q is the share of aspect-sharing first places that go to the
target. A scorer that ignores the condition ranks identically under A and B, so each of its aspect-sharing first places
is right under exactly one condition: q = 0.5 and the gain is 0 (cosine: either 25.92, R@1 12.96). Reading the condition
raises q (N6's reader alone: 16.51 / 28.61 = 58%; label probes told the aspect: 30.66 / 40.27 = 76%), and the either rate
is the base that q multiplies. The identity also holds for differences, ΔR@1 = (Δeither + Δgain) / 2, so a fused score
clears its control by 0.5 R@1 only if its gain exceeds its either loss by 1 point.

### 2.4 Uncertainty, seeds and pre-registration

Intervals resample anchor paintings (clusters), so an episode's dependence on its painting is respected. Development
used episode seed 42; tests used fresh seeds (43 for method A; 45, 47 and 48 for N1). A ledger records every use of
every seed. Each decision rule was committed to git before the numbers it governs were read; two exceptions are disclosed in Section 12 (Addendum 1 was committed two minutes after the N1 test had been computed but before its outputs were read; N6c's comparison with its declared control (cosine plus A3's centered term, R@1 17.94; +0.55) was already known from an exploratory run), and each stage ended
with a whole-branch review on the most capable model that re-derived every load-bearing number from the stored
per-anchor arrays with independent code.

### 2.5 Baselines and controls

- **Backbone only (cosine)**: CLIP cosine of query and candidate; condition gain 0 by construction.
- **Raw metric from pairs** (E1, Section 4): nine ways to turn example pairs into a metric (KISSME, RCA, Xing and CVS from the literature; the others are generic or our own). The best on seed
  42, **RCA**, is the *GO bar*.
- **Condition-free control**: the method's own score with the condition removed. A **matched** control removes only the
  condition and keeps every other ingredient of the score (same terms, same weight budget). Section 9.2 shows why the
  word "matched" matters.
- **Best condition-free score (B)**: the strongest condition-free score measured in the stage (R@1 18.34 on seed 42;
  defined in Section 14). B is a control built from our own ingredients, not a published baseline. For a new
  configuration the GO comparators are cosine, RCA, B (extended to B′ by any new condition-free ingredient) and the
  configuration's matched counterpart, which can sit above B (Section 11.3).
- **Label-probe reference** (diagnostic, never a method): logistic probes trained on the evaluation labels of 60,000
  scorer-train rows; an item is represented by its class posteriors.

*Sources: spec `docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md` §2 to §6 and §10;
[E0 evaluation setup](../auto/v2/2026-10-29_aspect_eval_setup.md); episode ledger `docs/superpowers/episode_seed_ledger.md`;
code `src/eval/aspect_episodes.py`, `src/eval/aspect_metrics.py`, `src/data/artelingo_splits.py`; episode and painting
counts from `src/test/20261109_fix_diagnostics/results/diagnose_fixes.txt`.*

## 3. How the task was arrived at

**Where the method came from: the buddy graph and the first v2 design.** The project began with the *buddy* line (June
to mid September): a small trainable condition vector per sample, initialised from a *buddy graph* (two items are
linked when each is among the other's nearest neighbours) and mixed into the frozen CLIP feature by a combiner. Its last
experiment raised cluster separation (silhouette 0.55 to 0.70) while image to caption R@1 fell to about 10.4, against
17.8 for CLIP. The *percept* line (15 to 28 September) then put a buddy graph in place of the topic-forming stage
(Stage 1) of PercepT, a published emotion-aware topic method. A student network was trained on two teacher graphs, a
content graph and an affect graph built from the 28 GoEmotions probabilities of a RoBERTa classifier read over the
captions, and Leiden clustered the student's space into communities. On held-out paintings these communities matched
our PercepT replication in agreement with the labels (emotion AMI 0.1241 against 0.1252, genre 0.2406 against 0.2486,
4 seeds), and a matched head-to-head of PercepT's image-only topic mapper (Stage 2) detected no difference at equal
emotion. Across buddy trials, Stage 2 AUC fell as emotion AMI rose (r = −0.70 and −0.74), so a high AUC said that
topics were easy to predict from the image, not that they captured conditions.

On 28 September we rebuilt CoSiR as v2 around a score s(I, T | c) whose condition c is given by example pairs. The
first design had a Block 1, the buddy Stage 1 rebuilt with the content graph only, and a Block 2 (*Candidate A*): 32
shared image and caption factors trained on raw CLIP features, one of whose losses keeps a code consistent with its
neighbours in the buddy graph. The pieces of the buddy pipeline then left one at a time:

- The Stage 2 topic mapper was not carried into v2, because topic classification is not the target.
- The affect graph was planned for Block 1 and never built.
- Block 1's student restated CLIP (emotion AMI 0.0369 against 0.0358 for raw CLIP), and its communities fed nothing
  downstream; only its graph entered the factor loss.
- As a source of training conditions for the stage (d) scorer, Block 1 communities scored best on the selection split
  (+3.00) but lost the pre-registered tie-break to CLIP feature clusters (+2.54) and were never tested on held rows. A
  later probe found them aligned with art style at AMI 0.227, below CLIP image k-means at 0.318, and with emotion at
  0.035, like every label-free source then available.
- Factors trained on value-condition episodes from k-means clusters of the GoEmotions affect vectors and of CLIP image
  features (SE) were the one cell of that stage confirmed on held rows (emotion +2.08 [1.51, 2.62] over the matched
  control C0). On
  2 October the user approved k-means partitions (affect, image, caption) as the training signal of method A
  (Section 5.1), with the risk that pseudo-aspects are only proxies recorded as R-pseudo.

Two buddy ingredients survive. Method A's factor recipes (R3, C0, A3) keep the graph-consistency term at weight 1 on a
mutual nearest-neighbour graph of image features (`lambda_graph` in `src/train/train_factors.py`; E2's `graph.npz`), and
the affect partition reads the same kind of GoEmotions signal as the percept line's affect graph, as k-means clusters
instead of a graph clustered by Leiden. N6 (Section 10) uses neither graph nor factors. One difference in goal is our
reading and was not tested: the percept line tuned one partition to carry emotion and genre together (its pass bar
asked for both AMIs at once), while an aspect reader needs a separate partition per aspect, so that agreement within a
pair on a partition points at one aspect.

**Value episodes measured recognition, not similarity.** The first benchmark conditioned on a *value* ("sad, like
these"): supports carried the query's own label. A prototype of the supports that ignored the query entirely scored
R@1 22.83 on these episodes, and adding the query raised it by only 0.57 [0.20, 0.96]; the best raw-CLIP probe (24.10)
beat our factor model SE (the C0 factor recipe plus value-condition episodes from GoEmotions affect clusters; 21.22) by 2.87 [2.14, 3.64]. The supports gave the answer away, so we redefined the condition to
show an *aspect* through values the query does not have.

The same reading applies to the first v2 result. On held value episodes the repaired factors R3 with the naive rule had
reached R@1 17.6 / 20.9 (image to caption / caption to image) against 11.5 / 14.9 for CLIP. R3 with uniform weights,
which ignores the condition, already reached 15.6 / 15.7, and the part that depends on the condition (+2.0 [0.5, 3.6] /
+5.2 [3.5, 6.8]) measures recognition of the value from the supports, the shortcut described above. On aspect episodes the condition-free part of that lift
reappears in the uniform factor term (Section 4) and later in B (Section 11.3).

**On aspect episodes the factor models did nothing, but labels showed headroom.** On 4,096 emotion × style aspect
episodes (no third-aspect constraint), CLIP cosine reached 11.13 R@1 and SE with the agreement rule 11.32 (+0.20
[−0.12, 0.50]); every other factor model and the raw-feature rule stayed at CLIP level. Label probes told the aspect
reached 23.09. Only a privileged reference given the label *names* as text prompts gained (12.63, +1.51 [0.94, 2.09]).

**The modalities carry the aspects unevenly.** Probes predict emotion from captions far better than from images (56.9%
against 35.2%) and style from images far better than from captions (60.8% against 25.4%). Four frozen backbones (CLIP,
SigLIP 2, PE-Core, Qwen3-VL-Embedding-2B) moved the weak side by at most 1.5 points (emotion from images 35.1 to 36.6;
style from captions 25.4 to 26.3) while moving the strong side by up to 11, so the asymmetry is not specific to one backbone; whether it comes from the modalities or from how the ArtEmis captions were collected is not separated. A cross-modal match is capped by the weaker side. We kept CLIP for all method work.

**Novelty.** A literature search found no paper that defines the task, but every ingredient exists: per-query weights
from a few examples (Contextual Visual Similarity, Wang et al. 2016), metric learning from pairs (Xing 2002, RCA, KISSME
2012), one example pair defining a relation (MARS, ICLR 2023), in-context text embedders. Our agreement rule reads as "a
few-shot, cross-modal, diagonal KISSME", so the novelty claim rests on the task and the combination, stated "to our
knowledge" within the search bounds.

*Sources: [weekly report of 30 September](../weekly/2026-09-30_percept_buddy_to_v2.md) §2 to §5 and its slides;
handoff `docs/superpowers/handoffs/2026-09-30-candidate-a-factor-learning-handoff.md`;
[held evaluation of the repaired factors](../auto/v2/2026-10-12_candidate_a_condition_eval_repaired_factors.md);
[factor headroom probe](../auto/v2/2026-10-15_candidate_a_factor_headroom_probe.md) Result 4;
[affect factor learning](../auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md) and its
[held test](../auto/v2/2026-10-19_candidate_a_affect_factor_learning_held.md); `src/train/train_factors.py`;
[support-baseline spike](../auto/v2/2026-10-22_support_baseline_spike.md);
[aspect-episode spike](../auto/v2/2026-10-23_aspect_episode_spike.md); [novelty check](../auto/v2/2026-10-24_aspect_task_novelty_check.md);
[backbone check](../auto/v2/2026-10-25_backbone_check.md); [literature review](../auto/v2/2026-10-21_cvpr_literature_review.md).*

## 4. Baselines: nine ways to read pairs, none separates the aspects (E1)

E1 ran nine raw-feature baselines on the 12,288 seed-42 and seed-43 episodes: the agreement rule on raw CLIP dimensions
(signed and rectified), a bilinear form, KISSME and RCA on a 32-component whitened PCA basis, Xing's metric, Wang et
al.'s per-query weights, a per-episode logistic probe on pair products, and a Tip-Adapter style cache. Each was fused with
cosine as z(cos) + λ·z(term), λ cross-fitted on the two halves of the episodes.

*Table 3. E1 on seed 42 (seed 43 in brackets where it matters). Cosine R@1 12.96 (13.53).*

| Scorer | R@1 | Condition gain |
|---|---|---|
| cosine | 12.96 [12.67, 13.26] | 0 |
| **RCA (GO bar)** | 13.38 [13.08, 13.69] | 0.10 [−0.08, 0.29] |
| Wang et al. (CVS) | 12.99 | 0.36 [0.06, 0.66] (seed 43: 0.30 [0.06, 0.55]) |
| per-episode probe | 12.79 | 0.35 [0.04, 0.65] |
| other six | 12.85 to 13.06 | −0.16 to 0.12 |
| SE codes, uniform weights (condition-free) | **16.30** [15.96, 16.63] | 0 |

No baseline separated the aspects: every gain stayed within 0.4 points of 0. The last row matters most. A factor
term with *uniform* weights, which ignores the condition, lifted R@1 by 3.3 points through the either rate alone. An
R@1 lift is therefore not evidence of reading the condition, and every method must beat its own condition-free control.
The ranking of the nine baselines was not stable across seeds (RCA fell below cosine on seed 43 by its mean of R@1 and
gain).

*Sources: [E1 baselines](../auto/v2/2026-10-30_aspect_baselines.md).*

## 5. Method A: a factor basis trained on pseudo-aspect episodes (E2, E3)

### 5.1 The method

![Method A](../assets/2026-11-01_aspect_factor_gonogo/method_diagram.png)

*Figure 3. Method A against the earlier factor models. Grey: unchanged from the C0 recipe; purple: the shared
architecture; teal: new in method A; orange (dashed): replaced.*

Two small encoders map frozen CLIP image and caption features into a shared space of 32 non-negative factors (the
*codes*). The *agreement rule* turns the 4 + 4 example pairs into factor weights without training,
w = ReLU(mean over supports of a_I ⊙ a_T − mean over contrasts of a_I ⊙ a_T), L1-normalised, where a_I and a_T are the
image and caption codes of a pair; the *agreement term* of a candidate is Σ_l w_l q_l c_l. The score is
z(cos) + λ·z(term). Training adds, to the earlier C0 recipe (InfoNCE pair agreement, decorrelation, sparsity), a
*pseudo-aspect episode loss*: ranking cross-entropy through the agreement rule on episodes whose "aspects" are label-free
k-means partitions of the scorer-train rows (E2): **affect** (k-means on GoEmotions probabilities of the captions),
**image** (k-means on CLIP image features) and **caption** (k-means on CLIP caption features), 64 clusters each, in banks of
65,536 episodes. No evaluation label is used. The partitions are, however, aspect-shaped (Table 4).

*Table 4. Adjusted mutual information between the label-free partitions and the evaluation labels (scorer-train rows,
shuffled-label null about 0.00001).*

| Partition | Emotion | Style | Genre |
|---|---:|---:|---:|
| affect | 0.204 | 0.016 | 0.023 |
| image | 0.037 | 0.318 | 0.397 |
| caption | 0.057 | 0.058 | 0.161 |

### 5.2 The GO test

Six runs (A1 to A6: loss weights, factor count, swap term, sparsity) were trained for 2,000 steps each; the pre-registered
pick rule chose A3 (aspect loss weight 3) on seed 42 (R@1 13.39, gain 0.52 [0.19, 0.87]). On fresh seed-43 episodes, scored
once:

*Table 5. E3's GO test, seed 43 (12,288 episodes, 4,575 anchor paintings).*

| Comparator | Its R@1 | A3 minus it, R@1 | A3 minus it, gain |
|---|---|---|---|
| cosine | 13.53 | +0.23 [0.00, 0.46] | +0.26 [−0.04, 0.56] |
| RCA (GO bar) | 13.52 | +0.24 [0.00, 0.48] | +0.32 [0.01, 0.64] |
| A3's uniform-weight control | **16.72** | **−2.96 [−3.29, −2.64]** | +0.26 [−0.04, 0.56] |

**NO-GO.** The gain halved from the pick draw (0.52 to 0.26) and A3 lost almost 3 points of R@1 to its own control. The
held-out-aspect test (K8: a run trained on the affect and image partitions only, with no caption partition and no genre label, though its image partition carries genre, AMI 0.397, tested on genre pairs) also failed, and so did its
positive control, so it did not isolate the held-out aspect. A 2B MLLM probe also failed (Section 7), and the
pre-registered map pointed to branch 3.

### 5.3 Why it failed (post hoc)

Scored alone, A3's agreement term did select the conditioned aspect: condition gain 0.97 [0.53, 1.41] on seed 43 and 0.99
on seed 42, about one point above SE and C0. But it ranked an aspect-sharing candidate first in only 21.4% of rankings,
against 27.1% for cosine and 33.4% for the uniform control. With R@1 = (either + gain) / 2, beating the control's 16.72 at
the term's either rate would need a gain of about 12 points. The training loss also stayed 1.0 to 4.4% below its
constant-score value throughout: the model barely fitted the pseudo-aspect episodes.

*Sources: [E2 partitions](../auto/v2/2026-10-31_pseudo_partitions.md); [E3 go/no-go](../auto/v2/2026-11-01_aspect_factor_gonogo.md)
§2 to §7, §6.1, §11.*

## 6. Repair A′: keep the condition-free term, add the conditioned one

The user chose to try a repair before branch 3. An automated two-seat methodology review of the repair order (ARS) returned Major
Revision with twelve rule changes, including a pick rule aligned with the binding comparison: the old criterion,
mean(R@1, gain) = either/4 + 3·gain/4, could prefer a cell that adds 1.0 gain while losing 2.9 either and 0.95 R@1.

**Method A′** kept A's codes and rule and changed the test score to a *nested score*
z(cos) + λ_u·z(T_u) + λ_a·z(T_a), where T_u is the uniform-weight term and T_a the agreement-weighted term, so that the
condition-free term's either rate is kept and the conditioned term is added on top. Weights came from a 56-cell grid,
cross-fitted with a *min-margin* rule: on each tuning half, pick the cell maximising min(R@1 − the control's R@1, gain).
The control is z(cos) + σ·z(T_u).

- **Pilot on A3 (seed 42):** R@1 16.52 against its control 16.55, gain −0.01 [−0.10, 0.07]. One tuning half picked the
  control itself; across the 56 cells no cell with any weight on T_a beat the control's best R@1, and every cell with a
  gain above 0.5 had R@1 at most 14.26 (Figure 4 of the A′ report).
- **Label-trained factors (diagnostic H3):** trained on the evaluation labels with A3's settings, the term's gain roughly
  doubled (L3: 1.85 against A3's 0.99 on seed 42) but its either rate stayed below cosine (22.93 against 25.92) and its
  nested gain was 0.01 to 0.05, below the pre-registered bar of 0.218.

The pre-registered decision was **branch 3, stop the repair**.

![A′ trade-off](../assets/2026-11-05_method_repair_diagnostics/h1_tradeoff.png)

*Figure 4. A′ on A3, seed 42: the 56 weight cells as seven rows; weight on the conditioned term buys gain only by losing
either rate at least as fast (from the A′ report).*

*Sources: [ARS repair-order review](../auto/v2/2026-11-04_ars_repair_order_review.md);
[A′ diagnostics](../auto/v2/2026-11-05_method_repair_diagnostics.md).*

## 7. In-context multimodal LLMs

We gave Qwen3-VL the episode in context: four example pairs, four counter-example pairs (each an image and a caption of
two artworks), the query and 13 lettered candidates, with the instruction to pick the candidate "alike to the query in
the same respect as the example pairs"; candidates were scored by the logit of their letter, with letters permuted per
episode.

*Table 6. MLLM probes against cosine on the same episodes (pre-registered rule: "works" if both lower bounds are above 0).*

| Model | Episodes | R@1 minus cosine | Gain | Either minus cosine | Works |
|---|---|---|---|---|---|
| Qwen3-VL-2B (fixed prompt) | seed 44, 900 | +0.22 [−1.13, 1.53] | −0.53 [−1.68, 0.66] | | no |
| Qwen3-VL-8B | seed 46, 1,800 | +1.07 [0.17, 1.93] | +0.21 [−0.51, 0.94] | +1.93 [0.25, 3.51] | no |

The 8B model found aspect-sharing candidates more often than cosine but did not choose the demonstrated aspect, so branch 2
was not met either.

*Sources: [E3 §9](../auto/v2/2026-11-01_aspect_factor_gonogo.md); `src/test/20261106_mllm_probe_8b/` (pre-registration and log).*

## 8. New candidates and the literature check

With the user's decision deferred, we drafted six candidates from the failure analysis and checked each against the
literature with an automated research pipeline (six scans and a synthesis):

*Table 7. Candidates (all "partially exist": every ingredient is published, the combination for this task was not found).*

| Id | Idea | Treatment |
|---|---|---|
| N1 | centered agreement rule: weights from the covariance of image and caption codes across the support pairs | run (a variant of KISSME and CCA) |
| N2 | find then select: rank by the condition-free score, reorder only the top k by the condition | run (retrieve then rerank) |
| N3 | concept-group basis from an LLM attribute vocabulary | deferred behind a privileged reference |
| N4 | meta-trained set-encoder conditioner | dropped (evidence that such adapters fit pseudo tasks, not real ones) |
| N5 | name the aspect with an MLLM, embed with the name | deferred behind a privileged reference |
| N6 | cross-modal classifier heads on the label-free partitions, head chosen by the pairs | run if the diagnostic D0 passes |

The synthesis ordered a diagnostic first (D0: can the aspect be read at all when items are well represented?), then N1 and
N2, then N6. A spec fixed the three checks and a decision table; the user approved it and set D0's threshold.

*Sources: `src/test/20261107_new_method_candidates/` (`candidates_draft.md`, `scan_N1.md` to `scan_N6.md`, `synthesis.md`);
spec `docs/superpowers/specs/2026-10-04-new-method-quick-checks-design.md`.*

## 9. Quick checks D0, N1, N2 and the fresh-seed test (night of 3 to 4 October)

### 9.1 D0: the aspect can be read from the pairs when items carry aspect blocks

D0 represents every item by label-probe posteriors. *Told* scores the posterior dot product on the conditioned aspect;
*Inferred* picks the aspect whose posteriors agree most within the support pairs minus within the contrast pairs.

*Table 8. D0 on seed 42.*

| Scorer | R@1 | Gain | Either |
|---|---|---|---|
| cosine | 12.96 | 0 | 25.92 |
| mean of the three probe scores (condition-free) | 22.54 [22.15, 22.93] | 0 | 45.08 |
| Told (labels, aspect given) | 30.66 [30.16, 31.16] | 21.06 [20.48, 21.61] | 40.27 |
| Inferred, hard pick | 25.02 [24.58, 25.48] | 13.33 [12.78, 13.91] | 36.70 |
| Inferred, soft weights | 24.97 [24.52, 25.41] | 9.62 [9.12, 10.11] | 40.31 |

Inferred kept 63% of Told's gain (the pre-registered threshold for "close" was 50%). The hard pick was right in 59% of
rankings (chance 33%), but only 38% on emotion × style, where Inferred kept 31% of Told's gain; on emotion × genre it kept
74% and on style × genre 67%. Emotion and style are weak in opposite modalities, so their within-pair agreements separate
poorly.

Told is given the aspect and never looks at the pairs, so it skips the step that defines the task; what it shows is
that matching on a named aspect across modalities can be recovered from frozen CLIP features. The stronger evidence
that the task is learnable is Inferred. With the condition ignored, the same probes reach R@1 22.54 (the mean of the
three probe scores), and reading the aspect from the pairs adds 2.48 [2.11, 2.84] R@1 for the hard pick (Told adds
8.12 [7.75, 8.50]). The three-probe mean is condition-free but differs from the matched counterpart of Section 14,
(T_a + T_b) / 2 over the two picked aspects, which was not computed for Inferred. The result has a limited scope. The
probes are trained on exactly the evaluation taxonomy, so each aspect has its own block of posteriors, which is the
representation spec §6 says the rule needs; whether a representation learned without these labels can supply such
blocks is the open question of the stage. Inferred chooses among the three aspects the benchmark contains, and within
an episode the gain measures a choice between A and B. Told's 30.66 is not a ceiling either: cross-modal matching is
capped by the weak side of each aspect and, for emotion, by labels given per viewer (Section 2.2).

### 9.2 N1: a pass against the wrong control

Alone on A3's codes, the centered rule found aspect-sharing candidates more often than the agreement rule (either 26.91
against 21.04; cosine 25.92) at a similar gain (1.25 against 0.99). Inside A′'s nested score, N1 on A3 reached R@1 16.79
against the declared control's 16.55: +0.24 [0.09, 0.39], gain +0.26 [0.05, 0.47]. It was the only one of nine
configurations to pass, so the table sent it to a test on three fresh seeds.

**The fresh-seed test was NO-GO** (seeds 45, 47, 48 pooled, 36,864 episodes): R@1 minus the control +0.07 [−0.04, 0.17],
gain minus RCA's +0.14 [−0.02, 0.29]. The gain halved again from the pick draw (0.26 to 0.15).

**The final review then showed the pass was spurious.** The declared control used the *uncentered* uniform term, so it
removed two things at once: the condition and N1's centering of the query on the episode's 8 example items. N1's own
condition-free version, the centered uniform term, is by itself the strongest condition-free scorer we had measured
(R@1 17.95). Against the *matched* control (the same nested family with N1's term replaced by its condition-free
version), N1 lost 1.15 [0.92, 1.38] R@1 on seed 42 and 1.19 [1.05, 1.33] on the test seeds. Re-applying the table with
matched controls sent the work to N6.

### 9.3 N2: reranking a short list did not protect the either rate

Both aspect candidates were in the control's top 2, 3 and 5 in 4.7, 10.9 and 25.8% of rankings. Reordering only the top k
by a conditioned term lost R@1 at every k (agreement term: −2.29 to −4.58; N1's term: −0.46 to −1.91) while adding
0.4 to 0.7 gain, because the term promoted negatives inside the short list.

*Sources: [quick-checks report](../auto/v2/2026-11-08_new_method_quick_checks.md) §4 to §7 and its addenda in
`src/test/20261108_new_method_quick_checks/`; condition-free probe mean and its differences from
`src/test/20261109_fix_diagnostics/results/diagnose_counterparts.txt` §1.*

## 10. N6 and N6c: a reader that works, a fusion that does not yet

**N6** trains, for each of E2's three label-free partitions (so the heads are trained without ArtELingo labels) and each modality, a logistic head (64 classes) on CLIP
features of 60,000 scorer-train rows; an item is represented by its three head posteriors. The reader is D0's rule over
the three partitions; the condition-free version averages the three heads.

*Table 9. N6 on seed 42.*

| Scorer | R@1 | Gain | Either |
|---|---|---|---|
| N6 reader term alone | 16.51 [16.15, 16.87] | **4.41 [3.96, 4.87]** | 28.61 |
| N6 condition-free term alone | 16.96 [16.60, 17.30] | 0 | 33.91 |
| N6 nested (cosine, condition-free term, reader term) | 17.30 [16.94, 17.66] | 1.69 [1.35, 2.02] | 32.91 |
| its control | 17.07 [16.73, 17.42] | 0 | 34.15 |

The reader's gain of 4.41 is the largest condition gain without ArtELingo labels measured in the project during the stage (the best factor rule 1.25;
factors trained on the evaluation labels 3.20 with N1's rule; the 8B MLLM 0.21); two exploratory contrastive readers scored alone after the stage reached 5.10 and 6.07 (`src/test/20261109_fix_diagnostics/results/diagnose_fixes.txt`), but they did worse once fused (Section 14). Inside the nested score, N6 beat its
control on gain but not reliably on R@1 (+0.23 [−0.01, 0.47]; the lower bound stayed at or below 0 for all of
bootstrap seeds 0 to 99), so it failed its pre-registered pass. The reader picked the affect partition for most
emotion-conditioned rankings and the image partition for most style- and genre-conditioned ones, except the style
condition of style × genre, where image clusters carry genre more than style and the reader picked affect.

**N6c** came from an exploratory look at seed 42: N6's reader term placed on the strongest condition-free base
(cosine plus A3's centered uniform term, R@1 17.94) reached 18.49. Committed as a new configuration after the exploratory look, before its matched control was computed, with a gate
against its matched control (the same score with N6's reader replaced by its averaged heads), it reached R@1 +0.15
[−0.06, 0.38] over that control (18.34), so the gate failed and no test seed was spent.

Two overnight rulings by the controller are disclosed here. Addendum 2 §4 said no further method would be built overnight after N6 failed; the controller overrode that line for N6c, adopted the matched control, and took the "build N6" step that the committed decision rule had not named (rulings in the quick-checks log).

![N6 trade-off](../assets/2026-11-08_new_method_quick_checks/n6_tradeoff.png)

*Figure 5. Either rate against condition gain for the N6 scorers on seed 42 (from the quick-checks report). Reading
scorers (orange) sit up and to the left of the condition-free ones (blue); dotted lines are constant R@1.*

*Sources: [quick-checks report](../auto/v2/2026-11-08_new_method_quick_checks.md) §8; `ADDENDUM_2_N6.md`,
`ADDENDUM_3_N6C.md`, `explore_n6.py` in `src/test/20261108_new_method_quick_checks/`.*

## 11. Synthesis: one frontier, one wall

### 11.1 All scorers on one plot

![Frontier](../assets/2026-10-04_aspect_conditioned_similarity_methods/frontier.png)

*Figure 6. Every scorer of the stage on the seed-42 development episodes: condition gain (y) against either rate (x). Dotted
lines are constant R@1 (cosine 12.96, E3's control 16.55, the best condition-free score 18.34). Hollow markers: label diagnostics (seed 42) and the MLLM (seed 46). See `FIGURES.md` in the assets folder for the full key.*

Figure 6 separates three groups. Condition-free scores (gain 0) moved right along the x axis, from cosine (either 25.9)
to the uniform factor term (33.1), the centered term (35.9) and the centered term plus averaged heads (36.7). Scorers that
read the condition moved up but also left: A3's rule alone at gain 1.0 and either 21.0, N6's reader at 4.4 and 28.6,
label probes at 13.3 and 36.7 (inferred) or 21.1 and 40.3 (told). Fusions sit within 0.25 R@1 of their own family's condition-free control (A′ −0.02, N6 +0.23, N6c +0.15), except N1, 1.15 below its matched control.

![Gain progression](../assets/2026-10-04_aspect_conditioned_similarity_methods/gain_progression.png)

*Figure 7. Condition gain of each reader scored alone (term only), with 95% intervals. Scored alone, except the raw baselines, which E1 fused with cosine; the C0 and SE bars are their agreement terms alone (seed 42). Hatched bars use evaluation labels and are diagnostics, not methods.*

### 11.2 Why every fusion lands on the wall

![Margins](../assets/2026-10-04_aspect_conditioned_similarity_methods/margins.png)

*Figure 8. R@1 (left) and condition gain (right) of each fused method minus its condition-free control, with 95%
intervals; where a declared and a matched control exist, both are shown. The controls of A′ and N6 are already matched (the same terms with the condition removed) and are drawn as matched squares.*

A fused score beats its matched control only if it adds more condition gain than it loses either rate. The measured
trades were close to even (Figure 8): N6c added 1.19 gain and lost 0.88 either (net +0.15 R@1); N6 nested added 1.69 and
lost 1.23 (net +0.23); A′ on A3 added nothing and lost nothing (gain −0.01, R@1 −0.02): one tuning half picked the control itself and the other put a small weight (0.25 against 4) on the conditioned term. Our reading, supported only by the exploratory analysis of Section 14: part of the loss comes from wrong aspect picks and from a representation that is weak in one modality (emotion from images, style from captions), which promote negatives; but reading costs either rate even when the pick is right (D0 below; Section 14). D0's condition-free counterpart (the mean of the three label-probe scores) has an either rate of 45.08 (R@1 22.54); Told keeps 40.27 and Inferred-hard 36.70, so reading the condition costs either rate even with label posteriors (−4.81 [−5.36, −4.26] for Told and −8.38 [−8.93, −7.85] for Inferred-hard), but the gain outruns the cost (Inferred-hard beats the counterpart by 2.48 [2.11, 2.84] R@1, Told by 8.12 [7.75, 8.50]). In every reader we measured, reading the condition cost either rate, and only a large enough gain outpaced that cost.

![Per pair](../assets/2026-10-04_aspect_conditioned_similarity_methods/per_pair.png)

*Figure 9. Condition gain by aspect pair for the main readers (seed 42).*

Where reading works is consistent across methods (Figure 9): gains were smallest on emotion × style for every reader and largest on emotion × genre for six of the seven (A3's agreement term peaked on style × genre, 1.57 against 1.35), matching D0's pick accuracy (77% against 38%).

### 11.3 The condition-free bar and its caveat

The strongest condition-free score rose from 16.55 (E3's uniform control) to 17.94 (centered factor term) and 18.34 (plus
averaged heads), against 13.38 for RCA, and the first replicated on the fresh seeds (17.75 pooled). These scores use the
examples but not the condition: both conditions share the same 8 example items. A probe by the final reviewer (seed 42,
decides nothing) found that centering on 8 random items instead gives 16.85 and on the global mean 17.14, against 17.95 on
the episode's own examples. Because the examples are value-disjoint from the query by construction, part of the lift may
come from the benchmark's construction rather than from the representation; this needs its own control before it is a
finding.

One mechanism is consistent with the probe; it has not been tested. Every example item comes from a painting without
the query's value of A or of B (Section 2.1), so the 8 items are a sample of the same population as the 11 negatives,
while p_A and p_B each share one of the query's values. Centering the query on them subtracts roughly what a typical
negative looks like and lifts p_A and p_B together: the either rate rises and the gain stays 0. For emotion, exclusion
is per painting while membership is per row (Section 2.2), which makes the example set more unlike the query than a
rule on rows would.

The caveat bears on B's absolute lift over cosine and not on the margins of Sections 9 to 14. A configuration and its
matched counterpart center the same way, so the effect, if it is an artefact, cancels in their difference. B is also a
control built from our own ingredients, not a published baseline. For a new configuration the GO comparators are
cosine, RCA, B (extended to B′ by any new condition-free ingredient) and the configuration's matched counterpart, which
can sit above B: the told partition's counterpart adds 0.35 [0.16, 0.55] to B (Section 14). B itself has not yet been
computed on fresh seeds; the 17.75 above is cosine with the centered factor term, without the heads. The split matters
for the paper's story. A method that cleared the development bar of +0.5 R@1 over max(B, its counterpart) would reach
about 18.8 against 12.96 for cosine, and about 5.4 of those points would come from condition-free ingredients, about
0.5 from reading the condition. The first v2 result showed the same pattern on value episodes (Section 3).

*Sources: final-review probe in the [quick-checks report](../auto/v2/2026-11-08_new_method_quick_checks.md); handoff
`docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md` §4 to §6;
`src/test/20261109_fix_diagnostics/results/diagnose_counterparts.txt` §2; `src/eval/aspect_episodes.py`.*

## 12. Process: what the reviews caught

- **An automated multi-persona LLM review of the plan** (ARS: five seats of one model family, no human reviewer; Major Revision) made condition gain co-primary, added the
  condition-free control, the clustered bootstrap, the held-out aspect test and the MLLM baseline. Without it, the
  condition-blind uniform term (16.30 R@1, Section 4) would have passed the original GO rule.
- **An automated two-seat methodology review of the repair order (ARS)** replaced a pick criterion that rewarded trading R@1 for gain.
- **Whole-branch final reviews** re-derived every number from the stored arrays with independent code. They found the
  confounded N1 control (Section 9.2), several overclaims in reports and two false sentences; all were corrected before
  the results were reported.
- **Disclosed exceptions to pre-registration:** Addendum 1 was committed two minutes after the N1 test had been computed but before its outputs were read; N6c's comparison with its declared control (cosine plus A3's centered term, R@1 17.94; +0.55) was already known from an exploratory run. Addendum 2 §4 said no further method would be built overnight after N6 failed; the controller overrode that line for N6c, adopted the matched control, and took the "build N6" step that the committed decision rule had not named (rulings in the quick-checks log).
- **Seeds:** development on seed 42; seed 43 spent on E3's test; 44 and 46 on MLLM probes; 45, 47 and 48 on N1's test.
  Fresh seeds start at 49.

## 13. Limitations

- One dataset (ArtELingo, English), one backbone (CLIP ViT-B/32) for all methods, three aspects. CUB, SemArt and GeneCIS
  were prepared but not run.
- Every method used one model seed; picks were made on one development draw (seed 42), and development seed 42 was
  reused across many analyses.
- Fresh seeds are fresh draws of episodes on the same 6,451 selection paintings; held rows were not read.
- Intervals cluster anchor paintings only; candidate and example paintings recur across episodes.
- The label-free partitions are shaped like the evaluation aspects (Table 4), and the affect partition uses an emotion
  classifier trained on external labels.
- The overnight follow-ups (N6, N6c) were chosen after seeing development results; their seed-42 numbers are
  exploratory, and N6c's gate failed before any test.
- The example pairs are not controlled on the third aspect (only the candidates are), and how often they share it has
  not been counted (Section 2.1).
- Emotion is labelled per viewer, style and genre per painting. The episode builder's emotion rules combine membership
  by row with exclusion by painting (Section 2.2), and how mixed the paintings' emotions are has not been measured.
- The condition selects among the three aspects the benchmark contains, and within an episode the gain measures a
  choice between two of them.

## 14. What a passing method has to do, and a first diagnosis

The numbers point to one requirement: add selection without losing aspect-finding. A method must add more condition gain than it loses either rate against the strongest condition-free score B (R@1 18.34, either 36.68): the told partition does (gain 4.56 for an either loss of 1.57); N6's reader does not by a reliable margin (0.79 for 0.41).

To find which part blocks this, we ran exploratory diagnostics on seed 42 after the stage (`src/test/20261109_fix_diagnostics/`; they decide nothing). Each variant was added on top of B (N6c's matched control: cosine, A3's centered uniform term and the averaged heads, max-R@1 cross-fit; R@1 18.34, either 36.68) with A′'s min-margin cross-fit against B. Each reader also got a condition-free counterpart T_cf, the mean of its two condition scores, fused on B with a max-R@1 cross-fit (the condition-free counterpart of the min-margin rule); the difference between the fused reader and its counterpart is the margin from reading the condition (the counterpart's gain is 0, so the margin's gain equals the fused gain).

*Table 10. Exploratory, seed 42: what N6's reader needs to clear the condition-free bar. "Fused" is the variant fused on B; "counterpart" is T_cf fused on B; all columns in pp.*

| Variant added on top of B | Fused minus B: R@1 | gain | either | Counterpart minus B: R@1 | either | Fused minus counterpart: R@1 | either |
|---|---|---|---|---|---|---|---|
| N6 reader (trained without ArtELingo labels, hard pick) | +0.19 [0.01, 0.38] | 0.79 [0.52, 1.05] | −0.41 [−0.70, −0.12] | +0.05 [−0.05, 0.15] | +0.10 [−0.09, 0.29] | **+0.14 [−0.04, 0.32]** | −0.51 [−0.76, −0.25] |
| contrastive reader (picked partition minus contrast-like partition) | +0.07 [−0.08, 0.22] | 0.53 | −0.39 | not computed | not computed | not computed | not computed |
| same heads, partition told (emotion to affect, style and genre to image) | +1.50 [1.22, 1.79] | 4.56 [4.21, 4.93] | −1.57 [−2.03, −1.08] | +0.35 [0.16, 0.55] | +0.70 [0.32, 1.09] | **+1.14 [0.90, 1.41]** | −2.27 [−2.66, −1.88] |
| label-probe reference (diagnostic, not a bound) | +12.28 [11.81, 12.75] | 20.55 [19.99, 21.09] | +4.00 [3.35, 4.68] | +5.62 [5.27, 5.97] | +11.24 [10.53, 11.95] | +6.66 [6.31, 7.00] | −7.23 [−7.70, −6.76] |

The heads are good enough; the reader's choice of partition is not. Told the right partition, the same heads trained without ArtELingo labels clear B by 1.50 R@1 against +0.19 for N6's reader fused the same way. Part of that comes from the condition-free side: the told term with the condition removed already adds 0.35 [0.16, 0.55] on its own. The margin from reading the condition is therefore +1.14 [0.90, 1.41] R@1 for the told partition against +0.14 [−0.04, 0.32] for N6's label-free reader. The told term pays 2.27 either for 4.56 gain, so its gain outruns its either cost; it is not free. The label-free reader picks the told partition in 52% of rankings and is right under both conditions in only 28% of episodes. In those episodes the fusion gains +1.16 [0.80, 1.54] R@1 and 2.21 gain with no detectable either loss (+0.12 [−0.43, 0.66]); in the others it loses 0.61 [0.29, 0.94] either. The wrong picks are concentrated where the partitions are ambiguous: under the style condition of style × genre the reader picks the image partition in only 17% of rankings, and even the told mapping cannot separate style from genre, because both live in the image clusters (Table 4). A contrastive reader and a top-k cascade with N6's reader did not help (cascades lost 0.55 to 2.59 R@1).

Read as oracles, the three rows of Table 10 that have a matched counterpart form layers. With nothing perfect, N6's
reader adds +0.14 R@1 over its counterpart; a perfect reader on the current label-free heads (the told partition) adds
+1.14; a perfect reader on a representation in the evaluation taxonomy (label probes told the aspect) adds +6.66. The
told row is a reader oracle for these three partitions only: its mapping was chosen with the labels (from AMI), and its
gain on style × genre is exactly 0. What an oracle picks is a block, one partition's 64-class posterior, and not a
single factor, because value-disjoint episodes need every value of an aspect to live on shared factors (spec §6). Method
A's 32 learned factors carry no aspect identity that could be told, so no reader oracle exists for them; L3 (Section 6)
is a different diagnostic, a better representation read by the same rule. We score oracles as R@1 margins over the
matched counterpart and not by gain: the told partition minus the other partition reaches a gain of 7.03 on its own
while its R@1 falls to 11.31.

Caveats on this analysis:

- For each partition h, the reader's statistic Δ_h (mean within-pair agreement over the support pairs minus over the contrast pairs) under condition b is exactly the negative of its value under condition a, and the told mapping sends style and genre to the image partition, so no style × genre episode can have both picks right. The 28% both-correct episodes come from emotion × style and emotion × genre only (1,576 and 1,883 of 3,459 episodes, none from style × genre), and the subset is selected after the fact.
- On those same 3,459 episodes the told-partition fusion, whose picks are right everywhere, lost 3.01 [2.09, 3.93] either, so the small either change of N6's reader there holds only at the small weight the cross-fit gave it.
- B's cross-fit picks were tuned on the same parity halves the fusion reuses (a small second-order leak), and seed 42 was reused for every analysis.
- N6's reader on B nominally clears B (+0.19 [0.01, 0.38]) in this exploratory look, but the lower bound sits at 0 under another bootstrap stream; it is not a pass.
- The pick accuracy of any reader that takes the argmax of Δ_h and uses the told mapping is capped at 83.3% pooled (100% on emotion × style and emotion × genre, 50% on style × genre, because of the antisymmetry), and both picks can be right in at most 66.7% of episodes; the label-free reader reached 52.4% pick accuracy and 28.1% both-correct.

The handoff (`docs/superpowers/handoffs/2026-10-04-fix-the-reader-handoff.md`) turns this into a plan: a reader that
chooses the aspect correctly (calibrated on label-free pseudo-aspect episodes), a partition that separates style from
genre, a matched control for each, and a test on fresh seeds 49 to 51.

## Appendix A. Glossary

| Term | Meaning |
|---|---|
| aspect episode | query, 4 support pairs, 4 contrast pairs, 13 candidates (Section 2.1); scored under both conditions and in both directions, four rankings |
| row | one image of a painting with one viewer's caption and emotion label; about five per painting |
| anchor | the row whose image (i2t) or caption (t2i) is the query |
| condition gain | R@1 minus other-aspect rate, counted on first places only; 0 for any condition-free scorer |
| swap success | p_A above p_B under condition A and p_B above p_A under B (spec §5.1); a different metric from condition gain |
| either rate | R@1 plus other-aspect rate; R@1 = (either + gain) / 2 |
| codes | 32 non-negative factors per item from method A's encoders |
| agreement rule | weights = ReLU(support agreement − contrast agreement) per factor |
| nested score | z(cos) + λ_u·z(condition-free term) + λ_a·z(conditioned term), min-margin cross-fit |
| matched control | the same score with only the condition removed |
| D0 | label-probe diagnostic: Told vs Inferred aspect |
| N1, N2, N6, N6c | centered rule; find-then-select cascade; partition-head reader; N6 on the centered base |
| GO bar | RCA, the best raw baseline on seed 42 |
| B | best condition-free score (cosine, A3's centered uniform term and the averaged heads; R@1 18.34, either 36.68) |
| C0, SE | earlier factor models: C0 = factor recipe without condition training; SE = C0 plus value-condition episodes from GoEmotions affect clusters |
| R3 | the repaired factor recipe of 29 September (InfoNCE pair agreement plus decorrelation, graph term kept); C0 is R3 refit on scorer-train rows |
| buddy graph | links two items when each is among the other's nearest neighbours; the basis of the buddy and percept lines (Section 3) |
| Block 1, Candidate A | the first v2 design (28 September): the buddy Stage 1 rebuilt with a content graph; shared factors with a condition interface (Section 3) |
| RCA | relevant component analysis, a metric learned from pairs |
| KISSME | a metric learned from similar and dissimilar pairs by comparing their covariances |
| z(·) | per-episode z-score over the 13 candidates |
| ARS | automated research and review pipeline of LLM personas (no human reviewer) |
| E0 to E3 | experiment numbers of the plan (E0 setup, E1 baselines, E2 partitions, E3 method A and its GO test) |
| L3 | factors trained on the evaluation labels (diagnostic) |
| T_cf | condition-free counterpart of a reader: the mean of its two condition scores (Section 14) |

## Appendix B. Reproducibility

- Code: `src/eval/aspect_episodes.py`, `aspect_metrics.py`, `aspect_scorers.py`, `aspect_nested.py`,
  `aspect_quick_checks.py`; runners under `src/test/20261030_aspect_baselines/`, `20261101_aspect_factor_gonogo/`,
  `20261105_method_repair_diagnostics/`, `20261106_mllm_probe_8b/`, `20261108_new_method_quick_checks/`, `20261109_fix_diagnostics/` (`diagnose_fixes.py`, `diagnose_counterparts.py`).
- Results: per-anchor arrays and JSON summaries in each folder's `results/` (gitignored); figures of this report are
  built by `docs/reports/assets/2026-10-04_aspect_conditioned_similarity_methods/prep_figure_data.py` and
  `make_figures.py` from those files. The fix diagnostics of Section 14 stored summaries only (no per-anchor arrays).
- Decision records: `DECISION_RULE.md`, `TEST_CONFIG.md`, `ADDENDUM_1.md` to `ADDENDUM_3_N6C.md` and the
  `PREREGISTRATION.md` files of E3, A′ and the 8B probe; episode ledger `docs/superpowers/episode_seed_ledger.md`.
