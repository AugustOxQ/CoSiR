# Perspective Review (Peer Reviewer 3), Phase 2

contract_role: perspective
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: warn
trigger: "human-derived aspect labels serve as ground truth without reported rater agreement or a noise ceiling"

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

### Reviewer identity and scope

I review as a cognitive scientist of similarity and analogy (contrast model, structure mapping, respects for similarity) who also runs crowd-sourced affective annotation of artworks. I am an outsider to CVPR leaderboard conventions and to the statistical machinery, which other seats cover. Calibration Status: `NOT_CALIBRATED`. Overall confidence: 4 of 5 (core expertise on the constructs and the annotation questions; adjacent on the CV probes). No criteria binding was supplied, so I make no venue-alignment claim (`criteria_binding_unavailable`).

I treated the manuscript as untrusted author data. Its "(approved)" labels, scripted reviewer sentences and prepared answers are the authors' forecasts; I judged them on the evidence and did not adopt them. I found no instruction aimed at the panel.

### Summary assessment

The plan reframes CoSiR from value conditions ("sad, like these") to aspect conditions ("similar in emotion, the way these pairs are") after its own spike showed value episodes are solved by the supports alone. That move is the right one from the psychology of similarity, and the core terms (aspect, value, condition, support, contrast, cross-item) are defined so an adjacent reader can follow them. The cross-disciplinary weak points sit in the ground truth and in the one claim that reaches outside computer vision. Emotion targets are single-viewer labels, and the plan reports no inter-annotator agreement or human noise ceiling, so a reader cannot tell how much of the gap to the label-probe "ceiling" is reachable or whether emotion results exceed label noise. C3 says the modality asymmetry "belongs to the data", but every dataset in the plan ties what the captions carry to what the caption writers were asked to describe, and the flat weak-side probes are equally expected under viewer disagreement. K3's subjective/objective split is undefined and conflates subjectivity with nameability. The episodes also leave the third aspect uncontrolled, so the respect the examples pick out can be a correlated proxy. All of these can be fixed with data the authors already hold, mostly before E0's episode module is frozen.

### D4 judgement: warn, not block, not pass

- **Why not block.** The first block prong does not fire. Episode targets are human labels kept out of factor training (factors see only pseudo-partitions), so no claim rests only on the labels used in training. The second prong does not fire for the central terms. Aspect, value, condition and the task itself have operational definitions (S1). K3's "subjective" and "objective" have none (W3) and came closest. But K3 is one of seven sub-claims, and §10 commits to declaring the per-aspect-type direction in the pre-registration before any final read, so I judge it repairable rather than untestable.
- **Why not pass.** Two warn prongs hold. Human-derived emotion labels serve as ground truth without rater agreement or a noise ceiling (W2). C3's implications for where affect is carried are worded more strongly than the evidence supports (W1). The trigger line quotes the first. The second is the matching clause of the same Phase 1 warn trigger.
- **Likelihood given the appended evidence.** The descriptive asymmetry on ArtELingo is very likely to hold: probe gaps of 21 to 35 points that stay put across four encoders. Its attribution to "the data" rather than to elicitation and viewer disagreement is not yet supported. For emotion, K3 currently points the other way: privileged names gained +3.2 R@1 over CLIP while SE with the agreement rule gained +0.2. The method is not trained yet, so this is weak evidence.

### Criterion-Bound Judgements

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| D4 cross_disciplinary_relevance | contract D4 description and my Phase 1 scoring plan; no bound venue criteria | PARTLY_MEETS | see S1 to S3 and W1 to W8 (one anchor each) | definitions are adequate; ground-truth validity, the C3 wording and the K3 construct fall short of what an affective-computing or cognitive-science reader would accept | pre-results plan; I may underweight CV benchmark norms | yes: D4 warn |

### S1: Aspect and value are defined so an adjacent reader can map them onto known constructs

The glossary, §3 and the worked example in §2.4 define aspect, value, condition, support, contrast and cross-item, and they show the swap. A cognitive scientist can read "aspect" as a respect for similarity and "value" as a feature value without translation. The change from value episodes to aspect episodes came from the authors' own finding that the supports alone solve value episodes. That is the same distinction the similarity literature draws between sharing a value and sharing a respect.
**Evidence Anchor**: text: Appendix A glossary "an aspect is a respect in which items can be similar"

### S2: The swap test is read next to correctness

Result 2 of the aspect spike shows that swap success alone is uninformative: antisymmetric scorers reach about 50% while R@1 sits at chance. The plan therefore reports swap only beside R@1. In human terms, a shift of respect counts only if the judgment both flips and is right, and the plan encodes exactly that.
**Evidence Anchor**: text: aspect-episode spike Result 2 "a high swap with a CLIP-level R@1 means a scorer that moves with the condition without being correct"

### S3: C3 comes with planned controls, and the label-noise caveat sits next to the evidence

Four encoders rule out the encoder explanation, CUB primary colour gives a measured symmetric case, and genre is kept as a within-dataset aspect that the authors expect to be visible in both modalities. The aspect spike also states openly that image emotion labels are inherited from rows and are noisy.
**Evidence Anchor**: text: §5.3 "ArtELingo aspect visible in both modalities, which C3 needs as its within-dataset control."

### W1: C3's "belongs to the data" cannot separate modality from elicitation protocol and viewer disagreement

**Problem**: ArtEmis-style captions are written by an annotator to explain the emotion that same annotator just chose, so the link between caption and emotion is built into the collection protocol. The annotators were never asked about style, so captions seldom carry it. Reed's CUB captions were elicited to describe visible parts, so colour symmetry is also what the protocol predicts. Flat weak-side probes across four encoders do rule out the encoder explanation, which is what the backbone check was built to do. They are equally expected under two other explanations. (a) What a text carries follows what its writers were asked to describe. (b) Image-level emotion is one viewer's label, so an image-only predictor is capped by how often viewers of one painting agree. The weak-side emotion probe sits only 3 to 5 points above the 31.8% majority class, which is the pattern a disagreement cap would produce. The backbone report's sentence "a better encoder cannot read what the pixels or words do not hold" claims more than this design can show.
**Evidence Anchor**: text: §4 C3 and backbone check Verdict "the effect persists across four backbones, so it belongs" and "so the modality asymmetry is a property of the data"
**Why it matters**: C3 is one of the three contributions and one of the three pillars of the NO-GO fallback (K1, K3, C3). Affective-computing and cognitive-science readers will take "belongs to the data" as a claim about where affect is carried, in pictures or in language. A reviewer who knows how ArtEmis was collected can dismiss it in one sentence.
**Suggestion**: The minimum fix is to narrow the wording, for example: "in datasets whose captions explain a self-reported emotion, emotion is carried by the text; style, which annotators were not asked to describe, is carried by the image." A stronger fix uses data already in the plan to separate the explanations. SemArt's catalogue texts were written by curators to describe paintings. If SemArt text carries type or school well while ArtEmis captions carry style poorly, the asymmetry follows how the text was elicited, not the modality. Also report the image-side emotion probe as a fraction of the viewer-agreement cap (W2). Measure genre's image and caption probes before using genre as the symmetric control (S3).
**Severity**: Major
**Confidence**: 4 (core expertise: affective annotation protocols; adjacent: the CV probes)

### W2: Emotion targets have no reported inter-annotator agreement or human noise ceiling

**Problem**: Each ArtELingo row is one viewer's emotion, and a painting has about five rows on average (308,723 rows over 61,402 paintings). Viewers of one painting often disagree. The episodes take an item's emotion from one row, so "shares the anchor's emotion" and "shares neither" are decided by that row. A "negative" painting may carry the anchor's emotion in other viewers' rows, and an image anchor's target depends on which viewer's row was drawn. The plan calls the label-probe score (about 23) the ceiling and sets Strong GO at a third of the way to it. It never reports the human reference: how often a second viewer of the same painting gives the same label. The noise is acknowledged in a caveat but never quantified.
**Evidence Anchor**: absence: §2.3, §5.1, §5.2 and §10 — expected an inter-annotator agreement statistic and a human noise ceiling for the emotion labels that define episode targets; checked §2.3 to §5.3, §8, §10 to §12, Appendix A, and the caveats of the support-baseline spike, aspect-episode spike and backbone check
**Why it matters**: Without this reference a reader cannot tell several things. How much of the gap between CLIP (about 10 on emotion) and the probe ceiling (about 24) is reachable at all. Whether emotion differences between methods exceed label noise. Whether "emotion is weak in images" is a model limit or a disagreement limit. Emotion is the aspect K3 and C3 lean on.
**Suggestion**: All of the following come from existing rows at negligible compute. (a) Per-painting agreement: the modal-label share, plus Fleiss' kappa or Krippendorff's alpha over the 8 emotions. (b) A leave-one-viewer-out reference: the R@1 an oracle reaches when it scores candidates by another viewer's label for the same painting. (c) The share of negative images whose painting has any row with the anchor's emotion. (d) A sensitivity read on high-agreement paintings (for example modal share of at least 0.6), declared in the pre-registration. Optionally, (e) a valence-level variant. Affect research treats emotion similarity as graded (contentment sits nearer awe than disgust), while exact-category matching scores near misses as plain negatives.
**Severity**: Major
**Confidence**: 5 (core expertise: annotator disagreement in emotion labels)

### W3: K3's subjective/objective split is undefined, and nameability, not subjectivity, predicts when examples beat names

**Problem**: K3 predicts that examples beat names on subjective aspects and match them on objective ones. §10 defers the per-aspect-type direction to the pre-registration. Nowhere does the plan say which of emotion, style, genre, colour, bill shape, type, school and timeframe count as subjective, or by what criterion. The risk table's mitigation (narrow K3 to where names fail) would make "subjective" mean "where names fail", which makes K3 true by definition. Two constructs are at stake, and here they come apart. Subjectivity means raters disagree, which is a property of the label. Nameability, or codability, means how readily a respect can be put into words that a text encoder handles. Emotion is highly subjective but highly nameable: eight named categories, and captions that are emotion explanations. Art style is fixed by curators but poorly nameable for a text encoder. The appended evidence already shows the split. Privileged names lifted emotion (13.31 against 10.14) but not style (11.96 against 12.11), although zero-shot emotion prompts sit at the majority-class rate.
**Evidence Anchor**: text: §4 K3 and §12 R-names "Examples beat naming the aspect on subjective aspects and match it on objective ones" and "narrow K3 to where names fail (subjective aspects)"
**Why it matters**: K3 carries the "why examples rather than names" argument, which is the motivation for the task (C1) for any reader outside the project. As worded, it cannot be refuted. If emotion is put in the subjective group, the current evidence points against it.
**Suggestion**: Fix, before E10, a measured criterion that does not depend on the outcome. For subjectivity: inter-annotator agreement from W2 on ArtELingo, and CUB's per-attribute certainty ratings as a proxy; SemArt catalogue fields are single-source, so state that they cannot be graded this way. Measure nameability separately as zero-shot name-prompt accuracy on the strong modality, which the spike already computes. State K3 as a prediction over both axes and pre-register each aspect's assignment. If one axis must be chosen, nameability is the one with a mechanism behind it.
**Severity**: Major
**Confidence**: 4 (core expertise: similarity and naming in cognitive psychology)

### W4: Each episode contrasts only two aspects and leaves the third uncontrolled, so the respect the examples pick out can be a correlated proxy

**Problem**: In an episode the supports share aspect A and the contrasts share aspect B. The third aspect is not controlled for the supports, for p_A or for the negatives. Correlated aspects are common in these datasets: style with genre in WikiArt, school with timeframe in SemArt (the literature appendix notes this), and colour and bill shape with species in CUB. Four Baroque support pairs may also share a genre and a palette, and the contrast set rules out only aspect B. In Tversky's terms, the contrast set fixes which features are diagnostic. The model therefore learns whatever separates S from C, which need not be aspect A. §5.3's claim that three aspects answer the binary-switch objection does not hold at the episode level, because every episode still chooses between two labelled aspects.
**Evidence Anchor**: text: §5.3 and §5.1 "answer the objection that two aspects make the condition a binary switch" and "condition A uses S = P_A, C = P_B; condition B swaps them"
**Why it matters**: Adjacent fields will read K6 (the rule selects aspect factors) and the swap test as evidence that a model infers a respect of similarity. If a correlated respect travels with the target, success can come from the proxy, and the demonstrated respect is underdetermined. This is Goodman's point that any two things share some respect. It is cheap to fix now, before the E0 module is frozen.
**Suggestion**: (a) Report the association between aspect labels on each dataset (Cramér's V or adjusted mutual information), and between each pseudo-partition and each label. (b) When sampling supports for A, require them to vary on every other labelled aspect, not only B, or report the third-aspect agreement rate inside S and for p_A. (c) Add a variant whose contrast set mixes B and C pairs, so the condition is no longer a binary switch. (d) Report R@1 per aspect pair, so a highly associated pair does not carry the mean.
**Severity**: Major
**Confidence**: 3 (core expertise on the construct; the size of the effect in these datasets is untested)

### W5: The framing is under-connected to the psychology of similarity, which already offers the "why it works" account the authors want

**Problem**: The task restates textbook constructs. "Similar in respect X" is Medin, Goldstone and Gentner's respects for similarity. The dependence of feature weights on the contrast set is Tversky's diagnosticity. Goodman argued that similarity without a respect is empty. The plan cites Tversky 1977 only as background in the novelty appendix and uses none of these constructs in §3 or in explaining the rule. The agreement rule (co-activation in S minus co-activation in C) is a diagnosticity estimator.
**Evidence Anchor**: text: novelty check §4 must-cite list "Tversky 1977 (similarity depends on the comparison context)"
**Why it matters**: The user's fourth requirement is an explanation of why the method works. Diagnosticity gives one that can be falsified. The rule should degrade when S and C differ on more than one respect (W4), and its weights should track how well each factor separates S from C. The same paragraph would let cognitive-science and IR readers place the task at once.
**Suggestion**: Add one paragraph to §3 mapping support and contrast to diagnosticity, and the task to respects for similarity. Add one E11 ablation that varies how many respects S and C differ on, as a prediction of that account.
**Severity**: Minor
**Confidence**: 4 (core expertise: similarity theory)

### W6: No user scenario is given, and the evaluation rules make the condition harder to supply than to name

**Problem**: §1 frames the task as a user showing in which respect two things are similar. The rules then require four cross-item image–caption pairs that agree on the aspect at values other than the query's, plus four contrast pairs that agree on a different aspect. The query's value may never be shown. These rules block the K1 shortcut, which is sound for evaluation. But a person able to assemble such pairs already knows the respect and could type it. The plan never says who holds such pairs (cross-item pairs need a captioned corpus) or why that person cannot name the respect.
**Evidence Anchor**: text: §1 and §3 "A user shows, by a few example image–caption pairs," and "The condition never shows the query's value and never names the aspect."
**Why it matters**: Practical significance and the K3 storyline both need a setting where naming fails and examples are at hand. Without one, IR and HCI readers will see a constructed benchmark rather than a user need.
**Suggestion**: Name a concrete setting, for example curating a collection by an unnamed stylistic or affective quality from a seed set drawn from that captioned collection. Report robustness to realistic conditions in the supplement: 1 or 2 supports, no contrast set, supports that include the query's value, noisy pairs. Tie the setting to the relevance-feedback line the plan already cites, where examples come from a user's judgments on retrieved items.
**Severity**: Minor
**Confidence**: 3 (adjacent field: human factors of retrieval)

### W7: "No labels" understates how much the emotion teacher shares the evaluation taxonomy

**Problem**: The emotion-like pseudo-partition is k-means over the 28 output probabilities of a GoEmotions classifier. GoEmotions names 6 of the 8 evaluation emotions (amusement, excitement, anger, disgust, fear, sadness; not awe or contentment), as the literature appendix also notes. The pseudo-partition is therefore a distilled emotion classifier, trained on Reddit text, that uses mostly the evaluation's own categories. "No labels" in §5.1 and "no labels from the evaluation taxonomy" in C2 are true at the dataset level, but affect researchers will read the setup as weak supervision.
**Evidence Anchor**: text: §5.1 Splits "Factor training uses image–caption pairs and pseudo-partitions only, no labels."
**Why it matters**: Readers from affective computing will discount emotion gains as transfer from the teacher. The teacher-only baseline has been moved out of the tables (§8), so the main results do not show the difference.
**Suggestion**: State the category overlap in the method section. Report emotion results separately for the six categories GoEmotions names and the two it does not, which is a free sensitivity check on the distant-supervision path. Keep the teacher-only text-side analysis next to the main emotion row.
**Severity**: Minor
**Confidence**: 4 (core expertise: emotion taxonomies)

### W8: Emotion is framed as a property of the painting, and the cultural scope of the English subset is not stated

**Problem**: The plan speaks of "a fearful image" and of the emotion a painting conveys. The label is one viewer's reported response. ArtELingo is multilingual by design, while the plan uses only its English part and never states the annotator population.
**Evidence Anchor**: text: §2.4 and §2.3 "each a fearful image matched with a fearful caption of" and "English part"
**Why it matters**: Statements such as "emotion lives in captions" will be read as general claims about viewers and media. Affective-science and art-history readers expect them scoped to the annotators who produced the labels. The scope also points to an unused test: the same paintings annotated by other viewers in other languages.
**Suggestion**: Say "the emotion annotators reported". State the English-only scope and the annotator population in the datasets paragraph. If ArtELingo's non-English annotations of the same paintings are available, run the C3 probes on them as a supplementary cross-cultural check (same images, different viewers and language).
**Severity**: Minor
**Confidence**: 4 (core expertise: cross-cultural affective annotation)

### Assumption audit

- **Explicit:** examples carry a respect better than names on subjective aspects (K3, see W3); the modality asymmetry is a property of the data (C3, see W1); clusters of a teacher's outputs stand in for an unknown respect (R-pseudo, flagged by the authors).
- **Implicit:** two items that share a label are similar in that respect, all or none, which is the graded-similarity issue in W2(e); one viewer's emotion is a property of the image (W2, W8); the person issuing a query can produce cross-item pairs but cannot name the respect (W6); one contrast aspect is enough to fix the respect (W4).
- **Paradigmatic:** ground truth comes from fixed label taxonomies, not human similarity judgments. That is standard for a CVPR benchmark, and I do not ask for a human study inside this six-week plan. The concurrent TPIPS line (human-voted, aspect-conditioned similarity, cited in the literature appendix) shows the alternative exists. A later paper could check on a small sample whether "shares the label" tracks what people call "similar in that respect".

### Cross-disciplinary reading recommendations

- Medin, Goldstone and Gentner (1993), "Respects for similarity", Psychological Review 100(2). The construct behind "aspect" (W5).
- Goodman (1972), "Seven strictures on similarity", in Problems and Projects. Why a respect must be fixed (W4, W5).
- Tversky (1977), already cited. Use the diagnosticity hypothesis, not only the general context claim (W5).
- Brown and Lenneberg (1954), "A study in language and cognition", Journal of Abnormal and Social Psychology. Codability as a separate axis from subjectivity (W3).
- Plank (2022), "The 'Problem' of Human Label Variation", EMNLP; Aroyo and Welty (2015), "Truth is a lie: Crowd truth and the seven myths of human annotation", AI Magazine. Disagreement as signal, and how to report it (W2).
- Peterson, Battleday, Griffiths and Russakovsky (2019), "Human uncertainty makes classification more robust", ICCV. Soft labels from many annotators (W2(d)).
- Russell (1980), "A circumplex model of affect", Journal of Personality and Social Psychology. Graded emotion similarity (W2(e)).
- Search lead [UNVERIFIED]: Mohammad and Kiritchenko (2018), "WikiArt Emotions", LREC. I believe its annotations were gathered under separate image-only and title-only conditions, which would test C3 directly; check before relying on it.
- Search lead [UNVERIFIED]: the annotator-agreement statistics reported in the ArtEmis paper (Achlioptas et al. 2021, already cited) for the share of artworks with a majority emotion.

### Questions for authors

1. Which aspects are subjective and which objective, and by what criterion fixed before E10? How does that criterion differ from "where names fail"?
2. What share of ArtELingo paintings have a majority emotion among their annotations, and what R@1 does a leave-one-viewer-out oracle reach on your aspect episodes?
3. Inside an episode, how often do the four supports for aspect A also share a value of the third aspect, and does R@1 differ between aspect pairs with high and low label association?
4. Which planned result would let you tell modality apart from elicitation for C3? If none, would you narrow C3 to the collection protocol?
5. Who, concretely, holds cross-item image–caption pairs that agree on a respect they cannot name?
