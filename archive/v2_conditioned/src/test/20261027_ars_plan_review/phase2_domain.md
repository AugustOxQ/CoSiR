contract_role: domain
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: warn
trigger: "a claim worded more strongly than the appended evidence supports while remaining testable as planned"
Basis: prior work is represented accurately in almost every row I could check, and the novelty statement is scoped with care. But four planned claims (K3, K5, K7 and C3) and the C2 supervision wording are stated more strongly than the plan's own tables and appended evidence support, and §8 drops example-conditioned baselines that the plan's own appendices rank as must-have. Each claim stays testable once repaired, and no finding meets a block trigger: no overlapping prior method is hidden, the named-instruction embedder is planned, and no headline claim lacks a revision path.

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: not_assessed

### D6: venue_fit_and_contribution
score: not_assessed

## Review Body

Calibration status: NOT_CALIBRATED. Venue binding: criteria_binding_unavailable, so this card makes no venue-alignment claim.

Summary. The manuscript is a pre-results plan for a CVPR paper on example-conditioned aspect similarity across modalities, with five appended evidence reports. I reviewed it as a researcher in conditional similarity and composed retrieval. For a plan, its domain scholarship is strong. It traces its rule and score back to Rocchio, CSN and KISSME, names Contextual Visual Similarity and MARS as the closest formulations, corrects its own misattribution of the PercepT teacher, and flags disputed GeneCIS reproductions. The prior-work rows I could check against my own knowledge (GeneCIS, CSN, SCE-Net, EmotionCLIP, the WikiArt genre ids, the ArtEmis emotion set, the GoEmotions overlap) are stated correctly. The weaknesses are mostly about claim calibration against the field. K5's GeneCIS bar uses the wrong column. The K7 basis ablation leaves out the dominant 2024 to 2026 family of sparse codes (sparse autoencoders). §8 drops the example-in-context alternatives that the plan's own novelty check lists. "No labels from the evaluation taxonomy" undersells how closely the ArtELingo pseudo-partitions match the evaluation aspects. K3's subjective versus objective direction has no definition and is not yet supported by the appended evidence. C3's backbone-invariance argument cannot tell modality content apart from how ArtEmis labels and captions were collected. All of these can be fixed before the October 23 freeze. I judged the scripted reviewer objections and the "(approved)" labels on the evidence, not as settled.

### S1: The novelty statement is scoped against the right lineages
The novelty check lists what must not be claimed, citing Xing, RCA, ITML and KISSME for pairs, Wang et al. 2016 and Category Traversal for test-time reweighting, and VISALOGY, VASR and MARS for relation by example. C1 is then written as a narrow conjunction under "to our knowledge". A domain reviewer would ask for exactly these lineages.
**Evidence Anchor**: text: novelty check §1 "We must not claim that inferring a notion of similarity from examples or pairs, test-time feature reweighting, or relation-by-example retrieval is new"

### S2: Self-correction of a prior-work attribution
The literature review checked PercepT's actual encoder (ModernBERT-base fine-tuned on GoEmotions, CLIP ViT-L/14) against the project's own label of the teacher. It corrected the label and asks for the teacher to be named exactly. This is the accuracy discipline a related-work section needs.
**Evidence Anchor**: text: literature review §2.4 "The teacher shares PercepT's label set but not its model"

### S3: The task redesign follows the field's lesson from GeneCIS on shortcuts
After the support spike showed that value episodes are solved without the query, the aspect episodes add an other-aspect hard distractor and a swap. That mirrors GeneCIS's gallery of reference-only and condition-only distractors. The plan also reads the swap metric correctly: it saw the antisymmetry artifact and requires swap success to be reported beside R@1.
**Evidence Anchor**: text: aspect spike Result 2 "a high swap with a CLIP-level R@1 means a scorer that moves with the condition without being correct"

### S4: The obvious "metric from pairs" objection is answered with baselines, not prose
Tier 1 runs diagonal and low-rank KISSME with shrinkage, RCA, a per-episode Xing fit and Wang et al.'s per-query weights on raw features. This is the right lineage for the rule and turns the most predictable reviewer line into a measured comparison.
**Evidence Anchor**: text: §8 Tier 1 metric-from-pairs row "your rule is few-shot KISSME"

### S5: GeneCIS numbers are sourced with unusual care
The review records five reporting formats, the OSrCIR reproduction dispute (17.4 against 14.0), the 0.2-point seed spread and two arithmetic errors in published averages. Few CIR papers handle the GeneCIS literature this carefully.
**Evidence Anchor**: text: literature review §2.2 "Two printed averages are arithmetically wrong"

### S6: Benchmark selection avoids label-templated captions
The benchmark survey excludes face and fashion sets whose captions are generated from the labels, and picks CUB with Reed captions, written without species names, as the symmetric control. For a cross-modal conditional task, that is the right domain judgement.
**Evidence Anchor**: text: literature review §2.7 "so the text side leaks the label and cannot test cross-modal conditional matching"

### W1: K5's GeneCIS bar quotes four-task averages for a focus-attribute-only evaluation
Problem. The plan evaluates only GeneCIS focus attribute, with focus object as a supplement. Yet the published bar it quotes is the average R@1 over all four GeneCIS tasks, including the two change tasks it never runs. The plan's own Table 2 gives the matching focus-attribute column for frozen ViT-B/32: SEARLE (CIReVL's run) 18.9, CIReVL 17.9, OSrCIR 19.4 and STiTch 18.4 average / 21.1 focus attribute. The training-free MLLM pipelines SQUARE (25.6) and DIOR (24.0, a frozen LVLM) are higher still. So the bar for "competitive" is 17.9 to 21.1 on the task actually run, not 14.4 to 17.4. The quoted set also leaves out STiTch, the strongest non-reranker frozen B/32 row in the plan's own list. Separately, §5.4 says the example protocol "is not comparable with published numbers". K5 is therefore decidable only through the text protocol, which is a stretch goal.
Why it matters. If left uncorrected, the only community-standard benchmark row in the paper would be judged against a bar 2.0 to 4.5 points too low, depending on the method. A GeneCIS-literate reviewer would catch this at once.
Suggestion. Restate K5 against the focus-attribute column, including STiTch, and report SQUARE and DIOR as out-of-class references. Either commit the text protocol as required work, or reword K5 to "example-protocol result next to our own image, text and image plus text baselines", with no published-row comparison.
**Evidence Anchor**: text: §5.4 item 2 "published frozen ViT-B/32 results: SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1"
**Severity**: Major
**Confidence**: 4 (core expertise: GeneCIS and zero-shot CIR reporting; numbers taken from the manuscript's own Table 2)

### W2: §8 drops the example-in-context alternatives that the plan's own appendices rank as must-have
Problem. The interface C1 defends is "examples, not names". §8 does test names on the same backbone (Qwen with the aspect in its instruction), which is good. But the strongest 2026 way to use the same examples without a learned basis is absent or demoted:
- An instruction embedder (Qwen3-VL-Embedding or GME) given the 4 plus 4 pairs in context is absent.
- Verbalise-then-name is absent. The novelty check says it "separates the value of examples from the value of naming".
- An in-context MLLM reranker appears only in the stretch row.
- The literature review's rank-2 Tip-Adapter cache and rank-3 linear probe on the eight pairs are also gone, although the probe was the strongest scorer on the earlier episodes (24.10 against SE 21.22).
Why it matters. Without these baselines, K2 can be won against a raw-feature, closed-form field of competitors. A CVPR reviewer can then answer C2's claim that the learned basis "is the prior that makes it possible to estimate a similarity from just four example pairs" with "a 2B embedder or an MLLM given the same pairs does this already". Norm evidence: the plan's own novelty check §5 Table 2 traces the in-context reranker to the VLM baselines of the Bongard and MARS lines, and its literature review §5 item 7 anticipates the "missing state-of-the-art comparisons" objection.
Suggestion. Promote verbalise-then-name and the in-context instruction embedder into Tier 2. Run the MLLM reranker on a fixed, pre-registered subsample. Restore the per-episode probe and the Tip-Adapter cache in Tier 1.
**Evidence Anchor**: absence: §8 baseline table and §10 pre-declared primary comparisons — expected a required, non-stretch baseline that gives an instruction embedder or open MLLM the same support and contrast pairs in context, and a verbalise-then-name baseline; checked §8 tiers 1 to 3 and the stretch row, §10 statistics, §11 E10 and E17, novelty check §5 Table 2, literature review §3 Table 8
**Severity**: Major
**Confidence**: 4 (core expertise: conditional similarity baselines and instruction-following multimodal embedders)

### W3: K7 compares the learned basis only with weak or concept-vocabulary bases, not with sparse autoencoders
Problem. K7 ("the gain comes from the learned basis") is tested against PCA, NMF and SpLiCE. Since 2024 the dominant unsupervised sparse code on frozen CLIP is the sparse autoencoder. The plan's Table 5 itself lists Discover-then-Name, Matryoshka SAE, the multimodal SAE analysis of Papadimitriou et al., and the shared-dictionary MGSAE and SPARC. A joint image-text SAE on the same features is a "shared sparse image-text basis" built without episodes. It is the direct ablation of C2's training signal. The literature review's own answer to the CSN objection already relies on a comparison "against a split dictionary, as MGSAE warns", but no dictionary baseline is planned.
Why it matters. If a generic SAE basis with the agreement rule matches method A, the contribution moves from the basis to the episodes, or disappears. If SAE latents split by modality, as Papadimitriou et al. report, the result directly supports K6. Either outcome is decision-relevant, and a CV reviewer working on sparse concept codes will ask for it.
Suggestion. Add a TopK SAE trained jointly on CLIP image and caption features of the scorer-train rows, with sparsity matched to the factor codes. Optionally add the MGSAE recipe. Run the agreement rule on its latents in the K7 table.
**Evidence Anchor**: text: §8 Tier 1 unsupervised-bases row "PCA-32/64, NMF-32 and SpLiCE sparse concept codes instead of our factors"
**Severity**: Major
**Confidence**: 3 (core expertise: sparse concept codes on CLIP; whether SAE latents work with the rule at 4 plus 4 pairs is untested)

### W4: "No labels from the evaluation taxonomy" undersells how closely the ArtELingo pseudo-partitions match the evaluation aspects
Problem. C2 claims the factors are trained "with no labels from the evaluation taxonomy". On ArtELingo:
- The emotion partition is k-means over a GoEmotions classifier whose label set contains 6 of the 8 evaluation emotions (amusement, anger, disgust, excitement, fear, sadness).
- The image partition is chosen as "style-like" (adjusted mutual information 0.32 with style).
- The caption-content partition is "genre-like".

So each training pseudo-aspect is hand-matched to one evaluation aspect. The literature review states the reviewer objection correctly, but its prepared answer ("claim no labels from the evaluation taxonomy") does not answer it, because the overlap is with the evaluation taxonomy itself.
Why it matters. Read this way, the emotion-aspect results are partly supervised by an emotion classifier, and the aspect-selection claim is weaker than "pseudo-aspects" suggests. CUB, with per-sentence caption clusters, is the cleaner test of the label-free story.
Suggestion. Word the claim as "no ArtELingo labels; distant supervision from a GoEmotions classifier whose label set overlaps 6 of the 8 evaluation emotions; partitions chosen to resemble the evaluation aspects". Report the emotion aspect with and without the affect partition, and lead the label-free argument with CUB.
**Evidence Anchor**: text: literature review §5 item 5 "a supervised GoEmotions classifier that names 6 of the 8 emotions"
**Severity**: Major
**Confidence**: 4 (core expertise: distant supervision in affective vision-language work; the label overlap is checkable in both taxonomies)

### W5: K3's subjective/objective direction is undefined and not yet supported by the appended evidence
Problem. K3 predicts that examples beat names on subjective aspects and match them on objective ones, and §10 pre-declares "beats" or "matches" per aspect type. No definition says which aspects count as subjective (is style subjective? genre? bill shape?). The only appended evidence runs the other way: privileged names gained on emotion (+3.2 over CLIP) and not on style. The R-names mitigation ("narrow K3 to where names fail (subjective aspects)") assumes names fail where, so far, they help most. The planned names baselines (Qwen instruction, CRL) will be stronger than the zero-shot CLIP prompts in the spike. The one outside data point offered for the direction, CLAY's +4.9 mAP on mood, is mild support at best.
Why it matters. K3 carries the "examples instead of names" positioning that sets C1 apart from CLAY, CRL, COCO-Facet and TPIPS. A pre-declared primary comparison whose partition and direction are not fixed cannot be read cleanly.
Suggestion. Define aspect types before the October 23 pre-registration, for example by inter-annotator agreement or nameability measured on development data. State K3 as a hypothesis with both outcomes reportable. Rewrite R-names so the fallback does not presuppose the direction.
**Evidence Anchor**: text: §4 claims table K3 "Examples beat naming the aspect on subjective aspects and match it on objective ones"
**Severity**: Major
**Confidence**: 3 (core expertise: text-conditioned similarity baselines; the direction is my reading of a single spike on untrained factors)

### W6: C3's "belongs to the data" cannot separate modality content from how ArtEmis labels were collected
Problem. C3 infers from flat weak-side probes across four encoders that the asymmetry is "a property of the data, not of the encoder". Two dataset facts predict the same flatness whatever the encoder:
- In ArtEmis, an emotion label belongs to one viewer's annotation. A painting carries several annotations with differing emotions, and the image inherits its row's label. Image-side emotion accuracy is therefore capped by label ambiguity (Bayes error), which does not depend on the encoder.
- Captions were elicited to explain the chosen emotion, not to describe style, so style-from-captions is capped by the elicitation protocol.

The CUB result (colour symmetric, with captions elicited to describe appearance) fits "asymmetry follows what annotators were asked to write" just as well. The backbone check notes the label noise but still draws the stronger conclusion.
Why it matters. C3 is one of three contributions and the core of the NO-GO fallback paper. As worded, it states a property of images and texts that the evidence cannot isolate from properties of the annotation protocol.
Suggestion. Recompute the image-side emotion probe on paintings with high annotator agreement, using painting-level majority or distribution targets. Phrase C3 as "in these datasets, under these elicitation protocols". Let CUB and SemArt serve as protocol contrasts rather than as confirmation.
**Evidence Anchor**: text: backbone check Verdict "The weaker-modality probes barely move across four backbones, so the modality asymmetry is a property of the data"
**Severity**: Major
**Confidence**: 3 (adjacent expertise: affective annotation datasets; the ArtEmis per-annotation design is stated in the manuscript's §2.3)

### W7: The SEARLE 14.4 row is given two different backbones
Problem. Table 2 lists 14.4 as CIReVL's SEARLE re-run at ViT-B/32. The pitfalls paragraph says 14.4 is CIReVL's SEARLE run at ViT-L/14, and RTD's B/32 SEARLE run is 12.19. The plan quotes 14.4 as a B/32 row. One of the two statements is wrong, and the B/32 SEARLE reference is either 14.4 or 12.19 depending on the source.
Suggestion. Re-check CIReVL Table 3 and cite one backbone with its source table.
**Evidence Anchor**: text: literature review §2.2 pitfalls "SEARLE at ViT-L/14 circulates as 14.4 (CIReVL's run) and 12.26 (LinCIR's run)"
**Severity**: Minor
**Confidence**: 4 (internal inconsistency, verifiable in the manuscript)

### W8: The agreement rule's lineage is labelled Rocchio/KISSME, but it is a cross-modal second-moment contrast
Problem. Rocchio and the earlier naive rule use first moments (mean support code minus mean contrast code). The agreement rule uses the mean elementwise product of image and caption codes, a difference of uncentred diagonal cross-modal second moments. KISSME uses inverse covariances of pair differences. For non-negative codes, a_I a_T equals (a_I² + a_T² − (a_I − a_T)²)/2, so the rule rewards both low within-pair difference (the KISSME part) and high joint activation (absent from KISSME). This is closer to a diagonal cross-covariance (PLS or CCA style) contrast, which the novelty check itself suggests in §4. The glossary still names Rocchio and CSN as the ancestors of "our rule".
Suggestion. Describe the rule as a diagonal cross-covariance contrast, cite Rasiwasia et al. 2010 next to KISSME, and use the decomposition above to answer "diagonal KISSME" precisely.
**Evidence Anchor**: text: Appendix A glossary "the classic ancestors of our rule and score"
**Severity**: Minor
**Confidence**: 4 (core expertise: metric learning from pairs)

### W9: "No paired swap test anywhere" is overstated; C-STS rates the same pair under contrasting conditions
Problem. C-STS (Deshpande et al., EMNLP 2023, arXiv 2305.15093, verified in this session) scores the same sentence pair under different natural-language conditions that give high and low similarity. That is a text-condition analogue of the swap test. CLAY's single gallery under three conditions, which the review already marks "partly", is another. C1's conjunction survives, because C-STS is text-only and names its conditions. But the novelty check's absolute statement does not hold, and C-STS is the obvious conditional-similarity precedent outside vision.
Suggestion. Cite C-STS and say the swap test differs by being paired, example-defined and cross-modal.
**Evidence Anchor**: text: novelty check §2 "we found no paired swap test (f) anywhere"
**Severity**: Minor
**Confidence**: 4 (core expertise: conditional similarity; reference checked at its arXiv page)

### W10: The label-free condition-discovery line misses multiview triplet embedding
Problem. The plan positions its pseudo-partition training against SCE-Net, DiscoverNet and multi-clustering (IC|TC, Multi-MaP). It does not cite Amid and Ukkonen's Multiview Triplet Embedding (ICML 2015, PMLR 37, verified in this session), which discovers several attribute-specific similarity maps from triplets without attribute labels. That is the earliest close ancestor of "several hidden notions of similarity learned without condition labels".
Suggestion. Add it to the §2.1 condition-learning paragraph and the must-cite list.
**Evidence Anchor**: absence: literature review §2.1 and §6, novelty check thread 1 and §6 — expected multiview triplet embedding (Amid and Ukkonen, ICML 2015) as prior work that discovers several attribute-specific notions of similarity without attribute labels; checked literature review Table 1, the §2.1 paragraph on learning conditions without labels, both reference lists, novelty check Table 1
**Severity**: Minor
**Confidence**: 3 (core expertise: conditional similarity lineage; relevance is my judgement)

### Questions for Authors
1. Against which GeneCIS focus-attribute numbers will K5 be judged, and is K5 dropped or reworded if the text protocol is not delivered by October 23?
2. Will the K7 table include a jointly trained image-caption SAE basis, and will the emotion aspect be reported for a factor model trained without the GoEmotions partition?
3. How is each aspect assigned to "subjective" or "objective" before pre-registration, and which emotion result would count against K3?
4. What is the image-side emotion probe accuracy on ArtELingo paintings with high annotator agreement, compared with the 35 to 37 reported now?

### Missing Key References
- Deshpande, Jimenez, Chen, Murahari, Graf, Rajpurohit, Kalyan, Chen, Narasimhan. C-STS: Conditional Semantic Textual Similarity. EMNLP 2023. arXiv 2305.15093 (checked in this session). Relevant to the swap test and to "condition" terminology (W9).
- Amid, Ukkonen. Multiview Triplet Embedding: Learning Attributes in Multiple Maps. ICML 2015, PMLR 37 (checked in this session). Relevant to label-free discovery of several notions of similarity (W10).
- Search lead [UNVERIFIED]: 2024 to 2026 work that gives open MLLMs interleaved image-text demonstrations for few-shot ranking or retrieval, beyond Bongard-OpenWorld, to calibrate the in-context baselines of W2.

### Minor Issues
- A short map from the plan's terms to the field's would help readers from the CSN and GeneCIS lines: aspect corresponds to CSN's notion or condition and to GeneCIS's attribute type; value corresponds to a condition value.
- The plan rightly avoids "few-shot cross-modal retrieval" as a name; the paper should keep its own term for the task from the abstract onward.
