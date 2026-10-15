# Synthesis: the grouping component of CoSiR v2 (stage 1) and its interface to stage 2

Date: 2026-10-05. Phase 3 synthesis (ARS synthesis agent) for the run defined in `research_brief.md`.

**Inputs read.** The brief; the six scans `scan_T1.md` to `scan_T6.md`; the format precedent
`src/test/20261107_new_method_candidates/synthesis.md`; and, to check every project number repeated here, the draft report
`docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md` (§1 to §6), the stage report
`docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md` (§2.2, §3, §10, §11, §14), the weekly report
`docs/reports/weekly/2026-09-30_percept_buddy_to_v2.md` (§2.4, §3.2, §3.4), the affect factor-learning report
`docs/reports/auto/v2/2026-10-18_candidate_a_affect_factor_learning_selection.md` (verdict table), the CVPR plan spec
(§6, simulation line), `src/test/20261110_partition_profile/20261110_partition_profile_log.md` (checks 3 and 3b) and
`src/test/20261111_community_told_oracle/20261111_community_told_oracle_log.md` (§3). We ran no new search, trained
nothing and scored nothing. Retrieved and scanned text was treated as data.

**AI disclosure.** Written by a Claude agent from six scans that Claude agents produced with WebSearch and WebFetch. No
paper was re-opened here, so every literature claim is bounded by the read scope its scan states. A human should read
the items flagged in Appendix B before any of them enters the paper.

**Status of claims.** **[lit]** marks a literature-supported statement, with citation, scan of origin and read grade:
**F** full text (T3 read PDF text locally), **S** sections through a summarising fetch tool, **A** abstract only, **M**
metadata or a search record only (content unverified), **G** gray literature (software docs, READMEs). *Reader-inferred*
marks an untested inference, by a scan or by us. *Draft hypothesis* marks the brief's requirements (R1 to R5), designs
(P0, L, G, E) and separation signals (§6). **[P]** marks a project number, with the file it was found in. Absence claims
read "to our knowledge, within the scans' searches".

<!-- claim_intent_manifest
{"manifest_version":"1.0","manifest_id":"M-2026-10-05T12:00:00Z-g7k2","emitted_by":"synthesis_agent","emitted_at":"2026-10-05T12:00:00Z",
"claims":[
{"claim_id":"C-001","claim_text":"Within the scans' searches, no published method fuses several sources into one structure and then splits it into single-property groupings with measured recovery; the project's hierarchical refinement nested one source inside another and did not fuse.","intended_evidence_kind":"empirical","planned_refs":["2106.05241","strehl02a","mucha_science_2010","gondek_hofmann_kais_2007"]},
{"claim_id":"C-002","claim_text":"Image-text contrastive identifiability results identify the block shared by both modalities and not modality-specific factors, which predicts that cross-modal placeability objectives favour content over image-only style and viewer-level emotion.","intended_evidence_kind":"theoretical","planned_refs":["2303.09166","2106.07115","2311.04056","1807.06653"],"negative_constraints":[{"constraint_id":"NC-C002-1","rule":"Do not state that the theory was tested on artworks or on affect."}]},
{"claim_id":"C-003","claim_text":"Placeability used as an objective or as a cross-source selection criterion conflicts with dominance for aspects with a weak modality; it remains usable within one source.","intended_evidence_kind":"theoretical","planned_refs":["2303.09166"]},
{"claim_id":"C-004","claim_text":"Without labels, which property a grouping follows cannot be identified, so dominance can be met only by an a priori source choice and verified only by a disclosed label read.","intended_evidence_kind":"theoretical","planned_refs":["1811.12359"]},
{"claim_id":"C-005","claim_text":"Recommended ranking P0, L, G, E, with the three queued checks first, L before G, G reduced to a label-free dominance diagnostic and E moved to the factor-learning discussion.","intended_evidence_kind":"normative","planned_refs":[]},
{"claim_id":"C-006","claim_text":"CSD is the only scanned style encoder with a published WikiArt zero-shot evaluation and no WikiArt labels in training; no label-free style descriptor has a published movement versus genre measurement.","intended_evidence_kind":"empirical","planned_refs":["2404.01292","2608.14435","1611.05368"]},
{"claim_id":"C-007","claim_text":"No scan found evidence that ArtEmis captions name art style, so style groupings will be placed from images and their caption side is the binding constraint.","intended_evidence_kind":"empirical","planned_refs":["2101.07396"]},
{"claim_id":"C-008","claim_text":"Long-CLIP is low priority because ArtEmis captions average about 15.8 words, near CLIP's effective length, and the project's rows are English.","intended_evidence_kind":"empirical","planned_refs":["2403.15378","2101.07396"]},
{"claim_id":"C-009","claim_text":"Clustering stability depends on the objective and misses over-coarse solutions, so it is a gate, not a selector; no scan found a label-free criterion shown to predict downstream usefulness.","intended_evidence_kind":"empirical","planned_refs":["ben-david_colt_2006","2006.08530","arbelaitz_pr_2013"]},
{"claim_id":"C-010","claim_text":"One label-free group-similarity matrix should serve both the reader's agreement and the bank's sibling-excluded negatives, with the identity matrix as matched control in both.","intended_evidence_kind":"normative","planned_refs":["sidorov_cys_2014","2011.11765","2106.03719"]},
{"claim_id":"C-011","claim_text":"SE's per-aspect gains were smaller than each single-source specialist's, so mixing groupings bought coverage rather than strength.","intended_evidence_kind":"empirical","planned_refs":["1810.02334"]},
{"claim_id":"C-012","claim_text":"The Leiden sweep's picked cell was chosen by reader margin on labelled episodes, which requirement R5 now rules out for grouping choice.","intended_evidence_kind":"normative","planned_refs":[]}],
"manifest_negative_constraints":[
{"constraint_id":"MNC-1","rule":"No draft hypothesis or reader-inferred statement is restated as a finding."},
{"constraint_id":"MNC-2","rule":"No claim of exhaustive absence; every absence is bounded by the scans' searches."},
{"constraint_id":"MNC-3","rule":"No citation's verification status is upgraded beyond what its scan reported."},
{"constraint_id":"MNC-4","rule":"No project number that was not found in the cited project files."},
{"constraint_id":"MNC-5","rule":"No claim that an audit or external review was run."}]}
-->

## Summary

The six scans agree on two absences. No published method fuses several sources into one structure and then splits it
into single-property groupings (T1), and no identifiability result splits one shared block into two correlated
properties, such as style and genre, without an extra signal that changes one and holds the other (T2). The positive
literature backs the designs that keep sources apart: one grouping per source (P0) and conditional refinement of kept
groupings (L), with heads on frozen features trained by neighbour-consistency losses (T3) and banks built from several
partitions (T6). It warns against both fusion designs. G's shared multiplex partition gives every layer one membership,
which by our reading makes it the dropped consensus partition built on graphs; and by the image-text identifiability
results E's cross-modal heads would keep what image and caption share, which in our data is content, not image-only
style or viewer-level emotion (a reader-inferred application). The same argument exposes a conflict inside the
requirements: optimising or selecting on cross-modal placeability (R2) pulls groupings toward content and against
dominance (R1). We recommend the
ranking P0, L, G, E: the three queued checks first, then P0 with a style source and sibling-aware agreement, then L on
a frozen trunk, G only as a label-free dominance diagnostic, and E handed to the factor-learning discussion. By our
reading, the same-painting ceiling (check 3) bounds how much image-side affect placement can ever recover.

**Terms** (the draft report's, §2). A *grouping* splits the 183,694 scorer-train rows into *groups* without evaluation
labels. A *head* places a lone image or caption into a grouping (logistic regression on frozen CLIP ViT-B/32 features).
*Agreement* is p_img · p_txt. The *reader* picks the grouping with the largest Δ (support minus contrast agreement).
*Told* gives the scorer the right grouping from the labels (a diagnostic). A *margin* is R@1 over the matched
condition-free counterpart. *B* is the best condition-free score (R@1 18.34 on seed 42). *S* is a group-similarity matrix.

## 1. Evidence per requirement

| | Literature supports | Literature contradicts or warns | Open |
|---|---|---|---|
| **R1 dominance** | One clustering returns the dominant structure and hides the others; keeping several groupings is the remedy: Niu et al. (2010) <!--ref:niu_icml_2010--><!--anchor:section:abstract--> (T1, A), ENRC (Miklautz et al., 2020) <!--ref:miklautz_aaai_2020--><!--anchor:section:abstract--> (T1, A), MFCVAE (Falck et al., 2021) <!--ref:2106.05241--><!--anchor:section:Table1--> (T1, S). [P] mixed codes give 0.00 condition gain against 0.70 to 0.95 for aspect-block codes (spec §6); SE kept both aspects where each single source lost one (affect report) | Which factor a representation follows cannot be identified without inductive bias or supervision: Locatello et al. (2019) <!--ref:1811.12359--><!--anchor:section:abstract--> (T2, A). The more discriminative view dominates joint learning: BMvC (Li et al., 2025) <!--ref:2501.02564--><!--anchor:section:abstract--> (T1, A); a dominant criterion needs an explicit override: IC\|TC (Kwon et al., 2023) <!--ref:2310.18297--><!--anchor:section:abstract--> (T3, A) | Style. No label-free descriptor has a published movement-versus-genre measurement (T4). [P] the image grouping's style-versus-genre contrast is 0.44 (profile log, check 3); told gain on style × genre is exactly 0 (stage report §14) |
| **R2 placeability** | Heads on frozen features with neighbour losses beat k-means: TEMI (Adaloglou et al., 2023) <!--ref:2303.17896--><!--anchor:section:abstract--> +6.1 ImageNet, +12.2 CIFAR100 (T3, F); SCAN (Van Gansbeke et al., 2020) <!--ref:2005.12320--><!--anchor:section:2.2--> (T3, F). Cross-modal pseudo-labels beat same-modality ones by 8 to 12 points: XDC (Alwassel et al., 2020) <!--ref:1911.12667--><!--anchor:section:4--> (T3, F) | Image-text contrastive learning identifies the block shared by both modalities and not the modality-specific factors: Daunhawer et al. (2023) <!--ref:2303.09166--><!--anchor:section:Theorem1--> (T2, S). [P] across buddy trials Stage 2 AUC fell as emotion AMI rose (r = −0.70, −0.74; PercepT's trials −0.10, −0.06; stage report §3 and weekly §3.6); in the affect sweep cluster lift rose 1.99 to 3.45 while lift through the heads stayed 1.12 to 1.16 and image-head accuracy fell 21.2% to 4.8% (draft §5) | The ceiling. [P] the affect image head is right 13.5% of the time (draft Table 2), 1.24 times the majority rate (told-oracle log §3). No source gives finite-sample retention of a weakly shared factor (T2) |
| **R3 siblings** | A similarity matrix inside a bilinear score that falls back to the plain form: soft cosine (Sidorov et al., 2014) <!--ref:sidorov_cys_2014--><!--anchor:section:abstract--> (T5, A); hierarchy-aware scoring (Bertinetto et al., 2020) <!--ref:1912.09393--><!--anchor:section:abstract--> (T5, A); OT with a ground cost (Kusner et al., 2015 <!--ref:kusnerb15--><!--anchor:section:abstract-->; Cuturi, 2013 <!--ref:1306.0895--><!--anchor:section:abstract-->) (T5, A). Bank side: same-concept negatives are the main failure of contrast (FNC, Huynh et al., 2022 <!--ref:2011.11765--><!--anchor:section:abstract-->; IFND, Chen et al., 2022 <!--ref:2106.03719--><!--anchor:section:abstract-->) (T6, A) | Every published S is external (a dictionary, WordNet); a label-free S is our extension (T5). A co-membership S can inflate agreement for random pairs (T5, reader-inferred) | Whether S moves margins (check 2). [P] the case for it: sadness sits 35% in cluster 38, about 21% in five smaller mostly sad clusters, about 19% in two large mixed ones (draft §3) |
| **R4 non-redundancy** | Methods exist: Niu et al. <!--ref:niu_icml_2010--><!--anchor:section:abstract-->, ENRC <!--ref:miklautz_aaai_2020--><!--anchor:section:abstract-->, MFCVAE <!--ref:2106.05241--><!--anchor:section:Table1-->, Cui et al. (2007) <!--ref:cui_icdm_2007--><!--anchor:none:--> (T1; Cui M, content unverified); variation of information as a metric (Meilă, 2007) <!--ref:meila_jmva_2007--><!--anchor:none:--> (T5, M). [P] affect and image groupings are already near independent (AMI 0.026; affect report) | An independence penalty without an inductive bias does not recover factors (Locatello et al., 2019 <!--ref:1811.12359--><!--anchor:section:abstract-->; T2, reader-inferred application). MFCVAE's evidence is on independent-by-construction facets (T1, reader-inferred). No T3 loss supplies it; IIC between heads has the wrong sign (T3) | Conditional non-redundancy (L) has no theory and no evidence (T2); its closest method, Gondek and Hofmann (2007) <!--ref:gondek_hofmann_kais_2007--><!--anchor:none:-->, is metadata only (T1, M). *Reader-inferred:* a random grouping is maximally non-redundant, so R4 is never a criterion alone |
| **R5 label-free choice** | Stability across resamples is mature for choosing K: Ben-Hur et al. (2002) <!--ref:ben-hur_psb_2002--><!--anchor:none:--> (M), Lange et al. (2004) <!--ref:lange_neco_2004--><!--anchor:none:--> (M), von Luxburg (2010) <!--ref:1007.1075--><!--anchor:section:abstract--> (A) (T5) | For large samples stability is fixed by the objective (Ben-David et al., 2006) <!--ref:ben-david_colt_2006--><!--anchor:section:abstract--> (A); it misses over-coarse solutions (Mourer et al., 2023) <!--ref:2006.08530--><!--anchor:section:abstract--> (A); no validity index is universally best (Arbelaitz et al., 2013) <!--ref:arbelaitz_pr_2013--><!--anchor:none:--> (M) (T5). [P] silhouette rose 0.55 to 0.70 while image-to-caption R@1 fell to about 10.4 against 17.8 for CLIP (stage report §3); PercepT's clearer clusters gave no emotion advantage downstream (0.0221 against 0.0231; weekly §3.2) | No scan found a label-free criterion validated against downstream usefulness (T5). The generic menu is untested on CUB, SemArt and GeneCIS |

**Theoretical integration** (reader-inferred, ours, from T2 and T3). Two results organise the table. First, by
Locatello et al. (2019) <!--ref:1811.12359--><!--anchor:section:abstract--> (T2, A), which property a grouping follows
cannot be identified without an inductive bias or supervision. Without labels, R1 can therefore be *met* only by choosing
sources that carry one property, and *verified* only by reading labels once, disclosed. The fixed generic source menu is
exactly such an inductive bias, written down in advance. Second, by Daunhawer et al. (2023)
<!--ref:2303.09166--><!--anchor:section:Theorem1--> (T2, S), an image-caption objective keeps the block both modalities
share. In our data that block is content (genre, subject) and the painting-level part of emotion, while style is mostly
image-only (60.8% from images against 25.4% from captions; stage report §3) and the viewer-level part of emotion is
caption-only. R2 therefore pulls against R1 for exactly the two aspects with a weak modality.

## 2. Signal map (brief §6)

| Signal | Theory | Empirical evidence | Project evidence | Verdict |
|---|---|---|---|---|
| source provenance | By analogy only: views of shared latents (Gresele et al., 2019 <!--ref:1905.06642--><!--anchor:section:abstract-->; Lyu et al., 2022 <!--ref:2106.07115--><!--anchor:section:abstract-->; Yao et al., 2024 <!--ref:2311.04056--><!--anchor:section:abstract-->) or an auxiliary variable that must change the latent distribution (iVAE, Khemakhem et al., 2020 <!--ref:1907.04809--><!--anchor:section:Theorem1-->) (T2). Our sources are derived from the same inputs, so view assumptions are doubtful (T2, reader-inferred) | Layer-specific communities under a multilayer block model (Wilson et al., 2017 <!--ref:1610.06511--><!--anchor:section:abstract-->; T1, A); a label source different from the input prevents collapse (XDC <!--ref:1911.12667--><!--anchor:section:4-->; T3, F) | Hierarchical refinement kept emotion AMI 0.107 against 0.036 for a random split; union fusion merged content communities (weekly §2.4) | theory by analogy; no evidence for derived sources |
| modality | Shared block identified, modality-specific not (Daunhawer et al. <!--ref:2303.09166--><!--anchor:section:Theorem1-->; Lyu et al. <!--ref:2106.07115--><!--anchor:section:abstract-->; Yao et al. <!--ref:2311.04056--><!--anchor:section:abstract-->) (T2) | Shared factors recovered on Multimodal3DIdent (Daunhawer, S); XDC +8 to 12 points (T3, F) | Weak modality per aspect; four backbones move the weak side by at most 1.5 points (stage report §3) | theory and evidence, for separating shared from private, not for placing the private part |
| painting membership | Analogue of pairs sharing some factors (Locatello et al., 2020 <!--ref:2002.02886--><!--anchor:section:Theorem1-->) and content-invariant pairs (von Kügelgen et al., 2021 <!--ref:2106.04619--><!--anchor:section:Theorem4.4-->): painting-level block against viewer-level remainder (T2, reader-inferred) | none found | About 5 rows per painting; how mixed a painting's emotions are has not been measured (stage report §2.2) | theory by analogy, block level; does not split style from genre |
| non-redundancy | negative (Locatello et al., 2019 <!--ref:1811.12359--><!--anchor:section:abstract-->) | MFCVAE facet probes 94 to 100% against 17 to 73% <!--ref:2106.05241--><!--anchor:section:Table1--> (T1, S), on independent facets | affect × image AMI 0.026 | methods and synthetic evidence; not identifying alone |
| conditional non-redundancy | none found (T2) | the method exists, content unverified (Gondek and Hofmann <!--ref:gondek_hofmann_kais_2007--><!--anchor:none:-->; T1, M) | none | none |
| agreement versus disagreement | Information shared by subsets of views (Yao et al. <!--ref:2311.04056--><!--anchor:section:abstract-->; T2, reader-inferred mapping); co-regularisation enforces one shared partition (Kumar et al., 2011 <!--ref:kumar_nips_2011--><!--anchor:section:abstract-->; T1, A) | none for groupings | Graph intersection left 98.96% of paintings without an edge, CCA 0.73 against 0.07 shuffled (weekly §2.4) | theory by analogy |
| head competition | none found (T2; mixture-of-experts identifiability not searched) | SCE-Net masks align with four Zappos attributes (Tan et al., 2019 <!--ref:1908.08589--><!--anchor:section:Fig4-->); best embedding count near the condition count (DiscoverNet, Ye et al., 2022 <!--ref:2204.04053--><!--anchor:section:Table3-->); both with attribute-sampled triplets (T1, S). Multi-head clustering heads differ only by initialisation or K (SeLa §3.5, Asano et al., 2020 <!--ref:1911.05371--><!--anchor:section:3.5-->; T3, F) | none | evidence only under attribute-structured sampling |
| augmentation invariance | The invariant block is content; style is not identified (von Kügelgen et al. <!--ref:2106.04619--><!--anchor:section:Theorem4.4-->; T2). Isolating style inverts the theorem, not shown | CSD trains style with tag contrast plus a self-supervised term (Somepalli et al., 2024 <!--ref:2404.01292--><!--anchor:section:4-->; T4, S); ALADIN-NST on synthetic style-fixed pairs (Ruta et al., 2023 <!--ref:2304.05755--><!--anchor:section:4.2-->; T4, S) | none | theory for content only |
| anchoring to the source | none found (T2) | KL to a known prior (SCAN §2.2 <!--ref:2005.12320--><!--anchor:section:2.2-->); teacher graph as anchor, joint fine-tuning helped only under a domain gap (TEMI §5 <!--ref:2303.17896--><!--anchor:section:5-->, author-acknowledged) (T3, F) | Up-weighting the affect teacher 2 to 8 times lowered emotion AMI from 0.1306 to at most 0.1180 and genre to 0.035 or less (weekly §3.4) | method precedents; no theory; weighting is not an anchor |

## 3. Tensions between the scans

| # | Claim A | Claim B | Resolution |
|---|---|---|---|
| 1 | T3 (reader-inferred): SwAV, IIC and TAC across image and caption reward what both towers share, so content wins | T2 [lit]: Daunhawer et al. <!--ref:2303.09166--><!--anchor:section:Theorem1--> identify the shared block, not modality-specific factors; FactorCL (2023) <!--ref:2306.05268--><!--anchor:section:abstract--> keeps unique information with explicit extra terms (A, no proof) | **Convergent, not contradictory.** Inference and theory predict the same thing. What they contradict is the hope (E, and L's placeability term) that a cross-modal objective makes style or viewer-level emotion placeable. FactorCL is the only escape route found, empirical only. Open: how much of a weakly shared factor a finite encoder keeps |
| 2 | Brief R2: groupings must be placeable from both modalities | Brief R1 with row 1: placeability favours content | **Conditional.** Placeability is a usable criterion within one source and a misleading one across sources; as an objective it pulls toward content. [P] Stage 2 AUC against emotion AMI, r = −0.70 and −0.74 (stage report §3) |
| 3 | T4: no source shows ArtEmis captions naming style; ArtEmis does not quantify it (Achlioptas et al., 2021 <!--ref:2101.07396--><!--anchor:section:abstract-->) | Brief P0: a style grouping placed and agreed from both modalities | **Reconcilable.** The style grouping will be placed from images; the caption head is the binding side ([P] style from captions 25.4%). Open: whether a caption head places a style grouping above shuffled pairs (step 1 measures it without labels) |
| 4 | T1: no fuse-then-split precedent; the only positive is internal | Brief §5: G's precedent is hierarchical refinement | **Resolved against the brief's label.** Hierarchical refinement nested affect inside content communities; it fused nothing. It is a precedent for conditional splitting (L; Gondek and Hofmann <!--ref:gondek_hofmann_kais_2007--><!--anchor:none:-->), not for a shared partition. The internal fused-graph result (union) was negative |
| 5 | T6: diversity across groupings beat quality of one (SE, CACTUs <!--ref:1810.02334--><!--anchor:section:5-->) | [P] affect report: SE gained emotion +1.31 [0.54, 2.06] and style +1.12 [0.29, 1.93]; affect alone gained emotion +3.99 [3.14, 4.83] and lost style −1.23; image alone gained style +3.60 [2.69, 4.48] and lost emotion −0.85 | **Resolved as coverage, not strength.** Mixing kept every aspect and cut each one's gain to about a third. T6's claim holds as "no aspect lost" |
| 6 | T5: stability is objective-dependent and misses over-coarse solutions; no criterion predicts usefulness | Brief §3 and R5: tune group count and sources by label-free criteria | **Conditional.** Label-free criteria can gate and compare within a source; none is validated as a selector. Check 1 would be new evidence (T5) |
| 7 | T1 [lit]: competing masks recover attributes (SCE-Net <!--ref:1908.08589--><!--anchor:section:Fig4-->, DiscoverNet <!--ref:2204.04053--><!--anchor:section:Table3-->); MFCVAE recovers facets <!--ref:2106.05241--><!--anchor:section:Table1--> | T2 [lit]: no theory for head competition; nothing is identified without an inductive bias (Locatello et al., 2019 <!--ref:1811.12359--><!--anchor:section:abstract-->) | **Resolved.** The successes carry their inductive bias: attribute-sampled triplets with K near the attribute count, or independent priors on independent facets. E's pooled, source-tagged edges over correlated properties lie outside it |
| 8 | T3 [lit]: frozen-feature heads are competitive; joint fine-tuning helped only under a domain gap (TEMI §5 <!--ref:2303.17896--><!--anchor:section:5-->) | User (brief §3): updating representation and groups is wanted | **Open, with a default.** [P] Every internal attempt to reshape a fused representation either traded one property for the other or lowered both: Attention-h1 (emotion 0.135, genre 0.240, training split; weekly §2.4; a trade by the brief's reading), widening the student raised genre (0.29, 0.33) and lowered emotion (0.115, 0.109), six DEC hybrids lowered separation and agreement together (weekly §3.2). Update heads and groups on a frozen trunk first (L) |
| 9 | T2 (reader-inferred): painting pairs identify the painting-level block, including a painting's consensus emotion | Brief §6: painting membership separates style and genre from emotion | **Partial.** The split is painting-level against viewer-idiosyncratic, not style and genre against emotion. Check 3 measures the painting-level part of affect |
| 10 | T4 [lit]: CSD is the best-evidenced style source <!--ref:2404.01292--><!--anchor:section:6--> | T4 and T2: CSD starts from CLIP and is scored with the artist as style proxy; augmentation invariance alone identifies content (von Kügelgen <!--ref:2106.04619--><!--anchor:section:Theorem4.4-->) | **Open.** A label-free redundancy check against the CLIP image grouping (step 1) is the first test |

#### Cross-Paper Tension Inventory (#262)

```yaml
cross_paper_tensions:
  - {pair_id: CP-001, paper_a: "2303.09166", paper_b: "2306.05268", candidate_basis: "shared construct/outcome/measure",
     overlap_topic: "Does a label-free image-text objective keep modality-unique information?",
     a_finding: "Symmetric multimodal contrast block-identifies shared factors; modality-specific ones are not identified.",
     a_evidence_pointer: "scan_T2.md, Daunhawer entry, WHAT (Theorem 1, fetch summary)",
     b_finding: "Shared plus unique parts per modality, unique task-relevant information kept via augmentations.",
     b_evidence_pointer: "scan_T2.md, FactorCL entry (abstract only)",
     pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis,
     resolution_pointer: "Synthesis > §3 Tensions, row 1", scholar_confirmation: pending}
  - {pair_id: CP-002, paper_a: "1908.08589", paper_b: "1811.12359", candidate_basis: "opposite finding direction",
     overlap_topic: "Can single-property structure be recovered without labels?",
     a_finding: "Latent condition masks align with the four Zappos attributes.",
     a_evidence_pointer: "scan_T1.md, SCE-Net entry, WHAT (Fig. 4, Table 2)",
     b_finding: "Unsupervised disentanglement is impossible without inductive biases.",
     b_evidence_pointer: "scan_T2.md, Locatello 2019 entry (abstract)",
     pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis,
     resolution_pointer: "Synthesis > §3 Tensions, row 7", scholar_confirmation: pending}
  - {pair_id: CP-003, paper_a: "2106.05241", paper_b: "1811.12359", candidate_basis: "opposite finding direction",
     overlap_topic: "Unsupervised recovery of several clusterable properties",
     a_finding: "Facets recovered on MNIST, 3DShapes, SVHN; matching facet probes 94 to 100%.",
     a_evidence_pointer: "scan_T1.md, MFCVAE entry, WHAT (Tables 1 and 2)",
     b_finding: "Identification needs inductive bias or supervision.",
     b_evidence_pointer: "scan_T2.md, Locatello 2019 entry (abstract)",
     pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis,
     resolution_pointer: "Synthesis > §3 Tensions, row 7", scholar_confirmation: pending}
  - {pair_id: CP-004, paper_a: "2501.02564", paper_b: "leidenalg_multiplex_docs", candidate_basis: "shared RQ subtopic",
     overlap_topic: "Does the strongest view dominate a fused clustering?",
     a_finding: "In deep joint training, more discriminative views dominate.",
     a_evidence_pointer: "scan_T1.md, BMvC entry (abstract)",
     b_finding: "Multiplex Leiden shares membership across layers; weights matter; no dominance reported.",
     b_evidence_pointer: "scan_T1.md, leidenalg docs entry (gray literature)",
     pair_assessment: conditional_difference, resolution_status: flagged_unresolved, scholar_confirmation: pending}
  - {pair_id: CP-005, paper_a: "1611.05368", paper_b: "2608.14435", candidate_basis: "shared construct/outcome/measure",
     overlap_topic: "Do frozen generic features carry art style apart from content?",
     a_finding: "One Gram layer of ImageNet VGG gives 33.46% style top-1 with a label-trained probe.",
     a_evidence_pointer: "scan_T4.md, Johnson entry, WHAT (§4, summary; class count unconfirmed)",
     b_finding: "Movement kNN on frozen embeddings drops 0.869 to 0.766 artist-disjoint; style stays entangled with content.",
     b_evidence_pointer: "scan_T4.md, Ashton entry, WHAT (§5 to §7, summary)",
     pair_assessment: conditional_difference, resolution_status: flagged_unresolved, scholar_confirmation: pending}
  - {pair_id: CP-006, paper_a: "2404.01292", paper_b: "2608.14435", candidate_basis: "shared construct/outcome/measure",
     overlap_topic: "Does a WikiArt style score measure style or artist signature?",
     a_finding: "CSD ViT-L 64.56 against CLIP ViT-L 59.4 mAP@1, artist used as style proxy.",
     a_evidence_pointer: "scan_T4.md, CSD entry, WHAT (§6, Table 1, summary)",
     b_finding: "Artist-disjoint evaluation lowers movement accuracy.",
     b_evidence_pointer: "scan_T4.md, Ashton entry, WHAT (§5 to §6)",
     pair_assessment: conditional_difference, resolution_status: flagged_unresolved, scholar_confirmation: pending}
  - {pair_id: CP-007, paper_a: "2303.17896", paper_b: "1911.12667", candidate_basis: "shared RQ subtopic",
     overlap_topic: "Freeze the encoder or train it from cluster targets?",
     a_finding: "Frozen features plus a head beat k-means; joint fine-tuning helped only under a domain gap.",
     a_evidence_pointer: "scan_T3.md, TEMI entry, WHAT and weaknesses (§3.3, §5)",
     b_finding: "Encoders trained on the other modality's clusters beat same-modality training by 8 to 12 points.",
     b_evidence_pointer: "scan_T3.md, XDC entry, WHAT (§4, Study 1)",
     pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis,
     resolution_pointer: "Synthesis > §3 Tensions, row 8", scholar_confirmation: pending}
  - {pair_id: CP-008, paper_a: "1810.02334", paper_b: "2011.14663", candidate_basis: "shared RQ subtopic",
     overlap_topic: "Transfer from cluster-built pseudo tasks to real tasks",
     a_finding: "Many partitions beat one; a large gap to the oracle remains.",
     a_evidence_pointer: "scan_T6.md, finding 1 and 2 (§5, Fig. 2, summary)",
     b_finding: "An adapter fitted on pseudo tasks helped pseudo and not real tasks.",
     b_evidence_pointer: "scan_T6.md, finding 5 (Table IX, relayed from scan_N4)",
     pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis,
     resolution_pointer: "Synthesis > §7 Interface, last bullet", scholar_confirmation: pending}
  - {pair_id: CP-009, paper_a: "2011.11765", paper_b: "2207.11163", candidate_basis: "shared construct/outcome/measure",
     overlap_topic: "Handling same-concept negatives",
     a_finding: "Cancel or attract false negatives.", a_evidence_pointer: "scan_T6.md, finding 7 (abstract)",
     b_finding: "Replace one-hot targets with soft relations.", b_evidence_pointer: "scan_T6.md, finding 7 (abstract)",
     pair_assessment: no_material_conflict, resolution_status: not_applicable, scholar_confirmation: pending}
  - {pair_id: CP-010, paper_a: "ben-hur_psb_2002", paper_b: "ben-david_colt_2006", candidate_basis: "opposite finding direction",
     overlap_topic: "Is stability a criterion for choosing a clustering?",
     a_finding: "Choose K by reproducibility under resampling.", a_evidence_pointer: "scan_T5.md, Finding 1 (bibliographic record only)",
     b_finding: "For large samples stability reflects the objective's minimiser, not usefulness.",
     b_evidence_pointer: "scan_T5.md, Finding 1 (abstract)",
     pair_assessment: conditional_difference, resolution_status: resolved_in_synthesis,
     resolution_pointer: "Synthesis > §3 Tensions, row 6", scholar_confirmation: pending}
  - {pair_id: CP-011, paper_a: "2106.04619", paper_b: "2404.01292", candidate_basis: "agent-noted cross-cluster",
     overlap_topic: "Can augmentation invariance isolate style?",
     a_finding: "The augmentation-invariant block is content; style is not identified.",
     a_evidence_pointer: "scan_T2.md, von Kugelgen entry (Theorem 4.4, fetch summary)",
     b_finding: "Style encoder trained by style-tag contrast plus a self-supervised term.",
     b_evidence_pointer: "scan_T4.md, CSD entry, HOW (§4, §5)",
     pair_assessment: conditional_difference, resolution_status: flagged_unresolved, scholar_confirmation: pending}
```

**Coverage Note**: 81 sources shortlisted across the six scans; 11 candidate pairs considered (basis: shared RQ
subtopic, shared construct or measure, opposite finding direction, one agent-noted cross-cluster pair). This is a
**scoped advisory scan, not complete pairwise contradiction detection**; cross-neighbourhood pairs not surfaced here may
exist and are not claimed absent. Not checked exhaustively: the T3 clustering losses among themselves, the T2
identifiability papers among themselves, the T4 style encoders against T3 losses. Bibliographic coupling was not
available and excluded no pair. The scholar confirms each `resolution_pointer` and may flag more pairs. Scan-to-brief
tensions are in the table above.

**Evidence convergence map.**

```
Moderate:  [======    ] One partition or fused objective follows the dominant property (T1 common WHY; BMvC; percept fusion, Attention-h1, widening)
Moderate:  [======    ] Cross-modal objectives keep the shared block (Daunhawer, Lyu, Yao; T3 by inference; Stage 2 AUC r = -0.70)
Moderate:  [=====     ] Heads on frozen features with neighbour losses beat k-means (SCAN, TEMI full text; Zhou and Zhang abstract)
Emerging:  [===       ] Several partitions beat one for pseudo-task banks (CACTUs at section level; SE internal)
Emerging:  [===       ] Label-free style encoders exist (CSD, ALADIN, Gram); none measured on movement versus genre
Emerging:  [==        ] Sibling-aware forms exist, all with external S (soft cosine, hierarchy, OT; abstract level)
Gap:       [          ] Fuse-then-split recovery; style split from genre without an extra signal; criteria that predict usefulness
```

## 4. Designs P0, L, G, E

| | Support | Main risks | Closest precedent | Must be invented | Cost |
|---|---|---|---|---|---|
| **P0** one grouping per source, Leiden each, sibling-aware agreement | Keeping several groupings ([lit] T1 row R1); several partitions for banks (CACTUs <!--ref:1810.02334--><!--anchor:section:5-->, T6, S); heads on frozen features (TEMI <!--ref:2303.17896--><!--anchor:section:abstract-->; Zhou and Zhang, 2022 <!--ref:2207.13364--><!--anchor:section:abstract-->; T3). [P] Leiden beat k-means on affect at every matched count, told +0.48 to +0.88 (draft §5) | Leaves the placement loss untouched ([P] lift 2.17 on the groups, 1.11 through heads; draft §3). Leiden's advantage is measured on affect only: on content, communities of the Block 1 student (which restated CLIP) scored +3.00 against +2.54 for CLIP k-means in a tie band and aligned with style at AMI 0.227 against 0.318 (stage report §3). A style source may stay genre-heavy or unplaceable from captions | XDC's per-modality clusterings <!--ref:1911.12667--><!--anchor:section:4-->; CACTUs' multiple partitions | label-free S; the style grouping | low: CPU minutes; one GPU pass over 36,518 paintings for a style encoder |
| **L** kept groupings refined jointly (non-redundancy given the others, placeability, anchor) | Clustering conditioned on a known clustering (Gondek and Hofmann <!--ref:gondek_hofmann_kais_2007--><!--anchor:none:-->, M); joint non-redundant clusterings (ENRC, Niu, MFCVAE; T1); several heads on one trunk (SeLa §3.5 <!--ref:1911.05371--><!--anchor:section:3.5-->); anchor as KL to a known assignment (SCAN §2.2 <!--ref:2005.12320--><!--anchor:section:2.2-->) (T3, F). [P] the only internal positive for splitting is conditional: refinement inside content communities kept emotion AMI 0.107 against 0.036 (weekly §2.4) | No theory for conditional non-redundancy or anchoring (T2); a placeability term pulls toward content (§1); the same refinement's genre AMI equalled its random-split control (0.195 against 0.201), so fragmentation set the cost; no stopping rule (T1); sharper placement may shrink the gain from S (T5) | Gondek and Hofmann; ENRC; SeLa heads with a SCAN anchor | refinement of existing groupings with an anchor (not found, T1) | low to medium: frozen trunk, CPU or one GPU, hours |
| **G** multiplex graph, shared Leiden partition, then layer-specific sub-groupings | Implemented machinery: multislice modularity (Mucha et al., 2010 <!--ref:mucha_science_2010--><!--anchor:section:abstract-->, A) and leidenalg multiplex <!--ref:leidenalg_multiplex_docs--><!--anchor:section:multiplex--> (G); layer-specific membership exists (leidenalg slices; Wilson et al. <!--ref:1610.06511--><!--anchor:section:abstract-->); multi-resolution levels supply F3's hierarchy (T5) | Shared membership ("cannot have different communities for different graphs", leidenalg docs) makes step one a consensus partition, the design the user dropped (*reader-inferred*, ours); view dominance in fusion (BMvC <!--ref:2501.02564--><!--anchor:section:abstract-->) is untested for multiplex modularity (T1); [P] union fusion merged content communities 28 to 21 and left emotion 0.124, genre 0.139 (weekly §2.4); one partition removes the multi-partition bank benefit (T6); kNN layers over derived features are not views of latents (T2) | Wilson et al.'s layer-specific extraction is nearer to what G needs than a shared partition | the split step and the placement of its sub-groupings | low to medium: CPU |
| **E** pooled source-tagged edges, competing property heads, two-tower trunk | Latent condition masks align with attributes (SCE-Net <!--ref:1908.08589--><!--anchor:section:Fig4-->, DiscoverNet <!--ref:2204.04053--><!--anchor:section:Table3-->; T1, S); unique per-modality information kept by FactorCL <!--ref:2306.05268--><!--anchor:section:abstract--> (T2, A); two-tower losses (IIC <!--ref:1807.06653--><!--anchor:section:3.1-->, SwAV <!--ref:2006.09882--><!--anchor:section:3-->, TAC <!--ref:2310.11989--><!--anchor:section:4.3.2-->; T3) | Image-only style is not identified by an image-caption objective (Daunhawer <!--ref:2303.09166--><!--anchor:section:Theorem1-->); four internal negatives (row 8 of §3; up-weighting, §2); K must be set by hand (SCE-Net, DiscoverNet, MFCVAE, author-acknowledged); alignment shown only with attribute-sampled triplets (T1); the pseudo-to-real gap grows when a conditioner trains on its own heads (Ye et al., 2020 <!--ref:2011.14663--><!--anchor:section:V-B-->, relayed by T6); overlaps stage 2 | SCE-Net and DiscoverNet, plus FactorCL | source-tagged head competition with a weak-property anchor; lone-modality towers | high: GPU training and design work |

**Ranking (a recommendation; the user decides).**

1. **P0.** Every component has published or project support, it is the matched control every fusion needs, it keeps
   provenance by construction, and it carries the two cheapest open questions (a style source, S) in minutes of CPU.
2. **L.** The one fusion point whose split mechanism has a published form (clustering conditioned on an existing
   clustering), and the design the only internal splitting success actually resembles. It keeps sources by
   construction, which sidesteps the dominance failure, and runs on a frozen trunk. Its terms lack theory and its
   placeability term conflicts with R1, so it must be judged on criteria it does not optimise (§6).
3. **G.** Cheap and implemented, but its first step is a consensus, view dominance is untested for it, and the nearest
   internal result (union fusion) was negative. Its value now is as a label-free dominance diagnostic and as the source
   of the hierarchy for S. A slice-coupled variant (layer-specific membership with inter-layer coupling, as in Mucha et
   al. and leidenalg slices) would let groupings interact while keeping provenance; it is *reader-inferred* and untested.
4. **E.** Its central bet, that cross-modal heads recover weak and modality-specific properties, runs against the
   block-identifiability theory and four internal negative results; it is the most expensive, overlaps stage 2, and
   cannot be built and controlled before 10 November.

**Working order.** We recommend changing brief §5's order in three ways: run the three queued checks before P0, because
they decide where R2 effort can pay; run L before G rather than side by side, with G reduced to a diagnostic; and move E
to the factor-learning discussion, since its heads are aspect-block codes (brief §5).

## 5. Style source and backbone (T4)

| Candidate | Training data | WikiArt labels / images | Evidence | Open risk |
|---|---|---|---|---|
| CSD ViT-L (Somepalli et al., 2024) <!--ref:2404.01292--><!--anchor:section:6--> | LAION-Styles, 511,921 images, 3,840 style tags; CLIP initialisation | labels no (evaluation only); images unclear | WikiArt zero-shot mAP@1 64.56 against 59.4 for CLIP ViT-L, artist as proxy (Table 1, S); arXiv preprint, venue unconfirmed | CLIP start may keep genre; score may reflect artist signature (CP-006); web tags may name movements (*reader-inferred*, unchecked) |
| Gram matrices of ImageNet VGG (Gatys et al., 2016 <!--ref:gatys_cvpr_2016--><!--anchor:none:-->, M; Johnson, 2017 <!--ref:1611.05368--><!--anchor:section:4-->, S) | ImageNet only | no / no | One Gram layer, 33.46% top-1 with a label-trained probe (class count unconfirmed); texture bias of ImageNet CNNs (Geirhos et al., 2019 <!--ref:geirhos_iclr_2019--><!--anchor:section:abstract-->, A) | no genre measurement; descriptor large (needs PCA) |
| ALADIN (Ruta et al., 2021) <!--ref:2103.09776--><!--anchor:section:3--> | BAM-FG (Behance), project co-membership | no / no | separate content and style branches (§3, S) | domain gap to classical paintings; project groups share subject (*reader-inferred*); no licence stated |
| ALADIN-NST <!--ref:2304.05755--><!--anchor:section:4.2--> | Behance styles plus style-transfer images | no / no | strongest content-removal evidence, synthetic tests only (S) | weights unchecked |
| GOYA (Wu et al., 2024) <!--ref:goya_jimaging_2024--><!--anchor:none:--> | Stable Diffusion images on CLIP features | unclear / unclear | search summary only (M) | not usable as evidence |
| CLIP ViT-B/32 image (control) | undisclosed | no / unclear | [P] genre lift 9.75, style 4.59 (profile log, check 3) | the problem it is meant to fix |

**Verdict.** For the hand-matched style source, CSD has the strongest case: it is the only candidate with a published
WikiArt zero-shot evaluation and no WikiArt labels in training. Gram statistics have the cleanest provenance (no art
data at all) and the weakest evidence, which makes them the natural generic-menu entry. Both are *reader-inferred*
choices. CSD shares CLIP's initialisation with the image grouping, so the first test is label-free: its redundancy
(variation of information) with the CLIP image grouping (§8, step 1). Measuring several candidates' style-versus-genre
profiles and keeping the best would select the grouping by labels (brief §9). The profile may describe the one chosen
grouping, once, disclosed.

**Captions.** No scanned source shows viewer captions naming style; ArtEmis does not quantify it
<!--ref:2101.07396--><!--anchor:section:abstract-->, and SemArt's expert comments
<!--ref:1810.09617--><!--anchor:section:abstract--> do name movements but are curator prose (T4). With [P] style from
captions at 25.4% (stage report §3), style must come from images in every design, and caption placement is the binding
side of any style grouping (*reader-inferred*).

**Long-CLIP: low priority.** ArtEmis explanations average 15.8 words (T4, via summary, section not recorded)
<!--ref:2101.07396--><!--anchor:none:-->, about 20 BPE tokens by T4's untested conversion, close to the "merely 20
tokens" Long-CLIP reports as CLIP's effective length (Zhang et al., 2024 <!--ref:2403.15378--><!--anchor:section:3.1-->,
S) and far below 77. T4's multilingual truncation risk does not apply: the project uses ArtELingo's English rows (stage
report §2.2). The weak side for style is captions, and there the limit is what captions say, not encoder length ([P] four
backbones moved the weak side by at most 1.5 points). One CPU check before dropping it: the token-length histogram of
the scorer-train captions. If used, disclose that ShareGPT4V's seed set has 500 WikiArt images
<!--ref:2311.12793--><!--anchor:section:appendixA-->.

## 6. Label-free criteria to design against

| Criterion | Evidence | Misleads when | Use |
|---|---|---|---|
| Placeability: chance-corrected MI of image-head and caption-head assignments, same row against shuffled rows (check 1) | IIC's objective <!--ref:1807.06653--><!--anchor:section:3.1--> (T3, F; T5); not validated as a predictor | compared across sources (favours content, §1); raw head accuracy falls with K while told stays flat ([P] draft §5); MI grows with many fine groups (T5) | within one source, to compare settings |
| Stability over seeds and subsamples; head-posterior stability over two 60,000-row draws | Ben-Hur <!--ref:ben-hur_psb_2002--><!--anchor:none:-->, Lange <!--ref:lange_neco_2004--><!--anchor:none:-->, von Luxburg <!--ref:1007.1075--><!--anchor:section:abstract--> (T5) | objective-dependent (Ben-David <!--ref:ben-david_colt_2006--><!--anchor:section:abstract-->); misses over-coarse solutions (Mourer <!--ref:2006.08530--><!--anchor:section:abstract-->); content-dominated partitions can be stable (T5, reader-inferred) | gate only |
| Non-redundancy: variation of information between groupings | Meilă <!--ref:meila_jmva_2007--><!--anchor:none:--> (T5, M) | a random grouping is maximally non-redundant | paired with stability, against a size-matched random grouping |
| Same-painting ceiling (painting ids come from the split; check 3) | block theory by analogy (T2) | trivially 1 for groupings built from images | bounds image placement of viewer-level groupings |
| Anchor distance: variation of information to the source grouping | none | no external benchmark | L only; drift bound stated in advance |

**Known to mislead in this project** ([P]): silhouette and cluster clarity (§1, R5 row); topic predictability from the
image, anti-correlated with emotion capture (r = −0.70, −0.74); cluster lift on the groups, which rose 1.99 to 3.45
while lift through the heads stayed flat.

**Circularity.** T5 warns that E would be judged on placeability it trains on; the same holds for L's placeability and
non-redundancy terms. For each design, name in advance the criteria it trains on and judge it on the others.

**Two consequences for the queued checks** (*reader-inferred*, ours). First, check 1's informative contrast is the
method family: all nine Leiden told margins lie inside each other's intervals, as do the k-means ones, while Leiden beat
k-means by +0.48 to +0.88 at every matched count ([P] draft §5). A rank correlation over 17 cells would be driven by that
split, so pre-state the test as "does placeability rank each Leiden cell above its matched k-means control". Check 1
reads told margins once, to validate a criterion, not to pick a grouping; disclose it. Second, the sweep's picked cell
(graph k 40, resolution 1.0) was chosen by the largest reader margin on labelled episodes (draft §5), a choice R5 now
rules out. Either the pre-specified default (k 20, resolution 1.0, 41 groups; draft §4) or a reselection by the criteria
above should replace it; that is the user's call (§10).

## 7. Sibling-aware agreement and the stage-2 interface

**Formulas** (T5; all *reader-inferred* constructions, forms from the cited papers). Notation: p, q are image and
caption posteriors over K groups.

- **F1, centroid soft cosine** (form after Sidorov et al. <!--ref:sidorov_cys_2014--><!--anchor:section:abstract-->):
  S_jk = max(0, cos(μ_j, μ_k))^γ with μ the group centroid in the source space; a = pᵀSq. S = I gives the plain dot
  product. Not guaranteed PSD. About 1.4 GFLOP for 100,000 pairs at K = 120 (T5 arithmetic).
- **F2, co-membership**: (a) column cosine of soft assignments within a source; (b) cross-modal C = P_imgᵀP_txt over
  paired rows, row-normalised plus I. Risk of inflating random-pair agreement where placement is noisy.
- **F3, hierarchy kernel** (after Bertinetto et al. <!--ref:1912.09393--><!--anchor:section:abstract-->):
  S = Σ_ℓ w_ℓ A_ℓᵀA_ℓ over nested levels; PSD by construction; S = I when only the finest level has weight. Leiden
  levels need not nest; agglomerative merging of centroids guarantees nesting.
- **OT** (Kusner <!--ref:kusnerb15--><!--anchor:section:abstract-->; Cuturi <!--ref:1306.0895--><!--anchor:section:abstract-->):
  no exact reduction to the dot product (S = I gives the overlap Σ min(p_j, q_j)); about 50 times costlier; a check only.

**Recommendation** (ours, *reader-inferred*). F1 as the primary for check 2: one parameter, cheapest, fixed before any
labelled episode is scored. F3 built by agglomerative merging of the same centroids as the secondary, because it also
gives the bank a discrete sibling rule (below). F2(b) descriptive only: it learns which image group accompanies which
caption group, so it partly re-learns placement and mixes R2 with R3. If F2(b) is used, build C on the 10,000
scorer-train rows the heads were not fitted on ([P] draft Table 2) and never on support pairs.

**Interface options** (T6). H, hard ids per grouping (today, the control); H+X, hard ids plus negatives filtered by S;
Soft, posteriors with p_imgᵀSp_txt in episodes; S as soft targets; W, per-grouping sampling weights.

**How they connect** (*reader-inferred*, ours).

- **One S, two uses.** The reader's agreement and the bank's sibling-excluded negatives take the same S, frozen before
  either is scored. Stage 1 then hands stage 2 a triple per grouping: hard ids, S, heads.
- **F3 settles H+X's threshold.** T6 found no label-free threshold for excluding sibling negatives. With F3 the rule is a
  level: exclude negatives sharing an ancestor at level ℓ with the support group.
- **Matched controls in both places:** S = I for the reader (with the condition-free counterpart built on the same S,
  since B averages agreement over groupings) and H for the bank.
- **Calibration protects both.** An S that inflates shuffled-pair agreement would also over-exclude negatives; check 2's
  shuffled-pair gate covers both uses.
- **Soft** carries little affect from the image side ([P] 13.5% head accuracy). **W** has no label-free criterion; sample
  groupings uniformly (CACTUs <!--ref:1810.02334--><!--anchor:section:5-->; SE), accepting the per-aspect cost of row 5
  in §3. Cluster semantics may matter less than task variety: augmentation-built tasks matched CACTUs (UMTRA, Khodadadeh
  et al., 2019 <!--ref:1811.11819--><!--anchor:section:abstract-->; T6, A).
- **Keep a condition-free pathway**, since a conditioner can fit the bank and not the aspect (Ye et al.
  <!--ref:2011.14663--><!--anchor:section:V-B-->, relayed by T6; CACTUs' acknowledged pseudo-to-real gap).

## 8. First experiments

All on seed 42 for development; the final configuration once on fresh seeds 49, 50 and 51 pooled (brief §9). No grouping
is chosen by evaluation labels. CPU unless stated; GPU work takes the shared lock or runs on DAS6.

| Step | What | Matched control | Label-free check: continue if | Labels read |
|---|---|---|---|---|
| **0a** (check 3, seconds) | Share of row pairs from one painting in the same affect group, for k-means 64 and the Leiden default | random row pairs from different paintings; a size-matched random grouping | Decides where R2 effort goes: near the random rate, image-side affect has little to recover, so skip head-loss work for affect and test placing captions by GoEmotions directly (draft §6); clearly above it, step 3 has room | none |
| **0b** (check 1, minutes) | Placeability and head-posterior stability for the 9 Leiden cells and 8 k-means controls | shuffled-row pairs; k-means at matched counts | Adopt placeability as the within-source criterion if it ranks Leiden above its matched k-means in a pre-stated share of pairs; otherwise use stability plus non-redundancy only | told margins, once, to validate the criterion (disclosed) |
| **0c** (check 2, minutes) | F1 (primary), F3, F2 on the Leiden default grouping | S = I in the term and in its condition-free counterpart | same-row over shuffled-row agreement under S not below that under I | told and reader margins (development) |
| **1** (week 1 to 2) | Write the generic source menu first (user). Then P0 plus a style grouping: style features for 36,518 paintings (GPU), Leiden at pre-specified defaults, image and caption heads | a size-matched random grouping in the style slot (controls for a fourth option in the reader), plus the matched counterpart with B′ | variation of information to the CLIP image grouping clearly above that of a reseeded CLIP image grouping; stable; both heads above shuffled pairs | style-versus-genre profile of the one chosen grouping, once, disclosed; margins (development) |
| **2** (minutes, in parallel) | G as a dominance diagnostic: leidenalg multiplex shared partition over the affect, image and style layers | P0's single-layer partitions | proceed to G proper only if the shared partition is not markedly closer to one layer than to the others | none |
| **3** (week 2 to 3) | L on a frozen trunk as a ladder: (a) P0 logistic heads; (b) per-source heads with a SCAN or TEMI consistency term on the source's own graph plus a KL anchor; (c) (b) plus a cross-grouping non-redundancy penalty. Arm (b) runs only if 0a leaves room | (b) against (a) isolates the head loss; (c) against (b) isolates non-redundancy | judged on stability, the same-painting ceiling and anchor drift (bound stated in advance), not on placeability or non-redundancy, which (b) and (c) train | margins (development) |
| **4** (week 4) | One pre-registered configuration on seeds 49 to 51 pooled | cosine, RCA, B (B′) and its matched counterpart | none | the GO test |

The development bar is the draft's: reader margin at least +0.5 with a lower bound above 0 on seed 42, because every
fresh-seed test so far roughly halved the development margin ([P] draft §2). E and G proper are not scheduled before
the 10 November abstract.

## 9. Open risks and gaps

1. **Placement may stay the binding loss.** [P] Emotion lift 2.17 on the groups fell to 1.11 through the heads, and the
   sweep's better groups did not survive placement. If check 3 shows a low ceiling, no grouping redesign lifts the
   image side of affect.
2. **Style may be out of reach.** No label-free descriptor's movement-versus-genre balance is published; CSD starts from
   CLIP and was scored on artist retrieval; captions do not name style. Even a style-dominant grouping will be placed
   from images only.
3. **R1 against R2.** Placeability, as objective or cross-source criterion, favours content (§1, §3 row 2).
4. **Label leakage by selection.** Choosing a style source or a Leiden cell by labelled margins; the sweep pick already
   did. CLIP's pretraining data is undisclosed, so image overlap of CLIP-initialised encoders with WikiArt cannot be
   ruled out. [P] GoEmotions' labels name 6 of the 8 evaluation emotions (draft §1); hand-matching must be disclosed.
5. **Seed 42 overuse and a winner's curse.** [P] The sweep pick beat the untuned default by only +0.15 [−0.01, 0.31].
6. **Evidence depth.** Most literature is abstract level; the theory papers were read through fetch summaries; the two
   conditional-clustering papers are metadata only.
7. **No precedent** for fuse-then-split, and no identifiability route that splits style from genre without an extra
   signal.
8. **S inflation** would propagate from the reader into the bank.
9. **Time.** Five weeks to the abstract leave room for steps 0 to 4 and little else.

Synthesis limitations: we did not re-open papers or re-derive project numbers from per-anchor arrays; formulas and
experiment rules are untested; the tension inventory covers 11 pairs and is not a complete pairwise check.

## 10. Decisions that remain the user's

1. Adopt the ranking and the changed order (checks first, L before G, G as diagnostic, E to factor learning), or keep L
   and G side by side.
2. The generic source menu, written before hand-matched results, and whether Gram statistics are on it.
3. The hand-matched style source (CSD, Gram or ALADIN), and whether a CLIP-initialised encoder meets "not trained on
   WikiArt style labels".
4. The affect setting: the pre-specified Leiden default, or a reselection by label-free criteria.
5. Whether check 1 may read told margins to validate placeability (a disclosed, meta-level label read).
6. The primary S formula and its parameter rule; whether S goes into stage 2 (H+X) now or after check 2.
7. Whether G stays a diagnostic, or the slice-coupled variant is tried.
8. Whether E moves to the factor-learning discussion.
9. The time budget for steps 1 to 3 before 10 November.

## Appendix A. Citations used

Grades as in the header. "Bears on" lists designs or sections.

| Source | Id (slug) | Scan | Grade | Bears on |
|---|---|---|---|---|
| Niu, Dy, Jordan (2010) <!--ref:niu_icml_2010--><!--anchor:section:abstract--> | ICML 2010, pp. 831 to 838 | T1 | A | R1, R4, L |
| Miklautz et al. (2020), ENRC <!--ref:miklautz_aaai_2020--><!--anchor:section:abstract--> | AAAI 2020, DOI 10.1609/aaai.v34i04.5961 | T1 | A | R1, R4, L, E |
| Falck et al. (2021), MFCVAE <!--ref:2106.05241--><!--anchor:section:Table1--> | arXiv 2106.05241, NeurIPS 2021 | T1 | S | R1, R4, L |
| Gondek, Hofmann (2004; 2007) <!--ref:gondek_hofmann_kais_2007--><!--anchor:none:--> | KAIS 12, 1 to 24 | T1 | M | L |
| Cui, Fern, Dy (2007) <!--ref:cui_icdm_2007--><!--anchor:none:--> | ICDM 2007, DOI 10.1109/ICDM.2007.94 | T1 | M | R4, L |
| Kumar, Rai, Daumé (2011) <!--ref:kumar_nips_2011--><!--anchor:section:abstract--> | NIPS 2011 | T1 | A | G |
| Li et al. (2025), BMvC <!--ref:2501.02564--><!--anchor:section:abstract--> | arXiv 2501.02564 (preprint) | T1 | A | R1, G |
| Mucha et al. (2010) <!--ref:mucha_science_2010--><!--anchor:section:abstract--> | Science 328, 876 to 878 | T1 | A | G |
| leidenalg multiplex docs <!--ref:leidenalg_multiplex_docs--><!--anchor:section:multiplex--> | leidenalg.readthedocs.io | T1 | G | G |
| Wilson et al. (2017) <!--ref:1610.06511--><!--anchor:section:abstract--> | arXiv 1610.06511, JMLR | T1 | A | G |
| Tan et al. (2019), SCE-Net <!--ref:1908.08589--><!--anchor:section:Fig4--> | arXiv 1908.08589, ICCV 2019 | T1 | S | E |
| Ye, Shi, Zhan (2022), DiscoverNet <!--ref:2204.04053--><!--anchor:section:Table3--> | arXiv 2204.04053, CVPR 2022 | T1 | S | E |
| Locatello et al. (2019) <!--ref:1811.12359--><!--anchor:section:abstract--> | arXiv 1811.12359, ICML 2019 | T2 | A | R1, R4 |
| Locatello et al. (2020) <!--ref:2002.02886--><!--anchor:section:Theorem1--> | arXiv 2002.02886, ICML 2020 | T2 | S | painting signal |
| Khemakhem et al. (2020), iVAE <!--ref:1907.04809--><!--anchor:section:Theorem1--> | arXiv 1907.04809, AISTATS 2020 | T2 | S | provenance signal |
| Gresele et al. (2019) <!--ref:1905.06642--><!--anchor:section:abstract--> | arXiv 1905.06642, UAI 2019 | T2 | A | provenance signal |
| Lyu et al. (2022) <!--ref:2106.07115--><!--anchor:section:abstract--> | arXiv 2106.07115, ICLR 2022 | T2 | A | R2, E |
| von Kügelgen et al. (2021) <!--ref:2106.04619--><!--anchor:section:Theorem4.4--> | arXiv 2106.04619, NeurIPS 2021 | T2 | S | augmentation, style |
| Daunhawer et al. (2023) <!--ref:2303.09166--><!--anchor:section:Theorem1--> | arXiv 2303.09166, ICLR 2023 | T2 | S | R1, R2, L, E |
| Yao et al. (2024) <!--ref:2311.04056--><!--anchor:section:abstract--> | arXiv 2311.04056, ICLR 2024 | T2 | A | G, provenance |
| FactorCL (2023; authors not listed by the scan) <!--ref:2306.05268--><!--anchor:section:abstract--> | arXiv 2306.05268, NeurIPS 2023 | T2 | A | E |
| Van Gansbeke et al. (2020), SCAN <!--ref:2005.12320--><!--anchor:section:2.2--> | arXiv 2005.12320, ECCV 2020 | T3 | F | R2, L |
| Caron et al. (2020), SwAV <!--ref:2006.09882--><!--anchor:section:3--> | arXiv 2006.09882, NeurIPS 2020 | T3 | F | E |
| Asano et al. (2020), SeLa <!--ref:1911.05371--><!--anchor:section:3.5--> | arXiv 1911.05371, ICLR 2020 | T3 | F | L |
| Ji et al. (2019), IIC <!--ref:1807.06653--><!--anchor:section:3.1--> | arXiv 1807.06653, ICCV 2019 | T3, T5 | F | R2, check 1, E |
| Alwassel et al. (2020), XDC <!--ref:1911.12667--><!--anchor:section:4--> | arXiv 1911.12667, NeurIPS 2020 | T3 | F | R2, P0 |
| Adaloglou et al. (2023), TEMI <!--ref:2303.17896--><!--anchor:section:5--> | arXiv 2303.17896, BMVC 2023 | T3 | F | R2, P0, L |
| Li et al. (2024), TAC <!--ref:2310.11989--><!--anchor:section:4.3.2--> | arXiv 2310.11989, ICML 2024 | T3 | A (plus summary) | E |
| Kwon et al. (2023), IC\|TC <!--ref:2310.18297--><!--anchor:section:abstract--> | arXiv 2310.18297 (venue unconfirmed) | T3 | A | R1 |
| Zhou, Zhang (2022) <!--ref:2207.13364--><!--anchor:section:abstract--> | arXiv 2207.13364 | T3 | A | P0 |
| Somepalli et al. (2024), CSD <!--ref:2404.01292--><!--anchor:section:6--> | arXiv 2404.01292 (preprint) | T4 | S | style source |
| Gatys et al. (2016) <!--ref:gatys_cvpr_2016--><!--anchor:none:--> | CVPR 2016, DOI 10.1109/CVPR.2016.265 | T4 | M | style source |
| Johnson (2017) <!--ref:1611.05368--><!--anchor:section:4--> | arXiv 1611.05368 | T4 | S | style source |
| Geirhos et al. (2019) <!--ref:geirhos_iclr_2019--><!--anchor:section:abstract--> | ICLR 2019 | T4 | A | style source |
| Ruta et al. (2021), ALADIN <!--ref:2103.09776--><!--anchor:section:3--> | arXiv 2103.09776, ICCV 2021 | T4 | S | style source |
| Ruta et al. (2023), ALADIN-NST <!--ref:2304.05755--><!--anchor:section:4.2--> | arXiv 2304.05755 | T4 | S | style source |
| Wu, Nakashima, Garcia (2024), GOYA <!--ref:goya_jimaging_2024--><!--anchor:none:--> | J. Imaging 10(7):156 | T4 | M | style source |
| Ashton (2026) <!--ref:2608.14435--><!--anchor:section:7--> | arXiv 2608.14435, VISART workshop | T4 | S | R1, style |
| Zhang et al. (2024), Long-CLIP <!--ref:2403.15378--><!--anchor:section:3.1--> | arXiv 2403.15378, ECCV 2024 | T4 | S | backbone |
| Chen et al. (2023), ShareGPT4V <!--ref:2311.12793--><!--anchor:section:appendixA--> | arXiv 2311.12793 | T4 | S | backbone overlap |
| Achlioptas et al. (2021), ArtEmis <!--ref:2101.07396--><!--anchor:section:abstract--> | arXiv 2101.07396, CVPR 2021 (venue from listing) | T4 | S | captions |
| Garcia, Vogiatzis (2018), SemArt <!--ref:1810.09617--><!--anchor:section:abstract--> | arXiv 1810.09617 | T4 | A (plus summary) | captions |
| Ben-Hur et al. (2002) <!--ref:ben-hur_psb_2002--><!--anchor:none:--> | PSB 2002, 6 to 17 | T5 | M | R5 |
| Lange et al. (2004) <!--ref:lange_neco_2004--><!--anchor:none:--> | Neural Computation 16(6) | T5 | M | R5 |
| von Luxburg (2010) <!--ref:1007.1075--><!--anchor:section:abstract--> | arXiv 1007.1075 | T5 | A | R5 |
| Ben-David et al. (2006) <!--ref:ben-david_colt_2006--><!--anchor:section:abstract--> | COLT 2006 | T5 | A | R5 |
| Mourer et al. (2023) <!--ref:2006.08530--><!--anchor:section:abstract--> | arXiv 2006.08530, PAKDD 2023 | T5 | A | R5 |
| Arbelaitz et al. (2013) <!--ref:arbelaitz_pr_2013--><!--anchor:none:--> | Pattern Recognition 46(1) | T5 | M | R5 |
| Meilă (2007) <!--ref:meila_jmva_2007--><!--anchor:none:--> | JMVA 98(5) | T5 | M | R4 |
| Sidorov et al. (2014) <!--ref:sidorov_cys_2014--><!--anchor:section:abstract--> | Computación y Sistemas 18(3) | T5 | A | R3 |
| Bertinetto et al. (2020) <!--ref:1912.09393--><!--anchor:section:abstract--> | arXiv 1912.09393, CVPR 2020 | T5 | A | R3 |
| Kusner et al. (2015) <!--ref:kusnerb15--><!--anchor:section:abstract--> | ICML 2015, PMLR 37 | T5 | A | R3 |
| Cuturi (2013) <!--ref:1306.0895--><!--anchor:section:abstract--> | arXiv 1306.0895, NeurIPS 2013 | T5 | A | R3 |
| Hsu, Levine, Finn (2019), CACTUs <!--ref:1810.02334--><!--anchor:section:5--> | arXiv 1810.02334, ICLR 2019 | T6 | S | interface |
| Khodadadeh et al. (2019), UMTRA <!--ref:1811.11819--><!--anchor:section:abstract--> | arXiv 1811.11819 (venue unconfirmed) | T6 | A | interface |
| Ye, Han, Zhan (2020) <!--ref:2011.14663--><!--anchor:section:V-B--> | arXiv 2011.14663, TPAMI (year unconfirmed) | T6 | S (relayed via scan_N4) | E, interface |
| Huynh et al. (2022), FNC <!--ref:2011.11765--><!--anchor:section:abstract--> | arXiv 2011.11765 (venue unconfirmed) | T6 | A | R3, interface |
| Chen et al. (2022), IFND <!--ref:2106.03719--><!--anchor:section:abstract--> | arXiv 2106.03719, ICLR 2022 | T6 | A | R3, interface |
| Feng, Patras (2022), ASCL <!--ref:2207.11163--><!--anchor:section:abstract--> | arXiv 2207.11163, ICPR 2022 | T6 | A | interface |

## Appendix B. Unverified items and corrections

**Unverified, blocked, metadata-only or summary-only items used above** (flagged where used; status not upgraded).

- Metadata or search record only (grade M, `anchor:none`): Gondek and Hofmann (content from a search summary; abstract
  page did not load), Cui et al. (no content read), Gatys et al. (CVF page 403; Semantic Scholar record), GOYA (MDPI 403;
  method from a search summary), Ben-Hur et al., Lange et al., Meilă, Arbelaitz et al. (publisher pages blocked).
- Fetch-summary details not checked against the paper: Daunhawer et al.'s theorem locator, assumption wording and the
  48% to 80% discrete text-factor accuracy; the assumption lists of Locatello et al. (2020), iVAE and von Kügelgen et
  al.; MFCVAE's §5 limitation quote; SCE-Net and DiscoverNet figures; TAC's §3 and §4.3.2 details; CSD's Table 1;
  Johnson's 33.46% (class count reported as 70 against WikiArt's 27, unresolved); Ashton's numbers; ArtEmis's 15.8
  words (section not recorded, so its anchor is `none`); ShareGPT4V's appendix counts.
- Relayed numbers: Ye et al.'s Table IX and CACTUs' 73.36 against 96.29 come from `scan_N4.md` through T6; SCE-Net's
  Zappos 7.53 against 10.73 came from the project literature review (not used here).
- Venue not confirmed: CSD (arXiv preprint), BMvC, IC|TC, UMTRA, FNC, ALADIN-NST, StyleBabel; ArtEmis's CVPR venue from a
  listing; Ye et al.'s TPAMI year; Wilson et al.'s JMLR volume.
- Gray literature: leidenalg multiplex docs; CSD and ALADIN READMEs (weights, licence).
- Preprint, context only, not cited for a claim: arXiv 2605.19135 (T2).
- Seen only in search results, not used: CoMVC (Trosten et al., CVPR 2021; 403), DR-Tune, a fine-tuning claim of 92.1
  to 65.7 (T3); arXiv 2411.03755, 2604.05462, 2206.06593, 2102.08850 (T2); "Coarse-to-fine pseudo supervision" (T6).
- Unmeasured facts T4 named: maximum caption length and share over 77 tokens; image overlap of LAION-Styles, GOYA or
  Long-CLIP's pairs with our paintings; weights and licences of GOYA and ALADIN-NST.

**Corrections the scans made to the brief.** DiscoverNet is the method in "Identifying Ambiguous Similarity Conditions
via Semantic Matching" (Ye, Shi, Zhan, CVPR 2022) (T1). Gondek and Hofmann is ICDM 2004 or KAIS 2006/2007 (T1).
Hyvärinen and Morioka's methods are two papers (TCL, NeurIPS 2016; PCL, AISTATS 2017); GCL is Hyvärinen, Sasaki and
Turner (T2). Locatello et al. 2020 is ICML 2020 (T2). Swapped prediction across image and caption is our adaptation;
SwAV uses augmentations (T3). TAC uses retrieved WordNet nouns, not paired captions (T3). SCAN mines neighbours once and
keeps the lowest-loss of ten heads (T3). CSD's venue is unconfirmed; cite it as a preprint (T4). ArtEmis counts differ
by version (439K or 455K) (T4). CACTUs is ICLR 2019; UMTRA's title is "Unsupervised Meta-Learning for Few-Shot Image
Classification"; Meta-GMVAE has no arXiv id (T6).

**Corrections and notes from this synthesis.**

1. Brief §5 names hierarchical refinement as G's precedent; it nested affect inside content communities and fused
   nothing, so it is a precedent for conditional splitting (L), not for a shared partition (§3 row 4).
2. T4 left open which languages the 183,694 rows contain; the stage report (§2.2) states ArtELingo (English), so the
   multilingual truncation risk does not apply.
3. T6's "diversity beat quality" reads as "no aspect lost": SE's gains were about a third of each single-source
   specialist's (§3 row 5; affect report).
4. The Leiden sweep's pick was made by reader margin on labelled episodes, which R5 rules out for grouping choice (§6).
5. Two homonyms: "PCL" in T2 is permutation-contrastive learning (Hyvärinen and Morioka); in T6 it is prototypical
   contrastive learning (Li et al., 2021).
