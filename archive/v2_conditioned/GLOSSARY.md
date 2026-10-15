# CoSiR glossary of the conditioned line (archived)

This is the conditioned line's glossary (CoSiR v2, rounds 1 to 9), archived on 2026-10-15; the live [GLOSSARY.md](../../GLOSSARY.md) keeps the terms that are not specific to that line.

The project's terms, written by the grill (research loop step 5) and used by briefs and reports. Terms of the
method-A era (factors, agreement rule, pseudo-partition, KISSME, …) are in Appendix A of the
[CVPR plan](docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md). Created 2026-10-09 for round 6.

| Term | Meaning |
|---|---|
| aspect, value | a respect in which items can be similar (emotion, style, genre) / one setting of it (sad, Baroque, landscape) |
| aspect episode | one ranking problem: a query (image or caption), 4 support pairs, 4 contrast pairs, and 13 candidates in the other modality. Each support pair is an image and a caption from two paintings that share one value of aspect A; the 4 pairs show 4 different values, never the query's own (value-disjoint). Contrast pairs are built the same way for aspect B. The target is the candidate that shares the query's value of the aspect the supports show |
| aspect pair | the two aspects of an episode: emotion × style, emotion × genre or style × genre |
| condition a / b | supports show aspect A (target p_A) / supports and contrasts swapped (target p_B) |
| R@1 | share of rankings where the target ranks strictly first, in percentage points |
| condition gain | R@1 under the correct condition minus R@1 for the same target under the swapped one; 0 for any condition-free scorer |
| condition-free | a scorer whose scores are identical under condition a and b, so its condition gain is 0. It may use the episode's examples (B centres on all 8 example items), but not which side shows which aspect. B, B′(A0), B′(A1) and every matched control are condition-free |
| grouping, head | a label-free partition of the training rows (affect, image, caption, CSD style) / a classifier on frozen CLIP features predicting an item's group |
| reader | round 1's learned model that reads an episode's support and contrast pairs and says which grouping they share |
| R1 | the reader's weighted grouping term, added to the score when the reader is confident |
| AFF (affect steering) | R1 with its term added only when the reader picks the affect grouping; the current best method |
| matched control (counterpart) | a scorer's identical score with only the condition removed (constitution C2) |
| B, B′(A0), B′(A1) | the condition-free scorers: B the project's best before the reader; B′(A0) B rebuilt on the reader's three groupings; B′(A1) B′ with the CSD style grouping added, the strongest condition-free scorer so far |
| bar margin | R@1 of a method minus the strongest of B, B′(A0) and its matched control |
| selection, scorer-train, held rows | ArtELingo's development, training and final-test rows (held: 12,281 paintings) |
| seed 42, fresh seeds | the development episode seed / new episode seeds for a test (constitution C4) |
| held read | one pre-registered scoring of the held rows by one final method and backbone, ledgered (constitution C5) |
| describe-then-score comparator | *(new, round 6)* a vision-language model states in a phrase what an episode's supports share and its contrasts lack; CRL (NeurIPS 2025) turns the phrase into a text basis on frozen CLIP, where query and candidates are compared; fused with cosine like every baseline |
| RCA | relevant component analysis: a classic pair-metric baseline that re-weights CLIP features by the covariance of the episode's example-pair differences (`rca_term`); condition-aware, fused with cosine at a frozen λ like every baseline |
| DTS, DTS-CF, DTS-N | the describe-then-score comparator, its matched control, and its variant told the true aspect name (a privileged ceiling) |
| Holm | Holm's step-down correction across a read's checks, holding the family-wise error at the stated level (round 6: one-sided α = 0.025 over P1 to P7) |
| ceiling check | *(new, round 7)* a told run that uses true labels to build one grouping, to bound what any label-free grouping in its place could give; it can stop a round, never pick |
| content grouping | *(new, round 7)* the reader's CLIP image k-means grouping (64 groups), used as the "content" that design L's style groupings are made to cut across |
| extended AFF | *(new, round 7)* AFF on a reader with a fourth, style grouping, its gate open when the reader picks affect or that style grouping |
| own assignment | *(new, round 8)* placing a gallery image into a grouping by the grouping's own map (its head or nearest centroid), not by a CLIP head fitted to copy the grouping |
| paired-image reference | *(new, round 8)* a privileged ceiling on caption placement: each caption takes its own painting's image-head posterior; no real caption has this, so it is not a method ([step 3](src/test/20261129_r8_grouping/step3_caption_ref/step3_caption_ref.md)) |
| positive pair | *(new, round 8)* two items a contrastive grouping head pulls together; in round 8, views of one painting or the nearest CLIP neighbour outside the query's content group |
| held tiers | *(new, round 8)* the reading of a held read with AFF's seven checks plus "beats AFF": Tier 1, all eight pass (replaces AFF); Tier 2, beats AFF but fails a condition-gain check; Tier 3, the seven pass but not AFF (AFF stays in 2 and 3) |
| fixed recipe | *(new, round 9)* AFF re-fitted on a new backbone or dataset with the same code and settings: every part above the frozen encoder is fitted again, and the same automatic seed-42 pick is re-run; no human choice |
| recipe replay | *(new, round 9)* the fixed recipe run on CLIP from scratch; it must give back CLIP's frozen files and picks, which shows the recipe is the procedure AFF went through |
| switched off | *(new, round 9)* the fixed recipe's seed-42 pick sets λ_a = 0 in both halves, so AFF equals B on that backbone; its held reads are then not run |
| language check | *(new, round 9)* the ArtELingo-28 read: AFF trained on English ArtELingo, scored on captions from 27 other languages, machine-translated to English, on 746 paintings (248 held + 498 new) |
| two-level test | *(new, round 9)* a read whose rule tests P4 and P6 first (Holm over two) and the other five checks only if both pass (Holm over five); serial gatekeeping keeps the family-wise error at α |
| development probe | *(new, round 9)* CLIP AFF re-fitted on a new dataset and scored on its development split only, to see whether its aspect episodes work; no claim |
