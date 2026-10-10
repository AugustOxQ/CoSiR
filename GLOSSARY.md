# CoSiR glossary

The project's terms, written by the grill (research loop step 5) and used by briefs and reports. Terms of the
method-A era (factors, agreement rule, pseudo-partition, KISSME, …) are in Appendix A of the
[CVPR plan](docs/superpowers/specs/2026-10-02-cosir-v2-cvpr-publication-plan-design.md). Created 2026-10-09 for round 6.

| Term | Meaning |
|---|---|
| aspect, value | a respect in which items can be similar (emotion, style, genre) / one setting of it (sad, Baroque, landscape) |
| aspect episode | one ranking problem: a query (image or caption), 4 support pairs that share its value of aspect A, 4 contrast pairs that share its value of aspect B, and 13 candidates in the other modality; the target is the candidate sharing the aspect the supports show |
| aspect pair | the two aspects of an episode: emotion × style, emotion × genre or style × genre |
| condition a / b | supports show aspect A (target p_A) / supports and contrasts swapped (target p_B) |
| R@1 | share of rankings where the target ranks strictly first, in percentage points |
| condition gain | R@1 under the correct condition minus R@1 for the same target under the swapped one; 0 for any scorer that ignores the examples |
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
