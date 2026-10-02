# Revision Roadmap: immutable core

CoSiR v2 CVPR plan review, panel `cosir-v2-cvpr-plan-review-round-1`, review round 1. Editorial decision: **Major Revision** (mechanical; F2 selected; see `editorial_decision.md`).

## Bindings and validation

- **Schema**: `revision-roadmap/1.0`, the reviewer-owned core only. Author triage, author reasons, display order, work order and claim-strength or collateral authorization are not part of this core; they belong in the separate author-adjudication sidecar built later by `scripts/revision_roadmap.py`.
- **Base draft**: `roadmap_base.md`, SHA-256 `16c9912faec08401d9cd71162e7ca192f181b30ea58d238a43bcbbc74b88487f`. The manuscript (`manuscript.md`, SHA-256 `d1ebaef8055ca882db4a1c8ba9090ae608a735d3ed96906d4b140eaed5a62342`) carries no block anchors, and no block manifest was supplied with the panel inputs. To give `proposed_targets` exact block ids without editing `manuscript.md`, the synthesizer ran `scripts/ars_anchorize_draft.py` on a copy. That copy is `roadmap_base.md`: the manuscript plus 321 `block:BNNNN` marker lines, and the anchorizer is content-neutral by its tested contract. `manuscript.md` is unchanged.
- **Block manifest**: `roadmap_base.block-manifest.json`, SHA-256 `5bece1904c274f793e305e2fd93e5a84d5e053c0688edbab775b984f00e22c4e` (321 blocks).
- **Validation**: `scripts/revision_roadmap.py validate-roadmap revision_roadmap.json --base roadmap_base.md --block-manifest roadmap_base.block-manifest.json` printed `revision roadmap ok`. The JSON at the end of this file is that validated object.
- **Counts**: 58 items; 23 must_fix, 19 should_fix, 16 consider.

## Conventions

- **Source order** (immutable; never a rank): seat order EIC, R1, R2, R3, DA, then `ordinal`, then `subclaim_ordinal`, then channel. `ordinal` is the card's W number. For the DA, CRITICAL C1 is ordinal 1 and MAJOR M1 to M5 are ordinals 2 to 6. Editorial-channel lists take the ordinal after the seat's last finding: EIC 17, R2 11, R3 9, DA 7. `subclaim_ordinal` 0 means the whole finding; n of 1 or more means its n-th decomposed sub-claim.
- **Transport refs**: `R<n>` numbers the must_fix items in source order and `S<n>` the remaining items in source order. Neither is a work rank.
- **Severity, anchor and confidence** are transported from each item's driving finding, named in `severity_source`. Corroborating seats keep their own values in `corroborating_sources`. Where severities differ, the item is a SPLIT and the arbitration is recorded in the letter.
- **Obligation rule** (from the decision contract):
  - `must_fix`: the driving finding is cited by its seat as a ground of that seat's score on D1, D2, D3 or D6, the mandatory dimensions whose verdicts fire F2 and F3 (EIC W1 to W6; R1 W1 to W5, W11, W12; R2 W1 to W6), or the item is the validated DA C1.
  - `should_fix`: other Major findings; findings a seat ties to D4 or D5, which fire only F5; Minor findings the EIC tags to D6; and Minor findings raised by two or more non-DA seats.
  - `consider`: everything else.
- **[CONSENSUS-LEVEL-ENUM-GAP]**: the schema's `consensus_level` enum has no value for a two-seat corroborated or a single-reviewer first-round finding. Those items carry `SINGLE-VERIFIER`, and their descriptions state the exact seat count.
- **Sub-claims**: SC-1 to SC-53, as in the Step 1b inventory of the letter. DA-only items (REV-55 to REV-57) and editorial lists carry no sub-claim id (shown as —).

## Required Revisions (Must Fix)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Consensus | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | REV-01: Condition-attributable GO statistic and condition-removed control | SC-1, SC-2 | major | text: §5.1 Common protocol "R@1, the mean over both directions and all conditions / one sharing aspect A with the anchor (p_A), one sharing aspect B (p_B), and 11" | 4 — core expertise: shortcut analysis of episodic retrieval benchmarks | EIC W1.1, EIC W1.2, R1 W1.1, R1 W1.2 | SINGLE-VERIFIER | must_fix | section: §6 Go/no-go rule, §8 Tier-1 controls, §10 primary comparisons | claim_scope_unsupported → claim: K2 and the GO decision (§6, §10) |
| R2 | REV-02: K3 has no outcome-neutral branch while current evidence favours names on emotion | SC-3, SC-4 | major | table: aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14 | 4 — significance is core to my remit; numbers read from the appended spike | EIC W2.1, EIC W2.2, R2 W5.2, R2 W5.3, R3 W3.2, R3 W3.3, DA M3.3 | CONSENSUS-3 | must_fix | section: §4 K3 row and Fallback, §12 R-names row | claim_scope_unsupported → claim: K3 and the motivation of C1 |
| R3 | REV-03: A use case in which pairs exist but the aspect cannot be named | SC-5 | major | table: aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14 | 4 — significance is core to my remit; numbers read from the appended spike | EIC W2.3, R3 W6.1 | SPLIT | must_fix | section: §1 goal paragraph and §3 problem definition | claim_scope_unsupported → claim: significance of C1 |
| R4 | REV-04: K5 is decidable only through a stretch protocol | SC-6 | major | text: §5.4 example protocol "It is not comparable with published numbers." | 4 — benchmark expectations in this subfield are core to my remit | EIC W3.1, R2 W1.2, DA M4.2 | SINGLE-VERIFIER | must_fix | section: §5.4 GeneCIS protocols, §11 E7, §4 K5 row | evidence_gap_remains → claim: K5 |
| R5 | REV-05: GeneCIS measures transfer, not the cross-modal task | SC-7 | major | text: §5.4 example protocol "It is not comparable with published numbers." | 4 — benchmark expectations in this subfield are core to my remit | EIC W3.2, DA M4.3 | SINGLE-VERIFIER | must_fix | sentence: §3 'Against GeneCIS' paragraph and §5.4 | claim_scope_unsupported → claim: K5 |
| R6 | REV-06: Complete the novelty search before the title and abstract are fixed | SC-8 | major | text: novelty-check appendix, "Searched, nothing found" paragraph "This is a negative result from an incomplete search (no web search engine, Semantic Scholar rate-limited, no Google Scholar)" | 4 — the gap is documented by the authors; I cannot tell whether a missed paper exists | EIC W4 | SINGLE-VERIFIER | must_fix | other: §11 schedule: a literature-completion row before E15 | claim_scope_unsupported → claim: C1 novelty statement |
| R7 | REV-07: In-context alternatives that use the same examples | SC-9 | major | absence: §8 baselines table and §10 primary comparisons — expected the in-context baselines of novelty-check §5 Table 2 (an embedder or MLLM given the same support and contrast pairs, and verbalise-then-name) as scheduled baselines; checked §8 tiers 1 to 3 and the stretch row, §10, §11 E1, E10 and E17 | 4 — what the 2027 readership will expect is core to my remit | EIC W5, R2 W2.1 | SINGLE-VERIFIER | must_fix | re_analysis: §8 Tier 2 and §11 E10 on existing episodes | evidence_gap_remains → claim: C2 and the significance of C1 |
| R8 | REV-08: Define the NO-GO fallback paper | SC-10 | major | text: §4 Fallback "If K2 fails, K1, K3 and C3 could carry a task, benchmark and analysis paper but not the method paper." | 4 — storyline robustness across branches is core to my remit | EIC W6 | SINGLE-VERIFIER | must_fix | section: §4 Fallback paragraph | interpretive_ambiguity_remains → section: §4 Fallback |
| R9 | REV-10: C3's 'belongs to the data' outruns the backbone test | SC-12 | major | text: §4 C3 and backbone check Result 1 "the effect persists across four backbones / a better encoder cannot read what the pixels or words do not" | 3 — adjacent: label-noise ceilings; construct validity is Reviewer 3's remit | EIC W8, R1 W5.1, R2 W6.1, R3 W1.1, DA M1.1 | SPLIT | must_fix | re_analysis: §4 C3 wording and the E12 C3 analysis with SemArt curator text as a protocol contrast | claim_scope_unsupported → claim: C3 |
| R10 | REV-11: K2 is worded beyond the pre-declared tests | SC-13 | major | text: §4 K2 and §10 "the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive." | 4 — core expertise: aligning pre-registered tests with claim wording | EIC W9, R1 W4.1, DA M2.1 | SPLIT | must_fix | section: §4 K2 row, §10 primary comparisons, E13 pre-registration | claim_scope_unsupported → claim: K2 |
| R11 | REV-20: Cluster the bootstrap over reused items | SC-21 | major | text: §10 Statistics "paired bootstrap over anchors (5,000 resamples)" | 4 — core expertise: cluster-robust bootstrap for ranking metrics | R1 W2 | SINGLE-VERIFIER | must_fix | re_analysis: §10 statistics: resampling unit for every pre-declared comparison and the GO test | claim_scope_unsupported → claim: K2 and the GO decision |
| R12 | REV-21: An equivalence margin for 'matches' | SC-22 | major | text: §4 claims table, K3 "Examples beat naming the aspect on subjective aspects and match it on objective ones" | 5 — core expertise: equivalence testing for matches-the-baseline claims | R1 W3, DA M3.1 | SINGLE-VERIFIER | must_fix | section: §10 statistics and E13 pre-registration | claim_scope_unsupported → claim: K3 |
| R13 | REV-22: K7 is a descriptive comparison | SC-23 | major | text: §4 K2 and §10 "the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive." | 4 — core expertise: aligning pre-registered tests with claim wording | R1 W4.2, DA M2.2 | SINGLE-VERIFIER | must_fix | section: §10 primary comparisons | claim_scope_unsupported → claim: K7 and C2 |
| R14 | REV-23: Per-viewer labels cap the image-side emotion probe | SC-24 | major | text: §4 C3 and backbone check Result 1 "the effect persists across four backbones / a better encoder cannot read what the pixels or words do not" | 3 — adjacent: label-noise ceilings; construct validity is Reviewer 3's remit | R1 W5.2, R2 W6.2, R3 W1.2, DA M1.2 | CONSENSUS-3 | must_fix | re_analysis: E12 probes on scorer-train rows | claim_scope_unsupported → claim: C3 |
| R15 | REV-28: Define the multiplicity family | SC-29 | minor | absence: §10 Statistics — expected a declared family and correction or intersection rule for the primary comparisons across datasets, backbones and aspect types; checked §4 claims table, §10, §11 E13, §12 risks | 4 — core expertise: multiplicity in pre-registered ML comparisons | R1 W11 | SINGLE-VERIFIER | must_fix | section: §10 primary comparisons and E13 | claim_scope_unsupported → claim: K2 and K3 |
| R16 | REV-29: Specify the power calculation | SC-30 | minor | text: §10 "come from a power calculation on selection variance (default 4,096 anchors per aspect pair)" | 4 — core expertise: power analysis for episodic benchmarks | R1 W12 | SINGLE-VERIFIER | must_fix | re_analysis: E13 power simulation | method_reproducibility_unresolved → section: §10 Statistics |
| R17 | REV-35: An outcome-independent criterion for subjective and objective aspects | SC-36 | major | text: §4 claims table K3 "Examples beat naming the aspect on subjective aspects and match it on objective ones" | 3 — core expertise: text-conditioned similarity baselines; the direction is my reading of a single spike on untrained factors | R1 W18, R2 W5.1, R3 W3.1, DA M3.2 | SPLIT | must_fix | section: §4 K3, §10 per-aspect-type declaration, §11 E10 and E13 | claim_scope_unsupported → claim: K3 |
| R18 | REV-37: K5's bar uses the wrong GeneCIS column | SC-38 | major | text: §5.4 item 2 "published frozen ViT-B/32 results: SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1" | 4 — core expertise: GeneCIS and zero-shot CIR reporting; numbers taken from the manuscript's own Table 2 | R2 W1.1, DA M4.1 | SINGLE-VERIFIER | must_fix | sentence: §5.4 item 2 and §4 K5 row | claim_scope_unsupported → claim: K5 |
| R19 | REV-38: Restore the per-episode probe and the Tip-Adapter cache | SC-39 | major | absence: §8 baseline table and §10 pre-declared primary comparisons — expected a required, non-stretch baseline that gives an instruction embedder or open MLLM the same support and contrast pairs in context, and a verbalise-then-name baseline; checked §8 tiers 1 to 3 and the stretch row, §10 statistics, §11 E10 and E17, novelty check §5 Table 2, literature review §3 Table 8 | 4 — core expertise: conditional similarity baselines and instruction-following multimodal embedders | R2 W2.2 | SINGLE-VERIFIER | must_fix | re_analysis: §8 Tier 1 on existing episodes | evidence_gap_remains → claim: K2 |
| R20 | REV-39: K7 needs a sparse-autoencoder basis | SC-40 | major | text: §8 Tier 1 unsupervised-bases row "PCA-32/64, NMF-32 and SpLiCE sparse concept codes instead of our factors" | 3 — core expertise: sparse concept codes on CLIP; whether SAE latents work with the rule at 4 plus 4 pairs is untested | R2 W3, DA M5.1 | SINGLE-VERIFIER | must_fix | re_analysis: K7 table: a jointly trained TopK SAE basis; E11 split-dictionary ablation | claim_scope_unsupported → claim: K7 and C2 |
| R21 | REV-40: 'No labels' understates GoEmotions distant supervision | SC-41 | major | text: literature review §5 item 5 "a supervised GoEmotions classifier that names 6 of the 8 emotions" | 4 — core expertise: distant supervision in affective vision-language work; the label overlap is checkable in both taxonomies | R2 W4.1, R3 W7 | SPLIT | must_fix | re_analysis: §4 C2 wording, §5.1 Splits and emotion results split by GoEmotions coverage | claim_scope_unsupported → claim: C2 |
| R22 | REV-41: ArtELingo partitions are hand-matched to the evaluation aspects | SC-42 | major | text: literature review §5 item 5 "a supervised GoEmotions classifier that names 6 of the 8 emotions" | 4 — core expertise: distant supervision in affective vision-language work; the label overlap is checkable in both taxonomies | R2 W4.2 | SINGLE-VERIFIER | must_fix | re_analysis: emotion aspect with and without the affect partition | claim_scope_unsupported → claim: C2 |
| R23 | REV-55: Held-out-aspect test for the mechanism claim (DA C1, VALIDATED) | — | critical | text: §6 pseudo-partition table, ArtELingo row, against §4 C2 "GoEmotions affect k-means of captions (emotion-like; distant supervision); backbone image k-means (style-like; adjusted mutual information 0.32 with style); caption-content k-means (genre-like) / is the prior that makes it possible to estimate a similarity from just four example pairs" | 4 — adjacent expertise: few-shot and metric-learning evaluation design | DA C1 | DA-CRITICAL | must_fix | re_analysis: a held-out-aspect run on ArtELingo and generic partitions on CUB | claim_scope_unsupported → claim: C2 mechanism claim and the motivation of C1 |

## Suggested Revisions (Should Fix and Consider)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source | Consensus | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | REV-09: Novelty statement counts protocol choices as task properties | SC-11 | minor | text: §4 C1 novelty statement "evaluated in both directions with a paired swap test" | 4 — novelty framing is core to my remit | EIC W7 | SINGLE-VERIFIER | should_fix | sentence: §4 C1 novelty statement | claim_scope_unsupported → claim: C1 novelty statement |
| S2 | REV-12: Release plan for the task and benchmark | SC-14 | minor | absence: plan body §4 to §12 — expected a release plan for the aspect-episode builder, episode files and split lists; checked §4 C1 and the fallback, §5.1 to §5.4, the §10 ledger, §11 E0 and E17, §12 R-licence | 4 — a standard expectation for new-task papers at vision venues | EIC W10 | SINGLE-VERIFIER | should_fix | section: §11 E17 and §12 R-licence | reporting_requirement_unmet → section: §11 E17 |
| S3 | REV-13: A full cut order that keeps a never-read test set | SC-15 | minor | text: §12 R-time "staged priority; SemArt moves to the supplementary first" | 3 — planning judgement; the compute estimates are the authors' | EIC W11 | SINGLE-VERIFIER | should_fix | sentence: §12 R-time row | interpretive_ambiguity_remains → section: §12 R-time |
| S4 | REV-14: Final review falls after the abstract freeze | SC-16 | minor | text: §11 E15 and E16 rows "title and abstract fixed Nov 7 / Nov 11 to 14" | 4 — read directly from the schedule | EIC W12 | SINGLE-VERIFIER | should_fix | sentence: §11 E15 and E16 rows | claim_scope_unsupported → claim: abstract headline numbers |
| S5 | REV-15: Main-paper outline and page budget | SC-17 | minor | absence: plan body — expected a main-paper outline with a page budget and the planned figures and tables; checked §4, §5.1, §10, §11 E15 and E17, §12, Appendix B | 4 — exposition and venue conventions are core to my remit | EIC W13 | SINGLE-VERIFIER | should_fix | section: a new outline section after §11 | editorial_conformance_unmet → manuscript: main-paper outline |
| S6 | REV-16: Project vocabulary and internal recipes | SC-18 | minor | text: §8 Tier-1 row for C0, SE and R3 "our earlier factor recipes" | 3 — a judgement about exposition, not correctness | EIC W14 | SINGLE-VERIFIER | should_fix | section: §8 Tier-1 internal-recipe row and Appendix A | reader_traceability_reduced → section: Appendix A glossary |
| S7 | REV-17: Dates that blur the order of evidence | SC-19 | minor | text: §2.4 opening "Six investigations, all on selection rows (held rows untouched), changed the plan." | 4 — read directly from the document | EIC W15 | SINGLE-VERIFIER | should_fix | section: §2.4, §5.3 and the held ledger | reader_traceability_reduced → section: §2.4 |
| S8 | REV-18: 'Ceiling' names a drifting diagnostic | SC-20 | minor | text: aspect-episode spike appendix, caveats "It is a diagnostic, not a bound in the strict sense; a better probe could score higher." | 4 — terminology and claim framing are core to my remit | EIC W16, R1 W8 | SINGLE-VERIFIER | should_fix | sentence: §5.2, §6 Strong GO and §12 R-ceiling | interpretive_ambiguity_remains → section: §6 Strong GO |
| S9 | REV-19: EIC minor issues | — | n/a (editorial list) | n/a (editorial list) | n/a | EIC editorial list | SINGLE-VERIFIER | consider | sentence: figure captions and §5.1 | editorial_conformance_unmet → manuscript: figure captions; §5.1 swap definition |
| S10 | REV-24: GeneCIS is read during development | SC-25 | major | text: §11 E7 "GeneCIS focus attribute: example protocol, COCO-trained factors, baselines; text protocol as stretch / GeneCIS table" | 3 — core expertise: test-set reuse audits; E7's intended output is not fully specified | R1 W6 | SINGLE-VERIFIER | should_fix | section: §11 E7, E9 and E10 rows; §10 ledger | method_reproducibility_unresolved → dataset: GeneCIS focus attribute |
| S11 | REV-25: Best-of-ten selection still inflates the GO test | SC-26 | minor | text: §6 Go/no-go "from the same rows, so picking the best of about ten runs does not inflate the result" | 4 — core expertise: selection bias in model selection | R1 W7 | SINGLE-VERIFIER | consider | section: §6 Go/no-go Picking and GO bullets | interpretive_ambiguity_remains → section: §6 Go/no-go |
| S12 | REV-26: Swap success: scorer-dependent null and unit | SC-27 | minor | text: aspect-episode spike Result 2 "about 50% for any ordering: it reached 51.6 and 51.8 while R@1 was 8.0 and 8.6" | 4 — core expertise: metric design for paired conditional tests | R1 W9 | SINGLE-VERIFIER | consider | sentence: §5.1 secondary metrics | interpretive_ambiguity_remains → section: §5.1 |
| S13 | REV-27: Training-seed variance in the headline CI | SC-28 | minor | text: §10 Seeds "The headline is the 3-seed mean, with its CI from bootstrapping anchors" | 4 — core expertise: seed variance in ML evaluation | R1 W10 | SINGLE-VERIFIER | consider | section: §10 Seeds | reporting_requirement_unmet → section: §10 Seeds |
| S14 | REV-30: Baseline tuning grids and the grid-edge rule | SC-31 | minor | text: support-baseline spike, Caveats "every pick for the prototype and probe terms sat at that edge" | 3 — core expertise: tuning parity; the baseline grids may exist outside the plan | R1 W13 | SINGLE-VERIFIER | consider | section: §8 baselines | evidence_gap_remains → claim: K2 |
| S15 | REV-31: One score form and its transfer across datasets | SC-32 | minor | text: §8 with §6 against the aspect spike "they are chosen on the development split and frozen. GeneCIS, which has no development / s = β·cos + Σ_l w_l a_I,l(x) a_T,l(y) / The final score is z(cos) + λ · z(term)" | 3 — core expertise: score fusion; the final form may be intended but is not stated | R1 W14 | SINGLE-VERIFIER | consider | section: §6 model score and §8 fusion weights | method_reproducibility_unresolved → claim: K5 |
| S16 | REV-32: Exclusivity and third-aspect balance of candidates | SC-33 | major | text: §5.3 and §5.1 "answer the objection that two aspects make the condition a binary switch / condition A uses S = P_A, C = P_B; condition B swaps them" | 3 — core expertise on the construct; the size of the effect in these datasets is untested | R1 W15, R3 W4.1 | SPLIT | should_fix | re_analysis: E0 episode module: exclusivity rule and third-aspect rates | interpretive_ambiguity_remains → section: §5.1 aspect episodes |
| S17 | REV-33: A guessed genre name handicaps the names baseline | SC-34 | minor | text: §5.3 "is inferred by elimination and alphabetical order" | 3 — adjacent: label provenance | R1 W16 | SINGLE-VERIFIER | consider | sentence: §5.3 genre id list | evidence_gap_remains → claim: K3 on genre |
| S18 | REV-34: Backbone-appendix prose disagrees with its tables | SC-35 | minor | table: backbone check Result 3, Primary colour (15, 21.0) row, SigLIP 2 cell 61.9 / 64.5 | 5 — direct recomputation from the manuscript's tables | R1 W17 | SINGLE-VERIFIER | consider | sentence: backbone-check appendix prose | reader_traceability_reduced → section: backbone-check appendix |
| S19 | REV-36: CUB final-test species were read in the backbone check | SC-37 | minor | text: backbone check Result 3 "Retrieval sanity check, 5,794 test images, first caption each" | 3 — inference: 5,794 matches CUB's standard test split size | R1 W19 | SINGLE-VERIFIER | consider | re_analysis: E0 CUB third-aspect choice and C3's CUB evidence on training species | method_reproducibility_unresolved → dataset: CUB-200-2011 unseen-species test split |
| S20 | REV-42: SEARLE 14.4 is given two backbones | SC-43 | minor | text: literature review §2.2 pitfalls "SEARLE at ViT-L/14 circulates as 14.4 (CIReVL's run) and 12.26 (LinCIR's run)" | 4 — internal inconsistency, verifiable in the manuscript | R2 W7 | SINGLE-VERIFIER | consider | sentence: literature review §2.2 Table 2 and pitfalls paragraph | reader_traceability_reduced → table: literature review Table 2 |
| S21 | REV-43: The agreement rule's lineage | SC-44 | minor | text: Appendix A glossary "the classic ancestors of our rule and score" | 4 — core expertise: metric learning from pairs | R2 W8 | SINGLE-VERIFIER | consider | sentence: §4 C2 and Appendix A glossary | interpretive_ambiguity_remains → section: Appendix A glossary |
| S22 | REV-44: C-STS already pairs contrasting conditions | SC-45 | minor | text: novelty check §2 "we found no paired swap test (f) anywhere" | 4 — core expertise: conditional similarity; reference checked at its arXiv page | R2 W9 | SINGLE-VERIFIER | consider | sentence: novelty-check §2 and §4 C1 | claim_scope_unsupported → claim: C1 novelty statement |
| S23 | REV-45: Multiview triplet embedding is missing | SC-46 | minor | absence: literature review §2.1 and §6, novelty check thread 1 and §6 — expected multiview triplet embedding (Amid and Ukkonen, ICML 2015) as prior work that discovers several attribute-specific notions of similarity without attribute labels; checked literature review Table 1, the §2.1 paragraph on learning conditions without labels, both reference lists, novelty check Table 1 | 3 — core expertise: conditional similarity lineage; relevance is my judgement | R2 W10 | SINGLE-VERIFIER | consider | sentence: literature review §2.1 and the novelty-check must-cite list | reader_traceability_reduced → section: literature review §2.1 |
| S24 | REV-46: R2 minor issues and search lead | — | n/a (editorial list) | n/a (editorial list) | n/a | R2 editorial list | SINGLE-VERIFIER | consider | sentence: Appendix A glossary | reader_traceability_reduced → section: Appendix A glossary |
| S25 | REV-47: Probe genre before using it as the symmetric control | SC-47 | major | text: §4 C3 and backbone check Verdict "the effect persists across four backbones, so it belongs / so the modality asymmetry is a property of the data" | 4 — core expertise: affective annotation protocols; adjacent: the CV probes | R3 W1.3, DA M1.3 | SINGLE-VERIFIER | should_fix | re_analysis: genre probes on scorer-train rows | evidence_gap_remains → claim: C3 within-dataset control |
| S26 | REV-48: Inter-annotator agreement and a human noise ceiling | SC-48 | major | absence: §2.3, §5.1, §5.2 and §10 — expected an inter-annotator agreement statistic and a human noise ceiling for the emotion labels that define episode targets; checked §2.3 to §5.3, §8, §10 to §12, Appendix A, and the caveats of the support-baseline spike, aspect-episode spike and backbone check | 5 — core expertise: annotator disagreement in emotion labels | R3 W2 | SINGLE-VERIFIER | should_fix | re_analysis: existing ArtELingo rows | evidence_gap_remains → dataset: ArtELingo emotion labels |
| S27 | REV-49: Subjectivity is conflated with nameability | SC-49 | major | text: §4 K3 and §12 R-names "Examples beat naming the aspect on subjective aspects and match it on objective ones / narrow K3 to where names fail (subjective aspects)" | 4 — core expertise: similarity and naming in cognitive psychology | R3 W3.4 | SINGLE-VERIFIER | should_fix | section: §4 K3 and E13 | interpretive_ambiguity_remains → claim: K3 |
| S28 | REV-50: The examples can pick out a correlated respect | SC-50 | major | text: §5.3 and §5.1 "answer the objection that two aspects make the condition a binary switch / condition A uses S = P_A, C = P_B; condition B swaps them" | 3 — core expertise on the construct; the size of the effect in these datasets is untested | R3 W4.2 | SINGLE-VERIFIER | should_fix | re_analysis: E0 episode module and E11 ablations | interpretive_ambiguity_remains → claim: K6 |
| S29 | REV-51: Connect the task to the psychology of similarity | SC-51 | minor | text: novelty check §4 must-cite list "Tversky 1977 (similarity depends on the comparison context)" | 4 — core expertise: similarity theory | R3 W5 | SINGLE-VERIFIER | should_fix | section: §3 and §11 E11 | interpretive_ambiguity_remains → section: §3 |
| S30 | REV-52: Robustness to realistic conditions | SC-52 | minor | text: §1 and §3 "A user shows, by a few example image–caption pairs, / The condition never shows the query's value and never names the aspect." | 3 — adjacent field: human factors of retrieval | R3 W6.2 | SINGLE-VERIFIER | should_fix | re_analysis: supplementary robustness runs | evidence_gap_remains → claim: practical setting of C1 |
| S31 | REV-53: Emotion labels are viewers' reports, and their scope | SC-53 | minor | text: §2.4 and §2.3 "each a fearful image matched with a fearful caption of / English part" | 4 — core expertise: cross-cultural affective annotation | R3 W8 | SINGLE-VERIFIER | should_fix | sentence: §2.3 and §2.4 wording; §5.2 datasets paragraph | claim_scope_unsupported → claim: C3 and emotion statements |
| S32 | REV-54: R3 reading list | — | n/a (editorial list) | n/a (editorial list) | n/a | R3 editorial list | SINGLE-VERIFIER | consider | sentence: related-work paragraph and Appendix B sources | reader_traceability_reduced → section: related work |
| S33 | REV-56: Decision rules for K6 and for emotion | — | major | text: §4 K2 against §10 "Our method beats backbone-only, raw-feature metric-from-pairs baselines and the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive." | 5 — direct comparison of two sections of the plan | DA M2.3 | SINGLE-VERIFIER | should_fix | section: §10 statistics | claim_scope_unsupported → claim: K6 |
| S34 | REV-57: Controls that separate basis structure from training signal | — | major | text: §8; literature review §5 "PCA-32/64, NMF-32 and SpLiCE sparse concept codes instead of our factors / GoEmotions cannot read images, so it cannot score cross-modal pairs / factors shared by image and text (against a split dictionary, as MGSAE warns)" | 4 — ablation design; checked §8, §11 E11 and literature review §5 | DA M5.2 | SINGLE-VERIFIER | should_fix | re_analysis: E11 ablations | claim_scope_unsupported → claim: C2 |
| S35 | REV-58: DA minor issues not carried elsewhere | — | n/a (editorial list) | n/a (editorial list) | n/a | DA editorial list | SINGLE-VERIFIER | consider | sentence: §4 K1, §5.1, §6 GO and §11 E0 | interpretive_ambiguity_remains → section: §6 Go/no-go |

## Source-Traceability Checklist

> Immutable source order. This list does not suggest a work order. The author chooses `will_address`, `wont_address` or `not_on_point` later, at the separate author-adjudication checkpoint.

- [ ] R1 (REV-01) — obligation `must_fix`: Condition-attributable GO statistic and condition-removed control
- [ ] R2 (REV-02) — obligation `must_fix`: K3 has no outcome-neutral branch while current evidence favours names on emotion
- [ ] R3 (REV-03) — obligation `must_fix`: A use case in which pairs exist but the aspect cannot be named
- [ ] R4 (REV-04) — obligation `must_fix`: K5 is decidable only through a stretch protocol
- [ ] R5 (REV-05) — obligation `must_fix`: GeneCIS measures transfer, not the cross-modal task
- [ ] R6 (REV-06) — obligation `must_fix`: Complete the novelty search before the title and abstract are fixed
- [ ] R7 (REV-07) — obligation `must_fix`: In-context alternatives that use the same examples
- [ ] R8 (REV-08) — obligation `must_fix`: Define the NO-GO fallback paper
- [ ] S1 (REV-09) — obligation `should_fix`: Novelty statement counts protocol choices as task properties
- [ ] R9 (REV-10) — obligation `must_fix`: C3's 'belongs to the data' outruns the backbone test
- [ ] R10 (REV-11) — obligation `must_fix`: K2 is worded beyond the pre-declared tests
- [ ] S2 (REV-12) — obligation `should_fix`: Release plan for the task and benchmark
- [ ] S3 (REV-13) — obligation `should_fix`: A full cut order that keeps a never-read test set
- [ ] S4 (REV-14) — obligation `should_fix`: Final review falls after the abstract freeze
- [ ] S5 (REV-15) — obligation `should_fix`: Main-paper outline and page budget
- [ ] S6 (REV-16) — obligation `should_fix`: Project vocabulary and internal recipes
- [ ] S7 (REV-17) — obligation `should_fix`: Dates that blur the order of evidence
- [ ] S8 (REV-18) — obligation `should_fix`: 'Ceiling' names a drifting diagnostic
- [ ] S9 (REV-19) — obligation `consider`: EIC minor issues
- [ ] R11 (REV-20) — obligation `must_fix`: Cluster the bootstrap over reused items
- [ ] R12 (REV-21) — obligation `must_fix`: An equivalence margin for 'matches'
- [ ] R13 (REV-22) — obligation `must_fix`: K7 is a descriptive comparison
- [ ] R14 (REV-23) — obligation `must_fix`: Per-viewer labels cap the image-side emotion probe
- [ ] S10 (REV-24) — obligation `should_fix`: GeneCIS is read during development
- [ ] S11 (REV-25) — obligation `consider`: Best-of-ten selection still inflates the GO test
- [ ] S12 (REV-26) — obligation `consider`: Swap success: scorer-dependent null and unit
- [ ] S13 (REV-27) — obligation `consider`: Training-seed variance in the headline CI
- [ ] R15 (REV-28) — obligation `must_fix`: Define the multiplicity family
- [ ] R16 (REV-29) — obligation `must_fix`: Specify the power calculation
- [ ] S14 (REV-30) — obligation `consider`: Baseline tuning grids and the grid-edge rule
- [ ] S15 (REV-31) — obligation `consider`: One score form and its transfer across datasets
- [ ] S16 (REV-32) — obligation `should_fix`: Exclusivity and third-aspect balance of candidates
- [ ] S17 (REV-33) — obligation `consider`: A guessed genre name handicaps the names baseline
- [ ] S18 (REV-34) — obligation `consider`: Backbone-appendix prose disagrees with its tables
- [ ] R17 (REV-35) — obligation `must_fix`: An outcome-independent criterion for subjective and objective aspects
- [ ] S19 (REV-36) — obligation `consider`: CUB final-test species were read in the backbone check
- [ ] R18 (REV-37) — obligation `must_fix`: K5's bar uses the wrong GeneCIS column
- [ ] R19 (REV-38) — obligation `must_fix`: Restore the per-episode probe and the Tip-Adapter cache
- [ ] R20 (REV-39) — obligation `must_fix`: K7 needs a sparse-autoencoder basis
- [ ] R21 (REV-40) — obligation `must_fix`: 'No labels' understates GoEmotions distant supervision
- [ ] R22 (REV-41) — obligation `must_fix`: ArtELingo partitions are hand-matched to the evaluation aspects
- [ ] S20 (REV-42) — obligation `consider`: SEARLE 14.4 is given two backbones
- [ ] S21 (REV-43) — obligation `consider`: The agreement rule's lineage
- [ ] S22 (REV-44) — obligation `consider`: C-STS already pairs contrasting conditions
- [ ] S23 (REV-45) — obligation `consider`: Multiview triplet embedding is missing
- [ ] S24 (REV-46) — obligation `consider`: R2 minor issues and search lead
- [ ] S25 (REV-47) — obligation `should_fix`: Probe genre before using it as the symmetric control
- [ ] S26 (REV-48) — obligation `should_fix`: Inter-annotator agreement and a human noise ceiling
- [ ] S27 (REV-49) — obligation `should_fix`: Subjectivity is conflated with nameability
- [ ] S28 (REV-50) — obligation `should_fix`: The examples can pick out a correlated respect
- [ ] S29 (REV-51) — obligation `should_fix`: Connect the task to the psychology of similarity
- [ ] S30 (REV-52) — obligation `should_fix`: Robustness to realistic conditions
- [ ] S31 (REV-53) — obligation `should_fix`: Emotion labels are viewers' reports, and their scope
- [ ] S32 (REV-54) — obligation `consider`: R3 reading list
- [ ] R23 (REV-55) — obligation `must_fix`: Held-out-aspect test for the mechanism claim (DA C1, VALIDATED)
- [ ] S33 (REV-56) — obligation `should_fix`: Decision rules for K6 and for emotion
- [ ] S34 (REV-57) — obligation `should_fix`: Controls that separate basis structure from training signal
- [ ] S35 (REV-58) — obligation `consider`: DA minor issues not carried elsewhere

## Response Letter Template

Respond to every item, R1 to R23 and S1 to S35, using `templates/revision_response_template.md` (academic-paper-reviewer).

## Machine artifact (`revision-roadmap/1.0`)

```json
{
 "schema_version": "revision-roadmap/1.0",
 "revision_round": 1,
 "base_draft_sha256": "16c9912faec08401d9cd71162e7ca192f181b30ea58d238a43bcbbc74b88487f",
 "block_manifest_sha256": "5bece1904c274f793e305e2fd93e5a84d5e053c0688edbab775b984f00e22c4e",
 "items": [
  {
   "id": "REV-01",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 1
    },
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 2
    },
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 1
    },
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 2
    }
   ],
   "description": "The GO rule and the three §10 primary comparisons are decided on pooled aspect R@1 alone, which a scorer that ignores the condition can raise (to about 50% under R1's construction), and no Tier-1 control runs the trained factors with the condition removed. Corroborated finding, 2 of 4 non-DA seats (EIC W1, R1 W1); R2 and R3 silent.",
   "reviewer": "EIC W1, R1 W1",
   "sub_claim_ids": [
    "SC-1",
    "SC-2"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R1 W1 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§5.1 Common protocol",
    "quote": "R@1, the mean over both directions and all conditions / one sharing aspect A with the anchor (p_A), one sharing aspect B (p_B), and 11"
   },
   "confidence": 4,
   "competence_basis": "core expertise: shortcut analysis of episodic retrieval benchmarks",
   "confidence_source": "per-finding Confidence of R1 W1",
   "corroborating_sources": [
    {
     "reviewer": "EIC W1",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§6 Go/no-go, GO bullet",
      "quote": "if, on those fresh episodes, pooled aspect R@1 has a 95% CI lower bound above 0 against"
     },
     "confidence": 4,
     "competence_basis": "evaluation-design reading checked against §5.1 and the aspect-spike CLIP rows; the full shortcut audit belongs to the methodology seat"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§6 Go/no-go rule, §8 Tier-1 controls, §10 primary comparisons"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K2 and the GO decision (§6, §10)"
    }
   },
   "target_section": "§6 Go/no-go; §8 Baselines; §10 Statistics",
   "suggested_action": "Add a condition-attributable statistic to the GO rule and to K2 with the same CI rule (for example condition-correct minus condition-swapped R@1, or R@1 minus the other-aspect rate); add 'ours with uniform weights' and 'ours under the swapped condition' as Tier-1 controls; pre-declare ours versus ours without the condition as a primary comparison; report a condition-blind probe reference beside the aspect-aware one.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§6 and §10 state a condition-sensitivity requirement with its CI rule, §8 lists both condition-removed controls, and the backbone-only value of the new statistic is reported on selection episodes.",
   "proposed_targets": [
    {
     "block_id": "B0073",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0080",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-02",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 1
    },
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 2
    },
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 2
    },
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 3
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 2
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 3
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 3
    }
   ],
   "description": "K3's subjective half meets adverse appended evidence (privileged names reach 13.31 on emotion against 10.14 for CLIP only, while every example-based scorer stays flat), and R-names narrows K3 'to where names fail (subjective aspects)', so the plan has no branch for names winning everywhere and K3 can survive any outcome. CONSENSUS-3: EIC W2, R2 W5, R3 W3; R1 silent. DA M3 also raises it.",
   "reviewer": "EIC W2, R2 W5, R3 W3, DA M3",
   "sub_claim_ids": [
    "SC-3",
    "SC-4"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from EIC W2 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "table",
    "locator": "aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14"
   },
   "confidence": 4,
   "competence_basis": "significance is core to my remit; numbers read from the appended spike",
   "confidence_source": "per-finding Confidence of EIC W2",
   "corroborating_sources": [
    {
     "reviewer": "R2 W5",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 claims table K3",
      "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones"
     },
     "confidence": 3,
     "competence_basis": "core expertise: text-conditioned similarity baselines; the direction is my reading of a single spike on untrained factors"
    },
    {
     "reviewer": "R3 W3",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 K3 and §12 R-names",
      "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones / narrow K3 to where names fail (subjective aspects)"
     },
     "confidence": 4,
     "competence_basis": "core expertise: similarity and naming in cognitive psychology"
    },
    {
     "reviewer": "DA M3",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 K3; §12; aspect spike",
      "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones / narrow K3 to where names fail (subjective aspects) / on emotion only (+3.2 over CLIP only, 13.31 against 10.14)"
     },
     "confidence": 4,
     "competence_basis": "statistical logic; evidence from the appended spike"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§4 K3 row and Fallback, §12 R-names row"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K3 and the motivation of C1"
    }
   },
   "target_section": "§4 Contributions and claims; §12 Risks",
   "suggested_action": "Rewrite R-names so its mitigation does not presuppose the direction; pre-declare what the paper claims if names win on every aspect (for example examples that compose with names, or groupings with no stable name); run the Qwen-instruction naming comparison on selection rows before the Oct 9 go/no-go and report it beside the example scorers.",
   "consensus_level": "CONSENSUS-3",
   "verification_criteria": "§4 and §12 state outcome-neutral claims for both directions of K3, and the selection-row naming comparison is reported with paired CIs.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0048",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0097",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-03",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 3
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 1
    }
   ],
   "description": "The plan names no setting in which a person holds value-disjoint, cross-item demonstration and contrast pairs but cannot name the aspect. SPLIT on severity (EIC W2 Major, R3 W6 Minor); arbitrated to the Journal-Fit seat's Major because the use case carries the significance argument for C1, which is that seat's remit.",
   "reviewer": "EIC W2, R3 W6",
   "sub_claim_ids": [
    "SC-5"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from EIC W2 (driving finding); R3 W6 rates it minor; SPLIT arbitrated to major",
   "evidence_anchor": {
    "anchor_type": "table",
    "locator": "aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14"
   },
   "confidence": 4,
   "competence_basis": "significance is core to my remit; numbers read from the appended spike",
   "confidence_source": "per-finding Confidence of EIC W2",
   "corroborating_sources": [
    {
     "reviewer": "R3 W6",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§1 and §3",
      "quote": "A user shows, by a few example image–caption pairs, / The condition never shows the query's value and never names the aspect."
     },
     "confidence": 3,
     "competence_basis": "adjacent field: human factors of retrieval"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§1 goal paragraph and §3 problem definition"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "significance of C1"
    }
   },
   "target_section": "§1 Goal; §3 Problem definition",
   "suggested_action": "State a concrete setting with its data source in which demonstration pairs exist but the aspect resists naming (for example curating a captioned collection by an unnamed stylistic or affective quality from a seed set), and tie it to the relevance-feedback line the plan already cites.",
   "consensus_level": "SPLIT",
   "verification_criteria": "§1 or §3 names the setting, who supplies the pairs and from which corpus.",
   "proposed_targets": [
    {
     "block_id": "B0042",
     "allowed_operations": [
      "insert_after"
     ]
    }
   ]
  },
  {
   "id": "REV-04",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 1
    },
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 2
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 2
    }
   ],
   "description": "K5's comparison with published GeneCIS rows is decidable only through the text protocol, which §5.4 and E7 keep as a stretch goal, because the example protocol 'is not comparable with published numbers'. Corroborated finding, 2 of 4 (EIC W3, R2 W1); DA M4 also.",
   "reviewer": "EIC W3, R2 W1, DA M4",
   "sub_claim_ids": [
    "SC-6"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from EIC W3 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§5.4 example protocol",
    "quote": "It is not comparable with published numbers."
   },
   "confidence": 4,
   "competence_basis": "benchmark expectations in this subfield are core to my remit",
   "confidence_source": "per-finding Confidence of EIC W3",
   "corroborating_sources": [
    {
     "reviewer": "R2 W1",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§5.4 item 2",
      "quote": "published frozen ViT-B/32 results: SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1"
     },
     "confidence": 4,
     "competence_basis": "core expertise: GeneCIS and zero-shot CIR reporting; numbers taken from the manuscript's own Table 2"
    },
    {
     "reviewer": "DA M4",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§5.4; literature review Table 2",
      "quote": "SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1 / It is not comparable with published numbers. / ViT-B/32 17.9 / 14.8 / 14.6 / 16.1 / 15.9"
     },
     "confidence": 5,
     "competence_basis": "arithmetic check against the plan's own Table 2"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§5.4 GeneCIS protocols, §11 E7, §4 K5 row"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "claim",
     "locator": "K5"
    }
   },
   "target_section": "§4 K5; §5.4 GeneCIS protocols; §11 E7",
   "suggested_action": "Either commit the GeneCIS text protocol as required work in §5.4 and E7 with a defined threshold for 'competitive', or reword K5 so the example-protocol result is reported next to the plan's own image, text and image-plus-text baselines with no published-row comparison.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "K5 names its protocol, its comparator rows and its threshold, or no longer claims a published-row comparison.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0064",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-05",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 2
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 3
    }
   ],
   "description": "GeneCIS is image to image, so even its comparable row measures transfer and cannot test the cross-modal task; every cross-modal result rests on aspect episodes the authors build. Single-reviewer finding (EIC W3); DA M4 also states it.",
   "reviewer": "EIC W3, DA M4",
   "sub_claim_ids": [
    "SC-7"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from EIC W3 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§5.4 example protocol",
    "quote": "It is not comparable with published numbers."
   },
   "confidence": 4,
   "competence_basis": "benchmark expectations in this subfield are core to my remit",
   "confidence_source": "per-finding Confidence of EIC W3",
   "corroborating_sources": [
    {
     "reviewer": "DA M4",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§5.4; literature review Table 2",
      "quote": "SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1 / It is not comparable with published numbers. / ViT-B/32 17.9 / 14.8 / 14.6 / 16.1 / 15.9"
     },
     "confidence": 5,
     "competence_basis": "arithmetic check against the plan's own Table 2"
    }
   ],
   "cost_scope": {
    "kind": "sentence",
    "locator": "§3 'Against GeneCIS' paragraph and §5.4"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K5"
    }
   },
   "target_section": "§3; §5.4",
   "suggested_action": "State in §3 and §5.4 that GeneCIS measures transfer, not the cross-modal task, and make the self-built aspect episodes reusable (see the release item) so they can serve as the external anchor.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§3 and §5.4 describe GeneCIS as a transfer check, and the release item covers the episode files.",
   "proposed_targets": [
    {
     "block_id": "B0043",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0065",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-06",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The 'first' claim of C1 rests on searches the authors call incomplete (about 25 arXiv API queries, no web search engine, Semantic Scholar rate-limited, no Google Scholar), and §11 schedules no completion before E15 fixes the title and abstract on Nov 7. R2 W9 and W10 each name one omitted precedent; those are separate items.",
   "reviewer": "EIC W4",
   "sub_claim_ids": [
    "SC-8"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from EIC W4 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "novelty-check appendix, \"Searched, nothing found\" paragraph",
    "quote": "This is a negative result from an incomplete search (no web search engine, Semantic Scholar rate-limited, no Google Scholar)"
   },
   "confidence": 4,
   "competence_basis": "the gap is documented by the authors; I cannot tell whether a missed paper exists",
   "confidence_source": "per-finding Confidence of EIC W4",
   "cost_scope": {
    "kind": "other",
    "locator": "§11 schedule: a literature-completion row before E15",
    "surface_id": "literature_search"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C1 novelty statement"
    }
   },
   "target_section": "§4 C1; §11 Experiment plan",
   "suggested_action": "Add an E-row before E15 for a full search (Google Scholar, Semantic Scholar, forward citations of KISSME, Wang et al. 2016, MARS, BGE-EN-ICL and GeneCIS, and the CVPR 2026 and ECCV 2026 proceedings), with a pre-declared rewording of C1 if a match appears.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§11 contains the search row, its report lists queries and sources, and C1 carries the pre-declared fallback wording.",
   "proposed_targets": [
    {
     "block_id": "B0046",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-07",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 0
    },
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 1
    }
   ],
   "description": "Alternatives that use the same support and contrast pairs without a learned basis are absent (an instruction embedder given the pairs in context; verbalise-then-name) or stretch-only (an in-context MLLM reranker), although the plan's own novelty check lists them. Corroborated finding, 2 of 4 (EIC W5, R2 W2); the DA's ignored-alternatives list names the reranker too.",
   "reviewer": "EIC W5, R2 W2",
   "sub_claim_ids": [
    "SC-9"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from EIC W5 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "absence",
    "locator": "§8 baselines table and §10 primary comparisons",
    "absence_scope": "the in-context baselines of novelty-check §5 Table 2 (an embedder or MLLM given the same support and contrast pairs, and verbalise-then-name) as scheduled baselines",
    "check_performed": "§8 tiers 1 to 3 and the stretch row, §10, §11 E1, E10 and E17"
   },
   "confidence": 4,
   "competence_basis": "what the 2027 readership will expect is core to my remit",
   "confidence_source": "per-finding Confidence of EIC W5",
   "corroborating_sources": [
    {
     "reviewer": "R2 W2",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "absence",
      "locator": "§8 baseline table and §10 pre-declared primary comparisons",
      "absence_scope": "a required, non-stretch baseline that gives an instruction embedder or open MLLM the same support and contrast pairs in context, and a verbalise-then-name baseline",
      "check_performed": "§8 tiers 1 to 3 and the stretch row, §10 statistics, §11 E10 and E17, novelty check §5 Table 2, literature review §3 Table 8"
     },
     "confidence": 4,
     "competence_basis": "core expertise: conditional similarity baselines and instruction-following multimodal embedders"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "§8 Tier 2 and §11 E10 on existing episodes"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "claim",
     "locator": "C2 and the significance of C1"
    }
   },
   "target_section": "§8 Baselines; §11 E10",
   "suggested_action": "Move verbalise-then-name and the in-context instruction embedder (Qwen3-VL-Embedding-2B, the chosen backbone) into Tier 2 and E10; run the in-context MLLM reranker on a fixed, pre-registered subsample; report them beside the primary comparisons.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§8 lists the three in-context baselines outside the stretch row, E10 schedules them, and the pre-registration fixes the reranker subsample.",
   "proposed_targets": [
    {
     "block_id": "B0080",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-08",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The NO-GO fallback (K1, K3, C3) has no claim table, target venue or switching criteria, although NO-GO is a live outcome with the factors at CLIP level on aspect episodes today.",
   "reviewer": "EIC W6",
   "sub_claim_ids": [
    "SC-10"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from EIC W6 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 Fallback",
    "quote": "If K2 fails, K1, K3 and C3 could carry a task, benchmark and analysis paper but not the method paper."
   },
   "confidence": 4,
   "competence_basis": "storyline robustness across branches is core to my remit",
   "confidence_source": "per-finding Confidence of EIC W6",
   "cost_scope": {
    "kind": "section",
    "locator": "§4 Fallback paragraph"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§4 Fallback"
    }
   },
   "target_section": "§4 Fallback",
   "suggested_action": "Write the fallback's claim table now (a released benchmark, strong in-context and named baselines, a defined subjective versus objective split, C3 tested across annotation protocols), name the venue it targets, and pre-declare the criteria for switching.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§4 Fallback carries a claim table, a named venue and switching criteria fixed before the Oct 9 decision.",
   "proposed_targets": [
    {
     "block_id": "B0048",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-09",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 7,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The C1 novelty statement makes 'both directions with a paired swap test' part of what is first, though these are evaluation choices any prior method could adopt.",
   "reviewer": "EIC W7",
   "sub_claim_ids": [
    "SC-11"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W7 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 C1 novelty statement",
    "quote": "evaluated in both directions with a paired swap test"
   },
   "confidence": 4,
   "competence_basis": "novelty framing is core to my remit",
   "confidence_source": "per-finding Confidence of EIC W7",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§4 C1 novelty statement"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C1 novelty statement"
    }
   },
   "target_section": "§4 C1",
   "suggested_action": "State the novelty on problem properties (an aspect fixed only by value-disjoint, cross-item image and caption demonstrations with a contrast aspect) and present both directions and the swap test as the protocol that makes the task measurable.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The C1 statement no longer lists protocol choices as novelty conditions.",
   "proposed_targets": [
    {
     "block_id": "B0046",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-10",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 8,
     "subclaim_ordinal": 0
    },
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 1
    },
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 1
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 1
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 1
    }
   ],
   "description": "C3's causal wording ('so it belongs to the data') outruns the backbone-invariance test, which cannot separate what images and captions contain from how ArtEmis captions and labels were elicited (each caption explains one viewer's emotion; Reed captions describe parts). All four non-DA seats raise it (EIC W8, R1 W5, R2 W6, R3 W1); SPLIT on severity only (EIC Minor, the other three Major), arbitrated Major. DA M1 also.",
   "reviewer": "EIC W8, R1 W5, R2 W6, R3 W1, DA M1",
   "sub_claim_ids": [
    "SC-12"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R1 W5 (driving finding); EIC W8 rates it minor; SPLIT arbitrated to major",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 C3 and backbone check Result 1",
    "quote": "the effect persists across four backbones / a better encoder cannot read what the pixels or words do not"
   },
   "confidence": 3,
   "competence_basis": "adjacent: label-noise ceilings; construct validity is Reviewer 3's remit",
   "confidence_source": "per-finding Confidence of R1 W5",
   "corroborating_sources": [
    {
     "reviewer": "EIC W8",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 C3",
      "quote": "the effect persists across four backbones, so it belongs to the data"
     },
     "confidence": 3,
     "competence_basis": "adjacent to my remit; I rely on the appendices' descriptions of the captions"
    },
    {
     "reviewer": "R2 W6",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "backbone check Verdict",
      "quote": "The weaker-modality probes barely move across four backbones, so the modality asymmetry is a property of the data"
     },
     "confidence": 3,
     "competence_basis": "adjacent expertise: affective annotation datasets; the ArtEmis per-annotation design is stated in the manuscript's §2.3"
    },
    {
     "reviewer": "R3 W1",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 C3 and backbone check Verdict",
      "quote": "the effect persists across four backbones, so it belongs / so the modality asymmetry is a property of the data"
     },
     "confidence": 4,
     "competence_basis": "core expertise: affective annotation protocols; adjacent: the CV probes"
    },
    {
     "reviewer": "DA M1",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 C3; backbone check; aspect spike",
      "quote": "the effect persists across four backbones, so it belongs / a better encoder cannot read what the pixels or words do not / an image takes the emotion of its row, which is a noisy image label"
     },
     "confidence": 4,
     "competence_basis": "confound reasoning; ArtEmis protocol as the plan itself describes it"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "§4 C3 wording and the E12 C3 analysis with SemArt curator text as a protocol contrast"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C3"
    }
   },
   "target_section": "§4 C3; §11 E12; backbone-check verdict",
   "suggested_action": "Narrow C3 to the annotated data and its elicitation protocol (for example 'in datasets whose captions explain a self-reported emotion, emotion is carried by the text'), unless the protocol-contrast analysis still shows a modality effect; use SemArt's curator-written catalogue text, and CUB, as elicitation-protocol contrasts in E12 rather than as confirmation.",
   "consensus_level": "SPLIT",
   "verification_criteria": "§4 C3 and the backbone-check verdict no longer attribute the asymmetry to modality alone, or E12 reports a protocol-contrast result that supports the stronger wording.",
   "proposed_targets": [
    {
     "block_id": "B0046",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0289",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-11",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 9,
     "subclaim_ordinal": 0
    },
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 1
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 1
    }
   ],
   "description": "K2 claims wins 'in both directions, on every dataset' over three baseline families, but §10 pools directions into one primary metric, sets no rule for partial passes (some datasets, one backbone), and GeneCIS (image to image, no aspect episodes) cannot satisfy the wording. EIC W9 (Minor) and R1 W4 (Major); SPLIT on severity, arbitrated Major. DA M2 also.",
   "reviewer": "EIC W9, R1 W4, DA M2",
   "sub_claim_ids": [
    "SC-13"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R1 W4 (driving finding); EIC W9 rates it minor; SPLIT arbitrated to major",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 K2 and §10",
    "quote": "the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive."
   },
   "confidence": 4,
   "competence_basis": "core expertise: aligning pre-registered tests with claim wording",
   "confidence_source": "per-finding Confidence of R1 W4",
   "corroborating_sources": [
    {
     "reviewer": "EIC W9",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 claims table, K2",
      "quote": "in both directions, on every dataset"
     },
     "confidence": 4,
     "competence_basis": "structural coherence of claims is core to my remit"
    },
    {
     "reviewer": "DA M2",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 K2 against §10",
      "quote": "Our method beats backbone-only, raw-feature metric-from-pairs baselines and the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive."
     },
     "confidence": 5,
     "competence_basis": "direct comparison of two sections of the plan"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§4 K2 row, §10 primary comparisons, E13 pre-registration"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K2"
    }
   },
   "target_section": "§4 K2; §10 Statistics; §11 E13",
   "suggested_action": "Restate K2 on the pooled primary metric per dataset, list the datasets K2 covers (leaving GeneCIS to K5), add per-direction tests or report directions as secondary, and pre-register in E13 the headline wording for each pattern of passes.",
   "consensus_level": "SPLIT",
   "verification_criteria": "K2's wording matches the §10 primary comparisons, and the E13 pre-registration lists the claim made under each pass pattern.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-12",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 10,
     "subclaim_ordinal": 0
    }
   ],
   "description": "C1 is a task and protocol and the fallback is a benchmark paper, yet no release plan covers the aspect-episode builder, episode files, split lists or trained factors, or how non-commercial dataset terms constrain them.",
   "reviewer": "EIC W10",
   "sub_claim_ids": [
    "SC-14"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W10 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "absence",
    "locator": "plan body §4 to §12",
    "absence_scope": "a release plan for the aspect-episode builder, episode files and split lists",
    "check_performed": "§4 C1 and the fallback, §5.1 to §5.4, the §10 ledger, §11 E0 and E17, §12 R-licence"
   },
   "confidence": 4,
   "competence_basis": "a standard expectation for new-task papers at vision venues",
   "confidence_source": "per-finding Confidence of EIC W10",
   "cost_scope": {
    "kind": "section",
    "locator": "§11 E17 and §12 R-licence"
   },
   "consequence_if_unaddressed": {
    "code": "reporting_requirement_unmet",
    "target": {
     "kind": "section",
     "locator": "§11 E17"
    }
   },
   "target_section": "§11 E17; §12 R-licence",
   "suggested_action": "Commit to releasing episode indices, split lists and the builder (which avoids redistributing images), state the terms of the derived files, and add the release to E17.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "E17 and §12 R-licence name the released artefacts and their terms.",
   "proposed_targets": [
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0097",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-13",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 11,
     "subclaim_ordinal": 0
    }
   ],
   "description": "R-time's only mitigation moves SemArt, a never-read test set, to the supplement; no full cut order is pre-declared, so time pressure can push the headline toward ArtELingo's pre-read held rows.",
   "reviewer": "EIC W11",
   "sub_claim_ids": [
    "SC-15"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W11 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§12 R-time",
    "quote": "staged priority; SemArt moves to the supplementary first"
   },
   "confidence": 3,
   "competence_basis": "planning judgement; the compute estimates are the authors'",
   "confidence_source": "per-finding Confidence of EIC W11",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§12 R-time row"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§12 R-time"
    }
   },
   "target_section": "§12 Risks",
   "suggested_action": "Pre-declare a full cut order (for example Qwen runs on secondary datasets and stretch baselines go before any never-read test set) and keep at least one never-read test set beside ArtELingo in the main paper.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§12 R-time lists a cut order that keeps one never-read test set in the main paper.",
   "proposed_targets": [
    {
     "block_id": "B0097",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-14",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 12,
     "subclaim_ordinal": 0
    }
   ],
   "description": "E16, the whole-branch review that may trigger a reserve read, runs Nov 11 to 14, after E15 fixes the title and abstract on Nov 7 and registration on Nov 10.",
   "reviewer": "EIC W12",
   "sub_claim_ids": [
    "SC-16"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W12 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§11 E15 and E16 rows",
    "quote": "title and abstract fixed Nov 7 / Nov 11 to 14"
   },
   "confidence": 4,
   "competence_basis": "read directly from the schedule",
   "confidence_source": "per-finding Confidence of EIC W12",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§11 E15 and E16 rows"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "abstract headline numbers"
    }
   },
   "target_section": "§11 E15 and E16",
   "suggested_action": "Review the load-bearing numbers before the abstract is fixed on Nov 7, or add an abstract-revision step before the paper deadline.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§11 places a load-bearing-number review before E15 or adds an abstract-revision row after E16.",
   "proposed_targets": [
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-15",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 13,
     "subclaim_ordinal": 0
    }
   ],
   "description": "There is no main-paper outline, page budget or list of main tables and figures, while four datasets, two backbones, three baseline tiers, both directions, swap and other-aspect metrics, the K6 and K7 ablations, C3 and a GeneCIS table compete for eight pages.",
   "reviewer": "EIC W13",
   "sub_claim_ids": [
    "SC-17"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W13 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "absence",
    "locator": "plan body",
    "absence_scope": "a main-paper outline with a page budget and the planned figures and tables",
    "check_performed": "§4, §5.1, §10, §11 E15 and E17, §12, Appendix B"
   },
   "confidence": 4,
   "competence_basis": "exposition and venue conventions are core to my remit",
   "confidence_source": "per-finding Confidence of EIC W13",
   "cost_scope": {
    "kind": "section",
    "locator": "a new outline section after §11"
   },
   "consequence_if_unaddressed": {
    "code": "editorial_conformance_unmet",
    "target": {
     "kind": "manuscript",
     "locator": "main-paper outline"
    }
   },
   "target_section": "§11 Planning scope",
   "suggested_action": "Add a one-page outline with page counts per section and the two or three main tables and figures, each tied to a K-claim with its baseline beside it.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The plan contains the outline with per-section page counts and named main tables and figures.",
   "proposed_targets": [
    {
     "block_id": "B0095",
     "allowed_operations": [
      "insert_after"
     ]
    }
   ]
  },
  {
   "id": "REV-16",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 14,
     "subclaim_ordinal": 0
    }
   ],
   "description": "Twenty-five glossary entries carry project history (naive versus agreement rule; R0, R3, C0, SE; condition versus pseudo-aspect episodes), and §8 keeps three internal recipes as Tier-1 baselines that readers never saw.",
   "reviewer": "EIC W14",
   "sub_claim_ids": [
    "SC-18"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W14 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§8 Tier-1 row for C0, SE and R3",
    "quote": "our earlier factor recipes"
   },
   "confidence": 3,
   "competence_basis": "a judgement about exposition, not correctness",
   "confidence_source": "per-finding Confidence of EIC W14",
   "cost_scope": {
    "kind": "section",
    "locator": "§8 Tier-1 internal-recipe row and Appendix A"
   },
   "consequence_if_unaddressed": {
    "code": "reader_traceability_reduced",
    "target": {
     "kind": "section",
     "locator": "Appendix A glossary"
    }
   },
   "target_section": "§8 Baselines; Appendix A",
   "suggested_action": "Fold the internal recipes into one ablation (E11 already compares no episodes with value episodes), define only the terms the main paper uses, and move the history to the supplement.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§8 has no internal-recipe Tier-1 row and the glossary is limited to main-paper terms.",
   "proposed_targets": [
    {
     "block_id": "B0080",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0102",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-17",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 15,
     "subclaim_ordinal": 0
    }
   ],
   "description": "Sequence labels that look like dates blur the order of evidence: §2.4 reports changes made on 2026-10-02 while citing reports dated Oct 20 to 25, the genre folder is dated 20261026 while §5.3 says resolved on Oct 2, and the six investigations 'all on selection rows' include two literature reviews that read no rows.",
   "reviewer": "EIC W15",
   "sub_claim_ids": [
    "SC-19"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W15 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§2.4 opening",
    "quote": "Six investigations, all on selection rows (held rows untouched), changed the plan."
   },
   "confidence": 4,
   "competence_basis": "read directly from the document",
   "confidence_source": "per-finding Confidence of EIC W15",
   "cost_scope": {
    "kind": "section",
    "locator": "§2.4, §5.3 and the held ledger"
   },
   "consequence_if_unaddressed": {
    "code": "reader_traceability_reduced",
    "target": {
     "kind": "section",
     "locator": "§2.4"
    }
   },
   "target_section": "§2.4; §5.3; §10 Ledger",
   "suggested_action": "Use real dates or explicit sequence numbers throughout, and log every held-row read with its real date in the ledger.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "Every report reference and ledger entry carries a real date or a sequence number labelled as such.",
   "proposed_targets": [
    {
     "block_id": "B0036",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0037",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0086",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-18",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "finding",
     "ordinal": 16,
     "subclaim_ordinal": 0
    },
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 8,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The label-probe value is called a 'ceiling' and anchors the strong-GO bar ('a third of the way to the ceiling of about 23'), though the appendix calls it a diagnostic rather than a bound, it drifts by up to 0.25 points between reruns, it differs by backbone, and a condition-blind scorer can exceed it. Corroborated finding, 2 of 4 (EIC W16, R1 W8, both Minor).",
   "reviewer": "EIC W16, R1 W8",
   "sub_claim_ids": [
    "SC-20"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from EIC W16 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "aspect-episode spike appendix, caveats",
    "quote": "It is a diagnostic, not a bound in the strict sense; a better probe could score higher."
   },
   "confidence": 4,
   "competence_basis": "terminology and claim framing are core to my remit",
   "confidence_source": "per-finding Confidence of EIC W16",
   "corroborating_sources": [
    {
     "reviewer": "R1 W8",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§6 Strong GO",
      "quote": "a third of the way to the ceiling of about 23"
     },
     "confidence": 4,
     "competence_basis": "core expertise: benchmark headroom analysis"
    }
   ],
   "cost_scope": {
    "kind": "sentence",
    "locator": "§5.2, §6 Strong GO and §12 R-ceiling"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§6 Strong GO"
    }
   },
   "target_section": "§6 Go/no-go; §12 Risks",
   "suggested_action": "Rename it a label-probe reference, report it per backbone with converged probes, and state strong GO as a pre-declared absolute effect with its CI; EIC proposes the gain over backbone-only and R1 the effect over the best baseline, and the plan states which reference it uses.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "No section uses 'ceiling' for the probe value, and strong GO is an absolute effect with a stated reference.",
   "proposed_targets": [
    {
     "block_id": "B0073",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0097",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-19",
   "source_refs": [
    {
     "seat": "EIC",
     "channel": "editorial",
     "ordinal": 17,
     "subclaim_ordinal": 0
    }
   ],
   "description": "EIC minor issues: internal reconciliation notes (for example the controller's 46.43 against the recomputed 46.42) and build-script paths in figure captions should not reach the paper; name the swap-success variant (pairwise or strict) that §5.1 adopts wherever swap numbers appear.",
   "reviewer": "EIC (Minor Issues list)",
   "obligation_class": "consider",
   "source_kind": "editorial",
   "cost_scope": {
    "kind": "sentence",
    "locator": "figure captions and §5.1"
   },
   "consequence_if_unaddressed": {
    "code": "editorial_conformance_unmet",
    "target": {
     "kind": "manuscript",
     "locator": "figure captions; §5.1 swap definition"
    }
   },
   "target_section": "Figure captions; §5.1",
   "suggested_action": "Remove internal reconciliation notes and script paths from paper-facing text and name the adopted swap-success variant in §5.1.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "Paper-facing captions carry no internal notes or paths, and §5.1 names one swap-success variant.",
   "proposed_targets": [
    {
     "block_id": "B0051",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-20",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 0
    }
   ],
   "description": "§10 bootstraps anchors only, but items recur heavily across episodes (about 10 episodes per held ArtELingo painting per aspect pair, about 115 per SemArt test painting; CUB's 50 test species carry largely species-level attributes), so 'CI lower bound above 0' is anti-conservative for claims about new paintings or species. The DA's minor issues raise the same point.",
   "reviewer": "R1 W2",
   "sub_claim_ids": [
    "SC-21"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R1 W2 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§10 Statistics",
    "quote": "paired bootstrap over anchors (5,000 resamples)"
   },
   "confidence": 4,
   "competence_basis": "core expertise: cluster-robust bootstrap for ranking metrics",
   "confidence_source": "per-finding Confidence of R1 W2",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "§10 statistics: resampling unit for every pre-declared comparison and the GO test"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K2 and the GO decision"
    }
   },
   "target_section": "§10 Statistics",
   "suggested_action": "State the inference target; use a painting-level cluster bootstrap (species-level on CUB) that resamples items and regenerates or reweights episodes, or a two-way anchor-by-item bootstrap; base the power calculation on the same unit and report CUB results per species.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§10 names the resampling unit and the inference target, and the GO test and primary comparisons use it.",
   "proposed_targets": [
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-21",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 0
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 1
    }
   ],
   "description": "§10 defines 'beats' as a CI lower bound above 0 but gives 'matches' no criterion, so K3's 'match it on objective ones' passes by default when underpowered and can be refuted only by a significant loss. DA M3 also.",
   "reviewer": "R1 W3, DA M3",
   "sub_claim_ids": [
    "SC-22"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R1 W3 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 claims table, K3",
    "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones"
   },
   "confidence": 5,
   "competence_basis": "core expertise: equivalence testing for matches-the-baseline claims",
   "confidence_source": "per-finding Confidence of R1 W3",
   "corroborating_sources": [
    {
     "reviewer": "DA M3",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 K3; §12; aspect spike",
      "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones / narrow K3 to where names fail (subjective aspects) / on emotion only (+3.2 over CLIP only, 13.31 against 10.14)"
     },
     "confidence": 4,
     "competence_basis": "statistical logic; evidence from the appended spike"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§10 statistics and E13 pre-registration"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K3"
    }
   },
   "target_section": "§4 K3; §10 Statistics",
   "suggested_action": "Pre-declare an equivalence margin in R@1 points per dataset before E14, test it with two one-sided tests (or require the 90% CI inside the margin), and power the episode count for it; otherwise word K3 as 'not significantly different' and drop 'match'.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§10 states the margin, the test and its power, or K3 no longer uses 'match'.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-22",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 2
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 2
    }
   ],
   "description": "The K7 test (the agreement rule on the best unsupervised basis against the learned factors) is descriptive in §10, although it carries C2. DA M2 also notes that K6 and K7 have no decision rule.",
   "reviewer": "R1 W4, DA M2",
   "sub_claim_ids": [
    "SC-23"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R1 W4 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 K2 and §10",
    "quote": "the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive."
   },
   "confidence": 4,
   "competence_basis": "core expertise: aligning pre-registered tests with claim wording",
   "confidence_source": "per-finding Confidence of R1 W4",
   "corroborating_sources": [
    {
     "reviewer": "DA M2",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 K2 against §10",
      "quote": "Our method beats backbone-only, raw-feature metric-from-pairs baselines and the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive."
     },
     "confidence": 5,
     "competence_basis": "direct comparison of two sections of the plan"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§10 primary comparisons"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K7 and C2"
    }
   },
   "target_section": "§4 K7; §10 Statistics",
   "suggested_action": "Promote 'ours versus the rule on the best unsupervised basis, chosen on development' to a pre-declared primary comparison, or state K7 as descriptive and word C2 accordingly.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§10 lists the K7 comparison as primary with its baseline fixed on development, or K7 and C2 are worded as descriptive.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-23",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 2
    },
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 2
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 2
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 2
    }
   ],
   "description": "Each ArtELingo row is one viewer's emotion and a painting carries about five rows that can disagree (308,723 rows over 61,402 paintings), so an image-only predictor is capped by the share of rows matching the painting's modal emotion whatever the encoder; flat weak-side emotion probes (35.1 to 36.6 against a 31.8% majority) are what the label construction predicts. CONSENSUS-3: R1 W5, R2 W6, R3 W1; EIC silent. DA M1 also.",
   "reviewer": "R1 W5, R2 W6, R3 W1, DA M1",
   "sub_claim_ids": [
    "SC-24"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R1 W5 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 C3 and backbone check Result 1",
    "quote": "the effect persists across four backbones / a better encoder cannot read what the pixels or words do not"
   },
   "confidence": 3,
   "competence_basis": "adjacent: label-noise ceilings; construct validity is Reviewer 3's remit",
   "confidence_source": "per-finding Confidence of R1 W5",
   "corroborating_sources": [
    {
     "reviewer": "R2 W6",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "backbone check Verdict",
      "quote": "The weaker-modality probes barely move across four backbones, so the modality asymmetry is a property of the data"
     },
     "confidence": 3,
     "competence_basis": "adjacent expertise: affective annotation datasets; the ArtEmis per-annotation design is stated in the manuscript's §2.3"
    },
    {
     "reviewer": "R3 W1",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 C3 and backbone check Verdict",
      "quote": "the effect persists across four backbones, so it belongs / so the modality asymmetry is a property of the data"
     },
     "confidence": 4,
     "competence_basis": "core expertise: affective annotation protocols; adjacent: the CV probes"
    },
    {
     "reviewer": "DA M1",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 C3; backbone check; aspect spike",
      "quote": "the effect persists across four backbones, so it belongs / a better encoder cannot read what the pixels or words do not / an image takes the emotion of its row, which is a noisy image label"
     },
     "confidence": 4,
     "competence_basis": "confound reasoning; ArtEmis protocol as the plan itself describes it"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "E12 probes on scorer-train rows"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C3"
    }
   },
   "target_section": "§4 C3; §11 E12",
   "suggested_action": "On scorer-train rows, compute the label-construction ceiling for image-to-emotion (mean share of each painting's modal emotion), report weak-side probes as a fraction of it, and score image-side emotion against painting-level majority or distribution labels and on high-agreement paintings.",
   "consensus_level": "CONSENSUS-3",
   "verification_criteria": "E12 reports the ceiling, the normalised probe values and the painting-level results, and C3's wording follows them.",
   "proposed_targets": [
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-24",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 0
    }
   ],
   "description": "GeneCIS has no development split and a 1 + 1 read budget, yet E7 (Oct 13 to 20) runs the example protocol and baselines on GeneCIS focus attribute and outputs a 'GeneCIS table' before the E14 final reads, and E9 runs Qwen 'on every dataset'; as scheduled, E7 is a test read outside the ledger or a step with undefined output.",
   "reviewer": "R1 W6",
   "sub_claim_ids": [
    "SC-25"
   ],
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from R1 W6 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§11 E7",
    "quote": "GeneCIS focus attribute: example protocol, COCO-trained factors, baselines; text protocol as stretch / GeneCIS table"
   },
   "confidence": 3,
   "competence_basis": "core expertise: test-set reuse audits; E7's intended output is not fully specified",
   "confidence_source": "per-finding Confidence of R1 W6",
   "cost_scope": {
    "kind": "section",
    "locator": "§11 E7, E9 and E10 rows; §10 ledger"
   },
   "consequence_if_unaddressed": {
    "code": "method_reproducibility_unresolved",
    "target": {
     "kind": "dataset",
     "locator": "GeneCIS focus attribute"
    }
   },
   "target_section": "§10 Held budget; §11 E7, E9, E10",
   "suggested_action": "Limit E7 to pipeline checks on COCO, plus at most a small, disclosed slice of GeneCIS templates excluded from the final read; move every reported GeneCIS number to E14 and log any earlier read in the ledger; apply the same rule to E9 and E10.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§11 E7 produces no GeneCIS metric outside the ledger, and any earlier slice is disclosed and excluded from the final read.",
   "proposed_targets": [
    {
     "block_id": "B0087",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-25",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 7,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The GO test's fresh seed-43 episodes recombine the same selection paintings (about 19 appearances each per episode set), so best-of-ten selection can still inflate a GO; the method's own fusion weight is not stated as cross-fitted, and 'seed 43' names both an episode seed and a training seed. The DA's minor issues raise the same point.",
   "reviewer": "R1 W7",
   "sub_claim_ids": [
    "SC-26"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W7 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§6 Go/no-go",
    "quote": "from the same rows, so picking the best of about ten runs does not inflate the result"
   },
   "confidence": 4,
   "competence_basis": "core expertise: selection bias in model selection",
   "confidence_source": "per-finding Confidence of R1 W7",
   "cost_scope": {
    "kind": "section",
    "locator": "§6 Go/no-go Picking and GO bullets"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§6 Go/no-go"
    }
   },
   "target_section": "§6 Go/no-go",
   "suggested_action": "Split the selection paintings into two disjoint halves (pick on one, run the GO test on the other), cross-fit the method's weight like the baselines', and rename the two seed roles.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§6 states disjoint pick and test halves, the method's cross-fitting and distinct seed names.",
   "proposed_targets": [
    {
     "block_id": "B0073",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-26",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 9,
     "subclaim_ordinal": 0
    }
   ],
   "description": "Swap success is 0 for condition-blind scorers, about 25% for independent random scores and about 50% for antisymmetric ones, so its null depends on the scorer; its unit (anchors or anchor-directions) is unstated, and C0's 4.43 is reachable only over 8,192 anchor-directions (receipt AR11).",
   "reviewer": "R1 W9",
   "sub_claim_ids": [
    "SC-27"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W9 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "aspect-episode spike Result 2",
    "quote": "about 50% for any ordering: it reached 51.6 and 51.8 while R@1 was 8.0 and 8.6"
   },
   "confidence": 4,
   "competence_basis": "core expertise: metric design for paired conditional tests",
   "confidence_source": "per-finding Confidence of R1 W9",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§5.1 secondary metrics"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§5.1"
    }
   },
   "target_section": "§5.1 Common protocol",
   "suggested_action": "Add the condition-averaged pairwise accuracy (exactly 50% for every condition-blind scorer) and state whether swap success is computed per anchor or per anchor-direction.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§5.1 defines the pairwise accuracy and the swap-success unit.",
   "proposed_targets": [
    {
     "block_id": "B0051",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-27",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 10,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The headline CI bootstraps anchors over per-anchor seed means, so between-seed training variance never enters the interval.",
   "reviewer": "R1 W10",
   "sub_claim_ids": [
    "SC-28"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W10 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§10 Seeds",
    "quote": "The headline is the 3-seed mean, with its CI from bootstrapping anchors"
   },
   "confidence": 4,
   "competence_basis": "core expertise: seed variance in ML evaluation",
   "confidence_source": "per-finding Confidence of R1 W10",
   "cost_scope": {
    "kind": "section",
    "locator": "§10 Seeds"
   },
   "consequence_if_unaddressed": {
    "code": "reporting_requirement_unmet",
    "target": {
     "kind": "section",
     "locator": "§10 Seeds"
    }
   },
   "target_section": "§10 Statistics",
   "suggested_action": "Require each pre-declared comparison to pass for every seed, or report the minimum-seed effect, beside the pooled CI, and report the between-seed SD of each effect.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§10 states the per-seed rule and the seed SD reporting.",
   "proposed_targets": [
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-28",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 11,
     "subclaim_ordinal": 0
    }
   ],
   "description": "§10 does not say whether K2 is an intersection test over datasets and backbones or a claim on any subset, how the per-aspect-type 'beats' and 'matches' declarations enter the family, or which raw metric-from-pairs baseline counts as 'the best' before the final read. R1 cites this gap as decision-bearing for D1.",
   "reviewer": "R1 W11",
   "sub_claim_ids": [
    "SC-29"
   ],
   "obligation_class": "must_fix",
   "severity": "minor",
   "severity_source": "transported from R1 W11 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "absence",
    "locator": "§10 Statistics",
    "absence_scope": "a declared family and correction or intersection rule for the primary comparisons across datasets, backbones and aspect types",
    "check_performed": "§4 claims table, §10, §11 E13, §12 risks"
   },
   "confidence": 4,
   "competence_basis": "core expertise: multiplicity in pre-registered ML comparisons",
   "confidence_source": "per-finding Confidence of R1 W11",
   "cost_scope": {
    "kind": "section",
    "locator": "§10 primary comparisons and E13"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K2 and K3"
    }
   },
   "target_section": "§10 Statistics",
   "suggested_action": "Declare the family: an intersection-union rule for 'every dataset' and Holm across backbones and aspect types for anything weaker; fix 'the best' baseline on development.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§10 names the family, the correction or intersection rule and the development-chosen baseline.",
   "proposed_targets": [
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-29",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 12,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The power calculation has no target effect, alpha, power or resampling unit, and computed on anchor-level selection variance it overstates power under item reuse; beyond the item pool extra anchors add little (SemArt has 1,069 test paintings). R1 cites this gap as decision-bearing for D1.",
   "reviewer": "R1 W12",
   "sub_claim_ids": [
    "SC-30"
   ],
   "obligation_class": "must_fix",
   "severity": "minor",
   "severity_source": "transported from R1 W12 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§10",
    "quote": "come from a power calculation on selection variance (default 4,096 anchors per aspect pair)"
   },
   "confidence": 4,
   "competence_basis": "core expertise: power analysis for episodic benchmarks",
   "confidence_source": "per-finding Confidence of R1 W12",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "E13 power simulation"
   },
   "consequence_if_unaddressed": {
    "code": "method_reproducibility_unresolved",
    "target": {
     "kind": "section",
     "locator": "§10 Statistics"
    }
   },
   "target_section": "§10 Statistics; §11 E13",
   "suggested_action": "Specify the minimal effect of interest and the equivalence margin, compute power by simulation under the cluster bootstrap, and report item counts as well as anchor counts.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The E13 pre-registration states effect, alpha, power, unit and item counts per dataset.",
   "proposed_targets": [
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-30",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 13,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The method gets about ten grid runs plus seeds, but §8 fixes no grids for the baselines' internal hyperparameters (KISSME shrinkage and rank, RCA shrinkage, Xing steps, PCA and NMF dimension, SpLiCE sparsity), and both spikes picked lambda at a grid edge.",
   "reviewer": "R1 W13",
   "sub_claim_ids": [
    "SC-31"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W13 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "support-baseline spike, Caveats",
    "quote": "every pick for the prototype and probe terms sat at that edge"
   },
   "confidence": 3,
   "competence_basis": "core expertise: tuning parity; the baseline grids may exist outside the plan",
   "confidence_source": "per-finding Confidence of R1 W13",
   "cost_scope": {
    "kind": "section",
    "locator": "§8 baselines"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "claim",
     "locator": "K2"
    }
   },
   "target_section": "§8 Baselines",
   "suggested_action": "Pre-register a grid for each baseline at least as large as the method's, extend any grid whose pick lands on an edge, and report every pick.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§8 or the pre-registration lists each baseline grid and the edge rule.",
   "proposed_targets": [
    {
     "block_id": "B0079",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-31",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 14,
     "subclaim_ordinal": 0
    }
   ],
   "description": "§6 writes the score at raw scale while the spikes fuse per-episode z-scores with lambda, and raw-scale artifacts of both kinds have been seen; GeneCIS inherits fusion weights from other datasets though its image-to-image cosines sit on a different scale.",
   "reviewer": "R1 W14",
   "sub_claim_ids": [
    "SC-32"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W14 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§8 with §6 against the aspect spike",
    "quote": "they are chosen on the development split and frozen. GeneCIS, which has no development / s = β·cos + Σ_l w_l a_I,l(x) a_T,l(y) / The final score is z(cos) + λ · z(term)"
   },
   "confidence": 3,
   "competence_basis": "core expertise: score fusion; the final form may be intended but is not stated",
   "confidence_source": "per-finding Confidence of R1 W14",
   "cost_scope": {
    "kind": "section",
    "locator": "§6 model score and §8 fusion weights"
   },
   "consequence_if_unaddressed": {
    "code": "method_reproducibility_unresolved",
    "target": {
     "kind": "claim",
     "locator": "K5"
    }
   },
   "target_section": "§6 Method A; §8 Baselines",
   "suggested_action": "Fix one fusion form for every method and baseline (per-episode z-scoring makes weights transferable) and report GeneCIS sensitivity to the transferred weight.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§6 and §8 state one fusion form, and the GeneCIS table reports weight sensitivity.",
   "proposed_targets": [
    {
     "block_id": "B0068",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0079",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-32",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 15,
     "subclaim_ordinal": 0
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 1
    }
   ],
   "description": "§5.1 does not state that p_A must differ from the anchor on aspect B, nor balance the third or a correlated aspect across p_A, p_B and the negatives (style with genre, colour with species), which can tilt condition-blind scorers toward one condition. R1 W15 (Minor) and R3 W4 (Major); SPLIT on severity, left unresolved because the evidence that would settle it (the episode code and the realised imbalance) is not in the manuscript.",
   "reviewer": "R1 W15, R3 W4",
   "sub_claim_ids": [
    "SC-33"
   ],
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from R3 W4 (driving finding); R1 W15 rates it minor; severity dissent unresolved",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§5.3 and §5.1",
    "quote": "answer the objection that two aspects make the condition a binary switch / condition A uses S = P_A, C = P_B; condition B swaps them"
   },
   "confidence": 3,
   "competence_basis": "core expertise on the construct; the size of the effect in these datasets is untested",
   "confidence_source": "per-finding Confidence of R3 W4",
   "corroborating_sources": [
    {
     "reviewer": "R1 W15",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§5.1",
      "quote": "With three aspects, all three aspect pairs"
     },
     "confidence": 3,
     "competence_basis": "core expertise: episode construction; the code may already enforce exclusivity"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "E0 episode module: exclusivity rule and third-aspect rates"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§5.1 aspect episodes"
    }
   },
   "target_section": "§5.1 Common protocol; §11 E0",
   "suggested_action": "State the exclusivity rule for p_A and p_B, constrain or balance the third aspect across p_A, p_B and the negatives, and report R@1 split by whether the target shares the anchor's third-aspect value.",
   "consensus_level": "SPLIT",
   "verification_criteria": "§5.1 states exclusivity and the third-aspect control, and E0 reports third-aspect agreement rates by candidate role.",
   "proposed_targets": [
    {
     "block_id": "B0051",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-33",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 16,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The name for genre id 5 was inferred by elimination, and a wrong name lowers only the privileged-names side of the K3 comparison on genre.",
   "reviewer": "R1 W16",
   "sub_claim_ids": [
    "SC-34"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W16 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§5.3",
    "quote": "is inferred by elimination and alphabetical order"
   },
   "confidence": 3,
   "competence_basis": "adjacent: label provenance",
   "confidence_source": "per-finding Confidence of R1 W16",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§5.3 genre id list"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "claim",
     "locator": "K3 on genre"
    }
   },
   "target_section": "§5.3",
   "suggested_action": "Verify the name against the ArtGAN or WikiArt metadata, or exclude id 5 from genre episodes, and report K3 on genre without it as a check.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§5.3 cites the verified name or states the exclusion, and the K3 genre check is reported.",
   "proposed_targets": [
    {
     "block_id": "B0060",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-34",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 17,
     "subclaim_ordinal": 0
    }
   ],
   "description": "Four statements in the backbone appendix disagree with its tables or §2.3: CUB colour 'within 2.3 points' (the SigLIP 2 gap is 2.6); 'gaps of 21 to 35 points' omits style gaps of 44.8 and 45.6; '62 minutes' against a timing table that sums to about 87; '37,738 selection paintings with 92,413 captions' against §2.3's 6,451 paintings and 32,413 rows.",
   "reviewer": "R1 W17",
   "sub_claim_ids": [
    "SC-35"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W17 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "table",
    "locator": "backbone check Result 3, Primary colour (15, 21.0) row, SigLIP 2 cell 61.9 / 64.5"
   },
   "confidence": 5,
   "competence_basis": "direct recomputation from the manuscript's tables",
   "confidence_source": "per-finding Confidence of R1 W17",
   "cost_scope": {
    "kind": "sentence",
    "locator": "backbone-check appendix prose"
   },
   "consequence_if_unaddressed": {
    "code": "reader_traceability_reduced",
    "target": {
     "kind": "section",
     "locator": "backbone-check appendix"
    }
   },
   "target_section": "Backbone-check appendix",
   "suggested_action": "Correct the four statements and generate appendix prose numbers from the JSON.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "Each corrected statement matches its table or §2.3.",
   "proposed_targets": [
    {
     "block_id": "B0292",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0305",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0314",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-35",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 18,
     "subclaim_ordinal": 0
    },
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 1
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 1
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 2
    }
   ],
   "description": "K3's split into subjective and objective aspects has no outcome-independent criterion, and §11 places the per-aspect-type declaration at E13 (Oct 21 to 23), after E10's naming comparisons on development (Oct 12 to 20), so the assignment can be fitted to the outcomes. R2 W5 and R3 W3 (Major), R1 W18 (Minor, on the timing); SPLIT on severity, arbitrated Major. DA M3 also.",
   "reviewer": "R1 W18, R2 W5, R3 W3, DA M3",
   "sub_claim_ids": [
    "SC-36"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R2 W5 (driving finding); R1 W18 rates the timing part minor; SPLIT arbitrated to major",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 claims table K3",
    "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones"
   },
   "confidence": 3,
   "competence_basis": "core expertise: text-conditioned similarity baselines; the direction is my reading of a single spike on untrained factors",
   "confidence_source": "per-finding Confidence of R2 W5",
   "corroborating_sources": [
    {
     "reviewer": "R1 W18",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§11 E13",
      "quote": "Pre-registration and power for the final reads"
     },
     "confidence": 3,
     "competence_basis": "core expertise: pre-registration timing"
    },
    {
     "reviewer": "R3 W3",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 K3 and §12 R-names",
      "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones / narrow K3 to where names fail (subjective aspects)"
     },
     "confidence": 4,
     "competence_basis": "core expertise: similarity and naming in cognitive psychology"
    },
    {
     "reviewer": "DA M3",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 K3; §12; aspect spike",
      "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones / narrow K3 to where names fail (subjective aspects) / on emotion only (+3.2 over CLIP only, 13.31 against 10.14)"
     },
     "confidence": 4,
     "competence_basis": "statistical logic; evidence from the appended spike"
    }
   ],
   "cost_scope": {
    "kind": "section",
    "locator": "§4 K3, §10 per-aspect-type declaration, §11 E10 and E13"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K3"
    }
   },
   "target_section": "§4 K3; §10 Statistics; §11 E10 and E13",
   "suggested_action": "Fix, before E10, each aspect's assignment from a criterion that does not depend on model results (for example inter-annotator agreement, or nameability measured as zero-shot name-prompt accuracy), and keep it whatever E10 shows.",
   "consensus_level": "SPLIT",
   "verification_criteria": "The plan lists each aspect's type and the criterion, recorded before E10 runs.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-36",
   "source_refs": [
    {
     "seat": "R1",
     "channel": "finding",
     "ordinal": 19,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The backbone check ran CUB probes and retrieval on the standard 5,794-image test split, which includes the 50 unseen species reserved for the final CUB test; these reads informed the backbone choice and C3's 'colour is symmetric', and §10 discloses prior reads only for ArtELingo.",
   "reviewer": "R1 W19",
   "sub_claim_ids": [
    "SC-37"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R1 W19 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "backbone check Result 3",
    "quote": "Retrieval sanity check, 5,794 test images, first caption each"
   },
   "confidence": 3,
   "competence_basis": "inference: 5,794 matches CUB's standard test split size",
   "confidence_source": "per-finding Confidence of R1 W19",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "E0 CUB third-aspect choice and C3's CUB evidence on training species"
   },
   "consequence_if_unaddressed": {
    "code": "method_reproducibility_unresolved",
    "target": {
     "kind": "dataset",
     "locator": "CUB-200-2011 unseen-species test split"
    }
   },
   "target_section": "§10 Ledger; §11 E0",
   "suggested_action": "Log and disclose the read, run E0's choice and all CUB probes on training species only, and recompute C3's CUB evidence on training species.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The ledger records the read, and E0 and C3's CUB evidence use training species only.",
   "proposed_targets": [
    {
     "block_id": "B0086",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-37",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 1
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 1
    }
   ],
   "description": "K5's bar quotes four-task average R@1 (SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4) for an evaluation that runs only focus attribute; the same rows' focus-attribute column in the plan's own Table 2 reads 18.9, 17.9 and 19.4, and STiTch (21.1 focus attribute) is left out, so the bar is 2.0 to 4.5 points too low. DA M4 also.",
   "reviewer": "R2 W1, DA M4",
   "sub_claim_ids": [
    "SC-38"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R2 W1 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§5.4 item 2",
    "quote": "published frozen ViT-B/32 results: SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1"
   },
   "confidence": 4,
   "competence_basis": "core expertise: GeneCIS and zero-shot CIR reporting; numbers taken from the manuscript's own Table 2",
   "confidence_source": "per-finding Confidence of R2 W1",
   "corroborating_sources": [
    {
     "reviewer": "DA M4",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§5.4; literature review Table 2",
      "quote": "SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4 average R@1 / It is not comparable with published numbers. / ViT-B/32 17.9 / 14.8 / 14.6 / 16.1 / 15.9"
     },
     "confidence": 5,
     "competence_basis": "arithmetic check against the plan's own Table 2"
    }
   ],
   "cost_scope": {
    "kind": "sentence",
    "locator": "§5.4 item 2 and §4 K5 row"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K5"
    }
   },
   "target_section": "§4 K5; §5.4",
   "suggested_action": "Restate K5 against the focus-attribute column including STiTch, and report SQUARE and DIOR as out-of-class references.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§5.4 quotes focus-attribute values with sources, and K5 names its threshold against them.",
   "proposed_targets": [
    {
     "block_id": "B0064",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-38",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 2
    }
   ],
   "description": "§8 also drops the literature review's rank-2 Tip-Adapter cache and rank-3 linear probe on the eight support and contrast pairs, although the probe was the strongest scorer on the earlier episodes (24.10 against SE's 21.22).",
   "reviewer": "R2 W2",
   "sub_claim_ids": [
    "SC-39"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R2 W2 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "absence",
    "locator": "§8 baseline table and §10 pre-declared primary comparisons",
    "absence_scope": "a required, non-stretch baseline that gives an instruction embedder or open MLLM the same support and contrast pairs in context, and a verbalise-then-name baseline",
    "check_performed": "§8 tiers 1 to 3 and the stretch row, §10 statistics, §11 E10 and E17, novelty check §5 Table 2, literature review §3 Table 8"
   },
   "confidence": 4,
   "competence_basis": "core expertise: conditional similarity baselines and instruction-following multimodal embedders",
   "confidence_source": "per-finding Confidence of R2 W2",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "§8 Tier 1 on existing episodes"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "claim",
     "locator": "K2"
    }
   },
   "target_section": "§8 Baselines",
   "suggested_action": "Restore the per-episode linear probe and the Tip-Adapter cache in Tier 1.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§8 Tier 1 lists both baselines.",
   "proposed_targets": [
    {
     "block_id": "B0080",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-39",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 0
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 1
    }
   ],
   "description": "K7 compares the learned basis only with PCA, NMF and SpLiCE, leaving out sparse autoencoders, the dominant unsupervised sparse code on CLIP since 2024 (the plan's Table 5 lists several, including the shared-dictionary MGSAE), and the split-dictionary comparison that the literature review's prepared answer relies on is not planned. DA M5 makes the split-dictionary point too.",
   "reviewer": "R2 W3, DA M5",
   "sub_claim_ids": [
    "SC-40"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R2 W3 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§8 Tier 1 unsupervised-bases row",
    "quote": "PCA-32/64, NMF-32 and SpLiCE sparse concept codes instead of our factors"
   },
   "confidence": 3,
   "competence_basis": "core expertise: sparse concept codes on CLIP; whether SAE latents work with the rule at 4 plus 4 pairs is untested",
   "confidence_source": "per-finding Confidence of R2 W3",
   "corroborating_sources": [
    {
     "reviewer": "DA M5",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§8; literature review §5",
      "quote": "PCA-32/64, NMF-32 and SpLiCE sparse concept codes instead of our factors / GoEmotions cannot read images, so it cannot score cross-modal pairs / factors shared by image and text (against a split dictionary, as MGSAE warns)"
     },
     "confidence": 4,
     "competence_basis": "ablation design; checked §8, §11 E11 and literature review §5"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "K7 table: a jointly trained TopK SAE basis; E11 split-dictionary ablation"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K7 and C2"
    }
   },
   "target_section": "§8 Baselines; §11 E11",
   "suggested_action": "Add a TopK SAE trained jointly on CLIP image and caption features of the scorer-train rows, with sparsity matched to the factor codes (optionally the MGSAE recipe), run the agreement rule on its latents in the K7 table, and add the split-dictionary ablation to E11.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The K7 table and E11 include the SAE basis and the split-dictionary ablation.",
   "proposed_targets": [
    {
     "block_id": "B0080",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-40",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 1
    },
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 7,
     "subclaim_ordinal": 0
    }
   ],
   "description": "'No labels from the evaluation taxonomy' understates distant supervision: the emotion pseudo-partition is k-means over a GoEmotions classifier whose label set names 6 of the 8 evaluation emotions. R2 W4 (Major) and R3 W7 (Minor); SPLIT on severity, arbitrated Major because the overlap is confirmed by the manuscript's own literature review.",
   "reviewer": "R2 W4, R3 W7",
   "sub_claim_ids": [
    "SC-41"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R2 W4 (driving finding); R3 W7 rates it minor; SPLIT arbitrated to major",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "literature review §5 item 5",
    "quote": "a supervised GoEmotions classifier that names 6 of the 8 emotions"
   },
   "confidence": 4,
   "competence_basis": "core expertise: distant supervision in affective vision-language work; the label overlap is checkable in both taxonomies",
   "confidence_source": "per-finding Confidence of R2 W4",
   "corroborating_sources": [
    {
     "reviewer": "R3 W7",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§5.1 Splits",
      "quote": "Factor training uses image–caption pairs and pseudo-partitions only, no labels."
     },
     "confidence": 4,
     "competence_basis": "core expertise: emotion taxonomies"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "§4 C2 wording, §5.1 Splits and emotion results split by GoEmotions coverage"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C2"
    }
   },
   "target_section": "§4 C2; §5.1; §8 Out of the tables",
   "suggested_action": "Word the claim as 'no ArtELingo labels; distant supervision from a GoEmotions classifier whose label set overlaps 6 of the 8 evaluation emotions', report emotion results separately for the six covered and the two uncovered categories, and keep the teacher-only text-side analysis next to the main emotion row.",
   "consensus_level": "SPLIT",
   "verification_criteria": "C2 and §5.1 state the overlap, and the emotion table splits covered and uncovered categories.",
   "proposed_targets": [
    {
     "block_id": "B0046",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0051",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0082",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-41",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 2
    }
   ],
   "description": "Each ArtELingo training pseudo-partition is chosen to resemble one evaluation aspect (emotion-like, style-like with AMI 0.32, genre-like), so aspect selection on ArtELingo is weaker evidence of label-free learning than 'pseudo-aspects' suggests; CUB, with per-sentence caption clusters, is the cleaner test. DA C1 builds its critical on the same premise (separate item).",
   "reviewer": "R2 W4",
   "sub_claim_ids": [
    "SC-42"
   ],
   "obligation_class": "must_fix",
   "severity": "major",
   "severity_source": "transported from R2 W4 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "literature review §5 item 5",
    "quote": "a supervised GoEmotions classifier that names 6 of the 8 emotions"
   },
   "confidence": 4,
   "competence_basis": "core expertise: distant supervision in affective vision-language work; the label overlap is checkable in both taxonomies",
   "confidence_source": "per-finding Confidence of R2 W4",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "emotion aspect with and without the affect partition"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C2"
    }
   },
   "target_section": "§6 Pseudo-partitions; R-pseudo",
   "suggested_action": "State that the ArtELingo partitions were chosen to resemble the evaluation aspects, report the emotion aspect with and without the affect partition, and lead the label-free argument with CUB.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§6 discloses the matching, results include the run without the affect partition, and the label-free argument cites CUB ahead of ArtELingo.",
   "proposed_targets": [
    {
     "block_id": "B0071",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0074",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-42",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 7,
     "subclaim_ordinal": 0
    }
   ],
   "description": "Table 2 lists SEARLE 14.4 as CIReVL's ViT-B/32 re-run, while the pitfalls paragraph says 14.4 is CIReVL's ViT-L/14 run and RTD's B/32 run is 12.19, so the B/32 reference is 14.4 or 12.19 depending on the source.",
   "reviewer": "R2 W7",
   "sub_claim_ids": [
    "SC-43"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R2 W7 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "literature review §2.2 pitfalls",
    "quote": "SEARLE at ViT-L/14 circulates as 14.4 (CIReVL's run) and 12.26 (LinCIR's run)"
   },
   "confidence": 4,
   "competence_basis": "internal inconsistency, verifiable in the manuscript",
   "confidence_source": "per-finding Confidence of R2 W7",
   "cost_scope": {
    "kind": "sentence",
    "locator": "literature review §2.2 Table 2 and pitfalls paragraph"
   },
   "consequence_if_unaddressed": {
    "code": "reader_traceability_reduced",
    "target": {
     "kind": "table",
     "locator": "literature review Table 2"
    }
   },
   "target_section": "Literature review §2.2",
   "suggested_action": "Re-check CIReVL Table 3 and cite one backbone with its source table.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "Table 2, the pitfalls paragraph and §5.4 give one consistent SEARLE value and backbone.",
   "proposed_targets": [
    {
     "block_id": "B0121",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0123",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-43",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 8,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The agreement rule is a difference of uncentred diagonal cross-modal second moments (a_I a_T = (a_I^2 + a_T^2 - (a_I - a_T)^2)/2), closer to a diagonal cross-covariance contrast than to Rocchio or KISSME, yet the glossary names Rocchio and CSN as its ancestors.",
   "reviewer": "R2 W8",
   "sub_claim_ids": [
    "SC-44"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R2 W8 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "Appendix A glossary",
    "quote": "the classic ancestors of our rule and score"
   },
   "confidence": 4,
   "competence_basis": "core expertise: metric learning from pairs",
   "confidence_source": "per-finding Confidence of R2 W8",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§4 C2 and Appendix A glossary"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "Appendix A glossary"
    }
   },
   "target_section": "§4 C2; Appendix A",
   "suggested_action": "Describe the rule as a diagonal cross-covariance contrast, cite Rasiwasia et al. 2010 next to KISSME, and use the decomposition to answer the 'diagonal KISSME' objection precisely.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§4 C2, §6 and the glossary describe the rule's lineage consistently with the decomposition.",
   "proposed_targets": [
    {
     "block_id": "B0102",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-44",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 9,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The novelty check's 'we found no paired swap test (f) anywhere' is overstated: C-STS (EMNLP 2023) rates the same sentence pair under contrasting conditions, a text-condition analogue; C1's conjunction survives because C-STS is text-only with named conditions.",
   "reviewer": "R2 W9",
   "sub_claim_ids": [
    "SC-45"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R2 W9 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "novelty check §2",
    "quote": "we found no paired swap test (f) anywhere"
   },
   "confidence": 4,
   "competence_basis": "core expertise: conditional similarity; reference checked at its arXiv page",
   "confidence_source": "per-finding Confidence of R2 W9",
   "cost_scope": {
    "kind": "sentence",
    "locator": "novelty-check §2 and §4 C1"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C1 novelty statement"
    }
   },
   "target_section": "Novelty-check §2; §4 C1",
   "suggested_action": "Cite C-STS and state that the swap test differs by being paired, example-defined and cross-modal.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The novelty statement and related work cite C-STS with the stated distinction.",
   "proposed_targets": [
    {
     "block_id": "B0255",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-45",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "finding",
     "ordinal": 10,
     "subclaim_ordinal": 0
    }
   ],
   "description": "Multiview triplet embedding (Amid and Ukkonen, ICML 2015), which discovers several attribute-specific similarity maps from triplets without attribute labels, is missing from the label-free condition-discovery lineage.",
   "reviewer": "R2 W10",
   "sub_claim_ids": [
    "SC-46"
   ],
   "obligation_class": "consider",
   "severity": "minor",
   "severity_source": "transported from R2 W10 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "absence",
    "locator": "literature review §2.1 and §6, novelty check thread 1 and §6",
    "absence_scope": "multiview triplet embedding (Amid and Ukkonen, ICML 2015) as prior work that discovers several attribute-specific notions of similarity without attribute labels",
    "check_performed": "literature review Table 1, the §2.1 paragraph on learning conditions without labels, both reference lists, novelty check Table 1"
   },
   "confidence": 3,
   "competence_basis": "core expertise: conditional similarity lineage; relevance is my judgement",
   "confidence_source": "per-finding Confidence of R2 W10",
   "cost_scope": {
    "kind": "sentence",
    "locator": "literature review §2.1 and the novelty-check must-cite list"
   },
   "consequence_if_unaddressed": {
    "code": "reader_traceability_reduced",
    "target": {
     "kind": "section",
     "locator": "literature review §2.1"
    }
   },
   "target_section": "Literature review §2.1; novelty-check §4",
   "suggested_action": "Add it to the §2.1 condition-learning paragraph and the must-cite list.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The literature review §2.1 paragraph and the must-cite list include it.",
   "proposed_targets": [
    {
     "block_id": "B0117",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0266",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-46",
   "source_refs": [
    {
     "seat": "R2",
     "channel": "editorial",
     "ordinal": 11,
     "subclaim_ordinal": 0
    }
   ],
   "description": "R2 minor issues: add a short map from the plan's terms to the field's (aspect to CSN's notion or condition and GeneCIS's attribute type; value to a condition value), and keep the plan's own task term from the abstract onward. R2 also lists an UNVERIFIED search lead on open MLLMs given interleaved demonstrations for few-shot ranking.",
   "reviewer": "R2 (Minor Issues list)",
   "obligation_class": "consider",
   "source_kind": "editorial",
   "cost_scope": {
    "kind": "sentence",
    "locator": "Appendix A glossary"
   },
   "consequence_if_unaddressed": {
    "code": "reader_traceability_reduced",
    "target": {
     "kind": "section",
     "locator": "Appendix A glossary"
    }
   },
   "target_section": "Appendix A",
   "suggested_action": "Add the terminology map to the glossary, keep one task name throughout, and verify the search lead before relying on it.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The glossary maps aspect and value to CSN and GeneCIS terms, and the task name is used consistently.",
   "proposed_targets": [
    {
     "block_id": "B0102",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-47",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 3
    },
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 3
    }
   ],
   "description": "§5.3 asserts that genre is 'the one ArtELingo aspect visible in both modalities' without image or caption probes, although C3 relies on genre as its within-dataset symmetric control. DA M1 makes the same point.",
   "reviewer": "R3 W1, DA M1",
   "sub_claim_ids": [
    "SC-47"
   ],
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from R3 W1 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 C3 and backbone check Verdict",
    "quote": "the effect persists across four backbones, so it belongs / so the modality asymmetry is a property of the data"
   },
   "confidence": 4,
   "competence_basis": "core expertise: affective annotation protocols; adjacent: the CV probes",
   "confidence_source": "per-finding Confidence of R3 W1",
   "corroborating_sources": [
    {
     "reviewer": "DA M1",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§4 C3; backbone check; aspect spike",
      "quote": "the effect persists across four backbones, so it belongs / a better encoder cannot read what the pixels or words do not / an image takes the emotion of its row, which is a noisy image label"
     },
     "confidence": 4,
     "competence_basis": "confound reasoning; ArtEmis protocol as the plan itself describes it"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "genre probes on scorer-train rows"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "claim",
     "locator": "C3 within-dataset control"
    }
   },
   "target_section": "§5.3; §11 E12",
   "suggested_action": "Measure genre's image and caption probes before using genre as the symmetric control.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "E0 or E12 reports genre probe accuracy from image and from caption with the majority rate.",
   "proposed_targets": [
    {
     "block_id": "B0062",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-48",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 2,
     "subclaim_ordinal": 0
    }
   ],
   "description": "Emotion episode targets are single-viewer labels with about five rows per painting, yet the plan reports no inter-annotator agreement or human noise ceiling, so a reader cannot tell how much of the CLIP-to-probe gap is reachable, whether emotion differences between methods exceed label noise, or whether 'negative' paintings carry the anchor's emotion in other viewers' rows. The DA's unexamined-premise note (exact label equality) is related.",
   "reviewer": "R3 W2",
   "sub_claim_ids": [
    "SC-48"
   ],
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from R3 W2 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "absence",
    "locator": "§2.3, §5.1, §5.2 and §10",
    "absence_scope": "an inter-annotator agreement statistic and a human noise ceiling for the emotion labels that define episode targets",
    "check_performed": "§2.3 to §5.3, §8, §10 to §12, Appendix A, and the caveats of the support-baseline spike, aspect-episode spike and backbone check"
   },
   "confidence": 5,
   "competence_basis": "core expertise: annotator disagreement in emotion labels",
   "confidence_source": "per-finding Confidence of R3 W2",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "existing ArtELingo rows"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "dataset",
     "locator": "ArtELingo emotion labels"
    }
   },
   "target_section": "§5.2 Datasets; §11 E12",
   "suggested_action": "From existing rows, report per-painting modal-label share and Fleiss' kappa or Krippendorff's alpha over the 8 emotions, a leave-one-viewer-out oracle R@1 on the aspect episodes, the share of negative images whose painting has any row with the anchor's emotion, and a pre-registered sensitivity read on high-agreement paintings; optionally add a valence-level variant.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The datasets paragraph and E12 report agreement, the oracle R@1, the negative-contamination share and the high-agreement sensitivity read.",
   "proposed_targets": [
    {
     "block_id": "B0053",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-49",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 4
    }
   ],
   "description": "K3 conflates subjectivity (raters disagree, a property of the label) with nameability (how readily a respect can be put into words a text encoder handles); emotion is subjective but nameable, art style is curator-fixed but poorly nameable, and the appended evidence already splits along nameability (names lift emotion, not style).",
   "reviewer": "R3 W3",
   "sub_claim_ids": [
    "SC-49"
   ],
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from R3 W3 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 K3 and §12 R-names",
    "quote": "Examples beat naming the aspect on subjective aspects and match it on objective ones / narrow K3 to where names fail (subjective aspects)"
   },
   "confidence": 4,
   "competence_basis": "core expertise: similarity and naming in cognitive psychology",
   "confidence_source": "per-finding Confidence of R3 W3",
   "cost_scope": {
    "kind": "section",
    "locator": "§4 K3 and E13"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "claim",
     "locator": "K3"
    }
   },
   "target_section": "§4 K3; §11 E13",
   "suggested_action": "State K3 as a prediction over both axes, measuring subjectivity by inter-annotator agreement (CUB certainty ratings as a proxy) and nameability by zero-shot name-prompt accuracy on the strong modality, and pre-register each aspect's position on both.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "K3 and the pre-registration name both axes and each aspect's measured position.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-50",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 4,
     "subclaim_ordinal": 2
    }
   ],
   "description": "Supports for aspect A are not required to vary on the other labelled aspects and the contrast set rules out only aspect B, so the examples can pick out a correlated respect (style with genre, school with timeframe, colour and bill shape with species); every episode still chooses between two labelled aspects, so three aspects do not answer the binary-switch objection at the episode level.",
   "reviewer": "R3 W4",
   "sub_claim_ids": [
    "SC-50"
   ],
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from R3 W4 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§5.3 and §5.1",
    "quote": "answer the objection that two aspects make the condition a binary switch / condition A uses S = P_A, C = P_B; condition B swaps them"
   },
   "confidence": 3,
   "competence_basis": "core expertise on the construct; the size of the effect in these datasets is untested",
   "confidence_source": "per-finding Confidence of R3 W4",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "E0 episode module and E11 ablations"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "claim",
     "locator": "K6"
    }
   },
   "target_section": "§5.1; §5.3; §11 E0 and E11",
   "suggested_action": "Report the association between aspect labels and between each pseudo-partition and each label (Cramér's V or AMI); require supports to vary on every other labelled aspect, or report the third-aspect agreement rate inside S and for p_A; add a variant whose contrast set mixes B and C pairs; report R@1 per aspect pair.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "E0 or E11 reports label associations, third-aspect agreement in S, the mixed-contrast variant and per-aspect-pair R@1.",
   "proposed_targets": [
    {
     "block_id": "B0051",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0062",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-51",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 5,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The task restates respects for similarity (Medin, Goldstone and Gentner), and the dependence of factor weights on the contrast set is Tversky's diagnosticity, but §3 uses neither and cites Tversky 1977 only as background; the agreement rule is a diagnosticity estimator, which gives a falsifiable account of why the method works.",
   "reviewer": "R3 W5",
   "sub_claim_ids": [
    "SC-51"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from R3 W5 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "novelty check §4 must-cite list",
    "quote": "Tversky 1977 (similarity depends on the comparison context)"
   },
   "confidence": 4,
   "competence_basis": "core expertise: similarity theory",
   "confidence_source": "per-finding Confidence of R3 W5",
   "cost_scope": {
    "kind": "section",
    "locator": "§3 and §11 E11"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§3"
    }
   },
   "target_section": "§3 Problem definition; §11 E11",
   "suggested_action": "Add one paragraph to §3 mapping support and contrast to diagnosticity and the task to respects for similarity, and add one E11 ablation varying how many respects S and C differ on.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§3 contains the mapping and E11 lists the ablation.",
   "proposed_targets": [
    {
     "block_id": "B0042",
     "allowed_operations": [
      "insert_after"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-52",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 2
    }
   ],
   "description": "No robustness to realistic conditions is planned (1 or 2 supports, no contrast set, supports that include the query's value, noisy pairs), though the evaluation rules make the condition harder to supply than to name.",
   "reviewer": "R3 W6",
   "sub_claim_ids": [
    "SC-52"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from R3 W6 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§1 and §3",
    "quote": "A user shows, by a few example image–caption pairs, / The condition never shows the query's value and never names the aspect."
   },
   "confidence": 3,
   "competence_basis": "adjacent field: human factors of retrieval",
   "confidence_source": "per-finding Confidence of R3 W6",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "supplementary robustness runs"
   },
   "consequence_if_unaddressed": {
    "code": "evidence_gap_remains",
    "target": {
     "kind": "claim",
     "locator": "practical setting of C1"
    }
   },
   "target_section": "§11 E17",
   "suggested_action": "Report robustness to these conditions in the supplement.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The E17 supplement list includes the robustness runs.",
   "proposed_targets": [
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-53",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "finding",
     "ordinal": 8,
     "subclaim_ordinal": 0
    }
   ],
   "description": "The plan speaks of 'a fearful image' and of the emotion a painting conveys, but the label is one viewer's reported response; only ArtELingo's English part is used and the annotator population is never stated.",
   "reviewer": "R3 W8",
   "sub_claim_ids": [
    "SC-53"
   ],
   "obligation_class": "should_fix",
   "severity": "minor",
   "severity_source": "transported from R3 W8 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§2.4 and §2.3",
    "quote": "each a fearful image matched with a fearful caption of / English part"
   },
   "confidence": 4,
   "competence_basis": "core expertise: cross-cultural affective annotation",
   "confidence_source": "per-finding Confidence of R3 W8",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§2.3 and §2.4 wording; §5.2 datasets paragraph"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C3 and emotion statements"
    }
   },
   "target_section": "§2.3; §2.4; §5.2",
   "suggested_action": "Say 'the emotion annotators reported', state the English-only scope and the annotator population in the datasets paragraph, and, if available, run the C3 probes on ArtELingo's non-English annotations of the same paintings as a supplementary cross-cultural check.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The datasets paragraph states scope and population, and emotion statements are worded as reported responses.",
   "proposed_targets": [
    {
     "block_id": "B0023",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0039",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-54",
   "source_refs": [
    {
     "seat": "R3",
     "channel": "editorial",
     "ordinal": 9,
     "subclaim_ordinal": 0
    }
   ],
   "description": "R3 reading list for the framing and annotation items: Medin, Goldstone and Gentner 1993; Goodman 1972; Tversky 1977 (diagnosticity); Brown and Lenneberg 1954; Plank 2022; Aroyo and Welty 2015; Peterson et al. 2019; Russell 1980. R3 marks two leads UNVERIFIED (WikiArt Emotions' separate image-only and title-only conditions; the ArtEmis agreement statistics).",
   "reviewer": "R3 (reading recommendations)",
   "obligation_class": "consider",
   "source_kind": "editorial",
   "cost_scope": {
    "kind": "sentence",
    "locator": "related-work paragraph and Appendix B sources"
   },
   "consequence_if_unaddressed": {
    "code": "reader_traceability_reduced",
    "target": {
     "kind": "section",
     "locator": "related work"
    }
   },
   "target_section": "Related work; Appendix B",
   "suggested_action": "Verify the two UNVERIFIED leads at their sources before relying on them, and cite the verified works where the framing items use them.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "Any cited lead is verified at its source.",
   "proposed_targets": [
    {
     "block_id": "B0104",
     "allowed_operations": [
      "insert_after"
     ]
    }
   ]
  },
  {
   "id": "REV-55",
   "source_refs": [
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 1,
     "subclaim_ordinal": 0
    }
   ],
   "description": "DA C1, adjudicated VALIDATED: every ArtELingo evaluation aspect has a training pseudo-partition chosen to resemble it (one built on a supervised emotion classifier), CUB's partitions are part-focused caption clusters, and no cross-modal dataset holds an aspect out of training under a decision rule; the planned R-pseudo tests (cross-dataset and unseen-species results) keep the same aspects, so the design cannot tell example-conditioned inference of a respect from a K-way selector among trained aspect subspaces. Repairable (block_class repairable).",
   "reviewer": "DA C1",
   "obligation_class": "must_fix",
   "severity": "critical",
   "severity_source": "transported from DA C1 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§6 pseudo-partition table, ArtELingo row, against §4 C2",
    "quote": "GoEmotions affect k-means of captions (emotion-like; distant supervision); backbone image k-means (style-like; adjusted mutual information 0.32 with style); caption-content k-means (genre-like) / is the prior that makes it possible to estimate a similarity from just four example pairs"
   },
   "confidence": 4,
   "competence_basis": "adjacent expertise: few-shot and metric-learning evaluation design",
   "confidence_source": "per-finding Confidence of DA C1",
   "corroborating_sources": [
    {
     "reviewer": "R2 W4",
     "severity": "major",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "literature review §5 item 5",
      "quote": "a supervised GoEmotions classifier that names 6 of the 8 emotions"
     },
     "confidence": 4,
     "competence_basis": "core expertise: distant supervision in affective vision-language work; the label overlap is checkable in both taxonomies"
    },
    {
     "reviewer": "R3 W7",
     "severity": "minor",
     "evidence_anchor": {
      "anchor_type": "text",
      "locator": "§5.1 Splits",
      "quote": "Factor training uses image–caption pairs and pseudo-partitions only, no labels."
     },
     "confidence": 4,
     "competence_basis": "core expertise: emotion taxonomies"
    }
   ],
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "a held-out-aspect run on ArtELingo and generic partitions on CUB"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C2 mechanism claim and the motivation of C1"
    }
   },
   "target_section": "§6 Pseudo-partitions and R-pseudo; §10 Statistics",
   "suggested_action": "Add a leave-one-aspect-out run on ArtELingo (for example train without the caption-content partition and test genre episodes) and generic, attribute-agnostic partitions on CUB, each with a decision rule fixed in E13 against the raw metric-from-pairs baseline; optionally a partition bank larger than the evaluation aspects built without looking at them. If no held-out-aspect test is run, narrow C2 and the C1 motivation to selecting among aspects represented in the training partitions.",
   "consensus_level": "DA-CRITICAL",
   "verification_criteria": "§6 and §10 list the held-out-aspect test or tests with a pre-registered rule, or C1 and C2 are reworded to the selector reading.",
   "proposed_targets": [
    {
     "block_id": "B0071",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0074",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-56",
   "source_refs": [
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 3,
     "subclaim_ordinal": 3
    }
   ],
   "description": "DA M2, DA-only part: K6 (the rule selects aspect factors and the factors are shared) has no decision rule, swap analysis alone cannot carry it (about 50% for any antisymmetric scorer at chance R@1), and pooling lets a genre-only gain pass the GO and primary comparisons while emotion, the lead example, has no protection.",
   "reviewer": "DA M2",
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from DA M2 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§4 K2 against §10",
    "quote": "Our method beats backbone-only, raw-feature metric-from-pairs baselines and the same rule on unsupervised bases, on aspect episodes, in both directions, on every dataset / All other comparisons are descriptive."
   },
   "confidence": 5,
   "competence_basis": "direct comparison of two sections of the plan",
   "confidence_source": "per-finding Confidence of DA M2",
   "cost_scope": {
    "kind": "section",
    "locator": "§10 statistics"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "K6"
    }
   },
   "target_section": "§4 K6; §10 Statistics",
   "suggested_action": "Pre-declare a decision rule for K6 and a per-aspect rule at least for emotion, with a multiplicity plan, or reword K6 to what a descriptive analysis supports.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "§10 lists the K6 and per-aspect rules, or K6 is worded as descriptive.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0090",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-57",
   "source_refs": [
    {
     "seat": "DA",
     "channel": "finding",
     "ordinal": 6,
     "subclaim_ordinal": 2
    }
   ],
   "description": "DA M5, DA-only part: every planned K7 control either has no training signal or is fit on eight pairs, so a K7 win mixes structure with the pseudo-partition training signal; missing are a per-partition probe trained on scorer-train rows (for emotion, the GoEmotions teacher distilled into an image probe, which §8 excludes on the premise that GoEmotions cannot read images) and a dense, unconstrained projection trained on the same episodes.",
   "reviewer": "DA M5",
   "obligation_class": "should_fix",
   "severity": "major",
   "severity_source": "transported from DA M5 (driving finding)",
   "evidence_anchor": {
    "anchor_type": "text",
    "locator": "§8; literature review §5",
    "quote": "PCA-32/64, NMF-32 and SpLiCE sparse concept codes instead of our factors / GoEmotions cannot read images, so it cannot score cross-modal pairs / factors shared by image and text (against a split dictionary, as MGSAE warns)"
   },
   "confidence": 4,
   "competence_basis": "ablation design; checked §8, §11 E11 and literature review §5",
   "confidence_source": "per-finding Confidence of DA M5",
   "cost_scope": {
    "kind": "re_analysis",
    "locator": "E11 ablations"
   },
   "consequence_if_unaddressed": {
    "code": "claim_scope_unsupported",
    "target": {
     "kind": "claim",
     "locator": "C2"
    }
   },
   "target_section": "§8 Baselines and Out of the tables; §11 E11",
   "suggested_action": "Add the distilled per-partition probe scored by the agreement rule and a dense projection trained on the same episodes to the K7 table or E11.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "The K7 table or E11 includes both controls.",
   "proposed_targets": [
    {
     "block_id": "B0080",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0082",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0092",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  },
  {
   "id": "REV-58",
   "source_refs": [
    {
     "seat": "DA",
     "channel": "editorial",
     "ordinal": 7,
     "subclaim_ordinal": 0
    }
   ],
   "description": "DA minor issues not carried by other items: K1's 'solved by the supports alone' overstates the evidence (a query-free prototype beats the factor model and the query adds 0.57); CUB's third aspect is chosen by the highest min(image, caption) probe gain, so CUB as C3's symmetric control partly reflects that selection; 'swap success is reported at matched R@1' does not say how R@1 is matched; the backbone-only GO reference of 11.1 comes from the two-aspect spike and should be recomputed on the new three-aspect episodes.",
   "reviewer": "DA (Minor issues list)",
   "obligation_class": "consider",
   "source_kind": "editorial",
   "cost_scope": {
    "kind": "sentence",
    "locator": "§4 K1, §5.1, §6 GO and §11 E0"
   },
   "consequence_if_unaddressed": {
    "code": "interpretive_ambiguity_remains",
    "target": {
     "kind": "section",
     "locator": "§6 Go/no-go"
    }
   },
   "target_section": "§4 K1; §5.1; §6 Go/no-go",
   "suggested_action": "Reword K1, disclose the CUB selection criterion beside C3, define how R@1 is matched for swap reporting, and recompute the backbone-only GO reference on the E0 episodes.",
   "consensus_level": "SINGLE-VERIFIER",
   "verification_criteria": "K1, C3's CUB evidence, the swap reporting rule and the GO reference reflect these corrections.",
   "proposed_targets": [
    {
     "block_id": "B0047",
     "allowed_operations": [
      "replace_block"
     ]
    },
    {
     "block_id": "B0073",
     "allowed_operations": [
      "replace_block"
     ]
    }
   ]
  }
 ],
 "total_items": 58,
 "obligation_counts": {
  "must_fix": 23,
  "should_fix": 19,
  "consider": 16
 },
 "editorial_decision": "Major Revision",
 "consensus_summary": "Mechanical decision major_revision (F2 fired: D1 block from R1, D3 block from the DA; F3 and F5 also fired). No CONSENSUS-4 item. CONSENSUS-3: K3's adverse evidence and outcome-proof fallback (EIC, R2, R3; R1 silent) and the per-viewer cap on the image-side emotion probe (R1, R2, R3; EIC silent). Six SPLITs, five on severity alone and arbitrated (use case, C3 wording, K2 wording, K3 aspect-type criterion, GoEmotions overlap) and one left unresolved (third-aspect balance). DA C1 VALIDATED as a repairable design gap.",
 "dissenting_opinions": [
  "R1 S6 credits the CUB unseen-species split as a clean test of R-pseudo; DA C1 holds that it cannot test aspect-level generalisation. Arbitrated for the DA on scope: unseen species test new items, not unseen aspects.",
  "Severity of SC-33 (exclusivity and third-aspect balance of candidates): R1 W15 Minor, R3 W4 Major; unresolved, the evidence needed (episode code, realised imbalance) is not in the manuscript.",
  "SC-20 strong-GO reference: EIC W16 proposes the gain over backbone-only, R1 W8 the effect over the best baseline; left to the authors, who state the reference they use.",
  "EIC W8 rates the C3 wording Minor ('the fix costs one sentence'); R1, R2 and R3 rate it Major. Arbitrated Major; the EIC's position is recorded."
 ]
}
```
