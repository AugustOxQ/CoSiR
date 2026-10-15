contract_role: eic
## Dimension Scores

### D1: methodology_rigor
score: not_assessed

### D2: domain_accuracy
score: not_assessed

### D3: argumentative_coherence
score: not_assessed

### D4: cross_disciplinary_relevance
score: not_assessed

### D5: writing_and_structure
score: warn
trigger: "Localised problems that still leave every claim traceable"

### D6: venue_fit_and_contribution
score: warn
trigger: "The contribution is testable but under-argued"

## Review Body

### Reviewer Identity and Scope

Journal-Fit Reviewer, configured as a CVPR 2027 area chair for vision and language, multimodal retrieval and representation learning. Venue binding: `criteria_binding_unavailable`. My fit judgements below are readership and significance judgements for a top computer-vision conference, not alignment with bound official criteria. Calibration status: `NOT_CALIBRATED`.

I read the manuscript as a pre-results plan. Missing results are not a defect by themselves. The "(approved)" labels and the scripted reviewer objections with prepared answers are author material, and I judged them on the evidence. I do not audit the bootstrap, the resampling unit or the ledger mechanics (methodology seat), or the construct validity of affective labels (perspective seat), except where they change what the eventual paper can claim.

### Summary Assessment

The plan defines a new task, example-conditioned aspect similarity across modalities. A few cross-item image and caption pairs show the respect in which a query should match candidates of the other modality, contrast pairs show a second respect, and a swap test exchanges the two. The method trains a shared sparse factor basis on pseudo-aspect episodes and reads factor weights from the examples with a Rocchio/KISSME-style rule. The document is honest and well organised. It retired its own earlier benchmark after finding a query-free shortcut, it places the rule next to its closest ancestors, and it backs the final reads with a ledger. The topic sits inside CVPR's vision-language retrieval readership.

The contribution is testable but under-argued for a top-venue headline. The go/no-go and the primary comparisons rest on R@1, which a condition-blind gain can raise (W1). The case for examples over names (K3) carries the significance argument, meets adverse appended evidence and has no failure branch (W2). The only community-standard comparison depends on a stretch protocol (W3). The "first" claim rests on searches the authors call incomplete (W4), in-context alternatives given the same examples are stretch-only or missing (W5), and the NO-GO fallback is not yet a CVPR paper (W6). Each is repairable before the Oct 9 decision and none is fatal, so I score D6 warn; the exposition issues are localised, so D5 is warn too. Preliminary signal, as input only: substantial revision of the plan before E3 runs.

### Fit, Originality and Significance

Fit is good. Conditional similarity and composed retrieval on frozen vision-language encoders are an active CVPR topic (GeneCIS, CSN and its successors, CLAY), and a cross-modal variant with an example interface would interest that audience.

Originality is a combination, and the plan says so. The novelty-check appendix found no paper that defines the task, while every ingredient exists (metric learning from pairs, contextual visual similarity, relation by example, in-context text embedders). Claiming only the combination is the right posture.

Significance is the open question. A combination clears an AC meta-review only if the problem matters. The plan never states who holds value-disjoint, cross-item demonstration pairs plus contrast pairs but cannot name the aspect. Its own literature appendix states the stakes: "Examples must win where the aspect is hard to name, or the paper needs another argument". That argument is K3, which is open and currently disfavoured (W2). The method contribution (C2) is significant only if K7 shows that the learned basis, not the rule, carries the gain; the plan correctly makes the rule on raw and unsupervised bases a Tier-1 comparison.

### Storyline Across Branches

A strong GO with the CUB check passed yields a coherent paper: a new task, a lightweight method that beats closed-form metrics from pairs and a named-aspect instruction embedder, and an analysis of modality asymmetry. A GO that narrows to ArtELingo after the CUB check leaves a method paper on one affective dataset whose held rows were read three times before and whose label-probe reference sits near 23 R@1; I expect an AC to read that as below the bar for a main-conference paper. The NO-GO branch is treated in W6. With the factors at CLIP level today, the narrowed and NO-GO branches are live outcomes, so their headlines should be written before Oct 9, not after.

### Structure and Exposition

As a planning document the structure works. Problem, claims with status, benchmarks, method, baselines, read budget, schedule and risks follow in order; a glossary defines the terms; and §11 ties each experiment to a claim. For the eventual eight-page paper the problems are local: no outline or page budget, a vocabulary inherited from project history, a "ceiling" that is not a bound, and dates that blur the order of the evidence (W13 to W16). Every claim stays traceable to its planned test, so D5 is a warn and not a block.

### S1: Claims are tied to evidence and status
The §4 table lists each claim with its evidence and status, separating the one finished claim (K1) from six open ones, and §11 names the claim each experiment feeds. This is the discipline that keeps an abstract honest.
**Evidence Anchor**: table: §4 claims table — K1 status "done (support spike)" beside K2 to K7 marked open, each with its evidence column

### S2: Negative evidence redirected the problem
The support-baseline appendix found that a scorer ignoring the query solves the old value episodes. The plan redefined the task around that finding instead of explaining it away, and a second spike confirmed that value baselines fail on the new aspect episodes.
**Evidence Anchor**: text: support-baseline spike verdict "A prototype scorer that never looks at the query reaches 22.83, already above SE"

### S3: Prior-art positioning is candid
The plan names the rule's ancestors (Rocchio, KISSME, CSN), lists what it will not claim, and builds the expected "diagonal KISSME" objection into its Tier-1 baselines.
**Evidence Anchor**: text: §4 C2 "presented openly as a Rocchio/KISSME-style estimator, not as the novelty"

### S4: Final reads are budgeted and enforced
One main and one reserve read per dataset, a ledger with script and episode hashes, scripts that refuse a second run, and a disclosure of the earlier ArtELingo reads give the headline a protocol a CVPR reader can trust.
**Evidence Anchor**: text: §10 ledger "Each final script refuses a second run."

### S5: The task definition is precise and guards against a known artifact
Input, output and rules are stated in §3, and §5.1 explains why swap success must sit next to R@1, a lesson taken from the antisymmetry artifact in the aspect spike.
**Evidence Anchor**: text: §5.1 secondary metrics "because a scorer can flip with the condition without being right"

### S6: CUB with captions adds a recognised dataset, a clean test split and a symmetric control
CUB-200-2011 with Reed captions brings a dataset the community knows, an unseen-species test split, captions written without species names, and a symmetric aspect (colour) that C3 needs as its contrast.
**Evidence Anchor**: table: §5.2 datasets table, CUB-200-2011 row — test split the standard zero-shot 50 unseen species, development on 30 of the 150 training species

### S7: Tier-1 baselines answer the likeliest objection
Low-rank KISSME with shrinkage, RCA, a per-episode Xing fit, Wang et al.'s per-query weights, and the same rule on PCA, NMF and SpLiCE bases test whether the learned basis (K7) carries any gain. The named-aspect instruction embedder runs on the same backbone.
**Evidence Anchor**: table: §8 baselines table, Tier-1 metric-from-pairs row — diagonal agreement rule, low-rank KISSME with shrinkage, RCA, Xing-style per-episode fit, Wang et al. per-query weights

### S8: A glossary fixes the terms
Appendix A defines the vocabulary once, which keeps the plan readable despite its density and gives the paper a starting point for its definitions.
**Evidence Anchor**: text: Appendix A swap entry "exchanging supports and contrasts must flip which aspect candidate wins"

### W1: The go/no-go and primary comparisons can pass a condition-blind gain (D6)
**Problem**: The GO rule and the three pre-declared primary comparisons of §10 use pooled aspect R@1 only. Swap success is "reported at matched R@1" and the other-aspect rate is secondary. Because both conditions share the anchor and the candidates, a scorer that ignores the condition has pooled R@1 equal to its other-aspect rate; the aspect spike shows exactly this for CLIP only (11.13 and 11.13). Such a scorer can still raise R@1 toward 50% by ranking both aspect candidates above the 11 negatives. A factor model that improves cross-modal value matching on both aspects at once could reach the strong-GO mark of about 15 without selecting any aspect.
**Evidence Anchor**: text: §6 Go/no-go, GO bullet "if, on those fresh episodes, pooled aspect R@1 has a 95% CI lower bound above 0 against"
**Why it matters**: The headline would then report a cross-modal matching gain, not example-conditioned similarity, and C1 and C2 would rest on the K6 analyses instead of the pre-registered decision.
**Suggestion**: Gate the GO and K2 on a condition-attributable statistic with the same CI rule, for example R@1 minus the other-aspect rate, or the paired difference between the model under its condition and under the swapped or uniform-weight condition. Add "our factors, condition removed" as a Tier-1 control.
**Severity**: Major
**Confidence**: 4 — evaluation-design reading checked against §5.1 and the aspect-spike CLIP rows; the full shortcut audit belongs to the methodology seat

### W2: The case for examples over names meets adverse evidence and has no failure branch (D6)
**Problem**: K3 (examples beat naming on subjective aspects and match it on objective ones) is the plan's answer to "why examples?". The only appended test points the other way on the subjective aspect: on aspect episodes the privileged names reference reaches 13.31 on emotion, against 10.14 for CLIP only and 10.34 for SE with the agreement rule, and no example-based scorer leaves the CLIP band. That namer sat at majority-class accuracy, so stronger namers (CRL, Qwen with the aspect in its instruction) may widen the gap. The R-names mitigation, "narrow K3 to where names fail (subjective aspects)", points at the aspect where the evidence currently favours names, so the plan has no branch for names winning everywhere. It also names no use case in which demonstration pairs exist but the aspect cannot be named.
**Evidence Anchor**: table: aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14
**Why it matters**: Without K3 or a concrete use case, the task reads as a harder way to specify what a 2026 instruction embedder accepts as text. That is the first objection an AC will raise against C1.
**Suggestion**: Pre-declare what the paper claims if names win on every aspect (for example, examples that compose with names, or groupings with no stable name), state a use case with its data source, and run the Qwen-instruction comparison on selection rows before Oct 9, which the backbone check's selection-row Qwen pipeline makes cheap.
**Severity**: Major
**Confidence**: 4 — significance is core to my remit; numbers read from the appended spike

### W3: The only community-standard comparison depends on a stretch protocol (D6)
**Problem**: K5 promises competitiveness with the published frozen-B/32 GeneCIS rows, but §5.4 declares the example protocol not comparable with published numbers and makes the text protocol, the only comparable one, a stretch. GeneCIS is also image to image, so even its comparable row measures transfer, not the cross-modal task. Every cross-modal result is therefore on aspect episodes the authors build themselves (ArtELingo, CUB, SemArt).
**Evidence Anchor**: text: §5.4 example protocol "It is not comparable with published numbers."
**Why it matters**: The user's own third requirement asks for clear baselines on comparable benchmarks. An AC weighs a home-built-only evaluation of a new task heavily, and the published frozen-B/32 bar is itself soft: OSrCIR's 17.4 was reproduced at 14.0, and training-free LVLM pipelines report higher focus-attribute numbers.
**Suggestion**: Either take the text protocol off the stretch list and fix it in the pre-registration, or drop K5 and say plainly that GeneCIS measures transfer. In both cases, make the home-built episodes reusable (W10) so they can serve as the external anchor.
**Severity**: Major
**Confidence**: 4 — benchmark expectations in this subfield are core to my remit

### W4: The "first" claim rests on searches the authors call incomplete, and no step completes them (D6)
**Problem**: Both literature appendices report exhausted search budgets. The novelty check ran about 25 arXiv API queries with no web search engine, a rate-limited Semantic Scholar and no Google Scholar. §11 schedules no literature completion before the title and abstract are fixed on Nov 7.
**Evidence Anchor**: text: novelty-check appendix, "Searched, nothing found" paragraph "This is a negative result from an incomplete search (no web search engine, Semantic Scholar rate-limited, no Google Scholar)"
**Why it matters**: C1 is a priority claim. One missed paper on example-conditioned cross-modal retrieval would remove it, and with heavy concurrent 2026 activity (TPIPS, SteerViT, CLAY, COCO-Facet, Fioresi et al.) that risk is not small.
**Suggestion**: Add an E-row before Nov 7 for a full search (Google Scholar, Semantic Scholar, forward citations of KISSME, Wang et al. 2016, MARS, BGE-EN-ICL and GeneCIS, and the CVPR 2026 and ECCV 2026 proceedings), with a pre-declared rewording if a match appears.
**Severity**: Major
**Confidence**: 4 — the gap is documented by the authors; I cannot tell whether a missed paper exists

### W5: In-context alternatives given the same examples are stretch-only or missing (D6)
**Problem**: The novelty check's own baseline table lists an embedder given the same support and contrast pairs in context, an in-context MLLM reranker and verbalise-then-name as the baselines this task implies. §8 keeps only the reranker, in the stretch row beside CLAY, and drops the other two. None appears among the §10 primary comparisons.
**Evidence Anchor**: absence: §8 baselines table and §10 primary comparisons — expected the in-context baselines of novelty-check §5 Table 2 (an embedder or MLLM given the same support and contrast pairs, and verbalise-then-name) as scheduled baselines; checked §8 tiers 1 to 3 and the stretch row, §10, §11 E1, E10 and E17
**Why it matters**: "Would a current vision-language model given the same examples solve this?" is the obvious question for an example-conditioned task at a 2027 venue. Without it, a GO shows that learned factors beat closed-form estimators and named instructions, which supports C2 but leaves the significance of the lightweight approach open.
**Suggestion**: Schedule the off-label in-context embedder on Qwen3-VL-Embedding-2B (the backbone is already chosen) and verbalise-then-name in E10, on a subsample if cost requires, and report them descriptively beside the primary comparisons.
**Severity**: Major
**Confidence**: 4 — what the 2027 readership will expect is core to my remit

### W6: The NO-GO fallback is not yet a CVPR paper (D6)
**Problem**: The fallback rests on K1, K3 and C3. K1 is a shortcut in the authors' own earlier benchmark that, as the support-baseline appendix notes, GeneCIS's gallery design already avoids, so it has little standalone value for readers. K3 needs a working example-based scorer, and under NO-GO none exists: every example scorer in the aspect appendix sits at CLIP level while privileged names gain on emotion (W2). C3 is an analysis of probe accuracies on one affective dataset plus three CUB attribute groups. The plan names no target venue or bar for this paper and defers the choice to the Oct 9 discussion.
**Evidence Anchor**: text: §4 Fallback "If K2 fails, K1, K3 and C3 could carry a task, benchmark and analysis paper but not the method paper."
**Why it matters**: With the factors at CLIP level on aspect episodes today, NO-GO is a live outcome, and it would leave about five weeks. An undefined fallback is where the schedule is most likely to break.
**Suggestion**: Write the fallback's claim table now (a released benchmark, strong in-context and named baselines, a defined subjective versus objective split, C3 tested across annotation protocols), name the venue it targets, and pre-declare the criteria for switching.
**Severity**: Major
**Confidence**: 4 — storyline robustness across branches is core to my remit

### W7: The novelty statement counts evaluation choices as task properties (D6)
**Problem**: The C1 statement makes "first" a conjunction that includes testing in both directions with a paired swap test, and the novelty check scores this protocol property as one of the six on which no prior row reaches more than three. Directions and a swap test are evaluation choices any prior method could adopt.
**Evidence Anchor**: text: §4 C1 novelty statement "evaluated in both directions with a paired swap test"
**Why it matters**: An AC reads a priority claim that leans on protocol qualifiers as defensive, which undercuts positioning that is candid elsewhere (S3).
**Suggestion**: State the novelty on the problem properties (an aspect fixed only by value-disjoint, cross-item image and caption demonstrations with a contrast aspect), and present both directions and the swap test as the protocol that makes the task measurable.
**Severity**: Minor
**Confidence**: 4 — novelty framing is core to my remit

### W8: C3's "belongs to the data" outruns the backbone evidence (D6)
**Problem**: Flat weak-side probes across four encoders show that better encoders do not recover an aspect from its weak modality. They cannot separate what images and captions contain from how the captions were collected: each ArtELingo caption explains one viewer's emotion, and ArtEmis already reports 65.7% text-to-emotion accuracy. CUB's symmetric colour fits the same reading, since its captions name colours.
**Evidence Anchor**: text: §4 C3 "the effect persists across four backbones, so it belongs to the data"
**Why it matters**: C3 supports both the GO storyline and the fallback; an overstated causal phrase is an easy target, and the fix costs one sentence.
**Suggestion**: Say "belongs to the annotations as collected", and use SemArt's catalogue text as a third collection protocol in the analysis. The construct question belongs to the perspective seat.
**Severity**: Minor
**Confidence**: 3 — adjacent to my remit; I rely on the appendices' descriptions of the captions

### W9: K2 is worded more strongly than §10 tests it (D6)
**Problem**: K2 claims wins in both directions on every dataset, while the §10 primary metric pools over directions and aspect pairs, and §11 already plans to narrow claims if the CUB check fails. No rule says what the paper claims if one final read (GeneCIS or SemArt) misses.
**Evidence Anchor**: text: §4 claims table, K2 "in both directions, on every dataset"
**Why it matters**: The abstract is fixed before the final review and any reserve read (W12); a conjunctive headline invites a promise the tables may not keep.
**Suggestion**: Restate K2 on the pooled primary metric per dataset, report directions as secondary, and pre-declare in E13 the headline wording for partial success.
**Severity**: Minor
**Confidence**: 4 — structural coherence of claims is core to my remit

### W10: No release plan for a task and benchmark contribution (D6)
**Problem**: C1 is a task and protocol, and the fallback is a benchmark paper, yet the plan never says whether the aspect-episode builder, the episode files, the split lists or the trained factors will be released, or how the non-commercial dataset terms constrain that.
**Evidence Anchor**: absence: plan body §4 to §12 — expected a release plan for the aspect-episode builder, episode files and split lists; checked §4 C1 and the fallback, §5.1 to §5.4, the §10 ledger, §11 E0 and E17, §12 R-licence
**Why it matters**: A new task gains readers only if others can run it. An AC will ask, and the answer sets how much C1 is worth.
**Suggestion**: Commit to releasing episode indices and the builder (which avoids redistributing images), state the terms of the derived files, and add the release to E17.
**Severity**: Minor
**Confidence**: 4 — a standard expectation for new-task papers at vision venues

### W11: The cut list has one item, and it is a clean test set (D6)
**Problem**: R-time's only mitigation moves SemArt to the supplement first. SemArt's official test and CUB's unseen species are the never-read test sets, while ArtELingo's held rows, the primary set, were read three times and shaped earlier design. No second cut is named across four datasets, two backbones, about twenty baselines and ablations, and three seeds in about five weeks.
**Evidence Anchor**: text: §12 R-time "staged priority; SemArt moves to the supplementary first"
**Why it matters**: Under time pressure the headline drifts toward the pre-read set, and the order of cuts decides which evidence the abstract rests on.
**Suggestion**: Pre-declare a full cut order (for example, Qwen runs on secondary datasets and stretch baselines go before any never-read test set), and keep at least one never-read test set beside ArtELingo in the main paper.
**Severity**: Minor
**Confidence**: 3 — planning judgement; the compute estimates are the authors'

### W12: The final review falls after the abstract freeze (D5)
**Problem**: E15 fixes the title and abstract on Nov 7 and registers on Nov 10, while E16, the whole-branch review that may trigger a reserve read, runs Nov 11 to 14.
**Evidence Anchor**: text: §11 E15 and E16 rows "title and abstract fixed Nov 7" and "Nov 11 to 14"
**Why it matters**: A review finding that changes a headline number cannot reach the abstract under the plan's own schedule.
**Suggestion**: Review the load-bearing numbers before Nov 7, or add an abstract-revision step before the paper deadline.
**Severity**: Minor
**Confidence**: 4 — read directly from the schedule

### W13: No main-paper outline or page budget (D5)
**Problem**: The plan lists what goes to the supplement (E17) but not what fills the main paper: four datasets, two backbones, three baseline tiers, both directions, swap success and the other-aspect rate, the K6 and K7 ablations, the C3 analysis and a GeneCIS table. There is no page allocation and no list of figures or tables, and writing starts Oct 26.
**Evidence Anchor**: absence: plan body — expected a main-paper outline with a page budget and the planned figures and tables; checked §4, §5.1, §10, §11 E15 and E17, §12, Appendix B
**Why it matters**: The claims-to-evidence discipline of §4 is the plan's strength; without an outline, which claims reach the eight-page paper is decided under deadline pressure.
**Suggestion**: Add a one-page outline with page counts per section and the two or three main tables and figures, each tied to a K-claim with its baseline beside it.
**Severity**: Minor
**Confidence**: 4 — exposition and venue conventions are core to my remit

### W14: Vocabulary and project history will crowd the paper (D5)
**Problem**: The glossary carries twenty-five entries, several from project history (naive rule versus agreement rule; R0, R3, C0 and SE; condition episodes versus pseudo-aspect episodes), and §8 keeps three internal recipes as Tier-1 baselines.
**Evidence Anchor**: text: §8 Tier-1 row for C0, SE and R3 "our earlier factor recipes"
**Why it matters**: Each internal recipe in a final table needs its own paragraph in a paper whose readers never saw it.
**Suggestion**: Fold the internal recipes into one ablation (E11 already compares no episodes with value episodes), define only the terms the main paper uses, and move the history to the supplement.
**Severity**: Minor
**Confidence**: 3 — a judgement about exposition, not correctness

### W15: Dates blur the order of the evidence (D5)
**Problem**: §2.4 reports what changed on 2026-10-02 yet cites reports dated Oct 20 to 25 and a GeneCIS feasibility row marked Oct 1. The appendices say the work was done on Oct 2, the genre coverage folder is dated 20261026 while §5.3 says resolved on Oct 2, and the six investigations "all on selection rows" include two literature reviews that read no rows.
**Evidence Anchor**: text: §2.4 opening "Six investigations, all on selection rows (held rows untouched), changed the plan."
**Why it matters**: The paper's disclosure that ArtELingo's held rows shaped earlier design depends on a trustworthy order of reads; sequence labels that look like dates undercut it.
**Suggestion**: Use real dates or explicit sequence numbers throughout, and log every held-row read with its real date in the ledger.
**Severity**: Minor
**Confidence**: 4 — read directly from the document

### W16: "Ceiling" names a drifting diagnostic (D5)
**Problem**: The plan treats the label-probe value as a ceiling (§5.2, the strong-GO bar of "a third of the way to the ceiling", R-ceiling), while its own appendix says it is not a bound, the backbone check shows it drifting by up to 0.25 points between reruns, and it differs by backbone (about 23.2 for CLIP and 25.5 for Qwen, as means of the emotion and style values).
**Evidence Anchor**: text: aspect-episode spike appendix, caveats "It is a diagnostic, not a bound in the strict sense; a better probe could score higher."
**Why it matters**: Readers take "ceiling" as an upper bound, so the strong-GO phrasing will read as a firmer statement than the evidence supports.
**Suggestion**: Call it a label-probe reference, report it per backbone with converged probes (as the backbone check recommends), and state the strong-GO bar as an absolute gain over backbone only.
**Severity**: Minor
**Confidence**: 4 — terminology and claim framing are core to my remit

### Questions for Authors

1. Which condition-attributable statistic (for example R@1 minus the other-aspect rate) will the GO and K2 use, and what does CLIP only score on it?
2. If Qwen with the aspect in its instruction beats examples on emotion as well as style, what does the paper claim, and which users hold demonstration pairs but cannot name the aspect?
3. Will the GeneCIS text protocol leave the stretch list, or will K5 be removed?
4. Under NO-GO, which venue does the fallback target, and what is its claim table?

### Minor Issues

- Internal reconciliation notes (for example the controller's 46.43 against the recomputed 46.42 in the backbone check) and build-script paths in figure captions should not reach the paper.
- The aspect spike defines two swap-success variants (pairwise and strict); the paper should name the one §5.1 adopts wherever swap numbers appear.
