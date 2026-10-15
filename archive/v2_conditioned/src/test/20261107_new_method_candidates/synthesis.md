# Literature synthesis of the six new-method candidates (N1 to N6) for CoSiR v2

Date: 2026-10-04. Phase 3 synthesis (ARS synthesis agent). We read the controller's draft (`candidates_draft.md`), the six
three-way scans (`scan_N1.md` to `scan_N6.md`) and, to cross-check the draft's numbers, the project reports E1
(`docs/reports/auto/v2/2026-10-30_aspect_baselines.md`), E3 (`2026-11-01_aspect_factor_gonogo.md`), the method-repair
diagnostics A′ (`2026-11-05_method_repair_diagnostics.md`), the 8B probe log
(`src/test/20261106_mllm_probe_8b/20261106_mllm_probe_8b_log.md`), the aspect-episode spike
(`2026-10-23_aspect_episode_spike.md`), the earlier literature review (`docs/reports/literature/2026-10-21_cvpr_literature_review.md`), the
episode builder `src/eval/aspect_episodes.py` and the episode-seed ledger. We ran no new search, trained nothing and
scored nothing. Retrieved and scanned text was treated as data.

**Status of claims.** Every rationale in the draft stays a hypothesis here and is marked *(draft hypothesis)*.
Statements we derived from reported numbers are marked *reader-derived* and are untested. Literature claims are only as
strong as each scan's read scope, which for most entries is an abstract seen through a summarising fetch tool (§3.2).
Citations carry the verified id given in the scan; ids a scan marked unverified or index-only are flagged where they
appear and listed in Appendix A.

<!-- claim_intent_manifest
{"manifest_version":"1.0","manifest_id":"M-2026-10-04T12:00:00Z-a7c3","emitted_by":"synthesis_agent","emitted_at":"2026-10-04T12:00:00Z",
"claims":[
{"claim_id":"C-001","claim_text":"Within each scan's stated search bound, no scan found published evidence for the regime of about four value-disjoint cross-item image-caption pairs.","intended_evidence_kind":"empirical","planned_refs":[]},
{"claim_id":"C-002","claim_text":"Every candidate partially exists: each ingredient is published and the exact combination was not found within the bound.","intended_evidence_kind":"empirical","planned_refs":["koestinger_cvpr_2012","1612.02534","2305.16304","2312.08924","2510.04564","1812.03664","2011.14663","2312.02974","1911.12667","1603.07810"]},
{"claim_id":"C-003","claim_text":"A FEAT-style set-to-set adapter trained on pseudo tasks improved pseudo-task but not real-task accuracy (Ye, Han and Zhan), which bears against N4 and qualifies E3's pseudo-partition training.","intended_evidence_kind":"empirical","planned_refs":["2011.14663","1812.03664","1810.02334"],"negative_constraints":[{"constraint_id":"NC-C003-1","rule":"Do not attribute the pseudo versus real result to the FEAT paper itself."}]},
{"claim_id":"C-004","claim_text":"The privileged-names gain of +1.51 bounds CLIP label prompting, not an instruction-following embedder, because the spike itself reports zero-shot emotion accuracy at the majority-class level.","intended_evidence_kind":"empirical","planned_refs":["2601.04720"]},
{"claim_id":"C-005","claim_text":"CVS, the per-query weighting a reviewer would cite against N1, already ran in E1 as wang and matched A3's fused gain.","intended_evidence_kind":"empirical","planned_refs":["1612.02534"]},
{"claim_id":"C-006","claim_text":"Because R@1 equals half the either rate plus half the gain, a conditioned term with a term-only either rate near 21 to 23 needs a gain near 10 to 12 points to beat the uniform control.","intended_evidence_kind":"theoretical","planned_refs":[]},
{"claim_id":"C-007","claim_text":"VisDiff appears in two scans (N3, N5) and CSN in three (N1, N3, N6), making N3 and N5 one competitor family and N1, N3, N6 another.","intended_evidence_kind":"empirical","planned_refs":["2312.02974","1603.07810","1908.08589"]},
{"claim_id":"C-008","claim_text":"Recommended order: a label-head selection diagnostic with N2 and N1 first, then N6, then the privileged CRL and Qwen3-VL-Embedding references that gate N3 and N5, with N4 dropped for the November 10 window.","intended_evidence_kind":"normative","planned_refs":["2510.04564","2601.04720","2011.14663"]},
{"claim_id":"C-009","claim_text":"The draft's N4 rationale cites the aspect loss level as evidence of a weak rule, while A′ found the loss level does not track the term's selection signal.","intended_evidence_kind":"empirical","planned_refs":[]},
{"claim_id":"C-010","claim_text":"A pre-registration must name one configuration for the single seed-45 test, exclude label-trained checkpoints, and compare against the strongest condition-free control on the same codes.","intended_evidence_kind":"normative","planned_refs":[]}],
"manifest_negative_constraints":[
{"constraint_id":"MNC-1","rule":"No draft hypothesis is restated as a finding."},
{"constraint_id":"MNC-2","rule":"No claim of exhaustive absence; every absence is bounded by a scan's search."},
{"constraint_id":"MNC-3","rule":"No paper is cited that a scan did not verify, except in the flagged list."},
{"constraint_id":"MNC-4","rule":"No claim that an audit or external review was run."}]}
-->

## Summary

- **Novelty.** All six scans returned *partially exists*: every ingredient is published and no scan found the exact
  combination within its search bound. For each candidate the novelty would rest on the task itself (about four
  value-disjoint, cross-item image-caption pairs, both retrieval directions), not on the mechanism.
- **Regime.** No scan found published evidence for that regime. Every published example-conditioned method we were
  shown uses examples that share the query's value or relation, or names the condition in text (§3.1).
- **Strongest evidence against a candidate.** Ye, Han and Zhan (arXiv 2011.14663)<!--ref:2011.14663--><!--anchor:section:V-B-->
  found that a FEAT-style set-to-set adapter improved pseudo-task and not real-task accuracy, which bears directly on N4.
  For N1, the per-query weighting a reviewer would cite (CVS, arXiv 1612.02534)<!--ref:1612.02534--><!--anchor:section:Table1-->
  already ran in E1 as "wang" and matched A3's fused gain.
- **Our own bar.** Since R@1 = (either + gain) / 2, every conditioned term measured so far (term-only gains 0.97 to
  1.85, either rates 21.0 to 22.9) is an order of magnitude short of what beating the condition-free control needs,
  unless the method keeps that control's either rate of about 33 (§4.1).
- **Revised order.** First a label-head selection diagnostic (D0) together with N2 and N1 on seed-42 episodes (minutes of
  CPU); then N6 if D0 shows the condition can be read from the pairs; the privileged CRL (Liu et al., 2025; arXiv
  2510.04564)<!--ref:2510.04564--><!--anchor:section:abstract--> and Qwen3-VL-Embedding (Li et al., 2026; arXiv
  2601.04720)<!--ref:2601.04720--><!--anchor:section:abstract--> references next, which gate N3 and N5 and serve branch 3
  anyway; N4 dropped for the November 10 window (§4.2).

**Terms used throughout.** *R@1*: share of rankings whose target ranks strictly first. *Other-aspect rate*: share where
the other aspect's candidate ranks first. *Condition gain*: R@1 minus the other-aspect rate (0 for any condition-free
scorer). *Either rate*: R@1 plus the other-aspect rate, so R@1 = (either + gain) / 2. *Term-only*: the
agreement-weighted factor term scored alone. *Nested uniform control*: z(cos) + σ·z(uniform factor term), the strongest
condition-free score on the factor codes (A′). Evidence grades for literature: **S** sections read through a summarising
fetch tool, **A** abstract only, **I** index page or search listing only (unverified). Project evidence: **P-pre**
pre-registered, **P-post** post-hoc and descriptive.

## 1. Cross-candidate table

Seeds: 42 is the development draw (reused many times), 43 is spent (E3's GO test), 44 and 46 were the MLLM probes, 45 is
reserved for one GO test. Cosine R@1 is 12.96 on seed 42 and 13.53 on seed 43.

| Candidate | Novelty verdict and search bound | Strongest prior work a reviewer would cite (verified id) | Baselines this implies | Published evidence for our regime (≈4 pairs, value-disjoint, cross-modal) | Our own evidence, for or against | Cost to a first development test |
|---|---|---|---|---|---|---|
| **N1** centered cross-modal covariance rule | Partially exists; the exact combination (support minus contrast cross-modal covariance on a learned shared basis) not found. Bound: 7 WebSearch queries; 6 pages fetched (KISSME PDF, RCA JMLR page, CSN arXiv, CVS ar5iv, CRML arXiv, Rasiwasia PDF unparsed); no Scholar, Semantic Scholar, kernel-alignment or PLS search | KISSME, Koestinger et al. (2012), TU Graz PDF<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16-->; then CVS, Wang et al. (2016), arXiv 1612.02534<!--ref:1612.02534--><!--anchor:section:Table1-->; CCA, Rasiwasia et al. (2010)<!--ref:rasiwasia-etal-2010-acm--><!--anchor:section:abstract--> (PDF located, not parsed; grade A from search) | Diagonal KISSME (within-pair difference variances) from the same 4+4 pairs on the same codes; N1's rule on raw CLIP or E1's PCA-32 basis (K7-type comparator); CVS per-query weights (E1 "wang", already run); E1's per-episode pair probe (already run); the uncentered A3 term; cosine, RCA, N1's own centered uniform control and A3's nested uniform control | None for the full regime. Few examples: CVS rises with k = 1, 3, 5 (MAP 0.440, 0.519, 0.557 with fc7, as returned by the tool), but image only, gradient-fitted, positives share the query's value. Value-disjoint: none. Cross-modal: CCA, offline, many pairs | Against the KISSME family on raw features: E1 KISSME gain −0.16 [−0.34, 0.01] (s42), −0.09 [−0.27, 0.10] (s43) (P-pre). The closest relative already works at A3's level: "wang" gain 0.36 [0.06, 0.66] (s42), 0.30 [0.06, 0.55] (s43); A3 minus wang on gain −0.04 [−0.40, 0.34] (s43). The term N1 replaces: A3 term-only gain 0.99 at either 21.04 vs cosine 25.92 (s42, P-post). With 4 pairs a per-factor covariance has 3 degrees of freedom (scan, reader-inferred) | Minutes of CPU; training-free; existing checkpoints (label-trained L3, LT only as diagnostics, §4.5) |
| **N2** find-then-select cascade | Partially exists: the architecture is standard in composed retrieval; the instantiation (aspect read from 4+4 pairs, displacement motivation) not found. Bound: 6 queries, abstract pages only; no Scholar, Semantic Scholar, ACM, IEEE, OpenReview or 2026 proceedings | Liu et al. (2023), arXiv 2305.16304<!--ref:2305.16304--><!--anchor:section:abstract-->; Sun et al. (2023), arXiv 2312.08924<!--ref:2312.08924--><!--anchor:section:abstract--> (training-free form); Baumgartner et al. (2022), arXiv 2210.10695<!--ref:2210.10695--><!--anchor:section:abstract--> (few-example feedback reranking) | Tuned convex fusion of the same terms (A′'s nested score is one); the first stage alone (k = 1), which is N2's own condition-free control; the share of rankings with both aspect candidates in the top k; cosine, RCA | None for a weak conditioning signal: no abstract compares cascade and fusion when the condition term is weak or reports first-stage survival as a ceiling. Only verified fusion result: tuned convex combination is strong in hybrid text retrieval, fusion-function analysis (2022), arXiv 2210.11934<!--ref:2210.11934--><!--anchor:section:abstract--> | For the target: A′ located the failure in composition, not in absent selection. A3's nested profile reached gains near 1 point (1.04 at (λ_u, λ_a) = (0, 2) and (4, 8)), but every cell with gain above 0.5 had R@1 ≤ 14.26 against the control's 16.55; 14 of 16 tuning halves set λ_a = 0 (P-pre). Against: the term-only either rate (21.04) is below the cosine's (25.92), so inside a short list the term can also put a negative above an aspect candidate | Minutes of CPU; the both-in-top-k diagnostic first |
| **N3** grouped concept basis | Partially exists; the full combination (pre-grouped vocabulary, group chosen by example agreement, value-disjoint cross-modal test) not found. Bound: about ten query themes (SpLiCE, LaBo, label-free CBM, CSN, condition-aware vocabularies, CLAY, CRL, VisDiff, CLIP on art style and emotion); arXiv and CVPR pages; no Scholar, Semantic Scholar, OpenReview, ACL or unindexed 2026 proceedings | CRL, Liu et al. (2025), arXiv 2510.04564<!--ref:2510.04564--><!--anchor:section:abstract-->; SpLiCE, Bhalla et al. (2024), arXiv 2402.10376<!--ref:2402.10376--><!--anchor:section:abstract-->; SCE-Net, Tan et al. (2019), arXiv 1908.08589<!--ref:1908.08589--><!--anchor:section:abstract-->; VisDiff (2024), arXiv 2312.02974<!--ref:2312.02974--><!--anchor:section:abstract--> | CRL with the true aspect name (privileged reference); SpLiCE flat codes with the agreement rule (K7 comparator); the grouped codes with uniform group weights (own condition-free control); cosine, RCA | None for example-selected concept groups. Text-named gains on concrete attributes: CRL and CLAY, Lim et al. (2026), arXiv 2604.11539<!--ref:2604.11539--><!--anchor:section:abstract-->. Against abstract aspects (all grade A): CLIP hallucinates concepts, Kazmierczak et al. (2025), arXiv 2510.07115<!--ref:2510.07115--><!--anchor:section:abstract-->; CLIP's zero-shot emotion on abstract art is modest, Widhoelzl and Takmaz (2024), arXiv 2405.06319<!--ref:2405.06319--><!--anchor:section:abstract-->; criterion projections leak, SP-CRL, Wang et al. (2026), arXiv 2602.05464<!--ref:2602.05464--><!--anchor:section:abstract--> | Against on emotion: the spike's CLIP emotion prompts on captions reached 31.3% against a 31.8% majority class; style prompts on images 27.5% against 16.1%; privileged label names +1.51 [0.94, 2.09], on emotion only. For the premise: aspect values already spread over shared factors (value spread 0.41 to 0.65, E3 §6.2), which a grouped basis keeps by construction | Hours: an LLM-written attribute bank and grouping frozen before any episode is scored, concept coding of the selection rows, then CPU scoring |
| **N4** amortized, meta-trained conditioner | Partially exists (set-conditioned embedding, FiLM conditioning, k-means pseudo-episodes and their combination are all published); the task-level novelty is the task's own, so N4 adds little. Bound: about 12 queries (standard and extended), 16 arXiv abstract pages, ar5iv summaries for six; no Scholar, Semantic Scholar, OpenReview, CVF or ACL listings | FEAT, Ye et al. (2020), arXiv 1812.03664<!--ref:1812.03664--><!--anchor:section:4.1-->; CTM, Li et al. (2019), arXiv 1905.11116<!--ref:1905.11116--><!--anchor:section:3.2.2-->; CACTUs, Hsu et al. (2019), arXiv 1810.02334<!--ref:1810.02334--><!--anchor:section:2.3-->; BGE-EN-ICL, Li et al. (2024), arXiv 2409.15700<!--ref:2409.15700--><!--anchor:section:3.1-->. Against soundness: Ye, Han and Zhan (2020), arXiv 2011.14663<!--ref:2011.14663--><!--anchor:section:V-B--> | The same model with the conditioner switched off (residual, identity-initialised), which is its condition-free control; the fixed agreement rule on the same backbone; a FEAT-style or CTM-style set encoder on the same banks; cosine, RCA | Against: the set-to-set adapter improved pseudo-task and not real-task accuracy (Ye et al., Table IX, §V-B, grade S)<!--ref:2011.14663--><!--anchor:section:V-B-->; CACTUs reached 73.36 against an oracle's 96.29 (Omniglot 20-way 5-shot) and flags a task-distribution mismatch (§5)<!--ref:1810.02334--><!--anchor:section:5-->. Transfer to unseen tasks only in text with shared-value demonstrations: BGE-EN-ICL +1.43 QA<!--ref:2409.15700--><!--anchor:section:3.1-->. No image-caption pair conditioner | Against: K8 failed (H1 genre gain +0.10 [−0.21, 0.40], P-pre); pseudo-aspect loss stayed 1.0% to 4.4% below its constant-score value; matched-granularity pseudo-partitions lowered the term (MK3 minus A3 −0.88 [−1.44, −0.34], P-pre). Labels helped (L3 minus A3 +0.86 [0.28, 1.42]) but are forbidden for the method (spec C2) | About one GPU hour per bank per run (draft estimate), plus architecture design; leaves the training-free framing |
| **N5** infer the name, then embed with it | Partially exists; not found as one pipeline. Bound: about 14 queries; arXiv abstract pages and the VisDiff v2 HTML page; GME, E5-V, VLM2Vec, MM-Embed not opened; no OpenReview or Semantic Scholar | VisDiff (2024), arXiv 2312.02974<!--ref:2312.02974--><!--anchor:section:results-->; CIReVL, Karthik et al. (2024), arXiv 2310.09291<!--ref:2310.09291--><!--anchor:section:abstract-->; TTE (2025), arXiv 2510.05014<!--ref:2510.05014--><!--anchor:section:abstract-->. The embedder N5 would call: Qwen3-VL-Embedding, Li et al. (2026), arXiv 2601.04720<!--ref:2601.04720--><!--anchor:section:abstract--> | Qwen3-VL-Embedding with the true aspect in the instruction (privileged ceiling; already a K3 baseline in E3 §11); the same embedder with its default instruction (own condition-free control); CRL with the true name; VisDiff-style generate-and-verify naming; cosine, RCA | Naming works for describable differences: VisDiff Acc@1 88, 75, 61% on Easy, Medium, Hard, and image-based LLaVA proposing 28% on Hard; abstract concepts are hard (author-acknowledged, via summary)<!--ref:2312.02974--><!--anchor:section:limitations-->. MLLMs are weak on classical Bongard problems (arXiv 2411.01173, grade A)<!--ref:2411.01173--><!--anchor:section:abstract-->. Attribute prompts help on concrete attributes (+15% R@5), Li et al. (2025), arXiv 2505.15877<!--ref:2505.15877--><!--anchor:section:abstract-->. No test of abstract-aspect instructions | 8B in context (s46): R@1 +1.07 [0.17, 1.93], gain +0.21 [−0.51, 0.94], either +1.93 [0.25, 3.51] (P-pre). 2B (s44): gain −0.53 [−1.68, 0.66]. Privileged CLIP names +1.51 [0.94, 2.09], which bounds CLIP label prompting only (§2.2, row 4) | GPU hours: embedder passes over the selection rows under three aspect instructions and the default; 8B naming on a few hundred conditions (the 8B ranking prompts took 3.6 s per four-prompt episode on node404 and 5.3 s locally, peak 18.9 GB) |
| **N6** cross-modal partition heads | Partially exists; the combination (per-partition cross-modal heads, head chosen by within-pair agreement) not found. Bound: 6 queries, abstract pages; no Scholar, Semantic Scholar, ACL, OpenReview or mixture-of-experts gating query | XDC, Alwassel et al. (2020), arXiv 1911.12667<!--ref:1911.12667--><!--anchor:section:abstract-->; CSN, Veit et al. (2017), arXiv 1603.07810<!--ref:1603.07810--><!--anchor:section:abstract-->; MFCVAE (2021), NeurIPS poster page<!--ref:neurips2021-poster-26795--><!--anchor:section:abstract--> (no arXiv id in the scan) | D0: the same heads trained on true labels with the same selection rule (diagnostic ceiling, §4.2); the label-probe reference told the aspect, recomputed on the E1 episode stack; N6 with uniform head weights (own condition-free control); the factor agreement rule on the same partitions (A3); cosine, RCA | None for affect or style aspects or for few-pair head choice. Cluster-derived supervision transfers to action recognition and zero-shot video retrieval: XDC and MCN (2021), arXiv 2104.12671<!--ref:2104.12671--><!--anchor:section:abstract--> | For: the label-probe reference (true labels, aspect told) reached 23.09 against CLIP's 11.13 on the spike's episodes. Against: partition-to-aspect AMI 0.16 to 0.42 (E2 and MK partitions); the image partition carries genre (0.397) and style (0.318) at once; K8 failed | Minutes of CPU: linear heads on frozen CLIP over scorer-train rows; the E2 partitions exist |

## 2. Convergences and contradictions across the scans

### 2.1 Convergences: where the scans point to the same competitor

1. **Condition-selected subspaces.** CSN (arXiv 1603.07810)<!--ref:1603.07810--><!--anchor:section:abstract--> appears in
   three scans (N1, N3, N6) and SCE-Net (arXiv 1908.08589)<!--ref:1908.08589--><!--anchor:section:abstract--> in two (N3,
   N4). A reviewer will read N1's factor weights, N3's group choice and N6's head choice as one idea, a condition that
   selects a subspace, given by an id in CSN and inferred from the compared items in SCE-Net. What remains distinct is
   the source of the condition (support and contrast cross-item pairs) and the value-disjoint cross-modal test, which
   belong to the task rather than to any one candidate.
2. **Set-difference inference.** VisDiff (arXiv 2312.02974)<!--ref:2312.02974--><!--anchor:section:abstract--> appears in
   N3 and N5. N3's group selection and N5's naming are both "infer what separates two example sets"; VisDiff is the
   published version, with a contrast set and a CLIP verifier. N3 and N5 are therefore one competitor family that differs
   only in output (a group index or a phrase).
3. **Text-named conditions.** CLAY (arXiv 2604.11539)<!--ref:2604.11539--><!--anchor:section:abstract--> appears in N3, N4
   and N5, and GeneCIS (arXiv 2306.07969)<!--ref:2306.07969--><!--anchor:section:abstract--> in N2 and N5. Both put the
   question "why not name the aspect" that the K3 analysis must answer.
4. **Pair-statistic metrics.** KISSME<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16--> appears only in N1, but
   it converges with the project's own record: the 2026-10-24 novelty check expected the reviewer line "a few-shot
   cross-modal diagonal KISSME", and E1 ran KISSME (low-rank, with shrinkage) and RCA on raw features. CVS (arXiv
   1612.02534)<!--ref:1612.02534--><!--anchor:section:Table1--> converges the same way: N1 and N4 both cite it, and E1 ran
   it as "wang", where it had the best mean of R@1 and gain on seed 43 (6.85) with a gain interval above zero on both
   seeds.
5. **Naming baselines.** CRL (arXiv 2510.04564)<!--ref:2510.04564--><!--anchor:section:abstract--> appears only in N3
   and Qwen3-VL-Embedding (arXiv 2601.04720)<!--ref:2601.04720--><!--anchor:section:abstract--> only in N5, but both are
   already the K3 naming baselines of E3 §11 and baselines 4 and 5 of the 2026-10-21 review. They are needed under branch
   3 as well as for gating N3 and N5, so running them early is not wasted whichever branch the user picks.
6. **Pseudo-task transfer.** The N4 and N6 scans and our own runs converge on one reading: pseudo-partitions are weak
   proxies for real aspects. In the literature, CACTUs (arXiv 1810.02334)<!--ref:1810.02334--><!--anchor:section:4.2-->
   and Ye, Han and Zhan (arXiv 2011.14663)<!--ref:2011.14663--><!--anchor:section:V-B-->; in our data, the K8 failure,
   MK3's −0.88 against A3 and the LAB lift of +0.86 when true labels replaced the partitions.
7. **The regime gap.** All six scans report the same absence (§3.1).

### 2.2 Contradictions and their resolution

| # | Claim A | Claim B | Resolution |
|---|---|---|---|
| 1 | N4 *(draft hypothesis)*: a learned set encoder can learn what "agree within pairs, vary across pairs" looks like, trained on pseudo-aspect episodes | Ye, Han and Zhan (arXiv 2011.14663)<!--ref:2011.14663--><!--anchor:section:V-B-->: with pseudo-class tasks, a FEAT-style Transformer adapter did better on pseudo tasks and the pre-adapted embedding did better on real tasks (Table IX; grade S). FEAT itself (arXiv 1812.03664)<!--ref:1812.03664--><!--anchor:section:5.2.1--> reports gains with supervised episodes, so the negative result belongs to Ye et al., not to the FEAT paper | **Reconcilable as a prediction, unresolved as a test.** The literature predicts that N4's learned reader is the component most likely to fit pseudo partitions and not real aspects, the K8 concern. Training regime explains FEAT against Ye et al. (real labelled episodes against pseudo tasks). If N4 is ever run, the scan's design applies: keep a condition-free pathway and add the conditioner as an identity-initialised residual |
| 2 | E3 §6.4 (P-post): what A3 fit on pseudo-aspects carried over to labelled aspects (training-score gain on fresh pseudo episodes 0.95 [0.28, 1.62] against a labelled term-only gain of 0.97 [0.53, 1.41]); "the amount fit was the limit" | Ye et al.<!--ref:2011.14663--><!--anchor:section:V-B-->: a stronger fit to pseudo tasks can lower real-task accuracy; CACTUs<!--ref:1810.02334--><!--anchor:section:4.2-->: a large gap to the oracle (73.36 against 96.29) | **Conditional.** E3's carry-over held at a weak fit (loss 1.0% to 4.4% below constant). The literature says it need not hold once the pseudo fit gets stronger, which is what any further pseudo-partition training aims for. Our own MK3 (−0.88 [−1.44, −0.34]) and LAB (+0.86 [0.28, 1.42]) results say partition content limits the term independently of fit. E3's reading "the amount fit was the limit" should therefore be read as "a limit", not "the only limit" |
| 3 | N4 *(draft hypothesis)*: the fixed rule is a weak estimator, since "even label training barely moved its loss" | A′ §4.3 (P-post): the loss level did not track the term's selection signal; LT ended 0.4% *above* its constant-score value and had the largest fit (+1.34 [0.77, 1.90] over A3) | **Resolved against the draft's evidence, not against its conclusion.** The loss level is not evidence about the estimator. The measured evidence (term-only gain 1.85 with labels against 0.99 without) leaves the rule's weakness and the basis's weakness unseparated. D0 (§4.2) separates them |
| 4 | Scan N5 (reader-inferred): "a perfect naming step through a CLIP-class embedder is worth about +1.5", so N5's ceiling depends on the embedder | The spike (P-post) reported that zero-shot emotion prompts on captions were at the majority-class level (31.3% against 31.8%) and said the names gain "is not a measure of what names can give" | **Resolved: the +1.51 bounds CLIP label prompting, not naming.** N5's ceiling with an instruction-following embedder is unmeasured. The privileged run of Qwen3-VL-Embedding<!--ref:2601.04720--><!--anchor:section:abstract--> with the true aspect in its instruction measures it, and it is already a K3 baseline |
| 5 | N5 *(draft hypothesis)*: the 8B model's failure is selection inside a long prompt; naming is easier than 13-way scoring | VisDiff<!--ref:2312.02974--><!--anchor:section:limitations-->: abstract concepts are hard (author-acknowledged), image-based LLaVA proposing reached 28% on Hard; Bongard study (arXiv 2411.01173)<!--ref:2411.01173--><!--anchor:section:abstract-->: MLLMs are poor on classical Bongard problems (grade A) | **Unresolved.** The evidence makes "naming is easier" less likely for abstract aspects, and VisDiff's high accuracies came from a caption proposer with a strong LLM and one dominant difference, not a contrast set that shares a second aspect. The draft's naming-accuracy diagnostic decides it, but only after the privileged ceiling (row 4) shows naming would be worth having |
| 6 | N2 *(draft hypothesis)*: inside a short list that holds p_A and p_B, the term only has to choose between them | E3 and A′ (P-post): the term-only either rate (21.04 on seed 42) is below the cosine's (25.92), so the term also ranks negatives above aspect candidates. Convex fusion is strong in hybrid retrieval (arXiv 2210.11934)<!--ref:2210.11934--><!--anchor:section:abstract-->, and the cascade papers do not test weak condition signals (grade A) | **Reconcilable.** N2 confines displacement to the top k; it does not remove it. With k ≥ 2 a list holding one aspect candidate and one negative can lose the aspect candidate. The both-in-top-k share and the either-rate loss at each k, measured on seed 42, decide whether gain exceeds loss (§4.1) |
| 7 | N1 *(draft hypothesis)*: displacement comes from the uncentered term rewarding generally active candidates; centering fixes it | E3 §6.1 (P-post, untested reading): the weights concentrate on the few factors the pairs agree on and drop the shared similarity that finds aspect-sharing candidates | **Unresolved; two untested readings.** N1's centered score keeps a concentrated weight vector, so if E3's reading holds, N1's either rate stays below the cosine. N1's own kill criterion (either rate below cosine) tests this directly |
| 8 | Draft N1 risk: KISSME and RCA "failed on raw CLIP in E1", so N1's claim rests on the learned basis | E1 (P-pre): CVS ("wang")<!--ref:1612.02534--><!--anchor:section:Table1--> and the per-episode pair probe had gain intervals above zero on seed 42 (0.36 [0.06, 0.66]; 0.35 [0.04, 0.65]), and wang again on seed 43 (0.30 [0.06, 0.55]), level with A3's fused gain | **Reconcilable.** The KISSME family failed the GO bar, but the closest few-example per-dimension method already reached A3's fused gain on raw features. N1 must beat CVS and the pair probe, not only KISSME and RCA |
| 9 | N6 *(draft hypothesis)*: it "mirrors the label-probe reference (23.09)" with pseudo-labels in place of labels | The spike: the reference was trained on human labels and **told the aspect**, on a different episode set (emotion and style only, CLIP 11.13) | **Resolved as a decomposition.** 23.09 is a reference for N6 with true labels *and* an oracle condition. N6 replaces both. D0 removes only the oracle, so N6's gap to D0 measures the partition cost and D0's gap to the told reference measures the reader cost |
| 10 | N3 *(draft hypothesis)*: a grouped vocabulary has aspect blocks by construction, so value-disjoint conditions can transfer | The spike (emotion prompts at the majority-class level); Widhoelzl and Takmaz<!--ref:2405.06319--><!--anchor:section:abstract-->; Kazmierczak et al.<!--ref:2510.07115--><!--anchor:section:abstract-->; SP-CRL<!--ref:2602.05464--><!--anchor:section:abstract--> (all grade A) | **Conditional.** Blocks in vocabulary space do not imply value information in CLIP space. Emotion groups are predicted to be weak, style groups to carry some signal (27.5% against a 16.1% majority). Expect per-pair differences, and read N3 on emotion pairs as the hard case |

#### Cross-Paper Tension Inventory (#262)

```yaml
cross_paper_tensions:
  - pair_id: CP-001
    paper_a: "1812.03664"
    paper_b: "2011.14663"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "Does a Transformer set-to-set adapter improve the task-specific embedding on real tasks?"
    a_finding: "FEAT: the Transformer set function beat BiLSTM, DeepSets and GCN, and helped under a Clipart to Real World shift (30.89 vs ProtoNet 29.47)."
    a_evidence_pointer: "scan_N4.md, FEAT entry, WHAT (§5.2.1 Table 1, §5.3.1; sections via fetch summary)"
    b_finding: "With pseudo-class tasks, the adapted embedding did better on pseudo tasks and the pre-adapted one on real tasks."
    b_evidence_pointer: "scan_N4.md, Ye et al. entry, WHAT (Table IX, §V-B; sections via fetch summary)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 1"
    scholar_confirmation: "pending"
  - pair_id: CP-002
    paper_a: "1810.02334"
    paper_b: "2011.14663"
    candidate_basis: "shared RQ subtopic"
    overlap_topic: "Transfer from clustering-built pseudo tasks to real tasks"
    a_finding: "CACTUs beats embedding-only baselines but reaches 73.36 against an oracle's 96.29 (Omniglot 20-way 5-shot); task-distribution mismatch flagged."
    a_evidence_pointer: "scan_N4.md, CACTUs entry, WHAT and weaknesses (Tables 1 to 2, §4.2, §5)"
    b_finding: "A set-to-set adapter on pseudo tasks helped pseudo-task and not real-task accuracy."
    b_evidence_pointer: "scan_N4.md, Ye et al. entry, WHAT (Table IX, §V-B)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 2"
    scholar_confirmation: "pending"
  - pair_id: CP-003
    paper_a: "2312.02974"
    paper_b: "2411.01173"
    candidate_basis: "opposite finding direction"
    overlap_topic: "Can models infer the concept that separates or unites image sets?"
    a_finding: "VisDiff: caption proposer plus CLIP ranker Acc@1 88, 75, 61% by difficulty; image-based LLaVA proposing 28% on Hard; abstract concepts hard."
    a_evidence_pointer: "scan_N5.md, VisDiff entry, WHAT and weaknesses (results and limitations, v2 HTML, fetch summary)"
    b_finding: "Eight MLLMs find shared concepts better on real-world than synthetic problems and do poorly on classical Bongard problems."
    b_evidence_pointer: "scan_N5.md, supporting papers, Bongard entry (abstract only)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 5"
    scholar_confirmation: "pending"
  - pair_id: CP-004
    paper_a: "2510.04564"
    paper_b: "2602.05464"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "Does projecting CLIP embeddings onto an LLM-written criterion basis isolate the criterion?"
    a_finding: "CRL reports gains on customized classification, clustering and retrieval with a criterion-specific basis."
    a_evidence_pointer: "scan_N3.md, CRL entry, WHAT (abstract only)"
    b_finding: "SP-CRL reports that VLMs do not disentangle criteria, so the projection leaks semantics."
    b_evidence_pointer: "scan_N3.md, CRL entry, follow-up line (abstract only)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 10"
    scholar_confirmation: "pending"
  - pair_id: CP-005
    paper_a: "2604.11539"
    paper_b: "2405.06319"
    candidate_basis: "shared RQ subtopic"
    overlap_topic: "Does CLIP's space support affect-like or abstract conditions?"
    a_finding: "CLAY reports competitive training-free conditional retrieval from text conditions on frozen VLM embeddings."
    a_evidence_pointer: "scan_N3.md and scan_N5.md, CLAY entries, WHAT (abstract only)"
    b_finding: "CLIP's zero-shot emotion recognition on abstract art is above baseline but modest and misaligned with humans."
    b_evidence_pointer: "scan_N3.md, verdict, known failure modes (abstract only)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 10"
    scholar_confirmation: "pending"
  - pair_id: CP-006
    paper_a: "2312.08924"
    paper_b: "2210.11934"
    candidate_basis: "shared RQ subtopic"
    overlap_topic: "Cascade reranking versus one-stage score fusion"
    a_finding: "Training-free global retrieval followed by local concept reranking is above other training-free methods on four CIR benchmarks."
    a_evidence_pointer: "scan_N2.md, Sun et al. entry, WHAT (abstract only)"
    b_finding: "Convex combination beats Reciprocal Rank Fusion in and out of domain; cascades not tested."
    b_evidence_pointer: "scan_N2.md, fusion-function entry, WHAT (abstract only)"
    pair_assessment: "insufficient_overlap"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
  - pair_id: CP-007
    paper_a: "1612.02534"
    paper_b: "koestinger_cvpr_2012"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "Can a metric be estimated from very few similar and dissimilar examples?"
    a_finding: "CVS: per-dimension weights fitted at query time from k = 1, 3, 5 positives raise MAP 0.440, 0.519, 0.557 (fc7)."
    a_evidence_pointer: "scan_N1.md, CVS entry, WHAT (Table 1 as returned by the tool)"
    b_finding: "KISSME: closed form from similar minus dissimilar pair covariances, shown with thousands of pairs; small-n regime not found in the text."
    b_evidence_pointer: "scan_N1.md, KISSME entry, HOW and weaknesses (sections, fetch summary)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 8"
    scholar_confirmation: "pending"
  - pair_id: CP-008
    paper_a: "1603.07810"
    paper_b: "1908.08589"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "Condition-selected embedding subspaces"
    a_finding: "CSN: learned masks selected by a given notion id beat separately trained specialists."
    a_evidence_pointer: "scan_N1.md, CSN entry (abstract only)"
    b_finding: "SCE-Net: the condition is a latent inferred from the compared items and beats supervised conditional methods on three fashion datasets."
    b_evidence_pointer: "scan_N3.md, SCE-Net entry (abstract only)"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
  - pair_id: CP-009
    paper_a: "2409.15700"
    paper_b: "2011.14663"
    candidate_basis: "agent-noted cross-cluster"
    overlap_topic: "Does a learned in-context or set conditioner generalise to unseen tasks?"
    a_finding: "BGE-EN-ICL gains on tasks absent from training (+1.43 QA, +1.08 long-doc), text only, demonstrations share the relation."
    a_evidence_pointer: "scan_N4.md, BGE-EN-ICL entry, WHAT (Tables 2 to 3, fetch summary)"
    b_finding: "A set-to-set adapter trained on pseudo tasks did not help real tasks."
    b_evidence_pointer: "scan_N4.md, Ye et al. entry, WHAT (Table IX, §V-B)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 1"
    scholar_confirmation: "pending"
  - pair_id: CP-010
    paper_a: "2505.15877"
    paper_b: "2405.06319"
    candidate_basis: "agent-noted cross-cluster"
    overlap_topic: "Does naming the attribute help an embedder focus on it?"
    a_finding: "Prompting MLLM retrievers with the needed attribute gives 15% R@5 on COCO-Facet with author-written concrete prompts."
    a_evidence_pointer: "scan_N5.md, supporting papers, Promptable Embeddings entry (abstract only)"
    b_finding: "CLIP's zero-shot emotion recognition on abstract art is modest."
    b_evidence_pointer: "scan_N3.md, verdict, known failure modes (abstract only)"
    pair_assessment: "conditional_difference"
    resolution_status: "resolved_in_synthesis"
    resolution_pointer: "Synthesis Report > Contradictions & Resolutions (§2.2), row 4"
    scholar_confirmation: "pending"
  - pair_id: CP-011
    paper_a: "2601.04720"
    paper_b: "2510.05014"
    candidate_basis: "shared construct/outcome/measure"
    overlap_topic: "Which instruction-aware embedder leads MMEB-V2?"
    a_finding: "Qwen3-VL-Embedding 8B scores 77.8 on MMEB-V2, reported first as of the paper's date."
    a_evidence_pointer: "scan_N5.md, Qwen3-VL-Embedding entry, WHAT (abstract only)"
    b_finding: "TTE reports state of the art on MMEB-V2, above proprietary models."
    b_evidence_pointer: "scan_N5.md, TTE entry, WHAT (abstract only)"
    pair_assessment: "no_material_conflict"
    resolution_status: "not_applicable"
    scholar_confirmation: "pending"
```

**Coverage Note**: 50 papers in the scans' corpus (Appendix A); 11 candidate pairs considered (basis: shared RQ subtopic,
shared construct or measure, opposite finding direction, and agent-noted cross-cluster pairs). This is a **scoped
advisory scan, not complete pairwise contradiction detection**; cross-neighbourhood pairs not surfaced here may exist and
are not claimed absent. Pair classes not checked exhaustively: the self-supervised clustering papers among themselves
(SwAV, DeepCluster, SeLa), the abstract-level related works of scan N4 among themselves, and pairs across candidates with
no shared construct (for example XDC against the rerankers). Bibliographic coupling was not available and was not used to
exclude any pair. The scholar confirms each `resolution_pointer` and may flag further pairs. Tensions between the
draft's hypotheses and project evidence are in the table above (rows 3, 4, 6 to 9), not in the paper-pair inventory.

## 3. Gaps

### 3.1 What no scan found: evidence for our regime

No scan found a paper that reads a condition from about four example pairs whose values never include the query's, with
image and caption from different items, scored across modalities. The nearest published settings each drop one property:
CVS (arXiv 1612.02534)<!--ref:1612.02534--><!--anchor:section:4.3--> has few examples but shared values and one modality;
BGE-EN-ICL (arXiv 2409.15700)<!--ref:2409.15700--><!--anchor:section:4.3--> has unseen tasks but shared-relation
demonstrations and text only; CCA<!--ref:rasiwasia-etal-2010-acm--><!--anchor:section:abstract--> is cross-modal but
offline with many pairs; CRL<!--ref:2510.04564--><!--anchor:section:abstract--> and
CLAY<!--ref:2604.11539--><!--anchor:section:abstract--> name the condition in text; VisDiff
(arXiv 2312.02974)<!--ref:2312.02974--><!--anchor:section:results--> infers a difference from sets but outputs a phrase
and does not condition retrieval. The absence is bounded by each scan's search:

| Scan | Queries | Sources opened | Not searched |
|---|---|---|---|
| N1 | 7 WebSearch queries | KISSME PDF<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16-->, RCA JMLR page<!--ref:bar-hillel05a--><!--anchor:section:abstract-->, CSN arXiv<!--ref:1603.07810--><!--anchor:section:abstract--> (CVF 403), CVS ar5iv<!--ref:1612.02534--><!--anchor:section:4.3-->, CRML arXiv<!--ref:2211.07116--><!--anchor:section:abstract-->, Rasiwasia PDF<!--ref:rasiwasia-etal-2010-acm--><!--anchor:section:abstract--> (not parsed) | Google Scholar, Semantic Scholar, kernel alignment, PLS |
| N2 | 6 | arXiv and mlanthology abstract pages | Google Scholar, Semantic Scholar, ACM, IEEE, OpenReview, 2026 CVPR and ECCV in depth |
| N3 | about 10 themes | arXiv abstract pages, CVPR 2017 page (403) | Google Scholar, Semantic Scholar, OpenReview, ACL, unindexed ICCV and ECCV 2026 |
| N4 | about 12 (standard and extended) | 16 arXiv abstract pages; ar5iv summaries for 6 | Google Scholar, Semantic Scholar, OpenReview, ACL and CVF listings, anything after early October 2026 |
| N5 | about 14 | arXiv abstract pages; VisDiff v2 HTML<!--ref:2312.02974--><!--anchor:section:results--> | Google Scholar, ACL, OpenReview, Semantic Scholar; GME, E5-V, VLM2Vec, MM-Embed not opened |
| N6 | 6 | arXiv, CVF and NeurIPS abstract pages | Google Scholar, Semantic Scholar, ACL, OpenReview, mixture-of-experts gating |

Every absence claim in this synthesis therefore reads "to our knowledge, within these searches". The 2026-10-21 review
reached the same negative result for example-conditioned cross-modal similarity from a separate, also incomplete search.

### 3.2 Read-scope limits

Of the 50 sources, 9 were read at section level through a summarising fetch tool: KISSME (2012)<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16-->,
CVS (2016)<!--ref:1612.02534--><!--anchor:section:4.3-->, FEAT (2020)<!--ref:1812.03664--><!--anchor:section:5.2.1-->,
CTM (2019)<!--ref:1905.11116--><!--anchor:section:3.2.2-->, TADAM (2018)<!--ref:1805.10123--><!--anchor:section:2.4-->,
CACTUs (2019)<!--ref:1810.02334--><!--anchor:section:4.2-->, Ye et al. (2020)<!--ref:2011.14663--><!--anchor:section:V-B-->,
BGE-EN-ICL (2024)<!--ref:2409.15700--><!--anchor:section:4.3--> and VisDiff (2024)<!--ref:2312.02974--><!--anchor:section:limitations-->.
36 were read at abstract level. 4 were seen only through an index page or search listing: Shen et al. (2021)<!--ref:shen2021neurips-reranking--><!--anchor:none:-->,
label-free CBM (2023)<!--ref:label-free-cbm--><!--anchor:none:-->, SeLa (2020)<!--ref:1911.05371--><!--anchor:none:-->
and Dual-disentangled Deep Multiple Clustering (2024)<!--ref:2402.05310--><!--anchor:none:-->. One, arXiv 2605.18974
(2026)<!--ref:2605.18974--><!--anchor:section:abstract-->, was opened without yielding a usable figure. No PDF was read
directly in any scan this session; the KISSME equations were checked earlier by the 2026-10-28 citation check.
Consequences: method weaknesses are "not assessed" for most entries; every number quoted above from a paper (CVS MAP,
CACTUs 73.36, VisDiff 88, 75, 61%, Ye et al. Table IX) is second-hand and must be checked against the PDF before it
enters a paper; the CtrlBench numbers in scan N5, attached to ABC (2025)<!--ref:2503.00329--><!--anchor:section:abstract-->,
come from a search snippet and are not used here.

### 3.3 Gaps in our own evidence

1. **The label-probe reference is on a different episode set.** 23.09 against CLIP's 11.13 was measured on the spike's
   emotion and style episodes, with no genre probe and with the aspect told. The E1 episode stack (three aspect pairs,
   cosine 12.96 on seed 42) has no such reference, so the headroom on the episodes every candidate will be scored on is
   unmeasured.
2. **Reader against representation is unseparated.** The draft's two readings (the condition must be read from
   within-pair agreement; the representation must hold aspect blocks) are both untested, and no run separates them.
3. **Aspect blocks were not measured** (E3 §6.2): value spread shows shared factors, not aspect-specific ones.
4. **N2's both-in-top-k share and the privileged instruction-embedder ceiling are unmeasured.**
5. **One model seed per run**, and seed 42 has been scored by E1, E3's pick and post-hoc profile, and A′'s pilot and
   transfer; development numbers on it carry selection effects (E3's picked gain fell from 0.52 to 0.26 on fresh episodes).

### Evidence convergence map

```
Strong:      [          ] none at the level of our regime
Moderate:    [======    ] Pseudo-task transfer is weak (CACTUs, Ye et al. at section level; E3, K8, MK3 in our data)
Moderate:    [======    ] Pair statistics need many pairs; few-example weights work with shared values (KISSME, CVS at section level; RCA, CCA abstract; E1)
Moderate:    [=====     ] Examples define conditions when they share the query's value or relation (CVS, FEAT, CTM, TADAM, BGE-EN-ICL at section level; 8 more abstract)
Emerging:    [====      ] Condition-selected subspaces or bases exist with given or text conditions (10 sources, abstract level)
Emerging:    [===       ] Language as the condition interface (9 sources; VisDiff at section level)
Emerging:    [===       ] Abstract aspects are hard for CLIP-class concept and naming pipelines (5 sources, abstract level; spike zero-shot numbers)
Emerging:    [==        ] Cascade versus fusion under a weak condition signal (6 sources, abstract level; no direct comparison)
Gap:         [          ] Few-pair, value-disjoint, cross-modal regime (0 sources in any scan)
```

## 4. Revised recommendation

### 4.1 The arithmetic every candidate faces (theoretical integration)

Two quantities and two components organise the evidence. The quantities: because R@1 = (either + gain) / 2, a scorer
raises R@1 by finding an aspect-sharing candidate more often (either rate) or by choosing the conditioned one more often
(gain). The components: a *representation* (the factor codes, concept groups, partition heads, an embedder) and a
*reader* that turns the 4+4 pairs into a condition (the agreement rule, a learned encoder, a generated name).

What we measured places every factor-basis variant on the wrong side of the arithmetic. On seed 42, A3's nested uniform
control reached either 33.09 and R@1 16.55; A3's term alone reached either 21.04 and gain 0.99; even the label-trained L3
term reached only either 22.93 and gain 1.85. Reader-derived: at either 21 to 23, beating 16.55 needs a gain of about 10
to 12 points; at the control's own either rate, any reliable positive gain would do. E3 §11 made the same calculation on
seed 43 (gains of 6.2 to 12.1 needed). So a candidate can pass only by (a) keeping the condition-free score's either rate
while adding selection, or (b) producing gains an order of magnitude above anything measured. N2 aims at (a). N1 aims at
the gain and, through its kill criterion, at the either rate. N3 and N6 change the representation, so their own uniform
controls set new either rates; N4 and N5 change the reader.

The candidates also map onto one decomposition the project has not yet measured. The label-probe reference uses a
true-label representation with an *oracle* reader (told the aspect). A3 and L3 use a factor representation with the fixed
agreement reader. Nothing uses a true-label representation with the *fixed* reader. That missing cell decides which
candidates can work: if the fixed reader recovers most of the oracle's gain when the representation has clean aspect
blocks, the bottleneck is the representation (favouring N6, then N3); if it does not, the bottleneck is reading the
condition from four pairs (sinking N1, N3 and N6 together and leaving N4 and N5, the two with the strongest prior
evidence against them).

### 4.2 Order

**Step 1, day 1, minutes of CPU, seed-42 development episodes. Run these together.**

- **D0, label-head selection diagnostic (not a candidate).** Fit the spike's multinomial logistic probes (human labels,
  60,000 scorer-train rows, one per modality and aspect, adding genre). On seed-42 episodes score (i) the aspect-told
  reference (posterior dot product on the condition's head), which gives the E1 stack the headroom number it lacks, and
  (ii) the inferred-aspect variant, which picks the head by N6's rule (within-pair posterior agreement over supports minus
  contrasts) and scores as in (i). Like the LAB runs it reads labels on training rows, so it is a diagnostic only and is
  excluded from every candidate set. Reason: it is the cheapest test of the draft's two readings (§3.3, item 2) and it
  orders everything after it. The spike's ceiling script ran in about two minutes.
- **N2.** First the diagnostic the draft already names: on A3's nested uniform control, the share of rankings with both
  aspect candidates in the top k, k ∈ {2, 3, 5}. Then the cascade with T_a from A3, reporting gain and the either-rate
  loss at each k. Reason: it is the only candidate built to keep the condition-free either rate, which is the binding
  quantity in §4.1. Its novelty is low (retrieve-then-rerank, Liu et al., arXiv 2305.16304)<!--ref:2305.16304--><!--anchor:section:abstract-->,
  so frame it as a cascade-versus-fusion control.
- **N1.** Term-only gain and either rate of the centered rule on A3, C0 and SE (and on L3, LT as diagnostics), against the
  uncentered term, diagonal KISSME (2012)<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16--> on the same codes,
  N1 on raw CLIP or PCA-32, CVS (2016)<!--ref:1612.02534--><!--anchor:section:4.3--> ("wang") and the pair probe; then N1
  as T_a inside N2. Reason: it tests a specific failure reading in minutes, and its kill criterion (either rate below
  cosine) also tests E3's competing reading (§2.2, row 7).

**Step 2, day 1 to 2, minutes, conditional on D0.** **N6** with heads trained on the frozen E2 partitions (AIC), only if
D0's inferred-aspect variant clears the cosine on gain by a margin stated before D0 is scored. Expected weak spot
(reader-derived, descriptive): the image partition carries genre and style at once (AMI 0.397 and 0.318), so style ×
genre episodes should show the least selection.

**Step 3, GPU hours, in parallel with Steps 1 and 2 under the GPU lock.** The privileged references: CRL
(arXiv 2510.04564)<!--ref:2510.04564--><!--anchor:section:abstract--> and Qwen3-VL-Embedding
(arXiv 2601.04720)<!--ref:2601.04720--><!--anchor:section:abstract--> with the true aspect name, each next to its
condition-free version. Reason: they are K3 baselines under branch 3 anyway (§2.1, item 5), and they gate N3 and N5.

**Step 4, conditional.**

- **N3**, demoted below N6: run only if N6 fails for a reason a grouped vocabulary could fix (for example the style ×
  genre merge in the image partition) and privileged CRL (2025)<!--ref:2510.04564--><!--anchor:section:abstract--> clears
  the cosine on gain. Reason: the spike's zero-shot emotion
  accuracy at the majority-class level predicts weak emotion groups (§2.2, row 10), and N3 costs hours against N6's
  minutes.
- **N5**, reframed as a K3 analysis: run the 8B naming step only if the privileged Qwen3-VL-Embedding
  (2026)<!--ref:2601.04720--><!--anchor:section:abstract--> run clears its default-instruction control and the cosine on
  gain. Reason: without that, perfect names would not help, and the 8B in-context gain of +0.21 [−0.51, 0.94] plus
  VisDiff's (2024)<!--ref:2312.02974--><!--anchor:section:limitations--> note on abstract concepts make naming unlikely to
  be the easy step.

### 4.3 Drops and demotions

- **N4: drop for the November 10 window.** It carries the strongest published evidence against its soundness (Ye et al.,
  arXiv 2011.14663)<!--ref:2011.14663--><!--anchor:section:V-B-->, it would learn from the pseudo-partitions that E3, K8 and
  MK3 found weak, its novelty is the task's own (scan N4), and it leaves the training-free framing of C2. It returns only if
  D0 shows the fixed reader is the bottleneck and the user accepts a learned-reader framing; then use the scan's design (an
  identity-initialised residual conditioner with the condition-free pathway kept).
- **N3 and N5: demoted to conditional** (Step 4), each behind a privileged reference that is cheap relative to the
  candidate and needed for branch 3 anyway.
- **N1 and N2 as method claims: demoted to repairs.** Both can go to a GO test, but scans N1 and N2 advise writing them as
  "a centered cross-modal diagonal variant of KISSME (2012)<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16-->
  and CCA (2010)<!--ref:rasiwasia-etal-2010-acm--><!--anchor:section:abstract--> on a learned basis" and "a cascade
  control", not as new mechanisms.

### 4.4 Required baselines each candidate brings

| Candidate | Already run (reuse) | New |
|---|---|---|
| N1 | Cosine, RCA<!--ref:bar-hillel05a--><!--anchor:section:abstract-->, KISSME<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16-->, CVS<!--ref:1612.02534--><!--anchor:section:4.3--> ("wang"), pair probe, diag rule (E1); A3 term-only and nested control (E3, A′) | Diagonal KISSME on the same codes; N1's rule on raw CLIP or PCA-32; N1's centered uniform control |
| N2 | Nested score as the tuned convex-fusion comparator (arXiv 2210.11934)<!--ref:2210.11934--><!--anchor:section:abstract-->; nested uniform control (A′) | Both-in-top-k share; first stage at k = 1 as its own control; either-rate loss per k |
| N3 | Cosine, RCA | CRL<!--ref:2510.04564--><!--anchor:section:abstract--> with the true name; SpLiCE<!--ref:2402.10376--><!--anchor:section:abstract--> flat codes with the agreement rule; grouped codes with uniform weights |
| N4 | Cosine, RCA; fixed agreement rule on A3 | Conditioner-off pathway; FEAT-style<!--ref:1812.03664--><!--anchor:section:4.1--> or CTM-style<!--ref:1905.11116--><!--anchor:section:3.2.2--> set encoder on the same banks |
| N5 | 8B and 2B in-context probes; privileged CLIP names (spike) | Qwen3-VL-Embedding<!--ref:2601.04720--><!--anchor:section:abstract--> with the true aspect and with the default instruction; CRL<!--ref:2510.04564--><!--anchor:section:abstract--> with the true name |
| N6 | Cosine, RCA; A3 on the same partitions | D0 (told and inferred); N6 with uniform head weights |

### 4.5 What the pre-registration for the first test(s) must include

The first pre-registration would cover N1 and N2 as one candidate set (and N6 in an addendum if Step 2 runs). It must be
committed before any seed-42 score of these scorers is read, or the seed-42 work must be declared exploratory.

1. **Seeds.** Development on seed 42, with its reuse disclosed from the ledger. Seed 45 is built and scored once, for
   one configuration (one scorer, one checkpoint by SHA-256, one k or λ rule). Developing several candidates does not
   license several seed-45 tests. Recommended, given E3's winner's curse: confirm the picked configuration on a free seed
   (47 or later) under the same rules before seed 45 is built.
2. **Eligible checkpoints.** A1 to A6, C0 and SE. The label-trained L3, L5 and LT are excluded by SHA-256
   (`label_checkpoints.json`), as A′ did; the draft lists LT and L3 for N1's first test, which is fine as diagnosis only.
   R3 is excluded as transductive.
3. **Pick rule.** A′'s min-margin cross-fit against the condition-free control, not E3's mean of R@1 and gain (which
   favoured cells that lost R@1 to the control).
4. **Comparators.** Cosine, RCA and the method's own condition-free control (N2: the first stage at k = 1; N1: its
   centered score with uniform weights), *and* A3's nested uniform control, the strongest condition-free score on these
   codes, so that a weaker own control cannot produce a pass. Intersection-union test, painting-clustered bootstrap
   (5,000 resamples, seed 42), lower bounds above zero; bootstrap seeds 0 to 99 reported as a sensitivity check, since E3
   had boundary cases.
5. **A gate before seed 45.** A′'s "promising" rule (both seed-42 margins at least 2.8 SE) with the predicted chance of
   clearing zero on a fresh draw, Φ(m / (2·SE) − 1.96); if not promising, seed 45 stays unbuilt.
6. **Kill criteria, numerically stated in advance.** N1: term-only gain not above the uncentered term (paired), or
   either rate below the cosine. N2: a minimum both-in-top-k share for some k ∈ {2, 3, 5}.
7. **Named baselines** of §4.4, reported descriptively, and per-pair results with the note that E3's per-pair order was
   unstable across draws.
8. **Framing fixed in advance.** The novelty claim is the task and the combination, "to our knowledge" with the scans'
   bounds; N1 and N2 are described as in §4.3.
9. **Provenance.** SHA-256 of episode sets, checkpoints, partitions and every script in each result file (A′ §7 found
   gaps here).
10. **For an N6 addendum.** Frozen partition hashes; heads trained on scorer-train rows only; D0's numbers as N6's
    ceiling; disclosure that the partitions were chosen to resemble evaluation aspects; the style × genre expectation as
    descriptive.

## 5. What would change the recommendation

1. **D0's inferred-aspect variant recovers most of the told reference's gain**: promote N6 to Step 1 and N3 above N5.
   **It stays in the 1 to 2 point band of every reader measured so far**: the fixed reader is the bottleneck; drop N1, N3
   and N6, and the remaining routes are N4 and N5, both against strong prior evidence, which points toward branch 3.
2. **N2's both-in-top-k share is small for every k**: drop N2; the A′ diagnosis then has no cheap remedy.
3. **N1 lifts the term-only either rate to the cosine's level or above while keeping A3's gain**: N1 becomes T_a in the
   N2 cascade or the nested score and is the first GO configuration.
4. **The privileged Qwen3-VL-Embedding (2026)<!--ref:2601.04720--><!--anchor:section:abstract--> run clears its default-instruction control on gain by a wide margin**: N5 moves to
   Step 2 and its naming-accuracy diagnostic follows.
5. **A full-text read of Ye et al. (2020)<!--ref:2011.14663--><!--anchor:section:V-B--> Table IX shows the pseudo-real gap is specific to augmentation-based pseudo-classes**:
   the soundness argument against N4 weakens (its other costs remain).
6. **A search outside the scans' bounds (Google Scholar, Semantic Scholar, OpenReview) finds a paper in our regime**:
   the novelty framing of every candidate changes, and that paper becomes a required baseline.
7. **The user chooses branch 3 or the deadline moves**: the candidates become analysis items (D0, N1 and N2 as mechanism
   checks, N5's privileged run as K3), and seed 45 need not be spent.

## 6. Synthesis limitations

- Most literature evidence is abstract-level and second-hand through a summarising tool (§3.2); we did not re-open any
  paper.
- Project numbers were checked against the project reports, not re-derived from per-anchor arrays.
- The arithmetic of §4.1 and the expectations about N2's losses and N6's style × genre weakness are reader-derived and
  untested.
- D0 is our proposal, built from the spike's existing probe; its outcome is unknown and it is not a finding.
- The tension inventory covers 11 paper pairs and is not a complete pairwise check.
- We did not assess the timeline beyond the draft's own §4: a GO on ArtELingo alone would support a narrow method claim,
  not branch 1's full claim table.

## Appendix A. Literature matrix (every source in the six scans)

Themes: T1 example-defined conditions; T2 pair-statistic metrics; T3 condition-selected subspaces or bases; T4
pseudo-task and cluster supervision; T5 cascade and fusion; T6 language as the condition interface; T7 abstract-aspect
failure modes. Grades: S, A, I as defined in the Summary. "Flag" marks ids a scan did not fully verify.

| # | Source (id) | Scan | Grade | Themes | Bears on | Flag |
|---|---|---|---|---|---|---|
| 1 | KISSME, Koestinger et al. (2012), TU Graz PDF<!--ref:koestinger_cvpr_2012--><!--anchor:section:eq12-16--> | N1 | S | T2 | N1 | |
| 2 | RCA, Bar-Hillel et al. (2005), JMLR 6<!--ref:bar-hillel05a--><!--anchor:section:abstract--> | N1 | A (full text checked 2026-10-28) | T2 | N1 | |
| 3 | CSN, Veit et al. (2017), arXiv 1603.07810<!--ref:1603.07810--><!--anchor:section:abstract--> | N1, N3, N6 | A | T3 | N1, N3, N6 | CVF page returned 403; arXiv page opened |
| 4 | CVS, Wang et al. (2016), arXiv 1612.02534<!--ref:1612.02534--><!--anchor:section:4.3--> | N1, N4 | S | T1, T2 | N1, N4 | |
| 5 | CRML (2022), arXiv 2211.07116<!--ref:2211.07116--><!--anchor:section:abstract--> | N1 | A | T1 | N1, N4 | |
| 6 | CCA, Rasiwasia et al. (2010), ACM MM<!--ref:rasiwasia-etal-2010-acm--><!--anchor:section:abstract--> | N1 | A (from search) | T2 | N1 | PDF located, binary not parsed |
| 7 | GeneCIS (2023), arXiv 2306.07969<!--ref:2306.07969--><!--anchor:section:abstract--> | N2, N5 | A | T5, T6 | N2, N5 | |
| 8 | Liu et al. (2023), arXiv 2305.16304<!--ref:2305.16304--><!--anchor:section:abstract--> | N2 | A | T5 | N2 | |
| 9 | Sun et al. (2023), arXiv 2312.08924<!--ref:2312.08924--><!--anchor:section:abstract--> | N2 | A | T5 | N2 | |
| 10 | Baumgartner et al. (2022), arXiv 2210.10695<!--ref:2210.10695--><!--anchor:section:abstract--> | N2 | A | T1, T5 | N2 | |
| 11 | Shen et al. (2021), NeurIPS<!--ref:shen2021neurips-reranking--><!--anchor:none:--> | N2 | I | T5 | N2 | index page only (mlanthology); proceedings page not opened |
| 12 | Fusion-function analysis (2022), arXiv 2210.11934<!--ref:2210.11934--><!--anchor:section:abstract--> | N2 | A | T5 | N2 | |
| 13 | SpLiCE, Bhalla et al. (2024), arXiv 2402.10376<!--ref:2402.10376--><!--anchor:section:abstract--> | N3 | A | T3 | N3 | |
| 14 | CRL, Liu et al. (2025), arXiv 2510.04564<!--ref:2510.04564--><!--anchor:section:abstract--> | N3 | A | T3, T6 | N3, N5 | |
| 15 | SP-CRL, Wang et al. (2026), arXiv 2602.05464<!--ref:2602.05464--><!--anchor:section:abstract--> | N3 | A | T3, T7 | N3 | |
| 16 | CLAY, Lim et al. (2026), arXiv 2604.11539<!--ref:2604.11539--><!--anchor:section:abstract--> | N3, N4, N5 | A | T3, T6 | N3, N5 | |
| 17 | SCE-Net, Tan et al. (2019), arXiv 1908.08589<!--ref:1908.08589--><!--anchor:section:abstract--> | N3, N4 | A | T1, T3 | N3, N4 | |
| 18 | LaBo (2023), arXiv 2211.11158<!--ref:2211.11158--><!--anchor:section:abstract--> | N3 | A | T3 | N3 | |
| 19 | Label-free CBM (2023), ICLR<!--ref:label-free-cbm--><!--anchor:none:--> | N3 | I | T3 | N3 | listed by search; ICLR page not opened; no id in the scan |
| 20 | Discover-then-Name (2024), arXiv 2407.14499<!--ref:2407.14499--><!--anchor:section:abstract--> | N3 | A | T3 | N3 | verified, not shortlisted |
| 21 | VisDiff (2024), arXiv 2312.02974<!--ref:2312.02974--><!--anchor:section:limitations--> | N3, N5 | S (N5), A (N3) | T1, T6, T7 | N3, N5 | |
| 22 | Kazmierczak et al. (2025), arXiv 2510.07115<!--ref:2510.07115--><!--anchor:section:abstract--> | N3 | A | T7 | N3 | |
| 23 | Widhoelzl and Takmaz (2024), arXiv 2405.06319<!--ref:2405.06319--><!--anchor:section:abstract--> | N3 | A | T7 | N3, N5 | |
| 24 | CLIP art-style source (2026), arXiv 2605.18974<!--ref:2605.18974--><!--anchor:section:abstract--> | N3 | A | T7 | none | opened, no figure stated; no claim made |
| 25 | FEAT, Ye et al. (2020), arXiv 1812.03664<!--ref:1812.03664--><!--anchor:section:5.2.1--> | N4 | S | T1 | N4 | |
| 26 | CTM, Li et al. (2019), arXiv 1905.11116<!--ref:1905.11116--><!--anchor:section:3.2.2--> | N4 | S | T1 | N4 | |
| 27 | TADAM, Oreshkin et al. (2018), arXiv 1805.10123<!--ref:1805.10123--><!--anchor:section:2.4--> | N4 | S | T1 | N4 | |
| 28 | CACTUs, Hsu et al. (2019), arXiv 1810.02334<!--ref:1810.02334--><!--anchor:section:4.2--> | N4 | S | T4 | N4, N6 | |
| 29 | Ye, Han and Zhan (2020), arXiv 2011.14663<!--ref:2011.14663--><!--anchor:section:V-B--> | N4 | S | T4 | N4 | |
| 30 | BGE-EN-ICL, Li et al. (2024), arXiv 2409.15700<!--ref:2409.15700--><!--anchor:section:4.3--> | N4 | S | T1, T6 | N4 | |
| 31 | DiscoverNet (2022), arXiv 2204.04053<!--ref:2204.04053--><!--anchor:section:abstract--> | N4 | A | T3 | N4 | |
| 32 | MARS (2022), arXiv 2210.00312<!--ref:2210.00312--><!--anchor:section:abstract--> | N4 | A | T1 | N4 | |
| 33 | UMTRA (2018), arXiv 1811.11819<!--ref:1811.11819--><!--anchor:section:abstract--> | N4 | A | T4 | N4 | |
| 34 | PsCo (2023), arXiv 2303.00996<!--ref:2303.00996--><!--anchor:section:abstract--> | N4 | A | T4 | N4 | |
| 35 | Adaptive cross-modal few-shot learning (2019), arXiv 1902.07104<!--ref:1902.07104--><!--anchor:section:abstract--> | N4 | A | T1 | N4 | |
| 36 | CIReVL, Karthik et al. (2024), arXiv 2310.09291<!--ref:2310.09291--><!--anchor:section:abstract--> | N5 | A | T6 | N5 | |
| 37 | Qwen3-VL-Embedding, Li et al. (2026), arXiv 2601.04720<!--ref:2601.04720--><!--anchor:section:abstract--> | N5 | A | T6 | N5 | |
| 38 | TTE (2025), arXiv 2510.05014<!--ref:2510.05014--><!--anchor:section:abstract--> | N5 | A | T6 | N5 | |
| 39 | ABC (2025), arXiv 2503.00329<!--ref:2503.00329--><!--anchor:section:abstract--> | N5 | A | T6 | N5 | CtrlBench R@1 numbers from a search snippet, unverified, not used |
| 40 | Promptable Embeddings, Li et al. (2025), arXiv 2505.15877<!--ref:2505.15877--><!--anchor:section:abstract--> | N5 | A | T6 | N5 | |
| 41 | Bongard study (2024), arXiv 2411.01173<!--ref:2411.01173--><!--anchor:section:abstract--> | N5 | A | T6, T7 | N5 | |
| 42 | VICIS (2026), arXiv 2607.02402<!--ref:2607.02402--><!--anchor:section:abstract--> | N5 | A | T1 | N5 | |
| 43 | Visual Instruction Inversion (2023), arXiv 2307.14331<!--ref:2307.14331--><!--anchor:section:abstract--> | N5 | A | T1, T6 | N5 | |
| 44 | XDC, Alwassel et al. (2020), arXiv 1911.12667<!--ref:1911.12667--><!--anchor:section:abstract--> | N6 | A | T4 | N6 | |
| 45 | MCN (2021), arXiv 2104.12671<!--ref:2104.12671--><!--anchor:section:abstract--> | N6 | A | T4 | N6 | |
| 46 | SwAV (2020), arXiv 2006.09882<!--ref:2006.09882--><!--anchor:section:abstract--> | N6 | A | T4 | N6 | |
| 47 | DeepCluster (2018), arXiv 1807.05520<!--ref:1807.05520--><!--anchor:section:abstract--> | N6 | A | T4 | N6 | |
| 48 | SeLa, Asano et al. (2020), arXiv 1911.05371<!--ref:1911.05371--><!--anchor:none:--> | N6 | I | T4 | N6 | search-result summary only, not opened; context only |
| 49 | MFCVAE (2021), NeurIPS poster page<!--ref:neurips2021-poster-26795--><!--anchor:section:abstract--> | N6 | A | T3, T4 | N6 | |
| 50 | Dual-disentangled Deep Multiple Clustering (2024), arXiv 2402.05310<!--ref:2402.05310--><!--anchor:none:--> | N6 | I | T4 | N6 | seen in search results only |

## Appendix B. The draft's numbers, checked against the project files

| Draft §1 row | Draft value | Project file value | Note |
|---|---|---|---|
| Label-probe reference | 23.09 vs CLIP 11.13 | Spike Result 3: pooled 23.09; CLIP 11.13 | Spike episodes (emotion and style, aspect told, human-label probes on 60,000 scorer-train rows), not the E1 stack |
| Privileged names | +1.51 [0.94, 2.09] | Spike Result 1: 12.63, +1.51 [+0.94, +2.09] | Emotion only (+3.2); zero-shot emotion at the majority-class level, so not a bound on naming in general |
| Factor uniform term | either 33.4 vs 27.1, gain 0 (s43) | E3 §6.1: 33.44 vs 27.05 | Matches |
| A3 term | gain 0.97, either 21.4 (s43) | E3 §6.1: 0.97 [0.53, 1.41], 21.37 | Matches; seed 42: 0.99, 21.04 |
| Nested score | 16.52 vs 16.55, gain −0.01 (s42) | A′ §3: 16.52 [16.18, 16.87], −0.01 [−0.10, 0.07]; control 16.55 | Matches; one exception cell (4, 0.25) rose 0.004 R@1 |
| Label-trained factors | gain 1.85 vs 0.99; loss within 2% | A′ §4.2: L3 1.85 [1.39, 2.33]; §4.3: L3 1.8% below, LT 0.4% above | Matches; A′ also found the loss level does not track the fit (§2.2, row 3) |
| 8B in context | +1.07 [0.17, 1.93], +0.21 [−0.51, 0.94] (s46) | Probe log: same; either +1.93 [0.25, 3.51] | Matches |
| N6 AMI | 0.20 to 0.40 | E2 and MK: 0.159 (caption with genre) to 0.418 (image23 with genre) | Slightly wider range |
| Episode construction (§1 text) | four different values of A, never the anchor's | `aspect_episodes.py`: four distinct values drawn without replacement; each pair shares its value of A and differs on B | Matches; this supports N1's "vary across pairs" premise as a property of the data, not of the method |
