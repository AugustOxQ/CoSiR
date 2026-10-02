# Editorial Decision Package

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from a candidate or prose-named profile. `PROFILE_MEASURED` remains unavailable until a closed profile artifact and replay validator bind the exact target fields to the completed panel's `execution_topology_sha256`.

## Manuscript Information

- **Title**: CoSiR v2: CVPR publication plan (design) with evidence appendices
- **Manuscript type**: pre-results research plan for a CV conference paper, with five appended evidence reports (contract stage note)
- **Manuscript SHA-256**: `d1ebaef8055ca882db4a1c8ba9090ae608a735d3ed96906d4b140eaed5a62342` (`manuscript.md`, unchanged by this synthesis)
- **Panel / round**: `cosir-v2-cvpr-plan-review-round-1`, review round 1
- **Mode and contract**: `reviewer_full`, `reviewer/reviewer_full/v2` (baseline v3.20.0, panel size 5)
- **Plan date**: 2026-10-02. **Decision date**: 2026-10-02
- **Venue binding**: unbound run. All five Phase 1 cards emit `criteria_binding_unavailable`, and no Phase 2 card claims alignment with bound venue criteria (the Journal-Fit seat states that its fit judgements concern readership and significance only). This letter makes no venue-alignment claim.

## Review Panel Provenance (#540/#740)

- **Typed artifact**: `provenance.json` (`review-panel-provenance/1.0`)
- **Artifact SHA-256**: `95fe0c737c9c93ceda9e00abcab3120f053c5aa8fd3aa108b16d0cf661c709d3`
- **Panel ID**: `cosir-v2-cvpr-plan-review-round-1`
- **Normalized manifest SHA-256**: `9fe74134917b5b7354877e432692f16f5ac47cf0699517397700598ac1f02303`
- **Execution topology SHA-256**: `ba4de2cf08761c7f478a4d22252e1c234f07e2b96cfc887197ff296a1c11b059`
- **Fresh-context scope**: `within_panel_attempt_only` (it does not compare retries or prior rounds)
- **Replay**: `scripts/review_panel_provenance.py validate provenance.json` re-run during synthesis: PASS.

| Seat | Role ID | Actor type | Context ID | Peer outputs visible | Model family | Provider | Human reviewer ID |
|---|---|---|---|---|---|---|---|
| EIC | eic | model | phase2-a5a59a0411f99c004 | false | claude | anthropic | null |
| R1 | methodology | model | phase2-acac9543481c0c7b8 | false | claude | anthropic | null |
| R2 | domain | model | phase2-af78edff540b9c5db | false | claude | anthropic | null |
| R3 | perspective | model | phase2-ae5bd1e9ef38d6b5b | false | claude | anthropic | null |
| DA | da | model | phase2-a07a9bb249a8d27a4 | false | claude | anthropic | null |

| Provenance axis | Status (`true` / `false` / `unknown`) |
|---|---|
| Role-separated | true |
| Within-panel invocation-context separation | true |
| Blind to peer outputs | true |
| Model-family distinct | false |
| Provider distinct | false |
| Human-reviewer distinct | false |

- **Binary independence claim**: Not computed (`independence_claim: not_computed_from_personas`). Persona or role diversity proves only `role_separated`; the panel is not relabelled as independent.
- **Correlated-error disclosure**: All model-executed review seats used one model family; role separation does not remove correlated-error risk.
- **Orchestrator observation (recorded fact)**: the EIC seat reported that reading `phase0_reviewer_configuration.md` exposed the other seats' configuration cards (the Phase 0 identity cards, not peer review outputs) before it committed. No seat opened another seat's Phase 1 or Phase 2 output. The artifact records `peer_outputs_visible: false` for every seat; configuration cards are not peer review outputs, so this letter renders the recorded value unchanged and does not re-derive the axis.
- **Schema 6 carrier**: this run emits a letter and a roadmap only; no Schema 6 package was produced, so no carrier was built. The letter prose is not the machine carrier.
- **Cross-model decision check (Step 4b)**: `ARS_CROSS_MODEL` is unset, so the check did not run; this is a single-model synthesis.

## Mechanical Synthesis (v3.6.2 sprint contract)

The decision engine is the contract's three-step protocol (`editorial_decision_standards.md` §0: under a sprint contract the mechanical synthesizer governs, and the recommendation matrix cannot override it). All five cards passed `check_phase_conformance.py` and the panel layer-1 check before synthesis; the panel is complete (5 of 5 usable).

### Step 1: role-scoped scoring matrix

| Dimension | Priority | Eligible roles | Assessed eligible scores | Verdict |
|---|---|---|---|---|
| D1 methodology_rigor | mandatory | methodology | R1 block (repairable) | block |
| D2 domain_accuracy | mandatory | domain | R2 warn | warn |
| D3 argumentative_coherence | mandatory | da, methodology | R1 warn; DA block (repairable) | block |
| D4 cross_disciplinary_relevance | high | perspective | R3 warn | warn |
| D5 writing_and_structure | normal | eic | EIC warn | warn |
| D6 venue_fit_and_contribution | mandatory | eic | EIC warn | warn |

Every other cell is a structural `not_assessed` from an ineligible seat and is excluded from numerator and denominator. No eligible seat abstained, and no seat declared `block_class: fatal`.

### Step 2: failure conditions

| ID | Severity | Quantifier | Expression | Per-dimension evaluation | Fired |
|---|---|---|---|---|---|
| F1 | 95 | any | any mandatory dimension has a fatal block | D1, D2, D3, D6: no fatal block | no |
| F2 | 90 | any | any mandatory dimension scores 'block' | D1: R1 block, true; D3: DA block, true | yes |
| F3 | 70 | majority | two or more mandatory dimensions score 'warn' or worse | D1 (n=1, owner R1 block) true; D2 (n=1) true; D3 (n=2, both seats required: warn and block) true; D6 (n=1) true; 4 of 4 dimensions, at least 2 required | yes |
| F4 | 60 | any | any high-priority dimension scores 'block' | D4 only: R3 warn, not block | no |
| F5 | 40 | any | any dimension scores 'warn' or worse | every dimension is warn or block | yes |
| F0 | 10 | all | every dimension scores 'pass' | D1 is block | no |

### Step 3: precedence and audit lines

Fired: F2 (severity 90), F3 (70), F5 (40). The highest severity is F2, whose action is Major Revision. The single DA CRITICAL (C1) is adjudicated below. The decision is not Accept, so no DA-consistency marker is emitted.

```
dimension_verdicts: [D1=block, D2=warn, D3=block, D4=warn, D5=warn, D6=warn]
fired_conditions: [F2, F3, F5]
da_critical_adjudications: [C1=VALIDATED]
editorial_decision=major_revision
```

---

## Part 1: Editorial Decision Letter

Dear Author(s),

Thank you for submitting "CoSiR v2: CVPR publication plan (design) with evidence appendices" for review against a CVPR main-conference bar. Your plan was reviewed through five role-separated review seats, including a Journal-Fit Reviewer role. Their execution provenance is reported above and is not reduced to a binary independence claim. Following the contract's stage note, the panel read the document as a pre-results plan: missing results were not counted against it, while design flaws that leave a claim untestable were.

### Decision: Major Revision

Major Revision here means that the decision rules and several claims of the plan must be redesigned or reworded before the E13 pre-registration and the final reads. It does not reflect missing results. The revised plan returns for another round of review.

### Blocking Issues (immutable source order)

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| R1; R11; R12 | D1 block (fires F2): the planned decision rules cannot separate success from failure. A condition-blind gain can pass the GO rule and K2; the anchor-only bootstrap is anti-conservative under item reuse; "matches" in K3 has no equivalence margin. | R1 (EIC W1 raises the first part too) | text: §5.1 Common protocol "R@1, the mean over both directions and all conditions" (R1 W1; the W2 and W3 anchors travel with REV-20 and REV-21) | REV-01; REV-20; REV-21 |
| R23 | D3 block (fires F2): no held-out-aspect test, so example-conditioned inference of a respect cannot be told from a K-way selector among trained aspect subspaces (DA C1, VALIDATED). | DA | text: §6 pseudo-partition table, ArtELingo row "GoEmotions affect k-means of captions (emotion-like; distant supervision)" against §4 C2 "is the prior that makes it possible to estimate a similarity from just four example pairs" | REV-55 |

F3 also fired, on the warn-or-worse verdicts of D1, D2, D3 and D6. Clearing these two rows alone would therefore still leave a Major Revision; the other must_fix items (R2 to R10, R13 to R22) carry the remaining D1, D2, D3 and D6 grounds.

### Reviewer Summary

| Reviewer | Role | Dimension scores (v2 cards carry no per-seat recommendation) | Confidence |
|---|---|---|---|
| Journal-Fit Reviewer (EIC) | CVPR 2027 area chair, vision and language, multimodal retrieval | D5 warn; D6 warn. Its preliminary signal, input only: "substantial revision of the plan before E3 runs" | per finding, 3 to 4 |
| Reviewer 1 (methodology) | ML evaluation methodologist for episodic retrieval benchmarks and test-set reuse | D1 block (repairable); D3 warn | per finding, 3 to 5 |
| Reviewer 2 (domain) | researcher in conditional similarity and composed retrieval | D2 warn | per finding, 3 to 4 |
| Reviewer 3 (perspective) | cognitive scientist of similarity, with affective annotation of artworks | D4 warn | overall 4 of 5; per finding, 3 to 5 |
| Devil's Advocate | fixed adversarial seat | D3 block (repairable); findings only | per finding, 4 to 5 |

Confidence values are self-reported scope disclosures. They were not totalled, averaged or used to weight any finding.

### Consensus Analysis

#### Step 1a: reviewer summary matrix

| Item | Journal-Fit Reviewer | R1 (methodology) | R2 (domain) | R3 (perspective) |
|---|---|---|---|---|
| Scores | D5 warn, D6 warn | D1 block, D3 warn | D2 warn | D4 warn |
| Scope disclosure | does not audit statistics or label construct validity | label-noise ceilings adjacent; construct validity deferred to R3 | affective datasets adjacent | outsider to CVPR leaderboard and statistics conventions |
| Key strengths | claims tied to evidence and status (S1); negative evidence redirected the task (S2); budgeted final reads (S4) | single-read hash ledger (S1); negative diagnostics acted on (S2); CUB unseen-species split (S6) | novelty scoped against the right lineages (S1); careful GeneCIS sourcing (S5) | aspect and value map onto known constructs (S1); swap read beside correctness (S2) |
| Findings | 16 (6 Major, 10 Minor) | 19 (6 Major, 13 Minor); receipts AR1 to AR12: 9 consistent, 3 not computable, 0 mismatch | 10 (6 Major, 4 Minor) | 8 (4 Major, 4 Minor) |
| Questions | 4 | 5 | 4 | 5 |
| Minor-issue lists | 2 bullets | none (Minor findings carry severity tags) | 2 bullets, 3 references (1 lead UNVERIFIED) | reading list (2 leads UNVERIFIED) |

The Devil's Advocate card carries 1 CRITICAL, 5 MAJOR, 6 minor issues and 3 non-defect observations; it is tracked outside the consensus count.

#### Step 1b: weakness sub-claim inventory

Each weakness bundle was split into atomic sub-claims (SC-1 to SC-53). The table lists every raised or disputed position of the four non-DA seats; every (sub-claim, seat) pair not listed is `not-mentioned`, which is silence and not opposition. Severity and confidence are transported per finding from the cards; no row needed a fallback tag. In a blind panel each seat reached its position independently, so agreeing positions are recorded as `raised`. A `disputed (severity)` row agrees that the problem exists but rates it in a different severity band.

| sub_claim_id | parent_weakness | reviewer_id | position | evidence_pointer (type: locator; full anchor in the roadmap) | severity | confidence | roadmap item |
|---|---|---|---|---|---|---|---|
| SC-1 | EIC W1.1 | EIC | raised | text: §6 Go/no-go, GO bullet | major | 4 | REV-01 |
| SC-1 | R1 W1.1 | R1 | raised | text: §5.1 Common protocol | major | 4 | REV-01 |
| SC-2 | EIC W1.2 | EIC | raised | text: §6 Go/no-go, GO bullet | major | 4 | REV-01 |
| SC-2 | R1 W1.2 | R1 | raised | text: §5.1 Common protocol | major | 4 | REV-01 |
| SC-3 | EIC W2.1 | EIC | raised | table: aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14 | major | 4 | REV-02 |
| SC-3 | R2 W5.2 | R2 | raised | text: §4 claims table K3 | major | 3 | REV-02 |
| SC-3 | R3 W3.2 | R3 | raised | text: §4 K3 and §12 R-names | major | 4 | REV-02 |
| SC-4 | EIC W2.2 | EIC | raised | table: aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14 | major | 4 | REV-02 |
| SC-4 | R2 W5.3 | R2 | raised | text: §4 claims table K3 | major | 3 | REV-02 |
| SC-4 | R3 W3.3 | R3 | raised | text: §4 K3 and §12 R-names | major | 4 | REV-02 |
| SC-5 | EIC W2.3 | EIC | raised | table: aspect-episode spike appendix, Result 1 table — Names (privileged, cv) emotion 13.31 against SE agree (cv) 10.34 and CLIP only 10.14 | major | 4 | REV-03 |
| SC-5 | R3 W6.1 | R3 | disputed (severity) | text: §1 and §3 | minor | 3 | REV-03 |
| SC-6 | EIC W3.1 | EIC | raised | text: §5.4 example protocol | major | 4 | REV-04 |
| SC-6 | R2 W1.2 | R2 | raised | text: §5.4 item 2 | major | 4 | REV-04 |
| SC-7 | EIC W3.2 | EIC | raised | text: §5.4 example protocol | major | 4 | REV-05 |
| SC-8 | EIC W4 | EIC | raised | text: novelty-check appendix, "Searched, nothing found" paragraph | major | 4 | REV-06 |
| SC-9 | EIC W5 | EIC | raised | absence: §8 baselines table and §10 primary comparisons | major | 4 | REV-07 |
| SC-9 | R2 W2.1 | R2 | raised | absence: §8 baseline table and §10 pre-declared primary comparisons | major | 4 | REV-07 |
| SC-10 | EIC W6 | EIC | raised | text: §4 Fallback | major | 4 | REV-08 |
| SC-11 | EIC W7 | EIC | raised | text: §4 C1 novelty statement | minor | 4 | REV-09 |
| SC-12 | EIC W8 | EIC | disputed (severity) | text: §4 C3 | minor | 3 | REV-10 |
| SC-12 | R1 W5.1 | R1 | raised | text: §4 C3 and backbone check Result 1 | major | 3 | REV-10 |
| SC-12 | R2 W6.1 | R2 | raised | text: backbone check Verdict | major | 3 | REV-10 |
| SC-12 | R3 W1.1 | R3 | raised | text: §4 C3 and backbone check Verdict | major | 4 | REV-10 |
| SC-13 | EIC W9 | EIC | disputed (severity) | text: §4 claims table, K2 | minor | 4 | REV-11 |
| SC-13 | R1 W4.1 | R1 | raised | text: §4 K2 and §10 | major | 4 | REV-11 |
| SC-14 | EIC W10 | EIC | raised | absence: plan body §4 to §12 | minor | 4 | REV-12 |
| SC-15 | EIC W11 | EIC | raised | text: §12 R-time | minor | 3 | REV-13 |
| SC-16 | EIC W12 | EIC | raised | text: §11 E15 and E16 rows | minor | 4 | REV-14 |
| SC-17 | EIC W13 | EIC | raised | absence: plan body | minor | 4 | REV-15 |
| SC-18 | EIC W14 | EIC | raised | text: §8 Tier-1 row for C0, SE and R3 | minor | 3 | REV-16 |
| SC-19 | EIC W15 | EIC | raised | text: §2.4 opening | minor | 4 | REV-17 |
| SC-20 | EIC W16 | EIC | raised | text: aspect-episode spike appendix, caveats | minor | 4 | REV-18 |
| SC-20 | R1 W8 | R1 | raised | text: §6 Strong GO | minor | 4 | REV-18 |
| SC-21 | R1 W2 | R1 | raised | text: §10 Statistics | major | 4 | REV-20 |
| SC-22 | R1 W3 | R1 | raised | text: §4 claims table, K3 | major | 5 | REV-21 |
| SC-23 | R1 W4.2 | R1 | raised | text: §4 K2 and §10 | major | 4 | REV-22 |
| SC-24 | R1 W5.2 | R1 | raised | text: §4 C3 and backbone check Result 1 | major | 3 | REV-23 |
| SC-24 | R2 W6.2 | R2 | raised | text: backbone check Verdict | major | 3 | REV-23 |
| SC-24 | R3 W1.2 | R3 | raised | text: §4 C3 and backbone check Verdict | major | 4 | REV-23 |
| SC-25 | R1 W6 | R1 | raised | text: §11 E7 | major | 3 | REV-24 |
| SC-26 | R1 W7 | R1 | raised | text: §6 Go/no-go | minor | 4 | REV-25 |
| SC-27 | R1 W9 | R1 | raised | text: aspect-episode spike Result 2 | minor | 4 | REV-26 |
| SC-28 | R1 W10 | R1 | raised | text: §10 Seeds | minor | 4 | REV-27 |
| SC-29 | R1 W11 | R1 | raised | absence: §10 Statistics | minor | 4 | REV-28 |
| SC-30 | R1 W12 | R1 | raised | text: §10 | minor | 4 | REV-29 |
| SC-31 | R1 W13 | R1 | raised | text: support-baseline spike, Caveats | minor | 3 | REV-30 |
| SC-32 | R1 W14 | R1 | raised | text: §8 with §6 against the aspect spike | minor | 3 | REV-31 |
| SC-33 | R1 W15 | R1 | disputed (severity) | text: §5.1 | minor | 3 | REV-32 |
| SC-33 | R3 W4.1 | R3 | raised | text: §5.3 and §5.1 | major | 3 | REV-32 |
| SC-34 | R1 W16 | R1 | raised | text: §5.3 | minor | 3 | REV-33 |
| SC-35 | R1 W17 | R1 | raised | table: backbone check Result 3, Primary colour (15, 21.0) row, SigLIP 2 cell 61.9 / 64.5 | minor | 5 | REV-34 |
| SC-36 | R1 W18 | R1 | disputed (severity) | text: §11 E13 | minor | 3 | REV-35 |
| SC-36 | R2 W5.1 | R2 | raised | text: §4 claims table K3 | major | 3 | REV-35 |
| SC-36 | R3 W3.1 | R3 | raised | text: §4 K3 and §12 R-names | major | 4 | REV-35 |
| SC-37 | R1 W19 | R1 | raised | text: backbone check Result 3 | minor | 3 | REV-36 |
| SC-38 | R2 W1.1 | R2 | raised | text: §5.4 item 2 | major | 4 | REV-37 |
| SC-39 | R2 W2.2 | R2 | raised | absence: §8 baseline table and §10 pre-declared primary comparisons | major | 4 | REV-38 |
| SC-40 | R2 W3 | R2 | raised | text: §8 Tier 1 unsupervised-bases row | major | 3 | REV-39 |
| SC-41 | R2 W4.1 | R2 | raised | text: literature review §5 item 5 | major | 4 | REV-40 |
| SC-41 | R3 W7 | R3 | disputed (severity) | text: §5.1 Splits | minor | 4 | REV-40 |
| SC-42 | R2 W4.2 | R2 | raised | text: literature review §5 item 5 | major | 4 | REV-41 |
| SC-43 | R2 W7 | R2 | raised | text: literature review §2.2 pitfalls | minor | 4 | REV-42 |
| SC-44 | R2 W8 | R2 | raised | text: Appendix A glossary | minor | 4 | REV-43 |
| SC-45 | R2 W9 | R2 | raised | text: novelty check §2 | minor | 4 | REV-44 |
| SC-46 | R2 W10 | R2 | raised | absence: literature review §2.1 and §6, novelty check thread 1 and §6 | minor | 3 | REV-45 |
| SC-47 | R3 W1.3 | R3 | raised | text: §4 C3 and backbone check Verdict | major | 4 | REV-47 |
| SC-48 | R3 W2 | R3 | raised | absence: §2.3, §5.1, §5.2 and §10 | major | 5 | REV-48 |
| SC-49 | R3 W3.4 | R3 | raised | text: §4 K3 and §12 R-names | major | 4 | REV-49 |
| SC-50 | R3 W4.2 | R3 | raised | text: §5.3 and §5.1 | major | 3 | REV-50 |
| SC-51 | R3 W5 | R3 | raised | text: novelty check §4 must-cite list | minor | 4 | REV-51 |
| SC-52 | R3 W6.2 | R3 | raised | text: §1 and §3 | minor | 3 | REV-52 |
| SC-53 | R3 W8 | R3 | raised | text: §2.4 and §2.3 | minor | 4 | REV-53 |

#### Step 1c: surface-form parity

Each sub-claim was assessed on its substance against the plan text. Informal framings (for example R3's "who, concretely, holds cross-item pairs") and technical ones (for example R1's GRIM receipts) were held to the same evidential test. No sub-claim was marked unevaluable.

#### Points of agreement

The denominator is the four non-DA seats.

- **[CONSENSUS-4]**: none. The one sub-claim all four raise (SC-12, the wording of C3) carries a severity conflict and routes to SPLIT under the precedence rule.
- **[CONSENSUS-3] SC-3 and SC-4 (REV-02)**: K3's subjective half meets adverse appended evidence (privileged names 13.31 on emotion against 10.14 for CLIP only, example scorers flat), and R-names narrows K3 to where names fail, so K3 has no outcome-neutral branch. Raised by EIC W2, R2 W5 and R3 W3; **R1 is silent**. DA M3 agrees.
- **[CONSENSUS-3] SC-24 (REV-23)**: per-viewer emotion labels (about five rows per painting) cap any image-only emotion probe whatever the encoder, so the flat weak-side probes are predicted by label construction. Raised by R1 W5, R2 W6 and R3 W1; **the EIC is silent**. DA M1 agrees.
- **Corroborated findings (2 of 4, below the consensus bar, no conflict)**: SC-1 and SC-2, a condition-blind gain passes the GO rule and K2 and no condition-removed control exists (EIC W1, R1 W1; REV-01). SC-6, K5 is decidable only through a stretch protocol (EIC W3, R2 W1; REV-04). SC-9, in-context alternatives that use the same examples are missing or stretch-only (EIC W5, R2 W2; REV-07). SC-20, "ceiling" is neither a bound nor stable (EIC W16, R1 W8; REV-18).
- **Single-reviewer findings**: every other sub-claim. Each is retained and judged against its named dimension and anchored evidence, not by a confidence weight.

#### Points of disagreement

The Journal-Fit Reviewer would normally arbitrate SPLITs, but in this blind panel the EIC card was committed without sight of the other cards and is itself a party to three of the six. The editor therefore arbitrated each SPLIT under the §3b principles (evidence first, then role expertise, otherwise unresolved dissent) and cites the EIC card wherever it speaks to the point.

**Disagreement 1: a use case for examples over names (SC-5, REV-03)**
- **EIC W2 view**: no use case is named; this is the first objection an AC will raise against C1 (Major).
- **R3 W6 view**: no user scenario is given and the rules make the condition harder to supply than to name (Minor; R3 states adjacent competence).
- **Disagreement type**: severity.
- **Editor's Resolution**: Major, must_fix.
- **Resolution Rationale**: role expertise. The use case carries the significance argument for C1, which is the Journal-Fit seat's remit (D6). Both seats ask for the same remedy.

**Disagreement 2: C3's "belongs to the data" (SC-12, REV-10)**
- **EIC W8 view**: overstated, but "the fix costs one sentence" (Minor).
- **R1 W5, R2 W6, R3 W1 view**: overstated, and C3 is a contribution and a pillar of the fallback, so the wording and its analysis need repair (Major).
- **Disagreement type**: severity.
- **Editor's Resolution**: Major, must_fix.
- **Resolution Rationale**: all four agree on existence. The EIC itself defers the construct question to the perspective seat. The three seats with label-construction or annotation competence rate it Major, and §4 C3 states the causal reading outright. The EIC's minimum remedy, rewording, remains an acceptable way to close the item.

**Disagreement 3: K2 worded beyond its tests (SC-13, REV-11)**
- **EIC W9 view**: K2 is worded more strongly than §10 tests it (Minor).
- **R1 W4 view**: the same, plus no rule for partial passes, and GeneCIS cannot satisfy "both directions, on every dataset" (Major).
- **Disagreement type**: severity.
- **Editor's Resolution**: Major, must_fix.
- **Resolution Rationale**: aligning pre-registered tests with claim wording is a methodology question, so it defers to R1, whose GeneCIS point is a concrete inconsistency in the plan. DA M2 independently rates it Major (not counted).

**Disagreement 4: exclusivity and third-aspect balance of candidates (SC-33, REV-32)**
- **R1 W15 view**: not stated, and it can tilt condition-blind scorers; "the code may already enforce exclusivity" (Minor).
- **R3 W4 view**: the third aspect is uncontrolled, so a correlated respect can travel with the target (Major).
- **Disagreement type**: severity.
- **Editor's Resolution**: unresolved dissent. The item stays should_fix and the authors must address it.
- **Resolution Rationale**: the evidence that would settle the severity (the episode code and the realised imbalance) is not in the manuscript. Neither seat's finding grounds a dimension that fires a Major condition, so the obligation does not depend on resolving the dispute. The panel did not resolve it.

**Disagreement 5: the subjective and objective split of K3 (SC-36, REV-35)**
- **R2 W5 and R3 W3 view**: no definition or criterion exists (Major). R2 sets the deadline at the Oct 23 pre-registration; R3 sets it before E10.
- **R1 W18 view**: the assignment is scheduled after E10's development comparisons, so it can be fitted to their outcomes (Minor).
- **Disagreement type**: severity, with a difference in timing.
- **Editor's Resolution**: Major, must_fix; fix the assignment before E10.
- **Resolution Rationale**: evidence. §11 places E10 (Oct 12 to 20) before E13 (Oct 21 to 23), so R1's circularity risk is real under the later deadline. R2's own criteria (rater agreement, nameability) do not depend on model results, so fixing them before E10 satisfies all three seats. The severity follows R2's D2 ground and R3's core competence; R1 rated only the timing part.

**Disagreement 6: "no labels" and the GoEmotions overlap (SC-41, REV-40)**
- **R2 W4 view**: the wording undersells distant supervision from a classifier that names 6 of the 8 evaluation emotions (Major).
- **R3 W7 view**: the same overlap; affect readers will read it as weak supervision (Minor).
- **Disagreement type**: severity.
- **Editor's Resolution**: Major, must_fix.
- **Resolution Rationale**: evidence and role. The manuscript's literature review confirms the 6-of-8 overlap and the §6 table labels the partition "distant supervision". Accuracy of the C2 claim is R2's remit. R3's remedies (per-category split, teacher-only row) are folded into the item.

**Recorded dissent outside the SPLIT count**
- R1 S6 credits the CUB unseen-species split as "a clean out-of-distribution test of R-pseudo", while DA C1 holds that no planned test can check aspect-level generalisation. This is resolved in the DA adjudication below: both positions hold at their own scope.
- SC-20 (REV-18): EIC W16 proposes stating strong GO as a gain over backbone-only, R1 W8 as an effect over the best baseline. These are compatible variants, not a conflict; the authors state which reference they use.

### DA-CRITICAL Adjudication

**C1 (DA, D3; Confidence 4, adjacent expertise in few-shot and metric-learning evaluation design; block_class repairable)**
- **DA's argument**: every ArtELingo and CUB evaluation aspect has a training pseudo-partition built to mirror it (one from a supervised emotion classifier), and no cross-modal dataset holds an aspect out of training. The design therefore cannot tell example-conditioned similarity from a K-way selector among pre-trained aspect subspaces, and the mechanism claim C2, with the motivation behind C1, cannot be refuted as planned. The DA's repair: leave-one-aspect-out on ArtELingo and generic partitions on CUB, each under a pre-registered rule.
- **Corroboration**: R2 W4 independently raises the premise (the ArtELingo partitions are hand-matched to the evaluation aspects; Major), and R3 W7 raises it for emotion (Minor). R1 W1 and EIC W1 raise a related but distinct threat (a condition-blind gain), which does not separate selection from inference.
- **Contrary positions**: R1 S6 calls the CUB unseen-species split a clean test of R-pseudo, and R1's D1 note says no evaluation label is circular with the training signal.
- **Journal-Fit Reviewer's assessment**: the EIC card was written blind to the DA and does not assess C1. Its nearest positions are W1 and its condition that C2 is significant only if K7 shows the learned basis carries the gain. The adjudication therefore rests with the editor, on the manuscript and the other cards.
- **Editor's check against the manuscript**: the §6 table labels the ArtELingo partitions emotion-like (distant supervision), style-like (adjusted mutual information 0.32 with style) and genre-like. The CUB partitions are per-sentence caption clusters motivated by part descriptions. R-pseudo names "the cross-dataset and unseen-species results" as its test, and §5.2's unseen CUB species keep the same attribute groups. One scope correction: §6 lists generic image and description k-means for SemArt, so the DA's phrase "every other dataset gets its own aspect-matched partitions" overstates SemArt. SemArt still holds no aspect out, and the plan attaches no decision rule to it, so the gap stands. R1 S6 is right that unseen species test generalisation to new items; it does not test unseen aspects, so it does not rebut C1. R1's D1 note concerns label circularity, which C1 does not allege.
- **Adjudication**: **VALIDATED**, as a repairable design gap. It is not a fatal flaw: the DA scored it repairable and no seat declared a fatal block.
- **Effect on the decision**: none beyond the mechanical result. D3's block comes from this finding, and D1's block alone would also fire F2.
- **Required author response**: REV-55 (R23). Add the held-out-aspect test or tests with a rule fixed in E13, or narrow C2 and the C1 motivation to selection among aspects represented in the training partitions.

### Decision Rationale

The decision is mechanical under the sprint contract. Two mandatory dimensions are at block, both repairable: methodology rigor (R1) and argumentative coherence (the DA, with R1 at warn). F2 therefore fires and sets Major Revision. F3 also fires, because four mandatory dimensions (D1, D2, D3, D6) are at warn or worse, and F5 fires on every warn. No seat declared a fatal block, so F1 (Reject) does not fire, and D4, the high-priority dimension, is at warn, so F4 does not.

The cards agree on why. Every seat credits the plan's discipline: the single-read ledger, the retirement of the old benchmark after the support spike, and the candid positioning against prior art. The problems lie in what the planned decision rules can distinguish and in claims worded beyond their tests. A condition-blind gain can pass the GO rule and K2 (EIC, R1). The bootstrap ignores item reuse, and "matches" has no margin (R1). K3 has no outcome-neutral branch while the only appended test favours names on emotion (EIC, R2, R3). C3's "belongs to the data" cannot be separated from how the ArtEmis labels and captions were collected (all four non-DA seats). K5 uses the wrong GeneCIS column and is decidable only through a stretch protocol (R2, EIC). No held-out-aspect test separates example-conditioned inference from selection among trained aspect subspaces (DA C1, validated).

A stricter decision is not supported: every block is repairable before the E13 pre-registration, and no seat found a claim that cannot be made testable. A lighter one is excluded because F2 and F3 both fire. None of this reflects missing results. It reflects design and claim wording that the revised plan must settle before the final reads.

---

## Part 2: Revision Roadmap (summary)

The immutable roadmap core is in `revision_roadmap.md`: 58 items in source order (23 must_fix, 19 should_fix, 16 consider), with the `revision-roadmap/1.0` JSON. `scripts/revision_roadmap.py validate-roadmap` passed against the anchored base and its block manifest. `R<n>` numbers the must_fix items in roadmap source order and `S<n>` the rest; both are transport references, not ranks.

**Obligation rule (derived from the decision contract, not a work rank).**
- `must_fix`: the driving finding is cited by its seat as a ground of that seat's score on D1, D2, D3 or D6, the mandatory dimensions whose verdicts fire F2 and F3. These are EIC W1 to W6, R1 W1 to W5, R1 W11 and W12 (cited as decision-bearing for D1's statistical reporting) and R2 W1 to W6. The validated DA C1 is also must_fix.
- `should_fix`: other Major findings; findings a seat ties to D4 or D5, which fire only F5; Minor findings the EIC tags to D6; and Minor findings raised by two or more non-DA seats.
- `consider`: everything else, including the untagged minor-issue lists.

**[CONSENSUS-LEVEL-ENUM-GAP]**: the `revision-roadmap/1.0` `consensus_level` enum has no value for a two-seat corroborated or a single-reviewer first-round finding. Such items carry `SINGLE-VERIFIER`, and their descriptions state the exact seat count.

### Required Revisions (Must Fix)

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

### Required Item Details

**R1: Condition-attributable GO statistic and condition-removed control** (REV-01; SINGLE-VERIFIER)
- **Problem**: The GO rule and the three §10 primary comparisons are decided on pooled aspect R@1 alone, which a scorer that ignores the condition can raise (to about 50% under R1's construction), and no Tier-1 control runs the trained factors with the condition removed. Corroborated finding, 2 of 4 non-DA seats (EIC W1, R1 W1); R2 and R3 silent.
- **Source**: EIC W1.1, EIC W1.2, R1 W1.1, R1 W1.2
- **Requirement**: Add a condition-attributable statistic to the GO rule and to K2 with the same CI rule (for example condition-correct minus condition-swapped R@1, or R@1 minus the other-aspect rate); add 'ours with uniform weights' and 'ours under the swapped condition' as Tier-1 controls; pre-declare ours versus ours without the condition as a primary comparison; report a condition-blind probe reference beside the aspect-aware one.
- **Acceptance criteria**: §6 and §10 state a condition-sensitivity requirement with its CI rule, §8 lists both condition-removed controls, and the backbone-only value of the new statistic is reported on selection episodes.

**R2: K3 has no outcome-neutral branch while current evidence favours names on emotion** (REV-02; CONSENSUS-3)
- **Problem**: K3's subjective half meets adverse appended evidence (privileged names reach 13.31 on emotion against 10.14 for CLIP only, while every example-based scorer stays flat), and R-names narrows K3 'to where names fail (subjective aspects)', so the plan has no branch for names winning everywhere and K3 can survive any outcome. CONSENSUS-3: EIC W2, R2 W5, R3 W3; R1 silent. DA M3 also raises it.
- **Source**: EIC W2.1, EIC W2.2, R2 W5.2, R2 W5.3, R3 W3.2, R3 W3.3, DA M3.3
- **Requirement**: Rewrite R-names so its mitigation does not presuppose the direction; pre-declare what the paper claims if names win on every aspect (for example examples that compose with names, or groupings with no stable name); run the Qwen-instruction naming comparison on selection rows before the Oct 9 go/no-go and report it beside the example scorers.
- **Acceptance criteria**: §4 and §12 state outcome-neutral claims for both directions of K3, and the selection-row naming comparison is reported with paired CIs.

**R3: A use case in which pairs exist but the aspect cannot be named** (REV-03; SPLIT)
- **Problem**: The plan names no setting in which a person holds value-disjoint, cross-item demonstration and contrast pairs but cannot name the aspect. SPLIT on severity (EIC W2 Major, R3 W6 Minor); arbitrated to the Journal-Fit seat's Major because the use case carries the significance argument for C1, which is that seat's remit.
- **Source**: EIC W2.3, R3 W6.1
- **Requirement**: State a concrete setting with its data source in which demonstration pairs exist but the aspect resists naming (for example curating a captioned collection by an unnamed stylistic or affective quality from a seed set), and tie it to the relevance-feedback line the plan already cites.
- **Acceptance criteria**: §1 or §3 names the setting, who supplies the pairs and from which corpus.

**R4: K5 is decidable only through a stretch protocol** (REV-04; SINGLE-VERIFIER)
- **Problem**: K5's comparison with published GeneCIS rows is decidable only through the text protocol, which §5.4 and E7 keep as a stretch goal, because the example protocol 'is not comparable with published numbers'. Corroborated finding, 2 of 4 (EIC W3, R2 W1); DA M4 also.
- **Source**: EIC W3.1, R2 W1.2, DA M4.2
- **Requirement**: Either commit the GeneCIS text protocol as required work in §5.4 and E7 with a defined threshold for 'competitive', or reword K5 so the example-protocol result is reported next to the plan's own image, text and image-plus-text baselines with no published-row comparison.
- **Acceptance criteria**: K5 names its protocol, its comparator rows and its threshold, or no longer claims a published-row comparison.

**R5: GeneCIS measures transfer, not the cross-modal task** (REV-05; SINGLE-VERIFIER)
- **Problem**: GeneCIS is image to image, so even its comparable row measures transfer and cannot test the cross-modal task; every cross-modal result rests on aspect episodes the authors build. Single-reviewer finding (EIC W3); DA M4 also states it.
- **Source**: EIC W3.2, DA M4.3
- **Requirement**: State in §3 and §5.4 that GeneCIS measures transfer, not the cross-modal task, and make the self-built aspect episodes reusable (see the release item) so they can serve as the external anchor.
- **Acceptance criteria**: §3 and §5.4 describe GeneCIS as a transfer check, and the release item covers the episode files.

**R6: Complete the novelty search before the title and abstract are fixed** (REV-06; SINGLE-VERIFIER)
- **Problem**: The 'first' claim of C1 rests on searches the authors call incomplete (about 25 arXiv API queries, no web search engine, Semantic Scholar rate-limited, no Google Scholar), and §11 schedules no completion before E15 fixes the title and abstract on Nov 7. R2 W9 and W10 each name one omitted precedent; those are separate items.
- **Source**: EIC W4
- **Requirement**: Add an E-row before E15 for a full search (Google Scholar, Semantic Scholar, forward citations of KISSME, Wang et al. 2016, MARS, BGE-EN-ICL and GeneCIS, and the CVPR 2026 and ECCV 2026 proceedings), with a pre-declared rewording of C1 if a match appears.
- **Acceptance criteria**: §11 contains the search row, its report lists queries and sources, and C1 carries the pre-declared fallback wording.

**R7: In-context alternatives that use the same examples** (REV-07; SINGLE-VERIFIER)
- **Problem**: Alternatives that use the same support and contrast pairs without a learned basis are absent (an instruction embedder given the pairs in context; verbalise-then-name) or stretch-only (an in-context MLLM reranker), although the plan's own novelty check lists them. Corroborated finding, 2 of 4 (EIC W5, R2 W2); the DA's ignored-alternatives list names the reranker too.
- **Source**: EIC W5, R2 W2.1
- **Requirement**: Move verbalise-then-name and the in-context instruction embedder (Qwen3-VL-Embedding-2B, the chosen backbone) into Tier 2 and E10; run the in-context MLLM reranker on a fixed, pre-registered subsample; report them beside the primary comparisons.
- **Acceptance criteria**: §8 lists the three in-context baselines outside the stretch row, E10 schedules them, and the pre-registration fixes the reranker subsample.

**R8: Define the NO-GO fallback paper** (REV-08; SINGLE-VERIFIER)
- **Problem**: The NO-GO fallback (K1, K3, C3) has no claim table, target venue or switching criteria, although NO-GO is a live outcome with the factors at CLIP level on aspect episodes today.
- **Source**: EIC W6
- **Requirement**: Write the fallback's claim table now (a released benchmark, strong in-context and named baselines, a defined subjective versus objective split, C3 tested across annotation protocols), name the venue it targets, and pre-declare the criteria for switching.
- **Acceptance criteria**: §4 Fallback carries a claim table, a named venue and switching criteria fixed before the Oct 9 decision.

**R9: C3's 'belongs to the data' outruns the backbone test** (REV-10; SPLIT)
- **Problem**: C3's causal wording ('so it belongs to the data') outruns the backbone-invariance test, which cannot separate what images and captions contain from how ArtEmis captions and labels were elicited (each caption explains one viewer's emotion; Reed captions describe parts). All four non-DA seats raise it (EIC W8, R1 W5, R2 W6, R3 W1); SPLIT on severity only (EIC Minor, the other three Major), arbitrated Major. DA M1 also.
- **Source**: EIC W8, R1 W5.1, R2 W6.1, R3 W1.1, DA M1.1
- **Requirement**: Narrow C3 to the annotated data and its elicitation protocol (for example 'in datasets whose captions explain a self-reported emotion, emotion is carried by the text'), unless the protocol-contrast analysis still shows a modality effect; use SemArt's curator-written catalogue text, and CUB, as elicitation-protocol contrasts in E12 rather than as confirmation.
- **Acceptance criteria**: §4 C3 and the backbone-check verdict no longer attribute the asymmetry to modality alone, or E12 reports a protocol-contrast result that supports the stronger wording.

**R10: K2 is worded beyond the pre-declared tests** (REV-11; SPLIT)
- **Problem**: K2 claims wins 'in both directions, on every dataset' over three baseline families, but §10 pools directions into one primary metric, sets no rule for partial passes (some datasets, one backbone), and GeneCIS (image to image, no aspect episodes) cannot satisfy the wording. EIC W9 (Minor) and R1 W4 (Major); SPLIT on severity, arbitrated Major. DA M2 also.
- **Source**: EIC W9, R1 W4.1, DA M2.1
- **Requirement**: Restate K2 on the pooled primary metric per dataset, list the datasets K2 covers (leaving GeneCIS to K5), add per-direction tests or report directions as secondary, and pre-register in E13 the headline wording for each pattern of passes.
- **Acceptance criteria**: K2's wording matches the §10 primary comparisons, and the E13 pre-registration lists the claim made under each pass pattern.

**R11: Cluster the bootstrap over reused items** (REV-20; SINGLE-VERIFIER)
- **Problem**: §10 bootstraps anchors only, but items recur heavily across episodes (about 10 episodes per held ArtELingo painting per aspect pair, about 115 per SemArt test painting; CUB's 50 test species carry largely species-level attributes), so 'CI lower bound above 0' is anti-conservative for claims about new paintings or species. The DA's minor issues raise the same point.
- **Source**: R1 W2
- **Requirement**: State the inference target; use a painting-level cluster bootstrap (species-level on CUB) that resamples items and regenerates or reweights episodes, or a two-way anchor-by-item bootstrap; base the power calculation on the same unit and report CUB results per species.
- **Acceptance criteria**: §10 names the resampling unit and the inference target, and the GO test and primary comparisons use it.

**R12: An equivalence margin for 'matches'** (REV-21; SINGLE-VERIFIER)
- **Problem**: §10 defines 'beats' as a CI lower bound above 0 but gives 'matches' no criterion, so K3's 'match it on objective ones' passes by default when underpowered and can be refuted only by a significant loss. DA M3 also.
- **Source**: R1 W3, DA M3.1
- **Requirement**: Pre-declare an equivalence margin in R@1 points per dataset before E14, test it with two one-sided tests (or require the 90% CI inside the margin), and power the episode count for it; otherwise word K3 as 'not significantly different' and drop 'match'.
- **Acceptance criteria**: §10 states the margin, the test and its power, or K3 no longer uses 'match'.

**R13: K7 is a descriptive comparison** (REV-22; SINGLE-VERIFIER)
- **Problem**: The K7 test (the agreement rule on the best unsupervised basis against the learned factors) is descriptive in §10, although it carries C2. DA M2 also notes that K6 and K7 have no decision rule.
- **Source**: R1 W4.2, DA M2.2
- **Requirement**: Promote 'ours versus the rule on the best unsupervised basis, chosen on development' to a pre-declared primary comparison, or state K7 as descriptive and word C2 accordingly.
- **Acceptance criteria**: §10 lists the K7 comparison as primary with its baseline fixed on development, or K7 and C2 are worded as descriptive.

**R14: Per-viewer labels cap the image-side emotion probe** (REV-23; CONSENSUS-3)
- **Problem**: Each ArtELingo row is one viewer's emotion and a painting carries about five rows that can disagree (308,723 rows over 61,402 paintings), so an image-only predictor is capped by the share of rows matching the painting's modal emotion whatever the encoder; flat weak-side emotion probes (35.1 to 36.6 against a 31.8% majority) are what the label construction predicts. CONSENSUS-3: R1 W5, R2 W6, R3 W1; EIC silent. DA M1 also.
- **Source**: R1 W5.2, R2 W6.2, R3 W1.2, DA M1.2
- **Requirement**: On scorer-train rows, compute the label-construction ceiling for image-to-emotion (mean share of each painting's modal emotion), report weak-side probes as a fraction of it, and score image-side emotion against painting-level majority or distribution labels and on high-agreement paintings.
- **Acceptance criteria**: E12 reports the ceiling, the normalised probe values and the painting-level results, and C3's wording follows them.

**R15: Define the multiplicity family** (REV-28; SINGLE-VERIFIER)
- **Problem**: §10 does not say whether K2 is an intersection test over datasets and backbones or a claim on any subset, how the per-aspect-type 'beats' and 'matches' declarations enter the family, or which raw metric-from-pairs baseline counts as 'the best' before the final read. R1 cites this gap as decision-bearing for D1.
- **Source**: R1 W11
- **Requirement**: Declare the family: an intersection-union rule for 'every dataset' and Holm across backbones and aspect types for anything weaker; fix 'the best' baseline on development.
- **Acceptance criteria**: §10 names the family, the correction or intersection rule and the development-chosen baseline.

**R16: Specify the power calculation** (REV-29; SINGLE-VERIFIER)
- **Problem**: The power calculation has no target effect, alpha, power or resampling unit, and computed on anchor-level selection variance it overstates power under item reuse; beyond the item pool extra anchors add little (SemArt has 1,069 test paintings). R1 cites this gap as decision-bearing for D1.
- **Source**: R1 W12
- **Requirement**: Specify the minimal effect of interest and the equivalence margin, compute power by simulation under the cluster bootstrap, and report item counts as well as anchor counts.
- **Acceptance criteria**: The E13 pre-registration states effect, alpha, power, unit and item counts per dataset.

**R17: An outcome-independent criterion for subjective and objective aspects** (REV-35; SPLIT)
- **Problem**: K3's split into subjective and objective aspects has no outcome-independent criterion, and §11 places the per-aspect-type declaration at E13 (Oct 21 to 23), after E10's naming comparisons on development (Oct 12 to 20), so the assignment can be fitted to the outcomes. R2 W5 and R3 W3 (Major), R1 W18 (Minor, on the timing); SPLIT on severity, arbitrated Major. DA M3 also.
- **Source**: R1 W18, R2 W5.1, R3 W3.1, DA M3.2
- **Requirement**: Fix, before E10, each aspect's assignment from a criterion that does not depend on model results (for example inter-annotator agreement, or nameability measured as zero-shot name-prompt accuracy), and keep it whatever E10 shows.
- **Acceptance criteria**: The plan lists each aspect's type and the criterion, recorded before E10 runs.

**R18: K5's bar uses the wrong GeneCIS column** (REV-37; SINGLE-VERIFIER)
- **Problem**: K5's bar quotes four-task average R@1 (SEARLE 14.4, CIReVL 15.9, OSrCIR 17.4) for an evaluation that runs only focus attribute; the same rows' focus-attribute column in the plan's own Table 2 reads 18.9, 17.9 and 19.4, and STiTch (21.1 focus attribute) is left out, so the bar is 2.0 to 4.5 points too low. DA M4 also.
- **Source**: R2 W1.1, DA M4.1
- **Requirement**: Restate K5 against the focus-attribute column including STiTch, and report SQUARE and DIOR as out-of-class references.
- **Acceptance criteria**: §5.4 quotes focus-attribute values with sources, and K5 names its threshold against them.

**R19: Restore the per-episode probe and the Tip-Adapter cache** (REV-38; SINGLE-VERIFIER)
- **Problem**: §8 also drops the literature review's rank-2 Tip-Adapter cache and rank-3 linear probe on the eight support and contrast pairs, although the probe was the strongest scorer on the earlier episodes (24.10 against SE's 21.22).
- **Source**: R2 W2.2
- **Requirement**: Restore the per-episode linear probe and the Tip-Adapter cache in Tier 1.
- **Acceptance criteria**: §8 Tier 1 lists both baselines.

**R20: K7 needs a sparse-autoencoder basis** (REV-39; SINGLE-VERIFIER)
- **Problem**: K7 compares the learned basis only with PCA, NMF and SpLiCE, leaving out sparse autoencoders, the dominant unsupervised sparse code on CLIP since 2024 (the plan's Table 5 lists several, including the shared-dictionary MGSAE), and the split-dictionary comparison that the literature review's prepared answer relies on is not planned. DA M5 makes the split-dictionary point too.
- **Source**: R2 W3, DA M5.1
- **Requirement**: Add a TopK SAE trained jointly on CLIP image and caption features of the scorer-train rows, with sparsity matched to the factor codes (optionally the MGSAE recipe), run the agreement rule on its latents in the K7 table, and add the split-dictionary ablation to E11.
- **Acceptance criteria**: The K7 table and E11 include the SAE basis and the split-dictionary ablation.

**R21: 'No labels' understates GoEmotions distant supervision** (REV-40; SPLIT)
- **Problem**: 'No labels from the evaluation taxonomy' understates distant supervision: the emotion pseudo-partition is k-means over a GoEmotions classifier whose label set names 6 of the 8 evaluation emotions. R2 W4 (Major) and R3 W7 (Minor); SPLIT on severity, arbitrated Major because the overlap is confirmed by the manuscript's own literature review.
- **Source**: R2 W4.1, R3 W7
- **Requirement**: Word the claim as 'no ArtELingo labels; distant supervision from a GoEmotions classifier whose label set overlaps 6 of the 8 evaluation emotions', report emotion results separately for the six covered and the two uncovered categories, and keep the teacher-only text-side analysis next to the main emotion row.
- **Acceptance criteria**: C2 and §5.1 state the overlap, and the emotion table splits covered and uncovered categories.

**R22: ArtELingo partitions are hand-matched to the evaluation aspects** (REV-41; SINGLE-VERIFIER)
- **Problem**: Each ArtELingo training pseudo-partition is chosen to resemble one evaluation aspect (emotion-like, style-like with AMI 0.32, genre-like), so aspect selection on ArtELingo is weaker evidence of label-free learning than 'pseudo-aspects' suggests; CUB, with per-sentence caption clusters, is the cleaner test. DA C1 builds its critical on the same premise (separate item).
- **Source**: R2 W4.2
- **Requirement**: State that the ArtELingo partitions were chosen to resemble the evaluation aspects, report the emotion aspect with and without the affect partition, and lead the label-free argument with CUB.
- **Acceptance criteria**: §6 discloses the matching, results include the run without the affect partition, and the label-free argument cites CUB ahead of ArtELingo.

**R23: Held-out-aspect test for the mechanism claim (DA C1, VALIDATED)** (REV-55; DA-CRITICAL)
- **Problem**: DA C1, adjudicated VALIDATED: every ArtELingo evaluation aspect has a training pseudo-partition chosen to resemble it (one built on a supervised emotion classifier), CUB's partitions are part-focused caption clusters, and no cross-modal dataset holds an aspect out of training under a decision rule; the planned R-pseudo tests (cross-dataset and unseen-species results) keep the same aspects, so the design cannot tell example-conditioned inference of a respect from a K-way selector among trained aspect subspaces. Repairable (block_class repairable).
- **Source**: DA C1
- **Requirement**: Add a leave-one-aspect-out run on ArtELingo (for example train without the caption-content partition and test genre episodes) and generic, attribute-agnostic partitions on CUB, each with a decision rule fixed in E13 against the raw metric-from-pairs baseline; optionally a partition bank larger than the evaluation aspects built without looking at them. If no held-out-aspect test is run, narrow C2 and the C1 motivation to selecting among aspects represented in the training partitions.
- **Acceptance criteria**: §6 and §10 list the held-out-aspect test or tests with a pre-registered rule, or C1 and C2 are reworded to the selector reading.

### Suggested Revisions (Should Fix and Consider)

S1 to S35 are listed with full metadata in `revision_roadmap.md` (19 should_fix, 16 consider).

## Journal-Supplied Deadline (Optional Transport)

- **Exact deadline from source letter**: NOT PROVIDED. No deadline, duration or work estimate is inferred. The plan's own internal dates are the authors' schedule, not a review deadline.

## Response Letter Instructions

Please respond to every roadmap item, R1 to R23 and S1 to S35, in the format of `templates/revision_response_template.md`. The response must include:
1. a response and revision description for each Required Revision;
2. a response for each Suggested Revision, adopted or with the reason for not adopting it;
3. change markup in the revised plan;
4. a cross-reference table from roadmap item to revised section or block.

Declining an item is allowed; the decline stays visible and unresolved in the author-adjudication sidecar. For DA C1 a response is required even if you dispute it.

## Closing

We encourage you to consider the reviewers' comments carefully and to submit a substantially revised plan. The revised plan will undergo another round of review.

---

## Part 3: Reviewer Report Summary (Appendix)

### Journal-Fit Review Report Summary
- Scores: D5 warn, D6 warn | Confidence: 3 to 4 per finding
- Key point: the topic fits CVPR and the combination is claimed honestly, but the contribution is under-argued. The GO rule can pass a condition-blind gain, K3 meets adverse evidence with no failure branch, K5 rests on a stretch protocol, the novelty search is incomplete, in-context alternatives are missing, and the NO-GO fallback is not yet a paper.

### Reviewer 1 (Methodology) Summary
- Scores: D1 block (repairable), D3 warn | Confidence: 3 to 5 per finding
- Key point: the held-out hygiene is strong, but three design gaps (a condition-blind pass, an anchor-only bootstrap, no equivalence margin) leave claims testable only in a weaker sense; all can be repaired at E13. No arithmetic receipt is a mismatch; the inconsistencies found are in prose (W17).

### Reviewer 2 (Domain) Summary
- Scores: D2 warn | Confidence: 3 to 4 per finding
- Key point: the prior-art scholarship is strong, but K3, K5, K7, C3 and the C2 supervision wording are stated beyond the evidence (wrong GeneCIS column, no SAE basis, the GoEmotions overlap), and §8 drops the example-in-context baselines the plan's own appendices rank as must-have.

### Reviewer 3 (Perspective) Summary
- Scores: D4 warn | Confidence: 4 of 5 overall
- Key point: the aspect and value constructs are sound, but the emotion ground truth has no rater agreement or noise ceiling, C3 confounds modality with elicitation and viewer disagreement, K3 conflates subjectivity with nameability, and the third aspect is uncontrolled.

### Devil's Advocate Summary
- Recommendation: N/A, findings only
- Key challenge: C1, VALIDATED. Without a held-out-aspect test the design cannot tell example-conditioned similarity from a K-way selector among trained aspect subspaces; repairable.


## Attachment: Acronym Check (advisory, #849)

### Acronym check (advisory; no reply needed)
Coverage: body (partial)
Not in this input: English abstract, Chinese abstract.
Not checked:
- Body, line 527: POLAR (the initials before its parentheses do not spell it)
- Body, line 533: CSN (the initials before its parentheses do not spell it)
- Body, line 580: mAP (the initials before its parentheses do not spell it)
- Body, line 671: CLEVR4 (the initials before its parentheses do not spell it)
- Body, line 1234: CPU (the initials before its parentheses do not spell it)
- Body, line 1324: MARS (the initials before its parentheses do not spell it)
- Body, line 1432: PE (the initials before its parentheses do not spell it)
- Body, line 1448: CUB (the initials before its parentheses do not spell it)

| Scope | Line | Rule | Acronym | Uses |
|---|---|---|---|---|
| Body | 9 | Not defined | CVPR | 13 |
| Body | 9 | Not defined | CoSiR | 11 |
| Body | 46 | Not defined | DAS6 | 1 |
| Body | 46 | Not defined | GPU | 8 |
| Body | 48 | Not defined | CLI | 1 |
| Body | 49 | Not defined | RTX | 2 |
| Body | 94 | Not defined | CLIP | 75 |
| Body | 94 | Not defined | ViT | 17 |
| Body | 174 | Not defined | CLAY | 10 |
| Body | 174 | Not defined | CRL | 8 |
| Body | 181 | Not defined | KISSME | 16 |
| Body | 237 | Not defined | COCO | 6 |
| Body | 243 | Not defined | ArtGAN | 2 |
| Body | 276 | Not defined | CIReVL | 8 |
| Body | 276 | Not defined | OSrCIR | 9 |
| Body | 276 | Not defined | SEARLE | 8 |
| Body | 279 | Not defined | CC3M | 2 |
| Body | 289 | Not defined | ReLU | 7 |
| Body | 318 | Not defined | GO | 6 |
| Body | 324 | Not defined | NO | 2 |
| Body | 333 | Not defined | VL | 13 |
| Body | 380 | Not defined | SHA | 3 |
| Body | 520 | Not defined | API | 3 |
| Body | 520 | Not defined | HTML | 2 |
| Body | 520 | Not defined | TEVI | 3 |
| Body | 527 | Not defined | ASIF | 4 |
| Body | 527 | Not defined | GME | 3 |
| Body | 527 | Not defined | LiT | 3 |
| Body | 527 | Not defined | SAIL | 3 |
| Body | 527 | Not defined | SCE | 7 |
| Body | 549 | Not defined | IC | 1 |
| Body | 549 | Not defined | ICLR | 2 |
| Body | 549 | Not defined | KL | 1 |
| Body | 549 | Not defined | LLM | 2 |
| Body | 549 | Not defined | TC | 1 |
| Body | 549 | Not defined | UT | 2 |
| Body | 553 | Not defined | CIRCO | 1 |
| Body | 553 | Not defined | CIRR | 1 |
| Body | 553 | Not defined | ZS | 4 |
| Body | 576 | Not defined | CoLLM | 1 |
| Body | 576 | Not defined | CoVR | 1 |
| Body | 576 | Not defined | DeCIR | 1 |
| Body | 576 | Not defined | GENIUS | 1 |
| Body | 576 | Not defined | ISA | 1 |
| Body | 576 | Not defined | KED | 1 |
| Body | 576 | Not defined | LDRE | 1 |
| Body | 576 | Not defined | LinCIR | 2 |
| Body | 576 | Not defined | LoRA | 2 |
| Body | 576 | Not defined | MLLM | 3 |
| Body | 576 | Not defined | MM | 3 |
| Body | 576 | Not defined | PACT | 2 |
| Body | 576 | Not defined | RTD | 1 |
| Body | 576 | Not defined | SEIZE | 1 |
| Body | 576 | Not defined | SQUARE | 2 |
| Body | 576 | Not defined | STiTch | 1 |
| Body | 576 | Not defined | UniIR | 1 |
| Body | 578 | Not defined | FoCo | 1 |
| Body | 578 | Not defined | README | 1 |
| Body | 580 | Not defined | CORE | 1 |
| Body | 580 | Not defined | EVAL | 1 |
| Body | 580 | Not defined | FLUX | 1 |
| Body | 580 | Not defined | MCMR | 1 |
| Body | 606 | Not defined | FSIR | 2 |
| Body | 606 | Not defined | GCRDP | 1 |
| Body | 649 | Not defined | SAE | 1 |
| Body | 671 | Not defined | DARN | 1 |
| Body | 671 | Not defined | HQ | 2 |
| Body | 675 | Not defined | MMEB | 4 |
| Body | 675 | Not defined | VQA | 1 |
| Body | 703 | Not defined | LamRA | 2 |
| Body | 703 | Not defined | UniME | 1 |
| Body | 705 | Not defined | DINOv2 | 2 |
| Body | 705 | Not defined | MLP | 1 |
| Body | 727 | Not defined | RN50x4 | 1 |
| Body | 752 | Not defined | MGSAE | 1 |
| Body | 757 | Not defined | SigLIP | 10 |
| Body | 758 | Not defined | TPIPS | 1 |
| Body | 1250 | Not defined | ASEN | 1 |
| Body | 1251 | Not defined | HTTP | 1 |
| Body | 1251 | Not defined | JMLR | 1 |
| Body | 1257 | Not defined | BGE | 8 |
| Body | 1257 | Not defined | EN | 8 |
| Body | 1257 | Not defined | ICL | 8 |
| Body | 1257 | Not defined | ITML | 3 |
| Body | 1257 | Not defined | RCA | 9 |
| Body | 1257 | Not defined | RICE | 8 |
| Body | 1312 | Not defined | VASR | 4 |
| Body | 1316 | Not defined | MAP | 1 |
| Body | 1320 | Not defined | AAAI | 1 |
| Body | 1322 | Not defined | AIR | 1 |
| Body | 1322 | Not defined | BEIR | 1 |
| Body | 1322 | Not defined | HyDE | 1 |
| Body | 1322 | Not defined | MTEB | 1 |
| Body | 1322 | Not defined | MuCo | 1 |
| Body | 1322 | Not defined | QA | 1 |
| Body | 1322 | Not defined | nDCG | 1 |
| Body | 1324 | Not defined | MRR | 1 |
| Body | 1324 | Not defined | MarT | 1 |
| Body | 1334 | Not defined | HOI | 1 |
| Body | 1338 | Not defined | CCA | 1 |
| Body | 1486 | Not defined | JSON | 2 |
| Body | 1527 | Not defined | OMP | 1 |
