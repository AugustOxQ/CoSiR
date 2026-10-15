# Rule check: round 4 decision rule (draft, before commit)

Written 2026-10-07 01:23 (Amsterdam) by a fresh Opus reviewer. Checked: `../DECISION_RULE.md` (uncommitted draft, 450
lines, SHA-256 54bc3007…2e99 at the time of the check) against the spec (7f0a00c), round 3's rule, rule check and final
review, the handoff, and the code the rule names. My script `rc4_seed42_check.py` (output `rc4_seed42_check.json`,
gitignored) read only seed-42 arrays and round 3's seed-42 bundle, on CPU. It computed no number of V4, V2 or V24: no
fused or counterpart score, cell, margin, Δ_k or candidate gate or open count. No seed was built and no test seed was
touched. I wrote nothing outside `rule_check/`, and no `__pycache__` was created.

## Verdict: FIX BEFORE COMMIT (1 blocking, 7 should-fix, 14 nits)

All 17 SHA-256s of D11 match. So do the spec's, the 36 inputs of round 3's `r3_common.INPUTS` and all 30 rows of round
3's D15 table. Every seed-42 constant and regression target reproduces at full precision through the float32 path:
v₇₅, the 9,216 count, B′(A1), the A1 reader against `cand_R1_A1.npz`, R1, AFF and IMGABST_q75. Item 3 has no float32
against float64 near-tie. The blocking item is a call the rule prescribes that cannot run on the test seeds.

## Blocking

**B1. §4 item 1 (line 194): `r3_bundle.build_bundle(s, smoke)` refuses seeds 52, 53 and 54.** `r3_bundle.py:133-136`
(`_check_seed`, called first in `build_bundle`) admits only `(42,) + r3_common.TEST_SEEDS` = (42, 49, 50, 51) and, in
smoke mode, 9001 to 9003. I called `_check_seed` directly, which reads no data. Seeds 52, 53 and 54 each raise
`ValueError: episode seed 52 (smoke False) is not one of this round's seeds (42, 49, 50, 51)`, while 9001 (smoke) and 42
pass. So the wiring smoke test of §10 passes. The failure first appears in the GO pass, after the seeds are built, and
§9 then says "stop … nothing is improvised". *Add to §4 item 1, after the bundle sentence:* "Round 3's seed guard
(`r3_bundle._check_seed`, which reads `r3_common.TEST_SEEDS` when it is called) admits only 42, 49 to 51 and the smoke
seeds. Before its first bundle call, this round's code sets `r3_common.TEST_SEEDS = (52, 53, 54)` in its own process.
No file of round 3 changes, and no other round-3 function that reads `TEST_SEEDS` or `EARLIER_SEEDS`
(`run_r3_build.py`, `r3_apply_rule.py`) is called. A unit test shows that the guard then admits 42, 52, 53 and 54 and
refuses 55."

## Should-fix

**S1. Lines 9-13 (precedence): the rule depends on round-3 sections that, as written, are "cited for provenance only".**
Only "D2 to D12 and its §4 pipeline" are part of the rule by reference. The rule also relies on these round-3 sections:
- §2 (metrics, bootstrap);
- D1 and D7 (line 86);
- D15, including its text: `told_oracle.json`, and `results/build_seed{s}.json` asserted by every later step;
- §5 items 1 to 3 (lines 218, 224);
- §6.1 (the formula, line 276) and §6.2 (provenance check, crash handling, lines 287-288);
- §6.11 (line 337), §7 item 7 (line 358), §9 (line 415) and §10.

*Replace "(its definitions D2 to D12 and its §4 pipeline, unchanged)" with:* "(every section of it that this file cites,
with the text of that section unchanged unless this file says otherwise: §2, D1 to D12, D15 with its text, §4, §5 items
1 to 3, §6.1, §6.2, §6.11, §7 item 7, §9 and §10). In particular, D15's build records `results/build_seed{s}.json` are
written for seeds 52 to 54 and asserted by every later step. Round 3's D1 sentence 'The csd grouping and configuration
A1 are not used in this round' does not apply."

**S2. §5 order (lines 214-216), item 5 (252-254), §10 test (430-431): "number of V4, V2 or V24" is undefined, and item 5
contradicts it.** Item 5 is one of the five checks. It computes the candidates' gates and open counts, which are numbers
of the candidates, before "all five pass", and the order test needs a precise target. *Replace the second sentence of
§5 with:* "No *candidate result* of V4, V2 or V24 is computed, written or printed before items 1 to 5 pass. A candidate
result is a score, per-anchor array, chosen cell, R@1, margin, gain, either rate, bar margin or Δ_k. Item 5 computes the
candidates' gates and their τ_0 open counts, which are gate statistics and not candidate results, once items 1 to 4 have
passed." *In §10, replace "no candidate number" with* "no candidate result (§5)".

**S3. §8 phase 1 (lines 374-376): the re-derivation can compute candidate results before the regression checks pass.**
Phase 1 "may start once this file is committed" and includes every candidate's development numbers. Nothing orders it
after §5 items 1 to 5. In round 3 this was harmless, because phase 1's AFF numbers were the rule's own. Here they are
new. *Replace "Phase 1 (may start once this file is committed; its result is needed before the carry is acted on)"
with:* "Phase 1 (may start once this file is committed). It computes its own §5 items 1 to 5 first, and it computes no
candidate result before those match this file's targets and the pipeline's record shows items 1 to 5 passed. Its result
is needed before §6.1 runs and before a kill is reported. Its x is compared after step (5) and before step (6)."

**S4. §4 items 3 and 4 (lines 200-203) against §6.4 (lines 296-302): non-carried candidates on the test seeds.** §6.4
permits "§4 items 1 to 4" before the verdict, and §4 item 4 computes "each candidate's gates". The same paragraph then
says the candidates that were not carried "are never computed on a test seed". Separately, the N11 assertion (line 305)
computes τ_0 open counts, while gate shares come "only after the verdict" (line 303). *Add to §4 item 4:* "On a test
seed, only AFF's gates and the carried candidate's gates are computed (the A1 reader in item 3 only if the carried
candidate reads CSD)." *Add to §6.4's last sentence:* "The open counts this assertion uses are held in memory and are
not written or printed before the verdict."

**S5. §10 tests (lines 428-433) cover only part of round 3's final review.** The rule's list covers M06, M09, M10, M20,
T3-1 and T3-5. It leaves out:
- **M27**, the descriptive pass's check that the cache reproduces the GO pass (`run_r3_test.py:296`);
- **M31**, the GO pass re-running the build records' episode-hash cross check;
- **M11**, which has no unit test: the GO pass must not cross-fit AFF's counterpart;
- **M28**, pick ties to the first grouping. This is decision-relevant now: V2 reads π_A1, and seed 42 has 0 exact A1
  arg-max ties, so the data never exercise the rule;
- **T2-M4**, a bar-comparator test where the per-seed, pooled and per-pair comparators differ. D8's four-way order is
  new code;
- **T3-10**, the agreement record bound to the SHA-256 of the GO pass's `go_pooled.json`;
- **T1-1**, the bundle guards that were never triggered.

The rule also leaves open what "each family sees its own τ_0 open counts" asserts.
*Replace the bullet's list with:* "… the GO-pass assertions of §6.4, each shown to fire under a mutation. (a) The
clusters equal `groups[anchor]`. (b) Before each family call, the gates passed in equal, at every τ index and in both
conditions, the gates recomputed from that method's D6 definition and the seed's margins, picks, A1 picks and v. (c) The
build records' file and episode-hash checks are re-run (M31). (d) AFF's counterpart is not cross-fitted before the
verdict (M11). (e) The descriptive pass checks that its cached arrays reproduce the GO pass's (M27). Also: a test that the
seed-42 runner writes and prints no candidate result before §5 items 1 to 5 pass; a criterion test for ρ_ctrl (M06);
tie tests for π and π_A1 with constructed exact ties (M28); a bar-comparator test with D8's four-way order in which the
per-seed, pooled and per-pair choices differ (T2-M4); tests of the bundle guards (head identity, finiteness, parity,
selection; T1-1); the boundary-reported path and the phase-2 agreement record each bound to the SHA-256 of the current
`go_pooled.json` (T3-1, T3-10); and a leak check in the wiring smoke test that catches any decimal number, one-decimal
numbers included."

**S6. §6.11 (lines 336-342): the multiplicity disclosure miscounts and leaves out the closest earlier variants.**
"Seven detectors" fits neither file: `bs_07_detector.json` holds 7 (6 label-free and the oracle SUP18), and
`bs_08_csd_evidence.json` holds 4 (3 label-free and the oracle SUP24). Two related reads are missing:
- `bs_05_aff.py` read four A1-reader variants (`A1.*`: "R1/A1 (all)", "A1 affect only", "A1 affect + csd", "A1 csd
  only"). "A1 affect only" is the nearest earlier relative of V2.
- `bs_11_visual_side.py` read `AFF_and_VIS`: AFF's affect pick AND Δ_image^c < 0, with the term z(s_affect), +0.553
  against B′(A0). That is a visual veto on AFF's pick, the nearest earlier relative of V4.

*Replace "(the brainstorm's … seven detectors)" with:* "(the brainstorm's `bs_04_readers.py`: four abstention variants;
`bs_07_detector.py` and `bs_08_csd_evidence.py`: nine label-free detectors and two declared oracles; `bs_05_aff.py`: four
variants of the A1 reader's own gate; `bs_11_visual_side.py`: a visual veto on AFF's affect pick, `AFF_and_VIS`)".

**S7. §5 items 2, 3 and 5 (lines 229-234, 252-253): the regression path does not exercise the candidates' factors.**
Item 2 sets both factors to 1. Item 3 multiplies R1's gates by a_v, but nothing says it uses the function that builds
V4's gate, so a parallel implementation would pass. No item checks the V2 factor against an independent source, and item
5's algebra still passes if π_A1 is taken from the wrong condition. *Append to item 3's first sentence:* "computed by the
same gate-factor function that builds V4's gate, with R1's gates in place of AFF's". *Append to item 5:* "V2's gate
equals AFF's gate times 1[`pick__c` = 0], with `pick__c` from `cand_R1_A1.npz` per condition (an independent source of
π_A1). V4's gate equals AFF's gate times 1[v < 0.021043562795966864], with v from the A0 features of round 3's bundle."

## Nits

- **N1.** Prior, line 40: "the same abstention moved the in-sample best fused cell by +0.045". From `bs_04_readers.json`
  `in_sample`, the best fused cell moved from 19.1162109375 to 19.142659505208332 (+0.026). The +0.045 is the in-sample
  margin, best fused minus best counterpart: 0.537109375 to 0.581868489583332. This is the brainstorm's "the margins
  moved by +0.045". Write "moved the in-sample margin (best fused minus best counterpart cell) by +0.045".
- **N2.** Line 25, "the spec approved at about 01:05": the log says 01:01, and 7f0a00c was committed at 01:02:40. Write
  "approved and committed at 01:02 (7f0a00c)".
- **N3.** D1, line 86: round 3's D1 also says "The csd grouping and configuration A1 are not used in this round". Exclude
  that sentence (the S1 text does).
- **N4.** D8, lines 145-149: the spec's bar comparator for V2 and V24 is "the strongest of the three" (B, floor,
  counterpart). The rule adds B′(A0). Say it is a refinement: it can only raise the bar, and it changes nothing on seed
  42, where B′(A1) is 18.804931640625 and B′(A0) 18.436686197916664.
- **N5.** D6 and D7: state that every gate is a float32 0/1 array per condition, as `r3_fusion.gates_aff` returns. A
  float64 or other gate would make `rc_core.gated_terms` promote the product, so the term would leave D8's float32 path.
- **N6.** §5 item 4, lines 248-250: confirmed. `run_r2_fusion.round1_probs` (lines 78-92) averages round 1's two A1
  half-readers' `predict_proba` on `rb_eval.seed42_features(load_bundle(), A1)`, which is D3's definition. The values
  are equal exactly (maximum absolute difference 0.0), and the picks are the stored int8 picks. Replace the hedge with
  "(confirmed by the rule check)". Seed 42 has no exact arg-max tie in P_A1 under either condition.
- **N7.** §6.1 and §6.7: for the AFF check, x is projected from a single seed-42 pair of cross-fits. When the two
  methods' cells agree on seed 42 but differ on a test seed, the realised variance is larger. In round 3 the analogous
  paired check (AFF minus R1) realised a half-width of 0.108 against a projected 0.091 (+18%), while the other checks
  came within 5% (final review §2.3). Add to §6.7: "For the AFF check, x may understate the margin needed: …".
- **N8.** §7 item 3: the spec asks for "the share of AFF's open values each veto closes". If V24 is carried, also report
  the share for each factor.
- **N9.** §7 item 4, line 361: if the carried candidate's pooled bar comparator is its own counterpart, say whether the
  control is compared with the candidate's counterpart or with its own. Suggest: "the same condition-free scorer as the
  candidate's pooled bar comparator, or the control's own counterpart when that comparator is the counterpart".
- **N10.** §9, line 413: define "matches or beats" and "B′(A1) is above AFF". Suggest: the control's pooled bar-margin
  point is at or above the candidate's, or the candidate-minus-control interval contains 0; and the pooled mean R@1 of
  B′(A1) is at or above AFF's. Round 3's final review (S2) applied the row in this sense.
- **N11.** §8 order: phase 1's x is listed at step (4), but x is computed at step (5). S3's text fixes this.
- **N12.** §10, lines 440-442: "(or the candidate's gate passed without its veto factor)" lets the weaker mutation stand
  in for the AFF-check swap. Require all three mutations.
- **N13.** D11: add provenance rows for the sources that §6.11 and item 4 cite:
  - `20261118_reader_fix_round2/run_r2_fusion.py` fe6fb9ff870c3942bef98327ff23894782522e516fc100cb1635aeec6e7e5b65
  - `20261120_r1_levers_brainstorm/bs_07_detector.py` 1b287bcc725270af9b3b0a0b41b9021d9ed3ea844ef404df52ca4f4e9e47088a
  - `…/results/bs_07_detector.json` e9d52462a2c128b39cb5a2fe7b78b425e5ceac982b091c334d99d5d951a485e1
  - `…/bs_11_visual_side.py` cb27cd636bae5f92ba016602148ad2508ed8a50867fe4c6b3b0c7c7885704ad2
  - `…/results/bs_11_visual_side.json` a1b51be306750af10a2634488fdf775ce096588e36ac143c61194630282dc6c7
- **N14.** D9 and item 8: the code should use the integer literal 24 and integer sums, never `floor(0.05*4*12288/100)`.
  In float64 that expression is 24.576000000000004, which happens to be harmless. Δ_k is best summed from
  `r2_fusion.as_int4` of the per-anchor R@1, which also asserts that every value is a multiple of 0.25.

## Statistical and logical points checked (no finding)

- **D9 and item 8.** 0.05 pp × 4·12,288 / 100 = 24.576. A gap of 24 is 0.048828125 pp (≤ 0.05) and a gap of 25 is
  0.0508626302 pp (> 0.05), so "M − Δ_k ≤ 24" is exactly the spec's "within 0.05", in the right direction. The carry
  compares integers only, so no float comparison can game it. Ties among E fall to the order V4, V2, V24. If two
  candidates are identical, the earlier wins; a candidate identical to AFF has Δ = 0 and is not in E.
- **D7 and run_family.** `r3_fusion.run_family(bundle, T, gates)` reads only `bundle.B`, `bundle.parity`, T and four
  gate dicts. It has no AFF-specific assertion, and `fused_only=True` gives §6.4's AFF arm. The AFF-specific code is
  `r3_stats.go_checks` (seven checks plus R1), `bar_info_pooled` (a three-way comparator) and round 3's runners, so
  round 4 needs its own versions (S5, T2-M4). σ* depends on B and parity only, so AFF and the candidate share it on
  every seed.
- **The AFF check.** It is well defined when many d_k are 0. The percentile cluster bootstrap needs only that the
  non-zero values spread over many paintings. A candidate identical to AFF on a seed gives d ≡ 0, a lower bound of 0,
  and the reading "did not beat AFF". The bar comparator for V2 and V24 when B′(A0) > B′(A1) is defined by the maximum
  and the tie order. "AFF frozen as tested" means its definition is frozen while its cross-fit is rerun per seed (glossary
  and §6.3), and AFF needs no σ* of its own.
- **D5.** The neighbouring v values sit 1.25e-6 below and 3.75e-6 above v₇₅, far outside the 1e-9 relative agreement
  tolerance. No v equals v₇₅.
- **Spec against rule.** Candidates, floors, development bar, carry, kill, the build and hash list, freezes, the order
  of computation, the GO lists (8 and 9 checks), the readings, the claim, the descriptive pass, the disclosure and the
  process all match the spec. The rule's additions (B′(A0) in V2's bar set, the smoke seeds, the boundary items) are
  refinements: N4, S6, N8.

## Verified (values at full precision)

- **SHA-256s (`sha256sum -c`).**
  - All 17 D11 rows match: round 3's rule 2d311dbe…5925 (also equal to the committed HEAD version), r3_common,
    r3_bundle, r3_fusion, r3_stats, rb_reader_A1.{pkl,json,npz}, run_step1.py, step1_heads_style.npz,
    cand_R1_A1.{npz,json}, bs_08_csd_evidence.{json,py} and baselines_seed{49,50,51}.json.
  - The spec matches d1584d8c…7d5b, and `git show 7f0a00c:` gives the same.
  - The 36 inputs of `r3_common.INPUTS` and the 30 rows of round 3's D15 table match, including `bs_04_readers.json`
    42bf6f20…691e and `codes_provenance.json` 8e6a517b…bfaf.
  - `common.INPUT_FILES` hashes `step1_group_style.npz` (b04d96b4…20e2) and `step1_heads_style.npz`, as D11 says.
- **D5.**
  - v = min(feature 6, feature 7) of condition a is float64.
  - S_img^b = C_img^a and C_img^b = S_img^a exactly, so v^b = v^a exactly.
  - `numpy.percentile(v, 75)` = 0.021043562795966864, equal to the rule's literal (its repr round-trips). 9,216 of
    12,288 episodes lie below it and none equal it.
  - The bundle's v equals the cache's.
- **D3.**
  - `rb_eval.seed42_features(SimpleNamespace(ctx, post with csd), A1)` on a fresh round-3 seed-42 bundle gives 24
    float64 columns, equal exactly to the cache's `feat_A1__a/b`.
  - The first 18 columns equal round 3's A0 features exactly, in both conditions.
  - The A1 `grouping_stack` equals the cache's `stack4`.
  - The A1 pickle has feature names equal to `rf.feature_names(A1)`, scikit-learn 1.6.1; the environment runs numpy
    2.2.6 and scikit-learn 1.6.1.
- **D4.**
  - `uniform_probe_scores` gives identical T_6u (float32) with the key "csd" and with step 1's key "style_csd".
  - `crossfit_condition_free(cos, T_N1u, T_6u(A1), parity)` reproduces round 1's `pBp["A1"]` per-anchor arrays
    exactly (all 5 metrics).
  - Mean R@1 = 18.804931640625, equal to the rule; it is 0.3682454427083357 above B′(A0).
- **§5 item 4.** P_A1 equals `probs__a/b` exactly and π_A1 equals `pick__a/b` (int8) exactly. The stored picks are the
  arg max of the stored probabilities. The anchors, parity and pair index are aligned.
- **§5 item 2 (r3_fusion float32).**
  - R1: T, margins and picks equal the stored values. Cells 116/119 and 58/123, σ* 0/0 (ρ_ctrl 4,532 and 4,483).
    Bar margin 0.4435221354166667 [0.21646171563312194, 0.6735669710776852] against the counterpart. Gain statistic
    2.667236328125 [2.325087836946873, 3.012361650695922]. The 10 per-anchor arrays and `bar_v` are bit-identical.
  - AFF: fused 19.136555989583336, counterpart 18.39599609375, comparator B′(A0). Bar margin 0.6998697916666667
    [0.4598852740816973, 0.9371680126852968]. Gain statistic 3.110758463541667 [2.780005709854805,
    3.4559584315470384]. Cells 39/119 and 149/10, σ* 0/0, τ_0 open 9,941 and 3,627. AFF minus R1 fused
    0.21769205729166666 [0.06425880757348419, 0.3709597330984391].
  - Gates times a factor of 1 equal AFF's gates exactly.
- **§5 item 3 (R1 × a_v through `run_family`, float32), equal to `bs_04_readers.json` IMGABST_q75 and to the rule.**
  - Fused 19.059244791666664 and counterpart 18.49365234375; the comparator is the counterpart.
  - Bar margin = margin = 0.5655924479166667 [0.3448683992591827, 0.79821625538382].
  - Gain statistic 2.878824869791667 [2.554983173204304, 3.2105685950938248]; either −1.7476399739583333.
  - Per pair: 0.677490234375, 1.28173828125, −0.262451171875.
  - Cells: the brainstorm's [2, 0, 4, 4] and [2, 0, 16, 16] convert to 117 and 119, and [1, 0, 0.5] and [1, 0.5, 1]
    to 58 and 67, by (t·7+u)·8+a with NESTED_U and NESTED_A. σ* 0/0.
  - Float32 against float64 (the brainstorm's tied family, cell by cell): 0 differing entries in all 224 × 12,288 of
    fused R@1, fused gain and counterpart R@1.
  - Tie sets at the maxima: fused h0 {117, 134} (147 against 146), h1 {119} (268 against 261); counterpart h0 {58, 75}
    (4,617), h1 {67, 84} (4,515). The lowest cell wins, as the brainstorm recorded.
- **§6.11.** IMGABST_q75 (+0.5656) is the best of the four abstention variants (CSDABST q50 +0.4781, q75 +0.4801, q90
  +0.3723). No script outside the brainstorm and this check computed an abstention or A1-pick variant, and no earlier
  file holds a V4, V2 or V24 number.
- **Other.**
  - The arithmetic of the prior holds: 18.805 + 0.5 = 19.305 is about +0.17 above 19.137.
  - Seeds 52 and later are free in the ledger. The 8 earlier seeds of §6.2 are every seed with a `baselines_seed*.json`
    (44 and 46 were MLLM probes without one).
  - `run_baselines.py --smoke` overwrites the existing 9001 to 9003 smoke files without `--overwrite`.
  - The `.gitignore` is identical to round 3's.
  - 7 October 2026 is a Wednesday, and so is 14 October.

Files: `rule_check/rc4_seed42_check.py` and `rule_check/rc4_seed42_check.json` (9 KB). Nothing over 1 GB was written.
