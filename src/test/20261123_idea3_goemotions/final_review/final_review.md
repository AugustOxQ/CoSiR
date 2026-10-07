# Round 5 (idea 3: GoEmotions placement of captions on AFF): whole-branch final review

**Verdict: CONFIRMED WITH FIXES.** The kill is true. My own third derivation reproduces every decision quantity of seed
42 exactly: the GE head (86.17%, 139 iterations, no fallback), Q_GE bit for bit, stack_G, F_G, B′_G, both candidates'
readers, τ′, gates, cells, σ*, per-anchor arrays, bar comparators, D10 clauses, Δ_k (−169, −134) and the empty carry
set. It also reproduces the measured diagnostics (a) to (d) and every load-bearing number of the report's §6. The fixes
are two report passages. One is a why-section verdict that its own numbers contradict. The other is a set of timing
statements left stale by the run-log correction 4a1a22c. Neither changes the kill.

**Counts:** 0 blocking, 2 should-fix, 8 nits.

Reviewer: Opus (fresh context), 2026-10-07 19:02 to 19:40 (Amsterdam). Binding rule `DECISION_RULE.md` (SHA-256
19e59fc7…735e, asserted by my scripts). CPU only (`CUDA_VISIBLE_DEVICES=`, 8 threads, `PYTHONDONTWRITEBYTECODE=1` on
every call), at most 2 processes. I wrote only under `final_review/` and the session scratch folder
(`…/scratchpad/fr5/`). I made no git write and dispatched no subagent. I built nothing of seeds 52 to 54 and opened no
build log.

## 1. The third derivation

**Code and imports** (`fr5_derive.py`, 106 s, 19:11 to 19:13). It imports only what rule §8 allows a re-derivation:

- round 1's `common.load_bundle()` (seed 42: episodes, parity, clusters, cosine, the CLIP affect heads through
  `run_told_oracle.fit_one_head`, the image and caption posteriors, B, B′(A0), B′(A1), the method-A term);
- `rb_build.load_readers("A0")`, `zscore_rows`, `crossfit_condition_free`, `uniform_probe_scores`, `cluster_bootstrap`;
- `src.data.artelingo_splits` (the aspect labels for diagnostic (c)), `src.data.affect` and `src.data.artelingo` (the
  spot check).

`rb_build` itself imports `rb_features` (transitively). An import-time assertion refuses any `r5_*`, `rd*`, `r2_*` to
`r4_*`, `rb_eval`, `bs_*` or `run_r*` module. My own code covers:

- the GE head: draw, check rows, labels, fit with scikit-learn's `LogisticRegression` called directly, the fallback
  rule, scatter and accuracy;
- the 18 features, P, T, margins, picks, τ and τ′, every gate, the z-scored terms, G_cf;
- the 224 cells' integer statistics, σ*, both cross-fits, the parity assembly and the per-anchor metrics;
- the comparators and bar comparator, margins, gain statistics, either changes, per-pair bar margins, D10, Δ_k and
  the carry;
- diagnostics (a) to (d), and the §6 numbers (pair lifts, sharpness, redundancy, the affect score alone, the
  same-painting agreement, B′ weights, per-cell net rankings, the fixed-cell decomposition, the in-sample family, G-TF's
  reader against AFF's).

Results were written and hashed before any comparison (`out/fr5_results.json` 4f13f12c…, `out/fr5_arrays.npz`
f114887a…, 19:13; the comparison ran at 19:14).

**Checked first, before any GE number:** round 1's stored R-c T, margins and picks; τ recomputed equal to `rc_tau.json`
and to the rule; σ* 0 / 0 (ρ_ctrl 4,532 and 4,483). Then R1 (cells 116, 119 / 58, 123) and AFF (cells 39, 119 / 149,
10) reproduced every target of rule §5 item 1 exactly, together with all of round 4's `r1_*` and `aff_*` arrays. My
`crossfit_condition_free` call on the CLIP placement reproduced B′(A0)'s per-anchor arrays exactly, and the six D7
redundancy values matched the rule exactly.

**Comparison** (`fr5_compare.py` → `out/fr5_agreement.json`). There were 976 comparisons under rule §8's tolerances
(discrete and arrays identical; τ, AUCs and ratios within 1e-9 relative; pp within 1e-9). **None failed.**

| Compared against | Discrete | pp values | τ / AUC / ratios | Arrays | Failed | Largest difference |
|---|---|---|---|---|---|---|
| the rule's stated targets (§5 item 1, D7, item 4's 1.1445 and 0.7871, D8 beside) | 13 | 33 | 10 | 0 | 0 | 0 |
| round 4's `seed42_arrays.npz` (`r1_*`, `aff_*`) | 0 | 0 | 0 | 28 | 0 | 0 |
| the implementation (`placement.json`, the GE file, `dev_seed42.json`, `carry.json`, every key of `seed42_arrays.npz` but cosine and RCA, `diagnostics_seed42.json`) | 60 | 438 | 15 | 83 | 0 | 5.6e-17 pp (Δ_k point from integers against the bootstrap point) |
| the re-derivation (`rd5_stageB.json`, `rd5_stageB_arrays.npz`) | 2 | 14 | 4 | 30 | 0 | 0 |
| the report's §6 rebuild (`why_rebuild.json`) | 53 | 114 | 71 | 6 | 0 | 2.2e-16 (same-row ratio) |
| orientation: AFF's per-direction margin against `bs_09_direction.json` | 0 | 2 | 0 | 0 | 0 | 4.7e-15 pp (the brainstorm's float64 path) |
| **total** | **128** | **601** | **100** | **147** | **0** | |

The seed-42 decision, three times identical:

| Seed 42 | G-T | G-TF | AFF (reference) |
|---|---|---|---|
| fused / counterpart R@1 | 18.792724609375 / 18.328857421875 | 18.863932291666664 / 18.235270182291664 | 19.136555989583336 / 18.39599609375 |
| fused cells; counterpart cells; σ* | 46, 117; 101, 25; 0, 0 | 12, 117; 67, 164; 0, 0 | 39, 119; 149, 10; 0, 0 |
| bar comparator (means: B′_G / B′(A0) / cf / B) | B′(A0) (18.3980 / 18.4367 / 18.3289 / 18.3411) | B′(A0) (18.3980 / 18.4367 / 18.2353 / 18.3411) | B′(A0) |
| bar margin | 0.35603841145833337 [0.12398419582712294, 0.5829741018096795] | 0.42724609375 [0.18483149832466372, 0.666759228901012] | 0.6998697916666667 [0.4598852740816973, 0.9371680126852968] |
| gain statistic | 2.530924479166667 [2.232038210094649, 2.8370856032504883] | 2.872721354166667 [2.558691997060608, 3.1995123231983036] | 3.110758463541667 […] |
| D10 c1 / c2 / c3 | **no** / yes / yes | **no** / yes / yes | yes / yes / yes |
| Δ_k (integer); point [95%] | −169; −0.3438313802083333 [−0.5222166846883708, −0.16297832514540103] | −134; −0.2726236979166667 [−0.4565247245743068, −0.09238454151969994] | |

E = ∅, M = none, nothing tied, nothing carried: **kill**. No rule §8 boundary applies. Two more checks:

- **Ties in the cross-fits.** Three of the four candidate fused picks were ties at the maximum criterion: G-T half 1
  (cells 117 and 134), G-TF half 0 (12 and 29) and G-TF half 1 (117 and 134). Each tie is between rank-equivalent cells
  with the same relative weight λ_a / (1 + λ_u) (4 and 4; 1.33 and 1.33). Every alternative gives the same fused R@1
  and the same Δ_k (`out/fr5_extra.json`), so the tie rule changes nothing, including §6.5's reading of the weights.
- **The GoEmotions file.** My own CPU spot check (`fr5_goemo_spot.py`) used 512 selection rows drawn with rng 11, a
  sample neither the implementation (rng 5, scorer-train) nor the re-derivation (rng 6) used. The largest difference was
  3.43e-7 against the 1e-4 tolerance. The same captions scored against the next row's stored probabilities differ by
  0.963, so the check would catch a misalignment.

**`build_figures.py`** ran on a scratch copy with only `ROOT` patched. It passed all its assertions, and its
`figure_data.json` is byte-identical to the stored one. The figures carry titles that state the finding, labelled axes
and legends; Figure 6's circles sit at the chosen cells' λ_a (16, 8, 2 on τ_0; 16, 4, 4 on τ_2).

## 2. The report, claim by claim

Every number in the Summary, Tables S1, 1 to 11 and the text of §4 to §6 matches a file or my derivation at the
precision shown:

- Tables 3, 4 and 6 checked programmatically against `dev_seed42.json` and `diagnostics_seed42.json`;
- Tables 7 to 10 and the §6 text against my own §6 computations;
- the integer sums 9,406 / 9,237 / 9,272 / 9,243 / 9,062 / 9,043 / 9,015; the better and worse counts 814 / 967 and
  864 / 979;
- B′_G against B′(A0) (−19 rankings; per pair −0.110, −0.037, +0.031) and B′_G against B (+0.057 [−0.072, +0.192]);
- the B′ weights; G-TF's pick agreement (95.0%, 92.3 to 98.1 per cell, affect share −0.6 to −1.9 points; open as
  scored 8,473 / 2,311 against 8,623 / 2,448; 535 against 585);
- the same-painting ratios 1.260 / 1.283 / 1.350 (75,210 pairs) and the sharpness numbers.

I also checked:

- the earlier-round citations: round 3's +0.591 [+0.462, +0.729] (`test_verdict.txt`), round 4's 00:17 to 02:36, and
  round 4 §6.5 on the detector;
- the 361 regression comparisons (184 / 20 / 21 / 136, all pass) and the re-derivation's 492 (222 / 142 / 24 / 104,
  largest difference 0);
- the rule check's 67 hashes. I re-verified all **38** D12 rows of the committed rule (all match).

Every disclosure the brief lists is present (§7), with the costs of all 19 ledger rulings in §8.4:

- development data;
- the 2026-09-22 percept-branch pilot;
- GoEmotions at inference;
- that captions explain the annotator's emotion;
- CPU because MultiMAE (pid 959679) held the GPU;
- the `.pyc` lapse;
- Task 3's in-place mutation window (16:27 to 16:33);
- the `_HEADS` shape bug that two reviews missed.

Every headline number has its baseline beside it.

The why-section's hypotheses get these verdicts:

- **H1** (the image side caps the agreement): supported by direct measurement. The image × caption lift moved
  1.145 → 1.207, against 1.656 → 2.575 on captions alone.
- **H3** ("consistent, within noise"): fair.
- **H2** and **H4**: see N2 and N3.
- **H5**: its verdict does not follow from its numbers (S1).

### Blocking

None.

### Should-fix

**S1. §6.5 and Table 11 answer the spec's question wrongly. The sharper term does need less weight for its gain; what
fails is that the gain is not cheaper.** Location: §6.5, the last two sentences of "No weight closes the gap"; Table
11, row 5.

The report says: "So the answer to the spec's question "does the sharper term need less weight for its gain?" is no:
lighter weights cost the GE term less either rate, but its family's ceiling is lower at every weight". Table 11 marks
hypothesis 5 **not supported**. The report's own numbers say the opposite on the first half:

- At AFF's cells, G-T's term bought +0.822 [+0.547, +1.100] pp more gain than AFF's (Table 10).
- In sample (my 224-cell statistics, equal to `why_rebuild.json`'s `in_sample_family`), G-T reaches about 3 pp of
  condition gain at about half of AFF's relative weight λ_a / (1 + λ_u). On τ_0 that is 1.78 (cell 55, gain 3.06)
  against 3.2 (cell 39, gain 3.02); on τ_2 it is 2 (cell 116, gain 2.96) against 4 (cells 117 and 134, gain 3.02).

What fails is the brainstorm's premise that less weight means less either cost. At that matched gain G-T's either rate
is lower: 34.76 against 35.44 on τ_0 and 34.93 against 35.51 on τ_2. Its fused R@1 is therefore about 0.3 lower:
18.91 against 19.23, and 18.95 against 19.27.

Second, "its family's ceiling is lower at every weight" is false at λ_a = 0.25 on τ_0. The best-λ_u curves there are
18.618 (G-T) and 18.630 (G-TF) against AFF's 18.612, and all three are equal at λ_a = 0. The same paragraph already
says they are "within 0.04 pp" at 0.25. This is a paper-facing conclusion about why idea 3 failed, so it should be
right.

Replace the paragraph's last sentence ("So the answer … at every weight.") with:

> So the answer to the spec's question "does the sharper term need less weight for its gain?" is yes, but the lighter
> weight did not make the gain cheaper. In sample, G-T reaches about 3 pp of condition gain at about half of AFF's
> relative term weight (1.78 against 3.2 at τ_0; 2 against 4 at τ_2), yet at that gain its either rate is about 0.6 pp
> lower (34.76 against 35.44; 34.93 against 35.51), so its fused R@1 is about 0.3 pp lower (18.91 against 19.23;
> 18.95 against 19.27). From λ_a = 0.5 up the GE families stay below AFF's at every weight; at λ_a 0.25 on τ_0 they
> sit 0.006 (G-T) and 0.018 (G-TF) above it.

Replace Table 11 row 5 with:

> | 5. The sharper term needs less weight for its gain, and so pays less either rate (the brainstorm's §3.3) | chosen
> cells, fixed-cell decomposition, in-sample family (Table 10, Figure 6) | **first half supported, second not**: at
> AFF's cells +0.82 gain, and in sample about half the relative weight for a 3 pp gain; but at AFF's cells −1.42
> either, and in sample 0.6 pp less either at matched gain, so a family ceiling about 0.3 pp lower from λ_a = 2 up |

Optionally add to the Summary's third bullet, after "So the GE term cost more either rate per unit of gain.": "It
needed less weight for its gain, but each unit of gain cost more either rate."

**S2. Timing statements left stale by the run-log correction, and a missing commit.** Location: the header's commit
list, Table 1, §8.3 and §8.4 ("Small timing differences…").

At 19:01 the controller corrected the run log (4a1a22c). The placement step was launched at 17:48, and phase 1 ran
from 17:49 to 17:50 (it had first been written as 17:55, an estimate). The draft still says:

- Table 1: "17:55 | re-derivation phase 1 finished";
- §8.3: "the log recorded only its results file's SHA-256 at 17:55";
- §8.4: "the run log puts the placement step's launch at 17:50 … while the run log records phase 1 as finished at 17:55
  (the controller's note time)".

All three now contradict the committed log. The header's commit list also omits 4a1a22c. The ledger still says
"~17:55", which is the source of the old estimate. Fixes:

- Table 1: replace the 17:55 row with "| 17:49 to 17:50 | re-derivation phase 1 (stage B) ran; its numbers withheld
  until the implementation's `carry.json` existed |". Move it above the 17:50 placement row or merge the two.
- §8.3: replace "(withheld from the ledger and the log until `carry.json` existed; the log recorded only its results
  file's SHA-256 at 17:55)" with "(withheld from the ledger and the log until `carry.json` existed; the log recorded
  only that phase 1 had finished, 17:49 to 17:50, and its results file's SHA-256)".
- §8.4: replace the "Small timing differences…" bullet with: "The controller first logged the placement step's
  launch as 17:50 and phase 1's finish as 17:55 (an estimate; the ledger keeps '~17:55'). The report draft found the
  mismatch with the step's own log (item 3 passing at 17:49:43) and the re-derivation's report (17:49 to 17:50). The
  log was corrected at 19:01 (4a1a22c) to 17:48 and 17:49 to 17:50. No order or number depends on it."
- Header: after "345fbb0 (re-derivation, its agreement and the kill in the log)" add ", 4a1a22c (run-log time
  corrections)".

### Nits

- **N1** (§6.2 heading and Table 11, row 2): "supported" for hypothesis 2 rests on descriptive association. The overlap
  with B and cosine fell by a third, and p_A and p_B were ranked first less often. No variant isolates the lost
  similarity as the cause of the R@1 loss. Lower redundancy with B is, on its own, what round 3's D7 called a virtue.
  Replace "**supported** for similarity across paintings" with "**consistent** (descriptive) with lost similarity
  across paintings", and the §6.2 heading's "supported in part" with "consistent in part".
- **N2** (§6.4, hypothesis 4): for G-T, "the losses sit where steering happens" holds by construction. Its gates as
  scored are AFF's, and I checked that no ranking of G-T differs from AFF's where the as-scored gate is shut (0 of the
  shut rankings). The informative part is the asymmetry per open value. Add after "…only on values where steering
  happens.": "(checked on seed 42: no ranking differs where the gate is shut). Per open value, condition a lost 178
  rankings on 8,623 open values and condition b gained 9 on 2,448." In Table 11, row 4, say "supported (by construction
  for G-T; the content is the a/b asymmetry)".
- **N3** (§6 intro): rule §5 says "No other variant is computed on seed 42". The fixed-cell assemblies and the
  "affect score alone" rankings are scorings outside the pre-registered family. They were computed after the kill and
  decide nothing, which the intro already says. Add one sentence: "They are scorings outside the rule's family, made
  after the kill for explanation only; rule §5's 'no other variant' sentence governs the development step, which they
  do not touch."
- **N4** (§8.1): "It verified the SHA-256 of all 36 rows of D12 as drafted." The committed D12 has 38 rows (N5 added
  `run_gonogo.py` and `wikiart_genre.py`). Add: "(the committed rule has 38; `r5_common` asserts all 38, and the final
  review re-verified them)".
- **N5** (terms): "CSD" (§1, B′(A1) row), "R-c" (Table 2, item 1) and "the method-A factor term" (§1, B row) are
  undefined. Use round 4's wording:
  - CSD: "the csd style grouping (Leiden communities of the paintings' CSD embeddings; CSD, Contrastive Style
    Descriptors, is a pretrained style-embedding model)";
  - R-c: "round 1 called this scorer R-c, and 'round-1 R-c' means its stored arrays";
  - method-A term: "the centred factor term of the method-A checkpoint, a factor model trained on scorer-train rows".
- **N6** (§9, "Our view"): it is marked as ours, but two sentences claim more than one recipe on development data shows.
  Replace "A better caption placement cannot fix an agreement whose other factor…" with "We doubt a better caption
  placement can fix an agreement whose other factor…". Replace "which supports reporting AFF's gain as coming from
  where it steers" with "which, on development data, is consistent with AFF's gain coming from where it steers".
- **N7** (§8.2, the deferred minors): add one sentence: "One of the twenty (Task 0: `clip_from_bundle` raising
  KeyError on an odd `_HEADS` shape) was closed by Task 0's second fix round. The final review's mutation run confirmed
  that two of Task 4's minors (the cell descriptions, and the order of `sharper_term`'s AFF difference) survive every
  test. Neither changed a number of this run, and both must be fixed before this code runs on a test seed (§8.5)."
- **N8** (tests, before reuse; no effect on this run): three mutations survive the 403 list-A tests (§3).
  - G3 removes `require()`'s second check that a `clip` placement does not carry the GE array. It fails safe: the two
    mint-time checks in `clip_from_bundle` are pinned (G7 and G8 killed), and forging a Placement is refused by test.
    It needs a test that mints a clip object around the GE array through `r5_guard._MINT`.
  - T1 swaps fused and counterpart picks in `dev_record`'s `cell_text` (deferred minor T4-1). This run's `cell_text`
    equals the cells (my comparison), so no number moved. Fix it with a test where fpick ≠ cpick.
  - T3 swaps the arguments of `minus()` at `sharper_term`'s call site (deferred minor T4-2). My derivation equals
    every minus_aff value of `diagnostics_seed42.json`, so Table 6 is right. Fix it with a test where AFF's family
    differs from the candidate's.

## 3. Mutation table

Each mutation ran on a scratch copy (modules, runners, tests, rule, run log, `cache/`, `results/`) in
`…/scratchpad/fr5/mut/<id>/`, deleted afterwards. Only `r5_common`'s `ROOT` was patched. The tests were the list-A
files relevant to the site, with pytest `-x`. The unmutated copy passed all 403 tests in 178 s. Logs are in
`out/mutations/`, the summary in `out/mutations.json`. **38 of 41 mutations killed; 3 survive.**

| Guard | Mutation | Result (first failing test) |
|---|---|---|
| D11 guard | G1 `require()` never refuses a ge placement before release | killed (`test_require_refuses_ge_before_release_and_accepts_clip`) |
| | G2 `release()` accepts any record with the rule's SHA | killed (`test_release_refuses_bad_records`) |
| | G3 `require()`'s clip branch accepts the GE array | **survives** (redundant; fails safe, see N8) |
| | G4 `require_carry()` accepts a missing `carry.json` (diagnostics before the carry) | killed (`test_require_carry`) |
| | G5 items marked out of order; G6 items 3 and 4 swapped | killed (`test_order_refuses_an_item_out_of_order…`, `test_items_are_the_rules_items_in_the_rules_order`) |
| | G7 `clip_from_bundle` without its GE-fingerprint check; G8 without the by-value check against `_HEADS` | killed (`test_placements_are_minted_only_by_the_two_constructors`, `test_clip_from_bundle_needs_an_equal_cached_head`) |
| D10 at its thresholds | D1 clause 1 strict; D2 clause 2 `>=`; D3 clause 3 reads the bar; D4 threshold 0.4; D5 boundary band 0 | all killed (`test_d10_point_half_passes_and_flags`, `…_bar_lower_bound_exactly_zero…`, `…_clause3_reads_gain_statistic_not_bar`, `…_point_049_fails…`, `…_flag_within_1e12_only`) |
| Δ_k | K1 sign reversed; K2 from the counterpart; K3 Δ_k = 0 not flagged | all killed (`test_delta_k_from_integers`, `test_dev_record_identical_to_aff_has_delta_zero_flag`) |
| Carry band | C1 band exclusive; C2 Δ_k = 0 in E; C3 last tied carried; C4 E ignores D10; C5 band 25; C6 gap 24 not flagged | all killed (`test_carry_tie_band_24_inclusive_25_exclusive`, `test_carry_delta_zero_not_in_e`, `test_carry_requires_all_clauses_recomputed_not_stored`) |
| G-T / G-TF wiring | W1 G-T reads F_G; W2 G-T's term on the CLIP stack; W3 G-TF reads F | all killed (`test_g_t_reads_p_m_pi_from_F_and_T_from_stack_G`, `test_g_tf_reads_p_and_T_from_F_G_and_stack_G`) |
| | W4 G-TF uses AFF's τ on seed 42 | killed (`test_tau_prime_on_seed_42_is_computed_from_g_tf_margins`) |
| | W5 family from AFF's gates; W6 family from AFF's term | killed (`test_counterpart_is_built_from_the_candidates_own_term_and_gates`) |
| | W7 D7's gate check never fires | killed (`test_a_swap_of_the_reader_input_or_the_term_stack_fires_d7`) |
| D5 positive check, no-assign | P1 to P3 each check vacuous; P4 the runner ignores it; P5 `post_q` assigns into `bundle.post` | all killed (`test_an_extension_that_ignores_its_q_fails_the_positive_check`, `test_a_failed_positive_check_stops_before_any_development_number`, `test_post_q_is_a_new_dict_in_a0_order`) |
| Item 1 / item 4 pins | N1 pins never differ; N2 the real run adds no pin row | killed (`test_a_missing_or_extra_name_stops_the_real_run_before_release`) |
| Agreement-line gate | A1 a "pending" line opens it; A2 any carry SHA opens it | killed (`test_the_agreement_record_has_one_exact_format[pending]`, `[wrong_sha]`) |
| Deferred-minor probes | T1 `cell_text` swaps fused and counterpart picks | **survives** (deferred minor T4-1; N8) |
| | T2 `r5_diag.minus` sign reversed | killed (`test_minus_sign_is_candidate_minus_aff`) |
| | T3 `sharper_term` calls `minus(aff, candidate)` (arguments swapped at the call site) | **survives** (deferred minor T4-2; N8) |

## 4. Deferred-minor triage

The ledger has 20 `minor (deferred)` lines and no `parked` line. None must be fixed before the report is committed:
the third derivation confirms every number they could touch. "Before reuse" means before this code runs on a test seed
or in a later round.

| Minor (ledger) | Triage |
|---|---|
| T0 M3: `require` re-fingerprints the whole Q on every call | Leave (speed only; the real run took 2 min) |
| T0 M5: brittle `len(INPUTS) == 38` test | Leave (D12 changes only with the user; the test fails loudly) |
| T0: `_GE_FPS` process-global, only grows | Leave (it only refuses more) |
| T0: `clip_from_bundle` raises KeyError on an odd `_HEADS` shape | **Closed** by Task 0's fix round 2 (`_cached_txt_equals` returns False, then GuardError; G8 pinned) |
| T0: 2 scipy deprecation warnings; 35 s test without a slow marker | Leave |
| T1: `N_SCORER_TRAIN` local to `r5_goemo` | Leave (one-shot step done; the file is frozen) |
| T1: npz written before json | Leave (one-shot step done; the record exists) |
| T2: `test_misaligned_mapping_refused` mostly tests the non-finite guard | Leave for this run: the real draw mapping is confirmed by item 3 and by my Q_GE, bit for bit. Before reuse |
| T2: item 3 checks `fit_one_head`'s draw SHA, not `place`'s own | Leave (posterior equality implies it; my draw SHA equals 7be956c0…) |
| T2: GE npz before `placement.json` can orphan an npz | Leave (one-shot step done) |
| T2: `placement_failure.json` not in the early refusal list | Leave |
| T4: `cell_text` / `chosen_cells` fused vs counterpart picks untested | **Confirmed by survivor T1.** No effect here (my comparison of the cells). Fix before reuse |
| T4: `sharper_term`'s minus_aff sign not pinned | **Confirmed by survivor T3** (the call-site order; `minus()` itself is pinned, T2 killed). No effect here: my derivation equals every minus_aff value of `diagnostics_seed42.json`. Fix before reuse |
| T4: `pair_lift`'s Pi not bound to a placement | Leave (the runner always passes the CLIP image head; (c) confirmed) |
| T4: carry copies stored boundaries, discards recomputed D10 flags | Leave (the stored boundaries come from the same `d10`; before reuse compute both from `k10`) |
| T4: reassembly key-set check's missing-metric direction untested | Leave; before reuse |
| T3: `load_ext` does not cross-check a cached bundle's record against `r4_shas` | Before reuse (per-seed caches exist only for test seeds) |
| T3: `_check_q`'s dtype clause not pinned separately | Leave (the Placement constructor enforces float32) |
| T3: `extend`'s checks hold `F_Q_computed` False with `features=False` | Leave |
| T5: agreement regex not anchored to the `\| HH:MM \|` frame | Before reuse (it guards only a carried candidate's sensitivity path; A1 and A2 are killed) |

## 5. What else I verified

- **Seeds and ledgers.** No `episodes_seed5{2,3,4}`, `per_anchor_seed5*` or `baselines_seed5*` file exists. The smoke
  files 9001 to 9003 date from 2026-10-06. `codes_provenance.json` is still 8e6a517b…bfaf. The seed ledger last
  changed in round 3 (2f8c784; "52 and later | free"), and the held ledger has been unchanged since a6eba60. This
  round's `results/` has no `build_seed*`, `sensitivity.json` or `boundary_seed42.json`.
- **Rule, spec and code.** The rule (19e59fc7…) and the spec are unchanged since 750e06f and 393c3c2. The GoEmotions
  and placement code did not change after their one-shot runs (`git diff e6bd95a 7542d36`). Work-tree tracked files are
  clean.
- **Dry runs.** `run_r5_seed42_dry*.log` and `seed42_dry.json` hold no decimal number (a regex check) and record a stop
  at the guard (359 comparisons, `guard_closed` true).
- **Process order** (rule §8), confirmed from the log, the ledger and git:
  1. list A, 373 passing, at 17:39 to 17:41, before the GoEmotions step (17:42);
  2. the GoEmotions step and its constant (e6bd95a, 17:48);
  3. the placement step and its constant (ad7fc3d, 17:50);
  4. list A again, 403 passing, then the real run at 7542d36 (18:23 to 18:25), the import check (18:27), the agreement
     (18:34) and the kill (345fbb0).

  Phase 1 was blind: its numbers stayed out of the ledger and the log until `carry.json` existed. Its imports conform
  to rule §8 (I read every import line of `rd5_*.py`).
- **`.pyc`.** None from this round remains in a read-only folder. Round 1's `__pycache__` holds only `rc_core.pyc` of
  2026-10-06. Round 4's folder still holds five `.pyc` files from round 4's own run (01:54 to 02:25, round 4's N11).
  They are harmless, and not this round's.
- **Timestamps.** Every time in the log, the ledger and the draft is Amsterdam local time with no offset. The commit
  times quoted in the draft match `git log` in Amsterdam time.
- **Storage left behind**, all gitignored except the `.py` and `.md` files below, nothing near 1 GB:
  - `cache/` 9.3 MB, `results/` 2.0 MB, `rederive/results/` 11 MB;
  - the report assets 1.3 MB;
  - `final_review/out/` 27 MB (mostly `fr5_arrays.npz`, 27.6 MB);
  - the scratch folder 1.3 MB (the `build_figures.py` copy; the mutation copies were deleted).

  The untracked `.claude/` hook-state folders in this folder and in `rederive/` (40 KB) are not this round's; the
  ledger notes them.

## 6. Files written (all under `src/test/20261123_idea3_goemotions/final_review/`)

| File | What |
|---|---|
| `final_review.md` | this review |
| `fr5_derive.py` | the third derivation (own code) |
| `fr5_compare.py` | the 976 comparisons (rule targets, implementation, round 4 arrays, re-derivation, §6 rebuild) |
| `fr5_goemo_spot.py` | the CPU spot check of the GoEmotions file (512 rows, rng 11) and its misalignment control |
| `fr5_extra.py` | the tied cross-fit picks and Δ_k under each alternative |
| `fr5_mutate.py` | the mutation harness (scratch copies, deleted after each run) |
| `out/fr5_results.json`, `out/fr5_arrays.npz` | derivation results (4f13f12c…, f114887a…) |
| `out/fr5_agreement.json`, `out/fr5_extra.json`, `out/fr5_goemo_spot.json` | comparisons, ties, spot check |
| `out/fr5_derive.log`, `out/fr5_goemo_spot.log`, `out/mut_*.log` | run logs |
| `out/mutations.json`, `out/mutations/*.{log,json}` | mutation results (41 mutations and the baseline) |

## 7. Re-review of the fix wave (2026-10-07 19:45 to 19:52, Amsterdam)

Scope: S1, S2, N1 to N8 and the filled §8.5 of the uncommitted report, plus the test commit 2668601. The fixer's two
departures from my proposal were re-derived from my own arrays (`out/fr5_results.json`).

- **S1 ADDRESSED.** §6.5 now reads: "So the answer … is yes, but the lighter weight did not make the gain cheaper …
  (1.78 against 3.2 at τ_0, cells 47 and 39; 2 against 4 at τ_2, cells 116 and 117)". Table 11 row 5 reads "**first
  half supported, second not** … from λ_a = 4 up", and the Summary has the added sentence.
  - The fixer is right and my "cell 55" was a slip. Cell 47 is λ_u 8, λ_a 16 (weight 1.778, gain 3.064, either 34.758,
    R@1 18.911). Cell 55 is λ_u 16, λ_a 16 (gain 1.770). Cells 116 (G-T: 2.962, 34.933, 18.947) and 117 (AFF: 3.023,
    35.506, 19.265) are G-T against AFF on τ_2, as cited.
  - "From λ_a = 4 up" is the correct tightening: the gaps are 0.18 to 0.22 at λ_a 2 and 0.295 to 0.342 at 4 to 16 on
    τ_0 and τ_2.
  - G-TF at λ_a 0.25 on τ_1 is 0.020 above AFF, as cited. From λ_a 0.5 up both GE families are below AFF on all four
    τ indices.
- **S2 ADDRESSED.** Table 1 has "17:48 to 17:50 | placement step" and "17:49 to 17:50 | re-derivation phase 1 (stage B)
  ran". §8.3 has "finished, 17:49 to 17:50". §8.4 explains the 17:55 estimate and 4a1a22c. The header lists 4a1a22c and
  2668601. No other "17:55" remains.
- **N1 ADDRESSED.** The §6.2 heading reads "consistent in part"; Table 11 row 2 reads "**consistent** (descriptive)"; and
  "Our view" reads "seems to have been useful partly".
- **N2 ADDRESSED.** §6.4 has "(the final review checked this on seed 42 …) … condition a lost 178 rankings on 8,623 open
  values and condition b gained 9 on 2,448". Table 11 row 4 says "by construction for G-T".
- **N3 ADDRESSED.** The §6 introduction has "scorings outside the rule's family, made after the kill for explanation
  only".
- **N4 ADDRESSED.** §8.1 has "(the committed rule has 38; `r5_common` asserts all 38, and the final review re-verified
  them)".
- **N5 ADDRESSED.** §1 defines R-c (in the R1 row), the method-A factor term (in the B row) and CSD (in the B′ row).
- **N6 ADDRESSED.** §9 has "We doubt a better caption placement can fix …" and "on development data, is consistent
  with".
- **N7 ADDRESSED.** §8.2 has "One of the twenty (Task 0 …) was closed … two of Task 4's minors … survived every test …
  the fix wave added tests that kill both".
- **N8 ADDRESSED.** On fresh scratch copies at 2668601 (full env prefix), the unmutated suite passes **407** tests
  (403 + 4 new). Each survivor now dies under its new test:
  - G3 by `test_require_refuses_a_minted_clip_carrying_the_ge_array`;
  - T1 by `test_dev_record_cell_text_keeps_fused_and_counterpart_picks_apart`;
  - T3 by `test_sharper_term_minus_aff_is_candidate_minus_aff`.

  The fourth new test, `test_chosen_cells_keep_fused_and_counterpart_picks_apart`, exists.
- **§8.5 is consistent with this review.** It gives:
  - the verdict, the counts (0 / 2 / 8) and 976 comparisons (128 / 601 / 100 / 147);
  - the largest differences, 4.7e-15 and 5.6e-17;
  - the ties, the spot check (3.43e-7 and 0.963) and 38 of 41 mutations killed, with the three survivors named;
  - the 20 deferred minors with no parked item: one closed earlier, two closed now, five before reuse.

  These match §1 to §5 above.

**New problems introduced by the fix:** none that changes a number or overclaims. One wording nit: §8.5's findings
table describes N7 as "the deferred-minor triage was missing". N7 actually asked §8.2 to name the minor already closed
and the two minors the mutations confirmed. Suggested cell: "§8.2 did not say which deferred minors were closed or
confirmed by the mutation run".

**Verdict: all findings addressed.**
