# Final review A2b: area A2 finished (held runners, C5 order, ArtELingo-28 episodes and bundle, descriptive numbers)

Reviewer A2b (Opus 5.5), snapshot /project/CoSiR-wt/r9-fr-a2b at 8db7ceb (r9-scope, detached), 2026-10-14 03:05.
Starts from A2's partial report (`/project/CoSiR-wt/r9-fr-a/src/test/20261202_r9_scope/final_review/fr_a2.md`, at
4458634). Between 4458634 and 8db7ceb, only `run_r9_held.py` and its tests changed in my scope (ticket 22's launch
gate and the `inputs_only` path of `setup_a28`). The descriptive, episode and bundle files did not change, so A2's
line numbers and mutant plan still apply.
Scripts and outputs in this folder: `fr_a2b_c5.py` (A2's C5 script, rerun here), `fr_a2b_smoke_h8.py`,
`fr_a2b_mut_bundle.py` and their `.json` files, and `fr_a2b_probe_tests.py` (F2's test and F3's
parts a and e, written out and tried). Temporary files are in the session scratchpad `fr_a2b/`.

## 1. Verdict

**CONFIRMED WITH FIXES.** The C5 order holds: no held or ArtELingo-28 number can be computed before that read's own
verdict. A2's finding F1 (MEDIUM) and F2 (LOW) are still present at 8db7ceb. With F1 open, rule §8 cannot run on any
real read. I add F3 (LOW): five descriptive checks that nothing tests and that no other check backs up.
F4 (LOW, process) is a gap in A2's sweep, not in the code: the five r9_bundle "survivors" are caught by
`test_r9_context.py`, which the sweep did not run. The CLIP H8 smoke pass equals r9_stats recomputed on its own
arrays. All fixes are outside the 17 closure modules.

## 2. Findings

**F1 MEDIUM (A2's, still present). The descriptive runner refuses every real verdict, so rule §8 cannot run on a real read.**
`run_r9_descriptive.py:75-77` (`bind_read`) requires `verdict['backbone'] == RH.READ_BACKBONES[read]`, the code name
("qwen3vl_emb_2b"). `run_r9_apply_rule.py:623` writes `READS[read]["backbone"]`, the ledger name
("Qwen3-VL-Embedding-2B"). I reran A2's script here (`fr_a2b_c5.py` → `fr_a2b_c5.json`, identical to A2's output
apart from temp paths). On verdicts that the real `apply` wrote for H6 and H8, `bind_read` refuses with "verdict
identity". H9 and H10 also fail inside `a28_report`'s `bind_read(H8)` (:341) and `paired_reads` (:301).
The failure is fail-safe (nothing leaks), but it blocks rule §8 items 1 to 6 and spec D13, D21 and D27 in real mode.
The smokes miss it because they bind with `smoke=True` (no verdict). The unit test misses it because its fixture
(`test_r9_descriptive.py:41`) builds the verdict with the code name.
- Fix (one line, outside the closure): `verdict.get('backbone') == RH.LEDGER_NAMES[RH.READ_BACKBONES[read]]`. I checked
  that this equals `AR.READS[read]['backbone']` for all five reads.
- Test, written first: in `test_r9_descriptive.py:41`, use `RD.AR.READS[read]["backbone"]`. This fails on the current
  code. Better: build the verdict with `AR.apply` (as `fr_a2b_c5.py` does with `test_r9_apply_rule.stage` and
  `call`) on a staged pass that carries the runner's `read`, `kind` and `backbone` fields, for H6 and H8. Without those
  fields the staged pass stops at "pass read/kind" (shown in `fr_a2b_c5.json`).
- Order: `run_r9_descriptive.py` is hashed in `R.r9_module_shas()`. That hash is recorded in every started and pass
  file and in ticket 22's canonical smoke record. Land this fix (and F3's) **before** the CLIP pre-read on final bytes.
  Otherwise that pre-read must be run again (about 10 to 20 min of CPU).

**F2 LOW (A2's, still present). `score_seed`'s before-verdict guards check only the verdict file's name.**
`r9_score.py:256-269` accepts any existing file whose name fits the mode. `fr_a2b_c5.json` shows six cases where a
wrong file is accepted, including H7's verdict for a held bundle and a null "not run" verdict. There is no live path:
`run_r9_held` calls `score_seed` with PM and the secondary scorers off. `run_r9_descriptive.bind_read` checks status,
read, mode, variant, pass and agreement before it calls `score_seed`.
- Fix (a test only; r9_score is in the closure): `test_bind_read_refuses_a_foreign_or_null_verdict_before_any_array`
  in `test_r9_descriptive.py`. Monkeypatch `RD.D.load_arrays` and `RD.S.score_seed` to raise if they are called. Then
  assert that `bind_read` refuses three cases:
  - H7's verdict copied to `held_verdict_H6.json`: "verdict identity".
  - A verdict with status "not run: switched off" and verdict null: "read has no run verdict".
  - A pass rewritten after the verdict with only its `time` changed: "pass SHA binding" (this case also closes F3a).

  This test covers mutants 151, 152 and 159. I wrote it out in `fr_a2b_probe_tests.py` with F1's code-name
  workaround. It passes at 8db7ceb (6 cases, with F3e's), and it fails on scratch copies with 151, 152, 159 or
  145 deleted. For 151 the failure is "regex did not match": :81 refuses instead, so no array is loaded.

**F3 LOW. Five descriptive checks have no test, and no other check backs them up.** Ticket 23 built the descriptive
runner as plumbing with no unit tests, proven by its smokes. That fits the lean build for most of its checks (see
§3). These five are different: each is the only protection of a C5 binding or of a number in the descriptive tables,
and deleting it leaves every test green.

| Part | Mutant, line | What it alone protects | Test that would catch its deletion |
|---|---|---|---|
| a | 159, `run_r9_descriptive.py:103` | The described pass is byte-identical to the one the verdict decided on (C5 binding). 160 and 161 compare only seeds and counts. | F2's third case |
| b | 191, `:210` | The re-scored primary and raw arrays equal the read's saved arrays. Every primary row of the descriptive tables comes from the re-score. Without this check, a change to the non-closure code (`run_r9_held`, `r9_a28_context`) between the read and the descriptive run passes silently, while `consistency.rebuilt_primary_and_raw_arrays_equal` is hard-coded True. | Move the loop into `r9_descriptive.check_rebuilt(got, arrays, sl, n)`, call it from `score_inputs`, then unit-test it on a synthetic array dict with one value changed by 1e-12 (expect BoundaryError "rebuilt primary/raw array"). Add a case with a NaN. |
| c | 172, `:211` | The per-direction rows (rule §8 item 1) average to the core metrics. | Move it to `r9_descriptive.check_directions(sc, rows)`. Test: `S.score_seed(test_r9_score.make_bundle(), …, include_pm=True, verdict=v6, on_scores=collect)` passes, and the same with one direction row changed raises. |
| d | 139, `:327` | The `cos_092` count. `near_duplicates_a28.json` is not pinned (t21 LOW-5), and only this SHA binds it to the read. 140 checks only `cos_099` and `cos_098` against the pass. | Move lines 326-331 to a helper `near_counts(rec)`. Test: a temp file whose SHA differs from `rec['near_duplicates_source']` raises "near-duplicates source binding". |
| e | 145, `:410` | The stage 0b changed-episode counts (rule §8 item 6). `stage0b.json` is not pinned, and this identity check (rule SHA, `scored`) is its only binding. | No refactor: `RD.stage0b(SimpleNamespace(smoke=True, out=tmp_path))` on a `stage0b.json` with another `rule_sha256`, then with `scored: false`. Expect BoundaryError "stage0b record". |

All of these changes are in `run_r9_descriptive.py`, `r9_descriptive.py` and their test, outside the closure. Same
order as F1: land them before the pre-read on final bytes.

**F4 LOW (process). A2's sweep did not run the tests of the five r9_bundle guards it reported as survivors.**
In `fr_a2_mut.py`, `TESTS["r9_bundle.py"]` lists `test_r9_a28_bundle.py` and `test_r9_held_a28.py`. Each of the
five guards (`fit_rows_sha`, `pm_raw`, `readers`, `cl_groups`, `scores`) has its own deletion test in
`test_r9_context.py:411-510`. I ran A2's own `mutate()` on a scratch copy of the round folder, never on the snapshot
(`fr_a2b_mut_bundle.py` → `fr_a2b_mut_bundle.json`). Each deletion fails its `test_r9_context.py` test: 4 with "DID
NOT RAISE"; `pm_raw` with a numpy.linalg error, which also fails closed. No code fix is needed. A future sweep should
add `test_r9_context.py` to `TESTS["r9_bundle.py"]`.

## 3. Guard-sweep triage (item 1)

The sweep found 92 mutants not caught (90 survived, 2 timed out). The controller reran the 12 of r9_episodes and
r9_bundle with the full test set (`fr_a2_mut_results.json` in r9-fr-a, `"full": true`).
- r9_episodes: all 7 caught (107, 109, 111 to 115; e.g. `test_tampered_file_fires_on_load_and_the_guard_matters`).
- r9_bundle: the 5 survive A2's full set but are caught by `test_r9_context.py` (F4).
- The descriptive files: 80 survivors. 5 are real gaps (F3). The other 75 are acceptable, grouped in the table below.

The "behind" column names the check that still fails closed when the mutant's line is gone.

| Group (IDs; `run_r9_descriptive.py` lines unless noted) | Class | Behind which check, or why plumbing |
|---|---|---|
| Verdict file and status: 150 (:70), 151 (:73) | acceptable; tested by F2's test | 150: `read_json` raises FileNotFoundError (exit 5) and `score_seed`'s `pm_before_verdict`. 151: a not-run verdict has no `pass_file` or `agreement_file`, so :81 refuses. |
| Verdict identity: 152 (:75) | acceptable as a deletion; the check itself is F1 | `load_agreement` (read, pass_file), :90 and :103 |
| Verdict-to-file names and copies: 153 (:81), 154 (:83), 155 (:86), 157 (:92), 158 (:93), 160 (:104), 161 (:105) | acceptable | 153: IndexError at :82. 154: the kind comes from the matched filename. 155, 157, 158: the agreement is found only by the verdict's SHA (:89-90). 160, 161: behind :103 (F3a). |
| Pass identity and CLI: 124 (:53), 125 (:97), 126 (:99), 127 (:101), 146 to 149 (:427-435) | acceptable (plumbing) | Real mode: :103 with :81 (the bytes the verdict decided on, under this read's name). 125: `read_bytes` raises. 147: `AR.load_pass(smoke=True)` refuses held passes. 149: `open('x')` raises FileExistsError (exit 4). 146, 148: operator input; writes are exclusive. |
| Started, arrays and sensitivity bindings: 128 (:117), 129 (:122), 130 (:124), 131 (:136), 132 (:138), 162 (:121), 163 (:131), 164 (:141) | acceptable | 128: IndexError (attempt numbers are unique). 129: :161. 130: `check_pass` (`r9_descriptive.py:74`) and :210. 162: :161 and :210. 163: :182, :188 and :210. 131, 132, 164: provenance only; no descriptive number reads the sensitivity file. |
| Re-score inputs: 133 to 135 (:151-153), 136 (:161), 165 (:163) and 168 (:174) (the same `require` twice), 166 (:168), 169 (:175), 187 (:166) | acceptable | Primary numbers: :210. PM inputs: `R.assert_inputs` (:150) and r9_bundle's tested `fit_rows_sha`. 165 and 168 cover each other. 166: the value sets drive the draws, so :182. 169: `load_read_set`'s pin SHA. 187: `a28_report`'s own `_pinned`. |
| Rebuilt A28 episodes: 188 (:181), 189 (:182), 170 (:188), 171 (:190), 190 (:195) | acceptable | 188, 189 and 170 back each other up, and :210. 171 (cl to groups[anchor]): :210's `cl` equality and r9_bundle's `cl_groups` (tested; mutant 122). 190: smoke only. |
| Paired reads and H5: 173 to 180 (:237-273), 181 (:308), 184 (:375); `r9_descriptive.py` 196 (:174) | acceptable | H5 is the closed round-6 read (no C5 order at stake); its pass is reproduced by `D6.check_pass`. 181 (episode hashes) and 196 (anchor alignment) back each other up. 184: 181 and 196. |
| ArtELingo-28 report: 137 (:324), 138 (:325), 140 (:329), 141 (:334), 142 (:337), 143 (:369), 144 (:399), 182 (:342), 183 (:358) | acceptable | Flags, coverage and read set come through `AD._pinned` (SHA). 138: pins_a28 is SHA-bound to the pass at :163. 140: :327. 141: shape by construction. 142 and 144: :181 (part records hold `set_aside`). 182: 181 and the held runner's `a28_reference` guards. |
| Stage 0b shape: 185 (:416), 186 (:422) | acceptable (plumbing) | Type checks on copied counts, after F3e's identity check |
| `r9_descriptive.py` array loader: 198 (:45), 199 (:49), 200 (:51), 201 (:57), 192 (:52), 204 (:74) | acceptable | 198 to 201: `run_r9_descriptive.py:124` (exact bytes) and :210 (shape, dtype and value per seed slice; NaN fails `array_equal`). 192: `seed_index` is never read (the slices are positional; :210 replaces it with zeros). 204 and :124 back each other up. |
| `r9_descriptive.py` scopes: 193 (:110), 194 (:118), 195 (:127), 197 (:190), 202 (:83), 203 (:137) | acceptable (plumbing) | numpy raises on mismatched shapes or an empty concatenate. Every caller builds boolean masks, one per seed (:361, :365, :383). Anchors are validated by `E.load_episodes` and :182. `score_seed` calls `summarize` once per scorer. |
| Real gaps: 159, 191, 172, 139, 145 | **gap** | F3 |

## 4. The CLIP H8 smoke of build 9 (item 2)

**The pass file equals r9_stats recomputed on its own arrays.** `fr_a2b_smoke_h8.py` rebuilds each seed from
`seed_index` and calls `ST.pass_record_a28(…, "smoke", …)`. The disclosure fields are taken from the record, then
checked against their sources (below). Output in `fr_a2b_smoke_h8.json`. Pass SHA-256
7549a10bdbcf980e3e9e145cd83da7942dbfbbb2ec52e5f100e60465473a8d3e.

| Check | Result |
|---|---|
| Every recomputed key (checks, Holm orders, n_episodes, n_clusters, design, seeds, mode, rule SHA, provenance) | equal; only `time` differs |
| Anchors | 576 = 3 seeds × 3 pairs × 64; 513 clusters (pass, sensitivity and my count agree) |
| Output SHAs (arrays, sensitivity) | equal to the pass's `outputs` |
| Episode files (9001 to 9003) | `load_episodes` hashes equal the pass's; `pair_index` equals the arrays' slices |
| Started file | 1 attempt; equal to its folder copy; mode, read, backbone, runner, module, coef, pins and input SHAs equal the pass's |
| Disclosures | `flagged_languages` (Emakhuwa, IsiNdebele) equals the pinned flags file; `near_duplicates` (0, 0) equals its source, whose SHA matches |
| Arrays | no share, rate, B1, R1 or PM key |

This smoke ran at 4458634. `setup_a28`'s smoke path changed at 8db7ceb (ticket 22: label-free parquet projection).
The smoke at final bytes is part of ticket 22's CLIP pre-read, which is still the controller's to run.

## 5. Hard rule 6 for area A2 (item 3)

"Runner" means my scope's part of an ID; stats (Holm, n*) is area A1.

| ID | Code | Test or run that proves it | Gap |
|---|---|---|---|
| SC1, SC2 (runner) | `run_r9_held.run_read`: pooled P1 to P7, seeds 52 to 54 on H5's episodes, comparators cosine, RCA, B, B′(A0), CF (`READ_SCORERS`) | `test_run_read_order_and_every_score_and_pass_argument`; A2's `fr_a2_e2e` (H6 path, real r9_stats); A2's H6 smoke | "Pre-read checks pass": the gate is now in the code (`preread_guard`, 62 tests). The CLIP pre-read run is outstanding: partial, by design until the run |
| SC3 (runner) | `pass_record_a28` path, seeds 55 to 57, H9 and H10 on H8's episodes (`a28_order_guard`, `a28_reference`) | `test_r9_held_a28` (`test_rebuilt_h8_hashes_checked_before_scores`, `test_h8_reference_recovers_complete_parts_and_hashes`); A2's `fr_a2_s51` (rule §5.1 exact); §4 above | none |
| D1, D9 | `a28_order_guard`, `prior_verdict` | `test_stage3_verdicts_required`; NO-GO at H6 or H7 opens H8 (`test_r9_held_a28.py:827`) | none |
| D7, D21 (read set) | pinned read set; descriptive partition check :369 | `load_read_set` pin SHA; B-area tickets 14 and 16 | none in my scope |
| D12, D17 | `check_pins`, `AD.check_pins`, ledger row checks | `GUARDS` `pins`, `test_c5_guard`, `test_ledger_a28_sha_*` | none (ledger rows: ticket 24) |
| D16, D29 | EpisodeRecorder's H5 hashes; `AE.builder_seeds`, `assert_seeds_free` | `test_h5_hash_parser_on_real_ledger`, `test_builder_seeds_are_1000s_plus_l`, `fr_a2_s51` | none |
| D25, D30 | `prior_verdict` status words; `read_guard` switched-off | `test_prior_verdict_words_of_each_stage`; `GUARDS['switched_off']` | none |
| Rule §7.4 (no per-pair, B1, R1 or PM before the verdict) | `allowed_keys` guard, `score_seed` guards, `bind_read` | `test_held_with_a_verdict_path_scores_no_pm_b1_or_r1_before_it`; `fr_a2b_c5.json` | F2 (no live path) |
| Rule §8 items 1 to 6; D13, D21 (per language), D27 (B′(A1), R1, PM) | `run_r9_descriptive.run`, `r9_descriptive.describe`, `paired_reads`, `a28_report`, `stage0b` | Smoke only: ticket 23's CLIP H6 and H8 descriptive smokes and ticket 22's pre-read smoke. Item 4's labels are tested (`test_paired_aff_labels_*`). | **contradicts** in real mode until F1 is fixed; partial tests (F3) |
| Unrequested | the chosen-cell gate share (`describe`, beside item 2's τ_0 share); per-direction rows also split per seed | – | unrequested, descriptive only; report them with no claim |

## 6. F1 and F2 on this commit (item 4)

Both are present at 8db7ceb, with the same lines as at 4458634: `run_r9_descriptive.py:75-77` and
`r9_score.py:256-269` did not change. `fr_a2b_c5.json` equals A2's `fr_a2_c5.json` once the temp paths are removed.
The smallest fixes outside the closure are in §2: one line in `bind_read` plus a test-fixture change (F1), and one
new test in `test_r9_descriptive.py` (F2).

## 7. Not done, and the long run

- Not re-derived: the chosen-cell gate share and the per-language thin-value counts (no real descriptive output
  exists yet).
- Long run for the controller: ticket 22's CLIP pre-read on final bytes, **after** the F1 and F3 fixes (command in
  `t22_handback.md`, about 10 to 20 min of CPU).
- To remove later (my session scratchpad only): `fr_a2b/wt` and `fr_a2b/wt2` (scratch copies of the round folder for
  F4's mutants).
