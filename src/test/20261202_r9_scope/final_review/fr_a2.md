# Final review A2: held runners H6 to H10, C5 order, ArtELingo-28 episodes and bundle, descriptive numbers

Reviewer A2 (Opus 5.5), snapshot /project/CoSiR-wt/r9-fr-a at 4458634 (r9-scope), started 2026-10-14.
Scripts and JSON outputs: this folder (`fr_a2_*.py`, `fr_a2_*.json`). Temp files: the session scratchpad `fr_a2/`.

## 1. Verdict

**NOT CONFIRMED (partial review; stopped at the 200k context mark).** One MEDIUM blocks rule §8 in real mode
(F1); one LOW (F2). The C5 order holds on every path I tried: no held or ArtELingo-28 number can be computed
before that read's own verdict through the paths that exist. Section 5.1 re-derives exactly. Left for a fresh
reviewer: the rest of the guard sweep (154 of 205 mutants), hard rule 6 in full, and the CLIP H8 smoke.

## 2. What I re-derived

| Check | How | Mine vs the code's | Result |
|---|---|---|---|
| Which verdict files open `score_seed`'s before-verdict guards | `fr_a2_c5.py`: synthetic `make_bundle` (held, a28), `include_pm=True`, verdict files of another read, null, empty, reserve-only | The guard needs only a file whose name fits the mode; content and read are not read | All six wrong files accepted; missing file refused (`fr_a2_c5.json`) |
| Does `run_r9_descriptive.bind_read` accept the verdict that `run_r9_apply_rule.apply` writes | `fr_a2_c5.py`: `apply` on test_r9_apply_rule's staged H6 and H8 passes, then `bind_read` | apply writes `backbone` = ledger name ("Qwen3-VL-Embedding-2B", "CLIP ViT-B/32"); bind_read wants the code name ("qwen3vl_emb_2b", "clip_b32") | Refused "verdict identity" for both; with the code name patched in, binding passes identity, agreement and SHA steps |
| Rule §5.1 items 1 to 6 (languages, shares, seeds 1000·s+ℓ, set-aside passes, stop, assembly, part hashes) | `fr_a2_s51.py`: my own pass loop and Fraction-based largest-remainder shares from the rule text, the same R6 builder; row space = the real read set's row_id, painting, dropped (74,569 rows, 746 paintings); language = row_id mod 27; synthetic labels (no real label or language column read) | Per-pair SHA, 27 part SHAs, shares, set-aside list and reasons, per-episode language, parity | Equal on every pair of seeds 55, 56, 57 (base); on seed 55 with Emakhuwa set aside in two pairs (values subset), Kyrgyz set aside in three pairs; both stop on three set aside (Emakhuwa, Hindi, IsiXhosa). Shares: 0 mismatches in 2,000 random size lists (ties included). Rule's language list is Python sort order and equals `AE.LANGUAGES`; builder seeds 55001 to 57027 equal (`fr_a2_s51*.json`). Not exercised by me: a failed-draw set-aside (the tests cover its classification) |
| Held runner real mode end to end, H6 and H8 paths | `fr_a2_e2e.py` (pytest): test_r9_held's tmp_path ledger and stubs, my synthetic per-anchor scorer, the real `r9_stats` (bootstrap included; H6 at 128 per pair, H8 at the stub's 1 per pair); pass recomputed from the saved `held_arrays` by my reassembly | `checks` and the other pass keys | Equal for H6 and H8; sensitivity n_paintings equals mine; no secondary key; a second run and an after-crash run after the verdict both refuse (exit 4) (`fr_a2_e2e.json`) |
| CLIP H6 smoke | `run_smoke('clip_b32','H6', results=<scratchpad>)`, 161 s on CPU; `fr_a2_smoke_check.py` | Pass vs `ST.pass_record` on its own arrays | Every key equal except `time`; started file equals its folder copy; no share, rate or secondary key in the arrays (`fr_a2_smoke_check.json`) |
| CLIP H8 smoke | same, read H8 | – | **Not done**: killed by my 235 s limit after the head fits (machine load about 30 from other sessions); nothing was written |

## 3. Findings

**F1 MEDIUM. The descriptive runner refuses every real verdict, so rule §8 can never run on a real read.**
`run_r9_descriptive.py:75-76` (`bind_read`) requires `verdict['backbone'] == RH.READ_BACKBONES[read]` (code name,
e.g. "qwen3vl_emb_2b"), but `run_r9_apply_rule.py:623` writes `READS[read]["backbone"]` (ledger name,
"Qwen3-VL-Embedding-2B"); `run_r9_held.prior_verdict` (`:623`) also expects the ledger name. Scenario: after H6's
verdict, `run_r9_descriptive.py --read H6` exits 4 "verdict identity"; the same for H7 to H10, and H9/H10 also
fail inside `a28_report`'s `bind_read(H8)`. Shown in `fr_a2_c5.json` with verdicts written by the real `apply`.
Fail-safe (no number leaks), but §8 items 1 to 6 are blocked. The tests miss it because the fixture
(`test_r9_descriptive.py:41`) builds the verdict with the code name. Fix: compare with
`RH.LEDGER_NAMES[RH.READ_BACKBONES[read]]` in `bind_read`, and build the test verdict through `AR.apply` (or
`AR.READS[read]["backbone"]`). Outside the stage 2 closure (run_r9_descriptive and its test only).

**F2 LOW. `score_seed`'s before-verdict guards check only a file name, not the read.** `r9_score.py:256-269`
(`read_verdict_name`, `pm_before_verdict`, `secondary_before_verdict`, `on_scores_before_verdict`) accept any
existing file whose name fits the mode. `fr_a2_c5.py`: a held bundle scores PM, B1, R1 and calls `on_scores` with
H7's verdict, a null "not run" verdict, an empty file, or a reserve-only name; an a28 bundle does so with H8's
verdict. No live path: the only gated callers are `run_r9_held` (`include_pm=False, secondary=False`, no
`on_scores`) and `run_r9_descriptive`, whose `bind_read` checks status, read, mode, variant, pass and agreement SHA
before `score_seed`. `test_held_verdict_file_of_another_read_does_not_count` only tests a different name in the
same folder. Fix outside the closure: a test in `test_r9_descriptive.py` that `bind_read` refuses another read's
and a null verdict before any array is loaded (keeps r9_score untouched). Fixing r9_score itself would touch the
closure (stage 0b re-compare and both stage 2 chains).

**Checked, no finding.** Descriptive smoke pointed at real results is refused by `AR.load_pass(smoke=True)` (mode
and seeds) before any array is read; `paired_reads` binds each paired read through `bind_read` (its own verdict);
the A28 smoke uses ArtELingo selection rows with the synthetic language column; `run_read` prints only file names
and SHAs; error messages carry shapes and names, not results. Item 4 of the first note (`near_duplicates_a28.json`
not pinned) is task review t21's open LOW-5; I did not check C13's comparison of it.

## 4. Hard rule 6 gaps (spec §6, SC1 to SC3, my scope)

Partial (not finished). SC1/SC2 (H6, H7: P1 to P7 pooled, seeds 52 to 54, comparators COS, RCA, B, B′(A0), CF):
the runner computes them (E2E above). "The pre-read checks pass" is ticket 22, not in this snapshot: missing here by
design. SC3 (two levels, seeds 55 to 57, per backbone, H9/H10 on H8's episodes): runner present, §5.1 exact.
Rule §8 (spec §3.4, §3.6, not an SC): contradicts in real mode (F1). Not done: claim wording and set-aside
disclosure against spec §6 (area A1's apply step).

## 5. Guards whose deletion leaves all tests green

`fr_a2_mut.py` (in place in scratch copies of the round folder, never in the snapshot) enumerates 205 mutants: 105
`# guard:` statements, 5 guard `if` lines (both directions for the two `continue` exemptions), 3 guarded
assignments, 11 guard call sites in run_r9_held and 81 `refuse_unless`/`require` boundaries in the descriptive
runner (it has no markers). First pass runs a fast test set per file (`FAST` in the script); survivors would rerun
on the full set (`FR_FULL=1`). Done so far: 75 of 205 (merged after hand-back): 72 caught, 0 survived, 3 timed out at 130 s under load (r9_score `include_pm_bool`, `admitted`, `parity`: rerun them alone) (`fr_a2_mut_results.json`). Left: 130.
Known open survivors from task reviews (not re-run by me): t21 LOW-1 to LOW-8 (ledger pins_a28 row variants,
order guard for continuations, `prior_verdict` clauses, `a28_reference` clauses, provenance fields, `READ_BACKBONES`
literal, r9_stats sensitivity inputs); t23_fix LOW-A (agreement with wrong SHA).
