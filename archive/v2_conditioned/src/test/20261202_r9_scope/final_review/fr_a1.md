# Final review A1: statistics, apply-rule step, C13 phase 2 (r9 build, snapshot 4458634)

Reviewer A1, 2026-10-14 01:03 to 01:41. Snapshot /project/CoSiR-wt/r9-fr-a1 (detached at 4458634, r9-scope).
F = src/test/20261202_r9_scope/. Scripts and JSON outputs: F/final_review/fr_a1_*. Scratch files:
/tmp/claude-0/-project-CoSiR/18057a5a-4e60-48cd-83b0-ff14c8a319ae/scratchpad/fr_a1/.

**Stopped at the 200k context limit. One part is not done:** the guard mutations of r9_stats.py, indices 11 to 31
of `STATS` in fr_a1_guards.py (21 mutants, listed in §5). Everything else in the brief is done.

## 1. Verdict

**CONFIRMED WITH FIXES.** I found no wrong verdict, kind, count, Holm field or claim text. I compared my own code
with r9_stats and run_r9_apply_rule on 107 pass records and 56 apply runs, and found no field difference. One
MEDIUM finding: C13 never compares the detectable margins x (SE, x, x95) or their seed-42 inputs. A wrong x would
pass both the comparer and the apply step into a NO-GO reading. I showed this with a mutation. The fix is outside
the stage 2 closure. There are five LOW findings.

The progress note's items: item 1 is **confirmed** (all of it re-derived, below). Item 2 is **confirmed, refined**:
the pass file has no `read` or `backbone` field, so `load_pass` cannot check one. The binding rests on the
agreement's read, pass_file and SHA guards (all tested), and phase 2's array comparison would differ for another
backbone (finding L3). Item 5 is **confirmed** as a LOW wording point (finding L4).

## 2. What I re-derived

My code (fr_a1_lib.py) is written from R6 rule §3 and r9 rule §4. It forms the draws as a 5,000 × k multiplicity
matrix (W·sums, a different path from the code's gather-sum), decides counts on the integer 4·Σ, and uses Holm in
exact fractions. It does not import r9_stats.

| Check | How | Mine vs the code's | Result |
|---|---|---|---|
| Holm own-count test, all n = 0..5,000, m = 2, 5, 7, every k | integer form vs `Fraction` vs float `(n+1)/5001 ≤ 0.025/(m+1−k)` | 0 mismatches (5001 is odd, so equality never occurs) | agree |
| n* tables | search on the fraction inequality vs `ST.boundary` | 16, 19, 24, 30, 40, 61, 124 / 61, 124 / 24, 30, 40, 61, 124 on both sides | agree |
| `ST.holm` order, k, passes, own_count_passes, level, near_boundary | 30,000 random count sets (uniform, small, all-equal, ±1 around n*), families of 7, 2 (P4, P6) and 5 | 0 differences | agree |
| z table | `NormalDist`: z(1−0.025/7), z(1−0.025/2), z(1−0.025/5), z(0.975), each + z(0.8) | 3.5317, 3.0830, 3.4175, 2.8016 round to 3.532, 3.083, 3.417, 2.80 | agree |
| `pass_record` (H6/H7 shape) | 3 seeds × 12,288 episodes, 248 Pareto-sized anchor clusters with non-contiguous IDs; per-check quarter differences planted to give exact counts; 49 scenarios: n*−1, n*, n*+1 at every rank, names reversed, all-equal at 0, 15, 16, 17, 40, 5000, ties | n, point, ci95, ci_holm (1e-9 points), holm_k, level_two_sided, passes, own_count_passes, near_boundary, holm_order, n_episodes, n_clusters, no `secondary`: 0 differences; float count = integer count everywhere | agree |
| `pass_record_a28` (H8 to H10 shape) | 746 clusters, seeds 55 to 57, 58 scenarios: both levels at every boundary, P4 = P6 tie (P4 first), level-2 ties, both levels failing, level 1 failing with level 2 passing | as above, plus `level`, `design`, set_aside, flagged, near_duplicates pass-through, level-2 Holm present after a level-1 fail: 0 differences | agree |
| `sigma_split`, `sensitivity_record` | my one-way split (R3 §6.1) on 100·d; SE² = (σ_a²·ΣM_p² + σ_ε²·N)/N²; x = z(kind)·SE | 0 differences (rtol 1e-12), both designs | agree |
| `decide` / `apply`, H6, H7 | 28 real pass files (from `pass_record`) with agreements made by ticket 25's own comparer from ticket 25's own `rd9_stats.pass_record`; each check failing alone with point > 0, < 0, = 0 exactly; Holm stopping at ranks 1, 2, 4; all at n*; one at n*+1 | verdict, kind, failed, per-check pass, reading kind, "not reached" target and text, x in the reading, boundary list with re-derivation counts, claim text: 0 differences; `rd9_stats.decide` agrees on every verdict | agree |
| `decide` / `apply`, H8 to H10 | 28 cases: full; full at every n*; level 1 failing (P4, P6, both, mixed sign, tie at 62); level 2 failing per check × point sign; level-2 stop at rank 2; three set-aside entries on two languages | verdict (full / small / NO-GO), kind from level-1 failures only, level 2 untested (no reading, no boundary) after a level-1 fail, level-2 x (3.417) and level-1 x (3.083), N = 27 − distinct languages (25), claim phrases: 0 differences | agree |
| `main` | tmp results, tmp agreement directory, tmp ledger copy | GO run exit 0; second run exit 4 "exists; a verdict is never overwritten", file SHA-256 unchanged | agree |
| Not-run rows | tmp ledger | H10 switched off before H7: refused; H7, then H10 switched off: written; H8 switched off: refused (D30); H9 dropped: written; H6 dropped with row "(pending)": refused; H9 again: refused | agree with rule §10, D25, D30 |
| C13 independence | import rd9_phase2, rd9_compare, rd9_phase1 and every lazily imported `src` module in a fresh interpreter | 0 r9_/run_r9/round modules in `sys.modules`; CLIP's `rb_reader_A0.pkl` references only numpy and sklearn (read with `pickletools`, not loaded) | independent |
| C13 comparer sensitivity | mutate 17 pass-file fields (H6) and 22 (H9), remake the agreement with `rd9_compare` | n, passes, own, holm_k, near_boundary, point (also +1e-6), ci95, ci_holm, level_two_sided, holm_order, n_clusters, seeds, set_aside, flagged: all disagree. mode, rule_sha256, quantity, level, design: comparer agrees, and the apply step refuses all except `quantity` (fr_a1_c13b.json) | see M1, L2 |
| Sensitivity file under C13 | SE ×2 (x, x95 kept consistent) in `sensitivity_H6.json` and `sensitivity_H9.json` | agreement true, verdict written with x = 1.784 instead of 0.892 | **gap (M1)** |

## 3. Findings

**M1 (MEDIUM): C13 never compares the detectable margins or their seed-42 inputs.**
- Where: `rederive/rd9_compare.py:70-90` (`compare_stats` compares only n, point, ci95, tested and the Holm fields)
  and `:259-290` (`compare_phase1` compares no σ). `rd9_phase2.py:165-172` computes SE, x and x95, and the comparer
  then drops them. The comparer never opens `sensitivity_<Hn>.json` or `sensitivity_seed42_<backbone>.json`.
  `run_r9_apply_rule.load_sensitivity` (:319) checks only x = z·SE and x95 = 2.80·SE.
- Rule: R6 §8.4 (inherited by r9 §7.5 "as R6 rule §8.4, per read"): phase 1 re-derives "the sensitivity inputs".
- How it fails: a wrong σ_a² or σ_ε² for Qwen or SigLIP 2 (stage 2 writes them), or a wrong M_p, gives a wrong x.
  The NO-GO reading then says "inconclusive at a detectable margin of x" with that wrong x, and C13 agrees. Shown:
  fr_a1_c13b.json. The verdict and kind do not depend on x. The CLIP dry run of stage 2 does reproduce CLIP's frozen σ
  file, which lowers the risk but does not cover the new backbones.
- Fix: in rd9_compare's per-read mode, take `--reported-sensitivity` and compare SE, x and x95 per check within
  1e-9. In phase 1, compare the re-derived σ_a² and σ_ε² with `sensitivity_seed42_<backbone>.json`. Optionally make
  the apply step require agreement items with these keys.
- Closure: rederive/ and run_r9_apply_rule are outside the 17 modules. Fix it before the H6 or H7 read. It is cheap
  and needs no stage-0b re-compare and no stage 2 rerun.

**L1 (LOW): `main` prints a boundary without the re-derivation's count.**
- Where: run_r9_apply_rule.py:690-692. Rule §7.6 and R6 §8.5 say the count goes to the user "with the runner's and
  the re-derivation's counts". The verdict file has `rederivation_n`, but the printed line does not.
- Fix: add `rederivation {b['rederivation_n']}` to the print. Outside the closure.

**L2 (LOW): `quantity` is not checked.**
- Where: `load_pass` / `_check_entry` (:242-256) accept any string, and the comparer does not compare it. A pass file
  with P3 labelled "R@1, AFF - B0" gives a verdict file that carries that label (fr_a1_c13b.json). The readings use
  the fixed LABELS, so no reading is wrong.
- Fix: require `quantity == ST.QUANTITIES[c][2]` in `_check_entry`. Outside the closure.

**L3 (LOW, progress-note item 2): the pass file names no read or backbone.**
- Where: `ST.pass_record` and `pass_record_a28` (r9_stats.py:248-271) write no `read` or `backbone` field.
  `load_pass` (:257) therefore binds a pass file to its read only through the agreement's `read`, `pass_file` and
  `held_pass_sha256`. All three are guarded and tested.
- How it fails: a pass file copied under another read's name. Phase 2 for that read would rebuild with its own
  backbone and its arrays would differ, so the agreement would be false.
- Fix: run_r9_held adds `read` and `backbone` to the record it writes, and `load_pass` checks them. This is outside
  the closure: do not change r9_stats for it.

**L4 (LOW, wording; progress-note item 5; the user decides):** the full ArtELingo-28 claim (rule §4.3, spec §6)
says AFF "keeps the same sentence" but does not name the sentence (run_r9_apply_rule.py:136-139, 413).
- How it fails: under D9, H9 or H10 can run after an H6 or H7 NO-GO. A full claim would then say "keeps" a sentence
  that did not hold on ArtELingo for that backbone.
- Fix (report writer or the user): name the sentence, or print the sibling read's verdict (H5, H6 or H7) beside every
  full claim. The code matches the rule as written.

**L5 (LOW): no runtime check that rd9_phase2 stays independent after unpickling.**
- Where: rd9_phase2.py:103 and rd9_inputs.py:223 unpickle files. Static imports are clean. The Qwen and SigLIP 2
  reader pickles do not exist yet; if one held an r9 class, loading it would import an r9 module unseen.
- Fix: assert `not [m for m in sys.modules if m.startswith(("r9_", "run_r9"))]` at the end of phase 2. Outside the
  closure.

## 4. Hard rule 6 gaps (SC1, SC2, SC3)

| SC | What the spec words | What the apply step delivers | Gap |
|---|---|---|---|
| SC1, SC2 | GO iff the pre-read checks pass and all seven pass (Holm, one-sided 0.025); claim verbatim | Holm verified on 49 + 28 cases; claim verbatim with "Qwen" / "SigLIP 2" and the "Not ..." line; "pre-read checks" covered here only by the smoke-record guard and the phase-2 agreement | none in this step. The §6 items (stage 0b, head refits, regression) are the runner's checks, which I did not review (area A2 or B). |
| SC1, SC2 NO-GO | each failed check read by C11: "inconclusive at x" or "did not beat" | correct readings and kinds, "not reached" with its own reading | **partial**: x is not independently re-derived (M1) |
| SC3 | full / small / NO-GO per backbone, level 1 (P4, P6) then level 2 | verified, including serial gatekeeping and the kind from level-1 failures only | none |
| SC3 claims | full: "with captions from 27 other languages ... keeps the same sentence"; small: "beats B′(A0) ..." | the rule's binding N form ("the N languages that entered the read"), set-aside languages with pair and seed, flagged languages, near-duplicate counts, the "Not ..." line | the spec's "27" vs the rule's N is not a gap (the rule governs). **Partial** (wording): "the same sentence" is unnamed (L4). Unrequested: none. |
| Switched off / dropped | "a result, no read" | status texts written, verdict null, ledger wording and D30 checked | none |

## 5. Guards whose deletion survives

I deleted or weakened each guard, ran the in-scope tests (test_r9_apply_rule.py; for r9_stats also
test_r9_stats.py and test_r9_a28_stats.py), and restored the file at once (SHA-256 checked; the snapshot's git
status is clean). Results: fr_a1_guards_apply_*.json, fr_a1_guards_stats_*.json.

run_r9_apply_rule.py, 56 mutants: all 19 `# GUARD:` markers are killed (including the smoke_guard call site), and
so are 30 of 37 unmarked checks and semantic mutants: gatekeeping, any/all kind, point ≥ 0, N by entries, write_once,
level-2 boundary, not-run checks, sensitivity checks, set-aside checks. **Seven survive:**
- `load_pass` design-key check (:271). It sits behind `_check_entry` levels and `assert_consistent`.
- `load_pass` checks-set check (:274). It sits behind `_check_entry` "missing". A non-dict `checks` would crash
  instead of refusing.
- `load_pass` seven-check holm_order schema (:280). It sits behind the holm_order equality in `decide` (exit 5
  instead of 4).
- `assert_rule`: the rule SHA-256 (:229), the R6 rule SHA-256 (:234), the fixed-phrase check (:238), and the
  `assert_rule()` call on the read path (:630). No test edits the rule file or a phrase through `apply`. This is
  partly covered: the pass file's `rule_sha256` is checked, and test_fixed_phrases_are_in_the_rule checks the
  constants. Fix: one test that points `AR.RULE_PATH` at an edited copy and expects exit 4, and one that changes a
  phrase constant and expects exit 5.

r9_stats.py, 11 of 32 mutants run:
- Killed: finite, |v| ≤ 2, quarter, exact sums, ci95 copy, point_ci cross-check, CF gain 0, seeds, per-seed count.
- Survived: the `k < 2` check (:97), which sits behind `cluster_bootstrap`'s own ValueError, and the per-key shape
  check in `check_diffs` (:200), which sits behind the shape check in `bootstrap_draws`.
- **Not run (left):** indices 11 to 31 of `STATS`: extra keys, extra seeds, rule SHA, Holm names, Holm count range,
  mode/design, the three sensitivity_record validations, M_p ints, ΣM_p = N, and the semantic mutants (strict
  integer form, Holm factor, tie order, Z level 2, Holm-interval level, near-boundary width, level-1 tie order, a28
  set_aside pass-through, a28 level field, sensitivity z kind). Command, about 3.5 minutes per run, 7 runs:
  `cd /project/CoSiR-wt/r9-fr-a1/src/test/20261202_r9_scope/final_review && timeout 235 /root/miniconda3/envs/CoSiR/bin/python fr_a1_guards.py --which stats --batch S --size 3`
  for S = 11, 14, 17, 20, 23, 26, 29. Each run restores r9_stats.py itself. Check `git status` after each run.
