# Handoff: run round 2 of the reader fix (follow-ups of the confidence-gated reader), results today

Written 2026-10-06 (Amsterdam) by the run tab of round 1 for a fresh agent, because that tab's context was over half
full. The user drives the decisions; the ones already taken are in §2 and are not to be reopened. **The user wants the
result as early as possible, ideally today (Tuesday 6 October).**

## 1. Read in this order

1. **The approved spec:** `docs/superpowers/specs/2026-10-06-reader-fix-round2-design.md` (commit 7659034). It is short
   and is the design: three readers on the configuration without the style grouping (A0), one fusion family with the
   confidence gate and a new top-k restriction built in, round 1's rule inherited, a regression check.
2. **The draft rule:** `src/test/20261118_reader_fix_round2/DECISION_RULE.md` (see §5 for its state). It was drafted
   from the spec by a subagent and has **not** been checked or committed yet.
3. **Round 1, the work this builds on:**
   - quick overview: the user-read briefing `docs/user_read/2026-10-06_reader_fix.md`;
   - full report (final-reviewed): `docs/reports/auto/v2/2026-11-18_reader_fix_csd.md`, especially §1 (terms), §3
     (results), §4.2 and §4.3 (learned reader and gate), §6 (what follows);
   - round 1's rule (the template for round 2's): `src/test/20261117_reader_fix_csd/DECISION_RULE.md`;
   - round 1's run log, which shows the process that worked: `src/test/20261117_reader_fix_csd/20261117_reader_fix_csd_log.md`.
4. Project memory `project_v2-publication-plan-pending.md` (top entry), `v2-matched-control-lesson.md`,
   `feedback_seed-handling-light.md`; global rules in `~/.claude/rules/` (notably `agent-routing.md`, `final-review.md`,
   `shared-resources.md`, `storage.md`, `timestamps.md`, `debug-and-changelog.md`, `reports-layout.md`,
   `report-writing.md`, `user-read-reports.md`).

## 2. Decided by the user (2026-10-06), not to be reopened

- Round 1's verdict stands: no candidate cleared the bar (best: confidence-gated reader on the learned reader's weighted
  term, A0, +0.444 [+0.216, +0.674]); no fresh-seed test was built; seeds 49, 50, 51 are still free.
- **Next: more reader improvements now; design L later.** Test both families: the picks fix (adapted reader R2,
  realistic-practice reader R3) and the top-k restriction, with the gate built into every candidate (3 candidates:
  R1, R2, R3). Develop on A0; the style grouping (A1) only as a descriptive ablation of the carried candidate.
- **Rule review: a fresh Opus check** (as on 6 October 02:38), not a full ARS round.
- **The spec is approved** (user, 2026-10-06 ~15:35: "That's good"), after section-by-section approval in chat,
  including the regression check (R1 with top-k off must reproduce round 1's +0.444 exactly) and the timeline.
- Unchanged from round 1: the development bar (+0.5, lower bound above 0, gain lower bound above 0), the comparators
  (B, B′, matched counterpart), the GO list of the fresh-seed test, seed handling (develop on seed 42, test the one
  carried candidate on seeds 49, 50, 51, pooled and per seed, light ceremony), held rows not read.
- **Commits:** allowed on `main` without asking each time, scoped: the rule (and the spec, already done) before any
  code, then one commit per finished step. Stage by explicit path; never `git add -A`; never push; `bin/`,
  `docs/paper/` and `.DS_Store` stay untracked; `.claude/` is gitignored. End commit messages with your session's
  attribution lines.

## 3. The work, in order (target: verdict today)

1. **Finish the rule** (see §5): make sure the draft is complete and matches the spec; a fresh **Opus** subagent checks
   it against the spec and round 1's rule (matched controls, outcomes without an action, ambiguities two implementers
   would resolve differently, every number written out); fix; commit the rule before any code; send it to the user
   with SendUserFile (`proactive`).
2. **Plan:** the brainstorming flow requires a written implementation plan the user reviews and an execution choice.
   Keep it short (`docs/superpowers/plans/2026-10-06-reader-fix-round2.md`, the steps of this section) and ask the user
   in one message; subagent-driven execution has been the norm here.
3. **Implement** with two subagents in parallel, reusing round 1's code by import and changing nothing committed:
   - *fusion stream* (Sonnet): the top-k restriction on the gated fusion, the 896-cell min-margin cross-fit and the
     G_cf counterpart over the same cells, extending `src/test/20261117_reader_fix_csd/rc_core.py`; unit tests,
     including the regression check and tests that fail under a broken guard (round 1's final review found two
     unguarded tests);
   - *reader stream* (Opus): R2 (re-standardisation on seed-42 features, EM prior correction) and R3 (impure banks from
     round 1's A0 banks, label-free choice of k, retraining with round 1's recipe), on top of `rb_features.py`,
     `rb_build.py`, `rb_eval.py`; also the A1 versions for the ablation.
   Scripts assert the rule's SHA-256 and refuse to overwrite results; smoke modes first.
4. **Run** (the main session launches; CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, at most 3
   processes, check `uptime` and `free -g` first; `run_in_background`). The regression check first.
5. **Re-derive** every decision number with an independent agent that writes its own code and does not read the
   implementation (round 1's `rederive/` code can be reused by that agent).
6. **Apply the rule** mechanically (adapt round 1's `apply_rule.py`). If no candidate clears: no test; report to the
   user. If one clears: the sensitivity projection, seeds 49 to 51 via
   `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`, hash check, the test, the verdict, an
   independent re-derivation of the GO quantities, update `docs/superpowers/episode_seed_ledger.md`.
7. **Whole-branch final review** on the most capable model (re-derives every load-bearing number; one fix wave and a
   scoped re-review), then the full report in `docs/reports/auto/v2/` (next free sequence date after 2026-11-18) with a
   row in `docs/reports/reports_sum.md` and `python scripts/check_reports_sum.py`. A user-read briefing only if the user
   asks (`~/.claude/rules/user-read-reports.md`).
8. **Records:** the run log `src/test/20261118_reader_fix_round2/20261118_reader_fix_round2_log.md`; a change log
   `.claude/<yyyymmdd>_log.md` for edits to existing source files; project memory at milestones.

## 4. Code and pitfalls from round 1

| What | Where |
|---|---|
| Shared loader and evaluation (bundle with regression check, `evaluate`, `evaluate_fused`, `save_candidate`, `bar_info`, `clears_bar`, `pair_agreements`, `grouping_stack`, `expected_term`) | `src/test/20261117_reader_fix_csd/common.py` |
| Gated fusion, 224 cells, G_cf counterpart, out-of-half assembly | `src/test/20261117_reader_fix_csd/rc_core.py`, `run_rc.py` |
| Learned reader: halves, cross-fitted heads, banks, features, training, evaluation | `rb_build.py`, `rb_features.py`, `rb_eval.py`; outputs `results/rb_*` (heads, banks, `rb_reader_A0.pkl`) |
| Round 1's best candidate (regression target) | `results/cand_Rc_Rb_expected_A0.{json,npz}`, `results/rc_tau.json` |
| Independent re-derivation code | `src/test/20261117_reader_fix_csd/rederive/` |
| Rule application | `src/test/20261117_reader_fix_csd/apply_rule.py` |

`results/` folders are gitignored; round 1's results hold about 1 GB, reusable as inputs.

- `build_episode_bank` sorts the partition keys and seeds block i with `seed + i`: key names change the banks (the
  rule check of round 1 caught this).
- z-score per ranking row first, then gate. The matched counterpart of anything gated is the two-condition mean of the
  whole gated term (G_cf), never the gate times the mean term.
- `common.py` puts `HERE.parents[2]` first on `sys.path`: a copy of a module at another folder depth silently imports
  the original (a mutation test once "survived" for this reason).
- When a subagent edited a committed markdown report, an editor formatter rewrote the whole file (padded tables,
  italic markers moved). After any report edit, check `git diff --stat` and restore unintended changes.
- Write times with `TZ=Europe/Amsterdam date '+%F %H:%M'`; do not estimate them (round 1 wrote times ahead of the
  clock twice).
- Fresh-seed tests have so far roughly halved development effects; R@1 = (either + gain) / 2.

## 5. State at handoff

- Commits on `main` from round 1 and its reports: f25c48f, bb0a2a6, 9a11680, 31df8ba, 9e471e0, 7152a01, c36efea,
  616740e, 01e3bc3, a70f64b; spec 7659034. Nothing pushed.
- `src/test/20261118_reader_fix_round2/`: the draft `DECISION_RULE.md` (569 lines; all 36 round-1 input SHA-256s in
  its inputs table D14 match the files on disk) and `.gitignore`, written by a subagent of the round-1 tab, **not yet
  checked or committed**. The drafter resolved four spec ambiguities, which the Opus check should judge:
  1. top-k: inside the top-k set the fused order is kept and B's order applies only outside it (R@1, other-aspect
     rate and gain depend only on first place; only the swap metric could differ);
  2. R3's impure banks: replacement positions and pairs are drawn once (seeds 21700 and 21800), so the banks for
     k = 1, 2, 3 are nested;
  3. the A1 ablation runs on the carried candidate, or, if nothing is carried, on the candidate with the largest
     bar margin, labelled "not carried";
  4. it defines a shift report for R2 (the spec did not), and if R3's label-free choice gives k = 4, R3 is R1 and
     cannot be carried.
- No round-2 code exists. Seeds 49 to 51 are free. The local GPU and node404 are not needed.
