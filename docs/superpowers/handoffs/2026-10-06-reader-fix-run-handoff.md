# Handoff: run plan (a), the reader fix with the CSD grouping, under the ARS-reviewed rule

Written 2026-10-06 02:30 (Amsterdam) by the review tab of this Herdr workspace for a fresh agent in a new tab. The
user drives the decisions; the decisions already taken are in §2 and are not to be reopened. Your job: write the
decision rule, have it checked, commit it, then implement and run the plan, ending with the user's GO / NO-GO on
Friday 9 October 2026 (CVPR abstract 10 November).

## 1. Read in this order

1. **The review report, self-contained, with nine figures:**
   `docs/reports/auto/v2/2026-11-17_ars_reader_fix_plan_review.md`. It explains the task, every term, where the work
   stands, the plan, the review's verdict (Major Revision, both blocks repairable by rule text), every finding and its
   fix, and Figure 4, the decision rule as it stands after the fixes. Start here.
2. **The editorial decision letter:** `src/test/20261117_reader_fix_csd/ars_review/editorial_decision.md`, sections
   "Required Item Details" (R1 to R5, each with Requirement and Acceptance criteria) and "Suggested Item Details"
   (S1 to S17). These are the exact texts to implement. The two reviewer reports (`phase2_methodology.md`,
   `phase2_eic.md`) and the review log (`20261117_ars_reader_fix_review_log.md`) are in the same folder.
3. **The plan as drafted:** `docs/superpowers/handoffs/2026-10-06-reader-fix-with-csd-handoff.md`. Its §5 is the
   draft the review changed; its §3 (fixed items), §6 (code, data, environment table) and §7 (pitfalls already paid
   for) still hold.
4. Templates for rule files: `src/test/20261108_new_method_quick_checks/DECISION_RULE.md` and `ADDENDUM_3_N6C.md`;
   the step-1 plan `src/test/20261116_grouping_step1_style/PLAN.md` (style of a pre-registered plan in this project).
5. Background only if needed: the grouping report `docs/reports/auto/v2/2026-11-12_partition_quality_leiden_communities.md`
   (§2 terms, §9 step 1) and the stage report `docs/reports/stage/2026-10-04_aspect_conditioned_similarity_methods.md`
   (§2, §14).
6. Project memory `project_v2-publication-plan-pending.md`, `v2-matched-control-lesson.md`,
   `feedback_seed-handling-light.md`; global rules in `~/.claude/rules/` (notably `agent-routing.md`,
   `final-review.md`, `shared-resources.md`, `storage.md`, `timestamps.md`, `debug-and-changelog.md`,
   `reports-layout.md`, `report-writing.md`).

## 2. Decided by the user (2026-10-06 02:25), not to be reopened

- **Adopt every review fix as proposed:** must-fix R1 to R5, should-fix S1 to S17, and the controller's AR check. In
  particular: the development bar alone decides whether a test is built (pick accuracy and R-b's bank accuracy are
  diagnostics that decide nothing); one definitions block; the reader fused on B with `crossfit_nested(B, B, T,
  parity)` as in step 1; B′ = B rebuilt (`crossfit_condition_free(cos, T_N1u, T_6u over the configuration's own
  groupings, parity)`); the bar comparator and the GO list include B, B′ and the matched counterpart; **no kill for
  R-b** (a label-free shift report instead); R-c with the hard gate, the τ grid and the counterpart
  G_cf = (g_a·z(T_a) + g_b·z(T_b)) / 2 over 224 cells; R-a divides by the noise-only spread (pooled within-episode
  standard error), not the RMS; R-b frozen before any seed-42 number (60,000-row cross-fitted heads per painting half,
  fixed bank, C by five-fold CV on bank episodes, A0's own three-grouping bank).
- **Carry priority:** the best A1 candidate that clears the bar; an A0 candidate only if no A1 candidate clears.
  Ties within 0.05 of the largest bar margin go to R-a, then R-b arg-max, R-b expected, R-c.
- **Cutoff: Thursday 8 October 12:00 (Amsterdam).** Candidates without development numbers by then are dropped from
  this round; R-c is built on the best completed candidate. Fallback order if time runs short: R-a and R-c on R-a
  first, then R-b arg-max on A1, the remaining configurations last.
- **An A0 GO counts as a GO for plan (a)**, reported as a reader-fix GO that leaves the CSD question open.
- **Commits:** you may commit to `main` without asking each time, scoped: first the review records, report, figures,
  both 2026-10-06 handoffs, `docs/reports/reports_sum.md` and `DECISION_RULE.md`, before any code; then one commit per
  finished step. Stage by explicit path; never `git add -A`; never push; `bin/`, `docs/paper/` and `.DS_Store` stay
  untracked; `.claude/` folders are gitignored. End commit messages with the attribution lines your session gives you.
- Still fixed from the plan: the four groupings (affect Leiden 41, image and caption k-means 64, CSD Leiden 17); no
  grouping choice reads evaluation labels; matched controls for everything; develop on episode seed 42, test the one
  carried configuration on fresh seeds 49, 50, 51 (each reported and pooled; light seed ceremony). Held rows are not
  read.

## 3. The work, in order

1. **Write `src/test/20261117_reader_fix_csd/DECISION_RULE.md`** (the folder and its `.gitignore` exist; the review
   is in `ars_review/` inside it). It must be self-contained and carry:
   - a precedence line ("where this file differs from the handoffs, the memo or the appendices, this file governs"),
     a status line (items on seed 42 are development selection; the fresh-seed test is the only confirmatory step),
     a short glossary, and a note that folder dates are sequence numbers;
   - the definitions block (R2, S11, S16), the candidates and their exact specifications (R-a S9, R-b S10, R-c R5,
     S1, S14), the measures (§5.3 of the plan plus pick accuracy as a diagnostic, A1 minus A0 under each reader, the
     AR check, R-b's label-free shift report), the development bar, the carry and tie rules, the kill (R1), the cutoff
     and fallback order, the test (seeds, SHA-256 check, per-seed cross-fitting declared part of the method, only GO
     quantities computed before the verdict, per-seed and per-pair results descriptive, the projected sensitivity
     computed from a seed-42 variance split before the seeds are built, the reading of a NO-GO with a positive pooled
     point, the "claim licensed" line), and the outcome-to-action table (S2);
   - every number it needs written out (grids, τ percentiles, bank size, head draw size, comparators).
2. **Check the rule before committing it:** dispatch a fresh subagent (most capable model) with the rule file and the
   letter, asking it to check every R and S item's acceptance criteria and look for any outcome without an action.
   Fix what it finds. This is a scoped check, not a new ARS round. Then make the first commit (records and rule) and
   send the rule to the user with SendUserFile (`proactive`).
3. **Implement and run on seed 42** (CPU only; `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, at most
   3 processes, check `uptime` and `free -g` first; nothing needs the GPU or node404). Subagents implement (sonnet for
   well-specified code with tests, opus for design or debugging; two in parallel is fine: R-a with the AR check and
   R-c in one stream, R-b's cross-fitted heads, banks and reader in the other). The main session reviews every result,
   launches long jobs itself with `run_in_background`, and re-derives every load-bearing number with its own code.
   Scripts assert the rule file's SHA-256 and refuse to overwrite results, as in step 1. Reuse step 1's code
   (`src/test/20261116_grouping_step1_style/run_step1.py`: `evaluate_general`, `bar_margin`, B′) and the plan's §6
   table.
4. **Apply the rule** at or before the cutoff. If no candidate clears the bar, no test is built: report to the user.
5. **The test** (Thursday afternoon): build seeds 49, 50, 51 with
   `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`, check the 9 new episode SHA-256s against
   all earlier ones, rerun every cross-fit on each seed's halves, compute only the GO quantities, write the verdict,
   then the descriptive numbers. Update `docs/superpowers/episode_seed_ledger.md`.
6. **Thursday evening:** the controller re-derives the test numbers with independent code. **Friday morning:** a
   whole-branch final review on the most capable model (re-derives every load-bearing number; one fix wave and a
   scoped re-review), then the report and the user's decision.
7. **Records:** a log `src/test/20261117_reader_fix_csd/20261117_reader_fix_csd_log.md`; a report
   `docs/reports/auto/v2/<next free sequence date>_<topic>.md` written to `~/.claude/rules/report-writing.md` (baseline
   beside every number, figures, matched counterpart and B′ always shown) with one row in `docs/reports/reports_sum.md`
   and `python scripts/check_reports_sum.py`; a change log `.claude/yyyymmdd_log.md` for any edit to an existing
   source file; project memory updated at milestones.

## 4. Talking to the user

The user may be away and follows from another device. Send files with SendUserFile (`proactive` when you surface
them unasked): the committed rule, the seed-42 development results, the test verdict, the final report. Any decision
the rule does not cover goes to the user (AskUserQuestion); do not decide it yourself, and do not change the rule
after its commit without the user. Timestamps in Amsterdam local time, plain (`TZ=Europe/Amsterdam date '+%F %H:%M'`).

## 5. State at handoff

- Nothing of the plan has run. No `DECISION_RULE.md` exists yet. Seeds 49 to 51 are free.
- Uncommitted: `docs/superpowers/handoffs/2026-10-06-reader-fix-with-csd-handoff.md`, this handoff,
  `src/test/20261117_reader_fix_csd/` (`.gitignore`, `ars_review/` with memo, manuscript, Phase 0 to 2 cards,
  editorial decision, log; `contract.json` and `metadata.json` are gitignored by `*.json`), the review report and its
  assets folder `docs/reports/assets/2026-11-17_ars_reader_fix_plan_review/`, and the new row in
  `docs/reports/reports_sum.md`. Last commit on main: 6b67c7f.
- The local GPU and node404 are not needed. Another discussion tab ("leiden-plan") covers Leiden in factor learning;
  do not duplicate it.
