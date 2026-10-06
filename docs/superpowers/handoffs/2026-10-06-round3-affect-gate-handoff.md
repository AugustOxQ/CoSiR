# Handoff: round 3, one-sided affect steering on R1, straight to the fresh-seed test

Written 2026-10-06 19:10 (Amsterdam) by the reader-fix-round2 tab of this Herdr workspace for a fresh agent in a new tab.
The user drives the decisions; the ones already taken are in §2 and are not to be reopened. The user wants the round run
as early as possible.

## 1. Read in this order

1. **The idea and its evidence:** `docs/reports/auto/v2/2026-11-20_r1_levers_brainstorm.md` (commit c749ee1), in
   particular the Summary, §2 (where R1's margin comes from), §3.1 (rank 1, one-sided affect steering), §5 (what not to
   do), §6 (disclosures). Its code is `src/test/20261120_r1_levers_brainstorm/bs_*.py` (harness `bs_lib.py`, idea 1 in
   `bs_04_readers.py`, `bs_05_aff.py`, `bs_06_aff_ceiling.py`; results in the gitignored `results/`). The harness
   reproduced round-1 R-c exactly but was **not reviewed independently**: treat its numbers as exploratory.
2. **Round 2, the work this builds on:** report `docs/reports/auto/v2/2026-11-19_reader_fix_round2.md` (final-reviewed),
   rule `src/test/20261118_reader_fix_round2/DECISION_RULE.md` (the template for round 3's rule: §2 data and
   intervals, D1 to D14, §4.5 and §4.6 fusion and counterpart with the exact integer cross-fit criteria, §6 the
   fresh-seed test, which round 2 specified but never ran, §7 to §9), its run log
   `src/test/20261118_reader_fix_round2/20261118_reader_fix_round2_log.md` (the process that worked), and its final
   review `src/test/20261118_reader_fix_round2/final_review/final_review.md`.
3. **Round 1:** report `docs/reports/auto/v2/2026-11-18_reader_fix_csd.md` (R-b learned reader, R-c confidence gate)
   and rule `src/test/20261117_reader_fix_csd/DECISION_RULE.md`.
4. **The no-caption spike** (context for why CSD is not used): `src/test/20261119_no_caption_csd_spike/20261119_no_caption_csd_spike_log.md`.
5. Project memory: `project_v2-publication-plan-pending.md` (top entries), `v2-matched-control-lesson.md`,
   `feedback_seed-handling-light.md`, `feedback_final-review-catches-real-issues.md`; global rules in `~/.claude/rules/`
   (`agent-routing.md`, `final-review.md`, `shared-resources.md`, `storage.md`, `timestamps.md`, `debug-and-changelog.md`,
   `reports-layout.md`, `report-writing.md`).

## 2. Decided by the user (2026-10-06), not to be reopened

- Round 2's verdict stands: no candidate cleared the development bar; R1 (round 1's learned reader with the confidence
  gate on A0) is the best so far: bar margin +0.472 in round 2's 896-cell family, +0.444 (= round-1 R-c exactly) in the
  224-cell family.
- **Design L (grouping redesign) is not pursued now. Keep improving based on R1.** The idea of dropping the CLIP caption
  grouping is parked (the spike gave style × genre +0.073 but pooled +0.256 below A0's +0.444).
- **Round 3 = the brainstorm's rank-1 idea, one-sided affect steering ("AFF"), as a pre-registered round with R1 beside
  it, going straight to the fresh seeds 49, 50, 51** (it was found by a search over about 50 variants on seed 42, so
  seed 42 cannot serve as its development data). Discussed with the user on 2026-10-06 ~19:00 and approved ("go with
  things"); the user asked for this round to start in a fresh tab.
- Process as in round 2: a short spec the user approves, then the rule written from it, a **fresh Opus check** of the
  rule (not an ARS round), fixes, commit before any code, rule sent to the user with SendUserFile (`proactive`); a short
  implementation plan and one question to the user for the execution method (subagent-driven has been the norm);
  subagents implement; the main session launches every real run (CPU only); an independent agent re-derives every
  decision number with its own code; the whole-branch final review on the most capable model; the full report.
- **Commits:** allowed on `main` without asking each time, scoped: spec and rule before any code, then one commit per
  finished step. Stage by explicit path; never `git add -A`; never push; `bin/`, `docs/paper/` and `.DS_Store` stay
  untracked; `.claude/` is gitignored. End commit messages with your session's attribution lines.

## 3. The candidate (as the brainstorm defined it; the spec fixes it exactly)

AFF keeps everything of R1 and changes one factor of the gate:

- reader: round 1's two A0 half-readers (`src/test/20261117_reader_fix_csd/results/rb_reader_A0.pkl`), probabilities
  averaged over halves; weighted term T^c = Σ_h P^c(h)·s_h on A0 = (affect, image, caption);
- thresholds: R1's τ_0..τ_3 (round 1's `results/rc_tau.json`: 3.8684538364530674e-05, 0.21702129490553143,
  0.47973989883399526, 0.7502585816077211), frozen;
- **gate g^c = 1[m^c ≥ τ] · 1[arg max_h P^c(h) = affect]** (R1's gate is the first factor alone);
- fusion: R1's 224 cells (k_top 13 only; top-k never helped), z-scoring before the gate, integer min-margin cross-fit
  (round 2's rule §4.5 item 8), matched counterpart G_cf = two-condition mean of the gated term under the same gates,
  integer max-R@1 cross-fit (§4.6).

Seed-42 (exploratory, from the brainstorm): fused 19.137, counterpart 18.396, bar margin +0.700 [+0.460, +0.937]
against B′ (18.437), gain statistic +3.111 [+2.780, +3.456], either −1.630; per pair e×s +0.977, e×g +1.453, s×g −0.330;
paired AFF minus R1 +0.256 [+0.043, +0.462] (bar margin), +0.218 [+0.064, +0.371] (fused R@1). Sixteen one-sided variants
spanned +0.42 to +0.72 (median +0.63). A label-reading random gate with AFF's per-condition open share (80.9% of
condition a, 29.5% of b) reached +0.56 to +0.67: the gain comes from steering the side where the reader sees affect.

## 4. Open points the spec must settle with the user (our proposals)

Ask about these in the spec, one at a time or as a short sectioned spec; do not decide them alone.

1. **How "affect" is fixed label-free.** Proposal: state the criterion (the grouping whose score is least redundant with
   B: lowest mean row correlation of z(s_h) with z(B) on seed 42, which uses no labels; affect 0.35 to 0.38 against 0.62
   to 0.71), record its seed-42 values, and freeze affect.
2. **Seed 42's role.** Proposal: round-3 code recomputes AFF on seed 42 and must reproduce the brainstorm's numbers
   exactly (a regression check of the new code, alongside R1 = round-1 R-c exactly); the development bar (D12) is
   applied formally and recorded, with the disclosure that AFF was selected on seed 42; no further variant is read.
3. **The GO list on seeds 49 to 51.** Proposal: round 2's seven checks unchanged (pooled 95% lower bounds above 0 for
   R@1 against cosine, RCA, B, B′ and the matched counterpart; the gain statistic; the gain against RCA). **R1 beside
   it:** either the paired AFF minus R1 difference as an eighth GO check, or descriptive only. Our proposal: descriptive
   (the claim is that AFF beats its comparators; the paired difference says whether the one-line change is worth it).
   This is the user's call.
4. **R1 on the test seeds:** computed with the same pipeline and reported descriptively (its own seven checks), never a
   second verdict.
5. **Sensitivity projection** before the seeds are built (round 2's rule §6.1 formula). On seed 42 the projection for R1
   gave SE about 0.067, detectable margin about 0.19 (computed by the round-2 tab, exploratory); compute AFF's in the run.
6. **Ride-alongs:** none as candidates. The brainstorm's ideas 2 to 4 stay out of round 3 (idea 3, GoEmotions placement
   of captions, can be measured later as a detector AUC).
7. **Disclosure:** AFF was found among about 50 label-free variants on seed 42; the median of its cluster (+0.63) is the
   better guide than +0.700; the fresh seeds are the protection.

## 5. Code, pitfalls and carry-forward items

| What | Where |
|---|---|
| Round-2 shared constants and round-1 imports | `src/test/20261118_reader_fix_round2/r2_common.py` (asserts round 2's rule SHA; round 3 needs its own copy with its own rule SHA and D14-style input table) |
| Fusion pieces (top-k sets, cells, integer cross-fits, counterpart, assembly) | `src/test/20261118_reader_fix_round2/r2_fusion.py`, `run_r2_fusion.py` |
| Rule application template | `src/test/20261118_reader_fix_round2/r2_apply_rule.py` |
| Brainstorm harness (unreviewed; reference for AFF) | `src/test/20261120_r1_levers_brainstorm/bs_lib.py`, `bs_04_readers.py` (`AFF`) |
| Round 1: bundle, B, B′, evaluation | `src/test/20261117_reader_fix_csd/common.py` (`load_bundle` is **seed-42 specific**: it asserts step-1's stored arrays) |
| Fresh-seed machinery used before (N1 test on seeds 45, 47, 48) | `src/test/20261108_new_method_quick_checks/test_seeds.py`; evaluation context `run_gonogo.EvalContext(seed, smoke)` in `src/test/20261101_aspect_factor_gonogo/`; seed builder `src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s>`; ledger `docs/superpowers/episode_seed_ledger.md` (49 and later free) |

The test-seed pipeline is the main new code: on each test seed build the episodes, B and B′(A0) with that seed's parity
halves, the standard-head features of R1's reader (round 1's `rb_eval.seed42_features` logic on the seed's episodes),
the frozen half-readers' probabilities, the gates against the frozen τ, the 224-cell cross-fits on the seed's halves
and the counterpart. Only the GO quantities before the verdict (round 2's rule §6.4).

Pitfalls (from rounds 1 and 2):
- `build_episode_bank` sorts partition keys and seeds block i with `seed + i`: key names change banks.
- z-score per ranking row first, then gate; the matched counterpart of anything gated is G_cf, never ḡ·z(T_cf).
- Round 1's folder is on `sys.path`: no round-3 module may share a name with a round-1 or round-2 module (`common`,
  `rc_core`, `rb_*`, `r2_*`, `run_*`, `apply_rule`, `test_*` of those rounds).
- Cross-fit ties: compare the integer sums ρ = Σ 4·R@1 and γ = Σ 4·gain, never float means (round 2's blocking finding).
- Round 2's slip: smoke runs printed new-cell numbers before the regression check passed. Either forbid printing
  candidate numbers in smoke before the check, or say in the rule that smoke numbers are not results.
- Final review carry-forwards (round 2 N8, N9): add one end-to-end wiring smoke test of the run scripts before any
  test-seed run (two wiring mutations survived every unit test in round 2); compare at the 0.05 carry-tie boundary with
  a +1e-12 tolerance; give τ an explicit tolerance (absolute 1e-15 or relative 1e-9) in the re-derivation agreement.
- After any subagent edit of a committed markdown report, check `git diff --stat` for formatter churn.
- Times with `TZ=Europe/Amsterdam date '+%F %H:%M'`; do not estimate them (round 2's log wrote two times off by
  minutes and had to be corrected).
- CPU only: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`, at most 3 processes, `uptime` and `free -g`
  first, `run_in_background` for real runs. The local GPU and node404 are not needed.

## 6. State at handoff

- Commits on `main` since round 2 started: dc9fac6 (rule), a221b64, cdf9b4d, ee48015, e0981ce (verdict), 9fc1c6d,
  e50da1e (report), dcea147 (final review and fixes), beda15d, 9f218bd; a3e8003 (no-caption spike); c749ee1
  (brainstorm). Nothing pushed.
- Seeds 49, 50, 51 are free (never built). No process of this line is running.
- No round-3 folder exists yet. Suggested: `src/test/20261121_round3_affect_gate/` (folder dates are sequence numbers,
  next free after 20261120), report `docs/reports/auto/v2/2026-11-21_round3_affect_gate.md`, spec
  `docs/superpowers/specs/2026-10-06-round3-affect-gate-design.md`.
