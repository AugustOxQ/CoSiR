# Handoff: r8 run: get the user's M4 answer, then run the two queues from the main checkout

## Brief
- Done: the r8 build is complete. All 13 tickets are merged into main. The final review passed after one fix wave.
- State: 244 tests pass. The end-to-end smoke reached the gate verdict, and the C13 re-derivation agreed at both steps.
- Next: ask the user about the M4 view check first. Then start the GPU and CPU queues from the main checkout.
- Blocked: the M4 threshold (it fails by 0.00007 on the smoke) waits for the user. No real step runs before the answer.

> Written 2026-10-12 23:27 by the `r8 build 5` chat. Loop step reached: end of step 7 (build); next steps 8 and 9.
> Direction log entry: [2026-10-12 · Round 8](../direction_log.md).

## 1. Read in this order

1. This handoff, §3 to §5.
2. `src/test/20261129_r8_grouping/DECISION_RULE.md` §8 (outcomes) and §9 (order of work). The rule governs.
3. `.scratch/r8-grouping/notes/build_log.md`, section "r8 build 5": the smokes, the reviews, the fixes, with times.
4. `.scratch/r8-grouping/notes/final_review.md` and `final_review_2.md`: what the final review re-derived and its
   minor rows (some belong in the run's report).
5. `.scratch/r8-grouping/issues/13-queue-smoke.md` and `notes/handback_13.md` ("Commands", "Timing estimates").

## 2. Decided by the user, not to be reopened

As in the r8 build handoff ([2026-10-12-r8-build-handoff.md](2026-10-12-r8-build-handoff.md) §2) and both specs'
§4 rows marked "user". The cut-off for a held-read verdict is Fri 2026-10-23.

## 3. Where things stand

- **Code:** `r8-grouping` (tip 12619c6) merged into main at 8e556d6, pushed. 244 tests pass, 1 skipped (each test
  file alone, from the round folder, with `--rootdir=.` in a worktree).
- **Stage 0 (SC1):** passed; record `src/test/20261129_r8_grouping/results/stage0_regression.json` (untracked; do not
  commit). The final review re-derived it: 128 comparisons, every difference 0.0, 43 input hashes equal at both ends.
- **Final review** (Opus, xhigh, two parts because of the context limit): no blocker. Majors fixed in one wave and
  re-reviewed (Sonnet): the NBC-dropped branch, the CPU queue crash on a missing views `meta.json`, the FCMI load-wait
  cap, 12's `harm` for an NBC target. Re-derived: the stage 0 record, the known answer of 11 (65 of R7's arrays
  exact), the FCMI loader identity (labels, spaces and losses equal, old vs new loader), the placement and the
  random controls on the smoke tree (0 differences).
- **End-to-end smoke** (fresh trees under `/data/SSD2/cosir_r8/`): smoke 2 ran every step to `gate_verdict.json`,
  with the NBC trigger fired and both C13 records `agree` (0 disagreements). Smoke 3, on the merged code (`e2e_smoke_c`, 23:08 to 23:27): GPU 12 ok + views dropped under the rule, CPU 58 ok + 3 skipped, both C13 `agree` (0), `harm` ran (target NB100), `descriptive.json` and `gate_verdict.json` written. The real dry-runs from main print 13 and 81 steps.
- **FCMI time (D21, 60 min):** about 54.5 min per run on a quiet machine, about 58 min under small load.

## 4. Open points for the next step

- **Key, for the user, before any real step: the M4 view check.** On the smoke the minimum cosine was 0.99983
  against the rule's 0.9999 (1 of 256 paintings fails). The real check is the same test, so V100, V67 and V33 will
  drop under the rule as written (§3 M4: "reported as a fair mechanic"). Two choices:
  (a) keep the rule: V drops, the run starts at once;
  (b) amend M4 (C7: a rule change after seeing a check result; record why): edit `DECISION_RULE.md`, the
  `RULE_SHA256` constant in `r8_common.py` and the threshold in `r8_views.py`, rerun stage 0 (about 13 min), then run.
  Every script asserts the rule's SHA-256, and the views `meta.json` is write-once, so nothing that writes a record
  may run before the answer. Ask the user for their choice and one line of why before showing a recommendation.
- **Minor, process (for the wrap-up):** build 4's fixer read the context meter's hand-back message as untrusted
  output and ran to about 236k tokens. This chat's briefs said "the context meter is ours", and both Opus reviewers
  handed back on it. The `worker` body could carry that line.
- **Minor, for the run's report:** D2 "also reported as a comparator" and ED18 "at most 3 h" are not enforced in
  code (final review 2). 06 → 07 and 04 → 07 never ran at full size in the smoke: the real run is their first.

## 5. Code and pitfalls

- **Commands** (main checkout, round folder, `R8_RESULTS` unset; the queue asserts the real results root):
  `./run_r8_queue_gpu.sh --dry-run`, `./run_r8_queue_cpu.sh --dry-run`, then each queue as one background Bash
  (`run_in_background: true`) that ends with one summary line. Logs in `results/logs/`. Do not pipe the wrapper
  into `tail`: that hides its exit code (0 ok, 1 a step failed, 3 SC1 stop, 4 GPU load cap).
- **FCMI needs a quiet machine.** The queue keeps heavy CPU steps away from each FCMI run and waits for a 1-minute
  load below 4.0 (`R8_LOAD_MAX`), at most 2 h (`R8_LOAD_WAIT_MAX`); at the cap the GPU queue exits 4 and a restart of
  both queues resumes (steps with outputs are skipped). Check `uptime` before the start; nothing else heavy (reviews,
  other sessions' jobs) beside an FCMI run.
- **A crash without a record holds everything.** One FCMI or encoder run that dies without its stage record fails
  its `part_<X>` step, and `collect` blocks the whole CPU chain, the main line included (rule §8 "not covered: stop";
  final review m4). Fix and restart; the resume skips finished steps.
- **A C13 disagreement stops the queue** before finalize, by design. Do not edit 10 or 11 to make them agree.
- **Rule-covered drops** (M4 views, an ESC1 failure) show in the summary as "dropped under the rule", not as failures.
- **The queues share** `/tmp/gpu0.lock` and `/tmp/r8_queue_{turnstile,quiet}.lock`; check `nvidia-smi` first.
- **Tests in a worktree:** run from the round folder with `--rootdir=.` and a scratch `--basetemp`; otherwise R7's
  loader refuses the worktree's modules.

## 6. State at handoff

- Running: nothing.
- Uncommitted: on main, untracked `src/test/20261129_r8_grouping/results/` (the real stage 0 record); do not commit it.
- Worktrees: `/project/CoSiR-wt/r8-{grouping,01,...,13,fix5,fix5b}`; scratch worktrees in the build 5 session's
  scratchpad (`wt_fcmi_old`, `wt_smoke_b`). All are listed on `.scratch/pending_deletions.md`.
- On disk over 1 GB: the three smoke trees `/data/SSD2/cosir_r8/e2e_smoke{,_b,_c}` (about 1 GB each) and 20 FCMI smoke
  checkpoints of this chat in `/data/SSD2/cosir_r8/fcmi/smoke/` (about 2.2 GB each); on `.scratch/pending_deletions.md`.
