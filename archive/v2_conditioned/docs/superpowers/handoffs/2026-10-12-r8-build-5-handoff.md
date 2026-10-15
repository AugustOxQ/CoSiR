# Handoff: r8 build 5: read and merge ticket 13, run the end-to-end smoke, then the final whole-branch review

## Brief
- Done: 12 of 13 tickets are merged (08, 11, 12 this chat). The FCMI loader change is merged; FCMI labels did not change.
- State: FCMI now projects to about 54.5 to 58 min per run against D21's 60. 11 reproduces 65 of R7's arrays exactly.
- Next: read ticket 13's diff, merge it, run the end-to-end smoke, then the final review on Opus at xhigh.
- Blocked: nothing. The view-check threshold (M4) still waits for the user in the morning.

> Written 2026-10-12 22:27 by the `r8 build 4` chat (cut at 200k tokens of context). Loop step reached: step 7, the
> build. Direction log entry: [2026-10-12 · Round 7 design L](../direction_log.md).

## 1. Read in this order

1. `.scratch/r8-grouping/notes/build_log.md`, section "r8 build 4": everything this chat did, with times and commits.
2. `.scratch/r8-grouping/notes/handback_13.md`: ticket 13 as built (queues, smoke mode, smoke limits, timings).
3. `.scratch/r8-grouping/notes/rereview_11.md` and `handback_11_fix.md`: ticket 11's state.
4. `.scratch/r8-grouping/issues/00-index.md`: two new shared items, "Centres" and "Late drops".
5. The previous handoffs, [build 4](2026-10-12-r8-build-4-handoff.md) §4 to §5 and
   [build 3](2026-10-12-r8-build-3-handoff.md) §3 to §5. Their view-check facts and pitfalls still apply.

## 2. Decided by the user, not to be reopened

As in the r8 build handoff (`2026-10-12-r8-build-handoff.md` §2) and both specs' §4 rows marked "user".

## 3. Where things stand

Integration branch `r8-grouping` at 8d33ce7, pushed; worktree `/project/CoSiR-wt/r8-grouping`.

| # | State | Commits | Check |
|---|---|---|---|
| 01 to 07, 09, 10 | merged | see the build 3 and build 4 handoffs | as there |
| 06 follow-up | merged 7e95fe1 | 3f005e5 | batched loader; byte-identical batches over 20 real epochs; CPU smoke labels and losses identical to the old loader |
| 08 | merged 3e4d6e9 | 76ed6aa, febc87a | Opus review (no blocker; arrays re-derived exactly), Sonnet fix (`--h-only`, 3 guards, 5 minors), Sonnet re-review: merge |
| 12 | merged ff4e6ea, b94dde6 | 43eb847, bd1ac40, ... | plumbing, controller read; aligned (see below) |
| 11 | merged 8d33ce7 | 19d6988, c71da06 | Opus review (6 blockers), Sonnet fix, known answer 65 arrays exact (controller rerun), Sonnet re-review: merge (15 of 18 mutations caught; survivors covered by the known answer or smoke-only) |
| 13 | built, branch `r8-13` at ca91ff7 | 13a0116 | plumbing: controller read of `run_r8_queue.py` (717 lines) still to do, then merge |

- **Alignments by the controller** (r8-12, merged): `r8_scoring_io.load_groupings` reads H<X> as X's labels and finds
  NBC/ZNBC in `r8_groupings_nbc.npz`; `run_r8.py` `grouping_records` merges `r8_groupings_nbc.json` and
  `split_dropped` resolves H<X> to X (09 would have crashed on the `nbc` and `hx` batches);
  `run_r8_descriptive.py` reads late-drop files, skips late-dropped IDs in the label part, and its smoke `harm` needs
  `--root results/smoke`. These touch verdict-adjacent code (02's reader adapter): the final review covers them.
- **FCMI time (D21):** with the new loader, clean epochs 10.8 to 11.0 s, epochs under small load 12.4 to 13.0 s: about
  54.5 to 58 min per 300-epoch run. Ticket 13's queue keeps heavy CPU jobs away from every FCMI run (turnstile
  locks), waits for a 1-minute load below 4.0 before each run, and runs CPU jobs under `nice -n 10`.
- **Codex:** 16% of the five-hour window left at 21:56 (reset 01:37), 72% weekly.

## 4. Open points for the next step

- **Ticket 13 read and merge:** check the order and stops against rule §8 and §9, the turnstile and load guard,
  the `# interface: 08` and `# interface: 11` call lines against the merged 08 and 11 CLIs, and its two smoke-only
  edits (`r8_head.py` line 300, `r8_groupings.py` `check_views_meta(path)`; both read).
- **End-to-end smoke** (controller, background, GPU parts under the lock): the commands are in `handback_13.md`, on
  a fresh tree `R8_RESULTS=/data/SSD2/cosir_r8/e2e_smoke/results`. Its limits: 07 and 08 run at full painting count
  (09 to 12 read all scorer-train rows), FC1 to FC6 and the encoders leave the smoke gate (listed as dropped), the
  views smoke runs on all 36,518 paintings. A smoke failure goes to a Sonnet fixer (plumbing) or the ticket's own
  route (verdict code).
- **Final whole-branch review** (`final-review.md`): `worker`, Opus at `xhigh`, brief naming both specs, the rule,
  the tickets, the index, hard rule 6 and the constitution; it re-derives load-bearing numbers (stage 0's record,
  the known answer, the FCMI label identity). Then one fix wave and a scoped Sonnet re-review; then merge
  `r8-grouping` into main and push; then the handoff to the run chat (`loop-next ... --label "r8 run"`).
- **For the run chat** (put in the build's final handoff): run the GPU and CPU queues from the main checkout with
  `R8_RESULTS` unset; FCMI needs a quiet machine; the M4 threshold decision comes from the user first.
- **For the morning, key, unchanged:** the M4 view check fails by 0.00007 on 1 of 256 paintings; V100 to V33 drop
  under the rule. Amending the threshold is the user's call (C7).
- **Process (for the wrap-up):** ticket 11's fixer ignored the context meter's hand-back message (it read it as
  untrusted tool output) and ran to about 236k tokens. The `worker` body could say that the context meter is ours.

## 5. Code and pitfalls

- As build 4 §5. Also: the FCMI worker asserts `CUBLAS_WORKSPACE_CONFIG=:4096:8`, even on the CPU.
- Run 11's tests from the round folder: `test_r8_rederive.py` sits there, not in `rederive/`.

## 6. State at handoff

- Running: nothing. All subagents have finished.
- Uncommitted: nothing of this chat. On main, untracked `src/test/20261129_r8_grouping/results/` holds the real
  stage 0 record; do not commit it.
- Worktrees: `/project/CoSiR-wt/r8-{grouping,01,...,13}`.
- On disk over 1 GB: two FCMI CPU smoke checkpoints (4.4 GB, `/data/SSD2/cosir_r8/fcmi/smoke/`), on
  `.scratch/pending_deletions.md` with the build 3 items.
