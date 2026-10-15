# Handoff: r8 build 3: review tickets 07 and 11, run stage 0, profile FCMI, then build 04, 08, 09, 12 and 13

## Brief
- Done: tickets 01, 02, 03, 05, 06 and 10 are merged on `r8-grouping`; each verdict ticket was reviewed by Opus and fixed. Tickets 07 and 11 are built but not reviewed.
- State: two build checks hit the rule. The M4 view check fails (1 of 256 paintings at 0.99983, below 0.9999), so V100 to V33 drop. FCMI projects to about 64 min against D21's 60.
- Next: Opus reviews of 07 and 11, the full stage 0, an FCMI profile, then tickets 04, 08, 09, 12 and 13.
- Blocked: nothing. The view-check threshold is the user's call (a rule amendment), flagged for the morning.

> Written 2026-10-12 21:30 by the `r8 build 2` chat (cut at 200k tokens of context). Loop step reached: step 7, the
> build, 8 of 13 tickets built. Direction log entry: [2026-10-12 · Round 7 design L](../direction_log.md).

## 1. Read in this order

1. `.scratch/r8-grouping/notes/build_log.md`: the two rule hits (view check, FCMI time), with numbers and logs, and
   the notes for the report.
2. `.scratch/r8-grouping/issues/00-index.md`: the tickets and the shared interfaces. This chat added "Views file",
   "NBC trigger" and "Gate inputs" (the file contracts that ticket 10's reader sets), plus `r8_common`'s new names.
3. `.scratch/r8-grouping/notes/ticket_rules.md`: it now says how to run tests in a worktree (`--rootdir=.`).
4. The hand-backs `.scratch/r8-grouping/notes/handback_07.md` and `handback_11.md` (rule gaps to judge).
5. The previous handoff, [2026-10-12-r8-build-2-handoff.md](2026-10-12-r8-build-2-handoff.md), §3 and §5: the ticket
   plan, the pitfalls and the outside-code facts still apply.

## 2. Decided by the user, not to be reopened

As in the r8 build handoff (`2026-10-12-r8-build-handoff.md` §2) and both specs' §4 rows marked "user".

## 3. Where things stand

Integration branch `r8-grouping` at b4208a6, pushed; worktree `/project/CoSiR-wt/r8-grouping`. 105 tests pass there
(stage 0, gate, head, common; run from the round folder with `--rootdir=.`), plus 15 for FCMI.

| # | State | Commits | Review |
|---|---|---|---|
| 01 | merged | dd15ce2, bb640d6, f7955e6 | Opus: 1 major (test order), 8 minor, all fixed; Sonnet re-review passed; its one new minor fixed by the controller |
| 02 | merged | e5bcd65, ce036d8 | Opus: 2 major (reader inputs not on SC1's path; an unneeded hunk), 4 minor, fixed; no re-review: the full stage 0 is the proof |
| 03 | merged | 616d633 | plumbing; the controller read the crop code against M4 and ran the GPU smoke |
| 05 | merged | 3aff63e, 8f4d90b | Opus: merge, test gaps closed with an independent M6 reference test; the controller checked it |
| 06 | merged | c1eee35, b0edfa6 | Opus: 1 blocker (`pass` key), 2 major (`drop_last` as upstream, agent default; smoke past warm-up), fixed |
| 10 | merged | 2ee41e1, 0d2e1d8 | Opus: 2 major test gaps, minors, fixed; the controller re-ran one mutation (it now fails 2 tests) |
| 07 | built, branch `r8-07` | a277263 | not reviewed (Claude worker, sonnet high) |
| 11 | built, branch `r8-11` | 19d6988 | not reviewed (Codex). Check its pending-file layout (`inputs`, `input_sha256`, `expected`) and `--stage-evidence` manifest against ticket 10's writer; one side must change |
| 04, 08, 09, 12, 13 | not started | | |

- **View check (M4):** min cosine 0.999833 at p = 172; the next two are 0.999937 and 0.999939, the median 0.999997.
  The CPU gives the same value, so the cause is the cached 224 image. It is neither the GPU nor ticket 03's code.
  Under the rule, V100 to V33 drop (M16). The full views run records this as the official check. It is a run-chat
  step (ticket 13's queue): do not change the threshold.
- **FCMI time:** at full size, post-warm-up epochs take 12.8 to 13.9 s on the GPU, with CPU faiss at 8 threads.
  300 epochs project to about 64 min against the strict 60-min limit of SC2 and D21. First profile one epoch
  (k-means, network, evaluation pass). Speedups that leave the labels identical are implementation choices, for
  example faiss threads or loader workers. Prove identical labels on the smoke. If none fits, SC2 decides as the rule
  says.
- **Codex:** 19% of the five-hour window left at 21:29 (reset 01:37), 72% weekly. Until the reset, give the
  implementers to Claude `worker`s. The ledger rows are added in `claude-config/codex/jobs.md` (not committed: that
  file also holds another session's edit; sync at wrap-up).
- Timings for the run chat: head settings take about 18 min each on CPU (ticket 05 review). Groupings take about 4 min
  per ID (ticket 07). Stage 0 takes about 10 min on CPU.

## 4. Open points for the next step

- None for the user tonight (night mode).
- For the morning, key: the view check fails by 0.00007 on 1 of 256 paintings. The threshold was an agent default.
  Amending it after the result is the user's call (C7, an amendment row in spec A). Keep the rule as it is until then.
- For the report: rule M14 asks for "only the whitelist change" in the copies. They also need import adapters and the
  R0 head root, which change no number. The controller accepted this (agent default); see `build_log.md`.

## 5. Code and pitfalls

- Run tests in a worktree from the round folder with `--rootdir=.`. Otherwise R7's loader refuses the worktree's
  `src`. Pass `--basetemp` inside the session scratchpad if a test writes temp files.
- Ticket 07 accepts `passed` or `pass` in FCMI records. Drop the `passed` fallback now that 06 writes `pass`.
- Ticket 08 must use `r8_groupings.control_draws` (07, gap 6). Ticket 09 uses `r8_scoring_io.reader_inputs` (02's fix;
  noted in ticket 09). Ticket 13 must handle smoke stage records (noted in ticket 13). Ticket 04 must define its
  `meta.json` keys in the index (07, gap 7).
- `subagent-watch` reads a subagent as finished when its last transcript line has no tool call (it misfired at
  20:41). Check with `subagent-watch --all`.
- Codex fix rounds by `resume <thread_id>` worked for all four tickets. The thread ids are in the first line of
  `notes/briefs/codex_NN.log`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing of this chat in git. On main, `palette.md`, `reports_sum.md` and the stage-report files belong
  to another session; do not commit them.
- Worktrees: `/project/CoSiR-wt/r8-{grouping,01,02,03,05,06,07,10,11}`. Their `results/` hold small test leftovers.
- On disk over 1 GB: `/data/SSD2/cosir_r8/fcmi/smoke/` (8.0 GB) and `RF/results/smoke/gpu_wiring_06/` (1.6 GB), both
  on `.scratch/pending_deletions.md`.
