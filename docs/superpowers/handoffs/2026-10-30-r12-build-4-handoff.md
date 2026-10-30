# Handoff: finish the r12 build (the GPU smoke of the queue, the final review, merge to main), then open the run chat

## Brief
- Done: all 13 r12 build tickets merged into `r12-affection` (08 fixes, 09, 10, 12, 13 in this chat); 515 fast tests pass.
- State: the queue script `ac_queue.sh` passes `bash -n` and both dry-runs; it has not run yet. Diag has not run on data.
- Next: the end-to-end smoke (`ac_queue.sh smoke <scratch>`, light GPU under the lock), fixes, the final whole-branch
  review (Opus `xhigh`), one fix wave and a re-review, merge to main, then the run handoff and the run chat.
- Blocked: nothing.

> Written 2026-10-30 22:55 by the r12 build chat 3 (cut at the context meter, 200k). Loop step reached: step 7 (build),
> all tickets merged; the smoke and the final review are left. Direction log entry: [2026-10-30](../direction_log.md).

## 1. Read in this order

1. The ticket index `.scratch/r12-affection/issues/00-index.md` and ticket 12 `.scratch/r12-affection/issues/12-queue.md`
   (`Status:` lines: every ticket is done except 12's smoke).
2. The spec, from §1 on: [2026-10-30-affection-confirmation-design.md](../specs/2026-10-30-affection-confirmation-design.md).
3. `~/.claude/references/matt-chain.md` ("The final whole-branch review"), and `~/.claude/references/codex-jobs.md`
   before a Codex job.
4. The round log on the integration branch: `src/test/20261030_affection_confirmation/20261030_affection_confirmation_log.md`
   (build decisions, readings, every check so far). The earlier build handoffs' pitfalls still hold:
   [build 1](2026-10-30-r12-build-handoff.md) §5, [build 2](2026-10-30-r12-build-2-handoff.md) §5,
   [build 3](2026-10-30-r12-build-3-handoff.md) §5.

## 2. Decided by the user, not to be reopened

All in the spec's §4 and the direction log, 2026-10-30; the spec was approved from its brief at 21:17.

## 3. Where things stand

| # | Ticket | State | Who built it |
|---|---|---|---|
| 01 to 07, 11 | store, data, extractor, exports, CRL, clustering, selection, C13 check | merged (build chats 1 and 2) | see the round log |
| 08 | seed ranges, SC table, verdict | merged; the review's four should-fix test gaps fixed (`306aaf6`); 128 tests pass with the slow replays | Sonnet `high` |
| 09 | runner `ac_run.py`, stages 0a to 3, guard | merged (diff read); guard tests with a mutation check per stage | Sonnet `medium` |
| 10 | `ac_diag.py`: `--stage diag` and `--stage 4` | merged (diff not read by the controller; its stage 4 PNG checked) | Sonnet `medium` |
| 12 | queue `ac_queue.sh` | merged; dry-runs checked; **the GPU smoke is open** | Sonnet `medium`, one fix by the controller |
| 13 | smoke mode (`AC_SMOKE`, `ac_data` truncation, `smoke_rows`, vh `--views-root` and smoke records) | merged after an Opus `high` review: pass, no must-fix | Sonnet `high` |

- Codex had no room all chat (5-hour window 58% left at 22:53, an implementer needs 80%); it resets at 23:52.
- GPU check at 22:53: driver 595.99.02 answers, module and library agree, GPU free, no lock. Run mode: night.

## 4. Open points for the next step

None for the user. For the controller:

0. **Read ticket 10's diff** (`git diff 26b124d 407990c -- src/test/20261030_affection_confirmation/`: `ac_diag.py` and
   two dispatch lines in `ac_run.py`). It was merged at the cut without the controller's diff read; the worker's
   report: diag.json keys `smoke`, `reading`, `r0_rows`, `ndcg`, `switch_column`, `per_source_map`, `per_class_ap`,
   `cstar_seed_spread`, `rows_not_picked`, `chance_map`, `percept_core`, `untied`, `inputs`.
1. **The smoke** (ticket 12's last box). From the integration worktree, in a background Bash:
   `bash src/test/20261030_affection_confirmation/ac_queue.sh smoke <scratchpad>/r12_smoke`. It sets `AC_SMOKE=2100`,
   uses draws 200 and 500, and runs every step: FI with `--limit 64` (under the scratch folder), the two parity checks,
   the three views on 2 × 2,100 keys (about 8,400 instructed images at about 13 images/s, about 11 minutes, plus CLIP),
   exports, CRL, stages 0a to 3 with vh after each, diag and stage 4. GPU steps wait for the lock. Expect the first
   run to stop at a mismatch: the stages and vh were built in parallel, and this is their first contact. A step stops
   the queue on a nonzero exit; finished steps are skipped on a rerun. Record each step's time in the round log.
2. **Readings to keep** (round log): smoke size 2,100 per split (whitening needs more than 1,024 rows per half);
   Impressions has 822 photo IDs in the store, not the spec's 827, so the figure uses 6 queries and 816 gallery.
3. **Review nits for the fix wave** (ticket 13 review): `AC_SMOKE=''` crashes at import (`ac_paths.py:22`; read it as
   `int(... or 0)`); `ac_export.py impressions` and `reinfer` skip the `--smoke` = `AC_SMOKE` check; `ac_vh.py
   --smoke N` is not checked against `AC_SMOKE`; the `real_data` test checks only that `rt_data.names` is sorted, not
   that it equals vh's `frozen_art_names`; `ac_paths.RANGES_FILE` and `PICKS_FILE` still name the round folder under
   smoke (the runner's `record_copy` redirects them; a second guard there is optional). The index and ticket 13
   should record vh's smoke records path (`records_dir`).
4. **The final whole-branch review** after the smoke passes: a `worker` on `opus` at `xhigh`, per `matt-chain.md`,
   then one fix wave and a scoped re-review (Sonnet `high`), then merge `r12-affection` into main and push.
5. **Then the run handoff** with the ticket list (index table) and the run chat (`loop-next`), after the GPU check
   (night mode: the check only). The run is planned for Saturday 2026-10-31 (spec §9).

## 5. Code and pitfalls

- Integration worktree: `/project/CoSiR/.claude/worktrees/r12-int` on `r12-affection` (pushed, `92d6496`). Main is
  untouched. Agent-tool worktrees start on main: each brief tells the implementer to `git checkout -B r12-tNN
  r12-affection`.
- Smoke mode needs both env vars: `AC_SMOKE=N` and a non-default `AC_OUTPUTS` (import asserts it). `ac_run.py
  --smoke N` must equal `AC_SMOKE`. In smoke the runner writes the record copies to `OUTPUTS/records/`, and vh reads
  them there; the committed files are never touched.
- Stage 0a in smoke records the replays, the FI merge, 0a(ii)(b) and the label counts as "skipped in smoke". In a full
  run the replays (about 30 minutes) run only when every other item passes.
- Workers cannot run `bash <script>` here (the sandbox guard refuses a script that contains `git`): the controller
  runs the queue and its dry-runs.
- Committed records need `git add -f` (the repo ignores `*.json`); the queue does it in full mode.
- Workers must not run CPU jobs over about 4 minutes; say so in each brief. Slow tests run under the controller.
- Unattended: delete nothing; this chat's worktrees and branches are on `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no GPU lock, no background command).
- Uncommitted: nothing of this chat's in CoSiR (`.scratch/` is outside git). `.claude/20261030_log.md` has the
  per-day change log entries of tickets 08 fixes, 10 and 13 on `r12-affection`.
- On disk over 1 GB: nothing new.
