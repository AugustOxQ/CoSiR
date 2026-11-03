# Handoff: ticket 14, the final whole-branch review, the fix wave, the merge into main, then the run chat

## Brief
- Done: tickets 12 and 13 merged (integration tip d2e87d7); node402 probe (59 inputs, 0 failures) and node smoke pass.
- State: tickets 01 to 13 are merged; ticket 14 is unblocked because r17's code is now on main.
- Next: merge main into the branch, ticket 14, final review on Opus `xhigh`, one fix wave, merge into main, run chat.
- Blocked: nothing.

> Written 2026-11-04 00:48 by tab "r19 build 4" (cut at the 200k context meter). Loop step reached: step 7, build
> nearly done. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets "The user's reading of
> the experiment overview" and "Decided (A)".

## 1. Read in this order

1. `/project/CoSiR/.scratch/r19-past-picks/issues/00-index.md` (binding), then `14-r17-metrics.md` beside it.
2. `/project/CoSiR/.scratch/r19-past-picks/notes/controller_notes.md` (binding).
3. `~/.claude/references/matt-chain.md` ("The final whole-branch review", merge) and `~/.claude/rules/spec-templates.md`
   (hard rule 6), which the review brief must name. `codex-jobs.md` before a Codex job.
4. `src/test/20261103_past_picks/pp_run_sheet.md` on `r19-past-picks`: the run chat's order and commands.
5. The spec [2026-11-03-past-picks-design.md](../specs/2026-11-03-past-picks-design.md), only where a review needs it.

## 2. Decided by the user, not to be reopened

- All of the spec's §4 (D1 to D25) and A1; the run's end time (CPU scoring may run to about 13:00 on 11-04; GPU work
  ends by 10:00 on a weekday). See the r19 build handoff §2.

## 3. Where things stand

**Tickets** (integration branch `r19-past-picks`, worktree `/project/CoSiR/.claude/worktrees/r19-int`, pushed):

| # | State | Check | Merge |
|---|---|---|---|
| 01 to 11 | merged | see the build 3 and build 4 handoffs | |
| 13 diagnostics | merged | diff read; one fix (ArtELingo and Affection rows); smoke below | 7b9db85 |
| 12 runner | merged | diff read; one fix round on a Sonnet `medium` worker (no Codex room); 8 tests | d2e87d7 |
| 14 r17 metrics | not started | unblocked: r17's code is on main (`src/test/20261103_new_scores/ns_metrics.py`) | |

**The next steps, in order:**

1. **Merge main into `r19-past-picks`** in `r19-int`, so r17's modules exist there. Expect conflicts at most in handoff
   files. `git log main` shows r17's run chat is live, so main moves; take the merge once.
2. **Ticket 14**: a Sonnet `medium` worker (plumbing; D13: context rows only, no claim). Codex needs 80% for the room
   check, and it has 57% until 04:35. Add the fix from "Known gaps" item a in the same round.
3. **Final whole-branch review**: Opus `xhigh` `worker`, then one fix wave and a Sonnet `high` scoped re-review
   (`matt-chain.md`). Keep the review on ticket 12's verdict parts. Point it at "Known gaps" below.
4. **Merge** into main, push. The merge subject must carry `cluster run`, because `cluster sync` refuses a HEAD without
   it. List the worktrees and ticket branches on `.scratch/pending_deletions.md`. Write the run handoff and cut.

**Node (ticket 04's real check), done on commit 001d3d1:**
- Probe, tag `20261103-232618-001d3d1-9a592cff`: 59 inputs, 0 failures; `pp_node` and `pp_matched` import in the node
  env; 87 s.
- Smoke, tag `20261103-232644-001d3d1-78737588`: `clip_b32` seed 0, 2 epochs, c* and photo both `ok` (127 s, 82 s).
- Both are pulled. `res/r19_pp`, `res/r19_pp_smoke` and `res/r19_pp_probe` exist in `r19-int`.
- The data map covers the probe's inputs (data sync ran at 00:26). Gate (ii) and the real jobs are the run chat's.

**The smokes of tickets 12 and 13:**
- Ticket 12's smoke (`pp_run_smoke.py` 0 to 4, tree `/tmp/claude-0/-project-CoSiR/f230deb2-5b4b-4531-9369-0d454a9ed956/scratchpad/t12_smoke`):
  each stage passed; both refusals recorded (stage 2 uncommitted, rc 2; wrong status, rc 1). After the fix round only
  stage 2 was rerun.
- Ticket 13 on that tree, `pp_diag.py run`: 6 of 11 kinds ok (`semart_inner`, `r6_5_held`, `gain_share`,
  `percept_strongest`, `cost_rows`, `node_time`); 5 failed by assertion. `r0`, `clustering_kclass` and `seed_spreads`:
  the smoke's 80-key stand-in exports against MMArt's real 16,384 keys. `semart_close` and `semart_ap_type`: the
  stand-in SemArt record holds 2 of 7 encoders. Neither gave a wrong number; the first is a limit of the smoke tree.

**Known gaps (for the fix round and the review):**
- a. **`pp_diag` needs all seven SemArt entries.** `semart_close` and `semart_ap_type` reach `pp_semart_score.check_stage1`
  through `stage1_for`, which asserts seven encoders. `pp_run`'s `semart-score` sets `s.ENCODERS = tuple(doc)` for a
  partial record; `pp_diag` does not, so one failed encoder fails both kinds for all. Do the same in `pp_diag`.
- b. **ArtELingo and Affection never ran through `pp_run`.** Their scorers are proven by the r14 gates on CLIP L/14, but
  the smoke's stage 1 blocks are r14's SigLIP 1 record renamed, and its stage 2 has MMArt and SemArt only. The controller
  checked by reading: `merge_records` gives the keys `pp_score14.check_record` asks for, and the CLIs in `tasks()` match
  the `pp_score14`, `pp_mmart`, `pp_export` and `pp_matched` parsers. The review should re-derive this.
- c. **`collect_gates` ran only on stand-in gate files.** File names and pass keys were checked by reading the writers
  (`gate_i/art_*`, `mmart_*`, `semart_*`, `aff_clip_l14`, `genre_word`, `store_*`; `gate_iii.json` with `r10`, `r14`;
  `passed` for iv, vii). Real files to test against, build 3's stage 0 outputs:
  `/tmp/claude-0/-project-CoSiR/8ed85e38-0792-46e4-b566-1bfe2941a14b/scratchpad/gates/out/stage0/`.
- d. **`skip_voided`** has a unit test, but the smoke never reaches a skip.
- e. **Five `pp_diag` kinds and all held rows** have not run end to end on real-shaped data. A failed kind can be rerun
  with `--only` in the run chat; it decides nothing.

**Choices of this chat that shape a result or the run (agent default; the review and the run report show them):**
- **`PP_CPU_JOBS` default 3:** `pp_run_cpu.sh` keeps three `res-queue cpu` reservations open, because serial steps take
  about 8 h and would miss 13:00. Barriers: names ending `_merge`, `merge`, `merge_encoders`, `collect`, `decide`,
  `diagnostics`.
- **`PP.file_lock`** is new in `pp_paths.py`, and the three cells writers (`pp_score14`, `pp_mmart`, `pp_semart_score`)
  take it, after their ticket reviews. Those lines are small, but verdict files pass through them: read them.
- **Order `3 → 0 → 1 → commit → 2 → C13 → 4`:** stage 3 first since `status.json` needs gate (viii). C13 runs before
  outcomes (spec §3.4, "before any verdict is written"); `c13_check` binds it to sha256 of the cells, the probe's files
  and the records.
- **`pp_semart.record()`** gained `artists_set` after its review.
- **`pp_diag` scope:** the old four's MMArt rows are not recomputed (r15 holds them). Affection R0 is the photo model's
  validation R0, as in r14.
- **The Codex and worker notes** in `controller_notes.md` stay binding; nothing new was added there.

**Resources at 00:48:** GPU idle; CPU shared with r17's run (load about 20). node402: three GPUs free, 109 h left. GPU
check 00:48: driver 595.99.02 in the module and the library. Night mode. Codex: five-hour window 57%, resets 04:35;
weekly 83%.

**Timing:** about 2 h ahead of the build 3 estimate. Ticket 14 about 20 min, the review and its fix wave about 1 h, so
the build ends about 02:15. The run chat then starts with parity and node402's jobs. The 07:00 limit for the heavy
extraction still holds.

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- The earlier handoffs' §5 hold. New ones:
- **A `cd` in a Bash call moves the chat,** also inside a compound call. Use `( cd ...; )` or absolute paths with `git -C`.
- **zsh takes a bare `==` in `echo == x` as a command.** Use `echo '--- x'`.
- **`cluster watch` and `pull` need the worktree's cwd** in a subshell; `--node node402` on every call.
- **Codex's room check counts 40% for a fix round,** so a fix round after a large job goes to Claude.
- **Codex writes its own note files** (`pp_findings.md`, `pp_progress.md`, `pp_task_plan.md`) when it is not told not to:
  say so in the brief.

## 6. State at handoff

- Running: nothing of r19 (no subagent, no Codex job, no queued job, no cluster job).
- Uncommitted: nothing of r19's in git (`.scratch/r19-past-picks/` is outside git by design).
- On disk: node smoke outputs 249 MB and the two node job folders, on `.scratch/pending_deletions.md`; this chat's
  scratchpad 2 MB, kept for the review.
