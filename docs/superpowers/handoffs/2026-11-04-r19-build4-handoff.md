# Handoff: node probe and smoke, tickets 12 and 13, then the final review and the run chat

## Brief
- Done: tickets 04, 08, 10 built, reviewed, fixed and merged; tickets 01 to 11 are all merged (integration tip 001d3d1).
- State: gates (iv) and (vii) r15 parts and gate (iv) SemArt part replay at 0.0; all three Affection stores written.
- Next: node402 probe and smoke; ticket 12 (Codex), ticket 13 (Claude); final review; merge; run chat.
- Blocked: nothing.

> Written 2026-11-04 00:20 by tab "r19 build 3" (cut at the 200k context meter). Loop step reached: step 7, build
> nearly done. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets "The user's reading of
> the experiment overview" and "Decided (A)".

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r19-past-picks/issues/00-index.md` (binding), then tickets `12-runner.md`,
   `13-diag.md`, `14-r17-metrics.md` beside it.
2. **`/project/CoSiR/.scratch/r19-past-picks/notes/controller_notes.md`**: binding, above all the new section "From
   build 3" (Affection stores, node commands, probe order, the quiet CPU rule, gate file keys) and the two lines from
   the t08 review. Point every new implementer at it.
3. [2026-11-03-r19-build3-handoff.md](2026-11-03-r19-build3-handoff.md) §5, and the build 2 handoff §2 and §5.
4. `~/.claude/references/matt-chain.md` (final review, merge) and `codex-jobs.md` before a Codex job.
5. The spec [2026-11-03-past-picks-design.md](../specs/2026-11-03-past-picks-design.md), only where a review needs it.

Skills: `cluster-run` (read its SKILL.md before the first cluster step).

## 2. Decided by the user, not to be reopened

- All of the spec's §4 (D1 to D25) and A1; the run's end time (CPU scoring may run to about 13:00 on 11-04; GPU work
  ends by 10:00 on a weekday). See the build handoff §2.

## 3. Where things stand

**Tickets** (integration branch `r19-past-picks`, worktree `/project/CoSiR/.claude/worktrees/r19-int`, pushed):

| # | State | Review | Merge |
|---|---|---|---|
| 01, 02, 03, 05, 06, 07, 09, 11 | merged | see the build 3 handoff | |
| 04 node jobs | merged | controller diff read; gate (ii) self-compare on r14's files 0.0; M5's 28 reads covered | acc6822 |
| 10 cost probe | merged | CHANGES NEEDED, no blocker; nine findings fixed on Codex; 46 tests; GPU re-smoke ran | 61f7e2d |
| 08 MMArt | merged | APPROVE with notes; picks and draw guards fixed on Codex; 32 tests | 001d3d1 |
| 12 runner, 13 diagnostics | not started | unblocked now | |
| 14 | waits for r17's merge into main | | |

**The next steps, in order:**

1. **Node (ticket 04's real check).** Work in `r19-int`; its HEAD subject has the `cluster run` marker `cluster sync`
   asks for. Use `CL=~/.claude/skills/cluster-run/cluster`, always `--node node402`. `$CL sync`, then `check
   --sync-data` and `launch --queue` for the probe, then the same for the smoke; `bash
   src/test/20261103_past_picks/pp_node_plan.sh` prints every command. The probe is the first time the node env
   imports `pp_node` and `pp_matched`. Before the first pull, set the `res/r19_pp*` symlinks (plan step 6). Then
   `watch`, `pull --tag`. Gate (ii) and the real jobs are the run chat's.
2. **Ticket 12**: Codex if `codex-usage --room implementer` says room (80% left at 00:18, needs 80%; it resets 04:35),
   otherwise a Sonnet `medium` worker. The run sheet must carry the quiet CPU rule and skip the Affection step.
3. **Ticket 13**: a Sonnet `medium` worker (the check allows one Codex implementer at a time), after or with 12; it
   smokes on ticket 12's smoke outputs.
4. **Final whole-branch review**: Opus `xhigh`, then one fix wave and a Sonnet `high` scoped re-review
   (`matt-chain.md`). Keep the review on 12's verdict parts. Then merge `r19-past-picks` into main, push, list the
   worktrees and ticket branches on `.scratch/pending_deletions.md`, write the run handoff, cut. Ticket 14 goes to the
   run chat if r17 has not merged.

**Gate checks run in this chat** (outputs in `/tmp/claude-0/-project-CoSiR/8ed85e38-0792-46e4-b566-1bfe2941a14b/scratchpad/gates/out/stage0/`, not the real OUT; from
commit 13572b5, before the ticket 08 guards, which changed no statistic):
- Gate (iv), r15 part: passes; max difference 0.0 over 180 items, intervals included (30 min).
- Gate (vii), r15 part: passes; max difference 0.0 over 367 items (17 min).
- Gate (iv), SemArt part: passes; max difference 0 (5 min).
- All replay gates now passed at least once in the build. The run chat reruns stage 0 on the final code.
- Gate (viii) has not passed yet; see below.

**The Affection stores** (`/data/SSD2/pre_extract/affection_local/{clip_b32,clip_b16,dinov2_large}`) are written in
the real OUT's status file and pass their store checks (85,007 rows each; 662, 350 and 420 s).

**Choices of this chat that shape a result (agent default; the final review and the run report should show them):**
- **Gate (viii) and CPU load.** The GPU smokes gave Qwen default end to end 225 ms at load about 45 and 167 ms at load
  about 30, against the 86.3 ms reference (bar 69.0 to 103.6). Load is the likely cause, not proven: no run at low load
  has happened. The code records load per repeat and never discards on it, since the spec names only GPU processes.
  Rule set here: start the probe at a 1-minute load under about 16; a fail with recorded load over 16 is rerun on a
  quiet CPU, and only a quiet fail counts for the voids table (SC15 void). If the quiet run also fails, the cause is
  in the code, not the load: debug it, do not widen the bar.
- **The extraction ran in the build**, Affection only, since the node smoke's photo step needs `clip_b32`'s store.
- **The data map** (`~/.claude/skills/cluster-run/cluster.conf`, local, not in git) has 16 new `/local/wding/r19_in/*`
  entries; the copy before the edit is `.../scratchpad/cluster.conf.before_r19` in this chat's scratchpad.
- **`passes_per_aspect`** in the cost table: c* 1/3 of an encoder pass, teacher 1 (Codex's reading of D8).

**Resources at 00:18:** GPU idle; CPU shared with r17's build (load about 30). node402 at 23:15: three GPUs free, 110 h
left. GPU check 00:18: driver 595.99.02 in the module and the library. Night mode. Codex: five-hour window 80%, resets
04:35; weekly 87%.

**Timing:** about 1 h ahead of the build 3 estimate. Node probe and smoke about 30 min, 12 and 13 about 1 h, the final
review and its fix wave about 1 h, so the build ends about 02:30. The run chat then starts with node402's jobs; the rest
of the extraction and the probe after those; the 07:00 limit for the heavy extraction still holds.

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- The earlier handoffs' §5 hold. New ones:
- **A reviewer's sabotages must not touch a gate job:** run gates and GPU smokes from a detached checkout (`r19-gates`,
  `r19-t10-smoke` exist; `git -C <it> checkout --detach r19-past-picks` moves one).
- **A `cd` in a foreground Bash call persists** and moves the chat into a worktree: use `( cd ...; )` or `git -C`.
- **`cluster sync` refuses a HEAD without `cluster run`** in its subject; put it in a real merge's subject, no empty
  commit. So each later merge that must reach the node carries it.
- **Codex fix rounds cost 4 to 14% of the window**, far under the 40% estimate, but the room check counts 40.
- Tests of `pp_mmart` run piecewise (`-k`) under load.

## 6. State at handoff

- Running: nothing of r19 (no subagent, no Codex job, no queued job, no cluster job).
- Uncommitted: nothing of r19's in git (`.scratch/r19-past-picks/` is outside git by design).
- On disk over 1 GB: the build 2 chat's scratchpad (already on `.scratch/pending_deletions.md`); this chat left 239 MB.
