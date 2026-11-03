# Handoff: build r19 from its 14 tickets (dispatch 01, 03, 11 first), then the final review and the run chat

## Brief
- Done: code maps of the reused code, the shared paths module (committed), 14 tickets with shared interfaces.
- State: integration branch r19-past-picks pushed; no ticket started; r17 not merged yet (ticket 14 waits).
- Next: dispatch tickets 01, 03 and 11 (Codex has room again after 23:34), then the frontier per the index.
- Blocked: nothing.

> Written 2026-11-03 21:27 by tab "r19 build" (cut at the 200k context meter). Loop step reached: step 7, tickets
> written, build not started. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullets "The
> user's reading of the experiment overview" and "Decided (A)".

## 1. Read in this order

1. The ticket index `/project/CoSiR/.scratch/r19-past-picks/issues/00-index.md`: the ticket table, dispatch order,
   routing, the shared interfaces and the controller decisions (binding). Then each ticket as you dispatch it
   (`issues/01-*.md` to `14-*.md`; about 1.5 KB each). `.scratch` lives only in the main checkout.
2. The spec from §1 on: [2026-11-03-past-picks-design.md](../specs/2026-11-03-past-picks-design.md) (rule §6 with
   §3.4 and §3.5). The previous handoff, [2026-11-03-r19-build-handoff.md](2026-11-03-r19-build-handoff.md), §2, §3
   and §5 (decisions, stores, timing, pitfalls) still hold.
3. `~/.claude/references/matt-chain.md` (the chain; you dispatch, review, merge, final review) and
   `~/.claude/references/codex-jobs.md` before any Codex job.
4. The common briefs `notes/implementer_brief.md` and `notes/reviewer_brief.md` beside the issues: every dispatch
   points the subagent at them and at its ticket, nothing more.
5. Only when a ticket needs it: the code maps `notes/code_map_{a_stores,b_training,c_scoring}.md` (B and C are
   partial: their "Not checked" lists are open). The implementers read them; you need not.

Skills: `cluster-run` (tickets 04 and 05's node smoke; read its SKILL.md before the first cluster step).

## 2. Decided by the user, not to be reopened

- All of the spec's §4 (D1 to D25) and A1; the run's end time (CPU scoring may run to about 13:00 on 11-04; GPU work
  ends by 10:00 on a weekday). See the previous handoff §2.

## 3. Where things stand

- Integration branch `r19-past-picks` (from main d8a054d), worktree `/project/CoSiR/.claude/worktrees/r19-int`, pushed;
  one commit: `pp_paths.py` (37467ce + the RB_OUTPUTS line, see the branch tip). It points `BS_OUTPUTS`, `MT_OUTPUTS`,
  `EG_OUTPUTS`, `RB_OUTPUTS` at r19's OUT before the older modules load (checked: importing it then `bs_paths`,
  `mt_paths`, `eg_paths`, `rb_paths` gives r19's paths).
- Code maps (from three Sonnet `medium` workers, 21:15 to 21:22): register-then-call works for r14's and r15's code
  (module tables read at call time); the store readers, `mt_paths.ENCODERS` (a tuple) and some asserts need patches
  or copies; DINOv2's M5 exists only at sample scale; `eg_matched.run(..., gated=False)` makes it in seconds per seed;
  gate (ii) reference values are in code map B Q4; the SemArt rule recomputes to 11,162 / 2,717 and 11,159 after md5.
- Codex at 21:25: five-hour window 37% (no room for an implementer: needs 80%), resets 23:34; weekly 90%. Until then
  Claude workers (verdict code Sonnet `high`, plumbing Sonnet `medium`; reviews Opus `high`).
- Queues at 21:25: GPU idle; CPU shared with r17's build (two short jobs). node402 not checked since 20:44.
- r17's build 2 chat is mid-build (tickets 04 to 11 open); its module will likely not merge before r19's final review,
  so ticket 14 goes to the run chat (it skips those rows and notes it).
- GPU check at 21:25: driver 595.99.02 in the module and the library, no compute process. Night mode.

## 4. Open points for the next step

None for the user.

## 5. Code and pitfalls

- The previous handoff §5 holds (keys, git, queues, `cluster` on PATH is Graphviz's, frozen inputs only in
  `/project/CoSiR`, `git add -f` for JSON records, commit only your own paths).
- Implementers run in background worktrees (`isolation: "worktree"`), at most three at once; each resets onto
  `r19-past-picks` and branches `r19-tNN-<slug>`. Merge clean branches yourself; a merger subagent only on a conflict.
- Every verdict-code ticket (01, 03, 05 to 11) gets an Opus `high` review before its merge (common reviewer brief).
- Keep `subagent-watch --watch` running in a background Bash while subagents run.
- Timing estimate: wall time so far was short (tickets done by 21:27), so the build should end by about 02:00 to
  03:00; the run chat then starts the Affection extraction first. Consider starting the extraction queue from the
  build only if the build runs late enough to threaten the 07:00 start of the heavy extraction; record it if so.

## 6. State at handoff

- Running: nothing of r19 (no subagent, no Codex job, no queued job).
- Uncommitted: nothing of r19's in git (`.scratch/r19-past-picks/` is outside git by design).
- On disk over 1 GB: nothing new.
