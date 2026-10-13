# Handoff: r9 build 6: reviews of 09, 10, 18 and 11's fix round, then merges; 20a, 20b and 25d

## Brief
- Done: 06, 08, 16, 17b, 25c merged; stage 0a passed on 06's fixed code; 08's L1 passed both identity gates on the GPU.
- State: 09, 10, 11, 18 built and checked on CLIP; 11 needs one fix round; 20 split into 20a and 20b.
- Next: Opus reviews of 09, 10, 18; 11's fix round; then 20a, 20b and 25c's LOW follow-ups (25d).
- Blocked: nothing.

> Written 2026-10-13 19:58 by `r9 build 5` (cut at the context meter). Loop step reached: step 7, the build.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build5.md`**: the open work per ticket (09, 10, 11, 18, 20a/20b, 25d), what each
   review must check, and the frontier.
2. `.scratch/r9-scope/notes/t20_progress.md`: the plan for 20a and 20b (the first implementer's reading, no code).
3. The tickets index `.scratch/r9-scope/issues/00-index.md` (new rows 20a, 20b). Build 5 notes were added at the end of
   tickets 10, 12, 13, 20, 21 and 22.
4. The round log `src/test/20261202_r9_scope/20261202_r9_scope_log.md`, "Build 5".
5. As before: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/codex_common.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34), the 18:01 wording changes and the 18:58 rule wording changes (rule SHA f9b07b34).
- Build 5 agent defaults, told to the user at 19:24 with no objection: rule §3.3's "every translation shard" is pinned
  literally (four DAS6 shards plus the merged folder); `near_duplicates_a28.json` stays out of `pins_a28.json` (25d
  records its SHA in the C13 agreement instead); 08's DAS6 input gap is a note for ticket 13.

## 3. Where things stand

- **On `r9-scope` @ f3f716f** (pushed): 01 to 08, 14 to 17, 17b, 19, 24, 25, 25b, 25c, 26, 28 merged.
- **Built, waiting:** `r9-t09` @ 2a8f085 and `r9-t11` @ e937c4e (both from 06's pre-fix branch 01b6fe3: merge
  `r9-scope` in first), `r9-t10` @ ec3f176 (from 06's fixed branch 4d65990), `r9-t18` @ 70fc8a7 (from d7219bb).
  `r9-t20` @ d7219bb is clean (use it for 20a).
- **CLIP checks done in build 5:** 09's chain byte-equal to `pins_clip.json` (τ exact; reader JSON differs); 10's
  means and λ picks exact (picks file SHA ccc35c41… vs pinned df1fff78…: to judge); 11's cells byte-equal.
- **Frontier:** reviews of 09, 10, 18; 11's fixes; 20a; 25d. Then 12 (after 09 to 11), 20b, 13, 21, 22, 23, 27.
- **Codex:** 4% of the five-hour window at 19:52 (reset 21:37), weekly 82%. After the reset it can take 20b, 25d or
  the plumbing of 12/13.

## 5. Code and pitfalls

- `r9_module_shas` hashes every `r9_*.py`: each merge makes a stage 0a pass stale. Ticket 22 reruns it on the final
  bytes; rerun it after any change to `r9_score`, `r9_bundle` or `r9_context` (20a, 18).
- Merges conflict only in the round log: keep both sides, ours first (`union_merge.py` pattern: drop the markers).
- Workers sometimes write logs into the main checkout or the gitignored `.claude/`: check `git status` after each.
- Not ours, leave alone: `docs/reports/stage/2026-10-12_devlog_primer.md` (modified) and
  `docs/reports/assets/2026-10-12_devlog_primer/` (untracked) in the main checkout, from another session.
- Implementers of tickets with long real-data runs (09, 10, 11) used 150k to 210k tokens; 20 ran out reading. Keep
  briefs pointing at notes, and give the long CLIP runs to the controller.

## 6. State at handoff

- Running: nothing (no subagents, no Codex, no GPU work).
- Uncommitted: nothing of ours (the two round 8 results trees and the devlog files above are not ours).
- On disk over 1 GB: nothing new; build 5's scratchpad (138 MB) and merged worktrees are on `.scratch/pending_deletions.md`.
