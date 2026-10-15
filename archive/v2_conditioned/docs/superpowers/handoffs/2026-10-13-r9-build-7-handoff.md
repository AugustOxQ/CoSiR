# Handoff: r9 build 7: 20a's fix round and merge, then 21a, 20b, 12, and the LOW fixers on Codex

## Brief
- Done: 25d, 10, 09, 18 and 11 merged; each CLIP check equals the round 6 pins byte for byte.
- State: 25 of 33 tickets merged; 20a built, stage 0a passed on it, review asks 2 small fixes.
- Next: 20a's fix round and merge; then 21a (new) and 20b; LOW fixers to Codex after 21:37.
- Blocked: nothing.

> Written 2026-10-13 20:53 by `r9 build 6` (cut at the context meter). Loop step reached: step 7, the build.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build6.md`**: what was merged, every check's number, the open LOWs per ticket,
   and the 20a fix list (section "20a: Opus review").
2. `.scratch/r9-scope/notes/review_t20a.md` (the 20a findings M1, M2, L1 to L5, and "Where the ArtELingo-28
   statistics go").
3. The tickets index `.scratch/r9-scope/issues/00-index.md` (new row 21a; 21 now blocked by 21a) and the new ticket
   `21a-a28-pass-record.md`. Build 6 notes were added at the end of tickets 12, 13, 20 and 21.
4. `.scratch/r9-scope/notes/t20_progress.md` sections 3 and 4 (the plan for 20b).
5. As before: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/codex_common.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34), the 18:01 wording changes and the 18:58 rule wording changes (rule SHA f9b07b34).
- Build 5 agent defaults (told to the user at 19:24, no objection) stand.
- Build 6 agent defaults (in `reviews_build6.md`): 11's L2 = call `RF.run_family` directly, stage 0b runs
  `cells_bundle`; 11's L4 done (float64 record, decides nothing); 18's M2 = the a28 seeds in `r9_context.ADMITTED`;
  20a's M2 = `score_seed` requires `verdict=` for held and a28; 21a split from 21 as verdict code; 10's picks file
  accepted (bytes differ from the pin only in fields no later code reads).

## 3. Where things stand

- **On `r9-scope` @ a097ffe** (pushed): 01 to 11, 14 to 19, 17b, 24, 25, 25b, 25c, 25d, 26, 28 merged (25 of 33).
- **Built, waiting:** `r9-t20` @ f657048 (20a, from e54b8b2): merge `r9-scope` in first (conflict only in
  `.claude/20261013_log.md`; `git add -f`), then the fix round in `reviews_build6.md`, stage 0a, a Sonnet re-review.
- **Frontier:** 20a fix → 21a and 20b (both after 20a) → 12 (stage 0b; 09, 10, 11 merged, so it is unblocked now),
  13, 21, 22, 23, 27.
- **Held for Codex** (five-hour window resets 21:37; weekly 82% at 19:52): 10b (10's L1, L2, L3, L5, L6), 09b (L-a,
  L-b, L-c), 11b (F1, F2, F3), 18's L-B, 25d's LOW A. All verdict code: `gpt-6.1-sol` high, one worktree each.
- **Deadline:** rule §10, a first held read not started by 2026-10-18 does not start.

## 5. Code and pitfalls

- Every merge changes `r9_module_shas`: ticket 22 reruns stage 0a on the final bytes; rerun it after any change to
  `r9_score`, `r9_bundle` or `r9_context` (20a's fix).
- A reviewer whose worktree hosts a live controller run must mutate only in a scratch copy (11's re-review spoiled a
  record's `script_sha256` that way; the rerun on the merged bytes is the valid one).
- Merges conflict in the round log and `.claude/20261013_log.md` (tracked in an ignored folder: `git add -f`). The
  union helper: `/tmp/claude-0/-project-CoSiR/2f29e134-1907-458c-b32a-999b8c5a13f7/scratchpad/union_merge.py <files>`
  (keeps both sides, ours first).
- The rederive tests are `rederive/rd9_test_*.py` (pass the glob; the folder alone collects nothing). The full
  mutation runner `rd9_mutations.py` runs on a quiet machine only (SIGINT to stop).
- Not ours, leave alone: `docs/reports/stage/2026-10-12_devlog_primer.md`, `docs/reports/assets/2026-10-12_devlog_primer/`,
  and the two round 8 results trees in the main checkout.

## 6. State at handoff

- Running: nothing (no subagents, no Codex, no GPU or CPU runs).
- Uncommitted: nothing of ours.
- On disk over 1 GB: nothing new; build 6's worktrees (r9-t09 holds 877M of CLIP replays) and its scratchpad (87M)
  are on `.scratch/pending_deletions.md`.
