# Handoff: r9 build 5: Opus review and merge of 06; 25c's fix round and review; 17's small fixes; then 08 and 16

## Brief
- Done: 05, 07, 14, 15, 17, 19, 25b merged; ArtELingo-28 stores built and stage-1 checks passed on three backbones.
- State: 06 built (stage 0a PASSED, 131 items, 0 differences), awaiting its Opus review; 25c built, awaiting one fix.
- Next: Opus review of 06, then merge; fix 25's `value_codes` bug inside 25c, then its Opus review; 17's LOW fixes.
- Blocked: nothing. The user approved two rule wording changes (painting join by nearest match; graph GPU-made).

> Written 2026-10-13 19:08 by `r9 build 4` (cut at the context meter). Loop step reached: step 7, the build.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build4.md`**: the open work per ticket (06, 25c with 25's MEDIUM bug, 17's LOW
   follow-ups), what each review must check, and the frontier.
2. The tickets index `.scratch/r9-scope/issues/00-index.md` (new row 25c). Build 4 notes were added at the end of
   tickets 11, 12, 15, 20, 21 and 23.
3. The round log `src/test/20261202_r9_scope/20261202_r9_scope_log.md`: "Rule wording changes" (both) and "Build 4".
4. As before: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/codex_common.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34) and the 18:01 wording changes (handoff of build 4).
- 2026-10-13, after the recommendation: rule §3.3, the painting join is a nearest-match test over our 61,402 ArtELingo
  paintings (option A; B was a 0.92 threshold, C dropping H8 to H10); the stage-1 record also counts near-duplicates
  at ≥ 0.92, stated in the report with the version gap. Rule §3.2 item 1: the content graph is GPU-made. Rule SHA-256
  now f9b07b34 (pins updated in `r9_common.py`, `test_r9_stats.py`, `rederive/rd9_inputs.py`).

## 3. Where things stand

- **On `r9-scope` @ d21a464:** 01 to 05, 07, 14, 15, 17, 19, 24, 25, 25b, 26, 28 merged.
- **Waiting:** `r9-t06` @ 9b0e3a4 (on e8d16ee: rebase before the merge), `r9-t25c` @ 7328543 (on 366cc2f).
- **Run outputs (main checkout, gitignored):** `results/read_set_a28.parquet` and the other prepare files,
  `stage1_checks_<B>.json`, `near_duplicates_a28.json`. Stores in `/data/SSD2/pre_extract/artelingo28/<B>/` (553 MB).
- **Frontier:** 08 and 16 now; 09, 10, 11, 20 after 06; 18 after 06 and 16.
- **Codex:** 4% of the five-hour window at 19:00 (reset 21:37), weekly 82%. Ledger rows written for 15 and 25c
  (25c: about 7% of the window).

## 5. Code and pitfalls

- Merges conflict only in the round log so far: keep both sides, ours first.
- 06a's worker wrote its log entry into the main checkout (with a stray shell fragment; reverted): check `git status`
  in the main checkout after each hand-back.
- `test_r9_context.py` takes 226 s: split it or run it in the background.

## 6. State at handoff

- Running: nothing (no subagents, no Codex, no GPU work). The 30-minute timer of the last question may still fire:
  ignore it.
- Uncommitted: nothing on `r9-scope` except the untracked round 8 results trees, as before.
- On disk over 1 GB: the build 4 session scratchpad (1.1 GB, reviewers' copies), listed on
  `.scratch/pending_deletions.md` with the merged worktrees.
