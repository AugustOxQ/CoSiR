# Handoff: r9 build 9: two Codex fix rounds (13, 23), then the stage 0b re-compare, CLIP's dry run and the stage 2 chains

## Brief
- Done: stage 0b PASSED (case 1, 131 items exact) after a user-confirmed r9_bundle fix; 20b fix, 21, 21c merged.
- State: 13 and 23 are built and reviewed; each needs one small Codex fix round (briefs ready). GPU idle.
- Next: launch both fix rounds, merge, rerun the stage 0b compare, CLIP dry run, the two stage 2 chains; build 22.
- Blocked: nothing.

> Written 2026-10-14 00:04 by `r9 build 8` (cut at the context meter, 200k). Loop step reached: step 7, the build.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build8.md`**: every merge, check and number of build 8, and the order of the
   next steps (the "After 13 merges, in order" item).
2. The two fix briefs, ready to launch: `.scratch/r9-scope/notes/job_13_fix2.txt` (Codex resume of session
   `01a136d4-871f-7e02-ad2e-26f5d1358f73`, worktree `/project/CoSiR-wt/r9-t13`) and `job_23_fix.txt` (resume of
   `01a13713-02e4-74f3-8cd4-b58cca93b1cc`, worktree `/project/CoSiR-wt/r9-t23`). The reviews behind them:
   `review_t13_fix.md`, `review_t23.md`.
3. Ticket 22: `.scratch/r9-scope/issues/22-preread-smoke.md`, plus the two build 8 additions in `reviews_build8.md`
   (21's LOW-6, 20b's LOW-D). Its brief is not written yet.
4. As before: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/codex_common.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34), the 2026-10-13 wording changes (rule SHA f9b07b34), and the build 5 to 7 agent defaults.
- **2026-10-13 22:4x, user, after seeing the recommendation:** the stage 0b replay's cells-step crash (KeyError
  'smoke' in `r9_bundle.model_inputs`; 08's picked.json has no "smoke" key) is a code bug, not a failed replay:
  fixed as `picked.get("smoke", False) is False` (as 10b), step 5 rerun only, then the compare. Done.

## 3. Where things stand

- **`r9-scope`** @ d9405ce (pushed): the r9_bundle fix (ee7a76d), stage 0b log, 20b fix (e708a2b), 21 (3151dd5),
  21c (d9405ce). Tickets left: 13 (fix 2), 23 (fix), 22 (not built).
- **Stage 0b:** PASSED, case 1 (A3 L1 state-equal to the pin; 131 items exact; 505 s). Record in the main checkout
  `src/test/20261202_r9_scope/results/stage0b.json` (made on ee7a76d bytes). The replay folder stays in worktree
  `r9-replay` (`results/replay/clip_b32/`).
- **Why the compare reruns:** 13's M3 check compares the recipe scripts' 17-module import closure with
  `stage0b.json`'s `module_sha256`. 13 changes `r9_pins` and 23 changes `r9_score`, both in the closure. So: merge both
  fix rounds first; then merge r9-scope into r9-replay; `mv` the old record to `results/stage0b_ee7a76d.json` (kept);
  rerun the compare from r9-replay with `--out` the main checkout's `results/stage0b.json` (~9 min, CPU).
- **Then:** CLIP dry run from the main checkout (commands in 13's fix-2 hand-back; `--stage0b <path>`); then the live
  chains in the main checkout (review L4: the checkout that runs the reads), `run_r9_stage2.py --backbone
  qwen3vl_emb_2b --step all`, then `siglip2_so400m`, about 65 to 90 min each under the GPU lock. Expected end about
  04:00 to 05:00 if started by 01:30, well before 11:00.
- **N5 (operational):** after the stage 2 chains, no closure module may change (the 17 modules include r9_pins,
  r9_stats, r9_score, r9_bundle, r9_common, run_r9_stage0a). Ticket 22's brief must forbid closure edits; if 22 needs
  one, it lands before the compare.
- **Probes:** the queue finished ok (SemArt and CUB); results in the main checkout `results/probes/`; for stage 3's
  decide chat, not read here.
- **Codex** at 00:04: five-hour window 88% left (resets 04:38), weekly 98% (the counters reset during build 8).
- **Deadline:** rule §10, a read not started by 2026-10-18 does not start.

## 5. Code and pitfalls

- `r9-scope` is shared with other chats in the main checkout: merge with `--no-ff`, check `git status` first. The
  round log and `.claude/<day>_log.md` conflict on almost every merge: union helper
  `/tmp/claude-0/-project-CoSiR/2f29e134-1907-458c-b32a-999b8c5a13f7/scratchpad/union_merge.py <files>`, then
  `git add -f`.
- Codex launch: the Bash command must start with `codex e --dangerously-bypass-approvals-and-sandbox` (a resume:
  `codex e --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -c mcp_servers={} -m gpt-6.1-sol -c
  model_reasoning_effort=high -C <worktree> --json resume <session> - < <brief> > <log> 2>&1`). Read its log with
  `grep '"type":"agent_message"' | tail -1`.
- Tests that stub `model_inputs` or use copies of CLIP's files miss recipe-only file shapes (the smoke crash): the
  replay, the dry run and the live chains are the real checks.
- `test_r9_context.py` takes about 16 min in full: split it with `-k` inside a subagent.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no detached run). GPU idle.
- Uncommitted: nothing of ours (notes live in the gitignored `.scratch/`). `/project/claude-config` has two new
  Codex ledger rows (`codex/jobs.md`), committed by `sync.sh` at the wrap-up.
- New worktrees this chat: r9-t0bfix, r9-t21, r9-t21c, r9-t23; review scratch worktrees in this chat's scratchpad
  (`t13_review`, `t20b_fix_review`, `t21_review`, `t13_fix_review`, `t13_base`, `t23_review`). All go on
  `.scratch/pending_deletions.md` at the build's end, with build 7's list. Keep `r9-replay` while stage 0b's record
  points at it.
