# Handoff: r9 build 4: re-review and merge 05, 14, 19, 25b; fix and merge 07; then 06

## Brief
- Done: 25 (C13 re-derivation) and 26 (probe corpora) merged; 05, 07, 14, 19 built and Opus-reviewed; two rule wording changes approved by the user.
- State: fix rounds done for 05, 19 and 25b; 14 ready; 07 needs one fix round (GPU guard for the content graph, records, tests).
- Next: Sonnet re-reviews of 05, then 19 with 25b; merge 05, 14, 19, 25b; 07's fix round; then 06 (Codex if the window allows).
- Blocked: nothing. For the user (minor): rule §3.2 calls the content graph "CPU-made"; it is GPU fp16 (bit-equal on the GPU only).

> Written 2026-10-13 18:16 by `r9 build 3` (cut at the context meter). Loop step reached: step 7, the build.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build3.md`**: the open work per ticket, with each review's findings, the fixes
   done, what each re-review must check and what to do at each merge. It is the plan for this chat's first hour.
2. The tickets index `.scratch/r9-scope/issues/00-index.md` (list, "Shared interfaces"; one new item at its end:
   ArtELingo-28 bootstrap clusters and part hashes). Notes added to tickets 08, 13, 15, 21 and 22 at their ends.
3. The round log `src/test/20261202_r9_scope/20261202_r9_scope_log.md`: "Rule wording changes" and "Build 3".
4. As in build 3: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/codex_common.md`
   (append it to every Codex brief; Claude `worker` briefs point to it). C13's phase-1 inputs:
   `.scratch/r9-scope/notes/phase1_inputs.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34).
- 2026-10-13, after the recommendation (round log, "Rule wording changes"): rule §7.1, H4's ledger cell names the four
  backbones; rule §3.3, the train side is train + val (790 + 122 = 912), and the near-duplicate check compares against
  all 912. The rule's SHA-256 is now 69a365c3 (was 7e2a6dde): update the pin in `r9_common.py` (05) and
  `run_r9_apply_rule.py` (19) at their merges; `rederive/rd9_inputs.py` is done.
- D22: the user checks ArtELingo-28's licence personally.

## 3. Where things stand

- **On `r9-scope`:** 01 to 04, 24, 25, 26, 28 merged. Ticket files' `Status:` lines say `done` for these.
- **Branches waiting** (worktrees `/project/CoSiR-wt/r9-tNN`): `r9-t05` @ b8b3d8a (fixes done), `r9-t14` @ a34ae06,
  `r9-t19` @ 4fe0d24, `r9-t25b` @ b409331 (from `r9-scope`), `r9-t07` @ 9be0c4b. 14, 19 and 07 sit on 05's first
  commit 922c79f: rebase them onto `r9-scope` after 05 merges.
- **Frontier after these merges:** 06 (Codex brief `.scratch/r9-scope/codex/t06_brief.md`, with the 05 and 25 points
  at its end); then 08, 09, 10, 11, 15, 16, 17, 20 as the index says.
- **Codex:** 15% of the five-hour window left at the cut (reset 21:37), weekly 84%. Job-ledger rows written for 05,
  05 fix, 25 fix, 26. Fix rounds are expensive (05: 9.8M input tokens; 25: 11.8M): prefer fresh Claude fixers when
  the window is short.

## 4. Open points for the next step

- **For the user (minor):** rule §3.2 item 1 lists the content graph among the "CPU-made" parts. The fact: the graph
  is built in fp16 on the GPU; on the GPU it equals CLIP's `graph.npz` bit for bit (1,873,347 edges), on the CPU it
  does not (1,873,192). The bit-equal gate holds. The wording change needs the user's approval.
- **For the user (minor):** rule §7.1 does not say what a switched-off or dropped ledger row looks like; 19's
  `--not-run` matches the wording anywhere in the row.

## 5. Code and pitfalls

- The shell for Bash tool calls is zsh: `$VAR` holding a command is not word-split. Put multi-step commands in a script.
- GPU steps (07's graph) run locally under `flock -n -o -E 75 /tmp/gpu0.lock`; nothing else in 07 uses the GPU.
- `git commit -a` in the main checkout can pick up other chats' files (the paper chat commits to `r9-scope`): add
  paths explicitly.
- Codex resumes: 05 thread `01a135d2-70ad-7042-98ef-7fc6951fc053`, 25 thread `01a135b2-ac45-73f2-b5b6-b5bc4881571b`
  (each has used its one fix round).

## 6. State at handoff

- Running: nothing (no subagents, no Codex jobs, no GPU work).
- Uncommitted: nothing on `r9-scope` except the untracked round 8 results trees, as before.
- On disk over 1 GB: `/project/CoSiR-wt/r9-t25/src/test/20261202_r9_scope/rederive/rd9_out/` (1.9 GB, gitignored,
  Codex's CLIP re-derivation outputs). Small: `results/replay_subset/` in the r9-t07 worktree; `/tmp/claude-0/gen`
  (92 KB) and `/tmp/claude-0/{apply,rd9_compare}.bak` from a worker. List them on `.scratch/pending_deletions.md` at
  the build's end with the worktrees.
