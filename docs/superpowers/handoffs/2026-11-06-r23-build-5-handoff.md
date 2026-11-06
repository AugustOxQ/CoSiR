# Handoff: r23 build 5: final whole-branch review, one fix wave, scoped re-review, merge to main, cut to the run chat

## Brief
- Done: tickets 9 and 10 are merged, so all 10 tickets are done. All 443 round tests pass on the merged code.
- State: no full held score exists. The end-to-end smoke on 200 paintings passes from stage 0 to stage 3, with all four gates passing and 76 cells written.
- Next: final whole-branch review, one fix wave, scoped re-review, merge to main, cut to the run chat.
- Blocked: nothing. The cut came from the context meter (past 200k tokens), not from a problem.

> Written 2026-11-06 07:57 by tab "r23 build 4". Loop step reached: inside step 7, all tickets merged. Direction log entry: [Pair-hit checks read](../direction_log.md), "Grill and spec".

## 1. Read in this order

1. `/project/CoSiR/.scratch/r23-text-claim/issues/00-index.md`: the ticket list and shared interfaces. The tickets sit beside it, outside git.
2. The [spec](../specs/2026-11-06-text-claim-design.md), from §1. Its §3.4, §3.5 and §6 are the rule.
3. The [third build handoff](2026-11-06-r23-build-3-handoff.md) §4: four notes for the final review, all still open.
4. `~/.claude/references/matt-chain.md`, "The final whole-branch review". `codex-jobs.md` only if a fix goes to Codex.

Skills: none.

## 2. Decided by the user, not to be reopened

All of the spec's §4, D1 to D14. A change to the rule is an amendment (hard rule 5).

## 3. Where things stand

| Item | State | Evidence |
|---|---|---|
| Tickets 1 to 8 | merged | third and fourth build handoffs, §3 |
| 9 runner, guard, queue | merged | 15 new tests. The controller reran the mutation: without the guard call in `tcl_score.run()`, `test_run_on_held_set_with_uncommitted_record_raises_before_any_row` fails |
| 10 diagnostics | merged | 17 tests. The worker's 8 mutants were all caught; not rerun by the controller |
| All round tests | pass | 443 passed in 3.9 minutes on `r23-text-claim`, 273dd4da, pushed |
| End-to-end smoke | passes | `TCL_SMOKE=200` through the real `tcl_queue.sh`, chains `val`, `held`, `post`, 34 steps, none failed |

Smoke details: gate (i) 39 of 39 stores; gate (ii) largest difference 6.1e-5 against 1e-3; gates (iii) and (iv) pass; stage 1 written for both groups; 72 label cells and 4 pair cells; verdicts copied without C13 (smoke only); all 8 diagnostic parts written with nothing missing; part 2's self-check at a = 0.7 passes. The smoke's verdict words come from 200 paintings and mean nothing: do not quote them.

Smoke output: `/tmp/claude-0/-project-CoSiR/a6e0cc2d-4e46-4685-a827-2c56524eb359/scratchpad/e2e/` (95 MB; `out/`, `stores/`, and the logs `val.log`, `held.log`, `post.log`). It stays for the review.

## 4. Open points for the next step

None for the user. Notes for the final review, beside the four in the third handoff. Each is a choice made in this chat, and the review should check it against the spec:

a. Stage 2 scores the bar rows and the bar text only. Stage 3 scores the side rows and MMArt's other four texts, after the verdicts. The reason is §3.3 ("read in stage 3") and §3.5.
b. `held_guard` needs the set's own stage 1 record and `status.json`. It checks the other group's record only when that file exists. The ticket says "both", but §3.5 and §6 void only the failed group's verdicts, so one group's failure must not stop the other.
c. A stage exits 0 when a step is skipped for a missing stage 1 record, so `decide` still runs and voids that group.
d. Only GPU jobs with `--est` wait for 19:00. `--short` jobs and the CPU `decide` step do not (controller's reading of D13).
e. The first failed step stops the chain, and a gate that reports a fail stops it too. To go past a gate fail that voids only part (§6), the run chat runs the remaining steps with `tcl_run.py <stage> --only <step>` under `res-queue`. A rerun of the chain stops at the same gate.
f. `tcl_diag.py` crashes when a stage 1 record is missing, at the guard or at its record read. A voided group therefore needs a manual route for the other group's diagnostics. Decide whether that needs a fix.
g. Part 1 reads c* at late fusion's pair pick head and read-out, since the spec gives c* no pair pick. Its JSON says so.
h. Part 2's self-check holds at 1e-6 only on the device that wrote the scores. On CPU against GPU files it failed at 7.4e-6, from near-tie order flips (worker's measurement). So stage 3 runs on `cuda`. This bears on the C13 note about ties in the third handoff.

The review also applies hard rule 6: `~/.claude/templates/spec_core.md` and `~/.claude/rules/spec-templates.md`, with the four gap types.

## 5. Code and pitfalls

Who does what:
- Final review: a `worker`, `opus` `xhigh`. Its brief names the spec, the tickets, `matt-chain.md`'s review section, hard rule 6 and `docs/superpowers/constitution.md`. It re-derives load-bearing numbers from the data or the code. It has no wait over 4 minutes, so GPU work runs under the controller.
- One fix wave, then a scoped re-review: a `worker`, `sonnet` `high`.
- Codex at 07:56: 35% left of five hours, reset 09:53; weekly 39%. `--room bug` needs 20%, so a fix round can go to Codex; check first. Ask for fixes and tests only.

After the re-review:
- Merge `r23-text-claim` into main, push main.
- List these on `.scratch/pending_deletions.md`: worktrees `r23-int` and `r23-t01` to `r23-t10`, their branches, and the smoke folder above. `/tmp/claude-0/sm` is already listed.
- Cut to the run chat, with the ticket list from `00-index.md` in its handoff (`matt-chain.md`, "The record").

What the run chat needs:
- Chains: `tcl_queue.sh val`; commit `records/stage1_*.json` and `status.json` with `git add -f`; `tcl_queue.sh held`; C13 as a Codex job; `tcl_run.py verdicts`; `tcl_queue.sh post`.
- Run the chains from the main checkout: `read_committed_record` compares with HEAD of the repo that holds the round folder.
- `tcl_run.py stage2`, `verdicts` and `stage3` never rewrite `status.json`, since the guard compares bytes with HEAD. Stage 0, stage 1 and `status` do rewrite it. If C13 adds `c13_void`, commit the file again.
- All `--est` values in `tcl_queue.sh` are estimates, not measured. By the worker's timing of single rankings, part 2 takes about 10 minutes and part 1 about 5 at full size.
- Run timing (D13): 2026-11-06 is a Friday. Heavy jobs that would fall between 10:00 and 19:00 wait for 19:00. Report by 2026-11-07 12:00, at the latest 2026-11-08.

Working here:
- The integration worktree is `/project/CoSiR/.claude/worktrees/r23-int`, branch `r23-text-claim`.
- A chained command stops only with `&&` or `set -e`. In this chat a merge went through after a failed test, since `;` does not stop. The test caught a cosmetic swap of mine and I reverted it, 273dd4da.
- In zsh, `echo =====` fails. An unquoted heredoc eats backslashes in Python patterns: use `<<'EOF'`.
- `*.json` is gitignored: results files need `git add -f`. `.claude/20261106_log.md` is tracked although `.claude` is ignored: `git add -f`.

## 6. State at handoff

- Running: nothing. No queue job, Codex job or subagent.
- Uncommitted: nothing of this chat, apart from `chats.tsv` and `briefs.md`, committed right after the cut.
- On disk over 1 GB: nothing new. The scratchpad holds 95 MB of smoke output, no file over 100 MB.
- GPU check before this unattended chat: passed. Driver 595.99.02 in the module and the library. Run mode: night, so no restart question.
