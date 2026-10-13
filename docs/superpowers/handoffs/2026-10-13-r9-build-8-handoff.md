# Handoff: r9 build 8: the stage 0b compare on the finished replay, then 13's review, 20b's fix check, and ticket 21 on Codex

## Brief
- Done: 20a, 12, 21a, 20b and 27 merged, plus four LOW rounds; every review passed.
- State: the stage 0b replay and the CUB/SemArt probes run detached; 13 and 20b's fix are built, not yet checked.
- Next: run the stage 0b compare on the replay, then 13's Opus review and dry run, 20b's smoke and re-check, ticket 21.
- Blocked: nothing.

> Written 2026-10-13 22:16 by `r9 build 7` (cut at the context meter, 300k). Loop step reached: step 7, the build.
> Direction log entry: [2026-10-13 · Round 8](../direction_log.md).

## 1. Read in this order

1. **`.scratch/r9-scope/notes/reviews_build7.md`**: every merge, check and number of build 7, the open LOWs per
   ticket (12b, 21b), and the commands for the replay compare and the probe queue.
2. `.scratch/r9-scope/notes/review_t20b.md`, `review_t21a.md`, `review_t12_fix.md` (open LOWs and notes for 21, 22, 25).
3. Ticket 13's hand-back `src/test/20261202_r9_scope/t13_handback.md` in worktree `/project/CoSiR-wt/r9-t13`.
4. `.scratch/r9-scope/notes/t27_progress.md` (probe design, choices and commands); the briefs for Codex jobs are
   `.scratch/r9-scope/notes/job_*.txt` (`job_21.txt` is ready to launch).
5. As before: `~/.claude/references/matt-chain.md`, `codex-jobs.md`, `.scratch/r9-scope/notes/codex_common.md`.

## 2. Decided by the user, not to be reopened

- All of spec §4 (D1 to D34), the 18:01 wording changes and the 18:58 rule wording changes (rule SHA f9b07b34).
- Build 5 and build 6 agent defaults stand. Build 7 agent defaults are in `reviews_build7.md`; the one to tell the
  user: **rule §10's date box also refuses a reserve read after 2026-10-18** (the row says "A read not started by
  2026-10-18 … does not start; the user decides"); a fix pass or an after-crash rerun continues a started read and is
  not boxed. Built in 20b's fix round 38252b2.

## 3. Where things stand

- **`r9-scope`** (pushed): build 7 merged 20a, 18b+25e, 09b+11b, 10b, 12, 21a, 20b, 27. Tickets left: 13 (built),
  21 (brief ready), 22, 23; small LOW jobs 12b and 21b (in `reviews_build7.md`).
- **Stage 0b replay (detached, started 21:43)** in worktree `/project/CoSiR-wt/r9-replay` (branch `r9-replay`,
  f66f554 = 12's fix + r9-scope 9245fc6; the five recipe scripts byte-equal to r9-scope). Log
  `src/test/20261202_r9_scope/results/stage0b_replay.log` there; it ends with one summary line. At 22:16 it was at
  step 5 of 5. Then run the compare from that worktree (command in `reviews_build7.md`; about 10 min; exit 0 pass,
  3 not a pass, 2 incomplete). Rule §3.2 item 4: any 0b failure: stop, trace, ask the user.
- **Probe queue (detached, 22:00)**: `src/test/20261202_r9_scope/results/probes/probe_queue.log` (main checkout);
  SemArt finished all eight stages (`results/probes/semart/probe_semart.json`, `probe_note.md`); CUB was at A3.
  The last line reads `probe queue done: <status>, <s> s`.
- **13** (Codex, `r9-t13` @ 24cadc0, 33 tests): needs an Opus `high` review (it extends `r9_pins.build_pins` and widens
  an independence test), then its CLIP dry run (`--step record --dry-run --replay <replay>`, then `--step pins
  --dry-run`; it must equal CLIP's stored means, pass counts, 96 arrays and sigma). Settle where the canonical
  `results/stage0b.json` lives (agent default: the main checkout's round folder, where stage 2 runs).
- **20b fix** (Codex resume, `r9-t20b` @ 38252b2, 170 tests, 73 mutation cases): rerun the CLIP smoke
  (`--backbone clip_b32 --read H6 --smoke`, about 3 min), a Sonnet `high` re-check, merge.
- **21**: launch Codex with `.scratch/r9-scope/notes/job_21.txt` in a worktree `r9-t21` made from r9-scope after
  the 20b fix merges (21 extends `run_r9_held.py`).
- Codex at 22:16: five-hour window about 60% left (resets 02:38), weekly about 78%.
- **Deadline:** rule §10, a read not started by 2026-10-18 does not start.

## 5. Code and pitfalls

- `r9-scope` is shared with other chats in the main checkout (the outside-pages chat commits there): merge with
  `--no-ff`, never fast-forward, and check `git status` first.
- Merges conflict in the round log and `.claude/20261013_log.md` (`git add -f`). The union helper is still
  `/tmp/claude-0/-project-CoSiR/2f29e134-1907-458c-b32a-999b8c5a13f7/scratchpad/union_merge.py`.
- Every merge that touches `r9_score`, `r9_bundle`, `r9_context`, `r9_picks` or `r9_common` changes
  `r9_module_shas`: ticket 22 reruns stage 0a on the final bytes.
- Run long jobs in a dedicated worktree, not in the shared main checkout (other chats' merges change files mid-run).
- Reviewers mutate only in a scratch worktree, never where a controller run is live.
- Auto mode refused two shell status commands about a running Codex job ("Create Unsafe Agents"); the launch
  passed. Read Codex logs with the Read tool. A Bash `cat` of a process's command line can dump a whole heredoc:
  cut it.
- The full mutation runner `rd9_mutations.py` (62 cases now) still waits for a quiet machine.

## 6. State at handoff

- Running: the stage 0b replay (detached, r9-replay worktree) and the probe queue (detached, main checkout); no
  subagents, no Codex job. A stray idle `cat` (pid 1078636, cwd r9-t27) from a subagent waits on stdin; harmless.
- Uncommitted: nothing of ours (notes live in the gitignored `.scratch/`).
- On disk over 1 GB: nothing new. New worktrees r9-t0911b, r9-t10b, r9-tsmall, r9-t12, r9-t20b, r9-t21a, r9-t13,
  r9-t27, r9-replay and this chat's scratchpad review copies go on `.scratch/pending_deletions.md` at the build's end.
