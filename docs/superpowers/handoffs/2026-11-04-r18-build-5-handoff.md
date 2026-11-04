# Handoff: r18 build 5: fix ticket 07, run gate (iii), build tickets 10, 12, 11, final review, merge, cut to run

## Brief
- Done: tickets 08 and 09 merged; ticket 07 reviewed, 3 fixes open. TPIPS gate (iv) fails on node402.
- State: 8 of 12 tickets merged. Node vectors do not count: lowest row cosine 0.9987 against 0.999, with identical packages and images. The threshold stays.
- Next: ticket 07 fixes, then gate (iii), then tickets 10, 12, 11. TPIPS stores now extract on the local 3090, shard 0 of 6 running since 01:38.
- Blocked: nothing. TURTLE time is 12.3 h, not the spec's 1 to 2 h: the route is open, recommendation in the handoff. GPU check passes.

> Written 2026-11-04 02:02 by tab "r18 build 4". Loop step reached: step 7, tickets 01 to 06, 08, 09 merged, cut at the 200k
> context meter. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 spec approved".

## 1. Read in this order

1. The ticket index, all of it: `/project/CoSiR/.scratch/r18-outside-methods/issues/00-index.md`. Its last section,
   "Controller decisions of build 4", is new and binding. Tickets beside it; common briefs in `../notes/`.
2. §3 and §4 below, then `../notes/t07_review.md`.
3. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md`, before the first dispatch.
4. Only when needed: the round log's "Build 4" section; the spec,
   [2026-11-03-outside-methods-design.md](../specs/2026-11-03-outside-methods-design.md) §3.2 to §6.

Keep the context small: builds 1 to 4 all reached 200k. `om_facts_brief.md` is 43 KB: give line ranges, never print it
whole. Subagents read the files; give them paths.

Skills: cluster-run (`SKILL.md`), only if §4 item 5 goes to the node.

## 2. Decided by the user, not to be reopened

As build 1's handoff §2 (D12 code route, D17 order, spec approved 2026-11-03 20:45, no container restart).

## 3. Where things stand

Integration branch `r18-outside-methods`, worktree `/project/CoSiR/.claude/worktrees/r18-int`, pushed.

| # | Ticket | State |
|---|---|---|
| 01 to 06 | | merged |
| 07 | Replays, gate (iii) | built, 12 tests; Opus `high` review APPROVE with notes; **3 should-fixes open**, not merged |
| 08 | Stage 1 | merged; Opus `high` review, one fix round, Sonnet `high` re-review APPROVE; 47 tests |
| 09 | TURTLE runner | merged; timing run done |
| 10 | Cells and labels | open, **unblocked; dispatch next** |
| 12 | Diagnostics | blocked by 10 |
| 11 | Stage runner | blocked by all |

## 4. Open points for the next step (agent defaults)

1. **Ticket 07 first.** Fresh Sonnet `high` fixer on `r18-t07-replays` in worktree
   `/project/CoSiR/.claude/worktrees/agent-a63100f1ddf83f50c`, findings 1 to 3 and notes 5 to 7 of `notes/t07_review.md`; a
   Sonnet `high` scoped re-review; merge. Then the full gate, about 2.5 h on the CPU, with the command in that file (a
   background `res-queue cpu`). Its hash then covers `om_data`, so any later edit of those files means a rerun.
2. **Ticket 10:** Sonnet `high`, then an Opus `high` review. For the brief: pins as the index says; `unit=False` is the
   clustering form of c*; `om_stage1.range_keys()` follows the same key rule, so compare the two; repeat c* + TURTLE [5, 1, n]
   to [5, 3, n] (index).
3. **Then 12 and 11**, Sonnet `medium`, Codex first if `codex-usage --room implementer` says room (57% left at 02:00; window
   resets 04:35). Ticket 11: the index's build 4 lines. Every GPU TURTLE process needs `LD_LIBRARY_PATH` (index, build 3).
4. **TPIPS stores, running by themselves.** Shard 0 runs under `res-queue` (label `r18:tpips_local0`). A detached loop, pid
   2887562, log `OUT/tpips_local_1_5.log`, queues shards 1 to 5, one at a time, and starts none on a weekday from 06:00 to
   19:00, since the job draws 256 W (heavy). It survives this cut; it is ours. A rerun skips finished chunks. So far about
   1.8 s per row, 2.5 to 4 h per shard (not settled), so the stores end about Thursday morning. Other chats' GPU jobs run
   between shards. After shard 5, in the tpips venv: `merge --set artelingo --factor all --of 6`, the same for mmart, then
   `gate-iv --path local=$OUT/stage0/probe_b1_rerun`, then the real-store check of `om_data.tpips` (build 3's handoff §4
   item 7). The run chat does this. `OUT/stage0/gate_iv.json` holds the node's failure until then; a copy stays as
   `gate_iv_das6_failed.json`.
5. **TURTLE route, open (agent decision, record it).** The timing run gives 12.3 h for 412 fits on one 3090, against the
   spec's 1 to 2 h (`OUT/smoke/turtle_timing.json`; it errs high). Mean 186 W, so light, and Qwen alone 199.6 W.
   Recommendation: DAS6 first, as `shared-resources.md` says, about 4 h over three GPUs, since TPIPS holds the local GPU for
   two nights. It needs a small node script, the c* exports and plain stores on the node (check `DATA_MAP`), and
   `tpips_pylibs/full` with `LD_LIBRARY_PATH` prepended gives the same torch 2.11 as gate (ii). A fit on the node is not
   checked yet: run one, and compare its loss and AMI with the local fit of the same K. Gate (iv) does not apply to TURTLE,
   since nothing compares its vectors across machines; say so in the log. If that takes too long, the 16 jobs run locally.
   Put the end time in the run handoff.
6. **Tell the final review and the report:** build 3's items (handoff §4 item 8), and gate (iv)'s story in the round log,
   build 4. The user reads the report's line on the 0.999 miss: the threshold was not changed. TPIPS vectors move with the
   transformers version (median row cosine 0.9988 between 5.6.2 and 5.16.1); a method paper's numbers rest on the version.
7. **r17's ranges:** as before. Stage 1 runs in the run chat, after main is merged. TPIPS picks wait for the stores, so
   stage 1 and the cells that use TPIPS cannot end before Thursday (not checked against the other chats' jobs).
8. **The build's handoff to the run chat** carries the ticket list (`matt-chain.md`) and the end-time estimate.

## 5. Code and pitfalls

- As build 3's §5.
- `cluster sync` needs "cluster run" in the HEAD subject: after a merge, add an empty commit with it.
- A `cluster launch` needs the HEAD on the node: sync after every merge.
- Node reruns skip finished chunks: a result equal to the last run in every digit, with no `probe shard` line, is not a run.
- Do not set `LD_LIBRARY_PATH` outright on the node: it holds the driver path; prepend.
- In a zsh Bash call, `$VAR` does not split on spaces: use `bash -c` or a loop.
- Scratchpad: subagents overwrote each other's `mut.py`. Each brief names its own subfolder.

## 6. State at handoff

- Running: r18's `tpips_local0` and the detached loop (pid 2887562). Beside it: r17's CPU jobs (`pick_untold_*`) and r19's build.
- Node402: no r18 job running.
- Uncommitted: other chats' lines in `briefs.md` and `chats.tsv` (not ours).
- Shared files changed: none this chat (no change to `cluster.conf`).
- On disk over 1 GB: `tpips_pylibs/full`, 7.3 GB, here and on the node (on `.scratch/pending_deletions.md`, keep until §4 item 5
  is settled); TPIPS weights in `/data/SSD2/HF_home/hub` (32 GB, keep); both snapshots on the node (30.5 GB, not needed
  for TPIPS any more; `cluster du` at the end).
- Worktrees to list on `.scratch/pending_deletions.md` at the build's end: `.claude/worktrees/agent-*` of tickets 01 to 09,
  and the ticket branches.
