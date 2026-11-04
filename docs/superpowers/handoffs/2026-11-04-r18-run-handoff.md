# Handoff: r18 run: launch the 16 TURTLE fits on node402, then stages 0 to 2, C13, labels, diagnostics and the report

## Brief
- Done: build finished. All 12 tickets are merged into main, the final review and the fix wave passed, and the full smoke passes.
- State: the node and the local GPU give the same TURTLE partition (every row equal, loss 2.911379). Gates (ii), (iii) pass. TPIPS shard 1 of 6 runs.
- Next: launch the 16 TURTLE jobs on node402. TPIPS stores end about Thursday 23:00, so stage 1 runs after that.
- Blocked: nothing. GPU check passes. No fit, cell or label exists yet.

> Written 2026-11-04 05:46 by tab "r18 build 6". Loop step reached: step 7 done, step 8 next. Direction log entry:
> [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 spec approved".

## 1. Read in this order

1. §3 and §4 below.
2. `src/test/20261103_outside_methods/om_queue.sh`, its printed plan: `bash om_queue.sh all`. It starts nothing without `--run`.
3. Binding decisions: `/project/CoSiR/.scratch/r18-outside-methods/issues/00-index.md`, builds 3, 4, 5.
4. Only when needed: the round log's "Build 6" section, spec [2026-11-03-outside-methods-design.md](../specs/2026-11-03-outside-methods-design.md) §3.5, §6.

Skills: cluster-run (`SKILL.md`) for the TURTLE jobs.

## 2. Decided by the user, not to be reopened

As build 1's handoff §2 (D12 code route, D17 order, spec approved 2026-11-03 20:45, no container restart).

## 3. Where things stand

Build record (`matt-chain.md`). All merged; verdict tickets had their own Opus `high` review.

| # | Ticket | Blocked by | Spec IDs | Verdict code |
|---|---|---|---|---|
| 01 | CLAY math `om_clay.py`, gate (i) | none | §3.4, (i), D4, D6, D12, D23 | yes |
| 02 | TURTLE loop `om_turtle.py`, gate (ii) | none | §3.4, (ii), D9, D12, D21, D23 | yes |
| 03 | TPIPS extractor `om_tpips.py`, stores, gate (iv) | none | D2, D18, D20, (iv), D23 | no |
| 04 | TPIPS on node402 | 03 | D3, §5 | no |
| 05 | Text set B `om_textset_b.py` | none | D5, §3.2 | no |
| 06 | Loaders and partition files `om_data.py` | r17, 01, 03 | §3.3, §3.4, C6 | yes |
| 07 | Replays, gate (iii) a, b, c | 06 | (iii), D23 | yes |
| 08 | Stage 1 `om_stage1.py` | 06 | §3.4, D7, D8, D14 | yes |
| 09, 09b | TURTLE runner `om_turtle_run.py`, node route | 02, 06 | §3.3, D9, D21, §3.6 | no |
| 10 | Cells and labels `om_cells.py` | 06, 08 | §6, §3.4, §3.5, D11, D13, D15, D24 | yes |
| 11 | Stage runner `om_run.py`, `om_queue.sh`, smoke | all | §3.5, D16, D24, §3.7 | no |
| 12 | Diagnostics `om_diag.py` | 06, 09, 10 | §3.6, D22 | no |

Main and `r18-int` both hold c29073c9 (subject has "cluster run"). 250 round tests passed before the fix wave; the 4 test files the
fix wave changed pass, 168 tests. The whole round's suite has not run since the fix wave: run it once, through `res-queue cpu`.

Gate state, read from `/project/CoSiR`: i `missing` (no file yet), ii `ok`, iii a, b, c `ok`, iv `failed`. Gate (iv)'s file still holds the node's
failure, and the local rerun replaces it. No hashed file changed on main, so gate (iii) needs no rerun.

Reference fit: MMArt Qwen plain, K 20, start 0, on the 3090 and on node402: all 16,384 rows equal, loss 2.911379 on both, AMI against style 0.22189 on
both. So the node route holds.

**End-time estimate (not checked against the other chats' jobs):**
- TURTLE: 16 jobs of 50 to 60 min, about 5 h over three GPUs. r19's node jobs share them.
- TPIPS: a shard takes about 3.9 h, not 2.5 h. Shard 1 ends about 09:25. The loop starts no shard on a weekday from 06:00 to 19:00. So shards 2, 3, 4
  run Wednesday 19:00 to Thursday 06:45, and shard 5 Thursday 19:00 to 23:00.
- Then merges and gate (iv), about 30 min; stage 1, about 2 h; stage 2, 28 units on 4 slots, about 3 to 4 h. Times per unit are from smoke draws
  (SC1 18 min, SC3 15, SC6 12, SC2 5, SC4 4 to 14, SC5 1); full draws were not measured, so correct `--est` from the first real units.
- So cells about Friday 05:00, C13 and labels before 11:00 on Friday.

## 4. What the run chat does, in order

1. **TURTLE.** Work in `/project/CoSiR/.claude/worktrees/r18-int`: a `--queue` launch refuses a changed HEAD, and main moves with the other chats. Pass `OM_EST_FIT=1h`
   to `om_turtle_node_plan.sh`. Steps 3, 4, 6, 7, 8 of its plan; pack, `DATA_MAP` and probe are done. Results land in `res/r18_turtle`. `check-pulled` must print 412 of
   412 and now also refuses mixed hosts, torch versions, devices or settings; then run the rsync line it prints.
2. **Stage 0.** `om_run.py stage0` (gate (i) and gate (ii) CPU, writes `gate_i.json`). Do not use `om_queue.sh stage0 --run`: it reruns gate (iii) for 45 min, which is not needed.
   Gate (ii) CUDA already passed.
3. **TPIPS, after shard 5.** Pid 2887562 queues the shards by itself. Do not merge into r18-int in a way that changes `om_tpips.py` or `om_tpips_local.sh`. Then the
   lines of `om_queue.sh stage0`: the two merges, `om_tpips_check.py`, gate-iv, `status`. `om_tpips_check.py` exits 0 while shards are pending: read `complete` in its
   JSON, which must be true. It also lists packages changed since shard 0 (none at 05:01).
4. **Stage 1.** First merge main into r18-int (r17's records), and confirm with `git diff --stat` that no hashed file moved: `om_gate_iii`, `om_data`, `om_paths`, `ns_clu`,
   `ns_metrics`, `ns_ret`, `ns_data`, `rg_held`. Then `bash om_queue.sh stage1 --run`, then commit `records/stage1_om_ranges.json`, `stage1_om_picks.json` and `status.json`.
   Gate (iv) must run before this step. If a gate file changes after the commit, rebuild status and recommit: `assemble` and `labels` now refuse otherwise.
5. **Stage 2.** `bash om_queue.sh stage2 --run`. It skips finished units: after any re-extraction or refit pass `--force`, since parts files bind no input hash. A range of family
   r17 must equal r17's record at `assemble`, so r17's stage 1 record must be final.
6. **C13, labels, diagnostics.** C13 is a Codex job from the saved outputs, writing `OUT/c13/c13_cells.json`. `labels` refuses a stub or a mismatch, and then removes an older `labels.json`.
   `om_diag.py` after the labels, one item per job. `sweep` has no TEMI rows, and `r17_kmeans` stays empty until r17's stage 3 files exist.
7. **Report.** Tell it: the gate reach and final review numbers in the round log's "Build 6"; build 5's handoff §4 item 6 (the 0.999 miss, the threshold not changed, TPIPS vectors move with the
   transformers version); set B's style list has 44 values with "gothic" (the index says 43), kept by D5; the interval's widening in `adds` is not measured.

## 5. Code and pitfalls

- As build 5's handoff §5 and build 3's §5.
- `om_run.py` and `om_cells.py` refuse OM_OUTPUTS without OM_SMOKE=1 and the reverse. `om_turtle_run.py` may run with OM_OUTPUTS alone.
- A status refusal comes before `labels` removes an old file, so after a stale-status refusal an older `labels.json` still stands. Not fixed; low.
- `cluster sync` needs "cluster run" in the HEAD subject; an empty commit does it.
- The final review's two files could not be written to `notes/`: worktree isolation refused the path. They are in the worktrees `agent-a4d5a1e64710d2db7` and `agent-a449a49cb0b23ede4`, under
  `.scratch/r18-outside-methods/notes/`. Their pending-deletion row says so; the findings are in the round log.

## 6. State at handoff

- Running: r18's `tpips_local1` and the detached loop (pid 2887562). r17's and r19's chats go on beside it.
- Node402: no r18 job running; both packs of inputs are on the node.
- Uncommitted: other chats' files only.
- On disk over 1 GB: `OUT/turtle_pack` 14 GB, here and on the node (keep until the fits pass); `tpips_pylibs/full` 7.3 GB, here and on the node (keep: the node's TURTLE jobs use it);
  this chat's scratchpad 3 GB (on `.scratch/pending_deletions.md`).
