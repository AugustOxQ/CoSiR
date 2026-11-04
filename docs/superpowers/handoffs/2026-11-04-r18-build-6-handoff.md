# Handoff: r18 build 6: build ticket 11, smoke of 11 and 12, reference fit, final review, merge, cut to run

## Brief
- Done: tickets 07, 10, 12 merged, gate (iii) passes (largest difference 5.55e-17), TURTLE route settled on node402.
- State: 11 of 12 tickets merged. One node fit takes 119 s against 105 s locally; loss and AMI against a local fit are not compared yet.
- Next: ticket 11 with its smoke, then the local reference fit, the final review, the merge, the cut to run.
- Blocked: nothing. GPU check passes. TPIPS stores still extract locally, shard 0 of 6.

> Written 2026-11-04 03:20 by tab "r18 build 5". Loop step reached: step 7, all tickets but 11 merged, cut at the 200k context
> meter. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 spec approved".

## 1. Read in this order

1. The ticket index, `/project/CoSiR/.scratch/r18-outside-methods/issues/00-index.md`: at least "Shared interfaces" and the
   decisions of builds 3, 4 and 5 (the last is new and binding). Ticket 11 beside it; common briefs in `../notes/`.
2. §3 and §4 below.
3. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md`, before the first dispatch.
4. Only when needed: the round log's "Build 5" section; `notes/t10_review.md`, `notes/t10_handback.md`, `notes/t07_review.md`; the
   spec, [2026-11-03-outside-methods-design.md](../specs/2026-11-03-outside-methods-design.md) §3.2 to §6.

Keep the context small: builds 1 to 5 all reached 200k. Subagents read the files; give them paths.

Skills: cluster-run (`SKILL.md`), only for the reference comparison's node files.

## 2. Decided by the user, not to be reopened

As build 1's handoff §2 (D12 code route, D17 order, spec approved 2026-11-03 20:45, no container restart).

## 3. Where things stand

Integration branch `r18-outside-methods`, worktree `/project/CoSiR/.claude/worktrees/r18-int`, pushed, main merged at 7252750b.

| # | Ticket | State |
|---|---|---|
| 01 to 10, 12, 09b | | merged |
| 11 | Stage runner, records, queue script, smoke | open, **unblocked; dispatch next** |

Gate (iii): `OUT/stage0/gate_iii.json`, pass, parts a, b, c hold the current `code_sha256`. Gates (i) and (ii) passed in earlier
builds. Gate (iv) waits for the stores.

## 4. Open points for the next step (agent defaults)

1. **Ticket 11.** Sonnet `medium`, Codex first if `codex-usage --room implementer` says room (29% left at 03:08; window resets
   04:35). For the brief:
   - `iii_a`, `iii_b`, `iii_c` compare each part's `code_sha256`, not part `pass` alone.
   - `records/status.json` `stage1` may name only `tpips/<set>`, `clay/<set>`, `clay_textset/ret/<set>`, `clay_textset/clu/<set>` and
     range keys. A loop over every key of `stage1_om_picks.json` would add `clay_text_sha256`, which `check_status` refuses.
   - `r17` holds exactly `path` and `sha256`. r17's `records/status.json` is now in this checkout. `om_cells.load_r17_status` crashes if the path is absent.
   - CLI: `om_cells.py cells`, `assemble`, `labels`. `cells` exits 1 when a unit fails and rescoring a parts file with `missing` needs no `--force`.
     `labels` rebuilds `assemble`, so it needs the parts, the committed records and both status files, and it refuses a `not scored` cell.
   - Smoke: `OM_OUTPUTS=<smoke out> OM_SMOKE=1 python om_diag.py all`, after the TURTLE and TPIPS smoke outputs exist, never inside a
     pool worker. This is the first run of `om_diag`'s TURTLE and TPIPS paths and of `om_cells` on those inputs: use real code paths.
   - `om_queue.sh`: TURTLE lines come from `om_turtle_node_plan.sh` (node); TPIPS from §4 item 4. Do not edit `om_data.py`, `om_paths.py`
     or `ns_*`: gate (iii) hashes them, and any change means a rerun of 45 min.
2. **The local reference fit**, stopped at this cut, not run. Queue it (about 2 min on the GPU; it goes between TPIPS shards, next one about 04:08):
   `res-queue gpu --short --label r18:turtle_ref -- env OM_OUTPUTS=<your scratchpad>/turtle_local_ref LD_LIBRARY_PATH=/root/miniconda3/envs/CoSiR/lib/python3.11/site-packages/nvidia/cu13/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python <r18-int round folder>/om_turtle_run.py fit --set mmart --enc qwen --arm plain --device cuda --only-k 20 --only-start 0`.
   Compare it with `res/r18_turtle_probe/parts/turtle/mmart/qwen/plain/k20__start0.{npz,json}` in r18-int (loss 2.9114, 20 clusters): the loss,
   the AMI of each against MMArt's style labels, and the AMI between the two partitions. Exact equality is not expected across
   GPU types. Say the numbers in the log. If they differ much, the 16 jobs run locally instead (mean 186 W, light; 12.3 h).
3. **The final review:** Opus `xhigh`, on the whole branch, after ticket 11's smoke (`matt-chain.md`); then one fix wave and a
   Sonnet `high` re-review; merge into main; push. Tell the review: build 3's items (its handoff §4 item 8), gate (iv)'s story
   (round log, build 4), the choices in `notes/t10_handback.md`, and that `om_diag` paths have only ticket 11's smoke behind them.
   A review fix in a hashed file means rerunning gate (iii) (`notes/t07_review.md`, last lines; 45 min).
4. **TPIPS stores, running by themselves**, as build 5's handoff §4 item 4: shard 0 under `res-queue`, the detached loop pid 2887562 queues
   shards 1 to 5, none on a weekday from 06:00 to 19:00. The merge, gate (iv) and the real-store check are the run chat's.
5. **TURTLE on the node:** the run chat launches the 16 jobs with `om_turtle_node_plan.sh`, after step 2. Pass `OM_EST_FIT=1h`: the
   plan's `--est 30m` is a guess, and 24 to 28 fits at about 2 min give 50 to 60 min a job. So about 5 h over three GPUs
   (not checked against r19's node jobs). Sync after the merge: a `cluster launch` needs the HEAD on the node, and `cluster sync`
   needs "cluster run" in the HEAD subject.
6. **Tell the report:** as build 5's handoff §4 item 6 (the 0.999 miss, the threshold not changed, TPIPS vectors move with the
   transformers version).
7. **r17's ranges** are now committed on main and merged here; stage 1 runs in the run chat, and TPIPS picks wait for the stores, so
   stage 1 and the TPIPS cells cannot end before Thursday (not checked against the other chats' jobs).
8. **The build's handoff to the run chat** carries the ticket list (`matt-chain.md`) and the end-time estimate.

## 5. Code and pitfalls

- As build 5's handoff §5 and build 3's §5.
- Each subagent brief names its own scratchpad subfolder.
- A `--queue` launch refuses a changed worktree HEAD: wait until it has launched before you merge in that worktree.
- r18-int's `res-queue` jobs read the working files: do not merge into r18-int while a gate needs the hashed files unchanged.

## 6. State at handoff

- Running: r18's `tpips_local0` and the detached loop (pid 2887562). r17's and r19's run chats go on beside it.
- Node402: no r18 job running; both packs of inputs are on the node.
- Shared file changed: `cluster.conf`, one `DATA_MAP` pair added (`/local/wding/r18_in/turtle_pack`).
- Uncommitted: other chats' lines in `briefs.md` and `chats.tsv` (not ours).
- On disk over 1 GB: `OUT/turtle_pack` 14 GB, here and on the node (on `.scratch/pending_deletions.md`, keep until the fits pass);
  `tpips_pylibs/full` 7.3 GB, here and on the node (keep: the node's TURTLE jobs use it).
- Worktrees to list on `.scratch/pending_deletions.md` at the build's end: `.claude/worktrees/agent-*` of tickets 01 to 12, and the
  ticket branches.
