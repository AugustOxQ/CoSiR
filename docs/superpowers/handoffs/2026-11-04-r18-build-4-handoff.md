# Handoff: r18 build 4: rerun the node probe, then tickets 07, 08, 09, then 10, 11, 12, final review, merge, cut to run

## Brief
- Done: tickets 02 (after one fix round), 04 and 06 merged; gate (i) passes; gate (ii) passes, 2.54e-7 against 1e-5 and GPU against CPU AMI 1.0; text set B frozen; D20 set to batch 1.
- State: 6 of 12 tickets merged. Node route works, only the probe's own weight check failed; it is fixed, rerun due.
- Next: node probe, node gate probe, then tickets 07, 08, 09.
- Blocked: nothing. GPU check passes.

> Written 2026-11-04 00:52 by tab "r18 build 3". Loop step reached: step 7, tickets 01 to 06 merged, cut at the 200k
> context meter. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 spec approved".

## 1. Read in this order

1. The ticket index, all of it: `/project/CoSiR/.scratch/r18-outside-methods/issues/00-index.md`. Its last section,
   "Controller decisions of build 3", is new and binding. Tickets beside it; common briefs in `../notes/`.
2. §3 and §4 below.
3. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md`, before the first dispatch.
4. Only when needed: the round log's "Build 3" section; the spec,
   [2026-11-03-outside-methods-design.md](../specs/2026-11-03-outside-methods-design.md) §3.2 to §6.

Keep the context small: builds 1 to 3 all reached 200k. `om_facts_brief.md` is 43 KB: give line ranges, never print it
whole. Subagents read the files; give them paths.

Skills: cluster-run (`SKILL.md`, the node steps).

## 2. Decided by the user, not to be reopened

As build 1's handoff §2 (D12 code route, D17 order, spec approved 2026-11-03 20:45, no container restart).

## 3. Where things stand

Integration branch `r18-outside-methods`, worktree `/project/CoSiR/.claude/worktrees/r18-int`, at b28f21d5, pushed.

| # | Ticket | State |
|---|---|---|
| 01 | CLAY math, gate (i) | merged |
| 02 | TURTLE loop, gate (ii) | merged; Opus `high` review, one fix round, Sonnet `high` re-review APPROVE; 16 tests |
| 03 | TPIPS extractor | merged |
| 04 | TPIPS on node402 | merged; plan and script changed by the controller (§4 item 1) |
| 05 | Text set B | merged, generated, frozen |
| 06 | Loaders, partition files | merged; Opus `high` review APPROVE with notes; 24 tests |
| 07, 08, 09 | Replays; stage 1; TURTLE runner | open, unblocked; **dispatch next**, at most three at once |
| 10 | Cells and labels | open; blocked by 08 |
| 11, 12 | Stage runner; diagnostics | open; 11 blocked by all, 12 by 09 and 10 |

`OUT` is `/data/SSD2/cosir_r18/20261103_outside_methods/outputs`. `OUT/stage0/gate_ii.json` has top-level `pass: true`,
tied by hash to `om_turtle.py` at b28f21d5: any later edit of that file needs both gate parts rerun.

## 4. Open points for the next step (agent defaults)

1. **Node probe, rerun first.** Tag 20261103-234529-a498178-5397d97e failed only on the probe's check: the TPIPS weights
   are `model.pt`, and the check looked for safetensors. Everything else passed: torch 2.14, transformers 5.16.1, peft
   0.19.1, tpips 0.1.2, RTX A6000, 34,817 images with 0 missing, both snapshots fetched (15.3 and 15.2 GB). From r18-int:
   `cluster sync --node node402`, `check --sync-data`, then `launch --queue --short --label r18:tpips -- bash
   scripts/run_om_tpips_node.sh probe`. The CLI is `~/.claude/skills/cluster-run/cluster`. Look for `PROBE_ENV_OK` and
   `PROBE_EMBED_OK`. Then `gate-probe --batch 1`. `om_node_plan.sh` has the order; steps 1 to 3 are done.
2. **Route (agent default):** no new env. The node's CoSiR env runs with tpips and peft from
   `/data/SSD2/cosir_r18/20261103_outside_methods/node_inputs/tpips_pylibs` on PYTHONPATH. Node and local versions differ
   (torch 2.14 against 2.11), so gate (iv) on the node's probe store decides whether the node's vectors count. If it
   fails: the local 3090 at night, and say so in the log.
3. **TPIPS time is not settled, and 2.5 h looks low (not checked).** Local batch 1: 46 s for 144 embeddings at mean 436
   visual tokens. The real sets have mean 723 tokens and 104,451 embeddings: about 15 GPU h on the 3090, so about 5 h over
   three node GPUs. The node's gate probe gives the ratio on the same 48 images: set `OM_EST_SHARD` from it. A local
   fallback does not end by 11:00 from any one night.
4. **Shard jobs (recommendation):** launch them from the build as soon as the node's gate probe passes, when 3 slots are
   free (`shared-resources.md`, DAS6 first; ticket 04 says the controller runs the plan). Nothing in the build depends
   on them, but the run chat gains about 5 h, and the real-store check of `om_data.tpips` (item 7) needs them. If you do,
   put the tags in the handoff; the run chat pulls and merges.
5. **Notes from the reviews, for the briefs of these tickets:**
   - 09: `save_parts` has no way to store the torch version and `om_paths.TURTLE`: add `extra=None`, merged into the
     sidecar. `input_sha256 = om_data.array_sha256(<exact fit input>)`. A crash between npz and sidecar leaves an npz
     alone, so the next save is refused: rewrite with `overwrite=True`. `_read_one` is private: 09 needs a public check
     of one fit for its resume.
   - 09, 11: every GPU TURTLE process sets `LD_LIBRARY_PATH` as the index says. Without it the fit crashes (nvrtc,
     from `torch.special.entr`). On the node, a TURTLE fit is not checked yet; torch differs there too.
   - 07: `method='replay'` is accepted. `save_parts` needs `loss`, `entropy`, `n_clusters`, `seconds`: pass nan or 0.
   - 08, 10: pin CLAY's `text_sha256` (index).
   - When `om_data.py` is next touched: `test_om_data.py:135` changes only the text, add an image-row change; no test
     covers a cache with reordered keys (`om_data.py:130`). `om_tpips.load` does not check `emb.npy` against `meta.json`.
6. **Dispatch:** run `codex-usage --room implementer` before each: 57% left at 00:50, an implementer needs 80%, the
   window resets at 04:35. Verdict tickets (07, 08, 10) get an Opus `high` review before merge. Sonnet `high` for
   07, 08, 10; Sonnet `medium` for 09, 11, 12. After a fix round, a Sonnet `high` scoped re-review.
7. **Open checks once the TPIPS stores exist:** real calls of `om_data.tpips` on all four sets, and MMArt `val` rows equal
   to the same painting's rows in a direct ArtELingo read. No synthetic test can show that.
8. **Tell the final review and the report:** gate (i)'s reach (build 3's handoff §4 item 4), and the review of 02: the gate
   at 100 steps on all rows cannot see batch draws, so the oracle test carries that claim. Set B on style: 43 of 100
   values are "gothic X" drift; emotion has 80 values. D5 keeps both.
9. **r17's ranges:** as before: `records/stage1_ranges.json` exists only after r17's run; ticket 08 gives `r: null` until
   then, and the run chat merges main before stage 1.
10. **The build's handoff to the run chat** carries the ticket list (`matt-chain.md`) and the end-time estimate.

## 5. Code and pitfalls

- As build 1's §5 and build 2's §5.
- `cluster check` needs node paths in the dry-run line; it maps them back through `DATA_MAP`. Local paths show as missing.
- A second CLI profile for CoSiR breaks every cluster call: `load_profiles` refuses a shared `REPO_URL`.
- The hook refuses any `rm` in this unattended chat, and the whole command with it: copy with `rsync --exclude` instead,
  and list deletions on `.scratch/pending_deletions.md`.
- Agent-tool worktrees start from main: the implementer resets to `r18-outside-methods` (the brief says so).
- Merging into main at the end: `git -C /project/CoSiR merge --ff-only r18-outside-methods`, then push. Never switch the
  branch of `/project/CoSiR`.
- In zsh a bare `======` in a command is an error.

## 6. State at handoff

- Running: nothing of r18. Beside it: r17's run (CPU queue) and r19's build.
- Uncommitted: other chats' lines in `briefs.md` and `chats.tsv` (not ours).
- Shared files changed: `cluster.conf` has two new `DATA_MAP` pairs (`r18_in`); the skill repo has
  `configs/CoSiR/dataset/om_node_cluster.yaml`, committed there by `cluster configs --adopt`.
- On node402 and DAS6: both weight snapshots in `/var/scratch/wding/cache/hub` (30.5 GB, keep until the TPIPS cells are
  done); the failed probe's job folder, for `cluster du` at the end.
- Worktrees to list on `.scratch/pending_deletions.md` at the build's end: `.claude/worktrees/agent-*` of tickets 01 to
  06, and the ticket branches.
- On disk over 1 GB: TPIPS weights in `/data/SSD2/HF_home/hub` (32 GB, keep).
