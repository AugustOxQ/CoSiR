# Handoff: r18 build 3: review ticket 02, run the short GPU jobs, then tickets 04 and 06 to 12, final review, merge, cut to run

## Brief
- Done: build 2 wrote all 12 tickets after r17's merge. Tickets 01, 03 and 05 are merged; gate (i) passes at 6.6e-7 against 1e-5.
- State: ticket 02 is built, gate (ii) CPU part 2.5e-7 against 1e-5, not reviewed. No GPU job has run yet.
- Next: Opus review of 02, the short GPU jobs, then tickets 04 and 06.
- Blocked: nothing. GPU check passes.

> Written 2026-11-04 00:35 by tab "r18 build 2". Loop step reached: step 7, tickets 01 to 05 built, cut at the 200k
> context meter. Direction log entry: [2026-11-03 · MMArt test read](../direction_log.md), bullet "r18 spec approved".

## 1. Read in this order

1. The ticket index, all of it: `/project/CoSiR/.scratch/r18-outside-methods/issues/00-index.md` (shared interfaces,
   coverage line, the controller decisions from r17's merged code). Tickets beside it; common briefs in `../notes/`.
2. §3 and §4 below.
3. `~/.claude/references/matt-chain.md` and `~/.claude/references/codex-jobs.md`, before the first dispatch.
4. Only when needed: the spec, [2026-11-03-outside-methods-design.md](../specs/2026-11-03-outside-methods-design.md)
   §3.2 to §6; pitfalls of build 1, [2026-11-03-r18-build-handoff.md](2026-11-03-r18-build-handoff.md) §5.

Keep the context small. Builds 1 and 2 both reached 200k. `om_facts_brief.md` is 43 KB: never print it whole, give
line ranges (§A CLAY 5 to 24, TURTLE 42 to 54). Subagents read the files; give them paths.

Skills: cluster-run (ticket 04, its `SKILL.md`).

## 2. Decided by the user, not to be reopened

As build 1's handoff §2 (D12 code route, D17 order, spec approved 2026-11-03 20:45, no container restart).

## 3. Where things stand

Integration branch `r18-outside-methods`, worktree `/project/CoSiR/.claude/worktrees/r18-int`, at cc7d2b6e, pushed.
Controller-owned `om_paths.py` is committed and imports r17 from this checkout; its fit counts give 192 + 220 = 412.

| # | Ticket | State |
|---|---|---|
| 01 | CLAY math, gate (i) | merged; Opus `high` review APPROVE with notes; 11 tests, 9 sabotages caught |
| 02 | TURTLE loop, gate (ii) | built on `r18-t02-turtle` (424031ad), worktree `.claude/worktrees/agent-a25ed8ad798747561`; 12 tests, 7 sabotages by the implementer; **review due** |
| 03 | TPIPS extractor | merged (controller's diff read); no GPU run yet |
| 04 | TPIPS on node402 | open; blocker 03 is merged |
| 05 | Text set B | merged (controller's diff read); no generation yet |
| 06 to 12 | | open, order in the index |

Status lines are in the ticket files. Build 2 ran only Claude workers: Sonnet `high` for 01 and 02, Sonnet `medium`
for 03 and 05, Opus `high` for the review of 01.

## 4. Open points for the next step (agent defaults)

1. **Review of 02** (Opus `high`, `notes/reviewer_brief.md`, ticket `issues/02-turtle.md`). Give it these points from
   the hand-back: the weight norm is written by hand (`g` and `v`), not torch's `weight_norm`; one inner init is drawn
   before the loop, then per step batch indices and a fresh init; batch indices come from a local
   `np.random.RandomState(seed)`; `fit` sets the global `torch.manual_seed`; `loss` and `entropy` are the last
   step's. Tests were not rerun after the last merge. Ask for a check that a gate at 100 steps on 2,000 rows (the full
   set) still proves the order at batch 10,000 of 16,384.
2. **Short GPU jobs after 02 merges** (each through `res-queue gpu --short --label r18:<step>`, from r18-int; first
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`):
   - a. gate (ii): `om_turtle.py gate-ii --device cuda` (env CoSiR). It writes the `cuda` block; the file's top-level
     `pass` is false until both parts exist. Ticket 11 reads that top-level `pass`.
   - b. text set B: `om_textset_b.py all` (env CoSiR; frozen once written). Read `lists.json` afterwards: counts per
     aspect, `short`, `capped`, header lines that the parser kept.
   - c. TPIPS: `/root/venvs/tpips/bin/python om_tpips.py prepare` (CPU, 20 s; 18,433, 16,384, 48 images), then
     `extract --set probe --batch 1 --store-root OUT/stage0/probe_b1` and `merge --set probe --of 1 --store-root …`.
     This is the first real call of the extractor.
   - d. D20: the same probe at batch 4 and 8 in length-sorted order. The earlier probe's batches were not sorted, so
     its result does not settle D20. Time them from the job log; use batch > 1 only when faster and within gate (iv).
3. **Dispatch next:** 04 and 06 (06's reader blocker, 03, is merged; its blocker 01 too), then 07, 08, 09, then 10,
   then 11, 12. Run `codex-usage --room implementer` before each: 65% left at 00:32, an implementer needs 80%, the
   five-hour window resets at 04:35. Verdict tickets (06, 07, 08, 10) get an Opus `high` review before merge.
4. **Gate (i) reach** (review of 01, note): the reference cannot see centring in `subspace`, and set B's truncated
   path has no official reference. Round log has it. Give it to the final review; the report states it. No new
   attended run of the clone is planned.
5. **Small review notes on `om_clay.py`, left as they are:** the reflection threshold is on the squared norm (cannot
   fire on real data); text rows accept a deviation of 1e-4 (tighten to 1e-5 if the file is touched again).
6. **r17's ranges:** `records/stage1_ranges.json` in r17's round exists only after r17's run, which is running now.
   Ticket 08 gives `r: null` until then. The run chat merges main before stage 1.
7. **The build's handoff to the run chat** carries the ticket list (`matt-chain.md`) and the run's end-time estimate.
   TPIPS: node402 (about 2.5 h), else the local 3090 at night. TURTLE: ticket 09's timing smoke sets the estimate.

## 5. Code and pitfalls

- As build 1's §5, and build 2's §5 for the `DATA_MAP` facts and the launchable forms.
- `ns_clu._boot` asserts a candidate of [5, 3, n]; ticket 10 repeats SC5's one start (index, round log).
- Agent-tool worktrees start from main: the implementer resets to `r18-outside-methods` (the brief says so).
- Merging into main at the end: `git -C /project/CoSiR merge --ff-only r18-outside-methods`, then push. Not
  `update-ref` (r17's build 4 handoff): runs import from the main checkout, which must hold the files. Never switch
  the branch of `/project/CoSiR`; other chats commit there.
- In zsh a bare `======` in a command is an error: use another separator.

## 6. State at handoff

- Running: nothing of r18. Beside it: r17's run (CPU queue) and r19's build.
- Uncommitted: other chats' lines in `briefs.md` and `chats.tsv` (not ours).
- Worktrees to list on `.scratch/pending_deletions.md` at the build's end: `.claude/worktrees/agent-*` of tickets 01,
  02, 03, 05, and the ticket branches.
- On disk over 1 GB: as build 1's §6 (TPIPS weights, 32 GB, keep).
