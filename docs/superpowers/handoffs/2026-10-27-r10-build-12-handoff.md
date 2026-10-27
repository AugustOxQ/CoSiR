# Handoff: re-review ticket 11's fixes and run its full self-check, merge it, then the final whole-branch review, then cut to the run chat

## Brief
- Done: ticket 15 and the MS3e test merged; ticket 11's fix wave done (113 tests, 0 disagree end to end); N1 and N2 pass on node402; the three removal smokes pass.
- State: 14 of 15 tickets merged. Ticket 11 waits for its re-review and full self-check.
- Next: the 11 re-review and self-check at once; merge; the final review (three Opus `xhigh` reviewers).
- Blocked: rule §4.2's 14:00 node deadline is past: the user decides (shift it or run locally).

> Written 2026-10-27 15:2x by "r10 build 11" (cut at the context meter's 200k note). Loop step reached: step 7, the
> build, 14 of 15 tickets merged. Direction log entry: [2026-10-27](../direction_log.md).

## 1. Read in this order

1. `.scratch/r10-recipe-sweep/notes/build8_reviews.md`, from "# Build 11 additions" on: every hand-back and decision of
   this chat, with paths.
2. `notes/review_11.md` (the review the fix wave answered) and `notes/brief_11fix.txt` (the fix brief); the fixer's
   record `rs_11_build.md` and its entry in `.claude/20261027_log.md`, both in 11's worktree.
3. `notes/brief_09rereview.txt` (a model re-review brief) and `notes/final_review_brief.md` (ready: fill `__TIP__`).
4. The build 11 handoff (`2026-10-27-r10-build-11-handoff.md`) §2 and §5: decisions and pitfalls that still hold.

Skills: `cluster-run` (only if the user decides on the node route before the run chat; the run chat does the sync).

## 2. Decided by the user, not to be reopened

- Decisions A, B, C, the grill and the brief: as in the build 7 handoff §2 (approved 2026-10-27 03:48).
- 14:2x: "start as quick as possible, but not at the cost of skipping processes": the final review runs before stage 1;
  the remaining work runs in parallel.
- Night mode (`run-mode`: night at the cut). GPU check at the cut passed (driver 595.99.02 on all three); no restart
  asked (night mode: check only).

## 3. Where things stand

| Item | State | Where |
|---|---|---|
| 01 to 10, 12 to 15, MS3e test | merged and pushed | `r10-recipe-sweep` 208f92e |
| 11 held re-derivation | fix wave done: all 9 review items; 113 tests; the fixer's 36 mutants all caught; e2e stage 0 (review fixture) 53 items, stage 4 (09's example) 66 items, ablation (10's example) 63 items, 0 disagree each; a permuted partition exits 1 | `worktree-agent-a9b6b08a3a20079fc`, 65d2d2c (merged 208f92e, not pushed) |
| rs_vp self-checks | draws 2,000: 45 items, 0 differ; points: 57 items, 0 differ (15:0x) | `/tmp/claude-0/r10-b11/vp_selfcheck_points.log` |
| N1, N2 on node402 | pass (14:43, 14:46), `gates/n1.json`, `n2.json` home | `rs_node_log.md` |
| Removal smokes, full scale | G-no_direct, G-no_content_correction, G-no_instructed_graphs: train and held ok | `/tmp/claude-0/r10-b11/removal_smoke{,2}.log` |
| Gates | V1a (13:04), V0a (13:13), V3a (12:10), V4a scorer and training (14:34, 14:35), k11, ablation graphs: pass | `outputs/gates/` |

## 4. Open points

a. **At once, in parallel:** a Sonnet `high` re-review of 11's fix wave (model `brief_09rereview.txt`: each item done or
   not; at least 10 mutants of its own, different from the fixer's list in `build8_reviews.md`; check the fixer's
   assumption that G's sample sidecars carry `arm: 'G'` against what `rs_stage0` writes, the reviewer may read the
   scorer, C13 binds only the implementer; review to `notes/rereview_11.md`), and the full self-check in the background
   (CPU, about 8 minutes; expect 77 items, 0 disagree, largest difference 0):
   `cd /project/CoSiR/.claude/worktrees/agent-a9b6b08a3a20079fc/src/test/20261027_recipe_sweep && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/CoSiR/bin/python rs_vh.py selfcheck --outputs /tmp/claude-0/r10-rs-11/full`.
   Codex: the five-hour window resets 18:02; the re-review stays with Claude (it checks code, `agent-routing.md`).
b. Merge 11 (a `.claude/20261027_log.md` conflict: keep both), push. No gate hashes `rs_vh.py`.
c. **The final review:** fill `__TIP__` in `notes/final_review_brief.md`, launch three Opus `xhigh` workers at once
   (parts H, D, A; each prompt names its part). One fix wave (Codex if `codex-usage --room` says room, else Sonnet
   `high`), a Sonnet `high` re-review, rerun any gate whose hashed files the fixes touch, merge into main, push.
d. **Key, for the user (sent to the phone at 15:1x, no answer yet):** rule §4.2 sends stages 1 and 2 to the local
   fallback "when stage 1 cannot start on the node by 14:00, when stage 2 has not ended by 22:00". The node start is
   about 20:00 at the earliest. Options: (1) an exception that keeps the node route and moves both deadlines by the
   real start (the deadlines guard the schedule; N1 and N2 pass); (2) the rule as written: stages 1 and 2 locally, G0 =
   12-27's local GS-0.02 seed 0, power check on the first run. Recommendation: (1), written as a rule exception with
   the user's words. The build does not wait for it; the run chat must have the answer before stage 1.
e. Then the cut to the run chat (`loop-next ... --label "r10 run"`); before it, the GPU check (night mode: check only).
   The run chat resyncs the node from a branch whose name holds "cluster-run" at the merged tip (`run_rs_node.sh`
   gained a `vp` mode in ticket 15).

## 5. Code and pitfalls

- `cluster launch` needs the node to hold HEAD and HEAD's subject or branch to name "cluster run"; a merge commit on
  the integration branch does not. Launch node jobs from a worktree on a branch named `...cluster-run...` (this chat
  used `/project/CoSiR-wt-node`, branch `r10-rs-cluster-run`, at ea1bd84).
- The cluster CLI dry-runs any `scripts/run_*.sh` that mentions `CLUSTER_DRY_RUN` and accepts only `python
  main_cosir.py` lines; `run_rs_node.sh` must never mention it again.
- The graph removals need `gates/ablation_graphs.json` in the outputs folder (a scratch smoke needs a copy).
- Ticket 15's stage 0 queue can only smoke the seed 4 training step (rs_stage0 and rs_vh have no smoke mode).
- The rest as in the build 11 handoff §5 (worktrees start at main; the log conflicts; F = G keys; gates bind SHA-256s).

## 6. State at handoff

- Running: nothing (every subagent handed back; no GPU lock, background run or cluster job of this chat).
- Uncommitted: nothing after the cut commit (`.scratch/` notes and briefs are outside git).
- On disk over 1 GB: `/tmp/claude-0/r10-b11/removal_smoke/` (1.2 GB), and the build 11 handoff's list. node402: two
  unpulled job folders (tags 20261027-134327-ea1bd84 and 20261027-134558-ea1bd84; code copies 107 MB each, N2's 500
  MB of smoke scratch). All on `.scratch/pending_deletions.md` (rows added at this cut).
