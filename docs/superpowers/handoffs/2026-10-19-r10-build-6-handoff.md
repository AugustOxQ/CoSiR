# Handoff: build A3 (PercepT core keeps a fixed 40 topics), then open the run chat that reruns the five arms

## Brief
- Done: decision C read; the user chose option b with a fixed K = 40; the spec is amended (A3); wrap-up done.
- State: run 2's one established arm (P2) scored 0.036 / 0.162 / 0.217, because two of its seeds kept 2 and 3 topics;
  our network 0.081 / 0.348 / 0.458.
- Next: one verdict ticket: keep the 40 of 100 centers with the most fit pairs; Algorithm 1 only reported. Then the
  final review, merge, and the run chat.
- Blocked: nothing.

> Written 2026-10-19 10:55 by "r10 decide 9". Loop step reached: step 6 (spec amendment A3) done; the next chat does
> step 7 (build). Direction log entry:
> [2026-10-19, PercepT core arms read](../direction_log.md).

## 1. Read in this order

1. The spec [2026-10-18-percept-core-design.md](../specs/2026-10-18-percept-core-design.md), from §1: §3.2 fix 3,
   §3.4 ("Beside it"), §4 (D9 and A3), §6 (the added unit test), §8 (the two count rows), §9 ("A3 rerun").
2. This handoff's §3 and §5, then the previous one's §5,
   [2026-10-19-r10-decide-9-handoff.md](2026-10-19-r10-decide-9-handoff.md).
3. The ticket index `.scratch/r10-percept-core/issues/00-index.md` ("Shared interfaces") and tickets
   `03-pc-fixes.md`, `04-pc-stage1.md`, `06-pc-score.md` (the code that A3 changes).
4. `~/.claude/references/matt-chain.md` (build 6 is small: one or two tickets, then the final review).

Skills: none.

## 2. Decided by the user, not to be reopened

- A3: a fixed K = 40 replaces Algorithm 1 (user, after seeing the recommendation, 2026-10-19 10:40, option b of the
  r10 decide 9 chat). No grill: the user waived it ("If ok go for it, no need to grill me").
- Which 40: the 40 of 100 centers with the most fit pairs, argmax over all 100 after the joint training, ties to the
  lower index (agent default, shown to the user; in A3). Not September's 60-to-40 lowest-norm pick: in the three
  99-topic runs of run 2, norm and pair share correlate positively (Spearman 0.40 to 0.44).
- The spec's D1 to D15 (2026-10-18 23:34), A1 and A2 (2026-10-19 08:20), descriptive mode, the comparator rule D15.
- The stage 2 decision of our model waits for the rerun (next decide chat).
- No container restart before the build (user, 2026-10-19 10:58); the GPU check passed (driver, module and library
  595.99.02).

## 3. Where things stand

Run 2 (γ = 2, Algorithm 1) per Stage 1 version, topics kept for seeds 0 / 1 / 2: CLIP + RoBERTa 99 / 2 / 3; Qwen +
Qwen 1 / 99 / 99; Qwen + RoBERTa 1 / 1 / 1. In the 99-topic runs about 40 topics hold at least 1% of the pairs.
Results, reports and the C13 re-derivation of run 2 are committed (`f25d83b`, `5fe91bd`).

**What A3 changes in the code** (facts from this chat; the tickets decide the form):
- `pc_fixes.kept_centers(center_norms, gamma=2)` is the Algorithm 1 mask. A3 needs a new verdict function: the top-40
  mask in original center order from per-center fit-pair counts, ties to the lower index, and a unit test on known
  counts (ties, and fewer than 40 non-empty centers). The Algorithm 1 tests stay.
- `pc_stage1.py` (about lines 230 to 240): sets `kept` from Algorithm 1 at γ = 2 and writes `K` as the dict of γ
  counts (`'1.5'`, `'2'`, `'3'`, `'5'`). It deletes `inputs` (the fit rows) before the cut, so the fit-pair argmax over
  all 100 centers must run before that. The JSON should also record K = 40, the kept topics with at least one fit pair
  and with at least 1% of fit pairs (§3.4), and the γ counts under their own key.
- `pc_score.py` (about lines 126 to 150) takes `k = counts['2']` and asserts `meta2['K'] == k`; its K < 2 branch
  explains the cut by Algorithm 1. After A3, K comes from the kept mask. Check `pc_stage2.py`, `pc_ablation.py` and
  the tables for the same `counts['2']` use.
- The C13 re-derivation brief (`src/test/20261018_percept_core/rederive_run2/task.txt`) has no "established" filter in
  its comparator; the run chat's copy for run 3 needs one.

**The run chat's work** (for its handoff): copy run 2's outputs first (`outputs/stage1/`, `outputs/stage2/`,
`outputs/repeat/`, `stage1_repeat.json`, `pc_results.json`, `pc_tables.md`, `pc_ablation_table.md` to `*_run2.*`;
the build may script it); the GPU queue only (stage 0, 9 Stage 1, 15 Stage 2, the repeat check; run 2 took 18
minutes, 09:50 to 10:08); `pc_score.py`; `pc_ablation.py`; the C13 re-derivation (Codex); the auto report and the
user-read report; then the next decide chat (`r10 decide 10`). The CPU queue (rows 1, 6, 10, 13) is not rerun.

**Timing.** Build about 1 to 2 hours; the run about 20 GPU minutes plus scoring, re-derivation and reports. It runs
during the day; the user accepted the GPU use ("go for it").

## 4. Open points for the next step

None for the user. Agent choices: ticket split, Codex or Claude per `agent-routing.md`, the `*_run2.*` copy as a
script or a run-chat command.

## 5. Code and pitfalls

- The decide 9 handoff's §5 holds: the queues write the main checkout's round folder; `pc_queue_status.txt` is
  appended (read the block after the last `PC_gpu_SUBMIT`); one GPU queue only; a rerun overwrites the outputs.
- Merges in the integration worktree, not the main checkout. Ticket worktrees need `res` and
  `src/test/20261017_prompted_aspect_views/sample.npz` linked from the main checkout (gitignored).
- The worktrees and local branches of builds 4 and 5 are deleted (user, 2026-10-19 10:58); `origin/r10-stage0-a1`
  stays. Build 6 starts a fresh integration branch from main.
- A GPU smoke of the queue in smoke mode before the merge (as build 5: `pc_queue_gpu.sh --foreground --smoke DIR`);
  every Stage 1 smoke run must keep 40 topics, and P5 must train.

## 6. State at handoff

- Running: nothing. GPU free; no lock.
- Uncommitted: nothing (after this chat's commit).
- On disk over 1 GB: `src/test/20261018_aspect_graph_sheet/outputs/` (1.1 GB, deferred in the ledger); the other
  deferred items are on `.scratch/pending_deletions.md`.
