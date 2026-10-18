# Handoff: build PercepT core (five arms, regression check) and the missing ablation rows, then open the run chat

## Brief
- Done: follow-up 1 read (decision C recorded); ablation table added to its user-read report; PercepT core grill (two
  rounds) and spec approved 23:34; no container restart (user).
- State: the best default-input network vs the prompted embedding: −0.018 emotion, −0.005 style, +0.003 genre. The
  PercepT rows (4, 5) and rows 1, 6, 10, 13 are empty.
- Next: tickets and build per the spec, an end-to-end smoke that times each run kind, then open the run chat.
- Blocked: nothing.

> Written 2026-10-18 23:38 by "r10 follow-up 1". Loop step reached: after the spec's approval (step 6), before step 7.
> Direction log entry: [2026-10-18, aspect graph follow-up](../direction_log.md).

## 1. Read in this order

1. [The spec](../specs/2026-10-18-percept-core-design.md), from §1 (the brief is the user's page). It is the build's
   only source of what to build.
2. The facts brief `.scratch/r10-percept-core/facts.md` (not in git): the September code, the four fixes' edit sites,
   feature stores, joins, risks. Cite it in the tickets.
3. The grill record `.scratch/r10-percept-core/grill.md`: who decided what, the user's words.
4. The code audit [2026-12-04_percept_code_audit.md](../../reports/auto/percept/2026-12-04_percept_code_audit.md):
   the four fixes, Eq. 2's reading, Algorithm 1.
5. Stage 1 code to reuse for rows 1, 6, 10, 13 and the scoring: `src/test/20261018_aspect_graph_sheet/`
   (`ags_rows.md`, `ags_head.py`, `ags_fu.py`, `ags_fu_queue.sh`).
6. The table to fill: [user-read report](../../user_read/2026-10-18_aspect_graph_followup.md), "Which part of the
   design does the work?".

Skills: to-tickets, implement-spec (`~/.claude/references/matt-chain.md`).

## 2. Decided by the user, not to be reopened

- The spec's §4 (D1 to D15, with who decided) and its approval, 2026-10-18 23:34.
- No container restart before this chat (23:34). Local runs only: DAS6 has no reservation.
- Storage of stage 1 `outputs/` (1.1 GB), follow-up `outputs_fu/` (1.2 GB) and `graphs/` (67 MB): decided at the end
  of the loop (user, 18:45); rows in the disk ledger.
- The user frames this as exploration: "as many trials as possible to enlarge the ablation table and then pick".

## 3. Where things stand

- GPU check at 23:31 passed (driver, module and library 595.99.02; no process, no lock). Run mode: day at 23:38;
  the main chat switches it when the user says so.
- Disk: system 260 GB free, SSD2 1.9 TB free.
- Codex: 5h 97%, weekly 94% left (23:13).

## 4. Open points for the next step

None for the user. Agent work the spec leaves to the build: locate the September snapshot script (K 60 to 40, seed
42; its report is `res/percept/percept_stage1_embedding_snapshot_pilot_report.md`) and the extended-seed Stage 2
record on branch `experiment/percept_topic_pipeline`; check the per-caption Qwen joins (`default`, `emotion_focus`)
and the patch-cache name join for the 19,889 frozen paintings.

## 5. Code and pitfalls

- New round folder `src/test/20261018_percept_core/`, module prefix `pc_` (round folders share `sys.path`).
- The September code lives in worktree `/project/CoSiR-buddy_prototype_conditioning`; read it with `git show`, copy
  what is needed into the round folder; never merge that branch.
- `ags_head.ami_scores` hard-codes `(5000, 2048)`: make the scoring dimension-neutral without changing its numbers
  (the spec's stage 0 checks row 2 to 1e-6).
- Shared caches are read-only: the patch cache `/data/SSD2/pre_extract/artelingo_percept_patch_features/` (6.3 GB
  train file), the Qwen stores in `/data/SSD2/pre_extract/artelingo_full/`. New RoBERTa features go in a new folder
  there.
- `PYTHONDONTWRITEBYTECODE=1`; CPU work at `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`; GPU under
  `flock -n -o -E 75 /tmp/gpu0.lock` (queued runs: `flock -o -w`).
- Bare `cluster` is the Graphviz tool; the cluster CLI is `~/.claude/skills/cluster-run/cluster` (not needed here).
- Smoke outputs go to the session scratchpad; delete nothing (unattended), list items in `.scratch/pending_deletions.md`.

## 6. State at handoff

- Running: nothing.
- Uncommitted: nothing (`.scratch/` is outside git).
- On disk over 1 GB: `src/test/20261018_aspect_graph_sheet/outputs/` (1.1 GB) and `outputs_fu/` (1.2 GB), deferred.
