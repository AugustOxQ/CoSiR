# Handoff: continue the r14 build: review tickets 07 and 09, build 08 and 10, then 11 and 12, run the GPU checks after the extraction queue, final review, then open the run chat

## Brief
- Done: tickets 01 to 06 merged, each with its Opus review and fixes; 07 and 09 built, not reviewed; every CPU replay passes with difference 0.
- State: extraction queue at CLIP L/14 train (Affection SigLIP 2 and SigLIP 1 done); six EmoSet views left, end about 08:00.
- Next: Opus reviews of 07 and 09; build 08 and 10 (both unblocked); then 11, 12; GPU checks 0(iv), 0(v) after the queue.
- Blocked: nothing. Codex has room again after 07:50.

> Written 2026-11-02 05:25 by "r14 build 2" (cut at the context meter, 200k). Loop step reached: step 7, building.
> Direction log entry: [2026-11-02](../direction_log.md).

## 1. Read in this order

1. The first build handoff, sections 2 to 5 only (decisions, stored inputs, verdict-code list, pitfalls):
   [2026-11-02-r14-build-handoff.md](2026-11-02-r14-build-handoff.md).
2. The spec from §1 on: [2026-11-02-backbones-sweep-design.md](../specs/2026-11-02-backbones-sweep-design.md).
3. `.scratch/r14-backbones-sweep/` (gitignored, main checkout only): `issues/00-index.md` and tickets 07 to 12;
   `notes/implementer_brief.md` (every implementer dispatch points at it); `notes/reviewer_brief.md` (every Opus
   ticket review points at it); `notes/controller_notes.md` (what later tickets must pick up from the reviews, the
   status schema of ticket 09, the GPU check commands with expected outputs).
4. `~/.claude/references/matt-chain.md` and, before a Codex job, `~/.claude/references/codex-jobs.md`.

Skills: `cluster-run` (ticket 10 and the node check).

## 3. Where things stand

- **Integration branch** `r14-backbones-sweep` (main checkout; pushed, 3f6026b): tickets 01, 03 (60cd8a7), 06
  (90027dd), 05 (a64cbcc), 02 (18e7c96), 04 (3f6026b) merged. Each verdict ticket had an Opus `high` review ("APPROVE
  with notes" every time) and a fix wave for its should-fix items before the merge. 134 tests in the five test files (ticket 01 is plumbing, proven by its smoke).
- **Built, not reviewed:**
  - Ticket 07 (Affection scores, SC4, SC5, SC6 photo cells): branch `r14-t07-affection-scores`, tip 3b28603,
    worktree `.claude/worktrees/agent-a3f03ef390a5af391`. 16 tests; replay 0(vi) r13 part PASS (51 items, diff 0,
    137 s); Qwen stand-in reproduces r13's SC1 to SC4 and r' ranges exactly. Choices for the reviewer: the stage 1
    pick sits in `OUT/stage1/affection_ranges_<enc>.json` beside the ranges; `write_record` refuses to change an
    existing file without `--overwrite`; SigLIP 2 and 1 frozen picks via `ac_select.arm_table` with r12's r_ret,mAP;
    CLIP L/14 and Qwen keep r12's pick; written before its tests (not strictly test-first; 9 sabotages caught).
  - Ticket 09 (outcome code): branch `r14-t09-outcomes`, tip 0868d55, worktree
    `.claude/worktrees/agent-a4e8bc6a37c1515e5`. 89 tests; 25 sabotages caught; written before its tests. Its
    choices (WIN-only cells "WIN" / "not WIN"; `r` required on SC6, SC9, SC10; per-encoder 19 cells; the part c
    qualified sentence adds the plain-image clause, rule review N1; the status schema) are in `controller_notes.md`.
    The reviewer must check them against §6 and the rule review's Q4.
- **Frontier:** 08 (01 and 02 merged: ready), 10 (03 and 04 merged: ready); then 11 (05, 06, 10), 12 (all). At most
  three implementers at once; the load reached about 39 with five workers.
- **Stage 0 checks done (CPU):** 0(ix) c* five seeds (max diff 6.1e-05) and photo 20 exports (0.0); 0(x) through the
  real cut path (2,328 keys, exact); 0(vi) 12-28, r11 and r13 parts (diff 0); 0(vii) matched arms, sha (9 arms), all
  eight removals (120 items) (diff 0); 0(xii) F (with the SC1 frozen margins, 27 items), K, r11 frozen mAPs (diff 0).
- **Stage 0 checks left:** 0(i), 0(ii) in the queue (parity passed at 04:28); 0(iii) once the EmoSet Qwen store
  exists: `bs_emoset.py nearcopy` (about 1 to 2 min CPU), then commit `RECORDS/nearcopy_emoset.json`; 0(iv) and 0(v)
  on the local GPU after the queue (commands and expected outputs in `controller_notes.md`); the SigLIP 2 c* smoke;
  `bs_artclu.py replay xii-f --recompute --loader bs_stores --encs siglip1` (and clip_l14) once the stores exist;
  0(viii) on node402 (ticket 10); 0(xi) (ticket 08).
- **Extraction queue:** detached (setsid nohup), not tied to a chat. Status file
  `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage0/extract_queue_status.txt`, ends `QUEUE DONE ok` or
  `QUEUE FAILED at <step>`. Affection SigLIP 2 took 1,583 s and SigLIP 1 1,784 s (slower than the worker's guess);
  left: CLIP L/14 train, the six EmoSet views (Qwen about 35 min each), check 0(ii). Expected end about 08:00. Watch
  it with a background until-loop on the status file (also exit when `pgrep -f bs_extract_queue.sh` finds nothing).
- **node402:** status ok at 04:40, 3 A6000 free, 61h28m left; pass `--node node402`.
- **Model routing:** Codex first when `codex-usage --room implementer` says room (5h resets 07:50); otherwise workers:
  verdict code Sonnet `high`, plumbing Sonnet `medium`; Opus `high` review before each verdict merge.

## 5. Code and pitfalls

- An `isolation: "worktree"` worker may start from a stale base: the implementer brief's reset step catches it
  (ticket 04's worker hit it). Finishers and fixers work in the existing worktree, no isolation.
- Ticket 10 inputs: `bs_cstar.py train --keep none` needs `--root` (refuses any encoder's keep-all root); build the
  k=10 and k=80 graph stores once before parallel jobs; `bs_photo.py train --encoder qwen --sweep [--rows ...]` writes
  `OUT/sweeps/photo/<row>/s<seed>/` (the sweep's own `ref` included); X photo runs need X-c*'s weights where
  `--cstar-root` points (default `OUT/cstar/<enc>`). More in `controller_notes.md`.
- Ticket 12 copies `OUT/stage1/artclu_ranges_<enc>.json`, `artret_{ranges,picks}_<enc>.json` and
  `affection_ranges_<enc>.json` into the committed `RECORDS/stage1_<enc>.json`, and writes `RECORDS/status.json` in
  ticket 09's schema.
- List each merged ticket's worktree and branches on `.scratch/pending_deletions.md` (01 to 06 are there).

## 6. State at handoff

- Running: the extraction queue (above). No subagents, no background job of this chat.
- Uncommitted: nothing.
- On disk over 1 GB: `/data/SSD/emoset/` val/test images (2.3 GB, keep: r14 input); `EmoSet-118K.zip` (11.3 GB,
  keep until r14's storage summary); the new stores under `/data/SSD2/pre_extract/affection_local/` (keep: r14 input).
