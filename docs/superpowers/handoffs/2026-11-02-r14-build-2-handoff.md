# Handoff: continue the r14 build: review and merge tickets 03 and 05, dispatch the rest of the 12 tickets, watch the extraction queue, final review, then open the run chat

## Brief
- Done: 12 tickets cut; ticket 01 (extraction code) merged; the extraction queue runs on the local GPU since 04:27.
- State: tickets 03 (c* trainer) and 05 (ArtELingo clustering) built on their branches, not yet reviewed; 9 tickets wait.
- Next: finish 03 and 05 (fresh workers), Opus reviews, merge; then 02, 04, 06, 07, 09 as the frontier allows.
- Blocked: nothing. Codex had no room until 07:50.

> Written 2026-11-02 04:35 by "r14 build" (cut at the context meter, 200k). Loop step reached: step 7, building.
> Direction log entry: [2026-11-02](../direction_log.md).

## 1. Read in this order

1. The first build handoff, sections 2 to 5 only (decisions, stored inputs, verdict-code list, pitfalls):
   [2026-11-02-r14-build-handoff.md](2026-11-02-r14-build-handoff.md).
2. The spec from §1 on: [2026-11-02-backbones-sweep-design.md](../specs/2026-11-02-backbones-sweep-design.md).
3. The tickets: `.scratch/r14-backbones-sweep/issues/` (gitignored, main checkout only): `00-index.md` (table,
   shared interfaces, coverage line, checked), then each ticket. The common implementer brief every dispatch points
   at: `.scratch/r14-backbones-sweep/notes/implementer_brief.md`.
4. `~/.claude/references/matt-chain.md` and, before a Codex job, `~/.claude/references/codex-jobs.md`.

Skills: `cluster-run` (ticket 10 and the node check); `to-tickets` and `implement-spec` already read (their SKILL.md,
with matt-chain's changes).

## 3. Where things stand

- **Integration branch** `r14-backbones-sweep` (checked out in the main checkout; pushed): `bs_paths.py`
  (controller-owned constants), ticket 01 merged (6d56715): `bs_stores.py` (store reader by key), `bs_extract.py`
  (EmoSet unpack, six-view extractor, parity 0(i), store check 0(ii)), `bs_extract_queue.sh`.
- **EmoSet unpacked**: 5,905 val and 17,716 test images and annotations in `/data/SSD/emoset/` (2.3 GB), no train.
- **Extraction queue**: started 04:27, detached (setsid nohup; it does not end with a chat). Status file
  `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage0/extract_queue_status.txt` (one line per step, ends
  `QUEUE DONE ok` or `QUEUE FAILED at <step>`), log `extract_queue.log` beside it. Order: parity 0(i) (6 cases; the
  worker's smoke parity passed all six, Qwen default min cosine 0.99943), Affection SigLIP 2, SigLIP 1, CLIP L/14
  train, the six EmoSet views, check 0(ii). The worker's timing suggests about 3 h (EmoSet Qwen about 35 min per
  view), so it ends near 07:30. Watch it with a Monitor until-loop on the status file. The 0(iv) and 0(v) check
  trainings wait for it (GPU lock).
- **Ticket 03** (c* trainer): code and 37 tests on branch `r14-t03-cstar-trainer` (6d4226a), not merged, not
  reviewed. The worker's context ran out: smokes, the integration merge and an optional mutation check are left.
  Its full progress note, commands for 0(iv) and 0(ix) and choices: `.scratch/r14-backbones-sweep/notes/t03_progress.md`.
  Next: a fresh Sonnet `high` worker finishes the left items (CPU smoke of `check-export` on one seed; merge the
  tip), then the Opus `high` review, then merge; then 0(iv) and 0(ix) on the GPU after the extraction queue.
- **Ticket 05** (ArtELingo clustering): code and 16 tests on branch `r14-t05-artelingo-clustering` (e835c75), not
  merged, not reviewed; the worker's context ran out and the controller stopped it. Replays 0(vi), the 0(vii) matched
  arms and the 0(xii) F part pass with difference 0. The 0(vii) sha replay fails on a reader bug (a sidecar name the
  removal arms do not use), not on data. Not run: the Qwen stand-in, the K part of 0(xii), the removals' held changes.
  Full note, commands and choices: `.scratch/r14-backbones-sweep/notes/t05_progress.md`. Next: a fresh Sonnet `high`
  worker fixes the sha reader and runs the rest, then the Opus `high` review, then merge.
- **Frontier after 03 and 05**: 02, 04, 07 (need 01: ready), 06, 09 (no blockers); then 08 (01, 02), 10 (03, 04),
  11 (05, 06, 10), 12 (all). At most three implementers at once.
- **Model routing**: Codex first when `codex-usage --room implementer` says room (none until 07:50 at 04:20);
  otherwise workers: verdict code Sonnet `high`, plumbing Sonnet `medium`; each verdict ticket gets an Opus `high`
  review before its merge; plumbing gets the controller's diff read and a smoke.
- **node402**: status ok at 04:23, 3 A6000 free, 61h44m left; the CLI's default node is node401, so pass
  `--node node402`.

## 5. Code and pitfalls

- The workers' brief tells them to base their worktree on `r14-backbones-sweep`, branch `r14-tNN-<slug>`, and
  merge the integration tip before hand-back. Merge with `git merge --no-ff r14-tNN-<slug>` in the main checkout.
- List each merged ticket's worktree and branches on `.scratch/pending_deletions.md` (ticket 01's is there).
- Ticket 01's choices: Affection key order is the Qwen store's `index.json` order; EmoSet keys in val.json then
  test.json order (ticket 02 sets the evaluation order); `bs_stores` view names are `qwen_default`, `emotion_focus`,
  `emotion_focus_photo`, `clip_l14`, `siglip2`, `siglip1`.
- The queue script re-executes itself under `flock -o -w 3600`; a leftover `<folder>.pending` stops it (never remove
  one without looking).

## 6. State at handoff

- Running: the extraction queue (above). No subagents (03 handed back; 05 stopped, its replay chain ended with it).
- Uncommitted: nothing.
- On disk over 1 GB: `/data/SSD/emoset/` val/test images (2.3 GB, keep: r14 input); `EmoSet-118K.zip` (11.3 GB,
  keep until r14's storage summary).
