# Handoff: continue the r14 build: fix the c* trainer for non-2048 encoders, merge tickets 10 and 11, build 12, GPU checks, final review, then open the run chat

## Brief
- Done: tickets 07, 08, 09 merged after Opus reviews and fixes; 10 (driver and node plumbing) and 11 built; 0(iii), 0(xi) and 0(xii) pass.
- State: blocker in merged ticket 03: the c* trainer fails for every encoder whose image is not 2048 wide (`rg_loss.py:88`), so part a cannot train yet.
- Next: fix that bug (verdict code, with review); merge 10a, 10b and 11; build 12; GPU checks after the queue (about 08:00).
- Blocked: nothing. Codex has room after 07:50.

> Written 2026-11-02 06:15 by "r14 build 3" (cut at the context meter, 200k). Loop step reached: step 7, building.
> Direction log entry: [2026-11-02](../direction_log.md).

## 1. Read in this order

1. The first build handoff, sections 2 to 5 only: [2026-11-02-r14-build-handoff.md](2026-11-02-r14-build-handoff.md).
2. The spec from §1 on: [2026-11-02-backbones-sweep-design.md](../specs/2026-11-02-backbones-sweep-design.md).
3. `.scratch/r14-backbones-sweep/` (gitignored, main checkout only): `issues/00-index.md`, tickets 03, 10, 11, 12;
   `notes/controller_notes.md` (all review notes for later tickets, the GPU check commands, the stage 0 results);
   `notes/t10_progress.md` (ticket 10's design, 10b's and 10a's hand-backs, the blocker); `notes/t11_handback.md`;
   `notes/implementer_brief.md` and `notes/reviewer_brief.md` (every dispatch points at them).
4. `~/.claude/references/matt-chain.md` and, before a Codex job, `~/.claude/references/codex-jobs.md`.

Skills: `diagnosing-bugs` (the blocker), `cluster-run` (the node smoke).

## 3. Where things stand

- **Integration branch** `r14-backbones-sweep` (main checkout; pushed, 1ebe97c): tickets 01 to 09 merged (07 f543e1b,
  09 7186eb7, 08 5f376c2), each after an Opus `high` review ("APPROVE with notes" every time) and a fix wave. The three
  verdict test files of this chat pass together: 151 tests (`test_bs_emoset_score`, `test_bs_outcome`,
  `test_bs_affection`).
- **The blocker (ticket 03, merged, verdict code):** `bs_cstar.train('siglip2', 'ref', ...)` fails at `rg_loss.py:88`
  (`assert teacher.shape == img.shape`): the genre head's loss compares the 2048-wide Qwen teacher with the student's
  image input (1152 for SigLIP, 768 for CLIP L/14). Ticket 03's width test checked only the first layer; the SigLIP 2
  c* smoke never ran (it waited for the GPU). Find what that loss term does with `img` and what spec §3.2 says the
  X-c* arm keeps ("Qwen's instructed graphs, default captions and teachers as in c*"; only the image input changes).
  The precedent: the 10-25 and 10-26 rounds trained students on these same encoders (`20261025_recipe_backbones`,
  `20261025_encoders_genre`); follow what their code fed this term. Routing: a hard bug's first attempt follows the
  default (Codex when `codex-usage --room bug` says room, else a Sonnet `high` worker), then an Opus `high` review
  (verdict code), then 0(iv) again (the Qwen path must stay exact) and the SigLIP 2 c* smoke on the GPU. If the fix
  changes what X-c* trains on beyond the image input, that is a spec question: record it for the run chat's handoff
  (anything that changes a number goes there, not to the user).
- **Built, not merged** (controller diff read, then merge; plumbing, so no ticket review):
  - Ticket 10a, the job driver: branch `r14-t10-node-jobs`, tip 5a84e06, worktree
    `.claude/worktrees/agent-a3c85e9e08a1e7398`. CPU smoke proved the rerun rule, the sweep queue and the folder
    format; details in `t10_progress.md`.
  - Ticket 10b, the node plumbing: branch `r14-t10b-node-plumbing`, tip c71c2f6, worktree
    `.claude/worktrees/r14-t10b-node-plumbing`. Probe job `20261102-044946-c800ed8` on node402 read all 17 input
    groups; 9 DATA_MAP entries added (`cluster.conf`, local); skill repo commit 86f2d79. The about 9 GB of Qwen stores
    are not synced yet (the node smoke's first `check --sync-data` does it; run it in the background).
  - After merging 10a and 10b: the GPU smoke on node402 (`bash scripts/run_bs_node.sh smoke`, via `cluster check
    --sync-data`, `launch`, `watch`, `pull`; `bs_node_plan.sh` prints the commands). It needs the blocker fixed
    (its first unit is SigLIP 2 X-c*). Untested: concurrent launches sharing `/local/wding/res/r14_bs`.
  - Ticket 11, the part b sheets: branch `r14-t11-partb-sheets`, tip 7ae6b42, worktree
    `.claude/worktrees/agent-a96c8c04ca41113aa`. Smoke on synthetic folders passed; real ablation held AMI for the
    relational removal equals r10's `held.json` (difference 0). Its choices to check are in `t11_handback.md`.
- **Frontier:** ticket 12 (blocked only by 10 and 11's merges; it may start now against their branches). Then the
  final whole-branch review (Opus `xhigh`), one fix wave, a Sonnet `high` re-review, merge into main, push.
- **Stage 0 done:** 0(iii) near-copy cut (val 135 of 5,905 removed, test 368 of 17,716; record committed, 1ebe97c);
  0(xi) wording rule, `--recompute` 62 items, difference 0 (scratch run; ticket 12's stage0 reruns it into OUT);
  0(xii) F recompute through `bs_stores` for SigLIP 1 and CLIP L/14 (difference 0); and from build 2: 0(i), 0(ii)
  parity, 0(vi), 0(vii), 0(ix), 0(x), 0(xii) F, K and r11 mAPs.
- **Stage 0 left:** 0(ii) the queue's final `check` step; 0(iv) and 0(v) on the local GPU after the queue (commands
  and expected outputs in `controller_notes.md`); the SigLIP 2 c* smoke (after the fix); 0(viii) on node402.
- **Extraction queue:** detached, not tied to a chat. Done: Affection SigLIP 2, SigLIP 1, CLIP L/14 train (361 s),
  EmoSet Qwen default (2,309 s, 06:09). Left: EmoSet `emotion_focus`, `emotion_focus_photo` (about 38 min each), CLIP
  L/14, SigLIP 2, SigLIP 1, then `check`; expected end about 08:00. Status file
  `/data/SSD2/cosir_r14/20261102_backbones_sweep/outputs/stage0/extract_queue_status.txt` (ends `QUEUE DONE ok` or
  `QUEUE FAILED at <step>`). A watcher script: `/tmp/claude-0/-project-CoSiR/1b6ad64e-1837-4f79-ba96-fb40cc313d4e/scratchpad/watch_queue.sh`
  (exits on each new status line or when the queue process is gone).
- **After the queue:** ticket 04's EmoSet exports (`bs_photo.py export --arm ARM --dataset emoset` for `photo_qwen`,
  `cstar_qwen`; `photo_<enc>` only after part a), then ticket 08's `stage2 --write-record` (CPU, minutes) once the
  stores and exports exist. Its record goes to git before any EmoSet test number (ticket 12).
- **node402:** reserved until 2026-11-04 18:06; 3 A6000; free at 05:49.
- **Model routing:** Codex first when `codex-usage --room <kind>` says room (5h resets 07:50; weekly 37% left at
  05:26, resets 11-04 08:27); otherwise workers: verdict code Sonnet `high`, plumbing Sonnet `medium`; Opus `high`
  review before each verdict merge. Context: each implementer of tickets 08 and 10 passed 200k; brief them to read
  code by function and to hand back a progress note near the limit.

## 5. Code and pitfalls

- **`*.json` is gitignored** (`.gitignore:154`): committed records need `git add -f` (`RECORDS/nearcopy_emoset.json`
  was; ticket 08's `stage2_emoset.json` and ticket 12's `stage1_<enc>.json`, `status.json`, `verdicts.json` must be;
  ticket 08's `require_record` compares with git HEAD).
- Ticket 12 must also take the review notes in `controller_notes.md`: the Qwen stand-in runs only under its own
  `BS_OUTPUTS` (the writers refuse the production folder now); `status.json` writes 'failed' under both stage 0 keys
  for a gate both paths use, and `photo_runs` 'failed' when the X-c* runs failed; commit the stage 1 files before
  `bs_affection score`; call gate 0(xi) in its `--recompute` form.
- The unattended-chat hook refuses `cluster clean` (it deletes): node leftovers go on `.scratch/pending_deletions.md`.
- Harness quirks on the cluster CLI (10b): literal paths, no `$VAR` or `~` as arguments, no `sleep` loops.
- An `isolation: "worktree"` worker may start from a stale base; the implementer brief's reset step catches it.
  Finishers and fixers work in the existing worktree, no isolation.
- List each merged ticket's worktree and branches on `.scratch/pending_deletions.md` (01 to 09 are there).

## 6. State at handoff

- Running: the extraction queue (above). No subagents; no background job of this chat.
- Uncommitted: nothing in git (the scratch notes are gitignored by design).
- On disk over 1 GB: `/data/SSD/emoset/` val/test images (2.3 GB, keep: r14 input); `EmoSet-118K.zip` (11.3 GB, keep
  until r14's storage summary); the new stores under `/data/SSD2/pre_extract/affection_local/` and
  `/data/SSD2/pre_extract/emoset/` (keep: r14 inputs); on node402 the synced inputs (keep: the runs need them) and the
  probe's folders (on pending_deletions).
- GPU check at 06:12: driver 595.99.02 answers; the kernel module and the library match; the GPU holds only the
  extraction queue (10 GB).
