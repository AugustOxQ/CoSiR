# Handoff: r10 sweep 1: build and run stages 0 and 1 of the aspect graph sweep (the graph number sheet) tonight; ask the user the open points when they return

## Brief
- Done: the purification pilot's reports, the direction log entry for r10, the spec draft of the exploration sweep.
- State: purified pair student 0.094 vs control 0.070 (bar 0.12 fails); the user said "firstly just go for testing different ideas".
- Next: build and run stage 0 (regression check) and stage 1 (about 60 graph rows: prune, fuse, guided) on CPU and about 50 min GPU.
- Blocked: decision A, spec approval and the storage summary wait for the user (at a workout).

> Written 2026-10-18 16:50 by `r10 decide 6` (cut near 200k context). Loop step reached: step 6 (spec) drafted, not
> approved; decision A still a draft; stages 0 and 1 started on the user's word (spec D10). Direction log entry:
> [2026-10-18](../direction_log.md). Run mode: night. GPU check at 16:45 passes (driver 595.99.02, GPU free, no lock).

## 1. Read in this order

1. **The spec**, [2026-10-18-aspect-graph-sweep-design](../specs/2026-10-18-aspect-graph-sweep-design.md): §3.2 (the
   graph rows), §3.4 (scores and which rows go downstream), §3.5 (stages and the regression gate).
2. **The pilot's code** to reuse: `src/test/20261018_purified_buddy_pilot/` (`pbp_graphs.py` builds the graphs and
   their lifts; `pbp_train.py` the pair student; `pbp_score.py` the scores) and its log. The CLIP content graph:
   `pbp_clip_content.py`.
3. **The pilot's report** for the numbers stage 0 must match:
   [2026-12-10_purified_buddy_pilot](../../reports/auto/percept/2026-12-10_purified_buddy_pilot.md).
4. Before any Codex job: `~/.claude/references/codex-jobs.md`. Before the build: `~/.claude/references/matt-chain.md`
   (exploration code, so no ticket reviews; the controller's diff read and one smoke run).

Skills: none.

## 2. Decided by the user, not to be reopened

- Everything in the [decide 6 handoff](2026-10-18-r10-decide-6-handoff.md), section 2 (decision B2, the grill
  answers, the parked ideas, the number-sheet guidance).
- Decision C on the purification pilot (user, before ours), recorded in the direction log 2026-10-18.
- 2026-10-18 16:20: "when encounter critical decision problem you can still ping me to notice me, but firstly just
  go for testing different ideas" (user). Ping the phone only for a critical decision.

## 3. Where things stand

- Reports: [auto](../../reports/auto/percept/2026-12-10_purified_buddy_pilot.md) and
  [user-read](../../user_read/2026-10-18_purified_buddy_pilot.md), both checked (Codex cold read and fidelity check,
  one fix wave); index row added; `check_reports_sum.py` OK.
- The aspect-prompt store (`/data/SSD2/pre_extract/artelingo_full/qwen3vl_emb_2b_aspect_prompts/<instruction>/`)
  holds images only for the 5,000 evaluation paintings (`img.npy` 5,000 x 2048) and all 99,999 captions
  (`txt.npy`). Training images exist only for emotion_focus (`emotion_focus_trainimg/`, 14,889). Default training
  images come from the main Qwen store, as in the pilot.
- So stage 1 first extracts style_word and genre_word for the 14,889 training images (about 23 min each at 11
  images per second, under the GPU lock; the emotion_focus_trainimg extraction of the night pilot is the pattern:
  `src/test/20261017_image_transfer_pilot/`).
- wrapup-check: only the known routing open item (auto-mode refusals of Codex launches; user: watch, no rule change).
  disk-check printed nothing. `sync.sh` not yet run in this chat.

## 4. Open points for the next step

1. **Key: decision A** (the user's): confirm the direction and its type and loop size. Agent default in the spec:
   exploration sweep. Ask answer-first when the user returns.
2. **Key: spec approval**: paste the brief with the file link; stages 2 and 3 wait for it.
3. **Key: the storage summary**: the aspect-prompt store (6.3 GB, ledger row deferred, recommend keep; it grows by
   about 120 MB with the two new extractions) and the six rows of `.scratch/pending_deletions.md`.
4. **Minor:** whether the number-sheet guidance becomes a global rule (`habit-scope.md`); now project memory.
5. **Run:** after stage 1, a short auto report and a half-page user-read report with the sheet; the user reads it
   first (decision C).

## 5. Code and pitfalls

- New round folder `src/test/20261018_aspect_graph_sheet/` with a new module prefix (taken: `apc_`, `pav_`, `itp_`,
  `ewm_`, `pbp_`); suggest `ags_`.
- Stage 0 gate: the emotion, purified and shared lifts 2.4205 / 2.3849 / 2.5529 exactly, before any new row.
- Graph rows that need both sides of one aspect (F-side, F-rank, G-agree, genre) need the training images of that
  instruction; emotion has them already.
- Weighted rows (G-down) need per-pair weights in the head's loss.
- The auto-mode check refused reads of Codex's json log before; read the files Codex writes instead.
- Gitignored: `*.json`, `*.npy`, `*.log`, `outputs/`. Tables `.md` are committed.
- Held data: none read; stay on the train side.

## 6. State at handoff

- Running: nothing (no subagent, no Codex job, no GPU lock).
- Uncommitted: Codex leftovers of earlier rounds (on `.scratch/pending_deletions.md`), `sample.npz`, the untracked
  archive result folders from before.
- On disk over 1 GB: the aspect-prompt store (6.3 GB, deferred).
