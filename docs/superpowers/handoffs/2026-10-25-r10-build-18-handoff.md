# Handoff: go on with the r10 build (tickets 05, 15 to review; 04, 06, 07, 13, 16 to build), the final review and the merge, then open "r10 run 14"

## Brief
- Done: tickets 03, 08, 10, 11, 14 merged (each verdict ticket reviewed by Opus `high`, fixes applied; GPU regressions pass).
- State: 9 of 16 tickets merged; 05 built, not reviewed; 15 built, not reviewed (it reproduces the CT stage 1 and 2 verdicts exactly).
- Next: read the part 1 agreement check, review 05 and 15, build 07, 13, 04, 06, then 16; final review; merge.
- Blocked: nothing. Night mode; GPU check passes; no container restart.

> Written 2026-10-25 05:00 by "r10 build 17", cut at the context meter (over 200k). Loop step reached: step 7 (build) in
> progress. Direction log entry: [2026-10-21 · D2 ablation read](../direction_log.md).

## 1. Read in this order

1. This handoff.
2. The ticket index `/project/CoSiR/.scratch/r10-build-16/issues/00-index.md` (states, binding shared interfaces,
   coverage line); tickets `NN-*.md` beside it. Ticket 16 has controller notes at its end.
3. The briefs every subagent gets: `.scratch/r10-build-16/notes/implementer_brief.md` (its session line is this
   chat's; leave it) and `reviewer_brief.md`. Review findings so far: `notes/review_{03,10,11,12,14}_findings.md`.
4. The spec `docs/superpowers/specs/2026-10-25-recipe-and-backbones-design.md` and the rule
   `src/test/20261025_recipe_backbones/DECISION_RULE.md` only as far as a ticket or review needs.
5. `~/.claude/references/matt-chain.md` (build, final review, merge) and `codex-jobs.md`.

Skills: none (`implement-spec` per `matt-chain.md`).

## 2. Decided by the user, not to be reopened

As the [build 16 handoff](2026-10-25-r10-build-16-handoff.md) §2: the spec's §4 (D1 to D19, D30). Night mode; no
restart.

## 3. Where things stand

- Integration branch `r10-recipe-backbones` (worktree `/project/CoSiR-wt/r10-recipe-backbones`, pushed, tip 327b4f6).
  Round tests there: 232 passed, 13 skipped (command in the build 17 handoff §3; `RB_SLOW=1` adds real-data tests).
- Merged in build 17 (controller decisions on the review findings are in each findings file):
  - 10 `rb_student`: six review fixes (shared V1 gate, V4 any-seed refusal, shared locked regression writer, switch
    test at full scale, K4/K5 tied to the pick, base-input assert). Stage 2 GPU regression rerun after the fixes:
    all four `student/` items pass.
  - 08 stores: diff read; GPU smoke of all four paths (CLIP B/32 full, DINOv2 full and ArtELingo-28 image-only, the
    instructed Qwen ArtELingo-28 stores) and `--verify` pass; metadata matches 09's readers. Change log copied to
    `/project/CoSiR/.claude/20261025_log.md`.
  - 11 `rb_matched`: GPU regressions pass at both scales (six `matched/` items each: retrain on the Qwen image and
    inference from the saved P3 and P2 mappers, bit for bit). Review fixes: V4 refusal; training of all seeds before
    any held inference (crash resume); seed and `eval_prob` tests.
  - 03 `rb_ami`, `rb_pick`: review PASS (hand case and 12-22's real exports re-derived exactly). Added:
    `--vp2-failed RECIPE=CAUSE` (a candidate still failing VP2 after its forced rerun is excluded; R0-5 stops part 1),
    `triggers {K4, K5}` and `label_sha256` in `pick.json`.
  - 14 `rb_a28`: review re-derived the labels (729 unique, 17 ties) and d_A28 exactly. Fixes: K1 always at full scale;
    only output faults count as V3; default model list = K1, K2, K3 plus K4/K5 from `pick.json`'s triggers;
    `scale_reason` when a model falls back to sample.
- Built, not reviewed: 05 `rb_vp1` on branch `r10rb-t05` (worktree `/project/CoSiR-wt/r10rb-t05`, commit 29d888c;
  35 tests). Points for its Opus `high` review: (a) the implementer read `test_rb_pick.py`'s key assertions for the
  `pick.json` layout (a small independence breach; judge it); (b) VP1's "agree" must require equal partitions and the
  same pick, not only e, lo, hi, gain; (c) its style codes (sorted 27 names) may differ from `rb_data`'s, so
  `label_sha256` could differ on style alone: compare the label partitions (up to relabelling), not the codes;
  (d) the reviewer should compare VP1's P labels with `rb_data.p_labels()` on the real names (`rb_data.p_names()`).
- Part 1 agreement check (started 04:51, detached, owned by build 18 now): fake P exports from real images
  (`normalize((1 - w) * default + w * instructed)`, w from 0 to 0.6 along `rb_rows.ORDER`), then `rb_pick` (03,
  integration branch) at 2,000 draws, then `rb_vp1` (05). Folder `/tmp/claude-0/rb17_e2e_p1/`: `status.txt` (one line
  per step, `end` when done), `pick.log`, `vp1.log`, `out/part1/pick.json`, `out/part1/rederive_vp1.json`. Expect
  agreement; a disagreement is a finding for 03 or 05, settled by the rule's text.
- Built, not reviewed: 15 `rb_v2` on branch `r10rb-t15` (worktree `/project/CoSiR-wt/r10rb-t15`, commit 6365693,
  merged with the integration tip at 48a8f94; 21 tests, plus two acceptance tests: `--ct-check 1` and `2` reproduce
  the CT's `verdict_stage1.json` and `verdict_stage2.json` exactly: d, every A_e, both bounds, all 2,000 bootstrap
  values, eligible seeds, all 35 saved CT partitions). Points for its Opus `high` review: (a) `read_block` guesses the
  verdict block layout (`candidates`/`models`/`blocks`, `d` or `d_A28`, ...): check it against `rb_score`'s and
  `rb_a28`'s real writers on synthetic runs of both, so the first real run does not fail on format; (b) at stage 5 it
  takes full scale when the model has any stage 4 block, while `rb_a28` needs a block with an outcome: align with
  the rule; (c) a missing saved partition counts as "unchecked", not a failure, although V2 asks for equal
  partitions; (d) style and genre labels are not re-derived (only emotion decides): judge against V2's text.
- Not started: 07 (sheet), 13 (part 2 reported), 04 and 06 (now unblocked by 03), 16 (needs all; read its controller
  notes).
- Codex: the five-hour window had 70% left at 04:55 and resets at 07:23; an implementer needs 80%. One Codex
  implementer at a time after the reset (07 or 13 fit); the rest go to Claude.

## 4. Open points for the next step (agent work; nothing for the user)

a. On start: read the agreement check's `status.txt`; if it ended, compare `rederive_vp1.json`'s agree flags; then
   review 05 (Opus `high`) with the points above, and 15 when built.
b. Dispatch at most three workers at once, each implementer in its own worktree from the integration branch
   (`git worktree add -b r10rb-tNN /project/CoSiR-wt/r10rb-tNN r10-recipe-backbones`); keep `subagent-watch --watch`
   in a background Bash. Plumbing tickets (04, 06, 07, 13, 16): Sonnet `medium`, controller diff read and smoke run.
c. Ticket 16's driver passes `--vp2-failed` to `rb_pick` after a forced rerun, reads `pick.json`'s `triggers`, and
   requires every expected regression item (student, matched, scorer) at stages 2 and 3a.
d. Final whole-branch review (Opus `xhigh`, `matt-chain.md`), one fix wave, a scoped re-review (Sonnet `high`); merge
   into main and push; worktrees and ticket branches go on `.scratch/pending_deletions.md`.
e. The run chat's handoff: ticket list (index table), the queue order (spec §3.5), commands from `rb_queue.md`, the
   end estimate against 11:00 (D32); store extraction commands are in ticket 08's report (run from the merged main
   checkout: `scripts/extract_features.py --dataset artelingo_full --backbone clip_b32`; `... dinov2_large
   --image-only` for `artelingo_full` and `artelingo28`; `rb_extract_a28.py`); then `loop-next <handoff> --label
   "r10 run 14" --unattended`.

## 5. Code and pitfalls

- As the build 17 handoff §5. Also: `rb_score.load_output` misreads B1/B6 ArtELingo-28 sidecars (it expects the CT's
  `held_sha256`); readers of those files call `rb_score.check_sidecar` (14 does).
- `rb_matched` and `rb_student` both refuse new training once any held output of the arm exists (V4); both resume
  verified seeds.
- The CT mappers' inference is bit for bit only on `cuda`; the matched-arm regression refuses another device.

## 6. State at handoff

- Running: the part 1 agreement check (`/tmp/claude-0/rb17_e2e_p1/run.sh`, nohup, CPU, 8 workers; about 30 minutes).
- Uncommitted: nothing (`.scratch/` is outside git).
- On disk over 1 GB: `/tmp/claude-0/rb17_e2e_p1/` (4.1 GB, fake exports; on `.scratch/pending_deletions.md`).
