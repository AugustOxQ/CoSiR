# Handoff: r16 report: answer the user's questions on r15 and r14, then the stage note, then the direction

## Brief
- Done: r16 decide recorded the user's reading of r15, ran a SemArt check, and finished the wrap-up and storage.
- State: r15 holds on four encoders (teacher 18 WIN, 4 LEVEL, 2 BELOW). Experiments are paused for the report to the collaborator and the supervisor.
- Next: answer the user's questions on r15 and r14; then the stage note; then the direction (decision A).
- Blocked: the user runs the deletion of five merged GitHub branches (refused by the permission check).

> Written 2026-11-03 14:51 by tab "r16 decide". Loop step reached: r16 after decision C and the wrap-up; decision A
> open, paused by the user. Direction log entry:
> [2026-11-03 · MMArt test read](../direction_log.md).

## 1. Read in this order

1. The direction log's r16 entry and "Now" ([direction log](../direction_log.md)): the user's reading, ours, and the
   open direction.
2. As the user's questions need them: the r15 auto report
   [`2027-01-03_mmart_test.md`](../../reports/auto/percept/2027-01-03_mmart_test.md) and the r14 auto report
   [`2027-01-02_backbones_sweep.md`](../../reports/auto/percept/2027-01-02_backbones_sweep.md); their user-read reports
   [`2026-11-03_mmart_test.md`](../../user_read/2026-11-03_mmart_test.md) and
   [`2026-11-02_backbones_sweep.md`](../../user_read/2026-11-02_backbones_sweep.md). Answer from the reports and the
   round records (`src/test/20261102_mmart_test/records/`, r14's round folder), never from memory; heavy reading goes
   to a subagent.
3. The SemArt check log [`20261103_semart_check_log.md`](../../../src/test/20261103_semart_check/20261103_semart_check_log.md),
   if SemArt comes up.
4. For the stage note: the last stage report [`2026-10-28_recipe_sweep.md`](../../reports/stage/2026-10-28_recipe_sweep.md)
   (with its `.html` outside-view page and `_devlog.md` post), then `~/.claude/rules/report-kinds.md`,
   `~/.claude/references/outside-view-report.md` and `~/.claude/templates/stage_report_template.md`.

Skills: none for the questions.

## 2. Decided by the user, not to be reopened

- No further experiments for now; the user reports to the collaborator and the supervisor first (2026-11-03, direction
  log r16 entry).
- The order: the user's questions on r15 and r14, then a stage note, then the direction, then r16 (user, 2026-11-03).
- The r15 storage (2026-11-03; disk ledger): `outputs/exports/` kept until the paper's numbers are frozen, the rest
  deleted.
- The r15 design and rule (spec approved 2026-11-02 18:37, e96a7cd; cut at 0.95, D19, 471859e). Deadlines: abstract
  2026-11-10, paper 2026-11-16, supplementary 2026-11-23.

## 3. Where things stand

- **r15 (MMArt, 16,384 paintings outside ArtELingo):** holds, GO on Qwen3-VL-Embedding-2B, SigLIP 2, SigLIP 1 and CLIP
  L/14. All nine plain-image and matched-PercepT cells WIN per encoder (Qwen emotion AMI 0.0609 against 0.0407).
  Teacher: 18 WIN, 4 LEVEL, 2 BELOW (r14 on ArtELingo: 18, 5, 1); BELOW is emotion retrieval, Qwen −0.0010
  [−0.0015, −0.0004] and CLIP L/14. Re-derivation 6,697 items, 0 failed. Unseen artists: 15 paintings, no cells.
- **The user's reading (C):** good, but the Qwen emotion retrieval cell was unexpected. Ours: expected, since emotion
  retrieval was the weakest teacher cell on ArtELingo too (Qwen −0.0023 with plain cosine, +0.0007 after the read-out
  pick; CLIP L/14 below in r14); on MMArt the gap sits mostly in contentment (AP 0.2413 against 0.2547).
- **r14:** c* holds on SigLIP 2, SigLIP 1 and CLIP L/14; the frozen photo model beats its teacher on EmoSet (AMI
  +0.0181, mAP +0.0320).
- **SemArt check (CPU, existing stores):** 11,162 paintings by 2,717 artists without ArtELingo train paintings, after
  the 0.95 cut; type labels only (10 classes, close to genre). The 0.95 cut misses copies across the two sources (Last
  Supper at 0.944), so the full kept set (17,840) still holds some. Parked in the direction log.

## 4. Open points for the next step

- Key: the user's questions on r15 and r14.
- Key: the stage note. "Stage note" is the user's word; ask its form and readers first. Likely the next stage report
  (since 2026-10-28: r11 to r15, with its outside-view page and dev-log post), or a shorter note for the collaborator
  and the supervisor.
- Key, after the stage note: the direction (decision A). Options seen: freeze and write; the SemArt genre test
  (parked); a second photo aspect (parked); the emotion retrieval cell stated in the paper as a limit (our
  recommendation) or examined (the contentment class).
- Minor: the five merged origin branches (`.scratch/pending_deletions.md`): the user runs
  `git push origin --delete r14-backbones-sweep r14-t03-cstar-trainer r14-t05-artelingo-clustering r14-t10b-node-plumbing r15-mmart-test`.
- Minor: `routing-check` prints the three items deferred to a tooling chat after 2026-11-09; nothing new.

## 5. Code and pitfalls

- zsh here: `echo ====` fails (`=word` expansion) and stops the command list; arrays start at 1; use
  `git commit -F - <<'EOF'`.
- `*.json` and `*.log` are gitignored: records need `git add -f`. The paper chat also commits on this repo: commit only
  your own paths.
- Auto report names keep CoSiR's sequence dates (r15: 2027-01-03); the stage reports in `docs/reports/stage/` carry
  the real date (2026-10-28_recipe_sweep).
- SemArt: the CSVs are latin-1 with `?` for lost letters; the Web Gallery of Art name forms differ from WikiArt's
  slugs (aliases and a wildcard in `sc_check.py`).

## 6. State at handoff

- Running: nothing of this chat. The r15 build 2 session (877af933) is still open as an idle process; its scratchpad
  was deleted with the user's approval.
- Uncommitted: nothing of this chat.
- On disk over 1 GB: `/data/SSD2/cosir_r15/20261102_mmart_test/outputs/exports/` 9.0 GB (keep; disk ledger).
