# r21: DEC loss on every head of c*: round log

Spec: `docs/superpowers/specs/2026-11-11-r21-dec-heads-design.md`. Decision lines: `decisions.md`.

## Build (2026-11-11, build chat)

- 2026-11-11 06:37: tickets 01 to 06 written (`.scratch/r21-dec-heads/issues/`); integration branch `r21-dec-heads`.
  Codex had no room for implementer jobs (five-hour window 61% left, 80% needed), so Claude workers build.

## Build chat 2 (2026-11-11, from 06:51)

- 06:54: **gate 0a PASS** (full run, `r21_score.py gate0a` at `r21/t05` a282ae62, 155 s on one CPU slot). Test AMI at
  the class count, 5-seed c* means equal r17's file: 0.102413 / 0.410926 / 0.510070; teacher 0.088875 / 0.406315 /
  0.501466; all diffs 0.0. c* mAP per seed equals r20's `ablation_ret.json` for all five seeds, max |diff| 0.0.
  Record: `/data/SSD2/cosir_r21/20261111_dec_heads/scores/gate0a.json`.
- 06:55: reviews of 02 and 05 (Opus `high` workers) and builds of 03 and 06 (Sonnet `high` workers) started in
  parallel; 03 and 06 branch from the integration tip with the unreviewed 02 and 05 merged in (review fixes merge later).
- 07:08: ticket 05 reviewed (Opus `high`: F1 to F9; F1 to F5 should-fix) and fixed (86095ff9, Sonnet `high`
  fixer; F9 kept); 72 tests pass; `val` on saved c* seed 0 still equals `stage1_ranges.json` (diff 0.0). Merged.
- 07:15: ticket 02 reviewed (Opus `high`: no code defect on the DEC-off path, which matches `ct2_d2.run` line by
  line; 7 of 16 hand mutations survived the tests) and fixed (ce46aa8e, Sonnet `high` fixer): new tests kill all 7;
  the teacher form now asserts its feature rows are in the head's fit order (C6); the parity test covers a distilled
  row and a patience stop. Merged; the three test files pass on the integration branch, each in its own process (07:19).
- 07:19: the three test files in one pytest process in the worktree give 1 failure and 11 errors in
  `test_r21_score.py` (each passes alone: 72, 29, 27). Cause: frozen `20261021_d2_ablation/ab_paths.py` (via r17's
  `ns_paths.add_rounds()`) pins the main checkout's round folders and asserts imports come from there, while
  `r21_train.py` adds the worktree's own folders. In a worktree the two copies clash; on main they are the same files.
  Rule for this build: run test files one per process in worktrees; the final check runs them together on main.
- 07:25: ticket 06 built (fb9ecc6c, Sonnet `high`; 37 tests). **Regression exact**: `table` on r10's removal arm
  `k11=40-no_buddy` gives AMI at 20 clusters (change, interval, both absolute values) equal to r10's `held.json` and mAP
  equal to `ablation_ret.json`, all 30 cells diff 0.0; c* against itself gives zero changes, zero intervals and "no
  clear change". 1,402 s on one CPU slot under load about 20 (two arms). Records in this chat's scratchpad
  (`t06/reg/reg_table.json`, `regress.log`). Review (Opus `high`) started.
