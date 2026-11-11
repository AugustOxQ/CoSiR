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
