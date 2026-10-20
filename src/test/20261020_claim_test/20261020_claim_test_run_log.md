# Claim test, stage 0 and stage 1: run log (r10 run 5)

Chat "r10 run 5", unattended (night mode), from the main checkout `/project/CoSiR`. Times are Amsterdam.

## Timeline

| Time | Step | Result |
|---|---|---|
| 04:08 | GPU check (driver 595.99.02 in all three places; GPU idle, no lock; load 0.6) | pass |
| 04:08 | End estimate: stage 0 by about 04:45, stage 1 by 06:00 to 07:15, CPU by 06:30 to 07:45; local run | well before 11:00 |
| 04:08 to 04:12 | `ct_queue_stage0.sh`: D2 seed 0 (55 s), core seed 0 (184 s), check (23 s) | PASS, 24 of 24 items; every regression in hash mode (bitwise equal); commit 1159209 |
| 04:13 to 04:55 | `ct_queue_stage1.sh`: D2 seeds 0 to 4 (174 s), core seeds 0 to 4 (761 s), paper settings Stage 1 (10 runs, 1,583 s) | failed at the end of the paper step, in the R-COL writer (below) |

## Failure 1: R-COL recheck refused `qwen_qwen` seed 2

**Problem.** `ct_rcol.read_stage1` recomputes the fit counts in float64 from the saved centers and fit latents and
asserts an L1 difference of at most 0.01% of fit pairs (6). For `percept/paper/stage1/qwen_qwen__s2.npz` it was
8,892. The paper step stopped before any paper Stage 2 training; no held output existed.

**Investigation** (scripts in the session scratchpad, `dbg/rcol_diag.py`, `dbg/rcol_ties.py`).
- That run stopped at the 200-epoch ceiling, with two pairs of duplicate centers: 9 and 43 (distance 0 to 1e-10)
  and 10 and 83 (distance 3.4e-7). About 10,000 fit pairs are ties between duplicates (top-2 Q margin under 1e-6).
- The GPU float32 argmax gave nearly all tied pairs to the lower index (saved counts 9: 3,420, 43: 0, 10: 3,235,
  83: 150), as the rule's "lowest index on a tie" says. The float64 recheck splits them by noise below float32
  precision (9: 1,640, 43: 1,780, 10: 569, 83: 2,816).
- With ties defined as Q within a relative 1e-7 to 1e-5 of the row maximum, 150 pairs remain, all between the
  near-duplicates 10 and 83 (float32 rounding). The 9 other runs agree exactly at every tolerance up to 1e-6.
- No judgment depends on the split: strict kept counts come from center norms; the largest share of this run is
  about 0.05, far from the 0.5 collapse threshold.

**Root cause.** The build's recheck (not part of the decision rule) compared exact argmax counts and did not allow
for ties between duplicate centers, which DEC can produce.

**Fix** (Sonnet `high` worker, test-first; review Opus `high`: PASS). `ct_rcol.py` now treats centers whose Q is
within a relative 1e-6 (`TIE_RTOL`) of a pair's maximum as ties: a saved topic disagrees only when it is not tied
for the maximum, and counts are compared summed per tie group (connected components of tied centers). Run entries
record two new keys, `tie_pairs` and `tie_groups`. Kept masks, kept counts, shares, collapse flags and entries are
computed as before (byte-identical code). Tests: `test_ct_rcol.py` 67 passed (60 old, 2 of them with changed
expectations that keep their intent, 7 new); the new tests fail on the old code, and the tolerance test fails at
`TIE_RTOL = 0`. Real artifacts: all 10 pass with count L1 0 and 0 topic disagreements; `qwen_qwen` seed 2 has
10,002 tie pairs in groups [9, 43], [10, 83], [25, 68]; no other run has a tie. Change log:
`.claude/20261020_log.md`. Not done (review's optional note): a per-center bound on counts inside a tie group.

**R-COL as the saved arrays give it** (read before any held output): every strict run is collapsed (kept counts
`clip_roberta` 1, 2, 2, 98, 98; `qwen_qwen` 98, 99, 99, 99, 99; the rule collapses under 10 or over 90), so B2 and
B3 leave the bar. No K = 40 run is collapsed (largest share 0.04 to 0.18), so B4 and B5 enter with all 5 seeds.
The R-COL file itself is written by the next stage 1 queue.

| Time | Step | Result |
|---|---|---|
| 05:02 to 05:06 | `ct_queue_stage0.sh` again (the edit changed the code identity): D2 seed 0 (53 s), core seed 0 (211 s), check (24 s) | PASS, 24 of 24, every regression bitwise equal; old seed 0 artifacts moved to `outputs/stale/` (306 MB, two folders) |

## Run 6 (chat "r10 run 6", unattended, night mode)

| Time | Step | Result |
|---|---|---|
| 05:08 | GPU check (driver 595.99.02 in all three places; GPU idle, no lock; load 1.7; 56 GB RAM free; 246 GB free on `/`) | pass |
| 05:08 | End estimate: paper Stage 2 20 to 40 min, held inference minutes, instructed images about 30 min, so stage 1 by about 06:30; CPU queue about 25 min, by about 07:00 | well before 11:00 |
| 05:09 | `ct_queue_stage1.sh` again (run 5's stage 1 logs copied to `outputs/logs/run5_stage1/`) | started |
| 05:09 to 05:10 | require_stage0, budget, frozen, held_list (0 to 4 s each); d2 (18 s) and core (20 s) verify the saved seeds 0 to 4, no retraining | pass |
| 05:10 | paper: the 10 saved Stage 1 runs verified complete, no retraining; Stage 2 starts; strict `clip_roberta` seed 0 skipped (1 kept topic, rule: at least 2) | running |
| 05:10 to 05:18 | paper Stage 2 (strict and K = 40, P1 and P4; 476 s) | pass |
| 05:10, 05:18 | `outputs/percept/rcol_stage1.json` written at 05:10 by the paper step (Stage 2 needs the kept masks), before any held output; the rcol step (6 s) found it and reused it; strict kept counts `clip_roberta` 1, 2, 2, 98, 98 and `qwen_qwen` 98, 99, 99, 99, 99, all collapsed, so B2 and B3 are not eligible; K = 40 largest shares 0.043 to 0.178, none collapsed, so B4 and B5 are eligible with seeds 0 to 4; `qwen_qwen` seed 2 has 10,002 tie pairs | pass; as read in run 5 |
| 05:18 to 05:19 | held inference (65 s): D2 heads and PercepT arms, seeds 0 to 4; strict `clip_roberta` seed 0 has no held file (no Stage 2) | pass |
| 05:19 | instructed held images start (GPU) | running |
| 05:21 | V2 starts: Codex had no room (5 h window 19% left, needs 20%, resets 07:01), so a Sonnet `high` worker re-derives from `.scratch/r10-claim-test/v2/v2_brief.md` (the rule and the data only, CPU) | running |
| 05:19 to 06:16 | instructed held images (3,449 s; three stores `emotion_focus_held`, `style_word_held`, `genre_word_held`, 54 MB each, in the shared cache) | pass; stage 1 queue done, failed=0 |
| 05:21 to 05:26 | V2 worker finished (CPU, about 4 min): `v2_result.json`, `v2_partitions/`, `v2_bootstrap.npz`; no check failed | compared after the verdict |
| 06:17 | CPU queue starts (verdict, then diagnostics); end estimate about 06:45 | running |
