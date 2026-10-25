# Recipe pick and backbone claim tests: run log (r10 run 14)

Chat "r10 run 14", unattended (night mode), from the main checkout `/project/CoSiR`. Times are Amsterdam (CET from
2026-10-25 03:00). Queue logs and status files: `outputs/` (gitignored); stdout of each queue in the session scratchpad.

## Timeline

| Time | Step | Result |
|---|---|---|
| 06:59 | Checks: load 1.4, 56 GB RAM free, GPU idle (33 MiB, no lock), driver 595.99.02; 211 GB free on `/`, 1.9 TB on `/data/SSD2` | pass |
| 06:59 | `rb_run.py budget` at start, nothing run: all four candidates 12:38 to 14:50; `RB_DEFER="K5"` 11:14 to 13:00; `"K4 K5"` 10:10 to 11:35 | past 11:00 in every option: split (D32), chosen after the pick (below) |
| 07:00 to 07:03 | `rb_queue_stage0.sh` (V0: R0-5, R6-3, R7-3 seed 0 on the old 5,000 against 12-22's exports; 126 s) | PASS 9/9 items bit for bit; genre labels on 23,682 of the 29,232; scoring time: partitions 7.1 s per export, AMI 0.0145 s per call, part 1 bootstrap about 1,568 s in one process (stand-in on a random unit-norm array, not an export) |
| 07:03 to 07:22 | `rb_queue_stage1.sh`: six recipes (07:03 to 07:11; 53 to 101 s each), pick (370 s), VP1 (351 s) | VP1 agrees; **P-R6-5** (no own-caption terms, teacher share 1/5); triggers K4 yes, K5 no; no VP2 failure, no forced rerun |
| 07:24 | Split chosen (below) | chain with K2, K3, K4 now; both part 2 report queues in the evening |
| 07:24 | `rb_queue_stage1x.sh` (background, CPU lock; probe on the GPU lock) | running |
| 07:24 | chain: `stores`, `stage2`, `stage3`, `stage3a`, `stage4`, `stage5`, one after another (no `RB_DEFER`) | running |
| 07:24 | `stores`: CLIP B/32 full and the three ArtELingo-28 instructed Qwen stores exist (skipped); K5 not triggered, no DINOv2 store | done |
| 07:24 to 07:27 | `stage2`: student regression (65 s), matched-arm regression (10 s), scorer check (122 s) | PASS, 27 of 27 items |
| 07:24 to 07:41 | `stage1x`: four-view sheet (133 s), part 1 reported readings (680 s), ceiling probe (209 s, GPU) | done, no step missing |
| 07:27 to 07:55 | `stage3`: per candidate train (179, 179, 192 s), infer (15, 16, 198 s; K4's waited for the probe's GPU lock), matched arm (M2 16 s, M3 14 s); scorer (441 s), check-verdict, V2 (375 s) | V2 agrees (no candidate skipped); K2 S1-GO d +0.0165 [+0.0110, +0.0177]; K3 S1-IN d +0.0046 [−0.0006, +0.0057]; K4 S1-GO d +0.0190 [+0.0137, +0.0201]; the bar is B1 own (0.0855) for all three |
| 07:55 to 08:00 | `stage3a`: student regression at full scale (178 s), matched arm (13 s), scorer check (115 s) | PASS, 27 of 27 items |
| 08:00 to 08:41 | `stage4`: K2, K3, K4 (none S1-NO) and M2, M3 at full scale, seeds 0 to 4: train (511, 499, 515 s), infer (15, 14, 15 s), matched arm (M2 32 s, M3 28 s); scorer (433 s), check-verdict, V2 (376 s) | V2 agrees; K2 S2-GO d +0.0150 [+0.0096, +0.0163]; K3 S2-IN d +0.0011 [−0.0043, +0.0025]; K4 S2-GO d +0.0154 [+0.0100, +0.0169]; the bar is B1 own (0.0874) for all three |
| 08:41 to 08:51 | `stage5`: ArtELingo-28, K1 to K4 at full scale (every stage 4 gave an outcome); K1, B1, B6 inference (6 to 7 s each), scorer (539 s), check, V2 (51 s) | V2 agrees; K1 A-QUAL d +0.0067 [−0.0200, +0.0116]; K2 A-QUAL +0.0244 [−0.0044, +0.0271]; K3 A-FAILS −0.0079 [−0.0332, +0.0001]; K4 A-QUAL +0.0167 [−0.0103, +0.0212] |
| 08:51 | chain done; GPU free (no process, no lock) | done |
| 08:52 to 09:52 | `rb_queue_cpu.sh 3` (stage 3 report queue), alone on the CPU; load 5.2, 56 GB free; gate v2_3 (3 s), report (3,613 s) | done, no step failed; `part2/reported_stage3.{json,md}` (decides nothing) |
| 09:52 | GPU check: driver, kernel module and library all 595.99.02; no GPU process, no lock of this chat; 188 GB free on `/` | pass |
| 09:55 | Cut to "r10 run 15" (context meter) | handoff `2026-10-25-r10-run-15-handoff.md` |
| 10:04 to 10:17 | (r10 run 15) report 12-24 drafted by Codex, one revision round | stage 4 readings left as a marked gap |
| 12:06 to 13:07 | `rb_queue_cpu.sh 4` (stage 4 report queue), alone on the CPU; load 0.2, 55 GB available; started at noon on the user's word (Sunday, machine idle), not at 19:00; gate v2_4 (3 s), report (3,617 s) | done, no step failed; `part2/reported_stage4.{json,md}` (decides nothing) |

## The split (spec D32)

The estimate at 06:59 assumed K4 and K5 and the measured part 1 time of 38 to 58 min; part 1 took 22 min. After the
pick (K5 not triggered), the remaining steps from 07:25, by the step table of `rb_run.py budget` (the Bash permission
check gave no verdict at 07:23, so this sum was made by hand from the 06:59 output; `budget` at 07:25 then gave the
same chain, 106 to 183 min, and an end with the reports of 11:01 to 12:27):

| Part | Minutes (low to high) | End |
|---|---|---|
| chain with K2, K3, K4: regressions and checks, students and matched arms at both scales, verdicts, V2, stage 5 | 106 to 183 | 09:11 to 10:28 |
| report queue stage 3 (3 candidates, 30 to 40 min each), after stage 3's V2 | 90 to 120 | about 10:30 |
| report queue stage 4, after stage 4's V2 and the stage 3 queue | 90 to 120 | about 12:00 to 12:30 |

Choice (agent, by D32): keep all three candidates in the morning chain; run both report queues (`rb_queue_cpu.sh 3`,
then `4`) in the evening window, from 19:00. Why: the chain alone ends by 10:28 at the high end with all three, so no
candidate needs deferral; the report queues decide nothing (spec §3.6) and alone push the end past 11:00. This keeps
every outcome in one pass (no deferral rerun, stage 5 computed once) and leaves only descriptive readings for the
evening. Reports stay due 2026-10-26 10:00.

At 07:58 the stage 3 queue had ended at 07:55, ahead of the high estimate (08:27). Running `rb_queue_cpu.sh 3` beside
the chain was weighed and not done: `rb_part2_report.py` writes every candidate's block only at its end (one step for
all three), its 30 to 40 min per candidate is an estimate not yet measured, and a run that crosses 11:00 could not be
stopped without losing it. The loop's main chat ("r10 main 3") showed a shell, not a running chat, at 07:58; no
message was sent to it.

At 08:51 the chain was done, 97 min before its high estimate. Change (agent): the stage 3 report queue starts now,
alone on the CPU (3 candidates, 90 to 120 min estimated: end 10:22 to 10:52); only the stage 4 report queue waits
for the evening window (19:00). Why: with the chain finished, the stage 3 queue fits before 11:00 at its high
estimate and halves the evening's work. If it runs past 11:00 (its estimate is not measured), it is a CPU-only job
at 8 of 32 cores and is let finish; the log says so.

At 09:52 the stage 3 report queue was done in 60 min for three candidates (estimate 90 to 120). The stage 4 report
queue (the same three candidates) stays in the evening window (19:00). Why (agent): started now, it would end about
10:53 at the measured time, with no margin before 11:00; the evening run delays only the descriptive readings, and
the reports stay due 2026-10-26 10:00.
