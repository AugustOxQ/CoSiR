# Run log: the recipe sweep round (r10), stages 0 to 4 and the ablation

## The chats

- Run 3, from 2026-10-27 17:05: stages 0 to 4, G's and c*'s analyses, then the detached ablation chain. Handoff: `docs/superpowers/handoffs/2026-10-27-r10-run-3-handoff.md` into the run and `2026-10-27-r10-run-4-handoff.md` out of it.
- Run 4, from 2026-10-27 21:38: continues the running chain. This document records completed stages only; ablation results belong to the later update.
- Code merged at eb1d04b; node launches at a87cb1d. Outputs: `/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs` (`<O>` below). Binding rule: `DECISION_RULE.md` @ 506c40c, first commit 35fcf68. Dates and times below are Amsterdam local time.

## Before the start (2026-10-27)

- 14:43 N1 passes, 42.94 s. Node402: RTX A6000; 1,773.3 GB free against 60 required; 71 input files (4,316,659,141 bytes) equal by SHA-256. Versions and stores pass. 14:46 N2 passes, 109.21 s, train exit 0 and score exit 0; smoke AMI emotion 0.0949, style 0.3376, genre 0.4988.
- 15:31 user exception to rule §4.2: keep the node route, shift deadlines to the real start and eight hours later. Applied as start-by 2026-10-27 17:08 and end-by 2026-10-28 01:08.
- Build gates pass: V1a trainer/scorer, V3a, V4a trainer/scorer, sample graph rebuild and ablation graph reproduction. Sample graph rebuild equals the frozen graphs in all 18 arrays. V0a was rerun after the stage 0 fix, passing 16:20; V4a scorer rerun passes 16:32. Build verification recorded 1,049 tests passing and the separate held scorer self-check agreeing on 77 items.
- 16:34 GPU check passed: driver 595.99.02 from nvidia-smi, kernel module and container library. Day mode. Restart question 16:35 had no answer by 17:04, so the recorded default was no restart. Repeat GPU check 17:04: same driver. At the later cut, user 21:38 chose no restart.
- 17:06 node402 has three free 48 GB GPUs and 79 h 44 min of reservation left. 17:07 sync a87cb1d and input verification pass: all 13 input items equal, 27.4 GiB. The status field's old checked-out branch is not the launched worktree; each launch records a87cb1d.
- Initial estimates: node stages about 3.5 h on three GPUs; local about 2.5 GPU h and 8 CPU h. Queue estimates: stage 0 about 35 min, stage 3 about 35, stage 4 about 65; development ablation about 40 and held about 110 min. Heavy local work stays outside weekday 10:00 to 19:00 and finishes before 11:00.

## Stage 0: G's emotion claim and analysis (2026-10-27)

- START 2026-10-27 17:08 smoke=0.
- gate V1a-train holds.
- gate V1a-scorer holds.
- gate V0a holds.
- `train_seed4`: 63 s, exit 0.
- `held_sample`: 16 s, exit 0.
- `v0c`: 10 s, exit 0.
- `v0d`: 5 s, exit 0.
- `verdict`: 257 s, exit 0.
- `v0b`: 227 s, exit 0.
- `analysis_g`: 120 s, exit 0.
- DONE failed=0 2026-10-27 17:20.
- sample: G AMI 0.1043; strongest PercepT entry B1-own 0.0855; margin +0.0188 [+0.0137, +0.0200], G-ES-GO.
- full: G AMI 0.1039; strongest PercepT entry B1-own 0.0874; margin +0.0164 [+0.0107, +0.0182], G-EF-GO.
- E0 true: the licensed emotion sentence holds. V0c and V0d pass; seed 4 trained once, no replacement. G full-scale held exports were reused after regression/hash checks. Sample held inference is new. Failures and reruns: none. Held ledger: H20.
- Re-derivation `stage0/rederive_stage0.json`: agree, 0 differences; 53 items, largest absolute difference 0, 0 failures.

## Stage 1: single-knob sweep on node402 (2026-10-27)

- Start 17:08, launches to 17:10; shards finish about 17:55, plan ends 18:29. Three shards contain 14, 14 and 13 runs including G and G-repeat, all launched at a87cb1d. Status, sync, launches, watches, pulls and fetches succeed. Node plan status lines do not record per-step elapsed seconds or shell exit codes; structured results say `ok: true`, watches `succeeded`. Exact missing durations are not inferred from file mtimes.
- N3 passes: G and G-repeat exports and best epochs equal bit for bit; repeat AMI differences 0.0000 for all aspects. Trainer parity, node minus frozen local seed 0: emotion +0.0003, style +0.0020, genre −0.0077 (about 0.6 genre seed ranges). Scorer parity: 0.0000 for every aspect. These are diagnostics only; no decision compares a node number with a local number.
- Separate partition check on the node passes for 41 runs; local sweep and decision re-derivation pass. GRID-4: graph neighbours, teacher row temperature, style relational weight and trunk width. The grid has 24 points, 18 new. All 39 candidate runs pass export validity; no training failure and no rerun. Guard failures are exclusions, not failed runs.
- Full node route, tags and transfers: [rs_node_log.md](rs_node_log.md). The curves below come directly from stage1.json.

| ID / knob | G | Value: mean gain (seed ranges) | Guard failures |
|---|---|---|---|
| k1 / emotion relational weight | 0 | 2: -0.559; 8: -0.700; 16: -1.197 | 2, 8, 16 |
| k2 / style relational weight | 0 | 2: +0.069; 8: +0.288; 16: -0.119 | 16 |
| k3 / genre relational weight | 8 | 4: +0.067; 12: -0.208; 16: -0.368 | 12, 16 |
| k4 / teacher row temperature | 0.02 | 0.015: -0.121; 0.025: -0.303; 0.035: +0.371 | 0.015, 0.025 |
| k5 / direct teacher weight | 0.25 | 0.0625: +0.158; 0.125: -0.225; 0.5: -0.454 | 0.125, 0.5 |
| k6 / learning rate | 0.001 | 0.0003: -0.391; 0.0005: -0.101; 0.002: -0.339 | 0.0003 |
| k7 / weight decay | 0.05 | 0: -0.239; 0.01: -0.299; 0.2: -0.146 | 0, 0.01, 0.2 |
| k8 / batch pairs | 512 | 256: -0.169; 1024: -0.146; 2048: +0.133 | none |
| k9 / trunk width | 1024 | 512: +0.164; 1536: -0.156; 2048: -0.032 | 1536 |
| k10 / dropout | 0.1 | 0: -0.414; 0.2: -0.136; 0.3: -0.854 | 0, 0.3 |
| k11 / graph neighbours | 20 | 10: -0.191; 30: -0.064; 40: +0.924 | 10 |
| k12 / buddy temperature | 0.05 | 0.03: +0.060; 0.07: -0.403; 0.1: -0.766 | 0.07, 0.1 |
| k13 / patience | 15 | 5: +0.001; 10: +0.000; 30: +0.000 | none |

## Stage 2: joint grid on node402 (2026-10-27)

- Start 18:30; three shards of six new configurations finish about 18:55. All 18 runs succeed and pass export validity. Pulls and scores fetch succeed.
- 19:27: plan stops on the verification watch's default 30-minute stall limit, with the CPU verification job still running and its log silent for 31 min 58 s. The watch reports `ok: false`, state `stalled`; this is not a trainer failure. The run chat watches the same job with `--stall-minutes 120`, then follows the plan's remaining order by hand (`stage2_tail.sh` in the run chat scratchpad).
- Tail: watch succeeds, pull succeeds, stage 1 home partition check is moved to `rederive_partitions.stage1.json`, fetch succeeds, node partition check passes all 59 runs, deadline still holds, local stage 2 scoring and decision re-derivation pass. Done 19:39. No fallback, job relaunch or training rerun; no result changed. The status files omit elapsed seconds and numeric shell exit codes for these node steps; structured results confirm success except the first watch.
- Pool contains 21 guarded positive candidates; TOP gives the following three. Details: [rs_node_log.md](rs_node_log.md).

| Rank / run | Emotion AMI | Style AMI | Genre AMI | Mean gain |
|---|---|---|---|---|
| 1 / k11=40 | 0.1107 | 0.4090 | 0.5060 | +0.924 |
| 2 / k2=2+k9=512+k11=40 | 0.1103 | 0.4031 | 0.5144 | +0.874 |
| 3 / k4=0.035+k11=40 | 0.1111 | 0.4018 | 0.5093 | +0.751 |

- Re-derivation `sweep/node402/rederive_stage2.json`: agree, 0 differences; 307 items, largest absolute difference 0, 0 failures.
- Node partitions `scores/node402/rederive_partitions.json`: agree, 0 differences; 413 items, largest absolute difference 0.

## Stage 3: confirmation and pick on seeds used in earlier rounds (2026-10-27)

- START 2026-10-27 19:39 smoke=0.
- gate V1a-train holds.
- gate V1a-scorer holds.
- gate V3a holds.
- gate V1c-a holds.
- gate V1c-b holds.
- `train_top`: 932 s, exit 0.
- `verdict`: 569 s, exit 0.
- `v3c`: 572 s, exit 0.
- DONE failed=0 2026-10-27 20:13.
- Nine new runs on local seeds 1 to 3; comparator G uses the frozen local runs. PICK: c* = k11=40 (graph neighbours 40, everything else G). Two configurations qualify; the third fails only the mean-gain lower-bound condition. No failed run or rerun.

| Run | Emotion mean / gain [95%] | Style mean / gain [95%] | Genre mean / gain [95%] | Mean gain [95%] | Qualifies |
|---|---|---|---|---|---|
| k11=40 | 0.1089 / +0.0026 [+0.0011, +0.0045] | 0.4072 / +0.0141 [+0.0121, +0.0160] | 0.5019 / -0.0031 [-0.0090, +0.0044] | +0.683 [+0.452, +0.959] | yes |
| k2=2+k9=512+k11=40 | 0.1085 / +0.0023 [+0.0009, +0.0036] | 0.4020 / +0.0089 [+0.0035, +0.0135] | 0.4990 / -0.0060 [-0.0098, -0.0012] | +0.369 [+0.001, +0.724] | yes |
| k4=0.035+k11=40 | 0.1082 / +0.0020 [+0.0002, +0.0040] | 0.3991 / +0.0060 [-0.0006, +0.0111] | 0.5006 / -0.0045 [-0.0096, +0.0003] | +0.272 [-0.187, +0.714] | no: mean lower <0 |

- c* genre is not distinguishable from G (interval crosses zero).
- Lead 0.314 against interval half-width 0.253 is descriptive; the mean-gain intervals overlap. The rule only reports this; the pick is the run with the largest mean gain. Pick verdict SHA-256: `aa708fd296714867d889e74640570ae08a9cfdd8275caab9e9f57eec314b643c`. Written before c*'s first full-scale run.
- Re-derivation `stage3/rederive_stage3.json`: agree, 0 differences; 213 items, largest absolute difference 0, 0 failures.

## Stage 4: held test and analysis (2026-10-27)

- START 2026-10-27 20:14 smoke=0.
- gate V1a-train holds.
- gate V1a-scorer holds.
- gate V4a-train holds.
- gate V4a-scorer holds.
- gate V3c holds.
- `train_c`: 1526 s, exit 0.
- `held_c`: 22 s, exit 0.
- `verdict`: 266 s, exit 0.
- `v4b`: 241 s, exit 0.
- `report`: 1019 s, exit 0.
- `analysis_c`: 219 s, exit 0.
- DONE failed=0 2026-10-27 21:09.
- Five full-scale seeds, held inference, verdict, independent check, reported margins and c* analysis finish in order. No failed run, non-finite-loss failure, held validity rerun or candidate replacement recorded. Held ledger: H21.

| Aspect | c* / T AMI | Difference [95%] | d/r | Rule outcome | Leiden AMI |
|---|---|---|---|---|---|
| emotion | 0.1066 / 0.0921 | +0.0145 [+0.0113, +0.0177] | 3.155 | T-WIN | not computed |
| style | 0.4066 / 0.4026 | +0.0041 [+0.0004, +0.0081] | 0.497 | T-WIN | not computed |
| genre | 0.5001 / 0.5062 | -0.0060 [-0.0103, -0.0015] | -0.472 | T-LEVEL | 0.5082 |

| Aspect | G AMI | c* minus G [95%] | Best PercepT entry / AMI | c* minus PercepT [95%] |
|---|---|---|---|---|
| emotion | 0.1039 | +0.0027 [+0.0001, +0.0054] | B1-own / 0.0874 | +0.0192 [+0.0137, +0.0207] |
| style | 0.4057 | +0.0009 [-0.0044, +0.0065] | B6-own / 0.2297 | +0.1769 [+0.1663, +0.1809] |
| genre | 0.5001 | 0.0000 [-0.0028, +0.0030] | B5-r20 / 0.4269 | +0.0732 [+0.0622, +0.0796] |

- Emotion over PercepT: +0.0192 [+0.0137, +0.0207], PE-GO. Genre is below the instructed image by less than one seed range (0.0128); its whole interval is below zero. Both round to genre AMI 0.5001; their held outputs differ in every head. Run 3 checked different held outputs in every head at 21:13.
- Re-derivation `stage4/rederive_stage4.json`: agree, 0 differences; 66 items, largest absolute difference 0, 0 failures.

## Best epochs and saved run records

Epochs are the selected checkpoints, read from the saved run/score JSON, in emotion/style/genre order. Seconds below are trainer-recorded durations, distinct from the queue step's wall time. No node step duration or exit code is inferred from them.

| Local run | Seed | Seconds | Status | Best epochs |
|---|---|---|---|---|
| k11=40 | 1 | 80.04 | ok | 2/3/3 |
| k11=40 | 2 | 77.00 | ok | 1/2/3 |
| k11=40 | 3 | 78.29 | ok | 2/2/3 |
| k2=2+k9=512+k11=40 | 1 | 71.06 | ok | 2/3/3 |
| k2=2+k9=512+k11=40 | 2 | 71.84 | ok | 3/3/3 |
| k2=2+k9=512+k11=40 | 3 | 73.55 | ok | 2/4/3 |
| k4=0.035+k11=40 | 1 | 78.78 | ok | 2/1/3 |
| k4=0.035+k11=40 | 2 | 76.24 | ok | 1/1/2 |
| k4=0.035+k11=40 | 3 | 79.22 | ok | 2/1/3 |
| full k11=40 | 0 | 316.97 | ok | 4/1/18 |
| full k11=40 | 1 | 243.47 | ok | 4/3/6 |
| full k11=40 | 2 | 212.37 | ok | 3/3/2 |
| full k11=40 | 3 | 296.09 | ok | 3/2/14 |
| full k11=40 | 4 | 241.31 | ok | 6/1/5 |

| Node seed-0 run | Status | Best epochs |
|---|---|---|
| G-repeat | ok | 3/2/4 |
| G | ok | 3/2/4 |
| k10=0.0 | ok | 3/2/4 |
| k10=0.2 | ok | 3/3/8 |
| k10=0.3 | ok | 3/3/8 |
| k11=10 | ok | 6/6/7 |
| k11=30 | ok | 3/2/5 |
| k11=40 | ok | 2/2/2 |
| k12=0.03 | ok | 3/2/4 |
| k12=0.07 | ok | 3/3/8 |
| k12=0.1 | ok | 5/3/20 |
| k13=10 | ok | 3/2/8 |
| k13=30 | ok | 3/2/4 |
| k13=5 | ok | 3/2/8 |
| k1=16 | ok | 15/2/4 |
| k1=2 | ok | 6/2/4 |
| k1=8 | ok | 7/2/4 |
| k2=16 | ok | 3/6/8 |
| k2=2+k11=40 | ok | 2/2/2 |
| k2=2+k4=0.035+k11=40 | ok | 2/2/2 |
| k2=2+k4=0.035+k9=512+k11=40 | ok | 2/2/2 |
| k2=2+k4=0.035+k9=512 | ok | 3/4/3 |
| k2=2+k4=0.035 | ok | 3/4/4 |
| k2=2+k9=512+k11=40 | ok | 2/2/4 |
| k2=2+k9=512 | ok | 3/6/5 |
| k2=2 | ok | 3/4/4 |
| k2=8+k11=40 | ok | 2/4/2 |
| k2=8+k4=0.035+k11=40 | ok | 2/3/2 |
| k2=8+k4=0.035+k9=512+k11=40 | ok | 2/4/2 |
| k2=8+k4=0.035+k9=512 | ok | 3/8/5 |
| k2=8+k4=0.035 | ok | 3/5/4 |
| k2=8+k9=512+k11=40 | ok | 2/4/5 |
| k2=8+k9=512 | ok | 4/8/8 |
| k2=8 | ok | 3/6/4 |
| k3=12 | ok | 3/2/8 |
| k3=16 | ok | 3/2/8 |
| k3=4 | ok | 3/2/3 |
| k4=0.015 | ok | 3/2/9 |
| k4=0.025 | ok | 3/2/4 |
| k4=0.035+k11=40 | ok | 2/2/2 |
| k4=0.035+k9=512+k11=40 | ok | 2/2/2 |
| k4=0.035+k9=512 | ok | 3/3/3 |
| k4=0.035 | ok | 3/2/4 |
| k5=0.0625 | ok | 3/2/4 |
| k5=0.125 | ok | 3/2/4 |
| k5=0.5 | ok | 3/4/4 |
| k6=0.0003 | ok | 3/3/9 |
| k6=0.0005 | ok | 3/3/3 |
| k6=0.002 | ok | 6/3/4 |
| k7=0.01 | ok | 3/2/4 |
| k7=0.0 | ok | 3/2/4 |
| k7=0.2 | ok | 3/2/4 |
| k8=1024 | ok | 3/4/9 |
| k8=2048 | ok | 6/8/6 |
| k8=256 | ok | 3/2/6 |
| k9=1536 | ok | 3/3/4 |
| k9=2048 | ok | 3/2/4 |
| k9=512+k11=40 | ok | 2/2/4 |
| k9=512 | ok | 3/3/5 |

## Diagnostic detail (completed after the relevant verdicts)

Key: T is the instructed Qwen image of the aspect; r is its seed range; d/r is the AMI difference in seed ranges. PercepT core is the project baseline; B1-own and B6-own are its P3 and P2 entries with their own clustering. B5-r20 is the paper-settings P4 entry reclustered at 20.

The analysis is descriptive: held full scale, five seeds; cluster galleries and low-overlap galleries use seed 0. G's analysis is within stage 0's 17:08 to 17:20 queue; c*'s is within stage 4's 20:14 to 21:09 queue.

| Head | Emotion labels, G → c* | Style labels, G → c* | Genre labels, G → c* |
|---|---|---|---|
| emotion | 0.1039 → 0.1066 | 0.1647 → 0.1641 | 0.4048 → 0.3889 |
| style | 0.0622 → 0.0610 | 0.4057 → 0.4066 | 0.2983 → 0.2903 |
| genre | 0.0695 → 0.0701 | 0.2668 → 0.2720 | 0.5001 → 0.5001 |

| Head pair | G AMI | c* AMI | Teacher AMI |
|---|---|---|---|
| emotion-style | 0.2401 | 0.2326 | 0.2853 |
| emotion-genre | 0.4234 | 0.4066 | 0.4263 |
| style-genre | 0.4033 | 0.4008 | 0.4640 |

| Head | Effective rank, G → c* | Hubness, G → c* | Alignment, G → c* | Uniformity, G → c* | Teacher overlap, G → c* |
|---|---|---|---|---|---|
| emotion | 216.645 → 225.750 | 1.043 → 1.087 | 1.252 → 1.308 | -2.253 → -2.155 | 0.464 → 0.462 |
| style | 159.794 → 161.736 | 1.144 → 1.139 | 1.442 → 1.415 | -2.844 → -2.887 | 0.444 → 0.453 |
| genre | 164.367 → 169.171 | 1.192 → 1.189 | 0.781 → 0.800 | -2.439 → -2.450 | 0.625 → 0.636 |

| Aspect / change | Class | G F1 | c* F1 | c* minus T F1 |
|---|---|---|---|---|
| emotion / largest rise | excitement | 0.1472 | 0.1551 | +0.0114 |
| emotion / largest fall | contentment | 0.2615 | 0.2464 | -0.0535 |
| style / largest rise | Early_Renaissance | 0.4354 | 0.4908 | +0.0747 |
| style / largest fall | Minimalism | 0.4592 | 0.3932 | -0.0071 |
| genre / largest rise | sketch_and_study | 0.3880 | 0.4189 | +0.0050 |
| genre / largest fall | cityscape | 0.6517 | 0.6090 | -0.0517 |

### G: all per-class F1 readings

emotion:

| Class | Count | Model F1 | T F1 | Difference |
|---|---|---|---|---|
| amusement | 1818 | 0.2076 | 0.1564 | +0.0512 |
| anger | 153 | 0.0622 | 0.0743 | -0.0121 |
| awe | 2448 | 0.2040 | 0.1846 | +0.0195 |
| contentment | 4244 | 0.2615 | 0.2999 | -0.0385 |
| disgust | 356 | 0.1308 | 0.1072 | +0.0237 |
| excitement | 543 | 0.1472 | 0.1437 | +0.0035 |
| fear | 967 | 0.2914 | 0.2650 | +0.0263 |
| sadness | 1021 | 0.2472 | 0.2341 | +0.0131 |
| something else | 731 | 0.2640 | 0.2651 | -0.0011 |

style:

| Class | Count | Model F1 | T F1 | Difference |
|---|---|---|---|---|
| Abstract_Expressionism | 392 | 0.5753 | 0.5172 | +0.0581 |
| Action_painting | 17 | 0.0558 | 0.0424 | +0.0135 |
| Analytical_Cubism | 15 | 0.0581 | 0.0561 | +0.0020 |
| Art_Nouveau_Modern | 623 | 0.4949 | 0.5291 | -0.0342 |
| Baroque | 647 | 0.4553 | 0.4251 | +0.0302 |
| Color_Field_Painting | 245 | 0.3854 | 0.3670 | +0.0184 |
| Contemporary_Realism | 71 | 0.0554 | 0.0543 | +0.0011 |
| Cubism | 331 | 0.5610 | 0.5751 | -0.0140 |
| Early_Renaissance | 221 | 0.4354 | 0.4161 | +0.0193 |
| Expressionism | 1003 | 0.3065 | 0.3148 | -0.0083 |
| Fauvism | 114 | 0.1131 | 0.1447 | -0.0316 |
| High_Renaissance | 218 | 0.2827 | 0.2636 | +0.0192 |
| Impressionism | 1949 | 0.5113 | 0.4548 | +0.0564 |
| Mannerism_Late_Renaissance | 202 | 0.2636 | 0.2287 | +0.0349 |
| Minimalism | 188 | 0.4592 | 0.4003 | +0.0590 |
| Naive_Art_Primitivism | 397 | 0.2600 | 0.2363 | +0.0237 |
| New_Realism | 44 | 0.0309 | 0.0324 | -0.0015 |
| Northern_Renaissance | 394 | 0.3995 | 0.3361 | +0.0634 |
| Pointillism | 112 | 0.1610 | 0.1629 | -0.0019 |
| Pop_Art | 239 | 0.4303 | 0.3273 | +0.1030 |
| Post_Impressionism | 1012 | 0.3233 | 0.2613 | +0.0621 |
| Realism | 1664 | 0.3124 | 0.3205 | -0.0080 |
| Rococo | 313 | 0.4674 | 0.4439 | +0.0235 |
| Romanticism | 1074 | 0.2833 | 0.2790 | +0.0043 |
| Symbolism | 599 | 0.2428 | 0.2067 | +0.0362 |
| Synthetic_Cubism | 22 | 0.0791 | 0.0802 | -0.0011 |
| Ukiyo_e | 175 | 0.9223 | 0.9663 | -0.0440 |

genre:

| Class | Count | Model F1 | T F1 | Difference |
|---|---|---|---|---|
| abstract_painting | 738 | 0.8782 | 0.8789 | -0.0008 |
| cityscape | 710 | 0.6517 | 0.6608 | -0.0091 |
| genre_painting | 1666 | 0.3872 | 0.3874 | -0.0002 |
| illustration | 296 | 0.4357 | 0.5068 | -0.0711 |
| landscape | 2007 | 0.6074 | 0.7175 | -0.1101 |
| nude_painting | 303 | 0.5115 | 0.5083 | +0.0032 |
| portrait | 2230 | 0.5473 | 0.4855 | +0.0618 |
| religious_painting | 985 | 0.7606 | 0.7131 | +0.0475 |
| sketch_and_study | 613 | 0.3880 | 0.4139 | -0.0259 |
| still_life | 401 | 0.7564 | 0.7686 | -0.0122 |

### k11=40: all per-class F1 readings

emotion:

| Class | Count | Model F1 | T F1 | Difference |
|---|---|---|---|---|
| amusement | 1818 | 0.2010 | 0.1564 | +0.0446 |
| anger | 153 | 0.0643 | 0.0743 | -0.0100 |
| awe | 2448 | 0.2019 | 0.1846 | +0.0173 |
| contentment | 4244 | 0.2464 | 0.2999 | -0.0535 |
| disgust | 356 | 0.1355 | 0.1072 | +0.0283 |
| excitement | 543 | 0.1551 | 0.1437 | +0.0114 |
| fear | 967 | 0.2788 | 0.2650 | +0.0138 |
| sadness | 1021 | 0.2512 | 0.2341 | +0.0171 |
| something else | 731 | 0.2714 | 0.2651 | +0.0063 |

style:

| Class | Count | Model F1 | T F1 | Difference |
|---|---|---|---|---|
| Abstract_Expressionism | 392 | 0.5379 | 0.5172 | +0.0208 |
| Action_painting | 17 | 0.0496 | 0.0424 | +0.0072 |
| Analytical_Cubism | 15 | 0.0531 | 0.0561 | -0.0030 |
| Art_Nouveau_Modern | 623 | 0.4574 | 0.5291 | -0.0716 |
| Baroque | 647 | 0.4902 | 0.4251 | +0.0651 |
| Color_Field_Painting | 245 | 0.3892 | 0.3670 | +0.0222 |
| Contemporary_Realism | 71 | 0.0600 | 0.0543 | +0.0057 |
| Cubism | 331 | 0.5387 | 0.5751 | -0.0364 |
| Early_Renaissance | 221 | 0.4908 | 0.4161 | +0.0747 |
| Expressionism | 1003 | 0.3132 | 0.3148 | -0.0016 |
| Fauvism | 114 | 0.1093 | 0.1447 | -0.0354 |
| High_Renaissance | 218 | 0.2693 | 0.2636 | +0.0058 |
| Impressionism | 1949 | 0.5278 | 0.4548 | +0.0730 |
| Mannerism_Late_Renaissance | 202 | 0.2983 | 0.2287 | +0.0696 |
| Minimalism | 188 | 0.3932 | 0.4003 | -0.0071 |
| Naive_Art_Primitivism | 397 | 0.2588 | 0.2363 | +0.0225 |
| New_Realism | 44 | 0.0310 | 0.0324 | -0.0015 |
| Northern_Renaissance | 394 | 0.4393 | 0.3361 | +0.1031 |
| Pointillism | 112 | 0.1318 | 0.1629 | -0.0311 |
| Pop_Art | 239 | 0.3751 | 0.3273 | +0.0478 |
| Post_Impressionism | 1012 | 0.3475 | 0.2613 | +0.0862 |
| Realism | 1664 | 0.3085 | 0.3205 | -0.0120 |
| Rococo | 313 | 0.4900 | 0.4439 | +0.0460 |
| Romanticism | 1074 | 0.2802 | 0.2790 | +0.0012 |
| Symbolism | 599 | 0.2515 | 0.2067 | +0.0448 |
| Synthetic_Cubism | 22 | 0.0744 | 0.0802 | -0.0057 |
| Ukiyo_e | 175 | 0.8931 | 0.9663 | -0.0732 |

genre:

| Class | Count | Model F1 | T F1 | Difference |
|---|---|---|---|---|
| abstract_painting | 738 | 0.8871 | 0.8789 | +0.0082 |
| cityscape | 710 | 0.6090 | 0.6608 | -0.0517 |
| genre_painting | 1666 | 0.3711 | 0.3874 | -0.0163 |
| illustration | 296 | 0.4567 | 0.5068 | -0.0500 |
| landscape | 2007 | 0.6121 | 0.7175 | -0.1055 |
| nude_painting | 303 | 0.4892 | 0.5083 | -0.0191 |
| portrait | 2230 | 0.5231 | 0.4855 | +0.0376 |
| religious_painting | 985 | 0.7219 | 0.7131 | +0.0088 |
| sketch_and_study | 613 | 0.4189 | 0.4139 | +0.0050 |
| still_life | 401 | 0.7748 | 0.7686 | +0.0062 |

The reported held style gain over G is +0.0009 [-0.0044, +0.0065], crossing zero.

Our reading: more graph neighbours can supply positives from more neighbours per painting. This may explain the development style gain, but that mechanism is not checked.  Visual interpretation of the saved galleries is not checked. Full class compositions, matrices against T and the default image, per-class neighbour overlaps, ranks, hub counts and seed values remain in `analysis/G/analysis.json` and `analysis/k11=40/analysis.json`.

## Provenance, files and retention

- Authoritative result files: `<O>/stage0/{verdict,rederive_stage0}.json`; `<O>/sweep/node402/{stage1,stage2,rederive_stage1,rederive_stage2}.json`; `<O>/scores/node402/`; `<O>/gates/`; `<O>/stage3/{verdict,rederive_stage3}.json`; `<O>/stage4/{verdict,rederive_stage4,reported}.json` and `leiden/`; both `analysis/` trees. Queue status files and step logs are in `<O>` and `<O>/logs/`.
- Run chat scratchpad, read only: `/tmp/claude-0/-project-CoSiR/197a25bd-5b5f-4a2e-be46-c0a7080cf8c8/scratchpad/`, containing `node_stage1.out`, `node_stage2.out`, `stage2_tail.sh`, `stage0.out`, `stage3.out`, `stage4.out` and deadline notes.
- G's node reference and TOP runs were copied home at 20:0x, four verified transfers of 13 files each: `<O>/node402/train/`, 1.9 GB. Keep these, G's sample seed 4 and all stage 3, stage 4 and ablation runs.
- Would remove after wrap-up review, never in this drafting job: `<O>/regression/v1a_s0`, `v1a_s1`, `v4a_run`; non-shortlisted node sweep weights and expendable node job scratch once sized and home copies verified. Prior pending rows also name build smoke/review scratch and merged fix worktrees; no new deletion or move is authorized here. The node's remaining output size still needs the wrap-up inventory.
Storage: decided at the wrap-up; see `.scratch/pending_deletions.md` and the disk ledger `/project/claude-config/storage/disk_ledger.md`.

## Ablation

F = k11=40 (c*). The removals and what each changes: spec §3.2, "Ablation". Rule: §2.5, §3.5, §4.5.

### Development (done 2026-10-27 22:50)

- Queue `rs_queue_ablation_dev.sh`: started 21:11, `dev_exit=0` at 22:50, `failed=0`. Gates V1a (train, scorer), V3a and V3c held.
- Training: 24 runs (8 removals × seeds 1 to 3), 3,763 s in total, no failure, no rerun. Scoring (`score_dev`): 1,528 s. Re-derivation (`v_ablation`, `rs_vp ablation`): 647 s.
- `ablation/dev.json` SHA-256 `c8218ceb3b85fd2340d44474f648fa3e3afe2cd674f2124bc988230054526be6` (written 22:39, code `da6fa381…`, commit a808d0e, clean).
- `ablation/rederive_dev.json` SHA-256 `01ba39ef65de77fde4e492a0010448143542adf9ffeaaaa59ca5bf1872db7077`: `pass: true`, 442 of 442 items pass, largest numeric difference 0.0.
- `inputs.ablation_graphs_sha256` = `698100b2fa4dea71f752ed03aea9c8bfd2189a89a745d74e4bd39bae99fefc97`; both graph checks passed, so the content-correction and instructed-graph removals ran.
- Draws: per removal `default_rng(0)`, I_b over the rows of P, then S_b over seeds 1 to 3, shared by the arm, F and the aspects; 2,000 draws.
- c*'s means on P (stage 3 runs, seeds 1 to 3): emotion 0.1089, style 0.4072, genre 0.5019.

| Removal / arm | Seconds (3 seeds) | Emotion arm / c* / ē [95%] | Style arm / c* / ē [95%] | Genre arm / c* / ē [95%] | Mean gain [95%] (seed ranges) |
|---|---|---|---|---|---|
| relational term / `k11=40-no_relational` | 297 | 0.1080 / 0.1089 / -0.0009 [-0.0035, +0.0013] | 0.4064 / 0.4072 / -0.0008 [-0.0050, +0.0052] | 0.4722 / 0.5019 / -0.0297 [-0.0339, -0.0258] | -0.869 [-1.150, -0.574] |
| sharper teacher row / `k11=40-no_sharp_row` | 329 | 0.1091 / 0.1089 / +0.0002 [-0.0015, +0.0018] | 0.4009 / 0.4072 / -0.0063 [-0.0134, -0.0006] | 0.5022 / 0.5019 / +0.0003 [-0.0082, +0.0073] | -0.233 [-0.355, -0.111] |
| direct teacher / `k11=40-no_direct` | 304 | 0.1111 / 0.1089 / +0.0022 [+0.0005, +0.0037] | 0.4065 / 0.4072 / -0.0007 [-0.0026, +0.0014] | 0.5030 / 0.5019 / +0.0011 [-0.0051, +0.0057] | +0.165 [-0.002, +0.327] |
| buddy terms / `k11=40-no_buddy` | 1374 | 0.0796 / 0.1089 / -0.0292 [-0.0319, -0.0264] | 0.2517 / 0.4072 / -0.1555 [-0.1591, -0.1506] | 0.5018 / 0.5019 / -0.0001 [-0.0020, +0.0018] | -8.441 [-8.733, -8.067] |
| caption endpoints / `k11=40-no_caption_endpoints` | 271 | 0.1089 / 0.1089 / 0.0000 [-0.0016, +0.0017] | 0.4114 / 0.4072 / +0.0042 [+0.0013, +0.0072] | 0.5075 / 0.5019 / +0.0056 [-0.0018, +0.0155] | +0.315 [+0.121, +0.534] |
| content correction / `k11=40-no_content_correction` | 422 | 0.0985 / 0.1089 / -0.0103 [-0.0131, -0.0074] | 0.3394 / 0.4072 / -0.0678 [-0.0737, -0.0599] | 0.5025 / 0.5019 / +0.0006 [-0.0039, +0.0067] | -3.491 [-3.896, -3.097] |
| instructed graphs / `k11=40-no_instructed_graphs` | 425 | 0.0743 / 0.1089 / -0.0346 [-0.0381, -0.0315] | 0.2078 / 0.4072 / -0.1994 [-0.2029, -0.1948] | 0.5033 / 0.5019 / +0.0015 [-0.0002, +0.0032] | -10.573 [-10.853, -10.258] |
| shared trunk / `k11=40-no_shared_trunk` | 341 | 0.1089 / 0.1089 / 0.0000 [-0.0025, +0.0023] | 0.4057 / 0.4072 / -0.0015 [-0.0058, +0.0019] | 0.5066 / 0.5019 / +0.0048 [+0.0029, +0.0067] | +0.062 [-0.143, +0.276] |

Key: arm / c* are mean AMIs over seeds 1 to 3 on P; ē = arm minus c*; mean gain = ē divided by the seed range (emotion 0.0046, style 0.0082, genre 0.0128), averaged over the three aspects.

### Held (running)

Queue `rs_queue_ablation_held.sh` started 22:50 (gates V3a, V3c, V4b hold; F = k11=40). Estimated end 2026-10-28 05:00 to 06:00. Results added when it ends.

## Deviations

- **Deadline exception:** user, 2026-10-27 15:31, rule §4.2: keep the node route and shift deadlines. Start 17:08; end-by 2026-10-28 01:08. Stage 2 ended 19:39.
- **Stage 2 watch:** at 19:27 the plan stopped at the default 30-minute silence limit while verification still ran. The run chat resumed the watch with `--stall-minutes 120`, then finished the same ordered tail by hand. No job reran and no result changed.
**Ablation timing correction.** Each development step trains one removal at seeds 1, 2 and 3, not one run.

| Development removal | Seconds, three seeds |
|---|---|
| relational | 297 |
| sharp row | 329 |
| direct | 304 |
| buddy | 1374 |
| caption endpoints | 271 |
| content correction | 422 |
| instructed graphs | 425 |
| shared trunk | 341 |

- Total: 3,763 s, about 63 GPU minutes for 24 runs, against about 17 estimated.
- Typical run: about 100 s; buddy removal: about 460 s per seed when its three-seed step is divided by three.
- Buddy metadata: emotion/style stop earlier in all seeds. Genre stops at epoch 165 in seed 1 and reaches the 200-epoch cap in seeds 2 and 3. Thus "never stops early" does not describe every head or seed.
- Held end **estimate**: 2026-10-28 05:00 to 06:00; the buddy removal alone is estimated at about 2 hours at full scale.
- The estimate is before 11:00 and outside weekday hours 10:00 to 19:00, but after the planned 23:00 end and the spec's 03:00 reporting time-box.

- **Stale scorer hash:** `gates/v1a_scorer.json` predates a metadata-only edit. Its hash starts 3f1803; the scorer hash starts 5f6fb8. No guard reads that hash; queues read its passing flag (queue note e).
- **Pending text:** the stage 4 verdict always writes `v4b: pending`; the held re-derivation file records the passing result. The queue's printed `None` is a return value, not a missing verdict. Stage 0 similarly retains pending text for its separate check.
- **Provenance:** JSON `code_commit` values vary as documentation commits advanced: stage 1 061e181, stages 2 and its sweep checks 15ce9c7, stage 3 7ec919c. The node launched a87cb1d; the code was merged at eb1d04b. These fields do not mean the trainer changed mid-run; code hashes and gates bind the code.
- **Stages 0, 1, 3 and 4:** no run failure, rerun or other execution deviation recorded. Stage 1 uses the planned exception: node trainer parity is a reading, not a gate. Leiden emotion/style and Leiden against the instructed image were not computed, as the binding rule (§3.4) narrows the spec's (§3.4) Leiden reading beside stage 4 to genre, as in 12-27; the rule governs.
