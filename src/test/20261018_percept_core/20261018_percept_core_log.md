# Round log: PercepT core (r10, from 2026-10-18)

Spec: `docs/superpowers/specs/2026-10-18-percept-core-design.md`. Tickets: `.scratch/r10-percept-core/issues/`
(outside git). Module prefix `pc_`. Outputs in `outputs/` (gitignored); `sept_stage0/` holds the first stage 0's
September re-fit (01:32 run, before A1; nothing writes it now).

## What the round does

1. Stage 0, the regression check (amended, A1): the September code at `1b70d7f` unchanged and copied text-identical,
   the saved snapshot re-scored, its k-means labels repeated on one thread, and our row 2 scoring.
2. Stage 1 of the spec, the baseline: PercepT core with the four fixes, arms P1 to P5, seeds 0 to 2 (GPU).
3. Stage 2 of the spec, the ablation: rows 1, 6, 10, 13 of the follow-up's table, after the head gate (CPU).

## Modules

| Module | Does |
|---|---|
| `pc_data.py` | frozen-sample readers, partitions fit / dev / eval (13,400 / 1,489 / 5,000 paintings) |
| `pc_roberta.py` | RoBERTa-GoEmotions per-caption store (`/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions/`) |
| `pc_fixes.py` | the four fixes (pair count, Eq. 2 fusion, the topic count, multi-hot targets): the top-40 rule `top_used` (A3) and Algorithm 1 `kept_centers` (reported only) |
| `pc_sept.py` | the September training functions, copied without change |
| `pc_stage1.py`, `pc_stage2.py` | PercepT core Stage 1 (topics) and Stage 2 (image mapper) |
| `pc_rows.py` | graphs and head rows 6, 10, 13; the head gate (`gate.json`) |
| `pc_score.py` | image-side scoring, row 1, row 2 check, the results sheet |
| `pc_stage0.py` | stage 0 (A1 checks (i) to (iv), row 2, the four-fix tests) |
| `pc_repeat.py` | the A2 repeat check: a second Stage 1 `clip_roberta` seed 0, compared bitwise (reported) |
| `pc_ablation.py` | the ablation table with rows 1 to 13 and the PercepT arm rows 4a to 5c |
| `pc_queue_gpu.sh`, `pc_queue_cpu.sh` | the two queues (how to call them: `pc_queue.md`) |

## Commands

From the repository root:

```bash
# tests (about 160 s; in the main checkout, ignore the gitignored sept_stage0/, whose September tests do not import)
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261018_percept_core --ignore=src/test/20261018_percept_core/sept_stage0
# the run: see pc_queue.md (GPU queue with --foreground in a background shell, CPU queue beside it)
```

## Build (2026-10-18 to 2026-10-19)

- Build 1 and 2: tickets 01 to 08 and 06b, merged on `r10-percept-core`; handoffs in `docs/superpowers/handoffs/`.
- Build 3: the stage 0 trace and the final whole-branch review (Opus `xhigh`: fix first, two queue fixes; all
  load-bearing numbers re-derived on P1 and P2 seed 0 to 1e-10). Fix wave: ticket 09.
- Build 4 (A1, A2; branch `r10-stage0-a1`): ticket 10, the stage 0 checks that repeat exactly and the one-thread
  k-means start (`pc_stage1.one_thread()`, config `kmeans_threads=1`; `pc_sept.py` byte-identical, blob `d7c064b`);
  ticket 11, the Stage 1 repeat step in the GPU queue (`pc_repeat.py`). Both by Codex, Opus ticket reviews, one fix
  round each.
- Build 5: GPU smoke (09:01 to 09:03, 125 s, exit 0; `outputs/queue_smoke_b5/` in the integration worktree): stage 0
  every check PASS in 53 s, 3 Stage 1 and 5 Stage 2 smoke runs, repeat `bitwise_equal` true. Ticket 12 (arm rows 4a
  to 5c in `pc_ablation.py`). Final whole-branch review (Opus `xhigh`): approve with fixes, no blocker; it re-derived
  checks (i) to (iv) with its own code (74 / 77 files, 0 differ; 24 copied parts identical; emotion
  0.12246902855745419, genre 0.22741920423971046 on 9,365 / 159 rows; 0 of 61,402 labels differ in 24.9 s) and the
  A2 hash (one thread `3b2b08fb7de82704` in both ways, 8 threads `b4eb501325c76f88`). Check (iv) also passes at 8
  threads (labels equal; only the centers' last bits differ), so the A2 hash test, not (iv), shows the limit works.
  Fix wave: ticket 13 (the repeat in the sheet, test gaps, the fix-test count); spec text for A1 and A2 (§3.2, §5,
  §7 C12 exception).
- Build 6 (A3, fixed K = 40; branch `r10-pc-a3`, 2026-10-19): ticket 14 by Codex: `pc_fixes.top_used` (the 40 of
  100 centers with the most fit pairs, ties to the lower index), `pc_stage1.count_fit_pairs` (argmax over all 100
  trained centers, fit rows), Stage 1 npz key `fit_counts`, sidecar `K` = 40, `K_gamma` (Algorithm 1, reported
  only), `kept_topics`; the K < 2 paths removed. Opus ticket review: merge. Final whole-branch review (Opus
  `xhigh`): merge after fixes; it re-derived the kept masks, `kept_topics` and `K_gamma` (3 smoke files, 9 run 2
  sidecars) with its own code. On run 2's three 99-topic runs the top 40 hold 57 to 63% of the fit pairs, with no tie
  at the cut. Fix wave: ticket 15 (count tests, `kept_topic_stats`, kept-mask asserts in the readers, results schema
  2 checked by `pc_ablation`, a fix-test collection check, sheet labels). GPU smoke twice (2,000 fit pairs): every
  Stage 1 keeps 40, all five arms train, stage 0 records 30 fix tests, repeat bitwise equal. Run 2's files are kept
  as `outputs/*_run2*`, `logs/run2/` and `pc_queue_gpu_run2.log` (copies checked equal).

## Stage 0 (2026-10-19)

FAIL as the spec defines it. The unchanged September re-fit gives held-out emotion AMI 0.1238 (recorded 0.1225) and
genre 0.2617 (recorded 0.2274); row 2 and the four-fix tests pass. Cause traced: the recorded values come from the
DAS6 node node404; this machine gives 0.1238 / 0.2617, as it did in September, with identical pretraining losses.
Trace: `src/test/20261019_stage0_trace/20261019_stage0_trace_log.md`. Per spec §3.5 the PercepT arms stop; the head
rows still run; the user decides.
The user amended stage 0 to the checks that repeat exactly (A1) and added the one-thread k-means start (A2), both
2026-10-19 08:20; builds 4 and 5 above.

## Topic count (spec §8 case)

Full-size Stage 1 timing runs, seed 0: clip_roberta K = 99 (the largest gap is at the top, 7.75 times the mean gap;
only the highest center is dropped), qwen_qwen K = 1 (the largest gap is at the bottom, 7.06 times the mean gap), the
same at every γ. Spec: report it, no rerun with another γ.

## Results

Run chat "r10 run", 2026-10-19, on `main` at `0f47271`. Reports:
[auto report](../../../docs/reports/auto/percept/2026-12-13_percept_core.md) and
[user-read report](../../../docs/user_read/2026-10-19_percept_core.md).

### The queues

| Queue | Start | End | Exit | Notes |
|---|---|---|---|---|
| GPU (`pc_queue_gpu.sh --foreground --lock-wait 14400`) | 01:32 | 01:44 | 1 (stage 0 FAIL) | stage 0 727 s; RoBERTa `skip`; all 9 Stage 1 and 15 Stage 2 runs `blocked` |
| CPU (`pc_queue_cpu.sh`) | 01:32 | 02:54 | 0 | gate verified (PASS, difference 0); 24 head runs; rows score 174 s; `gpu_wait=ok`; sheet |

The head runs took 400 to 680 s each (3 seeds per step: 1,182 to 2,030 s), not the 260 s of the build's timing: three
workers of 8 threads ran beside stage 0 and the load check below. The CPU queue ended at 02:54, not near 02:20.

### Stage 0: FAIL, and the September fit does not repeat itself

| Check | Recorded | Tonight | Status |
|---|---|---|---|
| Script 1 (Stage 1 snapshot) held-out emotion / genre AMI | 0.1225 / 0.2274 | 0.1244 / 0.2617 | FAIL |
| Script 2 (extended-seed Stage 2) held-out emotion / genre AMI | 0.1225 / 0.2274 | 0.1287 / 0.2804 | FAIL |
| Seed 42 Stage 2 macro AUC | 0.8296 | none (the script stops at its own re-fit check) | FAIL |
| Row 2, our scoring | 0.088556 / 0.170954 / 0.393076 | equal to 1e-6 | PASS |
| Four-fix unit tests | pass | 21 passed | PASS |

In the build (00:40 to 00:52) both scripts gave 0.1238 / 0.2617. The handoff's reading "this machine reproduces its
own September record bit for bit" does not hold. The load check
(`src/test/20261019_stage0_load_check/20261019_stage0_load_check_log.md`) traced it: the September k-means start
(K = 60, 10 restarts, 8 threads) gives different centers at the last float32 bit in every run, idle or loaded, with
the same labels. With one thread its centers are identical in every run. DEC then turns the last-bit differences into
one of a few outcomes. Eight local runs of the unchanged code:

| Outcome (emotion / genre) | DEC epoch 10 changed nodes | Runs |
|---|---|---|
| 0.1238 / 0.2617 | 2,973 | 5: September local sweep, build script 1 and 2, build thread test, tonight idle script 1 |
| 0.1244 / 0.2617 | 3,016 | 1: tonight script 1 (beside the CPU queue) |
| 0.1287 / 0.2804 | 3,140 | 2: tonight script 2 (beside the CPU queue, and idle) |

With one CPU thread (OMP, MKL and OpenBLAS at 1), two runs of script 1 gave identical logs and the same result,
0.1287 / 0.2804: one beside the Codex re-derivation (load 8.5), one idle.

PercepT core's Stage 1 (`pc_sept.py`) uses the same k-means start under `threadpool_limits(8)`, so its runs carry the
same noise. Per spec §3.5 the PercepT arms stay blocked; the user decides on stage 0.

### Head gate and rows 6, 10, 13 (complete, 3 seeds)

Gate: the build's rerun of genre R0 seed 0 (`outputs/heads/genre__R0__s0.npz`) verified and re-scored, target AMI
0.406099, difference 0 (PASS).

Target AMI (both-side mean for the aspect's label, k = 20, 5,000 evaluation paintings), mean ± population sd over
head seeds 0 to 2 (`outputs/pc_rows_tables.md`, `pc_rows_results.json`):

| Row | Graph, input | Emotion | Style | Genre |
|---|---|---:|---:|---:|
| 6 | default graph, default input | 0.0823 ± 0.0027 | 0.1630 ± 0.0018 | 0.3881 ± 0.0017 |
| 10 | other side's aspect graph (R0o), default input | 0.0857 ± 0.0003 | 0.1545 ± 0.0012 | (0.428, follow-up) |
| 13 | R0o, prompted input | 0.0889 ± 0.0013 | 0.1564 ± 0.0003 | 0.4342 ± 0.0026 |

Row 6's emotion and genre heads use the same default graph (pilot D, 47,516 fit pairs) and the same input, so their
outputs are byte-identical (same network, scored on two labels). Style's default graph is the image graph I (69,853).

### Row 1 (k-means on the September encoders; sd over k-means seeds)

| Caption encoder | Aspect | CLIP image | Caption | Both-side mean |
|---|---|---:|---:|---:|
| CLIP text | emotion | 0.0647 ± 0.0022 | 0.0667 ± 0.0033 | 0.0657 |
| CLIP text | style | 0.3000 ± 0.0027 | 0.1099 ± 0.0017 | 0.2050 |
| CLIP text | genre | 0.4116 ± 0.0098 | 0.3305 ± 0.0195 | 0.3710 |
| RoBERTa | emotion | 0.0647 ± 0.0022 | 0.1894 ± 0.0023 | 0.1271 |
| RoBERTa | style | 0.3000 ± 0.0027 | 0.0487 ± 0.0010 | 0.1743 |
| RoBERTa | genre | 0.4116 ± 0.0098 | 0.0680 ± 0.0007 | 0.2398 |

The updated ablation table: `pc_ablation.py` (reads the stored results only) writes `outputs/pc_ablation_table.md`.

### Re-derivation (C13)

Codex (`gpt-6.1-sol`, high) re-derived the head rows, the gate run and row 1 with its own code from the stored
outputs and the frozen sample, given the scoring rule and the data only (brief and outputs in the session
scratchpad; its script joins features by sample ID). All 150 per-run side scores, the 3-seed means and sds, and
every row 1 value equal ours (max abs difference 0). Its checks: 25 heads, no missing or extra file, largest norm
deviation 3.7e-5, label counts and ID joins. The PercepT comparator has no numbers to re-derive.

## Results, run 2 (PercepT arms, amended stage 0)

Run chat "r10 run 2", 2026-10-19, on `main` at `6cd4526`. Reports:
[auto report](../../../docs/reports/auto/percept/2026-12-14_percept_core_arms.md) and
[user-read report](../../../docs/user_read/2026-10-19_percept_core_arms.md). The first run's files are kept as
`pc_queue_gpu_run1.log`, `logs/stage0_run1.log` and `outputs/{stage0,pc_results,pc_tables,pc_ablation_table}_run1.*`.

### The queue

| Queue | Start | End | Exit | Notes |
|---|---|---|---|---|
| GPU (`pc_queue_gpu.sh --foreground --lock-wait 14400`) | 09:50:39 | 10:08:02 | 0 | stage 0 51 s; RoBERTa `skip`; 9 Stage 1 runs 76 to 99 s; 15 Stage 2 steps 3 to 8 s; `stage1_repeat` 77 s; `repeat_check` 0.1 s |

The CPU queue was not rerun. Score: `pc_score.py` (exit 0), `pc_ablation.py > outputs/pc_ablation_table.md` (exit 0;
rows other than 4a to 5c unchanged against `pc_ablation_table_run1.md`).

### Stage 0 (amended, A1): PASS

(i) 0 of 74 and 0 of 77 September files differ; (ii) 0 copied parts differ; (iii) snapshot 0.12246902855745419 /
0.22741920423971046 on 9,365 / 159 rows; (iv) 0 k-means labels differ (25 s); row 2 exact; 21 fix tests passed.
Stage 1 repeat (A2): `clip_roberta__s0` run twice, bitwise equal (sha256 `0f708d26...`), `fresh` on the sheet.

### Topic count (spec §8 case)

| Stage 1 version | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| clip_roberta (P1, P2) | 99 | 2 (100 at γ 5) | 3 |
| qwen_qwen (P3, P4) | 1 | 99 | 99 |
| qwen_roberta (P5) | 1 | 1 | 1 |

In every run the 100 center norms spread evenly (about 0.33 to 0.67); the largest adjacent gap (4.3 to 11.8 times
the mean gap) sits at an end, between sorted positions 1-2, 2-3, 3-4 or 99-100. At K = 99, 40 to 42 topics hold at
least 1% of the pairs. Stage 2 skips K < 2 (sidecar only): P3 s0, P4 s0, P5 s0 to s2.

### Arms, image-side AMI (k = 20, 3 k-means seeds, mean ± population sd over PercepT seeds)

| Arm | Emotion | Style | Genre | Dev AUC per seed | Established |
|---|---:|---:|---:|---|---|
| P1 | 0.02894 ± 0.01576 | 0.13682 ± 0.02772 | 0.23399 ± 0.08386 | 0.799 / 0.620 / 0.830 | no (seed 1 AUC) |
| P2 | 0.03552 ± 0.01973 | 0.16185 ± 0.07806 | 0.21697 ± 0.14261 | 0.896 / 0.752 / 0.901 | yes |
| P3 | seeds 1, 2: 0.07198, 0.07201 | 0.19151, 0.19702 | 0.44065, 0.45174 | — / 0.946 / 0.943 | no (seed 0 K = 1) |
| P4 | seeds 1, 2: 0.05055, 0.05304 | 0.15600, 0.15573 | 0.34854, 0.34748 | — / 0.784 / 0.791 | no (seed 0 K = 1) |
| P5 | — | — | — | — | no (K = 1 on every seed) |

Comparator (D15, established arms): P2 for every aspect. Over all arms with three trained seeds, genre goes to P1
(0.23399). K-native measures and per-seed values: `outputs/pc_tables.md`.

### Re-derivation (C13)

Codex (`gpt-6.1-sol`, high) re-derived the image-side and argmax AMI of the 10 arm files, the arm means and sds, and
the comparator from `outputs/stage2/*.npz` and the frozen sample, given the rule only. 90 per-file values and 24 arm
aggregates equal ours (max abs difference 0); K per file agrees. Its comparator rule had no "established" filter, so
its genre pick (P1) equals our all-arms pick. Files: `rederive_run2/` (brief `task.txt`, `rederive_summary.md`,
scripts, `compare.txt`).

## Results, run 3 (A3, fixed K = 40)

Run chat "r10 run 3", 2026-10-19, on `main` at `dfa3a85`. Reports:
[auto report](../../../docs/reports/auto/percept/2026-12-15_percept_core_k40.md) and
[user-read report](../../../docs/user_read/2026-10-19_percept_core_k40.md). Run 2's files are kept as
`outputs/*_run2*`, `logs/run2/` and `pc_queue_gpu_run2.log`.

### The queue

| Queue | Start | End | Exit | Notes |
|---|---|---|---|---|
| GPU (`pc_queue_gpu.sh --foreground --lock-wait 14400`) | 16:34:09 | 16:51:49 | 0 | every step 0; stage 0 51 s; RoBERTa `skip`; 9 Stage 1 runs 77 to 99 s; 15 Stage 2 steps 5 to 8 s; `stage1_repeat` 77 s; `repeat_check` 0.1 s |

The CPU queue was not rerun. Score: `pc_score.py && pc_ablation.py > outputs/pc_ablation_table.md` in one chain (exit
0, 16:52); rows other than 4a to 5c unchanged against `pc_ablation_table_run2.md` (diff checked).

### Stage 0 and the checks

Stage 0 PASS: (i) 0 of 74 and 0 of 77 September files differ; (ii) 0 copied parts differ; (iii) snapshot
0.12246902855745419 / 0.22741920423971046; (iv) 0 k-means labels differ (25 s); row 2 exact; 30 fix tests passed.
Stage 1 repeat (A2): bitwise equal (sha256 `255a5b03...`), `fresh`. Training check (final review item 7b): each Stage
1 file's `center_norms` and sidecar `K_gamma` equal run 2's (`K` there) in 9 of 9 files, bitwise.

### Arms, image-side AMI (k = 20, 3 k-means seeds, mean ± population sd over PercepT seeds)

| Arm | Emotion | Style | Genre | Dev AUC (mean ± sd) | Established |
|---|---:|---:|---:|---|---|
| P1 | 0.05296 ± 0.00071 | 0.16732 ± 0.00291 | 0.35583 ± 0.00078 | 0.852 ± 0.002 | yes |
| P2 | 0.06555 ± 0.00196 | 0.22856 ± 0.00491 | 0.40007 ± 0.00460 | 0.907 ± 0.003 | yes |
| P3 | 0.07717 ± 0.00124 | 0.18854 ± 0.00284 | 0.43588 ± 0.00374 | 0.959 ± 0.001 | yes |
| P4 | 0.05350 ± 0.00005 | 0.16248 ± 0.00317 | 0.36147 ± 0.00119 | 0.862 ± 0.003 | yes |
| P5 | 0.07375 ± 0.00199 | 0.18059 ± 0.00512 | 0.43505 ± 0.00690 | 0.961 ± 0.002 | yes |

Comparator (D15, established arms; the all-arms pick is the same): emotion P3, style P2, genre P3 (P5 within 0.003).
Kept topics: K = 40 in all 9 Stage 1 runs; 40 of 40 kept topics hold at least 1% of the fit pairs; share before the
cut 0.567 to 0.627; the 40th and 41st centers differ by 1 to 14 fit pairs. Algorithm 1's K per γ is unchanged from
run 2 (reported only). K-native measures and per-seed values: `outputs/pc_tables.md`.

### Re-derivation (C13)

Codex (`gpt-6.1-sol`, high) re-derived the image-side and argmax AMI of the 15 arm files, the arm means and sds, the
comparator over established and over all arms, and the Stage 1 kept-topic checks (top-40 pick from `fit_counts`,
coverage, share, Stage 1 argmax AMI), given the rule, the data, the fit rows and the established list only. 90
per-file values, 60 arm aggregates, 9 shares and 27 Stage 1 argmax values equal ours (max abs difference 0); top-40
pick equal in 9 of 9; 286 input checks pass. Files: `rederive_run3/` (brief `task.txt`, `rederive_summary.md`,
`rederive.py`, `compare.txt`).

## Storage (2026-10-19, r10 decide 10)

Deleted with the user's approval: `outputs/heads/*.npz` (977 MB; the JSONs stay), `sept_stage0/` (81 MB; every
stage 0 run exports it again), the build 6 worktrees and branches `r10-pc-14-k40` and `r10-pc-a3`, and
`.scratch/r10-percept-core/review_tmp/`. Kept: `outputs/stage1/` and `outputs/stage2/` (run 3; stage 2 reads them to
add V, ARI and FMI). Ledger: `/project/claude-config/storage/disk_ledger.md`.
