# Round log: PercepT core (r10, from 2026-10-18)

Spec: `docs/superpowers/specs/2026-10-18-percept-core-design.md`. Tickets: `.scratch/r10-percept-core/issues/`
(outside git). Module prefix `pc_`. Outputs in `outputs/` and `sept_stage0/` (both gitignored).

## What the round does

1. Stage 0, the regression check: the unchanged September PercepT (commit `1b70d7f`) and our row 2 scoring.
2. Stage 1 of the spec, the baseline: PercepT core with the four fixes, arms P1 to P5, seeds 0 to 2 (GPU).
3. Stage 2 of the spec, the ablation: rows 1, 6, 10, 13 of the follow-up's table, after the head gate (CPU).

## Modules

| Module | Does |
|---|---|
| `pc_data.py` | frozen-sample readers, partitions fit / dev / eval (13,400 / 1,489 / 5,000 paintings) |
| `pc_roberta.py` | RoBERTa-GoEmotions per-caption store (`/data/SSD2/pre_extract/artelingo_full/roberta_go_emotions/`) |
| `pc_fixes.py` | the four fixes (pair count, Eq. 2 fusion, Algorithm 1 cut, multi-hot targets) |
| `pc_sept.py` | the September training functions, copied without change |
| `pc_stage1.py`, `pc_stage2.py` | PercepT core Stage 1 (topics) and Stage 2 (image mapper) |
| `pc_rows.py` | graphs and head rows 6, 10, 13; the head gate (`gate.json`) |
| `pc_score.py` | image-side scoring, row 1, row 2 check, the results sheet |
| `pc_stage0.py` | stage 0 |
| `pc_queue_gpu.sh`, `pc_queue_cpu.sh` | the two queues (how to call them: `pc_queue.md`) |

## Commands

From the repository root:

```bash
# tests (about 61 s)
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q src/test/20261018_percept_core
# the run: see pc_queue.md (GPU queue with --foreground in a background shell, CPU queue beside it)
```

## Build (2026-10-18 to 2026-10-19)

- Build 1 and 2: tickets 01 to 08 and 06b, merged on `r10-percept-core`; handoffs in `docs/superpowers/handoffs/`.
- Build 3: the stage 0 trace and the final whole-branch review (Opus `xhigh`: fix first, two queue fixes; all
  load-bearing numbers re-derived on P1 and P2 seed 0 to 1e-10). Fix wave: ticket 09.

## Stage 0 (2026-10-19)

FAIL as the spec defines it. The unchanged September re-fit gives held-out emotion AMI 0.1238 (recorded 0.1225) and
genre 0.2617 (recorded 0.2274); row 2 and the four-fix tests pass. Cause traced: the recorded values come from the
DAS6 node node404; this machine gives 0.1238 / 0.2617, as it did in September, with identical pretraining losses.
Trace: `src/test/20261019_stage0_trace/20261019_stage0_trace_log.md`. Per spec §3.5 the PercepT arms stop; the head
rows still run; the user decides.

## Topic count (spec §8 case)

Full-size Stage 1 timing runs, seed 0: clip_roberta K = 99 (the largest gap is at the top, 7.75 times the mean gap;
only the highest center is dropped), qwen_qwen K = 1 (the largest gap is at the bottom, 7.06 times the mean gap), the
same at every γ. Spec: report it, no rerun with another γ.

## Results

To be filled by the run chat.
