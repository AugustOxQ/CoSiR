# r19 run log (past picks)

Run chat. All times Amsterdam. Worktree `r19-int`, branch `r19-past-picks`, code commit dd7e7c0b.
Run sheet: `pp_run_sheet.md`. Spec: `docs/superpowers/specs/2026-11-03-past-picks-design.md`.

## Records from the build, for the report

- `OUT/matched_sample` (gate (v), DINOv2) was copied at 01:13 on 11-04 from build 3's local rerun: 9 files, 6.8 MB,
  weights sha256 equal to the stored run's. The spec says node402; the controller accepted the local 3090 rerun. Not rerun.
- Node input probe (tag 20261103-232618-001d3d1-9a592cff, 59 inputs, 0 failures) and node smoke (tag
  20261103-232644-001d3d1-78737588) passed in the build on 001d3d1. `pp_node_inputs.json` has not changed since then, so
  neither was rerun.

## Events

| Time (11-04) | Event | Result |
|---|---|---|
| 02:24 | Run chat starts. Affection lines (3) exit 0, store checks (3) true | ok |
| 02:25 | Parity queued (`pp_extract_queue.sh parity`), behind r18 on the local GPU | waiting, start about 04:23 |
| 02:25 | Gate (ii) training, tag 20261104-012527-dd7e7c0-c2f8fa3c, GPU 0 | succeeded, 15 min |
| 02:45 | Gate (ii) comparison on the local CPU, `OUT/stage0/gate_ii.json` | `pass` true; every difference 0.0; weights sha256 equal |
| 02:46 | c* DINOv2, tag 20261104-014636-dd7e7c0-917eccfb, GPU 0 | running |
| 02:47 | c* CLIP B/32, tag 20261104-014658-dd7e7c0-84539bbe, GPU 1 | running |
| 02:47 | c* CLIP B/16, tag 20261104-014721-dd7e7c0-dd77f70c, GPU 2 | running |
| 03:27 | c* DINOv2 pulled | `ok`, 5 seeds, 40 min |
| 03:33 | c* CLIP B/32 pulled | `ok`, 5 seeds, 44 min |
| 03:37 | c* CLIP B/16 pulled | `ok`, 5 seeds, 47 min |
| 03:46 | Photo DINOv2, tag 20261104-022825-dd7e7c0-9ca7982a, pulled | `ok`, 5 seeds |
| 03:49 | M5 DINOv2 full scale, tag 20261104-024711-dd7e7c0-0b9a806a, pulled | job succeeded, about 2 min |
| 03:55 | Photo CLIP B/32, tag 20261104-023418-dd7e7c0-0dbee809, pulled | `ok`, 5 seeds |
| 03:59 | Photo CLIP B/16, tag 20261104-023800-dd7e7c0-ad0eeb06, pulled | `ok`, 5 seeds |
| 03:59 | Node work done | 15 c*, 15 photo, M5; every run on its first attempt; node402 free |
| 08:06 | Parity ended, 13 of 13 checks `pass`, minimum cosine 0.9997 (bar 0.999) | ok. The chain took 5.6 h, not about 1 h: other chats' GPU jobs ran between its steps (parity-dinov2_large 3.0 h, parity-clip_b16 2.5 h, in the queue) |
| 08:08 | MMArt and SemArt extraction started as two chains (agent default, so each holds a queue place) | running; planned start was about 07:00 |
| 08:29 | MMArt three and SemArt B/16, DINOv2 stores done | 5 stores `ok`, 5 to 10 min each |
| 09:18 | SemArt CLIP L/14, SigLIP 1, `genre_word` stores done | all 8 new stores `ok`; ended 42 min before the 10:00 limit |
| 09:19 | Probe chain started on a quiet CPU, load 2.3 | running |
| 10:22 | Probe chain ended, 63 min, load 1.8 to 10.5, 0 repeats discarded; the last steps ran past 10:00 as a short, light probe (`shared-resources.md`) | all rc 0 |
| 10:22 | Stage 3 (`cells_cost`, `cost_table`, gate (viii)) | `pass` true: Qwen default end to end 88.4 ms, reference 86.3 ms, difference 2.5% |
| 10:23 | Stage 0 CPU steps started, `PP_CPU_JOBS` 3 | running |
| 11:03 | Stage 0 ended, 44 of 44 steps rc 0, 40 min | gates (i) to (viii) all ok in `gate.json`; SemArt set 11,162 paintings, 2,717 artists, 11,159 after md5; D23 maximum difference 0.0; mapper replays at most 5.4e-07 |
| 11:05 | Stage 1 started, `PP_CPU_JOBS` 3 | running |
| 11:48 | Stage 1 ended, 19 of 19 steps rc 0, 42 min; `status.json` all ok; `missing_stores` empty | ok |
| 11:50 | Stage 1 records committed before any test number; the printed command plus attribution lines | committed |
