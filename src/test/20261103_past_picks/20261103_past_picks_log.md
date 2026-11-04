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
| 11:48 | Stage 1 records committed before any test number; the printed command plus attribution lines | committed |
| 11:49 | Stage 2 started, `PP_CPU_JOBS` 3 | running |
| 12:30 | Stage 2 ended, 22 of 22 steps rc 0, 41 min | 137 cells in 7 files; no number read yet |
| 12:34 | C13 by Codex started (`gpt-6.1-sol`, high) | running |
| 13:27 | Codex ended: 65 tests pass; subset smoke `C13 FAIL items=13538 failed=27 max_abs=1.58e-05` | all 27 are frozen-image pick `score` fields on ArtELingo (10) and Affection (17); every claim cell passes |
| 13:40 | Controller: imports and file reads clean. Sonnet `high` review: no way for a wrong result to pass; one must-fix (gates (vi), (viii) not cross-checked) | fix list written |
| 14:06 | Fix round by `resume` (nine items): 108 tests pass; Affection subset `C13 PASS items=394 failed=0 max_abs=1.1e-16` | cause of the 27 failures closed for Affection; ArtELingo's own check waits for the full derive |
| 14:13 | Sonnet `high` re-review: all nine fixed, independence holds; one new must-fix (digits in encoder names false-fail `check-outcomes`) | Claude fixed it (the one fix round was used), added slot-order checks for the cost sentence; 116 tests pass |
| 19:00 | C13 full derive started: 56 sections, one `res-queue cpu` reservation each, 3 at once (`pp_c13_queue.sh`) | running |
| 20:23 | Full derive ended, 83 min, 56 of 56 sections | merged |
| 20:25 | `compare`: `C13 FAIL items=18778 failed=6 max_abs=1.85e-09` | all six are one label, Affection's SC6 comparator: `teacher_emotion_focus` in the derive, `teacher_artwork` in the round. Numbers agree |
| 20:26 | Label fixed in `compare` and `check-outcomes` only (layout; spec §3.2 says r12's artwork wording; r12's record confirms), 119 tests. `C13 PASS items=18778 failed=0 max_abs=1.85393241037e-09 scope=full`; `records/c13.json` written | ok |
| 20:29 | Stage 4 started: `decide` under one minute, rc 0, then diagnostics | every verdict holds; decide refused nothing |
| 20:30 | `check-outcomes`: `C13 FAIL items=3987 failed=59 max_abs=1.85e-09` | 58 items: stage 4's file has no `status` key. One item: claims never compared, since claim rows have no `encoder` field. No number differs |
| 20:31 | Both fixed inside `check_outcomes`, 121 tests. `C13 PASS items=4211 failed=0 max_abs=1.85393241037e-09 scope=outcomes`; `source_identity` still equals the derive | six alterations of a copy of the real file (claim word, claim row, claim outcome, verdict, cell d, sentence number) each fail one item |
| 21:48 | Stage 4 ended: diagnostics 79 min, 12 of 12 `ok` (`r17_metrics` included); `PP_RUN_SUMMARY stage4 steps=1 failed=0` | ok; stage 4 wrote nothing under `records/` |
| 21:50 | `pp_report_tables.py`: `records/outcomes.json`, `outcomes_check.json`, `diag_tables.md`, `report_tables.md` | committed; the hashed records did not change |

## Choices of the run chat (agent default)

- **M5 ran before the last two photo jobs ended.** The spec's order puts M5 after the photo models. GPU 0 was free at 03:46, and
  M5 reads no photo output. Result: no effect on any input.
- **MMArt and SemArt extraction ran as two chains at once**, so each held a queue place. The queue serialized their steps.
- **The probe ended at 10:22, past the sheet's 10:00 GPU limit.** It started at 09:19 on a quiet CPU. `shared-resources.md`
  counts probes as short work that can stay local, and the spec calls this one short.
- **The stage 1 commit landed second.** A shell error stopped the first attempt, so the run log commit 35db5c7d came before the
  records commit a617831b. Both came before stage 2 started.
- **The C13 brief's normalisation fact did not carry over.** Fact 1 (twice) came from r15. On ArtELingo and Affection r14 normalises
  a frozen row once (`bs_c13.py:373`, `:411`, `:530`), and D21 reuses r14's rule, so the derive was wrong and the round's tables
  right. The effect on the 27 scores was at most 1.6e-5, on scores of size 1 to 40.
- **The end-to-end pass has no warm-up batches.** `pp_probe.py:487` records `warmup_batches=0`; its docstring and §3.4 name 3
  warm-up batches. It ran right after the three model-time repeats, so the model was warm. It touches only <c> and <d> of the cost
  sentence, not SC15, so the probe was not rerun (the handoff says not to without a reason). The report says so.
- **The six C13 label failures were C13's, not the round's.** C13 copied r14's name for Affection's teacher; the round uses r12's
  wording name, which is what the spec says. Both name one arm, and the values match, so only the name check gained an alias, read
  from r12's record. Any other label still fails.
- **`check-outcomes` failed first on its own layout, and the claims had never been compared.** Stage 4 writes `outcomes.json`
  from `pp_run.py`, which leaves out `status` (only `pp_outcome.py`'s own command adds it), and files claim rows per encoder with no
  `encoder` field. The checker's unit fixture had neither shape. It now reads the committed `records/status.json` when the key is
  absent, and takes the encoder from the path, so all 56 claims and their sentence numbers are compared. Fix inside `check_outcomes`
  only; stage 4 and its outcomes were not rerun.
- **C13's full derive waits for 19:00.** The user's limit for CPU scoring is about 13:00, and the code is not written yet. The
  report's date (about 11-06) allows this.

## User-read report (run chat 5, 2026-11-04)

- **Draft:** Codex, by resume of the report session, told to read the committed report again. The controller then rewrote it: a table of
  the smallest differences replaced a table of nine "It holds." cells, and prefetch and warm-up detail stayed in the full report.
- **Figure:** SemArt genre mAP per encoder, built from `records/outcomes.json` by the script beside it.
- **Checks:** cold read on Haiku, fidelity check on Sonnet `high`. No number was wrong. One fix wave, then the humanizer pass.
- **Findings applied:** "the teacher stays ahead" now says "in most new-encoder cells" (4 WIN and 4 LEVEL among the 48 new-encoder
  cells); SigLIP 2's LEVEL in the figure's caption; "matched by name" in the new-artist heading; exact copies stated as inside SemArt;
  the cost table marked as model time, with the with-decoding pass named; "a small output layer" dropped, since each head is a
  trunk-and-head network.
- **Findings not applied:** "published method" stays, since PercepT's published paper is the project's own definition. "The gates cover
  those" stays, since the claim review fixed that wording. D17's reason for the unset cells stays out: it explains the design and
  changes no result's meaning.
